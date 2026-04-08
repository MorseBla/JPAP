#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>
#include <filesystem>
#include <fstream>
#include <future>
#include <iostream>
#include <mutex>
#include <numeric>
#include <string>
#include <thread>
#include <vector>

#define CUDA_CHECK(call)                                                                           \
    do {                                                                                           \
        cudaError_t _e = (call);                                                                   \
        if (_e != cudaSuccess) {                                                                   \
            std::cerr << "CUDA error: " << cudaGetErrorString(_e) << " at " << __FILE__ << ":"    \
                      << __LINE__ << "\n";                                                         \
            std::exit(1);                                                                          \
        }                                                                                          \
    } while (0)

using Clock = std::chrono::steady_clock;

static unsigned int g_num_elements = 1 << 25;
static const unsigned int g_num_bins = 4096;
static int jobs = 1;
static float period_s = 0.1f;
static float duration_s = 100.0f;

static std::mutex sample_mutex;
static std::atomic<bool> monitor_running{false};
static std::vector<double> sampled_powers_w;
static std::vector<double> sampled_freqs_mhz;

static std::mutex timing_mutex;
static std::vector<float> response_times_ms;
static std::vector<float> kernel_exec_times_ms;
static std::vector<float> h2d_times_ms;
static std::vector<float> d2h_times_ms;
static std::vector<float> memory_times_ms;

static Clock::time_point program_start_time;

static std::vector<unsigned int> host_input;
static std::vector<unsigned int> host_bins;
static unsigned int* d_input = nullptr;
static unsigned int* d_bins = nullptr;

static std::ofstream responsetime_log;
static std::ofstream summary_log;

float average_of(const std::vector<float>& v) {
    if (v.empty()) return -1.0f;
    float s = 0.0f;
    for (float x : v) s += x;
    return s / static_cast<float>(v.size());
}

double average_of(const std::vector<double>& v) {
    if (v.empty()) return -1.0;
    double s = std::accumulate(v.begin(), v.end(), 0.0);
    return s / static_cast<double>(v.size());
}

double read_double_from_file(const std::string& path) {
    std::ifstream in(path);
    if (!in.is_open()) return -1.0;
    double val = -1.0;
    in >> val;
    return val;
}

double returnfreqjetson() {
    const std::vector<std::string> candidates_hz = {
        "/sys/class/devfreq/17000000.gpu/cur_freq",
        "/sys/class/devfreq/57000000.gpu/cur_freq"
    };

    for (const auto& p : candidates_hz) {
        double hz = read_double_from_file(p);
        if (hz > 0.0) return hz / 1e6;
    }
    return -1.0;
}

double returnpowerjetson() {
    const char* cmd = "tegrastats --interval 1 2>/dev/null | head -n 1";
    FILE* pipe = popen(cmd, "r");
    if (!pipe) return -1.0;

    char buffer[4096] = {};
    if (!fgets(buffer, sizeof(buffer), pipe)) {
        pclose(pipe);
        return -1.0;
    }
    pclose(pipe);

    std::string s(buffer);
    std::size_t pos = s.find("VDD_IN ");
    if (pos == std::string::npos) return -1.0;

    pos += 7;
    std::size_t end = s.find("mW", pos);
    if (end == std::string::npos) return -1.0;

    std::string mw_str = s.substr(pos, end - pos);
    std::size_t slash = mw_str.find('/');
    if (slash != std::string::npos) mw_str = mw_str.substr(0, slash);

    try {
        double mw = std::stod(mw_str);
        return mw / 1000.0;
    } catch (...) {
        return -1.0;
    }
}

void monitorpower_and_freq(int sample_period_ms = 100) {
    while (monitor_running.load(std::memory_order_relaxed)) {
        double p = returnpowerjetson();
        double f = returnfreqjetson();
        {
            std::lock_guard<std::mutex> lg(sample_mutex);
            if (p >= 0.0) sampled_powers_w.push_back(p);
            if (f >= 0.0) sampled_freqs_mhz.push_back(f);
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(sample_period_ms));
    }
}

__global__ void histogram_kernel(unsigned int* input, unsigned int* bins,
                                 unsigned int num_elements, unsigned int num_bins) {
    unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    for (unsigned int j = tid; j < num_elements; j += stride) {
        unsigned int position = input[j];
        if (position < num_bins) atomicAdd(&(bins[position]), 1);
    }
}

void init_memory() {
    host_input.resize(g_num_elements);
    host_bins.resize(g_num_bins, 0);

    std::srand(1);
    for (unsigned int i = 0; i < g_num_elements; ++i) {
        host_input[i] = std::rand() % g_num_bins;
    }

    CUDA_CHECK(cudaMalloc(&d_input, g_num_elements * sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&d_bins, g_num_bins * sizeof(unsigned int)));

    CUDA_CHECK(cudaMemcpy(
        d_input, host_input.data(),
        g_num_elements * sizeof(unsigned int),
        cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemcpy(
        d_bins, host_bins.data(),
        g_num_bins * sizeof(unsigned int),
        cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaDeviceSynchronize());
}

void free_memory() {
    if (d_input) cudaFree(d_input);
    if (d_bins) cudaFree(d_bins);
}

void task(int id) {
    auto task_start = Clock::now();

    cudaEvent_t startKernel, stopKernel, startH2D, stopH2D, startD2H, stopD2H;
    CUDA_CHECK(cudaEventCreate(&startKernel));
    CUDA_CHECK(cudaEventCreate(&stopKernel));
    CUDA_CHECK(cudaEventCreate(&startH2D));
    CUDA_CHECK(cudaEventCreate(&stopH2D));
    CUDA_CHECK(cudaEventCreate(&startD2H));
    CUDA_CHECK(cudaEventCreate(&stopD2H));

    dim3 blockDim(256);
    dim3 gridDim((g_num_elements + blockDim.x - 1) / blockDim.x);

    float total_kernel_ms = 0.0f;
    float total_h2d_ms = 0.0f;
    float total_d2h_ms = 0.0f;

    CUDA_CHECK(cudaEventRecord(startH2D));
    CUDA_CHECK(cudaMemcpy(
        d_bins, host_bins.data(),
        g_num_bins * sizeof(unsigned int),
        cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaEventRecord(stopH2D));
    CUDA_CHECK(cudaEventSynchronize(stopH2D));

    float h2d_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&h2d_ms, startH2D, stopH2D));
    total_h2d_ms += h2d_ms;

    for (int i = 0; i < jobs; ++i) {
        CUDA_CHECK(cudaEventRecord(startKernel));
        histogram_kernel<<<gridDim, blockDim>>>(d_input, d_bins, g_num_elements, g_num_bins);
        CUDA_CHECK(cudaEventRecord(stopKernel));
        CUDA_CHECK(cudaEventSynchronize(stopKernel));

        float kernel_ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&kernel_ms, startKernel, stopKernel));
        total_kernel_ms += kernel_ms;
    }

    CUDA_CHECK(cudaEventRecord(startD2H));
    CUDA_CHECK(cudaMemcpy(
        host_bins.data(), d_bins,
        g_num_bins * sizeof(unsigned int),
        cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaEventRecord(stopD2H));
    CUDA_CHECK(cudaEventSynchronize(stopD2H));

    float d2h_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&d2h_ms, startD2H, stopD2H));
    total_d2h_ms += d2h_ms;

    CUDA_CHECK(cudaDeviceSynchronize());

    auto task_finish = Clock::now();
    float response_ms =
        std::chrono::duration<float, std::milli>(task_finish - task_start).count();

    float memory_ms = total_h2d_ms + total_d2h_ms;
    float ke_ms = total_kernel_ms + memory_ms;

    {
        std::lock_guard<std::mutex> lg(timing_mutex);
        response_times_ms.push_back(response_ms);
        kernel_exec_times_ms.push_back(ke_ms);
        h2d_times_ms.push_back(total_h2d_ms);
        d2h_times_ms.push_back(total_d2h_ms);
        memory_times_ms.push_back(memory_ms);
    }

    if (responsetime_log.is_open()) {
        responsetime_log << "task_id=" << id
                         << " response_ms=" << response_ms
                         << " kernel_plus_memory_ms=" << ke_ms
                         << " kernel_only_ms=" << total_kernel_ms
                         << " h2d_ms=" << total_h2d_ms
                         << " d2h_ms=" << total_d2h_ms
                         << "\n";
    }

    CUDA_CHECK(cudaEventDestroy(startKernel));
    CUDA_CHECK(cudaEventDestroy(stopKernel));
    CUDA_CHECK(cudaEventDestroy(startH2D));
    CUDA_CHECK(cudaEventDestroy(stopH2D));
    CUDA_CHECK(cudaEventDestroy(startD2H));
    CUDA_CHECK(cudaEventDestroy(stopD2H));
}

void periodicTaskLauncher(float duration, float period) {
    int taskCounter = 0;
    auto start_time = program_start_time;
    std::future<void> asyncTask;

    while (Clock::now() - start_time < std::chrono::duration<float>(duration)) {
        asyncTask = std::async(std::launch::async, [=]() {
            task(taskCounter);
        });

        taskCounter++;
        std::this_thread::sleep_for(
            std::chrono::milliseconds(static_cast<long long>(period * 1000.0f)));
    }

    if (asyncTask.valid()) {
        asyncTask.get();
    }
}

static void save_results_json(const std::string& path) {
    std::ofstream out(path);
    if (!out.is_open()) {
        std::cerr << "Failed to open JSON output: " << path << "\n";
        return;
    }

    float avg_rt = average_of(response_times_ms);
    float avg_ke = average_of(kernel_exec_times_ms);
    float avg_h2d = average_of(h2d_times_ms);
    float avg_d2h = average_of(d2h_times_ms);
    float avg_mem = average_of(memory_times_ms);

    double avg_power = -1.0;
    double avg_freq = -1.0;
    size_t monitor_samples = 0;
    {
        std::lock_guard<std::mutex> lg(sample_mutex);
        avg_power = average_of(sampled_powers_w);
        avg_freq = average_of(sampled_freqs_mhz);
        monitor_samples = sampled_powers_w.size();
    }

    out << "{\n";
    out << "  \"platform\": \"jetson\",\n";
    out << "  \"workload\": \"histogram_releaseguard\",\n";
    out << "  \"num_elements\": " << g_num_elements << ",\n";
    out << "  \"num_bins\": " << g_num_bins << ",\n";
    out << "  \"period_s\": " << period_s << ",\n";
    out << "  \"duration_s\": " << duration_s << ",\n";
    out << "  \"jobs\": " << jobs << ",\n";
    out << "  \"power_sample_period_ms\": 100,\n";
    out << "  \"avg_response_ms\": " << avg_rt << ",\n";
    out << "  \"avg_ke_ms\": " << avg_ke << ",\n";
    out << "  \"avg_memory_ms\": " << avg_mem << ",\n";
    out << "  \"avg_h2d_ms\": " << avg_h2d << ",\n";
    out << "  \"avg_d2h_ms\": " << avg_d2h << ",\n";
    out << "  \"avg_measured_freq_mhz\": " << avg_freq << ",\n";
    out << "  \"avg_power_w\": " << avg_power << ",\n";
    out << "  \"monitor_samples\": " << monitor_samples << ",\n";
    out << "  \"num_released_jobs\": " << response_times_ms.size() << "\n";
    out << "}\n";
}

int main(int argc, char* argv[]) {
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0]
                  << " <num_elements> <period_s> <duration_s> [jobs] [out_json]\n";
        return 1;
    }

    g_num_elements = static_cast<unsigned int>(std::strtoul(argv[1], nullptr, 10));
    period_s = std::atof(argv[2]);
    duration_s = std::atof(argv[3]);
    if (argc >= 5) jobs = std::atoi(argv[4]);
    std::string out_json = "logs/histogram_releaseguard.json";
    if (argc >= 6) out_json = argv[5];

    std::filesystem::create_directories("logs");
    responsetime_log.open("logs/rt_histogram.txt", std::ios::out);
    summary_log.open("logs/summary_histogram.txt", std::ios::out);

    std::cout << "Num elements: " << g_num_elements << "\n";
    std::cout << "Period (s)  : " << period_s << "\n";
    std::cout << "Duration (s): " << duration_s << "\n";
    std::cout << "Jobs        : " << jobs << "\n";
    std::cout << "Output JSON : " << out_json << "\n";

    init_memory();
    program_start_time = Clock::now();

    {
        std::lock_guard<std::mutex> lg(sample_mutex);
        sampled_powers_w.clear();
        sampled_freqs_mhz.clear();
    }

    monitor_running.store(true, std::memory_order_relaxed);
    std::thread mon(monitorpower_and_freq, 100);

    std::thread launcher(periodicTaskLauncher, duration_s, period_s);
    launcher.join();

    monitor_running.store(false, std::memory_order_relaxed);
    mon.join();

    save_results_json(out_json);

    float avg_rt = average_of(response_times_ms);
    float avg_ke = average_of(kernel_exec_times_ms);
    float avg_mem = average_of(memory_times_ms);

    std::cout << "Response time = " << avg_rt
              << " ms, KE = " << avg_ke
              << " ms, Memory = " << avg_mem << " ms\n";

    if (summary_log.is_open()) {
        summary_log << "avg_response_ms " << avg_rt << "\n";
        summary_log << "avg_ke_ms " << avg_ke << "\n";
        summary_log << "avg_memory_ms " << avg_mem << "\n";
    }

    if (responsetime_log.is_open()) responsetime_log.close();
    if (summary_log.is_open()) summary_log.close();

    free_memory();
    return 0;
}