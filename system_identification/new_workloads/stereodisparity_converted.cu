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

#define BLOCK_X 32
#define BLOCK_Y 8
#define RAD 8
#define STEPS 3

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

static inline int iDivUp(int a, int b) { return (a + b - 1) / b; }

static int g_w = 640 * 8;
static int g_h = 480 * 8;
static const int minDisp = -16;
static const int maxDisp = 0;
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

static std::vector<uint32_t> h_left, h_right, h_out;
static uint32_t *d_left = nullptr, *d_right = nullptr, *d_out = nullptr;
static cudaTextureObject_t texLeft = 0, texRight = 0;
static size_t memSize = 0;

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

// PTX SAD on 4 packed bytes
__device__ __forceinline__ unsigned int usad4_u32(unsigned int A, unsigned int B,
                                                  unsigned int C = 0) {
    unsigned int r;
    asm("vabsdiff4.u32.u32.u32.add %0,%1,%2,%3;" : "=r"(r) : "r"(A), "r"(B), "r"(C));
    return r;
}

__global__ void stereoDisparityKernel(uint32_t *g_img0, uint32_t *g_img1, uint32_t *g_odata, int w,
                                      int h, int minDisp, int maxDisp,
                                      cudaTextureObject_t tex2Dleft,
                                      cudaTextureObject_t tex2Dright) {
    const int tidx = blockDim.x * blockIdx.x + threadIdx.x;
    const int tidy = blockDim.y * blockIdx.y + threadIdx.y;

    const int sidx = threadIdx.x + RAD;
    const int sidy = threadIdx.y + RAD;

    __shared__ unsigned int diff[BLOCK_Y + 2 * RAD][BLOCK_X + 2 * RAD];

    unsigned int imLeftA[STEPS], imLeftB[STEPS];

#pragma unroll
    for (int i = 0; i < STEPS; i++) {
        int offset = -RAD + i * RAD;
        float y = float(tidy + offset);
        imLeftA[i] = tex2D<unsigned int>(tex2Dleft, float(tidx - RAD), y);
        imLeftB[i] = tex2D<unsigned int>(tex2Dleft, float(tidx - RAD + BLOCK_X), y);
    }

    unsigned int bestCost = 0xFFFFFFFFu;
    unsigned int bestDisparity = 0;

    for (int d = minDisp; d <= maxDisp; d++) {
#pragma unroll
        for (int i = 0; i < STEPS; i++) {
            int offset = -RAD + i * RAD;
            float y = float(tidy + offset);
            unsigned int imLeft = imLeftA[i];
            unsigned int imRight = tex2D<unsigned int>(tex2Dright, float(tidx - RAD + d), y);
            unsigned int cost = usad4_u32(imLeft, imRight);
            diff[sidy + offset][sidx - RAD] = cost;
        }

#pragma unroll
        for (int i = 0; i < STEPS; i++) {
            int offset = -RAD + i * RAD;
            if (threadIdx.x < 2 * RAD) {
                float y = float(tidy + offset);
                unsigned int imLeft = imLeftB[i];
                unsigned int imRight =
                    tex2D<unsigned int>(tex2Dright, float(tidx - RAD + BLOCK_X + d), y);
                unsigned int cost = usad4_u32(imLeft, imRight);
                diff[sidy + offset][sidx - RAD + BLOCK_X] = cost;
            }
        }
        __syncthreads();

#pragma unroll
        for (int j = 0; j < STEPS; j++) {
            int offset = -RAD + j * RAD;
            unsigned int cost = 0;
#pragma unroll
            for (int i = -RAD; i <= RAD; i++)
                cost += diff[sidy + offset][sidx + i];
            __syncthreads();
            diff[sidy + offset][sidx] = cost;
            __syncthreads();
        }

        unsigned int cost = 0;
#pragma unroll
        for (int i = -RAD; i <= RAD; i++)
            cost += diff[sidy + i][sidx];

        if (cost < bestCost) {
            bestCost = cost;
            bestDisparity = d + 8;
        }
        __syncthreads();
    }

    if (tidx < w && tidy < h)
        g_odata[tidy * w + tidx] = bestDisparity;
}

void init_memory() {
    const size_t num = static_cast<size_t>(g_w) * g_h;
    memSize = num * sizeof(uint32_t);

    h_left.resize(num);
    h_right.resize(num);
    h_out.resize(num, 0);

    const int shift = 8;
    std::srand(1);
    for (int y = 0; y < g_h; ++y) {
        int base = y * g_w;
        for (int x = 0; x < g_w; ++x) {
            unsigned char r = static_cast<unsigned char>((x * 31 + y * 17 + std::rand()) & 0xFF);
            unsigned char g = static_cast<unsigned char>((x * 13 + y * 29 + std::rand()) & 0xFF);
            unsigned char b = static_cast<unsigned char>((x * 7 + y * 23 + std::rand()) & 0xFF);
            unsigned char a = 255;
            uint32_t px =
                (uint32_t(a) << 24) | (uint32_t(b) << 16) | (uint32_t(g) << 8) | uint32_t(r);
            h_left[base + x] = px;
        }
        for (int x = 0; x < g_w; ++x) {
            int xs = x + shift;
            if (xs >= g_w) xs = g_w - 1;
            h_right[base + x] = h_left[base + xs];
        }
    }

    CUDA_CHECK(cudaMalloc(&d_left, memSize));
    CUDA_CHECK(cudaMalloc(&d_right, memSize));
    CUDA_CHECK(cudaMalloc(&d_out, memSize));

    CUDA_CHECK(cudaMemcpy(d_left, h_left.data(), memSize, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_right, h_right.data(), memSize, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_out, 0, memSize));

    cudaChannelFormatDesc ch = cudaCreateChannelDesc<unsigned int>();

    cudaResourceDesc resL{};
    resL.resType = cudaResourceTypePitch2D;
    resL.res.pitch2D.devPtr = d_left;
    resL.res.pitch2D.desc = ch;
    resL.res.pitch2D.width = g_w;
    resL.res.pitch2D.height = g_h;
    resL.res.pitch2D.pitchInBytes = g_w * sizeof(uint32_t);

    cudaResourceDesc resR = resL;
    resR.res.pitch2D.devPtr = d_right;

    cudaTextureDesc tdesc{};
    tdesc.normalizedCoords = 0;
    tdesc.filterMode = cudaFilterModePoint;
    tdesc.addressMode[0] = cudaAddressModeClamp;
    tdesc.addressMode[1] = cudaAddressModeClamp;
    tdesc.readMode = cudaReadModeElementType;

    CUDA_CHECK(cudaCreateTextureObject(&texLeft, &resL, &tdesc, nullptr));
    CUDA_CHECK(cudaCreateTextureObject(&texRight, &resR, &tdesc, nullptr));

    CUDA_CHECK(cudaDeviceSynchronize());
}

void free_memory() {
    if (texLeft) cudaDestroyTextureObject(texLeft);
    if (texRight) cudaDestroyTextureObject(texRight);
    if (d_left) cudaFree(d_left);
    if (d_right) cudaFree(d_right);
    if (d_out) cudaFree(d_out);
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

    dim3 blockDim(BLOCK_X, BLOCK_Y);
    dim3 gridDim(iDivUp(g_w, blockDim.x), iDivUp(g_h, blockDim.y));

    float total_kernel_ms = 0.0f;
    float total_h2d_ms = 0.0f;
    float total_d2h_ms = 0.0f;

    CUDA_CHECK(cudaEventRecord(startH2D));
    CUDA_CHECK(cudaMemcpy(d_out, h_out.data(), memSize, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaEventRecord(stopH2D));
    CUDA_CHECK(cudaEventSynchronize(stopH2D));

    float h2d_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&h2d_ms, startH2D, stopH2D));
    total_h2d_ms += h2d_ms;

    for (int i = 0; i < jobs; ++i) {
        CUDA_CHECK(cudaEventRecord(startKernel));
        stereoDisparityKernel<<<gridDim, blockDim>>>(
            d_left, d_right, d_out, g_w, g_h, minDisp, maxDisp, texLeft, texRight);
        CUDA_CHECK(cudaEventRecord(stopKernel));
        CUDA_CHECK(cudaEventSynchronize(stopKernel));

        float kernel_ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&kernel_ms, startKernel, stopKernel));
        total_kernel_ms += kernel_ms;
    }

    CUDA_CHECK(cudaEventRecord(startD2H));
    CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, memSize, cudaMemcpyDeviceToHost));
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
    out << "  \"workload\": \"stereodisparity_releaseguard\",\n";
    out << "  \"width\": " << g_w << ",\n";
    out << "  \"height\": " << g_h << ",\n";
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
    if (argc < 5) {
        std::cerr << "Usage: " << argv[0]
                  << " <width> <height> <period_s> <duration_s> [jobs] [out_json]\n";
        return 1;
    }

    g_w = std::atoi(argv[1]);
    g_h = std::atoi(argv[2]);
    period_s = std::atof(argv[3]);
    duration_s = std::atof(argv[4]);
    if (argc >= 6) jobs = std::atoi(argv[5]);
    std::string out_json = "logs/stereodisparity_releaseguard.json";
    if (argc >= 7) out_json = argv[6];

    std::filesystem::create_directories("logs");
    responsetime_log.open("logs/rt_stereo.txt", std::ios::out);
    summary_log.open("logs/summary_stereo.txt", std::ios::out);

    std::cout << "Width       : " << g_w << "\n";
    std::cout << "Height      : " << g_h << "\n";
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