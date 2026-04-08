#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>
#include <fstream>
#include <iostream>
#include <mutex>
#include <numeric>
#include <string>
#include <thread>
#include <vector>
#include <cmath>

#define CUDA_CHECK(call)                                                                           \
    do {                                                                                           \
        cudaError_t _e = (call);                                                                   \
        if (_e != cudaSuccess) {                                                                   \
            std::cerr << "CUDA error: " << cudaGetErrorString(_e) << " at " << __FILE__ << ":"    \
                      << __LINE__ << "\n";                                                         \
            std::exit(1);                                                                          \
        }                                                                                          \
    } while (0)

const int Nparticles = 1 << 16;

static const std::vector<int> lut = {
    2100, 2085, 2070, 2055, 2040, 2025, 2010, 1995, 1980, 1965,
    1950, 1935, 1920, 1905, 1890, 1875, 1860, 1845, 1830, 1815,
    1800, 1785, 1770, 1755, 1740, 1725, 1710, 1695, 1680, 1665,
    1650, 1635, 1620, 1605, 1590, 1575, 1560, 1545, 1530, 1515,
    1500, 1485, 1470, 1455, 1440, 1425, 1410, 1395, 1380, 1365,
    1350, 1335, 1320, 1305, 1290, 1275, 1260, 1245, 1230, 1215,
    1200, 1185, 1170, 1155, 1140, 1125, 1110, 1095, 1080, 1065,
    1050, 1035, 1020, 1005, 990, 975, 960, 945, 930, 915,
    900, 885, 870, 855, 840, 825, 810, 795, 780, 765,
    750, 735, 720, 705, 690, 675, 660, 645, 630, 615,
    600, 585, 570, 555, 540, 525, 510, 495, 480, 465,
    450, 435, 420, 405, 390, 375, 360, 345, 330, 315,
    300, 285, 270, 255, 240, 225, 210
};

struct ResultRow {
    int requested_freq_mhz = -1;
    double avg_measured_freq_mhz = -1.0;
    double avg_power_w = -1.0;
    double avg_exec_ms = -1.0;
    int repeats = 0;
    size_t monitor_samples = 0;
};

class Average {
  public:
    double calculate(const std::vector<double>& v) const {
        if (v.empty()) return -1.0;
        double s = std::accumulate(v.begin(), v.end(), 0.0);
        return s / static_cast<double>(v.size());
    }
};

static std::atomic<bool> monitor_running{false};
static std::mutex sample_mutex;
static std::vector<double> sampled_powers;
static std::vector<double> sampled_freqs;

__global__ void particleKernel(double *arrayX, double *arrayY, double *CDF, double *u, double *xj,
                               double *yj, int Nparticles) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < Nparticles) {
        int index = -1;
        for (int x = 0; x < Nparticles; x++) {
            if (CDF[x] >= u[i]) {
                index = x;
                break;
            }
        }
        if (index == -1) index = Nparticles - 1;

        xj[i] = arrayX[index];
        yj[i] = arrayY[index];
    }
}

int returnpowerrtx() {
    const char* cmd =
        "nvidia-smi -i 0 --query-gpu=power.draw --format=csv,noheader,nounits";
    FILE* pipe = popen(cmd, "r");
    if (!pipe) return -1;

    char buffer[64] = {};
    if (!fgets(buffer, sizeof(buffer), pipe)) {
        pclose(pipe);
        return -1;
    }
    pclose(pipe);
    return std::atoi(buffer);
}

int returnfreqrtx() {
    const char* cmd =
        "nvidia-smi -i 0 --query-gpu=clocks.current.graphics --format=csv,noheader,nounits";
    FILE* pipe = popen(cmd, "r");
    if (!pipe) return -1;

    char buffer[64] = {};
    if (!fgets(buffer, sizeof(buffer), pipe)) {
        pclose(pipe);
        return -1;
    }
    pclose(pipe);
    return std::atoi(buffer);
}

void activate_freq_nvidia_rtx(int coreClock) {
    std::string command =
        "sudo nvidia-smi -i 0 -lgc " + std::to_string(coreClock) + "," + std::to_string(coreClock) +
        " > /dev/null 2>&1";
    (void)std::system(command.c_str());
}

void reset_freq_nvidia_rtx() {
    std::string command = "sudo nvidia-smi -i 0 -rgc > /dev/null 2>&1";
    (void)std::system(command.c_str());
}

void monitorpower_and_freq(int sample_period_ms = 5) {
    while (monitor_running.load(std::memory_order_relaxed)) {
        int p = returnpowerrtx();
        int f = returnfreqrtx();
        {
            std::lock_guard<std::mutex> lg(sample_mutex);
            if (p >= 0) sampled_powers.push_back(static_cast<double>(p));
            if (f >= 0) sampled_freqs.push_back(static_cast<double>(f));
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(sample_period_ms));
    }
}

class ParticleBenchmark {
  public:
    ParticleBenchmark(int repeats_, int jobs_) : repeats(repeats_), jobs(jobs_) {}

    void init() {
        h_arrayX.resize(Nparticles);
        h_arrayY.resize(Nparticles);
        h_CDF.resize(Nparticles);
        h_u.resize(Nparticles);
        h_xj.resize(Nparticles, 0.0);
        h_yj.resize(Nparticles, 0.0);

        std::srand(1);
        for (int i = 0; i < Nparticles; ++i) {
            h_arrayX[i] = std::sin(i * 0.01);
            h_arrayY[i] = std::cos(i * 0.01);
        }

        double sum = 0.0;
        for (int i = 0; i < Nparticles; ++i) {
            double val = (std::rand() % 1000) / 1000.0;
            sum += val;
            h_CDF[i] = sum;
        }
        for (int i = 0; i < Nparticles; ++i) h_CDF[i] /= sum;
        for (int i = 0; i < Nparticles; ++i) h_u[i] = (std::rand() % 1000) / 1000.0;

        CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        CUDA_CHECK(cudaEventCreate(&startEv));
        CUDA_CHECK(cudaEventCreate(&stopEv));

        CUDA_CHECK(cudaMalloc(&d_arrayX, Nparticles * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_arrayY, Nparticles * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_CDF, Nparticles * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_u, Nparticles * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_xj, Nparticles * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_yj, Nparticles * sizeof(double)));

        CUDA_CHECK(cudaMemcpy(d_arrayX, h_arrayX.data(), Nparticles * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_arrayY, h_arrayY.data(), Nparticles * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_CDF, h_CDF.data(), Nparticles * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_u, h_u.data(), Nparticles * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_xj, h_xj.data(), Nparticles * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_yj, h_yj.data(), Nparticles * sizeof(double), cudaMemcpyHostToDevice));

        blockDim = dim3(256);
        gridDim = dim3((Nparticles + blockDim.x - 1) / blockDim.x);
    }

    void destroy() {
        CUDA_CHECK(cudaEventDestroy(startEv));
        CUDA_CHECK(cudaEventDestroy(stopEv));
        CUDA_CHECK(cudaStreamDestroy(stream));

        CUDA_CHECK(cudaFree(d_arrayX));
        CUDA_CHECK(cudaFree(d_arrayY));
        CUDA_CHECK(cudaFree(d_CDF));
        CUDA_CHECK(cudaFree(d_u));
        CUDA_CHECK(cudaFree(d_xj));
        CUDA_CHECK(cudaFree(d_yj));
    }

    ResultRow run_one_frequency(int target_freq_mhz) {
        activate_freq_nvidia_rtx(target_freq_mhz);
        std::this_thread::sleep_for(std::chrono::milliseconds(300));

        std::vector<double> execs_ms;
        execs_ms.reserve(repeats);

        {
            std::lock_guard<std::mutex> lg(sample_mutex);
            sampled_powers.clear();
            sampled_freqs.clear();
        }

        monitor_running.store(true, std::memory_order_relaxed);
        std::thread mon(monitorpower_and_freq, 5);

        for (int rep = 0; rep < repeats; ++rep) {
            float sum_kernel_ms = 0.0f;

            CUDA_CHECK(cudaMemcpyAsync(d_xj, h_xj.data(), Nparticles * sizeof(double),
                                       cudaMemcpyHostToDevice, stream));
            CUDA_CHECK(cudaMemcpyAsync(d_yj, h_yj.data(), Nparticles * sizeof(double),
                                       cudaMemcpyHostToDevice, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));

            for (int j = 0; j < jobs; ++j) {
                CUDA_CHECK(cudaEventRecord(startEv, stream));
                particleKernel<<<gridDim, blockDim, 0, stream>>>(
                    d_arrayX, d_arrayY, d_CDF, d_u, d_xj, d_yj, Nparticles);
                CUDA_CHECK(cudaEventRecord(stopEv, stream));
                CUDA_CHECK(cudaEventSynchronize(stopEv));

                float kernel_ms = 0.0f;
                CUDA_CHECK(cudaEventElapsedTime(&kernel_ms, startEv, stopEv));
                sum_kernel_ms += kernel_ms;
            }

            execs_ms.push_back(sum_kernel_ms / static_cast<float>(jobs));
        }

        monitor_running.store(false, std::memory_order_relaxed);
        mon.join();

        Average avg;
        std::vector<double> local_powers, local_freqs;
        {
            std::lock_guard<std::mutex> lg(sample_mutex);
            local_powers = sampled_powers;
            local_freqs = sampled_freqs;
        }

        ResultRow row;
        row.requested_freq_mhz = target_freq_mhz;
        row.avg_measured_freq_mhz = avg.calculate(local_freqs);
        row.avg_power_w = avg.calculate(local_powers);
        row.avg_exec_ms = avg.calculate(execs_ms);
        row.repeats = repeats;
        row.monitor_samples = local_powers.size();
        return row;
    }

  private:
    int repeats;
    int jobs;

    std::vector<double> h_arrayX, h_arrayY, h_CDF, h_u, h_xj, h_yj;

    cudaStream_t stream{};
    cudaEvent_t startEv{}, stopEv{};
    dim3 gridDim{}, blockDim{};

    double *d_arrayX = nullptr, *d_arrayY = nullptr, *d_CDF = nullptr;
    double *d_u = nullptr, *d_xj = nullptr, *d_yj = nullptr;
};

static void save_results_json(const std::string& path, const std::vector<ResultRow>& rows) {
    std::ofstream out(path);
    if (!out.is_open()) {
        std::cerr << "Failed to open JSON output: " << path << "\n";
        return;
    }

    out << "{\n";
    out << "  \"gpu\": 0,\n";
    out << "  \"workload\": \"particle\",\n";
    out << "  \"num_particles\": " << Nparticles << ",\n";
    out << "  \"power_sample_period_ms\": 5,\n";
    out << "  \"frequency_sweep_stride\": 15,\n";
    out << "  \"runs_per_frequency\": 20,\n";
    out << "  \"results\": [\n";

    for (size_t i = 0; i < rows.size(); ++i) {
        const auto& r = rows[i];
        out << "    {\n";
        out << "      \"requested_freq_mhz\": " << r.requested_freq_mhz << ",\n";
        out << "      \"avg_measured_freq_mhz\": " << r.avg_measured_freq_mhz << ",\n";
        out << "      \"avg_power_w\": " << r.avg_power_w << ",\n";
        out << "      \"avg_execution_ms\": " << r.avg_exec_ms << ",\n";
        out << "      \"repeats\": " << r.repeats << ",\n";
        out << "      \"monitor_samples\": " << r.monitor_samples << "\n";
        out << "    }";
        if (i + 1 != rows.size()) out << ",";
        out << "\n";
    }

    out << "  ]\n";
    out << "}\n";
}

int main(int argc, char** argv) {
    std::string out_json = "particle_freq_power_exec";
    const int repeats = 100;
    const int jobs = 2;
    const int stride = 1;

    if (argc >= 2) out_json = argv[1];

    std::cout << "Output JSON : " << out_json << "\n";
    std::cout << "Repeats     : " << repeats << "\n";
    std::cout << "Jobs        : " << jobs << "\n";
    std::cout << "LUT stride  : " << stride << "\n";
    std::cout << "Power sample: 5 ms\n";

    ParticleBenchmark bench(repeats, jobs);
    bench.init();

    std::vector<ResultRow> rows;
    for (size_t i = 0; i < lut.size(); i += static_cast<size_t>(stride)) {
        int f = lut[i];
        std::cout << "Running frequency " << f << " MHz...\n";
        ResultRow row = bench.run_one_frequency(f);
        std::cout << "  requested=" << row.requested_freq_mhz
                  << " measured=" << row.avg_measured_freq_mhz
                  << " power=" << row.avg_power_w
                  << " exec_ms=" << row.avg_exec_ms << "\n";
        rows.push_back(row);
    }

    bench.destroy();
    reset_freq_nvidia_rtx();
    save_results_json(out_json, rows);

    std::cout << "Saved JSON to " << out_json << "\n";
    return 0;
}