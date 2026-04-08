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

#define BLOCK_SIZE 16
#define IN_RANGE(a, x, y) ((a) >= (x) && (a) <= (y))

#define CUDA_CHECK(call)                                                                           \
    do {                                                                                           \
        cudaError_t _e = (call);                                                                   \
        if (_e != cudaSuccess) {                                                                   \
            std::cerr << "CUDA error: " << cudaGetErrorString(_e) << " at " << __FILE__ << ":"    \
                      << __LINE__ << "\n";                                                         \
            std::exit(1);                                                                          \
        }                                                                                          \
    } while (0)

const int grid_rows = 1024 * 8;
const int grid_cols = 1024 * 8;
const int total_cells = grid_rows * grid_cols;
const int iteration = 2;
const float Cap = 0.5f;
const float Rx = 0.1f;
const float Ry = 0.1f;
const float Rz = 0.1f;
const float step = 0.0001f;
const float time_elapsed = 0.0f;

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

__global__ void calculate_temp(int iteration, float *power, float *temp_src, float *temp_dst,
                               int grid_cols, int grid_rows, int border_cols, int border_rows,
                               float Cap, float Rx, float Ry, float Rz, float step,
                               float time_elapsed) {
    __shared__ float temp_on_cuda[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float power_on_cuda[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float temp_t[BLOCK_SIZE][BLOCK_SIZE];

    float amb_temp = 80.0f;
    float step_div_Cap = step / Cap;
    float Rx_1 = 1.0f / Rx;
    float Ry_1 = 1.0f / Ry;
    float Rz_1 = 1.0f / Rz;

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int small_block_rows = BLOCK_SIZE - iteration * 2;
    int small_block_cols = BLOCK_SIZE - iteration * 2;

    int blkY = small_block_rows * by - border_rows;
    int blkX = small_block_cols * bx - border_cols;
    int blkYmax = blkY + BLOCK_SIZE - 1;
    int blkXmax = blkX + BLOCK_SIZE - 1;

    int yidx = blkY + ty;
    int xidx = blkX + tx;
    int index = grid_cols * yidx + xidx;

    if (IN_RANGE(yidx, 0, grid_rows - 1) && IN_RANGE(xidx, 0, grid_cols - 1)) {
        temp_on_cuda[ty][tx] = temp_src[index];
        power_on_cuda[ty][tx] = power[index];
    }
    __syncthreads();

    int validYmin = (blkY < 0) ? -blkY : 0;
    int validYmax =
        (blkYmax > grid_rows - 1) ? BLOCK_SIZE - 1 - (blkYmax - grid_rows + 1) : BLOCK_SIZE - 1;
    int validXmin = (blkX < 0) ? -blkX : 0;
    int validXmax =
        (blkXmax > grid_cols - 1) ? BLOCK_SIZE - 1 - (blkXmax - grid_cols + 1) : BLOCK_SIZE - 1;

    int N = ty - 1;
    int S = ty + 1;
    int W = tx - 1;
    int E = tx + 1;
    N = (N < validYmin) ? validYmin : N;
    S = (S > validYmax) ? validYmax : S;
    W = (W < validXmin) ? validXmin : W;
    E = (E > validXmax) ? validXmax : E;

    bool computed = false;
    for (int i = 0; i < iteration; i++) {
        computed = false;
        if (IN_RANGE(tx, i + 1, BLOCK_SIZE - i - 2) && IN_RANGE(ty, i + 1, BLOCK_SIZE - i - 2) &&
            IN_RANGE(tx, validXmin, validXmax) && IN_RANGE(ty, validYmin, validYmax)) {
            computed = true;
            temp_t[ty][tx] =
                temp_on_cuda[ty][tx] +
                step_div_Cap *
                    (power_on_cuda[ty][tx] +
                     (temp_on_cuda[S][tx] + temp_on_cuda[N][tx] - 2.0f * temp_on_cuda[ty][tx]) *
                         Ry_1 +
                     (temp_on_cuda[ty][E] + temp_on_cuda[ty][W] - 2.0f * temp_on_cuda[ty][tx]) *
                         Rx_1 +
                     (amb_temp - temp_on_cuda[ty][tx]) * Rz_1);
        }
        __syncthreads();
        if (i == iteration - 1) break;
        if (computed) temp_on_cuda[ty][tx] = temp_t[ty][tx];
        __syncthreads();
    }

    if (computed && IN_RANGE(yidx, 0, grid_rows - 1) && IN_RANGE(xidx, 0, grid_cols - 1))
        temp_dst[index] = temp_t[ty][tx];
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

class HotspotBenchmark {
  public:
    HotspotBenchmark(int repeats_, int jobs_) : repeats(repeats_), jobs(jobs_) {}

    void init() {
        std::vector<float> h_power(total_cells);
        std::vector<float> h_temp_src(total_cells);
        std::vector<float> h_temp_dst(total_cells, 0.0f);

        std::srand(1);
        for (int i = 0; i < total_cells; ++i) {
            h_power[i] = (std::rand() % 1000) / 1000.0f * 3.0f;
            h_temp_src[i] = 80.0f + (std::rand() % 1000) / 1000.0f * 5.0f;
            h_temp_init[i] = h_temp_src[i];
            h_power_init[i] = h_power[i];
            h_temp_zero[i] = 0.0f;
        }

        CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        CUDA_CHECK(cudaEventCreate(&startEv));
        CUDA_CHECK(cudaEventCreate(&stopEv));

        CUDA_CHECK(cudaMalloc(&d_power, total_cells * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_temp_src, total_cells * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_temp_dst, total_cells * sizeof(float)));

        CUDA_CHECK(cudaMemcpy(d_power, h_power_init.data(), total_cells * sizeof(float),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_temp_src, h_temp_init.data(), total_cells * sizeof(float),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_temp_dst, h_temp_zero.data(), total_cells * sizeof(float),
                              cudaMemcpyHostToDevice));

        blockDim = dim3(BLOCK_SIZE, BLOCK_SIZE);
        gridDim = dim3((grid_cols + BLOCK_SIZE - 1) / BLOCK_SIZE,
                       (grid_rows + BLOCK_SIZE - 1) / BLOCK_SIZE);
    }

    void destroy() {
        CUDA_CHECK(cudaEventDestroy(startEv));
        CUDA_CHECK(cudaEventDestroy(stopEv));
        CUDA_CHECK(cudaStreamDestroy(stream));
        CUDA_CHECK(cudaFree(d_power));
        CUDA_CHECK(cudaFree(d_temp_src));
        CUDA_CHECK(cudaFree(d_temp_dst));
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
            CUDA_CHECK(cudaMemcpy(d_power, h_power_init.data(), total_cells * sizeof(float),
                                  cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_temp_src, h_temp_init.data(), total_cells * sizeof(float),
                                  cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_temp_dst, h_temp_zero.data(), total_cells * sizeof(float),
                                  cudaMemcpyHostToDevice));

            float sum_kernel_ms = 0.0f;
            for (int j = 0; j < jobs; ++j) {
                CUDA_CHECK(cudaEventRecord(startEv, stream));
                calculate_temp<<<gridDim, blockDim, 0, stream>>>(
                    iteration, d_power, d_temp_src, d_temp_dst, grid_cols, grid_rows,
                    0, 0, Cap, Rx, Ry, Rz, step, time_elapsed);
                CUDA_CHECK(cudaEventRecord(stopEv, stream));
                CUDA_CHECK(cudaEventSynchronize(stopEv));

                float kernel_ms = 0.0f;
                CUDA_CHECK(cudaEventElapsedTime(&kernel_ms, startEv, stopEv));
                sum_kernel_ms += kernel_ms;

                std::swap(d_temp_src, d_temp_dst);
            }
            execs_ms.push_back(sum_kernel_ms);
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

    cudaStream_t stream{};
    cudaEvent_t startEv{}, stopEv{};
    dim3 gridDim{}, blockDim{};

    float *d_power = nullptr, *d_temp_src = nullptr, *d_temp_dst = nullptr;
    std::vector<float> h_power_init = std::vector<float>(total_cells);
    std::vector<float> h_temp_init = std::vector<float>(total_cells);
    std::vector<float> h_temp_zero = std::vector<float>(total_cells);
};

static void save_results_json(const std::string& path, const std::vector<ResultRow>& rows) {
    std::ofstream out(path);
    if (!out.is_open()) {
        std::cerr << "Failed to open JSON output: " << path << "\n";
        return;
    }

    out << "{\n";
    out << "  \"gpu\": 0,\n";
    out << "  \"workload\": \"hotspot\",\n";
    out << "  \"grid_rows\": " << grid_rows << ",\n";
    out << "  \"grid_cols\": " << grid_cols << ",\n";
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
    std::string out_json = "hotspot_freq_power_exec";
    const int repeats = 100;
    const int jobs = 2;
    const int stride = 1;

    if (argc >= 2) out_json = argv[1];

    std::cout << "Output JSON : " << out_json << "\n";
    std::cout << "Repeats     : " << repeats << "\n";
    std::cout << "Jobs        : " << jobs << "\n";
    std::cout << "LUT stride  : " << stride << "\n";
    std::cout << "Power sample: 5 ms\n";

    HotspotBenchmark bench(repeats, jobs);
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