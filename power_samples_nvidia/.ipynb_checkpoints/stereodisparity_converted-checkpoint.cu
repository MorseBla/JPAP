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

static inline int iDivUp(int a, int b) { return (a + b - 1) / b; }

static const int w = 640 * 8;
static const int h = 480 * 8;
static const int minDisp = -16;
static const int maxDisp = 0;
static const size_t num = size_t(w) * h;
static const size_t memSize = num * sizeof(uint32_t);

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
    double avg_execution_ms = -1.0;
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

class StereoBenchmark {
  public:
    StereoBenchmark(int repeats_, int jobs_) : repeats(repeats_), jobs(jobs_) {}

    void init() {
        h_left.resize(num);
        h_right.resize(num);
        h_out.resize(num, 0);

        const int shift = 8;
        std::srand(1);
        for (int y = 0; y < h; ++y) {
            int base = y * w;
            for (int x = 0; x < w; ++x) {
                unsigned char r = static_cast<unsigned char>((x * 31 + y * 17 + std::rand()) & 0xFF);
                unsigned char g = static_cast<unsigned char>((x * 13 + y * 29 + std::rand()) & 0xFF);
                unsigned char b = static_cast<unsigned char>((x * 7 + y * 23 + std::rand()) & 0xFF);
                unsigned char a = 255;
                uint32_t px =
                    (uint32_t(a) << 24) | (uint32_t(b) << 16) | (uint32_t(g) << 8) | uint32_t(r);
                h_left[base + x] = px;
            }
            for (int x = 0; x < w; ++x) {
                int xs = x + shift;
                if (xs >= w) xs = w - 1;
                h_right[base + x] = h_left[base + xs];
            }
        }

        CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        CUDA_CHECK(cudaEventCreate(&startEv));
        CUDA_CHECK(cudaEventCreate(&stopEv));

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
        resL.res.pitch2D.width = w;
        resL.res.pitch2D.height = h;
        resL.res.pitch2D.pitchInBytes = w * sizeof(uint32_t);

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

        blockDim = dim3(BLOCK_X, BLOCK_Y);
        gridDim = dim3(iDivUp(w, blockDim.x), iDivUp(h, blockDim.y));
    }

    void destroy() {
        CUDA_CHECK(cudaDestroyTextureObject(texLeft));
        CUDA_CHECK(cudaDestroyTextureObject(texRight));
        CUDA_CHECK(cudaEventDestroy(startEv));
        CUDA_CHECK(cudaEventDestroy(stopEv));
        CUDA_CHECK(cudaStreamDestroy(stream));
        CUDA_CHECK(cudaFree(d_left));
        CUDA_CHECK(cudaFree(d_right));
        CUDA_CHECK(cudaFree(d_out));
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

            CUDA_CHECK(cudaMemset(d_out, 0, memSize));

            for (int j = 0; j < jobs; ++j) {
                CUDA_CHECK(cudaEventRecord(startEv, stream));
                stereoDisparityKernel<<<gridDim, blockDim, 0, stream>>>(
                    d_left, d_right, d_out, w, h, minDisp, maxDisp, texLeft, texRight);
                CUDA_CHECK(cudaEventRecord(stopEv, stream));
                CUDA_CHECK(cudaEventSynchronize(stopEv));

                float kernel_ms = 0.0f;
                CUDA_CHECK(cudaEventElapsedTime(&kernel_ms, startEv, stopEv));
                sum_kernel_ms += kernel_ms;
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
        row.avg_execution_ms = avg.calculate(execs_ms);
        row.repeats = repeats;
        row.monitor_samples = local_powers.size();
        return row;
    }

  private:
    int repeats;
    int jobs;

    std::vector<uint32_t> h_left, h_right, h_out;

    cudaStream_t stream{};
    cudaEvent_t startEv{}, stopEv{};
    dim3 gridDim{}, blockDim{};

    uint32_t *d_left = nullptr, *d_right = nullptr, *d_out = nullptr;
    cudaTextureObject_t texLeft = 0, texRight = 0;
};

static void save_results_json(const std::string& path, const std::vector<ResultRow>& rows) {
    std::ofstream out(path);
    if (!out.is_open()) {
        std::cerr << "Failed to open JSON output: " << path << "\n";
        return;
    }

    out << "{\n";
    out << "  \"gpu\": 0,\n";
    out << "  \"workload\": \"stereodisparity\",\n";
    out << "  \"width\": " << w << ",\n";
    out << "  \"height\": " << h << ",\n";
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
        out << "      \"avg_execution_ms\": " << r.avg_execution_ms << ",\n";
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
    std::string out_json = "stereodisparity_freq_power_exec";
    const int repeats = 100;
    const int jobs = 2;
    const int stride = 1;

    if (argc >= 2) out_json = argv[1];

    std::cout << "Output JSON : " << out_json << "\n";
    std::cout << "Repeats     : " << repeats << "\n";
    std::cout << "Jobs        : " << jobs << "\n";
    std::cout << "LUT stride  : " << stride << "\n";
    std::cout << "Power sample: 5 ms\n";

    StereoBenchmark bench(repeats, jobs);
    bench.init();

    std::vector<ResultRow> rows;
    for (size_t i = 0; i < lut.size(); i += static_cast<size_t>(stride)) {
        int f = lut[i];
        std::cout << "Running frequency " << f << " MHz...\n";
        ResultRow row = bench.run_one_frequency(f);
        std::cout << "  requested=" << row.requested_freq_mhz
                  << " measured=" << row.avg_measured_freq_mhz
                  << " power=" << row.avg_power_w
                  << " exec_ms=" << row.avg_execution_ms << "\n";
        rows.push_back(row);
    }

    bench.destroy();
    reset_freq_nvidia_rtx();
    save_results_json(out_json, rows);

    std::cout << "Saved JSON to " << out_json << "\n";
    return 0;
}