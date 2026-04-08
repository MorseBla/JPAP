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

#define MAX_THREADS_PER_BLOCK 256

#define CUDA_CHECK(call)                                                                           \
    do {                                                                                           \
        cudaError_t _e = (call);                                                                   \
        if (_e != cudaSuccess) {                                                                   \
            std::cerr << "CUDA error: " << cudaGetErrorString(_e) << " at " << __FILE__ << ":"    \
                      << __LINE__ << "\n";                                                         \
            std::exit(1);                                                                          \
        }                                                                                          \
    } while (0)

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

struct Node {
    int starting;
    int no_of_edges;
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

void monitorpower_and_freq(int sample_period_ms = 5) {  // every 5 ms
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

__global__ void BFS_Kernel(Node *g_graph_nodes, int *g_graph_edges, unsigned char *g_graph_mask,
                           unsigned char *g_updating_graph_mask, unsigned char *g_graph_visited,
                           int *g_cost, int no_of_nodes) {
    int tid = blockIdx.x * MAX_THREADS_PER_BLOCK + threadIdx.x;
    if (tid < no_of_nodes && g_graph_mask[tid]) {
        g_graph_mask[tid] = 0;
        for (int i = g_graph_nodes[tid].starting;
             i < g_graph_nodes[tid].starting + g_graph_nodes[tid].no_of_edges; i++) {
            int id = g_graph_edges[i];
            if (!g_graph_visited[id]) {
                g_cost[id] = g_cost[tid] + 1;
                g_updating_graph_mask[id] = 1;
            }
        }
    }
}

__global__ void BFS_Kernel2(unsigned char *g_graph_mask, unsigned char *g_updating_graph_mask,
                            unsigned char *g_graph_visited, bool *g_over, int no_of_nodes) {
    int tid = blockIdx.x * MAX_THREADS_PER_BLOCK + threadIdx.x;
    if (tid < no_of_nodes && g_updating_graph_mask[tid]) {
        g_graph_mask[tid] = 1;
        g_graph_visited[tid] = 1;
        *g_over = true;
        g_updating_graph_mask[tid] = 0;
    }
}

class BFSBenchmark {
  public:
    BFSBenchmark(int iterations_, int jobs_, int repeats_)
        : iterations(iterations_), jobs(jobs_), repeats(repeats_) {}

    void init() {
        host_nodes.resize(no_of_nodes);
        host_edges.resize(no_of_edges);

        std::srand(1);
        for (int i = 0; i < no_of_nodes; ++i) {
            host_nodes[i].starting = i * edges_per_node;
            host_nodes[i].no_of_edges = edges_per_node;
            for (int j = 0; j < edges_per_node; ++j) {
                host_edges[i * edges_per_node + j] = std::rand() % no_of_nodes;
            }
        }

        CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        CUDA_CHECK(cudaEventCreate(&startEv));
        CUDA_CHECK(cudaEventCreate(&stopEv));

        CUDA_CHECK(cudaMalloc(&d_nodes, sizeof(Node) * no_of_nodes));
        CUDA_CHECK(cudaMalloc(&d_edges, sizeof(int) * no_of_edges));
        CUDA_CHECK(cudaMalloc(&d_mask, sizeof(unsigned char) * no_of_nodes));
        CUDA_CHECK(cudaMalloc(&d_updating_mask, sizeof(unsigned char) * no_of_nodes));
        CUDA_CHECK(cudaMalloc(&d_visited, sizeof(unsigned char) * no_of_nodes));
        CUDA_CHECK(cudaMalloc(&d_cost, sizeof(int) * no_of_nodes));
        CUDA_CHECK(cudaMalloc(&d_over, sizeof(bool)));

        CUDA_CHECK(cudaMemcpy(d_nodes, host_nodes.data(), sizeof(Node) * no_of_nodes,
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_edges, host_edges.data(), sizeof(int) * no_of_edges,
                              cudaMemcpyHostToDevice));

        blockDim = dim3(MAX_THREADS_PER_BLOCK);
        gridDim = dim3((no_of_nodes + blockDim.x - 1) / blockDim.x);
    }

    void destroy() {
        CUDA_CHECK(cudaEventDestroy(startEv));
        CUDA_CHECK(cudaEventDestroy(stopEv));
        CUDA_CHECK(cudaStreamDestroy(stream));

        CUDA_CHECK(cudaFree(d_nodes));
        CUDA_CHECK(cudaFree(d_edges));
        CUDA_CHECK(cudaFree(d_mask));
        CUDA_CHECK(cudaFree(d_updating_mask));
        CUDA_CHECK(cudaFree(d_visited));
        CUDA_CHECK(cudaFree(d_cost));
        CUDA_CHECK(cudaFree(d_over));
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

        for (int rep = 0; rep < repeats; ++rep) {  // 20 runs
            reset_graph_state();

            float sum_kernel_ms = 0.0f;
            for (int j = 0; j < jobs; ++j) {
                CUDA_CHECK(cudaEventRecord(startEv, stream));
                for (int iter = 0; iter < iterations; ++iter) {
                    bool over = false;
                    CUDA_CHECK(cudaMemcpyAsync(d_over, &over, sizeof(bool),
                                               cudaMemcpyHostToDevice, stream));
                    BFS_Kernel<<<gridDim, blockDim, 0, stream>>>(
                        d_nodes, d_edges, d_mask, d_updating_mask, d_visited, d_cost, no_of_nodes);
                    BFS_Kernel2<<<gridDim, blockDim, 0, stream>>>(
                        d_mask, d_updating_mask, d_visited, d_over, no_of_nodes);
                }
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
        row.avg_exec_ms = avg.calculate(execs_ms);
        row.repeats = repeats;
        row.monitor_samples = local_powers.size();
        return row;
    }

  private:
    void reset_graph_state() {
        std::vector<unsigned char> h_mask(no_of_nodes, 0);
        std::vector<unsigned char> h_updating_mask(no_of_nodes, 0);
        std::vector<unsigned char> h_visited(no_of_nodes, 0);
        std::vector<int> h_cost(no_of_nodes, 0);

        h_mask[0] = 1;
        h_visited[0] = 1;

        CUDA_CHECK(cudaMemcpy(d_mask, h_mask.data(), sizeof(unsigned char) * no_of_nodes,
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_updating_mask, h_updating_mask.data(),
                              sizeof(unsigned char) * no_of_nodes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_visited, h_visited.data(), sizeof(unsigned char) * no_of_nodes,
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_cost, h_cost.data(), sizeof(int) * no_of_nodes,
                              cudaMemcpyHostToDevice));
    }

    static constexpr int no_of_nodes = 1 << 24;
    static constexpr int edges_per_node = 8;
    static constexpr int no_of_edges = no_of_nodes * edges_per_node;

    int iterations;
    int jobs;
    int repeats;

    std::vector<Node> host_nodes;
    std::vector<int> host_edges;

    cudaStream_t stream{};
    cudaEvent_t startEv{}, stopEv{};
    dim3 gridDim{}, blockDim{};

    Node *d_nodes{};
    int *d_edges{};
    unsigned char *d_mask{}, *d_updating_mask{}, *d_visited{};
    int *d_cost{};
    bool *d_over{};
};

static void save_results_json(const std::string& path, const std::vector<ResultRow>& rows) {
    std::ofstream out(path);
    if (!out.is_open()) {
        std::cerr << "Failed to open JSON output: " << path << "\n";
        return;
    }

    out << "{\n";
    out << "  \"gpu\": 0,\n";
    out << "  \"workload\": \"bfs\",\n";
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
    std::string out_json = "bfs_freq_power_exec";
    const int repeats = 100;    // average of 20 runs
    const int iterations = 30;
    const int jobs = 2;
    const int stride = 1;     // granularity 15

    if (argc >= 2) out_json = argv[1];

    std::cout << "Output JSON : " << out_json << "\n";
    std::cout << "Repeats     : " << repeats << "\n";
    std::cout << "Iterations  : " << iterations << "\n";
    std::cout << "Jobs        : " << jobs << "\n";
    std::cout << "LUT stride  : " << stride << "\n";
    std::cout << "Power sample: 5 ms\n";

    BFSBenchmark bench(iterations, jobs, repeats);
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