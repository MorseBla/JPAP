#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>
#include <fstream>
#include <future>
#include <iostream>
#include <mutex>
#include <numeric>
#include <string>
#include <thread>
#include <vector>
#include <filesystem>

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

using Clock = std::chrono::steady_clock;

struct Node {
    int starting;
    int no_of_edges;
};

static int NO_OF_NODES = 1 << 24;
static int EDGES_PER_NODE = 8;
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
    BFSBenchmark(int no_of_nodes_, int edges_per_node_)
        : no_of_nodes(no_of_nodes_),
          edges_per_node(edges_per_node_),
          no_of_edges(no_of_nodes_ * edges_per_node_) {}

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
        CUDA_CHECK(cudaEventCreate(&startH2D));
        CUDA_CHECK(cudaEventCreate(&stopH2D));
        CUDA_CHECK(cudaEventCreate(&startD2H));
        CUDA_CHECK(cudaEventCreate(&stopD2H));

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
        CUDA_CHECK(cudaDeviceSynchronize());

        blockDim = dim3(MAX_THREADS_PER_BLOCK);
        gridDim = dim3((no_of_nodes + blockDim.x - 1) / blockDim.x);
    }

    void destroy() {
        CUDA_CHECK(cudaEventDestroy(startEv));
        CUDA_CHECK(cudaEventDestroy(stopEv));
        CUDA_CHECK(cudaEventDestroy(startH2D));
        CUDA_CHECK(cudaEventDestroy(stopH2D));
        CUDA_CHECK(cudaEventDestroy(startD2H));
        CUDA_CHECK(cudaEventDestroy(stopD2H));
        CUDA_CHECK(cudaStreamDestroy(stream));

        CUDA_CHECK(cudaFree(d_nodes));
        CUDA_CHECK(cudaFree(d_edges));
        CUDA_CHECK(cudaFree(d_mask));
        CUDA_CHECK(cudaFree(d_updating_mask));
        CUDA_CHECK(cudaFree(d_visited));
        CUDA_CHECK(cudaFree(d_cost));
        CUDA_CHECK(cudaFree(d_over));
    }

    void task(int id) {
        auto task_start = Clock::now();

        float total_kernel_ms = 0.0f;
        float total_h2d_ms = 0.0f;
        float total_d2h_ms = 0.0f;

        CUDA_CHECK(cudaEventRecord(startH2D, stream));
        reset_graph_state();
        CUDA_CHECK(cudaEventRecord(stopH2D, stream));
        CUDA_CHECK(cudaEventSynchronize(stopH2D));

        float h2d_ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&h2d_ms, startH2D, stopH2D));
        total_h2d_ms += h2d_ms;

        for (int j = 0; j < jobs; ++j) {
            CUDA_CHECK(cudaEventRecord(startEv, stream));

            bool over = false;
            int iter_guard = 0;
            const int max_iters = 4096;   // 防止异常情况下死转太久

            do {
                over = false;

                CUDA_CHECK(cudaMemcpyAsync(d_over, &over, sizeof(bool),
                                        cudaMemcpyHostToDevice, stream));

                BFS_Kernel<<<gridDim, blockDim, 0, stream>>>(
                    d_nodes, d_edges, d_mask, d_updating_mask, d_visited, d_cost, no_of_nodes);
                CUDA_CHECK(cudaGetLastError());

                BFS_Kernel2<<<gridDim, blockDim, 0, stream>>>(
                    d_mask, d_updating_mask, d_visited, d_over, no_of_nodes);
                CUDA_CHECK(cudaGetLastError());

                CUDA_CHECK(cudaMemcpyAsync(&over, d_over, sizeof(bool),
                                        cudaMemcpyDeviceToHost, stream));

                CUDA_CHECK(cudaStreamSynchronize(stream));

                iter_guard++;
                if (iter_guard > max_iters) {
                    std::cerr << "Warning: BFS iter_guard exceeded max_iters=" << max_iters
                            << ", forcing termination.\n";
                    break;
                }
            } while (over);

            CUDA_CHECK(cudaEventRecord(stopEv, stream));
            CUDA_CHECK(cudaEventSynchronize(stopEv));

            float kernel_ms = 0.0f;
            CUDA_CHECK(cudaEventElapsedTime(&kernel_ms, startEv, stopEv));
            total_kernel_ms += kernel_ms;
        }

        // BFS 这里没有必须的结果回传大数组，统一保留 d2h 统计项为 0
        total_d2h_ms = 0.0f;

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
    }

  private:
    void reset_graph_state() {
        std::vector<unsigned char> h_mask(no_of_nodes, 0);
        std::vector<unsigned char> h_updating_mask(no_of_nodes, 0);
        std::vector<unsigned char> h_visited(no_of_nodes, 0);
        std::vector<int> h_cost(no_of_nodes, 0);

        h_mask[0] = 1;
        h_visited[0] = 1;

        CUDA_CHECK(cudaMemcpyAsync(d_mask, h_mask.data(), sizeof(unsigned char) * no_of_nodes,
                                   cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(d_updating_mask, h_updating_mask.data(),
                                   sizeof(unsigned char) * no_of_nodes,
                                   cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(d_visited, h_visited.data(),
                                   sizeof(unsigned char) * no_of_nodes,
                                   cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(d_cost, h_cost.data(), sizeof(int) * no_of_nodes,
                                   cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }

  private:
    int no_of_nodes;
    int edges_per_node;
    int no_of_edges;

    std::vector<Node> host_nodes;
    std::vector<int> host_edges;

    cudaStream_t stream{};
    cudaEvent_t startEv{}, stopEv{};
    cudaEvent_t startH2D{}, stopH2D{};
    cudaEvent_t startD2H{}, stopD2H{};
    dim3 gridDim{}, blockDim{};

    Node *d_nodes{};
    int *d_edges{};
    unsigned char *d_mask{}, *d_updating_mask{}, *d_visited{};
    int *d_cost{};
    bool *d_over{};
};

static std::unique_ptr<BFSBenchmark> g_bench;

void task_wrapper(int id) {
    g_bench->task(id);
}

void periodicTaskLauncher(float duration, float period) {
    int taskCounter = 0;
    auto start_time = program_start_time;
    std::future<void> asyncTask;

    while (Clock::now() - start_time < std::chrono::duration<float>(duration)) {
        asyncTask = std::async(std::launch::async, [=]() {
            task_wrapper(taskCounter);
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
    out << "  \"workload\": \"bfs_releaseguard\",\n";
    out << "  \"num_nodes\": " << NO_OF_NODES << ",\n";
    out << "  \"edges_per_node\": " << EDGES_PER_NODE << ",\n";
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
                  << " <num_nodes> <period_s> <duration_s> [jobs] [out_json]\n";
        return 1;
    }

    NO_OF_NODES = std::atoi(argv[1]);
    period_s = std::atof(argv[2]);
    duration_s = std::atof(argv[3]);
    if (argc >= 5) jobs = std::atoi(argv[4]);

    std::string out_json = "logs/bfs_releaseguard.json";
    if (argc >= 6) out_json = argv[5];

    std::filesystem::create_directories("logs");
    responsetime_log.open("logs/rt1.txt", std::ios::out);
    summary_log.open("logs/summary_bfs.txt", std::ios::out);

    std::cout << "Num nodes   : " << NO_OF_NODES << "\n";
    std::cout << "Period (s)  : " << period_s << "\n";
    std::cout << "Duration (s): " << duration_s << "\n";
    std::cout << "Jobs        : " << jobs << "\n";
    std::cout << "Output JSON : " << out_json << "\n";

    g_bench = std::make_unique<BFSBenchmark>(NO_OF_NODES, EDGES_PER_NODE);
    g_bench->init();

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

    g_bench->destroy();
    g_bench.reset();
    return 0;
}