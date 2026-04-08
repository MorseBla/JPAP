#include <algorithm>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <cuda_runtime.h>
#include <filesystem>
#include <fstream>
#include <future>
#include <iostream>
#include <string>
#include <sys/ipc.h>
#include <sys/shm.h>
#include <thread>
#include <time.h>
#include <unistd.h>
#include <vector>

#include <errno.h>
#include <pthread.h>
#include <sched.h>
#include <sys/resource.h>

#include <string.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <sys/file.h>

#include "../Scheduler/shared_header.hpp"

#define SIGNAL_TYPE SIGHUP

bool keepRunning = true;
volatile sig_atomic_t gotSIGHUP = 0;

const int no_of_nodes = 1 << 24;
const int edges_per_node = 8;
const int no_of_edges = no_of_nodes * edges_per_node;
const int iterations = 30;

struct Node {
    int starting;
    int no_of_edges;
};

std::vector<Node> h_nodes(no_of_nodes);
std::vector<int> h_edges(no_of_edges);

static void handleSIGHUP(int sig) {
    if (sig == SIGHUP) gotSIGHUP = 1;
}
static void handleSIGINT(int sig) {
    if (sig == SIGINT) keepRunning = false;
}

#define CUDA_CHECK(call)                                                                           \
    do {                                                                                           \
        cudaError_t _e = (call);                                                                   \
        if (_e != cudaSuccess) {                                                                   \
            std::cerr << "CUDA error: " << cudaGetErrorString(_e) << " at " << __FILE__ << ":"    \
                      << __LINE__ << "\n";                                                         \
            std::exit(1);                                                                          \
        }                                                                                          \
    } while (0)

#define MAX_THREADS_PER_BLOCK 256

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

static int pick_core_for_task(int task_id, int base_core = 1) {
    int ncpu = (int)sysconf(_SC_NPROCESSORS_ONLN);
    if (ncpu <= 0) return 0;
    if (base_core >= ncpu) base_core = 0;
    return (base_core + task_id) % ncpu;
}

static void append_locked(const std::string &path, const std::string &line) {
    int fd = open(path.c_str(), O_WRONLY | O_CREAT | O_APPEND, 0644);
    if (fd < 0) return;

    flock(fd, LOCK_EX);
    write(fd, line.data(), line.size());
    write(fd, "\n", 1);
    flock(fd, LOCK_UN);

    close(fd);
}

static void pin_this_thread(int task_id, int base_core = 1) {
    int core = pick_core_for_task(task_id, base_core);

    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(core, &set);
    if (pthread_setaffinity_np(pthread_self(), sizeof(set), &set) != 0) {
        perror("pthread_setaffinity_np");
    }

    errno = 0;
    (void)setpriority(PRIO_PROCESS, 0, -5);

    std::cout << "[task " << task_id << "] pinned core=" << core << "\n";
}

static void try_mlockall() {
    if (mlockall(MCL_CURRENT | MCL_FUTURE) != 0) {
        std::cerr << "mlockall failed: " << strerror(errno)
                  << " (may need higher memlock ulimit). Continuing without mlockall.\n";
    } else {
        std::cout << "mlockall enabled\n";
    }
}

static void try_enable_realtime_fifo(int prio) {
    sched_param sp{};
    sp.sched_priority = prio;

    if (sched_setscheduler(0, SCHED_FIFO, &sp) == -1) {
        std::cerr << "sched_setscheduler(SCHED_FIFO) failed: " << strerror(errno)
                  << " (need CAP_SYS_NICE or root). Continuing with normal scheduling.\n";
    } else {
        std::cout << "SCHED_FIFO enabled, priority=" << prio << "\n";
    }
}

static void print_cuda_versions_or_exit() {
    int drv = 0, rt = 0;
    cudaError_t e1 = cudaDriverGetVersion(&drv);
    cudaError_t e2 = cudaRuntimeGetVersion(&rt);

    if (e1 != cudaSuccess || e2 != cudaSuccess) {
        std::cerr << "Failed to query CUDA versions. driverGetVersion=" << cudaGetErrorString(e1)
                  << " runtimeGetVersion=" << cudaGetErrorString(e2) << "\n";
        std::exit(1);
    }

    auto fmt = [](int v) {
        int major = v / 1000;
        int minor = (v % 1000) / 10;
        int patch = v % 10;
        std::string s = std::to_string(major) + "." + std::to_string(minor);
        if (patch != 0) s += "." + std::to_string(patch);
        return s;
    };

    std::cout << "CUDA driver API version: " << drv << " (" << fmt(drv) << ")\n";
    std::cout << "CUDA runtime version   : " << rt << " (" << fmt(rt) << ")\n";

    if (drv < rt) {
        std::cerr << "ERROR: NVIDIA driver is older than CUDA runtime.\n";
        std::cerr << "Fix: upgrade NVIDIA driver or use older CUDA runtime.\n";
        std::exit(1);
    }
}

class service {
  public:
    service(int task_, float setpoint_, float period_, float termination_, const std::string &path_,
            int pidsa)
        : task(task_), setpoint(setpoint_), period(period_), termination(termination_), path(path_),
          pids(pidsa) {}

    void runService();

  private:
    float kernellaunch(dim3 gridDim, dim3 blockDim, double carry_ms);

    int task;
    float setpoint;
    float period;
    float termination;
    std::string path;
    int pids;

    cudaStream_t stream{};
    cudaEvent_t startEv{}, stopEv{};

    Node *d_nodes{};
    int *d_edges{};
    unsigned char *d_mask{}, *d_updating_mask{}, *d_visited{};
    int *d_cost{};
    bool *d_over{};

    int jobs = 2;

    static constexpr size_t BUF_SZ = 1 << 20;
    char execBuf[BUF_SZ]{};
    char respBuf[BUF_SZ]{};
    std::string exec_log_path;
    std::string resp_log_path;
};

float service::kernellaunch(dim3 gridDim, dim3 blockDim, double carry_ms) {
    float sum_kernel_ms = 0.0f;

    for (int i = 0; i < jobs; ++i) {
        CUDA_CHECK(cudaEventRecord(startEv, stream));

        for (int iter = 0; iter < iterations; ++iter) {
            bool over = false;
            CUDA_CHECK(cudaMemcpyAsync(d_over, &over, sizeof(bool), cudaMemcpyHostToDevice, stream));

            BFS_Kernel<<<gridDim, blockDim, 0, stream>>>(d_nodes, d_edges, d_mask, d_updating_mask,
                                                         d_visited, d_cost, no_of_nodes);

            BFS_Kernel2<<<gridDim, blockDim, 0, stream>>>(d_mask, d_updating_mask, d_visited, d_over,
                                                          no_of_nodes);
        }

        CUDA_CHECK(cudaEventRecord(stopEv, stream));
        while (cudaEventQuery(stopEv) == cudaErrorNotReady) {
            asm volatile("pause" ::: "memory");
        }

        float kernel_ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&kernel_ms, startEv, stopEv));
        sum_kernel_ms += kernel_ms;
    }

    float exec_ms = sum_kernel_ms;
    float rt_ms = exec_ms + static_cast<float>(carry_ms);

    if (sharedData) {
        sharedData->executiontime[task] = exec_ms;
        sharedData->responsetime[task] = rt_ms;
    }

    char b1[64];
    snprintf(b1, sizeof(b1), "%.6f", exec_ms);
    append_locked(exec_log_path, b1);

    char b2[64];
    snprintf(b2, sizeof(b2), "%.6f", rt_ms);
    append_locked(resp_log_path, b2);

    return rt_ms;
}

void service::runService() {
    std::cout << "In service loop\n";
    print_cuda_versions_or_exit();
    pin_this_thread(task, 1);
    try_mlockall();
    try_enable_realtime_fifo(80);

    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    CUDA_CHECK(cudaEventCreate(&startEv));
    CUDA_CHECK(cudaEventCreate(&stopEv));

    srand((unsigned)time(nullptr));

    for (int i = 0; i < no_of_nodes; ++i) {
        h_nodes[i].starting = i * edges_per_node;
        h_nodes[i].no_of_edges = edges_per_node;
        for (int j = 0; j < edges_per_node; ++j) {
            h_edges[i * edges_per_node + j] = rand() % no_of_nodes;
        }
    }

    std::vector<unsigned char> h_mask(no_of_nodes, 0);
    std::vector<unsigned char> h_updating_mask(no_of_nodes, 0);
    std::vector<unsigned char> h_visited(no_of_nodes, 0);
    std::vector<int> h_cost(no_of_nodes, 0);

    h_mask[0] = 1;
    h_visited[0] = 1;

    CUDA_CHECK(cudaMalloc(&d_nodes, sizeof(Node) * no_of_nodes));
    CUDA_CHECK(cudaMalloc(&d_edges, sizeof(int) * no_of_edges));
    CUDA_CHECK(cudaMalloc(&d_mask, sizeof(unsigned char) * no_of_nodes));
    CUDA_CHECK(cudaMalloc(&d_updating_mask, sizeof(unsigned char) * no_of_nodes));
    CUDA_CHECK(cudaMalloc(&d_visited, sizeof(unsigned char) * no_of_nodes));
    CUDA_CHECK(cudaMalloc(&d_cost, sizeof(int) * no_of_nodes));
    CUDA_CHECK(cudaMalloc(&d_over, sizeof(bool)));

    CUDA_CHECK(cudaMemcpy(d_nodes, h_nodes.data(), sizeof(Node) * no_of_nodes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_edges, h_edges.data(), sizeof(int) * no_of_edges, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_mask, h_mask.data(), sizeof(unsigned char) * no_of_nodes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_updating_mask, h_updating_mask.data(), sizeof(unsigned char) * no_of_nodes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_visited, h_visited.data(), sizeof(unsigned char) * no_of_nodes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_cost, h_cost.data(), sizeof(int) * no_of_nodes, cudaMemcpyHostToDevice));

    dim3 blockDim(256);
    dim3 gridDim((no_of_nodes + blockDim.x - 1) / blockDim.x);

    gotSIGHUP = 0;
    std::signal(SIGHUP, handleSIGHUP);

    if (pids > 0) {
        kill(pids, SIGNAL_TYPE);
        while (!gotSIGHUP) pause();
    }
    std::cout << "Received Handshake\n";

    if (sharedData) sharedData->newperiods[task] = period;

    std::filesystem::create_directories(std::filesystem::path(path));
    exec_log_path = (std::filesystem::path(path) /
                     ("taskexecutiontime" + std::to_string(task) + ".txt")).string();
    resp_log_path = (std::filesystem::path(path) /
                     ("taskresponsetime" + std::to_string(task) + ".txt")).string();

    std::ofstream execFile(exec_log_path, std::ios::app);
    std::ofstream respFile(resp_log_path, std::ios::app);
    if (execFile.is_open()) execFile.rdbuf()->pubsetbuf(execBuf, sizeof(execBuf));
    if (respFile.is_open()) respFile.rdbuf()->pubsetbuf(respBuf, sizeof(respBuf));

    std::future<float> asyncJob;
    double prev_rt_ms = 0.0;
    bool first_job = true;

    timespec start_ts{};
    clock_gettime(CLOCK_MONOTONIC, &start_ts);

    while (keepRunning) {
        timespec now_ts{};
        clock_gettime(CLOCK_MONOTONIC, &now_ts);
        double elapsed_sec =
            (now_ts.tv_sec - start_ts.tv_sec) + (now_ts.tv_nsec - start_ts.tv_nsec) / 1e9;
        if (elapsed_sec > termination) break;

        double p = sharedData ? (double)sharedData->newperiods[task] : (double)period;
        if (p < 1e-6) p = (double)period;

        if (asyncJob.valid()) {
            prev_rt_ms = asyncJob.get();
        }

        double carry_ms = 0.0;
        if (!first_job) {
            carry_ms = std::max(0.0, prev_rt_ms - p * 1000.0);
        }

        asyncJob = std::async(std::launch::async, [&, carry_ms]() -> float {
            return kernellaunch(gridDim, blockDim, carry_ms);
        });

        first_job = false;

        std::this_thread::sleep_for(std::chrono::duration<double>(p));
    }

    if (asyncJob.valid()) {
        prev_rt_ms = asyncJob.get();
    }

    if (execFile.is_open()) execFile.close();
    if (respFile.is_open()) respFile.close();

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

int main(int argc, char *argv[]) {
    if (argc < 7) {
        std::cerr << "Usage: " << argv[0]
                  << " <task:int> <period:float> <setpoint:float> <termination:float> "
                     "<path:string> <pid:int>\n";
        return 1;
    }

    int task = atoi(argv[1]);
    float period = atof(argv[2]);
    float setpoint = atof(argv[3]);
    float termination = atof(argv[4]);
    std::string path = argv[5];
    int pid = atoi(argv[6]);

    std::signal(SIGINT, handleSIGINT);

    key_t key = ftok("shmfile", 65);
    int shmid = shmget(key, sizeof(SharedData), 0666);
    if (shmid < 0) {
        perror("shmget failed - segment does not exist. Did you start the server?");
        exit(1);
    }
    sharedData = (SharedData *)shmat(shmid, nullptr, 0);
    if (sharedData == (void *)-1) {
        perror("shmat failed");
        exit(1);
    }

    std::cout << "Successfully attached to existing shared memory.\n";
    service svc(task, setpoint, period, termination, path, pid);
    svc.runService();
    return 0;
}