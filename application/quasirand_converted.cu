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

#include <string.h>   // strerror
#include <sys/mman.h> // mlockall
#include <fcntl.h>
#include <sys/file.h>

#include "../Scheduler/shared_header.hpp"

#define SIGNAL_TYPE SIGHUP
//e
bool keepRunning = true;
volatile sig_atomic_t gotSIGHUP = 0;

// --- Quasirand Variables ---
#define BLOCK_SIZE 256
const unsigned int numSims = 16777216; 
//const unsigned int numSims = 1 << 28;
const unsigned int threadsPerBlock = BLOCK_SIZE;
const unsigned int blocksPerGrid = 256;

static void handleSIGHUP(int sig) {
    if (sig == SIGHUP)
        gotSIGHUP = 1;
}

static void handleSIGINT(int sig) {
    if (sig == SIGINT)
        keepRunning = false;
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

// intra-block reduction
__device__ unsigned int blockReduceSum(unsigned int val) {
    extern __shared__ unsigned int smem[];
    unsigned int tid = threadIdx.x;
    smem[tid] = val;
    __syncthreads();
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s)
            smem[tid] += smem[tid + s];
        __syncthreads();
    }
    return smem[0];
}

template <typename Real>
__global__ void computeValue(unsigned int *results, const Real *points, unsigned int numSims) {
    unsigned int bid = blockIdx.x;
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int step = gridDim.x * blockDim.x;

    const Real *pointx = points + tid;
    const Real *pointy = pointx + numSims; // second half of array holds Y coords

    unsigned int pointsInside = 0;
    for (unsigned int i = tid; i < numSims; i += step, pointx += step, pointy += step) {
        Real x = *pointx;
        Real y = *pointy;
        Real l2 = x * x + y * y;
        if (l2 < static_cast<Real>(1.0))
            pointsInside++;
    }

    unsigned int blockSum = blockReduceSum(pointsInside);
    if (threadIdx.x == 0)
        results[bid] = blockSum;
}

// --- Boilerplate Real-Time Functions ---
static inline timespec ts_add_ns(timespec t, long long ns) {
    t.tv_sec += ns / 1000000000LL;
    t.tv_nsec += ns % 1000000000LL;
    if (t.tv_nsec >= 1000000000L) {
        t.tv_sec++;
        t.tv_nsec -= 1000000000L;
    }
    return t;
}

static inline long long sec_to_ns(double s) { return (long long)(s * 1e9); }

static inline bool ts_ge(const timespec &a, const timespec &b) {
    if (a.tv_sec != b.tv_sec)
        return a.tv_sec > b.tv_sec;
    return a.tv_nsec >= b.tv_nsec;
}

static inline void sleep_until_monotonic_abs(const timespec &abs_ts) {
    clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, &abs_ts, nullptr);
}

static inline double ts_diff_ms(const timespec &end, const timespec &start) {
    long long sec = (long long)end.tv_sec - (long long)start.tv_sec;
    long long nsec = (long long)end.tv_nsec - (long long)start.tv_nsec;
    return (double)sec * 1000.0 + (double)nsec / 1e6;
}

static int pick_core_for_task(int task_id, int base_core = 1) {
    int ncpu = (int)sysconf(_SC_NPROCESSORS_ONLN);
    if (ncpu <= 0)
        return 0;
    if (base_core >= ncpu)
        base_core = 0;
    return (base_core + task_id) % ncpu;
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

static void append_locked(const std::string &path, const std::string &line) {
    int fd = open(path.c_str(), O_WRONLY | O_CREAT | O_APPEND, 0644);
    if (fd < 0)
        return;

    flock(fd, LOCK_EX);
    write(fd, line.data(), line.size());
    write(fd, "\n", 1);
    flock(fd, LOCK_UN);

    close(fd);
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
        if (patch != 0)
            s += "." + std::to_string(patch);
        return s;
    };

    std::cout << "CUDA driver API version: " << drv << " (" << fmt(drv) << ")\n";
    std::cout << "CUDA runtime version   : " << rt << " (" << fmt(rt) << ")\n";

    if (drv < rt) {
        std::cerr
            << "ERROR: NVIDIA driver is older than CUDA runtime. "
            << "This causes: 'CUDA driver version is insufficient for CUDA runtime version'.\n";
        std::exit(1);
    }
}

static inline timespec to_timespec(std::chrono::steady_clock::time_point tp) {
    using namespace std::chrono;
    auto ns = duration_cast<nanoseconds>(tp.time_since_epoch()).count();
    timespec ts{};
    ts.tv_sec = static_cast<time_t>(ns / 1000000000LL);
    ts.tv_nsec = static_cast<long>(ns % 1000000000LL);
    return ts;
}

static inline void sleep_until_steady(std::chrono::steady_clock::time_point tp) {
    timespec ts = to_timespec(tp);
    clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, &ts, nullptr);
}

// --- Service Class ---
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
    bool eventsReady = false;

    float *d_points = nullptr;
    unsigned int *d_results = nullptr;

    int jobs = 2;

    static constexpr size_t BUF_SZ = 1 << 20;
    char execBuf[BUF_SZ]{};
    char respBuf[BUF_SZ]{};
    std::string exec_log_path;
    std::string resp_log_path;
};

float service::kernellaunch(dim3 gridDim, dim3 blockDim, double carry_ms) {
    float sum_kernel_ms = 0.0f;
    GpuLockGuard lock(gpu_sem); 
    for (int i = 0; i < jobs; ++i) {
        CUDA_CHECK(cudaMemsetAsync(d_results, 0, blocksPerGrid * sizeof(unsigned int), stream));

        CUDA_CHECK(cudaEventRecord(startEv, stream));

        computeValue<float><<<gridDim, blockDim, blockDim.x * sizeof(unsigned int), stream>>>(
            d_results, d_points, numSims);

        CUDA_CHECK(cudaEventRecord(stopEv, stream));
        while (cudaEventQuery(stopEv) == cudaErrorNotReady) {
            #ifdef Jetson 
                asm volatile("yield" ::: "memory");
            #else
                asm volatile("pause" ::: "memory");
            #endif 
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

    // --- Host data generation ---
    std::vector<float> h_points(2 * numSims);
    srand(0);
    for (unsigned int i = 0; i < 2 * numSims; ++i)
        h_points[i] = static_cast<float>(rand()) / RAND_MAX;

    // --- Device memory allocations ---
    CUDA_CHECK(cudaMalloc(&d_points, 2 * numSims * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_results, blocksPerGrid * sizeof(unsigned int)));

    CUDA_CHECK(cudaMemcpy(d_points, h_points.data(), 2 * numSims * sizeof(float),
                          cudaMemcpyHostToDevice));

    dim3 blockDim(threadsPerBlock);
    dim3 gridDim(blocksPerGrid);

    // --- Handshake & System Setup ---
    gotSIGHUP = 0;
    std::signal(SIGHUP, handleSIGHUP);

    if (pids > 0) {
        kill(pids, SIGNAL_TYPE);
        while (!gotSIGHUP)
            pause();
    }
    std::cout << "Received Handshake\n";

    if (sharedData) {
        sharedData->newperiods[task] = period;
    }

    std::filesystem::create_directories(std::filesystem::path(path));

    exec_log_path =
        (std::filesystem::path(path) / ("taskexecutiontime" + std::to_string(task) + ".txt"))
            .string();
    resp_log_path =
        (std::filesystem::path(path) / ("taskresponsetime" + std::to_string(task) + ".txt"))
            .string();

    std::ofstream execFile(exec_log_path, std::ios::app);
    std::ofstream respFile(resp_log_path, std::ios::app);
    if (execFile.is_open())
        execFile.rdbuf()->pubsetbuf(execBuf, sizeof(execBuf));
    if (respFile.is_open())
        respFile.rdbuf()->pubsetbuf(respBuf, sizeof(respBuf));

    std::future<float> asyncJob;
    double prev_rt_ms = 0.0;
    bool first_job = true;

    timespec start_ts{};
    clock_gettime(CLOCK_MONOTONIC, &start_ts);

    // --- Periodic Loop ---
    while (keepRunning) {
        timespec now_ts{};
        clock_gettime(CLOCK_MONOTONIC, &now_ts);
        double elapsed_sec =
            (now_ts.tv_sec - start_ts.tv_sec) + (now_ts.tv_nsec - start_ts.tv_nsec) / 1e9;

        if (elapsed_sec > termination)
            break;

        double p = sharedData ? (double)sharedData->newperiods[task] : (double)period;
        if (p < 1e-6)
            p = (double)period;

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

    if (asyncJob.valid())
        prev_rt_ms = asyncJob.get();

    // --- Cleanup ---
    if (execFile.is_open())
        execFile.close();
    if (respFile.is_open())
        respFile.close();

    CUDA_CHECK(cudaEventDestroy(startEv));
    CUDA_CHECK(cudaEventDestroy(stopEv));
    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaFree(d_points));
    CUDA_CHECK(cudaFree(d_results));
}

// --- Main Standard Interface ---
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

    usleep(1000);
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
