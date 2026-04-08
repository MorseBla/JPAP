#include "hip/hip_runtime.h"
#include <algorithm>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <hip/hip_runtime.h>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <sys/ipc.h>
#include <sys/shm.h>
#include <time.h>
#include <unistd.h>

#include <errno.h>
#include <pthread.h>
#include <sched.h>
#include <sys/resource.h>

#include <string.h>   // strerror
#include <sys/mman.h> // mlockall
#include <fcntl.h>
#include <sys/file.h>

#include <future>
#include <thread>

#include "../Scheduler/shared_header.hpp"

#define SIGNAL_TYPE SIGHUP

bool keepRunning = true;
volatile sig_atomic_t gotSIGHUP = 0;

const unsigned int num_elements = 1 << 24;
const unsigned int num_bins = 4096;

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
        hipError_t _e = (call);                                                                   \
        if (_e != hipSuccess) {                                                                   \
            std::cerr << "CUDA error: " << hipGetErrorString(_e) << " at " << __FILE__ << ":"    \
                      << __LINE__ << "\n";                                                         \
            std::exit(1);                                                                          \
        }                                                                                          \
    } while (0)

// kernel
__global__ void histogram_kernel(unsigned int *input, unsigned int *bins, unsigned int num_elements,
                                 unsigned int num_bins) {
    for (unsigned int i = 0; i < 10; i++) {
        unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
        int stride = blockDim.x * gridDim.x;
        for (unsigned int j = tid; j < num_elements; j += stride) {
            unsigned int position = input[j];
            if (position < num_bins)
                atomicAdd(&(bins[position]), 1);
        }
    }
}

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

static void append_locked(const std::string& path, const std::string& line)
{
    int fd = open(path.c_str(), O_WRONLY | O_CREAT | O_APPEND, 0644);
    if (fd < 0) return;

    flock(fd, LOCK_EX);
    write(fd, line.data(), line.size());
    write(fd, "\n", 1);
    flock(fd, LOCK_UN);

    close(fd);
}

static void print_cuda_versions_or_exit() {
    int drv = 0, rt = 0;
    hipError_t e1 = hipDriverGetVersion(&drv);
    hipError_t e2 = hipRuntimeGetVersion(&rt);

    if (e1 != hipSuccess || e2 != hipSuccess) {
        std::cerr << "Failed to query CUDA versions. driverGetVersion=" << hipGetErrorString(e1)
                  << " runtimeGetVersion=" << hipGetErrorString(e2) << "\n";
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
        std::cerr
            << "Fix: upgrade NVIDIA driver (for CUDA 11.6 you typically need 510.39.01+ on Linux), "
            << "or run/build against an older CUDA runtime that matches the installed driver.\n";
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
    float period;      // seconds
    float termination; // seconds
    std::string path;
    int pids;

    hipStream_t stream{};
    hipEvent_t startEv{}, stopEv{};
    bool eventsReady = false;

    unsigned int *d_input{};
    unsigned int *d_bins{};
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
        CUDA_CHECK(hipMemsetAsync(d_bins, 0, num_bins * sizeof(unsigned int), stream));
        CUDA_CHECK(hipEventRecord(startEv, stream));
        histogram_kernel<<<gridDim, blockDim, 0, stream>>>(d_input, d_bins, num_elements, num_bins);
        CUDA_CHECK(hipEventRecord(stopEv, stream));
        while (hipEventQuery(stopEv) == hipErrorNotReady) {
            asm volatile("pause" ::: "memory");
        }

        float kernel_ms = 0.0f;
        CUDA_CHECK(hipEventElapsedTime(&kernel_ms, startEv, stopEv));
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

    // pin_this_thread(task, 1);
    // try_mlockall();
    // try_enable_realtime_fifo(80);

    CUDA_CHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
    CUDA_CHECK(hipEventCreate(&startEv));
    CUDA_CHECK(hipEventCreate(&stopEv));

    // host
    std::vector<unsigned int> h_input(num_elements);
    std::vector<unsigned int> h_bins(num_bins, 0);
    for (unsigned int i = 0; i < num_elements; ++i)
        h_input[i] = rand() % num_bins;

    // device allocations
    CUDA_CHECK(hipMalloc(&d_input, num_elements * sizeof(unsigned int)));
    CUDA_CHECK(hipMalloc(&d_bins, num_bins * sizeof(unsigned int)));
    CUDA_CHECK(hipMemcpy(d_input, h_input.data(), num_elements * sizeof(unsigned int),
                          hipMemcpyHostToDevice));
    CUDA_CHECK(hipMemset(d_bins, 0, num_bins * sizeof(unsigned int)));

    // block and grid size for kernel call
    dim3 blockDim(256);
    dim3 gridDim((num_elements + blockDim.x - 1) / blockDim.x);

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

    if (execFile.is_open())
        execFile.close();
    if (respFile.is_open())
        respFile.close();

    CUDA_CHECK(hipEventDestroy(startEv));
    CUDA_CHECK(hipEventDestroy(stopEv));
    CUDA_CHECK(hipStreamDestroy(stream));
    CUDA_CHECK(hipFree(d_input));
    CUDA_CHECK(hipFree(d_bins));
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