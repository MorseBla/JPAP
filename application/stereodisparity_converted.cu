#include <algorithm>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <cuda_runtime.h>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <sys/ipc.h>
#include <sys/shm.h>
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

#include <future>
#include <thread>

#include "../scheduler/shared_header.hpp"

#define SIGNAL_TYPE SIGHUP

bool keepRunning = true;
volatile sig_atomic_t gotSIGHUP = 0;

// Stereodisparity Variables
#define BLOCK_X 32
#define BLOCK_Y 8
#define RAD 8
#define STEPS 3

//const int w = 640 * 8;
//const int h = 480 * 8;
const int w = 2304;
const int h = 1728;
const int minDisp = -16;
const int maxDisp = 0;
const size_t num = size_t(w) * h;
const size_t memSize = num * sizeof(uint32_t);

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

// --- Helper Functions and Kernel ---
static inline int iDivUp(int a, int b) { return (a + b - 1) / b; }

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

// --- Boilerplate Real-Time Functions ---
//static inline timespec ts_add_ns(timespec t, long long ns) {
//    t.tv_sec += ns / 1000000000LL;
//    t.tv_nsec += ns % 1000000000LL;
//    if (t.tv_nsec >= 1000000000L) {
//        t.tv_sec++;
//        t.tv_nsec -= 1000000000L;
//    }
//    return t;
//}

//static inline long long sec_to_ns(double s) { return (long long)(s * 1e9); }

//static inline bool ts_ge(const timespec &a, const timespec &b) {
//    if (a.tv_sec != b.tv_sec)
//        return a.tv_sec > b.tv_sec;
//    return a.tv_nsec >= b.tv_nsec;
//}

//static inline void sleep_until_monotonic_abs(const timespec &abs_ts) {
//    clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, &abs_ts, nullptr);
//}

//static inline double ts_diff_ms(const timespec &end, const timespec &start) {
//    long long sec = (long long)end.tv_sec - (long long)start.tv_sec;
//    long long nsec = (long long)end.tv_nsec - (long long)start.tv_nsec;
//    return (double)sec * 1000.0 + (double)nsec / 1e6;
//}

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

#ifdef DEBUG  
    std::cout << "[task " << task_id << "] pinned core=" << core << "\n";
#endif
}

//static void try_mlockall() {
//    if (mlockall(MCL_CURRENT | MCL_FUTURE) != 0) {
//        std::cerr << "mlockall failed: " << strerror(errno)
//                  << " (may need higher memlock ulimit). Continuing without mlockall.\n";
//    } else {
//#ifdef DEBUG  
//        std::cout << "mlockall enabled\n";
//#endif
//    }
//}

static void try_enable_realtime_fifo(int prio) {
//    sched_param sp{};
//    sp.sched_priority = prio;
//
//    if (sched_setscheduler(0, SCHED_FIFO, &sp) == -1) {
//        std::cerr << "sched_setscheduler(SCHED_FIFO) failed: " << strerror(errno)
//                  << " (need CAP_SYS_NICE or root). Continuing with normal scheduling.\n";
//    } else {
//#ifdef DEBUG  
//        std::cout << "SCHED_FIFO enabled, priority=" << prio << "\n";
//#endif
//    }
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

    //auto fmt = [](int v) {
    //    int major = v / 1000;
    //    int minor = (v % 1000) / 10;
    //    int patch = v % 10;
    //    std::string s = std::to_string(major) + "." + std::to_string(minor);
    //    if (patch != 0)
    //        s += "." + std::to_string(patch);
    //    return s;
    //};

#ifdef DEBUG  
    std::cout << "CUDA driver API version: " << drv << " (" << fmt(drv) << ")\n";
    std::cout << "CUDA runtime version   : " << rt << " (" << fmt(rt) << ")\n";
#endif

    if (drv < rt) {
        std::cerr
            << "ERROR: NVIDIA driver is older than CUDA runtime. "
            << "This causes: 'CUDA driver version is insufficient for CUDA runtime version'.\n";
        std::exit(1);
    }
}

//static inline timespec to_timespec(std::chrono::steady_clock::time_point tp) {
//    using namespace std::chrono;
//    auto ns = duration_cast<nanoseconds>(tp.time_since_epoch()).count();
//    timespec ts{};
//    ts.tv_sec = static_cast<time_t>(ns / 1000000000LL);
//    ts.tv_nsec = static_cast<long>(ns % 1000000000LL);
//    return ts;
//}

//static inline void sleep_until_steady(std::chrono::steady_clock::time_point tp) {
//    timespec ts = to_timespec(tp);
//    clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, &ts, nullptr);
//}

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

    uint32_t *d_left = nullptr;
    uint32_t *d_right = nullptr;
    uint32_t *d_out = nullptr;
    cudaTextureObject_t texLeft = 0;
    cudaTextureObject_t texRight = 0;

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
        CUDA_CHECK(cudaEventRecord(startEv, stream));

        stereoDisparityKernel<<<gridDim, blockDim, 0, stream>>>(d_left, d_right, d_out, w, h,
                                                                minDisp, maxDisp, texLeft,
                                                                texRight);

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
#ifdef DEBUG  
    std::cout << "In service loop\n";
#endif
    print_cuda_versions_or_exit();
    pin_this_thread(task, 1);
    //try_mlockall();
    try_enable_realtime_fifo(80);

    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    CUDA_CHECK(cudaEventCreate(&startEv));
    CUDA_CHECK(cudaEventCreate(&stopEv));

    // --- Host data generation ---
    std::vector<uint32_t> h_left(num), h_right(num), h_out(num, 0);
    const int shift = 8;
    srand(0);
    for (int y = 0; y < h; ++y) {
        int base = y * w;
        for (int x = 0; x < w; ++x) {
            unsigned char r = (unsigned char)((x * 31 + y * 17 + rand()) & 0xFF);
            unsigned char g = (unsigned char)((x * 13 + y * 29 + rand()) & 0xFF);
            unsigned char b = (unsigned char)((x * 7 + y * 23 + rand()) & 0xFF);
            unsigned char a = 255;
            uint32_t px =
                (uint32_t(a) << 24) | (uint32_t(b) << 16) | (uint32_t(g) << 8) | uint32_t(r);
            h_left[base + x] = px;
        }
        for (int x = 0; x < w; ++x) {
            int xs = x + shift;
            if (xs >= w)
                xs = w - 1;
            h_right[base + x] = h_left[base + xs];
        }
    }

    // --- Device memory & Textures ---
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

    dim3 blockDim(BLOCK_X, BLOCK_Y);
    dim3 gridDim(iDivUp(w, blockDim.x), iDivUp(h, blockDim.y));

    // --- Handshake & System Setup ---
    gotSIGHUP = 0;
    std::signal(SIGHUP, handleSIGHUP);

    if (pids > 0) {
        kill(pids, SIGNAL_TYPE);
        while (!gotSIGHUP)
            pause();
    }
#ifdef DEBUG  
    std::cout << "Received Handshake\n";
#endif

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

    CUDA_CHECK(cudaDestroyTextureObject(texLeft));
    CUDA_CHECK(cudaDestroyTextureObject(texRight));
    CUDA_CHECK(cudaEventDestroy(startEv));
    CUDA_CHECK(cudaEventDestroy(stopEv));
    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaFree(d_left));
    CUDA_CHECK(cudaFree(d_right));
    CUDA_CHECK(cudaFree(d_out));
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

#ifdef DEBUG  
    std::cout << "Successfully attached to existing shared memory.\n";
#endif
    service svc(task, setpoint, period, termination, path, pid);
    svc.runService();
    return 0;
}
