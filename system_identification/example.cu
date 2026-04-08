#include <iostream>
#include <thread>
#include <chrono>
#include <vector>
#include <mutex>
#include <cstdlib>
#include <fstream>
#include <future>
#include <condition_variable>
#include <csignal>
#include <cuda_runtime.h>

using namespace std;

// Global variables
float period;
//int size;
size_t dataSize;
std::mutex mtx;
std::vector<std::pair<std::chrono::time_point<std::chrono::steady_clock>, std::chrono::time_point<std::chrono::steady_clock>>> timings;
std::vector<float> responseTimes;
std::vector<float> dh;
std::vector<float> hd;
std::vector<float> ke;

std::chrono::time_point<std::chrono::steady_clock> program_start_time;
std::chrono::time_point<std::chrono::steady_clock> preemptionlaunch;
std::ofstream responsetime("logs/rt1.txt");

std::condition_variable cv;
bool processed = false;
int pids;
float *d_A, *d_B, *d_C;
float *A, *B, *C;
int jobs = 2;

void init() {
    FILE *file = fopen("logs/proc2.txt", "r");
    if (file) {
        fscanf(file, "%d", &pids);
        fclose(file);
    }
    cout << "PID PARENT T2= " << pids << endl;
}

void handleSignal(int signal) {
    if (signal == SIGHUP) {
        processed = true;
    }
}

float calculateAverage(const std::vector<float>& values) {
    if (values.empty()) {
        return 0.0f;
    }
    float sum = 0.0f;
    for (float value : values) {
        sum += value;
    }
    return sum / values.size();
}

__global__ void matrixMul(float *A, float *B, float *C, int width) {
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * blockDim.y + ty;
    int col = blockIdx.x * blockDim.x + tx;

    float sum = 0.0f;
    for (int i = 0; i < width; ++i) {
        sum += A[row * width + i] * B[i * width + col];
    }

    C[row * width + col] = sum;
}

void task(int id, int width, float* d_A, float* d_B, float* d_C) {
    std::ofstream preemptiontime("logs/preemptiont1.txt", std::ios::app);
    auto start = std::chrono::steady_clock::now();
    auto preemption_start_duration = std::chrono::duration_cast<std::chrono::milliseconds>(start - preemptionlaunch).count();
    preemptiontime << preemption_start_duration << "\n";

    // CUDA Events for timing
    cudaEvent_t startEvent, stopEvent, startHD, stopHD, startDH, stopDH;
    cudaEventCreate(&startEvent);
    cudaEventCreate(&stopEvent);

    vector<float> hd_times, dh_times;
    float milliseconds = 0, hd_ms = 0, dh_ms = 0;
    dim3 dimBlock(32, 32);
    dim3 dimGrid((width + dimBlock.x - 1) / dimBlock.x, (width + dimBlock.y - 1) / dimBlock.y);

    for (int i = 0; i < jobs; i++) {
 

        cudaEventRecord(startEvent);
        matrixMul<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, width);
        cudaDeviceSynchronize();
        cudaEventRecord(stopEvent);
        cudaEventSynchronize(stopEvent);
        cudaEventElapsedTime(&milliseconds, startEvent, stopEvent);
        responsetime << "Kernel: " << milliseconds << " ms \n";

    }

    float average_ke = milliseconds;

    ke.push_back(average_ke);
    responseTimes.push_back(average_ke);
    cudaEventDestroy(startEvent);
    cudaEventDestroy(stopEvent);
    auto finish = std::chrono::steady_clock::now();
    timings.push_back({start, finish});
    preemptiontime.close();
}

void periodicTaskLauncher(float duration, float period, int width, float* d_A, float* d_B, float* d_C, float range) {
    int taskCounter = 0;
    auto start_time = program_start_time;
    auto controlperiod = std::chrono::high_resolution_clock::now();
    std::ofstream slack1("logs/s1.txt");
    std::ofstream period1("logs/p1.txt");
    std::ofstream rtj1("logs/rtj1.txt");
    std::future<void> asyncTask;

    while (std::chrono::steady_clock::now() - start_time < std::chrono::duration<float>(duration)) {
        auto now = std::chrono::steady_clock::now();
        auto end22 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> controlperiodaa = end22 - controlperiod;

        preemptionlaunch = std::chrono::steady_clock::now();

        asyncTask = std::async(std::launch::async, [&]() {
            task(taskCounter, width, d_A, d_B, d_C);
        });

        taskCounter++;
        std::chrono::milliseconds sleep_duration(static_cast<long long>(period * 1000.0));
        std::this_thread::sleep_for(sleep_duration);
    }
    if (asyncTask.valid()) {
        asyncTask.get(); // Wait for the last task to finish
    }
    slack1.close();
    period1.close();
}

int main(int argc, char* argv[]) {
    if (argc < 4) {
        cerr << "Usage: " << argv[0] << " <width> <range> <period> <duration>" << endl;
        return 1;
    }

    int width = atoi(argv[1]);
    period = atof(argv[2]);
    float duration = atof(argv[3]);

    program_start_time = chrono::steady_clock::now();

    A = (float*)malloc(width * width * sizeof(float));
    B = (float*)malloc(width * width * sizeof(float));
    C = (float*)malloc(width * width * sizeof(float));

    for (int i = 0; i < width * width; ++i) {
        A[i] = static_cast<float>(rand()) / RAND_MAX;
        B[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    dataSize = width * width * sizeof(float);
    cudaMalloc((void **)&d_A, dataSize);
    cudaMalloc((void **)&d_B, dataSize);
    cudaMalloc((void **)&d_C, dataSize);

    thread launcher(periodicTaskLauncher, duration, period, width, d_A, d_B, d_C, 0.0); // Note: 'range' is unused in your current setup
    launcher.join(); // Wait for the launcher to complete

    {
        lock_guard<mutex> lock(mtx);
        for (const auto& timing : timings) {
            auto start = chrono::duration_cast<chrono::seconds>(timing.first - program_start_time).count();
            auto finish = chrono::duration_cast<chrono::seconds>(timing.second - program_start_time).count();
        }
    }

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    float sx = calculateAverage(responseTimes);
    float ke1 = calculateAverage(ke);

    cout << "Response time = " << sx << " Period = " << period <<
    " KE = " << ke1 << endl;

    free(A);
    free(B);
    free(C);

    return 0;
}