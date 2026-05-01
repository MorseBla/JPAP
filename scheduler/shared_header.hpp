#pragma once
using namespace std;
#include <iostream>
#include <cstdlib>
#include <vector>
#include <ctime>
#include <cmath>
#include <signal.h>
#include <sys/wait.h>
#include <sstream>
#include <sys/file.h>
#include <unistd.h>
#include <fcntl.h>
#include <chrono>
#include <algorithm>   
#include <cmath>       
#include<vector>
#include <csignal>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <cstdlib> 
#include <semaphore.h>
#include <unistd.h>
#include<csignal>
#include <filesystem>
#include <string.h>
#include <float.h>
#include <sys/shm.h>
#include <sys/stat.h>
#include <fstream>
#include <thread>
#include <cassert>
#include <iomanip>
#include <cassert>
#include <future>
#include <queue>
#include <unordered_map>
#include <iomanip>
#include <vector>
#include <deque>

#include <errno.h>
#include <cstring>
#include <string>
#include <stdexcept>


#define SIGUSR3 30
#define SIGUSR4 31
#define old_range 5
extern std::atomic<float> current_freq;
extern float innerloopduration;
extern float outerloopduration;
extern std::atomic<bool> outerlooprunning;
extern std::atomic<bool> controllerlooping;
extern std::atomic<bool> monitorthread;
extern std::atomic<bool> stopsigma;
extern std::unordered_map<std::string, int> mappingTable;
std::vector<float>interpowervalues;
std::vector<float>interfreqvalues;
std::vector<float>finalpowervalues;
std::vector<float>finalfrequencyvalues;
std::vector<float>averagertrr;
inline std::mutex mtx_task;

float**gain;
struct logtasks {
    std::mutex mtx;
    std::vector<float> exec;
    std::vector<float> response;
    std::vector<float> period;
    std::vector<float> higherperiodbound;
    std::vector<float> lowerperiodbound;
    std::vector<float> rtr;
    std::vector<float> deadlinemiss;
    std::vector<float> outerexec;
    std::vector<float> outerresponse;
    std::vector<float> outerperiod;    
    std::vector<float> outerrtr;
    std::vector<float> filewritexec;
    std::vector<float> filewriteresponse;
    std::vector<float> filewriteperiod;
    std::vector<float> filewritehighperiodbound;
    std::vector<float> filewritelowperiodbound;
    std::vector<float> filewritertr;
    std::vector<float> filewritedeadlinemiss;
    std::vector<float> tasksetpoint;
    void add_exec(float v) { std::lock_guard<std::mutex> lk(mtx); exec.push_back(v); }
    void add_rt(float v) { std::lock_guard<std::mutex> lk(mtx); response.push_back(v); }
    void add_period(float v) { std::lock_guard<std::mutex> lk(mtx); period.push_back(v); }
    void add_rtr(float v) { std::lock_guard<std::mutex> lk(mtx); rtr.push_back(v); }
    void add_deadline(float v) { std::lock_guard<std::mutex> lk(mtx); deadlinemiss.push_back(v); }
    void add_higherperiodbound(float v) { std::lock_guard<std::mutex> lk(mtx);higherperiodbound.push_back(v); }
    void add_lowerperiodbound(float v) { std::lock_guard<std::mutex> lk(mtx);lowerperiodbound.push_back(v); }
    void outeradd_exec(float v) { std::lock_guard<std::mutex> lk(mtx); outerexec.push_back(v); }
    void outeradd_rt(float v) { std::lock_guard<std::mutex> lk(mtx); outerresponse.push_back(v); }
    void outeradd_period(float v) { std::lock_guard<std::mutex> lk(mtx); outerperiod.push_back(v); }
    void outeradd_rtr(float v) { std::lock_guard<std::mutex> lk(mtx); outerrtr.push_back(v); }
    void add_tasksetpoint(float v) { std::lock_guard<std::mutex> lk(mtx); tasksetpoint.push_back(v); }    
};

struct logpower{
    std::mutex mtx;
    std::vector<float>powersetpoints;
    std::vector<float>powercontrolperiod;
    std::vector<float>powerseterror;
    std::vector<float>measuredpower;
    void add_measuredpower(float v) { std::lock_guard<std::mutex> lk(mtx); measuredpower.push_back(v); }
    void add_powersetpoint(float v) { std::lock_guard<std::mutex> lk(mtx); powersetpoints.push_back(v); }
    void add_powercontrolperiod(float v) { std::lock_guard<std::mutex> lk(mtx); powercontrolperiod.push_back(v); }
    void add_powerseterror(float v) { std::lock_guard<std::mutex> lk(mtx); powerseterror.push_back(v); }
};
struct logging_sys{
    logpower power;
    explicit logging_sys(int solution) : solution_id(solution) {};
    std::unordered_map<int, std::string> solution = {
            {1, "FC_GPU"},
            {2, "N+1"},
            {3, "LQR"},
            {4, "OpenLoop"},
            {5, "Adhoc"},
            {6, "SISO"},
            {7, "DFS"},
        };
    int solution_id;
    std::string solution_name() const
    {
        auto it = solution.find(solution_id);
        if (it != solution.end()) return it->second;
        return "UNKNOWN";
    }
 
    void dump_files(const std::string& path)
    {
        std::string sol = solution_name();
        std::filesystem::create_directories(path);
        std::ofstream fpowersetpoint(path + "/" + sol + "_" +"_powersetpoint.txt", std::ios::app);
        std::ofstream fpowercontrolperiod(path + "/" + sol +"_" + "_powercontrolperiod.txt", std::ios::app);
        std::ofstream fpowerseterror(path + "/" + sol + "_" + "_powerseterror.txt", std::ios::app);

        for (float v : power.powersetpoints) fpowersetpoint << v << "\n";
        for (float v : power.powercontrolperiod) fpowercontrolperiod << v << "\n";
        for (float v : power.powerseterror) fpowerseterror << v << "\n";

        }
};



struct logging {
    std::vector<logtasks> tasks;
    explicit logging(int num_tasks,int solution)
        : tasks(static_cast<size_t>(num_tasks)), solution_id(solution) {}
    logtasks& operator[](size_t i)             { return tasks.at(i); }
    const logtasks& operator[](size_t i) const { return tasks.at(i); }
    std::unordered_map<int, std::string> solution = {
            {1, "FC_GPU"},
            {2, "N+1"},
            {3, "LQR"},
            {4, "OpenLoop"},
            {5, "Adhoc"},
            {6, "SISO"},
            {7, "DFS"},
        };
    int solution_id;

    std::string solution_name() const
    {
        auto it = solution.find(solution_id);
        if (it != solution.end()) return it->second;
        return "UNKNOWN";
    }
    void dump_files(const std::string& path)
    {
        std::string sol = solution_name();
        std::filesystem::create_directories(path);

        for (size_t i = 0; i < tasks.size(); i++) {
            std::string ti = "t" + std::to_string(i + 1);

            std::ofstream fexec(path + "/" + sol + "_" + ti + "_exec.txt", std::ios::app);
            std::ofstream fresp(path + "/" + sol + "_" + ti + "_response.txt", std::ios::app);
            std::ofstream fper(path + "/" + sol + "_" + ti + "_period.txt", std::ios::app);
            std::ofstream fperbound(path + "/" + sol + "_" + ti + "_higherperiodbound.txt", std::ios::app);
            std::ofstream fperbound1(path + "/" + sol + "_" + ti + "_lowerperiodbound.txt", std::ios::app);
            std::ofstream frtr(path + "/" + sol + "_" + ti + "_rtr.txt", std::ios::app);
            std::ofstream fdeadline(path + "/" + sol + "_" + ti + "_deadline.txt", std::ios::app);
            for (float v : tasks[i].filewritexec) fexec << v << "\n";
            for (float v : tasks[i].filewriteresponse) fresp << v << "\n";
            for (float v : tasks[i].filewriteperiod) fper << v << "\n";
            for (float v : tasks[i].filewritehighperiodbound)fperbound << v << "\n";
            for (float v : tasks[i].filewritelowperiodbound)fperbound1 << v << "\n";
            for (float v : tasks[i].filewritertr) frtr << v << "\n";
            for (float v : tasks[i].filewritedeadlinemiss) fdeadline << v << "\n";

        }
    }
    void dump_power(const std::string& path, const std::vector<float>& powervals)
    {
        std::string sol = solution_name();
        std::filesystem::create_directories(path);
        std::ofstream fp(path + "/" + sol + "_finalpowervalues.txt", std::ios::app);
        for (float p : powervals) fp << p << "\n";
    
    
    
    
    
    }
     void dump_power_setpoint(const std::string& path, const std::vector<float>& powervals)
    {
        std::string sol = solution_name();
        std::filesystem::create_directories(path);
        std::ofstream fp(path + "/" + sol + "_finalpower_setpoint.txt", std::ios::app);
        for (float p : powervals) fp << p << "\n";
    }
    
    void dump_freq(const std::string& path, const std::vector<float>& freqvals)
    {
        std::string sol = solution_name();
        std::filesystem::create_directories(path);
        std::ofstream fp(path + "/" + sol + "_finalfreqvalsvalues.txt", std::ios::app);
        for (float p : freqvals) fp << p << "\n";
    }
    
    void dump_rtr(const std::string& path, const std::vector<float>& averagertrr)
    {
        std::string sol = solution_name();
        std::filesystem::create_directories(path);
        std::ofstream fp(path + "/" + sol + "averagertr.txt", std::ios::app);
        for (float p : averagertrr) fp << p << "\n";
    }
    
    
    
};


struct taskConfig {
    float pid = 0.0f;
    float initial_rate = 0.0f;
    float setpoint = 0.0f;
};
struct tasks {
    std::vector<taskConfig> task;
    explicit tasks(int num_tasks) 
        : task(static_cast<size_t>(num_tasks)) {}

    taskConfig& operator[](size_t i)             { return task.at(i); }
    const taskConfig& operator[](size_t i) const { return task.at(i); }
};

struct average {
    
 float calculateAverage(const std::vector<float>& values) {
        if (values.empty()) {
            return 0.0;
        }
        float sum = 0.0;
        for (float value : values) {
            sum += value;
        }
        return sum / values.size();
    }
  
// };
    
//     float calculateAverage(const std::vector<float>& values) {
//     if (values.empty()) return 0.0f;

//     // ---- Compute mean of current batch (same as before) ----
//     float sum_now = 0.0f;
//     for (float v : values) sum_now += v;
//     float mean_now = sum_now / values.size();

//     // ---- Moving average over time (window) ----
//     static std::deque<float> window;
//     static float running_sum = 0.0f;
//     const size_t window_size = 5;   // smoothness (3–7 good)

//     window.push_back(mean_now);
//     running_sum += mean_now;

//     if (window.size() > window_size) {
//         running_sum -= window.front();
//         window.pop_front();
//     }

//     return running_sum / window.size();
// }
};



struct SharedData {
    float responsetime[4]; 
    float newperiods[4]; 
    float executiontime[4]; 
};
SharedData* sharedData;












#pragma once

#include <fcntl.h>
#include <semaphore.h>
#include <errno.h>
#include <cstring>
#include <string>

class GpuSemaphore {
public:
    GpuSemaphore() {
        sem_ = sem_open("/gpu_global_lock", O_CREAT, 0666, 1);
        if (sem_ == SEM_FAILED) {
            throw std::runtime_error(
                std::string("sem_open failed: ") + std::strerror(errno));
        }
    }

    void lock() {
        while (sem_wait(sem_) == -1) {
            if (errno != EINTR) {
                throw std::runtime_error(
                    std::string("sem_wait failed: ") + std::strerror(errno));
            }
        }
    }

    void unlock() {
        if (sem_post(sem_) == -1) {
            throw std::runtime_error(
                std::string("sem_post failed: ") + std::strerror(errno));
        }
    }

    ~GpuSemaphore() {
        if (sem_ != SEM_FAILED) {
            sem_close(sem_);
        }
    }

private:
    sem_t* sem_;
};

class GpuLockGuard {
public:
    explicit GpuLockGuard(GpuSemaphore& s) : sem_(s) { sem_.lock(); }
    ~GpuLockGuard() { sem_.unlock(); }

private:
    GpuSemaphore& sem_;
};

inline GpuSemaphore gpu_sem;
