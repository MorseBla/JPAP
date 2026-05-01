#include "shared_header.hpp"
#include <rocm_smi/rocm_smi.h>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <string>
#include <sys/file.h>
#include <sys/ipc.h>
#include <sys/mman.h>
#include <sys/shm.h>
#include <thread>
#include <unistd.h>
#include <unordered_map>
#include <vector>

std::atomic<float> current_freq{1235.0f};
float innerloopduration = 4.0f;
float outerloopduration = 5* innerloopduration;

std::atomic<bool> outerlooprunning{true};
std::atomic<bool> controllerlooping{true};
std::atomic<bool> monitorthread{true};
std::atomic<bool> stopsigma{false};

#ifdef CLAMP_PERIODS
#ifndef BOUND
#define BOUND 40
#endif
static const float BOUND_VALUE = BOUND / 100.0f;
#endif

class rocm_interface {
public:
    rocm_interface();
    ~rocm_interface();

    double get_power();
    double get_freq();
    int get_current_level();
    void set_freq(int target_level);
    void print_map();

    int nearest_level(float target_mhz) const;
    int floor_level(float target_mhz) const;
    int ceil_level(float target_mhz) const;
    double level_freq(int level) const;
    double min_freq() const;
    double max_freq() const;

private:
    uint32_t dev_idx{};
    std::unordered_map<int, double> sclk_map;
};

static std::unique_ptr<rocm_interface> rocm;

rocm_interface::rocm_interface() {
    dev_idx = 0;

    rsmi_status_t st = rsmi_init(0);
    if (st != RSMI_STATUS_SUCCESS) {
        std::cerr << "ROCm SMI init failed" << std::endl;
        std::exit(1);
    }

    rsmi_frequencies_t freq_info{};
    st = rsmi_dev_gpu_clk_freq_get(dev_idx, RSMI_CLK_TYPE_SYS, &freq_info);
    if (st != RSMI_STATUS_SUCCESS) {
        std::cerr << "Failed to read supported SCLK levels" << std::endl;
        std::exit(1);
    }

    for (uint32_t i = 0; i < freq_info.num_supported; ++i) {
        double mhz = static_cast<double>(freq_info.frequency[i]) / 1000000.0;
        sclk_map[static_cast<int>(i)] = mhz;
    }

    st = rsmi_dev_perf_level_set(dev_idx, RSMI_DEV_PERF_LEVEL_MANUAL);
    if (st != RSMI_STATUS_SUCCESS) {
        std::cerr << "Warning: failed to set MANUAL perf level" << std::endl;
    }

    std::cout << "Initial actual SCLK = " << get_freq() << " MHz" << std::endl;
}

rocm_interface::~rocm_interface() {
    rsmi_shut_down();
}

void rocm_interface::print_map() {
    std::cout << "\n--- SCLK Level to Frequency Map ---" << std::endl;
    for (const auto &kv : sclk_map) {
        std::cout << "Level " << std::setw(2) << kv.first
                  << ": " << kv.second << " MHz";
        if (kv.first == get_current_level()) std::cout << " <-- CURRENT";
        std::cout << std::endl;
    }
    std::cout << "-----------------------------------\n" << std::endl;
}

double rocm_interface::get_power() {
    uint64_t power_uw = 0;
    rsmi_status_t st = rsmi_dev_power_ave_get(dev_idx, 0, &power_uw);
    if (st == RSMI_STATUS_SUCCESS) {
        return power_uw / 1000000.0;
    }
    return 0.0;
}

double rocm_interface::get_freq() {
    rsmi_frequencies_t freq{};
    rsmi_status_t st = rsmi_dev_gpu_clk_freq_get(dev_idx, RSMI_CLK_TYPE_SYS, &freq);
    if (st == RSMI_STATUS_SUCCESS) {
        return freq.frequency[freq.current] / 1000000.0;
    }
    return 0.0;
}

int rocm_interface::get_current_level() {
    rsmi_frequencies_t freq{};
    rsmi_status_t st = rsmi_dev_gpu_clk_freq_get(dev_idx, RSMI_CLK_TYPE_SYS, &freq);
    if (st == RSMI_STATUS_SUCCESS) {
        return static_cast<int>(freq.current);
    }
    return -1;
}

void rocm_interface::set_freq(int target_level) {
    if (sclk_map.find(target_level) == sclk_map.end()) {
        std::cerr << "Invalid target SCLK level: " << target_level << std::endl;
        return;
    }

    rsmi_status_t st = rsmi_dev_perf_level_set(dev_idx, RSMI_DEV_PERF_LEVEL_MANUAL);
    // if (st != RSMI_STATUS_SUCCESS) {
    //     std::cerr << "set_freq: failed to set MANUAL perf level" << std::endl;
    //     return;
    // }

    uint64_t bitmask = (1ULL << target_level);
    st = rsmi_dev_gpu_clk_freq_set(dev_idx, RSMI_CLK_TYPE_SYS, bitmask);
    if (st != RSMI_STATUS_SUCCESS) {
        std::cerr << "set_freq: failed to set SCLK level " << target_level
                  << " (" << sclk_map[target_level] << " MHz). "
                  << "Try running with sudo if needed." << std::endl;
        return;
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(20));

    int actual_level = get_current_level();
    double actual_freq = get_freq();

    // std::cout << "[SCLK] requested level=" << target_level
    //           << " requested_freq=" << sclk_map[target_level] << " MHz"
    //           << " actual_level=" << actual_level
    //           << " actual_freq=" << actual_freq << " MHz" << std::endl;
}

double rocm_interface::level_freq(int level) const {
    auto it = sclk_map.find(level);
    if (it == sclk_map.end()) return 0.0;
    return it->second;
}

double rocm_interface::min_freq() const {
    double v = std::numeric_limits<double>::max();
    for (const auto &kv : sclk_map) v = std::min(v, kv.second);
    return (v == std::numeric_limits<double>::max()) ? 0.0 : v;
}

double rocm_interface::max_freq() const {
    double v = 0.0;
    for (const auto &kv : sclk_map) v = std::max(v, kv.second);
    return v;
}

int rocm_interface::floor_level(float target_mhz) const {
    int best_level = -1;
    double best_freq = -1.0;
    for (const auto &kv : sclk_map) {
        if (kv.second <= target_mhz && kv.second > best_freq) {
            best_freq = kv.second;
            best_level = kv.first;
        }
    }
    return best_level;
}

int rocm_interface::ceil_level(float target_mhz) const {
    int best_level = -1;
    double best_freq = std::numeric_limits<double>::max();
    for (const auto &kv : sclk_map) {
        if (kv.second >= target_mhz && kv.second < best_freq) {
            best_freq = kv.second;
            best_level = kv.first;
        }
    }
    return best_level;
}

int rocm_interface::nearest_level(float target_mhz) const {
    int best_level = -1;
    double best_diff = std::numeric_limits<double>::max();
    for (const auto &kv : sclk_map) {
        double diff = std::fabs(kv.second - target_mhz);
        if (diff < best_diff) {
            best_diff = diff;
            best_level = kv.first;
        }
    }
    return best_level;
}

static void append_lock(const std::string &path, const std::string &line) {
    int fd = open(path.c_str(), O_WRONLY | O_CREAT | O_APPEND, 0644);
    if (fd < 0) return;

    flock(fd, LOCK_EX);
    write(fd, line.data(), line.size());
    write(fd, "\n", 1);
    flock(fd, LOCK_UN);
    close(fd);
}

float calculate_deadline_miss(const std::vector<float> &local_resp,
                              const std::vector<float> &local_per,
                              float eps = 0.0f) {
    if (local_resp.empty() || local_per.empty())
        return 0.0f;

    size_t total = std::min(local_resp.size(), local_per.size());
    int miss_count = 0;

    for (size_t i = 0; i < total; ++i) {
        float period = local_per[i];
        if (period <= 0.0f)
            continue;

        float rtr = local_resp[i] / period;
        if (rtr > (1.0f + eps))
            miss_count++;
    }

    return (static_cast<float>(miss_count) / static_cast<float>(total)) * 100.0f;
}

void monitorpower() {
    average avg;
    interpowervalues.reserve(2048);
    using clock_t = std::chrono::steady_clock;

    while (monitorthread.load(std::memory_order_relaxed)) {
        interpowervalues.clear();
        auto windowStart = clock_t::now();

        while (true) {
            auto now = clock_t::now();
            std::chrono::duration<double> elapsed = now - windowStart;
            if (elapsed.count() >= innerloopduration)
                break;

            float power = rocm ? static_cast<float>(rocm->get_power()) : 0.0f;
            interpowervalues.push_back(power);
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }

        float finpower = avg.calculateAverage(interpowervalues);
        finalpowervalues.push_back(finpower);
    }
}

void monitorfreq() {
    average avg;
    interfreqvalues.reserve(2048);
    using clock_t = std::chrono::steady_clock;

    while (monitorthread.load(std::memory_order_relaxed)) {
        interfreqvalues.clear();
        auto windowStart = clock_t::now();

        while (true) {
            auto now = clock_t::now();
            std::chrono::duration<double> elapsed = now - windowStart;
            if (elapsed.count() >= innerloopduration)
                break;

            float freq = rocm ? static_cast<float>(rocm->get_freq()) : 0.0f;
            interfreqvalues.push_back(freq);
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }

        float finfreq = avg.calculateAverage(interfreqvalues);
        finalfrequencyvalues.push_back(finfreq);
    }
}

void sigmadelta_freq_step(float /*unused*/) {
    if (!rocm) return;

    static float acc = 0.0f;
    static int lastLower = -1;
    static int lastUpper = -1;

    while (!stopsigma.load(std::memory_order_relaxed)) {
        float target = current_freq.load(std::memory_order_relaxed);

        float clamped = std::clamp(
            target,
            static_cast<float>(rocm->min_freq()),
            static_cast<float>(rocm->max_freq())
        );

        int lower = rocm->floor_level(clamped);
        int upper = rocm->ceil_level(clamped);

        if (lower < 0 && upper < 0) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            continue;
        }
        if (lower < 0) lower = upper;
        if (upper < 0) upper = lower;

        if (lower == upper) {
            rocm->set_freq(lower);
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
            continue;
        }

        double lowerFreq = rocm->level_freq(lower);
        double upperFreq = rocm->level_freq(upper);

        if (lower != lastLower || upper != lastUpper) {
            acc = 0.0f;
            lastLower = lower;
            lastUpper = upper;
        }

        float delta = static_cast<float>((clamped - lowerFreq) / (upperFreq - lowerFreq));
        acc += delta;

        int selectedLevel;
        if (acc >= 1.0f) {
            selectedLevel = upper;
            acc -= 1.0f;
        } else {
            selectedLevel = lower;
        }

        rocm->set_freq(selectedLevel);
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
}

void outerloop(logging &log, tasks &t, int numtasks, std::atomic<bool> &outerlooprunning) {
    std::cout << "[outerloop] running with " << numtasks << " tasks\n";
    float kp = 250.0f;
    std::thread freqthread;

    float *error = new float[numtasks];
    float *rtr = new float[numtasks];
    float *executiontime = new float[numtasks];
    float *responsetime = new float[numtasks];
    float *taskperiod = new float[numtasks];
    float *newtaskperiod = new float[numtasks];

    average average;
    float outererror = 0.0f;
    float averagertr = 0.0f;
    float averagesetpoint = 0.0f;

    auto start = std::chrono::steady_clock::now();

    while (outerlooprunning.load(std::memory_order_relaxed)) {
        std::this_thread::sleep_for(std::chrono::duration<float>(outerloopduration));

        stopsigma.store(true, std::memory_order_relaxed);
        if (freqthread.joinable()) {
            freqthread.join();
        }

        auto end = std::chrono::steady_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::seconds>(end - start);
        std::cout << "Current time =\t" << duration.count() << "\n";

        averagertr = 0.0f;

        for (int i = 0; i < numtasks; ++i) {
            std::vector<float> local_rtr, local_resp, local_per, local_exec;
            {
                std::lock_guard<std::mutex> lk(log[i].mtx);
                local_rtr.swap(log[i].outerrtr);
                local_resp.swap(log[i].outerresponse);
                local_per.swap(log[i].outerperiod);
                local_exec.swap(log[i].outerexec);
            }

            rtr[i] = average.calculateAverage(local_rtr);
            responsetime[i] = average.calculateAverage(local_resp);
            taskperiod[i] = average.calculateAverage(local_per);
            executiontime[i] = average.calculateAverage(local_exec);
            newtaskperiod[i] = taskperiod[i];
            error[i] = t[i].setpoint - rtr[i];
            averagertr += rtr[i];

            std::cout << "[Outer] T" << (i + 1)
                      << " exec=" << executiontime[i]
                      << " resp=" << responsetime[i]
                      << " rtr=" << rtr[i]
                      << " err=" << error[i]
                      << " P=" << taskperiod[i]
                      << "\n";
        }

        float e_mag_sum = 0.0f;
        averagesetpoint = 0.0f;

        for (int i = 0; i < numtasks; i++) {
            e_mag_sum += error[i];
            averagesetpoint += t[i].setpoint;
        }

        e_mag_sum /= static_cast<float>(numtasks);
        averagesetpoint /= static_cast<float>(numtasks);
        averagertr /= static_cast<float>(numtasks);
        outererror = averagesetpoint - averagertr;

        float prev_avg = averagertrr.empty() ? averagertr : averagertrr.back();
        if (!averagertrr.empty() && std::fabs(averagertr - prev_avg) > 0.2f) {
            outererror = 0.0f;
        }
        averagertrr.push_back(averagertr);

        std::cout << "Combined Error (L3) = " << e_mag_sum << "\n";

        float delta = -kp * outererror;
        float measured = current_freq.load(std::memory_order_relaxed);
        float next = measured + delta;
        next = std::clamp(next,
                          static_cast<float>(rocm->min_freq()),
                          static_cast<float>(rocm->max_freq()));
        current_freq.store(next, std::memory_order_relaxed);

        std::cout << "Outer rtr = " << averagertr << std::endl;
        std::cout << "Current freq (SCLK) = " << measured << " MHz\n";
        std::cout << "Delta freq          = " << delta << " MHz\n";
        std::cout << "New freq (command)  = " << current_freq.load(std::memory_order_relaxed) << " MHz\n";
        std::cout << "Termination Outerloop iteration\n";

        stopsigma.store(false, std::memory_order_relaxed);
        freqthread = std::thread(sigmadelta_freq_step, next);
    }

    stopsigma.store(true, std::memory_order_relaxed);
    if (freqthread.joinable()) {
        freqthread.join();
    }

    delete[] error;
    delete[] rtr;
    delete[] executiontime;
    delete[] responsetime;
    delete[] taskperiod;
    delete[] newtaskperiod;
}

void dfs(std::string path, logging &log, tasks &t, int numtasks, float **gain,
         std::atomic<bool> &controllerlooping) {
    std::cout << "DFS running with " << numtasks << " tasks\n";
    float kp = 90.0f;
    std::thread freqthread;

    float *error = new float[numtasks];
    float *rtr = new float[numtasks];
    float *executiontime = new float[numtasks];
    float *responsetime = new float[numtasks];
    float *taskperiod = new float[numtasks];
    float *newtaskperiod = new float[numtasks];
    float *deadlinemiss = new float[numtasks];
    float *upperbound = new float[numtasks];
    float *lowerbound = new float[numtasks];

    int controlperiod = 0;
    float outererror = 0.0f;
    float averagertr = 0.0f;
    float averagesetpoint = 0.0f;

    average average;
    auto start = std::chrono::steady_clock::now();
    std::ofstream ctrlLog("logs/controller.txt", std::ios::app);

    std::filesystem::path base_path(path);
    std::vector<std::string> exec_paths(numtasks);
    std::vector<std::string> resp_paths(numtasks);

    for (int i = 0; i < numtasks; ++i) {
        exec_paths[i] = (base_path / ("taskexecutiontime" + std::to_string(i) + ".txt")).string();
        resp_paths[i] = (base_path / ("taskresponsetime" + std::to_string(i) + ".txt")).string();
    }

    while (controllerlooping.load(std::memory_order_relaxed)) {
        std::this_thread::sleep_for(std::chrono::duration<float>(outerloopduration));

        stopsigma.store(true, std::memory_order_relaxed);
        if (freqthread.joinable()) {
            freqthread.join();
        }

        auto end = std::chrono::steady_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::seconds>(end - start);
        std::cout << "Current time =\t" << duration.count() << "\n";
        std::cout << "Control Period = " << controlperiod << "\n";

        averagertr = 0.0f;

        for (int i = 0; i < numtasks; i++) {
            std::vector<float> local_rtr, local_resp, local_per, local_exec;
            {
                std::lock_guard<std::mutex> lk(log[i].mtx);
                local_rtr = std::move(log[i].rtr);
                local_resp = std::move(log[i].response);
                local_per = std::move(log[i].period);
                local_exec = std::move(log[i].exec);
            }

            rtr[i] = average.calculateAverage(local_rtr);
            responsetime[i] = average.calculateAverage(local_resp);
            taskperiod[i] = average.calculateAverage(local_per);
            executiontime[i] = average.calculateAverage(local_exec);
            newtaskperiod[i] = taskperiod[i];

            std::cout << "Size" << i
                      << " samples: rtr=" << local_rtr.size()
                      << " resp=" << local_resp.size()
                      << " period=" << local_per.size()
                      << " exec=" << local_exec.size()
                      << "\n";

            //float deadline = calculate_deadline_miss(local_resp, local_per, 0.05f);
            float deadline = calculate_deadline_miss(local_exec, local_per, 0.00f);

            deadlinemiss[i] = deadline;
            error[i] = t[i].setpoint - rtr[i];
            averagertr += rtr[i];

            append_lock(exec_paths[i], "Control Period");
            append_lock(resp_paths[i], "Control Period");
        }

        float e_mag_sum = 0.0f;
        averagesetpoint = 0.0f;
        for (int i = 0; i < numtasks; i++) {
            e_mag_sum += error[i];
            averagesetpoint += t[i].setpoint;
        }

        e_mag_sum /= static_cast<float>(numtasks);
        averagesetpoint /= static_cast<float>(numtasks);
        averagertr /= static_cast<float>(numtasks);
        outererror = averagesetpoint - averagertr;

        float prev_avg = averagertrr.empty() ? averagertr : averagertrr.back();
        if (!averagertrr.empty() && std::fabs(averagertr - prev_avg) > 0.2f) {
            outererror = 0.0f;
        }
        averagertrr.push_back(averagertr);

        std::cout << "Outer rtr = " << averagertr << std::endl;
        std::cout << "Combined Error (L3) = " << e_mag_sum << "\n";

        float delta = -kp * outererror;
        float measured = current_freq.load(std::memory_order_relaxed);
        float next = measured + delta;
        next = std::clamp(next,
                          static_cast<float>(731),
                          static_cast<float>(rocm->max_freq()));
        current_freq.store(next, std::memory_order_relaxed);

        std::cout << "Current freq (SCLK) = " << measured << " MHz\n";
        std::cout << "Delta freq          = " << delta << " MHz\n";
        std::cout << "New freq (command)  = " << current_freq.load(std::memory_order_relaxed) << " MHz\n";
        std::cout << "Termination Outerloop iteration\n";

        stopsigma.store(false, std::memory_order_relaxed);
        freqthread = std::thread(sigmadelta_freq_step, next);

        double ts = std::chrono::duration<double>(end - start).count();
        std::cout << "time=" << ts << "  control_period=" << controlperiod << "\n";
        ctrlLog << "time=" << ts << "  control_period=" << controlperiod << "\n";

        std::cout << "task\texec(ms)\tresp(ms)\trtr\terror\tPcur(s)\tPnext(ms)\tDeadline miss (%)\n";
        ctrlLog << "task\texec(ms)\tresp(ms)\trtr\terror\tPcur(s)\tPnext(ms)\tDeadline miss (%)\n";

        std::cout << std::fixed << std::setprecision(4);
        ctrlLog << std::fixed << std::setprecision(4);

        for (int i = 0; i < numtasks; i++) {
            float initial = t[i].initial_rate;
            float min_p = initial * 0.90f;
            float max_p = initial * 1.10f;
            upperbound[i] = max_p;
            lowerbound[i] = min_p;

            std::cout << "T" << (i + 1) << "\t"
                      << executiontime[i] << "\t\t"
                      << responsetime[i] << "\t\t"
                      << rtr[i] << "\t"
                      << error[i] << "\t"
                      << taskperiod[i] << "\t"
                      << newtaskperiod[i] << "\t"
                      << deadlinemiss[i] << "\n";

            ctrlLog << "T" << (i + 1) << "\t"
                    << executiontime[i] << "\t"
                    << responsetime[i] << "\t"
                    << rtr[i] << "\t"
                    << error[i] << "\t"
                    << taskperiod[i] << "\t"
                    << newtaskperiod[i] << "\n";

            {
                std::lock_guard<std::mutex> lk(log[i].mtx);
                log[i].filewritexec.push_back(executiontime[i]);
                log[i].filewriteresponse.push_back(responsetime[i]);
                log[i].filewriteperiod.push_back(taskperiod[i]);
                log[i].filewritertr.push_back(rtr[i]);
                log[i].filewritelowperiodbound.push_back(lowerbound[i] * 1000);
                log[i].filewritehighperiodbound.push_back(upperbound[i] * 1000);
                log[i].filewritedeadlinemiss.push_back(deadlinemiss[i]);
            }
        }

        std::cout << "*****************************************************************\n\n";
        ctrlLog << "*****************************************************************\n\n";
        ctrlLog.flush();

        controlperiod = controlperiod + 1;
    }

    stopsigma.store(true, std::memory_order_relaxed);
    log.dump_rtr(path, averagertrr);

    if (freqthread.joinable()) {
        freqthread.join();
    }

    delete[] error;
    delete[] rtr;
    delete[] executiontime;
    delete[] responsetime;
    delete[] taskperiod;
    delete[] newtaskperiod;
    delete[] deadlinemiss;
    delete[] upperbound;
    delete[] lowerbound;
}

void innerloop(std::string path, logging &log, tasks &t, int numtasks, float **gain,
               std::atomic<bool> &controllerlooping) {
    std::cout << "[innerloop] running with " << numtasks << " tasks\n";
    auto start = std::chrono::steady_clock::now();
    int controlperiod = 0;

    float *delta = new float[numtasks];
    float *error = new float[numtasks];
    float *rtr = new float[numtasks];
    float *upperbound = new float[numtasks];
    float *lowerbound = new float[numtasks];
    float *executiontime = new float[numtasks];
    float *responsetime = new float[numtasks];
    float *taskperiod = new float[numtasks];
    float *newtaskperiod = new float[numtasks];
    float *deadlinemiss = new float[numtasks];

    average average;
    std::ofstream ctrlLog("logs/controller.txt", std::ios::app);

    std::filesystem::path base_path(path);
    std::vector<std::string> exec_paths(numtasks);
    std::vector<std::string> resp_paths(numtasks);

    for (int i = 0; i < numtasks; ++i) {
        exec_paths[i] = (base_path / ("taskexecutiontime" + std::to_string(i) + ".txt")).string();
        resp_paths[i] = (base_path / ("taskresponsetime" + std::to_string(i) + ".txt")).string();
    }

    while (controllerlooping.load(std::memory_order_relaxed)) {
        std::this_thread::sleep_for(std::chrono::duration<float>(innerloopduration));
        auto end = std::chrono::steady_clock::now();
        std::cout << "Control Period = " << controlperiod << "\n";

        for (int i = 0; i < numtasks; i++) {
            std::vector<float> local_rtr, local_resp, local_per, local_exec;
            {
                std::lock_guard<std::mutex> lk(log[i].mtx);
                local_rtr = std::move(log[i].rtr);
                local_resp = std::move(log[i].response);
                local_per = std::move(log[i].period);
                local_exec = std::move(log[i].exec);
            }

            rtr[i] = average.calculateAverage(local_rtr);
            responsetime[i] = average.calculateAverage(local_resp);
            taskperiod[i] = average.calculateAverage(local_per);
            executiontime[i] = average.calculateAverage(local_exec);

            std::cout << "Size" << i
                      << " samples: rtr=" << local_rtr.size()
                      << " resp=" << local_resp.size()
                      << " period=" << local_per.size()
                      << " exec=" << local_exec.size()
                      << "\n";

            float deadline = calculate_deadline_miss(local_exec, local_per, 0.00f);
            deadlinemiss[i] = deadline;
            error[i] = t[i].setpoint - rtr[i];

            append_lock(exec_paths[i], "Control Period");
            append_lock(resp_paths[i], "Control Period");
        }

        for (int i = 0; i < numtasks; i++) {
            delta[i] = 0.0f;
            for (int j = 0; j < numtasks; j++) {
                delta[i] += error[j] * -gain[i][j];
            }
            std::cout << "delta[" << i << "]=" << delta[i] << std::endl;
        }

        for (int i = 0; i < numtasks; i++) {
            newtaskperiod[i] = (taskperiod[i] + delta[i]) / 1000.0f;
        }

#ifdef CLAMP_PERIODS
        for (int i = 0; i < numtasks; i++) {
            float initial = t[i].initial_rate;
            float min_p = initial * (1 - BOUND_VALUE);
            float max_p = initial * (1 + BOUND_VALUE);
            float original = newtaskperiod[i];
            float clamped = std::clamp(original, min_p, max_p);
            sharedData->newperiods[i] = clamped;
            upperbound[i] = max_p;
            lowerbound[i] = min_p;

            if (clamped != original) {
                printf("Task %d clamped: original=%f clamped=%f min=%f max=%f\n",
                       i, original, clamped, min_p, max_p);
            }
        }
#else
        for (int i = 0; i < numtasks; i++) {
            sharedData->newperiods[i] = newtaskperiod[i];
        }
#endif

        double ts = std::chrono::duration<double>(end - start).count();
        std::cout << "time=" << ts << "  control_period=" << controlperiod << "\n";
        ctrlLog << "time=" << ts << "  control_period=" << controlperiod << "\n";

        std::cout << "task\texec(ms)\tresp(ms)\trtr\terror\tPcur(s)\tPnext(ms)\tDeadline miss (%)\n";
        ctrlLog << "task\texec(ms)\tresp(ms)\trtr\terror\tPcur(s)\tPnext(ms)\tDeadline miss (%)\n";

        std::cout << std::fixed << std::setprecision(4);
        ctrlLog << std::fixed << std::setprecision(4);

        for (int i = 0; i < numtasks; i++) {
            std::cout << "T" << (i + 1) << "\t"
                      << executiontime[i] << "\t\t"
                      << responsetime[i] << "\t\t"
                      << rtr[i] << "\t"
                      << error[i] << "\t"
                      << taskperiod[i] << "\t"
                      << newtaskperiod[i] * 1000.0f << "\t"
                      << deadlinemiss[i] << "\n";

            ctrlLog << "T" << (i + 1) << "\t"
                    << executiontime[i] << "\t"
                    << responsetime[i] << "\t"
                    << rtr[i] << "\t"
                    << error[i] << "\t"
                    << taskperiod[i] * 1000.0f << "\t"
                    << newtaskperiod[i] << "\n";

            {
                std::lock_guard<std::mutex> lk(log[i].mtx);
                log[i].filewritexec.push_back(executiontime[i]);
                log[i].filewriteresponse.push_back(responsetime[i]);
                log[i].filewriteperiod.push_back(taskperiod[i]);
                log[i].filewritertr.push_back(rtr[i]);
                log[i].filewritelowperiodbound.push_back(lowerbound[i] * 1000.0f);
                log[i].filewritehighperiodbound.push_back(upperbound[i] * 1000.0f);
                log[i].filewritedeadlinemiss.push_back(deadlinemiss[i]);
            }
        }

        std::cout << "*****************************************************************\n\n";
        ctrlLog << "*****************************************************************\n\n";
        ctrlLog.flush();
        controlperiod = controlperiod + 1;
    }

    delete[] delta;
    delete[] error;
    delete[] rtr;
    delete[] lowerbound;
    delete[] upperbound;
    delete[] executiontime;
    delete[] responsetime;
    delete[] taskperiod;
    delete[] newtaskperiod;
    delete[] deadlinemiss;
}

void openloop(logging &log, tasks &t, int numtasks, std::atomic<bool> &controllerlooping) {
    std::cout << "[openloop] running\n";
}

void adhoc(logging &log, tasks &t, int numtasks, std::atomic<bool> &controllerlooping) {
    std::cout << "[adhoc] running\n";
}

void siso(logging &log, tasks &t, int numtasks, std::atomic<bool> &controllerlooping) {
    std::cout << "[siso] running\n";
}

void monitor(logging *log, int numtasks) {
    if (!log || !sharedData) return;
    std::cout << "Begin monitor\n";

    std::vector<float> prevResp(numtasks, std::numeric_limits<float>::quiet_NaN());
    std::vector<float> prevExec(numtasks, std::numeric_limits<float>::quiet_NaN());

    using clock = std::chrono::high_resolution_clock;
    while (monitorthread.load()) {
        auto start = clock::now();
        std::chrono::duration<double> elapsed(0);

        while (elapsed.count() < innerloopduration && outerlooprunning.load()) {
            elapsed = clock::now() - start;

            for (int i = 0; i < numtasks; ++i) {
                float resp = sharedData->executiontime[i];
                if (resp != prevResp[i]) {
                    (*log)[i].add_rt(resp);
                    (*log)[i].outeradd_rt(resp);

                    float period_s = sharedData->newperiods[i];
                    (*log)[i].add_period(period_s * 1000.0f);
                    (*log)[i].outeradd_period(period_s);

                    if (period_s > 0.0f) {
                        (*log)[i].add_rtr(resp / (period_s * 1000.0f));
                        (*log)[i].outeradd_rtr(resp / (period_s * 1000.0f));
                    } else {
                        (*log)[i].add_rtr(0.0f);
                        (*log)[i].outeradd_rtr(0.0f);
                    }

                    prevResp[i] = resp;
                }

                float exec = sharedData->executiontime[i];
                if (exec != prevExec[i]) {
                    (*log)[i].add_exec(exec);
                    (*log)[i].outeradd_exec(exec);
                    prevExec[i] = exec;
                }
            }

            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }
}

volatile sig_atomic_t readyCount = 0;
void signalHandler(int) {
    readyCount++;
}

int main(int argc, char *argv[]) {
    signal(SIGHUP, signalHandler);
    std::cout << "Controller PID: " << getpid() << "\n";

    int numtasks = std::atoi(argv[1]);
    int solution = std::atoi(argv[2]);
    int duration = std::atoi(argv[3]);
    std::string path = argv[4];

    int expected_argc = 5 + numtasks * 2;
    if (argc < expected_argc) {
        std::cerr << "ERROR: not enough arguments.\n"
                  << "Expected argc >= " << expected_argc << " but got " << argc << "\n";
        return 1;
    }

    key_t key = ftok("shmfile", 65);
    bool createdshm = false;

    int shmid = shmget(key, sizeof(SharedData), 0666 | IPC_CREAT | IPC_EXCL);
    if (shmid >= 0) {
        createdshm = true;
    } else {
        shmid = shmget(key, sizeof(SharedData), 0666 | IPC_CREAT);
        if (shmid < 0) {
            perror("shmget failed");
            return 1;
        }
    }

    sharedData = (SharedData *)shmat(shmid, nullptr, 0);
    if (sharedData == (void *)-1) {
        perror("shmat failed");
        return 1;
    }

    rocm = std::make_unique<rocm_interface>();
    rocm->print_map();

    std::cout << "Initial measured SCLK = " << rocm->get_freq() << " MHz\n";
    std::cout << "Nearest level to 1404 MHz = " << rocm->nearest_level(1404.0f) << "\n";

    std::cout << "tasks = " << numtasks << std::endl;

    float **gain = new float *[numtasks];
    for (int i = 0; i < numtasks; ++i) {
        gain[i] = new float[numtasks];
    }

    for (int i = 0; i < numtasks; ++i) {
        for (int j = 0; j < numtasks; ++j) {
            if (i == j) {
                gain[i][j] = 3.75f;
            } else {
                gain[i][j] = 0.3f;
            }
        }
    }

    for (int i = 0; i < numtasks; ++i) {
        for (int j = 0; j < numtasks; ++j) {
            std::cout << gain[i][j] << "\t";
        }
        std::cout << std::endl;
    }

    current_freq.store(1404.0f, std::memory_order_release);
    rocm->set_freq(rocm->nearest_level(current_freq.load(std::memory_order_relaxed)));

    logging log(numtasks, solution);
    tasks task(numtasks);

    int arg_index = 5;
    for (int i = 0; i < numtasks; i++) {
        task[i].initial_rate = std::atof(argv[arg_index++]);
        task[i].setpoint = std::atof(argv[arg_index++]);
    }

    for (int i = 0; i < numtasks; i++) {
        std::cout << "Task " << i
                  << " Rate=" << task[i].initial_rate * 1000
                  << " Setpoint=" << task[i].setpoint
                  << "\n";
    }

    std::unordered_map<int, std::string> mappings = {
        {1, "FC_GPU"},
        {2, "DFS"},
        {3, "Proposed"},
        {4, "OpenLoop"},
        {5, "Adhoc"},
        {6, "SISO"},
    };

    std::cout << "Solution = "
              << (mappings.count(solution) ? mappings[solution] : "UNKNOWN")
              << "\n";

    while (readyCount < numtasks) {
        usleep(10000);
    }

    std::ifstream in("pid.txt");
    if (!in) {
        std::cerr << "Error: cannot open pid.txt\n";
        return 1;
    }

    std::vector<pid_t> pids;
    for (int i = 0; i < numtasks; ++i) {
        pid_t p;
        if (!(in >> p)) {
            std::cerr << "pid.txt has fewer than " << numtasks << " PIDs\n";
            return 1;
        }
        pids.push_back(p);
    }

    for (pid_t p : pids) {
        if (kill(p, SIGHUP) != 0) {
            perror("kill");
        } else {
            std::cout << "Sent SIGHUP to " << p << "\n";
        }
    }

    std::cout << "Beginning\n";

    std::thread monitorThread(monitor, &log, numtasks);
    std::thread monitorpowerthread(monitorpower);
    std::thread monitorfreqthread(monitorfreq);
    std::thread controllerThread;
    std::thread controllerThread2;

    std::this_thread::sleep_for(std::chrono::seconds(10));

    switch (solution) {
    case 1:
        controllerThread = std::thread(innerloop, path, std::ref(log), std::ref(task),
                                       numtasks, gain, std::ref(controllerlooping));
        break;
    case 2:
        outerloopduration = innerloopduration;
        current_freq.store(1400.0f, std::memory_order_release);
        rocm->set_freq(rocm->nearest_level(current_freq.load(std::memory_order_relaxed)));
        controllerThread = std::thread(dfs, path, std::ref(log), std::ref(task),
                                       numtasks, gain, std::ref(controllerlooping));
        break;
    case 3:
        controllerThread = std::thread(innerloop, path, std::ref(log), std::ref(task),
                                       numtasks, gain, std::ref(controllerlooping));
        controllerThread2 = std::thread(outerloop, std::ref(log), std::ref(task),
                                        numtasks, std::ref(outerlooprunning));
        break;
    case 4:
        controllerThread = std::thread(openloop, std::ref(log), std::ref(task),
                                       numtasks, std::ref(controllerlooping));
        break;
    case 5:
        controllerThread = std::thread(adhoc, std::ref(log), std::ref(task),
                                       numtasks, std::ref(controllerlooping));
        break;
    case 6:
        controllerThread = std::thread(siso, std::ref(log), std::ref(task),
                                       numtasks, std::ref(controllerlooping));
        break;
    default:
        std::cerr << "ERROR: unknown solution id " << solution << "\n";
        outerlooprunning.store(false);
        controllerlooping.store(false);
        monitorthread.store(false);
        if (monitorThread.joinable()) monitorThread.join();
        return 1;
    }

    auto start = std::chrono::steady_clock::now();
    while (true) {
        auto now = std::chrono::steady_clock::now();
        std::chrono::duration<double> elapsed = now - start;
        if (elapsed.count() >= duration)
            break;

        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    outerlooprunning.store(false);
    controllerlooping.store(false);
    monitorthread.store(false);
    stopsigma.store(true);

    if (controllerThread.joinable()) controllerThread.join();
    if (controllerThread2.joinable()) controllerThread2.join();
    if (monitorThread.joinable()) monitorThread.join();
    if (monitorpowerthread.joinable()) monitorpowerthread.join();
    if (monitorfreqthread.joinable()) monitorfreqthread.join();

    for (int i = 0; i < numtasks; ++i) {
        delete[] gain[i];
    }
    delete[] gain;

    log.dump_files(path);
    log.dump_power(path, finalpowervalues);
    log.dump_freq(path, finalfrequencyvalues);

    if (sharedData) {
        shmdt(sharedData);
    }
    if (createdshm) {
        shmctl(shmid, IPC_RMID, NULL);
    }

    return 0;
}