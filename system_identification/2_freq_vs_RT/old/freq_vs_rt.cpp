// freq_vs_rt.cpp
//
// Runs each task at a set of GPU frequencies, parses the task's t1 response-time
// log, and writes summary CSVs.
//
// Assumptions:
//   1) Each task is launched like:
//        ./mm <problem_size> <task_period> <run_duration>
//   2) GPU frequency is changed with:
//        ./set_gpu_freq.sh <frequency_in_hz>
//   3) Each task writes a t1 log file containing response times.
//   4) We extract the LAST numeric value on each line of the log as the response time.
//      This is intentionally tolerant to different log formats.
//
// Edit the config section below. No command-line flags are used.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <regex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>
#include <unistd.h>

namespace fs = std::filesystem;

struct TaskConfig {
    std::string name;         // logical task name
    std::string binary;       // e.g. "./mm"
    std::string problem_size; // passed as argv[1]
    std::string log_path;     // t1 response-time log path
};

struct Stats {
    size_t count = 0;
    double mean = std::numeric_limits<double>::quiet_NaN();
    double max = std::numeric_limits<double>::quiet_NaN();
    double p95 = std::numeric_limits<double>::quiet_NaN();
};

static const std::string kWorkingDir = ".";   // set to your experiment directory if needed
static const std::string kPeriodStr  = "0.70";
static const int kRunDurationSec     = 20;    // edit if needed
static const int kSettleMsAfterFreq  = 500;   // let freq settle
static const int kSettleMsAfterRun   = 300;   // let logs flush

static const std::vector<long long> kFreqsHz = {
    306000000LL, 408000000LL, 510000000LL, 612000000LL,
    714000000LL, 816000000LL, 918000000LL, 1020000000LL
};

// ---------------------------
// EDIT THIS SECTION
// ---------------------------
static const std::vector<TaskConfig> kTasks = {
    // Replace problem_size and log_path with your real values/paths.
    //Usage: ../tasks/bfs <num_nodes> <period_s> <duration_s> [jobs] [out_json] 
    {"mm",       "../tasks/mm",       "1333", "logs/mm_t1.log"},
    {"stereo",   "../tasks/stereo",   "2304 1728", "logs/stereo_t1.log"},
    {"quasi",    "../tasks/quasi",    "16777216", "logs/quasi_t1.log"},
    {"hist",     "../tasks/hist",     "65536000", "logs/hist_t1.log"},
    {"particle", "../tasks/particle", "21948", "logs/particle_t1.log"},
    {"bfs",      "../tasks/bfs",      "2000000", "logs/bfs_t1.log"}
};
// ---------------------------

static std::string shellEscapeSingleQuotes(const std::string& s) {
    std::string out;
    for (char c : s) {
        if (c == '\'') out += "'\\''";
        else out += c;
    }
    return out;
}

static int runCommand(const std::string& cmd) {
    std::cout << "[CMD] " << cmd << std::endl;
    int ret = std::system(cmd.c_str());
    if (ret != 0) {
        std::cerr << "[WARN] Command returned non-zero status: " << ret << "\n";
    }
    return ret;
}

static bool fileExists(const std::string& path) {
    std::error_code ec;
    return fs::exists(path, ec);
}

static void removeIfExists(const std::string& path) {
    std::error_code ec;
    if (fs::exists(path, ec)) {
        fs::remove(path, ec);
    }
}

static std::vector<double> parseResponseTimesFromLog(const std::string& logPath) {
    std::vector<double> rts;

    std::ifstream fin(logPath);
    if (!fin) {
        std::cerr << "[WARN] Could not open log file: " << logPath << "\n";
        return rts;
    }

    // Match floats/ints, including scientific notation
    const std::regex num_re(R"(([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?))");

    std::string line;
    while (std::getline(fin, line)) {
        std::sregex_iterator it(line.begin(), line.end(), num_re);
        std::sregex_iterator end;

        if (it == end) continue;

        // take the LAST numeric value on the line
        std::string last;
        for (; it != end; ++it) last = (*it)[1].str();

        try {
            double v = std::stod(last);
            rts.push_back(v);
        } catch (...) {
            // ignore malformed conversion
        }
    }

    return rts;
}

static double percentile95(std::vector<double> v) {
    if (v.empty()) return std::numeric_limits<double>::quiet_NaN();
    std::sort(v.begin(), v.end());
    double idx = 0.95 * static_cast<double>(v.size() - 1);
    size_t lo = static_cast<size_t>(std::floor(idx));
    size_t hi = static_cast<size_t>(std::ceil(idx));
    double frac = idx - static_cast<double>(lo);
    if (lo == hi) return v[lo];
    return v[lo] * (1.0 - frac) + v[hi] * frac;
}

static Stats computeStats(const std::vector<double>& xs) {
    Stats s;
    s.count = xs.size();
    if (xs.empty()) return s;

    double sum = std::accumulate(xs.begin(), xs.end(), 0.0);
    s.mean = sum / static_cast<double>(xs.size());
    s.max = *std::max_element(xs.begin(), xs.end());
    s.p95 = percentile95(xs);
    return s;
}

struct ResultRow {
    std::string task;
    long long freq_hz;
    Stats stats;
};

int main() {
    //if (geteuid() != 0) {
    //    std::cerr << "Error: This program must be run as root.\n";
    //    std::cerr << "sudo ./freq_vs_rt\n";
    //    return 1;
    //} 

    std::vector<ResultRow> results;

    fs::create_directories("csv");
    std::ofstream detailedCsv("csv/freq_vs_rt_detailed.csv");
    if (!detailedCsv) {
        std::cerr << "Failed to open freq_vs_rt_detailed.csv for writing\n";
        return 1;
    }
    detailedCsv << "task,freq_hz,count,mean_rt,max_rt,p95_rt\n";

    // mean matrix: rows=freq, cols=tasks
    std::ofstream matrixCsv("csv/freq_vs_rt_mean_matrix.csv");
    if (!matrixCsv) {
        std::cerr << "Failed to open freq_vs_rt_mean_matrix.csv for writing\n";
        return 1;
    }

    matrixCsv << "freq_hz";
    for (const auto& task : kTasks) {
        matrixCsv << "," << task.name;
    }
    matrixCsv << "\n";

    for (long long freq : kFreqsHz) {
        std::cout << "\n========================================\n";
        std::cout << "Setting GPU frequency to " << freq << " Hz\n";
        std::cout << "========================================\n";

        {
            std::ostringstream cmd;
            cmd << "sudo ../scripts/set_gpu_freq.sh " << freq;
            if (runCommand(cmd.str()) != 0) {
                std::cerr << "[ERROR] Failed to set GPU frequency to " << freq << "\n";
                // continue anyway
            }
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(kSettleMsAfterFreq));

        matrixCsv << freq;

        for (const auto& task : kTasks) {
            std::cout << "\n--- Task: " << task.name << " @ " << freq << " Hz ---\n";

            removeIfExists(task.log_path);

            std::ostringstream cmd;
            cmd << "cd '" << shellEscapeSingleQuotes(kWorkingDir) << "' && "
                << task.binary << " "
                << task.problem_size << " "
                << kPeriodStr << " "
                << kRunDurationSec;

            int rc = runCommand(cmd.str());
            if (rc != 0) {
                std::cerr << "[WARN] Task returned non-zero status for " << task.name << "\n";
            }

            std::this_thread::sleep_for(std::chrono::milliseconds(kSettleMsAfterRun));

            auto rts = parseResponseTimesFromLog(task.log_path);
            Stats stats = computeStats(rts);

            if (stats.count == 0) {
                std::cerr << "[WARN] No response times parsed for task "
                          << task.name << " from log " << task.log_path << "\n";
            } else {
                std::cout << "count=" << stats.count
                          << " mean=" << stats.mean
                          << " max=" << stats.max
                          << " p95=" << stats.p95 << "\n";
            }

            results.push_back({task.name, freq, stats});

            detailedCsv
                << task.name << ","
                << freq << ","
                << stats.count << ","
                << stats.mean << ","
                << stats.max << ","
                << stats.p95 << "\n";

            matrixCsv << "," << stats.mean;
        }

        matrixCsv << "\n";
    }

    detailedCsv.close();
    matrixCsv.close();

    std::cout << "\nDone.\n";
    std::cout << "Wrote:\n";
    std::cout << "  freq_vs_rt_detailed.csv\n";
    std::cout << "  freq_vs_rt_mean_matrix.csv\n";

    return 0;
}
