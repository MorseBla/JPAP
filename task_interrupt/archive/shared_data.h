#ifndef SHARED_DATA_H
#define SHARED_DATA_H

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <deque>
#include <unordered_map>
#include <utility>
#include <atomic>
#include <cstdlib>
#include <cstdio>
#include <csignal>
#include <unistd.h>
#include <sys/ipc.h>
#include <sys/shm.h>
#include <chrono>
// If your system does not define SIGUSR3 / SIGUSR4, map them manually.
// Linux normally only has SIGUSR1 and SIGUSR2.
#ifndef SIGUSR3
#define SIGUSR3 (SIGRTMIN + 1)
#endif

#ifndef SIGUSR4
#define SIGUSR4 (SIGRTMIN + 2)
#endif

// You use indices 0 through 7 in the task mapping.
#define MAX_SHARED_VALUES 8

struct SharedData {
    float values[MAX_SHARED_VALUES];
    float newperiods[MAX_SHARED_VALUES];
};

// These are used by init() and the controller loops.
inline int sm1 = 0;
inline int sm2 = 0;
inline int sm3 = 0;
inline int sm4 = 0;

inline int sm11 = 0;
inline int sm22 = 0;
inline int sm33 = 0;
inline int sm44 = 0;

inline int proc11 = 0;
inline int proc22 = 0;
inline int proc33 = 0;
inline int proc44 = 0;

struct average {
    std::vector<float> taskexec1;
    std::vector<float> taskexec2;
    std::vector<float> taskexec3;
    std::vector<float> taskexec4;

    std::vector<float> taskexec11;
    std::vector<float> taskexec22;
    std::vector<float> taskexec33;
    std::vector<float> taskexec44;

    float calculateAverage(const std::vector<float>& data) {
        if (data.empty()) {
            return 0.0f;
        }

        float sum = 0.0f;
        for (float value : data) {
            sum += value;
        }

        return sum / static_cast<float>(data.size());
    }
};

#endif
