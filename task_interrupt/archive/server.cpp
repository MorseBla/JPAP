#include "../Scheduler/shared_header.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

#include <sys/ipc.h>
#include <sys/shm.h>
#include <unistd.h>

#define Jetson
using namespace std;

static constexpr int MAX_TASKS = 4;

static volatile sig_atomic_t reload_requested = 0;
static volatile sig_atomic_t keep_running = 1;

enum Solution {
    FC_GPU = 1,
    PROPOSED_LQR = 3,
    DFS = 7
};

std::atomic<float> current_freq{918.0f};

static void handleSIGINT(int sig) {
    if (sig == SIGINT) reload_requested = 1;
}

static void handleSIGTERM(int sig) {
    if (sig == SIGTERM) keep_running = 0;
}

static float avg_vec(const vector<float>& v) {
    if (v.empty()) return 0.0f;
    float s = 0.0f;
    for (float x : v) s += x;
    return s / static_cast<float>(v.size());
}

static float clamp_period(float p) {
    if (!std::isfinite(p)) return 1.0f;
    return std::clamp(p, 0.010f, 10.0f);
}

static float deadline_miss_percent(const vector<float>& rt_ms, float period_s) {
    if (rt_ms.empty()) return 0.0f;
    float deadline_ms = period_s * 1000.0f;
    int misses = 0;

    for (float rt : rt_ms) {
        if (rt > deadline_ms) misses++;
    }

    return 100.0f * misses / static_cast<float>(rt_ms.size());
}

static void write_server_pid() {
    ofstream f("logs/mainpid.txt");
    f << getpid() << "\n";
}

static int read_num_tasks_from_file(int fallback) {
    ifstream file("logs/task_values.txt");
    string s;
    int n = 0;

    while (file >> s) {
        if (!s.empty()) n++;
    }

    if (n <= 0) n = fallback;
    return std::clamp(n, 1, MAX_TASKS);
}

static int returnpowerrtx() {
#ifdef Jetson
    const char* cmd = "sudo ../scripts/get_power.sh";
#else
    const char* cmd = "nvidia-smi -i 0 --query-gpu=power.draw --format=csv,noheader,nounits";
#endif

    FILE* pipe = popen(cmd, "r");
    if (!pipe) return -1;

    char buffer[64];
    if (!fgets(buffer, sizeof(buffer), pipe)) {
        pclose(pipe);
        return -1;
    }

    pclose(pipe);
    return atoi(buffer);
}

static void activate_freq_nvidia_rtx(int coreClock) {
#ifdef Jetson
    string command = "sudo ../scripts/set_gpu_freq.sh " +
                     to_string(coreClock * 1000000) +
                     " > /dev/null 2>&1";
#else
    string command = "sudo nvidia-smi -i 0 -lgc " +
                     to_string(coreClock) + "," +
                     to_string(coreClock) +
                     " > /dev/null 2>&1";
#endif

    system(command.c_str());
}

static float clamp_freq(float f) {
#ifdef Jetson
    return std::clamp(f, 306.0f, 1020.0f);
#else
    return std::clamp(f, 210.0f, 2100.0f);
#endif
}

static void command_freq(float f) {
    f = clamp_freq(f);
    current_freq.store(f);
    activate_freq_nvidia_rtx(static_cast<int>(f));
}

static vector<float> read_setpoints(int num_tasks, int argc, char* argv[]) {
    vector<float> setpoints(num_tasks, 0.90f);

    for (int i = 0; i < num_tasks; i++) {
        int idx = 4 + i;
        if (idx < argc) {
            setpoints[i] = atof(argv[idx]);
        }
    }

    ifstream f("logs/setpoints.txt");
    if (f.is_open()) {
        for (int i = 0; i < num_tasks; i++) {
            if (!(f >> setpoints[i])) break;
        }
    }

    return setpoints;
}

static void init_shared(SharedData* sd) {
    for (int i = 0; i < MAX_TASKS; i++) {
        sd->responsetime[i] = 0.0f;
        sd->executiontime[i] = 0.0f;
        if (sd->newperiods[i] <= 0.0f || !std::isfinite(sd->newperiods[i])) {
            sd->newperiods[i] = 0.80f;
        }
    }
}

static vector<vector<float>> make_fc_gain(int n) {
    vector<vector<float>> K(n, vector<float>(n, 0.0f));

    for (int i = 0; i < n; i++) {
        K[i][i] = 80.0f;
        for (int j = 0; j < n; j++) {
            if (i != j) K[i][j] = 8.0f;
        }
    }

    return K;
}

static vector<vector<float>> make_lqr_gain(int n) {
    vector<vector<float>> K(n + 1, vector<float>(n + 1, 0.0f));

    for (int i = 0; i < n; i++) {
        K[i][i] = 0.20f;
        for (int j = 0; j < n; j++) {
            if (i != j) K[i][j] = 0.03f;
        }

        K[i][n] = 0.00002f;
    }

    K[n][n] = 0.05f;

    return K;
}

static void run_controller(SharedData* sd,
                           int num_tasks,
                           int solution,
                           float power_setpoint,
                           const vector<float>& setpoints) {
    cout << "\n========== SERVER START ==========\n";
    cout << "num_tasks=" << num_tasks
         << " solution=" << solution
         << " power_setpoint=" << power_setpoint << "\n";

    vector<float> prev_rt(num_tasks, -1.0f);
    vector<vector<float>> fcK = make_fc_gain(num_tasks);
    vector<vector<float>> lqrK = make_lqr_gain(num_tasks);

    ofstream ctrlLog("logs/controller.txt", ios::app);

    int control_period = 0;

    while (keep_running && !reload_requested) {
        vector<vector<float>> rt_samples(num_tasks);
        vector<vector<float>> exec_samples(num_tasks);
        vector<float> power_samples;

        auto start = chrono::steady_clock::now();

        while (keep_running && !reload_requested) {
            auto now = chrono::steady_clock::now();
            chrono::duration<double> elapsed = now - start;
            if (elapsed.count() >= 1.0) break;

            for (int i = 0; i < num_tasks; i++) {
                float rt = sd->responsetime[i];
                float ex = sd->executiontime[i];

                if (rt > 0.0f && rt != prev_rt[i]) {
                    rt_samples[i].push_back(rt);
                    exec_samples[i].push_back(ex);
                    prev_rt[i] = rt;
                }
            }

            int p = returnpowerrtx();
            if (p > 0) power_samples.push_back(static_cast<float>(p));

            this_thread::sleep_for(chrono::milliseconds(20));
        }

        if (reload_requested || !keep_running) break;

        vector<float> avg_rt(num_tasks, 0.0f);
        vector<float> avg_exec(num_tasks, 0.0f);
        vector<float> period_s(num_tasks, 0.0f);
        vector<float> period_ms(num_tasks, 0.0f);
        vector<float> rtr(num_tasks, 0.0f);
        vector<float> rtr_error(num_tasks, 0.0f);
        vector<float> rt_error_ms(num_tasks, 0.0f);
        vector<float> new_period_s(num_tasks, 0.0f);
        vector<float> deadline_miss(num_tasks, 0.0f);

        float measured_power = avg_vec(power_samples);
        float power_error = power_setpoint - measured_power;

        cout << "\n--------------------------------------------------\n";
        cout << "Control Period = " << control_period << "\n";
        cout << "Measured Power = " << measured_power
             << " Power Error = " << power_error << "\n";

        for (int i = 0; i < num_tasks; i++) {
            avg_rt[i] = avg_vec(rt_samples[i]);
            avg_exec[i] = avg_vec(exec_samples[i]);

            period_s[i] = sd->newperiods[i];
            if (period_s[i] <= 0.0f) period_s[i] = 0.80f;

            period_ms[i] = period_s[i] * 1000.0f;
            rtr[i] = avg_rt[i] / period_ms[i];

            rtr_error[i] = setpoints[i] - rtr[i];

            float desired_rt_ms = setpoints[i] * period_ms[i];
            rt_error_ms[i] = desired_rt_ms - avg_rt[i];

            deadline_miss[i] = deadline_miss_percent(rt_samples[i], period_s[i]);

            new_period_s[i] = period_s[i];
        }

        if (solution == FC_GPU) {
            cout << "[FC_GPU] period-only RTR control\n";

            for (int i = 0; i < num_tasks; i++) {
                float delta_ms = 0.0f;

                for (int j = 0; j < num_tasks; j++) {
                    delta_ms += -fcK[i][j] * rtr_error[j];
                }

                float next_ms = period_ms[i] + delta_ms;
                new_period_s[i] = clamp_period(next_ms / 1000.0f);
            }
        }
        else if (solution == PROPOSED_LQR) {
            cout << "[PROPOSED/LQR] task RT + power control\n";

            vector<float> error(num_tasks + 1, 0.0f);
            vector<float> delta(num_tasks + 1, 0.0f);

            for (int i = 0; i < num_tasks; i++) {
                error[i] = rt_error_ms[i];
            }

            error[num_tasks] = power_error;

            for (int i = 0; i < num_tasks + 1; i++) {
                for (int j = 0; j < num_tasks + 1; j++) {
                    delta[i] += -lqrK[i][j] * error[j];
                }
            }

            for (int i = 0; i < num_tasks; i++) {
                float next_ms = period_ms[i] + delta[i];
                new_period_s[i] = clamp_period(next_ms / 1000.0f);
            }

            float measured_freq = current_freq.load();
            float next_freq = clamp_freq(measured_freq + delta[num_tasks]);
            command_freq(next_freq);

            cout << "Current Freq = " << measured_freq
                 << " New Freq = " << next_freq << "\n";
        }
        else if (solution == DFS) {
            cout << "[DFS] frequency-only power control\n";

            float kp = 0.05f;
            float delta_freq = kp * power_error;

            float measured_freq = current_freq.load();
            float next_freq = clamp_freq(measured_freq + delta_freq);
            command_freq(next_freq);

            cout << "Current Freq = " << measured_freq
                 << " Delta Freq = " << delta_freq
                 << " New Freq = " << next_freq << "\n";

            for (int i = 0; i < num_tasks; i++) {
                new_period_s[i] = period_s[i];
            }
        }
        else {
            cerr << "Unknown solution " << solution << ". Defaulting to FC_GPU.\n";

            for (int i = 0; i < num_tasks; i++) {
                float delta_ms = -80.0f * rtr_error[i];
                new_period_s[i] = clamp_period((period_ms[i] + delta_ms) / 1000.0f);
            }
        }

        for (int i = 0; i < num_tasks; i++) {
            sd->newperiods[i] = new_period_s[i];

            cout << "T" << i
                 << " exec_ms=" << avg_exec[i]
                 << " rt_ms=" << avg_rt[i]
                 << " period_ms=" << period_ms[i]
                 << " new_period_ms=" << new_period_s[i] * 1000.0f
                 << " rtr=" << rtr[i]
                 << " setpoint=" << setpoints[i]
                 << " rtr_error=" << rtr_error[i]
                 << " rt_error_ms=" << rt_error_ms[i]
                 << " deadline_miss_pct=" << deadline_miss[i]
                 << "\n";

            ctrlLog << control_period << ","
                    << solution << ","
                    << i << ","
                    << avg_exec[i] << ","
                    << avg_rt[i] << ","
                    << period_ms[i] << ","
                    << new_period_s[i] * 1000.0f << ","
                    << rtr[i] << ","
                    << setpoints[i] << ","
                    << rtr_error[i] << ","
                    << rt_error_ms[i] << ","
                    << deadline_miss[i] << ","
                    << measured_power << ","
                    << power_error << ","
                    << current_freq.load()
                    << "\n";
        }

        ctrlLog.flush();
        control_period++;
    }

    ctrlLog.close();

    cout << "========== SERVER EXIT ==========\n";
}

int main(int argc, char* argv[]) {
    if (argc < 4) {
        cerr << "Usage:\n"
             << "  " << argv[0]
             << " <num_tasks> <solution> <power_setpoint> <setpoint1> <setpoint2> ...\n\n"
             << "solution: 1=FC_GPU, 3=PROPOSED/LQR, 7=DFS\n";
        return 1;
    }

    signal(SIGINT, handleSIGINT);
    signal(SIGTERM, handleSIGTERM);

    system("mkdir -p logs");
    ofstream shmfile("shmfile", ios::app);
    shmfile.close();

    write_server_pid();

    int fallback_num_tasks = atoi(argv[1]);
    int solution = atoi(argv[2]);
    float power_setpoint = atof(argv[3]);

    key_t key = ftok("shmfile", 65);
    if (key == -1) {
        perror("ftok failed");
        return 1;
    }

    int shmid = shmget(key, sizeof(SharedData), IPC_CREAT | 0666);
    if (shmid < 0) {
        perror("shmget failed");
        return 1;
    }

    SharedData* sd = static_cast<SharedData*>(shmat(shmid, nullptr, 0));
    if (sd == reinterpret_cast<void*>(-1)) {
        perror("shmat failed");
        return 1;
    }

    init_shared(sd);

    while (keep_running) {
        reload_requested = 0;

        int num_tasks = read_num_tasks_from_file(fallback_num_tasks);
        vector<float> setpoints = read_setpoints(num_tasks, argc, argv);

        run_controller(sd, num_tasks, solution, power_setpoint, setpoints);

        if (reload_requested) {
            cout << "\nReloading task list/setpoints after SIGINT...\n";
            this_thread::sleep_for(chrono::milliseconds(100));
        }
    }

    shmdt(sd);
    return 0;
}
