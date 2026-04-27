#include "../Scheduler/shared_header.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
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

using namespace std;

static constexpr int MAX_TASKS = 4;

static volatile sig_atomic_t reload_requested = 0;
static volatile sig_atomic_t keep_running = 1;

static void handleSIGINT(int sig)
{
    if (sig == SIGINT) {
        reload_requested = 1;
    }
}

static void handleSIGTERM(int sig)
{
    if (sig == SIGTERM) {
        keep_running = 0;
    }
}

static float average_vec(const vector<float>& v)
{
    if (v.empty()) return 0.0f;

    float sum = 0.0f;
    for (float x : v) sum += x;
    return sum / static_cast<float>(v.size());
}

static float deadline_miss_percent(const vector<float>& rt_ms, float period_s)
{
    if (rt_ms.empty()) return 0.0f;

    int misses = 0;
    float deadline_ms = period_s * 1000.0f;

    for (float rt : rt_ms) {
        if (rt > deadline_ms) misses++;
    }

    return 100.0f * static_cast<float>(misses) / static_cast<float>(rt_ms.size());
}

static float clamp_period(float p)
{
    const float min_period = 0.010f;
    const float max_period = 10.000f;

    if (!std::isfinite(p)) return max_period;
    return std::clamp(p, min_period, max_period);
}

static int read_num_tasks_from_file()
{
    ifstream file("logs/task_values.txt");
    string task_name;

    int count = 0;
    while (file >> task_name) {
        if (!task_name.empty()) count++;
    }

    if (count < 1) count = 1;
    if (count > MAX_TASKS) count = MAX_TASKS;

    return count;
}

static vector<float> read_setpoints_from_file_or_args(int num_tasks, int argc, char* argv[])
{
    vector<float> setpoints(num_tasks, 0.80f);

    ifstream f("logs/setpoints.txt");
    if (f.is_open()) {
        for (int i = 0; i < num_tasks; i++) {
            if (!(f >> setpoints[i])) {
                setpoints[i] = 0.80f;
            }
        }
        return setpoints;
    }

    /*
        Expected initial server call:
        ./server <num_tasks> <pid1> <pid2> <pid3> <setpoint1> <setpoint2> <setpoint3>

        For the old 3-task launch, setpoints begin at argv[5].
    */
    int setpoint_start = 5;
    for (int i = 0; i < num_tasks; i++) {
        int idx = setpoint_start + i;
        if (idx < argc) {
            setpoints[i] = static_cast<float>(atof(argv[idx]));
        }
    }

    return setpoints;
}

static void write_server_pid()
{
    ofstream f("logs/mainpid.txt");
    f << getpid() << "\n";
}

static void initialize_shared_data(SharedData* sharedData)
{
    for (int i = 0; i < MAX_TASKS; i++) {
        sharedData->responsetime[i] = 0.0f;
        sharedData->executiontime[i] = 0.0f;

        if (sharedData->newperiods[i] <= 0.0f || !std::isfinite(sharedData->newperiods[i])) {
            sharedData->newperiods[i] = 0.70f;
        }
    }
}

static void run_controller(SharedData* sharedData,
                           int num_tasks,
                           const vector<float>& setpoints)
{
    cout << "\n========== SERVER CONTROLLER START ==========\n";
    cout << "num_tasks = " << num_tasks << "\n";

    for (int i = 0; i < num_tasks; i++) {
        cout << "task " << i << " setpoint = " << setpoints[i]
             << ", initial period = " << sharedData->newperiods[i] << "\n";
    }

    vector<ofstream> rt_logs(num_tasks);
    vector<ofstream> rtr_logs(num_tasks);
    vector<ofstream> period_logs(num_tasks);
    vector<ofstream> deadline_logs(num_tasks);

    for (int i = 0; i < num_tasks; i++) {
        rt_logs[i].open("logs/server_rt_t" + to_string(i + 1) + ".txt", ios::app);
        rtr_logs[i].open("logs/server_rtr_t" + to_string(i + 1) + ".txt", ios::app);
        period_logs[i].open("logs/server_period_t" + to_string(i + 1) + ".txt", ios::app);
        deadline_logs[i].open("logs/server_deadline_miss_t" + to_string(i + 1) + ".txt", ios::app);
    }

    ofstream control_period_log("logs/server_control_period.txt", ios::app);

    vector<float> prev_rt(num_tasks, -1.0f);

    int control_period = 0;

    while (keep_running && !reload_requested) {
        vector<vector<float>> rt_samples(num_tasks);

        auto start = chrono::steady_clock::now();

        while (keep_running && !reload_requested) {
            auto now = chrono::steady_clock::now();
            chrono::duration<double> elapsed = now - start;

            if (elapsed.count() >= 4.0) break;

            for (int i = 0; i < num_tasks; i++) {
                float rt = sharedData->responsetime[i];

                if (rt > 0.0f && rt != prev_rt[i]) {
                    rt_samples[i].push_back(rt);
                    prev_rt[i] = rt;
                }
            }

            this_thread::sleep_for(chrono::milliseconds(5));
        }

        if (reload_requested || !keep_running) break;

        cout << "\n--------------------------------------------------\n";
        cout << "Control Period = " << control_period << "\n";

        vector<float> avg_rt(num_tasks, 0.0f);
        vector<float> rtr(num_tasks, 0.0f);
        vector<float> error(num_tasks, 0.0f);
        vector<float> delta(num_tasks, 0.0f);
        vector<float> old_period(num_tasks, 0.0f);
        vector<float> new_period(num_tasks, 0.0f);

        for (int i = 0; i < num_tasks; i++) {
            old_period[i] = sharedData->newperiods[i];
            if (old_period[i] <= 0.0f) old_period[i] = 0.70f;

            avg_rt[i] = average_vec(rt_samples[i]);
            rtr[i] = avg_rt[i] / (old_period[i] * 1000.0f);
            error[i] = setpoints[i] - rtr[i];

            float miss_pct = deadline_miss_percent(rt_samples[i], old_period[i]);

            rt_logs[i] << avg_rt[i] << "\n";
            rtr_logs[i] << rtr[i] << "\n";
            period_logs[i] << old_period[i] * 1000.0f << "\n";
            deadline_logs[i] << miss_pct << "\n";

            cout << "task " << i
                 << " avg_rt_ms=" << avg_rt[i]
                 << " period_s=" << old_period[i]
                 << " RTR=" << rtr[i]
                 << " setpoint=" << setpoints[i]
                 << " error=" << error[i]
                 << " deadline_miss_pct=" << miss_pct
                 << "\n";
        }

        /*
            Simple coupled proportional controller.

            Positive error means:
                setpoint > measured RTR
                task is below utilization/RT target
                period can be decreased to run more often

            Negative error means:
                measured RTR > setpoint
                task is too slow relative to period
                period should increase
        */

        for (int i = 0; i < num_tasks; i++) {
            float self_gain = 0.20f;
            float cross_gain = 0.03f;

            delta[i] = self_gain * error[i];

            for (int j = 0; j < num_tasks; j++) {
                if (i == j) continue;
                delta[i] += cross_gain * error[j];
            }
        }

        for (int i = 0; i < num_tasks; i++) {
            float scale = 1.0f - delta[i];

            if (scale < 0.50f) scale = 0.50f;
            if (scale > 1.50f) scale = 1.50f;

            new_period[i] = clamp_period(old_period[i] * scale);
        }

        for (int i = 0; i < num_tasks; i++) {
            sharedData->newperiods[i] = new_period[i];

            cout << "task " << i
                 << " old_period_s=" << old_period[i]
                 << " new_period_s=" << new_period[i]
                 << "\n";
        }

        control_period_log << control_period << "\n";
        control_period++;
    }

    for (int i = 0; i < num_tasks; i++) {
        rt_logs[i].close();
        rtr_logs[i].close();
        period_logs[i].close();
        deadline_logs[i].close();
    }

    control_period_log.close();

    cout << "========== SERVER CONTROLLER EXIT ==========\n";
}

int main(int argc, char* argv[])
{
    if (argc < 2) {
        cerr << "Usage: " << argv[0]
             << " <num_tasks> [pid1 pid2 pid3 ...] [setpoints...]\n";
        return 1;
    }

    signal(SIGINT, handleSIGINT);
    signal(SIGTERM, handleSIGTERM);

    write_server_pid();

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

    SharedData* sharedData = static_cast<SharedData*>(shmat(shmid, nullptr, 0));
    if (sharedData == reinterpret_cast<void*>(-1)) {
        perror("shmat failed");
        return 1;
    }

    initialize_shared_data(sharedData);

    while (keep_running) {
        reload_requested = 0;

        int num_tasks = read_num_tasks_from_file();
        vector<float> setpoints = read_setpoints_from_file_or_args(num_tasks, argc, argv);

        run_controller(sharedData, num_tasks, setpoints);

        if (reload_requested) {
            cout << "\nSIGINT received: reloading task list and setpoints...\n";
            this_thread::sleep_for(chrono::milliseconds(100));
        }
    }

    shmdt(sharedData);

    return 0;
}
