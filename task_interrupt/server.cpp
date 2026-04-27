#include "../Scheduler/shared_header.hpp"
#include <Eigen/Dense>
#include <stdexcept>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <thread>
#include <vector>
#include <mutex>

#include <sys/ipc.h>
#include <sys/shm.h>
#include <unistd.h>

#define Jetson

using namespace std;

static constexpr int MAX_TASKS = 4;
std::string logs_base = "a"; 

enum Solution {
    FC_GPU = 1,
    PROPOSED_LQR = 3,
    DFS = 7
};

#ifdef Jetson
std::atomic<float> current_freq{918.0f};
#else
std::atomic<float> current_freq{1200.0f};
#endif

float innerloopduration = 4.0f;
float outerloopduration = 4.0f;

static volatile sig_atomic_t reload_requested = 0;
static volatile sig_atomic_t keep_running = 1;
static atomic<bool> sigma_running{false};

using Matrix = Eigen::MatrixXf;

static void handleSIGUSR1(int sig)
{
    const char msg[] = "\nDEBUG: handleSIGUSR1 called\n";
    write(STDOUT_FILENO, msg, sizeof(msg) - 1);
    reload_requested = 1;
}

static void handleSIGTERM(int sig)
{
    if (sig == SIGTERM) keep_running = 0;
}

static float avg_vec(const vector<float>& v)
{
    if (v.empty()) return 0.0f;
    float sum = 0.0f;
    for (float x : v) sum += x;
    return sum / static_cast<float>(v.size());
}

static float clamp_period(float p)
{
    if (!std::isfinite(p)) return 1.0f;
    return std::clamp(p, 0.010f, 10.0f);
}

static float deadline_miss_percent(const vector<float>& rt_ms, float period_s)
{
    if (rt_ms.empty()) return 0.0f;

    float deadline_ms = period_s * 1000.0f;
    int misses = 0;

    for (float rt : rt_ms) {
        if (rt > deadline_ms) misses++;
    }

    return 100.0f * static_cast<float>(misses) / static_cast<float>(rt_ms.size());
}

static void write_server_pid()
{
    ofstream f(logs_base + "mainpid.txt");
    f << getpid() << "\n";
}

static int read_num_tasks_from_file(int fallback)
{
    ifstream file(logs_base + "task_values.txt");
    string s;
    int n = 0;

    while (file >> s) {
        if (!s.empty()) n++;
    }

    if (n <= 0) n = fallback;
    return std::clamp(n, 1, MAX_TASKS);
}

static int returnpowerrtx()
{
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

static int returnfreqrtx()
{
#ifdef Jetson
    const char* cmd = "sudo ../scripts/get_gpu_freq_mhz.sh";
#else
    const char* cmd = "nvidia-smi -i 0 --query-gpu=clocks.current.graphics --format=csv,noheader,nounits";
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

static void activate_freq_nvidia_rtx(int coreClock)
{
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

static float clamp_freq(float f)
{
#ifdef Jetson
    return std::clamp(f, 306.0f, 1020.0f);
#else
    return std::clamp(f, 210.0f, 2100.0f);
#endif
}

class PowerMonitor {
public:
    void start()
    {
        running.store(true, memory_order_relaxed);
        worker = thread(&PowerMonitor::loop, this);
    }

    void stop()
    {
        running.store(false, memory_order_relaxed);
        if (worker.joinable()) worker.join();
    }

    void clear()
    {
        lock_guard<mutex> lk(mtx);
        samples.clear();
    }

    vector<float> snapshot_and_clear()
    {
        lock_guard<mutex> lk(mtx);
        vector<float> out = samples;
        samples.clear();
        return out;
    }

private:
    void loop()
    {
        while (running.load(memory_order_relaxed)) {
            int p = returnpowerrtx();
            if (p > 0) {
                lock_guard<mutex> lk(mtx);
                samples.push_back(static_cast<float>(p));
            }

            this_thread::sleep_for(chrono::milliseconds(100));
        }
    }

    atomic<bool> running{false};
    thread worker;
    mutex mtx;
    vector<float> samples;
};

static void sigmadelta_freq_step(float desired_freq)
{
#ifdef Jetson
    static const vector<int> lut = {1020, 918, 816, 714, 612, 510, 408, 306};
#else
    static const vector<int> lut = {
        2100, 2085, 2070, 2055, 2040, 2025, 2010, 1995, 1980, 1965,
        1950, 1935, 1920, 1905, 1890, 1875, 1860, 1845, 1830, 1815,
        1800, 1785, 1770, 1755, 1740, 1725, 1710, 1695, 1680, 1665,
        1650, 1635, 1620, 1605, 1590, 1575, 1560, 1545, 1530, 1515,
        1500, 1485, 1470, 1455, 1440, 1425, 1410, 1395, 1380, 1365,
        1350, 1335, 1320, 1305, 1290, 1275, 1260, 1245, 1230, 1215,
        1200, 1185, 1170, 1155, 1140, 1125, 1110, 1095, 1080, 1065,
        1050, 1035, 1020, 1005, 990, 975, 960, 945, 930, 915,
        900, 885, 870, 855, 840, 825, 810, 795, 780, 765,
        750, 735, 720, 705, 690, 675, 660, 645, 630, 615,
        600, 585, 570, 555, 540, 525, 510, 495, 480, 465,
        450, 435, 420, 405, 390, 375, 360, 345, 330, 315,
        300, 285, 270, 255, 240, 225, 210
    };
#endif

    desired_freq = clamp_freq(desired_freq);

    float acc = 0.0f;
    int last_lower = 0;
    int last_upper = 0;

    while (sigma_running.load(memory_order_relaxed)) {
        if (desired_freq >= lut.front()) {
            activate_freq_nvidia_rtx(lut.front());
            this_thread::sleep_for(chrono::milliseconds(1));
            continue;
        }

        if (desired_freq <= lut.back()) {
            activate_freq_nvidia_rtx(lut.back());
            this_thread::sleep_for(chrono::milliseconds(1));
            continue;
        }

        auto it = lower_bound(
            lut.begin(),
            lut.end(),
            desired_freq,
            [](int lut_val, float value) {
                return lut_val > value;
            }
        );

        int lower = *it;
        int upper = *(it - 1);

        if (lower != last_lower || upper != last_upper) {
            acc = 0.0f;
            last_lower = lower;
            last_upper = upper;
        }

        float frac = (desired_freq - lower) / static_cast<float>(upper - lower);
        acc += frac;

        int selected_freq;
        if (acc >= 1.0f) {
            selected_freq = upper;
            acc -= 1.0f;
        } else {
            selected_freq = lower;
        }

        activate_freq_nvidia_rtx(selected_freq);
        this_thread::sleep_for(chrono::milliseconds(1));
    }
}

static void stop_sigma_thread(thread& freq_thread)
{
    sigma_running.store(false, memory_order_relaxed);

    if (freq_thread.joinable()) {
        freq_thread.join();
    }
}

static void start_sigma_thread(thread& freq_thread, float desired_freq)
{
    stop_sigma_thread(freq_thread);

    desired_freq = clamp_freq(desired_freq);

    sigma_running.store(true, memory_order_relaxed);
    freq_thread = thread(sigmadelta_freq_step, desired_freq);
}

static vector<float> read_setpoints(int num_tasks, int argc, char* argv[])
{
    vector<float> setpoints(num_tasks, 0.90f);

    for (int i = 0; i < num_tasks; i++) {
        int idx = 5 + i;
        if (idx < argc) {
            setpoints[i] = atof(argv[idx]);
        }
    }

    ifstream f(logs_base + "setpoints.txt");
    if (f.is_open()) {
        for (int i = 0; i < num_tasks; i++) {
            if (!(f >> setpoints[i])) break;
        }
    }

    return setpoints;
}

static void init_shared_periods_only(SharedData* sd)
{
    for (int i = 0; i < MAX_TASKS; i++) {
        if (sd->newperiods[i] <= 0.0f || !std::isfinite(sd->newperiods[i])) {
            sd->newperiods[i] = 0.80f;
        }
    }
}

static string solution_name(int solution)
{
    if (solution == FC_GPU) return "FC_GPU";
    if (solution == PROPOSED_LQR) return "LQR";
    if (solution == DFS) return "DFS";
    return "UNKNOWN";
}

static void open_controller_csv(ofstream& csv)
{   
    bool new_file = !filesystem::exists(logs_base + "controller.csv");

    csv.open(logs_base + "controller.csv", ios::app);

    if (new_file) {
        csv << "control_period,solution,task,exec_ms,rt_ms,period_ms,new_period_ms,"
            << "rtr,setpoint,rtr_error,rt_error_ms,deadline_miss_pct,"
            << "measured_power,power_error,current_freq_mhz,desired_freq_mhz,power_samples\n";
    }
}

static Matrix solveDARE(const Matrix& A,
                        const Matrix& B,
                        const Matrix& Q,
                        const Matrix& R,
                        int max_iters = 10000,
                        float tol = 1e-6f)
{
    Matrix P = Q;
    Matrix Pnext = Q;

    for (int k = 0; k < max_iters; ++k) {
        Matrix S = R + B.transpose() * P * B;
        Matrix Ktmp = S.inverse() * (B.transpose() * P * A);

        Pnext = A.transpose() * P * A
              - A.transpose() * P * B * Ktmp
              + Q;

        float err = (Pnext - P).cwiseAbs().maxCoeff();
        P = Pnext;

        if (err < tol) return P;
    }

    throw std::runtime_error("solveDARE: Riccati iteration did not converge");
}

static Matrix dlqr(const Matrix& A,
                   const Matrix& B,
                   const Matrix& Q,
                   const Matrix& R)
{
    Matrix P = solveDARE(A, B, Q, R);
    Matrix K = (R + B.transpose() * P * B).inverse()
             * (B.transpose() * P * A);
    return K;
}

static float** alloc_gain(int rows, int cols)
{
    float** gain = new float*[rows];

    for (int i = 0; i < rows; ++i) {
        gain[i] = new float[cols];
        for (int j = 0; j < cols; ++j) {
            gain[i][j] = 0.0f;
        }
    }

    return gain;
}

static void free_gain(float** gain, int rows)
{
    if (!gain) return;

    for (int i = 0; i < rows; ++i) {
        delete[] gain[i];
    }

    delete[] gain;
}

static void compute_FC_gain(float** gain, int numtasks)
{
    for (int i = 0; i < numtasks; ++i) {
        for (int j = 0; j < numtasks; ++j) {
            gain[i][j] = 0.0f;
        }
    }

    if (numtasks == 1) {
        gain[0][0] = 0.4968f * 2.0f * 10.0f;
    }
    else if (numtasks == 2) {
        gain[0][0] = 0.4968f * 2.0f * 10.0f;
        gain[0][1] = 3.3968f / 10.0f;

        gain[1][0] = 3.3968f / 10.0f;
        gain[1][1] = 0.7968f * 2.0f * 10.0f;
    }
    else if (numtasks == 3) {
        gain[0][0] = 0.4968f * 2.0f * 10.0f;
        gain[0][1] = 3.3968f / 10.0f;
        gain[0][2] = 3.3968f / 10.0f;

        gain[1][0] = 3.3968f / 10.0f;
        gain[1][1] = 0.7968f * 2.0f * 10.0f;
        gain[1][2] = 3.3968f / 10.0f;

        gain[2][0] = 2.2976f / 10.0f;
        gain[2][1] = 2.2976f / 10.0f;
        gain[2][2] = 0.6976f * 2.0f * 10.0f;
    }
    else if (numtasks == 4) {
        gain[0][0] = 2.6968f * 50.0f;
        gain[0][1] = 1.3968f / 10.0f;
        gain[0][2] = 0.3968f / 10.0f;
        gain[0][3] = 0.3968f / 10.0f;

        gain[1][0] = 0.3968f / 10.0f;
        gain[1][1] = 2.6968f * 50.0f;
        gain[1][2] = 0.3968f / 10.0f;
        gain[1][3] = 0.3968f / 10.0f;

        gain[2][0] = 0.2976f / 10.0f;
        gain[2][1] = 0.2976f / 10.0f;
        gain[2][2] = 2.6976f * 50.0f;
        gain[2][3] = 0.2976f / 10.0f;

        gain[3][0] = 0.2976f / 10.0f;
        gain[3][1] = 0.2976f / 10.0f;
        gain[3][2] = 0.2976f / 10.0f;
        gain[3][3] = 2.6976f * 50.0f;
    }
    else {
        throw std::runtime_error("compute_FC_gain: numtasks must be 1 to 4.");
    }
}

static void compute_LQR_gain_2(float** gain, int numtasks)
{
    if (numtasks < 1 || numtasks > 6) {
        throw std::runtime_error("compute_LQR_gain_2: numtasks must be between 1 and 6.");
    }

    const float base_last_col_full[6] = {
        0.06238100f,
        0.06804800f,
        0.07260400f,
        0.10549000f,
        0.02831200f,
        0.08079800f
    };

    const float base_last_row_full[6] = {
        0.02220753300f,
        0.01107223000f,
        0.02852230300f,
        0.0738135200f,
        0.01356965500f,
        0.01528722400f
    };

    const float base_bottom_right = 0.00116413f;

    const int N = numtasks;
    const int n = N + 1;

    Matrix A = Matrix::Identity(n, n);
    for (int i = 0; i < n; ++i) A(i, i) = 0.95f;

    Matrix B = Matrix::Zero(n, n);

    for (int i = 0; i < N; ++i) B(i, i) = 1.0f;
    for (int i = 0; i < N; ++i) B(i, N) = base_last_col_full[i];
    for (int j = 0; j < N; ++j) B(N, j) = base_last_row_full[j];

    B(N, N) = -base_bottom_right;

    const float q_scale = 4.0f;
    const float r_scale = 4.0f;

    Matrix Q = Matrix::Zero(n, n);
    for (int i = 0; i < N; ++i) Q(i, i) = 20.0f / q_scale;
    Q(N, N) = 10.0f / (q_scale * 4);

    Matrix R = Matrix::Zero(n, n);
    for (int i = 0; i < N; ++i) R(i, i) = 1.0f / r_scale;
    R(N, N) = 10.0f / (r_scale * 4);

    Matrix K = dlqr(A, B, Q, R);
    std::cout << "LQR Gain K:\n" << K << "\n\n";

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            gain[i][j] = K(i, j);
        }
    }
}

static void run_controller(SharedData* sd,
                           int num_tasks,
                           int solution,
                           float power_setpoint,
                           const vector<float>& setpoints)
{
    cout << "\n========== SERVER START ==========\n";
    cout << "num_tasks=" << num_tasks
         << " solution=" << solution_name(solution)
         << " power_setpoint=" << power_setpoint
         << " control_period_s=" << innerloopduration
         << "\n";

    vector<float> prev_rt(num_tasks, -1.0f);

    float** fcK = alloc_gain(num_tasks, num_tasks);
    float** lqrK = alloc_gain(num_tasks + 1, num_tasks + 1);

    compute_FC_gain(fcK, num_tasks);
    compute_LQR_gain_2(lqrK, num_tasks);

    thread freq_thread;

    PowerMonitor power_monitor;
    power_monitor.start();

    ofstream ctrlLog;
    open_controller_csv(ctrlLog);

    int control_period = 0;

    while (keep_running && !reload_requested) {
        vector<vector<float>> rt_samples(num_tasks);
        vector<vector<float>> exec_samples(num_tasks);

        power_monitor.clear();

        auto window_start = chrono::steady_clock::now();

        while (keep_running && !reload_requested) {
            auto now = chrono::steady_clock::now();
            chrono::duration<double> elapsed = now - window_start;

            if (elapsed.count() >= innerloopduration) break;

            for (int i = 0; i < num_tasks; i++) {
                float rt = sd->responsetime[i];
                float ex = sd->executiontime[i];

                if (rt > 0.0f && rt != prev_rt[i]) {
                    rt_samples[i].push_back(rt);
                    exec_samples[i].push_back(ex);
                    prev_rt[i] = rt;
                }
            }

            this_thread::sleep_for(chrono::milliseconds(5));
        }

        if (reload_requested || !keep_running) break;

        vector<float> power_samples = power_monitor.snapshot_and_clear();

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

        float cfreq = current_freq.load(memory_order_relaxed);
        float measured_freq = cfreq;
        float desired_freq = cfreq;

        int real_freq = returnfreqrtx();
        if (real_freq > 0) {
            measured_freq = static_cast<float>(real_freq);
        }

        for (int i = 0; i < num_tasks; i++) {
            avg_rt[i] = avg_vec(rt_samples[i]);
            avg_exec[i] = avg_vec(exec_samples[i]);

            period_s[i] = sd->newperiods[i];
            if (period_s[i] <= 0.0f || !std::isfinite(period_s[i])) {
                period_s[i] = 0.80f;
            }

            period_ms[i] = period_s[i] * 1000.0f;
            rtr[i] = avg_rt[i] / period_ms[i];

            rtr_error[i] = setpoints[i] - rtr[i];

            float desired_rt_ms = setpoints[i] * period_ms[i];
            rt_error_ms[i] = desired_rt_ms - avg_rt[i];

            deadline_miss[i] = deadline_miss_percent(rt_samples[i], period_s[i]);
            new_period_s[i] = period_s[i];
        }

        cout << "\n--------------------------------------------------\n";
        cout << "Control Period = " << control_period << "\n";
        cout << "Power samples = " << power_samples.size() << "\n";
        cout << "Measured Power = " << measured_power
             << " Power Error = " << power_error << "\n";

        if (solution == FC_GPU) {
            cout << "[FC_GPU] period-only RTR control\n";

            stop_sigma_thread(freq_thread);

            for (int i = 0; i < num_tasks; i++) {
                float delta_ms = 0.0f;

                for (int j = 0; j < num_tasks; j++) {
                    delta_ms += rtr_error[j] * (-fcK[i][j]);
                }

                float next_ms = period_ms[i] + delta_ms;
                new_period_s[i] = clamp_period(next_ms / 1000.0f);
            }
        }
        else if (solution == DFS) {
            cout << "[DFS] frequency-only power control with sigma-delta\n";

            float kp = 0.05f;
            float delta_freq = kp * power_error;

            desired_freq = clamp_freq(cfreq + delta_freq);
            current_freq.store(desired_freq, memory_order_relaxed);
            start_sigma_thread(freq_thread, desired_freq);

            for (int i = 0; i < num_tasks; i++) {
                new_period_s[i] = period_s[i];
            }

            cout << "Current freq        = " << cfreq << " MHz\n";
            cout << "Delta freq         = " << delta_freq << " MHz\n";
            cout << "Desired freq       = " << desired_freq << " MHz\n";
        }
        else if (solution == PROPOSED_LQR) {
            cout << "[LQR/Proposed] RT + power control with sigma-delta\n";

            vector<float> error(num_tasks + 1, 0.0f);
            vector<float> delta(num_tasks + 1, 0.0f);

            for (int i = 0; i < num_tasks; i++) {
                error[i] = rt_error_ms[i];
            }

            error[num_tasks] = power_error;

            for (int i = 0; i < num_tasks + 1; i++) {
                for (int j = 0; j < num_tasks + 1; j++) {
                    delta[i] += error[j] * (-lqrK[i][j]);
                }
            }

            for (int i = 0; i < num_tasks; i++) {
                float next_ms = period_ms[i] + delta[i];
                new_period_s[i] = clamp_period(next_ms / 1000.0f);
            }

            desired_freq = clamp_freq(cfreq + delta[num_tasks]);
            current_freq.store(desired_freq, memory_order_relaxed);
            start_sigma_thread(freq_thread, desired_freq);

            cout << "Current freq       = " << cfreq << " MHz\n";
            cout << "Delta freq         = " << delta[num_tasks] << " MHz\n";
            cout << "Desired freq       = " << desired_freq << " MHz\n";
        }
        else {
            cerr << "Unknown solution. Defaulting to FC_GPU.\n";

            stop_sigma_thread(freq_thread);

            for (int i = 0; i < num_tasks; i++) {
                float delta_ms = -80.0f * rtr_error[i];
                float next_ms = period_ms[i] + delta_ms;
                new_period_s[i] = clamp_period(next_ms / 1000.0f);
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
                    << solution_name(solution) << ","
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
                    << measured_freq << ","
                    << desired_freq << ","
                    << power_samples.size()
                    << "\n";
        }

        ctrlLog.flush();
        control_period++;
    }

    power_monitor.stop();
    stop_sigma_thread(freq_thread);

    ctrlLog.close();

    free_gain(fcK, num_tasks);
    free_gain(lqrK, num_tasks + 1);

    cout << "========== SERVER EXIT ==========\n";
}

int main(int argc, char* argv[])
{
    if (argc < 5) {
        cerr << "Usage:\n"
             << "  " << argv[0]
             << " <num_tasks> <solution> <power_setpoint> <control_period_sec> <setpoint1> <setpoint2> ...\n\n"
             << "solution: 1=FC_GPU, 3=LQR/Proposed, 7=DFS\n";
        return 1;
    }
    std::signal(SIGUSR1, handleSIGUSR1);
    std::signal(SIGTERM, handleSIGTERM);

    system("mkdir -p logs");

    ofstream shmfile("shmfile", ios::app);
    shmfile.close();

    write_server_pid();

    int fallback_num_tasks = atoi(argv[1]);
    int solution = atoi(argv[2]);
    if (solution == 1) logs_base = "logs/FC/";
    else if (solution == 3) logs_base = "logs/LQR/";
    else logs_base = "logs/DFS/";
    float power_setpoint = atof(argv[3]);

    innerloopduration = atof(argv[4]);
    if (innerloopduration <= 0.0f || !std::isfinite(innerloopduration)) {
        innerloopduration = 4.0f;
    }
    outerloopduration = innerloopduration;

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

    sharedData = static_cast<SharedData*>(shmat(shmid, nullptr, 0));
    if (sharedData == reinterpret_cast<void*>(-1)) {
        perror("shmat failed");
        return 1;
    }

    init_shared_periods_only(sharedData);

    while (keep_running) {
        reload_requested = 0;

        int num_tasks = read_num_tasks_from_file(fallback_num_tasks);
        vector<float> setpoints = read_setpoints(num_tasks, argc, argv);
        std::cout << "*****************************************************************************\n\nReloaded num_tasks = " << num_tasks << "\n\n";

        run_controller(sharedData, num_tasks, solution, power_setpoint, setpoints);

        if (reload_requested) {
            cout << "\nReloading task list/setpoints after SIGINT...\n";
            this_thread::sleep_for(chrono::milliseconds(100));
        }
    }

    sigma_running.store(false, memory_order_relaxed);
    shmdt(sharedData);

    return 0;
}
