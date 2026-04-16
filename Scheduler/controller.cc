#include "shared_header.hpp"
#include <Eigen/Dense>

#ifdef Jetson
    std::atomic<float> current_freq{918.0f};
#else 
    std::atomic<float> current_freq{1200.0f};
#endif
float innerloopduration = 0.500*2*2;
float outerloopduration = innerloopduration;
std::atomic<bool> outerlooprunning{true};
std::atomic<bool> controllerlooping{true};
std::atomic<bool> monitorthread{true};
std::atomic<bool> stopsigma{false};
float powersetpoint;

#ifdef CLAMP_PERIODS
    #ifndef BOUND
        #define BOUND 40  // Fallback default
    #endif
    static const float BOUND_VALUE = BOUND / 100.0f;
#endif



void activate_freq_nvidia_rtx(int coreClock) 
{  
    int memClock = 5001;  
    #ifdef Jetson
        std::string command ="sudo ../scripts/set_gpu_freq.sh " + std::to_string(coreClock * 1000000) + "> /dev/null 2>&1";
    #else
        std::string command ="sudo nvidia-smi -i 0 -lgc "+ std::to_string(coreClock) + "," + std::to_string(coreClock)+ "> /dev/null 2>&1";
    #endif
    int result = std::system(command.c_str());
    
}


// int returnfreq()
// {
//     const char* cmd ="nvidia-smi -i 0 --query-gpu=clocks.sm --format=csv,noheader,nounits";
//     FILE* pipe = popen(cmd, "r");
//     if (!pipe) {
//         return -1; 
//     }

//     char buffer[64];
//     if (!fgets(buffer, sizeof(buffer), pipe)) {
//         pclose(pipe);
//         return -1;  
//     }
//     pclose(pipe);
//     return std::atoi(buffer); 
// }

int returnpowerrtx()
{

    #ifdef Jetson
        const char* cmd ="sudo ../scripts/get_power.sh"; 
    #else
        const char* cmd ="nvidia-smi -i 0 --query-gpu=power.draw --format=csv,noheader,nounits";
    #endif

    FILE* pipe = popen(cmd, "r");
    if (!pipe) {
        return -1;  // failed to run command
    }

    char buffer[64];
    if (!fgets(buffer, sizeof(buffer), pipe)) {
        pclose(pipe);
        return -1;  // failed to read output
    }
    pclose(pipe);
    return std::atoi(buffer); 

}

int returnfreqrtx(){
   // const char* cmd ="nvidia-smi -i 0 --query-gpu=power.draw --format=csv,noheader,nounits";
    #ifdef Jetson
        const char* cmd ="sudo ../scripts/get_gpu_freq_mhz.sh";
    #else
        const char* cmd = "nvidia-smi -i 0 --query-gpu=clocks.current.graphics --format=csv,noheader,nounits";
    #endif
    FILE* pipe = popen(cmd, "r");
    if (!pipe) {
        return -1;  // failed to run command
    }

    char buffer[64];
    if (!fgets(buffer, sizeof(buffer), pipe)) {
        pclose(pipe);
        return -1;  // failed to read output
    }
    pclose(pipe);
    return std::atoi(buffer); 
    
    
}


static int append_lock(const std::string&path, const std::string &line){
    
    int fd= open(path.c_str(), O_WRONLY | O_CREAT | O_APPEND,0644);
    if (fd<0) return - 1;
    flock(fd,LOCK_EX);
     
    if (write(fd, line.data(), line.size()) < 0) return -1;
    if (write(fd, "\n", 1) < 0) return -1;
    
    flock(fd, LOCK_UN);
    close(fd);
    return 0;
}

float calculate_deadline_miss(const std::vector<float>& local_resp,
                              const std::vector<float>& local_per,
                              float eps = 0.0f)  // e.g., 0.01f for 1% tolerance
{
    if (local_resp.empty() || local_per.empty())
        return 0.0f;

    size_t total = std::min(local_resp.size(), local_per.size());
    int miss_count = 0;

    for (size_t i = 0; i < total; ++i)
    {
        float period = local_per[i];
        if (period <= 0.0f)
            continue;

        float rtr = local_resp[i] / period;
        if (rtr > (1.0f + eps))
            miss_count++;
    }

    return (static_cast<float>(miss_count) / total) * 100.0f;
}

void monitorpower()
{
    average avg;
    interpowervalues.reserve(2048);
    using clock_t = std::chrono::steady_clock;

    while (monitorthread.load(std::memory_order_relaxed))
    {
        interpowervalues.clear();
        auto windowStart = clock_t::now();

        while (true)
        {
            auto now = clock_t::now();
            std::chrono::duration<double> elapsed = now - windowStart;
            if (elapsed.count() >= innerloopduration)
                break;
            
            //#ifdef Jetson
            float power = static_cast<float>(returnpowerrtx());
            interpowervalues.push_back(power);
            std::this_thread::sleep_for(std::chrono::seconds(1));
            //#elif
            //float power = static_cast<float>(returnpowerrtx());
            //interpowervalues.push_back(power);
            //std::this_thread::sleep_for(std::chrono::milliseconds(1));
            
            //#endif
        }

        float finpower = avg.calculateAverage(interpowervalues);
        finalpowervalues.push_back(finpower);
    }
}

void monitorfreq()
{
    average avg;
    interfreqvalues.reserve(2048);
    using clock_t = std::chrono::steady_clock;
    while (monitorthread.load(std::memory_order_relaxed))
    {
        //interfreqvalues.clear();
        auto windowStart = clock_t::now();
        while (true)
        {
            auto now = clock_t::now();
            std::chrono::duration<double> elapsed = now - windowStart;
            if (elapsed.count() >= innerloopduration)
                break;

            //float power = static_cast<float>(returnfreqrtx());
            //interfreqvalues.push_back(power);
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }

        //float finfreq = avg.calculateAverage(interfreqvalues);
        float finfreq =current_freq.load(std::memory_order_relaxed);
        finalfrequencyvalues.push_back(finfreq);
    }
}







void sigmadelta_freq_step(float newfreq) 
{ 
    static const std::vector<int> lut = 
    #ifdef Jetson
        {
            1020, 918, 816, 714, 612, 510, 408, 306    
        };
   
    #else
         {
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

    static float acc = 0.0f;
    static int lastLower = 0;
    static int lastUpper = 0;
    acc = 0.0f;
    lastLower = 0;
    lastUpper = 0;

    while (!stopsigma.load(std::memory_order_relaxed))
    {
        if (newfreq >= lut.front()) {
            activate_freq_nvidia_rtx(lut.front());
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            continue;
        }
        if (newfreq <= lut.back()) {
            activate_freq_nvidia_rtx(lut.back());
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            continue;
        }
        auto it = std::lower_bound(
            lut.begin(), lut.end(), newfreq,
            [](int lutVal, float value) { return lutVal > value; }
        );

        int lower = *it;           
        int upper = *(it - 1);  
        if (lower != lastLower || upper != lastUpper) {
            acc = 0.0f;
            lastLower = lower;
            lastUpper = upper;
        }
        float delta = (newfreq - lower) / (float)(upper - lower);
        acc += delta;
        int selectedFreq;
        if (acc >= 1.0f) {
            selectedFreq = upper;
            acc -= 1.0f;
        } else {
            selectedFreq = lower;
        }
        activate_freq_nvidia_rtx(selectedFreq);
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
}





//void proposed(string path,logging& log, tasks& t, int numtasks,float** gain,std::atomic<bool>& controllerlooping, float powersetpoint, logging_sys& logsys)
//{
//    std::cout << "Proposed running\n";
//
//}



void proposed(std::string path,
              logging& log,
              tasks& t,
              int numtasks,
              float** gain,
              std::atomic<bool>& controllerlooping,
              float powersetpoint,
              logging_sys& logsys)
{
    std::cout << "Proposed running\n";

    auto start = std::chrono::steady_clock::now();
    int controlperiod = 0;
    int N = numtasks;

    float* delta = new float[N + 1];
    float* error = new float[N + 1];

    float* rtr = new float[N];
    float* upperbound = new float[N];
    float* lowerbound = new float[N];
    float* executiontime = new float[N];
    float* responsetime = new float[N];
    float* desiredresponsetime = new float[N];
    float* taskperiod = new float[N];
    float* newtaskperiod = new float[N];
    float* deadlinemiss = new float[N];

    std::thread freqthread;

    float measuredpower = 0.0f;
    float measuredpowererror = 0.0f;

    average average;
    std::ofstream ctrlLog("logs/controller.txt", std::ios::app);

    std::filesystem::path base_path(path);
    std::vector<std::string> exec_paths(N);
    std::vector<std::string> resp_paths(N);

    for (int i = 0; i < N; ++i) {
        exec_paths[i] = (base_path / ("taskexecutiontime" + std::to_string(i) + ".txt")).string();
        resp_paths[i] = (base_path / ("taskresponsetime" + std::to_string(i) + ".txt")).string();
    }

    while (controllerlooping.load(std::memory_order_relaxed))
    {
        std::this_thread::sleep_for(std::chrono::duration<float>(innerloopduration));

        stopsigma.store(true, std::memory_order_relaxed);
        if (freqthread.joinable()) {
            freqthread.join();
        }

        auto end = std::chrono::steady_clock::now();
        std::cout << "Control Period = " << controlperiod << "\n";

        for (int i = 0; i < N; i++) {
            std::vector<float> local_rtr, local_resp, local_per, local_exec;
            {
                std::lock_guard<std::mutex> lk(log[i].mtx);
                local_rtr  = std::move(log[i].rtr);
                local_resp = std::move(log[i].response);
                local_per  = std::move(log[i].period);
                local_exec = std::move(log[i].exec);
            }
            rtr[i]           = average.calculateAverage(local_rtr);
            responsetime[i]  = average.calculateAverage(local_resp);
            taskperiod[i]    = average.calculateAverage(local_per);
            executiontime[i] = average.calculateAverage(local_exec);


            //t[i].setpoint is RTR target, convert it into a response time target
            desiredresponsetime[i] = t[i].setpoint * taskperiod[i];

            deadlinemiss[i] = calculate_deadline_miss(local_resp, local_per, 0.00f);

            //LQR state/error with rt, not rtr 
            error[i] = desiredresponsetime[i] - responsetime[i];
            
            #ifdef DEBUG 
                std::cout << "Size" << i
                          << " samples: rtr=" << local_rtr.size()
                          << " resp=" << local_resp.size()
                          << " period=" << local_per.size()
                          << " exec=" << local_exec.size()
                          << "\n";
            #endif
            if (append_lock(exec_paths[i], "Control Period") < 0) cout << "Error \n\nin proposed\n\n";            
            if (append_lock(resp_paths[i], "Control Period") < 0) cout << "Error \n\nin proposed\n\n";
        }

        std::vector<float> local_power;
        {
            local_power = std::move(logsys.power.measuredpower);
        }

        measuredpower = average.calculateAverage(local_power);
        measuredpowererror = powersetpoint - measuredpower;
        error[N] = measuredpowererror;

        std::cout << "Measured Power = " << measuredpower
                  << " Error = " << measuredpowererror << "\n";

        for (int i = 0; i < N + 1; i++) {
            delta[i] = 0.0f;
            for (int j = 0; j < N + 1; j++) {
                delta[i] += error[j] * (-gain[i][j]) ;
            }
            #ifdef DEBUG
                std::cout << "delta[" << i << "] = " << delta[i] << "\n";
            #endif
        }

        for (int i = 0; i < N; i++)
        {
            // taskperiod is in ms here
            newtaskperiod[i] = taskperiod[i] + delta[i];

            // sharedData expects seconds
            newtaskperiod[i] = newtaskperiod[i] / 1000.0f;
        }

        #ifdef CLAMP_PERIODS
            for (int i = 0; i < N; i++) {
                float initial = t[i].initial_rate;   // assumed to be in seconds
                float min_p = initial * (1 - BOUND_VALUE);
                float max_p = initial * (1 + BOUND_VALUE);
                float original = newtaskperiod[i];
                float clamped = std::clamp(original, min_p, max_p);

                sharedData->newperiods[i] = clamped;
                upperbound[i] = max_p;
                lowerbound[i] = min_p;

                if (clamped != original) {
                    #ifdef DEBUG
                        printf("Task %d clamped: original=%f clamped=%f min=%f max=%f\n",
                           i, original, clamped, min_p, max_p);
                    #endif
                }
            }
        #else
            for (int i = 0; i < N; i++) {
                sharedData->newperiods[i] = newtaskperiod[i];
                //upperbound[i] = 0.0f;
                //lowerbound[i] = 0.0f;
            }
        #endif

            float measured = current_freq.load(std::memory_order_relaxed);
            float delta_freq = delta[N];
            float next = measured + delta_freq;

        #ifdef Jetson
            next = std::clamp(next, 306.0f, 1020.0f);
        #else
            next = std::clamp(next, 210.0f, 2100.0f);
        #endif

        current_freq.store(next, std::memory_order_relaxed);

        std::cout << "Current freq (SM) = " << measured << " MHz\n";
        std::cout << "Delta freq        = " << delta_freq << " MHz\n";
        std::cout << "New freq command  = " << next << " MHz\n";

        stopsigma.store(false, std::memory_order_relaxed);
        freqthread = std::thread(sigmadelta_freq_step, next);

        double ts = std::chrono::duration<double>(end - start).count();
        std::cout << "time=" << ts << "  control_period=" << controlperiod << "\n";
        ctrlLog  << "time=" << ts << "  control_period=" << controlperiod << "\n";

        std::cout << "task\t\tex(ms)\t\tRS(ms)\t\tRSn(ms)\t\trtr\n";
        //std::cout << "task\t\texec(ms)\t\tresp(ms)\t\tdes_resp(ms)\t\trtr\t\terror\t\tPcur(ms)\t\tPnext(ms)\tDeadline miss (%)\n";
        ctrlLog  << "task\texec(ms)\tresp(ms)\tdes_resp(ms)\trtr\terror\tPcur(ms)\tPnext(ms)\tDMR(%)\n";

        std::cout << std::fixed << std::setprecision(3);
        ctrlLog  << std::fixed << std::setprecision(3);
        for (int i = 0; i < N; i++)
        {
            std::cout << "T" << (i + 1) << "\t\t"
                      << executiontime[i] << "\t\t"
                      << responsetime[i]  << "\t\t"
                      << desiredresponsetime[i] << "\t\t"
                      << rtr[i]           << "\t\n";
        }
        std::cout << "task\t\terr\t\tP(ms)\t\tPn(ms)\tDeadline miss (%)\n";

        for (int i = 0; i < N; i++)
        {
            std::cout << "T" << (i + 1) << "\t\t"
                      << error[i]         << "\t\t"
                      << taskperiod[i]    << "\t\t"
                      << newtaskperiod[i] * 1000.0f << "\t\t"
                      << deadlinemiss[i]  << "\n";

            ctrlLog  << "T" << (i + 1) << "\t"
                     << executiontime[i] << "\t"
                     << responsetime[i]  << "\t"
                     << desiredresponsetime[i] << "\t"
                     << rtr[i]           << "\t"
                     << error[i]         << "\t"
                     << taskperiod[i]    << "\t"
                     << newtaskperiod[i] * 1000.0f << "\t"
                     << deadlinemiss[i]  << "\n";

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

        logsys.power.powersetpoints.push_back(powersetpoint);
        logsys.power.powerseterror.push_back(measuredpowererror);
        logsys.power.powercontrolperiod.push_back(measuredpower);

        ctrlLog << "Power\t" << measuredpower
                << "\tPerror\t" << measuredpowererror
                << "\tDeltaFreq\t" << delta_freq << "\n";

        std::cout << "*****************************************************************\n\n";
        ctrlLog  << "*****************************************************************\n\n";
        ctrlLog.flush();

        controlperiod++;
    }

    stopsigma.store(true, std::memory_order_relaxed);
    if (freqthread.joinable()) {
        freqthread.join();
    }

    delete[] delta;
    delete[] error;
    delete[] rtr;
    delete[] upperbound;
    delete[] lowerbound;
    delete[] executiontime;
    delete[] responsetime;
    delete[] desiredresponsetime;
    delete[] taskperiod;
    delete[] newtaskperiod;
    delete[] deadlinemiss;
}


void FC_GPU_and_power( string path, logging& log, tasks& t,int numtasks,float** doublegain,std::atomic<bool>& controllerlooping,float powersetpoint,
    logging_sys& logsys)
{
    std::cout << "[FC_GPU+Power] running with " << numtasks << " tasks\n";

    auto start = std::chrono::steady_clock::now();
    int controlperiod = 0;
    int N = numtasks;

    float* delta = new float[N + 1];        
    float* error = new float[N + 1];          
    float* rtr = new float[N];
    float* upperbound = new float[N];
    float* lowerbound = new float[N];
    float* executiontime = new float[N];
    float* responsetime = new float[N];
    float* taskperiod = new float[N];
    float* newtaskperiod = new float[N];
    float* deadlinemiss = new float[N];
    std::thread freqthread;

    float measuredpower = 0.0f;
    float measuredpowererror = 0.0f;

    average average;
    std::ofstream ctrlLog("logs/controller.txt", std::ios::app);
    std::filesystem::path base_path(path);
    std::vector<std::string> exec_paths(N);
    std::vector<std::string> resp_paths(N);

    for (int i = 0; i < N; ++i) {
        exec_paths[i] = (base_path / ("taskexecutiontime" + std::to_string(i) + ".txt")).string();
        resp_paths[i] = (base_path / ("taskresponsetime" + std::to_string(i) + ".txt")).string();
    }

    while (controllerlooping.load(std::memory_order_relaxed))
    {
        std::this_thread::sleep_for(std::chrono::duration<float>(innerloopduration));

        stopsigma.store(true, std::memory_order_relaxed);
        if (freqthread.joinable()) {
            freqthread.join();
        }

        auto end = std::chrono::steady_clock::now();
        std::cout << "Control Period = " << controlperiod << "\n";

        for (int i = 0; i < N; i++) {
            std::vector<float> local_rtr, local_resp, local_per, local_exec;
            {
                std::lock_guard<std::mutex> lk(log[i].mtx);
                local_rtr  = std::move(log[i].rtr);
                local_resp = std::move(log[i].response);
                local_per  = std::move(log[i].period);
                local_exec = std::move(log[i].exec);
            }

            rtr[i]           = average.calculateAverage(local_rtr);
            responsetime[i]  = average.calculateAverage(local_resp);
            taskperiod[i]    = average.calculateAverage(local_per);
            executiontime[i] = average.calculateAverage(local_exec);

            std::cout << "Size" << i
                      << " samples: rtr=" << local_rtr.size()
                      << " resp=" << local_resp.size()
                      << " period=" << local_per.size()
                      << " exec=" << local_exec.size()
                      << "\n";

            deadlinemiss[i] = calculate_deadline_miss(local_resp, local_per, 0.00f);

            error[i] = t[i].setpoint - rtr[i];

            if (append_lock(exec_paths[i], "Control Period") < 0) cout << "Error in n+1";
            if (append_lock(resp_paths[i], "Control Period") < 0)  cout << "Error in n+1";

        }

        std::vector<float> local_power;
        {
            local_power = std::move(logsys.power.measuredpower);
        }

        measuredpower = average.calculateAverage(local_power);
        measuredpowererror = powersetpoint - measuredpower;
        error[N] = measuredpowererror;
        std::cout << "Measured Power = " << measuredpower
                  << " Error = " << measuredpowererror << "\n";

        for (int i = 0; i < N + 1; i++) {
            delta[i] = 0.0f;
            for (int j = 0; j < N + 1; j++) {
                delta[i] += error[j] * (-doublegain[i][j]);
            }
            std::cout << "delta[" << i << "] = " << delta[i] << "\n";
        }

        for (int i = 0; i < N; i++)
        {
            newtaskperiod[i] = taskperiod[i] + delta[i];
            newtaskperiod[i] = newtaskperiod[i] / 1000.0f;  // sharedData expects seconds
        }

#ifdef CLAMP_PERIODS
        for (int i = 0; i < N; i++) {
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
        for (int i = 0; i < N; i++) {
            sharedData->newperiods[i] = newtaskperiod[i];
        }
#endif

        float measured = current_freq.load(std::memory_order_relaxed);
        float delta_freq = delta[N];
        float next = measured + delta_freq;
        #ifdef Jetson
            next = std::clamp(next, 306.0f, 1020.0f); // change to jetson frew
        #else
            next = std::clamp(next, 210.0f, 2100.0f);
        #endif
        current_freq.store(next, std::memory_order_relaxed);

        std::cout << "Current freq (SM) = " << measured << " MHz\n";
        std::cout << "Delta freq        = " << delta_freq << " MHz\n";
        std::cout << "New freq command  = " << next << " MHz\n";

        stopsigma.store(false, std::memory_order_relaxed);
        freqthread = std::thread(sigmadelta_freq_step, next);

        double ts = std::chrono::duration<double>(end - start).count();
        std::cout << "time=" << ts << "  control_period=" << controlperiod << "\n";
        ctrlLog  << "time=" << ts << "  control_period=" << controlperiod << "\n";

        std::cout << "task\texec(ms)\tresp(ms)\trtr\terror\tPcur(ms)\tPnext(ms)\tDeadline miss (%)\n";
        ctrlLog  << "task\texec(ms)\tresp(ms)\trtr\terror\tPcur(ms)\tPnext(ms)\tDeadline miss (%)\n";

        std::cout << std::fixed << std::setprecision(3);
        ctrlLog  << std::fixed << std::setprecision(3);

        for (int i = 0; i < N; i++)
        {
            std::cout << "T" << (i + 1) << "\t"
                      << executiontime[i] << "\t\t"
                      << responsetime[i]  << "\t\t"
                      << rtr[i]           << "\t"
                      << error[i]         << "\t"
                      << taskperiod[i]    << "\t"
                      << newtaskperiod[i] * 1000.0f << "\t"
                      << deadlinemiss[i]  << "\n";

            ctrlLog  << "T" << (i + 1) << "\t"
                     << executiontime[i] << "\t"
                     << responsetime[i]  << "\t"
                     << rtr[i]           << "\t"
                     << error[i]         << "\t"
                     << taskperiod[i]    << "\t"
                     << newtaskperiod[i] * 1000.0f << "\t"
                     << deadlinemiss[i]  << "\n";

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

        logsys.power.powersetpoints.push_back(powersetpoint);
        logsys.power.powerseterror.push_back(measuredpowererror);
        logsys.power.powercontrolperiod.push_back(measuredpower);

        ctrlLog << "Power\t" << measuredpower
                << "\tPerror\t" << measuredpowererror
                << "\tDeltaFreq\t" << delta_freq << "\n";

        std::cout << "*****************************************************************\n\n";
        ctrlLog  << "*****************************************************************\n\n";
        ctrlLog.flush();

        controlperiod++;
    }

    stopsigma.store(true, std::memory_order_relaxed);
    if (freqthread.joinable()) {
        freqthread.join();
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


void FC_GPU(string path,logging& log, tasks& t, int numtasks,float** gain,std::atomic<bool>& controllerlooping, float powersetpoint,
           logging_sys& logsys)
{
    std::cout << "[FC_GPU] running with " << numtasks << " tasks\n";
    auto start = std::chrono::steady_clock::now();
    int controlperiod=0;
    float*delta;
    float*error;
    float*rtr;
    float*measuredpower;
    float*measuredpowererror;
    float*upperbound;
    float*lowerbound;
    float*executiontime;
    float*responsetime;
    float*taskperiod;
    float*newtaskperiod;
    float*deadlinemiss;
    measuredpower= new float[1];
    measuredpowererror= new float[1];
    delta= new float[numtasks];
    error=new float[numtasks];
    rtr= new float[numtasks];
    executiontime=new float[numtasks];
    responsetime=new float[numtasks];
    taskperiod=new float[numtasks];
    newtaskperiod=new float[numtasks];
    deadlinemiss=new float[numtasks];
    upperbound=new float[numtasks];
    lowerbound=new float[numtasks];
    int lastracker=0;
    average average;
    std::ofstream ctrlLog("logs/controller.txt", std::ios::app);
    std::filesystem::path base_path(path);
    std::vector<std::string> exec_paths(numtasks);
    std::vector<std::string> resp_paths(numtasks);

    for (int i = 0; i < numtasks; ++i) {
        exec_paths[i] = (base_path / ("taskexecutiontime" + std::to_string(i) + ".txt")).string();
        resp_paths[i] = (base_path / ("taskresponsetime" + std::to_string(i) + ".txt")).string();
    }

    while (controllerlooping.load(std::memory_order_relaxed))
    {
        std::this_thread::sleep_for(std::chrono::duration<float>(innerloopduration));
        auto end = std::chrono::steady_clock::now();
        std::cout<<"Control Period = "<<controlperiod<<"\n";
        for (int i = 0; i < numtasks; i++) {
            std::vector<float> local_rtr, local_resp, local_per, local_exec;
            {
                std::lock_guard<std::mutex> lk(log[i].mtx);
                local_rtr  = std::move(log[i].rtr);
                local_resp = std::move(log[i].response);
                local_per  = std::move(log[i].period);
                local_exec = std::move(log[i].exec);
                // log[i].* are now empty
            }// unlock here
        

            rtr[i]          = average.calculateAverage(local_rtr);
            responsetime[i] = average.calculateAverage(local_resp);
            taskperiod[i]   = average.calculateAverage(local_per);
            executiontime[i]= average.calculateAverage(local_exec);
            std::cout << "Size" << i
              << " samples: rtr=" << local_rtr.size()
              << " resp=" << local_resp.size()
              << " period=" << local_per.size()
              << " exec=" << local_exec.size()
              << "\n";
            float deadline=calculate_deadline_miss(local_resp,local_per,0.00);
            deadlinemiss[i]=deadline;
            local_rtr.clear();
            local_resp.clear();
            local_per.clear();
            local_exec.clear();
            error[i] = t[i].setpoint - rtr[i];
            if (append_lock(exec_paths[i], "Control Period") < 0)  cout << "Error \n\nin FC\n\n";            
            if (append_lock(resp_paths[i], "Control Period") < 0) cout << "Error \n\nin FC\n\n";
       
        }
        std::vector<float> loacl_power, local_perror;
        {
        loacl_power=std::move(logsys.power.measuredpower);

        }
        measuredpower[0]= average.calculateAverage(loacl_power);
        measuredpowererror[0]=powersetpoint-measuredpower[0];
        std::cout<<"Meaured Power = "<<measuredpower[0]<<"Error = "<<measuredpowererror[0]<<endl;
        
        
        
        //fc-controller
        for(int i=0; i<numtasks; i++)
            {
            delta[i] = 0.0;
            for(int j=0; j<numtasks; j++)
            {   
             delta[i]= delta[i]+error[j]*-gain[i][j];  
            

            }
            std::cout<<"delta ["<<i<<"]="<<delta[i]<<endl;
        }
        
        
             //fc-actuator

        float sumP = 0.0f;
        for (int i = 0; i < numtasks; i++)
            sumP += taskperiod[i];

        for (int i = 0; i < numtasks; i++) 
        {
            float sumOthers = sumP - taskperiod[i];
            float denom = 1.0f + taskperiod[i] * sumOthers * delta[i];
            std::cout<<"denom = \t"<<denom<<"\n";
            //std::cout<<taskperiod[i] / denom<<"\n";
            //newtaskperiod[i]=taskperiod[i]+ (taskperiod[i] / denom);
            //newtaskperiod[i]=newtaskperiod[i]/1000;
           newtaskperiod[i]=taskperiod[i]+delta[i];
           newtaskperiod[i]=newtaskperiod[i]/1000;
           //newtaskperiod[i] = std::sqrt(newtaskperiod[i]);

        }
        
        #ifdef CLAMP_PERIODS
        for (int i = 0; i < numtasks; i++) {
            float initial = t[i].initial_rate;
            float min_p = initial * (1-BOUND_VALUE);
            float max_p = initial * (1+BOUND_VALUE);
            float original = newtaskperiod[i];
            float clamped = std::clamp(original, min_p, max_p);
            sharedData->newperiods[i] = clamped;
            //sharedData->newperiods[i] = std::clamp(newtaskperiod[i], min_p, max_p);  
            upperbound[i]=max_p;
            lowerbound[i]=min_p;
            if (clamped != original) 
            {
                printf("Task %d clamped: original=%f clamped=%f min=%f max=%f\n",i, original, clamped, min_p, max_p);}
                upperbound[i]=max_p;
                lowerbound[i]=min_p;
            }
        #else
        for (int i = 0; i < numtasks; i++) {
            sharedData->newperiods[i] = newtaskperiod[i];
        }
        #endif
   
        
        

        double ts = std::chrono::duration<double>(end - start).count();
        std::cout << "time=" << ts << "  control_period=" << controlperiod << "\n";
        ctrlLog  << "time=" << ts << "  control_period=" << controlperiod << "\n";

        std::cout << "task\texec(ms)\tresp(ms)\trtr\terror\tPcur(s)\tPnext(ms)\tDeadline miss (%)\n";
        ctrlLog  << "task\texec(ms)\tresp(ms)\trtr\terror\tPcur(s)\tPnext(ms)\tDeadline miss (%)\n";

        std::cout << std::fixed << std::setprecision(3);
        ctrlLog  << std::fixed << std::setprecision(3);

        for (int i = 0; i < numtasks; i++)
        {
            std::cout << "T" << (i + 1) << "\t"
                      << executiontime[i] << "\t\t"
                      << responsetime[i]  << "\t\t"
                      << rtr[i]           << "\t"
                      << error[i]         << "\t"
                      << taskperiod[i]    << "\t"
                      <<newtaskperiod[i]*1000 << "\t"
                      <<deadlinemiss[i]<<"\n";

            ctrlLog  << "T" << (i + 1) << "\t"
                     << executiontime[i] << "\t"
                     << responsetime[i]  << "\t"
                     << rtr[i]           << "\t"
                     << error[i]         << "\t"
                     << taskperiod[i]*1000    << "\t"
                     << newtaskperiod[i] << "\n";

            // Thread-safe append to filewrite buffers
            {
                std::lock_guard<std::mutex> lk(log[i].mtx);
                log[i].filewritexec.push_back(executiontime[i]);
                log[i].filewriteresponse.push_back(responsetime[i]);
                log[i].filewriteperiod.push_back(taskperiod[i]);
                log[i].filewritertr.push_back(rtr[i]);
                log[i].filewritelowperiodbound.push_back(lowerbound[i]*1000);
                log[i].filewritehighperiodbound.push_back(upperbound[i]*1000);
                log[i].filewritedeadlinemiss.push_back(deadlinemiss[i]);
            }
            
        }
        logsys.power.powersetpoints.push_back(powersetpoint);
        logsys.power.powerseterror.push_back(measuredpowererror[0]);
        logsys.power.powercontrolperiod.push_back(measuredpower[0]);
        std::cout << "*****************************************************************\n\n";
        ctrlLog  << "*****************************************************************\n\n";
        ctrlLog.flush();
        controlperiod=controlperiod+1;
      
    
    }
     // for (int i = 0; i < numtasks; ++i) {
       // exec_paths[i].close();
       // resp_paths[i].close();
    //}
 
    delete[]measuredpower;
    delete[]measuredpowererror;
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

void openloop(logging& log, tasks& t, int numtasks, std::atomic<bool>& controllerlooping, float powersetpoint, logging_sys& logsys)
{
    std::cout << "[openloop] running\n";
    auto start = std::chrono::steady_clock::now();
    int controlperiod=0;
    float*delta;
    float*error;
    float*rtr;
    float*executiontime;
    float*responsetime;
    float*taskperiod;
    float*newtaskperiod;
    float*deadlinemiss;
    delta= new float[numtasks];
    error=new float[numtasks];
    rtr= new float[numtasks];
    executiontime=new float[numtasks];
    responsetime=new float[numtasks];
    taskperiod=new float[numtasks];
    newtaskperiod=new float[numtasks];
    deadlinemiss=new float[numtasks];
    int lastracker=0;
    average average;
    std::ofstream ctrlLog("logs/controller.txt", std::ios::app);

    
    while (controllerlooping.load(std::memory_order_relaxed))
    {
        std::this_thread::sleep_for(std::chrono::duration<float>(innerloopduration));
        auto end = std::chrono::steady_clock::now();
        std::cout<<"Control Period = "<<controlperiod<<"\n";
        std::vector<float> local_rtr, local_resp, local_per, local_exec;

        for (int i = 0; i < numtasks; i++)
        {
            {
                std::lock_guard<std::mutex> lk(log[i].mtx);
                local_rtr.swap(log[i].rtr);
                local_resp.swap(log[i].response);
                local_per.swap(log[i].period);
                local_exec.swap(log[i].exec);
            } // unlock here

            rtr[i]          = average.calculateAverage(local_rtr);
            responsetime[i] = average.calculateAverage(local_resp);
            taskperiod[i]   = average.calculateAverage(local_per);
            executiontime[i]= average.calculateAverage(local_exec);
            std::cout << "Size" << i
              << " samples: rtr=" << local_rtr.size()
              << " resp=" << local_resp.size()
              << " period=" << local_per.size()
              << " exec=" << local_exec.size()
              << "\n";
            float deadline=calculate_deadline_miss(local_resp,local_per);
            deadlinemiss[i]=deadline;
            local_rtr.clear();
            local_resp.clear();
            local_per.clear();
            local_exec.clear();
            error[i] = t[i].setpoint - rtr[i];

        }
        double ts = std::chrono::duration<double>(end - start).count();
        std::cout << "time=" << ts << "  control_period=" << controlperiod << "\n";
        ctrlLog  << "time=" << ts << "  control_period=" << controlperiod << "\n";
        std::cout << "task\texec(ms)\tresp(ms)\trtr\terror\tPcur(s)\tPnext(s)\n";
        ctrlLog  << "task\texec(ms)\tresp(ms)\trtr\terror\tPcur(s)\tPnext(s)\n";
        std::cout << std::fixed << std::setprecision(4);
        ctrlLog  << std::fixed << std::setprecision(4);

        for (int i = 0; i < numtasks; i++)
        {
            std::cout << "T" << (i + 1) << "\t"
                      << executiontime[i] << "\t\t"
                      << responsetime[i]  << "\t\t"
                      << rtr[i]           << "\t"
                      << error[i]         << "\t"
                      << taskperiod[i]    << "\t"
                      << newtaskperiod[i] << "\n";

            ctrlLog  << "T" << (i + 1) << "\t"
                     << executiontime[i] << "\t"
                     << responsetime[i]  << "\t"
                     << rtr[i]           << "\t"
                     << error[i]         << "\t"
                     << taskperiod[i]    << "\t"
                     << newtaskperiod[i] << "\n";
            {
                std::lock_guard<std::mutex> lk(log[i].mtx);
                log[i].filewritexec.push_back(executiontime[i]);
                log[i].filewriteresponse.push_back(responsetime[i]);
                log[i].filewriteperiod.push_back(taskperiod[i]);
                log[i].filewritertr.push_back(rtr[i]);
                log[i].filewritedeadlinemiss.push_back(deadlinemiss[i]);
            }
        }
        std::cout << "*****************************************************************\n\n";
        ctrlLog  << "*****************************************************************\n\n";
        ctrlLog.flush();
        controlperiod=controlperiod+1;
    }
    delete[] delta;
    delete[] error;
    delete[] rtr;
    delete[] executiontime;
    delete[] responsetime;
    delete[] taskperiod;
    delete[] newtaskperiod;
    delete[] deadlinemiss;
}

    
void adhoc(logging& log, tasks& t, int numtasks, std::atomic<bool>& controllerlooping, float powersetpoint, logging_sys& logsys)
{
    std::cout << "[adhoc] running\n";
}

void siso(logging& log, tasks& t, int numtasks, std::atomic<bool>& controllerlooping, float powersetpoint, logging_sys& logsys)
{
    std::cout << "[siso] running\n";
}


void monitor(logging* log, logging_sys*plog, int numtasks)
{
    if (!log || !sharedData) return;
    std::cout<<"Begin monitor \n";
    std::vector<float> prevResp(numtasks, std::numeric_limits<float>::quiet_NaN());
    std::vector<float> prevExec(numtasks, std::numeric_limits<float>::quiet_NaN());
    float prevpower;

    using clock = std::chrono::high_resolution_clock;
    while (monitorthread.load())   
    {
        auto start = clock::now();
        std::chrono::duration<double> elapsed(0);
        while (elapsed.count() < innerloopduration && outerlooprunning.load())
        {
            elapsed = clock::now() - start;

            for (int i = 0; i < numtasks; ++i)
            {
                float resp = sharedData->responsetime[i];
                float gpu_exec = sharedData->executiontime[i];
                if (resp != prevResp[i])
                {
                    (*log)[i].add_rt(resp);   
                    (*log)[i].outeradd_rt(resp);         
                    float period_ms = sharedData->newperiods[i];
                    (*log)[i].add_period(period_ms*1000.0f); 
                    (*log)[i].outeradd_period(period_ms); 
                    if (period_ms > 0.0f){
                        (*log)[i].add_rtr(resp / (period_ms * 1000.0f));
                        (*log)[i].outeradd_rtr(resp / (period_ms * 1000.0f));
                    }
                    else{
                        (*log)[i].add_rtr(0.0f);
                        (*log)[i].outeradd_rtr(0.0f);
                    }
                    prevResp[i] = resp;
                }
                float exec = sharedData->executiontime[i];
                if (exec != prevExec[i])
                {
                    (*log)[i].add_exec(exec);
                    (*log)[i].outeradd_exec(exec);
                    prevExec[i] = exec;
                }
            }
            float power = static_cast<float>(returnpowerrtx());
            //if (power != prevpower)
                //{
                (*plog).power.add_measuredpower(power);
                // }
            //prevpower=power;
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }
}






volatile sig_atomic_t readyCount = 0;
void signalHandler(int) {
    readyCount++;
}



using Matrix = Eigen::MatrixXf;
// Discrete Algebraic Riccati Equation (iterative solve)
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

        if (err < tol) {
            return P;
        }
    }

    throw std::runtime_error("solveDARE: Riccati iteration did not converge");
}

// Discrete-time LQR solve
static Matrix dlqr(const Matrix& A,
                   const Matrix& B,
                   const Matrix& Q,
                   const Matrix& R)
{
    Matrix P = solveDARE(A, B, Q, R);
    Matrix K = (R + B.transpose() * P * B).inverse() * (B.transpose() * P * A);
    return K;
}

void compute_LQR_gain(float** gain, int numtasks)
{

    const float base_last_col_full[6] = {
        0.06238100f,
        0.06804800f,
        0.07260400f,
        0.10549000f,
        0.02831200f,
        0.08079800f
    };

    const float base_last_row_full[6] = {
        22.20753300f,
        11.07223000f,
        28.52230300f,
        7.38135200f,
        13.56965500f,
        15.28722400f
    };

    const float base_bottom_right = 0.00116413f;

    const int N = numtasks;
    const int n = N + 1;

    //matrix A
    Matrix A = Matrix::Identity(n, n);
    for (int i = 0; i < n; ++i) {
        A(i, i) = 0.95f;
    }

    //B matrix
    Matrix B = Matrix::Zero(n, n);

    //top-left nxn block (period -> RT error)
    //non-trained diagonal values.
    for (int i = 0; i < N; ++i) {
        B(i, i) = 1.0f;
    }

    //last column (freq -> RT error)
    for (int i = 0; i < N; ++i) {
        B(i, N) = base_last_col_full[i];
    }

    //bottom row  (period -> power error)
    for (int j = 0; j < N; ++j) {
        B(N, j) = base_last_row_full[j];
    }

    //Bottom right value (freq vs power)
    B(N, N) = -base_bottom_right;

    //Q matrix = state penalty
    Matrix Q = Matrix::Zero(n, n);
    for (int i = 0; i < N; ++i) {
        Q(i, i) = 20.0f;   // RT error weight
    }
    Q(N, N) = 10.0f;       // power error weight

    //R matrix = input penalty
    Matrix R = Matrix::Zero(n, n);
    for (int i = 0; i < N; ++i) {
        R(i, i) = 1.0f;    // period adjustment penalty
    }
    R(N, N) = 10.0f;       // frequency adjustment penalty

    //Solve LQR
    Matrix K = dlqr(A, B, Q, R);

    //move to float** gain
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            gain[i][j] = K(i, j);
        }
    }

    #ifdef DEBUG
        std::cout << "numtasks = " << numtasks << "\n\n";
        std::cout << "A =\n" << A << "\n\n";
        std::cout << "B =\n" << B << "\n\n";
        std::cout << "Q =\n" << Q << "\n\n";
        std::cout << "R =\n" << R << "\n\n";
        std::cout << "LQR gain K =\n" << K << "\n\n";
    #endif
}

void compute_LQR_gain_2(float** gain, int numtasks) //this one works for less than 6 tasks
{
    if (numtasks < 1 || numtasks > 6) {
        throw std::runtime_error("compute_lqr_gain_from_identified_B: numtasks must be between 1 and 6.");
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
        22.20753300f,
        11.07223000f,
        28.52230300f,
        7.38135200f,
        13.56965500f,
        15.28722400f
    };

    const float base_bottom_right = 0.00116413f;

    const int N = numtasks;
    const int n = N + 1;

    // -------------------------------------------------------------------------
    // A matrix
    // -------------------------------------------------------------------------
    Matrix A = Matrix::Identity(n, n);
    for (int i = 0; i < n; ++i) {
        A(i, i) = 0.95f;
    }

    // -------------------------------------------------------------------------
    // B matrix
    //
    // Top-left block is still placeholder unless you have measured
    // period-vs-response-time slopes.
    // -------------------------------------------------------------------------
    Matrix B = Matrix::Zero(n, n);

    for (int i = 0; i < N; ++i) {
        B(i, i) = 1.0f;   // replace later with measured -dRT/dT
    }

    for (int i = 0; i < N; ++i) {
        B(i, N) = base_last_col_full[i];
    }

    for (int j = 0; j < N; ++j) {
        B(N, j) = base_last_row_full[j];
    }

    // error_power = Pset - Pmeasured, so this term should be negative
    B(N, N) = -base_bottom_right;

    // -------------------------------------------------------------------------
    // Q and R
    // -------------------------------------------------------------------------
    Matrix Q = Matrix::Zero(n, n);
    for (int i = 0; i < N; ++i) {
        Q(i, i) = 20.0f;
    }
    Q(N, N) = 10.0f;

    Matrix R = Matrix::Zero(n, n);
    for (int i = 0; i < N; ++i) {
        R(i, i) = 1.0f;
    }
    R(N, N) = 10.0f;

    // -------------------------------------------------------------------------
    // Optional scaling to improve conditioning
    //
    // We normalize state and input channels so the Riccati recursion is less
    // sensitive to wildly different magnitudes in B.
    // -------------------------------------------------------------------------
    Matrix Sx = Matrix::Identity(n, n);
    Matrix Su = Matrix::Identity(n, n);

    // State scaling: RT errors ~ ms, power error ~ W
    for (int i = 0; i < N; ++i) {
        Sx(i, i) = 10.0f;   // assume ~10 ms characteristic RT error
    }
    Sx(N, N) = 1.0f;        // assume power error already ~1 W scale

    // Input scaling: period changes ~ ms, freq changes ~ 100 MHz scale
    for (int i = 0; i < N; ++i) {
        Su(i, i) = 1.0f;    // 1 ms period step scale
    }
    Su(N, N) = 100.0f;      // 100 MHz frequency step scale

    Matrix Sx_inv = Sx.inverse();
    Matrix Su_inv = Su.inverse();

    // Scaled system: x = Sx * xs, u = Su * us
    Matrix As = Sx_inv * A * Sx;
    Matrix Bs = Sx_inv * B * Su;

    Matrix Qs = Sx.transpose() * Q * Sx;
    Matrix Rs = Su.transpose() * R * Su;

    // -------------------------------------------------------------------------
    // Finite-horizon backward Riccati recursion
    // -------------------------------------------------------------------------
    const int horizon = 500;
    Matrix P = Qs;
    Matrix K = Matrix::Zero(n, n);

    for (int k = 0; k < horizon; ++k) {
        Matrix M = Rs + Bs.transpose() * P * Bs;

        Eigen::FullPivLU<Matrix> lu(M);
        if (!lu.isInvertible()) {
            throw std::runtime_error("LQR failed: (R + B^T P B) became singular.");
        }

        K = lu.solve(Bs.transpose() * P * As);

        Matrix Pnext = As.transpose() * P * As
                     - As.transpose() * P * Bs * K
                     + Qs;

        // Force symmetry to suppress numerical drift
        P = 0.5f * (Pnext + Pnext.transpose());

        if (!P.allFinite()) {
            throw std::runtime_error("LQR failed: Riccati recursion produced NaN/Inf.");
        }
    }

    // Convert scaled gain back to original coordinates:
    // us = -Ks xs
    // u  = Su us = -Su Ks Sx^{-1} x
    Matrix K_unscaled = Su * K * Sx_inv;

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            gain[i][j] = K_unscaled(i, j);
        }
    }

    std::cout << "numtasks = " << numtasks << "\n\n";
    std::cout << "A =\n" << A << "\n\n";
    std::cout << "B =\n" << B << "\n\n";
    std::cout << "Scaled LQR gain K =\n" << K_unscaled << "\n\n";
}

int main(int argc,char*argv[])
{
    signal(SIGHUP, signalHandler);
    std::cout << "Controller PID: " << getpid() << "\n";
    int numtasks= atoi(argv[1]);
    int solution = std::atoi(argv[2]);
    int duration = std::atoi(argv[3]);
    powersetpoint= std::atof(argv[4]);
    string path= argv[5];
    std::cout<<"Set point power = "<<powersetpoint<<endl;
    int expected_argc = 6 + numtasks * 2;
    if (argc < expected_argc) {
        std::cerr << "ERROR: not enough arguments.\n"
                      << "Expected argc >= " << expected_argc << " but got " << argc << "\n";
        return 1;
    }
    key_t key = ftok("shmfile", 65);
    int shmid = shmget(key, 1024, 0666 | IPC_CREAT);
    bool createdshm;
    if(shmid>0){
        sharedData = (SharedData*)shmat(shmid, nullptr, 0);
        createdshm=true;
    }
    else
    {
        shmid = shmget(key, sizeof(SharedData), 0666);
        sharedData = (SharedData*)shmat(shmid, nullptr, 0);
    }
    std::cout<<path<<endl;
   
    
    
    
    
    
    
    //key_t key = ftok("shmfile", 65);
    //int shmid = shmget(key, sizeof(SharedData), 0666);
    //SharedData* sharedData = (SharedData*)shmat(shmid, nullptr, 0);   
    std::cout<<"tasks = "<<numtasks<<endl;
    float** gain = new float*[numtasks];
    float** doublegain = new float*[numtasks+1];
    float** LQRGain = new float*[numtasks+1];

    for (int i = 0; i < numtasks + 1; ++i) {
        LQRGain[i] = new float[numtasks + 1];
    }
    if (solution ==3) compute_LQR_gain_2(LQRGain, numtasks);
    //compute_LQR_gain(LQRGain, numtasks);
    /*
    
.00000000   0.00000000   0.00000000   0.00000000   0.00000000   0.00000000  -0.06238100
  0.00000000   1.00000000   0.00000000   0.00000000   0.00000000   0.00000000  -0.06804800
  0.00000000   0.00000000   1.00000000   0.00000000   0.00000000   0.00000000  -0.07260400
  0.00000000   0.00000000   0.00000000   1.00000000   0.00000000   0.00000000  -0.10549000
  0.00000000   0.00000000   0.00000000   0.00000000   1.00000000   0.00000000  -0.02831200
  0.00000000   0.00000000   0.00000000   0.00000000   0.00000000   1.00000000  -0.08079800
-22.20753300 -11.07223000 -28.52230300  -7.38135200 -13.56965500 -15.28722400   0.00116413

    */
    
    for (int i = 0; i < numtasks + 1; ++i) {
        doublegain[i] = new float[numtasks + 1];
    }

    // values from your 7x7 example
    // these are used for the LAST COLUMN and LAST ROW
    const float base_last_col[6] = {
        0.06238100f,
        0.06804800f,
        0.07260400f,
        0.10549000f,
        0.02831200f,
        0.08079800f
    };

    const float base_last_row[6] = {
        22.20753300f,
        11.07223000f,
        28.52230300f,
        7.38135200f,
        13.56965500f,
        15.28722400f
    };

    
    const float base_bottom_right = 0.00116413f;

//     for(int i = 0; i < numtasks; ++i) {
//         doublegain[i] = new float[numtasks];
//     }    
    
    
    for (int i = 0; i < numtasks + 1; ++i) {
    for (int j = 0; j < numtasks + 1; ++j) {
        doublegain[i][j] = 0.0f;
    }
    }

    // top-left NxN block = your regular rule
    for (int i = 0; i < numtasks; ++i) {
        for (int j = 0; j < numtasks; ++j) {
            if (i == j) {
                doublegain[i][j] = (1.25f * 1.0f) * 1.0f * 3.0f;
            } else {
                doublegain[i][j] = 0.1f * 2.0f * 1.5f / 1.0f;
            }
        }
    }

    for (int i = 0; i < numtasks; ++i) {
        doublegain[i][numtasks] = base_last_col[i];
    }

    // last row: use posted values
    for (int j = 0; j < numtasks; ++j) {
        doublegain[numtasks][j] = base_last_row[j];
    }

    // bottom-right element
    doublegain[numtasks][numtasks] = base_bottom_right;
    for (int i = 0; i < numtasks + 1; ++i) {
        for (int j = 0; j < numtasks + 1; ++j) {
            doublegain[i][j] = doublegain[i][j] / 10;
        }
    }





    //fc-gpu gain
    for(int i = 0; i < numtasks; ++i) {
        gain[i] = new float[numtasks];
    }    
    
    for(int i = 0; i < numtasks; ++i) {
        for(int j = 0; j < numtasks; ++j){
            if( i==j){
            //gain[i][j]=(1.25*1)*1*4;
            gain[i][j]=(1.25*1)*1*4*2;
            }
            else{
            //gain[i][j]=1*2*1.5/(1);
            gain[i][j]=2*2*1.5/(1);
             }
        }
    }
    
    //gain[0][0]=9;
    //gain[1][1]=5.5;
   
    
//     for(int i = 0; i < numtasks; ++i) {
//         for(int j = 0; j < numtasks; ++j){
//         if( i==j){

//             gain[i][j]=0.35*1;
//         }
//             else
//             {
//             gain[i][j]=0*1;

//         }}}
    
//             else
//             {
            
//                         gain[i][j]=(3.5*1.2*5)/1;

//         }
             
//         }
//     }
    //*/
    
    for(int i = 0; i < numtasks; ++i) {
        for(int j = 0; j < numtasks; ++j){
           std::cout<<gain[i][j]<<"\t";
        }
         std::cout<<endl;
    }
    
    
    //std::atomic<float> current_freq{1200.0f};
    #ifdef Jetson
        std::atomic<float> current_freq{918.0f};
    #else
        std::atomic<float> current_freq{1800.0f};
    #endif
    activate_freq_nvidia_rtx(current_freq);
    logging log(numtasks,solution);
    logging_sys logsys(solution);
    tasks task(numtasks);  
    int arg_index = 6;  
    for (int i = 0; i < numtasks; i++) {
        //task[i].pid          = std::atoi(argv[arg_index++]);
        task[i].initial_rate = std::atof(argv[arg_index++]);
        task[i].setpoint     = std::atof(argv[arg_index++]);
    }
    std::cout<<"Power Set point = \t"<<powersetpoint<<endl;
    for (int i = 0; i < numtasks; i++) {
         std::cout << "Task " << i
          //<< " PID=" << task[i].pid
          << " Rate=" << task[i].initial_rate*1000
          << " Setpoint=" << task[i].setpoint
          << "\n";
    }
    
    std::unordered_map<int, std::string> mappings = {
            {1, "FC_GPU"},
            {2, "FC_GPU+1"},
            {3, "Proposed"},
            {4, "OpenLoop"},
            {5, "Adhoc"},
            {6, "SISO"},
        };
    
    std::cout << "Solution = " << (mappings.count(solution) ? mappings[solution] : "UNKNOWN") << "\n";
    
    while (readyCount < numtasks) 
    {
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

    std::cout<<"Begining \n";
    std::thread monitorThread(monitor, &log, &logsys, numtasks);
    std::thread monitorpowerthread(monitorpower);
    std::thread monitorfreqthread(monitorfreq);
    std::thread controllerThread;
    std::thread controllerThread2;
    std::this_thread::sleep_for(std::chrono::seconds(5));
    switch (solution) {
        case 1:
            controllerThread = std::thread(FC_GPU, path, std::ref(log), std::ref(task), numtasks, gain,std::ref(controllerlooping),powersetpoint,std::ref(logsys));
            break;
        case 2:
            controllerThread = std::thread(proposed, path, std::ref(log), std::ref(task), numtasks, doublegain,std::ref(controllerlooping),powersetpoint,std::ref(logsys));
            break;
        case 3:
            controllerThread = std::thread(proposed, path, std::ref(log), std::ref(task), numtasks, LQRGain,std::ref(controllerlooping),powersetpoint,std::ref(logsys));
            break;
        case 4:
            controllerThread = std::thread(openloop, std::ref(log), std::ref(task), numtasks, std::ref(controllerlooping),powersetpoint,std::ref(logsys));
            break;
        case 5:
            controllerThread = std::thread(adhoc, std::ref(log), std::ref(task), numtasks, std::ref(controllerlooping),powersetpoint,std::ref(logsys));
            break;
        case 6:
            controllerThread = std::thread(siso, std::ref(log), std::ref(task), numtasks, std::ref(controllerlooping),powersetpoint,std::ref(logsys));
            break;
        default:
            std::cerr << "ERROR: unknown solution id " << solution << "\n";
            outerlooprunning.store(false);
            controllerlooping.store(false);
            if (monitorThread.joinable()) monitorThread.join();
            return 1;
    }
   
    auto start = std::chrono::steady_clock::now();
    while (true) {
        auto now = std::chrono::steady_clock::now();
        std::chrono::duration<double> elapsed = now - start;
        if (10+elapsed.count() >= duration)
            break;

        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    // 1) Stop all controller/monitor loops so joins won't hang
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
     for (int i = 0; i < numtasks+1; ++i) {
        delete [] doublegain[i];
     }
    delete[] doublegain;
    log.dump_files(path);
    logsys.dump_files(path);
    log.dump_power(path, finalpowervalues);
    log.dump_freq(path, finalfrequencyvalues);
    if (createdshm == true) {
        shmdt(sharedData);
        shmctl(shmid, IPC_RMID, NULL);
    }

    return 0;
}
