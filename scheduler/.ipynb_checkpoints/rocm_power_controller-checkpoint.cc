#include <rocm_smi/rocm_smi.h>
#include <iostream>
#include <vector>
#include <unistd.h> 
#include <unordered_map>
#include <iomanip>

class rocm_interface {
    public:
        rocm_interface();
        ~rocm_interface();
        double get_power();
        double get_freq();
        void set_freq(int target_level);
        void print_map();

    private:
        uint32_t dev_idx;
        std::unordered_map<int, double> sclk_map;
};

rocm_interface::rocm_interface() {
    dev_idx = 0;
    if (rsmi_init(0) != RSMI_STATUS_SUCCESS) {
        std::cerr << "ROCm SMI Init Failed" << std::endl;
        exit(1);
    }
    
    rsmi_frequencies_t freq_info;   
    if (rsmi_dev_gpu_clk_freq_get(dev_idx, RSMI_CLK_TYPE_SYS, &freq_info) == RSMI_STATUS_SUCCESS) {
        for (uint32_t i = 0; i < freq_info.num_supported; ++i) {
            double mhz = static_cast<double>(freq_info.frequency[i]) / 1000000.0;
            sclk_map[i] = mhz;
        }
    }
    rsmi_dev_perf_level_set(dev_idx, RSMI_DEV_PERF_LEVEL_MANUAL);
}

rocm_interface::~rocm_interface() {
    rsmi_shut_down();
}

void rocm_interface::print_map() {
    std::cout << "\n--- SCLK Level to Frequency Map ---" << std::endl;
    for (int i = 0; i < sclk_map.size(); ++i) {
        std::cout << "Level " << std::setw(2) << i << ": " << sclk_map[i] << " MHz" << std::endl;
    }
    std::cout << "-----------------------------------\n" << std::endl;
}

double rocm_interface::get_power() {
    uint64_t power_uw = 0;
    if (rsmi_dev_power_ave_get(dev_idx, 0, &power_uw) == RSMI_STATUS_SUCCESS) {
        return (power_uw / 1000000.0);
    }
    return 0.0;
}

double rocm_interface::get_freq() {
    rsmi_frequencies_t freq;
    if (rsmi_dev_gpu_clk_freq_get(dev_idx, RSMI_CLK_TYPE_SYS, &freq) == RSMI_STATUS_SUCCESS) {
        return (freq.frequency[freq.current] / 1000000.0);
    }
    return 0.0;
}

void rocm_interface::set_freq(int target_level) {
    rsmi_dev_gpu_clk_freq_set(dev_idx, RSMI_CLK_TYPE_SYS, (1ULL << target_level));
}

int main() {
    if (geteuid() != 0) {
        std::cerr << "ERROR: This program must be run with sudo/root privileges." << std::endl;
        return 1;
    }

    rocm_interface gpu;
    gpu.print_map();
    std::cout << "Initial Freq: " << gpu.get_freq() << " MHz" << std::endl;
    std::cout << "Initial Power: " << gpu.get_power() << " W" << std::endl;
    gpu.set_freq(2);
    usleep(100000);
    std::cout << "New Freq: " << gpu.get_freq() << " MHz" << std::endl;
    std::cout << "New Power: " << gpu.get_power() << " W" << std::endl;
    return 0;
}