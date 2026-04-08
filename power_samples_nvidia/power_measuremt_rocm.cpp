#include <rocm_smi/rocm_smi.h>
#include <iostream>
#include <vector>
#include <unistd.h> // for usleep

void print_status(uint32_t dev_idx) {
    rsmi_frequencies_t freq;
    uint64_t power_uw;
    
    // 1. Get Power
    if (rsmi_dev_power_ave_get(dev_idx, 0, &power_uw) == RSMI_STATUS_SUCCESS) {
        std::cout << "  > Power Usage: " << (power_uw / 1000000.0) << " W" << std::endl;
    }

    // 2. Get Clock Info
    if (rsmi_dev_gpu_clk_freq_get(dev_idx, RSMI_CLK_TYPE_SYS, &freq) == RSMI_STATUS_SUCCESS) {
        std::cout << "  > Current SCLK Index: " << freq.current << std::endl;
        std::cout << "  > Current Frequency:  " << (freq.frequency[freq.current] / 1000000.0) << " MHz" << std::endl;
    }
    std::cout << "------------------------------------------" << std::endl;
}

int main() {
    // Initialize ROCm SMI
    if (rsmi_init(0) != RSMI_STATUS_SUCCESS) {
        std::cerr << "Failed to initialize ROCm SMI. Are drivers installed?" << std::endl;
        return 1;
    }

    uint32_t dev_idx = 0;
    rsmi_frequencies_t freq_info;

    // --- 1. SHOW SCLK RANGE ---
    std::cout << "=== GPU [0] SUPPORTED SCLK RANGE ===" << std::endl;
    if (rsmi_dev_gpu_clk_freq_get(dev_idx, RSMI_CLK_TYPE_SYS, &freq_info) == RSMI_STATUS_SUCCESS) {
        for (uint32_t i = 0; i < freq_info.num_supported; ++i) {
            std::cout << "Level [" << i << "]: " << (freq_info.frequency[i] / 1000000.0) << " MHz";
            if (i == freq_info.current) std::cout << " <-- CURRENT";
            std::cout << std::endl;
        }
    }
    std::cout << "------------------------------------------" << std::endl;

    // --- 2. MEASURE INITIAL STATE ---
    std::cout << "\n[Step 1] Initial State:" << std::endl;
    print_status(dev_idx);

    // --- 3. CHANGE FREQUENCY ---
    // Target Level 3 as an example (adjust based on your range output)
    uint32_t target_level = 3; 
    std::cout << "[Step 2] Setting Performance Level to MANUAL..." << std::endl;

    std::cout << "[Step 3] Requesting SCLK Level " << target_level << "..." << std::endl;

    if (ret != RSMI_STATUS_SUCCESS) {
        std::cerr << "!! Error setting frequency. Try running with sudo. Error code: " << ret << std::endl;
    } else {
        // Wait a moment for hardware to stabilize
        usleep(100000); 

        // --- 4. MEASURE FINAL STATE ---
        std::cout << "[Step 4] Final State after change:" << std::endl;
        print_status(dev_idx);
    }

    // Cleanup
    rsmi_shut_down();
    return 0;
}