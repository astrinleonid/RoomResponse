#pragma once

#include <SDL.h>
#include <vector>
#include <string>
#include <memory>
#include <map>

namespace sdl_audio {

struct DeviceInfo {
    int id;
    std::string name;
    bool is_input;
    SDL_AudioSpec spec;
    bool is_available;

    // Extended information
    std::vector<int> supported_sample_rates;
    std::vector<SDL_AudioFormat> supported_formats;
    int min_channels;
    int max_channels;

    DeviceInfo() : id(-1), is_input(false), is_available(false), min_channels(1), max_channels(1) {
        SDL_zero(spec);
    }

    DeviceInfo(int device_id, const std::string& device_name, bool input)
        : id(device_id), name(device_name), is_input(input), is_available(true)
        , min_channels(1), max_channels(1) {
        SDL_zero(spec);
    }
};

struct DeviceCapabilities {
    bool supports_simultaneous_io;
    bool supports_full_duplex;
    bool has_hardware_monitoring;
    bool supports_exclusive_mode;
    int min_buffer_size;
    int max_buffer_size;
    int preferred_buffer_size;
    double latency_estimate_ms;

    DeviceCapabilities()
        : supports_simultaneous_io(false)
        , supports_full_duplex(false)
        , has_hardware_monitoring(false)
        , supports_exclusive_mode(false)
        , min_buffer_size(64)
        , max_buffer_size(8192)
        , preferred_buffer_size(1024)
        , latency_estimate_ms(0.0) {}
};

class DeviceManager {
public:
    struct DevicePairScore {
        int input_id;
        int output_id;
        std::string input_name;
        std::string output_name;
        double score;
        bool can_open_both;
        bool has_acoustic_coupling;
        double coupling_strength;
        std::string notes;

        DevicePairScore() : input_id(-1), output_id(-1), score(0.0),
                          can_open_both(false), has_acoustic_coupling(false),
                          coupling_strength(0.0) {}
    };

private:
    std::vector<DeviceInfo> input_devices_;
    std::vector<DeviceInfo> output_devices_;
    std::map<int, DeviceCapabilities> device_capabilities_;

    bool is_initialized_;
    std::string last_error_;

    // Internal methods
    void refresh_device_list();
    DeviceCapabilities probe_device_capabilities(const DeviceInfo& device);
    bool test_device_opening(int device_id, bool is_input);
    std::vector<int> probe_sample_rates(int device_id, bool is_input);
    std::vector<SDL_AudioFormat> probe_formats(int device_id, bool is_input);

public:
    DeviceManager();
    ~DeviceManager();

    // Initialization
    bool initialize();
    void shutdown();
    bool is_initialized() const { return is_initialized_; }

    // Device enumeration
    const std::vector<DeviceInfo>& get_input_devices();
    const std::vector<DeviceInfo>& get_output_devices();
    void refresh_devices();

    // Device information
    DeviceInfo get_device_info(int device_id, bool is_input);
    DeviceCapabilities get_device_capabilities(int device_id, bool is_input);
    std::string get_device_driver(int device_id, bool is_input);

    // Default devices
    DeviceInfo get_default_input_device();
    DeviceInfo get_default_output_device();

    // Device testing
    bool test_device_pair(int input_id, int output_id);
    std::vector<DevicePairScore> score_all_device_pairs();
    DevicePairScore find_best_device_pair();

    // Acoustic coupling testing
    bool test_acoustic_coupling(int input_id, int output_id, double* coupling_strength = nullptr);

    // Device recommendations
    std::vector<DevicePairScore> get_recommended_pairs(int max_results = 5);
    std::vector<DeviceInfo> get_professional_audio_interfaces();
    std::vector<DeviceInfo> get_builtin_devices();

    // Utility methods
    std::string get_last_error() const { return last_error_; }
    void clear_error() { last_error_.clear(); }

    // Static utility functions
    static std::string format_to_string(SDL_AudioFormat format);
    static std::string spec_to_string(const SDL_AudioSpec& spec);
    static bool is_professional_interface(const std::string& device_name);
    static bool is_builtin_device(const std::string& device_name);

    // Debug and diagnostics
    void print_all_devices() const;
    void print_device_capabilities(int device_id, bool is_input) const;
    std::string generate_device_report() const;

    // Disable copy/move
    DeviceManager(const DeviceManager&) = delete;
    DeviceManager& operator=(const DeviceManager&) = delete;
    DeviceManager(DeviceManager&&) = delete;
    DeviceManager& operator=(DeviceManager&&) = delete;

private:
    void set_error(const std::string& error) { last_error_ = error; }
    void log(const std::string& message) const;
    void log_error(const std::string& message) const;
};

// Standalone utility functions
namespace device_utils {
    // Score calculation for device pairs
    double calculate_device_pair_score(const DeviceInfo& input, const DeviceInfo& output,
                                     bool can_open_both, bool has_coupling, double coupling_strength);

    // Device name analysis
    bool devices_likely_same_hardware(const std::string& input_name, const std::string& output_name);
    bool is_usb_device(const std::string& device_name);
    bool is_bluetooth_device(const std::string& device_name);

    // Common sample rates and formats
    std::vector<int> get_common_sample_rates();
    std::vector<SDL_AudioFormat> get_common_formats();

    // Device matching
    std::string extract_device_brand(const std::string& device_name);
    std::string extract_device_model(const std::string& device_name);
}

} // namespace sdl_audio