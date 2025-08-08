#include "device_manager.h"
#include <iostream>
#include <sstream>
#include <algorithm>
#include <cstring>
#include <thread>
#include <chrono>
#include <cmath>

namespace sdl_audio {

DeviceManager::DeviceManager() : is_initialized_(false) {
}

DeviceManager::~DeviceManager() {
    shutdown();
}

bool DeviceManager::initialize() {
    if (is_initialized_) {
        return true;
    }

    // Initialize SDL Audio if not already done
    if (SDL_InitSubSystem(SDL_INIT_AUDIO) < 0) {
        set_error("Failed to initialize SDL Audio: " + std::string(SDL_GetError()));
        return false;
    }

    log("DeviceManager initialized");
    is_initialized_ = true;

    // Initial device enumeration
    refresh_device_list();

    return true;
}

void DeviceManager::shutdown() {
    if (!is_initialized_) {
        return;
    }

    input_devices_.clear();
    output_devices_.clear();
    device_capabilities_.clear();

    // Note: We don't call SDL_QuitSubSystem here as other parts might be using it
    is_initialized_ = false;
    log("DeviceManager shutdown");
}

void DeviceManager::refresh_device_list() {
    if (!is_initialized_) {
        set_error("DeviceManager not initialized");
        return;
    }

    input_devices_.clear();
    output_devices_.clear();
    device_capabilities_.clear();

    log("Refreshing device list...");

    // Enumerate input devices
    int num_input_devices = SDL_GetNumAudioDevices(1);
    for (int i = 0; i < num_input_devices; ++i) {
        const char* device_name = SDL_GetAudioDeviceName(i, 1);
        if (device_name) {
            DeviceInfo device(i, std::string(device_name), true);

            // Try to get device spec
            SDL_AudioSpec spec;
            if (SDL_GetAudioDeviceSpec(i, 1, &spec) == 0) {
                device.spec = spec;
                device.max_channels = spec.channels;
                device.supported_sample_rates = probe_sample_rates(i, true);
                device.supported_formats = probe_formats(i, true);
            }

            input_devices_.push_back(device);

            // Probe capabilities
            device_capabilities_[i * 1000 + 1] = probe_device_capabilities(device);
        }
    }

    // Enumerate output devices
    int num_output_devices = SDL_GetNumAudioDevices(0);
    for (int i = 0; i < num_output_devices; ++i) {
        const char* device_name = SDL_GetAudioDeviceName(i, 0);
        if (device_name) {
            DeviceInfo device(i, std::string(device_name), false);

            // Try to get device spec
            SDL_AudioSpec spec;
            if (SDL_GetAudioDeviceSpec(i, 0, &spec) == 0) {
                device.spec = spec;
                device.max_channels = spec.channels;
                device.supported_sample_rates = probe_sample_rates(i, false);
                device.supported_formats = probe_formats(i, false);
            }

            output_devices_.push_back(device);

            // Probe capabilities
            device_capabilities_[i * 1000 + 0] = probe_device_capabilities(device);
        }
    }

    log("Found " + std::to_string(input_devices_.size()) + " input devices and " +
        std::to_string(output_devices_.size()) + " output devices");
}

const std::vector<DeviceInfo>& DeviceManager::get_input_devices() {
    if (!is_initialized_) {
        static std::vector<DeviceInfo> empty;
        return empty;
    }
    return input_devices_;
}

const std::vector<DeviceInfo>& DeviceManager::get_output_devices() {
    if (!is_initialized_) {
        static std::vector<DeviceInfo> empty;
        return empty;
    }
    return output_devices_;
}

void DeviceManager::refresh_devices() {
    refresh_device_list();
}

DeviceInfo DeviceManager::get_device_info(int device_id, bool is_input) {
    const auto& devices = is_input ? input_devices_ : output_devices_;

    for (const auto& device : devices) {
        if (device.id == device_id) {
            return device;
        }
    }

    // Return empty device if not found
    return DeviceInfo();
}

DeviceCapabilities DeviceManager::get_device_capabilities(int device_id, bool is_input) {
    int key = device_id * 1000 + (is_input ? 1 : 0);
    auto it = device_capabilities_.find(key);

    if (it != device_capabilities_.end()) {
        return it->second;
    }

    // Return default capabilities if not found
    return DeviceCapabilities();
}

DeviceCapabilities DeviceManager::probe_device_capabilities(const DeviceInfo& device) {
    DeviceCapabilities caps;

    // Test if device can be opened
    bool can_open = test_device_opening(device.id, device.is_input);
    if (!can_open) {
        return caps; // Return default (all false) if can't open
    }

    // Estimate latency based on buffer size
    caps.min_buffer_size = 64;
    caps.max_buffer_size = 4096;
    caps.preferred_buffer_size = 1024;
    caps.latency_estimate_ms = (caps.preferred_buffer_size / (double)device.spec.freq) * 1000.0;

    // Check if it's a professional interface
    if (is_professional_interface(device.name)) {
        caps.supports_simultaneous_io = true;
        caps.supports_full_duplex = true;
        caps.supports_exclusive_mode = true;
        caps.min_buffer_size = 32;
        caps.preferred_buffer_size = 256;
    }

    // Check if it's a built-in device
    if (is_builtin_device(device.name)) {
        caps.has_hardware_monitoring = true; // Usually has echo cancellation
    }

    // USB devices typically support full duplex
    if (device_utils::is_usb_device(device.name)) {
        caps.supports_full_duplex = true;
        caps.supports_simultaneous_io = true;
    }

    return caps;
}

bool DeviceManager::test_device_opening(int device_id, bool is_input) {
    SDL_AudioSpec desired_spec, obtained_spec;

    SDL_zero(desired_spec);
    desired_spec.freq = 48000;
    desired_spec.format = AUDIO_F32SYS;
    desired_spec.channels = 1;
    desired_spec.samples = 1024;
    desired_spec.callback = nullptr; // No callback for testing

    const char* device_name = SDL_GetAudioDeviceName(device_id, is_input ? 1 : 0);

    SDL_AudioDeviceID test_device = SDL_OpenAudioDevice(
        device_name,
        is_input ? 1 : 0,
        &desired_spec,
        &obtained_spec,
        SDL_AUDIO_ALLOW_ANY_CHANGE
    );

    if (test_device == 0) {
        return false;
    }

    SDL_CloseAudioDevice(test_device);
    return true;
}

std::vector<int> DeviceManager::probe_sample_rates(int device_id, bool is_input) {
    std::vector<int> supported_rates;
    std::vector<int> test_rates = device_utils::get_common_sample_rates();

    const char* device_name = SDL_GetAudioDeviceName(device_id, is_input ? 1 : 0);

    for (int rate : test_rates) {
        SDL_AudioSpec desired_spec, obtained_spec;

        SDL_zero(desired_spec);
        desired_spec.freq = rate;
        desired_spec.format = AUDIO_F32SYS;
        desired_spec.channels = 1;
        desired_spec.samples = 1024;
        desired_spec.callback = nullptr;

        SDL_AudioDeviceID test_device = SDL_OpenAudioDevice(
            device_name,
            is_input ? 1 : 0,
            &desired_spec,
            &obtained_spec,
            0 // Don't allow changes for this test
        );

        if (test_device != 0 && obtained_spec.freq == rate) {
            supported_rates.push_back(rate);
            SDL_CloseAudioDevice(test_device);
        } else if (test_device != 0) {
            SDL_CloseAudioDevice(test_device);
        }
    }

    return supported_rates;
}

std::vector<SDL_AudioFormat> DeviceManager::probe_formats(int device_id, bool is_input) {
    std::vector<SDL_AudioFormat> supported_formats;
    std::vector<SDL_AudioFormat> test_formats = device_utils::get_common_formats();

    const char* device_name = SDL_GetAudioDeviceName(device_id, is_input ? 1 : 0);

    for (SDL_AudioFormat format : test_formats) {
        SDL_AudioSpec desired_spec, obtained_spec;

        SDL_zero(desired_spec);
        desired_spec.freq = 48000;
        desired_spec.format = format;
        desired_spec.channels = 1;
        desired_spec.samples = 1024;
        desired_spec.callback = nullptr;

        SDL_AudioDeviceID test_device = SDL_OpenAudioDevice(
            device_name,
            is_input ? 1 : 0,
            &desired_spec,
            &obtained_spec,
            0 // Don't allow changes for this test
        );

        if (test_device != 0 && obtained_spec.format == format) {
            supported_formats.push_back(format);
            SDL_CloseAudioDevice(test_device);
        } else if (test_device != 0) {
            SDL_CloseAudioDevice(test_device);
        }
    }

    return supported_formats;
}

DeviceInfo DeviceManager::get_default_input_device() {
    if (input_devices_.empty()) {
        return DeviceInfo();
    }
    return input_devices_[0]; // SDL returns default device as first
}

DeviceInfo DeviceManager::get_default_output_device() {
    if (output_devices_.empty()) {
        return DeviceInfo();
    }
    return output_devices_[0]; // SDL returns default device as first
}

bool DeviceManager::test_device_pair(int input_id, int output_id) {
    if (!is_initialized_) {
        set_error("DeviceManager not initialized");
        return false;
    }

    log("Testing device pair: input=" + std::to_string(input_id) +
        ", output=" + std::to_string(output_id));

    // First check if both devices can be opened individually
    if (!test_device_opening(input_id, true)) {
        set_error("Cannot open input device " + std::to_string(input_id));
        return false;
    }

    if (!test_device_opening(output_id, false)) {
        set_error("Cannot open output device " + std::to_string(output_id));
        return false;
    }

    // Test simultaneous opening
    SDL_AudioSpec desired_spec, input_spec, output_spec;

    SDL_zero(desired_spec);
    desired_spec.freq = 48000;
    desired_spec.format = AUDIO_F32SYS;
    desired_spec.channels = 1;
    desired_spec.samples = 1024;
    desired_spec.callback = nullptr;

    const char* input_name = SDL_GetAudioDeviceName(input_id, 1);
    const char* output_name = SDL_GetAudioDeviceName(output_id, 0);

    SDL_AudioDeviceID input_device = SDL_OpenAudioDevice(
        input_name, 1, &desired_spec, &input_spec, SDL_AUDIO_ALLOW_ANY_CHANGE);

    if (input_device == 0) {
        set_error("Failed to open input device in pair test: " + std::string(SDL_GetError()));
        return false;
    }

    SDL_AudioDeviceID output_device = SDL_OpenAudioDevice(
        output_name, 0, &desired_spec, &output_spec, SDL_AUDIO_ALLOW_ANY_CHANGE);

    if (output_device == 0) {
        SDL_CloseAudioDevice(input_device);
        set_error("Failed to open output device in pair test: " + std::string(SDL_GetError()));
        return false;
    }

    // Both devices opened successfully
    SDL_CloseAudioDevice(input_device);
    SDL_CloseAudioDevice(output_device);

    log("Device pair test successful");
    return true;
}

bool DeviceManager::test_acoustic_coupling(int input_id, int output_id, double* coupling_strength) {
    if (!test_device_pair(input_id, output_id)) {
        return false;
    }

    log("Testing acoustic coupling between devices");

    // Simple coupling test - play a tone and see if we can detect it
    const int test_duration_ms = 500;
    const int sample_rate = 48000;
    const int buffer_size = 1024;
    const double test_freq = 1000.0; // 1kHz test tone

    // Generate test signal
    std::vector<float> test_signal;
    int num_samples = (sample_rate * test_duration_ms) / 1000;

    for (int i = 0; i < num_samples; ++i) {
        double t = i / double(sample_rate);
        test_signal.push_back(0.1f * std::sin(2.0 * M_PI * test_freq * t));
    }

    // Open devices for testing
    SDL_AudioSpec desired_spec, input_spec, output_spec;

    SDL_zero(desired_spec);
    desired_spec.freq = sample_rate;
    desired_spec.format = AUDIO_F32SYS;
    desired_spec.channels = 1;
    desired_spec.samples = buffer_size;
    desired_spec.callback = nullptr;

    const char* input_name = SDL_GetAudioDeviceName(input_id, 1);
    const char* output_name = SDL_GetAudioDeviceName(output_id, 0);

    SDL_AudioDeviceID input_device = SDL_OpenAudioDevice(
        input_name, 1, &desired_spec, &input_spec, SDL_AUDIO_ALLOW_FREQUENCY_CHANGE);

    SDL_AudioDeviceID output_device = SDL_OpenAudioDevice(
        output_name, 0, &desired_spec, &output_spec, SDL_AUDIO_ALLOW_FREQUENCY_CHANGE);

    if (input_device == 0 || output_device == 0) {
        if (input_device != 0) SDL_CloseAudioDevice(input_device);
        if (output_device != 0) SDL_CloseAudioDevice(output_device);
        return false;
    }

    // Start recording
    SDL_PauseAudioDevice(input_device, 0);

    // Wait a bit for input to stabilize
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    // Play test signal
    if (!test_signal.empty()) {
        SDL_QueueAudio(output_device, test_signal.data(), test_signal.size() * sizeof(float));
    }
    SDL_PauseAudioDevice(output_device, 0);

    // Wait for playback and recording
    std::this_thread::sleep_for(std::chrono::milliseconds(test_duration_ms + 200));

    // Stop devices
    SDL_PauseAudioDevice(input_device, 1);
    SDL_PauseAudioDevice(output_device, 1);

    // Get recorded data
    Uint32 queued_bytes = SDL_GetQueuedAudioSize(input_device);

    bool has_coupling = false;
    double strength = 0.0;

    if (queued_bytes > 0) {
        std::vector<float> recorded_data(queued_bytes / sizeof(float));
        SDL_DequeueAudio(input_device, recorded_data.data(), queued_bytes);

        // Simple energy-based coupling detection
        double rms = 0.0;
        for (float sample : recorded_data) {
            rms += sample * sample;
        }
        rms = std::sqrt(rms / recorded_data.size());

        strength = rms;
        has_coupling = (rms > 0.001); // Threshold for coupling detection

        log("Coupling test - RMS level: " + std::to_string(rms));
    }

    if (coupling_strength) {
        *coupling_strength = strength;
    }

    SDL_CloseAudioDevice(input_device);
    SDL_CloseAudioDevice(output_device);

    return has_coupling;
}

std::vector<DeviceManager::DevicePairScore> DeviceManager::score_all_device_pairs() {
    std::vector<DevicePairScore> scores;

    for (const auto& input_dev : input_devices_) {
        for (const auto& output_dev : output_devices_) {
            DevicePairScore score;
            score.input_id = input_dev.id;
            score.output_id = output_dev.id;
            score.input_name = input_dev.name;
            score.output_name = output_dev.name;

            // Test if devices can be opened together
            score.can_open_both = test_device_pair(input_dev.id, output_dev.id);

            if (score.can_open_both) {
                // Test acoustic coupling
                score.has_acoustic_coupling = test_acoustic_coupling(
                    input_dev.id, output_dev.id, &score.coupling_strength);

                // Calculate overall score
                score.score = device_utils::calculate_device_pair_score(
                    input_dev, output_dev, score.can_open_both,
                    score.has_acoustic_coupling, score.coupling_strength);

                // Add notes
                if (device_utils::devices_likely_same_hardware(input_dev.name, output_dev.name)) {
                    score.notes += "Same hardware; ";
                }
                if (is_professional_interface(input_dev.name) || is_professional_interface(output_dev.name)) {
                    score.notes += "Professional interface; ";
                }
            }

            scores.push_back(score);
        }
    }

    // Sort by score (highest first)
    std::sort(scores.begin(), scores.end(),
              [](const DevicePairScore& a, const DevicePairScore& b) {
                  return a.score > b.score;
              });

    return scores;
}

DeviceManager::DevicePairScore DeviceManager::find_best_device_pair() {
    auto all_scores = score_all_device_pairs();

    if (all_scores.empty()) {
        return DevicePairScore(); // Return default (empty) score
    }

    return all_scores[0]; // Return highest scored pair
}

std::vector<DeviceManager::DevicePairScore> DeviceManager::get_recommended_pairs(int max_results) {
    auto all_scores = score_all_device_pairs();

    // Filter to only include working pairs with decent scores
    std::vector<DevicePairScore> recommended;
    for (const auto& score : all_scores) {
        if (score.can_open_both && score.score > 0.1 && recommended.size() < max_results) {
            recommended.push_back(score);
        }
    }

    return recommended;
}

void DeviceManager::print_all_devices() const {
    std::cout << "\n=== Audio Device List ===\n";

    std::cout << "\nInput Devices:\n";
    for (const auto& device : input_devices_) {
        std::cout << "  [" << device.id << "] " << device.name << "\n";
        std::cout << "      " << spec_to_string(device.spec) << "\n";
    }

    std::cout << "\nOutput Devices:\n";
    for (const auto& device : output_devices_) {
        std::cout << "  [" << device.id << "] " << device.name << "\n";
        std::cout << "      " << spec_to_string(device.spec) << "\n";
    }
}

std::string DeviceManager::generate_device_report() const {
    std::stringstream ss;
    ss << "SDL Audio Device Report\n";
    ss << "========================\n\n";

    ss << "Input Devices (" << input_devices_.size() << "):\n";
    for (const auto& device : input_devices_) {
        ss << "  [" << device.id << "] " << device.name << "\n";
        ss << "      " << spec_to_string(device.spec) << "\n";
        if (is_professional_interface(device.name)) {
            ss << "      * Professional audio interface\n";
        }
        if (is_builtin_device(device.name)) {
            ss << "      * Built-in device\n";
        }
        ss << "\n";
    }

    ss << "Output Devices (" << output_devices_.size() << "):\n";
    for (const auto& device : output_devices_) {
        ss << "  [" << device.id << "] " << device.name << "\n";
        ss << "      " << spec_to_string(device.spec) << "\n";
        if (is_professional_interface(device.name)) {
            ss << "      * Professional audio interface\n";
        }
        if (is_builtin_device(device.name)) {
            ss << "      * Built-in device\n";
        }
        ss << "\n";
    }

    return ss.str();
}

void DeviceManager::log(const std::string& message) const {
    std::cout << "[DeviceManager] " << message << std::endl;
}

void DeviceManager::log_error(const std::string& message) const {
    std::cerr << "[DeviceManager ERROR] " << message << std::endl;
}

// Static utility functions
std::string DeviceManager::format_to_string(SDL_AudioFormat format) {
    switch (format) {
        case AUDIO_U8: return "U8";
        case AUDIO_S8: return "S8";
        case AUDIO_U16LSB: return "U16LSB";
        case AUDIO_S16LSB: return "S16LSB";
        case AUDIO_U16MSB: return "U16MSB";
        case AUDIO_S16MSB: return "S16MSB";
        case AUDIO_S32LSB: return "S32LSB";
        case AUDIO_S32MSB: return "S32MSB";
        case AUDIO_F32LSB: return "F32LSB";
        case AUDIO_F32MSB: return "F32MSB";
        default: return "Unknown";
    }
}

std::string DeviceManager::spec_to_string(const SDL_AudioSpec& spec) {
    std::stringstream ss;
    ss << format_to_string(spec.format)
       << ", " << spec.freq << " Hz"
       << ", " << static_cast<int>(spec.channels) << " ch"
       << ", " << spec.samples << " samples";
    return ss.str();
}

bool DeviceManager::is_professional_interface(const std::string& device_name) {
    std::vector<std::string> pro_keywords = {
        "RME", "Focusrite", "PreSonus", "MOTU", "Universal Audio",
        "Steinberg", "Roland", "TASCAM", "Zoom", "Behringer",
        "Scarlett", "Clarett", "FireStudio", "UltraLite", "FastTrack",
        "AudioBox", "Studio", "Interface", "ASIO"
    };

    std::string upper_name = device_name;
    std::transform(upper_name.begin(), upper_name.end(), upper_name.begin(), ::toupper);

    for (const auto& keyword : pro_keywords) {
        std::string upper_keyword = keyword;
        std::transform(upper_keyword.begin(), upper_keyword.end(), upper_keyword.begin(), ::toupper);
        if (upper_name.find(upper_keyword) != std::string::npos) {
            return true;
        }
    }

    return false;
}

bool DeviceManager::is_builtin_device(const std::string& device_name) {
    std::vector<std::string> builtin_keywords = {
        "Built-in", "Internal", "Integrated", "MacBook", "iMac",
        "Realtek", "IDT", "Conexant", "VIA", "Intel",
        "Microphone", "Speakers", "Headphones", "Line In", "Line Out"
    };

    std::string upper_name = device_name;
    std::transform(upper_name.begin(), upper_name.end(), upper_name.begin(), ::toupper);

    for (const auto& keyword : builtin_keywords) {
        std::string upper_keyword = keyword;
        std::transform(upper_keyword.begin(), upper_keyword.end(), upper_keyword.begin(), ::toupper);
        if (upper_name.find(upper_keyword) != std::string::npos) {
            return true;
        }
    }

    return false;
}

// Device utility functions implementation
namespace device_utils {

double calculate_device_pair_score(const DeviceInfo& input, const DeviceInfo& output,
                                 bool can_open_both, bool has_coupling, double coupling_strength) {
    double score = 0.0;

    // Base score for working devices
    if (!can_open_both) {
        return 0.0; // No score if devices can't be opened
    }

    score += 1.0; // Base score for working pair

    // Bonus for acoustic coupling
    if (has_coupling) {
        score += 2.0; // Strong bonus for coupling
        score += std::min(2.0, coupling_strength * 10.0); // Additional bonus based on strength
    }

    // Bonus for same hardware (likely to work well together)
    if (devices_likely_same_hardware(input.name, output.name)) {
        score += 1.5;
    }

    // Bonus for professional interfaces
    if (DeviceManager::is_professional_interface(input.name) ||
        DeviceManager::is_professional_interface(output.name)) {
        score += 1.0;
    }

    // Penalty for built-in devices (often have isolation issues)
    if (DeviceManager::is_builtin_device(input.name) &&
        DeviceManager::is_builtin_device(output.name)) {
        score -= 0.5;
    }

    // Bonus for matching sample rates
    if (input.spec.freq == output.spec.freq && input.spec.freq >= 44100) {
        score += 0.5;
    }

    // Penalty for Bluetooth (latency issues)
    if (is_bluetooth_device(input.name) || is_bluetooth_device(output.name)) {
        score -= 1.0;
    }

    return std::max(0.0, score);
}

bool devices_likely_same_hardware(const std::string& input_name, const std::string& output_name) {
    // Extract common parts of device names
    std::string input_brand = extract_device_brand(input_name);
    std::string output_brand = extract_device_brand(output_name);

    if (!input_brand.empty() && input_brand == output_brand) {
        return true;
    }

    // Check for common patterns
    std::vector<std::string> common_patterns = {
        "MacBook Pro", "Built-in", "Realtek", "USB Audio", "Scarlett"
    };

    for (const auto& pattern : common_patterns) {
        if (input_name.find(pattern) != std::string::npos &&
            output_name.find(pattern) != std::string::npos) {
            return true;
        }
    }

    return false;
}

bool is_usb_device(const std::string& device_name) {
    std::string upper_name = device_name;
    std::transform(upper_name.begin(), upper_name.end(), upper_name.begin(), ::toupper);

    return (upper_name.find("USB") != std::string::npos) ||
           (upper_name.find("AUDIO INTERFACE") != std::string::npos) ||
           (upper_name.find("EXTERNAL") != std::string::npos);
}

bool is_bluetooth_device(const std::string& device_name) {
    std::string upper_name = device_name;
    std::transform(upper_name.begin(), upper_name.end(), upper_name.begin(), ::toupper);

    return (upper_name.find("BLUETOOTH") != std::string::npos) ||
           (upper_name.find("WIRELESS") != std::string::npos) ||
           (upper_name.find("AIRPODS") != std::string::npos) ||
           (upper_name.find("BEATS") != std::string::npos);
}

std::vector<int> get_common_sample_rates() {
    return {8000, 11025, 16000, 22050, 32000, 44100, 48000, 88200, 96000, 176400, 192000};
}

std::vector<SDL_AudioFormat> get_common_formats() {
    return {AUDIO_S16LSB, AUDIO_S16MSB, AUDIO_S32LSB, AUDIO_S32MSB,
            AUDIO_F32LSB, AUDIO_F32MSB, AUDIO_U8, AUDIO_S8};
}

std::string extract_device_brand(const std::string& device_name) {
    std::vector<std::string> brands = {
        "Focusrite", "RME", "PreSonus", "MOTU", "Roland",
        "Steinberg", "TASCAM", "Zoom", "Behringer", "Scarlett",
        "Apple", "Realtek", "Creative", "ASUS", "Logitech"
    };

    std::string upper_name = device_name;
    std::transform(upper_name.begin(), upper_name.end(), upper_name.begin(), ::toupper);

    for (const auto& brand : brands) {
        std::string upper_brand = brand;
        std::transform(upper_brand.begin(), upper_brand.end(), upper_brand.begin(), ::toupper);
        if (upper_name.find(upper_brand) != std::string::npos) {
            return brand;
        }
    }

    return "";
}

std::string extract_device_model(const std::string& device_name) {
    // Simple model extraction - look for numbers and common model patterns
    std::string model;

    // Look for patterns like "Scarlett 2i2" or "RME Fireface 802"
    size_t pos = 0;
    bool found_digit = false;

    for (char c : device_name) {
        if (std::isdigit(c)) {
            found_digit = true;
            model += c;
        } else if (found_digit && (std::isalnum(c) || c == 'i' || c == 'x')) {
            model += c;
        } else if (found_digit && !std::isspace(c)) {
            break;
        }
    }

    return model;
}

} // namespace device_utils

} // namespace sdl_audio