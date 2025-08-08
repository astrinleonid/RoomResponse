#pragma once

#include "audio_engine.h"
#include <vector>
#include <memory>
#include <functional>

namespace sdl_audio {

class RoomResponseMeasurer {
public:
    struct Config {
        int sample_rate = 48000;
        int pulse_duration_ms = 10;        // Duration of each pulse in milliseconds
        int cycle_duration_ms = 100;       // Time between pulse starts
        int num_pulses = 5;                // Number of pulses to generate
        double pulse_frequency = 1000.0;   // Frequency of sine wave pulse
        double volume = 0.5;               // Output volume (0.0 - 1.0)

        // Processing parameters
        int averaging_start_pulse = 4;     // Which pulse to start averaging from (0-based)
        bool enable_onset_detection = true;
        double onset_threshold_factor = 2.0;
        int onset_window_size = 10;

        Config() = default;
    };

    struct MeasurementResult {
        std::vector<float> raw_recording;      // Complete raw recording
        std::vector<float> room_response;      // Averaged room response for one cycle
        std::vector<float> impulse_response;   // Time-aligned impulse response

        // Metadata
        int sample_rate;
        int actual_pulses_recorded;
        int onset_sample;
        double signal_to_noise_ratio;
        double max_amplitude;
        double rms_level;

        // Timing information
        std::vector<int> detected_pulse_positions;
        std::vector<double> pulse_timing_errors;

        bool success;
        std::string error_message;

        MeasurementResult() : sample_rate(0), actual_pulses_recorded(0),
                            onset_sample(-1), signal_to_noise_ratio(0.0),
                            max_amplitude(0.0), rms_level(0.0), success(false) {}
    };

private:
    std::unique_ptr<AudioEngine> audio_engine_;
    Config config_;

    std::vector<float> playback_signal_;
    std::vector<float> recorded_signal_;

    bool is_measuring_;
    size_t playback_position_;
    size_t record_position_;

    // Signal generation
    void generate_pulse_sequence();
    std::vector<float> generate_single_pulse(int samples) const;
    void apply_fade_envelope(std::vector<float>& pulse, int fade_samples) const;

    // Signal processing
    int find_sound_onset(const std::vector<float>& audio, int window_size, double threshold_factor) const;
    std::vector<float> calculate_moving_rms(const std::vector<float>& signal, int window_size) const;
    void analyze_pulse_timing(const std::vector<float>& recording, MeasurementResult& result) const;
    double calculate_snr(const std::vector<float>& signal) const;

    // Audio callbacks
    void input_callback(const float* samples, size_t count);
    void output_callback(float* samples, size_t count);

public:
    RoomResponseMeasurer();
    ~RoomResponseMeasurer();

    // Configuration
    bool initialize(const Config& config = Config{});
    void shutdown();

    // Device management
    std::vector<AudioDevice> get_input_devices() const;
    std::vector<AudioDevice> get_output_devices() const;
    bool set_devices(int input_device_id, int output_device_id);

    // Device testing
    struct DeviceTestResult {
        int input_device_id;
        int output_device_id;
        std::string input_device_name;
        std::string output_device_name;
        bool can_open_devices;
        bool has_acoustic_coupling;
        double coupling_strength;
        std::string error_message;
    };

    DeviceTestResult test_device_pair(int input_device_id, int output_device_id);
    std::vector<DeviceTestResult> test_all_device_combinations();
    DeviceTestResult find_best_device_pair();

    // Acoustic coupling test
    bool test_acoustic_coupling(int input_device_id, int output_device_id,
                               double* coupling_strength = nullptr);

    // Measurement
    MeasurementResult measure_room_response();
    MeasurementResult measure_room_response_with_devices(int input_device_id, int output_device_id);

    // Real-time measurement with progress callback
    using ProgressCallback = std::function<void(double progress, const std::string& status)>;
    MeasurementResult measure_room_response_with_progress(ProgressCallback progress_callback);

    // Utility methods
    Config get_config() const { return config_; }
    void set_config(const Config& config) { config_ = config; }

    bool is_initialized() const;
    std::string get_last_error() const;

    // Signal analysis utilities
    static std::vector<float> normalize_signal(const std::vector<float>& signal);
    static void save_wav_file(const std::vector<float>& signal, int sample_rate,
                             const std::string& filename);
    static std::vector<float> load_wav_file(const std::string& filename, int* sample_rate = nullptr);

    // Disable copy/move
    RoomResponseMeasurer(const RoomResponseMeasurer&) = delete;
    RoomResponseMeasurer& operator=(const RoomResponseMeasurer&) = delete;
    RoomResponseMeasurer(RoomResponseMeasurer&&) = delete;
    RoomResponseMeasurer& operator=(RoomResponseMeasurer&&) = delete;
};

// Standalone utility functions for signal processing
namespace signal_processing {
    std::vector<float> generate_sine_wave(double frequency, double duration, int sample_rate, double amplitude = 1.0);
    std::vector<float> generate_white_noise(double duration, int sample_rate, double amplitude = 1.0);
    std::vector<float> apply_window(const std::vector<float>& signal, const std::string& window_type = "hann");

    std::vector<float> cross_correlate(const std::vector<float>& signal1, const std::vector<float>& signal2);
    int find_max_correlation_lag(const std::vector<float>& correlation);

    double calculate_rms(const std::vector<float>& signal);
    double calculate_peak(const std::vector<float>& signal);
    double calculate_thd(const std::vector<float>& signal, double fundamental_freq, int sample_rate);
}

} // namespace sdl_audio