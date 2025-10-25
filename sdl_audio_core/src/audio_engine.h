#pragma once

#include <SDL.h>
#include <vector>
#include <memory>
#include <atomic>
#include <mutex>
#include <condition_variable>
#include <functional>
#include <string>

namespace sdl_audio {

struct AudioDevice {
    int device_id;
    std::string name;
    bool is_input;
    int max_channels;
    std::vector<int> supported_sample_rates;
    SDL_AudioSpec spec;

    AudioDevice(int id, const std::string& device_name, bool input)
        : device_id(id), name(device_name), is_input(input), max_channels(0) {}
};

struct AudioBuffer {
    std::vector<float> data;
    size_t write_pos;
    size_t read_pos;
    size_t size;
    mutable std::mutex mutex;  // Make mutex mutable so it can be locked in const methods

    AudioBuffer(size_t buffer_size) : size(buffer_size), write_pos(0), read_pos(0) {
        data.resize(buffer_size, 0.0f);
    }

    void write(const float* samples, size_t count);
    size_t read(float* samples, size_t count);
    void clear();
    size_t available_write() const;
    size_t available_read() const;
};

class AudioEngine {
public:
    struct Config {
        int sample_rate = 48000;
        int buffer_size = 1024;
        int input_device_id = -1;   // -1 for default
        int output_device_id = -1;  // -1 for default
        bool enable_logging = true;

        // NEW: Multi-channel configuration
        int input_channels = 1;     // Number of input channels (1-32)
        int output_channels = 1;    // Keep output mono for now

        Config() = default;
    };

    enum class State {
        Uninitialized,
        Initialized,
        Running,
        Stopped,
        Error
    };

private:
    Config config_;
    State state_;

    SDL_AudioDeviceID input_device_;
    SDL_AudioDeviceID output_device_;
    SDL_AudioSpec input_spec_;
    SDL_AudioSpec output_spec_;

    std::unique_ptr<AudioBuffer> input_buffer_;
    std::unique_ptr<AudioBuffer> output_buffer_;

    std::atomic<bool> is_running_;
    std::atomic<bool> should_stop_;

    mutable std::mutex state_mutex_;
    std::condition_variable state_cv_;

    // Callback function pointers for external processing
    std::function<void(const float*, size_t)> input_callback_;
    std::function<void(float*, size_t)> output_callback_;

    // Statistics
    std::atomic<uint64_t> input_samples_processed_;
    std::atomic<uint64_t> output_samples_processed_;
    std::atomic<uint32_t> buffer_underruns_;
    std::atomic<uint32_t> buffer_overruns_;

    // NEW: Recording and playback state
    std::atomic<bool> is_recording_;
    std::atomic<bool> is_playing_;

    // Multi-channel recording buffers
    std::vector<std::vector<float>> recording_buffers_;           // [channel_idx][samples]
    mutable std::vector<std::unique_ptr<std::mutex>> channel_mutexes_;  // Per-channel locks (mutable for const methods)
    int num_input_channels_;                                      // Cached from config
    int num_output_channels_;                                     // Cached from config

    std::vector<float> playback_signal_;
    std::atomic<size_t> recording_position_;             // Frames recorded (same for all channels)
    std::atomic<size_t> playback_position_;
    mutable std::mutex playback_mutex_;

    // SDL Audio callbacks (static functions)
    static void input_audio_callback(void* userdata, Uint8* stream, int len);
    static void output_audio_callback(void* userdata, Uint8* stream, int len);

    // Internal methods
    void handle_input_audio(Uint8* stream, int len);
    void handle_output_audio(Uint8* stream, int len);

    bool initialize_input_device();
    bool initialize_output_device();
    void cleanup_devices();

    void log(const std::string& message) const;
    void log_error(const std::string& message) const;

    // NEW: Internal recording/playback handlers
    void handle_recording_input(const float* samples, size_t count);
    void handle_playback_output(float* samples, size_t count);

public:
    AudioEngine();
    ~AudioEngine();

    // Core functionality
    bool initialize(const Config& config = Config{});
    bool start();
    bool stop();
    void shutdown();

    // Device management
    std::vector<AudioDevice> get_input_devices() const;
    std::vector<AudioDevice> get_output_devices() const;
    AudioDevice get_current_input_device() const;
    AudioDevice get_current_output_device() const;

    // Configuration
    bool set_input_device(int device_id);
    bool set_output_device(int device_id);
    bool set_sample_rate(int sample_rate);
    bool set_buffer_size(int buffer_size);

    // Audio data access
    void set_input_callback(std::function<void(const float*, size_t)> callback);
    void set_output_callback(std::function<void(float*, size_t)> callback);

    // Direct buffer access (for Python integration)
    size_t write_output_samples(const float* samples, size_t count);
    size_t read_input_samples(float* samples, size_t count);

    // NEW: Recording functionality
    bool start_recording(size_t max_samples = 0);
    bool stop_recording();
    std::vector<float> get_recorded_data();
    bool is_recording_active() const { return is_recording_.load(); }
    size_t get_recorded_samples() const { return recording_position_.load(); }
    void clear_recording_buffer();

    // NEW: Multi-channel data retrieval
    std::vector<std::vector<float>> get_recorded_data_multichannel();
    std::vector<float> get_recorded_data_channel(int channel_index);
    int get_num_input_channels() const { return num_input_channels_; }
    int get_num_output_channels() const { return num_output_channels_; }

    // NEW: Playback functionality
    bool start_playback(const std::vector<float>& signal);
    bool stop_playback();
    bool is_playback_active() const { return is_playing_.load(); }
    size_t get_playback_position() const { return playback_position_.load(); }

    // NEW: Synchronized recording and playback
    bool start_synchronized_recording_and_playback(const std::vector<float>& signal,
                                                   size_t max_recording_samples = 0);
    std::vector<float> stop_synchronized_and_get_data();

    // Synchronization and timing
    bool wait_for_input_data(size_t min_samples, int timeout_ms = 1000);
    bool wait_for_output_space(size_t min_samples, int timeout_ms = 1000);

    // NEW: Wait for recording/playback completion
    bool wait_for_recording_completion(int timeout_ms = 5000);
    bool wait_for_playback_completion(int timeout_ms = 5000);

    // State and status
    State get_state() const;
    bool is_running() const { return is_running_.load(); }

    // Statistics and diagnostics
    struct Stats {
        uint64_t input_samples_processed;
        uint64_t output_samples_processed;
        uint32_t buffer_underruns;
        uint32_t buffer_overruns;
        size_t input_buffer_level;
        size_t output_buffer_level;
        int actual_input_sample_rate;
        int actual_output_sample_rate;

        // NEW: Recording/playback stats
        bool is_recording;
        bool is_playing;
        size_t recording_position;
        size_t playback_position;
        size_t recording_buffer_size;
        size_t playback_signal_size;

        // NEW: Multi-channel info
        int num_input_channels;
        int num_output_channels;
        std::vector<size_t> channel_buffer_sizes;  // Samples per channel
    };

    Stats get_stats() const;
    void reset_stats();

    // Utility methods
    std::string get_error_string() const;
    static std::string get_sdl_version();
    static std::vector<std::string> get_audio_drivers();

    // Test methods
    bool test_device_combination(int input_id, int output_id);
    double measure_roundtrip_latency();

    // NEW: Room response measurement convenience method
    std::vector<float> measure_room_response(const std::vector<float>& test_signal,
                                           int input_device_id = -1,
                                           int output_device_id = -1);

    // Disable copy/move
    AudioEngine(const AudioEngine&) = delete;
    AudioEngine& operator=(const AudioEngine&) = delete;
    AudioEngine(AudioEngine&&) = delete;
    AudioEngine& operator=(AudioEngine&&) = delete;
};

// Helper functions
std::string audio_format_to_string(SDL_AudioFormat format);
std::string audio_spec_to_string(const SDL_AudioSpec& spec);
bool is_format_float(SDL_AudioFormat format);
size_t get_sample_size(SDL_AudioFormat format);

} // namespace sdl_audio