#include "audio_engine.h"
#include <iostream>
#include <cstring>
#include <algorithm>
#include <sstream>
#include <thread>
#include <chrono>

namespace sdl_audio {

// AudioBuffer implementation
void AudioBuffer::write(const float* samples, size_t count) {
    std::lock_guard<std::mutex> lock(mutex);

    for (size_t i = 0; i < count; ++i) {
        data[write_pos] = samples[i];
        write_pos = (write_pos + 1) % size;

        // If buffer is full, advance read position (overwrite old data)
        if (write_pos == read_pos) {
            read_pos = (read_pos + 1) % size;
        }
    }
}

size_t AudioBuffer::read(float* samples, size_t count) {
    std::lock_guard<std::mutex> lock(mutex);

    size_t samples_read = 0;

    while (samples_read < count && read_pos != write_pos) {
        samples[samples_read] = data[read_pos];
        read_pos = (read_pos + 1) % size;
        samples_read++;
    }

    // Fill remaining with silence
    while (samples_read < count) {
        samples[samples_read] = 0.0f;
        samples_read++;
    }

    return samples_read;
}

void AudioBuffer::clear() {
    std::lock_guard<std::mutex> lock(mutex);
    write_pos = 0;
    read_pos = 0;
    std::fill(data.begin(), data.end(), 0.0f);
}

size_t AudioBuffer::available_write() const {
    std::lock_guard<std::mutex> lock(mutex);
    if (write_pos >= read_pos) {
        return size - (write_pos - read_pos) - 1;
    } else {
        return read_pos - write_pos - 1;
    }
}

size_t AudioBuffer::available_read() const {
    std::lock_guard<std::mutex> lock(mutex);
    if (write_pos >= read_pos) {
        return write_pos - read_pos;
    } else {
        return size - (read_pos - write_pos);
    }
}

// AudioEngine implementation
AudioEngine::AudioEngine()
    : state_(State::Uninitialized),
      input_device_(0),
      output_device_(0),
      is_running_(false),
      should_stop_(false),
      input_samples_processed_(0),
      output_samples_processed_(0),
      buffer_underruns_(0),
      buffer_overruns_(0),
      is_recording_(false),
      is_playing_(false),
      recording_position_(0),
      playback_position_(0),
      num_input_channels_(1),     // NEW: Default to mono
      num_output_channels_(1) {   // NEW: Default to mono
}

AudioEngine::~AudioEngine() {
    shutdown();
}

bool AudioEngine::initialize(const Config& config) {
    std::lock_guard<std::mutex> lock(state_mutex_);

    if (state_ != State::Uninitialized) {
        log_error("AudioEngine already initialized");
        return false;
    }

    config_ = config;

    // NEW: Validate multi-channel configuration
    if (config_.input_channels < 1 || config_.input_channels > 32) {
        log_error("Invalid input_channels: must be 1-32");
        state_ = State::Error;
        return false;
    }
    if (config_.output_channels < 1 || config_.output_channels > 32) {
        log_error("Invalid output_channels: must be 1-32");
        state_ = State::Error;
        return false;
    }

    // NEW: Initialize multi-channel buffers
    num_input_channels_ = config_.input_channels;
    num_output_channels_ = config_.output_channels;

    recording_buffers_.resize(num_input_channels_);
    channel_mutexes_.resize(num_input_channels_);

    // Pre-allocate buffers for each channel (10 seconds @ 48kHz)
    for (int ch = 0; ch < num_input_channels_; ++ch) {
        channel_mutexes_[ch] = std::make_unique<std::mutex>();
        recording_buffers_[ch].reserve(48000 * 10);
    }

    // Initialize SDL audio subsystem
    if (SDL_InitSubSystem(SDL_INIT_AUDIO) < 0) {
        log_error("Failed to initialize SDL audio: " + std::string(SDL_GetError()));
        state_ = State::Error;
        return false;
    }

    log("SDL Audio subsystem initialized");
    log("SDL Version: " + get_sdl_version());
    log("Multi-channel config: " + std::to_string(num_input_channels_) +
        " input, " + std::to_string(num_output_channels_) + " output");

    // Create audio buffers (circular buffers remain mono for now)
    size_t buffer_samples = config_.buffer_size * 8; // 8x buffer size for safety
    input_buffer_ = std::make_unique<AudioBuffer>(buffer_samples);
    output_buffer_ = std::make_unique<AudioBuffer>(buffer_samples);

    // Reset statistics
    reset_stats();

    state_ = State::Initialized;
    log("AudioEngine initialized successfully");

    return true;
}

bool AudioEngine::start() {
    std::lock_guard<std::mutex> lock(state_mutex_);

    if (state_ != State::Initialized && state_ != State::Stopped) {
        log_error("AudioEngine must be initialized before starting");
        return false;
    }

    // Initialize devices
    if (!initialize_input_device() || !initialize_output_device()) {
        state_ = State::Error;
        return false;
    }

    is_running_ = true;
    should_stop_ = false;
    state_ = State::Running;

    log("AudioEngine started successfully");
    return true;
}

bool AudioEngine::stop() {
    std::lock_guard<std::mutex> lock(state_mutex_);

    if (state_ != State::Running) {
        return true; // Already stopped
    }

    should_stop_ = true;

    // Stop any ongoing recording/playback
    stop_recording();
    stop_playback();

    // Pause audio devices
    if (input_device_ > 0) {
        SDL_PauseAudioDevice(input_device_, 1);
    }
    if (output_device_ > 0) {
        SDL_PauseAudioDevice(output_device_, 1);
    }

    is_running_ = false;
    state_ = State::Stopped;

    log("AudioEngine stopped");
    return true;
}

void AudioEngine::shutdown() {
    std::lock_guard<std::mutex> lock(state_mutex_);

    if (state_ == State::Uninitialized) {
        return;
    }

    // Stop if running
    if (state_ == State::Running) {
        should_stop_ = true;
        stop_recording();
        stop_playback();
        is_running_ = false;
    }

    // Clean up devices
    cleanup_devices();

    // Clear buffers
    if (input_buffer_) {
        input_buffer_->clear();
    }
    if (output_buffer_) {
        output_buffer_->clear();
    }

    // Clear recording/playback data
    for (int ch = 0; ch < num_input_channels_; ++ch) {
        if (ch < (int)channel_mutexes_.size() && channel_mutexes_[ch]) {
            std::lock_guard<std::mutex> lock(*channel_mutexes_[ch]);
            recording_buffers_[ch].clear();
        }
    }
    {
        std::lock_guard<std::mutex> play_lock(playback_mutex_);
        playback_signal_.clear();
    }

    // Quit SDL audio subsystem
    SDL_QuitSubSystem(SDL_INIT_AUDIO);

    state_ = State::Uninitialized;
    log("AudioEngine shutdown complete");
}

std::vector<AudioDevice> AudioEngine::get_input_devices() const {
    std::vector<AudioDevice> devices;

    int device_count = SDL_GetNumAudioDevices(1); // 1 for input devices

    for (int i = 0; i < device_count; ++i) {
        const char* device_name = SDL_GetAudioDeviceName(i, 1);
        if (device_name) {
            AudioDevice device(i, device_name, true);

            // Get device specs
            SDL_AudioSpec spec;
            if (SDL_GetAudioDeviceSpec(i, 1, &spec) == 0) {
                device.spec = spec;
                device.max_channels = spec.channels;
            }

            devices.push_back(device);
        }
    }

    return devices;
}

std::vector<AudioDevice> AudioEngine::get_output_devices() const {
    std::vector<AudioDevice> devices;

    int device_count = SDL_GetNumAudioDevices(0); // 0 for output devices

    for (int i = 0; i < device_count; ++i) {
        const char* device_name = SDL_GetAudioDeviceName(i, 0);
        if (device_name) {
            AudioDevice device(i, device_name, false);

            // Get device specs
            SDL_AudioSpec spec;
            if (SDL_GetAudioDeviceSpec(i, 0, &spec) == 0) {
                device.spec = spec;
                device.max_channels = spec.channels;
            }

            devices.push_back(device);
        }
    }

    return devices;
}

AudioDevice AudioEngine::get_current_input_device() const {
    auto devices = get_input_devices();
    int device_id = config_.input_device_id;

    if (device_id >= 0 && device_id < static_cast<int>(devices.size())) {
        return devices[device_id];
    }

    // Return default device (first one)
    if (!devices.empty()) {
        return devices[0];
    }

    return AudioDevice(-1, "No Device", true);
}

AudioDevice AudioEngine::get_current_output_device() const {
    auto devices = get_output_devices();
    int device_id = config_.output_device_id;

    if (device_id >= 0 && device_id < static_cast<int>(devices.size())) {
        return devices[device_id];
    }

    // Return default device (first one)
    if (!devices.empty()) {
        return devices[0];
    }

    return AudioDevice(-1, "No Device", false);
}

bool AudioEngine::set_input_device(int device_id) {
    config_.input_device_id = device_id;

    if (state_ == State::Running) {
        // Reinitialize input device
        if (input_device_ > 0) {
            SDL_CloseAudioDevice(input_device_);
            input_device_ = 0;
        }
        return initialize_input_device();
    }

    return true;
}

bool AudioEngine::set_output_device(int device_id) {
    config_.output_device_id = device_id;

    if (state_ == State::Running) {
        // Reinitialize output device
        if (output_device_ > 0) {
            SDL_CloseAudioDevice(output_device_);
            output_device_ = 0;
        }
        return initialize_output_device();
    }

    return true;
}

bool AudioEngine::set_sample_rate(int sample_rate) {
    config_.sample_rate = sample_rate;

    if (state_ == State::Running) {
        log("Sample rate change requires restart");
        return false;
    }

    return true;
}

bool AudioEngine::set_buffer_size(int buffer_size) {
    config_.buffer_size = buffer_size;

    if (state_ == State::Running) {
        log("Buffer size change requires restart");
        return false;
    }

    return true;
}

void AudioEngine::set_input_callback(std::function<void(const float*, size_t)> callback) {
    input_callback_ = callback;
}

void AudioEngine::set_output_callback(std::function<void(float*, size_t)> callback) {
    output_callback_ = callback;
}

size_t AudioEngine::write_output_samples(const float* samples, size_t count) {
    if (!output_buffer_) return 0;

    size_t available = output_buffer_->available_write();
    size_t samples_to_write = std::min(count, available);

    output_buffer_->write(samples, samples_to_write);

    if (samples_to_write < count) {
        buffer_overruns_.fetch_add(1);
    }

    return samples_to_write;
}

size_t AudioEngine::read_input_samples(float* samples, size_t count) {
    if (!input_buffer_) return 0;

    return input_buffer_->read(samples, count);
}

bool AudioEngine::wait_for_input_data(size_t min_samples, int timeout_ms) {
    auto start_time = std::chrono::steady_clock::now();

    while (input_buffer_ && input_buffer_->available_read() < min_samples) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));

        auto elapsed = std::chrono::steady_clock::now() - start_time;
        if (std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count() > timeout_ms) {
            return false;
        }
    }

    return true;
}

bool AudioEngine::wait_for_output_space(size_t min_samples, int timeout_ms) {
    auto start_time = std::chrono::steady_clock::now();

    while (output_buffer_ && output_buffer_->available_write() < min_samples) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));

        auto elapsed = std::chrono::steady_clock::now() - start_time;
        if (std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count() > timeout_ms) {
            return false;
        }
    }

    return true;
}

// NEW: Recording methods
bool AudioEngine::start_recording(size_t max_samples) {
    if (state_ != State::Running) {
        log_error("AudioEngine must be running to start recording");
        return false;
    }

    if (is_recording_.load()) {
        log("Recording already active");
        return false;
    }

    for (int ch = 0; ch < num_input_channels_; ++ch) {
        if (ch < (int)channel_mutexes_.size() && channel_mutexes_[ch]) {
            std::lock_guard<std::mutex> lock(*channel_mutexes_[ch]);
            recording_buffers_[ch].clear();
            if (max_samples > 0) {
                recording_buffers_[ch].reserve(max_samples);
            }
        }
    }

    recording_position_.store(0);
    is_recording_.store(true);

    log("Recording started");
    return true;
}

bool AudioEngine::stop_recording() {
    if (!is_recording_.load()) {
        return true;  // Already stopped
    }

    is_recording_.store(false);

    size_t samples_recorded = recording_position_.load();
    log("Recording stopped. Captured " + std::to_string(samples_recorded) + " samples");

    return true;
}

std::vector<float> AudioEngine::get_recorded_data() {
    // Return channel 0 for backward compatibility
    if (num_input_channels_ > 0) {
        return get_recorded_data_channel(0);
    }
    return std::vector<float>();
}

// NEW: Multi-channel data retrieval
std::vector<std::vector<float>> AudioEngine::get_recorded_data_multichannel() {
    std::vector<std::vector<float>> result(num_input_channels_);

    for (int ch = 0; ch < num_input_channels_; ++ch) {
        if (ch < (int)channel_mutexes_.size() && channel_mutexes_[ch]) {
            std::lock_guard<std::mutex> lock(*channel_mutexes_[ch]);
            result[ch] = recording_buffers_[ch];
        }
    }

    return result;
}

// NEW: Single channel retrieval
std::vector<float> AudioEngine::get_recorded_data_channel(int channel_index) {
    if (channel_index < 0 || channel_index >= num_input_channels_) {
        throw std::out_of_range("Channel index " + std::to_string(channel_index) +
                                " out of range [0, " + std::to_string(num_input_channels_) + ")");
    }

    if (channel_index >= (int)channel_mutexes_.size() || !channel_mutexes_[channel_index]) {
        return std::vector<float>();  // Return empty if not initialized
    }

    std::lock_guard<std::mutex> lock(*channel_mutexes_[channel_index]);
    return recording_buffers_[channel_index];
}

void AudioEngine::clear_recording_buffer() {
    for (int ch = 0; ch < num_input_channels_; ++ch) {
        if (ch < (int)channel_mutexes_.size() && channel_mutexes_[ch]) {
            std::lock_guard<std::mutex> lock(*channel_mutexes_[ch]);
            recording_buffers_[ch].clear();
        }
    }
    recording_position_.store(0);
}

// NEW: Playback methods
bool AudioEngine::start_playback(const std::vector<float>& signal) {
    if (state_ != State::Running) {
        log_error("AudioEngine must be running to start playback");
        return false;
    }

    if (is_playing_.load()) {
        log("Playback already active");
        return false;
    }

    {
        std::lock_guard<std::mutex> lock(playback_mutex_);
        playback_signal_ = signal;
    }

    playback_position_.store(0);
    is_playing_.store(true);

    log("Playback started. Signal length: " + std::to_string(signal.size()) + " samples");
    return true;
}

bool AudioEngine::stop_playback() {
    if (!is_playing_.load()) {
        return true;  // Already stopped
    }

    is_playing_.store(false);
    log("Playback stopped");
    return true;
}

// NEW: Synchronized recording and playback
bool AudioEngine::start_synchronized_recording_and_playback(const std::vector<float>& signal,
                                                           size_t max_recording_samples) {
    if (state_ != State::Running) {
        log_error("AudioEngine must be running for synchronized operation");
        return false;
    }

    if (is_recording_.load() || is_playing_.load()) {
        log_error("Recording or playback already active");
        return false;
    }

    // Prepare recording buffer
    for (int ch = 0; ch < num_input_channels_; ++ch) {
        if (ch < (int)channel_mutexes_.size() && channel_mutexes_[ch]) {
            std::lock_guard<std::mutex> lock(*channel_mutexes_[ch]);
            recording_buffers_[ch].clear();
            size_t buffer_size = max_recording_samples > 0 ? max_recording_samples : signal.size() * 2;
            recording_buffers_[ch].reserve(buffer_size);
        }
    }

    // Prepare playback signal
    {
        std::lock_guard<std::mutex> lock(playback_mutex_);
        playback_signal_ = signal;
    }

    // Reset positions
    recording_position_.store(0);
    playback_position_.store(0);

    // Start both operations atomically
    is_recording_.store(true);
    is_playing_.store(true);

    log("Synchronized recording and playback started");
    log("  Playback signal: " + std::to_string(signal.size()) + " samples");

    return true;
}

std::vector<float> AudioEngine::stop_synchronized_and_get_data() {
    stop_playback();
    stop_recording();
    return get_recorded_data();
}

bool AudioEngine::wait_for_recording_completion(int timeout_ms) {
    auto start_time = std::chrono::steady_clock::now();

    while (is_recording_.load()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));

        auto elapsed = std::chrono::steady_clock::now() - start_time;
        if (std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count() > timeout_ms) {
            return false;  // Timeout
        }
    }

    return true;
}

bool AudioEngine::wait_for_playback_completion(int timeout_ms) {
    auto start_time = std::chrono::steady_clock::now();

    while (is_playing_.load()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));

        auto elapsed = std::chrono::steady_clock::now() - start_time;
        if (std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count() > timeout_ms) {
            return false;  // Timeout
        }
    }

    return true;
}

// NEW: Room response measurement convenience method
std::vector<float> AudioEngine::measure_room_response(const std::vector<float>& test_signal,
                                                     int input_device_id,
                                                     int output_device_id) {
    // Store original device settings
    int original_input = config_.input_device_id;
    int original_output = config_.output_device_id;

    // Set devices if specified
    if (input_device_id >= 0) {
        set_input_device(input_device_id);
    }
    if (output_device_id >= 0) {
        set_output_device(output_device_id);
    }

    // Perform synchronized recording and playback
    if (!start_synchronized_recording_and_playback(test_signal)) {
        log_error("Failed to start synchronized recording and playback");
        return {};
    }

    // Wait for completion
    double duration_seconds = static_cast<double>(test_signal.size()) / config_.sample_rate;
    int timeout_ms = static_cast<int>(duration_seconds * 1000) + 1000;  // Add 1 second buffer

    if (!wait_for_playback_completion(timeout_ms)) {
        log_error("Playback did not complete within timeout");
        stop_synchronized_and_get_data();
        return {};
    }

    // Wait a bit more for any remaining audio to be captured
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    // Get recorded data
    std::vector<float> recorded_data = stop_synchronized_and_get_data();

    // Restore original device settings
    set_input_device(original_input);
    set_output_device(original_output);

    log("Room response measurement completed. Recorded " +
        std::to_string(recorded_data.size()) + " samples");

    return recorded_data;
}

AudioEngine::State AudioEngine::get_state() const {
    std::lock_guard<std::mutex> lock(state_mutex_);
    return state_;
}

AudioEngine::Stats AudioEngine::get_stats() const {
    Stats stats{};

    stats.input_samples_processed = input_samples_processed_.load();
    stats.output_samples_processed = output_samples_processed_.load();
    stats.buffer_underruns = buffer_underruns_.load();
    stats.buffer_overruns = buffer_overruns_.load();

    if (input_buffer_) {
        stats.input_buffer_level = input_buffer_->available_read();
    }
    if (output_buffer_) {
        stats.output_buffer_level = output_buffer_->available_write();
    }

    stats.actual_input_sample_rate = input_spec_.freq;
    stats.actual_output_sample_rate = output_spec_.freq;

    // Recording/playback stats
    stats.is_recording = is_recording_.load();
    stats.is_playing = is_playing_.load();
    stats.recording_position = recording_position_.load();
    stats.playback_position = playback_position_.load();

    // NEW: Multi-channel stats
    stats.num_input_channels = num_input_channels_;
    stats.num_output_channels = num_output_channels_;

    stats.channel_buffer_sizes.resize(num_input_channels_);
    for (int ch = 0; ch < num_input_channels_; ++ch) {
        if (ch < (int)channel_mutexes_.size() && channel_mutexes_[ch]) {
            std::lock_guard<std::mutex> lock(*channel_mutexes_[ch]);
            stats.channel_buffer_sizes[ch] = recording_buffers_[ch].size();
        } else {
            stats.channel_buffer_sizes[ch] = 0;
        }
    }

    // For backward compatibility, set recording_buffer_size to channel 0 size
    if (num_input_channels_ > 0 && !channel_mutexes_.empty() && channel_mutexes_[0]) {
        std::lock_guard<std::mutex> lock(*channel_mutexes_[0]);
        stats.recording_buffer_size = recording_buffers_[0].size();
    } else {
        stats.recording_buffer_size = 0;
    }

    {
        std::lock_guard<std::mutex> lock(playback_mutex_);
        stats.playback_signal_size = playback_signal_.size();
    }

    return stats;
}

void AudioEngine::reset_stats() {
    input_samples_processed_.store(0);
    output_samples_processed_.store(0);
    buffer_underruns_.store(0);
    buffer_overruns_.store(0);
}

std::string AudioEngine::get_error_string() const {
    return SDL_GetError();
}

std::string AudioEngine::get_sdl_version() {
    SDL_version version;
    SDL_GetVersion(&version);

    std::stringstream ss;
    ss << static_cast<int>(version.major) << "."
       << static_cast<int>(version.minor) << "."
       << static_cast<int>(version.patch);

    return ss.str();
}

std::vector<std::string> AudioEngine::get_audio_drivers() {
    std::vector<std::string> drivers;

    int driver_count = SDL_GetNumAudioDrivers();
    for (int i = 0; i < driver_count; ++i) {
        const char* driver_name = SDL_GetAudioDriver(i);
        if (driver_name) {
            drivers.push_back(driver_name);
        }
    }

    return drivers;
}

bool AudioEngine::test_device_combination(int input_id, int output_id) {
    // Store original settings
    int orig_input = config_.input_device_id;
    int orig_output = config_.output_device_id;

    // Set test devices
    config_.input_device_id = input_id;
    config_.output_device_id = output_id;

    bool success = false;

    if (state_ == State::Running) {
        // Temporarily reinitialize devices
        if (input_device_ > 0) {
            SDL_CloseAudioDevice(input_device_);
            input_device_ = 0;
        }
        if (output_device_ > 0) {
            SDL_CloseAudioDevice(output_device_);
            output_device_ = 0;
        }

        success = initialize_input_device() && initialize_output_device();
    }

    // Restore original settings
    config_.input_device_id = orig_input;
    config_.output_device_id = orig_output;

    return success;
}

double AudioEngine::measure_roundtrip_latency() {
    // This is a placeholder implementation
    // Real latency measurement would require generating a test signal
    // and measuring the time between output and input detection
    log("Roundtrip latency measurement not yet implemented");
    return 0.0;
}

// Static audio callbacks
void AudioEngine::input_audio_callback(void* userdata, Uint8* stream, int len) {
    AudioEngine* engine = static_cast<AudioEngine*>(userdata);
    engine->handle_input_audio(stream, len);
}

void AudioEngine::output_audio_callback(void* userdata, Uint8* stream, int len) {
    AudioEngine* engine = static_cast<AudioEngine*>(userdata);
    engine->handle_output_audio(stream, len);
}

// Internal audio handlers
void AudioEngine::handle_input_audio(Uint8* stream, int len) {
    if (!input_buffer_) return;

    // Convert to float samples
    size_t sample_count = len / sizeof(float);
    float* samples = reinterpret_cast<float*>(stream);

    // Write to circular buffer
    input_buffer_->write(samples, sample_count);

    // Call external callback if set
    if (input_callback_) {
        input_callback_(samples, sample_count);
    }

    // Handle recording
    if (is_recording_.load()) {
        handle_recording_input(samples, sample_count);
    }

    input_samples_processed_.fetch_add(sample_count);
}

void AudioEngine::handle_output_audio(Uint8* stream, int len) {
    if (!output_buffer_) return;

    size_t sample_count = len / sizeof(float);
    float* samples = reinterpret_cast<float*>(stream);

    // Initialize with silence
    std::memset(samples, 0, len);

    // Handle playback first (this has priority)
    if (is_playing_.load()) {
        handle_playback_output(samples, sample_count);
    } else {
        // Read from circular buffer if not playing
        size_t samples_read = output_buffer_->read(samples, sample_count);

        if (samples_read < sample_count) {
            buffer_underruns_.fetch_add(1);
        }
    }

    // Call external callback if set
    if (output_callback_) {
        output_callback_(samples, sample_count);
    }

    output_samples_processed_.fetch_add(sample_count);
}

// NEW: Recording and playback handlers
void AudioEngine::handle_recording_input(const float* samples, size_t count) {
    // count = total samples received = num_frames * num_channels
    if (num_input_channels_ == 1) {
        // Fast path for mono (backward compatibility)
        std::lock_guard<std::mutex> lock(*channel_mutexes_[0]);
        size_t current_pos = recording_position_.load();

        for (size_t i = 0; i < count; ++i) {
            recording_buffers_[0].push_back(samples[i]);
        }

        recording_position_.store(current_pos + count);
        return;
    }

    // Multi-channel de-interleaving
    size_t num_frames = count / num_input_channels_;

    // Pre-reserve memory to avoid reallocations in audio thread
    for (int ch = 0; ch < num_input_channels_; ++ch) {
        std::lock_guard<std::mutex> lock(*channel_mutexes_[ch]);
        recording_buffers_[ch].reserve(recording_buffers_[ch].size() + num_frames);
    }

    // De-interleave: [L0, R0, L1, R1, ...] → [L0, L1, ...], [R0, R1, ...]
    for (size_t frame = 0; frame < num_frames; ++frame) {
        for (int ch = 0; ch < num_input_channels_; ++ch) {
            float sample = samples[frame * num_input_channels_ + ch];

            // Lock only this channel's mutex
            std::lock_guard<std::mutex> lock(*channel_mutexes_[ch]);
            recording_buffers_[ch].push_back(sample);
        }
    }

    // Update position atomically (frames recorded)
    size_t current_pos = recording_position_.load();
    recording_position_.store(current_pos + num_frames);
}

void AudioEngine::handle_playback_output(float* samples, size_t count) {
    std::lock_guard<std::mutex> lock(playback_mutex_);

    size_t current_pos = playback_position_.load();
    size_t signal_size = playback_signal_.size();

    if (current_pos >= signal_size) {
        // Playback completed
        is_playing_.store(false);
        std::memset(samples, 0, count * sizeof(float));
        return;
    }

    size_t samples_remaining = signal_size - current_pos;
    size_t samples_to_copy = std::min(count, samples_remaining);

    // Copy samples from playback signal
    std::memcpy(samples, &playback_signal_[current_pos], samples_to_copy * sizeof(float));

    // Fill remaining with silence if needed
    if (samples_to_copy < count) {
        std::memset(&samples[samples_to_copy], 0, (count - samples_to_copy) * sizeof(float));
        is_playing_.store(false);  // Mark as completed
    }

    playback_position_.store(current_pos + samples_to_copy);
}

// Device initialization
bool AudioEngine::initialize_input_device() {
    SDL_AudioSpec desired_spec, obtained_spec;
    SDL_zero(desired_spec);

    desired_spec.freq = config_.sample_rate;
    desired_spec.format = AUDIO_F32SYS;  // 32-bit float, system byte order
    desired_spec.channels = num_input_channels_;  // CHANGED: Use configured channel count
    desired_spec.samples = config_.buffer_size;
    desired_spec.callback = input_audio_callback;
    desired_spec.userdata = this;

    // Get device name
    const char* device_name = nullptr;
    if (config_.input_device_id >= 0) {
        device_name = SDL_GetAudioDeviceName(config_.input_device_id, 1);
    }

    input_device_ = SDL_OpenAudioDevice(
        device_name,     // Device name (nullptr for default)
        1,               // iscapture = 1 for input
        &desired_spec,   // Desired spec
        &obtained_spec,  // Obtained spec
        SDL_AUDIO_ALLOW_FREQUENCY_CHANGE | SDL_AUDIO_ALLOW_SAMPLES_CHANGE
    );

    if (input_device_ == 0) {
        log_error("Failed to open input device: " + std::string(SDL_GetError()));
        return false;
    }

    input_spec_ = obtained_spec;

    // NEW: Validate obtained channel count
    if (obtained_spec.channels != num_input_channels_) {
        log_error("Device does not support " + std::to_string(num_input_channels_) +
                  " channels (got " + std::to_string(obtained_spec.channels) + ")");
        SDL_CloseAudioDevice(input_device_);
        input_device_ = 0;
        return false;
    }

    // Start audio input
    SDL_PauseAudioDevice(input_device_, 0);

    log("Input device opened successfully");
    log("  Device: " + std::string(device_name ? device_name : "Default"));
    log("  Sample rate: " + std::to_string(obtained_spec.freq));
    log("  Channels: " + std::to_string(obtained_spec.channels));
    log("  Buffer size: " + std::to_string(obtained_spec.samples));

    return true;
}

bool AudioEngine::initialize_output_device() {
    SDL_AudioSpec desired_spec, obtained_spec;
    SDL_zero(desired_spec);

    desired_spec.freq = config_.sample_rate;
    desired_spec.format = AUDIO_F32SYS;  // 32-bit float, system byte order
    desired_spec.channels = 1;           // Mono
    desired_spec.samples = config_.buffer_size;
    desired_spec.callback = output_audio_callback;
    desired_spec.userdata = this;

    // Get device name
    const char* device_name = nullptr;
    if (config_.output_device_id >= 0) {
        device_name = SDL_GetAudioDeviceName(config_.output_device_id, 0);
    }

    output_device_ = SDL_OpenAudioDevice(
        device_name,     // Device name (nullptr for default)
        0,               // iscapture = 0 for output
        &desired_spec,   // Desired spec
        &obtained_spec,  // Obtained spec
        SDL_AUDIO_ALLOW_FREQUENCY_CHANGE | SDL_AUDIO_ALLOW_SAMPLES_CHANGE
    );

    if (output_device_ == 0) {
        log_error("Failed to open output device: " + std::string(SDL_GetError()));
        return false;
    }

    output_spec_ = obtained_spec;

    // Start audio output
    SDL_PauseAudioDevice(output_device_, 0);

    log("Output device opened successfully");
    log("  Device: " + std::string(device_name ? device_name : "Default"));
    log("  Sample rate: " + std::to_string(obtained_spec.freq));
    log("  Channels: " + std::to_string(obtained_spec.channels));
    log("  Buffer size: " + std::to_string(obtained_spec.samples));

    return true;
}

void AudioEngine::cleanup_devices() {
    if (input_device_ > 0) {
        SDL_PauseAudioDevice(input_device_, 1);
        SDL_CloseAudioDevice(input_device_);
        input_device_ = 0;
        log("Input device closed");
    }

    if (output_device_ > 0) {
        SDL_PauseAudioDevice(output_device_, 1);
        SDL_CloseAudioDevice(output_device_);
        output_device_ = 0;
        log("Output device closed");
    }
}

void AudioEngine::log(const std::string& message) const {
    if (config_.enable_logging) {
        std::cout << "[AudioEngine] " << message << std::endl;
    }
}

void AudioEngine::log_error(const std::string& message) const {
    std::cerr << "[AudioEngine] ERROR: " << message << std::endl;
}

// Helper functions implementation
std::string audio_format_to_string(SDL_AudioFormat format) {
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

std::string audio_spec_to_string(const SDL_AudioSpec& spec) {
    std::stringstream ss;
    ss << "Format: " << audio_format_to_string(spec.format)
       << ", Freq: " << spec.freq
       << ", Channels: " << static_cast<int>(spec.channels)
       << ", Samples: " << spec.samples;
    return ss.str();
}

bool is_format_float(SDL_AudioFormat format) {
    return format == AUDIO_F32LSB || format == AUDIO_F32MSB;
}

size_t get_sample_size(SDL_AudioFormat format) {
    switch (format) {
        case AUDIO_U8:
        case AUDIO_S8:
            return 1;
        case AUDIO_U16LSB:
        case AUDIO_S16LSB:
        case AUDIO_U16MSB:
        case AUDIO_S16MSB:
            return 2;
        case AUDIO_S32LSB:
        case AUDIO_S32MSB:
        case AUDIO_F32LSB:
        case AUDIO_F32MSB:
            return 4;
        default:
            return 0;
    }
}

} // namespace sdl_audio