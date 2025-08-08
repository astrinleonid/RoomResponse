#include "room_response.h"
#include "device_manager.h"
#include <cmath>
#include <algorithm>
#include <numeric>
#include <fstream>
#include <random>
#include <chrono>
#include <thread>

namespace sdl_audio {

RoomResponseMeasurer::RoomResponseMeasurer()
    : is_measuring_(false), playback_position_(0), record_position_(0) {
}

RoomResponseMeasurer::~RoomResponseMeasurer() {
    shutdown();
}

bool RoomResponseMeasurer::initialize(const Config& config) {
    config_ = config;

    // Create audio engine
    audio_engine_ = std::make_unique<AudioEngine>();

    AudioEngine::Config engine_config;
    engine_config.sample_rate = config_.sample_rate;
    engine_config.buffer_size = 1024;
    engine_config.enable_logging = true;

    if (!audio_engine_->initialize(engine_config)) {
        return false;
    }

    // Generate pulse sequence
    generate_pulse_sequence();

    return true;
}

void RoomResponseMeasurer::shutdown() {
    if (is_measuring_) {
        // Stop any ongoing measurement
        is_measuring_ = false;
    }

    if (audio_engine_) {
        audio_engine_->shutdown();
        audio_engine_.reset();
    }
}

void RoomResponseMeasurer::generate_pulse_sequence() {
    playback_signal_.clear();

    int pulse_samples = (config_.pulse_duration_ms * config_.sample_rate) / 1000;
    int cycle_samples = (config_.cycle_duration_ms * config_.sample_rate) / 1000;

    int total_samples = cycle_samples * config_.num_pulses;
    playback_signal_.resize(total_samples, 0.0f);

    // Generate single pulse
    auto single_pulse = generate_single_pulse(pulse_samples);

    // Place pulses at correct intervals
    for (int i = 0; i < config_.num_pulses; ++i) {
        int start_sample = i * cycle_samples;

        for (int j = 0; j < pulse_samples && (start_sample + j) < total_samples; ++j) {
            playback_signal_[start_sample + j] = single_pulse[j];
        }
    }
}

std::vector<float> RoomResponseMeasurer::generate_single_pulse(int samples) const {
    std::vector<float> pulse(samples);

    for (int i = 0; i < samples; ++i) {
        double t = i / double(config_.sample_rate);
        pulse[i] = config_.volume * std::sin(2.0 * M_PI * config_.pulse_frequency * t);
    }

    // Apply fade envelope to prevent clicks
    int fade_samples = std::min(samples / 10, config_.sample_rate / 1000); // 1ms or 10% of pulse
    if (fade_samples > 0) {
        apply_fade_envelope(pulse, fade_samples);
    }

    return pulse;
}

void RoomResponseMeasurer::apply_fade_envelope(std::vector<float>& pulse, int fade_samples) const {
    // Fade in
    for (int i = 0; i < fade_samples && i < pulse.size(); ++i) {
        double fade = i / double(fade_samples);
        pulse[i] *= fade;
    }

    // Fade out
    int start_fade_out = pulse.size() - fade_samples;
    for (int i = 0; i < fade_samples && start_fade_out + i < pulse.size(); ++i) {
        double fade = (fade_samples - i) / double(fade_samples);
        pulse[start_fade_out + i] *= fade;
    }
}

std::vector<AudioDevice> RoomResponseMeasurer::get_input_devices() const {
    if (!audio_engine_) {
        return {};
    }
    return audio_engine_->get_input_devices();
}

std::vector<AudioDevice> RoomResponseMeasurer::get_output_devices() const {
    if (!audio_engine_) {
        return {};
    }
    return audio_engine_->get_output_devices();
}

bool RoomResponseMeasurer::set_devices(int input_device_id, int output_device_id) {
    if (!audio_engine_) {
        return false;
    }

    bool success = true;
    success &= audio_engine_->set_input_device(input_device_id);
    success &= audio_engine_->set_output_device(output_device_id);

    return success;
}

RoomResponseMeasurer::DeviceTestResult RoomResponseMeasurer::test_device_pair(int input_device_id, int output_device_id) {
    DeviceTestResult result;
    result.input_device_id = input_device_id;
    result.output_device_id = output_device_id;

    if (!audio_engine_) {
        result.error_message = "Audio engine not initialized";
        return result;
    }

    // Get device names
    auto input_devices = get_input_devices();
    auto output_devices = get_output_devices();

    for (const auto& dev : input_devices) {
        if (dev.device_id == input_device_id) {
            result.input_device_name = dev.name;
            break;
        }
    }

    for (const auto& dev : output_devices) {
        if (dev.device_id == output_device_id) {
            result.output_device_name = dev.name;
            break;
        }
    }

    // Test if devices can be opened
    result.can_open_devices = audio_engine_->test_device_combination(input_device_id, output_device_id);

    if (result.can_open_devices) {
        // Test acoustic coupling
        result.has_acoustic_coupling = test_acoustic_coupling(
            input_device_id, output_device_id, &result.coupling_strength);
    }

    return result;
}

std::vector<RoomResponseMeasurer::DeviceTestResult> RoomResponseMeasurer::test_all_device_combinations() {
    std::vector<DeviceTestResult> results;

    auto input_devices = get_input_devices();
    auto output_devices = get_output_devices();

    for (const auto& input_dev : input_devices) {
        for (const auto& output_dev : output_devices) {
            auto result = test_device_pair(input_dev.device_id, output_dev.device_id);
            results.push_back(result);
        }
    }

    // Sort by coupling strength (best first)
    std::sort(results.begin(), results.end(),
              [](const DeviceTestResult& a, const DeviceTestResult& b) {
                  if (a.has_acoustic_coupling != b.has_acoustic_coupling) {
                      return a.has_acoustic_coupling > b.has_acoustic_coupling;
                  }
                  return a.coupling_strength > b.coupling_strength;
              });

    return results;
}

RoomResponseMeasurer::DeviceTestResult RoomResponseMeasurer::find_best_device_pair() {
    auto all_results = test_all_device_combinations();

    if (all_results.empty()) {
        return DeviceTestResult(); // Return default
    }

    // Return the best result (first after sorting)
    return all_results[0];
}

bool RoomResponseMeasurer::test_acoustic_coupling(int input_device_id, int output_device_id,
                                                double* coupling_strength) {
    if (!audio_engine_) {
        return false;
    }

    // Set devices
    if (!set_devices(input_device_id, output_device_id)) {
        return false;
    }

    // Start audio engine
    if (!audio_engine_->start()) {
        return false;
    }

    // Generate a simple test tone
    const double test_duration = 0.5; // 500ms
    const int test_samples = static_cast<int>(test_duration * config_.sample_rate);
    const double test_frequency = 1000.0; // 1kHz

    std::vector<float> test_signal(test_samples);
    for (int i = 0; i < test_samples; ++i) {
        double t = i / double(config_.sample_rate);
        test_signal[i] = 0.1f * std::sin(2.0 * M_PI * test_frequency * t);
    }

    // Clear any existing data
    std::vector<float> dummy(1024);
    audio_engine_->read_input_samples(dummy.data(), dummy.size());

    // Play test signal
    size_t samples_written = 0;
    size_t batch_size = 1024;

    while (samples_written < test_signal.size()) {
        size_t samples_to_write = std::min(batch_size, test_signal.size() - samples_written);

        // Wait for space in output buffer
        if (!audio_engine_->wait_for_output_space(samples_to_write, 1000)) {
            break;
        }

        audio_engine_->write_output_samples(test_signal.data() + samples_written, samples_to_write);
        samples_written += samples_to_write;

        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    // Wait for playback and recording
    std::this_thread::sleep_for(std::chrono::milliseconds(static_cast<int>(test_duration * 1000) + 200));

    // Read recorded data
    std::vector<float> recorded_data(test_samples);
    size_t samples_read = audio_engine_->read_input_samples(recorded_data.data(), test_samples);

    audio_engine_->stop();

    // Analyze coupling
    double rms = 0.0;
    if (samples_read > 0) {
        for (size_t i = 0; i < samples_read; ++i) {
            rms += recorded_data[i] * recorded_data[i];
        }
        rms = std::sqrt(rms / samples_read);
    }

    if (coupling_strength) {
        *coupling_strength = rms;
    }

    // Threshold for coupling detection
    return rms > 0.001;
}

RoomResponseMeasurer::MeasurementResult RoomResponseMeasurer::measure_room_response() {
    MeasurementResult result;

    if (!audio_engine_) {
        result.error_message = "Audio engine not initialized";
        return result;
    }

    if (playback_signal_.empty()) {
        result.error_message = "No playback signal generated";
        return result;
    }

    // Start audio engine
    if (!audio_engine_->start()) {
        result.error_message = "Failed to start audio engine: " + audio_engine_->get_error_string();
        return result;
    }

    // Set up callbacks
    playback_position_ = 0;
    record_position_ = 0;
    recorded_signal_.clear();
    recorded_signal_.resize(playback_signal_.size() + config_.sample_rate); // Extra buffer

    is_measuring_ = true;

    // Set up audio callbacks
    audio_engine_->set_input_callback([this](const float* samples, size_t count) {
        this->input_callback(samples, count);
    });

    audio_engine_->set_output_callback([this](float* samples, size_t count) {
        this->output_callback(samples, count);
    });

    // Wait for measurement to complete
    while (is_measuring_ && playback_position_ < playback_signal_.size()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    // Wait a bit more to capture the tail
    std::this_thread::sleep_for(std::chrono::milliseconds(500));

    is_measuring_ = false;
    audio_engine_->stop();

    // Process the results
    result.sample_rate = config_.sample_rate;
    result.actual_pulses_recorded = config_.num_pulses;

    // Copy raw recording
    result.raw_recording = std::vector<float>(recorded_signal_.begin(),
                                            recorded_signal_.begin() + record_position_);

    if (result.raw_recording.empty()) {
        result.error_message = "No audio was recorded";
        return result;
    }

    // Calculate basic statistics
    result.max_amplitude = *std::max_element(result.raw_recording.begin(), result.raw_recording.end(),
                                           [](float a, float b) { return std::abs(a) < std::abs(b); });
    result.max_amplitude = std::abs(result.max_amplitude);

    double rms_sum = 0.0;
    for (float sample : result.raw_recording) {
        rms_sum += sample * sample;
    }
    result.rms_level = std::sqrt(rms_sum / result.raw_recording.size());

    // Process pulse responses
    int cycle_samples = (config_.cycle_duration_ms * config_.sample_rate) / 1000;

    if (result.raw_recording.size() >= cycle_samples * config_.num_pulses) {
        // Reshape into pulses
        std::vector<std::vector<float>> pulse_responses;

        for (int i = 0; i < config_.num_pulses; ++i) {
            int start_idx = i * cycle_samples;
            int end_idx = std::min(start_idx + cycle_samples, static_cast<int>(result.raw_recording.size()));

            if (end_idx > start_idx) {
                pulse_responses.emplace_back(result.raw_recording.begin() + start_idx,
                                           result.raw_recording.begin() + end_idx);
            }
        }

        // Average the pulses (starting from the specified pulse)
        if (!pulse_responses.empty() && config_.averaging_start_pulse < pulse_responses.size()) {
            result.room_response.resize(cycle_samples, 0.0f);

            int pulses_averaged = 0;
            for (int i = config_.averaging_start_pulse; i < pulse_responses.size(); ++i) {
                for (int j = 0; j < std::min(cycle_samples, static_cast<int>(pulse_responses[i].size())); ++j) {
                    result.room_response[j] += pulse_responses[i][j];
                }
                pulses_averaged++;
            }

            // Normalize by number of pulses averaged
            if (pulses_averaged > 0) {
                for (float& sample : result.room_response) {
                    sample /= pulses_averaged;
                }
            }

            // Find onset and create impulse response
            if (config_.enable_onset_detection) {
                result.onset_sample = find_sound_onset(result.room_response,
                                                     config_.onset_window_size,
                                                     config_.onset_threshold_factor);

                if (result.onset_sample >= 0 && result.onset_sample < result.room_response.size()) {
                    // Create time-aligned impulse response
                    result.impulse_response.resize(result.room_response.size());

                    // Copy from onset to end
                    std::copy(result.room_response.begin() + result.onset_sample,
                            result.room_response.end(),
                            result.impulse_response.begin());

                    // Copy from beginning to onset (wraparound)
                    std::copy(result.room_response.begin(),
                            result.room_response.begin() + result.onset_sample,
                            result.impulse_response.begin() + (result.room_response.size() - result.onset_sample));
                }
            } else {
                result.impulse_response = result.room_response;
            }

            // Calculate SNR
            result.signal_to_noise_ratio = calculate_snr(result.room_response);

            result.success = true;
        } else {
            result.error_message = "Not enough pulses recorded or invalid averaging start pulse";
        }
    } else {
        result.error_message = "Recorded audio too short";
    }

    return result;
}

RoomResponseMeasurer::MeasurementResult RoomResponseMeasurer::measure_room_response_with_devices(
    int input_device_id, int output_device_id) {

    if (!set_devices(input_device_id, output_device_id)) {
        MeasurementResult result;
        result.error_message = "Failed to set devices";
        return result;
    }

    return measure_room_response();
}

RoomResponseMeasurer::MeasurementResult RoomResponseMeasurer::measure_room_response_with_progress(
    ProgressCallback progress_callback) {

    // This is a simplified version - in a full implementation, you'd
    // integrate progress reporting throughout the measurement process

    if (progress_callback) {
        progress_callback(0.0, "Starting measurement...");
    }

    auto result = measure_room_response();

    if (progress_callback) {
        progress_callback(1.0, result.success ? "Measurement complete" : "Measurement failed");
    }

    return result;
}

void RoomResponseMeasurer::input_callback(const float* samples, size_t count) {
    if (!is_measuring_) return;

    for (size_t i = 0; i < count && record_position_ < recorded_signal_.size(); ++i) {
        recorded_signal_[record_position_++] = samples[i];
    }
}

void RoomResponseMeasurer::output_callback(float* samples, size_t count) {
    if (!is_measuring_) {
        std::fill(samples, samples + count, 0.0f);
        return;
    }

    for (size_t i = 0; i < count; ++i) {
        if (playback_position_ < playback_signal_.size()) {
            samples[i] = playback_signal_[playback_position_++];
        } else {
            samples[i] = 0.0f;
        }
    }
}

int RoomResponseMeasurer::find_sound_onset(const std::vector<float>& audio, int window_size, double threshold_factor) const {
    if (audio.size() < window_size * 2) {
        return 0;
    }

    auto rms_signal = calculate_moving_rms(audio, window_size);

    // Calculate background noise level from the beginning
    double noise_level = 0.0;
    int noise_samples = std::min(window_size, static_cast<int>(rms_signal.size()) / 4);

    for (int i = 0; i < noise_samples; ++i) {
        noise_level += rms_signal[i] * rms_signal[i];
    }
    noise_level = std::sqrt(noise_level / noise_samples);

    double threshold = noise_level * threshold_factor;

    // Find first point above threshold
    for (int i = window_size; i < rms_signal.size(); ++i) {
        if (rms_signal[i] > threshold) {
            return i;
        }
    }

    return 0;
}

std::vector<float> RoomResponseMeasurer::calculate_moving_rms(const std::vector<float>& signal, int window_size) const {
    std::vector<float> rms(signal.size());

    for (int i = 0; i < signal.size(); ++i) {
        double sum = 0.0;
        int count = 0;

        int start = std::max(0, i - window_size / 2);
        int end = std::min(static_cast<int>(signal.size()), i + window_size / 2);

        for (int j = start; j < end; ++j) {
            sum += signal[j] * signal[j];
            count++;
        }

        rms[i] = std::sqrt(sum / count);
    }

    return rms;
}

double RoomResponseMeasurer::calculate_snr(const std::vector<float>& signal) const {
    if (signal.empty()) {
        return 0.0;
    }

    // Simple SNR calculation: ratio of signal power to noise power
    // Assumes first 10% is noise, rest is signal
    int noise_end = signal.size() / 10;

    double noise_power = 0.0;
    for (int i = 0; i < noise_end; ++i) {
        noise_power += signal[i] * signal[i];
    }
    noise_power /= noise_end;

    double signal_power = 0.0;
    for (int i = noise_end; i < signal.size(); ++i) {
        signal_power += signal[i] * signal[i];
    }
    signal_power /= (signal.size() - noise_end);

    if (noise_power > 0) {
        return 10.0 * std::log10(signal_power / noise_power);
    }

    return 0.0;
}

bool RoomResponseMeasurer::is_initialized() const {
    return audio_engine_ && audio_engine_->get_state() != AudioEngine::State::Uninitialized;
}

std::string RoomResponseMeasurer::get_last_error() const {
    if (audio_engine_) {
        return audio_engine_->get_error_string();
    }
    return "Audio