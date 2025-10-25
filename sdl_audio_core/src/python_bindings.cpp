// Updated pybind11 bindings - Replace your existing bindings

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <iostream>
#include <chrono>
#include <thread>
#include "audio_engine.h"
// #include "device_manager.h"  // Include if you have this

namespace py = pybind11;
using namespace sdl_audio;

PYBIND11_MODULE(sdl_audio_core, m) {
    m.doc() = R"pbdoc(
        SDL Audio Core - Low-level audio module for simultaneous playback and recording
        This package provides C++ SDL2-based audio functionality with Python bindings
        for precise audio device control, room impulse response measurement, and
        professional audio applications.

        Example usage:
            import sdl_audio_core

            # List all audio devices
            devices = sdl_audio_core.list_all_devices()

            # Create and configure audio engine
            engine = sdl_audio_core.AudioEngine()
            config = sdl_audio_core.AudioEngineConfig()
            config.sample_rate = 48000

            # Initialize and start
            engine.initialize(config)
            engine.start()

            # Setup devices
            engine.set_input_device(0)
            engine.set_output_device(0)

            # Perform room response measurement
            test_signal = [0.5] * 4800  # 100ms at 48kHz
            recorded_data = engine.measure_room_response(test_signal)
    )pbdoc";

    // AudioDevice struct
    py::class_<AudioDevice>(m, "AudioDevice")
        .def(py::init<int, const std::string&, bool>())
        .def_readwrite("device_id", &AudioDevice::device_id)
        .def_readwrite("name", &AudioDevice::name)
        .def_readwrite("is_input", &AudioDevice::is_input)
        .def_readwrite("max_channels", &AudioDevice::max_channels)
        .def_readwrite("supported_sample_rates", &AudioDevice::supported_sample_rates)
        .def("__repr__", [](const AudioDevice& d) {
            return "<AudioDevice id=" + std::to_string(d.device_id) +
                   " name='" + d.name + "' input=" + (d.is_input ? "True" : "False") + ">";
        });

    // AudioEngine::Config
    py::class_<AudioEngine::Config>(m, "AudioEngineConfig")
        .def(py::init<>())
        .def_readwrite("sample_rate", &AudioEngine::Config::sample_rate)
        .def_readwrite("buffer_size", &AudioEngine::Config::buffer_size)
        .def_readwrite("input_device_id", &AudioEngine::Config::input_device_id)
        .def_readwrite("output_device_id", &AudioEngine::Config::output_device_id)
        .def_readwrite("enable_logging", &AudioEngine::Config::enable_logging)
        .def_readwrite("input_channels", &AudioEngine::Config::input_channels)     // NEW
        .def_readwrite("output_channels", &AudioEngine::Config::output_channels)   // NEW
        .def("__repr__", [](const AudioEngine::Config& c) {
            return "<AudioEngineConfig sample_rate=" + std::to_string(c.sample_rate) +
                   " buffer_size=" + std::to_string(c.buffer_size) +
                   " input_channels=" + std::to_string(c.input_channels) +
                   " output_channels=" + std::to_string(c.output_channels) + ">";
        });

    // AudioEngine::State enum
    py::enum_<AudioEngine::State>(m, "AudioEngineState")
        .value("Uninitialized", AudioEngine::State::Uninitialized)
        .value("Initialized", AudioEngine::State::Initialized)
        .value("Running", AudioEngine::State::Running)
        .value("Stopped", AudioEngine::State::Stopped)
        .value("Error", AudioEngine::State::Error);

    // AudioEngine::Stats struct
    py::class_<AudioEngine::Stats>(m, "AudioEngineStats")
        .def_readwrite("input_samples_processed", &AudioEngine::Stats::input_samples_processed)
        .def_readwrite("output_samples_processed", &AudioEngine::Stats::output_samples_processed)
        .def_readwrite("buffer_underruns", &AudioEngine::Stats::buffer_underruns)
        .def_readwrite("buffer_overruns", &AudioEngine::Stats::buffer_overruns)
        .def_readwrite("input_buffer_level", &AudioEngine::Stats::input_buffer_level)
        .def_readwrite("output_buffer_level", &AudioEngine::Stats::output_buffer_level)
        .def_readwrite("actual_input_sample_rate", &AudioEngine::Stats::actual_input_sample_rate)
        .def_readwrite("actual_output_sample_rate", &AudioEngine::Stats::actual_output_sample_rate)
        .def_readwrite("is_recording", &AudioEngine::Stats::is_recording)
        .def_readwrite("is_playing", &AudioEngine::Stats::is_playing)
        .def_readwrite("recording_position", &AudioEngine::Stats::recording_position)
        .def_readwrite("playback_position", &AudioEngine::Stats::playback_position)
        .def_readwrite("recording_buffer_size", &AudioEngine::Stats::recording_buffer_size)
        .def_readwrite("playback_signal_size", &AudioEngine::Stats::playback_signal_size)
        .def_readwrite("num_input_channels", &AudioEngine::Stats::num_input_channels)        // NEW
        .def_readwrite("num_output_channels", &AudioEngine::Stats::num_output_channels)      // NEW
        .def_readwrite("channel_buffer_sizes", &AudioEngine::Stats::channel_buffer_sizes);   // NEW

    // AudioEngine main class
    py::class_<AudioEngine>(m, "AudioEngine")
        .def(py::init<>())

        // Core functionality
        .def("initialize", &AudioEngine::initialize,
             "Initialize the audio engine with given configuration",
             py::arg("config") = AudioEngine::Config{})
        .def("start", &AudioEngine::start,
             "Start the audio engine")
        .def("stop", &AudioEngine::stop,
             "Stop the audio engine")
        .def("shutdown", &AudioEngine::shutdown,
             "Shutdown the audio engine and release resources")

        // Device management
        .def("get_input_devices", &AudioEngine::get_input_devices,
             "Get list of available input devices")
        .def("get_output_devices", &AudioEngine::get_output_devices,
             "Get list of available output devices")
        .def("get_current_input_device", &AudioEngine::get_current_input_device,
             "Get currently selected input device")
        .def("get_current_output_device", &AudioEngine::get_current_output_device,
             "Get currently selected output device")

        // Configuration
        .def("set_input_device", &AudioEngine::set_input_device,
             "Set input device by ID",
             py::arg("device_id"))
        .def("set_output_device", &AudioEngine::set_output_device,
             "Set output device by ID",
             py::arg("device_id"))
        .def("set_sample_rate", &AudioEngine::set_sample_rate,
             "Set sample rate",
             py::arg("sample_rate"))
        .def("set_buffer_size", &AudioEngine::set_buffer_size,
             "Set buffer size",
             py::arg("buffer_size"))

        // Recording methods
        .def("start_recording", &AudioEngine::start_recording,
             "Start audio recording",
             py::arg("max_samples") = 0)
        .def("stop_recording", &AudioEngine::stop_recording,
             "Stop audio recording")
        .def("get_recorded_data", &AudioEngine::get_recorded_data,
             "Get recorded audio data as list of floats")
        .def("is_recording_active", &AudioEngine::is_recording_active,
             "Check if recording is currently active")
        .def("get_recorded_samples", &AudioEngine::get_recorded_samples,
             "Get number of samples recorded so far")
        .def("clear_recording_buffer", &AudioEngine::clear_recording_buffer,
             "Clear the recording buffer")

        // NEW: Multi-channel data retrieval
        .def("get_recorded_data_multichannel", &AudioEngine::get_recorded_data_multichannel,
             "Get recorded data for all channels as list of lists")
        .def("get_recorded_data_channel", &AudioEngine::get_recorded_data_channel,
             "Get recorded data for specific channel",
             py::arg("channel_index"))
        .def("get_num_input_channels", &AudioEngine::get_num_input_channels,
             "Get number of input channels")
        .def("get_num_output_channels", &AudioEngine::get_num_output_channels,
             "Get number of output channels")

        // Playback methods
        .def("start_playback", &AudioEngine::start_playback,
             "Start audio playback with given signal",
             py::arg("signal"))
        .def("stop_playback", &AudioEngine::stop_playback,
             "Stop audio playback")
        .def("is_playback_active", &AudioEngine::is_playback_active,
             "Check if playback is currently active")
        .def("get_playback_position", &AudioEngine::get_playback_position,
             "Get current playback position in samples")

        // Synchronized recording and playback
        .def("start_synchronized_recording_and_playback",
             &AudioEngine::start_synchronized_recording_and_playback,
             "Start synchronized recording and playback",
             py::arg("signal"), py::arg("max_recording_samples") = 0)
        .def("stop_synchronized_and_get_data",
             &AudioEngine::stop_synchronized_and_get_data,
             "Stop synchronized operation and return recorded data")

        // Synchronization and timing
        .def("wait_for_input_data", &AudioEngine::wait_for_input_data,
             "Wait for input data to be available",
             py::arg("min_samples"), py::arg("timeout_ms") = 1000)
        .def("wait_for_output_space", &AudioEngine::wait_for_output_space,
             "Wait for output buffer space",
             py::arg("min_samples"), py::arg("timeout_ms") = 1000)
        .def("wait_for_recording_completion", &AudioEngine::wait_for_recording_completion,
             "Wait for recording to complete",
             py::arg("timeout_ms") = 5000)
        .def("wait_for_playback_completion", &AudioEngine::wait_for_playback_completion,
             "Wait for playback to complete",
             py::arg("timeout_ms") = 5000)

        // State and status
        .def("get_state", &AudioEngine::get_state,
             "Get current engine state")
        .def("is_running", &AudioEngine::is_running,
             "Check if engine is running")
        .def("get_stats", &AudioEngine::get_stats,
             "Get engine statistics")
        .def("reset_stats", &AudioEngine::reset_stats,
             "Reset statistics counters")

        // Direct buffer access
        .def("write_output_samples", &AudioEngine::write_output_samples,
             "Write samples directly to output buffer",
             py::arg("samples"), py::arg("count"))
        .def("read_input_samples", &AudioEngine::read_input_samples,
             "Read samples directly from input buffer",
             py::arg("samples"), py::arg("count"))

        // Utility methods
        .def("get_error_string", &AudioEngine::get_error_string,
             "Get last error message")
        .def_static("get_sdl_version", &AudioEngine::get_sdl_version,
                   "Get SDL version string")
        .def_static("get_audio_drivers", &AudioEngine::get_audio_drivers,
                   "Get list of available audio drivers")

        // Test methods
        .def("test_device_combination", &AudioEngine::test_device_combination,
             "Test if input/output device combination works",
             py::arg("input_id"), py::arg("output_id"))
        .def("measure_roundtrip_latency", &AudioEngine::measure_roundtrip_latency,
             "Measure round-trip latency")

        // Room response measurement
        .def("measure_room_response", &AudioEngine::measure_room_response,
             "Measure room response with test signal",
             py::arg("test_signal"),
             py::arg("input_device_id") = -1,
             py::arg("output_device_id") = -1);

    // Module utility functions
    m.def("get_version", []() {
        return "SDL Audio Core v0.1.0";
    }, "Get module version");

    m.def("get_build_info", []() {
        py::dict info;
        info["module_loaded"] = true;
        info["version"] = "0.1.0";
        info["python_version"] = py::str(py::module_::import("sys").attr("version"));
        info["sdl_version"] = AudioEngine::get_sdl_version();
        return info;
    }, "Get build and system information");

    m.def("list_all_devices", []() {
        AudioEngine engine;
        AudioEngine::Config config;
        config.enable_logging = false;  // Reduce noise for device listing

        if (!engine.initialize(config)) {
            throw std::runtime_error("Failed to initialize audio engine");
        }

        auto input_devices = engine.get_input_devices();
        auto output_devices = engine.get_output_devices();

        engine.shutdown();

        py::dict result;
        result["input_devices"] = input_devices;
        result["output_devices"] = output_devices;

        return result;
    }, "List all available audio devices");

    m.def("check_installation", []() {
        try {
            AudioEngine engine;
            AudioEngine::Config config;
            config.enable_logging = true;

            bool init_success = engine.initialize(config);
            if (!init_success) {
                return false;
            }

            auto input_devices = engine.get_input_devices();
            auto output_devices = engine.get_output_devices();

            engine.shutdown();

            std::cout << "✅ SDL Audio Core " << "0.1.0" << " installed correctly" << std::endl;
            std::cout << "SDL Version: " << AudioEngine::get_sdl_version() << std::endl;
            std::cout << "Found " << input_devices.size() << " input and "
                      << output_devices.size() << " output devices" << std::endl;

            return true;
        } catch (const std::exception& e) {
            std::cerr << "❌ Installation check failed: " << e.what() << std::endl;
            return false;
        }
    }, "Check if the module is properly installed and working");

    // Convenience function for quick testing
    m.def("quick_device_test", [](int input_device_id, int output_device_id,
                                 const std::vector<float>& test_signal) {
        AudioEngine engine;
        AudioEngine::Config config;
        config.enable_logging = true;

        if (!engine.initialize(config)) {
            throw std::runtime_error("Failed to initialize audio engine");
        }

        if (!engine.start()) {
            engine.shutdown();
            throw std::runtime_error("Failed to start audio engine");
        }

        bool success = true;
        std::vector<float> recorded_data;
        std::string error_message;

        try {
            // Set devices
            if (!engine.set_input_device(input_device_id)) {
                throw std::runtime_error("Failed to set input device");
            }

            if (!engine.set_output_device(output_device_id)) {
                throw std::runtime_error("Failed to set output device");
            }

            // Perform synchronized recording and playback
            if (!engine.start_synchronized_recording_and_playback(test_signal)) {
                throw std::runtime_error("Failed to start synchronized operation");
            }

            // Wait for completion
            double duration_seconds = static_cast<double>(test_signal.size()) / config.sample_rate;
            int timeout_ms = static_cast<int>(duration_seconds * 1000) + 1000;

            if (!engine.wait_for_playback_completion(timeout_ms)) {
                throw std::runtime_error("Playback did not complete within timeout");
            }

            // Get results
            recorded_data = engine.stop_synchronized_and_get_data();

        } catch (const std::exception& e) {
            success = false;
            error_message = e.what();
        }

        engine.shutdown();

        py::dict result;
        result["success"] = success;
        result["recorded_data"] = recorded_data;
        result["input_device_id"] = input_device_id;
        result["output_device_id"] = output_device_id;
        result["samples_recorded"] = recorded_data.size();
        result["error_message"] = error_message;

        return result;
    }, "Quick device test with custom signal",
       py::arg("input_device_id"), py::arg("output_device_id"), py::arg("test_signal"));

    // Room response measurement with automatic device selection
    m.def("measure_room_response_auto", [](const std::vector<float>& test_signal,
                                          float volume = 0.3, int input_device = -1, int output_device = -1) {
        AudioEngine engine;
        AudioEngine::Config config;
        config.enable_logging = true;

        if (!engine.initialize(config)) {
            throw std::runtime_error("Failed to initialize audio engine");
        }

        if (!engine.start()) {
            engine.shutdown();
            throw std::runtime_error("Failed to start audio engine");
        }

        std::vector<float> recorded_data;
        bool success = false;
        std::string error_message;

        try {
            // Use default devices (-1)
            recorded_data = engine.measure_room_response(test_signal, input_device, output_device);
            success = !recorded_data.empty();

        } catch (const std::exception& e) {
            error_message = e.what();
        }

        engine.shutdown();

        py::dict result;
        result["success"] = success;
        result["recorded_data"] = recorded_data;
        result["test_signal_samples"] = test_signal.size();
        result["recorded_samples"] = recorded_data.size();
        result["error_message"] = error_message;

        return result;
    }, "Measure room response with automatic device selection",
       py::arg("test_signal"), py::arg("volume") = 0.3f, py::arg("input_device") = -1, py::arg("output_device") = -1);

    // NEW: Multi-channel room response measurement
    m.def("measure_room_response_auto_multichannel",
        [](const std::vector<float>& test_signal,
           float volume = 0.3f,
           int input_device = -1,
           int output_device = -1,
           int input_channels = 1) {

            AudioEngine engine;
            AudioEngine::Config config;
            config.enable_logging = true;
            config.sample_rate = 48000;
            config.input_channels = input_channels;
            config.output_channels = 1;

            if (!engine.initialize(config)) {
                throw std::runtime_error("Failed to initialize audio engine");
            }

            if (!engine.start()) {
                engine.shutdown();
                throw std::runtime_error("Failed to start audio engine");
            }

            std::vector<std::vector<float>> multichannel_data;
            bool success = false;
            std::string error_message;

            try {
                // Use measure_room_response (handles device switching)
                std::vector<float> response = engine.measure_room_response(test_signal, input_device, output_device);

                // Get multi-channel data
                multichannel_data = engine.get_recorded_data_multichannel();
                success = !multichannel_data.empty();

            } catch (const std::exception& e) {
                error_message = e.what();
            }

            engine.shutdown();

            // Return Python dict
            py::dict result;
            result["success"] = success;
            result["multichannel_data"] = multichannel_data;
            result["num_channels"] = (int)multichannel_data.size();
            result["samples_per_channel"] = multichannel_data.empty() ? 0 : (int)multichannel_data[0].size();
            result["test_signal_samples"] = (int)test_signal.size();
            result["error_message"] = error_message;

            return result;
        },
        "Measure room response with multi-channel input support",
        py::arg("test_signal"),
        py::arg("volume") = 0.3f,
        py::arg("input_device") = -1,
        py::arg("output_device") = -1,
        py::arg("input_channels") = 1
    );

    // Version info
    m.attr("__version__") = "0.1.0";
    m.attr("SDL_VERSION") = AudioEngine::get_sdl_version();
}