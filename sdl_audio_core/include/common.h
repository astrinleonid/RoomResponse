#pragma once

#include <cstdint>
#include <string>
#include <vector>

// SDL Audio Core version information
#define SDL_AUDIO_CORE_VERSION_MAJOR 0
#define SDL_AUDIO_CORE_VERSION_MINOR 1
#define SDL_AUDIO_CORE_VERSION_PATCH 0

// Common constants
namespace sdl_audio {

// Audio format constants
constexpr int DEFAULT_SAMPLE_RATE = 48000;
constexpr int DEFAULT_BUFFER_SIZE = 1024;
constexpr int DEFAULT_CHANNELS = 1;

// Measurement defaults
constexpr int DEFAULT_PULSE_DURATION_MS = 10;
constexpr int DEFAULT_CYCLE_DURATION_MS = 100;
constexpr int DEFAULT_NUM_PULSES = 5;
constexpr double DEFAULT_PULSE_FREQUENCY = 1000.0;
constexpr double DEFAULT_VOLUME = 0.5;

// Buffer limits
constexpr size_t MIN_BUFFER_SIZE = 64;
constexpr size_t MAX_BUFFER_SIZE = 8192;
constexpr size_t MAX_SAMPLE_RATE = 192000;
constexpr size_t MIN_SAMPLE_RATE = 8000;

// Timing constants
constexpr int DEFAULT_TIMEOUT_MS = 1000;
constexpr int MAX_MEASUREMENT_DURATION_S = 30;

// Mathematical constants
#ifndef M_PI
constexpr double M_PI = 3.14159265358979323846;
#endif

#ifndef M_2PI
constexpr double M_2PI = 2.0 * M_PI;
#endif

// Error codes
enum class ErrorCode : int {
    Success = 0,
    InitializationFailed = -1,
    DeviceNotFound = -2,
    DeviceOpenFailed = -3,
    InvalidParameter = -4,
    BufferUnderrun = -5,
    BufferOverrun = -6,
    Timeout = -7,
    NoAcousticCoupling = -8,
    MeasurementFailed = -9,
    FileIOError = -10,
    UnknownError = -99
};

// Utility functions
inline std::string error_code_to_string(ErrorCode code) {
    switch (code) {
        case ErrorCode::Success: return "Success";
        case ErrorCode::InitializationFailed: return "Initialization failed";
        case ErrorCode::DeviceNotFound: return "Device not found";
        case ErrorCode::DeviceOpenFailed: return "Device open failed";
        case ErrorCode::InvalidParameter: return "Invalid parameter";
        case ErrorCode::BufferUnderrun: return "Buffer underrun";
        case ErrorCode::BufferOverrun: return "Buffer overrun";
        case ErrorCode::Timeout: return "Timeout";
        case ErrorCode::NoAcousticCoupling: return "No acoustic coupling";
        case ErrorCode::MeasurementFailed: return "Measurement failed";
        case ErrorCode::FileIOError: return "File I/O error";
        default: return "Unknown error";
    }
}

// Version string
inline std::string get_version_string() {
    return std::to_string(SDL_AUDIO_CORE_VERSION_MAJOR) + "." +
           std::to_string(SDL_AUDIO_CORE_VERSION_MINOR) + "." +
           std::to_string(SDL_AUDIO_CORE_VERSION_PATCH);
}

// Common data types
using sample_t = float;
using buffer_t = std::vector<sample_t>;

// Device identifiers
using device_id_t = int;
constexpr device_id_t INVALID_DEVICE_ID = -1;
constexpr device_id_t DEFAULT_DEVICE_ID = -1;

// Audio specifications
struct AudioSpec {
    int sample_rate = DEFAULT_SAMPLE_RATE;
    int channels = DEFAULT_CHANNELS;
    int buffer_size = DEFAULT_BUFFER_SIZE;

    AudioSpec() = default;
    AudioSpec(int sr, int ch, int bs) : sample_rate(sr), channels(ch), buffer_size(bs) {}

    bool operator==(const AudioSpec& other) const {
        return sample_rate == other.sample_rate &&
               channels == other.channels &&
               buffer_size == other.buffer_size;
    }

    bool operator!=(const AudioSpec& other) const {
        return !(*this == other);
    }
};

// Logging levels
enum class LogLevel {
    Debug = 0,
    Info = 1,
    Warning = 2,
    Error = 3,
    None = 4
};

// Simple logging interface
class Logger {
private:
    LogLevel level_ = LogLevel::Info;

public:
    void set_level(LogLevel level) { level_ = level; }
    LogLevel get_level() const { return level_; }

    void debug(const std::string& message) const {
        if (level_ <= LogLevel::Debug) {
            log("DEBUG", message);
        }
    }

    void info(const std::string& message) const {
        if (level_ <= LogLevel::Info) {
            log("INFO", message);
        }
    }

    void warning(const std::string& message) const {
        if (level_ <= LogLevel::Warning) {
            log("WARNING", message);
        }
    }

    void error(const std::string& message) const {
        if (level_ <= LogLevel::Error) {
            log("ERROR", message);
        }
    }

private:
    void log(const std::string& level, const std::string& message) const;
};

// Global logger instance
extern Logger g_logger;

// Convenience macros
#define SDL_AUDIO_LOG_DEBUG(msg) sdl_audio::g_logger.debug(msg)
#define SDL_AUDIO_LOG_INFO(msg) sdl_audio::g_logger.info(msg)
#define SDL_AUDIO_LOG_WARNING(msg) sdl_audio::g_logger.warning(msg)
#define SDL_AUDIO_LOG_ERROR(msg) sdl_audio::g_logger.error(msg)

} // namespace sdl_audio