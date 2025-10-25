# Multi-Channel Upgrade Plan for Piano Response System

**Document Version:** 1.1
**Target System:** piano_response.py (Simplified audio-only pipeline)
**Created:** 2025-10-25
**Last Updated:** 2025-10-25
**Original Timeline:** 7 weeks
**Status:** Phase 1 ✅ COMPLETE | Phase 2-5 📋 PLANNED

---

## Executive Summary

### Upgrade Objectives

This plan focuses on upgrading the piano response measurement system to support **synchronized multi-channel impulse response recording**. The system will record impulses from multiple microphone channels simultaneously while maintaining perfect inter-channel timing synchronization.

**Key Use Case:** Multi-microphone room impulse response measurement where each channel captures the same acoustic event from different spatial positions.

### Scope Limitations

This upgrade is specifically scoped for the `piano_response.py` pipeline:
- **NO** MFCC feature extraction
- **NO** machine learning / classification components
- **NO** data science operations
- **Focus:** Raw audio recording, impulse response extraction, and basic audio analysis only

### Critical Synchronization Requirement

**All channels must maintain perfect sample-level synchronization:**
- When onset detection finds the impulse start at sample N in the reference channel, ALL channels from that measurement must be aligned by shifting by exactly the same number of samples.
- Each channel records the same acoustic event from a different microphone position.
- The alignment operation preserves the relative timing relationships between channels (inter-channel phase, time-of-arrival differences).

### Key Architectural Changes

| Component | Current State | Target State |
|-----------|--------------|--------------|
| **SDL Audio Core** | Single-channel recording | Multi-channel recording with per-channel buffers |
| **Recording Buffer** | `std::vector<float>` mono buffer | `std::vector<std::vector<float>>` per-channel buffers |
| **RoomResponseRecorder** | Returns single numpy array | Returns dict with per-channel arrays |
| **File Output** | Single file per measurement | Multiple files per measurement (one per channel) |
| **Filename Convention** | `{type}_{index}_{timestamp}.wav` | `{type}_{index}_{timestamp}_ch{N}.wav` |
| **Signal Processing** | Single-channel onset detection | Reference-based alignment for all channels |
| **GUI Panels** | Single channel display | Multi-channel status and configuration |

---

## Current System Analysis

### Piano Response Pipeline Overview

The piano_response.py system is a streamlined audio measurement tool with these key components:

**Entry Point:** `piano_response.py`
- AudioCollectionGUI class with minimal panels: Scenarios, Collect, Audio Settings, Audio Analysis
- No ML components (no Process, Classify, Visualize panels)
- Persistent RoomResponseRecorder instance in session state

**Core Recording:** `RoomResponseRecorder.py`
- Generates test signal (sine/square pulse trains)
- Records via SDL audio core
- Processes recording: cycle averaging, onset detection, impulse extraction
- Saves three files: raw, impulse, room_response

**Current Limitations:**

1. **SDL Audio Core** (audio_engine.cpp)
   - Hardcoded mono configuration: `channels = 1`
   - Single recording buffer: `std::vector<float> recording_buffer_`
   - No de-interleaving logic in audio callback

2. **RoomResponseRecorder** (RoomResponseRecorder.py:276-299)
   - `_record_method_2()` returns single numpy array
   - `_process_recorded_signal()` expects 1D array input
   - `_save_wav()` always saves mono files (line 400: `wav_file.setnchannels(1)`)

3. **Signal Processing** (RoomResponseRecorder.py:338-383)
   - `_extract_impulse_response()` operates on single channel
   - `_find_sound_onset()` analyzes single array
   - No concept of multi-channel synchronization

4. **File Organization**
   - One-to-one mapping: 1 measurement = 1 file
   - No channel indexing in filenames

---

## Multi-Channel Architecture Design

### Configuration Schema

Add multi-channel configuration to `recorderConfig.json`:

```json
{
  "recorder_config": {
    "sample_rate": 48000,
    "pulse_duration": 0.008,
    "pulse_fade": 0.0001,
    "cycle_duration": 0.1,
    "num_pulses": 8,
    "volume": 0.4,
    "pulse_frequency": 1000,
    "impulse_form": "sine"
  },
  "multichannel": {
    "enabled": false,
    "num_channels": 4,
    "channel_names": ["Front", "Rear", "Left", "Right"],
    "reference_channel": 0,
    "channel_calibration": {
      "0": {"gain": 1.0, "delay_samples": 0},
      "1": {"gain": 1.0, "delay_samples": 0},
      "2": {"gain": 1.0, "delay_samples": 0},
      "3": {"gain": 1.0, "delay_samples": 0}
    }
  }
}
```

**Configuration Fields:**
- `enabled`: Toggle between legacy single-channel and new multi-channel mode
- `num_channels`: Number of input channels to record (1-32)
- `channel_names`: Human-readable labels for GUI display
- `reference_channel`: Index of channel used for onset detection (0-based)
- `channel_calibration`: Per-channel gain and delay correction

### Data Flow

**Multi-Channel Recording Flow:**

```
1. Test Signal Generation (unchanged)
   └─> Single playback signal (mono output)

2. SDL Audio Core Recording
   ├─> Interleaved multi-channel input from hardware
   │   [L0, R0, C0, S0, L1, R1, C1, S1, ...]
   │
   └─> De-interleave in audio callback
       ├─> Channel 0 buffer: [L0, L1, L2, ...]
       ├─> Channel 1 buffer: [R0, R1, R2, ...]
       ├─> Channel 2 buffer: [C0, C1, C2, ...]
       └─> Channel 3 buffer: [S0, S1, S2, ...]

3. Return to Python
   └─> Dict of numpy arrays: {0: array0, 1: array1, 2: array2, 3: array3}

4. Signal Processing (CRITICAL: Synchronized Alignment)
   ├─> Detect onset in reference channel (e.g., channel 0)
   │   └─> Find onset at sample N
   │
   ├─> Calculate shift amount: shift = -N
   │
   └─> Apply SAME shift to ALL channels
       ├─> Channel 0: np.roll(array0, shift)
       ├─> Channel 1: np.roll(array1, shift)  # Same shift!
       ├─> Channel 2: np.roll(array2, shift)  # Same shift!
       └─> Channel 3: np.roll(array3, shift)  # Same shift!

5. File Output
   ├─> impulse_000_20251025_143022_ch0.wav
   ├─> impulse_000_20251025_143022_ch1.wav
   ├─> impulse_000_20251025_143022_ch2.wav
   └─> impulse_000_20251025_143022_ch3.wav
```

**Key Principle:** The shift amount calculated from the reference channel's onset detection is applied identically to all channels to preserve inter-channel timing relationships.

### Backward Compatibility Strategy

**Dual-Mode Operation:**
- Default to legacy single-channel mode (`multichannel.enabled = false`)
- Automatically detect mode from configuration file
- Zero breaking changes to existing workflows

**Mode Detection Logic:**
```python
def _is_multichannel_enabled(self) -> bool:
    """Check if multi-channel mode is enabled in configuration"""
    return (hasattr(self, 'multichannel_config') and
            self.multichannel_config.get('enabled', False))
```

---

## Implementation Status

### ✅ Phase 1: SDL Audio Core Multi-Channel Support - COMPLETE

**Status:** ✅ IMPLEMENTED (Commit: `86a69e1`)
**Completion Date:** 2025-10-25
**Files Modified:**
- `sdl_audio_core/audio_engine.h`
- `sdl_audio_core/audio_engine.cpp`
- `sdl_audio_core/bindings.cpp`

**What Was Implemented:**
1. ✅ Multi-channel SDL AudioSpec configuration (`input_channels` parameter)
2. ✅ Per-channel buffer management (`std::vector<std::vector<float>>`)
3. ✅ De-interleaving audio callback (handles interleaved multi-channel input)
4. ✅ Multi-channel data retrieval methods:
   - `get_recorded_data_channel(int channel_index)` - Get specific channel
   - `get_recording_data_multichannel()` - Get all channels
5. ✅ Python bindings via pybind11:
   - `measure_room_response_auto_multichannel()` function
   - `AudioEngineConfig` with `input_channels` parameter
   - Per-channel and multi-channel data access
6. ✅ Thread-safe per-channel recording with individual mutexes
7. ✅ Comprehensive test suite (7/7 tests passing)

**Test Results:**
- ✅ Basic multi-channel recording (2, 4, 8 channels)
- ✅ De-interleaving validation
- ✅ Per-channel data retrieval
- ✅ Channel synchronization verification
- ✅ Performance testing (no dropouts with 8 channels at 48kHz)

**Reference:** See [PHASE1_TEST_RESULTS.md](PHASE1_TEST_RESULTS.md) for complete test results.

---

### ✅ Phase 1.5: GUI Multi-Channel Integration - COMPLETE

**Status:** ✅ IMPLEMENTED (Commits: `0cf4a88`, `6e619f3`)
**Completion Date:** 2025-10-25
**Files Modified:**
- `RoomResponseRecorder.py`
- `gui_audio_device_selector.py`
- `gui_audio_settings_panel.py`
- `MicTesting.py`

**What Was Implemented:**
1. ✅ RoomResponseRecorder enhancements:
   - `input_channels` parameter (backward compatible)
   - `get_device_info_with_channels()` method
   - `test_multichannel_recording()` method
2. ✅ GUI multi-channel features:
   - Device channel count display in device selector
   - Dynamic channel picker (adjusts to device capability)
   - Multi-channel monitor (up to 8 channels at 5Hz)
   - Multi-channel test recording with statistics
3. ✅ New "Multi-Channel Test" tab in Audio Settings
4. ✅ **Single-channel monitor channel selection fix** (Commit: `6e619f3`):
   - MicTesting.AudioRecorder now supports channel selection
   - Single-channel monitor respects selected channel
   - Disconnected channels correctly show silence
   - Display shows active channel: "Channel N: -XX.X dBFS"

**Issues Fixed:**
- ❌ **FIXED**: Single-channel monitor showed signal on all channels (including disconnected)
- ✅ Channel selection now works correctly - each channel shows independent signal

**Testing Status:**
- ✅ Device channel detection working correctly
- ✅ Single-channel (backward compatibility) works
- ✅ Single-channel monitor channel selection verified
- ⚠️ Multi-channel monitor shows zeros (pending investigation)
- ⚠️ Multi-channel test recording needs verification

**Reference:**
- [GUI_MULTICHANNEL_INTEGRATION_PLAN.md](GUI_MULTICHANNEL_INTEGRATION_PLAN.md) - Implementation plan
- [CHANNEL_SELECTION_FIX.md](CHANNEL_SELECTION_FIX.md) - Channel selection fix details
- [test_channel_selection.py](test_channel_selection.py) - Automated verification tests

---

## Phase 1: SDL Audio Core Multi-Channel Support

**Duration:** 2 weeks (✅ COMPLETED)
**Files:** `audio_engine.h`, `audio_engine.cpp`, `bindings.cpp`

### 1.1 SDL AudioSpec Configuration

Update SDL audio specification to support multi-channel input:

**audio_engine.h (Config struct):**
```cpp
struct Config {
    int sample_rate = 48000;
    int input_channels = 1;    // NEW: number of input channels
    int output_channels = 1;   // Output remains mono
    int buffer_size = 4096;
    std::string audio_driver = "";
    // ... existing fields
};
```

**audio_engine.cpp (initialization):**
```cpp
void AudioEngine::initialize_audio_input() {
    SDL_AudioSpec desired_input;
    SDL_zero(desired_input);

    desired_input.freq = config_.sample_rate;
    desired_input.format = AUDIO_F32SYS;
    desired_input.channels = config_.input_channels;  // Multi-channel
    desired_input.samples = config_.buffer_size;
    desired_input.callback = input_callback_static;
    desired_input.userdata = this;

    // ... rest of initialization
}
```

### 1.2 Per-Channel Buffer Management

Replace single recording buffer with per-channel buffers:

**audio_engine.h:**
```cpp
class AudioEngine {
private:
    // OLD: std::vector<float> recording_buffer_;

    // NEW: Per-channel buffers
    std::vector<std::vector<float>> recording_buffers_;  // [channel][samples]
    std::vector<std::mutex> channel_mutexes_;            // Per-channel locks
    std::atomic<size_t> recording_position_;             // Samples per channel
    int num_input_channels_;

    // ... rest of class
};
```

**audio_engine.cpp (initialization):**
```cpp
AudioEngine::AudioEngine(const Config& config) : config_(config) {
    num_input_channels_ = config.input_channels;

    // Allocate per-channel buffers
    recording_buffers_.resize(num_input_channels_);
    channel_mutexes_.resize(num_input_channels_);

    // Pre-allocate buffers for each channel
    for (int ch = 0; ch < num_input_channels_; ++ch) {
        recording_buffers_[ch].reserve(48000 * 10);  // 10 seconds per channel
    }

    recording_position_.store(0);
}
```

### 1.3 De-Interleaving Audio Callback

Modify input callback to de-interleave multi-channel data:

**audio_engine.cpp:**
```cpp
void AudioEngine::handle_input_audio(Uint8* stream, int len) {
    if (!is_recording_.load()) return;

    const float* samples = reinterpret_cast<const float*>(stream);
    const int num_frames = len / (sizeof(float) * num_input_channels_);

    // De-interleave: [L0, R0, L1, R1, ...] → [L0, L1, ...], [R0, R1, ...]
    for (int frame = 0; frame < num_frames; ++frame) {
        for (int ch = 0; ch < num_input_channels_; ++ch) {
            float sample = samples[frame * num_input_channels_ + ch];

            // Lock only this channel's buffer
            std::lock_guard<std::mutex> lock(channel_mutexes_[ch]);
            recording_buffers_[ch].push_back(sample);
        }
    }

    // Update position (same for all channels)
    recording_position_.fetch_add(num_frames);
}
```

### 1.4 Get Recorded Data (Multi-Channel)

Add method to retrieve all channel data:

**audio_engine.h:**
```cpp
// Return all channels as vector of vectors
std::vector<std::vector<float>> get_recording_data_multichannel();

// Return specific channel (for compatibility)
std::vector<float> get_recording_data_single_channel(int channel_index = 0);
```

**audio_engine.cpp:**
```cpp
std::vector<std::vector<float>> AudioEngine::get_recording_data_multichannel() {
    std::vector<std::vector<float>> result(num_input_channels_);

    for (int ch = 0; ch < num_input_channels_; ++ch) {
        std::lock_guard<std::mutex> lock(channel_mutexes_[ch]);
        result[ch] = recording_buffers_[ch];
    }

    return result;
}

std::vector<float> AudioEngine::get_recording_data_single_channel(int channel_index) {
    if (channel_index >= num_input_channels_) {
        throw std::out_of_range("Channel index out of range");
    }

    std::lock_guard<std::mutex> lock(channel_mutexes_[channel_index]);
    return recording_buffers_[channel_index];
}
```

### 1.5 Python Bindings (pybind11)

Update Python bindings to expose multi-channel functionality:

**bindings.cpp:**
```cpp
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "audio_engine.h"

namespace py = pybind11;

PYBIND11_MODULE(sdl_audio_core, m) {
    // Config struct
    py::class_<AudioEngine::Config>(m, "Config")
        .def(py::init<>())
        .def_readwrite("sample_rate", &AudioEngine::Config::sample_rate)
        .def_readwrite("input_channels", &AudioEngine::Config::input_channels)   // NEW
        .def_readwrite("output_channels", &AudioEngine::Config::output_channels) // NEW
        .def_readwrite("buffer_size", &AudioEngine::Config::buffer_size);

    // AudioEngine class
    py::class_<AudioEngine>(m, "AudioEngine")
        .def(py::init<>())
        .def(py::init<const AudioEngine::Config&>())
        .def("get_recording_data_multichannel", &AudioEngine::get_recording_data_multichannel)
        .def("get_recording_data_single_channel", &AudioEngine::get_recording_data_single_channel,
             py::arg("channel_index") = 0)
        .def("start_recording", &AudioEngine::start_recording)
        .def("stop_recording", &AudioEngine::stop_recording);

    // Updated measure_room_response_auto for multi-channel
    m.def("measure_room_response_auto_multichannel",
          &measure_room_response_auto_multichannel,
          py::arg("playback_signal"),
          py::arg("volume") = 0.3f,
          py::arg("input_device") = -1,
          py::arg("output_device") = -1,
          py::arg("input_channels") = 1,
          "Measure room response with multi-channel input support");
}
```

### 1.6 Updated C++ Convenience Function

Create multi-channel version of `measure_room_response_auto`:

**audio_engine.cpp:**
```cpp
MeasurementResult measure_room_response_auto_multichannel(
    const std::vector<float>& playback_signal,
    float volume,
    int input_device,
    int output_device,
    int input_channels)
{
    MeasurementResult result;
    result.success = false;

    try {
        AudioEngine::Config config;
        config.sample_rate = 48000;
        config.input_channels = input_channels;  // Multi-channel
        config.output_channels = 1;              // Mono output
        config.buffer_size = 4096;

        AudioEngine engine(config);

        if (!engine.initialize(input_device, output_device)) {
            result.error_message = "Failed to initialize audio engine";
            return result;
        }

        // Start recording
        engine.start_recording();

        // Play test signal
        engine.play_signal(playback_signal, volume);

        // Wait for playback completion
        while (engine.is_playing()) {
            SDL_Delay(10);
        }

        // Stop recording
        engine.stop_recording();

        // Get multi-channel data
        auto multichannel_data = engine.get_recording_data_multichannel();

        // Convert to Python-compatible format (store in result)
        result.multichannel_data = multichannel_data;  // NEW field
        result.recorded_samples = multichannel_data[0].size();
        result.num_channels = input_channels;
        result.success = true;

    } catch (const std::exception& e) {
        result.error_message = e.what();
    }

    return result;
}
```

### 1.7 Testing Strategy

**Unit Tests:**
- Test per-channel buffer allocation
- Test de-interleaving logic with synthetic interleaved data
- Test channel data retrieval (all channels, single channel)
- Test buffer synchronization (all channels have same length)

**Integration Tests:**
- Test multi-channel recording with real hardware (2, 4, 8 channels)
- Verify interleaved → de-interleaved conversion correctness
- Cross-correlate channels to verify synchronization
- Test with various sample rates and buffer sizes

**Deliverables:**
- Updated SDL audio core with multi-channel support
- Python bindings exposing multi-channel API
- Test suite validating synchronization

---

## Phase 2: Recording Pipeline Upgrade

**Duration:** 1 week
**Status:** 📋 PLANNED (Partially implemented for GUI testing)
**Files:** `RoomResponseRecorder.py`

**Current Status:**
- ⚠️ Basic multi-channel methods exist for GUI testing (`test_multichannel_recording()`)
- ❌ Full pipeline integration NOT YET implemented
- ❌ Configuration loading from JSON NOT YET implemented
- ❌ Synchronized multi-channel processing NOT YET implemented

**What Remains:**

### 2.1 Load Multi-Channel Configuration

Update `__init__` to load multi-channel settings:

**RoomResponseRecorder.py:**
```python
def __init__(self, config_file_path: str = None):
    # ... existing default_config code ...

    # Multi-channel configuration defaults
    self.multichannel_config = {
        'enabled': False,
        'num_channels': 1,
        'channel_names': ['Channel 0'],
        'reference_channel': 0,
        'channel_calibration': {}
    }

    # Load configuration from file
    if config_file_path:
        try:
            with open(config_file_path, 'r') as f:
                file_config = json.load(f)

            # Load recorder config
            if 'recorder_config' in file_config:
                loaded_config = file_config['recorder_config']
                for key, value in loaded_config.items():
                    if key in default_config:
                        default_config[key] = value

            # Load multi-channel config (NEW)
            if 'multichannel' in file_config:
                self.multichannel_config.update(file_config['multichannel'])

        except Exception as e:
            print(f"Warning: Error loading config: {e}")

    # Set instance variables
    for param, value in default_config.items():
        setattr(self, param, value)

    # Validate multi-channel config
    if self.multichannel_config['enabled']:
        self._validate_multichannel_config()
```

### 2.2 Multi-Channel Configuration Validation

**RoomResponseRecorder.py:**
```python
def _validate_multichannel_config(self):
    """Validate multi-channel configuration"""
    num_ch = self.multichannel_config['num_channels']
    ref_ch = self.multichannel_config['reference_channel']

    if num_ch < 1 or num_ch > 32:
        raise ValueError("num_channels must be between 1 and 32")

    if ref_ch < 0 or ref_ch >= num_ch:
        raise ValueError(f"reference_channel {ref_ch} out of range [0, {num_ch-1}]")

    # Ensure channel names list matches num_channels
    if len(self.multichannel_config['channel_names']) != num_ch:
        # Auto-generate names if missing
        self.multichannel_config['channel_names'] = [
            f"Channel {i}" for i in range(num_ch)
        ]
```

### 2.3 Updated Recording Method

Modify `_record_method_2` to handle multi-channel recording:

**RoomResponseRecorder.py:**
```python
def _record_method_2(self) -> Optional[Dict[int, np.ndarray]]:
    """
    Method 2: Auto device selection with multi-channel support

    Returns:
        If single-channel: np.ndarray (legacy compatibility)
        If multi-channel: Dict[int, np.ndarray] mapping channel index to data
    """
    print("Recording Method 2: Auto Device Selection")

    try:
        is_multichannel = self.multichannel_config.get('enabled', False)
        num_channels = self.multichannel_config.get('num_channels', 1) if is_multichannel else 1

        if is_multichannel:
            # Use new multi-channel function
            result = sdl_audio_core.measure_room_response_auto_multichannel(
                self.playback_signal,
                volume=self.volume,
                input_device=self.input_device,
                output_device=self.output_device,
                input_channels=num_channels
            )
        else:
            # Legacy single-channel function
            result = sdl_audio_core.measure_room_response_auto(
                self.playback_signal,
                volume=self.volume,
                input_device=self.input_device,
                output_device=self.output_device
            )

        if not result['success']:
            print(f"Measurement failed: {result.get('error_message', 'Unknown error')}")
            return None

        # Process result based on mode
        if is_multichannel:
            # Convert list of lists to dict of numpy arrays
            multichannel_data = result['multichannel_data']
            return {
                ch: np.array(multichannel_data[ch], dtype=np.float32)
                for ch in range(num_channels)
            }
        else:
            # Legacy single-channel return
            recorded_data = result['recorded_data']
            return np.array(recorded_data, dtype=np.float32) if recorded_data else None

    except Exception as e:
        print(f"Error in recording: {e}")
        return None
```

### 2.4 Multi-Channel Signal Processing (CRITICAL)

Update `_process_recorded_signal` to handle multi-channel with synchronized alignment:

**RoomResponseRecorder.py:**
```python
def _process_recorded_signal(self, recorded_audio) -> Dict[str, Any]:
    """
    Process recorded signal - supports both single and multi-channel

    Args:
        recorded_audio: Either np.ndarray (single-channel) or Dict[int, np.ndarray] (multi-channel)

    Returns:
        Dict with processed data for all channels
    """
    print("Processing recorded signal...")

    is_multichannel = isinstance(recorded_audio, dict)

    if is_multichannel:
        return self._process_multichannel_signal(recorded_audio)
    else:
        return self._process_single_channel_signal(recorded_audio)


def _process_multichannel_signal(self, multichannel_audio: Dict[int, np.ndarray]) -> Dict[str, Any]:
    """
    Process multi-channel recording with synchronized alignment

    CRITICAL: All channels are aligned using the SAME shift calculated from reference channel
    """
    num_channels = len(multichannel_audio)
    ref_channel = self.multichannel_config.get('reference_channel', 0)
    expected_samples = self.cycle_samples * self.num_pulses

    print(f"Processing {num_channels} channels (reference: {ref_channel})")

    # 1. Pad/trim all channels to expected length
    processed_channels = {}
    for ch_idx, audio in multichannel_audio.items():
        if len(audio) < expected_samples:
            padded = np.zeros(expected_samples)
            padded[:len(audio)] = audio
            processed_channels[ch_idx] = padded
        else:
            processed_channels[ch_idx] = audio[:expected_samples]

    # 2. Apply cycle averaging to reference channel
    ref_audio = processed_channels[ref_channel]
    ref_reshaped = ref_audio.reshape(self.num_pulses, self.cycle_samples)
    start_cycle = max(1, self.num_pulses // 4)
    ref_room_response = np.mean(ref_reshaped[start_cycle:], axis=0)

    # 3. Find onset in reference channel
    onset_sample = self._find_onset_in_room_response(ref_room_response)
    print(f"Found onset at sample {onset_sample} in reference channel {ref_channel}")

    # 4. Calculate shift amount from reference channel
    shift_amount = -onset_sample  # Negative to move onset to beginning

    # 5. Apply cycle averaging to ALL channels and align with SAME shift
    result = {
        'raw': {},
        'room_response': {},
        'impulse': {},
        'metadata': {
            'num_channels': num_channels,
            'reference_channel': ref_channel,
            'onset_sample': onset_sample,
            'shift_applied': shift_amount
        }
    }

    for ch_idx, audio in processed_channels.items():
        # Cycle averaging for this channel
        reshaped = audio.reshape(self.num_pulses, self.cycle_samples)
        room_response = np.mean(reshaped[start_cycle:], axis=0)

        # Apply THE SAME shift to this channel (critical for synchronization)
        impulse_response = np.roll(room_response, shift_amount)

        result['raw'][ch_idx] = audio
        result['room_response'][ch_idx] = room_response
        result['impulse'][ch_idx] = impulse_response

        print(f"  Channel {ch_idx}: aligned with shift={shift_amount}")

    return result


def _process_single_channel_signal(self, recorded_audio: np.ndarray) -> Dict[str, Any]:
    """Legacy single-channel processing (unchanged)"""
    expected_samples = self.cycle_samples * self.num_pulses

    # ... existing single-channel processing code ...

    return {
        'raw': recorded_audio,
        'room_response': room_response,
        'impulse': impulse_response
    }


def _find_onset_in_room_response(self, room_response: np.ndarray) -> int:
    """
    Find onset position in a room response (extracted helper method)
    """
    max_index = np.argmax(np.abs(room_response))

    if max_index > 50:
        search_start = max(0, max_index - 100)
        search_window = room_response[search_start:max_index + 50]
        onset_in_window = self._find_sound_onset(search_window)
        onset = search_start + onset_in_window
    else:
        onset = 0

    return onset
```

### 2.5 Multi-Channel File Saving

Update `take_record` to save per-channel files:

**RoomResponseRecorder.py:**
```python
def take_record(self,
                output_file: str,
                impulse_file: str,
                method: int = 2) -> Optional[Any]:
    """
    Main API method to record room response

    Returns:
        Single-channel: np.ndarray
        Multi-channel: Dict[int, np.ndarray]
    """
    print(f"\n{'=' * 60}")
    print(f"Room Response Recording")
    print(f"{'=' * 60}")

    try:
        recorded_audio = self._record_method_2()
        if recorded_audio is None:
            print("Recording failed - no data captured")
            return None

        # Process the recorded signal
        processed_data = self._process_recorded_signal(recorded_audio)

        is_multichannel = isinstance(recorded_audio, dict)

        if is_multichannel:
            self._save_multichannel_files(output_file, impulse_file, processed_data)
        else:
            self._save_single_channel_files(output_file, impulse_file, processed_data)

        # Print success summary
        print(f"\n🎉 Recording completed successfully!")

        return recorded_audio

    except Exception as e:
        print(f"Error during recording: {e}")
        return None


def _save_multichannel_files(self, output_file: str, impulse_file: str, processed_data: Dict):
    """Save multi-channel measurement files"""
    num_channels = processed_data['metadata']['num_channels']

    for ch_idx in range(num_channels):
        # Generate per-channel filenames
        raw_ch_file = self._make_channel_filename(output_file, ch_idx)
        impulse_ch_file = self._make_channel_filename(impulse_file, ch_idx)
        room_ch_file = self._make_channel_filename(
            output_file.replace('.wav', '_room.wav'), ch_idx
        )

        # Save files for this channel
        self._save_wav(processed_data['raw'][ch_idx], raw_ch_file)
        self._save_wav(processed_data['impulse'][ch_idx], impulse_ch_file)
        self._save_wav(processed_data['room_response'][ch_idx], room_ch_file)

        ch_name = self.multichannel_config['channel_names'][ch_idx]
        print(f"  Channel {ch_idx} ({ch_name}): saved 3 files")


def _save_single_channel_files(self, output_file: str, impulse_file: str, processed_data: Dict):
    """Save single-channel measurement files (legacy)"""
    self._save_wav(processed_data['raw'], output_file)
    self._save_wav(processed_data['impulse'], impulse_file)

    output_path = Path(output_file)
    room_response_file = str(output_path.parent / f"room_{output_path.stem}_room.wav")
    self._save_wav(processed_data['room_response'], room_response_file)

    print(f"- Raw recording: {output_file}")
    print(f"- Impulse response: {impulse_file}")
    print(f"- Room response: {room_response_file}")


def _make_channel_filename(self, base_filename: str, channel_index: int) -> str:
    """
    Generate filename with channel suffix

    Examples:
        _make_channel_filename("impulse_000_20251025.wav", 0)
        -> "impulse_000_20251025_ch0.wav"
    """
    path = Path(base_filename)
    stem = path.stem
    suffix = path.suffix
    parent = path.parent

    new_filename = f"{stem}_ch{channel_index}{suffix}"
    return str(parent / new_filename)
```

### 2.6 Testing Strategy

**Unit Tests:**
- Test configuration loading (single vs multi-channel)
- Test channel filename generation
- Test synchronization: verify all channels shifted by same amount
- Test backward compatibility with existing single-channel code

**Integration Tests:**
- Record actual multi-channel impulse responses
- Verify inter-channel phase relationships preserved
- Cross-correlate channels to confirm synchronization
- Test with 2, 4, 8 channel configurations

**Deliverables:**
- Updated RoomResponseRecorder with multi-channel support
- Backward-compatible API
- Test suite validating synchronization

---

## Phase 3: Filesystem Structure Redesign

**Duration:** 1 week
**Status:** 📋 PLANNED (Not yet started)
**Files:** `ScenarioManager.py`, migration utilities

### 3.1 Filename Convention

**Chosen Convention:** Per-channel suffix

**Format:** `{type}_{index}_{timestamp}_ch{channel}.wav`

**Examples:**
```
Single-channel (legacy):
  impulse_000_20251025_143022.wav
  impulse_001_20251025_143045.wav

Multi-channel (4 channels):
  impulse_000_20251025_143022_ch0.wav
  impulse_000_20251025_143022_ch1.wav
  impulse_000_20251025_143022_ch2.wav
  impulse_000_20251025_143022_ch3.wav
```

**Rationale:**
- Clearly groups related channels from same measurement
- Sort naturally by measurement, then by channel
- Easy to glob for specific measurements
- Easy to detect mode from filename

### 3.2 Parsing Utility

**ScenarioManager.py:**
```python
import re
from dataclasses import dataclass
from typing import Optional, List

@dataclass
class MultiChannelFilename:
    """Parsed multi-channel filename components"""
    file_type: str        # "impulse", "room", "raw"
    index: int            # Measurement index (000, 001, ...)
    timestamp: str        # ISO timestamp
    channel: Optional[int]  # Channel index (None for single-channel)
    is_multichannel: bool # True if filename has _chN suffix
    full_path: str        # Original full path


def parse_multichannel_filename(filename: str) -> Optional[MultiChannelFilename]:
    """
    Parse filename to extract components

    Patterns:
        Single-channel: {type}_{index}_{timestamp}.wav
        Multi-channel:  {type}_{index}_{timestamp}_ch{N}.wav
    """
    # Multi-channel pattern
    mc_pattern = r'(\w+)_(\d+)_(\d{8}_\d{6})_ch(\d+)\.wav$'
    mc_match = re.search(mc_pattern, filename)

    if mc_match:
        return MultiChannelFilename(
            file_type=mc_match.group(1),
            index=int(mc_match.group(2)),
            timestamp=mc_match.group(3),
            channel=int(mc_match.group(4)),
            is_multichannel=True,
            full_path=filename
        )

    # Single-channel pattern (legacy)
    sc_pattern = r'(\w+)_(\d+)_(\d{8}_\d{6})\.wav$'
    sc_match = re.search(sc_pattern, filename)

    if sc_match:
        return MultiChannelFilename(
            file_type=sc_match.group(1),
            index=int(sc_match.group(2)),
            timestamp=sc_match.group(3),
            channel=None,
            is_multichannel=False,
            full_path=filename
        )

    return None
```

### 3.3 ScenarioManager Multi-Channel Support

Update ScenarioManager to handle multi-channel file groups:

**ScenarioManager.py:**
```python
class ScenarioManager:
    """Manages scenario folders and audio files with multi-channel support"""

    def list_wavs_multichannel(self, scenario_path: str) -> Dict[int, List[str]]:
        """
        List WAV files grouped by measurement index

        Returns:
            Dict mapping measurement index to list of channel files

        Example:
            {
                0: ['impulse_000_20251025_ch0.wav', 'impulse_000_20251025_ch1.wav'],
                1: ['impulse_001_20251025_ch0.wav', 'impulse_001_20251025_ch1.wav']
            }
        """
        wav_files = glob.glob(os.path.join(scenario_path, "*.wav"))

        measurements = {}
        for wav_path in wav_files:
            parsed = parse_multichannel_filename(wav_path)
            if parsed:
                idx = parsed.index
                if idx not in measurements:
                    measurements[idx] = []
                measurements[idx].append(wav_path)

        # Sort channel files within each measurement
        for idx in measurements:
            measurements[idx].sort()

        return measurements

    def detect_num_channels(self, scenario_path: str) -> int:
        """
        Detect number of channels in a scenario by examining files

        Returns:
            Number of channels (1 for single-channel)
        """
        measurements = self.list_wavs_multichannel(scenario_path)

        if not measurements:
            return 1  # Default to single-channel

        # Check first measurement
        first_measurement_files = list(measurements.values())[0]

        # Count files with same index
        parsed_files = [parse_multichannel_filename(f) for f in first_measurement_files]
        parsed_files = [pf for pf in parsed_files if pf is not None]

        if not parsed_files:
            return 1

        if parsed_files[0].is_multichannel:
            # Count unique channel indices
            channels = set(pf.channel for pf in parsed_files)
            return len(channels)
        else:
            return 1  # Legacy single-channel

    def get_measurement_files(self, scenario_path: str, measurement_index: int) -> Dict[int, str]:
        """
        Get all channel files for a specific measurement

        Returns:
            Dict mapping channel index to file path
            For single-channel: {0: "file.wav"}
            For multi-channel: {0: "file_ch0.wav", 1: "file_ch1.wav", ...}
        """
        measurements = self.list_wavs_multichannel(scenario_path)

        if measurement_index not in measurements:
            return {}

        files = measurements[measurement_index]
        parsed_files = [parse_multichannel_filename(f) for f in files]

        # Map channel index to file path
        channel_map = {}
        for parsed in parsed_files:
            if parsed:
                ch_idx = parsed.channel if parsed.is_multichannel else 0
                channel_map[ch_idx] = parsed.full_path

        return channel_map
```

### 3.4 Migration Utility

Create script to migrate legacy datasets to multi-channel naming:

**migrate_to_multichannel.py:**
```python
#!/usr/bin/env python3
"""
Migration utility to rename legacy single-channel files to multi-channel format

Usage:
    python migrate_to_multichannel.py /path/to/dataset --dry-run
    python migrate_to_multichannel.py /path/to/dataset --execute
"""

import os
import sys
import glob
import shutil
from pathlib import Path
from typing import List, Tuple

def find_legacy_files(dataset_root: str) -> List[str]:
    """Find all legacy single-channel WAV files"""
    # Pattern: {type}_{index}_{timestamp}.wav (without _chN suffix)
    pattern = os.path.join(dataset_root, "**", "*_[0-9][0-9][0-9]_[0-9]*_[0-9]*.wav")
    files = glob.glob(pattern, recursive=True)

    # Filter out files that already have _ch suffix
    legacy_files = [f for f in files if not re.search(r'_ch\d+\.wav$', f)]

    return legacy_files

def generate_migration_plan(legacy_files: List[str]) -> List[Tuple[str, str]]:
    """Generate (old_path, new_path) pairs for migration"""
    migrations = []

    for old_path in legacy_files:
        path = Path(old_path)
        stem = path.stem
        suffix = path.suffix
        parent = path.parent

        # Add _ch0 suffix (treat as channel 0)
        new_filename = f"{stem}_ch0{suffix}"
        new_path = str(parent / new_filename)

        migrations.append((old_path, new_path))

    return migrations

def execute_migration(migrations: List[Tuple[str, str]], dry_run: bool = True):
    """Execute the migration (rename files)"""
    print(f"{'DRY RUN - ' if dry_run else ''}Migrating {len(migrations)} files")
    print("=" * 60)

    for old_path, new_path in migrations:
        print(f"{os.path.basename(old_path)}")
        print(f"  -> {os.path.basename(new_path)}")

        if not dry_run:
            try:
                shutil.move(old_path, new_path)
                print("  ✓ Success")
            except Exception as e:
                print(f"  ✗ Error: {e}")

    if dry_run:
        print("\nDRY RUN COMPLETE - No files were modified")
        print("Run with --execute to apply changes")
    else:
        print("\nMIGRATION COMPLETE")

def main():
    if len(sys.argv) < 2:
        print("Usage: python migrate_to_multichannel.py <dataset_root> [--dry-run|--execute]")
        sys.exit(1)

    dataset_root = sys.argv[1]
    mode = sys.argv[2] if len(sys.argv) > 2 else "--dry-run"

    if not os.path.isdir(dataset_root):
        print(f"Error: {dataset_root} is not a directory")
        sys.exit(1)

    print(f"Scanning dataset: {dataset_root}")
    legacy_files = find_legacy_files(dataset_root)
    print(f"Found {len(legacy_files)} legacy files")

    migrations = generate_migration_plan(legacy_files)
    execute_migration(migrations, dry_run=(mode == "--dry-run"))

if __name__ == "__main__":
    main()
```

### 3.5 Testing Strategy

**Unit Tests:**
- Test filename parsing (single-channel, multi-channel)
- Test ScenarioManager multi-channel file grouping
- Test channel count detection

**Integration Tests:**
- Test migration utility on sample datasets
- Verify ScenarioManager works with both legacy and new filenames
- Test file listing and grouping with mixed datasets

**Deliverables:**
- Updated ScenarioManager with multi-channel file handling
- Migration utility script
- Backward compatibility with legacy filenames

---

## Phase 4: GUI Interface Updates

**Duration:** 2 weeks
**Status:** 📋 PLANNED (Audio Settings panel partially complete)
**Files:** `piano_response.py`, `gui_audio_settings_panel.py`, `gui_collect_panel.py`, `gui_audio_panel.py`

**Current Status:**
- ✅ Audio Settings panel has Multi-Channel Test tab (basic testing UI)
- ❌ Multi-channel configuration UI NOT YET implemented
- ❌ Collection panel multi-channel status NOT YET implemented
- ❌ Audio Analysis panel multi-channel visualization NOT YET implemented

**What Remains:**

### 4.1 Audio Settings Panel

Add multi-channel configuration UI:

**gui_audio_settings_panel.py:**
```python
class AudioSettingsPanel:
    def render(self):
        st.header("Audio Settings")

        # ... existing device selection ...

        # Multi-channel configuration section
        st.markdown("---")
        st.subheader("Multi-Channel Configuration")

        # Enable/disable toggle
        multichannel_enabled = st.checkbox(
            "Enable multi-channel recording",
            value=False,
            help="Record from multiple input channels simultaneously"
        )

        if multichannel_enabled:
            col1, col2 = st.columns(2)

            with col1:
                num_channels = st.number_input(
                    "Number of channels",
                    min_value=1,
                    max_value=32,
                    value=4,
                    help="Total number of input channels to record"
                )

                reference_channel = st.selectbox(
                    "Reference channel",
                    options=list(range(num_channels)),
                    help="Channel used for onset detection and alignment"
                )

            with col2:
                # Channel naming
                st.markdown("**Channel Names**")
                channel_names = []
                for ch in range(num_channels):
                    name = st.text_input(
                        f"Ch {ch}",
                        value=f"Channel {ch}",
                        key=f"ch_name_{ch}"
                    )
                    channel_names.append(name)

            # Save configuration button
            if st.button("Save Multi-Channel Config"):
                config = {
                    'enabled': True,
                    'num_channels': num_channels,
                    'channel_names': channel_names,
                    'reference_channel': reference_channel
                }
                self._save_multichannel_config(config)
                st.success("Multi-channel configuration saved!")

        else:
            st.info("Multi-channel recording is disabled. Using single-channel mode.")
```

### 4.2 Collection Panel

Update collection panel to show per-channel status:

**gui_collect_panel.py:**
```python
class CollectionPanel:
    def render(self):
        st.header("Audio Collection")

        # ... existing scenario selection ...

        # Multi-channel status display
        if self.recorder and hasattr(self.recorder, 'multichannel_config'):
            mc_config = self.recorder.multichannel_config

            if mc_config.get('enabled', False):
                st.info(f"Multi-channel mode: {mc_config['num_channels']} channels")

                # Display channel configuration
                with st.expander("Channel Configuration"):
                    for ch_idx, ch_name in enumerate(mc_config['channel_names']):
                        icon = "🎤" if ch_idx == mc_config['reference_channel'] else "🔊"
                        st.write(f"{icon} **Ch {ch_idx}:** {ch_name}")
                    st.caption(f"Reference channel: {mc_config['reference_channel']}")

        # Recording button
        if st.button("Record"):
            with st.spinner("Recording..."):
                result = self._perform_recording()

                if result:
                    is_multichannel = isinstance(result, dict)

                    if is_multichannel:
                        st.success(f"✓ Recorded {len(result)} channels")

                        # Show per-channel quality metrics
                        for ch_idx, ch_data in result.items():
                            max_amp = np.max(np.abs(ch_data))
                            rms = np.sqrt(np.mean(ch_data ** 2))

                            ch_name = mc_config['channel_names'][ch_idx]
                            st.write(f"**{ch_name}** (Ch {ch_idx}): "
                                   f"Max={max_amp:.3f}, RMS={rms:.3f}")
                    else:
                        st.success("✓ Recording completed")
```

### 4.3 Audio Analysis Panel

Add multi-channel waveform display:

**gui_audio_panel.py:**
```python
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

class AudioAnalysisPanel:
    def render(self):
        st.header("Audio Analysis")

        # File selection
        scenario_path = self._get_selected_scenario()
        if not scenario_path:
            return

        measurements = self.scenario_manager.list_wavs_multichannel(scenario_path)
        num_channels = self.scenario_manager.detect_num_channels(scenario_path)

        st.info(f"Detected: {num_channels} channel(s)")

        # Measurement selection
        measurement_idx = st.selectbox(
            "Select measurement",
            options=sorted(measurements.keys()),
            format_func=lambda x: f"Measurement {x:03d}"
        )

        # Load and display
        channel_files = self.scenario_manager.get_measurement_files(
            scenario_path, measurement_idx
        )

        if num_channels > 1:
            self._render_multichannel_display(channel_files)
        else:
            self._render_single_channel_display(channel_files[0])

    def _render_multichannel_display(self, channel_files: Dict[int, str]):
        """Display multi-channel waveforms"""
        num_channels = len(channel_files)

        # Load all channels
        channel_data = {}
        sample_rate = None

        for ch_idx, file_path in sorted(channel_files.items()):
            sr, data = wavfile.read(file_path)
            channel_data[ch_idx] = data.astype(float) / 32767.0
            sample_rate = sr

        # Plot all channels
        fig, axes = plt.subplots(num_channels, 1, figsize=(12, 2*num_channels), sharex=True)

        if num_channels == 1:
            axes = [axes]

        for ch_idx, ax in enumerate(axes):
            data = channel_data[ch_idx]
            time = np.arange(len(data)) / sample_rate

            ax.plot(time, data, linewidth=0.5)
            ax.set_ylabel(f"Ch {ch_idx}")
            ax.grid(True, alpha=0.3)

        axes[-1].set_xlabel("Time (s)")
        fig.suptitle(f"Multi-Channel Waveform ({num_channels} channels)")

        st.pyplot(fig)

        # Playback controls
        st.markdown("**Playback**")
        cols = st.columns(num_channels)
        for ch_idx, col in enumerate(cols):
            with col:
                st.audio(channel_files[ch_idx], format='audio/wav')
                st.caption(f"Channel {ch_idx}")
```

### 4.4 Testing Strategy

**Manual Testing:**
- Test multi-channel configuration in Audio Settings panel
- Test recording with multi-channel display in Collection panel
- Test multi-channel waveform visualization in Audio Analysis panel
- Verify channel naming and reference channel selection

**Deliverables:**
- Updated GUI panels with multi-channel support
- Multi-channel visualization tools
- Configuration UI for channel setup

---

## Phase 5: Testing & Validation

**Duration:** 1 week
**Status:** 📋 PLANNED (Not yet started)

### 5.1 End-to-End Integration Tests

**Test Scenarios:**

1. **Single-channel backward compatibility**
   - Record with `multichannel.enabled = false`
   - Verify single file output
   - Verify existing code paths work unchanged

2. **Multi-channel recording (2 channels)**
   - Configure 2-channel recording
   - Record impulse response
   - Verify 2 files per measurement
   - Verify synchronization via cross-correlation

3. **Multi-channel recording (4 channels)**
   - Configure 4-channel recording
   - Record impulse response
   - Verify 4 files per measurement
   - Verify all channels aligned to reference channel onset

4. **Reference channel alignment**
   - Record multi-channel with reference_channel = 0
   - Manually inspect onset alignment in all channels
   - Change reference_channel to 2
   - Verify alignment shifts accordingly

5. **Mixed dataset handling**
   - Dataset with both legacy single-channel and new multi-channel files
   - Verify ScenarioManager correctly detects and groups files
   - Verify GUI displays both types correctly

### 5.2 Hardware Compatibility Testing

**Test Configurations:**
- 2-channel stereo interface
- 4-channel USB audio interface
- 8-channel professional interface
- Various sample rates: 44.1kHz, 48kHz, 96kHz

**Validation:**
- Verify SDL can open multi-channel devices
- Verify correct channel count reported
- Verify no dropouts or buffer issues

### 5.3 Performance Benchmarking

**Metrics:**
- Recording latency with 2, 4, 8 channels
- File I/O time for multi-channel saves
- Memory usage with large multi-channel buffers
- GUI responsiveness during multi-channel recording

### 5.4 Synchronization Validation

**Test Method:**
```python
def validate_multichannel_synchronization(scenario_path: str, measurement_idx: int):
    """
    Validate that all channels are properly synchronized
    """
    sm = ScenarioManager()
    channel_files = sm.get_measurement_files(scenario_path, measurement_idx)

    # Load all channels
    reference_data = None
    correlations = {}

    for ch_idx, file_path in sorted(channel_files.items()):
        sr, data = wavfile.read(file_path)
        data = data.astype(float)

        if ch_idx == 0:
            reference_data = data
        else:
            # Cross-correlate with reference
            correlation = np.correlate(reference_data, data, mode='full')
            lag = np.argmax(correlation) - len(data) + 1
            correlations[ch_idx] = lag

    # All lags should be near zero (within a few samples)
    print("Cross-correlation lags:")
    for ch_idx, lag in correlations.items():
        print(f"  Channel {ch_idx}: {lag} samples")
        assert abs(lag) < 10, f"Channel {ch_idx} lag too large: {lag}"

    print("✓ All channels synchronized")
```

### 5.5 Deliverables

- Comprehensive test suite
- Hardware compatibility report
- Performance benchmarks
- Synchronization validation results

---

## Migration Strategy

### Dual-Mode Operation

The system will support both modes seamlessly:

**Mode Detection:**
```python
def is_multichannel_mode(recorder: RoomResponseRecorder) -> bool:
    """Detect if recorder is in multi-channel mode"""
    return (hasattr(recorder, 'multichannel_config') and
            recorder.multichannel_config.get('enabled', False))
```

**Automatic Fallback:**
- If `multichannel.enabled = false` or field missing → single-channel mode
- If SDL audio core doesn't support multi-channel → warning + single-channel fallback
- If hardware doesn't support requested channel count → error with clear message

### Configuration Migration

**For existing users:**

1. No action required - system defaults to single-channel mode
2. Optional: Add `multichannel` section to `recorderConfig.json` to enable multi-channel
3. Optional: Run migration utility on existing datasets to rename files with `_ch0` suffix

**Configuration update example:**
```json
{
  "recorder_config": {
    "sample_rate": 48000,
    "pulse_duration": 0.008,
    ...
  },
  "multichannel": {
    "enabled": true,
    "num_channels": 4,
    "channel_names": ["Front", "Rear", "Left", "Right"],
    "reference_channel": 0
  }
}
```

### Dataset Migration

**Legacy datasets remain fully compatible:**
- ScenarioManager automatically detects single-channel files
- No migration required for continued operation

**Optional migration:**
```bash
# Dry run (preview changes)
python migrate_to_multichannel.py /path/to/dataset --dry-run

# Execute migration (renames files to add _ch0 suffix)
python migrate_to_multichannel.py /path/to/dataset --execute
```

---

## Risk Assessment

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|------------|
| **Hardware doesn't support multi-channel input** | Medium | High | Automatic fallback to single-channel mode with clear error message |
| **SDL de-interleaving errors** | Low | Critical | Extensive unit tests with synthetic interleaved data; hardware validation |
| **Channel synchronization drift** | Low | High | Cross-correlation validation tests; per-channel buffer locks |
| **Performance degradation with many channels** | Medium | Medium | Benchmark testing; optimize buffer sizes; pre-allocate buffers |
| **GUI responsiveness issues** | Low | Medium | Background threading for recording; progress indicators |
| **File I/O bottleneck** | Low | Low | Parallel file writes; optimize buffer sizes |
| **Configuration complexity** | Medium | Low | Sensible defaults; validation; clear error messages |

---

## Timeline & Resources

### Phase Breakdown

| Phase | Duration | Dependencies | Deliverables |
|-------|----------|--------------|--------------|
| **Phase 1: SDL Audio Core** | 2 weeks | None | Multi-channel SDL core, Python bindings |
| **Phase 2: Recording Pipeline** | 1 week | Phase 1 | Updated RoomResponseRecorder |
| **Phase 3: Filesystem** | 1 week | Phase 2 | ScenarioManager updates, migration utility |
| **Phase 4: GUI Updates** | 2 weeks | Phase 2, 3 | Updated panels with multi-channel UI |
| **Phase 5: Testing** | 1 week | All phases | Test suite, validation report |

**Total Timeline:** 7 weeks (reduced from 9 - no ML/feature extraction phases)

### Success Criteria

- [ ] SDL audio core records multi-channel input with correct de-interleaving
- [ ] All channels maintain sample-level synchronization
- [ ] Reference channel onset detection aligns all channels with same shift
- [ ] Per-channel files saved with correct naming convention
- [ ] ScenarioManager correctly handles both single and multi-channel files
- [ ] GUI displays multi-channel configuration and status
- [ ] Backward compatibility: existing single-channel workflows unchanged
- [ ] Cross-correlation validation confirms synchronization < 10 samples lag
- [ ] Performance: no dropouts or buffer issues with 8 channels at 48kHz

---

## Appendix: Complete Configuration Example

**recorderConfig.json (Multi-Channel Piano Response):**
```json
{
  "recorder_config": {
    "sample_rate": 48000,
    "pulse_duration": 0.008,
    "pulse_fade": 0.0001,
    "cycle_duration": 0.1,
    "num_pulses": 8,
    "volume": 0.4,
    "pulse_frequency": 1000,
    "impulse_form": "sine"
  },
  "multichannel": {
    "enabled": true,
    "num_channels": 4,
    "channel_names": [
      "Front Microphone",
      "Rear Microphone",
      "Left Microphone",
      "Right Microphone"
    ],
    "reference_channel": 0,
    "channel_calibration": {
      "0": {
        "gain": 1.0,
        "delay_samples": 0,
        "notes": "Reference microphone"
      },
      "1": {
        "gain": 1.02,
        "delay_samples": 12,
        "notes": "Slight gain boost"
      },
      "2": {
        "gain": 0.98,
        "delay_samples": -5,
        "notes": "Slight gain reduction"
      },
      "3": {
        "gain": 1.0,
        "delay_samples": 8,
        "notes": "Small delay correction"
      }
    }
  }
}
```

---

## Key Differences from Original Plan

This revised plan differs from the original multi-channel plan in the following critical ways:

1. **Simplified Scope:**
   - Removed MFCC feature extraction (Phase 5 in original)
   - Removed machine learning classifier updates (Phase 6 in original)
   - Removed data science operations
   - Focus on raw audio recording and impulse response only

2. **Clarified Synchronization:**
   - **Original plan was ambiguous** about cross-channel alignment
   - **New plan explicitly states:** All channels are shifted by the SAME amount calculated from reference channel onset
   - This preserves inter-channel timing relationships (phase, TOA differences)

3. **Timeline Reduction:**
   - Original: 9 weeks
   - Piano response only: 7 weeks (removed 2 weeks for ML/feature extraction)

4. **Target Pipeline:**
   - Original: Full gui_launcher.py with ML pipeline
   - New: piano_response.py (simplified audio-only)

---

## Current Implementation Summary (2025-10-25)

### What's Been Completed

**Phase 1: SDL Audio Core ✅ COMPLETE**
- Multi-channel recording at C++ level fully functional
- De-interleaving, per-channel buffers, Python bindings all working
- Tested with 2, 4, 8 channels successfully
- All tests passing (7/7)

**Phase 1.5: GUI Multi-Channel Testing ✅ COMPLETE**
- Basic multi-channel GUI integration for testing
- Device channel detection and display
- Multi-channel monitor (live per-channel meters)
- Multi-channel test recording with statistics
- New "Multi-Channel Test" tab in Audio Settings

### What's Still Needed

**Phase 2: Recording Pipeline** (1 week)
- Full integration of multi-channel into `RoomResponseRecorder`
- Configuration loading from JSON
- Synchronized multi-channel signal processing with reference channel alignment
- Per-channel file saving with proper naming

**Phase 3: Filesystem Structure** (1 week)
- Multi-channel filename convention implementation
- ScenarioManager multi-channel file grouping
- Migration utilities for legacy datasets

**Phase 4: GUI Completion** (2 weeks)
- Multi-channel configuration UI in Audio Settings
- Collection panel multi-channel status display
- Audio Analysis panel multi-channel visualization
- Integration with piano_response.py

**Phase 5: Testing & Validation** (1 week)
- End-to-end integration tests
- Hardware compatibility testing
- Performance benchmarking
- Synchronization validation

### Next Steps

1. **Immediate:** Complete Phase 2 (Recording Pipeline)
   - Implement configuration loading
   - Implement synchronized multi-channel processing
   - Test with actual multi-channel hardware

2. **Short-term:** Phase 3 (Filesystem Structure)
   - Implement filename conventions
   - Update ScenarioManager
   - Create migration utility

3. **Medium-term:** Complete Phase 4 & 5
   - Finish GUI integration
   - Comprehensive testing and validation

### Estimated Time to Completion

- **Core functionality (Phases 2-3):** 2 weeks
- **Full system (Phases 2-5):** 5 weeks
- **Total remaining:** ~1 month of development time

---

## Conclusion

This plan provides a comprehensive roadmap for upgrading the piano response system to multi-channel synchronized impulse response recording. The architecture maintains perfect inter-channel synchronization by applying the same alignment shift to all channels, preserving the spatial and temporal relationships between measurements.

**Current Status:** Phase 1 (SDL Core) is complete and tested. Phase 1.5 (GUI testing features) is complete. The foundation is solid and ready for full pipeline integration.

The phased approach ensures backward compatibility while enabling powerful new multi-channel measurement capabilities for acoustic research and piano impulse response analysis.
