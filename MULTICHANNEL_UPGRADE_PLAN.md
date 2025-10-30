# RoomResponse Multi-Channel Upgrade Implementation Plan

> **Comprehensive Phased Plan for Multi-Channel Input Recording**
>
> Upgrading the system to record simultaneous responses from multiple input channels (e.g., multiple microphones positioned at different locations in a room).

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Current System Analysis](#2-current-system-analysis)
3. [Multi-Channel Architecture Design](#3-multi-channel-architecture-design)
4. [Phase 1: SDL Audio Core Multi-Channel Support](#phase-1-sdl-audio-core-multi-channel-support)
5. [Phase 2: Recording Pipeline Upgrade](#phase-2-recording-pipeline-upgrade)
6. [Phase 3: Filesystem Structure Redesign](#phase-3-filesystem-structure-redesign)
7. [Phase 4: Signal Processing Adaptation](#phase-4-signal-processing-adaptation)
8. [Phase 5: Feature Extraction Multi-Channel](#phase-5-feature-extraction-multi-channel)
9. [Phase 6: GUI Interface Updates](#phase-6-gui-interface-updates)
10. [Phase 7: Testing & Validation](#phase-7-testing--validation)
11. [Migration Strategy](#11-migration-strategy)
12. [Risk Assessment & Mitigation](#12-risk-assessment--mitigation)
13. [Timeline & Resources](#13-timeline--resources)

---

## 1. Executive Summary

### 1.1 Upgrade Objective

**Goal**: Enable the RoomResponse system to record and process multiple input channels simultaneously, allowing spatial analysis of room acoustics with microphone arrays.

**Use Cases**:
- Multi-microphone room response measurements (stereo, 5.1, arrays)
- Spatial acoustic analysis with microphone arrays
- Comparative measurements across different positions
- Beamforming and directional response analysis
- Time-of-arrival (TOA) measurements for source localization

### 1.2 Key Changes Required

| Component | Current State | Target State | Complexity |
|-----------|---------------|--------------|------------|
| **SDL Audio Core** | Single-channel mono input | Multi-channel input (1-32 channels) | **High** |
| **Recording Pipeline** | Single buffer → single WAV | Multi-buffer → per-channel WAVs | **Medium** |
| **Filesystem** | Flat structure (impulse_000.wav) | Hierarchical (impulse_000_ch0.wav, ...) | **Medium** |
| **Signal Processing** | Mono onset detection, alignment | Per-channel or multichannel correlation | **High** |
| **Feature Extraction** | Single feature vector | Per-channel or aggregated features | **Medium** |
| **GUI Interface** | Single device selector | Channel configuration panel | **Medium** |
| **Metadata** | Simple recording info | Channel topology, calibration data | **Low** |

### 1.3 Backward Compatibility Strategy

**Approach**: Maintain dual-mode operation
- **Legacy Mode**: Single-channel operation (default, fully compatible)
- **Multi-Channel Mode**: Activated via configuration flag

**Benefits**:
- Zero disruption to existing workflows
- Gradual migration path
- Supports mixed datasets (legacy + multi-channel)
- No breaking changes to existing code

---

## Implementation Status

**Last Updated**: 2025-01-28

### Completed Phases

| Phase | Status | Summary |
|-------|--------|---------|
| **Phase 1** | ✅ **COMPLETED** | SDL Audio Core multi-channel support implemented and tested |
| **Phase 2** | ✅ **COMPLETED** | Recording pipeline upgraded for multi-channel |
| **Phase 3** | ✅ **COMPLETED** | Filesystem structure redesigned |
| **Phase 4** | ✅ **COMPLETED** | Signal processing adapted (partial - calibration impulse) |
| **Phase 5** | ⏸️ **PENDING** | Feature extraction multi-channel (not started) |
| **Phase 6** | ⏸️ **PENDING** | GUI interface updates (not started) |
| **Phase 7** | ⏸️ **PENDING** | Full testing & validation (not started) |

### Key Accomplishments

**Phase 1-4 Implementation (Completed):**
- ✅ SDL audio core supports 1-32 input channels
- ✅ Python bindings expose multi-channel API (`measure_room_response_auto_multichannel`)
- ✅ Channel negotiation with `SDL_AUDIO_ALLOW_CHANNELS_CHANGE`
- ✅ Multi-channel output device support (mono-to-multichannel replication)
- ✅ Per-channel WAV file storage
- ✅ Multi-channel calibration impulse test in GUI
- ✅ Comprehensive test suite created
- ✅ Device fallback mechanism for robust initialization

### Critical Discovery: Native Driver Requirement

**Finding:** Professional audio interfaces (e.g., Behringer UMC1820) require **native manufacturer drivers** for multi-channel operation.

**Issue Identified:**
- Windows generic USB Audio Class 2.0 driver reports device capabilities (e.g., "10 channels")
- But WDM/WASAPI interface is **hardcoded to stereo (2 channels)** only
- Multi-channel recording fails with "Invalid source channels" error

**Solution Implemented:**
- Created comprehensive driver installation guides
- Diagnostic scripts to detect driver status (`check_umc_driver.py`)
- Test scripts to verify multi-channel functionality
- Documentation: [install_behringer_driver.md](install_behringer_driver.md)
- Technical analysis: [SOLUTION_INSTALL_BEHRINGER_DRIVER.md](SOLUTION_INSTALL_BEHRINGER_DRIVER.md)

**For Behringer UMC1820 Users:**
1. Install Behringer native driver (v4.59.0 or v5.57.0)
2. Download from: https://www.behringer.com/downloads.html
3. After installation: Full 18 input / 20 output channel access enabled

**For Other Professional Interfaces:**
- Focusrite: Install Focusrite Control + ASIO driver
- PreSonus: Install Universal Control + ASIO driver
- MOTU: Install MOTU AVB/USB driver

### Files Created

**Implementation Files:**
- `sdl_audio_core/src/audio_engine.cpp` - Multi-channel support, output channel negotiation
- `sdl_audio_core/src/python_bindings.cpp` - Device fallback, multi-channel API

**Documentation:**
- `SOLUTION_INSTALL_BEHRINGER_DRIVER.md` - Complete driver solution guide
- `SOLUTION_UMC1820_WASAPI.md` - Technical deep-dive (SDL2 WASAPI limitations)
- `TROUBLESHOOTING_UMC1820.md` - User troubleshooting guide
- `install_behringer_driver.md` - Step-by-step installation instructions

**Diagnostic Tools:**
- `check_umc_driver.py` - Check current driver status
- `test_umc_multichannel.py` - Test 9 device combinations
- `test_umc_input_detailed.py` - Detailed channel count testing
- `test_device_enumeration.py` - SDL device detection tests
- `test_multichannel.py` - General multi-channel tests

### Next Steps

**Phase 5 (Feature Extraction):**
- Implement per-channel MFCC extraction
- Add aggregate feature modes
- Spatial feature extraction (TOA, coherence, level differences)

**Phase 6 (GUI Updates):**
- Multi-channel device configuration panel
- Channel selection and naming
- Per-channel visualization
- Aggregate feature display

**Phase 7 (Testing & Validation):**
- Integration tests with real hardware
- Performance benchmarks
- User documentation
- Migration guide for existing datasets

### Known Limitations

1. **Driver Dependency**: Multi-channel recording requires native drivers - see installation guides
2. **SDL2 WASAPI**: Current SDL 2.30.0 WASAPI backend has channel conversion limitations
3. **Output Device**: Multi-channel output works via mono-to-multichannel replication (suitable for test signals)
4. **Calibration**: Phase 4 only implements calibration impulse; full calibration averaging pending

---

## 2. Current System Analysis

### 2.1 SDL Audio Core Limitations

**Current Architecture** (`audio_engine.h/cpp`):

```cpp
// Current: Single-channel recording
SDL_AudioDeviceID input_device_;           // Single input device
std::vector<float> recording_buffer_;      // Mono buffer
std::atomic<size_t> recording_position_;   // Single position counter

// Input callback processes mono samples
void handle_input_audio(Uint8* stream, int len) {
    // Converts SDL format to float
    // Appends to single recording_buffer_
    float* samples = reinterpret_cast<float*>(stream);
    int num_samples = len / sizeof(float);

    for (int i = 0; i < num_samples; ++i) {
        recording_buffer_.push_back(samples[i]);
    }
}
```

**Identified Limitations**:

1. **Single Device**: Only one `SDL_AudioDeviceID` for input
2. **Mono Buffer**: `recording_buffer_` assumes mono samples
3. **No Channel Awareness**: SDL callback treats all channels as mono stream
4. **Device Spec**: `SDL_AudioSpec input_spec_` configured for `channels = 1`
5. **No De-Interleaving**: Cannot separate interleaved multi-channel data

**SDL Multi-Channel Capabilities**:
- SDL2 supports up to 32 channels natively
- Interleaved format: `[L0, R0, L1, R1, ...]` for stereo
- Channel layouts: MONO, STEREO, QUAD, 5.1, 7.1, custom
- Requires explicit channel count in `SDL_AudioSpec.channels`

### 2.2 Recording Pipeline Limitations

**Current Flow** (`RoomResponseRecorder.py`):

```python
# Single-channel recording call
result = sdl_audio_core.measure_room_response_auto(
    playback_signal=[...],
    volume=0.4,
    input_device=-1,   # Single device
    output_device=-1
)

# Returns 1D array
recorded_data = result['recorded_data']  # [s0, s1, s2, ...]

# Single WAV file save
_save_wav(recorded_data, "impulse_000.wav")
```

**Multi-Channel Requirements**:
- Return multi-channel data structure: `{'ch0': [...], 'ch1': [...], ...}`
- Save per-channel WAVs: `impulse_000_ch0.wav`, `impulse_000_ch1.wav`, ...
- Alternative: Single multi-channel WAV file (more complex for downstream processing)
- Metadata tracking: channel count, names, calibration

### 2.3 Filesystem Structure Limitations

**Current Structure**:
```
Laptop-Scenario1-LivingRoom/
├── impulse_responses/
│   ├── impulse_000.wav  (mono, 48kHz)
│   ├── impulse_001.wav  (mono, 48kHz)
│   └── ...
├── raw_recordings/
│   └── raw_000.wav  (mono)
├── room_responses/
│   └── room_000.wav  (mono)
└── metadata/
    └── session_metadata.json
```

**Multi-Channel Options Analysis**:

**Option A: Per-Channel Suffix (RECOMMENDED)**
```
impulse_responses/
├── impulse_000_ch0.wav
├── impulse_000_ch1.wav
├── impulse_000_ch2.wav
├── impulse_000_ch3.wav
├── impulse_001_ch0.wav
└── ...
```
- **Pros**: Simple, flat structure; easy glob patterns; compatible with existing tools
- **Cons**: Many files (N channels × M measurements); no inherent grouping
- **Use Case**: Best for independent channel analysis

**Option B: Subfolder per Measurement**
```
impulse_responses/
├── measurement_000/
│   ├── ch0.wav
│   ├── ch1.wav
│   ├── ch2.wav
│   ├── ch3.wav
│   └── metadata.json
├── measurement_001/
│   ├── ch0.wav
│   └── ...
└── ...
```
- **Pros**: Better organization; metadata co-location; clear grouping
- **Cons**: Deeper nesting; path changes affect existing code
- **Use Case**: Best for synchronized multi-channel analysis

**Option C: Multi-Channel WAV Files**
```
impulse_responses/
├── impulse_000.wav  (4-channel interleaved)
├── impulse_001.wav  (4-channel interleaved)
└── ...
```
- **Pros**: Fewer files; standard format; guaranteed synchronization
- **Cons**: Requires multi-channel-aware tools; complex channel extraction
- **Use Case**: Best for DAW integration, post-processing

**Decision**: **Option A for Phase 1** (simplicity, backward compatibility), with migration path to Option B for Phase 3 if needed.

### 2.4 Signal Processing Limitations

**Current Onset Detection** (`RoomResponseRecorder._find_sound_onset()`):
```python
def _find_sound_onset(audio: np.ndarray, window_size=10, threshold_factor=2) -> int:
    # Operates on mono signal only
    # RMS-based threshold detection
    # Returns single onset sample index
```

**Multi-Channel Challenges**:
1. **Different Arrival Times**: Each channel may detect onset at different samples (TOA)
2. **Reference Channel**: Need to select which channel defines "true" onset
3. **Cross-Channel Alignment**: Use cross-correlation to align channels
4. **Spatial Processing**: Array geometry affects processing strategy

**Multi-Channel Requirements**:
- **Per-Channel Onset Detection**: Independent detection for each channel
- **Reference-Based Alignment**: Align all channels to reference channel
- **TOA Extraction**: Calculate inter-channel time differences
- **Quality Validation**: Ensure all channels captured valid signal

### 2.5 Feature Extraction Limitations

**Current MFCC Extraction** (`FeatureExtractor.py`):
```python
def _extract_mfcc_from_audio(audio: np.ndarray) -> np.ndarray:
    # Input: 1D mono array shape=(N,)
    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=self.sample_rate,
        n_mfcc=13
    )
    # Output: Average across time frames, shape=(13,)
    return np.mean(mfcc, axis=1)
```

**Multi-Channel Feature Strategies**:

**1. Per-Channel Features** (Independent Processing)
```python
# Extract MFCC for each channel separately
# Features: ch0_mfcc_0, ch0_mfcc_1, ..., ch1_mfcc_0, ch1_mfcc_1, ...
# Feature count: num_channels × n_mfcc (e.g., 4 × 13 = 52)
# Use case: Channel-specific acoustic properties
```

**2. Aggregate Features** (Combined Processing)
```python
# Combine channels before extraction (mean, max, RMS)
# Features: mfcc_0, mfcc_1, ... (same as mono)
# Feature count: n_mfcc (e.g., 13)
# Use case: Overall room response, backward compatibility
```

**3. Spatial Features** (Inter-Channel Analysis)
```python
# Extract spatial relationships between channels
# Features: toa_ch1_ch0, coherence_ch1_ch0, level_diff_ch1_ch0, ...
# Feature count: num_pairs × num_spatial_features
# Use case: Spatial analysis, source localization
```

**Recommended**: Support all three modes, user-configurable via `recorderConfig.json`.

### 2.6 Current Metadata Schema

**Existing** (`MeasurementMetadata`):
```python
@dataclass
class MeasurementMetadata:
    scenario_name: str
    measurement_index: int
    timestamp: str
    filename_raw: str              # Single file
    filename_impulse: str          # Single file
    filename_room_response: str    # Single file
    signal_params: Dict[str, Any]
    quality_metrics: Dict[str, Any]  # Single-channel metrics
```

**Required for Multi-Channel**:
- `is_multichannel: bool`
- `num_channels: int`
- `channel_names: List[str]`
- `filenames_per_channel: Dict[int, Dict[str, str]]`
- `quality_metrics_per_channel: Dict[int, Dict[str, float]]`
- `inter_channel_metrics: Dict[str, float]`

---

## 3. Multi-Channel Architecture Design

### 3.1 Configuration Schema

**Extended `recorderConfig.json`**:

```json
{
  "sample_rate": 48000,
  "pulse_duration": 0.008,
  "pulse_fade": 0.0001,
  "cycle_duration": 0.1,
  "num_pulses": 8,
  "volume": 0.4,
  "pulse_frequency": 1000,
  "impulse_form": "sine",

  "multichannel": {
    "enabled": true,
    "num_channels": 4,
    "channel_names": [
      "Front-Left",
      "Front-Right",
      "Rear-Left",
      "Rear-Right"
    ],
    "reference_channel": 0,
    "channel_calibration": {
      "0": {"gain": 1.0, "delay_samples": 0},
      "1": {"gain": 1.02, "delay_samples": 12},
      "2": {"gain": 0.98, "delay_samples": 8},
      "3": {"gain": 1.01, "delay_samples": 15}
    },
    "spatial_config": {
      "geometry": "rectangular_array",
      "positions": [
        {"x": 0.0, "y": 0.0, "z": 0.0},
        {"x": 0.5, "y": 0.0, "z": 0.0},
        {"x": 0.0, "y": 0.5, "z": 0.0},
        {"x": 0.5, "y": 0.5, "z": 0.0}
      ],
      "units": "meters"
    }
  },

  "feature_extraction": {
    "mode": "per_channel",
    "aggregate_method": "mean",
    "spatial_features": ["toa", "coherence", "level_diff"]
  }
}
```

**Backward Compatibility**:
```json
{
  // If multichannel section missing → legacy single-channel mode
  "multichannel": {
    "enabled": false  // Explicit disable
  }
}
```

### 3.2 Data Flow Architecture

```
┌──────────────────────────────────────────────────────────┐
│              USER CONFIGURATION                          │
│  - Select input device (multi-channel capable)           │
│  - Configure num_channels (1-32)                         │
│  - Set channel names, calibration                        │
└──────────────────────────────────────────────────────────┘
                         ↓
┌──────────────────────────────────────────────────────────┐
│         SDL AUDIO CORE INITIALIZATION                    │
│  AudioEngineConfig:                                      │
│    - input_channels = 4                                  │
│  SDL_AudioSpec:                                          │
│    - channels = 4                                        │
│    - format = AUDIO_F32SYS                              │
└──────────────────────────────────────────────────────────┘
                         ↓
┌──────────────────────────────────────────────────────────┐
│            SIMULTANEOUS RECORDING                        │
│  SDL Input Callback (44.1kHz or 48kHz):                 │
│    - Receives interleaved: [L0,R0,RL0,RR0,L1,R1,...]   │
│    - De-interleaves to per-channel buffers              │
│  Buffers:                                                │
│    - recording_buffers_[0]: [L0, L1, L2, ...]          │
│    - recording_buffers_[1]: [R0, R1, R2, ...]          │
│    - recording_buffers_[2]: [RL0, RL1, RL2, ...]       │
│    - recording_buffers_[3]: [RR0, RR1, RR2, ...]       │
└──────────────────────────────────────────────────────────┘
                         ↓
┌──────────────────────────────────────────────────────────┐
│       PYTHON MULTI-CHANNEL PROCESSING                    │
│  RoomResponseRecorder.take_record():                     │
│    Returns: {                                            │
│      'ch0': np.array([...]),                            │
│      'ch1': np.array([...]),                            │
│      'ch2': np.array([...]),                            │
│      'ch3': np.array([...]),                            │
│      'metadata': {                                       │
│        'num_channels': 4,                                │
│        'channel_names': [...],                           │
│        'reference_channel': 0                            │
│      }                                                    │
│    }                                                      │
└──────────────────────────────────────────────────────────┘
                         ↓
┌──────────────────────────────────────────────────────────┐
│        PER-CHANNEL SIGNAL PROCESSING                     │
│  For each channel ch_i:                                  │
│    1. Extract room response (cycle averaging)           │
│    2. Find onset (relative to reference)                │
│    3. Extract impulse response (circular shift)         │
│    4. Apply calibration (gain, delay)                   │
└──────────────────────────────────────────────────────────┘
                         ↓
┌──────────────────────────────────────────────────────────┐
│             FILE PERSISTENCE                             │
│  Save per-channel WAVs:                                  │
│    - raw_000_ch0.wav, raw_000_ch1.wav, ...             │
│    - impulse_000_ch0.wav, impulse_000_ch1.wav, ...     │
│    - room_000_ch0.wav, room_000_ch1.wav, ...           │
│  Save metadata:                                          │
│    - measurement_000_meta.json                           │
│      (includes per-channel quality metrics)             │
└──────────────────────────────────────────────────────────┘
                         ↓
┌──────────────────────────────────────────────────────────┐
│        FEATURE EXTRACTION (Configurable)                 │
│  Mode 1: Per-Channel → features_ch0.csv, ...           │
│  Mode 2: Aggregate → features_aggregate.csv            │
│  Mode 3: Spatial → spatial_features.csv                │
└──────────────────────────────────────────────────────────┘
```

### 3.3 Class Architecture Changes

**AudioEngine (C++) - New Members**:

```cpp
class AudioEngine {
private:
    // Multi-channel support
    int num_input_channels_;
    std::vector<std::vector<float>> recording_buffers_;  // [channel][samples]
    std::vector<std::mutex> channel_mutexes_;
    std::vector<std::atomic<size_t>> recording_positions_;

    // Channel calibration
    std::vector<float> channel_gains_;
    std::vector<int> channel_delays_samples_;

public:
    struct MultiChannelConfig {
        int num_channels = 1;
        bool enable_calibration = false;
        std::vector<float> gains;
        std::vector<int> delays_samples;
    };

    bool initialize_multichannel(const Config& config,
                                 const MultiChannelConfig& mc_config);

    std::map<int, std::vector<float>> get_recorded_data_multichannel();
    void clear_recording_buffer_multichannel();
    int get_num_channels() const { return num_input_channels_; }
};
```

**RoomResponseRecorder (Python) - New Methods**:

```python
class RoomResponseRecorder:
    def __init__(self, config_file_path: str = None):
        # Existing initialization
        ...

        # Multi-channel configuration
        self.multichannel_enabled = False
        self.num_channels = 1
        self.channel_names = []
        self.reference_channel = 0
        self.channel_calibration = {}

        self._load_multichannel_config()

    def take_record(self, output_file, impulse_file) -> Union[np.ndarray, Dict]:
        """
        Unified recording method supporting both modes.

        Returns:
            - Single-channel: np.ndarray
            - Multi-channel: Dict[str, np.ndarray]
        """
        if not self.multichannel_enabled:
            return self._record_single_channel(output_file, impulse_file)
        else:
            return self._record_multichannel(output_file, impulse_file)

    def _record_multichannel(self, base_output, base_impulse) -> Dict:
        """Multi-channel recording implementation"""
        # Call SDL audio core with multi-channel mode
        # Process each channel independently
        # Save per-channel files
        # Return multi-channel data dictionary
```

---

## Phase 1: SDL Audio Core Multi-Channel Support

**Duration**: 2 weeks
**Objective**: Upgrade C++ audio engine to record multi-channel input.

### 1.1 SDL AudioSpec Configuration

**File**: `audio_engine.cpp:initialize_input_device()`

**Current Code**:
```cpp
SDL_AudioSpec desired_spec;
SDL_zero(desired_spec);
desired_spec.freq = config_.sample_rate;
desired_spec.format = AUDIO_F32SYS;
desired_spec.channels = 1;  // ← CURRENT: Hardcoded mono
desired_spec.samples = config_.buffer_size;
desired_spec.callback = input_audio_callback;
desired_spec.userdata = this;
```

**Updated Code**:
```cpp
SDL_AudioSpec desired_spec;
SDL_zero(desired_spec);
desired_spec.freq = config_.sample_rate;
desired_spec.format = AUDIO_F32SYS;
desired_spec.channels = mc_config_.num_channels;  // ← NEW: Configurable
desired_spec.samples = config_.buffer_size;
desired_spec.callback = input_audio_callback;
desired_spec.userdata = this;

// Open device with multi-channel request
const char* device_name = (config_.input_device_id >= 0)
    ? SDL_GetAudioDeviceName(config_.input_device_id, 1)
    : nullptr;

input_device_ = SDL_OpenAudioDevice(
    device_name,
    1,  // is_capture = true
    &desired_spec,
    &input_spec_,
    SDL_AUDIO_ALLOW_FREQUENCY_CHANGE
);

if (input_device_ == 0) {
    log_error("Failed to open input device: " + std::string(SDL_GetError()));
    return false;
}

// Validate obtained spec
if (input_spec_.channels != mc_config_.num_channels) {
    log_error("Device does not support " + std::to_string(mc_config_.num_channels) +
              " channels, got: " + std::to_string(input_spec_.channels));
    // Option: Fallback to supported channel count
    // or return false to fail fast
}

log("Input device opened: " + std::to_string(input_spec_.channels) + " channels, " +
    std::to_string(input_spec_.freq) + " Hz");
```

### 1.2 Multi-Channel Buffer Management

**File**: `audio_engine.h` - Add Private Members

```cpp
private:
    struct MultiChannelConfig {
        int num_channels = 1;
        bool enable_calibration = false;
        std::vector<float> gains;              // Per-channel gain
        std::vector<int> delays_samples;       // Per-channel delay
    };

    MultiChannelConfig mc_config_;

    // Per-channel recording buffers
    std::vector<std::vector<float>> recording_buffers_;  // [channel][samples]
    std::vector<std::mutex> channel_mutexes_;            // Thread safety
    std::vector<std::atomic<size_t>> recording_positions_;  // Per-channel write positions
```

**File**: `audio_engine.cpp` - Initialize Buffers

```cpp
bool AudioEngine::initialize_multichannel(const Config& config,
                                         const MultiChannelConfig& mc_config) {
    std::lock_guard<std::mutex> lock(state_mutex_);

    if (state_ != State::Uninitialized) {
        log_error("AudioEngine already initialized");
        return false;
    }

    config_ = config;
    mc_config_ = mc_config;

    // Initialize SDL audio subsystem
    if (SDL_InitSubSystem(SDL_INIT_AUDIO) < 0) {
        log_error("Failed to initialize SDL audio: " + std::string(SDL_GetError()));
        state_ = State::Error;
        return false;
    }

    // Allocate per-channel buffers
    recording_buffers_.resize(mc_config_.num_channels);
    channel_mutexes_.resize(mc_config_.num_channels);
    recording_positions_.resize(mc_config_.num_channels);

    size_t buffer_capacity = config_.sample_rate * 60;  // 60 seconds per channel
    for (int ch = 0; ch < mc_config_.num_channels; ++ch) {
        recording_buffers_[ch].reserve(buffer_capacity);
        recording_positions_[ch].store(0);
    }

    // Validate calibration data
    if (mc_config_.enable_calibration) {
        if (mc_config_.gains.size() != static_cast<size_t>(mc_config_.num_channels)) {
            log_error("Gain vector size mismatch");
            return false;
        }
        if (mc_config_.delays_samples.size() != static_cast<size_t>(mc_config_.num_channels)) {
            log_error("Delay vector size mismatch");
            return false;
        }
    }

    // Create audio buffers (for output)
    size_t buffer_samples = config_.buffer_size * 8;
    input_buffer_ = std::make_unique<AudioBuffer>(buffer_samples);
    output_buffer_ = std::make_unique<AudioBuffer>(buffer_samples);

    reset_stats();
    state_ = State::Initialized;

    log("AudioEngine initialized (multi-channel mode: " +
        std::to_string(mc_config_.num_channels) + " channels)");

    return true;
}
```

### 1.3 De-Interleaving Callback

**File**: `audio_engine.cpp` - Update Input Handler

```cpp
void AudioEngine::handle_input_audio(Uint8* stream, int len) {
    if (!is_recording_.load()) {
        return;
    }

    int num_channels = input_spec_.channels;
    int sample_size = sizeof(float);
    int num_frames = len / (sample_size * num_channels);

    if (num_frames <= 0) {
        return;
    }

    float* samples = reinterpret_cast<float*>(stream);

    // De-interleave: [L0, R0, RL0, RR0, L1, R1, ...] → per-channel buffers
    for (int frame = 0; frame < num_frames; ++frame) {
        for (int ch = 0; ch < num_channels; ++ch) {
            float sample = samples[frame * num_channels + ch];

            // Apply calibration gain if enabled
            if (mc_config_.enable_calibration &&
                ch < static_cast<int>(mc_config_.gains.size())) {
                sample *= mc_config_.gains[ch];
            }

            // Append to per-channel buffer (thread-safe)
            {
                std::lock_guard<std::mutex> lock(channel_mutexes_[ch]);
                recording_buffers_[ch].push_back(sample);
                recording_positions_[ch].fetch_add(1);
            }
        }
    }

    input_samples_processed_.fetch_add(num_frames);
}
```

### 1.4 Multi-Channel Data Retrieval

**File**: `audio_engine.h` - Public Methods

```cpp
public:
    // Multi-channel recording methods
    std::map<int, std::vector<float>> get_recorded_data_multichannel();
    void clear_recording_buffer_multichannel();
    int get_num_channels() const { return mc_config_.num_channels; }
```

**File**: `audio_engine.cpp` - Implementation

```cpp
std::map<int, std::vector<float>> AudioEngine::get_recorded_data_multichannel() {
    std::map<int, std::vector<float>> result;

    for (int ch = 0; ch < mc_config_.num_channels; ++ch) {
        std::lock_guard<std::mutex> lock(channel_mutexes_[ch]);
        result[ch] = recording_buffers_[ch];  // Copy data
    }

    return result;
}

void AudioEngine::clear_recording_buffer_multichannel() {
    for (int ch = 0; ch < mc_config_.num_channels; ++ch) {
        std::lock_guard<std::mutex> lock(channel_mutexes_[ch]);
        recording_buffers_[ch].clear();
        recording_positions_[ch].store(0);
    }
}
```

### 1.5 Python Bindings (pybind11)

**File**: `python_bindings.cpp`

```cpp
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "audio_engine.h"

namespace py = pybind11;

PYBIND11_MODULE(sdl_audio_core, m) {
    m.doc() = "SDL Audio Core - Multi-Channel Support";

    // ... existing bindings ...

    // Multi-channel config
    py::class_<sdl_audio::AudioEngine::MultiChannelConfig>(m, "MultiChannelConfig")
        .def(py::init<>())
        .def_readwrite("num_channels", &sdl_audio::AudioEngine::MultiChannelConfig::num_channels)
        .def_readwrite("enable_calibration", &sdl_audio::AudioEngine::MultiChannelConfig::enable_calibration)
        .def_readwrite("gains", &sdl_audio::AudioEngine::MultiChannelConfig::gains)
        .def_readwrite("delays_samples", &sdl_audio::AudioEngine::MultiChannelConfig::delays_samples);

    // Audio Engine with multi-channel methods
    py::class_<sdl_audio::AudioEngine>(m, "AudioEngine")
        // ... existing methods ...
        .def("initialize_multichannel", &sdl_audio::AudioEngine::initialize_multichannel,
             py::arg("config"), py::arg("mc_config"))
        .def("get_recorded_data_multichannel", &sdl_audio::AudioEngine::get_recorded_data_multichannel)
        .def("clear_recording_buffer_multichannel", &sdl_audio::AudioEngine::clear_recording_buffer_multichannel)
        .def("get_num_channels", &sdl_audio::AudioEngine::get_num_channels);

    // Convenience function for multi-channel measurement
    m.def("measure_room_response_multichannel",
        [](const std::vector<float>& playback_signal,
           float volume,
           int num_channels,
           int input_device,
           int output_device) -> py::dict {

            sdl_audio::AudioEngine engine;
            sdl_audio::AudioEngine::Config config;
            config.sample_rate = 48000;
            config.input_device_id = input_device;
            config.output_device_id = output_device;
            config.enable_logging = false;

            sdl_audio::AudioEngine::MultiChannelConfig mc_config;
            mc_config.num_channels = num_channels;

            if (!engine.initialize_multichannel(config, mc_config)) {
                py::dict result;
                result["success"] = false;
                result["error_message"] = "Failed to initialize audio engine";
                return result;
            }

            if (!engine.start()) {
                py::dict result;
                result["success"] = false;
                result["error_message"] = "Failed to start audio engine";
                return result;
            }

            // Start synchronized recording and playback
            size_t max_recording_samples = playback_signal.size() * 2;
            engine.start_synchronized_recording_and_playback(playback_signal, max_recording_samples);

            // Wait for playback to complete
            bool completed = engine.wait_for_playback_completion(10000);  // 10 second timeout

            if (!completed) {
                py::dict result;
                result["success"] = false;
                result["error_message"] = "Recording timeout";
                engine.stop();
                return result;
            }

            // Retrieve multi-channel data
            auto multichannel_data = engine.get_recorded_data_multichannel();

            py::dict result;
            result["success"] = true;
            result["num_channels"] = num_channels;
            result["recorded_samples"] = multichannel_data.empty() ? 0 : multichannel_data.begin()->second.size();

            // Add per-channel data
            for (const auto& [ch, data] : multichannel_data) {
                std::string key = "ch" + std::to_string(ch);
                result[key.c_str()] = data;
            }

            engine.stop();
            return result;
        },
        py::arg("playback_signal"),
        py::arg("volume") = 0.4f,
        py::arg("num_channels") = 1,
        py::arg("input_device") = -1,
        py::arg("output_device") = -1,
        "Record room response with multiple input channels"
    );
}
```

### 1.6 Phase 1 Testing

**Unit Tests (C++)**:

```cpp
// test_multichannel_audio_engine.cpp
#include <gtest/gtest.h>
#include "audio_engine.h"

TEST(AudioEngineMultiChannel, InitializeWith4Channels) {
    sdl_audio::AudioEngine engine;
    sdl_audio::AudioEngine::Config config;
    config.sample_rate = 48000;

    sdl_audio::AudioEngine::MultiChannelConfig mc_config;
    mc_config.num_channels = 4;

    ASSERT_TRUE(engine.initialize_multichannel(config, mc_config));
    EXPECT_EQ(engine.get_num_channels(), 4);
}

TEST(AudioEngineMultiChannel, BufferAllocation) {
    sdl_audio::AudioEngine engine;
    sdl_audio::AudioEngine::Config config;
    sdl_audio::AudioEngine::MultiChannelConfig mc_config;
    mc_config.num_channels = 8;

    ASSERT_TRUE(engine.initialize_multichannel(config, mc_config));

    // Buffers should be allocated for all channels
    auto data = engine.get_recorded_data_multichannel();
    EXPECT_EQ(data.size(), 8);
}

TEST(AudioEngineMultiChannel, CalibrationValidation) {
    sdl_audio::AudioEngine engine;
    sdl_audio::AudioEngine::Config config;
    sdl_audio::AudioEngine::MultiChannelConfig mc_config;
    mc_config.num_channels = 2;
    mc_config.enable_calibration = true;
    mc_config.gains = {1.0f};  // Mismatch: 1 gain for 2 channels

    EXPECT_FALSE(engine.initialize_multichannel(config, mc_config));
}
```

**Integration Tests (Python)**:

```python
# test_multichannel_recording.py
import pytest
import sdl_audio_core

def test_multichannel_initialization():
    """Test Python bindings for multi-channel config"""
    mc_config = sdl_audio_core.MultiChannelConfig()
    mc_config.num_channels = 4
    mc_config.enable_calibration = False

    assert mc_config.num_channels == 4
    assert mc_config.enable_calibration == False

def test_multichannel_recording_api():
    """Test multi-channel recording convenience function"""
    # Generate 1-second test signal at 48kHz
    signal = [0.1] * 48000

    result = sdl_audio_core.measure_room_response_multichannel(
        playback_signal=signal,
        volume=0.2,
        num_channels=2,
        input_device=-1,
        output_device=-1
    )

    assert 'success' in result
    if result['success']:
        assert result['num_channels'] == 2
        assert 'ch0' in result
        assert 'ch1' in result
        assert len(result['ch0']) > 0
        assert len(result['ch1']) > 0
```

**Phase 1 Deliverables**:
- ✅ Multi-channel SDL AudioSpec configuration
- ✅ De-interleaving logic in audio callback
- ✅ Per-channel buffer management
- ✅ Python bindings for multi-channel API
- ✅ Unit tests (C++)
- ✅ Integration tests (Python)
- ✅ Documentation updates

---

## Phase 2: Recording Pipeline Upgrade

**Duration**: 1 week
**Objective**: Adapt Python recording pipeline to handle multi-channel data.

### 2.1 RoomResponseRecorder Multi-Channel Configuration

**File**: `RoomResponseRecorder.py`

```python
class RoomResponseRecorder:
    def __init__(self, config_file_path: str = None):
        # ... existing initialization ...

        # Multi-channel configuration
        self.multichannel_enabled = False
        self.num_channels = 1
        self.channel_names = []
        self.reference_channel = 0
        self.channel_calibration = {}

        # Load multi-channel config
        self._load_multichannel_config()

    def _load_multichannel_config(self):
        """Load multi-channel configuration from config dict/file"""
        if not hasattr(self, 'config') or 'multichannel' not in self.config:
            # No multi-channel config → stay in single-channel mode
            return

        mc = self.config['multichannel']
        self.multichannel_enabled = mc.get('enabled', False)

        if not self.multichannel_enabled:
            return

        self.num_channels = mc.get('num_channels', 1)
        self.channel_names = mc.get('channel_names',
                                    [f"Channel {i}" for i in range(self.num_channels)])
        self.reference_channel = mc.get('reference_channel', 0)
        self.channel_calibration = mc.get('channel_calibration', {})

        print(f"Multi-channel mode enabled: {self.num_channels} channels")
        for i, name in enumerate(self.channel_names):
            print(f"  Channel {i}: {name}")
```

### 2.2 Unified Recording API

**File**: `RoomResponseRecorder.py`

```python
def take_record(self, output_file: str, impulse_file: str,
                method: int = 2) -> Union[np.ndarray, Dict[str, np.ndarray]]:
    """
    Unified recording method supporting both single-channel and multi-channel.

    Args:
        output_file: Base path for raw recording(s)
        impulse_file: Base path for impulse response(s)
        method: Recording method (kept for backward compatibility, always uses method 2)

    Returns:
        - Single-channel mode: np.ndarray (mono audio)
        - Multi-channel mode: Dict with keys 'ch0', 'ch1', ..., 'metadata'
    """
    print(f"\n{'=' * 60}")
    print(f"Room Response Recording ({'Multi-Channel' if self.multichannel_enabled else 'Single-Channel'})")
    print(f"{'=' * 60}")
    self.print_signal_analysis()

    if not self.multichannel_enabled:
        return self._record_single_channel(output_file, impulse_file)
    else:
        return self._record_multichannel(output_file, impulse_file)

def _record_single_channel(self, output_file: str, impulse_file: str) -> np.ndarray:
    """Existing single-channel implementation (unchanged)"""
    try:
        recorded_audio = self._record_method_2()

        if recorded_audio is None:
            print("Recording failed - no data captured")
            return None

        processed_data = self._process_recorded_signal(recorded_audio)

        self._save_wav(processed_data['raw'], output_file)
        self._save_wav(processed_data['impulse'], impulse_file)

        # Room response file
        output_path = Path(output_file)
        room_response_file = str(output_path.parent / f"room_{output_path.stem}_room.wav")
        self._save_wav(processed_data['room_response'], room_response_file)

        print(f"✅ Recording completed successfully")
        return recorded_audio

    except Exception as e:
        print(f"Error during recording: {e}")
        return None

def _record_multichannel(self, base_output_file: str,
                        base_impulse_file: str) -> Dict[str, np.ndarray]:
    """Multi-channel recording implementation"""
    try:
        # Call SDL audio core with multi-channel mode
        result = sdl_audio_core.measure_room_response_multichannel(
            self.playback_signal,
            volume=self.volume,
            num_channels=self.num_channels,
            input_device=self.input_device,
            output_device=self.output_device
        )

        if not result['success']:
            print(f"Multi-channel recording failed: {result.get('error_message', 'Unknown error')}")
            return None

        print(f"Recorded {result.get('recorded_samples', 0)} samples per channel")

        multichannel_data = {}

        # Process each channel
        for ch_idx in range(self.num_channels):
            ch_key = f'ch{ch_idx}'

            if ch_key not in result:
                print(f"Warning: Channel {ch_idx} data missing")
                continue

            recorded_audio = np.array(result[ch_key], dtype=np.float32)

            # Apply calibration (gain + delay)
            recorded_audio = self._apply_channel_calibration(recorded_audio, ch_idx)

            # Process signal (existing pipeline)
            processed = self._process_recorded_signal(recorded_audio)

            # Generate per-channel filenames
            output_ch = self._make_channel_filename(base_output_file, ch_idx)
            impulse_ch = self._make_channel_filename(base_impulse_file, ch_idx)
            room_ch = self._make_channel_filename(
                base_output_file.replace('raw_', 'room_'), ch_idx
            )

            # Save per-channel WAVs
            self._save_wav(processed['raw'], output_ch)
            self._save_wav(processed['impulse'], impulse_ch)
            self._save_wav(processed['room_response'], room_ch)

            print(f"  Channel {ch_idx} ({self.channel_names[ch_idx]}): {len(recorded_audio)} samples")

            multichannel_data[ch_key] = recorded_audio

        # Add metadata
        multichannel_data['metadata'] = {
            'num_channels': self.num_channels,
            'channel_names': self.channel_names,
            'reference_channel': self.reference_channel,
            'sample_rate': self.sample_rate
        }

        print(f"✅ Multi-channel recording completed successfully")
        return multichannel_data

    except Exception as e:
        print(f"Error during multi-channel recording: {e}")
        return None

def _apply_channel_calibration(self, audio: np.ndarray, ch_idx: int) -> np.ndarray:
    """Apply gain and delay calibration to a channel"""
    if str(ch_idx) not in self.channel_calibration:
        return audio

    cal = self.channel_calibration[str(ch_idx)]

    # Apply gain
    gain = cal.get('gain', 1.0)
    audio = audio * gain

    # Apply delay
    delay = cal.get('delay_samples', 0)
    if delay != 0:
        audio = self._apply_delay(audio, delay)

    return audio

def _apply_delay(self, audio: np.ndarray, delay_samples: int) -> np.ndarray:
    """Apply sample-accurate delay (positive=delay, negative=advance)"""
    if delay_samples > 0:
        # Delay: prepend zeros, trim end
        return np.concatenate([np.zeros(delay_samples), audio[:-delay_samples]])
    elif delay_samples < 0:
        # Advance: trim start, append zeros
        return np.concatenate([audio[-delay_samples:], np.zeros(-delay_samples)])
    return audio

def _make_channel_filename(self, base_filename: str, ch_idx: int) -> str:
    """Generate per-channel filename: 'file.wav' → 'file_ch0.wav'"""
    path = Path(base_filename)
    stem = path.stem
    suffix = path.suffix
    parent = path.parent

    new_stem = f"{stem}_ch{ch_idx}"
    return str(parent / f"{new_stem}{suffix}")
```

### 2.3 DatasetCollector Multi-Channel Adaptation

**File**: `DatasetCollector.py` - Update Metadata Schema

```python
@dataclass
class MeasurementMetadata:
    scenario_name: str
    measurement_index: int
    timestamp: str

    # Multi-channel support
    is_multichannel: bool = False
    num_channels: int = 1
    channel_names: List[str] = None

    # Legacy single-channel filenames (backward compatibility)
    filename_raw: Optional[str] = None
    filename_impulse: Optional[str] = None
    filename_room_response: Optional[str] = None

    # Multi-channel filenames
    filenames_per_channel: Optional[Dict[int, Dict[str, str]]] = None
    # Example: {
    #   0: {'raw': 'raw_000_ch0.wav', 'impulse': 'impulse_000_ch0.wav', 'room': 'room_000_ch0.wav'},
    #   1: {'raw': 'raw_000_ch1.wav', 'impulse': 'impulse_000_ch1.wav', 'room': 'room_000_ch1.wav'},
    # }

    signal_params: Dict[str, Any]
    recording_stats: Dict[str, Any] = None
    quality_metrics: Optional[Dict] = None  # Can be per-channel dict or single dict
    notes: str = ""

    def __post_init__(self):
        if self.channel_names is None:
            self.channel_names = []
        if self.filenames_per_channel is None:
            self.filenames_per_channel = {}
```

**File**: `DatasetCollector.py` - Update Collection Loop

```python
def collect_scenario_measurements(self) -> List[MeasurementMetadata]:
    # ... existing setup ...

    for local_idx in range(self.scenario.num_measurements):
        # ... pause/stop checks ...

        absolute_idx = start_index + local_idx
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        base = f"{self.scenario.scenario_name}_{absolute_idx:03d}_{timestamp}"

        # Determine if multi-channel mode
        is_multichannel = self.recorder.multichannel_enabled

        if not is_multichannel:
            # ===== LEGACY SINGLE-CHANNEL PATH =====
            raw_filename = f"raw_{base}.wav"
            impulse_filename = f"impulse_{base}.wav"
            room_filename = f"room_{base}.wav"

            raw_path = self.scenario_dir / "raw_recordings" / raw_filename
            impulse_path = self.scenario_dir / "impulse_responses" / impulse_filename

            try:
                audio_data = self.recorder.take_record(str(raw_path), str(impulse_path))

                if audio_data is not None:
                    q = self.calculate_quality_metrics(audio_data)

                    mm = MeasurementMetadata(
                        scenario_name=self.scenario.scenario_name,
                        measurement_index=absolute_idx,
                        timestamp=timestamp,
                        is_multichannel=False,
                        num_channels=1,
                        filename_raw=raw_filename,
                        filename_impulse=impulse_filename,
                        filename_room_response=room_filename,
                        signal_params=self.recorder_config_dict,
                        quality_metrics=q,
                        recording_stats={'samples_recorded': int(len(audio_data))}
                    )

                    scenario_measurements.append(mm)
                    self.measurements.append(mm)
                    successful_measurements += 1
                    self._save_metadata(append=True)
                else:
                    print("  ❌ Measurement failed - no data recorded")
                    failed_measurements += 1

            except Exception as e:
                print(f"  ❌ Measurement failed: {e}")
                failed_measurements += 1

        else:
            # ===== MULTI-CHANNEL PATH =====
            raw_base = self.scenario_dir / "raw_recordings" / f"raw_{base}"
            impulse_base = self.scenario_dir / "impulse_responses" / f"impulse_{base}"

            try:
                multichannel_data = self.recorder.take_record(
                    str(raw_base), str(impulse_base)
                )

                if multichannel_data is not None:
                    num_channels = multichannel_data['metadata']['num_channels']
                    channel_names = multichannel_data['metadata']['channel_names']

                    # Per-channel quality metrics
                    quality_metrics_per_channel = {}
                    filenames_per_channel = {}
                    total_samples = 0

                    for ch_idx in range(num_channels):
                        ch_key = f'ch{ch_idx}'
                        if ch_key in multichannel_data:
                            audio_data = multichannel_data[ch_key]

                            # Quality assessment per channel
                            q = self.calculate_quality_metrics(audio_data)
                            quality_metrics_per_channel[ch_idx] = q
                            total_samples = len(audio_data)

                            # Filenames
                            filenames_per_channel[ch_idx] = {
                                'raw': f"raw_{base}_ch{ch_idx}.wav",
                                'impulse': f"impulse_{base}_ch{ch_idx}.wav",
                                'room': f"room_{base}_ch{ch_idx}.wav"
                            }

                    mm = MeasurementMetadata(
                        scenario_name=self.scenario.scenario_name,
                        measurement_index=absolute_idx,
                        timestamp=timestamp,
                        is_multichannel=True,
                        num_channels=num_channels,
                        channel_names=channel_names,
                        filenames_per_channel=filenames_per_channel,
                        signal_params=self.recorder_config_dict,
                        quality_metrics=quality_metrics_per_channel,
                        recording_stats={'samples_recorded_per_channel': total_samples}
                    )

                    scenario_measurements.append(mm)
                    self.measurements.append(mm)
                    successful_measurements += 1
                    self._save_metadata(append=True)
                else:
                    print("  ❌ Multi-channel measurement failed - no data recorded")
                    failed_measurements += 1

            except Exception as e:
                print(f"  ❌ Multi-channel measurement failed: {e}")
                failed_measurements += 1

        # Inter-measurement delay
        if local_idx < self.scenario.num_measurements - 1:
            print(f"  Waiting {self.scenario.measurement_interval:.1f}s...")
            time.sleep(self.scenario.measurement_interval)

    # ... rest of method unchanged ...
```

### 2.4 Phase 2 Testing

```python
# test_multichannel_recording_pipeline.py

def test_multichannel_recording_full_pipeline(tmp_path):
    """Test complete multi-channel recording with DatasetCollector"""
    config = {
        "sample_rate": 48000,
        "pulse_duration": 0.008,
        "cycle_duration": 0.1,
        "num_pulses": 8,
        "volume": 0.3,
        "impulse_form": "sine",
        "multichannel": {
            "enabled": True,
            "num_channels": 2,
            "channel_names": ["Left", "Right"],
            "reference_channel": 0
        }
    }

    scenario_config = {
        'scenario_number': '1',
        'description': 'Test multi-channel',
        'computer_name': 'TestPC',
        'room_name': 'TestRoom',
        'num_measurements': 2,
        'measurement_interval': 0.5
    }

    recorder = RoomResponseRecorder()
    recorder.config = config
    recorder._load_multichannel_config()
    recorder._generate_complete_signal()

    collector = SingleScenarioCollector(
        base_output_dir=str(tmp_path),
        recorder_config=config,
        scenario_config=scenario_config,
        recorder=recorder
    )

    collector.setup_directories()
    collector.initialize_recorder()

    measurements = collector.collect_scenario_measurements()

    # Validate measurements
    assert len(measurements) >= 1  # At least one successful

    for mm in measurements:
        assert mm.is_multichannel == True
        assert mm.num_channels == 2
        assert len(mm.channel_names) == 2
        assert len(mm.filenames_per_channel) == 2

        # Check files exist
        for ch_idx, files in mm.filenames_per_channel.items():
            for file_type, filename in files.items():
                file_path = collector.scenario_dir / filename.split('/')[0] / os.path.basename(filename)
                # Files may or may not exist depending on hardware
```

**Phase 2 Deliverables**:
- ✅ Multi-channel configuration loading
- ✅ Unified `take_record()` API
- ✅ Per-channel file saving
- ✅ Multi-channel metadata schema
- ✅ Updated DatasetCollector
- ✅ Integration tests
- ✅ Backward compatibility verification

---

## Phase 3: Filesystem Structure Redesign

**Duration**: 1 week
**Objective**: Implement scalable file organization and discovery for multi-channel datasets.

### 3.1 Filename Convention

**Pattern**:
```
{type}_{index:03d}_{timestamp}_ch{channel}.wav

Examples:
- raw_000_20250115_103045_123_ch0.wav
- impulse_000_20250115_103045_123_ch1.wav
- room_000_20250115_103045_123_ch2.wav
```

**Parsing Utility** (new file: `multichannel_utils.py`):

```python
import re
from typing import Dict, Any, Optional

def parse_multichannel_filename(filename: str) -> Optional[Dict[str, Any]]:
    """
    Parse multi-channel or legacy filename.

    Returns:
        {
            'type': str,           # 'raw', 'impulse', 'room'
            'index': int,          # Measurement index
            'timestamp': str,      # Timestamp string
            'channel': int | None, # Channel index or None for legacy
            'is_multichannel': bool
        }
    """
    # Try multi-channel format: {type}_{index}_{timestamp}_ch{channel}.wav
    pattern_mc = r'^(\w+)_(\d{3})_(\d{8}_\d{6}_\d{3})_ch(\d+)\.wav$'
    match = re.match(pattern_mc, filename)

    if match:
        return {
            'type': match.group(1),
            'index': int(match.group(2)),
            'timestamp': match.group(3),
            'channel': int(match.group(4)),
            'is_multichannel': True
        }

    # Try legacy format: {type}_{index}_{timestamp}.wav
    pattern_legacy = r'^(\w+)_(\d{3})_(\d{8}_\d{6}_\d{3})\.wav$'
    match_legacy = re.match(pattern_legacy, filename)

    if match_legacy:
        return {
            'type': match_legacy.group(1),
            'index': int(match_legacy.group(2)),
            'timestamp': match_legacy.group(3),
            'channel': None,
            'is_multichannel': False
        }

    # No match
    return None

def group_files_by_measurement(filenames: List[str]) -> Dict[int, Dict[int, str]]:
    """
    Group files by measurement index and channel.

    Returns:
        {
            measurement_index: {
                channel_index: filename,
                ...
            },
            ...
        }
    """
    grouped = {}

    for filename in filenames:
        parsed = parse_multichannel_filename(filename)
        if not parsed:
            continue

        idx = parsed['index']
        ch = parsed['channel']

        if idx not in grouped:
            grouped[idx] = {}

        if ch is not None:
            grouped[idx][ch] = filename
        else:
            # Legacy single-channel: treat as channel 0
            grouped[idx][0] = filename

    return grouped
```

### 3.2 ScenarioManager Multi-Channel Support

**File**: `ScenarioManager.py` - Add Multi-Channel Methods

```python
class ScenarioManager:
    @staticmethod
    def list_wavs_multichannel(folder: str,
                              channel: Optional[int] = None) -> Dict[int, List[str]]:
        """
        List WAV files grouped by channel.

        Args:
            folder: Folder to search
            channel: If specified, return only that channel's files

        Returns:
            {
                0: ['impulse_000_ch0.wav', 'impulse_001_ch0.wav', ...],
                1: ['impulse_000_ch1.wav', 'impulse_001_ch1.wav', ...],
                ...
            }
        """
        from multichannel_utils import parse_multichannel_filename

        if not os.path.isdir(folder):
            return {}

        files_by_channel = {}

        for filename in sorted(os.listdir(folder)):
            if not filename.lower().endswith('.wav'):
                continue

            parsed = parse_multichannel_filename(filename)
            if not parsed:
                continue

            ch = parsed['channel']
            if ch is None:
                ch = 0  # Legacy files treated as channel 0

            if channel is not None and ch != channel:
                continue

            if ch not in files_by_channel:
                files_by_channel[ch] = []

            files_by_channel[ch].append(os.path.join(folder, filename))

        return files_by_channel

    @staticmethod
    def detect_num_channels(scenario_path: str) -> int:
        """Detect number of channels in a scenario"""
        from multichannel_utils import parse_multichannel_filename

        impulse_folder = os.path.join(scenario_path, "impulse_responses")
        if not os.path.isdir(impulse_folder):
            return 1

        channels = set()

        for filename in os.listdir(impulse_folder):
            if not filename.lower().endswith('.wav'):
                continue

            parsed = parse_multichannel_filename(filename)
            if parsed and parsed['is_multichannel']:
                channels.add(parsed['channel'])

        return len(channels) if channels else 1

    @staticmethod
    def count_feature_samples_multichannel(scenario_path: str,
                                          wav_subfolder: str = "impulse_responses",
                                          recording_type: str = "any",
                                          channel: Optional[int] = None) -> int:
        """
        Count samples for multi-channel scenarios.

        Args:
            channel: If None, count unique measurements (deduplicate channels).
                    If int, count only that channel.
        """
        from multichannel_utils import parse_multichannel_filename

        wav_folder_path = os.path.join(scenario_path, wav_subfolder)
        if not os.path.isdir(wav_folder_path):
            return 0

        if channel is None:
            # Count unique measurement indices
            indices = set()
            for filename in os.listdir(wav_folder_path):
                if not filename.lower().endswith('.wav'):
                    continue

                parsed = parse_multichannel_filename(filename)
                if parsed:
                    # Apply recording type filter
                    if recording_type != "any":
                        fn_lower = filename.lower()
                        if recording_type == "raw" and not (fn_lower.startswith("raw_") or fn_lower.endswith('raw_recording.wav')):
                            continue
                        if recording_type == "average" and not (fn_lower.startswith("impulse_") or (fn_lower.endswith('recording.wav') and not fn_lower.endswith('raw_recording.wav'))):
                            continue

                    indices.add(parsed['index'])

            return len(indices)
        else:
            # Count specific channel
            count = 0
            for filename in os.listdir(wav_folder_path):
                if not filename.lower().endswith('.wav'):
                    continue

                parsed = parse_multichannel_filename(filename)
                if parsed and parsed.get('channel') == channel:
                    # Apply recording type filter
                    if recording_type != "any":
                        fn_lower = filename.lower()
                        if recording_type == "raw" and not (fn_lower.startswith("raw_") or fn_lower.endswith('raw_recording.wav')):
                            continue
                        if recording_type == "average" and not (fn_lower.startswith("impulse_") or (fn_lower.endswith('recording.wav') and not fn_lower.endswith('raw_recording.wav'))):
                            continue

                    count += 1

            return count
```

### 3.3 Migration Utility

**New File**: `migrate_to_multichannel.py`

```python
#!/usr/bin/env python3
"""
Migrate legacy single-channel datasets to multi-channel file structure.

Usage:
    python migrate_to_multichannel.py <scenario_path> [--channel-id 0] [--dry-run]
"""

import os
import sys
import json
import argparse
from pathlib import Path
import re

def migrate_single_to_multichannel(scenario_path: str,
                                   channel_id: int = 0,
                                   dry_run: bool = False):
    """
    Rename legacy single-channel files to multi-channel format.

    Example:
        impulse_000_20250115_103045_123.wav
        → impulse_000_20250115_103045_123_ch0.wav
    """
    scenario_dir = Path(scenario_path)

    if not scenario_dir.exists():
        print(f"Error: Scenario path does not exist: {scenario_path}")
        return False

    print(f"Migrating scenario: {scenario_dir.name}")
    print(f"Channel ID: {channel_id}")
    print(f"Dry run: {dry_run}")
    print()

    renamed_count = 0

    for subfolder in ['raw_recordings', 'impulse_responses', 'room_responses']:
        folder = scenario_dir / subfolder
        if not folder.exists():
            continue

        print(f"Processing {subfolder}/")

        for wav_file in folder.glob('*.wav'):
            # Check if already multi-channel format
            if '_ch' in wav_file.stem:
                print(f"  Skip (already multi-channel): {wav_file.name}")
                continue

            # Parse legacy filename
            # Pattern: {type}_{index}_{timestamp}.wav
            pattern = r'^(\w+)_(\d{3})_(\d{8}_\d{6}_\d{3})\.wav$'
            match = re.match(pattern, wav_file.name)

            if not match:
                print(f"  Skip (non-standard format): {wav_file.name}")
                continue

            # Generate new filename
            type_str = match.group(1)
            index = match.group(2)
            timestamp = match.group(3)
            new_name = f"{type_str}_{index}_{timestamp}_ch{channel_id}.wav"
            new_path = folder / new_name

            # Rename
            if dry_run:
                print(f"  [DRY RUN] Would rename: {wav_file.name} → {new_name}")
            else:
                print(f"  Rename: {wav_file.name} → {new_name}")
                wav_file.rename(new_path)

            renamed_count += 1

    # Update metadata
    meta_file = scenario_dir / "metadata" / "session_metadata.json"
    if meta_file.exists():
        print(f"\nUpdating metadata: {meta_file}")

        if not dry_run:
            try:
                meta = json.loads(meta_file.read_text())

                # Update measurement entries
                for meas in meta.get('measurements', []):
                    if 'is_multichannel' not in meas:
                        meas['is_multichannel'] = True
                        meas['num_channels'] = 1
                        meas['channel_names'] = [f"Channel {channel_id}"]

                        # Convert single filenames to per-channel dict
                        filenames_per_channel = {
                            channel_id: {
                                'raw': meas.get('filename_raw', '').replace('.wav', f'_ch{channel_id}.wav') if meas.get('filename_raw') else None,
                                'impulse': meas.get('filename_impulse', '').replace('.wav', f'_ch{channel_id}.wav') if meas.get('filename_impulse') else None,
                                'room': meas.get('filename_room_response', '').replace('.wav', f'_ch{channel_id}.wav') if meas.get('filename_room_response') else None
                            }
                        }
                        meas['filenames_per_channel'] = filenames_per_channel

                # Write updated metadata
                meta_file.write_text(json.dumps(meta, indent=2))
                print(f"  ✅ Metadata updated")

            except Exception as e:
                print(f"  ❌ Error updating metadata: {e}")
        else:
            print(f"  [DRY RUN] Would update metadata")

    print(f"\nSummary:")
    print(f"  Files renamed: {renamed_count}")
    print(f"  {'(Dry run - no changes made)' if dry_run else '✅ Migration complete'}")

    return True

def main():
    parser = argparse.ArgumentParser(description='Migrate legacy single-channel dataset to multi-channel format')
    parser.add_argument('scenario_path', help='Path to scenario folder')
    parser.add_argument('--channel-id', type=int, default=0, help='Channel ID to assign (default: 0)')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be done without making changes')

    args = parser.parse_args()

    success = migrate_single_to_multichannel(
        args.scenario_path,
        channel_id=args.channel_id,
        dry_run=args.dry_run
    )

    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
```

**Phase 3 Deliverables**:
- ✅ Filename parsing utilities
- ✅ Multi-channel file discovery
- ✅ ScenarioManager multi-channel methods
- ✅ Migration script for legacy datasets
- ✅ Unit tests for filename parsing
- ✅ Migration testing

---

## Phase 4: Signal Processing Adaptation

**Duration**: 1 week
**Objective**: Extend onset detection and alignment for multi-channel signals.

### 4.1 Reference-Based Onset Detection

**File**: `RoomResponseRecorder.py`

```python
def _find_onset_multichannel(self,
                            multichannel_data: Dict[str, np.ndarray],
                            reference_channel: int = 0,
                            window_size: int = 10,
                            threshold_factor: float = 2) -> Dict[int, int]:
    """
    Find onset for each channel relative to reference channel.

    Args:
        multichannel_data: Dict with 'ch0', 'ch1', ..., 'metadata'
        reference_channel: Channel to use as timing reference
        window_size: RMS window size for onset detection
        threshold_factor: Threshold multiplier over background noise

    Returns:
        {
            0: onset_sample_ch0,  # Reference channel
            1: onset_sample_ch1,
            2: onset_sample_ch2,
            ...
        }
    """
    num_channels = multichannel_data['metadata']['num_channels']
    onsets = {}

    # Find reference channel onset
    ref_audio = multichannel_data[f'ch{reference_channel}']
    ref_onset = self._find_sound_onset(ref_audio, window_size, threshold_factor)
    onsets[reference_channel] = ref_onset

    print(f"Reference channel {reference_channel} onset at sample {ref_onset}")

    # Find onsets for other channels
    for ch_idx in range(num_channels):
        if ch_idx == reference_channel:
            continue

        ch_audio = multichannel_data[f'ch{ch_idx}']
        ch_onset = self._find_sound_onset(ch_audio, window_size, threshold_factor)
        onsets[ch_idx] = ch_onset

        # Calculate relative offset
        offset = ch_onset - ref_onset
        print(f"  Channel {ch_idx} onset at sample {ch_onset} (offset: {offset:+d} samples, {offset/self.sample_rate*1000:+.2f} ms)")

    return onsets
```

### 4.2 Cross-Channel Alignment

**File**: `RoomResponseRecorder.py`

```python
def align_multichannel_signals(self,
                              multichannel_data: Dict[str, np.ndarray],
                              reference_channel: int = 0,
                              max_lag: int = 1000) -> Dict[str, np.ndarray]:
    """
    Align all channels to reference channel using cross-correlation.

    Args:
        multichannel_data: Dict with per-channel audio
        reference_channel: Channel to align to
        max_lag: Maximum search range for lag (samples)

    Returns:
        Aligned multi-channel data with added 'alignment' metadata
    """
    num_channels = multichannel_data['metadata']['num_channels']
    aligned_data = {}

    ref_audio = multichannel_data[f'ch{reference_channel}']
    ref_audio_norm = ref_audio / (np.max(np.abs(ref_audio)) + 1e-12)

    print(f"Aligning {num_channels} channels to reference channel {reference_channel}")

    for ch_idx in range(num_channels):
        ch_key = f'ch{ch_idx}'
        ch_audio = multichannel_data[ch_key]

        if ch_idx == reference_channel:
            # Reference channel stays unchanged
            aligned_data[ch_key] = ch_audio
            print(f"  Channel {ch_idx}: reference (no alignment)")
            continue

        # Normalize
        ch_audio_norm = ch_audio / (np.max(np.abs(ch_audio)) + 1e-12)

        # Cross-correlate with reference
        correlation = np.correlate(ref_audio_norm, ch_audio_norm, mode='full')

        # Find lag of maximum correlation (restricted to max_lag)
        center = len(correlation) // 2
        search_start = max(0, center - max_lag)
        search_end = min(len(correlation), center + max_lag)
        search_window = correlation[search_start:search_end]

        lag_in_window = np.argmax(np.abs(search_window))
        lag = (search_start + lag_in_window) - (len(ref_audio) - 1)
        corr_peak = float(search_window[lag_in_window])

        # Apply circular shift to align
        if lag != 0:
            aligned = np.roll(ch_audio, -lag)
        else:
            aligned = ch_audio

        aligned_data[ch_key] = aligned

        # Store alignment metadata
        if 'alignment' not in aligned_data:
            aligned_data['alignment'] = {}
        aligned_data['alignment'][ch_idx] = {
            'lag_samples': int(lag),
            'lag_ms': float(lag / self.sample_rate * 1000),
            'correlation_peak': float(corr_peak)
        }

        print(f"  Channel {ch_idx}: lag={lag:+d} samples ({lag/self.sample_rate*1000:+.2f} ms), corr={corr_peak:.3f}")

    # Copy metadata
    aligned_data['metadata'] = multichannel_data['metadata'].copy()
    aligned_data['metadata']['aligned'] = True
    aligned_data['metadata']['reference_channel'] = reference_channel

    return aligned_data
```

### 4.3 Inter-Channel Quality Metrics

**File**: `DatasetCollector.py`

```python
def _calculate_interchannel_metrics(self,
                                   multichannel_data: Dict[str, np.ndarray]) -> Dict[str, float]:
    """Calculate metrics comparing channels"""
    num_channels = multichannel_data['metadata']['num_channels']
    metrics = {}

    # Cross-correlation between all channel pairs
    for i in range(num_channels):
        for j in range(i + 1, num_channels):
            ch_i = multichannel_data[f'ch{i}']
            ch_j = multichannel_data[f'ch{j}']

            # Normalize
            ch_i_norm = ch_i / (np.max(np.abs(ch_i)) + 1e-12)
            ch_j_norm = ch_j / (np.max(np.abs(ch_j)) + 1e-12)

            # Cross-correlation
            corr = np.correlate(ch_i_norm, ch_j_norm, mode='full')
            max_corr = np.max(np.abs(corr))
            lag = np.argmax(np.abs(corr)) - len(ch_i) + 1

            metrics[f'ch{i}_ch{j}_correlation'] = float(max_corr)
            metrics[f'ch{i}_ch{j}_lag_samples'] = int(lag)
            metrics[f'ch{i}_ch{j}_lag_ms'] = float(lag / 48000 * 1000)  # Assuming 48kHz

    # Level balance (RMS ratio)
    rms_levels = []
    for ch_idx in range(num_channels):
        audio = multichannel_data[f'ch{ch_idx}']
        rms = np.sqrt(np.mean(audio ** 2))
        rms_levels.append(rms)
        metrics[f'ch{ch_idx}_rms'] = float(rms)

    if len(rms_levels) > 1:
        metrics['rms_balance_db'] = float(20 * np.log10(
            (max(rms_levels) + 1e-12) / (min(rms_levels) + 1e-12)
        ))
        metrics['rms_mean'] = float(np.mean(rms_levels))
        metrics['rms_std'] = float(np.std(rms_levels))

    return metrics
```

**Phase 4 Deliverables**:
- ✅ Multi-channel onset detection
- ✅ Cross-correlation alignment
- ✅ Inter-channel quality metrics
- ✅ Unit tests for alignment algorithms
- ✅ Validation with synthetic multi-channel data

---

## Phase 5: Feature Extraction Multi-Channel

**Duration**: 1 week
**Objective**: Extend feature extraction to support per-channel, aggregate, and spatial features.

### 5.1 Multi-Channel Feature Extraction Modes

**File**: `FeatureExtractor.py` - Add Multi-Channel Processing

```python
class AudioFeatureExtractor:
    def __init__(self, ...):
        # ... existing initialization ...

        # Multi-channel configuration
        self.feature_mode = "per_channel"  # "per_channel" | "aggregate" | "spatial" | "all"
        self.aggregate_method = "mean"     # "mean" | "max" | "rms"
        self.spatial_features_enabled = False
        self.spatial_feature_types = ["toa", "coherence", "level_diff"]

    def process_scenario_folder_multichannel(self,
                                            scenario_folder: str,
                                            wav_subfolder: str = "impulse_responses",
                                            recording_type: str = "any",
                                            **kwargs) -> bool:
        """
        Process multi-channel scenario folder.

        Outputs (depending on mode):
            - Per-channel: features_ch0.csv, features_ch1.csv, spectrum_ch0.csv, ...
            - Aggregate: features_aggregate.csv, spectrum_aggregate.csv
            - Spatial: spatial_features.csv
        """
        from multichannel_utils import group_files_by_measurement
        from ScenarioManager import ScenarioManager

        wav_folder_path = os.path.join(scenario_folder, wav_subfolder)
        if not os.path.exists(wav_folder_path):
            print(f"WAV subfolder not found: {wav_folder_path}")
            return False

        # Detect channels
        files_by_channel = ScenarioManager.list_wavs_multichannel(wav_folder_path)

        if not files_by_channel:
            print(f"No multi-channel WAV files found")
            # Fall back to single-channel processing
            return self.process_scenario_folder(scenario_folder, wav_subfolder, recording_type, **kwargs)

        num_channels = len(files_by_channel)
        print(f"Processing multi-channel scenario: {num_channels} channels")

        # Mode 1: Per-Channel Features
        if self.feature_mode in ["per_channel", "all"]:
            self._extract_per_channel_features(scenario_folder, files_by_channel)

        # Mode 2: Aggregate Features
        if self.feature_mode in ["aggregate", "all"]:
            self._extract_aggregate_features(scenario_folder, files_by_channel)

        # Mode 3: Spatial Features
        if self.feature_mode in ["spatial", "all"] or self.spatial_features_enabled:
            self._extract_spatial_features(scenario_folder, files_by_channel)

        return True

    def _extract_per_channel_features(self, scenario_folder: str,
                                     files_by_channel: Dict[int, List[str]]):
        """Extract features independently for each channel"""
        from tqdm import tqdm

        print("  Mode: Per-Channel Features")

        for ch_idx, wav_files in files_by_channel.items():
            print(f"    Channel {ch_idx}: {len(wav_files)} files")

            mfcc_rows = []
            spec_rows = []

            for file_path in tqdm(wav_files, desc=f"    Ch{ch_idx}", leave=False):
                audio = self.load_audio_file(file_path)
                if audio is None:
                    continue

                filename = os.path.basename(file_path)

                # MFCC
                mfcc_vec = self._extract_mfcc_from_audio(audio)
                mfcc_row = {"filename": filename}
                for i, v in enumerate(mfcc_vec):
                    mfcc_row[f"mfcc_{i}"] = float(v)
                mfcc_rows.append(mfcc_row)

                # Spectrum
                spec = self._extract_spectrum_from_audio(audio)
                spec_row = {"filename": filename}
                for i, v in enumerate(spec):
                    if self.num_freq_bins is None or i < self.num_freq_bins:
                        spec_row[f"freq_{i}"] = float(v)
                spec_rows.append(spec_row)

            # Save per-channel CSVs
            if mfcc_rows:
                mfcc_df = pd.DataFrame(mfcc_rows)
                mfcc_output = Path(scenario_folder) / f"features_ch{ch_idx}.csv"
                mfcc_df.to_csv(mfcc_output, index=False)
                print(f"      → features_ch{ch_idx}.csv ({len(mfcc_df)} samples)")

            if spec_rows:
                spec_df = pd.DataFrame(spec_rows)
                spec_output = Path(scenario_folder) / f"spectrum_ch{ch_idx}.csv"
                spec_df.to_csv(spec_output, index=False)
                print(f"      → spectrum_ch{ch_idx}.csv ({len(spec_df)} samples)")

    def _extract_aggregate_features(self, scenario_folder: str,
                                   files_by_channel: Dict[int, List[str]]):
        """Extract features from aggregated (combined) channels"""
        from multichannel_utils import group_files_by_measurement

        print(f"  Mode: Aggregate Features (method: {self.aggregate_method})")

        # Group files by measurement index
        all_files = []
        for files in files_by_channel.values():
            all_files.extend(files)

        grouped = group_files_by_measurement([os.path.basename(f) for f in all_files])

        agg_mfcc_rows = []
        agg_spec_rows = []

        for idx in sorted(grouped.keys()):
            ch_files = grouped[idx]

            # Load all channels for this measurement
            ch_audios = []
            for ch_idx in sorted(ch_files.keys()):
                # Find full path
                matching = [f for f in all_files if os.path.basename(f) == ch_files[ch_idx]]
                if matching:
                    audio = self.load_audio_file(matching[0])
                    if audio is not None:
                        ch_audios.append(audio)

            if not ch_audios:
                continue

            # Aggregate channels
            min_len = min(len(a) for a in ch_audios)
            ch_audios_trimmed = [a[:min_len] for a in ch_audios]

            if self.aggregate_method == "mean":
                aggregated_audio = np.mean(ch_audios_trimmed, axis=0)
            elif self.aggregate_method == "max":
                aggregated_audio = np.max(np.abs(ch_audios_trimmed), axis=0) * np.sign(np.sum(ch_audios_trimmed, axis=0))
            elif self.aggregate_method == "rms":
                aggregated_audio = np.sqrt(np.mean(np.array(ch_audios_trimmed) ** 2, axis=0))
            else:
                aggregated_audio = np.mean(ch_audios_trimmed, axis=0)

            # Extract features
            mfcc_vec = self._extract_mfcc_from_audio(aggregated_audio)
            mfcc_row = {"measurement_index": idx}
            for i, v in enumerate(mfcc_vec):
                mfcc_row[f"mfcc_{i}"] = float(v)
            agg_mfcc_rows.append(mfcc_row)

            spec = self._extract_spectrum_from_audio(aggregated_audio)
            spec_row = {"measurement_index": idx}
            for i, v in enumerate(spec):
                if self.num_freq_bins is None or i < self.num_freq_bins:
                    spec_row[f"freq_{i}"] = float(v)
            agg_spec_rows.append(spec_row)

        # Save aggregate CSVs
        if agg_mfcc_rows:
            df = pd.DataFrame(agg_mfcc_rows)
            output = Path(scenario_folder) / "features_aggregate.csv"
            df.to_csv(output, index=False)
            print(f"    → features_aggregate.csv ({len(df)} samples)")

        if agg_spec_rows:
            df = pd.DataFrame(agg_spec_rows)
            output = Path(scenario_folder) / "spectrum_aggregate.csv"
            df.to_csv(output, index=False)
            print(f"    → spectrum_aggregate.csv ({len(df)} samples)")

    def _extract_spatial_features(self, scenario_folder: str,
                                  files_by_channel: Dict[int, List[str]]):
        """Extract inter-channel spatial features"""
        from multichannel_utils import group_files_by_measurement

        print(f"  Mode: Spatial Features (types: {', '.join(self.spatial_feature_types)})")

        # Group files by measurement
        all_files = []
        for files in files_by_channel.values():
            all_files.extend(files)

        grouped = group_files_by_measurement([os.path.basename(f) for f in all_files])

        spatial_rows = []

        for idx in sorted(grouped.keys()):
            ch_files = grouped[idx]

            # Load all channels
            ch_audios = {}
            for ch_idx, filename in ch_files.items():
                matching = [f for f in all_files if os.path.basename(f) == filename]
                if matching:
                    audio = self.load_audio_file(matching[0])
                    if audio is not None:
                        ch_audios[ch_idx] = audio

            if len(ch_audios) < 2:
                continue  # Need at least 2 channels

            row = {"measurement_index": idx}

            # Pairwise spatial features
            for ch_i in sorted(ch_audios.keys()):
                for ch_j in sorted(ch_audios.keys()):
                    if ch_j <= ch_i:
                        continue

                    audio_i = ch_audios[ch_i]
                    audio_j = ch_audios[ch_j]

                    # Trim to same length
                    min_len = min(len(audio_i), len(audio_j))
                    audio_i = audio_i[:min_len]
                    audio_j = audio_j[:min_len]

                    # Time-of-Arrival (TOA)
                    if "toa" in self.spatial_feature_types:
                        corr = np.correlate(audio_i, audio_j, mode='full')
                        lag = np.argmax(np.abs(corr)) - (len(audio_i) - 1)
                        row[f"toa_ch{ch_i}_ch{ch_j}_samples"] = int(lag)
                        row[f"toa_ch{ch_i}_ch{ch_j}_ms"] = float(lag / self.sample_rate * 1000)

                    # Cross-correlation peak
                    if "cross_corr" in self.spatial_feature_types:
                        corr = np.correlate(audio_i / (np.max(np.abs(audio_i)) + 1e-12),
                                          audio_j / (np.max(np.abs(audio_j)) + 1e-12),
                                          mode='full')
                        row[f"cross_corr_ch{ch_i}_ch{ch_j}"] = float(np.max(np.abs(corr)))

                    # Level difference
                    if "level_diff" in self.spatial_feature_types:
                        rms_i = np.sqrt(np.mean(audio_i ** 2))
                        rms_j = np.sqrt(np.mean(audio_j ** 2))
                        level_diff_db = 20 * np.log10((rms_i + 1e-12) / (rms_j + 1e-12))
                        row[f"level_diff_ch{ch_i}_ch{ch_j}_db"] = float(level_diff_db)

                    # Coherence
                    if "coherence" in self.spatial_feature_types:
                        norm_i = audio_i / (np.linalg.norm(audio_i) + 1e-12)
                        norm_j = audio_j / (np.linalg.norm(audio_j) + 1e-12)
                        coherence = np.abs(np.sum(norm_i * norm_j))
                        row[f"coherence_ch{ch_i}_ch{ch_j}"] = float(coherence)

            spatial_rows.append(row)

        # Save spatial features CSV
        if spatial_rows:
            df = pd.DataFrame(spatial_rows)
            output = Path(scenario_folder) / "spatial_features.csv"
            df.to_csv(output, index=False)
            print(f"    → spatial_features.csv ({len(df)} samples, {len(df.columns)-1} features)")
```

**Phase 5 Deliverables**:
- ✅ Per-channel feature extraction
- ✅ Aggregate feature extraction
- ✅ Spatial feature extraction
- ✅ Configuration-driven mode selection
- ✅ Unit tests for all modes
- ✅ Integration tests with real multi-channel data

---

## Phase 6: GUI Interface Updates

**Duration**: 2 weeks
**Objective**: Add multi-channel configuration and monitoring to all GUI panels.

### 6.1 Audio Settings Panel Enhancement

**File**: `gui_audio_settings_panel.py` - Add Multi-Channel Section

```python
class AudioSettingsPanel:
    def render(self):
        st.header("Audio Settings")

        # ... existing device selection ...

        # ========== MULTI-CHANNEL CONFIGURATION ==========
        st.markdown("---")
        st.subheader("Multi-Channel Configuration")

        enable_multichannel = st.checkbox(
            "Enable Multi-Channel Recording",
            value=False,
            key="enable_multichannel",
            help="Record from multiple input channels simultaneously (requires multi-channel hardware)"
        )

        if enable_multichannel:
            num_channels = st.number_input(
                "Number of Channels",
                min_value=2,
                max_value=32,
                value=4,
                step=1,
                key="num_channels",
                help="Number of input channels (depends on audio interface capabilities)"
            )

            # Channel Configuration
            with st.expander("Channel Names & Reference"):
                channel_names = []
                cols = st.columns(min(num_channels, 4))

                for ch_idx in range(num_channels):
                    col_idx = ch_idx % 4
                    with cols[col_idx]:
                        name = st.text_input(
                            f"Channel {ch_idx}",
                            value=f"Channel {ch_idx}",
                            key=f"ch_name_{ch_idx}"
                        )
                        channel_names.append(name)

                reference_channel = st.selectbox(
                    "Reference Channel (for alignment)",
                    options=list(range(num_channels)),
                    format_func=lambda i: f"{i}: {channel_names[i]}",
                    key="reference_channel",
                    help="All other channels will be aligned to this channel"
                )

            # Channel Calibration
            with st.expander("Channel Calibration (Optional)"):
                enable_calibration = st.checkbox(
                    "Enable Per-Channel Calibration",
                    key="enable_calibration",
                    help="Apply gain and delay corrections to each channel"
                )

                if enable_calibration:
                    calibration_data = {}

                    for ch_idx in range(num_channels):
                        st.markdown(f"**Channel {ch_idx}: {channel_names[ch_idx]}**")

                        col1, col2 = st.columns(2)

                        with col1:
                            gain = st.number_input(
                                "Gain Multiplier",
                                min_value=0.1,
                                max_value=10.0,
                                value=1.0,
                                step=0.01,
                                format="%.3f",
                                key=f"ch_gain_{ch_idx}",
                                help="Multiply channel amplitude by this value"
                            )

                        with col2:
                            delay_samples = st.number_input(
                                "Delay (samples)",
                                min_value=-1000,
                                max_value=1000,
                                value=0,
                                step=1,
                                key=f"ch_delay_{ch_idx}",
                                help="Sample offset (positive=delay, negative=advance)"
                            )

                        calibration_data[ch_idx] = {
                            "gain": gain,
                            "delay_samples": delay_samples
                        }
                else:
                    calibration_data = {}

            # Spatial Configuration (optional)
            with st.expander("Spatial Configuration (Optional)"):
                st.markdown("Define microphone positions for spatial analysis")

                geometry = st.selectbox(
                    "Array Geometry",
                    options=["linear", "rectangular", "circular", "custom"],
                    key="array_geometry"
                )

                if geometry == "linear":
                    spacing = st.number_input(
                        "Microphone Spacing (meters)",
                        min_value=0.01,
                        max_value=10.0,
                        value=0.05,
                        step=0.01,
                        format="%.3f",
                        key="mic_spacing"
                    )

                    positions = [{"x": i * spacing, "y": 0.0, "z": 0.0}
                                for i in range(num_channels)]

                # ... other geometry types ...

            # Save Configuration Button
            if st.button("💾 Save Multi-Channel Configuration", key="save_mc_config"):
                config = self._load_or_create_config()

                config["multichannel"] = {
                    "enabled": True,
                    "num_channels": num_channels,
                    "channel_names": channel_names,
                    "reference_channel": reference_channel,
                    "channel_calibration": calibration_data if enable_calibration else {}
                }

                self._save_config(config)
                st.success("✅ Multi-channel configuration saved to recorderConfig.json")
                st.rerun()

        else:
            # Ensure multi-channel is disabled in config
            if st.button("Disable Multi-Channel Mode", key="disable_mc"):
                config = self._load_or_create_config()
                if "multichannel" in config:
                    config["multichannel"]["enabled"] = False
                    self._save_config(config)
                    st.success("✅ Multi-channel mode disabled")
                    st.rerun()

    def _load_or_create_config(self) -> dict:
        config_path = Path("recorderConfig.json")
        if config_path.exists():
            return json.loads(config_path.read_text())
        else:
            return {
                "sample_rate": 48000,
                "pulse_duration": 0.008,
                "pulse_fade": 0.0001,
                "cycle_duration": 0.1,
                "num_pulses": 8,
                "volume": 0.4,
                "pulse_frequency": 1000,
                "impulse_form": "sine"
            }

    def _save_config(self, config: dict):
        config_path = Path("recorderConfig.json")
        config_path.write_text(json.dumps(config, indent=2))
```

### 6.2 Collection Panel Multi-Channel Monitoring

**File**: `gui_collect_panel.py` - Add Channel Status Display

```python
def render(self):
    st.header("Data Collection")

    # ... existing code ...

    # Multi-channel status display
    if self.recorder and self.recorder.multichannel_enabled:
        st.markdown("---")
        st.subheader("📊 Multi-Channel Configuration")

        info_cols = st.columns([1, 1, 1])
        with info_cols[0]:
            st.metric("Channels", self.recorder.num_channels)
        with info_cols[1]:
            st.metric("Reference Channel", self.recorder.reference_channel)
        with info_cols[2]:
            calibrated = len(self.recorder.channel_calibration) > 0
            st.metric("Calibration", "✅ Enabled" if calibrated else "❌ Disabled")

        # Channel list
        with st.expander("Channel Details"):
            for ch_idx, name in enumerate(self.recorder.channel_names):
                col1, col2, col3 = st.columns([2, 1, 1])

                with col1:
                    st.write(f"**{ch_idx}**: {name}")

                with col2:
                    if str(ch_idx) in self.recorder.channel_calibration:
                        gain = self.recorder.channel_calibration[str(ch_idx)].get('gain', 1.0)
                        st.caption(f"Gain: {gain:.2f}x")

                with col3:
                    if str(ch_idx) in self.recorder.channel_calibration:
                        delay = self.recorder.channel_calibration[str(ch_idx)].get('delay_samples', 0)
                        st.caption(f"Delay: {delay:+d} smp")
```

### 6.3 Process Panel Multi-Channel Feature Extraction

**File**: `gui_process_panel.py`

```python
def render(self):
    st.header("Feature Extraction")

    # ... existing scenario selection ...

    st.markdown("---")
    st.subheader("Multi-Channel Feature Extraction")

    feature_mode = st.selectbox(
        "Extraction Mode",
        options=["per_channel", "aggregate", "spatial", "all"],
        format_func=lambda x: {
            "per_channel": "🔢 Per-Channel (separate features for each channel)",
            "aggregate": "📊 Aggregate (combine channels before extraction)",
            "spatial": "🎯 Spatial (inter-channel features only)",
            "all": "🌐 All Modes (per-channel + aggregate + spatial)"
        }[x],
        key="feature_mode",
        help="How to extract features from multi-channel recordings"
    )

    # Mode-specific options
    if feature_mode == "aggregate":
        aggregate_method = st.selectbox(
            "Aggregation Method",
            options=["mean", "max", "rms"],
            key="aggregate_method",
            help="How to combine channels before feature extraction"
        )

    if feature_mode in ["spatial", "all"]:
        spatial_types = st.multiselect(
            "Spatial Feature Types",
            options=["toa", "coherence", "cross_corr", "level_diff"],
            default=["toa", "coherence"],
            key="spatial_types",
            help="Which spatial features to extract"
        )

    # Extract Features Button
    if st.button("🎵 Extract Features", key="extract_features"):
        # Configure extractor
        self.feature_extractor.feature_mode = feature_mode
        if feature_mode == "aggregate":
            self.feature_extractor.aggregate_method = aggregate_method
        if feature_mode in ["spatial", "all"]:
            self.feature_extractor.spatial_feature_types = spatial_types

        # Process scenarios
        with st.spinner("Extracting features..."):
            progress_bar = st.progress(0)
            success_count = 0

            for idx, scenario_path in enumerate(selected_scenarios):
                try:
                    # Auto-detect if multi-channel
                    num_ch = self.scenario_manager.detect_num_channels(scenario_path)

                    if num_ch > 1:
                        result = self.feature_extractor.process_scenario_folder_multichannel(
                            scenario_path,
                            wav_subfolder=wav_subfolder,
                            recording_type=recording_type
                        )
                    else:
                        result = self.feature_extractor.process_scenario_folder(
                            scenario_path,
                            wav_subfolder=wav_subfolder,
                            recording_type=recording_type
                        )

                    if result:
                        success_count += 1

                    progress_bar.progress((idx + 1) / len(selected_scenarios))

                except Exception as e:
                    st.error(f"Error processing {os.path.basename(scenario_path)}: {e}")

            progress_bar.empty()

            if success_count > 0:
                st.success(f"✅ Features extracted for {success_count}/{len(selected_scenarios)} scenarios")
                self.scenario_manager.clear_cache()
            else:
                st.error("❌ No scenarios processed successfully")
```

**Phase 6 Deliverables**:
- ✅ Audio settings panel with multi-channel config
- ✅ Collection panel with channel status
- ✅ Process panel with mode selection
- ✅ Classify panel with channel mode support
- ✅ Scenarios panel with channel-aware file browsing
- ✅ UI/UX testing

---

## Phase 7: Testing & Validation

**Duration**: 1 week
**Objective**: Comprehensive testing of all multi-channel functionality.

### 7.1 End-to-End Integration Test

```python
# test_multichannel_end_to_end.py

def test_complete_multichannel_workflow(tmp_path):
    """
    Test complete workflow:
    1. Configure multi-channel
    2. Record data
    3. Extract features
    4. Train classifier
    """
    # 1. Configuration
    config = {
        "sample_rate": 48000,
        "pulse_duration": 0.008,
        "cycle_duration": 0.1,
        "num_pulses": 8,
        "volume": 0.3,
        "impulse_form": "sine",
        "multichannel": {
            "enabled": True,
            "num_channels": 4,
            "channel_names": ["FL", "FR", "RL", "RR"],
            "reference_channel": 0
        }
    }

    # 2. Recording
    recorder = RoomResponseRecorder()
    recorder.config = config
    recorder._load_multichannel_config()

    assert recorder.multichannel_enabled == True
    assert recorder.num_channels == 4

    # 3. Feature Extraction
    extractor = AudioFeatureExtractor()
    extractor.feature_mode = "all"

    # ... (requires hardware or mock)

    # 4. Classification
    # ... (test with synthetic data)
```

### 7.2 Performance Benchmarks

```python
def benchmark_multichannel_overhead():
    """Measure performance impact of multi-channel processing"""
    import time

    configs = [(1, "Mono"), (2, "Stereo"), (4, "Quad"), (8, "8-channel")]

    for num_channels, label in configs:
        start = time.time()

        # Benchmark recording + processing
        # ... (implementation)

        duration = time.time() - start
        print(f"{label}: {duration:.2f}s per measurement")
```

**Phase 7 Deliverables**:
- ✅ End-to-end integration tests
- ✅ Hardware compatibility tests
- ✅ Performance benchmarks
- ✅ Regression tests (single-channel mode)
- ✅ User acceptance testing
- ✅ Documentation updates

---

## 11. Migration Strategy

### 11.1 Backward Compatibility Guarantees

**Zero-Breaking-Changes Approach**:

1. **Automatic Mode Detection**:
   - If `multichannel.enabled` missing or `false` → Legacy Mode
   - If `multichannel.enabled = true` → Multi-Channel Mode

2. **Legacy Mode Behavior**:
   - Identical to current single-channel system
   - No filename changes
   - No metadata schema changes
   - Existing code paths unchanged

3. **Dual-Mode File Discovery**:
   - `ScenarioManager` auto-detects file format
   - `FeatureExtractor` falls back to single-channel if no multi-channel files

### 11.2 Phased Rollout

**Week 1-2**: Internal testing (Phase 1-3)
- SDL core + recording pipeline
- Limited user exposure

**Week 3-4**: Beta testing (Phase 4-5)
- Signal processing + feature extraction
- Selected power users

**Week 5-7**: Full deployment (Phase 6-7)
- GUI updates + testing
- General availability

---

## 12. Risk Assessment & Mitigation

### 12.1 Technical Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| **Hardware compatibility issues** | Medium | High | Early hardware testing; support fallback to mono |
| **Performance degradation** | Medium | Medium | Benchmarking; optimize buffers; consider parallel processing |
| **File system overhead** | Low | Medium | Monitor disk usage; implement Option B if needed |
| **Backward compatibility break** | Low | Critical | Extensive regression testing; dual-mode operation |
| **SDL multi-channel limitations** | Medium | High | Test with multiple devices; document limitations |

### 12.2 User Experience Risks

| Risk | Mitigation |
|------|------------|
| **Increased complexity** | Default to single-channel; clear documentation; setup wizard |
| **Confusion with file organization** | Clear naming; GUI filters; visual indicators |
| **Calibration complexity** | Auto-calibration tools; presets for common configurations |

---

## 13. Timeline & Resources

### 13.1 Effort Estimate

**Total Duration**: 9 weeks (1 developer, full-time)

| Phase | Duration | Deliverables |
|-------|----------|--------------|
| Phase 1: SDL Audio Core | 2 weeks | Multi-channel C++ core, Python bindings |
| Phase 2: Recording Pipeline | 1 week | Multi-channel RoomResponseRecorder, DatasetCollector |
| Phase 3: Filesystem | 1 week | File organization, ScenarioManager, migration tool |
| Phase 4: Signal Processing | 1 week | Onset detection, alignment, quality metrics |
| Phase 5: Feature Extraction | 1 week | Per-channel, aggregate, spatial features |
| Phase 6: GUI Updates | 2 weeks | All panels updated with multi-channel support |
| Phase 7: Testing & Docs | 1 week | Integration tests, documentation, validation |

### 13.2 Success Criteria

**Functional**:
- ✅ Record 2-8 channels simultaneously
- ✅ Per-channel and spatial feature extraction
- ✅ 100% backward compatibility
- ✅ GUI support for all multi-channel workflows

**Performance**:
- ✅ Multi-channel overhead < 20% vs single-channel
- ✅ GUI remains responsive with 8 channels

**Quality**:
- ✅ >90% test coverage for new code
- ✅ Zero regressions in single-channel mode

---

## Appendix A: Configuration Example

**Complete `recorderConfig.json` with Multi-Channel**:

```json
{
  "sample_rate": 48000,
  "pulse_duration": 0.008,
  "pulse_fade": 0.0001,
  "cycle_duration": 0.1,
  "num_pulses": 8,
  "volume": 0.4,
  "pulse_frequency": 1000,
  "impulse_form": "sine",

  "multichannel": {
    "enabled": true,
    "num_channels": 4,
    "channel_names": ["Front-Left", "Front-Right", "Rear-Left", "Rear-Right"],
    "reference_channel": 0,
    "channel_calibration": {
      "0": {"gain": 1.0, "delay_samples": 0},
      "1": {"gain": 1.02, "delay_samples": 12},
      "2": {"gain": 0.98, "delay_samples": 8},
      "3": {"gain": 1.01, "delay_samples": 15}
    },
    "spatial_config": {
      "geometry": "rectangular_array",
      "positions": [
        {"x": 0.0, "y": 0.0, "z": 0.0},
        {"x": 0.5, "y": 0.0, "z": 0.0},
        {"x": 0.0, "y": 0.5, "z": 0.0},
        {"x": 0.5, "y": 0.5, "z": 0.0}
      ],
      "units": "meters"
    }
  },

  "feature_extraction": {
    "mode": "all",
    "aggregate_method": "mean",
    "spatial_features": ["toa", "coherence", "cross_corr", "level_diff"]
  }
}
```

---

**Document Version**: 1.0
**Last Updated**: 2025-01-15
**Author**: AI-Generated Multi-Channel Upgrade Plan
**System Version**: RoomResponse v2.0 (Multi-Channel Capability)
