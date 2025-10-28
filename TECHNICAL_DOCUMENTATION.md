# RoomResponse System - Comprehensive Technical Documentation

> **Advanced Acoustic Room Response Measurement & Analysis System**
>
> A sophisticated audio analysis platform combining low-latency C++ audio engine, Python signal processing, and machine learning for acoustic scenario classification.

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [System Architecture Overview](#system-architecture-overview)
3. [Core Components Deep Dive](#core-components-deep-dive)
4. [Data Flow Architecture](#data-flow-architecture)
5. [Technology Stack](#technology-stack)
6. [File Organization](#file-organization)
7. [Entry Points & Usage](#entry-points--usage)
8. [Signal Processing Pipeline](#signal-processing-pipeline)
9. [Machine Learning Framework](#machine-learning-framework)
10. [GUI Architecture](#gui-architecture)
11. [Configuration & Metadata](#configuration--metadata)
12. [Error Handling & Quality Assurance](#error-handling--quality-assurance)
13. [Performance Considerations](#performance-considerations)
14. [Extension Points](#extension-points)
15. [Development Guidelines](#development-guidelines)

---

## 1. Executive Summary

### 1.1 System Purpose

**RoomResponse** is a professional-grade acoustic measurement system designed to:
- **Record room impulse responses** with microsecond-precision timing
- **Extract acoustic features** (MFCC, frequency spectrum) for machine learning
- **Classify acoustic scenarios** using trained ML models (SVM, Logistic Regression)
- **Manage datasets** of room response measurements with rich metadata
- **Provide interactive visualization** and analysis through a web-based GUI

### 1.2 Key Capabilities

- **Low-latency audio engine**: Custom C++ SDL2 module with Python bindings (pybind11)
- **Configurable signal generation**: Sine/square pulse trains with precise timing control
- **Automated data collection**: Series recording with pause/resume, quality metrics
- **Feature extraction**: MFCC and spectrum analysis via librosa
- **ML classification**: Binary and multi-class classification with cross-validation
- **Streamlit GUI**: Modular panels for collection, processing, classification, visualization

### 1.3 Typical Workflow

```
┌──────────────┐       ┌──────────────┐       ┌──────────────┐       ┌──────────────┐
│   COLLECT    │ ────> │   PROCESS    │ ────> │   CLASSIFY   │ ────> │  VISUALIZE   │
│ (Audio Data) │       │  (Features)  │       │  (ML Train)  │       │  (Results)   │
└──────────────┘       └──────────────┘       └──────────────┘       └──────────────┘
     ↓                      ↓                      ↓                      ↓
  WAV files          features.csv           trained model          interactive
  metadata.json      spectrum.csv           metrics.json              charts
```

---

## 2. System Architecture Overview

### 2.1 Architectural Layers

```
┌─────────────────────────────────────────────────────────────────┐
│                    PRESENTATION LAYER                           │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ Streamlit Web GUI (gui_launcher.py, piano_response.py)  │   │
│  │  - Scenarios Panel  - Process Panel  - Visualize Panel  │   │
│  │  - Collect Panel    - Classify Panel  - Settings Panel  │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      BUSINESS LOGIC LAYER                       │
│  ┌─────────────────┐  ┌─────────────────┐  ┌────────────────┐  │
│  │ ScenarioManager │  │FeatureExtractor │  │   Classifier   │  │
│  │  - Metadata     │  │  - MFCC         │  │  - SVM Model   │  │
│  │  - Filtering    │  │  - Spectrum     │  │  - Evaluation  │  │
│  │  - Caching      │  │  - Alignment    │  │  - Persistence │  │
│  └─────────────────┘  └─────────────────┘  └────────────────┘  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │         DatasetCollector & SeriesWorker                 │   │
│  │  - Collection orchestration  - Quality assessment       │   │
│  │  - Pause/resume/stop        - Metadata persistence      │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      AUDIO ENGINE LAYER                         │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │         RoomResponseRecorder (Python)                   │   │
│  │  - Signal generation    - Audio processing              │   │
│  │  - Device management    - Onset detection               │   │
│  │  - Configuration        - Quality metrics               │   │
│  └─────────────────────────────────────────────────────────┘   │
│                              ▼                                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │     sdl_audio_core (C++ with pybind11 bindings)         │   │
│  │  - SDL2 audio engine    - Low-latency I/O               │   │
│  │  - Device enumeration   - Simultaneous playback/record  │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                       HARDWARE LAYER                            │
│  Audio Input Devices (Microphones) ←→ Audio Output (Speakers)  │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 Design Patterns

| Pattern | Application | Purpose |
|---------|-------------|---------|
| **Module Pattern** | GUI panels, managers | Encapsulation and reusability |
| **Repository Pattern** | ScenarioManager | Data access abstraction |
| **Strategy Pattern** | FeatureExtractor | Pluggable feature types (MFCC/spectrum) |
| **Builder Pattern** | Signal generation | Configurable pulse train construction |
| **Observer Pattern** | SeriesWorker queues | Event-driven background processing |
| **Facade Pattern** | RoomResponseRecorder | Simplified audio engine interface |
| **Template Method** | DatasetCollector | Extensible collection workflow |

---

## 3. Core Components Deep Dive

### 3.1 RoomResponseRecorder (Python Audio Interface)

**File**: `RoomResponseRecorder.py`

#### Purpose
High-level Python interface for room impulse response measurement. Abstracts SDL audio core complexity and provides signal generation, recording, and processing.

#### Key Responsibilities

1. **Configuration Management**
   - Load parameters from JSON files (`recorderConfig.json`)
   - Validate signal generation parameters
   - Manage audio device selection

2. **Signal Generation**
   - Generate pulse trains (sine, square, or voice_coil waveforms)
   - Apply fade envelopes to prevent clicks
   - Ensure exact sample-level timing
   - Voice coil mode: combines positive pulse with negative pull-back for actuator control

3. **Audio Recording**
   - Coordinate simultaneous playback and recording
   - Handle SDL audio core lifecycle
   - Manage recording buffers

4. **Signal Processing**
   - Extract room response via cycle averaging
   - Detect sound onset using RMS analysis
   - Extract impulse response through circular shift alignment

#### Core Methods

```python
class RoomResponseRecorder:
    def __init__(self, config_file_path: str = None)
    def take_record(output_file, impulse_file, method=2) -> np.ndarray
    def test_mic(duration=10.0, chunk_duration=0.1)
    def list_devices() -> Dict
    def set_audio_devices(input=None, output=None)
    def get_sdl_core_info() -> dict

    # Private internals
    def _generate_complete_signal() -> list
    def _generate_single_pulse(exact_samples) -> np.ndarray
    def _record_method_2() -> np.ndarray
    def _process_recorded_signal(recorded_audio) -> Dict[str, np.ndarray]
    def _extract_impulse_response(room_response) -> np.ndarray
    def _find_sound_onset(audio, window_size=10) -> int
    def _save_wav(audio_data, filename)
```

#### Signal Generation Algorithm

```python
# Pulse Train Configuration
pulse_duration = 8ms       # Duration of each pulse
cycle_duration = 100ms     # Time between pulse starts
num_pulses = 8            # Total number of pulses
pulse_fade = 0.1ms        # Fade in/out duration
volume = 0.4              # Amplitude (0.0-1.0)
pulse_frequency = 1000Hz  # Carrier frequency (sine mode)

# Total signal duration = num_pulses × cycle_duration = 800ms

# Pulse Placement (exact sample-level timing)
for i in range(num_pulses):
    start_sample = i × cycle_samples
    end_sample = start_sample + pulse_samples
    signal[start_sample:end_sample] = single_pulse

# Fade Envelope (prevent clicks)
fade_in = linspace(0, 1, fade_samples)
fade_out = linspace(1, 0, fade_samples)
pulse[:fade_samples] *= fade_in
pulse[-fade_samples:] *= fade_out
```

#### Room Response Extraction

```python
# 1. Reshape recording into cycles (num_pulses × cycle_samples)
reshaped = recorded_audio.reshape(num_pulses, cycle_samples)

# 2. Skip first cycle(s) for system settling
start_cycle = max(1, num_pulses // 4)  # Skip first 25%

# 3. Average cycles to extract room response
room_response = mean(reshaped[start_cycle:], axis=0)

# 4. Find onset via RMS threshold
onset_index = find_sound_onset(room_response)

# 5. Rotate to put onset at beginning (impulse response)
impulse_response = concatenate([room_response[onset_index:],
                                room_response[:onset_index]])
```

#### Configuration Parameters

**File**: `recorderConfig.json`

```json
{
  "sample_rate": 48000,
  "pulse_duration": 0.008,
  "pulse_fade": 0.0001,
  "cycle_duration": 0.1,
  "num_pulses": 8,
  "volume": 0.4,
  "pulse_frequency": 1000,
  "impulse_form": "sine"  // Options: "sine", "square", "voice_coil"
}
```

---

### 3.2 SDL Audio Core (C++ Module)

**Directory**: `sdl_audio_core/`
**Technology**: C++ with pybind11 bindings
**Compiled Binary**: `sdl_audio_core.cp312-win_amd64.pyd` (Windows)

#### Architecture

```cpp
// Main classes exposed to Python via pybind11

class AudioEngine {
public:
    // Lifecycle management
    void initialize(const AudioEngineConfig& config);
    void start();
    void stop();

    // Device management
    void set_input_device(int device_id);
    void set_output_device(int device_id);

    // Recording control
    void start_recording();
    void stop_recording();
    std::vector<float> get_recorded_data();
    void clear_recording_buffer();

    // Static helpers
    static std::string get_sdl_version();
    static std::vector<std::string> get_audio_drivers();
};

class AudioEngineConfig {
public:
    int sample_rate = 48000;
    bool enable_logging = false;
};

// Device enumeration
struct AudioDeviceInfo {
    int device_id;
    std::string name;
    bool is_input;
};

// Free functions
DeviceList list_all_devices();
AudioDeviceInfo* get_input_devices();
AudioDeviceInfo* get_output_devices();
MeasurementResult measure_room_response_auto(
    const std::vector<float>& playback_signal,
    float volume,
    int input_device,
    int output_device
);
```

#### Key Features

1. **Low-Latency I/O**: SDL2 provides cross-platform low-latency audio
2. **Simultaneous Playback/Recording**: Single engine instance handles both
3. **Device Enumeration**: Lists all available audio devices with metadata
4. **Thread Safety**: Internal locking for concurrent access
5. **Error Handling**: Exceptions propagated to Python layer

#### Python Bindings (pybind11)

```python
# Exposed to Python as:
import sdl_audio_core

# Device listing
devices = sdl_audio_core.list_all_devices()
# Returns: {'input_devices': [...], 'output_devices': [...]}

# Convenience measurement function
result = sdl_audio_core.measure_room_response_auto(
    playback_signal=[...],  # List[float]
    volume=0.4,
    input_device=-1,        # -1 = auto-select
    output_device=-1
)
# Returns: {'success': bool, 'recorded_data': List[float], ...}

# Manual control
engine = sdl_audio_core.AudioEngine()
config = sdl_audio_core.AudioEngineConfig()
config.sample_rate = 48000
engine.initialize(config)
engine.set_input_device(1)
engine.set_output_device(2)
engine.start()
# ... record ...
data = engine.get_recorded_data()
engine.stop()
```

---

### 3.3 DatasetCollector (Automated Collection)

**File**: `DatasetCollector.py`

#### Purpose
Orchestrates automated dataset collection with robust error handling, quality assessment, and metadata persistence.

#### Core Classes

**1. ScenarioConfig** (dataclass)

```python
@dataclass
class ScenarioConfig:
    scenario_number: str           # e.g., "1", "0.5", "7a"
    description: str               # Human-readable description
    computer_name: str             # System identifier
    room_name: str                 # Location identifier
    num_measurements: int = 30     # Samples to collect
    measurement_interval: float = 2.0  # Seconds between measurements
    warm_up_measurements: int = 0  # Discarded initial measurements
    additional_metadata: Dict[str, Any] = field(default_factory=dict)
```

**2. SingleScenarioCollector**

```python
class SingleScenarioCollector:
    def __init__(self, scenario_config, recorder, base_output_dir,
                 folder_behavior="append", config_mismatch_policy="warn")

    def collect_scenario() -> bool

    # Private methods
    def _create_scenario_folder() -> Path
    def _handle_existing_folder() -> str  # append/overwrite/abort
    def _save_session_metadata()
    def _collect_single_measurement(index) -> bool
    def _assess_quality(audio_data) -> Dict[str, float]
    def _check_sentinel_files() -> str  # "continue"/"pause"/"stop"
```

#### Collection Workflow

```python
# 1. Folder Management
scenario_folder = f"{computer}-Scenario{num}-{room}"
# Example: "Laptop-Scenario1.5-LivingRoom"

# Folder structure:
scenario_folder/
├── raw_recordings/          # Original microphone captures
├── impulse_responses/       # Extracted impulse responses
├── room_responses/          # Averaged room responses
├── metadata/
│   ├── session_metadata.json
│   ├── measurement_000_meta.json
│   └── ...
└── <scenario>_SUMMARY.txt

# 2. Measurement Loop
for index in range(num_measurements):
    # Check for pause/stop sentinel files
    status = check_sentinel_files()
    if status == "pause": wait_for_resume()
    if status == "stop": break

    # Record measurement
    raw_file = f"raw_recordings/recording_{index:03d}.wav"
    impulse_file = f"impulse_responses/impulse_{index:03d}.wav"
    audio = recorder.take_record(raw_file, impulse_file)

    # Quality assessment
    quality_metrics = assess_quality(audio)
    # - SNR (signal-to-noise ratio)
    # - Clipping percentage
    # - Dynamic range

    # Save metadata
    save_measurement_metadata(index, quality_metrics)

    # Inter-measurement delay
    time.sleep(measurement_interval)

# 3. Session Summary
save_session_metadata()
generate_summary_report()
```

#### Quality Assessment Metrics

```python
def _assess_quality(audio_data: np.ndarray) -> Dict[str, float]:
    # 1. Signal-to-Noise Ratio (SNR)
    signal_power = np.mean(audio_data ** 2)
    noise_region = audio_data[-1000:]  # Last 1000 samples
    noise_power = np.mean(noise_region ** 2)
    snr_db = 10 * np.log10(signal_power / noise_power)

    # 2. Clipping Detection
    max_abs = np.max(np.abs(audio_data))
    clipping_threshold = 0.99
    clipping_percentage = 100 * np.sum(np.abs(audio_data) > clipping_threshold) / len(audio_data)

    # 3. Dynamic Range
    rms = np.sqrt(np.mean(audio_data ** 2))
    peak = np.max(np.abs(audio_data))
    dynamic_range_db = 20 * np.log10(peak / rms) if rms > 0 else 0

    return {
        "snr_db": snr_db,
        "clipping_percentage": clipping_percentage,
        "dynamic_range_db": dynamic_range_db,
        "rms_level": rms,
        "peak_amplitude": peak
    }
```

#### Pause/Resume Mechanism

```python
# User drops sentinel files in scenario folder
# PAUSE file → pauses collection, waits for resume
# STOP file → gracefully stops collection

def _check_sentinel_files() -> str:
    pause_file = scenario_folder / "PAUSE"
    stop_file = scenario_folder / "STOP"

    if stop_file.exists():
        stop_file.unlink()
        return "stop"

    if pause_file.exists():
        # Wait until PAUSE file is removed
        while pause_file.exists():
            time.sleep(1)
        return "paused"

    return "continue"
```

---

### 3.4 ScenarioManager (Dataset Management)

**File**: `ScenarioManager.py`

#### Purpose
Centralized management of scenario datasets with parsing, filtering, caching, and metadata operations.

#### Core Capabilities

1. **Scenario Parsing**
   - Parse folder names: `<computer>-Scenario<num>-<room>`
   - Handle variants: `Scenario1`, `Scenario0.5`, `Scenario7a`
   - Extract computer, room, scenario number

2. **Feature Detection**
   - Check for `spectrum.csv`, `features.csv`, audio files
   - Count samples matching FeatureExtractor logic

3. **Metadata Management**
   - Read/write labels to `scenario_meta.json`
   - Read descriptions from session metadata
   - Bulk label operations

4. **Dataset Analysis**
   - Build enriched pandas DataFrame of all scenarios
   - Cache results in Streamlit session state
   - Integrate with ScenarioSelector if available

5. **Filtering & Sorting**
   - Regex filtering by scenario number, computer, room
   - Smart numeric sorting (1 < 1.5 < 2 < 2a < 3)

#### Key Methods

```python
class ScenarioManager:
    # File utilities
    @staticmethod
    def list_wavs(folder: str) -> List[str]
    def check_features_available(path: str) -> Dict[str, bool]
    def count_feature_samples(scenario_path, wav_subfolder, recording_type) -> int

    # Metadata operations
    @staticmethod
    def read_label(path: str) -> Optional[str]
    def read_description(path: str) -> Optional[str]
    def write_label(path: str, label: Optional[str]) -> bool

    # Scenario parsing
    @classmethod
    def parse_scenario_folder_name(folder_name: str) -> Tuple[str, str, str]
    # Returns: (number_str, computer, room)

    # Dataset analysis
    @classmethod
    def build_scenarios_df(root: str, force_refresh=False) -> pd.DataFrame
    def analyze_dataset_filesystem(root: str) -> pd.DataFrame

    # Filtering & sorting
    @staticmethod
    def apply_filters(df, text, computer, room) -> pd.DataFrame
    def sort_scenarios_df(df: pd.DataFrame) -> pd.DataFrame

    # Bulk operations
    @classmethod
    def bulk_apply_label(df: pd.DataFrame, label: Optional[str]) -> int

    # Utilities
    @staticmethod
    def validate_dataset_root(root: str) -> Tuple[bool, str]
    def get_unique_labels(df: pd.DataFrame) -> Set[str]
    def format_features_status(features_dict) -> str

    # Caching
    @classmethod
    def clear_cache()
```

#### Scenario Folder Parsing

```python
# Regex patterns for structured matching
_NAME_FULL_RE = re.compile(
    r'^(?P<computer>.+?)-Scenario(?P<num>[A-Za-z0-9]+(?:\.[A-Za-z0-9]+)*)-(?P<room>.+)$',
    re.IGNORECASE
)

# Examples:
"Laptop-Scenario1-LivingRoom"    → ("1", "Laptop", "LivingRoom")
"PC-Scenario0.5-Studio"          → ("0.5", "PC", "Studio")
"Desktop-Scenario7a-Office"      → ("7a", "Desktop", "Office")
"Workstation-Scenario1.2.3-Lab"  → ("1.2.3", "Workstation", "Lab")

# Fallback regex for partial matches
_NAME_NUM_RE = re.compile(r'(?i)-Scenario(?P<num>[A-Za-z0-9]+(?:\.[A-Za-z0-9]+)*)-')
```

#### DataFrame Schema

```python
# Columns in scenarios DataFrame
{
    "scenario": str,           # Folder name
    "path": str,              # Absolute path
    "sample_count": int,      # Number of audio files
    "features_available": {   # Feature file status
        "spectrum": bool,
        "mfcc": bool,
        "audio": bool
    },
    "label": Optional[str],   # User-assigned label(s)
    "description": Optional[str],
    "number_str": str,        # Parsed scenario number
    "computer": str,          # Parsed computer name
    "room": str              # Parsed room name
}
```

#### Smart Filtering

```python
# Auto-mode: tries scenario number first, then falls back to name/path
apply_filters(df, text="1.5", computer="", room="")
# → Matches scenarios with number_str == "1.5"

apply_filters(df, text="^7\\.", computer="", room="")
# → Matches scenarios starting with "7." (7.1, 7.2, 7a, etc.)

apply_filters(df, text="name:Studio", computer="", room="")
# → Matches scenarios with "Studio" in name

apply_filters(df, text="", computer="laptop", room="living")
# → Matches scenarios from "laptop" computer in "living" room
```

---

### 3.5 FeatureExtractor (Signal Analysis)

**File**: `FeatureExtractor.py`

#### Purpose
Extract acoustic features (MFCC, frequency spectrum) from audio recordings for machine learning classification.

#### Supported Feature Types

**1. MFCC (Mel-Frequency Cepstral Coefficients)**
- **Count**: 13 coefficients by default (configurable)
- **Use Case**: Speech recognition, acoustic classification
- **Column Names**: `mfcc_0`, `mfcc_1`, ..., `mfcc_12`

**2. Frequency Spectrum**
- **Count**: Variable (depends on audio length and FFT size)
- **Use Case**: Spectral analysis, resonance detection
- **Column Names**: `freq_0`, `freq_1`, ..., `freq_N`

#### Core Methods

```python
class AudioFeatureExtractor:
    def __init__(self, sample_rate=16000, n_mfcc=13, config_filename=None,
                 max_spectrum_freq=None)

    # Batch processing (training data preparation)
    def process_dataset(dataset_path, wav_subfolder, recording_type, ...)
    def process_scenario_folder(scenario_folder, ...) -> bool
    def find_wav_files(folder_path, recording_type) -> List[str]

    # Single-sample extraction (inference)
    def build_feature_vector_from_audio(audio, feature_type, feature_names) -> np.ndarray
    def build_feature_vector_from_wav(file_path, feature_type, feature_names) -> np.ndarray

    # Internals
    def _extract_mfcc_from_audio(audio: np.ndarray) -> np.ndarray
    def _extract_spectrum_from_audio(audio: np.ndarray) -> np.ndarray
    def _adapt_parameters_for_audio_length(audio_length) -> Tuple[int, int]
```

#### MFCC Extraction Algorithm

```python
def _extract_mfcc_from_audio(audio: np.ndarray) -> np.ndarray:
    # 1. Adapt FFT parameters to audio length
    n_fft, hop_length = _adapt_parameters_for_audio_length(len(audio))

    # 2. Compute MFCC via librosa
    mfcc = librosa.feature.mfcc(
        y=audio.astype(float),
        sr=self.sample_rate,
        n_mfcc=self.n_mfcc,  # 13 by default
        n_fft=n_fft,
        hop_length=hop_length
    )
    # Shape: (n_mfcc, time_frames)

    # 3. Filter valid frames (non-NaN, non-zero)
    valid_mask = ~(np.isnan(mfcc[0]) | (np.sum(np.abs(mfcc), axis=0) == 0))

    # 4. Average across time to get single feature vector
    mfcc_vector = np.mean(mfcc[:, valid_mask], axis=1)
    # Shape: (n_mfcc,)

    return mfcc_vector.astype(float)
```

#### Spectrum Extraction Algorithm

```python
def _extract_spectrum_from_audio(audio: np.ndarray) -> np.ndarray:
    # 1. Compute FFT
    fft_result = np.fft.fft(audio)

    # 2. Take positive half (real signal symmetry)
    positive_half = fft_result[:len(audio) // 2 + 1]

    # 3. Magnitude spectrum
    magnitude = np.abs(positive_half)

    # 4. Normalize to [0, 1]
    if np.max(magnitude) > 0:
        magnitude = magnitude / np.max(magnitude)

    return magnitude.astype(float)
```

#### Adaptive FFT Parameters

```python
def _adapt_parameters_for_audio_length(audio_length: int) -> Tuple[int, int]:
    """
    Adapt FFT size and hop length to audio duration.
    Prevents librosa errors on very short audio.
    """
    if audio_length < 512:
        # Very short audio (< 32ms at 16kHz)
        n_fft = max(64, 2 ** int(np.log2(max(4, audio_length // 2))))
        hop_length = max(16, n_fft // 4)
    elif audio_length < 2048:
        # Short audio (< 128ms at 16kHz)
        n_fft = 512
        hop_length = 128
    else:
        # Standard audio
        n_fft = 2048
        hop_length = 512

    # Ensure n_fft doesn't exceed audio length
    n_fft = min(n_fft, max(4, audio_length))
    hop_length = min(hop_length, max(1, audio_length // 4))

    return n_fft, hop_length
```

#### Feature Alignment for Inference

```python
# Problem: Training and inference must use same feature set
# Solution: Explicitly align features to trained model's feature_names

def build_feature_vector_from_audio(
    audio: np.ndarray,
    feature_type: str,
    feature_names: List[str]  # From trained model
) -> np.ndarray:
    """
    Extract features and align to expected feature_names order.
    Handles missing indices with 0.0 padding.
    """
    if feature_type == "spectrum":
        # Compute full spectrum
        magnitude = _extract_spectrum_from_audio(audio)

        # Align to feature_names (e.g., ["freq_0", "freq_1", ...])
        aligned_vector = []
        for name in feature_names:
            index = _suffix_int(name)  # Extract integer from "freq_42"
            if index is None or index >= len(magnitude):
                aligned_vector.append(0.0)  # Pad missing bins
            else:
                aligned_vector.append(float(magnitude[index]))

        return np.array(aligned_vector, dtype=float)

    elif feature_type == "mfcc":
        # Compute MFCC
        mfcc_vector = _extract_mfcc_from_audio(audio)

        # Map to feature_names (e.g., ["mfcc_0", "mfcc_1", ...])
        vec_map = {f"mfcc_{i}": float(mfcc_vector[i])
                   for i in range(len(mfcc_vector))}

        return np.array([vec_map.get(name, 0.0) for name in feature_names],
                        dtype=float)
```

#### Batch Processing Output

```python
# For each scenario folder, creates two CSV files:

# 1. features.csv (MFCC)
# filename,mfcc_0,mfcc_1,mfcc_2,...,mfcc_12
# impulse_000.wav,12.34,-5.67,8.90,...,1.23
# impulse_001.wav,11.22,-4.55,7.88,...,2.34
# ...

# 2. spectrum.csv (Frequency Spectrum)
# filename,freq_0,freq_1,freq_2,...,freq_N
# impulse_000.wav,0.123,0.456,0.789,...,0.012
# impulse_001.wav,0.234,0.567,0.890,...,0.023
# ...
```

---

### 3.6 ScenarioClassifier (Machine Learning)

**File**: `ScenarioClassifier.py`

#### Purpose
Train, evaluate, and persist machine learning models for acoustic scenario classification.

#### Supported Models

**1. SVM (Support Vector Machine)**
- **Algorithm**: `sklearn.svm.SVC` with RBF kernel
- **Pros**: Non-linear boundaries, robust to outliers
- **Cons**: Slower on large datasets
- **Parameters**: `C=1.0`, `gamma='scale'`, `probability=True`

**2. Logistic Regression**
- **Algorithm**: `sklearn.linear_model.LogisticRegression`
- **Pros**: Fast, interpretable, probabilistic
- **Cons**: Assumes linear separability
- **Parameters**: `max_iter=5000`

#### Training Modes

**1. Single Pair** (`run_single_pair`)
- Train binary classifier on two scenarios
- Example: "Quiet Room" vs "Noisy Room"

**2. Group vs Group** (`run_group_vs_group`)
- Train on grouped scenarios (balanced subsampling)
- Example: ["Room1", "Room2"] vs ["Room3", "Room4"]

**3. All Pairs** (`run_all_pairs`)
- Train on all N(N-1)/2 pairs of scenarios
- Returns accuracy matrix for comparison

#### Core Methods

```python
class ScenarioClassifier:
    def __init__(self, model_type="svm", feature_type="mfcc")

    # High-level training APIs
    def run_single_pair(path_a, path_b, label_a, label_b, params, dataset_root) -> Dict
    def run_group_vs_group(scenarios_a, scenarios_b, label_a, label_b, params, dataset_root) -> Dict
    def run_all_pairs(scenarios, params, dataset_root) -> Dict

    # Model persistence
    def save_model(path=None, dataset_root=None) -> str
    def download_model() -> Tuple[str, bytes]
    @staticmethod
    def load_model(path=None, file_bytes=None) -> "ScenarioClassifier"

    # Inference
    def predict_from_features(X: np.ndarray) -> Dict[str, Any]
    def predict_from_audio(audio, extractor, feature_names) -> Dict[str, Any]

    # State queries
    def is_trained() -> bool
    def get_model_info() -> Dict[str, Any]
```

#### Training Pipeline

```python
def _train_eval_enhanced(X: np.ndarray, y: np.ndarray, params: Dict) -> Dict:
    # 1. Train/test split (stratified)
    test_size = params.get("test_size", 0.2)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=42
    )

    # 2. Feature scaling (StandardScaler)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 3. Train model
    if model_type == "svm":
        model = SVC(kernel="rbf", probability=True, gamma="scale", C=1.0)
    else:
        model = LogisticRegression(max_iter=5000)

    model.fit(X_train_scaled, y_train)

    # 4. Evaluate
    y_pred_train = model.predict(X_train_scaled)
    y_pred_test = model.predict(X_test_scaled)

    train_accuracy = accuracy_score(y_train, y_pred_train)
    test_accuracy = accuracy_score(y_test, y_pred_test)

    # 5. Cross-validation (5-fold by default)
    cv_folds = params.get("cv_folds", 5)
    cv_scores = cross_val_score(model, X_train_scaled, y_train,
                                 cv=cv_folds, scoring="accuracy")

    # 6. Metrics
    confusion_mat = confusion_matrix(y_test, y_pred_test)
    classification_rep = classification_report(y_test, y_pred_test)

    return {
        "train_accuracy": train_accuracy,
        "test_accuracy": test_accuracy,
        "cv_mean": np.mean(cv_scores),
        "cv_std": np.std(cv_scores),
        "cv_scores": cv_scores,
        "confusion_matrix": confusion_mat,
        "classification_report": classification_rep,
        "X_train": X_train_scaled,
        "X_test": X_test_scaled,
        "y_train": y_train,
        "y_test": y_test
    }
```

#### Model Persistence

```python
# Save model with complete metadata
def save_model(path: Optional[str] = None, dataset_root: Optional[str] = None) -> str:
    # Package everything needed for inference
    package = {
        "model_type": self.model_type,
        "feature_type": self.feature_type,
        "model": self.model,                # Trained sklearn model
        "scaler": self.scaler,              # StandardScaler instance
        "label_encoder": self.label_encoder, # LabelEncoder instance
        "feature_names": self.feature_names, # Column order for alignment
        "meta": {
            "mode": "single_pair",
            "dataset_root": dataset_root,
            "scenarios": ["Scenario1", "Scenario2"],
            "labels": ["quiet", "noisy"],
            "params": {"test_size": 0.2, "cv_folds": 5},
            "trained_at": time.time(),
            "metrics": {
                "train_accuracy": 0.95,
                "test_accuracy": 0.92,
                "cv_mean": 0.91,
                "cv_std": 0.03
            }
        }
    }

    # Serialize with joblib (compression level 3)
    joblib.dump(package, path, compress=3)
    return path
```

#### Inference Workflow

```python
# 1. Load trained model
clf = ScenarioClassifier.load_model("trained_model.joblib")

# 2. Extract features from new audio
extractor = AudioFeatureExtractor(sample_rate=48000, n_mfcc=13)
audio_data = librosa.load("new_recording.wav", sr=48000)[0]

# 3. Align features to trained model's feature_names
feature_vector = extractor.build_feature_vector_from_audio(
    audio=audio_data,
    feature_type=clf.feature_type,        # "mfcc" or "spectrum"
    feature_names=clf.feature_names       # Column order from training
)

# 4. Predict
result = clf.predict_from_features(feature_vector.reshape(1, -1))
# Returns: {
#     "label": "quiet",
#     "proba": {"quiet": 0.87, "noisy": 0.13}
# }
```

---

## 4. Data Flow Architecture

### 4.1 Collection Data Flow

```
┌────────────────────────────────────────────────────────────────┐
│                        USER INITIATES                          │
│  GUI: Collect Panel → Single or Series Mode                   │
└────────────────────────────────────────────────────────────────┘
                               ↓
┌────────────────────────────────────────────────────────────────┐
│                   CONFIGURATION LOADING                        │
│  1. Load recorderConfig.json (sample rate, pulse params)      │
│  2. Select audio devices (input/output)                       │
│  3. Define scenario metadata (number, description, room)      │
└────────────────────────────────────────────────────────────────┘
                               ↓
┌────────────────────────────────────────────────────────────────┐
│                    SIGNAL GENERATION                           │
│  RoomResponseRecorder._generate_complete_signal()             │
│  → Pulse train: 8 pulses × 100ms cycle = 800ms total         │
│  → Apply fade envelopes to prevent clicks                     │
│  → Convert to float32 list for SDL core                       │
└────────────────────────────────────────────────────────────────┘
                               ↓
┌────────────────────────────────────────────────────────────────┐
│              SIMULTANEOUS PLAYBACK + RECORDING                 │
│  sdl_audio_core.measure_room_response_auto()                  │
│  → Initialize SDL audio engine                                │
│  → Set input/output devices                                   │
│  → Start recording stream                                     │
│  → Play generated signal through speakers                     │
│  → Capture microphone input simultaneously                    │
│  → Return recorded samples as float32 array                   │
└────────────────────────────────────────────────────────────────┘
                               ↓
┌────────────────────────────────────────────────────────────────┐
│                    SIGNAL PROCESSING                           │
│  RoomResponseRecorder._process_recorded_signal()              │
│  1. Reshape into cycles (num_pulses × cycle_samples)         │
│  2. Skip first cycle for system settling                      │
│  3. Average cycles → room_response                            │
│  4. Find onset via RMS threshold                              │
│  5. Circular shift to align onset → impulse_response          │
└────────────────────────────────────────────────────────────────┘
                               ↓
┌────────────────────────────────────────────────────────────────┐
│                     QUALITY ASSESSMENT                         │
│  DatasetCollector._assess_quality()                           │
│  → SNR calculation                                             │
│  → Clipping percentage                                         │
│  → Dynamic range analysis                                      │
│  → RMS level check                                             │
└────────────────────────────────────────────────────────────────┘
                               ↓
┌────────────────────────────────────────────────────────────────┐
│                      FILE PERSISTENCE                          │
│  1. Save raw recording → raw_recordings/recording_NNN.wav     │
│  2. Save impulse response → impulse_responses/impulse_NNN.wav │
│  3. Save room response → room_responses/room_NNN.wav          │
│  4. Write metadata → metadata/measurement_NNN_meta.json       │
└────────────────────────────────────────────────────────────────┘
                               ↓
┌────────────────────────────────────────────────────────────────┐
│                     ITERATION CONTROL                          │
│  → Check sentinel files (PAUSE, STOP)                         │
│  → Inter-measurement delay                                     │
│  → Repeat for num_measurements                                │
└────────────────────────────────────────────────────────────────┘
                               ↓
┌────────────────────────────────────────────────────────────────┐
│                    SESSION FINALIZATION                        │
│  1. Write session_metadata.json (scenario config, timestamps) │
│  2. Generate summary report (sample count, quality stats)     │
│  3. Invalidate ScenarioManager cache                          │
└────────────────────────────────────────────────────────────────┘
```

### 4.2 Processing Data Flow (Feature Extraction)

```
┌────────────────────────────────────────────────────────────────┐
│                        USER INITIATES                          │
│  GUI: Process Panel → Select scenarios + feature type         │
└────────────────────────────────────────────────────────────────┘
                               ↓
┌────────────────────────────────────────────────────────────────┐
│                   SCENARIO ITERATION                           │
│  For each selected scenario folder:                           │
│  1. Locate wav_subfolder (impulse_responses by default)       │
│  2. Filter by recording_type (average/raw/any)                │
│  3. Load recorderConfig.json if present (override sample_rate)│
└────────────────────────────────────────────────────────────────┘
                               ↓
┌────────────────────────────────────────────────────────────────┐
│                      AUDIO FILE LOADING                        │
│  For each WAV file in subfolder:                              │
│  → librosa.load(file_path, sr=sample_rate, mono=True)        │
│  → Returns audio as float32 numpy array                       │
└────────────────────────────────────────────────────────────────┘
                               ↓
         ┌────────────────────┴─────────────────────┐
         ▼                                           ▼
┌──────────────────────┐                  ┌──────────────────────┐
│  MFCC EXTRACTION     │                  │ SPECTRUM EXTRACTION  │
│  (feature_type=mfcc) │                  │ (feature_type=spec)  │
└──────────────────────┘                  └──────────────────────┘
         │                                           │
         │ librosa.feature.mfcc()                   │ np.fft.fft()
         │ → (n_mfcc, time_frames)                  │ → complex array
         │                                           │
         │ Average across time                       │ Magnitude (abs)
         │ → (n_mfcc,) vector                        │ → positive half
         │                                           │ → normalized
         │                                           │
         │ Column names:                             │ Column names:
         │ mfcc_0, mfcc_1, ..., mfcc_12             │ freq_0, freq_1, ...
         │                                           │
         └────────────────────┬─────────────────────┘
                               ↓
┌────────────────────────────────────────────────────────────────┐
│                      CSV GENERATION                            │
│  Accumulate feature rows:                                     │
│  [{filename: "impulse_000.wav", mfcc_0: 12.3, ...}, ...]     │
│                                                                │
│  Convert to pandas DataFrame                                  │
│  Save to scenario_folder/features.csv (MFCC)                 │
│  Save to scenario_folder/spectrum.csv (Spectrum)             │
└────────────────────────────────────────────────────────────────┘
                               ↓
┌────────────────────────────────────────────────────────────────┐
│                     METADATA WRITING                           │
│  Write features_meta.json:                                    │
│  {                                                             │
│    "sample_rate": 48000,                                      │
│    "fft_len": null,  // varies per file                      │
│    "bin_hz": null    // varies per file                      │
│  }                                                             │
└────────────────────────────────────────────────────────────────┘
                               ↓
┌────────────────────────────────────────────────────────────────┐
│                    CACHE INVALIDATION                          │
│  ScenarioManager.clear_cache()                                │
│  → Forces GUI to reload scenario list with updated features   │
└────────────────────────────────────────────────────────────────┘
```

### 4.3 Classification Data Flow

```
┌────────────────────────────────────────────────────────────────┐
│                        USER INITIATES                          │
│  GUI: Classify Panel → Select mode, scenarios, parameters     │
│  Modes: single_pair | group_vs_group | all_pairs              │
└────────────────────────────────────────────────────────────────┘
                               ↓
┌────────────────────────────────────────────────────────────────┐
│                     DATA PREPARATION                           │
│  ScenarioClassifier._prepare_pair() or _prepare_groups()     │
│                                                                │
│  1. Load feature CSVs from scenario folders                   │
│     - features.csv (MFCC) or spectrum.csv (Spectrum)         │
│                                                                │
│  2. Extract feature columns                                   │
│     - MFCC: mfcc_0, mfcc_1, ..., mfcc_12                     │
│     - Spectrum: freq_0, freq_1, ..., freq_N                  │
│                                                                │
│  3. Align columns (sort by numeric suffix)                    │
│     - Ensure consistent feature order across scenarios        │
│                                                                │
│  4. Create label arrays                                       │
│     - LabelEncoder: string labels → integers                  │
│                                                                │
│  5. Concatenate into X (features) and y (labels)              │
│     - X shape: (num_samples, num_features)                    │
│     - y shape: (num_samples,)                                 │
└────────────────────────────────────────────────────────────────┘
                               ↓
┌────────────────────────────────────────────────────────────────┐
│                    TRAIN/TEST SPLIT                            │
│  train_test_split(X, y, test_size=0.2, stratify=y)           │
│  → Ensures balanced class distribution in both sets           │
│  → Random state=42 for reproducibility                        │
└────────────────────────────────────────────────────────────────┘
                               ↓
┌────────────────────────────────────────────────────────────────┐
│                   FEATURE SCALING                              │
│  scaler = StandardScaler()                                    │
│  X_train_scaled = scaler.fit_transform(X_train)              │
│  X_test_scaled = scaler.transform(X_test)                    │
│                                                                │
│  → Zero mean, unit variance normalization                     │
│  → Critical for SVM and logistic regression                   │
└────────────────────────────────────────────────────────────────┘
                               ↓
         ┌────────────────────┴─────────────────────┐
         ▼                                           ▼
┌──────────────────────┐                  ┌──────────────────────┐
│   SVM TRAINING       │                  │ LOGISTIC REGRESSION  │
│                      │                  │      TRAINING        │
└──────────────────────┘                  └──────────────────────┘
         │                                           │
         │ SVC(kernel="rbf",                         │ LogisticRegression(
         │     probability=True,                     │     max_iter=5000)
         │     gamma="scale",                        │
         │     C=1.0)                                │
         │                                           │
         └────────────────────┬─────────────────────┘
                               ↓
┌────────────────────────────────────────────────────────────────┐
│                     MODEL TRAINING                             │
│  model.fit(X_train_scaled, y_train)                           │
│  → Fit model to training data                                 │
└────────────────────────────────────────────────────────────────┘
                               ↓
┌────────────────────────────────────────────────────────────────┐
│                    EVALUATION METRICS                          │
│  1. Training Accuracy                                         │
│     y_pred_train = model.predict(X_train_scaled)             │
│     train_acc = accuracy_score(y_train, y_pred_train)        │
│                                                                │
│  2. Test Accuracy                                             │
│     y_pred_test = model.predict(X_test_scaled)               │
│     test_acc = accuracy_score(y_test, y_pred_test)           │
│                                                                │
│  3. Cross-Validation (5-fold default)                         │
│     cv_scores = cross_val_score(model, X, y, cv=5)           │
│     cv_mean = mean(cv_scores)                                 │
│     cv_std = std(cv_scores)                                   │
│                                                                │
│  4. Confusion Matrix                                          │
│     confusion_matrix(y_test, y_pred_test)                    │
│                                                                │
│  5. Classification Report                                     │
│     precision, recall, F1-score per class                     │
└────────────────────────────────────────────────────────────────┘
                               ↓
┌────────────────────────────────────────────────────────────────┐
│                  FEATURE IMPORTANCE                            │
│  If model supports:                                           │
│  - SVM: Derive from support vector coefficients               │
│  - Logistic: Use model.coef_ (weight magnitudes)             │
│                                                                │
│  Returns array matching feature_names order                   │
└────────────────────────────────────────────────────────────────┘
                               ↓
┌────────────────────────────────────────────────────────────────┐
│                    MODEL PERSISTENCE                           │
│  Package for serialization:                                   │
│  {                                                             │
│    "model": trained sklearn model,                            │
│    "scaler": StandardScaler instance,                         │
│    "label_encoder": LabelEncoder instance,                    │
│    "feature_names": column order list,                        │
│    "meta": {                                                   │
│      "model_type": "svm" or "logistic",                       │
│      "feature_type": "mfcc" or "spectrum",                    │
│      "mode": "single_pair" etc.,                              │
│      "dataset_root": path,                                    │
│      "scenarios": [list of scenario names],                   │
│      "labels": [list of label strings],                       │
│      "params": {test_size, cv_folds, ...},                    │
│      "trained_at": timestamp,                                 │
│      "metrics": {train_acc, test_acc, cv_mean, ...}          │
│    }                                                           │
│  }                                                             │
│                                                                │
│  Serialize: joblib.dump(package, path, compress=3)           │
└────────────────────────────────────────────────────────────────┘
                               ↓
┌────────────────────────────────────────────────────────────────┐
│                  SESSION STATE STORAGE                         │
│  Store in Streamlit session_state:                            │
│  - SK_CLASSIFIER_OBJ: ScenarioClassifier instance             │
│  - SK_CLASSIFICATION_RESULTS: metrics dict                    │
│  - SK_LAST_MODEL_INFO: ModelInfo dataclass                    │
│  - SK_CLASSIFICATION_ARTIFACTS: predictions, importance, etc. │
│                                                                │
│  → Persists across GUI interactions                           │
│  → Enables Visualize panel to display results                 │
└────────────────────────────────────────────────────────────────┘
```

---

## 5. Technology Stack

### 5.1 Core Dependencies

| Category | Library | Version | Purpose |
|----------|---------|---------|---------|
| **Audio Processing** | librosa | 0.11.0 | MFCC extraction, audio loading |
| | sounddevice | 0.5.2 | Python audio I/O (alternative) |
| | soundfile | 0.13.1 | WAV file I/O utility |
| | soxr | 0.5.0 | High-quality audio resampling |
| **Signal Processing** | numpy | 2.2.6 | Array operations, FFT |
| | scipy | 1.16.1 | Signal processing, convolution |
| **Machine Learning** | scikit-learn | 1.7.1 | SVM, LogisticRegression, metrics |
| | joblib | 1.5.1 | Model serialization |
| | numba | 0.61.2 | JIT compilation for librosa |
| **Data Analysis** | pandas | 2.3.1 | DataFrame operations, CSV I/O |
| **Visualization** | matplotlib | 3.10.5 | Static plots, fallback charts |
| | seaborn | 0.13.2 | Statistical visualization |
| | plotly | (via streamlit) | Interactive charts, heatmaps |
| | altair | 5.5.0 | Declarative visualizations |
| **GUI Framework** | streamlit | 1.48.0 | Web app framework, session state |
| **C++ Bindings** | pybind11 | 3.0.0 | Python/C++ interface |
| **Utilities** | tqdm | 4.67.1 | Progress bars |
| | pyarrow | 21.0.0 | Efficient data serialization |

### 5.2 C++ SDL2 Audio Engine

**SDL Audio Core Module**:
- **Language**: C++17
- **Audio Library**: SDL2 (Simple DirectMedia Layer)
- **Binding Tool**: pybind11 3.0.0
- **Compiler**: MSVC (Windows), GCC/Clang (Linux/macOS)

**Build Dependencies**:
- SDL2 development libraries (headers + .lib/.so)
- CMake or setuptools for build orchestration
- Python development headers (Python.h)

**Compiled Output**:
- Windows: `sdl_audio_core.cp312-win_amd64.pyd`
- Linux: `sdl_audio_core.cpython-312-x86_64-linux-gnu.so`
- macOS: `sdl_audio_core.cpython-312-darwin.so`

### 5.3 Python Environment

**Minimum Requirements**:
- Python 3.12+ (for dataclass improvements, type hints)
- pip 24.0+
- virtualenv or conda recommended

**Installation**:
```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
.venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt

# Build SDL audio core (if source available)
cd sdl_audio_core
python setup.py build_ext --inplace
```

---

## 6. File Organization

### 6.1 Repository Structure

```
RoomResponse/
├── CORE AUDIO ENGINE
│   ├── RoomResponseRecorder.py          # Python audio interface
│   ├── MicTesting.py                    # Audio recording workers
│   └── sdl_audio_core/                  # C++ module
│       ├── src/                         # C++ source
│       ├── setup.py                     # Build script
│       └── *.pyd / *.so                 # Compiled binary
│
├── DATA COLLECTION
│   ├── DatasetCollector.py              # Single scenario collector
│   ├── gui_collect_panel.py             # Collection GUI panel
│   ├── gui_series_worker.py             # Background series worker
│   ├── gui_single_pulse_recorder.py     # Single measurement UI
│   └── collect_dataset.py               # CLI wrapper
│
├── DATASET MANAGEMENT
│   ├── ScenarioManager.py               # Scenario parsing, filtering
│   ├── ScenarioSelector.py              # Dataset analysis helper
│   └── scenario_explorer.py             # Inspection utilities
│
├── SIGNAL PROCESSING
│   ├── signal_alignment.py              # Cross-correlation alignment
│   └── FirFilterFileIO.py               # FIR filter serialization
│
├── FEATURE EXTRACTION
│   ├── FeatureExtractor.py              # MFCC and spectrum extraction
│   ├── FeatureAnalyser.py               # Feature analysis utilities
│   └── gui_process_panel.py             # Processing GUI panel
│
├── MACHINE LEARNING
│   ├── ScenarioClassifier.py            # Train/eval/persist models
│   └── gui_classify_panel.py            # Classification GUI panel
│
├── VISUALIZATION
│   ├── gui_visualize_panel.py           # Results visualization panel
│   └── gui_audio_visualizer.py          # Waveform plotting
│
├── GUI APPLICATION
│   ├── gui_launcher.py                  # Main Streamlit app (full)
│   ├── piano_response.py                # Simplified app (audio only)
│   ├── gui_scenarios_panel.py           # Scenario browser panel
│   ├── gui_audio_settings_panel.py      # System settings panel
│   ├── gui_audio_device_selector.py     # Device picker with monitor
│   └── gui_series_settings_panel.py     # Series config panel
│
├── CONFIGURATION & DATA
│   ├── recorderConfig.json              # Signal generation params
│   ├── requirements.txt                 # Python dependencies
│   └── room_response_dataset/           # Output data root (default)
│       └── <computer>-Scenario<N>-<room>/
│           ├── raw_recordings/
│           ├── impulse_responses/
│           ├── room_responses/
│           ├── metadata/
│           │   ├── session_metadata.json
│           │   └── measurement_*_meta.json
│           ├── features.csv             # MFCC features
│           ├── spectrum.csv             # Frequency spectrum
│           └── scenario_meta.json       # Labels, descriptions
│
├── UTILITIES & TESTING
│   ├── test_audio.py                    # Audio system tests
│   ├── test_data_generator.py           # Synthetic data generation
│   ├── debug_piano_scenarios.py         # Debugging utilities
│   ├── detect_paths.py                  # SDL2 path detection
│   └── copyFolder.py                    # Folder copy utility
│
└── DOCUMENTATION
    ├── Readme.md                        # Quick start guide
    ├── RoomResponse_GUI_Tech_Requirements.md  # Requirements doc
    └── TECHNICAL_DOCUMENTATION.md       # This file
```

### 6.2 Module Dependency Graph

```
┌─────────────────────────────────────────────────────────────────┐
│                    gui_launcher.py (Main App)                   │
└─────────────────────────────────────────────────────────────────┘
                               │
            ┌──────────────────┼──────────────────┐
            ▼                  ▼                  ▼
   ┌──────────────┐   ┌──────────────┐   ┌──────────────┐
   │ Scenarios    │   │  Collect     │   │  Process     │
   │   Panel      │   │   Panel      │   │   Panel      │
   └──────────────┘   └──────────────┘   └──────────────┘
            │                  │                  │
            │         ┌────────┴────────┐         │
            │         ▼                 ▼         ▼
            │  ┌──────────────┐  ┌──────────────┐
            │  │ DatasetColl. │  │FeatureExtrac.│
            │  └──────────────┘  └──────────────┘
            │         │                  │
            │         ▼                  │
            │  ┌──────────────┐          │
            │  │RoomResponse  │          │
            │  │  Recorder    │          │
            │  └──────────────┘          │
            │         │                  │
            │         ▼                  │
            │  ┌──────────────┐          │
            │  │sdl_audio_core│          │
            │  └──────────────┘          │
            │                            │
            └────────────┬───────────────┘
                         ▼
                ┌──────────────┐
                │ Scenario     │
                │  Manager     │
                └──────────────┘
                         │
            ┌────────────┼────────────┐
            ▼            ▼            ▼
   ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
   │  Classify    │ │  Visualize   │ │  Settings    │
   │   Panel      │ │   Panel      │ │   Panel      │
   └──────────────┘ └──────────────┘ └──────────────┘
            │            │
            ▼            │
   ┌──────────────┐      │
   │ Scenario     │      │
   │ Classifier   │      │
   └──────────────┘      │
            │            │
            └────────────┘
                         │
                         ▼
                 ┌──────────────┐
                 │  plotly /    │
                 │ matplotlib   │
                 └──────────────┘
```

---

## 7. Entry Points & Usage

### 7.1 GUI Mode (Recommended)

**Full Application** (all panels):
```bash
streamlit run gui_launcher.py
```
- Opens browser at `http://localhost:8501`
- Provides: Scenarios, Collect, Process, Classify, Visualize panels
- Session state persists across interactions
- Real-time progress updates

**Simplified Application** (audio collection only):
```bash
streamlit run piano_response.py
```
- Streamlined interface for recording only
- Excludes ML pipeline (Process, Classify, Visualize)
- Lighter weight, faster startup

### 7.2 CLI Mode

**Single Scenario Collection**:
```bash
python collect_dataset.py \
  --scenario-number 1 \
  --description "Living room with carpet" \
  --num-measurements 30 \
  --measurement-interval 2.0 \
  --config-file recorderConfig.json \
  --output-dir room_response_dataset \
  --quiet
```

**Series Collection** (multiple scenarios):
```bash
python collect_dataset.py \
  --series "0.1,0.2,1-3,7a" \
  --pre-delay 60 \
  --inter-delay 60 \
  --num-measurements 30 \
  --measurement-interval 2.0 \
  --beep-volume 0.2 \
  --beep-freq 880 \
  --output-dir room_response_dataset
```

**Series Expression Syntax**:
- Single: `5` → scenario 5
- Decimal: `0.1` → scenario 0.1
- Range: `1-3` → scenarios 1, 2, 3
- Alphanumeric: `7a` → scenario 7a
- Combined: `0.1,0.2,1-3,7a` → scenarios 0.1, 0.2, 1, 2, 3, 7a

### 7.3 Feature Extraction CLI

```bash
python FeatureExtractor.py \
  --dataset_path room_response_dataset \
  --wav_subfolder impulse_responses \
  --recording-type average \
  --sample-rate 16000 \
  --n-mfcc 13 \
  --config-filename recorderConfig.json \
  --max-spectrum-freq 8000 \
  --force  # Overwrite existing features
```

### 7.4 Programmatic Usage

**Recording**:
```python
from RoomResponseRecorder import RoomResponseRecorder

# Initialize recorder
recorder = RoomResponseRecorder(config_file_path="recorderConfig.json")

# List devices
recorder.list_devices()

# Set devices
recorder.set_audio_devices(input=1, output=2)

# Test microphone
recorder.test_mic(duration=10.0)

# Record room response
recorder.take_record(
    output_file="output/raw.wav",
    impulse_file="output/impulse.wav",
    method=2
)
```

**Feature Extraction**:
```python
from FeatureExtractor import AudioFeatureExtractor

# Initialize extractor
extractor = AudioFeatureExtractor(
    sample_rate=16000,
    n_mfcc=13,
    config_filename="recorderConfig.json"
)

# Process entire dataset
extractor.process_dataset(
    dataset_path="room_response_dataset",
    wav_subfolder="impulse_responses",
    recording_type="average",
    skip_existing=True
)

# Or process single scenario
extractor.process_scenario_folder(
    scenario_folder="room_response_dataset/Laptop-Scenario1-LivingRoom",
    wav_subfolder="impulse_responses",
    recording_type="average"
)
```

**Classification**:
```python
from ScenarioClassifier import ScenarioClassifier

# Initialize classifier
clf = ScenarioClassifier(model_type="svm", feature_type="mfcc")

# Train binary classifier
result = clf.run_single_pair(
    path_a="dataset/Scenario1",
    path_b="dataset/Scenario2",
    label_a="quiet",
    label_b="noisy",
    params={"test_size": 0.2, "cv_folds": 5},
    dataset_root="dataset"
)

print(f"Test Accuracy: {result['metrics']['test_accuracy']:.2%}")
print(f"CV Mean: {result['metrics']['cv_mean']:.2%}")

# Save model
model_path = clf.save_model()
print(f"Model saved to: {model_path}")

# Later: load and predict
clf_loaded = ScenarioClassifier.load_model(model_path)
prediction = clf_loaded.predict_from_audio(
    audio=audio_data,
    extractor=extractor,
    feature_names=clf_loaded.feature_names
)
print(f"Predicted label: {prediction['label']}")
print(f"Probabilities: {prediction['proba']}")
```

---

## 8. Signal Processing Pipeline

### 8.1 Onset Detection Algorithm

**Purpose**: Find the exact sample where the impulse response begins, enabling proper alignment.

**Method**: RMS-based threshold detection with moving average

```python
def _find_sound_onset(audio: np.ndarray,
                     window_size: int = 10,
                     threshold_factor: float = 2) -> int:
    """
    Detect sound onset using moving RMS and derivative analysis.

    Args:
        audio: Input signal
        window_size: Moving average window size (samples)
        threshold_factor: Multiplier for background noise level

    Returns:
        Sample index of detected onset
    """
    # 1. Compute moving RMS
    def moving_rms(signal, window):
        padded = np.pad(signal, window // 2, mode='constant')
        return np.sqrt(np.convolve(padded ** 2,
                                    np.ones(window) / window,
                                    mode='valid'))

    rms = moving_rms(audio, window_size)

    # 2. Compute RMS derivative (rate of change)
    rms_diff = np.diff(rms)

    # 3. Estimate background noise level
    background_level = np.std(rms[:window_size]) if len(rms) > window_size else np.std(rms)

    # 4. Set threshold
    threshold = threshold_factor * background_level

    # 5. Find first point exceeding threshold
    onset_candidates = np.where(rms_diff > threshold)[0]

    return onset_candidates[0] if len(onset_candidates) > 0 else 0
```

**Visualization**:
```
RMS Level
   │      ┌─────────────────  Signal region
   │     ╱
   │    ╱
   │   ╱
   │  ╱
   │ ╱
   │╱________  Noise floor
   │          threshold
   └────────────────────────> Time
          ↑
        Onset
```

### 8.2 Signal Alignment (Cross-Correlation)

**File**: `signal_alignment.py`

**Purpose**: Synchronize multiple impulse response recordings by aligning onsets.

**Algorithm**:
```python
class SignalAligner:
    def align_signals(file_paths: List[str],
                     reference_index: int = 0,
                     max_shift: int = 10000) -> List[AlignmentResult]:
        """
        Align multiple signals via cross-correlation.

        Returns:
            List of AlignmentResult with aligned signals and shift amounts
        """
        # 1. Load all signals
        signals = [load_wav_file(path) for path in file_paths]

        # 2. Normalize to unit peak
        signals_normalized = [normalize_signal(s) for s in signals]

        # 3. Apply threshold to remove noise
        signals_cleaned = [apply_threshold(s, threshold=0.01)
                          for s in signals_normalized]

        # 4. Select reference signal
        reference = signals_cleaned[reference_index]

        # 5. Align each signal to reference
        results = []
        for i, signal in enumerate(signals_cleaned):
            if i == reference_index:
                # Reference stays unchanged
                results.append(AlignmentResult(
                    original_signal=signal,
                    aligned_signal=signal,
                    shift_samples=0,
                    correlation_peak=1.0,
                    file_path=file_paths[i]
                ))
            else:
                # Compute cross-correlation
                correlation = np.correlate(reference, signal, mode='full')

                # Find lag of maximum correlation
                lag = np.argmax(correlation) - (len(reference) - 1)

                # Limit shift to max_shift
                lag = np.clip(lag, -max_shift, max_shift)

                # Apply circular shift
                aligned = np.roll(signal, -lag)

                results.append(AlignmentResult(
                    original_signal=signal,
                    aligned_signal=aligned,
                    shift_samples=-lag,
                    correlation_peak=float(np.max(correlation)),
                    file_path=file_paths[i]
                ))

        return results
```

**Cross-Correlation Visualization**:
```
Reference:  ─┬─┐
            ─┴─┴─────

Signal:     ───┬─┐
            ───┴─┴───
               ↑
            Lag = 2 samples

Correlation:
   │       ╱╲
   │      ╱  ╲
   │     ╱    ╲
   │____╱______╲____
   └──────────────────> Lag
           ↑
        Peak at lag=2
```

### 8.3 FIR Filter Export

**File**: `FirFilterFileIO.py`

**Purpose**: Export impulse responses as FIR filter coefficients for use in audio processing software (e.g., REW, Equalizer APO).

**Format** (.fir file):
```
# FIR filter coefficients
# Sample rate: 48000 Hz
# Length: 4800 samples (100ms)
0.0012345
-0.0023456
0.0034567
...
```

**Usage**:
```python
from FirFilterFileIO import save_fir_filter, load_fir_filter

# Export impulse response as FIR filter
impulse_response = np.array([...])
save_fir_filter(
    coefficients=impulse_response,
    sample_rate=48000,
    output_path="room_correction.fir",
    metadata={"room": "Living Room", "mic": "Calibrated"}
)

# Load FIR filter
coefficients, metadata = load_fir_filter("room_correction.fir")
```

---

## 9. Machine Learning Framework

### 9.1 Model Selection Criteria

| Model | Best For | Pros | Cons |
|-------|----------|------|------|
| **SVM (RBF kernel)** | Non-linear boundaries, small-medium datasets | High accuracy, robust to outliers | Slower training, requires scaling |
| **Logistic Regression** | Linear boundaries, large datasets, interpretability | Fast, probabilistic, simple | Assumes linear separability |

### 9.2 Hyperparameter Tuning

**SVM Parameters**:
- `C=1.0`: Regularization (higher = less regularization)
- `gamma='scale'`: RBF kernel width (auto-adjusted to feature count)
- `kernel='rbf'`: Radial Basis Function (non-linear)
- `probability=True`: Enable probability estimates

**Logistic Regression Parameters**:
- `max_iter=5000`: Maximum iterations for convergence
- `solver='lbfgs'`: Optimization algorithm (default)

**Custom Tuning** (via GUI or code):
```python
params = {
    "test_size": 0.2,       # 20% held out for testing
    "cv_folds": 5,          # 5-fold cross-validation
    "max_samples_per_scenario": 100,  # Subsample large scenarios
    "balance_groups": True   # Balance class sizes (group mode)
}
```

### 9.3 Evaluation Metrics

**Accuracy**:
- **Train Accuracy**: Performance on training set (should be high)
- **Test Accuracy**: Performance on held-out test set (generalization)
- **CV Mean**: Average cross-validation accuracy (robust estimate)
- **CV Std**: Cross-validation standard deviation (stability indicator)

**Confusion Matrix**:
```
              Predicted
           Quiet   Noisy
Actual
Quiet       45      5     ← 90% recall for Quiet
Noisy        3     47     ← 94% recall for Noisy

Precision:  94%    90%
```

**Classification Report**:
```
              precision    recall  f1-score   support

       quiet       0.94      0.90      0.92        50
       noisy       0.90      0.94      0.92        50

    accuracy                           0.92       100
   macro avg       0.92      0.92      0.92       100
weighted avg       0.92      0.92      0.92       100
```

### 9.4 Feature Importance

**SVM Feature Importance** (derived from coefficients):
```python
# For linear SVM or logistic regression:
importance = np.abs(model.coef_[0])
importance_normalized = importance / np.sum(importance)

# For RBF SVM (approximate):
# Use support vector weights
importance = np.abs(model.dual_coef_).sum(axis=0)
```

**Interpretation**:
- Higher values = feature contributes more to decision boundary
- Can identify which frequencies or MFCCs discriminate scenarios

---

## 10. GUI Architecture

### 10.1 Streamlit Session State Management

**Key Session State Variables**:

```python
# Dataset management
SK_DATASET_ROOT = "dataset_root"                  # Current dataset path
SK_DATASET_NAME = "dataset_folder_name"           # Folder name in text input
SK_DATASET_NAME_PENDING = "dataset_folder_name_pending"  # Pending update
SK_LAST_DATASET_ROOT = "last_dataset_root"        # Previous root (cache invalidation)

# Scenario selection
SK_SCN_SELECTIONS = "scenarios_selected_set"      # Set of selected scenario paths
SK_SCN_EXPLORE = "scenarios_explore_path"         # Currently explored scenario
SK_SCENARIOS_DF = "scenarios_df_cache"            # Cached scenarios DataFrame

# Filtering
SK_FILTER_TEXT = "filter_text"                    # Primary filter input
SK_FILTER_COMPUTER = "filter_computer"            # Computer name filter
SK_FILTER_ROOM = "filter_room"                    # Room name filter

# Classification
SK_CLASSIFIER_OBJ = "classifier_obj"              # ScenarioClassifier instance
SK_CLASSIFICATION_RESULTS = "classification_results"  # Metrics dict
SK_LAST_MODEL_INFO = "last_model_info"            # ModelInfo dataclass
SK_CLASSIFICATION_ARTIFACTS = "classification_artifacts"  # Predictions, etc.

# Series collection
SK_SERIES_EVT_Q = "series_event_q"                # Event queue from worker
SK_SERIES_CMD_Q = "series_cmd_q"                  # Command queue to worker
SK_SERIES_THREAD = "series_thread"                # Background thread object
SK_SERIES_LAST = "series_last_event"              # Last event received
SK_SERIES_STARTED_AT = "series_started_at"        # Start timestamp

# Recorder persistence
"recorder" = RoomResponseRecorder instance         # Persisted across reruns
```

### 10.2 Panel Lifecycle

**Panel Rendering Flow**:
```python
# 1. Main app initialization
app = RoomResponseGUI()
app._initialize_components()  # Create managers, panels

# 2. Session state setup
app._ensure_initial_state()   # Set defaults, handle pending updates

# 3. Dataset root UI
app._ensure_dataset_root_ui() # Text input, validation, status

# 4. Navigation sidebar
panel = app._render_sidebar_navigation()  # Radio button selection

# 5. Panel rendering
app._render_panel(panel)      # Delegate to specific panel render()

# 6. Streamlit reruns on interaction
# → Session state persists
# → Panel render() called again with updated state
```

**Panel Interface** (convention):
```python
class MyPanel:
    def __init__(self, scenario_manager, **kwargs):
        self.manager = scenario_manager
        # Store dependencies

    def render(self):
        """Main rendering method called by app."""
        st.header("My Panel")

        # Read session state
        selections = st.session_state.get(SK_SCN_SELECTIONS, set())

        # Render UI
        if st.button("Do Something"):
            # Update session state
            st.session_state[SK_SOME_KEY] = new_value

            # Trigger rerun
            st.rerun()
```

### 10.3 Background Workers (Series Collection)

**Architecture**:
```python
# Main thread (Streamlit GUI)
│
├─> Create event queue (thread-safe)
├─> Create command queue (thread-safe)
├─> Start SeriesWorker thread
│   │
│   ├─> [Background Thread]
│   │   ├─> Initialize recorder
│   │   ├─> For each scenario in series:
│   │   │   ├─> Send status event to GUI ("running_scenario")
│   │   │   ├─> Collect measurements (blocking)
│   │   │   ├─> Send progress events
│   │   │   ├─> Check command queue (pause/resume/stop)
│   │   │   └─> Inter-scenario delay
│   │   ├─> Send completion event ("done")
│   │   └─> Thread exits
│   │
├─> GUI polling loop
│   ├─> Check event queue (non-blocking)
│   ├─> Update progress UI
│   └─> Handle user commands (pause/resume/stop)
│
└─> Wait for thread completion or user stop
```

**Event Protocol**:
```python
# Events sent from worker to GUI
{
    "type": "status",
    "status": "running_scenario" | "paused" | "done" | "error",
    "scenario_index": 0,
    "scenario_number": "1",
    "timestamp": 1234567890.123
}

{
    "type": "progress",
    "current": 5,
    "total": 30,
    "scenario_number": "1"
}

{
    "type": "error",
    "message": "Failed to record: ...",
    "scenario_number": "1"
}

# Commands sent from GUI to worker
{"command": "pause"}
{"command": "resume"}
{"command": "stop"}
```

### 10.4 Multi-Channel Audio Support (GUI Integration)

**Overview**: The GUI now supports multi-channel audio input for testing and monitoring advanced audio interfaces.

**Key Components**:

1. **RoomResponseRecorder Enhancements**:
   ```python
   # Multi-channel support attributes
   recorder.input_channels = 1  # Default: mono (backward compatible)

   # Get device capabilities
   devices = recorder.get_device_info_with_channels()
   # Returns: {'input_devices': [{'device_id': 0, 'name': 'Mic', 'max_channels': 8}, ...]}

   # Test multi-channel recording
   result = recorder.test_multichannel_recording(
       duration=2.0,
       num_channels=4
   )
   # Returns: {'success': bool, 'multichannel_data': [[ch0_samples], [ch1_samples], ...],
   #           'channel_stats': [{'max': 0.3, 'rms': 0.02, 'db': -18.5}, ...]}
   ```

2. **AudioDeviceSelector Features**:
   - **Channel Count Display**: Device list shows "(N ch)" for multi-channel devices
   - **Dynamic Channel Picker**: Range automatically adjusts to device capability
   - **Multi-Channel Monitor**: Live per-channel monitoring (up to 8 channels at 5Hz)
   - **Test Recording UI**: Test multi-channel recording with statistics table

3. **AudioSettingsPanel - Multi-Channel Test Tab**:
   ```
   ┌────────────────────────────────────────────────────────┐
   │ Audio Settings                                         │
   ├────────────────────────────────────────────────────────┤
   │ [System Info] [Device Selection] [Multi-Channel Test]  │
   ├────────────────────────────────────────────────────────┤
   │ Selected Device: MOTU 8A (ID: 1) - 8 channels         │
   ├────────────────────────────────────────────────────────┤
   │ Multi-Channel Monitor                                  │
   │  Monitor Channels: [4] ▼                              │
   │  [Start Monitor]                                       │
   │                                                        │
   │  Channel 0: ████████░░ -18.5 dB ✓ Good               │
   │  Channel 1: ██████░░░░ -22.3 dB 🟡 Moderate          │
   │  Channel 2: ████████░░ -19.1 dB ✓ Good               │
   │  Channel 3: ██████░░░░ -24.7 dB 🟡 Moderate          │
   ├────────────────────────────────────────────────────────┤
   │ Multi-Channel Test Recording                           │
   │  Test Channels: [4] Duration: [2.0s]                  │
   │  [Run Test Recording]                                  │
   │                                                        │
   │  Results: 4 channels × 96000 samples                  │
   │  ┌─────────┬──────────┬──────┬───────┐               │
   │  │ Channel │ Max Ampl │ RMS  │ dB    │               │
   │  ├─────────┼──────────┼──────┼───────┤               │
   │  │ 0       │ 0.3245   │ 0.03 │ -18.2 │               │
   │  │ 1       │ 0.3108   │ 0.03 │ -19.8 │               │
   │  └─────────┴──────────┴──────┴───────┘               │
   └────────────────────────────────────────────────────────┘
   ```

4. **Implementation Details**:
   - **Thread-Safe Monitoring**: Background worker thread for per-channel data collection
   - **5Hz Update Rate**: Efficient UI updates using `st.rerun()` with 200ms sleep
   - **Error Handling**: Graceful degradation when device doesn't support requested channels
   - **Backward Compatibility**: Defaults to mono (1 channel), existing code unaffected
   - **Performance**: Supports up to 32 channels for testing, displays up to 8 simultaneously

5. **Files Modified** (2025-10-25):
   - `RoomResponseRecorder.py`: Added `input_channels`, `get_device_info_with_channels()`, `test_multichannel_recording()`
   - `gui_audio_device_selector.py`: Added multi-channel monitor, test UI, dynamic channel picker
   - `gui_audio_settings_panel.py`: Added "Multi-Channel Test" tab with device info display

**Usage Scenario**:
```python
# GUI automatically detects multi-channel devices
# User selects 8-channel audio interface
# Channel picker shows 0-7 range
# User starts 4-channel monitor → sees live levels per channel
# User runs test recording → gets per-channel statistics
```

**Reference**: See [GUI_MULTICHANNEL_INTEGRATION_PLAN.md](GUI_MULTICHANNEL_INTEGRATION_PLAN.md) for complete implementation details.

### 10.5 Multi-Channel Device Driver Requirements

**Critical Finding**: Professional audio interfaces require **native manufacturer drivers** for multi-channel operation.

#### 10.5.1 Driver Architecture Issue

**Windows Generic USB Audio Class 2.0 Driver Limitations:**

The Windows generic USB Audio driver has a fundamental limitation:
- **Device Enumeration**: Correctly reports device capabilities (e.g., "10 input channels available")
- **WDM/WASAPI Interface**: **Hardcoded to stereo (2 channels)** only
- **Result**: SDL can enumerate the device with correct channel count, but `IAudioClient::Initialize()` fails with "Invalid source channels"

**Technical Root Cause:**
```
Device Capabilities (USB descriptor)     ✓ Correct: Reports 10 channels
    ↓
SDL_GetAudioDeviceSpec()                ✓ Correct: Returns 10 max_channels
    ↓
IAudioClient::GetMixFormat()            ✗ WRONG: Returns 2-channel format only
    ↓
IAudioClient::Initialize()              ✗ FAILS: "Invalid source channels"
```

**Why This Happens:**
- Generic driver implements USB Audio Class 2.0 spec for basic compatibility
- WDM/WASAPI kernel interface is limited to stereo for generic devices
- Microsoft prioritizes broad compatibility over advanced features
- Multi-channel access requires manufacturer-specific WDM driver

#### 10.5.2 Native Driver Solution

**Native manufacturer drivers provide:**
- ✓ Full multi-channel WDM/WASAPI access (all input/output channels)
- ✓ ASIO driver for professional low-latency operation
- ✓ Proper channel mask and format reporting to Windows
- ✓ Device-specific control panel and configuration
- ✓ Better audio quality and stability

**Example: Behringer UMC1820**

| Driver Type | Input Channels | Output Channels | WDM/WASAPI | ASIO |
|-------------|----------------|-----------------|------------|------|
| Generic USB | Reports: 10    | Reports: 12     | ✗ Limited to 2 | ✗ Not available |
| Behringer Native | 18 | 20 | ✓ Full access | ✓ Available |

#### 10.5.3 Device-Specific Installation Guides

**Behringer UMC1820:**
- **Detection Script**: `python check_umc_driver.py`
- **Installation Guide**: [install_behringer_driver.md](install_behringer_driver.md)
- **Driver Download**: https://www.behringer.com/downloads.html
- **Recommended Version**: 4.59.0 or 5.57.0
- **Test Script**: `python test_umc_input_detailed.py`

**Other Professional Interfaces:**

| Manufacturer | Driver Name | Max Channels | Download |
|--------------|-------------|--------------|----------|
| Focusrite | Focusrite Control + ASIO | 2-18 (model dependent) | https://focusrite.com/downloads |
| PreSonus | Universal Control + ASIO | 2-32 (model dependent) | https://www.presonus.com/products |
| MOTU | MOTU AVB/USB Driver | 6-64 (model dependent) | https://motu.com/download |
| RME | RME ASIO + TotalMix FX | 8-192 (model dependent) | https://www.rme-audio.de/downloads |

#### 10.5.4 Diagnostic Tools

**Check Current Driver Status:**
```bash
python check_umc_driver.py
```

**Output Example (Generic Driver):**
```
CHECKING DEVICE MANAGER
✗ Using generic USB Audio driver (NEEDS FIXING)

CHECKING SDL AUDIO DEVICES
→ UMC1820 (10 channels) detected

DIAGNOSIS & RECOMMENDATIONS
✗ STATUS: PROBLEM - Using generic Windows USB Audio driver
Multi-channel recording (10 channels) will NOT work.

SOLUTION: Install Behringer native driver
1. Read: install_behringer_driver.md
2. Download from: https://www.behringer.com/downloads.html
3. Restart computer after installation
```

**Test Multi-Channel Functionality:**
```bash
# Test 9 device combinations (input/output, various channel counts)
python test_umc_multichannel.py

# Test detailed channel counts (1, 2, 4, 6, 8, 10 channels)
python test_umc_input_detailed.py

# Test SDL device enumeration
python test_device_enumeration.py
```

#### 10.5.5 Technical Deep Dive

**SDL2 WASAPI Backend Limitation:**

SDL 2.30.0 WASAPI backend (`src/audio/wasapi/SDL_wasapi.c`) uses `AUDCLNT_STREAMFLAGS_AUTOCONVERTPCM` flag only for **sample rate** mismatches, not **channel count** mismatches.

**Current SDL Code (line 740-750):**
```c
DWORD streamflags = 0;
if ((DWORD)device->spec.freq != waveformat->nSamplesPerSec) {
    streamflags |= (AUDCLNT_STREAMFLAGS_AUTOCONVERTPCM |
                    AUDCLNT_STREAMFLAGS_SRC_DEFAULT_QUALITY);
    waveformat->nSamplesPerSec = device->spec.freq;
}
// ❌ No channel count check here!
```

**Why This Matters:**
- Even with `SDL_AUDIO_ALLOW_CHANNELS_CHANGE` flag, SDL won't request channel conversion
- Generic driver returns incompatible channel format via `GetMixFormat()`
- `IAudioClient::Initialize()` fails because formats don't match

**Proposed SDL Fix (not implemented):**
```c
if ((DWORD)device->spec.freq != waveformat->nSamplesPerSec ||
    device->spec.channels != waveformat->nChannels) {
    streamflags |= (AUDCLNT_STREAMFLAGS_AUTOCONVERTPCM |
                    AUDCLNT_STREAMFLAGS_SRC_DEFAULT_QUALITY);
    // ... set both sample rate and channel count
}
```

**Solution Status**: Native drivers bypass this issue entirely by providing correct multi-channel WDM/WASAPI interface.

#### 10.5.6 Implementation Changes for Driver Support

**Files Modified:**
- `sdl_audio_core/src/audio_engine.cpp`:
  - Added `SDL_AUDIO_ALLOW_CHANNELS_CHANGE` to both input and output devices
  - Implemented multi-channel output playback (mono-to-multichannel replication)
  - Added channel negotiation logging

- `sdl_audio_core/src/python_bindings.cpp`:
  - Added output device fallback mechanism (default → Device 0)
  - Improved error reporting for device initialization failures

**Files Created:**
- `SOLUTION_INSTALL_BEHRINGER_DRIVER.md` - Complete driver installation guide
- `SOLUTION_UMC1820_WASAPI.md` - Technical analysis of SDL/WASAPI limitations
- `TROUBLESHOOTING_UMC1820.md` - User-focused troubleshooting
- `install_behringer_driver.md` - Step-by-step installation instructions
- `check_umc_driver.py` - Driver status detection script
- `test_umc_multichannel.py` - Comprehensive device combination tests
- `test_umc_input_detailed.py` - Detailed channel count tests

**Expected Behavior After Native Driver Installation:**
```bash
python test_umc_input_detailed.py

# With native driver:
Testing with 1 channels...  ✓ SUCCESS - got 1 channels
Testing with 2 channels...  ✓ SUCCESS - got 2 channels
Testing with 4 channels...  ✓ SUCCESS - got 4 channels
Testing with 8 channels...  ✓ SUCCESS - got 8 channels
Testing with 10 channels... ✓ SUCCESS - got 10 channels
```

**Reference Documents:**
- [SOLUTION_INSTALL_BEHRINGER_DRIVER.md](SOLUTION_INSTALL_BEHRINGER_DRIVER.md) - Complete solution guide
- [SOLUTION_UMC1820_WASAPI.md](SOLUTION_UMC1820_WASAPI.md) - Technical deep-dive
- [install_behringer_driver.md](install_behringer_driver.md) - Installation instructions

---

## 11. Configuration & Metadata

### 11.1 Recorder Configuration (recorderConfig.json)

**Schema**:
```json
{
  "sample_rate": 48000,          // Audio sample rate (Hz)
  "pulse_duration": 0.008,       // Pulse length (seconds)
  "pulse_fade": 0.0001,          // Fade in/out duration (seconds)
  "cycle_duration": 0.1,         // Time between pulse starts (seconds)
  "num_pulses": 8,               // Number of pulses in train
  "volume": 0.4,                 // Playback amplitude (0.0-1.0)
  "pulse_frequency": 1000,       // Carrier frequency for sine (Hz)
  "impulse_form": "sine"         // Options: "sine", "square", "voice_coil"
}
```

**Validation Rules**:
- `pulse_duration > 0`
- `cycle_duration >= pulse_duration`
- `pulse_fade < pulse_duration / 2`
- `volume` in `[0.0, 1.0]`
- `impulse_form` in `["sine", "square", "voice_coil"]`

**Impulse Form Details**:
- **sine**: Sinusoidal pulse at `pulse_frequency` Hz with fade in/out
- **square**: Rectangular pulse with fade in/out to prevent clicks
- **voice_coil**: Square pulse with negative pull-back for voice coil actuator control
  - Main pulse duration controlled by `pulse_duration`
  - Pull-back duration controlled by `pulse_fade` parameter
  - Pull-back structure: 1/3 delay (zeros) + 2/3 negative ramp (-0.5 to 0)
  - Optimized for electromagnetic actuator control with proper retraction

### 11.2 Session Metadata (session_metadata.json)

**Schema**:
```json
{
  "scenario_info": {
    "scenario_number": "1",
    "description": "Living room with carpet",
    "computer_name": "Laptop",
    "room_name": "LivingRoom",
    "num_measurements": 30,
    "measurement_interval": 2.0,
    "warm_up_measurements": 0,
    "additional_metadata": {}
  },
  "recorder_config": {
    "sample_rate": 48000,
    "pulse_duration": 0.008,
    ...
  },
  "collection_info": {
    "started_at": "2025-01-15T10:30:45.123Z",
    "completed_at": "2025-01-15T10:32:15.456Z",
    "total_duration_seconds": 90.333,
    "successful_measurements": 30,
    "failed_measurements": 0
  },
  "quality_summary": {
    "mean_snr_db": 28.5,
    "mean_clipping_percentage": 0.02,
    "mean_dynamic_range_db": 42.3
  }
}
```

### 11.3 Scenario Metadata (scenario_meta.json)

**Schema**:
```json
{
  "label": "quiet, baseline",     // Comma-separated labels
  "description": "Quiet room with minimal furniture",
  "notes": "Collected with door closed, windows shut",
  "created_at": "2025-01-15T10:30:00.000Z",
  "updated_at": "2025-01-20T14:15:30.000Z"
}
```

### 11.4 Measurement Metadata (measurement_NNN_meta.json)

**Schema**:
```json
{
  "measurement_index": 0,
  "timestamp": "2025-01-15T10:30:45.123Z",
  "files": {
    "raw_recording": "raw_recordings/recording_000.wav",
    "impulse_response": "impulse_responses/impulse_000.wav",
    "room_response": "room_responses/room_000.wav"
  },
  "quality_metrics": {
    "snr_db": 29.2,
    "clipping_percentage": 0.01,
    "dynamic_range_db": 43.5,
    "rms_level": 0.12,
    "peak_amplitude": 0.87
  },
  "signal_info": {
    "recorded_samples": 38400,
    "expected_samples": 38400,
    "sample_rate": 48000,
    "duration_seconds": 0.8
  }
}
```

---

## 12. Error Handling & Quality Assurance

### 12.1 Configuration Validation

**RoomResponseRecorder Validation**:
```python
def _validate_config(self):
    errors = []

    if self.pulse_samples <= 0:
        errors.append("Pulse duration too short")

    if self.gap_samples < 0:
        errors.append("Cycle duration shorter than pulse duration")

    if self.fade_samples >= self.pulse_samples // 2:
        errors.append("Fade duration too long for pulse duration")

    if not 0.0 <= self.volume <= 1.0:
        errors.append("Volume must be between 0.0 and 1.0")

    if self.impulse_form not in ["square", "sine", "voice_coil"]:
        errors.append("Impulse form must be 'square', 'sine', or 'voice_coil'")

    if errors:
        raise ValueError(f"Configuration errors: {'; '.join(errors)}")
```

### 12.2 Recording Quality Checks

**Signal Quality Warnings**:
```python
# After recording, check signal quality
max_amplitude = np.max(np.abs(recorded_audio))
rms_level = np.sqrt(np.mean(recorded_audio ** 2))

if max_amplitude < 0.01:
    print("⚠️  WARNING: Very low signal level detected!")
    print("   Try: Increase microphone gain or move closer to speaker")

elif max_amplitude > 0.95:
    print("⚠️  WARNING: Signal may be clipping!")
    print("   Try: Reduce volume or speaker level")

if rms_level < 0.005:
    print("⚠️  WARNING: Low RMS level - check audio connections")
```

**Quality Assessment Thresholds**:
```python
# DatasetCollector quality assessment
QUALITY_THRESHOLDS = {
    "min_snr_db": 15.0,            # Minimum signal-to-noise ratio
    "max_clipping_percent": 2.0,   # Maximum acceptable clipping
    "min_dynamic_range_db": 25.0   # Minimum dynamic range
}

def _assess_quality(audio_data):
    metrics = calculate_metrics(audio_data)

    warnings = []
    if metrics["snr_db"] < QUALITY_THRESHOLDS["min_snr_db"]:
        warnings.append(f"Low SNR: {metrics['snr_db']:.1f} dB")

    if metrics["clipping_percentage"] > QUALITY_THRESHOLDS["max_clipping_percent"]:
        warnings.append(f"Clipping: {metrics['clipping_percentage']:.1f}%")

    if metrics["dynamic_range_db"] < QUALITY_THRESHOLDS["min_dynamic_range_db"]:
        warnings.append(f"Low dynamic range: {metrics['dynamic_range_db']:.1f} dB")

    return metrics, warnings
```

### 12.3 Feature Extraction Safeguards

**Audio Length Adaptation**:
```python
# Adapt FFT parameters to audio length to prevent librosa errors
if audio_length < 512:
    # Very short audio → smaller FFT
    n_fft = max(64, 2 ** int(np.log2(max(4, audio_length // 2))))
    hop_length = max(16, n_fft // 4)
elif audio_length < 2048:
    n_fft = 512
    hop_length = 128
else:
    # Standard FFT
    n_fft = 2048
    hop_length = 512

# Ensure n_fft doesn't exceed audio length
n_fft = min(n_fft, max(4, audio_length))
hop_length = min(hop_length, max(1, audio_length // 4))
```

**NaN/Inf Handling**:
```python
# Filter invalid MFCC frames
valid_mask = ~(np.isnan(mfcc[0]) | (np.sum(np.abs(mfcc), axis=0) == 0))

if not np.any(valid_mask):
    # All frames invalid → return zeros
    return np.zeros(self.n_mfcc, dtype=float)

# Average only valid frames
return np.mean(mfcc[:, valid_mask], axis=1).astype(float)
```

### 12.4 ML Training Safeguards

**Feature Alignment**:
```python
# Ensure feature columns match across scenarios
def _prepare_pair(path_a, path_b, label_a, label_b):
    df1, cols1 = _load_and_align(path_a)
    df2, cols2 = _load_and_align(path_b)

    # Find common columns (intersection)
    common_cols = sorted(
        list(set(cols1).intersection(cols2)),
        key=lambda c: _suffix_int(c) if _suffix_int(c) is not None else 10**9
    )

    if not common_cols:
        raise ValueError("Feature columns do not overlap between scenarios.")

    # Use only common columns
    X1 = df1[common_cols].to_numpy()
    X2 = df2[common_cols].to_numpy()
    ...
```

**Class Imbalance Handling**:
```python
# Balance groups by subsampling
def _balance_groups(X, y):
    unique_labels, counts = np.unique(y, return_counts=True)
    min_count = int(counts.min())

    balanced_indices = []
    for label in unique_labels:
        label_indices = np.where(y == label)[0]
        if len(label_indices) > min_count:
            # Subsample to min_count
            label_indices = np.random.choice(label_indices, min_count, replace=False)
        balanced_indices.extend(label_indices.tolist())

    balanced_indices = np.array(balanced_indices)
    return X[balanced_indices], y[balanced_indices]
```

### 12.5 SDL Audio Core Error Handling

**Device Initialization Failures**:
```python
try:
    result = sdl_audio_core.measure_room_response_auto(
        playback_signal=self.playback_signal,
        volume=self.volume,
        input_device=self.input_device,
        output_device=self.output_device
    )

    if not result['success']:
        error_msg = result.get('error_message', 'Unknown error')
        print(f"Measurement failed: {error_msg}")
        return None

except Exception as e:
    print(f"SDL audio core error: {e}")
    print("Possible causes:")
    print("  - Device ID invalid or disconnected")
    print("  - Incompatible sample rate")
    print("  - Insufficient permissions (Linux)")
    return None
```

---

## 13. Performance Considerations

### 13.1 Audio Processing Performance

**Bottlenecks**:
1. **SDL Audio I/O**: ~10-50ms latency (depends on buffer size)
2. **MFCC Computation**: ~100-500ms per file (librosa, numba JIT)
3. **FFT Computation**: ~10-50ms per file

**Optimization Strategies**:
- **Numba JIT**: librosa uses numba for accelerated MFCC computation
- **Batch Processing**: Process multiple files in parallel (not implemented)
- **Caching**: FeatureExtractor skips existing CSV files by default

**Benchmarks** (typical laptop, single-threaded):
- Record 30 measurements (800ms each): ~90 seconds (with 2s intervals)
- Extract MFCC from 30 files: ~5-10 seconds
- Train SVM on 200 samples: ~1-5 seconds

### 13.2 GUI Responsiveness

**Streamlit Rerun Cycle**:
- Each user interaction triggers full script rerun
- Session state persists across reruns
- Heavy operations should use caching or background threads

**Caching Strategies**:
```python
# ScenarioManager caches DataFrame in session state
@classmethod
def build_scenarios_df(cls, root: str, force_refresh=False):
    if not force_refresh and cls.SK_SCENARIOS_DF in st.session_state:
        return st.session_state[cls.SK_SCENARIOS_DF]

    # Expensive filesystem scan
    df = cls.analyze_dataset_filesystem(root)

    # Cache result
    st.session_state[cls.SK_SCENARIOS_DF] = df
    return df

# Clear cache on dataset root change
if last_root != resolved:
    cls.clear_cache()
```

**Background Processing**:
- Series collection runs in background thread
- Event queue for progress updates
- GUI polls queue during rerun cycle

### 13.3 Memory Usage

**Typical Memory Footprint**:
- RoomResponseRecorder: ~50 MB (signal buffers)
- FeatureExtractor: ~10 MB per scenario during processing
- ScenarioClassifier: ~20-100 MB (depends on dataset size)
- Streamlit GUI: ~200-500 MB base + session state

**Memory Management**:
- WAV files loaded one at a time (not held in memory)
- Feature DataFrames loaded on demand
- Large datasets should use subsampling (`max_samples_per_scenario`)

---

## 14. Extension Points

### 14.1 Adding New Feature Types

**Steps**:
1. **Add extraction method** to `FeatureExtractor.py`:
   ```python
   def _extract_chroma_from_audio(self, audio: np.ndarray) -> np.ndarray:
       chroma = librosa.feature.chroma_stft(y=audio, sr=self.sample_rate)
       return np.mean(chroma, axis=1)  # Average across time
   ```

2. **Update feature column detection**:
   ```python
   def _feature_columns(df: pd.DataFrame, feature_type: str) -> List[str]:
       if feature_type == "chroma":
           return [c for c in df.columns if c.startswith("chroma_")]
       ...
   ```

3. **Update CSV saving logic** in `process_scenario_folder()`:
   ```python
   chroma_vec = self._extract_chroma_from_audio(audio)
   chroma_row = {"filename": filename}
   for i, v in enumerate(chroma_vec):
       chroma_row[f"chroma_{i}"] = float(v)
   chroma_rows.append(chroma_row)
   ```

4. **Update GUI panels** to include new feature type in dropdown

### 14.2 Adding New ML Models

**Steps**:
1. **Update ScenarioClassifier model types**:
   ```python
   class ScenarioClassifier:
       def __init__(self, model_type="svm", feature_type="mfcc"):
           if model_type not in ("svm", "logistic", "random_forest"):
               raise ValueError("...")
           ...
   ```

2. **Add model instantiation**:
   ```python
   def _make_model(self):
       if self.model_type == "random_forest":
           from sklearn.ensemble import RandomForestClassifier
           return RandomForestClassifier(n_estimators=100, random_state=42)
       ...
   ```

3. **Update GUI Classify Panel** dropdown options

### 14.3 Custom Signal Forms

**Built-in Impulse Forms**:

The system currently supports three impulse forms:

1. **sine**: Sinusoidal carrier wave at `pulse_frequency` Hz
2. **square**: Rectangular pulse with fade in/out
3. **voice_coil**: Square pulse with negative pull-back for actuator control
   - Main pulse + 1/3 delay + 2/3 negative ramp
   - Optimized for electromagnetic voice coil actuators

**Adding New Impulse Forms**:

To add a new impulse form (e.g., "chirp"):

1. **Add new impulse form** in `RoomResponseRecorder._generate_single_pulse()`:
   ```python
   def _generate_single_pulse(self, exact_samples: int) -> np.ndarray:
       if self.impulse_form == "sine":
           # ... existing sine code
       elif self.impulse_form == "voice_coil":
           # ... existing voice_coil code
       elif self.impulse_form == "chirp":
           # Linear frequency sweep
           t = np.linspace(0, exact_samples / self.sample_rate, exact_samples)
           f0, f1 = 500, 2000  # 500Hz to 2000Hz sweep
           phase = 2 * np.pi * (f0 * t + (f1 - f0) * t**2 / (2 * exact_samples / self.sample_rate))
           pulse = np.sin(phase)
       else:  # square
           # ... existing square code
   ```

2. **Update configuration validation** in `_validate_config()`:
   ```python
   if self.impulse_form not in ["square", "sine", "voice_coil", "chirp"]:
       raise ValueError("Impulse form must be 'square', 'sine', 'voice_coil', or 'chirp'")
   ```

3. **Update GUI panels** to include new option:
   - `gui_series_settings_panel.py`: Add to waveform_options list
   - `gui_single_pulse_recorder.py`: Add to waveform_options list

### 14.4 Additional GUI Panels

**Template**:
```python
# gui_custom_panel.py
import streamlit as st

class CustomPanel:
    def __init__(self, scenario_manager):
        self.manager = scenario_manager

    def render(self):
        st.header("Custom Panel")

        # Read session state
        dataset_root = st.session_state.get("dataset_root")

        # Your custom logic
        st.write("Panel content here")

        if st.button("Do Something"):
            # Update session state
            st.session_state["custom_key"] = "value"
            st.rerun()
```

**Integration** in `gui_launcher.py`:
```python
try:
    from gui_custom_panel import CustomPanel
except ImportError:
    CustomPanel = None

class RoomResponseGUI:
    def __init__(self):
        ...
        self.custom_panel = None
        ...

    def _initialize_components(self):
        ...
        if CustomPanel and self.scenario_manager:
            self.custom_panel = CustomPanel(self.scenario_manager)

    def _render_sidebar_navigation(self):
        options = [..., "Custom"]
        ...

    def _render_panel(self, panel: str):
        ...
        elif panel == "Custom":
            self.custom_panel.render() if self.custom_panel else st.error("...")
```

---

## 15. Development Guidelines

### 15.1 Code Style

**Python**:
- Follow PEP 8
- Use type hints where possible
- Docstrings for all public methods (Google style)
- Maximum line length: 100 characters

**Example**:
```python
def process_audio(
    audio_data: np.ndarray,
    sample_rate: int,
    feature_type: str = "mfcc"
) -> Dict[str, np.ndarray]:
    """
    Process audio data and extract features.

    Args:
        audio_data: Input audio signal as float32 numpy array
        sample_rate: Audio sample rate in Hz
        feature_type: Type of features to extract ("mfcc" or "spectrum")

    Returns:
        Dictionary with feature arrays:
        - "features": Extracted feature vector (1D array)
        - "metadata": Additional metadata dict

    Raises:
        ValueError: If feature_type is not supported

    Example:
        >>> audio = librosa.load("test.wav", sr=16000)[0]
        >>> result = process_audio(audio, 16000, "mfcc")
        >>> print(result["features"].shape)
        (13,)
    """
    ...
```

### 15.2 Testing Recommendations

**Unit Tests** (not yet implemented):
```python
# test_room_response_recorder.py
import pytest
import numpy as np
from RoomResponseRecorder import RoomResponseRecorder

def test_signal_generation():
    recorder = RoomResponseRecorder()
    signal = recorder._generate_complete_signal()

    # Check signal length
    expected_samples = recorder.cycle_samples * recorder.num_pulses
    assert len(signal) == expected_samples

    # Check signal range
    signal_array = np.array(signal)
    assert np.max(np.abs(signal_array)) <= recorder.volume

def test_onset_detection():
    recorder = RoomResponseRecorder()

    # Create synthetic signal with known onset
    audio = np.zeros(1000)
    audio[100:] = np.random.randn(900) * 0.1  # Onset at sample 100

    onset = recorder._find_sound_onset(audio)
    assert 90 <= onset <= 110  # Allow 10-sample tolerance
```

**Integration Tests**:
- Test full recording pipeline with loopback device
- Test feature extraction on synthetic audio
- Test classifier training with small dataset

### 15.3 Debugging Tips

**Enable Logging**:
```python
# RoomResponseRecorder
recorder = RoomResponseRecorder()
sdl_info = recorder.get_sdl_core_info()
print(json.dumps(sdl_info, indent=2))

# FeatureExtractor
extractor = AudioFeatureExtractor()
extractor.process_dataset(..., skip_existing=False)  # Force reprocess
```

**Inspect Session State** (Streamlit):
```python
# Add to GUI panel render() method
with st.expander("Debug: Session State"):
    st.json(dict(st.session_state))
```

**Audio Debugging**:
```python
# Save intermediate signals
recorder._save_wav(room_response, "debug_room_response.wav")
recorder._save_wav(impulse_response, "debug_impulse.wav")

# Plot signals
import matplotlib.pyplot as plt
plt.plot(recorded_audio)
plt.axvline(onset_index, color='red', linestyle='--', label='Detected Onset')
plt.legend()
plt.savefig("debug_signal.png")
```

### 15.4 Contributing Guidelines

**Pull Request Process**:
1. Fork repository
2. Create feature branch: `git checkout -b feature/my-feature`
3. Make changes with clear commit messages
4. Add tests if applicable
5. Update documentation
6. Submit pull request with description

**Commit Message Format**:
```
<type>: <short summary>

<optional detailed description>

<optional breaking changes>
```

**Types**: `feat`, `fix`, `docs`, `refactor`, `test`, `chore`

**Example**:
```
feat: Add chirp signal generation for impulse responses

- Implement linear frequency sweep from 500Hz to 2000Hz
- Add chirp parameters to recorderConfig.json schema
- Update GUI panel to expose chirp configuration

Breaking: Config files must include "chirp_f0" and "chirp_f1" keys
```

---

## Appendix A: Glossary

**Acoustic Terms**:
- **Impulse Response**: System's output to a brief input signal (characterizes room acoustics)
- **Room Response**: Average acoustic response over multiple measurement cycles
- **Onset**: First sample where signal significantly exceeds background noise
- **SNR (Signal-to-Noise Ratio)**: Power of signal vs noise (higher = cleaner)
- **Dynamic Range**: Ratio between loudest and quietest signal parts
- **Clipping**: Distortion from signal exceeding maximum amplitude

**Signal Processing**:
- **FFT (Fast Fourier Transform)**: Efficient algorithm to compute frequency spectrum
- **MFCC (Mel-Frequency Cepstral Coefficients)**: Audio features modeling human hearing
- **RMS (Root Mean Square)**: Measure of signal power/energy
- **Cross-Correlation**: Measure of similarity between two signals at different time lags
- **Convolution**: Mathematical operation combining two signals (filtering)

**Machine Learning**:
- **SVM (Support Vector Machine)**: Classifier finding optimal decision boundary
- **Logistic Regression**: Probabilistic linear classifier
- **Cross-Validation**: Evaluation technique splitting data into folds
- **Confusion Matrix**: Table showing prediction vs actual labels
- **Feature Importance**: Measure of feature contribution to predictions

**Software**:
- **Streamlit**: Python web framework for data apps
- **pybind11**: Library for Python/C++ interoperability
- **SDL2**: Cross-platform multimedia library (audio, video, input)
- **joblib**: Library for model serialization and parallelization

---

## Appendix B: Troubleshooting

### B.1 Audio Recording Issues

**Problem**: No audio recorded (empty buffer)

**Solutions**:
- Check device IDs: `recorder.list_devices()`
- Test microphone: `recorder.test_mic(duration=10)`
- Verify SDL installation: `sdl_audio_core.check_installation()`
- Check permissions (Linux): `sudo usermod -a -G audio $USER`

---

**Problem**: Very low signal level

**Solutions**:
- Increase microphone gain in OS settings
- Move microphone closer to speaker
- Increase recorder volume: `recorder.volume = 0.6`
- Check microphone not muted

---

**Problem**: Signal clipping

**Solutions**:
- Reduce recorder volume: `recorder.volume = 0.3`
- Lower speaker output level
- Increase distance between speaker and microphone

---

### B.2 Feature Extraction Issues

**Problem**: MFCC extraction fails with NaN

**Solutions**:
- Check audio file integrity: `librosa.load(file_path)`
- Verify sample rate matches config
- Increase audio length (too short → invalid MFCCs)
- Use adaptive FFT parameters (already implemented)

---

**Problem**: Feature column count mismatch

**Solutions**:
- Ensure consistent sample rate across recordings
- Reprocess all scenarios with same configuration
- Use `max_spectrum_freq` to limit spectrum bins

---

### B.3 Classification Issues

**Problem**: Low test accuracy despite high train accuracy

**Solutions**:
- **Overfitting**: Increase dataset size, reduce model complexity
- **Data leakage**: Ensure train/test split stratified by scenario
- **Insufficient features**: Try different feature type (MFCC vs spectrum)

---

**Problem**: "Feature columns do not overlap"

**Solutions**:
- Reprocess scenarios with identical `FeatureExtractor` configuration
- Check `recorderConfig.json` consistent across scenarios
- Delete old CSV files: `rm */features.csv */spectrum.csv`

---

### B.4 GUI Issues

**Problem**: Panel not updating after action

**Solutions**:
- Check `st.rerun()` called after state changes
- Verify session state key names correct
- Clear browser cache or restart Streamlit server

---

**Problem**: Background worker hangs

**Solutions**:
- Check for deadlock in queue operations
- Verify worker thread started: `thread.is_alive()`
- Look for exceptions in worker thread (not propagated to GUI)
- Restart Streamlit server

---

## Appendix C: Useful Commands

**Dataset Management**:
```bash
# Count scenarios
ls -d room_response_dataset/*Scenario* | wc -l

# Find scenarios without features
find room_response_dataset -type d -name "*Scenario*" ! -exec test -e "{}/features.csv" \; -print

# Delete all feature CSVs (reprocess from scratch)
find room_response_dataset -name "features.csv" -delete
find room_response_dataset -name "spectrum.csv" -delete
```

**Audio Conversion**:
```bash
# Convert WAV to different sample rate (ffmpeg)
ffmpeg -i input.wav -ar 16000 output.wav

# Convert stereo to mono
ffmpeg -i input.wav -ac 1 output.wav

# Normalize audio levels
ffmpeg -i input.wav -filter:a loudnorm output.wav
```

**Model Management**:
```bash
# List saved models
ls -lh room_response_dataset/*.joblib

# Check model size
du -h room_response_dataset/room_response_model_*.joblib

# Backup models
tar -czf models_backup_$(date +%Y%m%d).tar.gz room_response_dataset/*.joblib
```

---

## Appendix D: References

**Libraries & Frameworks**:
- [librosa Documentation](https://librosa.org/doc/latest/index.html)
- [scikit-learn User Guide](https://scikit-learn.org/stable/user_guide.html)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [SDL2 Wiki](https://wiki.libsdl.org/)
- [pybind11 Documentation](https://pybind11.readthedocs.io/)

**Academic Papers**:
- McFee et al. (2015). "librosa: Audio and Music Signal Analysis in Python"
- Pedregosa et al. (2011). "Scikit-learn: Machine Learning in Python"

**Related Projects**:
- [REW (Room EQ Wizard)](https://www.roomeqwizard.com/) - Room acoustics measurement
- [Audacity](https://www.audacityteam.org/) - Audio editing and analysis
- [SoX](http://sox.sourceforge.net/) - Sound processing utility

---

**Document Version**: 1.0
**Last Updated**: 2025-01-15
**Author**: AI-Generated Technical Documentation
**System Version**: RoomResponse v1.0 (based on git commit 0e7bd64)
