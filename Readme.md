# Room Response Recording Project

A Python-based acoustic room response measurement system using custom pulse train signals for precise audio analysis. The system uses a hybrid architecture combining C++ SDL audio core with Python signal processing for high-performance, cross-platform audio recording and analysis.

## 🎯 Project Overview

This project enables automated collection of room acoustic response datasets for research, audio analysis, and machine learning applications. It uses carefully designed pulse train signals to measure how rooms respond to audio, capturing characteristics like reverberation, echo, and acoustic signatures under different conditions.

## 🏗️ Architecture

- **C++ SDL Audio Core**: Low-level audio engine with pybind11 Python bindings
- **Python Interface**: High-level room response recorder with signal processing
- **Hybrid Approach**: SDL for device management + direct audio I/O for synchronized recording/playback

## 📁 Project Structure

```
RoomResponse/
├── sdl_audio_core/              # C++ module with Python bindings
│   ├── src/
│   │   ├── audio_engine.h/cpp        # Core audio engine
│   │   ├── device_manager.h/cpp      # Device enumeration
│   │   └── python_bindings.cpp      # pybind11 interface
│   └── build system files
├── RoomResponseRecorder.py      # Main recorder class (refactored)
├── dataSetCollector.py          # Automated dataset collection
├── room_response_dataset/       # Output directory (created automatically)
│   └── session_YYYYMMDD_HHMMSS/
│       ├── raw_recordings/           # Original recorded audio
│       ├── impulse_responses/        # Processed impulse responses
│       ├── room_responses/           # Averaged room responses
│       ├── metadata/                 # Session and measurement metadata
│       └── analysis/                 # Analysis outputs
└── README.md                    # This file
```

## 🚀 Quick Start

### Prerequisites

1. **Python 3.8+** with numpy, wave modules
2. **SDL2** audio library
3. **Compiled SDL audio core** (C++ module)
4. **Audio devices**: Working microphone and speakers

### Basic Usage

```python
from RoomResponseRecorder import RoomResponseRecorder

# Create recorder with default settings
recorder = RoomResponseRecorder()

# Record a single measurement
audio_data = recorder.take_record("recording.wav", "impulse.wav")
```

### Dataset Collection

```python
from dataSetCollector import RoomResponseDatasetCollector

# Create collector
collector = RoomResponseDatasetCollector()

# Run full dataset collection with default scenarios
collector.collect_full_dataset(recording_method=2)
```

## 📚 API Reference

### RoomResponseRecorder Class

The main recording interface with unified API for all recording methods.

#### Constructor

```python
RoomResponseRecorder(
    sample_rate: int = 48000,
    pulse_duration: float = 0.008,
    pulse_fade: float = 0.0001,
    cycle_duration: float = 0.1,
    num_pulses: int = 8,
    volume: float = 0.4,
    impulse_form: str = "square"
)
```

**Parameters:**
- `sample_rate`: Audio sample rate in Hz (default: 48000)
- `pulse_duration`: Duration of each pulse in seconds (default: 0.008)
- `pulse_fade`: Fade in/out duration to prevent clicks (default: 0.0001)
- `cycle_duration`: Time between pulse starts in seconds (default: 0.1)
- `num_pulses`: Number of pulses in test signal (default: 8)
- `volume`: Playback volume 0.0-1.0 (default: 0.4)
- `impulse_form`: Pulse type - "square" or "sine" (default: "square")

#### Main Methods

##### `take_record(output_file, impulse_file, method=2, **kwargs)`

Primary recording method with unified interface for all recording approaches.

```python
audio_data = recorder.take_record(
    output_file="recording.wav",
    impulse_file="impulse.wav", 
    method=2,                    # Recording method (1, 2, or 3)
    interactive=False,           # Interactive device selection (method 1)
    input_device_id=None,        # Input device ID (method 3)
    output_device_id=None        # Output device ID (method 3)
)
```

**Recording Methods:**
- **Method 1**: Manual AudioEngine setup with optional interactive device selection
- **Method 2**: Automatic device selection (recommended)
- **Method 3**: Specific device IDs

**Returns:** `numpy.ndarray` of recorded audio data, or `None` if failed

##### `list_devices()`

List all available audio input and output devices.

```python
devices = recorder.list_devices()
# Returns: {'input_devices': [...], 'output_devices': [...]}
```

##### `get_signal_info()`

Get detailed information about the generated test signal.

```python
info = recorder.get_signal_info()
# Returns: Dict with signal parameters and timing information
```

##### `print_signal_analysis()`

Print detailed analysis of the test signal configuration.

```python
recorder.print_signal_analysis()
```

### RoomResponseDatasetCollector Class

Automated dataset collection across multiple scenarios.

#### Constructor

```python
RoomResponseDatasetCollector(
    base_output_dir: str = "room_response_dataset",
    recorder_config: Dict[str, Any] = None
)
```

#### Main Methods

##### `collect_full_dataset(recording_method=2)`

Run complete dataset collection process with all configured scenarios.

```python
collector.collect_full_dataset(recording_method=2)
```

##### `add_scenario(scenario)`

Add a custom measurement scenario.

```python
from dataSetCollector import ScenarioConfig

scenario = ScenarioConfig(
    name="custom_scenario",
    description="Custom room configuration",
    num_measurements=25,
    measurement_interval=2.0
)
collector.add_scenario(scenario)
```

##### `setup_default_scenarios()`

Configure standard set of measurement scenarios:
- Empty room (40 measurements)
- Single person center (35 measurements)  
- Single person corner (35 measurements)
- Two people (35 measurements)
- Furniture added (35 measurements)
- Doors open (30 measurements)
- Moving person (25 measurements)

## 🎛️ Configuration Examples

### High-Quality Research Configuration

```python
recorder = RoomResponseRecorder(
    sample_rate=96000,          # Higher sample rate
    pulse_duration=0.005,       # Shorter pulses for better time resolution
    cycle_duration=0.15,        # Longer cycles for more reverberation capture
    num_pulses=12,              # More pulses for better averaging
    volume=0.3,                 # Conservative volume
    impulse_form="sine"         # Smoother sine wave pulses
)
```

### Fast Survey Configuration

```python
recorder = RoomResponseRecorder(
    sample_rate=44100,          # Standard sample rate
    pulse_duration=0.01,        # Longer pulses for robustness
    cycle_duration=0.08,        # Faster cycles
    num_pulses=6,               # Fewer pulses for speed
    volume=0.5,                 # Higher volume for noisy environments
    impulse_form="square"       # Sharp square wave pulses
)
```

### Custom Dataset Collection

```python
collector = RoomResponseDatasetCollector(
    base_output_dir="my_room_study",
    recorder_config={
        'sample_rate': 48000,
        'pulse_duration': 0.008,
        'cycle_duration': 0.1,
        'num_pulses': 8,
        'volume': 0.4,
        'impulse_form': 'square'
    }
)

# Add custom scenarios
collector.add_scenario(ScenarioConfig(
    name="curtains_closed",
    description="Room with heavy curtains closed",
    num_measurements=30,
    additional_metadata={"curtains": "closed", "absorption": "high"}
))

collector.collect_full_dataset(recording_method=2)
```

## 📊 Output Files

### Generated Files

For each measurement, the system creates:

1. **Raw Recording** (`raw_SCENARIO_XXX_TIMESTAMP.wav`)
   - Original recorded audio with all pulses
   - Full duration capture including reverberation

2. **Impulse Response** (`impulse_SCENARIO_XXX_TIMESTAMP.wav`)
   - Processed impulse response with onset correction
   - Single cycle representing room's acoustic signature

3. **Room Response** (`room_SCENARIO_XXX_TIMESTAMP_room.wav`)
   - Averaged response from multiple pulse cycles
   - Noise-reduced representation of room acoustics

### Metadata

Each session generates comprehensive metadata:

```json
{
  "session_info": {
    "timestamp": "20250809_194618",
    "recorder_config": {...},
    "device_info": {...},
    "quality_thresholds": {...}
  },
  "scenarios": [...],
  "measurements": [
    {
      "scenario_name": "empty_room",
      "measurement_index": 0,
      "timestamp": "20250809_194618_458",
      "filename_raw": "raw_empty_room_000_20250809_194618_458.wav",
      "filename_impulse": "impulse_empty_room_000_20250809_194618_458.wav",
      "quality_metrics": {
        "snr_db": 18.5,
        "max_amplitude": 0.23,
        "rms_level": 0.045,
        "clip_percentage": 0.0,
        "dynamic_range_db": 28.2
      }
    }
  ]
}
```

## 🔧 Troubleshooting

### Common Issues

#### Low SNR (Signal-to-Noise Ratio)

**Symptoms:** Quality warnings about low SNR (< 15dB)

**Solutions:**
- Increase microphone gain in system settings
- Move microphone closer to speakers
- Reduce background noise
- Check microphone is working properly

#### Audio Device Issues

**Symptoms:** "No suitable devices found" or device selection failures

**Solutions:**
```python
# List available devices
recorder = RoomResponseRecorder()
devices = recorder.list_devices()

# Use interactive device selection
audio_data = recorder.take_record("test.wav", "impulse.wav", 
                                 method=1, interactive=True)
```

#### File Permission Errors

**Symptoms:** Cannot save audio files

**Solutions:**
- Ensure output directory is writable
- Run with appropriate permissions
- Check disk space availability

#### Clipping Detection

**Symptoms:** "Signal may be clipping" warnings

**Solutions:**
- Reduce volume parameter in recorder configuration
- Lower speaker/system volume
- Check for audio driver gain settings

### Audio Quality Guidelines

**Optimal Recording Conditions:**
- Quiet environment (minimal background noise)
- Moderate volume levels (avoid too loud/quiet)
- Stable microphone and speaker positions
- Good acoustic separation between mic and speakers

**Quality Thresholds:**
- SNR: ≥ 15dB (good), ≥ 20dB (excellent)
- Clipping: < 2% (acceptable), < 1% (good)
- Dynamic Range: ≥ 25dB (acceptable), ≥ 30dB (good)

## 🔬 Technical Details

### Signal Processing

The system uses **pulse train signals** with the following characteristics:

- **Pulse Generation**: Square or sine wave pulses with smooth fade-in/out
- **Timing Precision**: Exact sample-level timing for consistent measurements
- **Cycle Averaging**: Multiple pulse cycles averaged to reduce noise
- **Onset Detection**: Automatic detection of signal arrival time
- **Impulse Extraction**: Signal rotation to align impulse response start

### Audio Engine

**SDL2-Based Core:**
- Thread-safe audio callbacks with atomic operations
- Circular buffers for real-time processing
- Synchronized recording and playback
- Cross-platform device management
- Low-latency audio I/O

**Quality Control:**
- Real-time signal level monitoring
- Automatic clipping detection
- SNR estimation and reporting
- Dynamic range analysis

## 📈 Research Applications

### Acoustic Analysis
- Room reverberation time measurement
- Acoustic signature classification
- Echo and reflection analysis
- Frequency response characterization

### Machine Learning Datasets
- Room type classification training data
- Occupancy detection model training
- Audio source localization datasets
- Acoustic scene analysis

### Audio Engineering
- Room acoustic optimization
- Speaker placement analysis
- Acoustic treatment effectiveness
- Sound system calibration

## 🤝 Contributing

### Development Setup

1. Clone repository with SDL audio core
2. Build C++ audio engine with pybind11
3. Install Python dependencies
4. Test with basic recording

### Code Style

- Follow PEP 8 for Python code
- Use type hints for all public methods
- Comprehensive docstrings for classes and methods
- Error handling with informative messages

## 📄 License

[Specify your license here]

## 🆘 Support

For issues, questions, or contributions:

1. Check troubleshooting section above
2. Review audio quality guidelines
3. Test with different recording methods
4. Verify audio device functionality

---

**Note:** This project requires properly configured audio hardware and the compiled SDL audio core module. Ensure all dependencies are installed and audio devices are functioning before use.