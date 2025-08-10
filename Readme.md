# Room Response Recording Project

A Python-based acoustic room response measurement system using custom pulse train signals for precise audio analysis. The system uses a hybrid architecture combining C++ SDL audio core with Python signal processing for high-performance, cross-platform audio recording and analysis.

## 🎯 Project Overview

This project enables automated collection of room acoustic response datasets for research, audio analysis, and machine learning applications. It uses carefully designed pulse train signals to measure how rooms respond to audio, capturing characteristics like reverberation, echo, and acoustic signatures under different conditions.

## 🏗️ Architecture

- **C++ SDL Audio Core**: Low-level audio engine with pybind11 Python bindings
- **Python Interface**: High-level room response recorder with signal processing
- **Hybrid Approach**: SDL for device management + direct audio I/O for synchronized recording/playback
- **Configuration-Based**: JSON configuration files for signal parameters and defaults

## 📁 Project Structure

```
RoomResponse/
├── sdl_audio_core/              # C++ module with Python bindings
│   ├── src/
│   │   ├── audio_engine.h/cpp        # Core audio engine
│   │   ├── device_manager.h/cpp      # Device enumeration
│   │   └── python_bindings.cpp      # pybind11 interface
│   └── build system files
├── RoomResponseRecorder.py      # Main recorder class (config-based)
├── DatasetCollector.py          # Single scenario dataset collector
├── collect_scenario.py          # Command-line wrapper script
├── batch_collect_scenarios.py   # Python batch collection script
├── batch_collect.ps1            # PowerShell batch collection script
├── recorderConfig.json          # Signal configuration file
├── room_response_dataset/       # Output directory (created automatically)
│   └── <computer>-Scenario<num>-<room>/
│       ├── raw_recordings/           # Original recorded audio
│       ├── impulse_responses/        # Processed impulse responses
│       ├── room_responses/           # Averaged room responses
│       ├── metadata/                 # Session and measurement metadata
│       ├── analysis/                 # Analysis outputs
│       └── <scenario>_SUMMARY.txt    # Collection summary
└── README.md                    # This file
```

## 🚀 Quick Start

### Prerequisites

1. **Python 3.8+** with numpy, wave modules
2. **SDL2** audio library
3. **Compiled SDL audio core** (C++ module)
4. **Audio devices**: Working microphone and speakers

### Configuration Setup

1. **Create or edit `recorderConfig.json`**:
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
  "computer": "MyLaptop",
  "room": "LivingRoom"
}
```

2. **Single Scenario Collection**:
```bash
# Interactive mode
python collect_scenario.py

# Quiet mode with parameters
python collect_scenario.py --quiet --scenario-number "1" --num-measurements 50

# Interactive device selection
python collect_scenario.py -i
```

3. **Batch Collection** (5 scenarios, 200 measurements each):
```bash
# Python version (cross-platform)
python batch_collect_scenarios.py

# PowerShell version (Windows)
.\batch_collect.ps1
```

## 📚 API Reference

### RoomResponseRecorder Class

The main recording interface with JSON configuration-based initialization.

#### Constructor

```python
RoomResponseRecorder(config_file_path: str = None)
```

**Parameters:**
- `config_file_path`: Path to JSON configuration file. If None, uses default config.

**Configuration File Format:**
```json
{
  "sample_rate": 48000,        # Audio sample rate in Hz
  "pulse_duration": 0.008,     # Duration of each pulse in seconds
  "pulse_fade": 0.0001,        # Fade in/out duration to prevent clicks
  "cycle_duration": 0.1,       # Time between pulse starts in seconds
  "num_pulses": 8,             # Number of pulses in test signal
  "volume": 0.4,               # Playback volume 0.0-1.0
  "pulse_frequency": 1000,     # Frequency for sine wave pulses
  "impulse_form": "sine",      # Pulse type - "square" or "sine"
  "computer": "MyLaptop",      # Default computer name (optional)
  "room": "LivingRoom"         # Default room name (optional)
}
```

#### Main Methods

##### `take_record(output_file, impulse_file, method=2, **kwargs)`

Primary recording method with unified interface for all recording approaches.

```python
recorder = RoomResponseRecorder("recorderConfig.json")
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

### SingleScenarioCollector Class

Automated single scenario data collection with configuration-based setup.

#### Constructor

```python
SingleScenarioCollector(
    base_output_dir: str = "room_response_dataset",
    recorder_config: str = "recorderConfig.json",
    scenario_config: Dict[str, Any] = None
)
```

**Scenario Configuration:**
```python
scenario_config = {
    "computer_name": "MyLaptop",
    "room_name": "LivingRoom", 
    "scenario_number": "1",
    "description": "Empty room measurement",
    "num_measurements": 30,
    "measurement_interval": 2.0
}
```

## 🛠️ Command Line Tools

### collect_scenario.py

Single scenario data collection with flexible configuration options.

```bash
# Basic usage
python collect_scenario.py

# All options
python collect_scenario.py \
    --config-file "myConfig.json" \
    --output-dir "my_datasets" \
    --quiet \
    --scenario-number "5" \
    --description "Furniture test" \
    --num-measurements 100 \
    --measurement-interval 1.5 \
    --interactive
```

**Arguments:**
- `--config-file`: Configuration file path (default: recorderConfig.json)
- `--output-dir`: Base output directory (default: room_response_dataset)
- `--quiet`: Skip interactive prompts, use defaults
- `--scenario-number`: Scenario number (e.g., "1", "0.1", "5a")
- `--description`: Scenario description
- `--num-measurements`: Number of measurements (default: 30)
- `--measurement-interval`: Seconds between measurements (default: 2.0)
- `--interactive`: Enable interactive audio device selection

### Batch Collection Scripts

#### Python Version (Cross-platform)
```bash
python batch_collect_scenarios.py
```

#### PowerShell Version (Windows)
```powershell
.\batch_collect.ps1
```

**Features:**
- Automatically runs scenarios 0.1, 0.2, 0.3, 0.4, 0.5
- 200 measurements per scenario
- ~100 minutes total collection time
- Progress tracking and error handling
- Comprehensive summary reports

## 🎛️ Configuration Examples

### High-Quality Research Configuration

```json
{
  "sample_rate": 96000,
  "pulse_duration": 0.005,
  "cycle_duration": 0.15,
  "num_pulses": 12,
  "volume": 0.3,
  "impulse_form": "sine",
  "computer": "ResearchLab",
  "room": "AnechoicChamber"
}
```

### Fast Survey Configuration

```json
{
  "sample_rate": 44100,
  "pulse_duration": 0.01,
  "cycle_duration": 0.08,
  "num_pulses": 6,
  "volume": 0.5,
  "impulse_form": "square",
  "computer": "FieldLaptop",
  "room": "Office"
}
```

## 📊 Output Files and Naming Convention

### Dataset Naming
Datasets use the convention: `<computer>-Scenario<number>-<room>`

Examples:
- `MyLaptop-Scenario1-LivingRoom`
- `ResearchPC-Scenario0.1-Lab`
- `FieldDevice-Scenario5a-Auditorium`

### Generated Files

For each measurement, the system creates:

1. **Raw Recording** (`raw_<scenario>_XXX_TIMESTAMP.wav`)
   - Original recorded audio with all pulses
   - Full duration capture including reverberation

2. **Impulse Response** (`impulse_<scenario>_XXX_TIMESTAMP.wav`)
   - Processed impulse response with onset correction
   - Single cycle representing room's acoustic signature

3. **Room Response** (`room_<scenario>_XXX_TIMESTAMP.wav`)
   - Averaged response from multiple pulse cycles
   - Noise-reduced representation of room acoustics

### Metadata

Each scenario generates comprehensive metadata:

```json
{
  "scenario_info": {
    "scenario_name": "MyLaptop-Scenario1-LivingRoom",
    "scenario_number": "1",
    "computer_name": "MyLaptop",
    "room_name": "LivingRoom",
    "description": "Empty room measurement",
    "collection_timestamp": "2025-08-10T17:30:00"
  },
  "recorder_config": {...},
  "device_info": {...},
  "measurements": [...],
  "summary": {
    "total_measurements": 30,
    "success_rate": 100.0
  }
}
```

## 🔧 Troubleshooting

### Common Issues

#### Configuration File Problems

**Symptoms:** "Config file not found" or "Invalid JSON"

**Solutions:**
- Ensure `recorderConfig.json` exists in the working directory
- Validate JSON syntax using online JSON validators
- Check file permissions

#### Low SNR (Signal-to-Noise Ratio)

**Symptoms:** Quality warnings about low SNR (< 15dB)

**Solutions:**
- Increase microphone gain in system settings
- Move microphone closer to speakers
- Reduce background noise
- Adjust `volume` parameter in config file

#### Audio Device Issues

**Symptoms:** "No suitable devices found" or device selection failures

**Solutions:**
```python
# List available devices
recorder = RoomResponseRecorder("recorderConfig.json")
devices = recorder.list_devices()

# Use interactive device selection
python collect_scenario.py --interactive
```

#### Windows PowerShell Execution Policy

**Symptoms:** Cannot run `.ps1` scripts

**Solutions:**
```powershell
# Check current policy
Get-ExecutionPolicy

# Set policy for current user
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# Or run with bypass
powershell -ExecutionPolicy Bypass -File .\batch_collect.ps1
```

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

## 🚀 Workflow Examples

### Research Data Collection

```bash
# Configure for high-quality research
# Edit recorderConfig.json with high sample rate, more pulses

# Collect baseline data
python collect_scenario.py --scenario-number "baseline" --description "Empty_room_baseline" --num-measurements 100

# Collect treatment data
python collect_scenario.py --scenario-number "treatment1" --description "Added_acoustic_panels" --num-measurements 100
```

### Automated Survey Collection

```bash
# Batch collection for standardized scenarios
python batch_collect_scenarios.py

# Results in:
# - MyLaptop-Scenario0.1-Office (200 measurements)
# - MyLaptop-Scenario0.2-Office (200 measurements)
# - MyLaptop-Scenario0.3-Office (200 measurements)
# - MyLaptop-Scenario0.4-Office (200 measurements)
# - MyLaptop-Scenario0.5-Office (200 measurements)
```

### Custom Measurement Campaign

```bash
# Day 1: Empty room
python collect_scenario.py --scenario-number "1a" --description "Empty_room_morning" --num-measurements 50

# Day 2: Furnished room
python collect_scenario.py --scenario-number "1b" --description "Furnished_room_morning" --num-measurements 50

# Day 3: Occupied room
python collect_scenario.py --scenario-number "1c" --description "Occupied_room_meeting" --num-measurements 30
```

## 🤝 Contributing

### Development Setup

1. Clone repository with SDL audio core
2. Build C++ audio engine with pybind11
3. Install Python dependencies
4. Create `recorderConfig.json` with your settings
5. Test with basic recording

### Code Style

- Follow PEP 8 for Python code
- Use type hints for all public methods
- Comprehensive docstrings for classes and methods
- Error handling with informative messages
- JSON configuration for all parameters

## 📄 License

[Specify your license here]

## 🆘 Support

For issues, questions, or contributions:

1. Check troubleshooting section above
2. Review audio quality guidelines
3. Test with different recording methods
4. Verify audio device functionality
5. Validate configuration file syntax

---

**Note:** This project requires properly configured audio hardware and the compiled SDL audio core module. Ensure all dependencies are installed, configuration files are set up correctly, and audio devices are functioning before use.