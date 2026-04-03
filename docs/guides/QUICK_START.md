# Quick Start

## Prerequisites

| Requirement | Notes |
|-------------|-------|
| Python 3.8+ | 3.12 tested |
| SDL2 development libraries | See [Build System](../architecture/BUILD_SYSTEM.md) |
| C++ compiler (MSVC/GCC/Clang) | For building `sdl_audio_core` |
| Audio device | Built-in mic/speaker minimum; multi-channel interface for advanced use |

## Setup

### 1. Python Dependencies

```bash
pip install numpy pybind11 streamlit pandas scipy scikit-learn matplotlib librosa tqdm joblib
```

### 2. Build sdl_audio_core

**Windows (one-step):**

```bash
scripts\build_sdl_audio_core.bat
```

**Manual (any platform):**

```bash
cd sdl_audio_core
python tools/detect_paths.py --auto
python setup.py build_ext --inplace
pip install -e .
```

### 3. Verify Installation

```python
import sdl_audio_core as sdl
print(sdl.get_version())
print(sdl.list_all_devices())
```

## First Run

### GUI Mode

```bash
streamlit run gui_launcher.py
```

1. **Scenarios panel**: Set dataset root directory (default: `room_response_dataset/`)
2. **Collect panel**: Configure recording parameters or load from `recorderConfig.json`
3. Start a single measurement or series collection

### CLI Mode

**Single measurement:**

```bash
python collect_dataset.py --quiet --scenario-number 1 --description "Test" --num-measurements 5
```

**Series collection:**

```bash
python collect_dataset.py --series "1-3" --num-measurements 10 --measurement-interval 2.0
```

## Output

Recordings land in `room_response_dataset/<computer>-Scenario<N>-<room>/`:

```
raw_recordings/       # Raw captured audio
impulse_responses/    # Extracted impulse responses
room_responses/       # Room response estimates
metadata/             # Session metadata JSON
```

## Next Steps

- [Configuration Reference](CONFIGURATION.md) -- customize signal parameters
- [GUI Overview](../modules/gui/OVERVIEW.md) -- panel descriptions
- [Data Flows](../architecture/DATA_FLOWS.md) -- understand the processing pipeline

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `ImportError: sdl_audio_core` | Rebuild the extension; ensure `SDL2.dll` is on PATH (Windows) |
| No audio devices listed | Check SDL2 installation; install manufacturer drivers for pro interfaces |
| 2-channel limit on multi-channel interface | Install manufacturer's native driver (not Windows generic USB Audio) |
| Low SNR warnings | Reduce background noise, adjust volume, improve mic placement |
