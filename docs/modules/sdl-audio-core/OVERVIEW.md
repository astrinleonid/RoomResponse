# SDL Audio Core

C++ audio engine with pybind11 bindings. Wraps SDL2 for low-latency, cross-platform audio I/O with multi-channel support.

## Directory

```
sdl_audio_core/
├── include/common.h
├── src/
│   ├── audio_engine.cpp / .h      # Core engine: init, start, record, playback
│   ├── device_manager.cpp / .h    # SDL2 device enumeration and selection
│   ├── room_response.cpp / .h     # Room response utils (disabled)
│   └── python_bindings.cpp        # pybind11 module definition
├── setup.py                       # Build script
├── build_config.json              # Auto-detected SDL2 paths
├── tools/detect_paths.py          # SDL2 path auto-detection
└── SDL2.dll                       # Runtime DLL (Windows)
```

## Python API

### Classes

| Class | Purpose |
|-------|---------|
| `AudioEngine` | Core engine -- initialize, start/stop, record/playback |
| `AudioEngineConfig` | Configuration struct (sample_rate, input_channels, logging) |
| `AudioEngineStats` | Runtime statistics |

### Key Methods

| Method | Description |
|--------|-------------|
| `AudioEngine.initialize(config)` | Initialize SDL2 with given config |
| `AudioEngine.start()` / `stop()` | Start/stop audio subsystem |
| `AudioEngine.start_recording()` / `stop_recording()` | Control recording |
| `AudioEngine.get_recorded_data_channel(ch)` | Get numpy array for specific channel |
| `AudioEngine.clear_recording_buffer()` | Clear all channel buffers |
| `AudioEngine.set_input_device(id)` | Select input device by index |

### Device Enumeration

| Function | Description |
|----------|-------------|
| `get_input_devices()` | List available input devices |
| `get_output_devices()` | List available output devices |
| `list_all_devices()` | List all audio devices |
| `get_audio_drivers()` | List available SDL2 audio drivers |
| `get_sdl_version()` | SDL2 version string |
| `get_build_info()` | Build configuration info |

## Multi-Channel Support

The engine supports up to 32 input channels. Channel count is set via `AudioEngineConfig.input_channels`. SDL2 negotiates the actual format with the hardware driver.

Professional interfaces (Behringer UMC1820, Focusrite, etc.) require manufacturer drivers for full channel access. Windows generic USB Audio driver limits to 2 channels.

## MicTesting Wrapper

`MicTesting.py` provides `AudioRecorder` -- a context manager wrapping `AudioEngine`:

```python
from MicTesting import AudioRecorder

with AudioRecorder(sample_rate=48000, input_channels=6, input_channel=2) as rec:
    chunk = rec.get_audio_chunk(min_samples=4800)
```

Handles init, start, recording, cleanup automatically.
