# Build System

## Prerequisites

| Dependency | Version | Purpose |
|-----------|---------|---------|
| Python | 3.8+ | Runtime |
| SDL2 dev libs | 2.x | Audio I/O backend |
| MSVC / GCC / Clang | C++17 | Compile `sdl_audio_core` |
| pybind11 | latest | C++ <-> Python bindings |
| numpy | latest | Array interop, signal processing |
| streamlit | latest | GUI framework |
| librosa | latest | Audio feature extraction |
| scikit-learn | latest | ML classification |

## Building sdl_audio_core

The C++ extension lives in `sdl_audio_core/` and builds via `setup.py` using pybind11.

### Step 1: Generate Build Config

```bash
cd sdl_audio_core
python tools/detect_paths.py --auto
```

Writes `build_config.json` with SDL2 include/lib paths for the current platform.

### Step 2: Build and Install

```bash
python setup.py build_ext --inplace
pip install -e .
```

### Windows Helper

```bash
scripts/build_sdl_audio_core.bat
```

Runs detect, build, install, and copies `SDL2.dll` to the working directory.

### Build Config Structure

`sdl_audio_core/build_config.json`:

```json
{
  "include_dirs": ["path/to/SDL2/include"],
  "library_dirs": ["path/to/SDL2/lib/x64"],
  "libraries": ["SDL2", "SDL2main"]
}
```

## Source Files

| File | Role |
|------|------|
| `src/audio_engine.cpp` / `.h` | Core audio engine (init, start, record, playback) |
| `src/device_manager.cpp` / `.h` | SDL2 device enumeration and selection |
| `src/room_response.cpp` / `.h` | Room response utilities (currently disabled) |
| `src/python_bindings.cpp` | pybind11 module definition |
| `include/common.h` | Shared headers |
| `setup.py` | Build script |
| `tools/detect_paths.py` | Auto-detect SDL2 paths |

## Python Package Installation

```bash
pip install numpy pybind11 streamlit pandas scipy scikit-learn matplotlib librosa tqdm joblib
```

## Platform Notes

| Platform | SDL2 Install | DLL Note |
|----------|-------------|----------|
| Windows | Download dev ZIP from libsdl.org | Copy `SDL2.dll` to working directory or add to `PATH` |
| macOS | `brew install sdl2` | -- |
| Linux | `apt install libsdl2-dev` | -- |
