# RoomResponse Documentation

Acoustic room response measurement system. Injects configurable pulse-train signals through a C++ SDL2 audio engine, records multi-channel responses, extracts impulse responses via signal processing, and classifies room acoustics with ML. Includes ESPRIT modal analysis for piano soundboard research.

---

## System Architecture

```mermaid
graph TD
    subgraph "C++ Layer"
        SDL[sdl_audio_core<br/>C++ / pybind11]
    end

    subgraph "Python Backend"
        REC[RoomResponseRecorder]
        SP[SignalProcessor]
        DC[DatasetCollector]
        FE[FeatureExtractor]
        SC[ScenarioClassifier]
        ESP[ESPRIT Core]
    end

    subgraph "Frontend"
        GUI[Streamlit GUI<br/>gui_launcher.py]
        PANELS[Panels: Scenarios, Collect,<br/>Process, Classify, Visualize]
    end

    subgraph "Data"
        CFG[recorderConfig.json]
        DS[room_response_dataset/]
    end

    GUI --> PANELS
    PANELS --> DC
    PANELS --> FE
    PANELS --> SC
    DC --> REC
    REC --> SDL
    REC --> SP
    SP --> DS
    DC --> DS
    FE --> DS
    CFG --> REC
    ESP --> DS
```

## Module Index

| Module | Language | Role |
|--------|----------|------|
| `sdl_audio_core` | C++ / pybind11 | Low-latency audio I/O via SDL2 |
| `RoomResponseRecorder` | Python | Recording orchestration, signal generation, playback+capture |
| `SignalProcessor` | Python | Cycle extraction, alignment, normalization, averaging |
| `DatasetCollector` | Python | Scenario-based dataset management, metadata, QA |
| `FeatureExtractor` | Python | MFCC and spectrum feature extraction |
| `ScenarioClassifier` | Python | SVM/LogReg training, evaluation, model persistence |
| `ESPRIT` | Python | TLS-ESPRIT modal identification for piano soundboards |
| `gui_launcher` | Python (Streamlit) | Multi-panel GUI application |

## Documentation Map

### Architecture

- [System Overview](architecture/SYSTEM_OVERVIEW.md) -- components, threading, lifecycle
- [Build System](architecture/BUILD_SYSTEM.md) -- toolchain, dependencies, build commands
- [Data Flows](architecture/DATA_FLOWS.md) -- end-to-end data flow traces

### Modules

- [SDL Audio Core](modules/sdl-audio-core/OVERVIEW.md) -- C++ audio engine with pybind11 bindings
- [Recorder](modules/recorder/OVERVIEW.md) -- recording orchestration and signal generation
- [Signal Processing](modules/signal-processing/OVERVIEW.md) -- cycle extraction, alignment, averaging
- [GUI](modules/gui/OVERVIEW.md) -- Streamlit panels and series worker
- [Dataset & Collection](modules/dataset/OVERVIEW.md) -- scenario management and dataset structure
- [ML Pipeline](modules/ml-pipeline/OVERVIEW.md) -- feature extraction and classification
- [ESPRIT](modules/esprit/OVERVIEW.md) -- modal identification algorithm

### Development

- [Testing](development/TESTING.md) -- test inventory and usage
- [Work in Progress](development/WORK_IN_PROGRESS.md) -- active investigations

### Guides

- [Quick Start](guides/QUICK_START.md) -- prerequisites, setup, first run
- [Configuration](guides/CONFIGURATION.md) -- recorderConfig.json reference
