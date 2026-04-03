# Data Flows

## 1. Recording Flow

Single measurement from signal generation through storage.

```mermaid
sequenceDiagram
    participant GUI as Streamlit GUI
    participant DC as DatasetCollector
    participant REC as RoomResponseRecorder
    participant SDL as sdl_audio_core
    participant SP as SignalProcessor
    participant FS as Filesystem

    GUI->>DC: collect measurement
    DC->>REC: take_record()
    REC->>REC: generate pulse-train signal
    REC->>SDL: start_synchronized_record_playback(signal)
    SDL-->>SDL: play signal + capture mic input
    SDL->>REC: recorded_data (numpy array)
    REC->>SP: process recording
    SP->>SP: reshape into cycles
    SP->>SP: align cycles (onset/correlation)
    SP->>SP: normalize + average
    SP->>REC: impulse_response, room_response
    REC->>DC: measurement results
    DC->>FS: save raw, impulse, room_response WAV/NPY
    DC->>FS: update metadata JSON
```

## 2. Signal Processing Pipeline

```
Raw recording (N samples, C channels)
    |
    v
Reshape into cycles (num_pulses x cycle_samples x channels)
    |
    v
Cycle alignment
    ├── Standard mode: cross-correlation alignment
    └── Calibration mode: calibration channel onset detection
    |
    v
Normalization (per-cycle or calibration-based)
    |
    v
Averaging (across valid cycles, skip warm-up)
    |
    v
Truncation (optional, configurable IR length + fade)
    |
    v
Output: impulse_response per channel
```

## 3. Multi-Channel Data Path

For professional audio interfaces (e.g., Behringer UMC1820):

```
sdl_audio_core captures N channels simultaneously
    |
    v
RoomResponseRecorder receives interleaved multi-channel data
    |
    v
SignalProcessor processes each channel independently
    |
    v
Calibration channel (if configured) used for:
    - Onset alignment reference
    - Amplitude normalization
    |
    v
Per-channel impulse responses saved as:
    impulse_<scenario>_IDX_TIMESTAMP_chN.npy
```

Key config fields in `recorderConfig.json`:

| Field | Purpose |
|-------|---------|
| `multichannel_config.enabled` | Enable multi-channel recording |
| `multichannel_config.num_channels` | Total input channels |
| `multichannel_config.calibration_channel` | Channel used for alignment reference |
| `multichannel_config.reference_channel` | Channel used for normalization |
| `multichannel_config.response_channels` | Channels containing response data |

## 4. Feature Extraction Flow

```mermaid
graph TD
    DS[Scenario folder<br/>impulse_responses/] --> FE[AudioFeatureExtractor]
    FE -->|MFCC| CSV1[features.csv]
    FE -->|Spectrum| CSV2[spectrum.csv]
    CSV1 --> SC[ScenarioClassifier]
    CSV2 --> SC
    SC --> MODEL[Trained model<br/>SVM / LogReg]
    MODEL --> PRED[Prediction on<br/>new samples]
```

## 5. Series Collection Flow

The `SeriesWorker` runs as a daemon thread for unattended multi-scenario collection.

```mermaid
stateDiagram-v2
    [*] --> INIT
    INIT --> PRECHECK: validate config
    PRECHECK --> RUNNING: pre-delay countdown
    RUNNING --> PAUSED: pause command
    PAUSED --> RUNNING: resume command
    RUNNING --> STOPPING: stop command / all done
    STOPPING --> DONE
    DONE --> [*]

    state RUNNING {
        [*] --> NextScenario
        NextScenario --> Recording: start measurements
        Recording --> InterDelay: scenario complete
        InterDelay --> NextScenario: next scenario
        Recording --> Recording: measurement loop
    }
```

## 6. ESPRIT Analysis Flow

```
Multi-channel impulse responses (NPY files)
    |
    v
Preprocessing (optional contact removal, windowing)
    |
    v
Build Hankel matrix (single or multi-channel stacked)
    |
    v
SVD (CPU or GPU via CuPy)
    |
    v
Model order estimation (singular value gap)
    |
    v
ESPRIT shift-invariance -> complex poles
    |
    v
Extract: frequencies, damping ratios, mode shapes
    |
    v
Stabilization diagram (optional, multi-order sweep)
```
