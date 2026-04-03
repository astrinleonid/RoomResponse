# Configuration

All recording parameters are stored in `recorderConfig.json`.

## Signal Parameters

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `sample_rate` | int | 48000 | Audio sample rate (Hz) |
| `pulse_duration` | float | 0.008 | Pulse width (seconds) |
| `pulse_fade` | float | 0.0001 | Fade in/out duration (seconds) |
| `cycle_duration` | float | 0.1 | Full cycle length (seconds) |
| `num_pulses` | int | 8 | Number of pulses per recording |
| `volume` | float | 0.4 | Output volume (0.0 - 1.0) |
| `pulse_frequency` | float | 1000 | Pulse frequency (Hz) |
| `impulse_form` | string | `"sine"` | Pulse shape: `sine`, `square`, `voice_coil` |

## Device Selection

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `input_device` | int | -1 | Input device index (-1 = system default) |
| `output_device` | int | -1 | Output device index (-1 = system default) |

## Scenario Metadata

| Field | Type | Description |
|-------|------|-------------|
| `computer` | string | Computer/location identifier |
| `room` | string | Room identifier |
| `num_measurements` | int | Measurements per scenario |

## Multi-Channel Configuration

`multichannel_config` object:

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `enabled` | bool | false | Enable multi-channel mode |
| `num_channels` | int | 1 | Total input channels |
| `channel_names` | list[str] | `["Channel 0"]` | Display names |
| `calibration_channel` | int/null | null | Channel for alignment reference |
| `reference_channel` | int | 0 | Channel for normalization |
| `response_channels` | list[int] | `[0]` | Channels containing response data |
| `normalize_by_calibration` | bool | false | Normalize by calibration channel |
| `alignment_correlation_threshold` | float | 0.45 | Minimum correlation for alignment |
| `alignment_target_onset_position` | int | 0 | Target onset sample position |

## Series Configuration

`series_config` object:

| Field | Type | Description |
|-------|------|-------------|
| `record_extra_time_ms` | float | Extra recording time after signal ends |
| `averaging_start_cycle` | int | First cycle to include in averaging |

## Calibration Quality Configuration

`calibration_quality_config` object:

| Field | Type | Description |
|-------|------|-------------|
| `min_negative_peak` / `max_negative_peak` | float | Acceptable negative peak range |
| `min_positive_peak` / `max_positive_peak` | float | Acceptable positive peak range |
| `min_aftershock` / `max_aftershock` | float | Acceptable aftershock range |
| `aftershock_window_ms` | float | Window for aftershock detection (ms) |
| `aftershock_skip_ms` | float | Skip period after main peak (ms) |
| `min_valid_cycles` | int | Minimum cycles passing quality check |
| `max_precursor_ratio` | float | Max pre-peak energy ratio |
| `min/max_negative_peak_width_ms` | float | Acceptable peak width range |
| `min/max_first_positive_time_ms` | float | Timing of first positive peak |
| `max_first_positive_ratio` | float | Amplitude ratio threshold |
| `max_highest_positive_ratio` | float | Amplitude ratio threshold |
| `max_secondary_negative_ratio` | float | Secondary negative peak limit |
| `secondary_negative_window_ms` | float | Window for secondary detection |

## Truncation Configuration

`truncate_config` object:

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `enabled` | bool | true | Enable IR truncation |
| `ir_working_length_ms` | float | 600.0 | Working IR length (ms) |
| `ir_fade_length_ms` | float | 20.0 | Fade-out length (ms) |

## Save Format

`save_format` object:

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `save_wav` | bool | false | Save as WAV files |
| `save_npy` | bool | true | Save as NumPy binary files |

## Example: Minimal Config

```json
{
  "sample_rate": 48000,
  "pulse_duration": 0.008,
  "cycle_duration": 0.1,
  "num_pulses": 8,
  "volume": 0.4,
  "pulse_frequency": 1000,
  "impulse_form": "sine",
  "computer": "MyPC",
  "room": "Studio"
}
```

## Example: Multi-Channel Piano Config

```json
{
  "sample_rate": 48000,
  "pulse_duration": 0.019,
  "pulse_fade": 0.018,
  "cycle_duration": 1.0,
  "num_pulses": 5,
  "volume": 0.65,
  "impulse_form": "voice_coil",
  "multichannel_config": {
    "enabled": true,
    "num_channels": 6,
    "calibration_channel": 2,
    "reference_channel": 5,
    "response_channels": [0, 1, 3, 4, 5],
    "normalize_by_calibration": true
  },
  "truncate_config": {
    "enabled": true,
    "ir_working_length_ms": 600.0,
    "ir_fade_length_ms": 20.0
  },
  "save_format": {
    "save_wav": false,
    "save_npy": true
  }
}
```
