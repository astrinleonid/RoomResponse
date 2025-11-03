# Multi-Channel Room Response System

**Document Version:** 4.2
**Created:** 2025-10-31
**Last Updated:** 2025-11-03 (cleaned up legacy content)
**Target:** Piano Response Measurement System
**Status:** Core Implementation Complete | GUI Integration Complete | Pipeline Refactored ✅ | Series Settings Calibration Mode Complete ✅

---

## Executive Summary

The Room Response system has been upgraded to support **synchronized multi-channel impulse response recording** with calibration-driven quality validation. The system can record from 1-32 channels simultaneously while maintaining sample-perfect synchronization across all channels.

**Implementation Status:**
- ✅ Multi-channel audio recording (C++/Python)
- ✅ Signal processing pipeline (single & multi-channel)
- ✅ Signal processing extracted to SignalProcessor class (clean architecture)
- ✅ Calibration quality validation system (V2 - comprehensive 7-criteria)
- ✅ Multi-channel file management
- ✅ Calibration testing GUI interface (Calibration Impulse panel)
- ✅ Multi-channel configuration GUI (fully implemented)
- ✅ Configuration profile management system (save/load/delete)
- ✅ Collection Panel multi-channel status display
- ✅ Calibration-based normalization system
- ✅ Multi-channel response review GUI (interactive visualization)
- ✅ Series Settings calibration mode integration (complete with cycle overlay, statistics)
- ✅ Code cleanup and refactoring (Phase 6 complete)
- ⚠️ Collect Panel calibration mode (needs review - may already be implemented)
- ❌ Scenarios Panel analysis integration (needs implementation)
- ❌ Full multi-channel scenario visualization GUI (planned for future)

---

## System Architecture

### Core Components

```
┌─────────────────────────────────────────────────────────────────┐
│                    CORE SIGNAL PROCESSING                        │
│                   RoomResponseRecorder.py                        │
│                     (1,385 lines, 31 methods)                    │
└─────────────────────────────────────────────────────────────────┘
                              ↓
        ┌─────────────────────┼─────────────────────┐
        ↓                     ↓                     ↓
  ┌──────────┐         ┌───────────┐        ┌──────────────┐
  │ SDL Core │         │ Validator │        │ File Manager │
  │  (C++)   │         │   (V2)    │        │   (Python)   │
  └──────────┘         └───────────┘        └──────────────┘
   Multi-channel      Calibration          Multi-channel
   recording          quality checking     file I/O

        ↓                     ↓                     ↓
  ┌──────────────────────────────────────────────────────┐
  │              CONFIGURATION SYSTEM                     │
  │           recorderConfig.json (Persistent)            │
  │   ├─ Recorder settings (pulse, cycle, volume)        │
  │   ├─ Multichannel config (channels, names, ref)      │
  │   └─ Calibration thresholds (min/max ranges)         │
  └──────────────────────────────────────────────────────┘

        ↓                     ↓                     ↓
  ┌──────────────────────────────────────────────────────┐
  │                   GUI LAYER                           │
  │   ├─ AudioSettingsPanel (79 KB) ✅                   │
  │   │   ├─ Device selection                             │
  │   │   ├─ Multi-channel configuration ✅              │
  │   │   ├─ Calibration Impulse testing ✅              │
  │   │   └─ Series Settings ✅ (calibration mode complete)│
  │   ├─ CollectionPanel (23 KB) ✅                      │
  │   │   └─ Multi-channel status display ✅             │
  │   ├─ ConfigProfileManager (NEW) ✅                   │
  │   │   └─ Save/load/delete profiles ✅                │
  │   ├─ ScenariosPanel (72 KB)                           │
  │   └─ SeriesWorker (background recording)             │
  └──────────────────────────────────────────────────────┘
```

---

## Signal Processing Pipeline

### Universal Three-Stage Architecture (Refactored ✅)

The system now implements a **clean three-stage universal pipeline** where Stages 1 and 3 are mode-independent:

**Pipeline Design:**
- **Stage 1 (Recording):** Universal - same for all modes
- **Stage 2 (Processing):** Mode-specific - different logic per mode
- **Stage 3 (Saving):** Universal - same file structure for all modes

---

#### PATH 1: Standard Mode (Averaged Room Response)

```
User calls: recorder.take_record(output_file, impulse_file, mode='standard')

    ↓
STAGE 1: Recording (UNIVERSAL)
    _record_audio()  # Renamed from _record_method_2
    ├─ If multichannel_config['enabled'] == False:
    │   sdl_audio_core.measure_room_response_auto()
    │   → Returns: np.ndarray [samples]
    │
    └─ If multichannel_config['enabled'] == True:
        sdl_audio_core.measure_room_response_auto_multichannel()
        → Returns: Dict[int, np.ndarray] {0: ch0_data, 1: ch1_data, ...}

    ↓
STAGE 2: Processing (MODE-SPECIFIC)
    _process_recorded_signal(recorded_audio)
    ├─ If isinstance(recorded_audio, np.ndarray):  # Single-channel
    │   _process_single_channel_signal()
    │   ├─ _extract_cycles() → [num_pulses, cycle_samples]
    │   ├─ _average_cycles(skip_first_25%) → [cycle_samples]  ← Averaging HERE
    │   └─ _extract_impulse_response() → [cycle_samples]
    │
    └─ If isinstance(recorded_audio, dict):  # Multi-channel
        _process_multichannel_signal()
        ├─ Process reference channel:
        │   ├─ _extract_cycles(ref_audio)
        │   ├─ _average_cycles(ref_cycles)  ← Averaging HERE
        │   └─ _find_onset_in_room_response() → onset_sample
        │
        └─ Process all channels with SAME shift:
            FOR each channel:
                ├─ _extract_cycles(channel_audio)
                ├─ _average_cycles(channel_cycles)  ← Averaging HERE
                └─ np.roll(room_response, -onset_sample)  # Same shift!

    → Returns: {
          'raw': Dict[int, np.ndarray],        # Original audio
          'room_response': Dict[int, np.ndarray],  # Averaged responses
          'impulse': Dict[int, np.ndarray],    # Impulse responses
          'metadata': Dict                     # Processing metadata
      }

    ↓
STAGE 3: File Saving (UNIVERSAL)
    _save_processed_data(processed_data, output_file, impulse_file)
    └─ _save_multichannel_files() OR _save_single_channel_files()
        ├─ raw_000_TIMESTAMP_ch0.wav
        ├─ impulse_000_TIMESTAMP_ch0.wav
        ├─ room_raw_000_room_TIMESTAMP_ch0.wav
        ├─ raw_000_TIMESTAMP_ch1.wav
        ├─ ...

    ↓
Returns: recorded_audio (np.ndarray or Dict[int, np.ndarray])
         BACKWARD COMPATIBLE - raw audio only
```

**Key Points:**
- ⚠️ **No cycle alignment** - assumes perfect timing from audio engine
- Cycles averaged "as is" without onset detection
- Works well for synthetic pulses with precise timing

#### PATH 2: Calibration Mode (Quality-Validated Averaged Response) ✅ REFACTORED

```
User calls: recorder.take_record(output_file, impulse_file, mode='calibration', save_files=True)
         OR: recorder.take_record_calibration()  # No files (testing only)

    ↓
STAGE 1: Recording (UNIVERSAL - Identical to Standard)
    _record_audio()
    → Returns: Dict[int, np.ndarray] (multichannel required)

    ↓
STAGE 2: Processing (MODE-SPECIFIC - Quality Pipeline)
    _process_calibration_mode(recorded_audio)

    STEP 1: Validate cycles
        Extract calibration channel:
            cal_raw = recorded_audio[calibration_channel]
            initial_cycles = _extract_cycles(cal_raw)  ✅ Uses helper

        Validate each cycle:
            validator = CalibrationValidatorV2(thresholds, sample_rate)
            FOR each cycle in initial_cycles:
                validation = validator.validate_cycle(cycle, index)
                ├─ Check negative peak in range [min, max]
                ├─ Check positive peak in range [min, max]
                ├─ Check aftershock in range [min, max]
                └─ Mark as valid/invalid

    STEP 2: Align cycles by onset
        alignment_result = align_cycles_by_onset(
            initial_cycles,
            validation_results,
            correlation_threshold=config['alignment_correlation_threshold']  # Tunable!
        )
        ├─ Filter to valid cycles only
        ├─ Detect onset in each valid cycle
        ├─ Align all to target position (config['alignment_target_onset_position'])  # Tunable!
        └─ Cross-correlation filtering (removes outliers)

    STEP 3: Apply alignment to all channels
        FOR each channel in recorded_audio:
            aligned_cycles[ch] = apply_alignment_to_channel(
                channel_data,
                alignment_result  # Uses SAME shifts from calibration channel
            )

    STEP 4: Normalize by impact magnitude (optional)
        IF config['normalize_by_calibration'] == True:
            FOR each cycle, each response channel:
                normalized[cycle] = aligned[cycle] / |negative_peak[cycle]|

    STEP 5: Average processed cycles ✅ NEW!
        FOR each channel:
            room_response[ch] = np.mean(processed_cycles[ch], axis=0)

        ← Averages aligned/normalized cycles (no skipping - already filtered)

    STEP 6: Output impulse responses
        FOR each channel:
            impulse[ch] = room_response[ch]  # Onset already at target position

    → Returns: {
          'raw': Dict[int, np.ndarray],        # Original audio
          'room_response': Dict[int, np.ndarray],  # Averaged aligned/normalized
          'impulse': Dict[int, np.ndarray],    # Impulse responses
          'metadata': Dict,                    # Processing metadata
          # BACKWARD COMPATIBILITY for GUI:
          'calibration_cycles': np.ndarray,
          'validation_results': List[Dict],
          'alignment_metadata': Dict,
          'aligned_multichannel_cycles': Dict[int, np.ndarray]
      }

    ↓
STAGE 3: File Saving (UNIVERSAL - Same as Standard Mode) ✅ NEW!
    IF save_files == True:
        _save_processed_data(processed_data, output_file, impulse_file)
        └─ _save_multichannel_files()
            ├─ raw_000_TIMESTAMP_ch0.wav       # Original recording
            ├─ impulse_000_TIMESTAMP_ch0.wav   # Averaged response
            ├─ room_raw_000_room_TIMESTAMP_ch0.wav
            ├─ raw_000_TIMESTAMP_ch1.wav
            ├─ ...

    ↓
Returns: Dict with averaged responses + cycle-level data (backward compatible)
```

**Key Improvements:**
- ✅ **Per-cycle alignment** - robust for physical impacts with timing variance
- ✅ **Quality validation** - filters out bad cycles before averaging
- ✅ **Averaging after alignment** - produces clean averaged response
- ✅ **Universal file structure** - same format as standard mode
- ✅ **Tunable parameters** - correlation threshold and onset position configurable

---

### Key Architecture Features ✅

**Universal Three-Stage Design:**
- Stage 1 (Recording): Mode-independent, handles single/multi-channel
- Stage 2 (Processing): Mode-specific (standard vs calibration)
- Stage 3 (Saving): Universal file structure for both modes

**Quality Features:**
- Tunable calibration parameters (correlation threshold, onset position)
- Optional calibration-based normalization
- SignalProcessor class for clean signal processing architecture
- 7-criteria validation system with comprehensive metrics

---

### Critical Synchronization Requirement

**All channels maintain sample-perfect synchronization:**

When the system finds the impulse onset at sample position N in the reference channel, **ALL channels** from that measurement are shifted by exactly the same number of samples (-N). This preserves:
- Inter-channel phase relationships
- Time-of-arrival differences between microphones
- Spatial information in multi-mic recordings

**Implementation:** Lines 753-801 in `RoomResponseRecorder.py`

---

## Configuration System

### recorderConfig.json Structure

**Current Active Configuration (User's System):**

```json
{
  "sample_rate": 48000,
  "pulse_duration": 0.019,
  "pulse_fade": 0.018,
  "cycle_duration": 0.5,
  "num_pulses": 4,
  "volume": 0.15,
  "pulse_frequency": 1000.0,
  "impulse_form": "voice_coil",
  "input_device": 5,
  "output_device": 6,

  "multichannel_config": {
    "enabled": true,                    // ✅ Multi-channel ACTIVE
    "num_channels": 8,                  // Recording 8 channels
    "channel_names": [
      "Channel 0", "Channel 1", "Channel 2", "Channel 3",
      "Channel 4", "Channel 5", "Channel 6", "Channel 7"
    ],
    "calibration_channel": 2,           // Ch2 = calibration sensor
    "reference_channel": 5,             // Ch5 = alignment reference
    "response_channels": [0,1,3,4,5,6,7], // All except calibration

    // ✅ NEW: Tunable calibration parameters
    "alignment_correlation_threshold": 0.7,     // Min correlation for cycle filtering
    "alignment_target_onset_position": 100,     // Target onset sample position
    "normalize_by_calibration": false           // Enable impact magnitude normalization
  }
}
```

**Configuration Loading:**
- File location: `recorderConfig.json` (root directory)
- Loaded by: `RoomResponseRecorder.__init__(config_file_path)`
- Fallback: Uses default config if file not found
- Supports legacy format: `multichannel` → `multichannel_config` (auto-migration)

**Key Fields:**

| Field | Purpose | Default | User's Value |
|-------|---------|---------|--------------|
| `enabled` | Toggle multi-channel mode | `false` | `true` ✅ |
| `num_channels` | Total input channels | `1` | `8` |
| `calibration_channel` | Channel with calibration sensor | `None` | `2` |
| `reference_channel` | Channel for onset alignment | `0` | `5` |
| `response_channels` | Channels to process (excl. calibration) | `[0]` | `[0,1,3,4,5,6,7]` |
| `alignment_correlation_threshold` ✅ NEW | Min correlation for cycle filtering | `0.7` | `0.7` |
| `alignment_target_onset_position` ✅ NEW | Target onset sample position | `100` | `100` |
| `normalize_by_calibration` ✅ NEW | Enable impact magnitude normalization | `false` | `false` |

---

## File Management

### Naming Convention

**Single-Channel (Legacy):**
```
raw_000_20251031_143022.wav
impulse_000_20251031_143022.wav
room_response_000_20251031_143022.wav
```

**Multi-Channel (New):**
```
raw_000_20251031_143022_ch0.wav
raw_000_20251031_143022_ch1.wav
raw_000_20251031_143022_ch2.wav
...
impulse_000_20251031_143022_ch0.wav
impulse_000_20251031_143022_ch1.wav
...
room_response_000_20251031_143022_ch0.wav
room_response_000_20251031_143022_ch1.wav
...
```

**Pattern:** `{type}_{index}_{timestamp}_ch{N}.wav`

### File Management Utilities

**multichannel_filename_utils.py** (7.6 KB)

Functions:
- `parse_multichannel_filename(filename)` → `ParsedFilename(type, index, timestamp, channel)`
- `group_files_by_measurement(files)` → `Dict[index, List[files]]`
- `group_files_by_channel(files)` → `Dict[channel, List[files]]`
- `detect_num_channels(files)` → `int`

**ScenarioManager.py** (28 KB) - Extended with multi-channel support

New methods:
- `is_multichannel_scenario(scenario_path)` → `bool`
- `detect_num_channels_in_scenario(scenario_path)` → `int`
- `get_measurement_files_from_scenario(scenario_path, measurement_index, file_type)` → `Dict[channel, filepath]`

---

## Calibration Quality Validation

### CalibrationValidatorV2 (Current - V3 Comprehensive Format)

**File:** `calibration_validator_v2.py` (25 KB)
**Replaced:** `calibration_validator.py` (V1, deprecated 2025-10-30)
**Updated to V3:** 2025-11-01
**Status:** ✅ Production Ready

**Validation Approach: Comprehensive 7-Criteria System**

V3 uses a comprehensive validation system with 11 parameters covering 7 quality criteria:

```python
@dataclass
class QualityThresholds:
    # 1. Negative Peak Range (absolute amplitude)
    min_negative_peak: float
    max_negative_peak: float

    # 2. Precursor (peaks before negative peak)
    max_precursor_ratio: float  # Relative to negative peak

    # 3. Negative Peak Width
    min_negative_peak_width_ms: float  # Width at 50% amplitude
    max_negative_peak_width_ms: float

    # 4. First Positive Peak After Negative
    max_first_positive_ratio: float  # Relative to negative peak

    # 5. First Positive Peak Timing
    min_first_positive_time_ms: float
    max_first_positive_time_ms: float

    # 6. Highest Positive Peak After Negative
    max_highest_positive_ratio: float  # Relative to negative peak

    # 7. Secondary Negative Peak (replaces aftershock)
    max_secondary_negative_ratio: float  # Relative to main negative
    secondary_negative_window_ms: float
```

**7 Quality Criteria:**

1. **Negative Peak Range**: Ensures impact amplitude is within acceptable bounds
2. **Precursor**: Detects pre-impact vibrations (excludes 1ms rise time)
3. **Negative Peak Width**: Validates impulse duration (width at 50% amplitude)
4. **First Positive Peak Magnitude**: Checks initial rebound intensity
5. **First Positive Peak Timing**: Validates rebound timing
6. **Highest Positive Peak**: Monitors maximum positive excursion
7. **Secondary Negative Peak**: Detects hammer bounces/rebounds

**Validation Logic:**

For each calibration cycle:
1. Find negative peak and its properties
2. Check for precursors (before negative peak, excluding rise time)
3. Measure negative peak width at 50% amplitude
4. Locate first positive peak after negative
5. Measure timing from negative to first positive
6. Find highest positive peak in entire response
7. Search for secondary negative peaks (bounces)
8. Mark cycle as valid only if ALL 7 criteria pass

**Threshold Learning Workflow:**

GUI allows user to:
1. Run calibration test → Get initial cycles with full metrics
2. Visually inspect waveforms with detailed validation results
3. Mark 3+ "good" cycles with checkboxes
4. Click "Calculate Thresholds" → System analyzes all 11 parameters
5. Thresholds saved to `recorderConfig.json` (V3 format)
6. Future recordings validated against comprehensive criteria

**Key Benefits:**
- ✅ Comprehensive impulse quality assessment
- ✅ Detects subtle issues (precursors, bounces, timing problems)
- ✅ User-driven quality definition with detailed metrics
- ✅ Backward compatible with V1/V2 config formats
- ✅ 5% safety margin for learned thresholds (reduced from 20%)

### Calibration-Based Normalization (NEW - 2025-11-02)

**Purpose:** Normalize response channel amplitudes by calibration signal magnitude for quantitative, reproducible measurements.

**File:** `RoomResponseRecorder.py` → `_normalize_by_calibration()`
**Status:** ✅ Implemented
**Configuration:** `multichannel_config.normalize_by_calibration` (default: False)

**Problem Solved:**
- Variations in calibration impulse magnitude across recording takes
- Differences in sensor sensitivity between channels
- Need for quantitative comparison between measurements

**Normalization Strategy:**

For each recording cycle `i`:
```
normalized_response[channel][cycle_i] = raw_response[channel][cycle_i] / |negative_peak[cycle_i]|
```

Where:
- `negative_peak[cycle_i]`: Calibration impulse magnitude (from validation results)
- Result: Response amplitude per unit impact strength

**Processing Pipeline:**

1. Record multi-channel audio
2. Extract cycles from calibration channel
3. Validate each cycle → Get `negative_peak` for each cycle
4. Align all cycles by onset
5. Apply alignment to all channels → `aligned_multichannel_cycles`
6. **Normalize response channels** (if enabled) → `normalized_multichannel_cycles`
7. Return both aligned (raw) and normalized cycles

**Output Structure:**

```python
{
    'calibration_cycles': np.ndarray,
    'validation_results': List[Dict],
    'aligned_multichannel_cycles': Dict[int, np.ndarray],      # Raw aligned
    'normalized_multichannel_cycles': Dict[int, np.ndarray],   # NEW: Calibrated
    'normalization_factors': List[float],                      # NEW: Negative peaks
    'alignment_metadata': Dict,
    'metadata': {
        'normalize_by_calibration': bool  # NEW: Flag indicating if normalization applied
    }
}
```

**Key Features:**
- ✅ **Optional:** Disabled by default, enabled via config
- ✅ **Per-cycle normalization:** Each cycle normalized by its own calibration magnitude
- ✅ **Calibration channel preserved:** Kept unnormalized (or normalized to 1.0)
- ✅ **Division-by-zero protection:** Skips cycles with peak < 1e-6
- ✅ **Dual output:** Returns both raw aligned and normalized cycles
- ✅ **Normalization factors logged:** Min/max/mean negative peaks printed

**Use Cases:**

1. **Piano Hammer Impact Studies:**
   - Measure string response per unit force
   - Compare different striking techniques
   - Normalize for varying impact strengths

2. **Room Acoustic Measurements:**
   - Normalize by source impulse magnitude
   - Compare responses across different sessions
   - Remove source strength variability

3. **Sensor Calibration:**
   - Account for sensor gain differences
   - Produce sensor-independent measurements
   - Enable cross-system comparisons

**Benefits:**
- ✅ **Quantitative Analysis:** Results in units of "response per unit impact"
- ✅ **Reproducibility:** Removes impact strength variability
- ✅ **Comparability:** Different measurements directly comparable
- ✅ **Physical Meaning:** Clear physical interpretation of results
- ✅ **Backward Compatible:** Optional feature, doesn't affect existing workflows

---

### Multi-Channel Response Review (NEW - 2025-11-02)

**Purpose:** Interactive GUI for reviewing aligned and normalized response cycles across all channels.

**File:** `gui_calibration_impulse_panel.py` → `_render_multichannel_response_review()`
**Status:** ✅ Implemented
**Location:** Calibration Impulse panel, appears after "Alignment Results Review" section

**Problem Solved:**
- Visual validation of alignment quality across response channels
- Comparison of raw vs normalized responses
- Per-cycle quality assessment via detailed metrics
- Side-by-side evaluation of normalization effectiveness

**User Interface Components:**

1. **Display Controls**
   ```
   ┌─────────────────────────────────────────────────────────┐
   │ Display Controls                                         │
   ├─────────────────────────────────────────────────────────┤
   │ Select Response Channel: [Channel 1 ▼]                  │
   │ Display Mode: ⚪ Aligned Only  ⚪ Normalized Only       │
   │               ⚪ Both (Side-by-Side)                     │
   └─────────────────────────────────────────────────────────┘
   ```

2. **Cycle Selection Table**
   - Checkbox selection for each cycle
   - Per-cycle metrics displayed:
     - Negative Peak (max absolute negative value)
     - Positive Peak (max positive value)
     - RMS (Raw) - root mean square amplitude
     - RMS (Norm) - normalized RMS (if available)
     - Norm Factor - calibration magnitude used for normalization
   - Adaptive columns: Shows normalized metrics only when available
   - Selection count display: "✓ Selected N cycle(s): 0, 1, 2..."

3. **Waveform Overlay Visualization**
   - Interactive zoom controls (persistent zoom state per channel)
   - View mode selector: Waveform / Spectrum
   - Reset zoom button
   - Analysis statistics display
   - Three display modes:
     - **Aligned Only:** Shows raw aligned cycles
     - **Normalized Only:** Shows calibration-normalized cycles
     - **Both (Side-by-Side):** Split view comparing raw vs normalized

4. **Detailed Cycle Information** (Expandable)
   - Per-cycle expandable sections
   - Side-by-side metrics:
     - **Aligned (Raw):** Original metrics
     - **Normalized:** Post-normalization metrics
   - Displays: Negative Peak, Positive Peak, RMS, Max Abs, Energy
   - Shows normalization factor used

**Implementation Details:**

**Helper Methods:**
```python
_compute_channel_cycle_metrics(cycle_data: np.ndarray) -> dict
    # Computes metrics for a single cycle
    # Returns: negative_peak, positive_peak, rms, max_abs, energy

_render_channel_cycles_table(channel_idx, channel_name, aligned_cycles,
                              normalized_cycles, normalization_factors) -> list
    # Renders checkbox table with per-cycle metrics
    # Returns list of selected cycle indices

_render_channel_cycles_overlay(selected_cycles, aligned_cycles, normalized_cycles,
                                channel_name, channel_idx, sample_rate, display_mode)
    # Renders waveform visualization based on display mode
    # Handles: Aligned Only / Normalized Only / Both Side-by-Side

_plot_cycle_overlay(selected_cycles, cycle_data, sample_rate,
                    label_prefix, component_id)
    # Plots cycle overlay using AudioVisualizer.render_multi_waveform_with_zoom()
    # Provides zoom controls, view mode, analysis display

_render_multichannel_response_review(test_results: Dict[str, Any])
    # Main integration method
    # Coordinates all sub-components
```

**Session State Management:**
- Per-channel cycle selection: `multichannel_review_selected_cycles_ch{channel_idx}`
- Channel selector: `multichannel_review_channel_selector`
- Display mode: `multichannel_review_display_mode`
- Zoom state per channel/view: `multichannel_ch{channel_idx}_{aligned|normalized}_viz_zoom_*`

**Visualization Features:**
- ✅ **Zoom Controls:** Interactive zoom with persistent state
- ✅ **View Modes:** Waveform and Spectrum display
- ✅ **Analysis Stats:** Automatic statistics display
- ✅ **Side-by-Side Comparison:** Compare raw vs normalized in split view
- ✅ **Multi-Cycle Overlay:** Plot multiple cycles simultaneously
- ✅ **Consistent UI:** Matches Alignment Results Review section

**User Workflow:**

1. Run calibration test with multi-channel enabled
2. Scroll to "Multi-Channel Response Review" section (appears after alignment review)
3. Select response channel from dropdown (excludes calibration channel)
4. Choose display mode (Aligned/Normalized/Both)
5. Review metrics table for all cycles
6. Check boxes to select cycles for visualization
7. View waveform overlays with zoom controls
8. Expand individual cycle details for in-depth analysis
9. Switch channels to review other response channels

**Integration:**
- Automatically appears when multi-channel mode enabled AND response channels exist
- Only shows if `aligned_multichannel_cycles` present in test results
- Filters out calibration channel (only shows response channels)
- Gracefully handles missing normalized data (falls back to aligned only)

**Benefits:**
- ✅ **Visual Quality Check:** Quickly assess alignment quality across channels
- ✅ **Normalization Validation:** See effect of calibration normalization
- ✅ **Per-Cycle Inspection:** Detailed metrics for individual cycles
- ✅ **Interactive Exploration:** Zoom into specific regions of interest
- ✅ **Quantitative Metrics:** Numerical values for objective assessment
- ✅ **Multi-Channel Support:** Review all response channels systematically

---

## GUI Implementation Status

### Implemented GUIs

#### 1. AudioSettingsPanel (79 KB) - ✅ COMPLETE

**File:** `gui_audio_settings_panel.py`

**Tabs:**
- **Device Selection:** Audio input/output device picker
- **Calibration Impulse:** ✅ **Multi-channel calibration testing**
  - Run calibration test (calls `recorder.take_record_calibration()`)
  - Display per-cycle validation results
  - Checkbox-based cycle selection
  - Automatic threshold learning
  - Manual threshold editing (tabular form)
  - Unified waveform visualization (1-N cycles)
  - Save thresholds to config
- **Series Settings:** Multi-pulse configuration (pulse duration, cycle duration, num pulses)

**Multi-Channel Features:**
- ✅ Calibration quality testing with V3 validator (7 criteria)
- ✅ Per-cycle validation metrics display (all 11 parameters)
- ✅ Multi-cycle waveform visualization
- ✅ Threshold learning from user-selected cycles
- ✅ **Multi-channel configuration UI** (enable/disable, channel setup)
- ✅ Device capability detection
- ✅ Per-channel naming and role assignment
- ✅ **Calibration-based normalization** (normalize response by calibration magnitude)
- ✅ **Multi-Channel Response Review** (interactive review of aligned/normalized cycles)

#### 2. SeriesSettingsPanel (19 KB)

**File:** `gui_series_settings_panel.py`

**Features:**
- Configure multi-pulse recording parameters
- Test recording and analysis
- Cycle consistency overlay plots
- Saves configuration to `recorderConfig.json`

**Multi-Channel Status:**
- ✅ Works with multi-channel recorder
- ✅ Uses `recorder.take_record()` (respects multichannel_config)
- ✅ Multi-channel configuration UI in Device Selection tab

#### 3. SeriesWorker (Background Recording)

**File:** `gui_series_worker.py`

**Features:**
- Background thread for series recording
- Multi-scenario support
- Measurement interval control
- Warm-up measurements
- File-based pause/stop control

**Multi-Channel Status:**
- ✅ Calls `recorder.take_record()` → Respects multichannel_config
- ✅ Saves multi-channel files if enabled
- ❌ No multi-channel-specific UI

#### 4. CollectionPanel (23 KB)

**File:** `gui_collect_panel.py`

**Multi-Channel Status:**
- ✅ **Multi-channel status display** (implemented in Phase 7)
- ✅ **Per-channel indicators** (shows all active channels)
- ✅ **Visual channel status** (green badges per channel)

### Missing GUIs

#### Multi-Channel Configuration Interface ✅ IMPLEMENTED

**Location:** AudioSettingsPanel → Device Selection tab → Multi-Channel Configuration section

**Status:** ✅ FULLY FUNCTIONAL (Implemented Phase 5)

**Features Implemented:**
- ✅ Enable/disable multi-channel recording toggle
- ✅ Device capability detection (max input channels)
- ✅ Number of channels selector (1-32, constrained by device)
- ✅ Per-channel name editing
- ✅ Reference channel selector
- ✅ Calibration channel selector
- ✅ Configuration profiles (save/load/delete named presets)
- ✅ Save to recorderConfig.json
- ✅ Real-time validation

#### Multi-Channel Scenario Visualization ❌ NOT IMPLEMENTED

**Should Exist In:** AudioAnalysisPanel or new panel

**Required Features:**
- Load multi-channel files from scenario
- Display stacked waveform plots (one per channel)
- Channel selection checkboxes (show/hide)
- Synchronized zoom/pan across channels
- Per-channel statistics (max amplitude, RMS, etc.)
- Cross-correlation display (verify synchronization)

**Effort Estimate:** 6-8 hours

---

## Current Usage Patterns

### 1. Calibration Testing (Audio Settings Panel) ✅ WORKING

**User Workflow:**
1. Open Audio Settings → Calibration Impulse tab
2. Ensure `multichannel_config.enabled = true` in JSON (manual edit)
3. Click "Run Calibration Test"
4. System calls `recorder.take_record_calibration()`
5. Displays per-cycle validation results
6. User selects "good" cycles with checkboxes
7. Click "Calculate and Apply Thresholds"
8. Thresholds saved to config

**Code Path:**
```
gui_audio_settings_panel.py:1291
    → recorder.take_record_calibration()
        → RoomResponseRecorder._take_record_calibration_mode()
            → _record_method_2() [multi-channel]
            → CalibrationValidatorV2.validate_cycle() [per cycle]
            → align_cycles_by_onset()
            → apply_alignment_to_channel() [all channels]
            → Returns cycle data dict
```

**Status:** ✅ FULLY FUNCTIONAL

### 2. Series Recording (Series Settings Panel) ✅ FULLY IMPLEMENTED

**User Workflow:**
1. Open Audio Settings → Series Controls tab
2. Configure pulse parameters (num_pulses, duration, frequency, etc.)
3. Select recording mode: **Standard** or **Calibration** (default: Calibration)
4. Click "🎵 Record Series"
5. Recording & Analysis section appears below with:
   - Cycle Statistics table (for calibration mode)
   - Cycle selection checkboxes
   - Cycle overlay chart with zoom controls
   - Display mode selector (Aligned/Normalized/Both)
   - Overlay statistics (peak range, range width %, std dev)

**Code Path:**
```
gui_series_settings_panel.py
    → _execute_series_recording()
        → recorder.take_record(output_file, impulse_file, method=2, mode=recording_mode)
            → RoomResponseRecorder.take_record(mode='calibration' or 'standard')
                → _record_audio() [universal recording stage]
                → _process_calibration_mode() [if mode='calibration']
                    → Validation → Alignment → Normalization
                → _process_standard_mode() [if mode='standard']
                    → Averaging → Spectral analysis
                → _save_multichannel_files() [if multichannel enabled]
                → Returns complete processed dict
```

**Status:**
- ✅ Single-channel recording: TESTED, WORKING
- ✅ Multi-channel recording: TESTED, WORKING
- ✅ Calibration mode integration: COMPLETE
- ✅ Cycle overlay visualization: COMPLETE (with zoom controls)
- ✅ Display mode selector: COMPLETE (Aligned/Normalized/Both)
- ✅ Cycle statistics display: COMPLETE (7-criteria validation results)
- ✅ Overlay statistics: COMPLETE (peak range, range width %, std dev)
- ✅ SignalProcessor integration: COMPLETE (settings auto-update processor)

### 3. Background Series Recording (Series Worker) ⚠️ PARTIALLY WORKING

**User Workflow:**
1. Series Worker runs in background thread
2. Calls `recorder.take_record()` periodically
3. Saves files automatically

**Code Path:**
```
gui_series_worker.py:302
    → recorder.take_record(raw_path, impulse_path, method=2, interactive=False)
        → [Same as Series Recording above]
```

**Status:** Same as Series Recording (works if JSON configured)

**Note:** `interactive=False` parameter is **not defined in signature** - likely ignored or legacy parameter

---

## Implementation History

**Summary:** Multi-channel system developed October-November 2025 in 7 phases.

### Core Implementation (Phases 1-4, Oct 2025) ✅
- SDL audio core multi-channel recording (C++)
- Python recording pipeline with calibration validation
- Multi-channel file management system
- GUI calibration interface with threshold learning

### Refactoring & Enhancement (Phases 5-7, Nov 2025) ✅
- Code cleanup (removed V1 validator, unified cycle extraction)
- SignalProcessor class extraction for clean architecture
- Configuration profile management
- Multi-channel response review GUI
- Series Settings calibration mode integration

**Current Status:** Core system complete and operational. All planned features implemented except Scenarios Panel integration.

---

## Roadmap

### Immediate: Scenarios Panel Integration (High Priority)

**Task:** Add Series Recording Analysis capabilities to Scenarios Panel for reviewing saved calibration recordings.

**Features:**
- Load NPZ files with aligned/normalized cycles
- Cycle Statistics table with validation results
- Cycle selection and overlay visualization
- Display mode selector (Aligned/Normalized/Both)
- Overlay statistics (peak range, width %, std dev)

**Effort:** 1-2 days

### Future: Extensible Validation System (Phase 6.5, Optional)

Plugin-based calibration validation allowing custom metrics and validation logic. See archived documentation for detailed design.

**Effort:** 2 weeks

### Phase 7: Multi-Channel GUI Integration ✅ **COMPLETE** (2025-11-01 to 2025-11-02)

**Task 1: Configuration Profile Management** ✅ **DONE** (commit 9da691f)
- ✅ Added ConfigProfileManager to sidebar (both `piano_response.py` and `gui_launcher.py`)
- ✅ Save/load/delete named configuration profiles
- ✅ **NEW FILE:** `gui_config_profiles.py` (407 lines)
- ✅ Profile storage in `configs/` directory with JSON files
- ✅ Includes: recorder settings, multichannel config, calibration thresholds
- ✅ In-memory recorder update on profile load (no restart required)
- ✅ Profile metadata tracking (creation date, description)
- ✅ Quick switch between configurations via dropdown
- **Sidebar UI Features:**
  ```
  📋 Active Profile Display
  📂 Profile Selector (dropdown)
  💾 Save Profile button
  📂 Load Profile button
  🗑️ Delete Profile button
  📁 Profile count display
  ```

**Task 2: Multi-Channel Configuration Interface** ✅ **ALREADY EXISTED**
- ✅ Multi-Channel Configuration section in Audio Settings → Device Selection tab
- ✅ Enable/disable toggle
- ✅ Channel count input (1-32)
- ✅ Device capability detection
- ✅ Per-channel naming with text inputs
- ✅ Reference/calibration channel selectors
- ✅ Save to config file button
- ✅ Configuration validation
- ✅ Channel role indicators with icons (🔨 🎤 🔊)
- **Location:** `gui_audio_settings_panel.py` → `_render_multichannel_configuration()`

**Task 3: Collection Panel Multi-Channel Status** ✅ **DONE** (commit a47ad09)
- ✅ Added `_render_recorder_status()` to CollectionPanel
- ✅ Expandable "📊 Recorder Configuration" section
- ✅ Displays recording mode (single/multi-channel indicator)
- ✅ Shows channel configuration with names and roles
- ✅ Per-channel role indicators (🔨 Calibration, 🎤 Reference, 🔊 Response)
- ✅ Recording parameters summary (sample rate, pulses, cycle duration)
- ✅ Direct link to Audio Settings configuration UI

**Task 4: Multi-Channel Visualization** ❌ **NOT IMPLEMENTED** (Future Work)
- Planned features:
  - Load multi-channel files from scenarios
  - Stacked waveform plots (one per channel)
  - Channel show/hide checkboxes
  - Synchronized zoom/pan
  - Per-channel statistics table
  - Cross-correlation display
- **Estimated effort:** 6-8 hours
- **Priority:** Low (can record and save multi-channel, visualization optional)

---

## Technical Specifications

### Multi-Channel Capabilities

| Feature | Specification | Status |
|---------|--------------|--------|
| **Max Channels** | 32 (configurable, tested to 8) | ✅ Implemented |
| **Channel Synchronization** | Sample-perfect | ✅ Verified |
| **Sample Rates** | 44.1, 48, 96 kHz | ✅ Supported |
| **Buffer Management** | Per-channel with mutexes | ✅ Implemented |
| **De-interleaving** | In SDL callback (C++) | ✅ Implemented |
| **Alignment Strategy** | Reference channel onset | ✅ Implemented |
| **Calibration Validation** | 7-criteria comprehensive (V3) | ✅ Implemented |
| **File Format** | WAV with `_chN` suffix | ✅ Implemented |
| **Configuration Profiles** | Save/load/delete named configs | ✅ Implemented |
| **GUI Configuration** | Full multi-channel setup UI | ✅ Implemented |
| **Status Display** | Collection Panel recorder info | ✅ Implemented |

### Configuration Schema

**recorderConfig.json:**

```json
{
  // Basic recorder settings
  "sample_rate": 48000,
  "pulse_duration": 0.019,        // seconds
  "cycle_duration": 0.5,          // seconds
  "num_pulses": 4,
  "volume": 0.15,
  "pulse_frequency": 1000.0,
  "impulse_form": "voice_coil",

  // Device selection
  "input_device": 5,
  "output_device": 6,

  // Multi-channel configuration
  "multichannel_config": {
    "enabled": true,                      // Toggle multi-channel mode
    "num_channels": 8,                    // Total input channels
    "channel_names": [...],               // Human-readable names
    "calibration_channel": 2,             // Calibration sensor channel (optional)
    "reference_channel": 5,               // Alignment reference channel
    "response_channels": [...],           // Channels to process (optional)
    "normalize_by_calibration": true      // Enable calibration-based normalization
  },

  // Calibration quality thresholds (V3 comprehensive format - 11 parameters)
  "calibration_quality_config": {
    // 1. Negative Peak Range
    "min_negative_peak": 0.1,
    "max_negative_peak": 0.95,

    // 2. Precursor
    "max_precursor_ratio": 0.2,

    // 3. Negative Peak Width
    "min_negative_peak_width_ms": 0.3,
    "max_negative_peak_width_ms": 3.0,

    // 4. First Positive Peak
    "max_first_positive_ratio": 0.3,

    // 5. First Positive Peak Timing
    "min_first_positive_time_ms": 0.1,
    "max_first_positive_time_ms": 5.0,

    // 6. Highest Positive Peak
    "max_highest_positive_ratio": 0.5,

    // 7. Secondary Negative Peak
    "max_secondary_negative_ratio": 0.3,
    "secondary_negative_window_ms": 10.0
  },

  // Note: V3 format is backward compatible with V1/V2 configs
  // Legacy parameters (max_positive_peak, max_aftershock) auto-converted on load
}
```

### Hardware Requirements

**Native Audio Drivers Required:**

Windows Generic USB Audio Class 2.0 drivers **DO NOT support multi-channel**. Professional audio interfaces require **native manufacturer drivers**.

**Tested Interfaces:**
- Behringer UMC1820 (18 in / 20 out) with native driver
- Generic stereo interfaces (2 channels)

**Driver Installation:**
1. Check current driver: `python check_umc_driver.py`
2. Download manufacturer driver (e.g., Behringer driver v4.59.0 or v5.57.0)
3. Install and reboot
4. Verify: `python test_umc_input_detailed.py`

**Expected Results (Native Driver):**
```
Testing with 1 channels...  ✓ SUCCESS
Testing with 2 channels...  ✓ SUCCESS
Testing with 8 channels...  ✓ SUCCESS
Testing with 18 channels... ✓ SUCCESS
```

---

## API Reference

### Public Methods

#### RoomResponseRecorder.take_record()

```python
def take_record(self,
                output_file: str,
                impulse_file: str,
                method: int = 2,
                mode: str = 'standard',
                return_processed: bool = False) -> Union[np.ndarray, Dict, None]
```

**Parameters:**
- `output_file`: Path for raw recording (e.g., `"raw_000.wav"`)
- `impulse_file`: Path for impulse response (e.g., `"impulse_000.wav"`)
- `method`: Recording method (2=auto, only option currently used)
- `mode`: `'standard'` or `'calibration'`
- `return_processed`: If True, return processed data dict (internal use)

**Returns:**
- **Standard mode:**
  - Single-channel: `np.ndarray` (raw audio)
  - Multi-channel: `Dict[int, np.ndarray]` (raw audio per channel)
- **Calibration mode:** `Dict` with cycle-level data

**Behavior:**
- Standard mode: ALWAYS saves 3 files per channel (raw, impulse, room_response)
- Calibration mode: NEVER saves files, returns analysis data
- Respects `multichannel_config.enabled` setting

#### RoomResponseRecorder.take_record_calibration()

```python
def take_record_calibration(self) -> Dict[str, Any]
```

**Convenience wrapper for:** `take_record("", "", mode='calibration')`

**Returns:**
```python
{
    'calibration_cycles': np.ndarray,           # [N, samples]
    'validation_results': List[Dict],           # Per-cycle validation
    'aligned_multichannel_cycles': Dict[int, np.ndarray],  # Aligned cycles per channel
    'alignment_metadata': Dict,                 # Alignment shifts, indices
    'num_valid_cycles': int,
    'num_aligned_cycles': int,
    'metadata': Dict                            # Recording metadata
}
```

**Requires:**
- `multichannel_config.enabled = true`
- `multichannel_config.calibration_channel` set

---

## File Inventory

### Core Python Files

| File | Size | Lines | Purpose |
|------|------|-------|---------|
| `RoomResponseRecorder.py` | ~40 KB | 1,385 | Signal processing pipeline |
| `calibration_validator_v2.py` | 16 KB | ~300 | Calibration quality validation |
| `multichannel_filename_utils.py` | 7.6 KB | ~200 | Filename parsing/grouping |
| `ScenarioManager.py` | 28 KB | ~700 | Dataset management (multi-channel aware) |
| `gui_audio_settings_panel.py` | 79 KB | ~2,000 | GUI: Calibration testing ✅ |
| `gui_audio_visualizer.py` | 39 KB | ~900 | Waveform/spectrum visualization |
| `gui_series_settings_panel.py` | 19 KB | ~500 | Series recording config |
| `gui_series_worker.py` | ~15 KB | ~400 | Background recording thread |
| `gui_collect_panel.py` | 23 KB | ~600 | Collection interface |

### Deprecated Files

| File | Size | Status | Replacement |
|------|------|--------|-------------|
| `calibration_validator.py` | 7 KB | ❌ DEPRECATED | `calibration_validator_v2.py` |

### C++ SDL Audio Core

| File | Purpose |
|------|---------|
| `sdl_audio_core/audio_engine.h` | Multi-channel recording interface |
| `sdl_audio_core/audio_engine.cpp` | De-interleaving, buffer management |
| `sdl_audio_core/bindings.cpp` | Python bindings (pybind11) |

---

## Known Limitations

### Hardware Requirements

**Native Drivers Required:**
- Generic Windows USB Audio Class 2.0 drivers do NOT support multi-channel
- Professional audio interfaces require manufacturer-specific drivers
- Tested: Behringer UMC1820 with native driver (18 channels working)

### Testing Status

**Validated:**
- ✅ Single-channel backward compatibility
- ✅ Multi-channel recording (up to 8 channels tested)
- ✅ Sample-perfect synchronization
- ✅ Calibration validation and quality filtering
- ✅ GUI workflows (Calibration Impulse, Series Settings)

**Pending:**
- ❌ Hardware testing with 16+ channel interfaces
- ❌ Extended performance benchmarking
- ❌ Cross-platform testing (macOS, Linux)

---

## System Status Summary

**Core Features:** ✅ Complete
- Multi-channel recording (1-32 channels)
- Calibration quality validation (7 criteria)
- Signal processing pipeline (standard & calibration modes)
- File management (multi-channel aware)
- Configuration profiles (save/load/delete)

**GUI Features:** ✅ Complete
- Audio device selection
- Multi-channel configuration UI
- Calibration testing panel
- Series Settings with calibration mode
- Multi-channel response review
- Cycle overlay visualization with statistics
- Configuration profile management

**Remaining Work:**
1. **Collection Panel Calibration Mode** - Needs review (may already work)
2. **Scenarios Panel Integration** - Add analysis capabilities for saved recordings
3. **Hardware Testing** - Validate with various multi-channel interfaces

**System Maturity:** Production-ready for single and multi-channel recording with calibration validation. Scenarios Panel integration is the only major feature gap.
