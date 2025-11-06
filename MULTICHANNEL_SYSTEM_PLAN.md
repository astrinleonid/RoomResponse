# Multi-Channel Room Response System

**Document Version:** 5.0
**Created:** 2025-10-31
**Last Updated:** 2025-11-03 (Phase 8 complete - Collection Panel calibration mode integration)
**Target:** Piano Response Measurement System
**Status:** Core Complete | GUI Complete | SignalProcessor Refactored ✅ | Collection Panel Calibration Mode ✅

---

## Executive Summary

The Room Response system supports **synchronized multi-channel impulse response recording** with calibration-driven quality validation. The system records from 1-32 channels simultaneously with sample-perfect synchronization.

**Implementation Status:**
- ✅ Multi-channel audio recording (C++/Python)
- ✅ Signal processing pipeline refactored (SignalProcessor class extraction)
- ✅ Calibration quality validation system (V2 - comprehensive 7-criteria)
- ✅ Multi-channel file management
- ✅ Multi-channel configuration GUI (fully implemented)
- ✅ Configuration profile management system (save/load/delete)
- ✅ Collection Panel calibration mode integration (recording mode selector)
- ✅ Series Settings calibration mode integration (cycle overlay, statistics)
- ✅ Multi-channel response review GUI (interactive visualization)
- ❌ Scenarios Panel analysis integration (**NEXT PRIORITY** - see detailed plan below)

---

## System Architecture

### Core Components

```
┌─────────────────────────────────────────────────────────────────┐
│                    SIGNAL PROCESSING LAYER                       │
│                   RoomResponseRecorder.py                        │
│                     (1,275 lines, 25 methods)                    │
│                  ↓ delegates to ↓                                 │
│                   SignalProcessor.py                             │
│                     (625 lines, 9 methods)                       │
│              ✅ Clean architecture separation                    │
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
  │   ├─ AudioSettingsPanel ✅                           │
  │   │   ├─ Device selection                             │
  │   │   ├─ Multi-channel configuration ✅              │
  │   │   ├─ Calibration Impulse testing ✅              │
  │   │   └─ Series Settings ✅ (calibration mode)      │
  │   ├─ CollectionPanel ✅                              │
  │   │   ├─ Multi-channel status display ✅             │
  │   │   └─ Recording mode selector ✅ (standard/calib)│
  │   ├─ ConfigProfileManager ✅                         │
  │   │   └─ Save/load/delete profiles ✅                │
  │   ├─ ScenariosPanel ⚠️                              │
  │   │   └─ NEEDS REFACTORING (see plan below)          │
  │   └─ SeriesWorker (background recording) ✅          │
  └──────────────────────────────────────────────────────┘
```

---

## Signal Processing Architecture

### SignalProcessor Class Extraction ✅ **COMPLETE** (Phase 6)

**Clean Separation of Concerns:**

```
Before:
RoomResponseRecorder (1,586 lines)
├─ Audio I/O (recording, playback, device management) ❌ MIXED
├─ Signal Processing (cycle extraction, averaging, FFT) ❌ MIXED
└─ File I/O (WAV, NPZ saving/loading) ❌ MIXED

After:
RoomResponseRecorder (1,275 lines) - Recording Orchestrator
├─ Audio I/O (recording, playback, device management) ✅
├─ File I/O (WAV, NPZ saving/loading) ✅
└─ Delegates signal processing to SignalProcessor ✅

SignalProcessor (625 lines) - Pure Signal Processing
├─ Universal methods (extract_cycles, average_cycles, spectral_analysis) ✅
├─ Standard mode methods (find_onset, extract_impulse) ✅
└─ Calibration mode methods (align_cycles, apply_alignment, normalize) ✅
```

**Key Benefits:**
- ✅ Independent testability (SignalProcessor tested without hardware)
- ✅ Reusable in CLI tools, web APIs, batch scripts
- ✅ 100% backward compatible (all existing code works)
- ✅ Reduced code duplication (all signal processing in one place)

**Delegation Pattern:**
- RoomResponseRecorder calls `self.signal_processor.extract_cycles()` instead of doing signal processing itself
- SignalProcessor is stateless - configured via `SignalProcessingConfig` dataclass
- Processor reinitializes when recorder config changes

---

## Recording Pipeline

### Universal Three-Stage Architecture

The system implements a **clean three-stage pipeline** where Stages 1 and 3 are mode-independent:

**Stage 1 (Recording):** Universal - same for all modes
**Stage 2 (Processing):** Mode-specific - different logic per mode
**Stage 3 (Saving):** Universal - same file structure for all modes

### PATH 1: Standard Mode (Averaged Room Response)

```
recorder.take_record(output_file, impulse_file, mode='standard')

STAGE 1: Recording (UNIVERSAL)
    _record_audio()
    ├─ Single-channel: sdl_audio_core.measure_room_response_auto()
    │   → Returns: np.ndarray [samples]
    └─ Multi-channel: sdl_audio_core.measure_room_response_auto_multichannel()
        → Returns: Dict[int, np.ndarray]

STAGE 2: Processing (MODE-SPECIFIC)
    _process_recorded_signal(recorded_audio)
    ├─ Single-channel: _process_single_channel_signal()
    │   ├─ signal_processor.extract_cycles() → [num_pulses, cycle_samples]
    │   ├─ signal_processor.average_cycles(skip_first_25%) → [cycle_samples]
    │   └─ signal_processor.extract_impulse_response() → [cycle_samples]
    └─ Multi-channel: _process_multichannel_signal()
        ├─ Process reference channel (find onset)
        └─ Apply SAME shift to all channels (sample-perfect sync)

STAGE 3: File Saving (UNIVERSAL)
    _save_processed_data()
    └─ Saves: raw_*.wav, impulse_*.wav, room_*.wav (per channel)

Returns: recorded_audio (backward compatible)
```

### PATH 2: Calibration Mode (Quality-Validated Response) ✅

```
recorder.take_record(output_file, impulse_file, mode='calibration', save_files=True)

STAGE 1: Recording (UNIVERSAL - same as standard)
    _record_audio() → Dict[int, np.ndarray]

STAGE 2: Processing (MODE-SPECIFIC - Quality Pipeline)
    _process_calibration_mode(recorded_audio)
    STEP 1: Validate cycles (CalibrationValidatorV2 - 7 criteria)
    STEP 2: Align cycles by onset (per-cycle alignment)
    STEP 3: Apply alignment to all channels (SAME shifts)
    STEP 4: Normalize by impact magnitude (optional)
    STEP 5: Average aligned/normalized cycles
    STEP 6: Output impulse responses

STAGE 3: File Saving (UNIVERSAL - same format as standard) ✅
    _save_processed_data()
    └─ Saves: raw_*.wav, impulse_*.wav, room_*.wav

Returns: Dict with averaged responses + cycle-level data
```

**Key Improvements:**
- ✅ Per-cycle alignment (robust for physical impacts)
- ✅ Quality validation (filters bad cycles)
- ✅ Averaging after alignment (clean averaged response)
- ✅ Universal file structure (same format as standard mode)
- ✅ Tunable parameters (correlation threshold, onset position)

---

## Collection Panel Integration ✅ **COMPLETE** (Phase 8)

### Recording Mode Selector

**Location:** CollectionPanel → Common Configuration section

**Features:**
- Recording mode radio selector: "Standard" / "Calibration"
- Real-time validation warnings:
  - Error if calibration mode selected without multi-channel enabled
  - Error if no calibration channel configured
  - Success message when calibration mode ready
- Mode parameter passed throughout collection chain:
  - CollectionPanel → SingleScenarioCollector → DatasetCollector → take_record(mode=...)
  - SeriesWorker also receives and passes mode parameter

**Implementation:** (commit 4b77f80)
- `gui_collect_panel.py`: Added recording mode UI and validation
- `DatasetCollector.py`: Added `recording_mode` parameter
- `gui_series_worker.py`: Added `recording_mode` parameter
- Full integration for both single scenario and series modes

---

## Configuration System

### recorderConfig.json Structure

```json
{
  "sample_rate": 48000,
  "pulse_duration": 0.019,
  "cycle_duration": 0.5,
  "num_pulses": 4,
  "volume": 0.15,
  "impulse_form": "voice_coil",
  "input_device": 5,
  "output_device": 6,

  "multichannel_config": {
    "enabled": true,
    "num_channels": 8,
    "channel_names": [...],
    "calibration_channel": 2,
    "reference_channel": 5,
    "response_channels": [0,1,3,4,5,6,7],
    "alignment_correlation_threshold": 0.7,
    "alignment_target_onset_position": 100,
    "normalize_by_calibration": false
  },

  "calibration_quality_config": {
    // V3 format: 11 parameters for 7 quality criteria
    "min_negative_peak": 0.1,
    "max_negative_peak": 0.95,
    "max_precursor_ratio": 0.2,
    "min_negative_peak_width_ms": 0.3,
    "max_negative_peak_width_ms": 3.0,
    "max_first_positive_ratio": 0.3,
    "min_first_positive_time_ms": 0.1,
    "max_first_positive_time_ms": 5.0,
    "max_highest_positive_ratio": 0.5,
    "max_secondary_negative_ratio": 0.3,
    "secondary_negative_window_ms": 10.0
  }
}
```

---

## Scenarios Panel Refactoring Plan

### Current State Analysis

**Existing Scenarios Panel** (`gui_scenarios_panel.py` - 72 KB, ~1,800 lines)

**Current Capabilities:**
- Browse scenarios (Computer-Scenario-Room hierarchy)
- Load and display single-channel measurements
- Waveform visualization (room response, impulse, spectrum)
- Measurement comparison (overlay multiple measurements)
- Basic statistics display
- NPZ file loading support

**Current Limitations:**
- ❌ No multi-channel support
- ❌ No calibration mode analysis
- ❌ No per-channel browsing
- ❌ No cycle-level visualization (only averaged responses)
- ❌ No overlay/averaging across channels
- ❌ Cannot view channels from same scenario side-by-side
- ❌ No validation results display

---

### Refactoring Requirements

#### **1. Multi-Channel Data Structure Support**

**Goal:** Handle both standard (averaged) and calibration (cycle-level) data formats.

**Data Formats to Support:**

**Standard Mode Files:**
```
raw_000_TIMESTAMP_ch0.wav
impulse_000_TIMESTAMP_ch0.wav
room_000_TIMESTAMP_ch0.wav
...
```

**Calibration Mode Files + NPZ:**
```
raw_000_TIMESTAMP_ch0.wav
impulse_000_TIMESTAMP_ch0.wav
room_000_TIMESTAMP_ch0.wav
...
calibration_cycles_000_TIMESTAMP.npz  # NEW: Cycle-level data
```

**NPZ File Structure (Calibration Mode):**
```python
{
    'aligned_multichannel_cycles': Dict[int, np.ndarray],  # [N_cycles, samples] per channel
    'normalized_multichannel_cycles': Dict[int, np.ndarray],  # Optional
    'validation_results': List[Dict],  # Per-cycle validation
    'alignment_metadata': Dict,
    'normalization_factors': List[float],  # If normalized
    'metadata': Dict
}
```

#### **2. Hierarchical Navigation Structure**

**New Navigation Hierarchy:**

```
📁 Scenario (Computer-ScenarioN-Room)
├─ 📊 Overview (scenario-level statistics)
├─ 📈 Measurements (browse by measurement index)
│   ├─ 📐 Measurement 000
│   │   ├─ Channel 0 (Calibration Sensor)
│   │   ├─ Channel 1 (Response)
│   │   ├─ Channel 2 (Response)
│   │   └─ ...
│   ├─ 📐 Measurement 001
│   └─ ...
├─ 🎨 Channels (browse by channel)
│   ├─ Channel 0 (all measurements)
│   ├─ Channel 1 (all measurements)
│   └─ ...
└─ ⚙️ Configuration (display scenario config)
```

**Implementation Strategy:**
- Use expandable sections (st.expander) for each level
- Session state to track current selection (scenario, measurement, channel)
- Sidebar navigation for quick scenario switching
- Breadcrumb display for current location

#### **3. Measurement Visualization Components**

**Component A: Single Measurement, Single Channel View**

**Purpose:** Display waveform/spectrum for one channel from one measurement.

**Features:**
- Tab selector: Waveform / Spectrum / Cycles (if calibration mode)
- Display mode: Averaged Response (standard mode) or Individual Cycles (calibration mode)
- Zoom controls (inherited from AudioVisualizer)
- Statistics panel (peak, RMS, energy)
- Validation results (if calibration mode)

**Component B: Single Measurement, Multi-Channel Overlay**

**Purpose:** Compare multiple channels from the same measurement side-by-side.

**Features:**
- Channel selection checkboxes (select which channels to overlay)
- Layout mode: Stacked (vertical) / Overlaid (single plot)
- Synchronized zoom/pan (linked axes)
- Per-channel statistics table
- Cross-correlation display (verify synchronization)
- Color coding per channel (consistent throughout panel)

**Component C: Multi-Measurement, Single Channel View**

**Purpose:** Compare same channel across multiple measurements (already exists, needs enhancement).

**Features:**
- Measurement selection checkboxes
- Overlay plot with automatic averaging option
- Statistics: mean, std dev, min/max envelope
- Outlier detection/filtering
- Export averaged response

**Component D: Multi-Measurement, Multi-Channel Analysis**

**Purpose:** Advanced analysis across measurements and channels.

**Features:**
- 2D selection grid: Measurements (rows) × Channels (columns)
- Analysis modes:
  - Channel-wise averaging (average same channel across measurements)
  - Measurement-wise averaging (average all channels from same measurement)
  - Grand average (all selected data)
- Heatmap visualization (amplitude distribution)
- Export results to CSV/NPZ

#### **4. Cycle-Level Visualization (Calibration Mode)**

**Purpose:** Display and analyze individual cycles from calibration recordings.

**Features:**

**Cycle Statistics Table:**
- Per-cycle metrics (negative peak, positive peak, RMS)
- Validation status (pass/fail with color coding)
- Normalization factors (if normalized)
- Checkbox selection for overlay

**Cycle Overlay Plot:**
- Display mode: Aligned / Normalized / Both Side-by-Side
- Interactive zoom controls (per-channel persistent state)
- Statistics overlay: peak range, std dev, range width %
- View mode: Waveform / Spectrum
- Analysis display: min/max/mean/std

**Validation Results Display:**
- Expandable per-cycle validation details
- 7-criteria status (pass/fail indicators)
- Threshold comparison (value vs threshold)
- Failure reason highlighting

**Implementation:** Reuse components from Series Settings panel (`gui_series_settings_panel.py` lines 200-400)

#### **5. Channel Grouping and Averaging**

**Purpose:** Analyze response characteristics across multiple channels.

**Averaging Modes:**

**A. Channel-Wise Averaging:**
- Select multiple measurements, one channel
- Compute: `mean_response = mean([meas_i[channel_j] for i in measurements])`
- Display: averaged waveform + confidence interval (std dev envelope)
- Use case: Representative response for a channel across session

**B. Measurement-Wise Averaging:**
- Select one measurement, multiple channels
- Compute: `mean_response = mean([meas_i[channel_j] for j in channels])`
- Display: averaged waveform across channels
- Use case: Overall response characteristic for a measurement

**C. Grand Averaging:**
- Select multiple measurements, multiple channels
- Compute: `grand_mean = mean([meas_i[channel_j] for i, j in selections])`
- Display: overall averaged response with variance
- Use case: System-level response characteristic

**UI Components:**
- Selection grid (measurements × channels)
- Averaging mode selector (channel-wise / measurement-wise / grand)
- "Compute Average" button
- Export button (save averaged response as WAV/NPZ)

#### **6. Side-by-Side Channel Comparison**

**Purpose:** View multiple channels from same measurement simultaneously.

**Layout Options:**

**Stacked Layout (Vertical):**
```
┌─────────────────────────────────────┐
│ Channel 0 (Calibration Sensor)      │
│ [waveform plot]                     │
├─────────────────────────────────────┤
│ Channel 1 (Response)                │
│ [waveform plot]                     │
├─────────────────────────────────────┤
│ Channel 2 (Response)                │
│ [waveform plot]                     │
└─────────────────────────────────────┘
```

**Overlaid Layout (Single Plot):**
```
┌─────────────────────────────────────┐
│ All Selected Channels               │
│ [overlaid waveforms, color-coded]  │
│ Legend: Ch0 (blue), Ch1 (red), ...  │
└─────────────────────────────────────┘
```

**Features:**
- Toggle between stacked/overlaid layouts
- Channel selection checkboxes
- Synchronized zoom (linked X-axis for stacked, shared for overlaid)
- Per-channel amplitude scaling (auto or manual)
- Color legend with channel names
- Export all channels to single figure (PNG/PDF)

#### **7. Configuration Display**

**Purpose:** Show recording and calibration configuration for the scenario.

**Display Sections:**

**A. Recording Configuration:**
- Sample rate, num_pulses, cycle_duration
- Input/output devices used
- Recording mode (standard / calibration)

**B. Multi-Channel Configuration:**
- Enabled status
- Number of channels
- Channel names and roles (calibration, reference, response)
- Alignment parameters (correlation threshold, target onset position)
- Normalization status

**C. Calibration Quality Configuration:**
- 11 threshold parameters
- Validation criteria summary

**Implementation:**
- Read from `session_metadata.json` (scenario metadata)
- Display in tabular format with expandable sections
- Copy-to-clipboard button for configuration JSON

---

### Implementation Plan

#### **Phase 1: Data Loading Infrastructure** (2-3 hours)

**Tasks:**
1. Extend `ScenarioManager` with multi-channel NPZ loading:
   - `load_calibration_cycles(scenario_path, measurement_index) -> Dict`
   - `get_measurement_mode(scenario_path, measurement_index) -> 'standard' | 'calibration'`
   - `get_channel_list(scenario_path, measurement_index) -> List[int]`

2. Create data models for loaded data:
   ```python
   @dataclass
   class MeasurementData:
       mode: str  # 'standard' or 'calibration'
       channels: Dict[int, np.ndarray]  # Channel index -> averaged response
       sample_rate: int
       metadata: Dict

   @dataclass
   class CalibrationCycleData:
       aligned_cycles: Dict[int, np.ndarray]  # Channel -> [N_cycles, samples]
       normalized_cycles: Optional[Dict[int, np.ndarray]]
       validation_results: List[Dict]
       alignment_metadata: Dict
       normalization_factors: Optional[List[float]]
   ```

3. Implement caching for loaded measurements (session state)

#### **Phase 2: Navigation UI Refactoring** (4-5 hours)

**Tasks:**
1. Replace flat scenario list with hierarchical navigation
2. Add measurement/channel selection UI:
   - Sidebar: Scenario selector dropdown
   - Main panel: Tab selector (Overview / Measurements / Channels / Config)
   - Measurements tab: Expandable sections per measurement
   - Channels tab: Channel selector dropdown + measurement grid

3. Implement session state management:
   ```python
   # Session keys
   'scenarios_selected_scenario': str
   'scenarios_selected_measurement': Optional[int]
   'scenarios_selected_channel': Optional[int]
   'scenarios_view_mode': 'single' | 'multi_channel' | 'multi_measurement'
   ```

4. Create breadcrumb display for current location

#### **Phase 3: Single Measurement Visualization** (6-8 hours)

**Tasks:**
1. **Component A: Single Channel View**
   - Implement tab selector (Waveform / Spectrum / Cycles)
   - Standard mode: Display averaged response
   - Calibration mode: Display cycles with statistics
   - Reuse `AudioVisualizer` components for plotting
   - Add validation results display (expandable)

2. **Component B: Multi-Channel Overlay**
   - Implement channel selection grid (checkboxes)
   - Add layout mode toggle (Stacked / Overlaid)
   - Create stacked plot layout (matplotlib subplots)
   - Create overlaid plot (single plot, color-coded)
   - Implement synchronized zoom (linked axes)
   - Add per-channel statistics table
   - Calculate and display cross-correlation matrix

#### **Phase 4: Cycle-Level Visualization** (4-5 hours)

**Tasks:**
1. Port cycle visualization components from Series Settings panel:
   - `_render_cycle_statistics_table()`
   - `_render_cycle_overlay_visualization()`
   - Adapt for Scenarios Panel context (loaded data vs fresh recording)

2. Add display mode selector (Aligned / Normalized / Both)

3. Implement per-cycle validation display:
   - Expandable sections per cycle
   - Color-coded pass/fail indicators
   - Threshold comparison display

#### **Phase 5: Averaging and Analysis** (5-6 hours)

**Tasks:**
1. **Channel-Wise Averaging:**
   - Measurement selection grid (checkboxes)
   - "Compute Average" button
   - Display averaged waveform with std dev envelope
   - Export averaged response (WAV/NPZ)

2. **Measurement-Wise Averaging:**
   - Channel selection grid
   - Compute and display averaged waveform

3. **Grand Averaging:**
   - 2D selection grid (measurements × channels)
   - Compute grand average with variance metrics
   - Export results

4. **Statistics Display:**
   - Mean, std dev, min/max
   - Outlier detection (Z-score, IQR method)
   - Filtering options (exclude outliers)

#### **Phase 6: Configuration Display** (2-3 hours)

**Tasks:**
1. Load configuration from `session_metadata.json`
2. Create configuration display UI (tabular format)
3. Add copy-to-clipboard functionality
4. Display channel roles and names
5. Show calibration thresholds (if applicable)

#### **Phase 7: Polish and Testing** (3-4 hours)

**Tasks:**
1. Add loading spinners and progress indicators
2. Error handling for missing files/data
3. Performance optimization (lazy loading, caching)
4. User documentation (inline help text)
5. Test with various scenarios:
   - Single-channel standard mode
   - Multi-channel standard mode
   - Multi-channel calibration mode
   - Missing files/corrupted data
6. UI refinements based on testing

---

### Total Effort Estimate

**Development Time:**
- Phase 1: 2-3 hours
- Phase 2: 4-5 hours
- Phase 3: 6-8 hours
- Phase 4: 4-5 hours
- Phase 5: 5-6 hours
- Phase 6: 2-3 hours
- Phase 7: 3-4 hours

**Total: 26-34 hours (~3-4 working days)**

---

### Implementation Priority

**Must-Have (MVP):**
1. Multi-channel data loading (Phase 1)
2. Basic navigation UI (Phase 2)
3. Single measurement, multi-channel view (Phase 3, Component B)
4. Cycle-level visualization (Phase 4)
5. Configuration display (Phase 6)

**Should-Have (Enhanced):**
6. Single channel view with tabs (Phase 3, Component A)
7. Channel-wise averaging (Phase 5, partial)

**Nice-to-Have (Advanced):**
8. Grand averaging and heatmaps (Phase 5, full)
9. Cross-correlation analysis
10. Export to publication-quality figures

---

### Success Criteria

**Functional Requirements:**
- ✅ Load and display multi-channel measurements (standard & calibration modes)
- ✅ Browse by scenario, measurement, and channel
- ✅ View channels from same measurement side-by-side (stacked or overlaid)
- ✅ Display cycle-level data for calibration recordings
- ✅ Overlay and average measurements by channel
- ✅ Show validation results and statistics

**Non-Functional Requirements:**
- ✅ Intuitive navigation (breadcrumbs, hierarchical sections)
- ✅ Responsive UI (lazy loading, caching)
- ✅ Consistent visual style (matches other panels)
- ✅ Backward compatible (still works with single-channel data)

---

## System Status Summary

**Core Features:** ✅ Complete
- Multi-channel recording (1-32 channels)
- Calibration quality validation (7 criteria)
- Signal processing pipeline (SignalProcessor class extraction)
- File management (multi-channel aware)
- Configuration profiles (save/load/delete)
- Collection Panel calibration mode integration

**GUI Features:** ✅ Complete
- Audio device selection
- Multi-channel configuration UI
- Calibration testing panel
- Series Settings with calibration mode
- Multi-channel response review
- Collection Panel recording mode selector
- Configuration profile management

**Remaining Work:**
1. **Scenarios Panel Refactoring** (**NEXT PRIORITY**)
   - Implement multi-channel browsing and visualization
   - Add cycle-level analysis for calibration data
   - Enable channel grouping and averaging
   - See detailed plan above
   - Estimated effort: 3-4 working days

**System Maturity:** Production-ready for recording and basic review. Scenarios Panel is the main feature gap for comprehensive analysis workflow.

---

## Technical Specifications

### Multi-Channel Capabilities

| Feature | Specification | Status |
|---------|--------------|--------|
| **Max Channels** | 32 (configurable, tested to 8) | ✅ |
| **Channel Synchronization** | Sample-perfect | ✅ |
| **Sample Rates** | 44.1, 48, 96 kHz | ✅ |
| **Recording Modes** | Standard, Calibration | ✅ |
| **Signal Processing** | SignalProcessor (clean architecture) | ✅ |
| **Calibration Validation** | 7-criteria comprehensive (V3) | ✅ |
| **File Format** | WAV with `_chN` suffix, NPZ for cycles | ✅ |
| **Configuration Profiles** | Save/load/delete named configs | ✅ |
| **Collection Panel** | Recording mode selector | ✅ |

---

## File Inventory

### Core Python Files

| File | Size | Lines | Purpose |
|------|------|-------|---------|
| `RoomResponseRecorder.py` | ~50 KB | 1,275 | Recording orchestrator |
| `signal_processor.py` | ~20 KB | 625 | Signal processing (independent) |
| `calibration_validator_v2.py` | 16 KB | ~300 | Quality validation |
| `multichannel_filename_utils.py` | 7.6 KB | ~200 | File parsing/grouping |
| `ScenarioManager.py` | 28 KB | ~700 | Dataset management |
| `gui_audio_settings_panel.py` | 79 KB | ~2,000 | Settings & calibration UI |
| `gui_collect_panel.py` | 25 KB | ~560 | Collection interface ✅ |
| `gui_scenarios_panel.py` | 72 KB | ~1,800 | Scenarios browser ⚠️ NEEDS REFACTORING |

---

## Known Limitations

### Hardware Requirements
- Native manufacturer drivers required (generic USB Audio Class 2.0 insufficient)
- Tested: Behringer UMC1820 with native driver (18 channels working)

### Testing Status
**Validated:**
- ✅ Single/multi-channel recording (up to 8 channels)
- ✅ Sample-perfect synchronization
- ✅ Calibration validation and filtering
- ✅ All GUI workflows (Settings, Series, Collection)

**Pending:**
- ❌ Hardware testing with 16+ channel interfaces
- ❌ Extended performance benchmarking
- ❌ Cross-platform testing (macOS, Linux)

---

**Document Revision History:**
- v5.0 (2025-11-03): Added Collection Panel calibration mode status, detailed Scenarios Panel refactoring plan
- v4.2 (2025-11-03): Cleaned up legacy content, updated Phase 6 status
- v4.0 (2025-11-02): Added Series Settings calibration mode, multi-channel response review
- v3.0 (2025-11-01): Updated to V3 validation format, added calibration normalization
- v2.0 (2025-10-31): Initial comprehensive documentation
