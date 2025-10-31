# RoomResponseRecorder Refactoring Plan

**Document Version:** 1.1
**Date:** 2025-10-31
**Status:** 📋 DEFERRED - Moved to Future Phase
**Priority:** Low (optional enhancement)

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Current State Analysis](#current-state-analysis)
3. [Problem Statement](#problem-statement)
4. [Proposed Architecture](#proposed-architecture)
5. [Refactoring Strategy](#refactoring-strategy)
6. [Implementation Plan](#implementation-plan)
7. [Risk Assessment](#risk-assessment)
8. [Migration Path](#migration-path)

---

## 1. Executive Summary

### Decision: Refactoring Deferred

**Date:** 2025-10-31
**Rationale:**
- Multi-channel system is working correctly after critical bug fixes
- All core functionality (recording, calibration, series analysis) operational
- Refactoring would introduce risk without immediate functional benefit
- Focus should be on Phase 5 (Testing & Validation) first

**Future Consideration:**
This refactoring plan remains valid as a future enhancement to improve code maintainability and reduce technical debt. It should be reconsidered after:
1. Phase 5 (Testing & Validation) is complete
2. System has been validated with hardware testing
3. All edge cases and workflows are well understood

### Current Situation

RoomResponseRecorder has evolved organically through multiple phases:
1. Single-channel recording with simple averaging
2. Multi-channel support added
3. Calibration validation system (V1 → V2)
4. Multiple cycle extraction/alignment approaches implemented in parallel
5. Onset-based alignment added (latest)

**Result:** Code contains **redundant methods**, **conflicting approaches**, and **unclear separation of concerns**.

### Core Issue

**Two fundamentally different recording modes are conflated:**

1. **Standard Recording Mode** (No Calibration Signal)
   - Record impulse response directly
   - Average multiple cycles
   - Apply onset detection/alignment
   - Used for: Piano response measurements

2. **Calibration Recording Mode** (With Calibration Signal)
   - Record WITH calibration channel
   - Validate calibration quality
   - Align all channels based on calibration
   - Used for: System calibration, quality assurance

### Refactoring Goals

1. **Separate** the two recording modes clearly
2. **Eliminate** redundant cycle extraction methods
3. **Unify** multi-channel support across both modes
4. **Clarify** which methods belong to which mode
5. **Maintain** backward compatibility where possible

---

## 2. Current State Analysis

### 2.1 Method Inventory

Total methods: **36**

#### Recording & Core API (4 methods)
```python
__init__(config_file_path)              # Constructor
take_record(output, impulse, method)    # Main API - uses _process_recorded_signal
_record_method_2()                      # SDL recording (multi-channel capable)
test_mic(duration, chunk_duration)      # Mic test utility
```

#### Signal Generation (2 methods)
```python
_generate_single_pulse(exact_samples)   # Generate one pulse
_generate_complete_signal()             # Generate full pulse train
```

#### Configuration & Validation (4 methods)
```python
_validate_config()                      # Validate signal parameters
_validate_multichannel_config()         # Validate multi-channel setup
_migrate_calibration_config_v1_to_v2()  # Config migration
get_signal_info()                       # Return signal parameters
```

#### Device Management (4 methods)
```python
set_audio_devices(input, output)        # Set device IDs
list_devices()                          # List available devices
get_device_info_with_channels()         # Query device channel counts
test_multichannel_recording(duration)   # Test multi-channel recording
```

#### Signal Processing Dispatch (3 methods)
```python
_process_recorded_signal(recorded_audio)           # Dispatcher (NEW - line 686)
_process_single_channel_signal(recorded_audio)     # Single-channel path
_process_multichannel_signal(multichannel_audio)   # Multi-channel path (NO calibration)
```

#### Onset Detection & Impulse Extraction (3 methods)
```python
_find_onset_in_room_response(room_response)  # Find onset in averaged signal
_extract_impulse_response(room_response)     # Extract impulse by rotation
_find_sound_onset(audio, window_size)        # Moving RMS onset detection
```

#### **CYCLE EXTRACTION/ALIGNMENT - Multiple Conflicting Approaches** (7 methods)

**Approach A: Onset-Based Alignment (NEW - FOR CALIBRATION)**
```python
align_cycles_by_onset(initial_cycles, validation_results, threshold)  # Line 872 ✅ CURRENT
apply_alignment_to_channel(channel_raw, alignment_metadata)           # Line 1007 ✅ CURRENT
```

**Approach B: Onset Detection + Cross-Correlation Alignment (OLDER)**
```python
extract_and_align_cycles(raw_audio, num_expected, onset_window_ms)   # Line 1092 ❓ REDUNDANT?
_detect_cycle_onsets(audio, num_expected, onset_window_ms)            # Line 1155 ❓ USED?
_select_reference_cycle(cycles)                                        # Line 1256 ✅ USED by 1092
_align_cycles_to_reference(cycles, reference, ref_idx)                # Line 1271 ✅ USED by 1092
_find_cycle_alignment_shift(reference, signal, max_lag)               # Line 1309 ✅ USED by 1271
```

**Approach C: Pre-calculated Positions Extraction (UNUSED?)**
```python
extract_cycles_with_shifts(raw_audio, cycle_positions, cycle_samples) # Line 1059 ❓ UNUSED?
```

**Helper Methods:**
```python
_calculate_rms_envelope(audio, window_samples)                         # Line 1209 ✅ USED by 1155
_extract_cycles_at_positions(audio, positions)                         # Line 1226 ✅ USED by 1092
_calculate_cycle_extraction_quality(aligned_cycles, onsets, shifts)   # Line 1338 ✅ USED by 1092
```

#### File Saving (4 methods)
```python
_save_wav(audio_data, filename)                            # WAV file writer
_save_multichannel_files(output, impulse, processed_data)  # Multi-channel save
_save_single_channel_files(output, impulse, processed_data) # Single-channel save
_make_channel_filename(base_filename, channel_index)       # Generate channel filename
```

#### Utility (3 methods)
```python
get_sdl_core_info()           # SDL version info
print_signal_analysis()       # Print signal parameters
```

### 2.2 Redundancy Analysis

#### Critical Finding: THREE DIFFERENT CYCLE ALIGNMENT APPROACHES

**Approach A: `align_cycles_by_onset()` (NEWEST - 2025-10-30)**
- **Used by:** GUI calibration test (`gui_audio_settings_panel._perform_calibration_test`)
- **Input:** Already extracted cycles from simple reshape
- **Method:** Direct negative peak detection (np.argmin)
- **Target:** Fixed position (100 samples)
- **Purpose:** Align calibration cycles by onset
- **Status:** ✅ **ACTIVELY USED**

**Approach B: `extract_and_align_cycles()` (OLDER)**
- **Used by:** Unknown (need to search codebase)
- **Input:** Raw audio
- **Method:** RMS envelope onset detection → Extract → Cross-correlation alignment
- **Target:** Align to reference cycle
- **Purpose:** Full pipeline from raw audio to aligned cycles
- **Status:** ❓ **USAGE UNCLEAR**

**Approach C: `extract_cycles_with_shifts()` (OLDEST)**
- **Used by:** Unknown (need to search codebase)
- **Input:** Raw audio + pre-calculated positions
- **Method:** Extract at specified positions
- **Purpose:** Two-stage extraction (calculate positions, then extract)
- **Status:** ❓ **LIKELY UNUSED**

### 2.3 Call Graph Analysis

```
take_record()
  └─> _record_method_2()
  └─> _process_recorded_signal()
      ├─> _process_single_channel_signal()
      │     └─> Simple reshape → Average → Find onset → Rotate
      └─> _process_multichannel_signal()
            └─> Per-channel: Simple reshape → Average → Find onset in ref → Apply to all
```

**Calibration Test (GUI):**
```
gui_audio_settings_panel._perform_calibration_test()
  └─> _record_method_2()
  └─> Simple reshape → Validate → align_cycles_by_onset() → apply_alignment_to_channel()
```

**Key Observation:**
- `take_record()` does NOT use any cycle alignment methods
- Cycle alignment methods ONLY used by GUI calibration test
- `extract_and_align_cycles()` and `extract_cycles_with_shifts()` appear UNUSED

### 2.4 Configuration Analysis

**Calibration Quality Config:**
```python
self.calibration_quality_config = {
    'min_negative_peak': 0.1,      # Used by CalibrationValidatorV2
    'max_negative_peak': 0.95,
    'min_positive_peak': 0.0,
    'max_positive_peak': 0.6,
    'min_aftershock': 0.0,
    'max_aftershock': 0.3,
    'aftershock_window_ms': 10.0,
    'aftershock_skip_ms': 2.0,
    'min_valid_cycles': 3
}
```
✅ **USED** by GUI calibration test

**Correlation Quality Config:**
```python
self.correlation_quality_config = {
    'ref_xcorr_threshold': 0.85,
    'ref_xcorr_min_pass_fraction': 0.75,
    'ref_xcorr_max_retries': 3,
    'min_valid_cycles_after_corr': 3
}
```
❌ **UNUSED** - was for removed `_filter_cycles_by_correlation` method

### 2.5 Multi-Channel Support Status

**Current Status:**

| Feature | Single-Channel | Multi-Channel (No Cal) | Multi-Channel (With Cal) |
|---------|---------------|----------------------|-------------------------|
| **Recording** | ✅ SDL | ✅ SDL | ✅ SDL |
| **Processing** | ✅ `_process_single_channel_signal` | ✅ `_process_multichannel_signal` | ✅ GUI `_perform_calibration_test` |
| **Cycle Extraction** | Simple reshape | Simple reshape | Simple reshape |
| **Averaging** | ✅ Direct | ✅ Per-channel | ❌ Not in recorder |
| **Onset Detection** | ✅ `_find_onset_in_room_response` | ✅ Reference channel | ❌ Not in recorder |
| **Alignment** | ✅ Rotation | ✅ Same shift to all | ✅ GUI `align_cycles_by_onset` |
| **File Saving** | ✅ Single file | ✅ Per-channel files | ❌ Not saved by recorder |

**Issue:** Calibration mode completely separate from `take_record()` pipeline!

---

## 3. Problem Statement

### 3.1 Core Problems

#### Problem 1: Two Recording Modes Conflated

**Standard Recording:**
```python
recorder.take_record("raw.wav", "impulse.wav")
# → Records, averages cycles, finds onset, saves averaged impulse
```

**Calibration Recording:**
```python
# Completely different code path!
gui._perform_calibration_test()
# → Records, validates each cycle, aligns, returns cycle data
# → NO file saving, NO averaging (done later by user)
```

**These are fundamentally different workflows but share the same class!**

#### Problem 2: Unclear Responsibilities

**What does RoomResponseRecorder do?**
- Generate test signal? ✅
- Record audio? ✅
- Process/average cycles? ✅ (for standard mode)
- Validate calibration quality? ❌ (done by CalibrationValidatorV2)
- Align cycles? ❓ (Approach A: yes, Approach B: unclear)
- Save files? ✅

**Confusion:** Cycle alignment methods exist but aren't used by `take_record()`

#### Problem 3: Redundant Cycle Alignment Methods

Three approaches coexist:
1. `align_cycles_by_onset()` - NEW, used by GUI
2. `extract_and_align_cycles()` - OLD, usage unclear
3. `extract_cycles_with_shifts()` - OLDER, likely unused

**Cost:** Maintenance burden, confusion, potential bugs

#### Problem 4: Multi-Channel Not Unified

**For standard recording:**
- Multi-channel supported in `_process_multichannel_signal()`
- Averages cycles per-channel
- Finds onset in reference channel
- Applies same shift to all channels
- Saves per-channel files

**For calibration recording:**
- Completely separate implementation in GUI
- Uses `align_cycles_by_onset()` and `apply_alignment_to_channel()`
- Does NOT average
- Does NOT save files

**No code reuse between the two modes!**

#### Problem 5: Configuration Bloat

`correlation_quality_config` exists but is UNUSED (was for deleted methods).

Should be removed.

---

## 4. Proposed Architecture

### 4.1 Conceptual Model

**Two Recording Modes Should Be Explicit:**

```python
class RoomResponseRecorder:
    """
    Handles two distinct recording workflows:

    1. Standard Recording (take_record):
       - Record impulse response
       - Average multiple cycles
       - Extract impulse (onset-aligned)
       - Save files
       - Used for: Piano response measurements

    2. Calibration Recording (calibration_test):
       - Record WITH calibration channel
       - Validate cycle quality
       - Align cycles by calibration
       - Return individual cycles (no averaging)
       - Used for: System calibration
    """
```

### 4.2 Proposed Method Structure

#### Core API (3 methods)
```python
def __init__(config_file_path)
    """Initialize recorder with configuration."""

def take_record(output_file, impulse_file, mode='standard')
    """
    Record and process room response.

    mode='standard': Average cycles, save impulse (current behavior)
    mode='calibration': Validate and align cycles, return cycle data
    """

def calibration_test() -> Dict
    """
    Specialized calibration recording.
    Validates and aligns cycles, returns data for GUI analysis.
    Delegates to take_record(mode='calibration').
    """
```

#### Recording & Processing (5 methods)
```python
def _record_audio() -> Union[np.ndarray, Dict[int, np.ndarray]]
    """Record audio using SDL (single or multi-channel)."""

def _process_standard_mode(recorded_audio) -> Dict
    """
    Standard recording processing:
    1. Extract cycles (simple reshape)
    2. Average cycles
    3. Find onset
    4. Align by rotation
    5. Return averaged impulse
    """

def _process_calibration_mode(recorded_audio) -> Dict
    """
    Calibration recording processing:
    1. Extract cycles (simple reshape)
    2. Validate each cycle (CalibrationValidatorV2)
    3. Align valid cycles (onset-based)
    4. Apply to all channels
    5. Return cycle data
    """

def _extract_cycles_simple(audio, num_cycles, cycle_samples) -> np.ndarray
    """Simple reshape extraction (used by both modes)."""

def _average_cycles(cycles, start_cycle=0) -> np.ndarray
    """Average cycles for standard mode."""
```

#### Cycle Alignment (2 methods - UNIFIED)
```python
def align_cycles_by_onset(initial_cycles, validation_results, threshold) -> dict
    """
    Align cycles by detecting onset (negative peak).
    Used by calibration mode.
    """

def apply_alignment_to_channel(channel_raw, alignment_metadata) -> np.ndarray
    """
    Apply calibration-derived shifts to any channel.
    Ensures uniform multi-channel alignment.
    """
```

#### Onset Detection (2 methods - UNIFIED)
```python
def _find_onset_in_averaged_signal(room_response) -> int
    """
    Find onset in an averaged signal.
    Used by standard mode for final impulse extraction.
    """

def _find_onset_by_rms(audio, window_ms, threshold_factor) -> int
    """
    Find onset using RMS envelope.
    Generic utility method.
    """
```

#### File Management (3 methods)
```python
def _save_files(output_file, impulse_file, processed_data)
    """Save files (handles both single and multi-channel)."""

def _save_wav(audio_data, filename)
    """WAV file writer."""

def _make_channel_filename(base_filename, channel_index)
    """Generate channel-specific filename."""
```

#### Configuration & Utilities (8 methods - unchanged)
```python
# Keep as-is:
_generate_single_pulse()
_generate_complete_signal()
_validate_config()
_validate_multichannel_config()
_migrate_calibration_config_v1_to_v2()
set_audio_devices()
list_devices()
get_device_info_with_channels()
# etc.
```

### 4.3 What Gets Removed

#### Remove These Methods (7 total)
```python
# OLD cycle extraction approach (unused)
extract_and_align_cycles()              # Line 1092
_detect_cycle_onsets()                  # Line 1155
_extract_cycles_at_positions()          # Line 1226
_select_reference_cycle()               # Line 1256
_align_cycles_to_reference()            # Line 1271
_find_cycle_alignment_shift()           # Line 1309
_calculate_cycle_extraction_quality()  # Line 1338

# Pre-calculated positions approach (unused)
extract_cycles_with_shifts()            # Line 1059

# Unused helper
_calculate_rms_envelope()               # Line 1209 (only used by _detect_cycle_onsets)
```

#### Remove This Configuration
```python
# UNUSED correlation config
self.correlation_quality_config = {...}
```

#### Consolidate These Methods
```python
# Merge into one:
_process_single_channel_signal()    # Line 706
_process_multichannel_signal()      # Line 741
# → _process_standard_mode(recorded_audio, is_multichannel)

# Keep separate dispatcher:
_process_recorded_signal()          # Line 686 (add mode parameter)
```

### 4.4 Method Count Comparison

| Category | Before | After | Change |
|----------|--------|-------|--------|
| **Core API** | 4 | 3 | -1 |
| **Recording & Processing** | 3 | 5 | +2 |
| **Cycle Alignment** | 9 | 2 | **-7** |
| **Onset Detection** | 3 | 2 | -1 |
| **File Management** | 4 | 3 | -1 |
| **Config & Utils** | 13 | 13 | 0 |
| **TOTAL** | **36** | **28** | **-8** |

**Net reduction: 8 methods (22% smaller)**

---

## 5. Refactoring Strategy

### 5.1 Guiding Principles

1. **Explicit > Implicit:** Make the two modes explicit in API
2. **Consolidate:** Remove redundant implementations
3. **Unify:** Single code path for multi-channel in both modes
4. **Preserve:** Keep existing behavior for `take_record()` by default
5. **Clarify:** Each method has ONE clear purpose

### 5.2 Key Decisions

#### Decision 1: Two Modes, One API

**Option A:** Separate classes
```python
StandardRecorder()
CalibrationRecorder()
```
❌ **Rejected:** Too much code duplication

**Option B:** Mode parameter ✅ **SELECTED**
```python
recorder.take_record(output, impulse, mode='standard')
recorder.take_record(output, impulse, mode='calibration')
```

**Rationale:**
- Shares common code (signal generation, recording, device management)
- Clear separation of processing logic
- Backward compatible (mode='standard' is default)

#### Decision 2: Consolidate Processing Methods

**Current:**
- `_process_single_channel_signal()` - 35 lines
- `_process_multichannel_signal()` - 64 lines
- 90% overlap (reshape, average, onset detection)

**Proposed:**
```python
def _process_standard_mode(recorded_audio):
    is_multichannel = isinstance(recorded_audio, dict)
    # Unified logic for both single and multi-channel
```

**Rationale:** Eliminates duplication, easier to maintain

#### Decision 3: Keep Only One Cycle Alignment Approach

**Remove:**
- `extract_and_align_cycles()` - Full pipeline (RMS onset → extract → align)
- All helper methods for this approach

**Keep:**
- `align_cycles_by_onset()` - Works on pre-extracted cycles
- `apply_alignment_to_channel()` - Multi-channel support

**Rationale:**
- Simpler approach (negative peak detection)
- Actively used by GUI
- Proven to work (0 sample alignment error)
- Other approach appears unused

#### Decision 4: Clarify Onset Detection Methods

**Current confusion:** Multiple onset detection methods with unclear purposes

**Proposed:**
- `_find_onset_in_averaged_signal()` - For standard mode (averaged signal)
- `_find_onset_by_rms()` - Generic utility (if needed elsewhere)

**Rationale:** Clear naming indicates purpose

#### Decision 5: Remove Unused Configuration

Remove `correlation_quality_config` - was for deleted methods.

---

## 6. Implementation Plan

### 6.1 Phase 1: Search & Verify (RISK MITIGATION)

**Goal:** Confirm which methods are actually unused

**Tasks:**
1. Search entire codebase for calls to suspected unused methods:
   - `extract_and_align_cycles`
   - `extract_cycles_with_shifts`
   - `_detect_cycle_onsets`
   - `_select_reference_cycle` (check both versions!)
   - etc.

2. Check if `correlation_quality_config` is used anywhere

3. Document findings

**Output:** Confirmed list of methods to remove

### 6.2 Phase 2: Add Mode Parameter

**Goal:** Add explicit mode support without breaking existing code

**Tasks:**

1. **Update `take_record()` signature:**
   ```python
   def take_record(self, output_file: str, impulse_file: str,
                   method: int = 2, mode: str = 'standard'):
       """
       mode: 'standard' (default) or 'calibration'
       """
   ```

2. **Update `_process_recorded_signal()` signature:**
   ```python
   def _process_recorded_signal(self, recorded_audio, mode='standard'):
       if mode == 'standard':
           return self._process_standard_mode(recorded_audio)
       elif mode == 'calibration':
           return self._process_calibration_mode(recorded_audio)
       else:
           raise ValueError(f"Unknown mode: {mode}")
   ```

3. **Test:** Verify existing code still works with default mode='standard'

### 6.3 Phase 3: Implement Standard Mode (Consolidation)

**Goal:** Unify single and multi-channel standard processing

**Tasks:**

1. **Create `_process_standard_mode()`:**
   ```python
   def _process_standard_mode(self, recorded_audio) -> Dict[str, Any]:
       is_multichannel = isinstance(recorded_audio, dict)

       if is_multichannel:
           # Multi-channel logic
           channels = {}
           ref_ch = self.multichannel_config['reference_channel']

           for ch_idx, audio in recorded_audio.items():
               # Extract cycles
               cycles = self._extract_cycles_simple(audio)
               # Average cycles
               averaged = self._average_cycles(cycles)
               channels[ch_idx] = averaged

           # Find onset in reference channel
           onset = self._find_onset_in_averaged_signal(channels[ref_ch])

           # Apply same shift to all channels
           impulses = {}
           for ch_idx, avg_signal in channels.items():
               impulses[ch_idx] = np.roll(avg_signal, -onset)

           return {
               'raw': recorded_audio,
               'room_response': channels,
               'impulse': impulses,
               'onset_sample': onset,
               'metadata': {
                   'num_channels': len(channels),
                   'reference_channel': ref_ch
               }
           }
       else:
           # Single-channel logic
           cycles = self._extract_cycles_simple(recorded_audio)
           room_response = self._average_cycles(cycles)
           onset = self._find_onset_in_averaged_signal(room_response)
           impulse = np.roll(room_response, -onset)

           return {
               'raw': recorded_audio,
               'room_response': room_response,
               'impulse': impulse,
               'onset_sample': onset
           }
   ```

2. **Create helper methods:**
   ```python
   def _extract_cycles_simple(self, audio: np.ndarray) -> np.ndarray:
       """Simple reshape extraction."""
       expected_samples = self.cycle_samples * self.num_pulses
       if len(audio) < expected_samples:
           padded = np.zeros(expected_samples)
           padded[:len(audio)] = audio
           audio = padded
       else:
           audio = audio[:expected_samples]
       return audio.reshape(self.num_pulses, self.cycle_samples)

   def _average_cycles(self, cycles: np.ndarray, start_cycle: int = 0) -> np.ndarray:
       """Average cycles starting from start_cycle."""
       return np.mean(cycles[start_cycle:], axis=0)
   ```

3. **Remove old methods:**
   - `_process_single_channel_signal()`
   - `_process_multichannel_signal()`

4. **Test:** Verify standard recording still works

### 6.4 Phase 4: Implement Calibration Mode

**Goal:** Move calibration test logic into recorder

**Tasks:**

1. **Create `_process_calibration_mode()`:**
   ```python
   def _process_calibration_mode(self, recorded_audio) -> Dict[str, Any]:
       """
       Process recording for calibration mode.

       Returns cycle-level data for quality analysis.
       """
       from calibration_validator_v2 import CalibrationValidatorV2, QualityThresholds

       # Get calibration channel
       cal_ch = self.multichannel_config.get('calibration_channel')
       if cal_ch is None:
           raise ValueError("Calibration mode requires calibration_channel in config")

       # Extract calibration channel
       is_multichannel = isinstance(recorded_audio, dict)
       if is_multichannel:
           cal_raw = recorded_audio[cal_ch]
       else:
           cal_raw = recorded_audio

       # Extract cycles
       initial_cycles = self._extract_cycles_simple(cal_raw)

       # Validate cycles
       thresholds = QualityThresholds.from_config(self.calibration_quality_config)
       validator = CalibrationValidatorV2(thresholds, self.sample_rate)

       validation_results = []
       for i, cycle in enumerate(initial_cycles):
           validation = validator.validate_cycle(cycle, i)
           validation_results.append({
               'cycle_index': i,
               'calibration_valid': validation.calibration_valid,
               'calibration_metrics': validation.calibration_metrics,
               'calibration_failures': validation.calibration_failures
           })

       # Align cycles
       alignment_result = self.align_cycles_by_onset(
           initial_cycles,
           validation_results,
           correlation_threshold=0.7
       )

       # Apply to all channels if multi-channel
       aligned_multichannel = {}
       if is_multichannel:
           for ch_name, ch_data in recorded_audio.items():
               aligned_multichannel[ch_name] = self.apply_alignment_to_channel(
                   ch_data,
                   alignment_result
               )
       else:
           aligned_multichannel[cal_ch] = alignment_result['aligned_cycles']

       return {
           'all_calibration_cycles': initial_cycles,
           'validation_results': validation_results,
           'aligned_multichannel_cycles': aligned_multichannel,
           'alignment_metadata': alignment_result,
           'calibration_channel': cal_ch,
           'sample_rate': self.sample_rate
       }
   ```

2. **Add `calibration_test()` convenience method:**
   ```python
   def calibration_test(self) -> Dict:
       """
       Perform calibration recording and analysis.

       Returns cycle-level data for GUI display and analysis.
       Does NOT save files (use take_record for that).
       """
       recorded_audio = self._record_audio()
       if recorded_audio is None:
           raise ValueError("Recording failed")

       return self._process_calibration_mode(recorded_audio)
   ```

3. **Update GUI to use new method:**
   ```python
   # In gui_audio_settings_panel.py:
   def _perform_calibration_test(self):
       # Old: Duplicated all the logic
       # New: Just call the recorder
       return self.recorder.calibration_test()
   ```

4. **Test:** Verify calibration test in GUI still works

### 6.5 Phase 5: Remove Redundant Methods

**Goal:** Clean up unused cycle alignment code

**Tasks:**

1. **Remove these methods (confirmed unused):**
   ```python
   extract_and_align_cycles()
   _detect_cycle_onsets()
   _extract_cycles_at_positions()
   _select_reference_cycle()  # (the OLD one at line 1256)
   _align_cycles_to_reference()
   _find_cycle_alignment_shift()
   _calculate_cycle_extraction_quality()
   extract_cycles_with_shifts()
   _calculate_rms_envelope()
   ```

2. **Remove unused configuration:**
   ```python
   # In __init__, remove:
   self.correlation_quality_config = {...}
   ```

3. **Test:** Verify nothing breaks

### 6.6 Phase 6: Rename & Clarify

**Goal:** Improve clarity of remaining methods

**Tasks:**

1. **Rename onset detection methods:**
   ```python
   # Old name (ambiguous):
   _find_onset_in_room_response()

   # New name (clear purpose):
   _find_onset_in_averaged_signal()
   ```

2. **Rename extraction methods:**
   ```python
   # Old name (not in code yet):
   _extract_impulse_response()

   # New name (if kept):
   _extract_impulse_by_rotation()
   ```

3. **Add clear docstrings** indicating which mode each method belongs to

4. **Update class docstring** to explain two modes

### 6.7 Phase 7: Documentation & Testing

**Goal:** Complete refactoring with docs and tests

**Tasks:**

1. **Update documentation:**
   - TECHNICAL_DOCUMENTATION.md - Document two modes
   - PIANO_MULTICHANNEL_PLAN.md - Update status
   - Add inline comments explaining mode-specific logic

2. **Create test script:**
   ```python
   # test_recorder_modes.py

   # Test 1: Standard mode (existing behavior)
   recorder.take_record("test_raw.wav", "test_impulse.wav")

   # Test 2: Calibration mode (new explicit API)
   recorder.take_record("cal_raw.wav", "cal_impulse.wav", mode='calibration')

   # Test 3: Calibration test (convenience method)
   results = recorder.calibration_test()
   ```

3. **Verify backward compatibility:**
   - Existing piano_response.py code should work unchanged
   - GUI calibration test should work with minimal changes

---

## 7. Risk Assessment

### 7.1 High Risk Items

#### Risk 1: Breaking Existing Code

**Likelihood:** Medium
**Impact:** High

**Mitigation:**
- Default mode='standard' preserves existing behavior
- Extensive testing before deployment
- Phase 1 search confirms no hidden dependencies

#### Risk 2: Incorrectly Identifying Unused Methods

**Likelihood:** Low
**Impact:** High

**Mitigation:**
- Comprehensive codebase search in Phase 1
- Test all functionality after removal
- Git allows rollback if needed

### 7.2 Medium Risk Items

#### Risk 3: Logic Errors in Consolidation

**Likelihood:** Medium
**Impact:** Medium

**Mitigation:**
- Incremental refactoring (one phase at a time)
- Test after each phase
- Compare outputs before/after for same inputs

#### Risk 4: GUI Integration Issues

**Likelihood:** Low
**Impact:** Medium

**Mitigation:**
- GUI changes minimal (just call new method)
- Test GUI after Phase 4

### 7.3 Low Risk Items

#### Risk 5: Configuration Migration Issues

**Likelihood:** Low
**Impact:** Low

**Mitigation:**
- Only removing unused config
- Existing configs unaffected

---

## 8. Migration Path

### 8.1 For Existing Code

**No changes required!**

```python
# This still works exactly as before:
recorder = RoomResponseRecorder("config.json")
recorder.take_record("raw.wav", "impulse.wav")
```

### 8.2 For New Code Using Calibration

**Before (using GUI):**
```python
# Calibration test logic embedded in GUI
results = gui._perform_calibration_test()
```

**After (using recorder):**
```python
# Explicit API
results = recorder.calibration_test()

# Or:
results = recorder.take_record("cal_raw.wav", "cal_impulse.wav",
                               mode='calibration')
```

### 8.3 Deprecation Strategy

**No deprecation needed** - existing API unchanged.

**New feature** - `mode` parameter is optional addition.

---

## 9. Expected Outcomes

### 9.1 Code Quality Improvements

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| **Total Methods** | 36 | 28 | -22% |
| **Lines of Code** | ~1600 | ~1400 | -12% |
| **Redundant Methods** | 7 | 0 | -100% |
| **Processing Paths** | 4 | 2 | -50% |
| **Clarity** | Low | High | ✅ |

### 9.2 Maintainability Improvements

1. **Clear Separation:** Standard vs. Calibration modes explicit
2. **Reduced Duplication:** Single multi-channel code path
3. **Easier Testing:** Each mode testable independently
4. **Better Documentation:** Purpose of each method clear
5. **Future Extensibility:** Easy to add new modes if needed

### 9.3 Functionality Improvements

1. **Calibration in Recorder:** No longer GUI-only
2. **Unified Multi-Channel:** Same code for both modes
3. **Explicit API:** Users can choose mode directly
4. **Better Error Messages:** Mode-specific validation

---

## 10. Open Questions

### Q1: Is `extract_and_align_cycles()` used anywhere?

**Action:** ✅ COMPLETED - Searched entire codebase

**Finding:** Only used in `test_cycle_alignment.py` (uncommitted old test file)

**Decision:** ✅ SAFE TO REMOVE - Not used in production code

### Q2: Should calibration mode save files?

**Current:** Calibration test in GUI doesn't save files
**Proposed:** `mode='calibration'` also doesn't save files

**Rationale:** Calibration is for analysis, not production recording

**Alternative:** Add `save_cycles=True` parameter

### Q3: What about the old `_select_reference_cycle()` at line 1256?

**Current:** Used by `extract_and_align_cycles()`
**If that method removed:** This can also be removed

**Action:** Confirm in Phase 1

### Q4: Should `_find_sound_onset()` be kept as utility?

**Current:** Generic RMS-based onset detection
**Usage:** Only by methods we're removing

**Decision:** Remove unless found to be useful elsewhere

---

## 11. Success Criteria

Refactoring is successful if:

1. ✅ All existing code works unchanged (backward compatible)
2. ✅ GUI calibration test works with new API
3. ✅ Standard recording (take_record) works for single and multi-channel
4. ✅ Calibration recording accessible via API
5. ✅ Code is 20%+ smaller (8+ methods removed)
6. ✅ No redundant implementations remain
7. ✅ Documentation updated
8. ✅ Tests pass

---

## 12. Next Steps

1. **Review this plan** with stakeholders
2. **Approve/adjust** proposed changes
3. **Execute Phase 1:** Search and verify unused methods
4. **Proceed** with phased implementation
5. **Test thoroughly** after each phase
6. **Deploy** when all phases complete

---

**Status:** ✅ Planning complete, ready for review and approval
