# RoomResponseRecorder Comprehensive Refactoring Plan V2

**Document Version:** 2.2
**Date:** 2025-10-31
**Status:** ✅ PHASE 2 COMPLETED SUCCESSFULLY
**Work Completed:** Phase 1 (Cleanup) ✅, Phase 2 (Mode Parameter) ✅

---

## ✅ SUCCESS: Phase 2 Implementation Outcome (Second Attempt)

### What Happened This Time

**Phase 2 was successfully re-implemented with ZERO breaking changes and ALL tests passing:**

**Commits on `phase2-risk-mitigated` branch:**
- `8267feb` - feat: Add explicit mode parameter to take_record() API (Phase 2)
- Merged to dev branch successfully

**Key Differences from Previous Failed Attempt:**
1. **Return Type PRESERVED:** Returns raw audio by default (backward compatible)
2. **Standard Mode UNTOUCHED:** Existing code path completely unchanged
3. **Comprehensive Testing:** Unit + Integration + Hardware tests all passed
4. **Bugs Fixed Early:** All issues caught in testing, not production

### Success Metrics

**Testing Results:**
- ✅ Level 1 (Unit Tests): 5/5 PASSED
- ✅ Level 2 (Integration Tests): 5/5 PASSED
- ✅ Level 3 (Hardware Tests): 4/4 PASSED

**Backward Compatibility:** 100% - All existing code works unchanged

**Production Ready:** Yes - Merged to dev branch

---

## 📚 HISTORICAL NOTE: First Phase 2 Attempt (Failed)

### What Happened in First Attempt

**Phase 2 was initially implemented but caused breaking changes that affected system stability:**

**Commits on `refactor/unified-recording-modes` branch:**
- `ecca566` - feat: Add explicit mode parameter to take_record() API (Phase 2)
- `aa54ab7` - fix: Update GUI to handle Phase 2 API changes
- `b6d5922` - fix: Correct parameter name in calculate_thresholds_from_marked_cycles
- `5d411ed` - fix: Update GUI and QualityThresholds for V2 format compatibility
- `b237632` - fix: Improve error reporting for config file save failures

**Issues Discovered:**
1. **API Return Type Change:** Changed `take_record()` to return `Dict` instead of raw audio
   - Required updates throughout GUI code
   - Created cascading compatibility issues
2. **Audio Recording Failure:** After Phase 2, recordings failed with timeout errors
   - "Playback did not complete within timeout"
   - 0 samples captured
   - Device channel mismatch (requested 1, got 2)
3. **Configuration Incompatibility:** Multiple config format mismatches between old/new code
4. **Testing Gaps:** Phase 2 tests didn't catch integration issues with actual hardware

### Decision: Revert to Last Working Version

**Reason:** The refactoring introduced more complexity and instability than it solved. The original architecture, while not ideal, is stable and working.

**Reverted to:** Commit `76cbc33` (Merge refactor/room-response-recorder into dev)
- This includes Phase 1 cleanup (9 unused methods removed)
- Excludes Phase 2 mode parameter changes
- Stable, tested, working state

### Lessons Learned

1. **Breaking Changes Are Risky:** Changing return types of core APIs has cascading effects
2. **Hardware Integration Testing Required:** Unit tests aren't enough for audio I/O code
3. **Incremental Changes Better:** Smaller, more incremental changes easier to debug
4. **Current Architecture Works:** While not perfect, it's stable and functional

### Recommended Path Forward

**Don't attempt full refactoring.** Instead:

1. **Keep existing architecture** - Two separate code paths work fine
2. **Improve documentation** - Document current behavior clearly
3. **Add targeted fixes only** - Fix specific bugs, don't redesign
4. **Accept technical debt** - Some duplication is acceptable for stability

**The perfect is the enemy of the good.** The current implementation works reliably for production use.

---

## Table of Contents

1. [Executive Summary (Original Plan)](#1-executive-summary-original-plan)
2. [Current State After Phase 1](#2-current-state-after-phase-1)
3. [Problem Statement](#3-problem-statement)
4. [Proposed Architecture](#4-proposed-architecture)
5. [Implementation Phases](#5-implementation-phases)
6. [Testing Strategy](#6-testing-strategy)
7. [Risk Assessment & Mitigation](#7-risk-assessment--mitigation)
8. [Success Criteria](#8-success-criteria)

---

## 1. Executive Summary (Original Plan)

### What Was Done (Phase 1 - Completed)

**Commits:**
- `292b62e` - Removed 9 unused cycle alignment methods
- `1e3d729` - Fixed multi-channel data handling in visualization

**Results:**
- Reduced from **36 to 27 methods** (-25% complexity)
- Removed **350 lines of code**
- Fixed critical bug in multi-channel series recording
- Eliminated all unused cycle alignment approaches

### What Remains (Phases 2-6)

The core architectural issue still exists:

**Two fundamentally different recording modes are conflated in a single API:**

1. **Standard Recording Mode** - Record → Average → Save impulse
2. **Calibration Recording Mode** - Record → Validate → Align → Return cycles

Currently, these modes use completely different code paths:
- Standard mode: `take_record()` → `_process_recorded_signal()`
- Calibration mode: GUI calls `_record_method_2()` directly, bypasses `take_record()`

### Refactoring Goals

1. **✅ DONE:** Remove redundant cycle alignment methods
2. **TODO:** Unify the two modes under a single API with explicit `mode` parameter
3. **TODO:** Consolidate single/multi-channel processing paths
4. **TODO:** Move calibration logic from GUI into recorder
5. **TODO:** Improve naming and separation of concerns

---

## 2. Current State After Phase 1

### 2.1 Method Inventory (27 methods)

#### Core API (4 methods)
```python
__init__(config_file_path)              # Constructor
take_record(output, impulse, method)    # Main API - standard mode only
_record_method_2()                      # SDL recording (multi-channel capable)
test_mic(duration, chunk_duration)      # Mic test utility
```

#### Signal Generation (2 methods)
```python
_generate_single_pulse(exact_samples)
_generate_complete_signal()
```

#### Configuration & Validation (4 methods)
```python
_validate_config()
_validate_multichannel_config()
_migrate_calibration_config_v1_to_v2()
get_signal_info()
```

#### Device Management (4 methods)
```python
set_audio_devices(input, output)
list_devices()
get_device_info_with_channels()
test_multichannel_recording(duration)
```

#### Signal Processing (3 methods)
```python
_process_recorded_signal(recorded_audio)        # Dispatcher
_process_single_channel_signal(audio)           # Single-channel path
_process_multichannel_signal(multichannel_audio) # Multi-channel path
```

#### Cycle Alignment (2 methods) ✅ CLEANED
```python
align_cycles_by_onset(initial_cycles, validation_results, threshold)
apply_alignment_to_channel(channel_raw, alignment_metadata)
```

#### Onset Detection (2 methods)
```python
_find_onset_in_room_response(room_response)
_extract_impulse_response(room_response)
```

#### File Saving (4 methods)
```python
_save_wav(audio_data, filename)
_save_multichannel_files(output, impulse, processed_data)
_save_single_channel_files(output, impulse, processed_data)
_make_channel_filename(base_filename, channel_index)
```

#### Utility (2 methods)
```python
get_sdl_core_info()
print_signal_analysis()
```

### 2.2 Current Call Graphs

**Standard Recording (take_record):**
```
take_record()
  └─> _record_method_2()
  └─> _process_recorded_signal()
      ├─> _process_single_channel_signal()
      │     └─> Simple reshape → Average → Find onset → Rotate
      └─> _process_multichannel_signal()
            └─> Per-channel: Simple reshape → Average → Find onset in ref → Apply to all
  └─> _save_single_channel_files() or _save_multichannel_files()
```

**Calibration Recording (GUI direct call):**
```
gui_audio_settings_panel._perform_calibration_test()
  └─> recorder._record_method_2()                    # Bypasses take_record!
  └─> Simple reshape
  └─> CalibrationValidatorV2.validate_cycle()         # Per-cycle validation
  └─> recorder.align_cycles_by_onset()                # Align cycles
  └─> recorder.apply_alignment_to_channel()           # Apply to all channels
  └─> Return cycle data (no saving, no take_record)
```

**Problem:** Calibration mode completely bypasses the main `take_record()` API!

---

## 3. Problem Statement

### 3.1 Architectural Issues

#### Issue 1: Two Recording Modes, No Unified API

**Current State:**
- Standard mode: Use `take_record()`
- Calibration mode: GUI bypasses `take_record()`, calls `_record_method_2()` directly

**Problems:**
- Inconsistent API usage
- Calibration logic scattered in GUI
- No way to do calibration recording from API
- Code duplication between GUI and recorder

#### Issue 2: Single vs Multi-Channel Duplication

**Current State:**
- `_process_single_channel_signal()` - 35 lines
- `_process_multichannel_signal()` - 64 lines
- 90% code overlap (reshape, average, onset detection)

**Problems:**
- Maintenance burden (fix bugs in two places)
- Inconsistent behavior risk
- Unnecessary complexity

#### Issue 3: Unclear Method Responsibilities

**Confusion:**
- What does `_process_recorded_signal()` do?
  - Only handles standard mode
  - Doesn't know about calibration mode
- When should `align_cycles_by_onset()` be used?
  - Only by GUI for calibration
  - Not integrated into recorder's main flow

#### Issue 4: GUI Does Recorder's Job

**Current State:**
The GUI `_perform_calibration_test()` method:
1. Calls `_record_method_2()` to record
2. Extracts cycles (simple reshape)
3. Validates cycles (CalibrationValidatorV2)
4. Aligns cycles (`align_cycles_by_onset()`)
5. Returns cycle data

**This is recorder logic living in the GUI!**

### 3.2 What Users Want

**Use Case 1: Standard Piano Recording**
```python
# Record piano response, get averaged impulse
recorder.take_record("raw.wav", "impulse.wav")
```

**Use Case 2: Calibration Test**
```python
# Record calibration, get cycle-by-cycle data
result = recorder.take_record_calibration()
# Returns: {
#     'calibration_cycles': [...],
#     'validation_results': [...],
#     'aligned_multichannel_cycles': {...},
#     'alignment_metadata': {...}
# }
```

**Use Case 3: Multi-Channel Standard Recording**
```python
# Record multi-channel, get per-channel impulses
recorder.take_record("raw.wav", "impulse.wav")
# Saves: raw_ch0.wav, raw_ch1.wav, impulse_ch0.wav, impulse_ch1.wav
```

---

## 4. Proposed Architecture

### 4.1 High-Level Design

**Unified API with Explicit Modes:**

```python
class RoomResponseRecorder:
    """
    Room response recording with two distinct modes:

    1. Standard Mode (default):
       - Record → Extract cycles → Average → Find onset → Save impulse
       - For: Piano response measurements

    2. Calibration Mode:
       - Record → Extract cycles → Validate → Align → Return cycle data
       - For: System calibration and quality assurance
    """

    def take_record(self, output_file, impulse_file, method=2, mode='standard'):
        """
        Unified recording API.

        Args:
            output_file: Raw recording output path
            impulse_file: Impulse response output path
            method: Recording method (1=manual, 2=auto)
            mode: 'standard' (default) or 'calibration'

        Returns:
            mode='standard': Dict with 'impulse', 'onset_sample', etc.
            mode='calibration': Dict with 'calibration_cycles', 'validation_results', etc.
        """

    def take_record_calibration(self):
        """
        Convenience method for calibration recording.
        Delegates to take_record(mode='calibration', output_file=None, impulse_file=None)
        """
```

### 4.2 Proposed Method Structure (Target: 24 methods)

#### Core API (3 methods) - Simplified
```python
def __init__(config_file_path)
def take_record(output_file, impulse_file, method=2, mode='standard')
def take_record_calibration()  # Convenience wrapper
```

#### Recording (1 method) - Keep as-is
```python
def _record_method_2() -> Union[np.ndarray, Dict[int, np.ndarray]]
```

#### Processing - Mode Dispatch (3 methods) - NEW STRUCTURE
```python
def _process_recorded_signal(recorded_audio, mode='standard') -> Dict
    """Dispatcher: calls _process_standard_mode or _process_calibration_mode"""

def _process_standard_mode(recorded_audio) -> Dict
    """Standard: Extract → Average → Onset → Rotate → Return impulse"""

def _process_calibration_mode(recorded_audio) -> Dict
    """Calibration: Extract → Validate → Align → Return cycles"""
```

#### Cycle Operations (4 methods) - NEW HELPERS
```python
def _extract_cycles(audio, num_cycles, cycle_samples) -> np.ndarray
    """Simple reshape extraction (used by both modes)"""

def _average_cycles(cycles, start_cycle=0) -> np.ndarray
    """Average cycles (used by standard mode)"""

def align_cycles_by_onset(initial_cycles, validation_results, threshold) -> dict
    """Align by onset (used by calibration mode)"""

def apply_alignment_to_channel(channel_raw, alignment_metadata) -> np.ndarray
    """Apply alignment to channel (used by calibration mode)"""
```

#### Onset Detection (2 methods) - Keep, rename for clarity
```python
def _find_onset_in_averaged_signal(room_response) -> int
    """Find onset in averaged signal (standard mode)"""

def _extract_impulse_by_rotation(room_response, onset) -> np.ndarray
    """Rotate to extract impulse (standard mode)"""
```

#### File Management (3 methods) - Consolidate
```python
def _save_files(output_file, impulse_file, processed_data)
    """Unified save (detects single vs multi-channel)"""

def _save_wav(audio_data, filename)
    """WAV file writer"""

def _make_channel_filename(base_filename, channel_index)
    """Generate channel-specific filename"""
```

#### Signal Generation (2 methods) - Keep as-is
```python
def _generate_single_pulse(exact_samples)
def _generate_complete_signal()
```

#### Configuration & Validation (4 methods) - Keep as-is
```python
def _validate_config()
def _validate_multichannel_config()
def _migrate_calibration_config_v1_to_v2()
def get_signal_info()
```

#### Device Management (4 methods) - Keep as-is
```python
def set_audio_devices(input, output)
def list_devices()
def get_device_info_with_channels()
def test_multichannel_recording(duration)
```

#### Utility (2 methods) - Keep as-is
```python
def get_sdl_core_info()
def print_signal_analysis()
```

### 4.3 Method Count Comparison

| Phase | Methods | Description |
|-------|---------|-------------|
| **Before Refactor** | 36 | Original state with 3 cycle alignment approaches |
| **After Phase 1** | 27 | Removed 9 unused cycle alignment methods |
| **After Phase 2-6** | 24 | Unified modes, consolidated processing |

**Net reduction: 12 methods (33% smaller than original)**

### 4.4 New Call Graphs

**Standard Mode:**
```
take_record(mode='standard')
  └─> _record_method_2()
  └─> _process_recorded_signal(mode='standard')
      └─> _process_standard_mode()
          ├─> Single-channel:
          │     └─> _extract_cycles() → _average_cycles() → _find_onset_in_averaged_signal()
          └─> Multi-channel:
                └─> Per-channel: _extract_cycles() → _average_cycles()
                └─> _find_onset_in_averaged_signal(reference_channel)
                └─> Apply same shift to all channels
  └─> _save_files()
```

**Calibration Mode:**
```
take_record_calibration()  OR  take_record(mode='calibration')
  └─> _record_method_2()
  └─> _process_recorded_signal(mode='calibration')
      └─> _process_calibration_mode()
          ├─> _extract_cycles(calibration_channel)
          ├─> CalibrationValidatorV2.validate_cycle() for each cycle
          ├─> align_cycles_by_onset()
          └─> apply_alignment_to_channel() for all channels
  └─> Return cycle data (no file saving in calibration mode)
```

---

## 5. Implementation Phases

### Phase 1: Cleanup Unused Methods ✅ COMPLETED

**Status:** ✅ **DONE** (commits 292b62e, 1e3d729)

**What was accomplished:**
- Removed 9 unused cycle alignment methods
- Removed `correlation_quality_config`
- Fixed multi-channel visualization bug
- Reduced from 36 to 27 methods

---

### Phase 2: Add Mode Parameter & Refactor Processing Dispatch

**Goal:** Add `mode` parameter to API without breaking existing code

**Duration:** 2-3 hours
**Risk:** LOW (additive changes, defaults preserve existing behavior)

#### Tasks

**2.1: Update take_record() signature**
```python
def take_record(self,
                output_file: str,
                impulse_file: str,
                method: int = 2,
                mode: str = 'standard') -> Dict[str, Any]:
    """
    Record and process room response.

    Args:
        output_file: Path for raw recording WAV file
        impulse_file: Path for impulse response WAV file
        method: Recording method (2=SDL auto)
        mode: 'standard' (default) or 'calibration'

    Returns:
        Standard mode: {'impulse': array, 'onset_sample': int, ...}
        Calibration mode: {'calibration_cycles': array, 'validation_results': list, ...}
    """
    # Validate mode
    if mode not in ['standard', 'calibration']:
        raise ValueError(f"Invalid mode: {mode}. Must be 'standard' or 'calibration'")

    # Record audio
    if method == 2:
        recorded_audio = self._record_method_2()
    else:
        raise ValueError(f"Recording method {method} not implemented")

    if recorded_audio is None:
        raise RuntimeError("Recording failed")

    # Process based on mode
    processed_data = self._process_recorded_signal(recorded_audio, mode=mode)

    # Save files (standard mode only)
    if mode == 'standard':
        if output_file and impulse_file:
            self._save_files(output_file, impulse_file, processed_data)

    return processed_data
```

**2.2: Update _process_recorded_signal() dispatcher**
```python
def _process_recorded_signal(self, recorded_audio, mode='standard') -> Dict[str, Any]:
    """
    Process recorded audio based on mode.

    Args:
        recorded_audio: Single-channel array or multi-channel dict
        mode: 'standard' or 'calibration'

    Returns:
        Processed data structure (varies by mode)
    """
    if mode == 'standard':
        return self._process_standard_mode(recorded_audio)
    elif mode == 'calibration':
        return self._process_calibration_mode(recorded_audio)
    else:
        raise ValueError(f"Unknown mode: {mode}")
```

**2.3: Add convenience method**
```python
def take_record_calibration(self) -> Dict[str, Any]:
    """
    Convenience method for calibration recording.

    Records audio, validates calibration cycles, and returns cycle data.
    Does NOT save files.

    Returns:
        {
            'calibration_cycles': np.ndarray,       # All cycles [N, samples]
            'validation_results': List[Dict],       # Per-cycle validation
            'aligned_multichannel_cycles': Dict,    # Per-channel aligned cycles
            'alignment_metadata': Dict,             # Alignment info
            'num_valid_cycles': int,
            'num_aligned_cycles': int
        }
    """
    return self.take_record(
        output_file=None,
        impulse_file=None,
        method=2,
        mode='calibration'
    )
```

**2.4: Testing**
- [ ] Test `take_record(mode='standard')` with single-channel
- [ ] Test `take_record(mode='standard')` with multi-channel
- [ ] Verify existing code still works (default mode)
- [ ] Test invalid mode raises ValueError

**Commit Message:**
```
feat: Add explicit mode parameter to take_record() API

Add 'mode' parameter to support both standard and calibration recording
modes through unified API.

Changes:
- take_record(): Add mode='standard'|'calibration' parameter
- _process_recorded_signal(): Add mode dispatch
- Add take_record_calibration() convenience method
- Mode validation with clear error messages

Backward compatible: mode='standard' is default, preserves existing behavior

Next: Implement _process_standard_mode() and _process_calibration_mode()
```

---

### Phase 3: Implement Unified Standard Mode Processing

**Goal:** Consolidate single-channel and multi-channel standard processing into one method

**Duration:** 3-4 hours
**Risk:** MEDIUM (refactoring existing logic, must preserve behavior exactly)

#### Tasks

**3.1: Create helper methods**

```python
def _extract_cycles(self, audio: np.ndarray) -> np.ndarray:
    """
    Extract cycles using simple reshape.

    Args:
        audio: Raw audio signal

    Returns:
        Cycles array [num_cycles, cycle_samples]
    """
    expected_samples = self.cycle_samples * self.num_pulses

    # Pad or trim to expected length
    if len(audio) < expected_samples:
        padded = np.zeros(expected_samples, dtype=audio.dtype)
        padded[:len(audio)] = audio
        audio = padded
    else:
        audio = audio[:expected_samples]

    # Reshape into cycles
    return audio.reshape(self.num_pulses, self.cycle_samples)


def _average_cycles(self, cycles: np.ndarray, start_cycle: int = None) -> np.ndarray:
    """
    Average cycles starting from start_cycle.

    Args:
        cycles: Cycles array [num_cycles, cycle_samples]
        start_cycle: Index to start averaging from (default: num_pulses // 4)

    Returns:
        Averaged signal [cycle_samples]
    """
    if start_cycle is None:
        start_cycle = max(1, self.num_pulses // 4)

    return np.mean(cycles[start_cycle:], axis=0)
```

**3.2: Implement _process_standard_mode()**

```python
def _process_standard_mode(self, recorded_audio) -> Dict[str, Any]:
    """
    Process recording in standard mode.

    Standard mode:
    1. Extract cycles (simple reshape)
    2. Average cycles (skip first few)
    3. Find onset in averaged signal
    4. Rotate to align onset
    5. Return impulse response

    Handles both single-channel and multi-channel recordings.

    Args:
        recorded_audio: np.ndarray (single) or Dict[int, np.ndarray] (multi)

    Returns:
        {
            'raw': recorded_audio,
            'room_response': averaged signal(s),
            'impulse': impulse response(s),
            'onset_sample': int,
            'metadata': dict
        }
    """
    is_multichannel = isinstance(recorded_audio, dict)

    if is_multichannel:
        return self._process_standard_multichannel(recorded_audio)
    else:
        return self._process_standard_single(recorded_audio)


def _process_standard_single(self, audio: np.ndarray) -> Dict[str, Any]:
    """Process single-channel standard recording."""
    # Extract and average cycles
    cycles = self._extract_cycles(audio)
    room_response = self._average_cycles(cycles)

    # Find onset and extract impulse
    onset_sample = self._find_onset_in_averaged_signal(room_response)
    impulse = self._extract_impulse_by_rotation(room_response, onset_sample)

    return {
        'raw': audio,
        'room_response': room_response,
        'impulse': impulse,
        'onset_sample': onset_sample,
        'metadata': {
            'mode': 'standard',
            'num_cycles': self.num_pulses,
            'cycle_samples': self.cycle_samples
        }
    }


def _process_standard_multichannel(self, recorded_audio: Dict[int, np.ndarray]) -> Dict[str, Any]:
    """Process multi-channel standard recording."""
    ref_ch = self.multichannel_config.get('reference_channel', 0)

    # Process each channel: extract cycles and average
    averaged_channels = {}
    for ch_idx, audio in recorded_audio.items():
        cycles = self._extract_cycles(audio)
        averaged_channels[ch_idx] = self._average_cycles(cycles)

    # Find onset in reference channel
    onset_sample = self._find_onset_in_averaged_signal(averaged_channels[ref_ch])

    # Apply SAME rotation to all channels
    impulse_channels = {}
    for ch_idx, room_response in averaged_channels.items():
        impulse_channels[ch_idx] = self._extract_impulse_by_rotation(room_response, onset_sample)

    return {
        'raw': recorded_audio,
        'room_response': averaged_channels,
        'impulse': impulse_channels,
        'onset_sample': onset_sample,
        'metadata': {
            'mode': 'standard',
            'num_channels': len(recorded_audio),
            'reference_channel': ref_ch,
            'num_cycles': self.num_pulses,
            'cycle_samples': self.cycle_samples
        }
    }
```

**3.3: Rename onset methods for clarity**

```python
def _find_onset_in_averaged_signal(self, room_response: np.ndarray) -> int:
    """
    Find onset in an averaged signal using moving RMS.

    Used by standard mode to find onset in averaged room response.

    Args:
        room_response: Averaged signal

    Returns:
        Onset sample index
    """
    # Keep existing _find_onset_in_room_response implementation
    # Just rename for clarity
    ...


def _extract_impulse_by_rotation(self, room_response: np.ndarray, onset: int) -> np.ndarray:
    """
    Extract impulse by rotating onset to beginning.

    Args:
        room_response: Averaged signal
        onset: Onset sample index

    Returns:
        Impulse response (rotated signal)
    """
    # Keep existing _extract_impulse_response implementation
    # Just rename for clarity
    return np.roll(room_response, -onset)
```

**3.4: Remove old methods**

Delete:
- `_process_single_channel_signal()`
- `_process_multichannel_signal()`

**3.5: Update _save_files() to handle both formats**

```python
def _save_files(self, output_file: str, impulse_file: str, processed_data: Dict[str, Any]):
    """
    Save files for standard mode recording.

    Detects single vs multi-channel from processed_data structure.
    """
    # Detect format from metadata
    is_multichannel = 'num_channels' in processed_data.get('metadata', {})

    if is_multichannel:
        self._save_multichannel_files(output_file, impulse_file, processed_data)
    else:
        self._save_single_channel_files(output_file, impulse_file, processed_data)
```

**3.6: Testing**
- [ ] Record single-channel, verify impulse matches previous behavior
- [ ] Record multi-channel, verify all channels aligned correctly
- [ ] Compare onset detection results with old implementation
- [ ] Verify file saving works correctly
- [ ] Test with different num_pulses and cycle_samples
- [ ] Run existing test suite

**Commit Message:**
```
refactor: Unify single and multi-channel standard mode processing

Consolidate _process_single_channel_signal() and _process_multichannel_signal()
into unified _process_standard_mode().

Changes:
- Add _extract_cycles() helper (simple reshape)
- Add _average_cycles() helper (skip first cycles)
- Implement _process_standard_mode() with single/multi-channel branches
- Rename onset methods for clarity:
  - _find_onset_in_room_response → _find_onset_in_averaged_signal
  - _extract_impulse_response → _extract_impulse_by_rotation
- Remove duplicate processing methods (2 methods → 1)
- Update _save_files() to auto-detect single vs multi-channel

Result: Unified standard mode processing, easier to maintain

Next: Implement _process_calibration_mode()
```

---

### Phase 4: Implement Calibration Mode Processing

**Goal:** Move calibration test logic from GUI into recorder

**Duration:** 4-5 hours
**Risk:** MEDIUM (new feature, must integrate CalibrationValidatorV2)

#### Tasks

**4.1: Implement _process_calibration_mode()**

```python
def _process_calibration_mode(self, recorded_audio) -> Dict[str, Any]:
    """
    Process recording in calibration mode.

    Calibration mode:
    1. Extract cycles (simple reshape)
    2. Validate each cycle (CalibrationValidatorV2)
    3. Align valid cycles by onset
    4. Apply same alignment to all channels
    5. Return cycle-level data (NO averaging, NO file saving)

    Requires multi-channel config with calibration_channel set.

    Args:
        recorded_audio: Dict[int, np.ndarray] (multi-channel required)

    Returns:
        {
            'calibration_cycles': np.ndarray,           # All cycles [N, samples]
            'validation_results': List[Dict],           # Per-cycle validation
            'aligned_multichannel_cycles': Dict[int, np.ndarray],  # Aligned cycles
            'alignment_metadata': Dict,                 # Alignment info
            'num_valid_cycles': int,
            'num_aligned_cycles': int,
            'metadata': Dict
        }
    """
    from calibration_validator_v2 import CalibrationValidatorV2, QualityThresholds

    # Validate calibration setup
    if not isinstance(recorded_audio, dict):
        raise ValueError("Calibration mode requires multi-channel recording")

    cal_ch = self.multichannel_config.get('calibration_channel')
    if cal_ch is None:
        raise ValueError("Calibration mode requires 'calibration_channel' in multichannel_config")

    if cal_ch not in recorded_audio:
        raise ValueError(f"Calibration channel {cal_ch} not found in recorded data")

    # Extract calibration channel cycles
    cal_raw = recorded_audio[cal_ch]
    initial_cycles = self._extract_cycles(cal_raw)

    # Validate each cycle
    thresholds = QualityThresholds.from_dict(self.calibration_quality_config)
    validator = CalibrationValidatorV2(thresholds, self.sample_rate)

    validation_results = []
    for i, cycle in enumerate(initial_cycles):
        validation = validator.validate_cycle(cycle, i)
        validation_dict = {
            'cycle_index': i,
            'is_valid': validation.is_valid,
            'negative_peak': validation.negative_peak_amplitude,
            'positive_peak': validation.positive_peak_amplitude,
            'aftershock': validation.aftershock_amplitude,
            'issues': validation.issues
        }
        validation_results.append(validation_dict)

    # Count valid cycles
    num_valid = sum(1 for v in validation_results if v['is_valid'])

    print(f"Calibration validation: {num_valid}/{len(validation_results)} valid cycles")

    # Align cycles by onset
    correlation_threshold = 0.7  # Could be configurable
    alignment_result = self.align_cycles_by_onset(
        initial_cycles,
        validation_results,
        correlation_threshold=correlation_threshold
    )

    # Apply alignment to all channels
    aligned_multichannel_cycles = {}
    for ch_idx, channel_data in recorded_audio.items():
        aligned_channel = self.apply_alignment_to_channel(
            channel_data,
            alignment_result
        )
        aligned_multichannel_cycles[ch_idx] = aligned_channel

    return {
        'calibration_cycles': initial_cycles,
        'validation_results': validation_results,
        'aligned_multichannel_cycles': aligned_multichannel_cycles,
        'alignment_metadata': alignment_result,
        'num_valid_cycles': num_valid,
        'num_aligned_cycles': alignment_result['num_aligned'],
        'metadata': {
            'mode': 'calibration',
            'calibration_channel': cal_ch,
            'num_channels': len(recorded_audio),
            'num_cycles': self.num_pulses,
            'cycle_samples': self.cycle_samples,
            'correlation_threshold': correlation_threshold
        }
    }
```

**4.2: Testing**
- [ ] Test calibration mode with multi-channel recording
- [ ] Verify validation results match GUI implementation
- [ ] Verify alignment matches GUI implementation
- [ ] Test with invalid calibration channel
- [ ] Test with single-channel (should raise error)
- [ ] Test error handling for missing config

**Commit Message:**
```
feat: Implement calibration mode in recorder

Move calibration test logic from GUI into RoomResponseRecorder.

Changes:
- Implement _process_calibration_mode()
- Integrate CalibrationValidatorV2 for per-cycle validation
- Apply onset-based alignment to all channels
- Return cycle-level data (no averaging, no file saving)
- Validate calibration setup (multi-channel, calibration_channel)

Result: Calibration mode now accessible via take_record(mode='calibration')

Next: Update GUI to use new API
```

---

### Phase 5: Update GUI to Use New API

**Goal:** Replace GUI's direct `_record_method_2()` calls with new `take_record_calibration()`

**Duration:** 2-3 hours
**Risk:** LOW (replacing implementation, not changing GUI behavior)

#### Tasks

**5.1: Update gui_audio_settings_panel.py**

**Before (current):**
```python
def _perform_calibration_test(self) -> Dict:
    """Calibration test with onset-based alignment."""

    # Record audio directly
    recorded_audio = self.recorder._record_method_2()

    # Extract cycles
    cal_raw = recorded_audio[cal_ch]
    initial_cycles = cal_raw.reshape(num_pulses, cycle_samples)

    # Validate
    validator = CalibrationValidatorV2(thresholds, sample_rate)
    validation_results = []
    for cycle in initial_cycles:
        validation = validator.validate_cycle(cycle, i)
        validation_results.append(validation_dict)

    # Align
    alignment_result = self.recorder.align_cycles_by_onset(...)

    # Apply to all channels
    aligned_multichannel_cycles = {}
    for channel_name, channel_data in recorded_audio.items():
        aligned_channel = self.recorder.apply_alignment_to_channel(...)
        aligned_multichannel_cycles[channel_name] = aligned_channel

    return {...}
```

**After (simplified):**
```python
def _perform_calibration_test(self) -> Dict:
    """
    Perform calibration test using recorder's calibration mode.

    Returns same data structure as before for backward compatibility.
    """
    try:
        # Use new calibration API
        result = self.recorder.take_record_calibration()

        # Result already contains everything we need:
        # - calibration_cycles
        # - validation_results
        # - aligned_multichannel_cycles
        # - alignment_metadata
        # - num_valid_cycles, num_aligned_cycles

        return result

    except Exception as e:
        print(f"Calibration test failed: {e}")
        raise
```

**5.2: Verify GUI behavior unchanged**
- [ ] Test calibration test in GUI
- [ ] Verify visualization shows same data
- [ ] Verify validation results display correctly
- [ ] Test with single-channel (should show error)

**5.3: Remove redundant code from GUI**

The GUI's `_perform_calibration_test()` method can now be 10 lines instead of 100+ lines.

**Commit Message:**
```
refactor: Update GUI to use recorder's calibration mode

Replace GUI's manual calibration test implementation with
take_record_calibration() API.

Changes:
- gui_audio_settings_panel._perform_calibration_test(): Use take_record_calibration()
- Remove ~100 lines of calibration logic from GUI
- Same behavior, much simpler code

Result: Calibration logic centralized in recorder, GUI is thin layer
```

---

### Phase 6: Documentation & Cleanup

**Goal:** Update documentation, add examples, final cleanup

**Duration:** 2-3 hours
**Risk:** NONE (documentation only)

#### Tasks

**6.1: Update TECHNICAL_DOCUMENTATION.md**

Add section:
```markdown
### 10.7 Recording Modes

RoomResponseRecorder supports two distinct recording modes:

#### Standard Mode (default)

For piano response measurements. Records impulse response, averages cycles,
finds onset, and saves impulse file.

**Usage:**
```python
recorder = RoomResponseRecorder("config.json")
result = recorder.take_record("raw.wav", "impulse.wav")
# Files saved: raw.wav, impulse.wav (or per-channel files)
```

**Process:**
1. Record audio (single or multi-channel)
2. Extract cycles (simple reshape)
3. Average cycles (skip first few for stabilization)
4. Find onset in averaged signal
5. Rotate to align onset to beginning
6. Save raw recording and impulse response

#### Calibration Mode

For system calibration and quality assurance. Records with calibration
channel, validates each cycle, aligns cycles, returns cycle data.

**Usage:**
```python
result = recorder.take_record_calibration()
# No files saved, returns cycle data

# Result contains:
# - calibration_cycles: All cycles [N, samples]
# - validation_results: Per-cycle validation data
# - aligned_multichannel_cycles: Aligned cycles per channel
# - alignment_metadata: Alignment info (shifts, correlations)
```

**Process:**
1. Record audio (multi-channel required)
2. Extract cycles from calibration channel
3. Validate each cycle (CalibrationValidatorV2)
4. Align valid cycles by onset detection
5. Apply same alignment to all channels
6. Return cycle-level data (no averaging)
```

**6.2: Update PIANO_MULTICHANNEL_PLAN.md**

Update status to reflect completion:
```markdown
## Status

Version: 1.6
Status: 85% complete
Last Updated: 2025-10-31

### Completed
- ✅ Multi-channel recording
- ✅ Calibration validation V2
- ✅ Onset-based cycle alignment
- ✅ Multi-channel visualization
- ✅ Unified recording modes API
- ✅ Code cleanup and consolidation

### In Progress
- 🔄 Testing and validation

### Remaining
- 📋 Performance optimization
- 📋 User documentation
```

**6.3: Add usage examples**

Create `RECORDING_EXAMPLES.md`:

```markdown
# RoomResponseRecorder Usage Examples

## Standard Recording - Single Channel

```python
from RoomResponseRecorder import RoomResponseRecorder

# Initialize
recorder = RoomResponseRecorder("config.json")

# Record piano response
result = recorder.take_record(
    output_file="recordings/piano_raw.wav",
    impulse_file="recordings/piano_impulse.wav"
)

print(f"Onset found at sample: {result['onset_sample']}")
print(f"Impulse length: {len(result['impulse'])} samples")
```

## Standard Recording - Multi-Channel

```python
# Configure multi-channel in config.json:
# {
#   "multichannel_config": {
#     "enabled": true,
#     "channels": [0, 1, 2, 3],
#     "reference_channel": 0
#   }
# }

recorder = RoomResponseRecorder("config.json")

result = recorder.take_record(
    output_file="recordings/piano_raw.wav",
    impulse_file="recordings/piano_impulse.wav"
)

# Files saved:
# - piano_raw_ch0.wav, piano_raw_ch1.wav, ...
# - piano_impulse_ch0.wav, piano_impulse_ch1.wav, ...

print(f"Recorded {result['metadata']['num_channels']} channels")
```

## Calibration Recording

```python
# Configure calibration in config.json:
# {
#   "multichannel_config": {
#     "enabled": true,
#     "channels": [0, 1, 2, 3],
#     "calibration_channel": 3
#   },
#   "calibration_quality_config": {
#     "min_negative_peak": 0.1,
#     "max_negative_peak": 0.95,
#     ...
#   }
# }

recorder = RoomResponseRecorder("config.json")

# Perform calibration test
result = recorder.take_record_calibration()

# Analyze results
print(f"Valid cycles: {result['num_valid_cycles']}/{len(result['calibration_cycles'])}")
print(f"Aligned cycles: {result['num_aligned_cycles']}")

# Check validation
for v in result['validation_results']:
    if not v['is_valid']:
        print(f"Cycle {v['cycle_index']}: {v['issues']}")

# Access aligned data
aligned_cal = result['aligned_multichannel_cycles'][3]
print(f"Aligned calibration shape: {aligned_cal.shape}")
```
```

**6.4: Update ROOMRESPONSERECORDER_REFACTORING_PLAN.md**

Add completion status:
```markdown
## Refactoring Status

**Phase 1:** ✅ COMPLETED - Cleanup unused methods (commits 292b62e, 1e3d729)
**Phase 2:** ✅ COMPLETED - Add mode parameter
**Phase 3:** ✅ COMPLETED - Unified standard mode
**Phase 4:** ✅ COMPLETED - Calibration mode
**Phase 5:** ✅ COMPLETED - Update GUI
**Phase 6:** ✅ COMPLETED - Documentation

**Final Result:**
- 36 methods → 24 methods (-33% complexity)
- ~500 lines removed
- Unified API for both modes
- Clear separation of concerns
- Easier to test and maintain
```

**Commit Message:**
```
docs: Update documentation for unified recording modes

Update all documentation to reflect completed refactoring.

Changes:
- TECHNICAL_DOCUMENTATION.md: Add section 10.7 Recording Modes
- PIANO_MULTICHANNEL_PLAN.md: Update status to 85% complete
- Add RECORDING_EXAMPLES.md with usage examples
- Update ROOMRESPONSERECORDER_REFACTORING_PLAN.md with completion status

Result: Complete documentation of new unified API
```

---

## 6. Testing Strategy

### 6.1 Unit Tests

Create `tests/test_recording_modes.py`:

```python
import pytest
import numpy as np
from RoomResponseRecorder import RoomResponseRecorder

class TestStandardMode:
    """Test standard recording mode."""

    def test_single_channel_recording(self):
        """Test standard mode with single channel."""
        recorder = RoomResponseRecorder()
        # Mock recording...
        result = recorder.take_record("raw.wav", "impulse.wav")

        assert 'impulse' in result
        assert 'onset_sample' in result
        assert result['metadata']['mode'] == 'standard'

    def test_multi_channel_recording(self):
        """Test standard mode with multi-channel."""
        # Configure multi-channel
        recorder = RoomResponseRecorder("config_multichannel.json")
        result = recorder.take_record("raw.wav", "impulse.wav")

        assert isinstance(result['impulse'], dict)
        assert result['metadata']['num_channels'] > 1

    def test_onset_detection(self):
        """Test onset detection in averaged signal."""
        recorder = RoomResponseRecorder()
        # Create test signal with known onset
        # Verify onset detection accuracy

class TestCalibrationMode:
    """Test calibration recording mode."""

    def test_calibration_recording(self):
        """Test calibration mode returns cycle data."""
        recorder = RoomResponseRecorder("config_calibration.json")
        result = recorder.take_record_calibration()

        assert 'calibration_cycles' in result
        assert 'validation_results' in result
        assert 'aligned_multichannel_cycles' in result
        assert result['metadata']['mode'] == 'calibration'

    def test_validation_integration(self):
        """Test CalibrationValidatorV2 integration."""
        recorder = RoomResponseRecorder("config_calibration.json")
        result = recorder.take_record_calibration()

        assert 'num_valid_cycles' in result
        assert len(result['validation_results']) == recorder.num_pulses

    def test_alignment_integration(self):
        """Test alignment integration."""
        recorder = RoomResponseRecorder("config_calibration.json")
        result = recorder.take_record_calibration()

        assert 'alignment_metadata' in result
        assert 'num_aligned_cycles' in result

    def test_requires_multichannel(self):
        """Test calibration mode requires multi-channel."""
        recorder = RoomResponseRecorder()  # Single-channel config

        with pytest.raises(ValueError, match="multi-channel"):
            recorder.take_record_calibration()

class TestAPIValidation:
    """Test API validation and error handling."""

    def test_invalid_mode(self):
        """Test invalid mode raises error."""
        recorder = RoomResponseRecorder()

        with pytest.raises(ValueError, match="Invalid mode"):
            recorder.take_record("raw.wav", "impulse.wav", mode="invalid")

    def test_calibration_without_cal_channel(self):
        """Test calibration mode without calibration_channel config."""
        recorder = RoomResponseRecorder("config_multichannel_no_cal.json")

        with pytest.raises(ValueError, match="calibration_channel"):
            recorder.take_record_calibration()
```

### 6.2 Integration Tests

**Test Scenarios:**

1. **Standard Recording Workflow**
   - Configure recorder with test signal
   - Record actual audio (or mock)
   - Verify impulse response extracted correctly
   - Verify files saved correctly

2. **Calibration Recording Workflow**
   - Configure recorder with calibration channel
   - Record actual audio (or mock)
   - Verify validation results
   - Verify alignment results
   - Verify no files saved

3. **Backward Compatibility**
   - Run existing piano_response.py application
   - Verify standard recording still works
   - Verify GUI calibration test still works

### 6.3 Manual Testing Checklist

- [ ] Standard recording - single channel - Record, check impulse file
- [ ] Standard recording - multi-channel - Record, check per-channel files
- [ ] Calibration recording - Run calibration test in GUI
- [ ] Calibration recording - Call take_record_calibration() directly
- [ ] Error cases - Invalid mode, missing config, wrong channels
- [ ] File saving - Verify WAV files playable
- [ ] Multi-channel alignment - Verify all channels aligned to reference

---

## 7. Risk Assessment & Mitigation

### 7.1 Risks by Phase

| Phase | Risk Level | Primary Risks | Mitigation |
|-------|-----------|---------------|------------|
| Phase 1 | ✅ DONE | N/A | Completed successfully |
| Phase 2 | LOW | Breaking existing code | Default mode preserves behavior, additive changes only |
| Phase 3 | MEDIUM | Processing behavior changes | Extensive testing, compare with old implementation |
| Phase 4 | MEDIUM | Integration with CalibrationValidatorV2 | Match GUI implementation exactly |
| Phase 5 | LOW | GUI behavior changes | Verify same results before/after |
| Phase 6 | NONE | Documentation only | No code changes |

### 7.2 Rollback Strategy

Each phase is committed separately. If issues arise:

1. **Identify problematic phase** from git log
2. **Revert specific commit:**
   ```bash
   git revert <commit-hash>
   ```
3. **Address issues** and re-implement
4. **Alternative:** Maintain refactor branch separate from main until fully tested

### 7.3 Backward Compatibility

**Guaranteed:**
- Existing `take_record()` calls work unchanged (mode='standard' is default)
- File formats unchanged
- Configuration file format unchanged (additive only)

**Breaking Changes:**
- None for external API
- Internal: `_process_single_channel_signal()` and `_process_multichannel_signal()` removed
  - Only affects code directly calling private methods (none known)

---

## 8. Success Criteria

### 8.1 Functional Success

- [ ] Standard recording works for single-channel
- [ ] Standard recording works for multi-channel
- [ ] Calibration recording returns correct data structure
- [ ] Calibration validation matches GUI implementation
- [ ] Calibration alignment matches GUI implementation
- [ ] GUI calibration test uses new API
- [ ] All files saved correctly in standard mode
- [ ] No files saved in calibration mode

### 8.2 Code Quality Success

- [ ] Method count reduced to 24 (from 36)
- [ ] No code duplication between single/multi-channel
- [ ] Clear separation of standard vs calibration mode
- [ ] All methods have single clear purpose
- [ ] Configuration properly validated

### 8.3 Documentation Success

- [ ] API documented with usage examples
- [ ] Recording modes explained clearly
- [ ] Migration guide for any breaking changes
- [ ] All documentation updated

### 8.4 Testing Success

- [ ] Unit tests pass for all modes
- [ ] Integration tests pass
- [ ] Manual testing checklist completed
- [ ] No regressions in existing functionality

---

## 9. Timeline Estimate

| Phase | Duration | Dependencies |
|-------|----------|--------------|
| Phase 1 | ✅ DONE | None |
| Phase 2 | 2-3 hours | Phase 1 |
| Phase 3 | 3-4 hours | Phase 2 |
| Phase 4 | 4-5 hours | Phase 3 |
| Phase 5 | 2-3 hours | Phase 4 |
| Phase 6 | 2-3 hours | Phase 5 |
| **Total** | **13-18 hours** | Sequential |

**Recommended Schedule:**
- Day 1: Phases 2-3 (5-7 hours)
- Day 2: Phases 4-5 (6-8 hours)
- Day 3: Phase 6 + testing (2-3 hours)

---

## 10. Next Steps

**To begin Phase 2:**

1. **Checkout new branch:**
   ```bash
   git checkout -b refactor/unified-recording-modes
   ```

2. **Start with Phase 2.1:** Update `take_record()` signature

3. **Follow implementation tasks** exactly as documented

4. **Test thoroughly** before moving to next phase

5. **Commit after each phase** with detailed commit message

**Ready to proceed?** 🚀
