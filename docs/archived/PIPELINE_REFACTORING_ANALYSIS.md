# RoomResponseRecorder Pipeline Architecture Analysis

**Document Version:** 1.0
**Created:** 2025-10-31
**Purpose:** Comprehensive analysis of signal processing pipeline architecture
**Status:** Architecture review for Phase 6 refactoring

---

## Executive Summary

The `RoomResponseRecorder` pipeline has evolved through multiple refactoring phases and currently contains **two parallel implementations** that diverge unnecessarily. This document analyzes the current state, identifies architectural issues, and proposes a unified pipeline design.

### Key Findings

1. **✅ Core Principle Correct**: All signal processing is properly centralized in `RoomResponseRecorder`
2. **⚠️ Unnecessary Divergence**: Standard and calibration modes duplicate recording and file I/O logic
3. **⚠️ Deprecated Code**: Old `CalibrationValidator` (V1) still exists alongside V2
4. **⚠️ Missing Abstractions**: No clear separation between universal and mode-specific processing steps

---

## Current Pipeline Architecture (AS-IS)

### High-Level Structure

```
RoomResponseRecorder
│
├─ PUBLIC API
│  ├─ take_record(output, impulse, method=2, mode='standard'|'calibration')
│  └─ take_record_calibration() → wrapper for mode='calibration'
│
├─ TWO DIVERGENT PATHS
│  │
│  ├─ PATH A: STANDARD MODE (mode='standard')
│  │  └─ take_record() → lines 1126-1160
│  │      ├─ Step 1: _record_method_2()
│  │      ├─ Step 2: _process_recorded_signal()
│  │      │           ├─ _process_single_channel_signal()
│  │      │           └─ _process_multichannel_signal()
│  │      ├─ Step 3: _save_multichannel_files() / _save_single_channel_files()
│  │      └─ Return: raw audio (backward compatible)
│  │
│  └─ PATH B: CALIBRATION MODE (mode='calibration')
│     └─ _take_record_calibration_mode() → lines 1162-1288
│         ├─ Step 1: _record_method_2() [DUPLICATED]
│         ├─ Step 2: Custom processing (validation + alignment)
│         │           ├─ Simple reshape (no _extract_cycles helper)
│         │           ├─ CalibrationValidatorV2.validate_cycle()
│         │           ├─ align_cycles_by_onset()
│         │           └─ apply_alignment_to_channel()
│         ├─ Step 3: NO FILE SAVING
│         └─ Return: cycle-level data dict
```

### Problem: Unnecessary Code Duplication

Both paths perform identical operations but with different implementations:

| Operation | Standard Mode | Calibration Mode | Issue |
|-----------|--------------|------------------|-------|
| **Recording** | `_record_method_2()` | `_record_method_2()` | ✅ Shared (correct) |
| **Cycle extraction** | `_extract_cycles()` helper | Inline pad/trim/reshape | ❌ Duplicated logic |
| **Validation** | None (just averaging) | `CalibrationValidatorV2` | Different approaches |
| **Alignment** | `_find_onset_in_room_response()` + `np.roll()` | `align_cycles_by_onset()` + `apply_alignment_to_channel()` | ❌ Two alignment systems |
| **File saving** | Always saves files | Never saves files | ⚠️ No flexibility |

---

## Architectural Issues

### Issue 1: Two Cycle Extraction Implementations

**Standard Mode** (lines 690-713):
```python
def _extract_cycles(self, audio: np.ndarray) -> np.ndarray:
    expected_samples = self.cycle_samples * self.num_pulses
    if len(audio) < expected_samples:
        padded = np.zeros(expected_samples, dtype=audio.dtype)
        padded[:len(audio)] = audio
        audio = padded
    else:
        audio = audio[:expected_samples]
    return audio.reshape(self.num_pulses, self.cycle_samples)
```

**Calibration Mode** (lines 1208-1221):
```python
# DUPLICATED LOGIC - should use _extract_cycles()
cal_raw = recorded_audio[cal_ch]
expected_samples = self.cycle_samples * self.num_pulses
if len(cal_raw) < expected_samples:
    padded = np.zeros(expected_samples, dtype=cal_raw.dtype)
    padded[:len(cal_raw)] = cal_raw
    cal_raw = padded
else:
    cal_raw = cal_raw[:expected_samples]
initial_cycles = cal_raw.reshape(self.num_pulses, self.cycle_samples)
```

**Impact**: Same logic implemented twice, potential for divergence and bugs.

---

### Issue 2: Two Alignment Implementations

**Standard Mode Alignment** (lines 803-841):
- Uses `_find_onset_in_room_response()` on averaged cycles
- Simple onset detection: finds peak, searches window for onset
- Applies `np.roll()` directly to averaged signal
- **Limitation**: No per-cycle alignment, no validation, no quality filtering

**Calibration Mode Alignment** (lines 870-1055):
- Uses `align_cycles_by_onset()` on individual cycles
- Sophisticated: onset detection per cycle, cross-correlation filtering
- Returns alignment metadata with valid cycle indices
- Uses `apply_alignment_to_channel()` to propagate shifts
- **Advantage**: Per-cycle quality control, detailed diagnostics

**Why Two Systems Exist**:
1. Standard mode evolved for simple averaging (no quality requirements)
2. Calibration mode needed per-cycle validation (quality-critical measurements)
3. Never unified after calibration system was added

**Impact**: Standard mode cannot benefit from quality filtering that calibration mode provides.

---

### Issue 3: Hardcoded File Saving Behavior

**Current Implementation**:
- Standard mode: **ALWAYS** saves 3 files per channel
- Calibration mode: **NEVER** saves files

**Problem**: No flexibility for:
- Dry-run recordings (standard mode without saving)
- Saving calibration cycles for later analysis
- Conditional saving based on quality metrics

---

### Issue 4: Deprecated Code Still Present

**CalibrationValidator V1** (`calibration_validator.py`):
- ❌ **DEPRECATED**: Replaced by V2 on 2025-10-30
- Still exists in codebase
- Used only in old test files (`test_phase2_implementation.py`, `test_calibration_visualizer.py`)
- Risk of accidental usage

**Migration Evidence**:
```python
# RoomResponseRecorder.py lines 110-121
if 'calibration_quality' in file_config:
    print("Info: Migrating calibration_quality_config from V1 to V2 format")
    v1_config = file_config['calibration_quality']
    v2_config = self._migrate_v1_to_v2_calibration_config(v1_config)
    self.calibration_quality_config.update(v2_config)
```

System migrates V1 configs at runtime, suggesting V1 is still in use somewhere.

---

## Proposed Universal Pipeline Architecture (SHOULD-BE)

### Design Principles

1. **Three Universal Stages**: Recording → Processing → Output
2. **Processing Divergence Only**: Standard vs calibration differ ONLY in signal processing stage
3. **Configurable Output**: Decouple file saving from processing logic
4. **Shared Helpers**: All cycle extraction, alignment, validation use same core functions
5. **Mode as Strategy**: Processing mode is a parameter, not a code branch

---

### Unified Pipeline Design

```
┌─────────────────────────────────────────────────────────────────────┐
│                    RoomResponseRecorder                              │
│                     Universal Pipeline                               │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│ STAGE 1: SOUND RECORDING (Universal - No Divergence)                │
└─────────────────────────────────────────────────────────────────────┘
    _record_audio()
    ├─ Input: None (uses instance config)
    ├─ Processing: _record_method_2()
    │   ├─ Single-channel: sdl_audio_core.measure_room_response_auto()
    │   └─ Multi-channel: sdl_audio_core.measure_room_response_auto_multichannel()
    └─ Output: RawRecording
        ├─ Single: np.ndarray
        └─ Multi: Dict[int, np.ndarray]

┌─────────────────────────────────────────────────────────────────────┐
│ STAGE 2: SIGNAL PROCESSING (Diverges by Mode)                       │
└─────────────────────────────────────────────────────────────────────┘
    _process_audio(raw_recording, mode: ProcessingMode)
    │
    ├─ ProcessingMode.STANDARD
    │   └─ _process_standard(raw_recording)
    │       ├─ _extract_cycles_all_channels(raw)
    │       ├─ _average_cycles_all_channels(cycles)
    │       ├─ _align_by_reference_channel(averaged)
    │       └─ Returns: ProcessedData
    │           ├─ raw: Dict[int, ndarray] or ndarray
    │           ├─ room_response: Dict[int, ndarray] or ndarray
    │           ├─ impulse: Dict[int, ndarray] or ndarray
    │           └─ metadata: Dict
    │
    └─ ProcessingMode.CALIBRATION
        └─ _process_calibration(raw_recording)
            ├─ _extract_cycles_all_channels(raw)
            ├─ _validate_calibration_cycles(cal_cycles)
            ├─ _align_cycles_by_onset_with_validation(cycles, validation)
            ├─ _apply_alignment_to_all_channels(raw, alignment)
            └─ Returns: CalibrationData
                ├─ calibration_cycles: ndarray [N, samples]
                ├─ validation_results: List[Dict]
                ├─ aligned_multichannel_cycles: Dict[int, ndarray]
                ├─ alignment_metadata: Dict
                └─ quality_metrics: Dict

┌─────────────────────────────────────────────────────────────────────┐
│ STAGE 3: OUTPUT (Universal - Configurable)                          │
└─────────────────────────────────────────────────────────────────────┘
    _handle_output(processed_data, output_config: OutputConfig)
    │
    ├─ OutputConfig.save_files: bool
    ├─ OutputConfig.output_path: str
    ├─ OutputConfig.impulse_path: str
    └─ OutputConfig.return_format: 'raw' | 'processed' | 'full'

    If save_files:
        ├─ Single-channel: _save_single_channel_files()
        └─ Multi-channel: _save_multichannel_files()

    Returns based on return_format:
        ├─ 'raw': raw audio (backward compatible)
        ├─ 'processed': processed data dict
        └─ 'full': complete data with all stages
```

---

### Shared Helper Methods (Refactored)

All processing modes use these universal helpers:

```python
# ────────────────────────────────────────────────────────────────────
# UNIVERSAL CYCLE EXTRACTION
# ────────────────────────────────────────────────────────────────────

def _extract_cycles(self, audio: np.ndarray) -> np.ndarray:
    """
    Universal cycle extraction: pad/trim + reshape
    Used by BOTH standard and calibration modes
    """
    expected_samples = self.cycle_samples * self.num_pulses
    if len(audio) < expected_samples:
        padded = np.zeros(expected_samples, dtype=audio.dtype)
        padded[:len(audio)] = audio
        audio = padded
    else:
        audio = audio[:expected_samples]
    return audio.reshape(self.num_pulses, self.cycle_samples)

def _extract_cycles_all_channels(self, raw_recording) -> Dict[int, np.ndarray]:
    """
    Extract cycles from all channels (single or multi)
    Returns: Dict[channel_idx, cycles_array [N, samples]]
    """
    if isinstance(raw_recording, dict):
        return {ch: self._extract_cycles(audio) for ch, audio in raw_recording.items()}
    else:
        return {0: self._extract_cycles(raw_recording)}

# ────────────────────────────────────────────────────────────────────
# UNIVERSAL CYCLE AVERAGING
# ────────────────────────────────────────────────────────────────────

def _average_cycles(self, cycles: np.ndarray,
                   valid_indices: List[int] = None) -> np.ndarray:
    """
    Average cycles with optional filtering

    Args:
        cycles: [N, samples] array
        valid_indices: Optional list of cycle indices to include
                      If None, uses default: skip first 25%
    """
    if valid_indices is None:
        start = max(1, self.num_pulses // 4)
        valid_indices = list(range(start, self.num_pulses))

    return np.mean(cycles[valid_indices], axis=0)

# ────────────────────────────────────────────────────────────────────
# UNIVERSAL ALIGNMENT SYSTEM (Unified)
# ────────────────────────────────────────────────────────────────────

def _align_cycles_by_onset(self,
                           cycles: np.ndarray,
                           validation_results: List[Dict] = None,
                           correlation_threshold: float = 0.7) -> AlignmentResult:
    """
    Universal cycle alignment with optional validation filtering

    Works for:
    - Standard mode: validation_results=None, uses all cycles
    - Calibration mode: validation_results provided, filters invalid cycles

    Returns: AlignmentResult with shifts, valid_indices, aligned_cycles
    """
    # Filter cycles based on validation (if provided)
    if validation_results:
        valid_indices = [v['cycle_index'] for v in validation_results if v['is_valid']]
    else:
        valid_indices = list(range(len(cycles)))

    # Detect onset in each valid cycle
    onsets = []
    for idx in valid_indices:
        onset = self._find_sound_onset(cycles[idx])
        onsets.append(onset)

    # Align to common position
    target_position = 100  # samples
    shifts = [target_position - onset for onset in onsets]

    # Apply shifts
    aligned = np.array([np.roll(cycles[idx], shifts[i])
                       for i, idx in enumerate(valid_indices)])

    # Cross-correlation filtering (if threshold provided)
    if correlation_threshold is not None:
        aligned, valid_indices, shifts = self._filter_by_correlation(
            aligned, valid_indices, shifts, correlation_threshold
        )

    return AlignmentResult(
        valid_cycle_indices=valid_indices,
        shifts=shifts,
        aligned_cycles=aligned,
        num_aligned=len(valid_indices)
    )

# ────────────────────────────────────────────────────────────────────
# MODE-SPECIFIC PROCESSING
# ────────────────────────────────────────────────────────────────────

def _process_standard(self, raw_recording) -> ProcessedData:
    """Standard mode: Simple averaging + alignment"""
    # Extract cycles (all channels)
    all_cycles = self._extract_cycles_all_channels(raw_recording)

    # Determine reference channel
    ref_ch = self.multichannel_config.get('reference_channel', 0)
    ref_cycles = all_cycles[ref_ch]

    # Align reference channel cycles (no validation)
    ref_alignment = self._align_cycles_by_onset(
        ref_cycles,
        validation_results=None,  # No validation in standard mode
        correlation_threshold=None  # No filtering
    )

    # Average aligned reference cycles
    ref_room_response = np.mean(ref_alignment.aligned_cycles, axis=0)

    # Find onset in averaged signal
    onset_sample = self._find_sound_onset(ref_room_response)
    shift_amount = -onset_sample

    # Apply SAME shift to all channels
    result = {}
    for ch_idx, cycles in all_cycles.items():
        averaged = self._average_cycles(cycles)  # Default: skip first 25%
        impulse = np.roll(averaged, shift_amount)
        result[ch_idx] = {
            'room_response': averaged,
            'impulse': impulse
        }

    return ProcessedData(
        raw=raw_recording,
        channels=result,
        metadata={'reference_channel': ref_ch, 'onset_sample': onset_sample}
    )

def _process_calibration(self, raw_recording) -> CalibrationData:
    """Calibration mode: Validation + per-cycle alignment"""
    from calibration_validator_v2 import CalibrationValidatorV2, QualityThresholds

    # Extract cycles (all channels)
    all_cycles = self._extract_cycles_all_channels(raw_recording)

    # Get calibration channel
    cal_ch = self.multichannel_config['calibration_channel']
    cal_cycles = all_cycles[cal_ch]

    # Validate each cycle
    thresholds = QualityThresholds.from_config(self.calibration_quality_config)
    validator = CalibrationValidatorV2(thresholds, self.sample_rate)

    validation_results = []
    for i, cycle in enumerate(cal_cycles):
        validation = validator.validate_cycle(cycle, i)
        validation_results.append({
            'cycle_index': i,
            'is_valid': validation.calibration_valid,
            'calibration_metrics': validation.calibration_metrics,
            'calibration_failures': validation.calibration_failures
        })

    # Align with validation filtering
    alignment = self._align_cycles_by_onset(
        cal_cycles,
        validation_results=validation_results,
        correlation_threshold=0.7
    )

    # Apply alignment to all channels
    aligned_multichannel = {}
    for ch_idx, cycles in all_cycles.items():
        # Apply same shifts to this channel's cycles
        aligned = np.array([
            np.roll(cycles[idx], alignment.shifts[i])
            for i, idx in enumerate(alignment.valid_cycle_indices)
        ])
        aligned_multichannel[ch_idx] = aligned

    return CalibrationData(
        calibration_cycles=cal_cycles,
        validation_results=validation_results,
        aligned_multichannel_cycles=aligned_multichannel,
        alignment_metadata=alignment,
        num_valid_cycles=sum(1 for v in validation_results if v['is_valid']),
        num_aligned_cycles=alignment.num_aligned
    )
```

---

### Refactored Public API

```python
def take_record(self,
                output_file: str = "",
                impulse_file: str = "",
                mode: str = 'standard',
                save_files: bool = True,
                return_format: str = 'raw') -> Union[np.ndarray, Dict, CalibrationData]:
    """
    Universal recording API with configurable processing and output

    Args:
        output_file: Path for raw audio file (if save_files=True)
        impulse_file: Path for impulse response file (if save_files=True)
        mode: 'standard' or 'calibration'
        save_files: Whether to save files (default True for backward compatibility)
        return_format: 'raw' (default), 'processed', or 'full'

    Returns:
        Depends on mode and return_format:
        - standard + raw: np.ndarray or Dict[int, ndarray] (BACKWARD COMPATIBLE)
        - standard + processed: ProcessedData
        - calibration: CalibrationData (save_files ignored)
    """
    # STAGE 1: SOUND RECORDING (Universal)
    raw_recording = self._record_audio()
    if raw_recording is None:
        return None

    # STAGE 2: SIGNAL PROCESSING (Mode-specific)
    if mode == 'standard':
        processed = self._process_standard(raw_recording)
    elif mode == 'calibration':
        processed = self._process_calibration(raw_recording)
    else:
        raise ValueError(f"Invalid mode: {mode}")

    # STAGE 3: OUTPUT (Configurable)
    if mode == 'standard' and save_files:
        is_multichannel = isinstance(raw_recording, dict)
        if is_multichannel:
            self._save_multichannel_files(output_file, impulse_file, processed)
        else:
            self._save_single_channel_files(output_file, impulse_file, processed)

    # Return based on format
    if mode == 'calibration':
        return processed  # Always CalibrationData
    elif return_format == 'raw':
        return raw_recording  # BACKWARD COMPATIBLE
    elif return_format == 'processed':
        return processed
    else:  # 'full'
        return {'raw': raw_recording, 'processed': processed}
```

---

## Benefits of Unified Architecture

### 1. Code Reuse
- ✅ Single `_extract_cycles()` implementation
- ✅ Single `_align_cycles_by_onset()` system
- ✅ Shared validation infrastructure

### 2. Flexibility
- ✅ Standard mode can optionally use validation
- ✅ Calibration mode can optionally save files
- ✅ Dry-run recordings for testing

### 3. Maintainability
- ✅ Changes to cycle extraction affect all modes consistently
- ✅ Clear separation: universal vs mode-specific code
- ✅ Easier to add new processing modes

### 4. Testing
- ✅ Test universal helpers independently
- ✅ Test mode-specific logic separately
- ✅ Reduced test duplication

---

## Migration Path

### Phase 1: Unify Cycle Extraction
1. Update `_take_record_calibration_mode()` to use `_extract_cycles()` helper
2. Remove inline pad/trim/reshape code (lines 1208-1221)
3. Test calibration mode still works

### Phase 2: Unify Alignment System
1. Refactor `align_cycles_by_onset()` to accept optional validation
2. Update standard mode to use unified alignment
3. Add correlation filtering option to standard mode
4. Test both modes with unified system

### Phase 3: Decouple File Saving
1. Add `save_files` parameter to `take_record()`
2. Make file saving conditional in standard mode
3. Add optional file saving to calibration mode
4. Update GUI calls to explicitly pass `save_files=True`

### Phase 4: Remove Deprecated Code
1. Delete `calibration_validator.py` (V1)
2. Update test files to use V2
3. Remove V1 migration code from `__init__`

### Phase 5: Refactor Public API
1. Implement `_record_audio()` wrapper
2. Implement `_process_standard()` and `_process_calibration()`
3. Update `take_record()` to use new architecture
4. Maintain backward compatibility

---

## Deprecated Code Inventory

### Files to Remove

| File | Status | Reason | Dependencies |
|------|--------|--------|--------------|
| `calibration_validator.py` | ❌ DEPRECATED | Replaced by V2 | `test_phase2_implementation.py`, `test_calibration_visualizer.py` |

### Code to Remove

| Location | Lines | Description | Replacement |
|----------|-------|-------------|-------------|
| `RoomResponseRecorder.py` | 110-121 | V1 config migration | Remove after tests updated |
| `RoomResponseRecorder.py` | 322-339 | `_migrate_v1_to_v2_calibration_config()` | Remove after tests updated |
| `_take_record_calibration_mode()` | 1208-1221 | Inline cycle extraction | Use `_extract_cycles()` |

### Tests to Update

| File | Current Status | Action Required |
|------|---------------|-----------------|
| `test_phase2_implementation.py` | Uses V1 validator | Update to V2 or mark deprecated |
| `test_calibration_visualizer.py` | Uses V1 validator | Update to V2 or mark deprecated |

---

## Implementation Priority

### High Priority (Blocking)
1. ✅ **Unify cycle extraction** - Prevents logic divergence
2. ✅ **Remove deprecated CalibrationValidator V1** - Reduces confusion

### Medium Priority (Quality of Life)
3. ⚠️ **Decouple file saving** - Enables dry-run testing
4. ⚠️ **Unify alignment system** - Enables validation in standard mode

### Low Priority (Nice to Have)
5. 📋 **Refactor public API** - Cleaner architecture
6. 📋 **Add processing mode enum** - Type safety

---

## Conclusion

The current pipeline architecture is **functionally correct** but suffers from **unnecessary code duplication** between standard and calibration modes. The proposed universal pipeline architecture:

1. **Preserves** all existing functionality
2. **Maintains** backward compatibility
3. **Eliminates** code duplication
4. **Enables** new capabilities (validation in standard mode, file saving in calibration mode)
5. **Simplifies** future maintenance

**Recommended Action**: Implement migration phases 1-4 before Phase 5 (hardware testing). This ensures a clean, maintainable codebase for production deployment.
