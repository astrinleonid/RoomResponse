# Deprecated Code Inventory

**Document Version:** 1.0
**Created:** 2025-10-31
**Purpose:** Track deprecated code for removal during Phase 6 refactoring

---

## Overview

This document identifies code that has been superseded by newer implementations but remains in the codebase. All items should be reviewed and removed during Phase 6 pipeline refactoring.

---

## Files to Remove

### 1. calibration_validator.py (V1)

**Status:** ❌ DEPRECATED (Replaced 2025-10-30)
**Location:** `d:\repos\RoomResponse\calibration_validator.py`
**Size:** ~7 KB
**Replacement:** `calibration_validator_v2.py`

**Reason for Deprecation:**
- Complex ratio-based validation logic (unreliable)
- Difficult to configure and understand
- Replaced by simpler min/max range checking in V2

**Current Dependencies:**
```
calibration_validator.py
├─ test_phase2_implementation.py (line unknown)
└─ test_calibration_visualizer.py (line unknown)
```

**Migration Required:**
1. Update `test_phase2_implementation.py` to import V2
2. Update `test_calibration_visualizer.py` to import V2
3. Verify all tests pass with V2
4. Delete `calibration_validator.py`

**Risk:** Low - V2 is fully functional and tested

---

## Code Sections to Remove

### 2. V1 Config Migration Logic

**Status:** ⚠️ TEMPORARY (Can be removed after test updates)
**Location:** `RoomResponseRecorder.py`

#### Section A: Config Loading Migration (lines 110-121)

```python
# Current code (TO BE REMOVED):
if 'calibration_quality' in file_config:
    print("Info: Migrating calibration_quality_config from V1 to V2 format")
    v1_config = file_config['calibration_quality']
    v2_config = self._migrate_v1_to_v2_calibration_config(v1_config)
    self.calibration_quality_config.update(v2_config)
elif 'calibration_quality_config' in file_config:
    self.calibration_quality_config.update(file_config['calibration_quality_config'])

# After tests updated (SHOULD BE):
if 'calibration_quality_config' in file_config:
    self.calibration_quality_config.update(file_config['calibration_quality_config'])
```

**Reason:** V1 configs will no longer exist after test updates

#### Section B: Migration Helper Method (lines 322-339)

```python
# TO BE REMOVED (entire method):
def _migrate_v1_to_v2_calibration_config(self, v1_config: Dict) -> Dict:
    """
    Migrate V1 calibration quality config to V2 format.

    V1: ratio-based thresholds (cal_min_amplitude, cal_max_amplitude, etc.)
    V2: min/max ranges (min_negative_peak, max_negative_peak, etc.)
    """
    # ... 17 lines of migration logic ...
```

**Reason:** No longer needed after V1 removal

**Dependencies:** None (only called from `__init__`)

---

### 3. Duplicated Cycle Extraction (Calibration Mode)

**Status:** ❌ DUPLICATE LOGIC
**Location:** `RoomResponseRecorder.py` lines 1208-1221
**Replacement:** Use existing `_extract_cycles()` helper (lines 690-713)

**Current Code:**
```python
# DUPLICATED LOGIC (in _take_record_calibration_mode):
cal_raw = recorded_audio[cal_ch]
expected_samples = self.cycle_samples * self.num_pulses

# Pad or trim
if len(cal_raw) < expected_samples:
    padded = np.zeros(expected_samples, dtype=cal_raw.dtype)
    padded[:len(cal_raw)] = cal_raw
    cal_raw = padded
else:
    cal_raw = cal_raw[:expected_samples]

# Simple reshape extraction
initial_cycles = cal_raw.reshape(self.num_pulses, self.cycle_samples)
```

**Should Be:**
```python
# Use existing helper (REFACTORED):
cal_raw = recorded_audio[cal_ch]
initial_cycles = self._extract_cycles(cal_raw)
```

**Risk:** None - `_extract_cycles()` does identical operation

---

## Code Patterns to Refactor

### 4. Hardcoded File Saving Behavior

**Status:** ⚠️ ARCHITECTURAL LIMITATION
**Location:** Multiple locations

#### Standard Mode (lines 1142-1145)
```python
# Current: ALWAYS saves files
if is_multichannel:
    self._save_multichannel_files(output_file, impulse_file, processed_data)
else:
    self._save_single_channel_files(output_file, impulse_file, processed_data)

# Proposed: Configurable
if save_files:  # New parameter
    if is_multichannel:
        self._save_multichannel_files(output_file, impulse_file, processed_data)
    else:
        self._save_single_channel_files(output_file, impulse_file, processed_data)
```

#### Calibration Mode (lines 1162-1288)
```python
# Current: NEVER saves files
# No file saving code at all

# Proposed: Optional saving
if save_files:
    # Save calibration cycles for later analysis
    self._save_calibration_files(output_file, processed_data)
```

**Benefit:** Enables dry-run testing and calibration cycle archiving

---

### 5. Two Alignment Systems

**Status:** ⚠️ DUPLICATION OF LOGIC
**Locations:**
- Standard: `_find_onset_in_room_response()` + `np.roll()` (lines 803-841)
- Calibration: `align_cycles_by_onset()` (lines 870-1003)

**Current State:**

**Standard Mode Alignment:**
```python
# Simple averaged-signal alignment
onset_sample = self._find_onset_in_room_response(ref_room_response)
shift_amount = -onset_sample
impulse_response = np.roll(room_response, shift_amount)
```

**Calibration Mode Alignment:**
```python
# Sophisticated per-cycle alignment with validation
alignment_result = self.align_cycles_by_onset(
    initial_cycles,
    validation_results,
    correlation_threshold=correlation_threshold
)
```

**Proposed Unified System:**
```python
def _align_cycles_by_onset(self,
                           cycles: np.ndarray,
                           validation_results: List[Dict] = None,
                           correlation_threshold: float = None) -> AlignmentResult:
    """
    Universal alignment - works for both modes

    Standard mode: validation_results=None, correlation_threshold=None
    Calibration mode: validation_results provided, correlation_threshold=0.7
    """
    # Single implementation supporting both use cases
```

**Benefit:** Standard mode can optionally use quality filtering

---

## Test Files Requiring Updates

### test_phase2_implementation.py

**Current Status:** Uses V1 validator
**Required Changes:**
1. Change import: `from calibration_validator import CalibrationValidator` → `from calibration_validator_v2 import CalibrationValidatorV2`
2. Update validator initialization (V1 → V2 config structure)
3. Update assertion expectations (ratio metrics → absolute metrics)
4. Run tests and verify pass

**Estimated Effort:** 30 minutes

---

### test_calibration_visualizer.py

**Current Status:** Uses V1 validator
**Required Changes:**
1. Same as above
2. Update visualization code if it displays V1 metrics
3. Run tests and verify pass

**Estimated Effort:** 30 minutes

---

## Configuration Files

### Old V1 Configuration Format

**Status:** ⚠️ DEPRECATED (but migrated at runtime)

**V1 Format (deprecated):**
```json
{
  "calibration_quality": {
    "cal_min_amplitude": 0.1,
    "cal_max_amplitude": 0.95,
    "cal_min_duration_ms": 2.0,
    "cal_max_duration_ms": 20.0,
    "cal_duration_threshold": 0.3,
    "cal_double_hit_window_ms": [10, 50],
    "cal_double_hit_threshold": 0.3,
    "cal_tail_start_ms": 30.0,
    "cal_tail_max_rms_ratio": 0.15,
    "min_valid_cycles": 3
  }
}
```

**V2 Format (current):**
```json
{
  "calibration_quality_config": {
    "min_negative_peak": 0.5144,
    "max_negative_peak": 0.7086,
    "min_positive_peak": 0.3813,
    "max_positive_peak": 0.5002,
    "min_aftershock": 0.1001,
    "max_aftershock": 0.1301,
    "aftershock_window_ms": 10.0,
    "aftershock_skip_ms": 2.0,
    "min_valid_cycles": 3
  }
}
```

**Action:** After V1 removal, delete migration code

---

## Unused/Dead Code

### Potential Recording Methods 1 & 3

**Status:** 🔍 INVESTIGATION REQUIRED
**Location:** References in `take_record(method=2)`

**Current Code:**
```python
def take_record(self, output_file, impulse_file, method: int = 2, ...):
    """
    Args:
        method: Recording method (1=manual, 2=auto, 3=specific devices)
    """
```

**Findings:**
- No `_record_method_1()` or `_record_method_3()` found in codebase
- Only `_record_method_2()` exists
- `method` parameter documented but not used

**Questions:**
1. Were methods 1 & 3 removed in previous refactoring?
2. Is `method` parameter deprecated?
3. Should parameter be removed or documented as "reserved for future use"?

**Recommendation:**
- If methods 1 & 3 never existed: Remove `method` parameter
- If they're planned for future: Document as "reserved"
- If they're truly deprecated: Add deprecation warning

---

## Interactive Mode Parameter

**Status:** 🔍 INVESTIGATION REQUIRED
**Location:** `take_record()` calls in GUI

**Evidence:**
```python
# gui_series_worker.py:170
_ = self._recorder.take_record("/dev/null", "/dev/null", method=2, interactive=False)

# gui_series_worker.py:302
result['audio'] = self._recorder.take_record(raw_path, impulse_path, method=2, interactive=False)
```

**Current Signature:**
```python
def take_record(self,
                output_file: str,
                impulse_file: str,
                method: int = 2,
                mode: str = 'standard',
                return_processed: bool = False):
```

**Issue:** `interactive` parameter used in calls but **not defined in signature**

**Possible Explanations:**
1. Parameter was removed but calls not updated (most likely)
2. Parameter handled by `**kwargs` somewhere (not seen)
3. Parameter ignored silently (error-prone)

**Recommendation:**
- Remove `interactive=False` from all GUI calls
- Verify no functionality breaks
- If interactive mode needed, re-add to signature with proper handling

---

## Summary of Deprecated Code

| Item | Type | Location | Effort | Risk | Priority |
|------|------|----------|--------|------|----------|
| `calibration_validator.py` | File | Root directory | 1 hour | Low | High |
| V1 config migration | Code section | `RoomResponseRecorder.py:110-121, 322-339` | 15 min | None | High |
| Duplicated cycle extraction | Code pattern | `RoomResponseRecorder.py:1208-1221` | 15 min | None | High |
| Hardcoded file saving | Architecture | Multiple locations | 4 hours | Low | Medium |
| Two alignment systems | Architecture | Multiple locations | 8 hours | Medium | Medium |
| `method` parameter | Parameter | `take_record()` signature | 30 min | Low | Low |
| `interactive` parameter | Ghost param | GUI calls | 15 min | Low | Medium |

**Total Cleanup Effort:** ~14 hours
**Total Refactoring Effort (Phases 6.1-6.4):** ~28 hours (3.5 days)

---

## Migration Checklist

### Phase 6.1: Code Cleanup (Day 1)
- [ ] Update `test_phase2_implementation.py` to use V2 validator
- [ ] Update `test_calibration_visualizer.py` to use V2 validator
- [ ] Run all tests with V2 validator - verify pass
- [ ] Remove `calibration_validator.py` file
- [ ] Remove V1 migration code from `RoomResponseRecorder.__init__` (lines 110-121)
- [ ] Remove `_migrate_v1_to_v2_calibration_config()` method (lines 322-339)
- [ ] Remove `interactive=False` from GUI calls
- [ ] Run regression tests - verify pass

### Phase 6.2: Unify Cycle Extraction (Day 1.5)
- [ ] Refactor `_take_record_calibration_mode()` to use `_extract_cycles()` helper
- [ ] Remove inline pad/trim/reshape code (lines 1208-1221)
- [ ] Test calibration mode - verify unchanged behavior
- [ ] Run calibration quality UI tests - verify pass

### Phase 6.3: Decouple File Saving (Day 2-2.5)
- [ ] Add `save_files: bool = True` parameter to `take_record()`
- [ ] Make file saving conditional in standard mode
- [ ] Add optional file saving to calibration mode
- [ ] Update all GUI calls to explicitly pass `save_files=True`
- [ ] Test dry-run mode (save_files=False)
- [ ] Run full integration tests

### Phase 6.4: Unify Alignment (Optional - Day 3-3.5)
- [ ] Refactor `align_cycles_by_onset()` to accept optional `validation_results`
- [ ] Update standard mode to use unified alignment
- [ ] Add optional correlation filtering to standard mode
- [ ] Test both modes with unified system
- [ ] Performance benchmark - verify no regression
- [ ] Run full test suite

---

## Post-Cleanup Verification

After all cleanup:

1. **Run Full Test Suite:**
   ```bash
   pytest test_phase2_implementation.py
   pytest test_calibration_visualizer.py
   pytest test_phase3_implementation.py
   # All other test files
   ```

2. **Manual GUI Testing:**
   - Audio Settings → Calibration Impulse tab
   - Series Settings → Record and analyze
   - Single Pulse Recorder → Record
   - Verify all workflows functional

3. **Code Quality:**
   ```bash
   # No references to V1 validator
   grep -r "from calibration_validator import" .
   # Should return no results

   # No V1 config keys
   grep -r "calibration_quality[^_]" recorderConfig.json
   # Should return no results
   ```

4. **Documentation:**
   - Update method docstrings (new parameters)
   - Update examples in comments
   - Update README if applicable

---

## Conclusion

This inventory identifies approximately **14 hours of cleanup work** and **28 hours total refactoring effort** to achieve a unified, maintainable pipeline architecture.

**Recommended Approach:** Implement Phases 6.1-6.3 (high/medium priority items) before Phase 5 hardware testing. Phase 6.4 (alignment unification) can be deferred to post-testing if time constrained.

**Key Benefits:**
- ✅ Removes confusing deprecated code
- ✅ Prevents future bugs from logic divergence
- ✅ Enables new capabilities (dry-run testing, optional validation)
- ✅ Cleaner, more maintainable codebase for production
