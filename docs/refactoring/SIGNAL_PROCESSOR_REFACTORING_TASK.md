# SignalProcessor Refactoring Task - Fresh Start

**Date Created:** 2025-11-03
**Status:** NOT STARTED
**Priority:** HIGH
**Complexity:** MEDIUM

---

## Problem Statement

The RoomResponseRecorder class (~1,660 lines) currently handles THREE distinct responsibilities:
1. **Recording orchestration** - Managing audio I/O, device configuration
2. **Signal processing** - All DSP algorithms for analysis and alignment
3. **File I/O** - Saving WAV files and JSON metadata

This violates the Single Responsibility Principle and makes the code:
- Hard to test (requires hardware for all tests)
- Hard to maintain (changes to one area can break others)
- Hard to reuse (signal processing algorithms locked inside recorder)
- Hard to understand (too many responsibilities in one class)

**Previous Attempt Failed:** An attempt to extract signal processing to a SignalProcessor class was made but the alignment algorithm was incorrectly reimplemented, breaking calibration mode functionality.

---

## Objective

Extract ONLY the signal processing methods from RoomResponseRecorder into a standalone SignalProcessor class while:
1. **Maintaining 100% backward compatibility** - All existing code must work unchanged
2. **Preserving exact algorithms** - Copy implementations byte-for-byte, DO NOT rewrite
3. **No functional changes** - This is pure code movement, not improvement
4. **Comprehensive testing** - Verify delegation produces identical results

---

## Signal Processing Methods to Extract

These are the methods that contain signal processing logic and should be extracted:

### Core Processing Methods (Used in Standard Mode)
1. **`_extract_cycles()`** - Line 705
   - Splits audio into individual cycles
   - Pure array manipulation, no recorder dependencies

2. **`_average_cycles()`** - Line 719
   - Averages multiple cycles together
   - Pure computation, no recorder dependencies

3. **`_compute_spectral_analysis()`** - Line 737
   - Performs FFT and spectral analysis
   - Depends on: `self.sample_rate` (can be passed as parameter)

4. **`_find_onset_in_room_response()`** - Line 869
   - Finds sound onset in room response
   - Pure computation

5. **`_extract_impulse_response()`** - Line 890
   - Extracts impulse response from room response
   - Calls `_find_onset_in_room_response()`

6. **`_find_sound_onset()`** - Line 898
   - Finds sound onset using energy detection
   - Pure computation

### Advanced Calibration Mode Methods (CRITICAL - DO NOT MODIFY ALGORITHMS)
7. **`align_cycles_by_onset()`** - Line 942
   - **CRITICAL:** This method's algorithm works correctly and must be preserved exactly
   - Aligns calibration cycles by finding negative peak (impact moment)
   - Uses configurable target_onset_position from `self.multichannel_config`
   - Complex multi-step algorithm - DO NOT rewrite, only copy
   - **WARNING:** Previous refactoring broke this by changing the alignment logic

8. **`apply_alignment_to_channel()`** - Line 1077
   - Applies calibration alignment to other channels
   - Must preserve exact shift calculation logic

9. **`_normalize_by_calibration()`** - Line 1129
   - Normalizes response channels by calibration magnitude
   - Complex logic with validation result handling

---

## What NOT to Extract

These methods should STAY in RoomResponseRecorder:

- **Recording methods** - `take_record()`, `_record_audio()`, `_play_audio()`
- **File I/O** - `_save_wav()`, `_save_json()`, `_save_multichannel_files()`
- **Configuration** - `load_config()`, `save_config()`, `set_audio_devices()`
- **GUI callbacks** - `set_audio_callback()`, `register_pulse_callback()`
- **Processing orchestration** - `_process_recording_standard_mode()`, `_process_recording_calibration_mode()`

---

## Implementation Strategy

### Phase 1: Create SignalProcessor Class (Read-Only Analysis)

**DO NOT CODE YET - Just read and understand:**

1. Read the current implementations of all 9 signal processing methods
2. Identify all dependencies on `self.*` attributes
3. Create a dependency map showing what each method needs
4. Design SignalProcessingConfig class to hold all needed parameters

**Deliverable:** Dependency analysis document

### Phase 2: Create SignalProcessor with Copied Implementations

**Critical Rules:**
- **COPY implementations exactly** - Do not rewrite algorithms
- **Preserve line-by-line logic** - Especially for `align_cycles_by_onset()`
- Use SignalProcessingConfig for parameters instead of `self.attribute`
- Do not change any computation logic
- Do not "improve" or "optimize" anything

**Steps:**
1. Create `signal_processor.py` with SignalProcessingConfig class
2. Create SignalProcessor class with all 9 methods
3. Copy implementations exactly, only changing:
   - `self.sample_rate` → `self.config.sample_rate`
   - `self.multichannel_config.get(...)` → `self.config.multichannel_config.get(...)`
   - `self.cycle_samples` → `self.config.cycle_samples`
   - `self.num_pulses` → `self.config.num_pulses`
4. Keep all docstrings, comments, and code structure identical

**Deliverable:** `signal_processor.py` with exact algorithm copies

### Phase 3: Add Delegation Wrappers to RoomResponseRecorder

**Critical Rules:**
- **Do not delete original implementations yet**
- Create new wrapper methods with `_delegated_` prefix
- Wrappers should be thin - just call SignalProcessor and return result
- Handle any format conversions needed for backward compatibility

**Example:**
```python
def _delegated_align_cycles_by_onset(self, initial_cycles, validation_results, correlation_threshold=0.7):
    """Delegates to SignalProcessor (NEW IMPLEMENTATION - TESTING)"""
    return self.signal_processor.align_cycles_by_onset(
        initial_cycles,
        validation_results,
        correlation_threshold
    )
```

**Deliverable:** Parallel implementations (original + delegated)

### Phase 4: Comprehensive Testing

**Test Strategy:**
1. Create unit tests for SignalProcessor methods with synthetic data
2. Create integration tests comparing original vs delegated implementations
3. Test with REAL calibration hardware data (10 nearly identical impulses)
4. Verify correlations are >0.9 for identical impulses
5. Verify all aligned cycles are kept (not filtered incorrectly)

**Test Files:**
- `test_signal_processor.py` - Unit tests for SignalProcessor
- `test_delegation.py` - Compare original vs delegated on same inputs
- `test_calibration_hardware.py` - Real hardware validation

**Pass Criteria:**
- All unit tests pass
- Delegation produces identical results to original (use `np.allclose`)
- Real hardware test shows >0.9 correlation for identical impulses
- No regressions in existing functionality

**Deliverable:** All tests passing with real hardware validation

### Phase 5: Switch to Delegated Implementations

**Only after Phase 4 passes completely:**
1. Replace original method bodies with delegation calls
2. Keep original implementations commented out for 1 release cycle
3. Update documentation
4. Run full test suite again

**Deliverable:** Working refactored code with commented backups

### Phase 6: Cleanup (Future)

After 1-2 release cycles with no issues:
1. Remove commented original implementations
2. Remove `_delegated_` prefix from method names
3. Final documentation update

---

## Critical Lessons from Previous Failed Attempt

### What Went Wrong

1. **Algorithm was rewritten instead of copied**
   - Original: Align ALL cycles to target_onset_position, then select reference from aligned cycles
   - Broken version: Select reference from unaligned cycles, align to reference
   - Result: Reference cycle in wrong coordinate system, correlations = 0.0001

2. **Onset detection was changed**
   - Original: Correctly finds minimum value (negative peak)
   - Broken version: Initially used energy-based detection (wrong for calibration mode)

3. **Correlation formula was changed**
   - Original: `cross_product / sqrt(ref_energy * cyc_energy)`
   - Broken version: Used mean-subtraction normalization formula
   - Result: Different numerical results

4. **Configuration access was changed**
   - Original: `self.multichannel_config.get('alignment_target_onset_position', 0)`
   - Broken version: Hardcoded `target_onset_position = 0`
   - Result: Lost configurability

### How to Avoid These Mistakes

1. **COPY, DON'T REWRITE** - Use copy/paste, not "implement from memory"
2. **Test early and often** - Compare delegated vs original on EVERY method
3. **Use real hardware** - Synthetic tests can pass while real usage fails
4. **Preserve configuration** - Don't hardcode values that were configurable
5. **Keep algorithm structure** - Don't reorder steps or "optimize"

---

## Success Criteria

The refactoring is successful when ALL of these are true:

1. ✅ All 9 signal processing methods extracted to SignalProcessor
2. ✅ RoomResponseRecorder delegates to SignalProcessor for all signal processing
3. ✅ Unit tests pass for all SignalProcessor methods
4. ✅ Integration tests show identical results (original vs delegated)
5. ✅ Real calibration hardware test shows >0.9 correlation for identical impulses
6. ✅ All existing tests still pass (no regressions)
7. ✅ Code reduced by ~350+ lines in RoomResponseRecorder
8. ✅ SignalProcessor can be used standalone (no recorder dependencies)
9. ✅ Documentation updated to reflect new architecture
10. ✅ 100% backward compatibility maintained

---

## Dependencies

- **NumPy** - For array operations
- **SciPy** - For FFT and spectral analysis
- **RoomResponseRecorder** - For integration
- **CalibrationValidatorV2** - For validation result format

---

## Timeline Estimate

- Phase 1 (Analysis): 2-3 hours
- Phase 2 (Implementation): 4-6 hours
- Phase 3 (Delegation): 2-3 hours
- Phase 4 (Testing): 4-6 hours
- Phase 5 (Switch): 1-2 hours
- **Total:** 13-20 hours

---

## Risk Assessment

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Break calibration mode alignment | MEDIUM | CRITICAL | Extensive testing with real hardware before switching |
| Introduce numerical differences | MEDIUM | HIGH | Use np.allclose for all test comparisons |
| Miss configuration dependencies | LOW | MEDIUM | Careful dependency analysis in Phase 1 |
| Break backward compatibility | LOW | HIGH | Keep original methods until fully validated |

---

## Next Steps

**Before starting implementation:**

1. Read this entire document
2. Read the current implementations of all 9 methods in RoomResponseRecorder
3. Run existing tests to establish baseline
4. Create a git branch: `git checkout -b refactor/signal-processor-v2`
5. Document all dependencies found in Phase 1

**DO NOT:**
- Skip Phase 1 analysis
- Rewrite algorithms "from scratch"
- Change any computation logic
- Remove original implementations until Phase 5
- Skip hardware testing

---

## Questions to Answer Before Starting

1. What dependencies does `align_cycles_by_onset()` have on recorder state?
2. What is the exact algorithm flow in `align_cycles_by_onset()`?
3. What configuration values are used by signal processing methods?
4. How can we test correlation calculation is identical?
5. What format do validation_results use (dict or object)?
6. Where does `alignment_target_onset_position` come from?

---

## Reference Files

- **Current Implementation:** [RoomResponseRecorder.py](RoomResponseRecorder.py) (HEAD commit: bb0580e)
- **Method Locations:**
  - `_extract_cycles()`: Line 705
  - `_average_cycles()`: Line 719
  - `_compute_spectral_analysis()`: Line 737
  - `_find_onset_in_room_response()`: Line 869
  - `_extract_impulse_response()`: Line 890
  - `_find_sound_onset()`: Line 898
  - `align_cycles_by_onset()`: Line 942
  - `apply_alignment_to_channel()`: Line 1077
  - `_normalize_by_calibration()`: Line 1129

---

## Contact

For questions or clarifications about this refactoring task, refer to:
- [ARCHITECTURE_REFACTORING_PLAN.md](ARCHITECTURE_REFACTORING_PLAN.md) - Overall architecture vision
- [TECHNICAL_DOCUMENTATION.md](TECHNICAL_DOCUMENTATION.md) - System design
- Git commit `bb0580e` - Known working implementation

---

**REMEMBER:** The goal is to MOVE code, not IMPROVE it. Copy implementations exactly. Test extensively. Maintain backward compatibility. Do not rewrite algorithms.
