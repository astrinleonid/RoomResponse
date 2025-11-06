# SignalProcessor Refactoring Rollback Summary

**Date:** 2025-11-03
**Status:** ROLLED BACK
**Reason:** Alignment algorithm incorrectly reimplemented

---

## What Happened

An attempt was made to extract signal processing functionality from RoomResponseRecorder into a standalone SignalProcessor class. The refactoring appeared successful in testing but failed when tested with real calibration hardware.

### What Was Attempted

1. Created `signal_processor.py` with 9 signal processing methods
2. Created delegation wrappers in RoomResponseRecorder
3. Created comprehensive test suite (15 tests, all passing)
4. Tested with synthetic data (tests passed)

### Why It Failed

When tested with real hardware (calibration mode with 10 nearly identical piano hammer impacts), the alignment algorithm produced incorrect results:

**Expected Behavior:**
- 10 nearly identical impacts should have correlation >0.9
- All or most cycles should pass correlation filter
- Aligned cycles should be properly synchronized

**Actual Behavior:**
- Correlations were ~0.0001 for nearly identical signals
- Only 1 out of 10 cycles passed correlation threshold
- Alignment was completely broken

### Root Cause Analysis

The **`align_cycles_by_onset()`** method was **reimplemented** instead of **copied**, introducing critical algorithm changes:

#### Original Working Algorithm (RoomResponseRecorder)
```python
# Step 1: Filter to valid cycles
valid_cycles = initial_cycles[valid_indices]

# Step 2: Find onset in each cycle (minimum value = negative peak)
onset_positions = [np.argmin(cycle) for cycle in valid_cycles]

# Step 3: Get target position from configuration
target_onset_position = self.multichannel_config.get('alignment_target_onset_position', 0)
aligned_onset_position = target_onset_position

# Step 4: Align ALL cycles to target position
for i, cycle in enumerate(valid_cycles):
    shift_needed = aligned_onset_position - onset_positions[i]
    aligned_cycle = np.roll(cycle, shift_needed)
    aligned_cycles_list.append(aligned_cycle)

# Step 5: Select reference from ALIGNED cycles (highest energy)
energies = np.sqrt(np.mean(aligned_cycles ** 2, axis=1))
reference_idx = int(np.argmax(energies))
reference_cycle = aligned_cycles[reference_idx]

# Step 6: Calculate correlation between aligned cycles
```

#### Broken Implementation (SignalProcessor - INCORRECT)
```python
# Step 1: Filter to valid cycles
valid_cycles = initial_cycles[valid_indices]

# Step 2: Find onset in each cycle
onsets = [np.argmin(cycle) for cycle in valid_cycles]

# Step 3: Select reference from UNALIGNED cycles  ← WRONG
reference_idx = 0  # or passed parameter
reference_onset = onsets[reference_idx]
reference_cycle = valid_cycles[reference_idx]  ← UNALIGNED!

# Step 4: Align cycles to reference's onset  ← WRONG APPROACH
shifts = reference_onset - onsets
aligned_cycles = [np.roll(cycle, shift) for cycle, shift in zip(valid_cycles, shifts)]

# Step 5: Calculate correlation with UNALIGNED reference  ← WRONG!
# Comparing aligned cycles against unaligned reference_cycle
```

**The Problem:** The broken version:
1. Selected reference from **unaligned** cycles
2. Aligned all cycles to the reference's onset position
3. Compared aligned cycles against the **unaligned reference cycle**
4. Lost configurability of `alignment_target_onset_position`

This caused complete coordinate system mismatch - aligned cycles were in one coordinate system, reference was in another, resulting in correlation values of 0.0001.

### Secondary Issues Found

1. **Onset Detection:** Initially tried energy-based detection instead of minimum value
2. **Correlation Formula:** Changed from energy-based to mean-subtraction formula
3. **Configuration Loss:** Hardcoded `target_onset_position = 0` instead of reading from config

---

## Actions Taken

### Rollback Steps

1. ✅ Reverted `RoomResponseRecorder.py` to commit 52757f3 (before refactoring started)
2. ✅ Reverted `gui_series_settings_panel.py` to commit 52757f3 (removed `_init_signal_processor()` call)
3. ✅ Deleted `signal_processor.py` (broken implementation)
4. ✅ Deleted test files:
   - `test_signal_processor.py`
   - `test_recorder_signal_processor_integration.py`
   - `test_calibration_dict_format.py`
5. ✅ Created task description for correct refactoring approach

**Note:** Commits bb0580e and later were made during the failed refactoring session and include
signal_processor integration. Reverted to 52757f3 which is the last known fully working state.

### Preserved Documentation

The following documentation files have been kept for reference:
- `SIGNAL_PROCESSOR_REFACTORING_TASK.md` - NEW: Detailed task description for correct approach
- `REFACTORING_IMPLEMENTATION_COMPLETE.md` - Documents what was attempted
- `ROOMRESPONSE_RECORDER_REFACTORING_PLAN.md` - Original plan
- `REFACTORING_VISUAL_SUMMARY.md` - Architecture diagrams

---

## Current State

### What's Working
- ✅ RoomResponseRecorder restored to working state (commit 52757f3)
- ✅ gui_series_settings_panel restored to working state (commit 52757f3)
- ✅ Calibration mode alignment works correctly with original algorithm
- ✅ All existing functionality intact
- ✅ No broken code in repository
- ✅ Application starts without errors

### What's Removed
- ❌ signal_processor.py (broken implementation)
- ❌ All refactoring test files
- ❌ Delegation wrappers from RoomResponseRecorder

### What Remains
- 📄 Documentation files (for reference and learning)
- 📄 New task description with correct approach
- 📄 This rollback summary

---

## Key Lessons Learned

### 1. Don't Rewrite Working Algorithms

**Wrong Approach:**
- "I understand the algorithm, I'll implement it fresh"
- "I can simplify this logic"
- "This looks inefficient, let me optimize"

**Correct Approach:**
- Copy implementation line-by-line
- Change only parameter access (`self.config.x` instead of `self.x`)
- Keep ALL logic identical, including configuration access

### 2. Test with Real Data Early

**What Happened:**
- 15 unit tests passed with synthetic data
- All integration tests passed
- Real hardware test failed immediately

**Lesson:**
- Synthetic tests can't catch algorithm changes
- Test with real calibration hardware BEFORE considering phase complete
- One real hardware test worth 100 synthetic tests

### 3. Preserve Configuration Access

**What Happened:**
```python
# Original (configurable)
target = self.multichannel_config.get('alignment_target_onset_position', 0)

# Broken (hardcoded)
target_onset_position = 0
```

**Lesson:**
- Don't assume default values are sufficient
- Configuration hooks may be used even if not currently set
- Preserve all configuration access patterns

### 4. Understand Before Extracting

**What Happened:**
- Didn't fully understand why ALL cycles (including reference) needed alignment
- Didn't understand coordinate system requirements for correlation
- Assumed simpler approach would work

**Lesson:**
- Spend time understanding the algorithm flow
- Draw diagrams of data transformations
- Don't extract code you don't fully understand

### 5. Keep Original Until Proven

**What Did Work:**
- Keeping original methods during testing phase
- Parallel implementation approach (`_delegated_*` methods)
- Ability to compare side-by-side

**What Should Have Been Done:**
- Test delegated vs original on SAME REAL DATA
- Use `np.allclose()` to verify identical results
- Don't switch until proven identical on hardware

---

## Next Steps (If Refactoring Is Attempted Again)

### Prerequisites

1. **Read the task description:** [SIGNAL_PROCESSOR_REFACTORING_TASK.md](SIGNAL_PROCESSOR_REFACTORING_TASK.md)
2. **Understand the failure:** This document
3. **Study the working algorithm:** Lines 942-1075 in RoomResponseRecorder.py (commit bb0580e)
4. **Have real hardware available:** For calibration mode testing

### Phase 1: Analysis (Do Not Code)

**Tasks:**
1. Read all 9 signal processing methods in RoomResponseRecorder
2. Document every `self.*` dependency
3. Trace the exact flow of `align_cycles_by_onset()` step by step
4. Understand WHY each step is done in that order
5. Create a dependency matrix

**Deliverable:** Written analysis document answering:
- Why are ALL cycles aligned to target position?
- Why is reference selected AFTER alignment?
- What is `alignment_target_onset_position` used for?
- Why is correlation computed on aligned cycles?

**Exit Criteria:** Can explain the algorithm to someone else without looking at code

### Phase 2: Copy (Not Implement)

**Tasks:**
1. Create `signal_processor.py`
2. **COPY** implementations using copy/paste
3. Only change parameter access patterns
4. Keep computation logic 100% identical
5. Preserve all configuration access

**Strict Rules:**
- ✅ Use copy/paste, not "implement from memory"
- ✅ Preserve every line of logic
- ✅ Keep the same variable names
- ✅ Preserve the same step order
- ❌ Don't "simplify" or "optimize"
- ❌ Don't change algorithm structure
- ❌ Don't hardcode configurable values

### Phase 3: Test with Real Hardware First

**Critical Change from Previous Attempt:**
Test with REAL HARDWARE **before** creating unit tests!

**Test Process:**
1. Create parallel `_delegated_*` methods in RoomResponseRecorder
2. Record 10 calibration impacts (nearly identical)
3. Process with original method, save results
4. Process with delegated method, save results
5. Compare using `np.allclose()`
6. **MUST PASS:** Correlation values identical, aligned cycles identical

**Only proceed if hardware test passes!**

### Phase 4: Comprehensive Testing

After hardware test passes:
1. Create unit tests
2. Create integration tests
3. Test edge cases
4. Test with different configurations

### Phase 5: Switch (Only After Everything Passes)

1. Replace original implementations with delegation
2. Keep originals commented for 1 release
3. Update documentation

---

## Files Reference

### Current Working Code
- [RoomResponseRecorder.py](RoomResponseRecorder.py) - Restored to commit bb0580e
  - `align_cycles_by_onset()`: Line 942 ← **WORKING ALGORITHM - DO NOT CHANGE**
  - `apply_alignment_to_channel()`: Line 1077
  - `_normalize_by_calibration()`: Line 1129

### Task Documentation
- [SIGNAL_PROCESSOR_REFACTORING_TASK.md](SIGNAL_PROCESSOR_REFACTORING_TASK.md) - Detailed task description

### Historical Documentation (For Reference)
- [REFACTORING_IMPLEMENTATION_COMPLETE.md](REFACTORING_IMPLEMENTATION_COMPLETE.md) - What was attempted
- [ROOMRESPONSE_RECORDER_REFACTORING_PLAN.md](ROOMRESPONSE_RECORDER_REFACTORING_PLAN.md) - Original plan

---

## Conclusion

The refactoring was rolled back because the alignment algorithm was incorrectly reimplemented, breaking calibration mode functionality. The original working code has been restored.

**If attempting refactoring again:**
1. Read [SIGNAL_PROCESSOR_REFACTORING_TASK.md](SIGNAL_PROCESSOR_REFACTORING_TASK.md) completely
2. Follow the phased approach strictly
3. **COPY implementations, don't rewrite them**
4. Test with real hardware BEFORE considering any phase complete
5. Use `np.allclose()` to verify identical results

**Key Principle:** This is code MOVEMENT, not code IMPROVEMENT. The goal is to reorganize existing working code, not to rewrite it.

---

**Status:** ✅ Rollback Complete - System Restored to Working State
**Date:** 2025-11-03
**Next Action:** Study task description before attempting refactoring again
