# Averaging Amplitude Fix - Calibration Mode Normalization

**Date:** 2025-11-02
**Issue:** Averaged impulse response amplitude significantly lower than individual cycles
**Status:** 🔴 CRITICAL BUG IDENTIFIED - Fix in progress
**Priority:** HIGH - Affects data accuracy in calibration mode

---

## Problem Description

### User Report
> "I see that the amplitude of the averaged impulse response is significantly lower than the average amplitude of the individual cycles. It can be for two reasons: or the cycles are not perfectly aligned - which is improbable? I checked the alignment multiple times or The sum is divided by wrong denominator, maybe the total number of cycles vs number of filtered cycles"

### Observed Behavior
When recording in calibration mode with normalization enabled:
- ❌ Individual normalized cycles show correct amplitudes (e.g., ~1.7 peak)
- ❌ Averaged impulse response shows much lower amplitude (e.g., ~0.8 peak)
- ❌ The averaged amplitude is approximately 50-60% of individual cycle amplitudes
- ✅ User confirmed alignment is correct (checked multiple times)

### Impact
**CRITICAL:** The averaging amplitude issue makes quantitative analysis unreliable. Users cannot trust that the averaged response accurately represents the calibration data.

---

## Root Cause Analysis

### The Bug: Index Mapping Mismatch

The normalization function incorrectly maps aligned cycle indices to original cycle indices when looking up normalization factors.

#### Data Flow Example

**Scenario: 8 recorded cycles, cycle #3 fails validation**

**Step 1: Validation (Lines 1315-1335)**
```python
# Records 8 cycles
validation_results = [
    {'cycle_index': 0, 'is_valid': True, 'calibration_metrics': {'negative_peak': -0.45}},
    {'cycle_index': 1, 'is_valid': True, 'calibration_metrics': {'negative_peak': -0.32}},
    {'cycle_index': 2, 'is_valid': True, 'calibration_metrics': {'negative_peak': -0.28}},
    {'cycle_index': 3, 'is_valid': False, 'calibration_metrics': {'negative_peak': -0.15}},  # ← REJECTED
    {'cycle_index': 4, 'is_valid': True, 'calibration_metrics': {'negative_peak': -0.50}},
    {'cycle_index': 5, 'is_valid': True, 'calibration_metrics': {'negative_peak': -0.38}},
    {'cycle_index': 6, 'is_valid': True, 'calibration_metrics': {'negative_peak': -0.41}},
    {'cycle_index': 7, 'is_valid': True, 'calibration_metrics': {'negative_peak': -0.35}}
]
```

**Step 2: Alignment Filtering (Lines 1339-1346)**
```python
alignment_result = self.align_cycles_by_onset(initial_cycles, validation_results, ...)

# Inside align_cycles_by_onset (line 911):
# Filters to valid cycles, then applies correlation filtering
# Result:
alignment_result = {
    'valid_cycle_indices': [0, 1, 2, 4, 5, 6, 7],  # ← Cycle 3 removed!
    'aligned_cycles': <7 cycles array>,
    ...
}
```

**Step 3: Apply Alignment (Lines 1348-1354)**
```python
# apply_alignment_to_channel uses valid_cycle_indices
aligned_multichannel_cycles = {
    ch0: [cycle0, cycle1, cycle2, cycle4, cycle5, cycle6, cycle7],  # 7 cycles
    ch1: [cycle0, cycle1, cycle2, cycle4, cycle5, cycle6, cycle7],  # 7 cycles
    ...
}

# Note: aligned_multichannel_cycles[ch][i] corresponds to ORIGINAL cycle valid_cycle_indices[i]
# Index mapping:
# aligned_idx=0 → original_idx=0
# aligned_idx=1 → original_idx=1
# aligned_idx=2 → original_idx=2
# aligned_idx=3 → original_idx=4  ← NOT 3!
# aligned_idx=4 → original_idx=5
# aligned_idx=5 → original_idx=6
# aligned_idx=6 → original_idx=7
```

**Step 4: Normalize (Lines 1361-1365) ← THE BUG**
```python
processed_cycles, normalization_factors = self._normalize_by_calibration(
    aligned_multichannel_cycles,  # ← 7 cycles, indexed 0-6
    validation_results,            # ← 8 validation entries, indexed 0-7
    cal_ch
)
```

**Inside _normalize_by_calibration (Lines 1092-1098):**
```python
# Extract normalization factors from ALL validation results
normalization_factors = []
for v_result in validation_results:  # ← ALL 8 cycles
    metrics = v_result.get('calibration_metrics') or {}
    neg_peak = abs(metrics.get('negative_peak', 0.0))
    normalization_factors.append(neg_peak)

# Result:
normalization_factors = [0.45, 0.32, 0.28, 0.15, 0.50, 0.38, 0.41, 0.35]
#                        idx0  idx1  idx2  idx3  idx4  idx5  idx6  idx7
#                        ✅    ✅    ✅    ❌    ✅    ✅    ✅    ✅
#                                         rejected
```

**The Normalization Loop (Lines 1130-1142):**
```python
for cycle_idx, cycle_data in enumerate(channel_cycles):  # ← Iterates 0-6 (7 cycles)
    if cycle_idx < len(normalization_factors):
        norm_factor = normalization_factors[cycle_idx]  # ← WRONG INDEX!
        normalized_cycle = cycle_data / norm_factor
        normalized_cycles.append(normalized_cycle)
```

**What Happens:**
```
Aligned cycle 0 (from original cycle 0) / normalization_factors[0] = 0.45  ✅ CORRECT
Aligned cycle 1 (from original cycle 1) / normalization_factors[1] = 0.32  ✅ CORRECT
Aligned cycle 2 (from original cycle 2) / normalization_factors[2] = 0.28  ✅ CORRECT
Aligned cycle 3 (from original cycle 4) / normalization_factors[3] = 0.15  ❌ WRONG!
                                                                      ^^^^
                                           Should use normalization_factors[4] = 0.50!

Aligned cycle 4 (from original cycle 5) / normalization_factors[4] = 0.50  ❌ WRONG!
                                           Should use normalization_factors[5] = 0.38!

Aligned cycle 5 (from original cycle 6) / normalization_factors[5] = 0.38  ❌ WRONG!
                                           Should use normalization_factors[6] = 0.41!

Aligned cycle 6 (from original cycle 7) / normalization_factors[6] = 0.41  ❌ WRONG!
                                           Should use normalization_factors[7] = 0.35!
```

**Result of Wrong Normalization:**
- Aligned cycle 3 is divided by 0.15 instead of 0.50 → 3.33x too large!
- Aligned cycle 4 is divided by 0.50 instead of 0.38 → 0.76x too small
- Aligned cycle 5 is divided by 0.38 instead of 0.41 → 0.93x too small
- Aligned cycle 6 is divided by 0.41 instead of 0.35 → 0.85x too small

When these incorrectly normalized cycles are averaged, the result is unpredictable and incorrect!

---

## The Code Locations

### Bug Location #1: Index Mapping Lost
**File:** `RoomResponseRecorder.py`
**Line:** 1361-1365

```python
# PROBLEM: valid_cycle_indices not passed to normalization function
processed_cycles, normalization_factors = self._normalize_by_calibration(
    aligned_multichannel_cycles,  # ← Cycles indexed by aligned position (0-6)
    validation_results,            # ← Validation indexed by original position (0-7)
    cal_ch
)  # ← Missing: alignment_result['valid_cycle_indices'] mapping!
```

### Bug Location #2: Wrong Index Used for Lookup
**File:** `RoomResponseRecorder.py`
**Line:** 1130-1136

```python
for cycle_idx, cycle_data in enumerate(channel_cycles):
    if cycle_idx < len(normalization_factors):
        norm_factor = normalization_factors[cycle_idx]  # ← Uses aligned index!
        #                                  ^^^^^^^^^^
        #                    Should use: normalization_factors[original_cycle_indices[cycle_idx]]

        if norm_factor > 1e-6:
            normalized_cycle = cycle_data / norm_factor
            normalized_cycles.append(normalized_cycle)
```

---

## The Solution

### Strategy
Pass the `valid_cycle_indices` mapping to `_normalize_by_calibration` so it can correctly look up normalization factors using original cycle indices.

### Implementation Plan

**Change 1: Update function signature (Line 1061)**
```python
def _normalize_by_calibration(self,
                               aligned_multichannel_cycles: Dict[int, np.ndarray],
                               validation_results: List[Dict],
                               calibration_channel: int,
                               valid_cycle_indices: List[int]) -> Tuple[Dict[int, np.ndarray], List[float]]:
                               # ↑ NEW PARAMETER
```

**Change 2: Use correct indices for normalization factor lookup (Lines 1130-1142)**
```python
for cycle_idx, cycle_data in enumerate(channel_cycles):
    # Map aligned cycle index to original cycle index
    if cycle_idx < len(valid_cycle_indices):
        original_idx = valid_cycle_indices[cycle_idx]  # ← NEW: Get original index

        if original_idx < len(normalization_factors):
            norm_factor = normalization_factors[original_idx]  # ← Use original index!

            if norm_factor > 1e-6:
                normalized_cycle = cycle_data / norm_factor
                normalized_cycles.append(normalized_cycle)
            else:
                print(f"Warning: Skipping cycle {original_idx} (aligned position {cycle_idx}) "
                      f"in channel {ch_idx} (normalization factor too small: {norm_factor})")
                normalized_cycles.append(cycle_data)
```

**Change 3: Pass valid_cycle_indices at call site (Line 1361)**
```python
processed_cycles, normalization_factors = self._normalize_by_calibration(
    aligned_multichannel_cycles,
    validation_results,
    cal_ch,
    alignment_result['valid_cycle_indices']  # ← NEW: Pass the mapping
)
```

---

## Expected Behavior After Fix

### With Correct Index Mapping

**Scenario: Same 8 cycles, cycle #3 rejected**

```python
valid_cycle_indices = [0, 1, 2, 4, 5, 6, 7]
normalization_factors = [0.45, 0.32, 0.28, 0.15, 0.50, 0.38, 0.41, 0.35]

# Normalization loop:
for cycle_idx in range(7):  # 0-6 (aligned indices)
    original_idx = valid_cycle_indices[cycle_idx]
    norm_factor = normalization_factors[original_idx]

# Results:
cycle_idx=0 → original_idx=0 → norm_factor=0.45  ✅ CORRECT
cycle_idx=1 → original_idx=1 → norm_factor=0.32  ✅ CORRECT
cycle_idx=2 → original_idx=2 → norm_factor=0.28  ✅ CORRECT
cycle_idx=3 → original_idx=4 → norm_factor=0.50  ✅ CORRECT (was 0.15!)
cycle_idx=4 → original_idx=5 → norm_factor=0.38  ✅ CORRECT (was 0.50!)
cycle_idx=5 → original_idx=6 → norm_factor=0.41  ✅ CORRECT (was 0.38!)
cycle_idx=6 → original_idx=7 → norm_factor=0.35  ✅ CORRECT (was 0.41!)
```

### Amplitude Calculation

**Before Fix (Wrong Factors):**
```
Response channel cycle 3: peak = 0.85
Divided by wrong factor 0.15 → normalized peak = 5.67  ← Way too high!

Average of all normalized cycles → inconsistent amplitudes → wrong average
```

**After Fix (Correct Factors):**
```
Response channel cycle 3: peak = 0.85
Divided by correct factor 0.50 → normalized peak = 1.70  ← Correct!

Average of all normalized cycles → consistent amplitudes → correct average
```

---

## Testing

### Test Case 1: Verify Index Mapping
**Setup:**
- Record 8 cycles
- Ensure one cycle fails validation or correlation
- Enable normalization

**Steps:**
1. Record calibration series
2. Check console output for:
   - Which cycles are rejected
   - Normalization factors for each cycle
3. Add debug logging to show:
   ```
   Aligned cycle 0 from original cycle 0, norm_factor=0.45
   Aligned cycle 1 from original cycle 1, norm_factor=0.32
   Aligned cycle 2 from original cycle 2, norm_factor=0.28
   Aligned cycle 3 from original cycle 4, norm_factor=0.50  ← Should skip cycle 3!
   ...
   ```

**Expected:**
- ✅ Each aligned cycle uses normalization factor from its original cycle
- ✅ No off-by-one errors after rejected cycles
- ✅ Console output shows correct mapping

---

### Test Case 2: Amplitude Consistency
**Setup:**
- Record with varying impact strengths
- Enable normalization

**Steps:**
1. Record series with intentionally varying impacts
2. Check individual normalized cycle amplitudes
3. Check averaged impulse response amplitude
4. Calculate: `avg_amplitude / mean(individual_amplitudes)`

**Expected:**
- ✅ Ratio should be ~1.0 (within 5%)
- ✅ Averaged amplitude matches individual cycle amplitudes
- ✅ No unexplained amplitude reduction

---

### Test Case 3: Edge Case - First Cycle Rejected
**Setup:**
- Record series where first cycle is rejected

**Expected:**
- ✅ Aligned cycle 0 uses normalization factor from original cycle 1
- ✅ No index out of bounds errors
- ✅ Correct normalization applied

---

### Test Case 4: Edge Case - Last Cycle Rejected
**Setup:**
- Record series where last cycle is rejected

**Expected:**
- ✅ Only first 7 cycles normalized
- ✅ No attempt to access invalid index
- ✅ Correct normalization applied

---

## Related Issues

### Issue #3 - Normalization Not Applied
- **Status:** ✅ Fixed in [PHASE1_NORMALIZATION_FIX.md](PHASE1_NORMALIZATION_FIX.md)
- **Relation:** That fix ensured normalization was enabled; this fix ensures it's correct

### Issue #5 - Raw Recording Display
- **Status:** ✅ Fixed (previous session)
- **Relation:** Independent issue with data display, not normalization math

---

## Code Statistics

### Files Modified
- `RoomResponseRecorder.py` - 3 locations modified

### Lines Changed
- **Line 1061:** +1 parameter in function signature
- **Lines 1130-1142:** ~15 lines modified (index mapping logic)
- **Line 1361:** +1 parameter in function call
- **Total:** ~17 lines modified

### Breaking Changes
- **None** - Function signature extended (backward compatible if called positionally)
- **Risk:** Low - fix is localized to normalization logic

---

## Why This Matters

### Calibration Mode Purpose
The entire point of normalization is to:
1. **Remove impact strength variability** by dividing by magnitude
2. **Enable quantitative comparison** across cycles and measurements
3. **Produce accurate averaged responses** for analysis

### Without This Fix
- ❌ Wrong normalization factors applied to most cycles
- ❌ Normalized amplitudes unpredictable and incorrect
- ❌ Averaged response unreliable for quantitative analysis
- ❌ Data quality compromised
- ❌ Research conclusions may be invalid

### With This Fix
- ✅ Correct normalization factors applied to all cycles
- ✅ Normalized amplitudes consistent and accurate
- ✅ Averaged response correctly represents the data
- ✅ Data quality ensured
- ✅ Quantitative analysis reliable

---

## Severity Assessment

### Priority: 🔴 CRITICAL
- **Functionality:** Normalization feature produces wrong results
- **User Impact:** Silent failure - incorrect data without warning
- **Data Quality:** Compromises research data integrity
- **Discoverability:** Subtle bug - user noticed amplitude inconsistency

### Why Critical
1. Normalization is a **core calibration mode feature**
2. Bug produces **wrong results** (not just no results)
3. Failure is **subtle** (data looks plausible but is wrong)
4. Users may have collected **invalid datasets**
5. Affects **research data integrity**

---

## Conclusion

This fix ensures that **normalization uses the correct factors** for each cycle by properly mapping aligned cycle indices to original cycle indices. The index mapping mismatch was causing most cycles to be divided by wrong normalization factors, producing incorrect amplitudes and unreliable averaged responses.

**Status:** 🔧 Fix ready to implement
**Impact:** Calibration mode normalization will produce correct results
**Data Quality:** Normalized responses will be accurate and reliable
**Breaking Changes:** None

---

## Next Steps

1. ✅ Bug identified and documented
2. 📋 Implement the fix (3 code changes)
3. 📋 Test with real calibration recording
4. 📋 Verify amplitude consistency
5. 📋 Update Phase 1 completion documentation

---

**Thank you for identifying this critical issue!** 🙏
