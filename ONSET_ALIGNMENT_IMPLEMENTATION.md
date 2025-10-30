# Onset-Based Cycle Alignment Implementation

**Date:** 2025-10-30
**Status:** ✅ COMPLETE

---

## Overview

This document describes the implementation of onset-based cycle alignment for piano impulse response measurements. This replaces the previous two-stage approach with a simpler, more direct method.

### Key Differences from Previous Approach

**Previous (Two-Stage):**
1. Extract cycles with simple reshape
2. Calculate shifts via cross-correlation
3. Re-extract from raw audio using shifts

**Current (Onset-Based):**
1. Extract cycles with simple reshape
2. Filter to only VALID cycles
3. Find onset (negative peak) in each cycle
4. Align ALL cycles by shifting peaks to common position
5. Filter again by correlation

---

## Implementation Details

### Step 1-4: Initial Extraction (AS-IS Logic)

**Location:** [gui_audio_settings_panel.py:1260-1293](gui_audio_settings_panel.py#L1260-L1293)

```python
# Pad or trim to expected length
expected_samples = self.recorder.cycle_samples * self.recorder.num_pulses
if len(cal_raw) < expected_samples:
    padded = np.zeros(expected_samples)
    padded[:len(cal_raw)] = cal_raw
    cal_raw = padded
else:
    cal_raw = cal_raw[:expected_samples]

# Simple reshape into individual cycles
initial_cycles = cal_raw.reshape(self.recorder.num_pulses, self.recorder.cycle_samples)

# Run validation on initial cycles
for cycle_idx in range(self.recorder.num_pulses):
    validation = validator.validate_cycle(initial_cycles[cycle_idx], cycle_idx)
    initial_validation_results.append(validation_dict)
```

### Step 5: Align Cycles by Onset

**Location:** [RoomResponseRecorder.py:1201-1330](RoomResponseRecorder.py#L1201-L1330)

**Method:** `align_cycles_by_onset(initial_cycles, validation_results, correlation_threshold=0.7)`

**Algorithm:**

#### 1. Filter to Valid Cycles Only
```python
valid_indices = [i for i, v in enumerate(validation_results)
                 if v.get('calibration_valid', False)]
valid_cycles = initial_cycles[valid_indices]
```

**Why:** Invalid cycles don't contain proper impulses and should not be included in alignment.

#### 2. Find Onset (Negative Peak) in Each Cycle
```python
onset_positions = []
for cycle in valid_cycles:
    onset_idx = int(np.argmin(cycle))  # Find minimum = negative peak
    onset_positions.append(onset_idx)
```

**Why:** The negative peak represents the hammer impact onset in piano measurements.

#### 3. Determine Common Onset Position
```python
aligned_onset_position = int(np.median(onset_positions))
```

**Why:** Using median is robust to outliers. All cycles will be shifted so their onsets align at this position.

#### 4. Align All Cycles by Circular Shift
```python
aligned_cycles_list = []
for i, cycle in enumerate(valid_cycles):
    shift_needed = aligned_onset_position - onset_positions[i]
    aligned_cycle = np.roll(cycle, shift_needed)  # Circular shift
    aligned_cycles_list.append(aligned_cycle)
```

**Why:** `np.roll()` performs circular shift, moving the onset to the common position without losing data.

#### 5. Select Reference Cycle
```python
energies = np.sqrt(np.mean(aligned_cycles ** 2, axis=1))
reference_idx = int(np.argmax(energies))
reference_cycle = aligned_cycles[reference_idx]
```

**Why:** Highest energy cycle is likely to have the clearest signal for correlation comparison.

#### 6. Calculate Cross-Correlation at Zero Lag
```python
for i, cycle in enumerate(aligned_cycles):
    if i == reference_idx:
        correlations.append(1.0)
    else:
        # Normalized cross-correlation
        ref_energy = np.sum(reference_cycle ** 2)
        cyc_energy = np.sum(cycle ** 2)
        cross_product = np.sum(reference_cycle * cycle)

        if ref_energy > 0 and cyc_energy > 0:
            corr_value = float(cross_product / np.sqrt(ref_energy * cyc_energy))
        else:
            corr_value = 0.0

        correlations.append(corr_value)
```

**Why:** After alignment, cycles should have similar waveforms. Low correlation indicates problems (noise, artifacts, wrong impulse detection).

#### 7. Filter by Correlation Threshold
```python
final_indices = [i for i, corr in enumerate(correlations)
                 if corr >= correlation_threshold]

final_aligned_cycles = aligned_cycles[final_indices]
final_valid_indices = [valid_indices[i] for i in final_indices]
```

**Why:** Only keep cycles that correlate well with the reference after alignment. This is the second quality filter.

### Return Data Structure

```python
{
    'aligned_cycles': np.ndarray,           # (num_kept, cycle_samples) - Only filtered, aligned cycles
    'valid_cycle_indices': List[int],       # Original indices of kept cycles
    'onset_positions': List[int],           # Onset position found in each original cycle
    'aligned_onset_position': int,          # Common onset position in aligned cycles
    'correlations': List[float],            # Correlation values for kept cycles
    'reference_cycle_idx': int,             # Index of reference (in kept set)
    'correlation_threshold': float          # Threshold used
}
```

---

## UI Implementation

**Location:** [gui_audio_settings_panel.py:1551-1739](gui_audio_settings_panel.py#L1551-L1739)

### Features

#### 1. Summary Metrics (4 columns)
- **Initial Cycles:** Total cycles extracted
- **Valid & Aligned:** Cycles passing both validation and correlation filters
- **Mean Correlation:** Average correlation among kept cycles
- **Aligned Onset Pos:** Common position where all onsets are aligned (in samples)

#### 2. Aligned Cycles Table
Displays only the cycles that passed all filters:
- **Cycle #:** Original cycle index
- **Original Onset:** Onset position before alignment (samples)
- **Aligned Onset:** Common onset position after alignment (samples)
- **Correlation:** Correlation value with reference
- **Valid:** Validation status (all shown cycles are valid)
- **Neg. Peak:** Negative peak magnitude
- **Note:** Shows "REF" for reference cycle

**Key Points:**
- Invalid cycles are NOT shown
- Only cycles passing correlation filter are shown
- Checkboxes allow selection for visualization

#### 3. Aligned Cycles Overlay Visualization
- Shows selected cycles overlaid
- All cycles should overlap precisely at the onset
- Negative peak should be near the beginning of the chart
- Uses AudioVisualizer component for interactive display

---

## Expected Behavior

### What User Should See

1. **Invalid Cycles Filtered Out**
   - Only valid cycles appear in the table
   - Invalid cycles completely excluded from analysis

2. **All Cycles Aligned to Each Other**
   - When viewing multiple cycles overlaid, waveforms should overlap exactly
   - Negative peaks should be at the same position in all cycles
   - Slight variations are normal (noise), but main features should align

3. **Cycles Aligned by Onset**
   - Negative peak (onset) should be near the beginning of the visualization
   - Position shown in "Aligned Onset Pos" metric
   - All cycles shifted so onset is at this common position

### Verification

Run the test script to verify alignment quality:

```bash
python test_two_stage_alignment.py
```

**Expected Output:**
- All valid cycles should have onset at same position
- Maximum alignment error should be 0-2 samples
- Correlations should be high (>0.7 for good cycles)
- Mean correlation typically >0.8 for clean signals

---

## Key Design Decisions

### 1. Why Onset-Based Alignment?

**Piano impulse responses have a clear onset:** The negative peak represents the hammer impact, which is the most reliable timing reference.

**Advantages:**
- Simple and robust
- Works even if overall timing varies between cycles
- Aligns the most important feature (hammer impact)
- No need for re-extraction from raw audio

### 2. Why Two-Stage Filtering?

**First filter (validation):** Removes cycles without proper impulse characteristics:
- Missing negative peak
- Wrong peak ratios
- Excessive noise

**Second filter (correlation):** Removes cycles that don't match reference after alignment:
- Artifacts
- Wrong onset detection
- Phase issues

### 3. Why Circular Shift with np.roll()?

**Circular shift preserves all data:** Unlike window-based extraction, circular shift doesn't lose samples.

**Pros:**
- No boundary effects
- Simple implementation
- Reversible operation

**Cons:**
- Wraps data around (end appears at beginning)
- User should ensure cycles have sufficient silence at boundaries

### 4. Why Median for Common Onset Position?

**Robust to outliers:** A few cycles with wrong onset detection won't affect the result.

**Alternative (mean):** Would be pulled by outliers.

---

## Testing

### Test Script

**File:** [test_two_stage_alignment.py](test_two_stage_alignment.py)

Generates synthetic impulse train with:
- 10 impulses
- Negative peak at 0.8
- Ringing after each impulse
- Small amount of noise

**Validates:**
1. All cycles pass validation
2. Onset detected at same position in all cycles (456 samples in test)
3. After alignment, all onsets at common position
4. Maximum alignment error = 0 samples (perfect)
5. High correlations (>0.8)

### Test Output Example

```
Validating 10 initial cycles...
Cycle 0: VALID (neg_peak=0.801)
...
Cycle 9: VALID (neg_peak=0.802)

Align cycles by onset (negative peak detection)...
- Valid cycles kept: 10/10
- Aligned onset position: 456 samples
- Reference cycle: 4 (in valid set)
- Correlation threshold: 0.3

Verifying Alignment Quality...
Cycle 0: onset at 456 samples (error=0) [OK]
...
Cycle 9: onset at 456 samples (error=0) [OK]
Maximum alignment error: 0 samples

Summary Statistics
Mean correlation: 0.844
Alignment accuracy: max error = 0 samples
```

---

## Comparison with Previous Two-Stage Approach

| Aspect | Two-Stage (Old) | Onset-Based (New) |
|--------|-----------------|-------------------|
| **Steps** | Extract → Calculate shifts → Re-extract | Extract → Filter → Align → Filter |
| **Alignment Method** | Cross-correlation search window | Direct onset detection |
| **Data Processing** | Re-extraction from raw audio | In-place circular shift |
| **Complexity** | Higher (3 stages) | Lower (1 alignment step) |
| **Filtering** | Validation + Correlation | Validation + Correlation |
| **UI Comparison** | Initial vs Realigned | Only Aligned (overlaid) |
| **Result** | Same quality | Same quality, simpler |

**Both approaches achieve the same goal:** Align cycles by their onset so they can be compared or averaged.

**New approach is preferred because:**
- Simpler to understand and maintain
- More direct (no re-extraction needed)
- Easier to visualize (just overlay aligned cycles)
- Equivalent quality

---

## Troubleshooting

### Problem: Only 1 cycle passes correlation filter

**Cause:** Cycles have high noise or are not similar to each other.

**Solutions:**
1. Lower correlation threshold (default 0.7)
2. Check if cycles actually contain similar impulses
3. Verify recording quality

### Problem: Alignment error > 2 samples

**Cause:** Onset detection finding wrong peak (noise instead of impulse).

**Solutions:**
1. Improve validation thresholds
2. Filter out noisy cycles earlier
3. Check if impulses are actually present

### Problem: Negative peak not at beginning of chart

**Cause:** Cycles have long silence before impulse.

**Expected:** This is normal! The aligned onset position shows where the peak is.

**Not a problem if:** All cycles align at that position.

---

## Future Enhancements

1. **Fractional Sample Alignment**
   - Use interpolation for sub-sample precision
   - Potential for even better alignment

2. **Adaptive Correlation Threshold**
   - Automatically adjust based on signal quality
   - Keep minimum number of cycles

3. **Multi-Channel Extension**
   - Apply same shifts to all channels
   - Maintain inter-channel timing relationships

4. **Visual Alignment Quality Indicator**
   - Show alignment error distribution
   - Highlight cycles with poor alignment

---

## Summary

The onset-based cycle alignment successfully implements the user's requirements:

1. ✅ Invalid cycles filtered out - only valid cycles analyzed
2. ✅ All cycles aligned to each other - charts should overlay exactly
3. ✅ Cycles aligned by onset - negative peak at common position

The implementation is simple, robust, and provides high-quality alignment for piano impulse response measurements.
