# Cycle Statistics Table - Quality Control Feature

**Date:** 2025-11-02
**Feature:** Comprehensive cycle statistics table for calibration mode
**Status:** ✅ IMPLEMENTED
**Priority:** HIGH - Enables data quality verification

---

## Overview

Added a comprehensive cycle statistics table to the Recording & Analysis section that displays detailed information about all cycles (valid and rejected) in calibration mode recordings. This enables users to verify data quality and understand why cycles were rejected.

---

## User Request

> "Add to the Research & Analysis section the table with cycle statistics similar to what is done in the Calibration Cycle section. It should include all cycles - valid and invalid, with indication why the cycles were rejected, peak amplitudes before and after normalizing, exact positioning (in number of samples) of the main peak after the alignment, and also the same statistics for the averaged impulse response. That way I would be able to check the quality of the record and control the correctness of the averaging."

---

## Features Implemented

### 1. Cycle Statistics Table

**Displays for Each Cycle:**
- ✅ **Cycle Number** - Original cycle index (0-based)
- ✅ **Status** - "✓ Kept" (green background) or "✗ Rejected" (red background)
- ✅ **Rejection Reason** - Detailed explanation:
  - `Validation: [failure1, failure2, ...]` - Failed quality validation
  - `Correlation filter (< threshold)` - Failed correlation check
  - `-` - Cycle was kept (no rejection)
- ✅ **Impact Magnitude** - Normalization factor (|negative_peak|) from calibration sensor
- ✅ **Peak Before** - Peak amplitude of aligned cycle (before normalization)
- ✅ **Peak After** - Peak amplitude after normalization (if enabled)
- ✅ **Onset Position** - Sample position of negative peak in original cycle
- ✅ **Aligned Position** - Sample position after alignment (same for all kept cycles)

**Visual Features:**
- ✅ Color-coded rows: Green for kept cycles, red for rejected cycles
- ✅ Full-width table display
- ✅ Only shown for calibration mode recordings
- ✅ Updates when user changes selected channel

### 2. Averaged Impulse Response Statistics

**Displays Below Table:**
- ✅ **Cycles Averaged** - Number of cycles included in average
- ✅ **Avg Peak Amplitude** - Peak amplitude of averaged response
- ✅ **Mean Individual Peak** - Mean peak amplitude of individual cycles
- ✅ **Peak Ratio** - Ratio of averaged peak to mean individual peak
  - Should be ~1.0 if normalization/averaging is correct
  - Shows warning if outside 0.95-1.05 range
- ✅ **Avg Peak Position** - Sample position of peak in averaged response

**Purpose:**
- Verify that averaged amplitude matches individual cycle amplitudes
- Detect normalization issues (Issue #6 fix validation)
- Ensure averaging is working correctly

---

## Implementation Details

### File Modified
- **`gui_series_settings_panel.py`** - Added ~180 lines

### New Function
**`_render_cycle_statistics_table()` (Lines 986-1161)**

**Logic Flow:**
1. Check if recording mode is calibration (skip if standard mode)
2. Extract metadata from `recorded_audio`:
   - `validation_results` - validation status for each cycle
   - `alignment_metadata` - alignment results and valid cycle indices
   - `aligned_multichannel_cycles` - pre-normalization cycle data
   - `normalized_multichannel_cycles` - post-normalization cycle data
   - `normalization_factors` - impact magnitudes
3. Get selected channel from session state
4. Build table data:
   - Iterate through all validation results
   - For each cycle:
     - Check if kept after alignment (in `valid_cycle_indices`)
     - Determine rejection reason (validation or correlation)
     - Extract peak amplitudes (before/after normalization)
     - Extract onset and aligned positions
5. Display styled DataFrame with color-coded rows
6. Calculate and display averaged impulse response statistics

### Integration Point
**Line 902 in `_render_recording_analysis()`:**
```python
if analysis:
    self._display_analysis_metrics(analysis)

# Show cycle statistics table for calibration mode
self._render_cycle_statistics_table()  # ← NEW

self._render_visualization_controls()
```

---

## Data Sources

### From `recorded_audio` Dictionary

**`metadata` Section:**
```python
metadata = {
    'mode': 'calibration',
    'validation_results': [
        {
            'cycle_index': 0,
            'calibration_valid': True,
            'calibration_metrics': {'negative_peak': -0.45, ...},
            'calibration_failures': []
        },
        ...
    ],
    'alignment_metadata': {
        'valid_cycle_indices': [0, 1, 2, 4, 5, 6, 7],
        'onset_positions': [125, 130, 120, 128, 132, 127, 129],
        'aligned_onset_position': 100,
        'correlations': [1.0, 0.95, 0.92, 0.88, 0.90, 0.93, 0.91]
    },
    'correlation_threshold': 0.7,
    'normalize_by_calibration': True
}
```

**Top-Level Keys:**
```python
recorded_audio = {
    'aligned_multichannel_cycles': {
        0: np.array([[...], [...]]),  # aligned cycles for channel 0
        1: np.array([[...], [...]]),  # aligned cycles for channel 1
        ...
    },
    'normalized_multichannel_cycles': {
        0: np.array([[...], [...]]),  # normalized cycles for channel 0
        1: np.array([[...], [...]]),  # normalized cycles for channel 1
        ...
    },
    'normalization_factors': [0.45, 0.32, 0.28, 0.50, ...],
    'metadata': { ... }
}
```

---

## Usage Example

### Scenario: 8 Cycles Recorded, 1 Rejected

**Cycle Statistics Table:**

| Cycle | Status | Rejection Reason | Impact Mag | Peak Before | Peak After | Onset Pos | Aligned Pos |
|-------|--------|------------------|------------|-------------|------------|-----------|-------------|
| 0 | ✓ Kept | - | 0.4500 | 0.8500 | 1.8889 | 125 | 100 |
| 1 | ✓ Kept | - | 0.3200 | 0.6100 | 1.9063 | 130 | 100 |
| 2 | ✓ Kept | - | 0.2800 | 0.5300 | 1.8929 | 120 | 100 |
| 3 | ✗ Rejected | Validation: impact_too_weak | 0.1500 | - | - | - | - |
| 4 | ✓ Kept | - | 0.5000 | 0.9500 | 1.9000 | 128 | 100 |
| 5 | ✓ Kept | - | 0.3800 | 0.7200 | 1.8947 | 132 | 100 |
| 6 | ✓ Kept | - | 0.4100 | 0.7800 | 1.9024 | 127 | 100 |
| 7 | ✓ Kept | - | 0.3500 | 0.6600 | 1.8857 | 129 | 100 |

*(Green background for kept cycles, red for rejected)*

**Averaged Impulse Response Statistics:**

| Cycles Averaged | Avg Peak Amplitude | Mean Individual Peak | Peak Ratio | Avg Peak Position |
|-----------------|-------------------|---------------------|------------|------------------|
| 7 | 1.8958 | 1.8958 | 1.000 | 100 samples |

**Interpretation:**
- ✅ Cycle 3 rejected due to weak impact (0.15 vs ~0.35-0.50 for others)
- ✅ All kept cycles have consistent normalized peaks (~1.89)
- ✅ Peak ratio is 1.000 - averaging is working correctly!
- ✅ All onsets aligned to position 100 samples

---

## Quality Control Benefits

### 1. Verify Rejection Reasons
Users can see exactly why cycles were rejected:
- **Validation failures:** impact_too_weak, impact_too_strong, noise_level_high, etc.
- **Correlation failures:** Cycles that don't align well with reference

### 2. Verify Normalization
Compare "Peak Before" and "Peak After" columns:
- Before: Varying amplitudes (0.53-0.95) due to varying impact strengths
- After: Consistent amplitudes (~1.89) - normalization working correctly!
- Impact Mag: Shows the actual impact strength measured by calibration sensor

### 3. Verify Alignment
Check "Onset Pos" and "Aligned Pos" columns:
- Onset Pos: Original positions vary (120-132 samples)
- Aligned Pos: All aligned to same position (100 samples)
- Confirms alignment is working correctly

### 4. Verify Averaging
Check "Peak Ratio" metric:
- **Ratio ~1.0:** Averaging is correct, no amplitude loss
- **Ratio < 0.95:** Possible issue with normalization or averaging
- **Ratio > 1.05:** Unusual, may indicate data quality issue

### 5. Identify Problematic Cycles
- Weak impacts: Low "Impact Mag" values
- Strong impacts: High "Impact Mag" values
- Inconsistent peaks: "Peak After" values varying significantly
- Alignment issues: "Onset Pos" values far from mean

---

## Testing Verification

### Test Case 1: All Cycles Valid
**Setup:** Record with consistent impacts
**Expected:**
- ✅ All rows show "✓ Kept" with green background
- ✅ "Peak After" values all similar (~within 5%)
- ✅ Peak Ratio ~1.0

### Test Case 2: Some Cycles Rejected
**Setup:** Record with one weak impact
**Expected:**
- ✅ Weak cycle shows "✗ Rejected" with red background
- ✅ Rejection reason: "Validation: impact_too_weak"
- ✅ Impact Mag shows low value for rejected cycle
- ✅ Remaining cycles have consistent Peak After values

### Test Case 3: Normalization Disabled
**Setup:** Uncheck normalization checkbox
**Expected:**
- ✅ "Peak After" column shows "N/A"
- ✅ "Peak Before" shows varying amplitudes
- ✅ Peak Ratio not calculated (no normalization to verify)

### Test Case 4: Channel Selection
**Setup:** Change selected channel in dropdown
**Expected:**
- ✅ Table updates to show peaks for new channel
- ✅ Status/rejection reasons stay same (based on calibration channel)
- ✅ Peak values change (different channel response)

---

## Code Statistics

### Lines Added
- **`_render_cycle_statistics_table()` function:** ~175 lines (986-1161)
- **Integration call:** 2 lines (901-902)
- **Total:** ~177 lines

### Dependencies
- `pandas` - DataFrame creation and styling
- `numpy` - Array operations and statistics
- `streamlit` - UI rendering

### Performance
- **Complexity:** O(n) where n = number of cycles (typically 8-12)
- **Impact:** Minimal - table generation is fast
- **Memory:** Small - only statistics, not full audio data

---

## Integration with Issue #6 Fix

This feature directly supports verification of the Issue #6 fix (averaging amplitude):

**Before Fix:**
- Peak Ratio might show 0.6-0.8 (too low)
- Warning would appear: "Peak ratio outside expected range"
- User could see in table that wrong normalization factors were used

**After Fix:**
- Peak Ratio shows ~1.0 (correct)
- No warning
- Table confirms each cycle normalized by correct impact magnitude

**Validation Flow:**
1. User records calibration series
2. Checks cycle statistics table
3. Verifies "Impact Mag" matches expected values
4. Verifies "Peak After" values are consistent
5. Checks "Peak Ratio" is ~1.0
6. ✅ Confirms normalization and averaging are correct!

---

## User Benefits

### Research Data Quality
- ✅ **Transparency:** See exactly what happened to each cycle
- ✅ **Reproducibility:** Document which cycles were used
- ✅ **Quality Control:** Verify data meets research standards

### Troubleshooting
- ✅ **Identify Issues:** See which cycles failed and why
- ✅ **Adjust Thresholds:** Decide if quality thresholds are appropriate
- ✅ **Verify Fixes:** Confirm that bug fixes work correctly

### Confidence
- ✅ **Trust the Data:** Verify that averaging produces correct results
- ✅ **Quantitative Analysis:** Ensure normalization is working
- ✅ **Publication Ready:** Document data quality in papers

---

## Future Enhancements (Optional)

### Potential Additions
1. **Export to CSV:** Allow users to save table for analysis
2. **Correlation Values:** Show actual correlation value for kept cycles
3. **RMS Level:** Show RMS level in addition to peak amplitude
4. **SNR Estimate:** Calculate signal-to-noise ratio per cycle
5. **Visual Indicators:** Color-code cells based on values (not just rows)
6. **Sorting:** Allow user to sort table by any column
7. **Filtering:** Show only kept or only rejected cycles

### Not Implemented (Out of Scope)
- Multi-channel comparison table (would be very wide)
- Time-series plots of metrics across cycles
- Statistical analysis (mean, std, outliers)

---

## Related Documentation

- [PHASE1_AVERAGING_AMPLITUDE_FIX.md](PHASE1_AVERAGING_AMPLITUDE_FIX.md) - Issue #6 fix that this table helps verify
- [PHASE1_NORMALIZATION_FIX.md](PHASE1_NORMALIZATION_FIX.md) - Issue #3 fix for normalization
- [PHASE1_IMPLEMENTATION_SUMMARY.md](PHASE1_IMPLEMENTATION_SUMMARY.md) - Overall Phase 1 implementation

---

## Conclusion

The cycle statistics table provides comprehensive quality control for calibration mode recordings. Users can now:

1. **See all cycles** - valid and rejected
2. **Understand rejections** - detailed reasons for each rejected cycle
3. **Verify normalization** - compare peaks before/after normalization
4. **Check alignment** - see onset positions and aligned positions
5. **Validate averaging** - confirm averaged amplitude matches individual cycles

This feature directly addresses the user's request and enables rigorous data quality verification for research applications.

**Status:** ✅ **Implemented and ready for testing**

---

**Next Step:** Test with real calibration recording to verify all statistics display correctly
