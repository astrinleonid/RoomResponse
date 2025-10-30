# Multi-Channel Alignment Implementation

**Date:** 2025-10-30
**Status:** ✅ COMPLETE

---

## Overview

The onset-based cycle alignment is applied **uniformly across ALL channels** to maintain time-synchronization in multi-channel recordings.

## Key Principle

**Alignment shifts are calculated from the calibration channel, but applied to ALL channels identically.**

This ensures:
1. All channels remain time-synchronized
2. Inter-channel timing relationships are preserved
3. The onset (hammer impact) is at the same position in all channels

---

## Implementation

### Step 1: Calculate Alignment from Calibration Channel

**Location:** [RoomResponseRecorder.py:1201-1334](RoomResponseRecorder.py#L1201-L1334)

**Method:** `align_cycles_by_onset()`

The calibration channel is analyzed to:
- Find onset (negative peak) position in each valid cycle
- Calculate shifts needed to align all onsets to position 100 samples
- Filter cycles by correlation quality

**Returns:** `alignment_metadata` containing:
- `valid_cycle_indices`: Which cycles passed filters
- `onset_positions`: Original onset position in each kept cycle
- `aligned_onset_position`: Target position (100 samples)
- `correlations`: Quality metrics

### Step 2: Apply Same Shifts to All Channels

**Location:** [RoomResponseRecorder.py:1336-1386](RoomResponseRecorder.py#L1336-L1386)

**Method:** `apply_alignment_to_channel(channel_raw, alignment_metadata)`

For each channel:
1. Extract initial cycles using simple reshape
2. Apply the SAME shifts calculated from calibration channel
3. Return only the cycles that passed validation/correlation

**Example:**
```python
# After calculating alignment from calibration channel
alignment_result = recorder.align_cycles_by_onset(cal_cycles, validation_results)

# Apply to all channels
for channel_name, channel_data in recorded_audio.items():
    aligned_channel = recorder.apply_alignment_to_channel(
        channel_data,
        alignment_result
    )
    aligned_multichannel_cycles[channel_name] = aligned_channel
```

### Step 3: Store Aligned Multi-Channel Data

**Location:** [gui_audio_settings_panel.py:1313-1341](gui_audio_settings_panel.py#L1313-L1341)

The calibration test returns:

```python
{
    'success': True,
    'num_cycles': num_pulses,
    'calibration_channel': cal_ch,
    'sample_rate': sample_rate,

    # FOR EXISTING UI (Quality Metrics Table)
    'all_calibration_cycles': initial_cycles,  # ALL cycles, unaligned
    'validation_results': initial_validation_results,

    # FOR DOWNSTREAM USE (Actual Measurements)
    'aligned_multichannel_cycles': {
        'Calibration': aligned_cal_cycles,  # (num_valid, cycle_samples)
        'Channel_1': aligned_ch1_cycles,
        'Channel_2': aligned_ch2_cycles,
        ...
    },
    'alignment_metadata': alignment_result,
    'aligned_validation_results': aligned_validation_results,
    'cycle_duration_s': cycle_duration
}
```

---

## Data Flow

```
1. Record Multi-Channel Audio
   ↓
   {
     'Calibration': [raw audio],
     'Channel_1': [raw audio],
     'Channel_2': [raw audio],
     ...
   }

2. Analyze Calibration Channel
   ↓
   - Extract cycles (simple reshape)
   - Validate cycles
   - Find onset in each cycle
   - Calculate shifts to align onsets to position 100
   - Filter by correlation
   ↓
   alignment_metadata = {
     'valid_cycle_indices': [0, 2, 3, 5, ...],
     'onset_positions': [4523, 4501, 4489, ...],
     'aligned_onset_position': 100,
     'correlations': [0.95, 0.92, 0.88, ...]
   }

3. Apply Same Shifts to ALL Channels
   ↓
   For each channel:
     - Extract cycles (simple reshape)
     - Keep only valid_cycle_indices
     - Apply same shifts (align onset to position 100)
   ↓
   aligned_multichannel_cycles = {
     'Calibration': [[cycle0], [cycle2], [cycle3], ...],  # Only valid, aligned
     'Channel_1': [[cycle0], [cycle2], [cycle3], ...],    # Same cycles, same shifts
     'Channel_2': [[cycle0], [cycle2], [cycle3], ...],
     ...
   }

4. Use Aligned Data Downstream
   ↓
   - All channels have cycles aligned to onset at position 100
   - All channels have same cycles (invalid ones filtered out)
   - Inter-channel timing preserved
   - Ready for averaging, analysis, or storage
```

---

## Verification

### Test: Verify Multi-Channel Alignment

To verify that alignment is applied uniformly:

1. **Check onset positions match across channels:**
   ```python
   for channel_name, cycles in aligned_multichannel_cycles.items():
       for i, cycle in enumerate(cycles):
           onset = np.argmin(cycle)
           print(f"{channel_name} cycle {i}: onset at {onset} samples")

   # All channels should show onset at position 100
   ```

2. **Check same cycles kept in all channels:**
   ```python
   num_cycles_per_channel = {
       ch: len(cycles)
       for ch, cycles in aligned_multichannel_cycles.items()
   }
   print(num_cycles_per_channel)

   # All channels should have same count (only valid cycles)
   ```

3. **Visual verification in UI:**
   - Run calibration test
   - Check "Aligned Cycles Overlay" visualization
   - All cycles should overlay with onset at beginning
   - Check multiple valid cycles to verify consistency

---

## Usage in Downstream Code

### For Piano Response Measurements

When recording piano responses, use the aligned data:

```python
# After calibration test
calibration_results = _perform_calibration_test()

# Get aligned multi-channel cycles
aligned_cycles = calibration_results['aligned_multichannel_cycles']

# Average valid cycles for each channel
averaged_responses = {}
for channel_name, cycles in aligned_cycles.items():
    if len(cycles) > 0:
        averaged_responses[channel_name] = np.mean(cycles, axis=0)

# Now averaged_responses has:
# - All channels aligned to onset at position 100
# - Only valid cycles averaged
# - Inter-channel timing preserved
```

### For Threshold Learning

When learning quality thresholds from user-marked "good" cycles:

```python
# User marks cycles 2, 3, 5 as "good"
good_cycle_indices = [2, 3, 5]

# Get aligned calibration cycles
aligned_cal = calibration_results['aligned_multichannel_cycles']['Calibration']

# Learn thresholds from aligned data
from calibration_validator_v2 import QualityThresholds

# Map user selection to valid cycle indices
alignment_metadata = calibration_results['alignment_metadata']
valid_indices = alignment_metadata['valid_cycle_indices']

# Find which aligned cycles correspond to user selection
good_aligned_indices = []
for user_idx in good_cycle_indices:
    if user_idx in valid_indices:
        aligned_idx = valid_indices.index(user_idx)
        good_aligned_indices.append(aligned_idx)

# Extract good cycles (already aligned)
good_cycles = aligned_cal[good_aligned_indices]

# Learn thresholds
thresholds = QualityThresholds.from_user_marked_cycles(
    cycles=good_cycles,
    marked_good=list(range(len(good_cycles))),
    sample_rate=sample_rate
)
```

---

## Important Notes

### Why Apply to All Channels?

**Problem:** If we only aligned the calibration channel, other channels would have:
- Onsets at different positions
- Invalid cycles still present
- Timing drift between channels

**Solution:** Apply the SAME shifts to all channels, ensuring:
- All onsets at position 100 across all channels
- Same cycles kept in all channels (invalid ones removed everywhere)
- Perfect time-synchronization maintained

### Circular Shift with np.roll()

**Behavior:** `np.roll(cycle, shift)` performs circular shift:
- Positive shift: moves data right, wraps end to beginning
- Negative shift: moves data left, wraps beginning to end

**Example:**
```python
cycle = [0, 0, 0, -0.8, 0.3, 0.1, 0]  # Onset at position 3
shifted = np.roll(cycle, -3 + 100)    # Shift to position 100

# Result: onset now at position 100
# Data before position 3 wraps to end
```

**Implication:** Ensure cycles have sufficient silence at boundaries to avoid wrap-around artifacts.

### Target Onset Position

Currently hardcoded to **100 samples** in [RoomResponseRecorder.py:1263](RoomResponseRecorder.py#L1263):

```python
target_onset_position = 100  # Position onset at 100 samples (near beginning)
aligned_onset_position = target_onset_position
```

**Rationale:**
- Position 100 is near the chart beginning (user requirement)
- Leaves small buffer (100 samples) for pre-onset data
- Consistent across all channels and all cycles

**Future Enhancement:** Could make this configurable if needed.

---

## Summary

✅ **Multi-channel alignment is fully implemented:**
1. Shifts calculated from calibration channel
2. Same shifts applied to ALL channels uniformly
3. Inter-channel timing relationships preserved
4. All onsets aligned to position 100 (near beginning)
5. Only valid, high-correlation cycles kept
6. Data ready for downstream averaging/analysis

The implementation ensures that when you visualize or process aligned cycles, ALL channels are time-synchronized and have their onset (hammer impact) at the same position.
