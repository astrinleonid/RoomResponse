# ESPRIT Preprocessing Analysis

## Existing RoomResponse Preprocessing

### Summary

Your measurement data from `RoomResponseRecorder.py` is **already fully preprocessed** and ready for ESPRIT analysis. The following processing has been applied:

1. ✅ **Onset Alignment** - Done by `signal_processor.py`
2. ✅ **Quality Filtering** - Done during measurement
3. ✅ **Fade-out Windowing** - Hann window applied

---

## Detailed Analysis

### 1. Onset Alignment ✅ COMPLETE

**Location:** [signal_processor.py:250-383](signal_processor.py#L250-L383)

**Method:** `align_cycles_by_onset()`

**Process:**
```python
# Step 1: Find negative peak (force maximum) in each cycle
onset_idx = np.argmin(cycle)  # Line 306

# Step 2: Align all cycles to common position
aligned_cycle = np.roll(cycle, shift_needed)  # Line 321
```

**Result:**
- All cycles aligned by force channel maximum
- Cross-correlation > 0.7 ensures good alignment
- Only aligned, valid cycles are averaged
- **No additional onset detection needed**

---

### 2. Quality Filtering ✅ COMPLETE

**Location:** `CalibrationValidatorV2` (validation during measurement)

**Checks Applied:**
- **Clipping detection**: Rejects overloaded signals
- **Double hit detection**: Rejects hammer bounce
- **Peak amplitude range**: Validates impact strength
- **Aftershock check**: Ensures clean impact
- **Minimum valid cycles**: Requires sufficient good data

**Configuration:** [RoomResponseRecorder.py:66-81](RoomResponseRecorder.py#L66-L81)
```python
calibration_quality_config = {
    'min_negative_peak': 0.1,
    'max_negative_peak': 0.95,
    'max_aftershock': 0.3,
    'min_valid_cycles': 3
}
```

**Result:**
- Only high-quality impacts are kept
- Bad cycles are rejected before averaging
- **No additional quality checks needed**

---

### 3. Fade-out Windowing ✅ COMPLETE

**Location:** [signal_processor.py:544-594](signal_processor.py#L544-L594)

**Method:** `truncate_with_fadeout()`

**Process:**
```python
# Apply Hann window (second half) for smooth fade-out
fade_window = np.hanning(fade_samples * 2)[fade_samples:]
truncated[fade_start:] *= fade_window  # Line 591-592
```

**Configuration:** [RoomResponseRecorder.py:84-88](RoomResponseRecorder.py#L84-L88)
```python
truncate_config = {
    'enabled': True/False,
    'ir_working_length_ms': 500.0,  # Total length
    'ir_fade_length_ms': 50.0        # Fade duration
}
```

**Example:** 500ms working length with 50ms fade
```
|←────── 450ms constant ──────→|←── 50ms Hann fade ──→|
```

**Result:**
- Smooth transition to zero at end
- Minimizes spectral leakage from finite observation
- **No additional windowing needed**

---

## ESPRIT Preprocessing Recommendation

### ⚠️ DO NOT Add Exponential Windowing

**Reason:** Your signals are already properly windowed with Hann fade.

**Adding exponential window would:**
1. **Corrupt damping estimates** - Artificially increases modal damping
2. **Duplicate fade functionality** - Hann fade already handles spectral leakage
3. **Violate assumptions** - ESPRIT expects free vibration, not artificially damped

### Mathematical Explanation

**Current (Correct):**
```
Signal: y(t) = Σ A_k · e^(-ζ_k·ω_k·t) · sin(ω_d·t)
Window: w(t) = 1 for t ∈ [0, T-50ms], Hann fade for t ∈ [T-50ms, T]
Result: y_windowed(t) = y(t) · w(t)
```

**If exponential window added (Incorrect):**
```
Additional: w_exp(t) = e^(-α·t), α = -ln(10^(-70/20))/T
Result: y_double_windowed(t) = y(t) · w(t) · w_exp(t)
Problem: System appears to have damping ζ_measured = ζ_actual + α/(2πf)
```

**Example corruption:**
- Mode at f = 200 Hz, actual ζ = 3%
- Exponential window α = 0.019 (for -70dB at 600ms)
- Measured ζ = 3% + 0.019/(2π×200) ≈ 3% + 1.5% = 4.5%
- **50% error in damping!**

---

## Recommended ESPRIT Workflow

### Use Minimal Preprocessing

**New module:** `preprocessing_minimal.py`

**Only performs:**
1. **Load data** from TSV file
2. **Optional DC removal** - High-pass filter at 1 Hz
3. **Return full signal** - No segmentation, no windowing

**Usage:**
```python
from preprocessing_minimal import load_measurement_file, preprocess_measurement

# Load pre-processed data
force, responses = load_measurement_file('measurement.txt', skip_channel=2)

# Minimal preprocessing (optional DC removal only)
processed_responses, metadata = preprocess_measurement(force, responses, fs=48000)

# Ready for ESPRIT
modal_params = esprit_modal_identification(processed_responses, fs=48000, ...)
```

---

## Comparison: Old vs New Preprocessing

### Old Preprocessing (preprocessing.py) - ❌ NOT NEEDED

```python
# STEP 1: Onset detection ← Already done by RoomResponse
onset = detect_onset(force)

# STEP 2: Contact end ← Not needed, signals aligned
contact_end = detect_contact_end(force)

# STEP 3: Segment extraction ← Not needed, use full signal
segment = extract_decay_segment(...)

# STEP 4: Exponential windowing ← HARMFUL - corrupts damping
windowed = segment * exp_window

# STEP 5: High-pass filtering ← OK, keep this
filtered = highpass_filter(...)
```

### New Preprocessing (preprocessing_minimal.py) - ✅ CORRECT

```python
# STEP 1: Load data
force, responses = load_measurement_file('measurement.txt')

# STEP 2: Optional DC removal (only needed step)
if use_highpass:
    responses = highpass_filter(responses, fs, cutoff=1.0)

# STEP 3: Done! Use full signal for ESPRIT
return responses
```

---

## Note for Main Package Update

### Current Hann Fade Configuration

**File:** `recorderConfig.json` or GUI settings

**Parameters:**
```json
{
  "truncate_config": {
    "enabled": true,
    "ir_working_length_ms": 500.0,
    "ir_fade_length_ms": 50.0
  }
}
```

### ⚠️ Important: Do NOT Disable Hann Fade

The current Hann fade is **appropriate and beneficial** for ESPRIT analysis:

**Benefits:**
- ✅ Smoothly brings signal to zero at end
- ✅ Minimizes spectral leakage in FFT operations
- ✅ Does not corrupt modal parameters
- ✅ Standard practice in modal analysis

**Recommendation:**
- **Keep** `truncate_config.enabled = true`
- **Keep** `ir_fade_length_ms = 50.0` (typical value)
- **Do NOT** add exponential windowing to ESPRIT

---

## Summary

| Processing Step | Status | Location | Action |
|----------------|---------|----------|--------|
| **Onset Alignment** | ✅ Done | `signal_processor.py:250` | None - use as-is |
| **Quality Filtering** | ✅ Done | Measurement time | None - use as-is |
| **Hann Fade** | ✅ Done | `signal_processor.py:544` | **Keep enabled** |
| **DC Removal** | ⚠️ Optional | ESPRIT preprocessing | Add if needed |
| **Exponential Window** | ❌ Not needed | N/A | **Do NOT add** |

---

## Conclusion

Your `RoomResponseRecorder` pipeline produces signals that are **ready for ESPRIT analysis** with minimal additional processing:

1. ✅ Load data using `preprocessing_minimal.py`
2. ✅ Optionally apply high-pass filter for DC removal
3. ✅ Run ESPRIT directly on full signal
4. ❌ **Do NOT** add exponential windowing
5. ✅ **Keep** existing Hann fade enabled

This approach:
- Preserves accurate modal damping estimates
- Avoids duplicate windowing
- Respects the preprocessing already done by RoomResponse
- Follows best practices in experimental modal analysis
