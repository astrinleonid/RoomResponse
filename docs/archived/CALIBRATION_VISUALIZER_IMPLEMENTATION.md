# Calibration Impulse Visualizer Implementation

**Date:** 2025-10-28
**Status:** ✅ COMPLETE
**Branch:** `feature/calibration-impulse-visualizer`

---

## Overview

This document describes the implementation of the Calibration Impulse Visualizer, a tool integrated into the Audio Settings panel that enables users to explore recorded calibration impulses visually, assess quality criteria interactively, and adjust validation thresholds based on actual data.

### Problem Statement

Previously, the calibration test would:
- Record impulses and immediately filter out invalid ones
- Only show a summary table of validation results
- Not provide visual feedback on waveform characteristics
- Require users to adjust quality thresholds blindly without seeing the actual data

### Solution

The new implementation:
- **Records ALL impulses** regardless of quality
- **Extracts and stores raw waveforms** for each cycle
- **Provides interactive visualization** using AudioVisualizer component
- **Enables iterative workflow:** record → visualize → assess → adjust → re-test

---

## Architecture

### Data Flow

```
User clicks "Run Calibration Test"
    ↓
_perform_calibration_test()
    ├─> Record multi-channel audio (_record_method_2)
    ├─> Extract calibration channel raw data
    ├─> Reshape into individual cycles (num_pulses x cycle_samples)
    ├─> Run quality validation on each cycle (CalibrationValidator)
    ├─> Store ALL cycles and metrics (no filtering)
    └─> Return results dict with:
        - all_calibration_cycles: np.ndarray (num_pulses, cycle_samples)
        - validation_results: List[Dict] with per-cycle metrics
        - sample_rate, num_cycles, calibration_channel
    ↓
_render_calibration_test_results(results)
    ├─> Display summary metrics
    ├─> Show quality metrics table (all cycles)
    ├─> Per-cycle visualization:
    │   ├─> Cycle selector dropdown
    │   ├─> Quality metrics display
    │   ├─> AudioVisualizer for waveform (with zoom, playback, analysis)
    │   └─> Valid/Invalid status with failure reasons
    ├─> Multi-cycle overlay comparison:
    │   ├─> Multi-select cycles
    │   ├─> AudioVisualizer.render_overlay_plot()
    │   └─> Normalized waveform overlay
    └─> User guidance with workflow tips
```

---

## Implementation Details

### 1. Modified `_perform_calibration_test()`

**File:** [gui_audio_settings_panel.py](gui_audio_settings_panel.py:917-1022)

**Changes:**
- No longer calls `_process_recorded_signal()` which would filter cycles
- Directly extracts calibration channel from recorded audio
- Reshapes raw data into individual cycles
- Runs validation but stores results without filtering
- Returns complete dataset for visualization

**Return Structure:**
```python
{
    'success': True,
    'num_cycles': 8,
    'calibration_channel': 0,
    'sample_rate': 48000,
    'all_calibration_cycles': np.ndarray,  # Shape: (8, 4800)
    'validation_results': [                 # One dict per cycle
        {
            'cycle_index': 0,
            'calibration_valid': True,
            'calibration_metrics': {
                'peak_amplitude': 0.5,
                'duration_ms': 5.2,
                'secondary_peak_ratio': 0.15,
                'tail_rms_ratio': 0.08
            },
            'calibration_failures': []
        },
        # ... 7 more cycles
    ],
    'cycle_duration_s': 0.1
}
```

### 2. Rewrote `_render_calibration_test_results()`

**File:** [gui_audio_settings_panel.py](gui_audio_settings_panel.py:1033-1226)

**UI Components:**

#### A. Summary Section
- **Metrics:** Total cycles, valid count, calibration channel
- **Table:** Per-cycle quality metrics (Peak Amp, Duration, Secondary Peak, Tail RMS, Issues)
- Uses pandas DataFrame for clean display

#### B. Per-Cycle Visualization
- **Cycle Selector:** Dropdown with format "Cycle N ✓ Valid / ✗ Invalid"
- **Quality Status:** Color-coded valid/invalid display with failure reasons
- **Metrics Display:** 4-column layout with key metrics
- **AudioVisualizer Integration:**
  - Unique component ID per cycle: `f"cal_cycle_{selected_cycle}"`
  - Waveform, spectrum, spectrogram views
  - Zoom, playback, and analysis features
  - Export functionality

#### C. Multi-Cycle Overlay Comparison
- **Multi-select:** Choose multiple cycles to compare
- **Overlay Plot:** Uses `AudioVisualizer.render_overlay_plot()` static method
  - Normalized waveforms for easy comparison
  - Color-coded with labels showing valid/invalid status
  - Interactive legend
  - Helps identify outliers and consistency issues

#### D. User Guidance
- Expandable section with:
  - Workflow steps
  - Quality criteria explanations
  - Tips for threshold adjustment

---

## Usage Workflow

### Step 1: Configure Multi-Channel Settings
1. Go to Audio Settings → Device Selection tab
2. Enable multi-channel recording
3. Set number of channels (e.g., 4)
4. Select calibration channel (e.g., Ch 0 for accelerometer)
5. Save configuration

### Step 2: Set Initial Quality Parameters
1. Go to Audio Settings → Calibration Impulse tab
2. Expand "Quality Parameter Settings"
3. Set initial thresholds (use defaults or adjust)
4. Save quality parameters

### Step 3: Run Calibration Test
1. Click "Run Calibration Test"
2. System records calibration impulses (e.g., 8 cycles)
3. All cycles recorded regardless of quality

### Step 4: Review Results
1. Check summary table to see pass/fail counts
2. Use cycle selector to examine individual waveforms
3. Look for patterns:
   - Are failures consistent (e.g., all weak)?
   - Are there outliers?
   - Is the threshold too strict/loose?

### Step 5: Compare Cycles
1. Select multiple cycles in multi-select
2. View overlay plot to compare waveforms
3. Identify consistency issues or outliers

### Step 6: Adjust Thresholds
1. Based on visual inspection, adjust quality parameters
2. Example: If good impulses fail amplitude check, lower `cal_min_amplitude`
3. Save new parameters

### Step 7: Re-test
1. Run another calibration test with new thresholds
2. Verify that valid impulses pass and invalid ones fail
3. Iterate until thresholds match your quality criteria

---

## Test Coverage

**Test File:** [test_calibration_visualizer.py](test_calibration_visualizer.py)

### Test 1: Calibration Data Extraction
- ✅ Create synthetic multi-channel recording
- ✅ Extract calibration channel
- ✅ Reshape into individual cycles (8 x 4800 samples)
- ✅ Run quality validation
- ✅ Verify data structure matches UI expectations
- ✅ Test individual cycle extraction
- ✅ Test multi-cycle overlay data preparation

### Test 2: Summary Table Generation
- ✅ Generate pandas DataFrame from validation results
- ✅ Format metrics for display
- ✅ Handle valid/invalid status

### Test Results
```
============================================================
TEST PASSED: All data extraction working correctly!
============================================================

Validation Summary: 5/8 cycles valid
- Cycle 0: ✓ VALID (good impulse)
- Cycle 1: ✗ INVALID (weak amplitude, tail noise)
- Cycle 2: ✓ VALID (good impulse)
- Cycle 3: ✗ INVALID (clipped, tail noise)
- Cycle 4: ✓ VALID (good impulse)
- Cycle 5: ✗ INVALID (long duration, tail noise)
- Cycle 6: ✓ VALID (good impulse)
- Cycle 7: ✓ VALID (good impulse)
```

---

## Technical Notes

### AudioVisualizer Integration

**Component ID Management:**
- Each cycle visualization uses unique ID: `f"cal_cycle_{selected_cycle}"`
- Prevents session state conflicts when switching between cycles
- Ensures proper audio data updates

**Static Methods Used:**
- `AudioVisualizer.render_overlay_plot()` for multi-cycle comparison
- Accepts list of signals, labels, and visualization parameters
- Returns matplotlib figure for display

### Session State Management
- Test results stored in `st.session_state['cal_test_results']`
- Persists across Streamlit reruns
- Cleared when user clicks "Clear Results"

### Performance Considerations
- Cycles stored as numpy arrays (efficient memory usage)
- Only selected cycle visualized at a time (not all 8 simultaneously)
- Matplotlib figures properly closed after rendering (`plt.close(fig)`)

---

## Dependencies

### Required Modules
- `numpy` - Array operations and waveform storage
- `streamlit` - UI framework
- `pandas` - Summary table display
- `matplotlib` - Overlay plot rendering
- `gui_audio_visualizer.py` - Waveform visualization component
- `calibration_validator.py` - Quality validation logic
- `RoomResponseRecorder.py` - Recording and processing

### Import Pattern
```python
# Audio visualizer (modular)
try:
    from gui_audio_visualizer import AudioVisualizer
    AUDIO_VISUALIZER_AVAILABLE = True
except ImportError:
    AudioVisualizer = None
    AUDIO_VISUALIZER_AVAILABLE = False
```

Graceful degradation if AudioVisualizer not available.

---

## Future Enhancements

### Potential Improvements
1. **Export Feature:**
   - Save all cycles as individual WAV files
   - Export validation report as JSON or CSV

2. **Statistics View:**
   - Show distribution of quality metrics across all cycles
   - Histogram of peak amplitudes, durations, etc.

3. **Threshold Suggestions:**
   - Automatically suggest thresholds based on recorded data
   - Calculate percentiles for each metric

4. **Real-time Feedback:**
   - Show quality validation while recording
   - Live waveform display during test

5. **Comparison Across Tests:**
   - Store multiple test sessions
   - Compare quality metrics over time

---

## Files Modified

1. **[gui_audio_settings_panel.py](gui_audio_settings_panel.py)**
   - Added `import numpy as np`
   - Added AudioVisualizer import with availability check
   - Modified `_perform_calibration_test()` (lines 917-1022)
   - Completely rewrote `_render_calibration_test_results()` (lines 1033-1226)

2. **[test_calibration_visualizer.py](test_calibration_visualizer.py)** (NEW)
   - Comprehensive test suite for calibration workflow
   - Synthetic data generation with varying quality
   - Validation testing
   - UI simulation

3. **[CALIBRATION_VISUALIZER_IMPLEMENTATION.md](CALIBRATION_VISUALIZER_IMPLEMENTATION.md)** (NEW)
   - This document

---

## Commits

### Commit 1: `531c7ae`
```
feat: Add AudioVisualizer integration to Calibration Impulse test

Enhanced the Calibration Impulse section in the Audio Settings panel to provide
comprehensive visualization and analysis tools for calibration impulse quality assessment.
```

### Commit 2: `290df10`
```
test: Add comprehensive test suite for calibration visualizer

Created test_calibration_visualizer.py to verify the calibration impulse
visualization workflow without launching the full GUI.
```

---

## Integration with Multi-Channel Plan

**Relates to:** [PIANO_MULTICHANNEL_PLAN.md](PIANO_MULTICHANNEL_PLAN.md) - Phase 4: GUI Updates

**Phase 4 Task:** Update Calibration Impulse section in Audio Settings panel ✅

**Next Phase 4 Tasks:**
- Multi-channel configuration UI (partially complete)
- Collection panel multi-channel status display
- Audio Analysis panel multi-channel visualization

---

## Conclusion

The Calibration Impulse Visualizer successfully implements the requested feature:
- ✅ Records all impulses regardless of quality
- ✅ Provides interactive visualization using AudioVisualizer
- ✅ Enables user exploration and quality assessment
- ✅ Supports iterative threshold adjustment workflow
- ✅ Comprehensive test coverage

This tool empowers users to make informed decisions about calibration quality thresholds based on actual recorded data rather than arbitrary defaults.

**Status:** Ready for integration and user testing.
