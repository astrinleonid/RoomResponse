# Phase 4 Implementation Plan - Move Calibration Logic from GUI

**Document Version:** 1.0
**Date:** 2025-10-31
**Status:** 🚧 PLANNING - Ready for Implementation
**Prerequisites:** Phase 2 (calibration mode) and Phase 3 (helpers) completed ✅

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Current State Analysis](#2-current-state-analysis)
3. [Duplication Analysis](#3-duplication-analysis)
4. [Proposed Solution](#4-proposed-solution)
5. [Implementation Steps](#5-implementation-steps)
6. [Testing Strategy](#6-testing-strategy)
7. [Risk Mitigation](#7-risk-mitigation)

---

## 1. Executive Summary

### Objective

Replace GUI's calibration test logic with calls to recorder's `take_record_calibration()` method to eliminate ~100 lines of duplicate code.

### Key Principle

**USE EXISTING RECORDER METHODS** - No duplication between GUI and recorder.

### Benefits

1. **Eliminated Duplication:** ~100 lines of duplicate calibration logic removed from GUI
2. **Single Source of Truth:** Calibration logic only in recorder
3. **Easier Maintenance:** Bug fixes in one place
4. **Consistent Behavior:** GUI and CLI use same code path

### Timeline

- **Planning:** 1 hour (this document)
- **Implementation:** 2-3 hours
- **Testing:** 1-2 hours
- **Total:** 4-6 hours

---

## 2. Current State Analysis

### Current GUI Implementation

**File:** `gui_audio_settings_panel.py`
**Method:** `_perform_calibration_test()` (lines 1202-1346, ~144 lines)

**What it does:**
1. Validates device capabilities (lines 1216-1240)
2. Records audio via `recorder._record_method_2()` (line 1243)
3. Extracts calibration channel (lines 1248-1261)
4. Pads/trims audio (lines 1264-1272)
5. Reshapes into cycles (line 1275)
6. Validates each cycle with CalibrationValidatorV2 (lines 1278-1296)
7. Aligns cycles by onset (lines 1298-1308)
8. Applies alignment to all channels (lines 1316-1328)
9. Returns results (lines 1330-1346)

### Recorder Implementation

**File:** `RoomResponseRecorder.py`
**Method:** `_take_record_calibration_mode()` (lines 1152-1276, ~125 lines)

**What it does:**
1. Validates calibration setup (lines 1176-1182)
2. Records audio via `_record_method_2()` (line 1186)
3. Extracts calibration channel (lines 1193-1195)
4. Extracts cycles via helper (line 1211)
5. Validates each cycle with CalibrationValidatorV2 (lines 1214-1228)
6. Aligns cycles by onset (lines 1234-1240)
7. Applies alignment to all channels (lines 1243-1249)
8. Returns results (lines 1255-1270)

### Duplication Percentage

**~95% duplicate** - Same logic, slightly different return format.

---

## 3. Duplication Analysis

| Operation | GUI Code | Recorder Code | Overlap |
|-----------|----------|---------------|---------|
| **Record audio** | `_record_method_2()` | `_record_method_2()` | ✅ 100% |
| **Extract cal channel** | Lines 1248-1261 | Lines 1193-1195 | ✅ 100% |
| **Pad/trim** | Lines 1264-1272 | Via `_extract_cycles()` | ✅ 100% |
| **Reshape** | Line 1275 | Via `_extract_cycles()` | ✅ 100% |
| **Validate cycles** | Lines 1278-1296 | Lines 1214-1228 | ✅ 100% |
| **Align cycles** | Lines 1298-1308 | Lines 1234-1240 | ✅ 100% |
| **Apply to all channels** | Lines 1316-1328 | Lines 1243-1249 | ✅ 100% |
| **Device validation** | Lines 1216-1240 | ❌ Not in recorder | GUI-specific |
| **Return format** | Dict with specific keys | Dict with specific keys | ⚠️ Slightly different |

**Duplication:** ~95% of logic is identical

---

## 4. Proposed Solution

### New Architecture

**GUI should call recorder's calibration mode:**

```python
def _perform_calibration_test(self) -> Dict:
    """
    Perform a calibration test using recorder's calibration mode.

    Returns:
        Dictionary with test results for GUI display
    """
    # Device validation (GUI-specific)
    self._validate_device_capabilities()

    # Use recorder's calibration mode (eliminates duplication)
    result = self.recorder.take_record_calibration()

    # Transform result for GUI compatibility (if needed)
    return self._format_calibration_result_for_gui(result)
```

### Benefits

1. **Eliminates ~100 lines** of duplicate code from GUI
2. **Single source of truth** for calibration logic
3. **Easier testing** - test recorder once, GUI just uses it
4. **Bug fixes propagate** - fix in recorder, GUI gets it automatically

---

## 5. Implementation Steps

### Step 1: Extract Device Validation

**Create new helper method in GUI:**

```python
def _validate_device_capabilities(self):
    """
    Validate that the selected device supports the configured number of channels.

    Raises:
        ValueError: If device doesn't support required channels
    """
    num_channels = self.recorder.multichannel_config.get('num_channels', 1)
    try:
        devices_info = self.recorder.get_device_info_with_channels()
        current_device_id = int(getattr(self.recorder, 'input_device', -1))

        if current_device_id == -1:
            max_device_channels = max((d['max_channels'] for d in devices_info['input_devices']), default=1)
        else:
            max_device_channels = 1
            for dev in devices_info['input_devices']:
                if dev['device_id'] == current_device_id:
                    max_device_channels = dev['max_channels']
                    break

        if num_channels > max_device_channels:
            raise ValueError(
                f"Device capability mismatch: Your input device only supports {max_device_channels} channels, "
                f"but multi-channel configuration is set to {num_channels} channels. "
                f"Please reduce the number of channels in Device Selection tab."
            )
    except Exception as e:
        if "capability mismatch" in str(e):
            raise
        # Continue if we can't check (device info might not be available)
        pass
```

### Step 2: Create Result Formatter (if needed)

**Check if GUI needs different format:**

Current GUI returns:
```python
{
    'success': True,
    'num_cycles': int,
    'calibration_channel': int,
    'sample_rate': int,
    'all_calibration_cycles': np.ndarray,       # Initial cycles
    'validation_results': List[Dict],
    'alignment_metadata': Dict,
    'aligned_cycles': np.ndarray,                # Calibration channel aligned
    'aligned_multichannel_cycles': Dict         # All channels aligned
}
```

Recorder returns:
```python
{
    'calibration_cycles': np.ndarray,           # Initial cycles
    'validation_results': List[Dict],
    'aligned_multichannel_cycles': Dict,
    'alignment_metadata': Dict,
    'num_valid_cycles': int,
    'num_aligned_cycles': int,
    'metadata': {
        'mode': 'calibration',
        'calibration_channel': int,
        'num_channels': int,
        'num_cycles': int,
        'cycle_samples': int,
        'correlation_threshold': float
    }
}
```

**Create formatter:**

```python
def _format_calibration_result_for_gui(self, recorder_result: Dict) -> Dict:
    """
    Format recorder's calibration result for GUI compatibility.

    Args:
        recorder_result: Result from recorder.take_record_calibration()

    Returns:
        Dict formatted for GUI display
    """
    cal_ch = recorder_result['metadata']['calibration_channel']

    return {
        'success': True,
        'num_cycles': recorder_result['metadata']['num_cycles'],
        'calibration_channel': cal_ch,
        'sample_rate': self.recorder.sample_rate,
        # Map recorder keys to GUI keys
        'all_calibration_cycles': recorder_result['calibration_cycles'],
        'validation_results': recorder_result['validation_results'],
        'alignment_metadata': recorder_result['alignment_metadata'],
        'aligned_cycles': recorder_result['aligned_multichannel_cycles'].get(cal_ch),
        'aligned_multichannel_cycles': recorder_result['aligned_multichannel_cycles']
    }
```

### Step 3: Replace _perform_calibration_test()

**New implementation:**

```python
def _perform_calibration_test(self) -> Dict:
    """
    Perform a calibration test using recorder's calibration mode.

    This method validates device capabilities, then delegates to the recorder's
    calibration mode implementation to avoid duplication.

    Returns:
        Dictionary with test results including:
        - all_calibration_cycles: Raw waveforms for each cycle
        - validation_results: Quality metrics for each cycle
        - aligned_cycles: Aligned calibration channel cycles
        - aligned_multichannel_cycles: Aligned cycles for all channels
        - sample_rate: Sample rate for waveform playback
    """
    # Validate device capabilities (GUI-specific check)
    self._validate_device_capabilities()

    # Use recorder's calibration mode (eliminates duplication)
    try:
        recorder_result = self.recorder.take_record_calibration()
    except Exception as e:
        # Re-raise with user-friendly message
        if "no data captured" in str(e).lower():
            raise ValueError("Recording failed - no data captured. Check your audio device connections.")
        raise

    # Format result for GUI compatibility
    return self._format_calibration_result_for_gui(recorder_result)
```

### Lines of Code Impact

| Metric | Before | After | Change |
|--------|--------|-------|--------|
| **_perform_calibration_test()** | 144 lines | ~20 lines | -124 lines (-86%) |
| **New helpers** | 0 | ~50 lines | +50 lines |
| **Net change** | 144 lines | ~70 lines | **-74 lines (-51%)** |

---

## 6. Testing Strategy

### Level 1: Unit Tests

**Test device validation:**
```python
def test_device_validation_passes():
    """Verify device validation accepts compatible devices"""

def test_device_validation_fails():
    """Verify device validation rejects incompatible devices"""
```

**Test result formatter:**
```python
def test_result_formatter():
    """Verify formatter correctly transforms recorder result"""
    mock_recorder_result = {...}
    gui_result = panel._format_calibration_result_for_gui(mock_recorder_result)

    assert 'all_calibration_cycles' in gui_result
    assert 'validation_results' in gui_result
    assert gui_result['success'] == True
```

### Level 2: Integration Tests

**Test full calibration flow:**
```python
def test_calibration_test_integration():
    """Verify GUI calibration test uses recorder's method"""
    panel = AudioSettingsPanel(...)

    result = panel._perform_calibration_test()

    assert result['success'] == True
    assert 'all_calibration_cycles' in result
    assert 'validation_results' in result
```

### Level 3: GUI Tests

**Manual testing checklist:**
- [ ] Open GUI → Audio Settings
- [ ] Click "Run Calibration Test"
- [ ] Verify cycles displayed correctly
- [ ] Verify validation results shown
- [ ] Verify aligned cycles displayed
- [ ] Check multi-channel visualization
- [ ] Compare output with pre-refactor version

---

## 7. Risk Mitigation

### Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| GUI display breaks | MEDIUM | HIGH | Careful result formatting |
| Different behavior | LOW | MEDIUM | Recorder already tested |
| Missing GUI-specific logic | MEDIUM | HIGH | Extract device validation first |

### Mitigation Strategies

#### Strategy 1: Preserve GUI-Specific Logic

**What:** Device validation is GUI-specific, keep it separate

**How:**
1. Extract to `_validate_device_capabilities()`
2. Call before recorder method
3. Don't move to recorder

#### Strategy 2: Careful Result Mapping

**What:** Ensure GUI gets data in expected format

**How:**
1. Create explicit formatter method
2. Map all required keys
3. Test with existing GUI code

#### Strategy 3: Incremental Testing

**What:** Test each step independently

**How:**
1. Test device validation alone
2. Test result formatter alone
3. Test full integration
4. Test GUI display

---

## 8. Success Criteria

### Functional Success

- ✅ GUI calibration test works identically to before
- ✅ All cycles displayed correctly
- ✅ Validation results shown correctly
- ✅ Multi-channel visualization works
- ✅ No user-visible changes

### Code Quality Success

- ✅ ~100 lines of duplicate code removed
- ✅ Single source of truth for calibration
- ✅ Clearer separation GUI/business logic
- ✅ Easier to maintain

---

## 9. Implementation Checklist

### Pre-Implementation
- [ ] Review this plan
- [ ] Create backup of gui_audio_settings_panel.py
- [ ] Create implementation branch
- [ ] Run baseline GUI test

### Implementation
- [ ] Step 1: Extract `_validate_device_capabilities()`
- [ ] Step 2: Create `_format_calibration_result_for_gui()`
- [ ] Step 3: Replace `_perform_calibration_test()`

### Testing
- [ ] Level 1: Unit tests for helpers
- [ ] Level 2: Integration test
- [ ] Level 3: Manual GUI test

### Finalization
- [ ] All tests passed
- [ ] Commit changes
- [ ] Merge to dev

---

**Status:** 🟢 **READY FOR IMPLEMENTATION**

**Estimated Duration:** 4-6 hours

**Next Step:** Begin implementation with Step 1
