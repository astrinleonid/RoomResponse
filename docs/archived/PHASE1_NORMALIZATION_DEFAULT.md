# Normalization Default and UI Control

**Date:** 2025-11-02
**Enhancement:** Make normalization enabled by default in calibration mode with UI toggle
**Status:** ✅ IMPLEMENTED

---

## Overview

Added a checkbox control in the Series Settings panel to enable/disable normalization in calibration mode, with normalization **enabled by default**. This makes the normalization feature more discoverable and user-friendly.

---

## Changes Made

### 1. Config File Update

**File:** `recorderConfig.json`

**Added:**
```json
"multichannel_config": {
  ...
  "normalize_by_calibration": true  // ← NEW: Default to enabled
}
```

**Impact:** Normalization is now enabled by default for all users

---

### 2. UI Control Addition

**File:** `gui_series_settings_panel.py` (Lines 255-279)

**Added checkbox control:**

```python
# Normalization toggle - default to True for calibration mode
current_normalize = mc_config.get('normalize_by_calibration', True)
normalize_enabled = st.checkbox(
    "Enable Normalization",
    value=current_normalize,
    key="series_normalize_by_calibration",
    help="Divide response amplitudes by impact magnitude for quantitative comparison"
)

# Update recorder config if changed
if normalize_enabled != current_normalize:
    self.recorder.multichannel_config['normalize_by_calibration'] = normalize_enabled
    if normalize_enabled:
        st.success("✅ Normalization enabled - responses will be normalized by impact magnitude")
    else:
        st.info("ℹ️ Normalization disabled - responses will show raw aligned amplitudes")

# Show current status
if normalize_enabled:
    st.caption("✅ Responses normalized by impact magnitude")
else:
    st.caption("⚠️ Raw aligned responses (not normalized)")
```

---

## UI Changes

### Before

```
🔨 Calibration Mode Configuration

[Sensor Setup]              [Processing Options]
🔨 Calibration: Ch 2        ✅ Normalization: Enabled
   Channel Name                (or ⚠️ Disabled)
🎤 Reference: Ch 5          [Static display only]
   Channel Name
```

### After

```
🔨 Calibration Mode Configuration

[Sensor Setup]              [Processing Options]
🔨 Calibration: Ch 2        ☑ Enable Normalization  [Checkbox - Interactive!]
   Channel Name             ✅ Responses normalized by impact magnitude
🎤 Reference: Ch 5             (or ⚠️ Raw aligned responses)
   Channel Name
```

---

## User Workflow

### New User Experience

1. Configure multi-channel with calibration sensor
2. Open Series Settings → Pulse Series Config
3. Select "Calibration (Physical Impact)" mode
4. Calibration Mode Configuration expander shows:
   - **Checkbox "Enable Normalization" is CHECKED by default** ✅
   - Help tooltip explains what normalization does
   - Status caption shows "Responses normalized by impact magnitude"
5. User can **uncheck to disable** if they want raw responses
6. Setting is **immediately applied** to recorder config
7. When recording, normalization is used based on checkbox state

### For Existing Users

- If config already has `normalize_by_calibration: false`:
  - Checkbox starts unchecked (respects existing config)
  - Can enable by checking the box
- If config is missing the field:
  - Defaults to `true` (enabled)
  - Checkbox starts checked

---

## Benefits

### ✅ Discoverability
- Users can see the normalization option directly
- Don't need to navigate to Device Selection tab
- Checkbox is in the context of calibration mode

### ✅ Immediate Feedback
- Success message when enabling/disabling
- Status caption always shows current state
- Help tooltip explains what normalization does

### ✅ Sensible Default
- Normalization enabled by default makes sense for calibration mode
- Primary use case (comparing responses) requires normalization
- Advanced users can disable if needed

### ✅ Flexibility
- Users can toggle on/off without editing config file
- Changes apply immediately to recorder
- Can experiment with both modes easily

---

## Default Behavior Rationale

### Why Default to Enabled?

**Calibration mode exists for quantitative comparison:**
- Varying impact strengths pollute data
- Normalization removes force variation
- Enables apple-to-apple comparison

**Most users want normalized data:**
- Research applications need normalized responses
- Piano hammer studies compare response characteristics
- Sensor calibration requires magnitude compensation

**Disabling is the exception:**
- Some users may want to see raw impact magnitude
- Useful for debugging sensor issues
- Can compare normalized vs un-normalized

**Better UX:**
- Users get the "right" behavior by default
- Don't have to hunt for the setting
- Can disable if needed (advanced use case)

---

## Technical Details

### Default Value Logic

```python
current_normalize = mc_config.get('normalize_by_calibration', True)
#                                                            ^^^^
#                                           Default to True if missing
```

**Behavior:**
- If config has the key: Use its value (respects user choice)
- If config missing key: Default to `True` (sensible default)
- Checkbox value = current_normalize on first render

### State Management

```python
if normalize_enabled != current_normalize:
    self.recorder.multichannel_config['normalize_by_calibration'] = normalize_enabled
```

**When user toggles checkbox:**
1. Detects value changed
2. Updates recorder's multichannel_config immediately
3. Shows confirmation message
4. Next recording will use new setting

**Not saved to file automatically:**
- Changes only affect recorder's in-memory config
- User must click "Save Configuration" to persist
- This is consistent with other settings in Series Settings

---

## Testing

### Test Case 1: Fresh Install (No Config)
**Setup:** No `normalize_by_calibration` in config

**Steps:**
1. Open Series Settings → Calibration Mode
2. Check normalization checkbox state

**Expected:**
- ✅ Checkbox is CHECKED
- ✅ Caption shows "Responses normalized by impact magnitude"

### Test Case 2: Existing Config (Enabled)
**Setup:** Config has `"normalize_by_calibration": true`

**Steps:**
1. Open Series Settings → Calibration Mode
2. Check normalization checkbox state

**Expected:**
- ✅ Checkbox is CHECKED
- ✅ Respects existing config

### Test Case 3: Existing Config (Disabled)
**Setup:** Config has `"normalize_by_calibration": false`

**Steps:**
1. Open Series Settings → Calibration Mode
2. Check normalization checkbox state

**Expected:**
- ✅ Checkbox is UNCHECKED
- ✅ Respects existing config
- ✅ Caption shows "Raw aligned responses (not normalized)"

### Test Case 4: Toggle Enable
**Setup:** Start with disabled

**Steps:**
1. Check the normalization checkbox
2. Observe UI feedback

**Expected:**
- ✅ Success message appears
- ✅ Caption updates to "Responses normalized"
- ✅ recorder.multichannel_config updated
- ✅ Next recording uses normalization

### Test Case 5: Toggle Disable
**Setup:** Start with enabled

**Steps:**
1. Uncheck the normalization checkbox
2. Observe UI feedback

**Expected:**
- ✅ Info message appears
- ✅ Caption updates to "Raw aligned responses"
- ✅ recorder.multichannel_config updated
- ✅ Next recording doesn't use normalization

### Test Case 6: Record with Normalization Enabled
**Setup:** Checkbox checked

**Steps:**
1. Record calibration series
2. Check console output
3. Check recorded data

**Expected:**
```
Console:
  Normalization: Enabled (dividing by impact magnitude)
  Impact magnitudes: min=X.XX, max=X.XX, mean=X.XX

GUI:
  Normalization: ✅ Enabled
  [Response amplitudes should be consistent across cycles]
```

### Test Case 7: Record with Normalization Disabled
**Setup:** Checkbox unchecked

**Steps:**
1. Record calibration series
2. Check console output
3. Check recorded data

**Expected:**
```
Console:
  Normalization: Disabled

GUI:
  Normalization: ❌ Disabled
  [Response amplitudes vary with impact strength]
```

---

## Code Statistics

### Files Modified
- `recorderConfig.json` - 1 line added
- `gui_series_settings_panel.py` - 24 lines modified (lines 255-279)

### Lines Changed
- **Config:** +1 line
- **GUI:** +24 lines (replaced 8 lines)
- **Net:** +17 lines

### Breaking Changes
- **None** - Fully backward compatible
- Existing configs with explicit values are respected
- Missing values default to enabled (better behavior)

---

## User Documentation

### How to Use

**To use normalized responses (default):**
1. Select Calibration mode
2. Ensure "Enable Normalization" checkbox is CHECKED ✅
3. Record series
4. Responses will be divided by impact magnitude

**To use raw aligned responses:**
1. Select Calibration mode
2. UNCHECK "Enable Normalization" checkbox
3. Record series
4. Responses will show raw amplitudes (varying with impact strength)

**To persist your choice:**
1. Change normalization setting as desired
2. Click "Save Configuration" button
3. Setting will be preserved across sessions

---

## Visual Example

### UI Layout

```
┌─────────────────────────────────────────────────────────────┐
│ 🔨 Calibration Mode Configuration                [Expanded] │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  [Sensor Setup]                [Processing Options]         │
│  ┌─────────────────────────┐  ┌──────────────────────────┐ │
│  │ 🔨 Calibration Sensor:  │  │ ☑ Enable Normalization   │ │
│  │    Ch 2 - Hammer        │  │                           │ │
│  │                          │  │ [?] Divide response       │ │
│  │ 🎤 Reference Channel:   │  │     amplitudes by impact  │ │
│  │    Ch 5 - Front Mic     │  │     magnitude             │ │
│  └─────────────────────────┘  │                           │ │
│                                │ ✅ Responses normalized   │ │
│                                │    by impact magnitude    │ │
│                                └──────────────────────────┘ │
│                                                              │
│  [Quality Validation]                                        │
│  ┌────────────────────────────────────────────────────────┐ │
│  │ Negative peak range: 0.30 - 0.50                       │ │
│  │ Correlation threshold: 0.70                             │ │
│  └────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

---

## Related Documentation

- [PHASE1_NORMALIZATION_FIX.md](PHASE1_NORMALIZATION_FIX.md) - Critical fix for normalization not being applied
- [PHASE1_IMPLEMENTATION_SUMMARY.md](PHASE1_IMPLEMENTATION_SUMMARY.md) - Overall Phase 1 summary
- [CALIBRATION_MODE_INTEGRATION_PLAN.md](CALIBRATION_MODE_INTEGRATION_PLAN.md) - Master plan

---

## Conclusion

This enhancement makes normalization:
1. **Enabled by default** (sensible for most use cases)
2. **Easy to toggle** (checkbox in calibration config)
3. **Well-documented** (help tooltip and status caption)
4. **Immediately effective** (updates recorder config on change)

Users no longer need to:
- Navigate to Device Selection tab
- Edit config file manually
- Wonder if normalization is enabled

The feature is now **discoverable, understandable, and controllable** directly in the context where it's used.

**Status:** ✅ IMPLEMENTED
**User Impact:** Much improved discoverability and usability
**Breaking Changes:** None
