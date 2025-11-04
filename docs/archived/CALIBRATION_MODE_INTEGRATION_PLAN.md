# Calibration Mode Integration Plan - Audio Settings & Recording Functionality

**Document Version:** 1.0
**Created:** 2025-11-02
**Status:** 📋 Planning Phase
**Target:** Integrate Calibration Mode into Audio Settings/Series Settings panels and Recording & Analysis workflow

---

## Executive Summary

This plan details the integration of Calibration Mode functionality into the Audio Settings panel (Series Settings tab) and the Recording & Analysis workflow. The goal is to enable users to easily switch between Standard and Calibration recording modes through the GUI interface, with proper configuration controls and workflow integration.

**Key Insight:** The backend (`RoomResponseRecorder.py`) **ALREADY SUPPORTS** calibration mode with file saving via the `save_files` parameter added in the recent pipeline refactoring. This plan focuses on **GUI/UX integration** rather than backend changes.

---

## Current State Analysis

### ✅ Backend Capabilities (Already Implemented)

1. **RoomResponseRecorder.take_record()** - Lines 1181-1260
   - ✅ Supports `mode='calibration'` parameter
   - ✅ Supports `save_files=True` parameter for calibration dataset collection
   - ✅ Universal three-stage pipeline (Record → Process → Save)
   - ✅ File saving logic unified for both standard and calibration modes
   - ✅ Backward compatible with existing code

2. **Calibration Mode Processing** - Already Implemented
   - ✅ Per-cycle quality validation (CalibrationValidatorV2)
   - ✅ Cycle alignment by onset detection
   - ✅ Optional normalization by calibration magnitude
   - ✅ Comprehensive metadata output

3. **File Management** - Already Implemented
   - ✅ Multi-channel file naming convention (`_chN` suffix)
   - ✅ Timestamp-based unique filenames
   - ✅ Metadata JSON export with validation results

### ❌ Missing GUI Integration

1. **Audio Settings Panel** - `gui_audio_settings_panel.py`
   - ❌ No recording mode selector (Standard vs Calibration)
   - ❌ Series Settings tab doesn't expose calibration mode
   - ❌ No UI controls for calibration-specific parameters
   - ⚠️ Calibration Impulse tab exists but is for **testing only** (no file saving)

2. **Series Settings Panel** - `gui_series_settings_panel.py`
   - ❌ No integration with calibration mode
   - ❌ Test recording always uses standard mode
   - ❌ No display of calibration-specific configuration

3. **Collection Panel** - `gui_collect_panel.py`
   - ✅ **Already has implementation plan** (see `CALIBRATION_MODE_COLLECT_PANEL_PLAN.md`)
   - 📋 Planned but not yet implemented

---

## Architecture Overview

### Current Structure

```
┌─────────────────────────────────────────────────────────────┐
│                     piano_response.py                        │
│                     (Main Entry Point)                       │
└─────────────────────────────────────────────────────────────┘
                              ↓
        ┌─────────────────────┼─────────────────────┐
        ↓                     ↓                     ↓
┌───────────────┐    ┌────────────────┐    ┌──────────────┐
│ AudioSettings │    │ CollectionPanel│    │ Scenarios    │
│    Panel      │    │                │    │   Panel      │
└───────────────┘    └────────────────┘    └──────────────┘
        ↓
  ┌─────┴──────┬──────────────┬─────────────┐
  │            │              │             │
Device     Calibration    Series       [Multi-Channel
Selection  Impulse        Settings      Configuration]
Tab        Tab (TEST)     Tab           Section
           ❌ No files    ❌ No cal mode ✅ Exists

        ↓                     ↓
┌────────────────────────────────────────────────┐
│         RoomResponseRecorder                   │
│  ✅ take_record(mode='standard'|'calibration') │
│  ✅ save_files parameter                       │
└────────────────────────────────────────────────┘
```

### Target Structure (After Implementation)

```
┌─────────────────────────────────────────────────────────────┐
│                     piano_response.py                        │
│                     (Main Entry Point)                       │
└─────────────────────────────────────────────────────────────┘
                              ↓
        ┌─────────────────────┼─────────────────────┐
        ↓                     ↓                     ↓
┌───────────────┐    ┌────────────────┐    ┌──────────────┐
│ AudioSettings │    │ CollectionPanel│    │ Scenarios    │
│    Panel      │    │  ✅ Mode Sel   │    │   Panel      │
└───────────────┘    └────────────────┘    └──────────────┘
        ↓
  ┌─────┴──────┬──────────────┬─────────────┐
  │            │              │             │
Device     Calibration    Series       [Multi-Channel
Selection  Impulse        Settings      Configuration]
Tab        Tab (TEST)     Tab           Section
           ❌ No files    ✅ Mode Sel   ✅ Exists
                          ✅ Cal params
                          ✅ Test with mode

        ↓                     ↓
┌────────────────────────────────────────────────┐
│         RoomResponseRecorder                   │
│  ✅ take_record(mode='standard'|'calibration') │
│  ✅ save_files parameter                       │
└────────────────────────────────────────────────┘
```

---

## Implementation Plan

### Phase 1: Series Settings Tab - Add Recording Mode Selection ⭐⭐⭐⭐⭐

**Goal:** Enable users to select Standard vs Calibration mode in Series Settings tab

**File:** `gui_series_settings_panel.py`

#### 1.1: Add Recording Mode Selector Component

**Location:** Add to `SeriesSettingsPanel` class

**New Method:**
```python
def _render_recording_mode_selection(self) -> str:
    """
    Render recording mode selection UI.

    Only shows Calibration option if multi-channel with calibration sensor configured.

    Returns:
        Selected mode: 'standard' or 'calibration'
    """
    st.markdown("### Recording Mode")

    # Check if calibration mode is available
    mc_config = getattr(self.recorder, 'multichannel_config', {})
    mc_enabled = mc_config.get('enabled', False)
    has_calibration = mc_config.get('calibration_channel') is not None

    if not mc_enabled or not has_calibration:
        st.info("ℹ️ **Standard Mode** (Room Response)")
        st.caption("Calibration mode requires multi-channel setup with calibration sensor. Configure in Device Selection tab.")
        return 'standard'

    # Get current mode from session state
    current_mode = st.session_state.get('series_recording_mode', 'standard')
    default_index = 0 if current_mode == 'standard' else 1

    mode_selection = st.radio(
        "Choose recording mode:",
        options=["Standard (Room Response)", "Calibration (Physical Impact)"],
        index=default_index,
        key="series_recording_mode_radio",
        help="""
        **Standard Mode:**
        - Record room acoustic responses using synthetic pulse train
        - Audio output from speaker, captured by microphones
        - Best for: Room impulse response measurements

        **Calibration Mode:**
        - Record physical impact responses (e.g., hammer strikes)
        - Requires calibration sensor (force/impact sensor)
        - Per-cycle quality validation
        - Automatic alignment and optional normalization
        - Best for: Piano hammer impact studies, sensor calibration
        """
    )

    selected_mode = 'calibration' if 'Calibration' in mode_selection else 'standard'
    st.session_state['series_recording_mode'] = selected_mode

    return selected_mode
```

**Effort:** 1 hour

---

#### 1.2: Display Mode-Specific Configuration Info

**New Method:**
```python
def _render_calibration_mode_info(self) -> None:
    """Display calibration mode configuration details when enabled."""
    current_mode = st.session_state.get('series_recording_mode', 'standard')

    if current_mode != 'calibration':
        return

    mc_config = getattr(self.recorder, 'multichannel_config', {})

    with st.expander("🔨 Calibration Mode Configuration", expanded=True):
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Sensor Setup**")
            cal_ch = mc_config.get('calibration_channel')
            channel_names = mc_config.get('channel_names', [])
            cal_name = channel_names[cal_ch] if cal_ch < len(channel_names) else f"Channel {cal_ch}"

            st.success(f"🔨 Calibration Sensor: Ch {cal_ch} - {cal_name}")

            ref_ch = mc_config.get('reference_channel', 0)
            ref_name = channel_names[ref_ch] if ref_ch < len(channel_names) else f"Channel {ref_ch}"
            st.info(f"🎤 Reference Channel: Ch {ref_ch} - {ref_name}")

        with col2:
            st.markdown("**Processing Options**")
            normalize_enabled = mc_config.get('normalize_by_calibration', False)
            if normalize_enabled:
                st.success("✅ Normalization: Enabled")
                st.caption("Responses normalized by impact magnitude")
            else:
                st.warning("⚠️ Normalization: Disabled")
                st.caption("Enable in Device Selection → Multi-Channel Configuration")

        # Display quality thresholds summary
        cal_config = getattr(self.recorder, 'calibration_quality_config', {})
        if cal_config:
            st.markdown("**Quality Validation**")
            with st.container():
                col_a, col_b = st.columns(2)
                with col_a:
                    st.caption(f"Negative peak range: {cal_config.get('min_negative_peak', 0):.2f} - {cal_config.get('max_negative_peak', 1):.2f}")
                with col_b:
                    st.caption(f"Correlation threshold: {mc_config.get('alignment_correlation_threshold', 0.7):.2f}")
```

**Effort:** 1 hour

---

#### 1.3: Update Test Recording to Use Selected Mode

**Modify:** Existing test recording button handler

**Current Code (approximately line 500-550):**
```python
if st.button("🎤 Test Recording and Analysis", ...):
    # ... existing code ...
    audio_data = self.recorder.take_record(
        str(raw_path),
        str(impulse_path)
    )
```

**Updated Code:**
```python
if st.button("🎤 Test Recording and Analysis", ...):
    # ... existing code ...

    # Get selected recording mode
    recording_mode = st.session_state.get('series_recording_mode', 'standard')

    # Display mode indicator
    if recording_mode == 'calibration':
        st.info("🔨 Testing with Calibration Mode (quality validation enabled)")

    # Call recorder with mode parameter
    audio_data = self.recorder.take_record(
        str(raw_path),
        str(impulse_path),
        mode=recording_mode,
        save_files=True  # Explicitly save files for testing
    )

    # Mode-specific result handling
    if recording_mode == 'calibration':
        # audio_data is a dict with calibration results
        if isinstance(audio_data, dict):
            st.success(f"✅ Calibration test completed")
            st.write(f"Valid cycles: {audio_data.get('num_valid_cycles', 0)}/{len(audio_data.get('validation_results', []))}")
            st.write(f"Aligned cycles: {audio_data.get('num_aligned_cycles', 0)}")

            # Show validation summary
            with st.expander("📊 Validation Results"):
                validation_results = audio_data.get('validation_results', [])
                for i, result in enumerate(validation_results):
                    status = "✅ Valid" if result['is_valid'] else "❌ Invalid"
                    st.write(f"Cycle {i}: {status}")
                    if not result['is_valid']:
                        st.caption(f"  Reasons: {', '.join(result.get('reasons', []))}")
```

**Effort:** 1.5 hours

---

#### 1.4: Integrate into Series Settings Render Method

**Modify:** `render()` method in `SeriesSettingsPanel`

**Add after configuration loading, before test recording section:**

```python
def render(self):
    st.header("Series Settings")

    # ... existing status and config loading ...

    # NEW: Recording mode selection
    recording_mode = self._render_recording_mode_selection()

    # NEW: Show calibration config if in calibration mode
    if recording_mode == 'calibration':
        self._render_calibration_mode_info()

    st.markdown("---")

    # ... rest of existing rendering ...
    # (configuration, test recording, etc.)
```

**Effort:** 30 minutes

---

### Phase 2: Audio Settings Panel - Mode Indicator in Recorder Status ⭐⭐⭐

**Goal:** Display current recording mode in the recorder status section

**File:** `gui_audio_settings_panel.py`

#### 2.1: Update Recorder Status Display

**Modify:** `_render_recorder_status()` method (line 221)

**Add after existing recorder info:**

```python
def _render_recorder_status(self):
    # ... existing recorder status code ...

    # NEW: Recording mode indicator
    if self.recorder:
        mc_config = getattr(self.recorder, 'multichannel_config', {})
        has_calibration = mc_config.get('calibration_channel') is not None

        if mc_config.get('enabled', False) and has_calibration:
            st.markdown("**Recording Modes Available:**")
            col_mode1, col_mode2 = st.columns(2)
            with col_mode1:
                st.success("✅ Standard (Room Response)")
            with col_mode2:
                st.success("✅ Calibration (Physical Impact)")
            st.caption("💡 Select mode in Series Settings tab or Collection Panel")
        else:
            st.markdown("**Recording Mode:**")
            st.info("📊 Standard (Room Response) - Only mode available")
            if not mc_config.get('enabled', False):
                st.caption("⚠️ Multi-channel disabled. Enable in Device Selection tab to use Calibration mode.")
            elif not has_calibration:
                st.caption("⚠️ No calibration sensor configured. Set calibration channel to use Calibration mode.")
```

**Effort:** 45 minutes

---

### Phase 3: Configuration Persistence ⭐⭐⭐

**Goal:** Save recording mode preference to config file

**File:** `gui_audio_settings_panel.py` and `gui_series_settings_panel.py`

#### 3.1: Add Recording Mode to Configuration

**Modify:** Configuration save/load methods

**In `gui_audio_settings_panel.py`:**

```python
def _save_config_to_file(self) -> bool:
    """Save current configuration to recorderConfig.json"""
    try:
        config = {
            # ... existing config fields ...

            # NEW: Add recording mode preferences
            "default_recording_mode": st.session_state.get('series_recording_mode', 'standard'),
        }

        # ... rest of save logic ...
    except Exception as e:
        st.error(f"Failed to save configuration: {e}")
        return False
```

**In `gui_series_settings_panel.py`:**

```python
def _load_configuration(self):
    """Load configuration including recording mode preference"""
    # ... existing load logic ...

    # NEW: Load default recording mode
    default_mode = config.get('default_recording_mode', 'standard')
    if 'series_recording_mode' not in st.session_state:
        st.session_state['series_recording_mode'] = default_mode
```

**Effort:** 1 hour

---

### Phase 4: Documentation & Help Text ⭐⭐

**Goal:** Add comprehensive help text and tooltips

#### 4.1: Add Mode Comparison Table

**New Component in Series Settings:**

```python
def _render_mode_comparison_table(self):
    """Display comparison table of Standard vs Calibration modes."""
    with st.expander("ℹ️ Recording Mode Comparison"):
        st.markdown("""
        | Feature | Standard Mode | Calibration Mode |
        |---------|---------------|------------------|
        | **Signal Source** | Synthetic pulse (speaker) | Physical impact (hammer) |
        | **Best For** | Room acoustics, reverb | Impact studies, piano research |
        | **Quality Validation** | ❌ No | ✅ Yes (per-cycle) |
        | **Cycle Alignment** | Basic (assumes perfect timing) | ✅ Advanced (onset detection) |
        | **Normalization** | ❌ No | ✅ Optional (by impact magnitude) |
        | **Calibration Sensor** | Not required | ✅ Required |
        | **File Output** | Raw + Impulse + Room Response | Aligned + Normalized + Metadata |
        | **Typical Use** | Room impulse responses | Piano hammer characterization |
        """)

        st.markdown("**When to use Calibration Mode:**")
        st.markdown("""
        - ✅ Recording physical impacts (hammer strikes, taps)
        - ✅ Need per-event quality validation
        - ✅ Varying impact magnitudes that need normalization
        - ✅ Precise alignment required for time-domain analysis

        **When to use Standard Mode:**
        - ✅ Synthetic audio signals (pulse trains)
        - ✅ Room acoustic measurements
        - ✅ Controlled signal source with consistent timing
        - ✅ Basic impulse response extraction
        """)
```

**Effort:** 45 minutes

---

### Phase 5: Integration Testing ⭐⭐⭐⭐

**Goal:** Comprehensive testing of all integration points

#### 5.1: Test Cases

**Test 1: Mode Selection in Series Settings**
- Open Audio Settings → Series Settings tab
- Verify mode selector appears when calibration sensor configured
- Select Calibration mode
- Verify calibration config info appears
- Run test recording
- Verify files saved with calibration format

**Test 2: Mode Persistence**
- Select Calibration mode
- Save configuration
- Restart application
- Verify Calibration mode is still selected
- Switch to Standard mode
- Verify mode changes correctly

**Test 3: Without Calibration Sensor**
- Disable multi-channel or remove calibration sensor config
- Open Series Settings
- Verify only Standard mode available
- Verify appropriate help text displayed

**Test 4: Test Recording in Both Modes**
- Run test recording in Standard mode
- Verify standard files created
- Switch to Calibration mode
- Run test recording
- Verify calibration files created with validation metadata

**Test 5: Recorder Status Display**
- Open Audio Settings panel
- Verify recorder status shows available modes correctly
- Change configuration (enable/disable calibration sensor)
- Verify status updates appropriately

**Test 6: Backward Compatibility**
- Test existing Calibration Impulse tab functionality
- Verify it still works (test only, no file saving)
- Ensure no conflicts with Series Settings calibration mode

**Effort:** 3-4 hours

---

## File Structure Changes

### Configuration File (recorderConfig.json)

**New Fields:**
```json
{
  "sample_rate": 48000,
  // ... existing fields ...

  "default_recording_mode": "standard",  // NEW: 'standard' or 'calibration'

  "multichannel_config": {
    "enabled": true,
    "calibration_channel": 0,
    // ... existing fields ...
  }
}
```

### Session State Keys

**New Keys Added:**
- `series_recording_mode` - Current recording mode ('standard' or 'calibration')
- `series_recording_mode_radio` - Radio button state

---

## API Changes

### No Breaking Changes

All changes are **additive and backward compatible**:

1. ✅ Existing `take_record()` calls without `mode` parameter default to 'standard'
2. ✅ Existing code continues to work unchanged
3. ✅ New `mode` parameter is optional
4. ✅ Calibration Impulse tab (testing) unaffected

---

## Implementation Summary

### Total Estimated Effort: **10-12 hours**

| Phase | Description | Priority | Effort |
|-------|-------------|----------|--------|
| 1 | Series Settings - Mode selection UI | ⭐⭐⭐⭐⭐ Critical | 4 hours |
| 2 | Audio Settings - Status display | ⭐⭐⭐ High | 45 min |
| 3 | Configuration persistence | ⭐⭐⭐ High | 1 hour |
| 4 | Documentation & help text | ⭐⭐ Medium | 45 min |
| 5 | Integration testing | ⭐⭐⭐⭐ High | 3-4 hours |

### Recommended Implementation Order:

1. **Phase 1.1-1.2** → UI components for mode selection
2. **Phase 1.3** → Test recording integration
3. **Phase 1.4** → Integrate into Series Settings render
4. **Phase 2** → Status display updates
5. **Phase 3** → Configuration persistence
6. **Phase 4** → Documentation
7. **Phase 5** → Comprehensive testing

---

## Key Benefits

✅ **Unified Interface** - Users can select recording mode in one place (Series Settings)
✅ **Clear Visual Feedback** - Mode-specific configuration displayed prominently
✅ **Backward Compatible** - Existing workflows unchanged
✅ **Configuration Persistence** - Mode preference saved to config file
✅ **Comprehensive Help** - Comparison tables and tooltips guide users
✅ **No Backend Changes** - All work is UI/UX integration
✅ **Testing Support** - Test recording works in both modes

---

## Risk Assessment

### Low Risk Areas
- ✅ Backend already supports calibration mode with file saving
- ✅ No changes to RoomResponseRecorder needed
- ✅ GUI changes are isolated to specific panels
- ✅ Backward compatibility maintained throughout

### Medium Risk Areas
- ⚠️ Configuration persistence might conflict with existing config save logic
- ⚠️ Session state management across multiple panels
- ⚠️ UI layout changes might affect existing users

### Mitigation Strategies
1. Test configuration save/load thoroughly
2. Use unique session state keys to avoid conflicts
3. Keep UI changes minimal and intuitive
4. Provide clear visual indicators for mode selection
5. Add comprehensive help text for new users

---

## Future Enhancements (Post-Implementation)

### Phase 6: Collection Panel Integration (Separate Project)
- Add recording mode selector to Collection Panel
- Enable calibration mode for Single Scenario and Series collection
- See: `CALIBRATION_MODE_COLLECT_PANEL_PLAN.md`

### Phase 7: Advanced Features
- Real-time quality metrics display during recording
- Mode-specific visualization in Audio Analysis panel
- Batch re-processing of calibration datasets
- Export calibration validation reports

---

## Success Criteria

### Must Have (Phase 1-3)
- ✅ Recording mode selector in Series Settings tab
- ✅ Calibration config info displayed when calibration mode selected
- ✅ Test recording works in both modes
- ✅ Files saved correctly for both modes
- ✅ Mode preference persists across sessions

### Should Have (Phase 4-5)
- ✅ Clear documentation and help text
- ✅ Comprehensive testing completed
- ✅ Recorder status shows available modes
- ✅ Backward compatibility verified

### Nice to Have (Future)
- 📋 Collection Panel integration
- 📋 Real-time quality metrics
- 📋 Advanced visualization tools

---

## Conclusion

This implementation plan provides a straightforward path to integrate Calibration Mode into the Audio Settings and Series Settings panels. Since the backend already supports calibration mode with file saving, the work focuses entirely on GUI/UX integration, making this a **low-risk, high-value enhancement**.

The phased approach ensures incremental progress with testable milestones at each stage. The estimated 10-12 hours of development time will deliver a polished, user-friendly interface for switching between Standard and Calibration recording modes.

**Next Steps:**
1. Review and approve plan
2. Begin Phase 1 implementation (Series Settings UI)
3. Test incrementally after each phase
4. Update user documentation with new mode selection workflow
