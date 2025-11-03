# Phase 1 Testing Guide - Calibration Mode Integration

**Date:** 2025-11-02
**Phase:** 1 - Recording Mode Selection in Series Settings
**Estimated Testing Time:** 1-2 hours

---

## Prerequisites

### Required Setup:
- ✅ Python environment with Streamlit installed
- ✅ RoomResponseRecorder configured
- ✅ Audio devices connected (input and output)
- ✅ SDL Audio Core available

### Optional Setup (for full calibration testing):
- Multi-channel audio interface
- Calibration sensor configured in recorderConfig.json
- Physical impact source (hammer, tapping device)

---

## Test Suite

### Test 1: Basic Syntax and Import
**Priority:** CRITICAL
**Time:** 2 minutes

#### Steps:
1. Open terminal in project directory
2. Run: `python -m py_compile gui_series_settings_panel.py`
3. Verify no syntax errors

#### Expected Result:
- ✅ No output (successful compilation)

#### If Failed:
- Check Python syntax in modified sections
- Verify imports at top of file
- Review line numbers in error message

---

### Test 2: Application Launch
**Priority:** CRITICAL
**Time:** 2 minutes

#### Steps:
1. Run: `.venv/Scripts/python.exe -m streamlit run piano_response.py`
2. Wait for browser to open
3. Navigate to Audio Settings

#### Expected Result:
- ✅ Application starts without errors
- ✅ Audio Settings panel loads
- ✅ No exceptions in console

#### If Failed:
- Check console for error messages
- Verify all imports are available
- Check session state initialization

---

### Test 3: Series Settings Panel - Single Channel Mode
**Priority:** HIGH
**Time:** 5 minutes

#### Setup:
- Multi-channel disabled OR no calibration sensor configured

#### Steps:
1. Navigate to Audio Settings → Series Settings
2. Go to "Pulse Series Config" tab
3. Look for "Recording Mode" section

#### Expected Result:
```
### Recording Mode

ℹ️ Standard Mode (Room Response)
Calibration mode requires multi-channel setup with
calibration sensor. Configure in Device Selection tab.
```

#### Verify:
- ✅ Only info message shown (no radio buttons)
- ✅ Message explains why calibration unavailable
- ✅ Link to Device Selection tab mentioned
- ✅ No errors in console

---

### Test 4: Multi-Channel Without Calibration Sensor
**Priority:** HIGH
**Time:** 5 minutes

#### Setup:
- Enable multi-channel in Device Selection
- Do NOT configure calibration sensor (leave as None)

#### Steps:
1. Navigate to Audio Settings → Device Selection
2. Enable multi-channel recording
3. Configure 2+ channels
4. Do NOT set calibration channel
5. Save configuration
6. Navigate to Series Settings → Pulse Series Config

#### Expected Result:
- Same as Test 3 (Standard mode only)
- ℹ️ Info message explains calibration sensor needed

#### Verify:
- ✅ No radio buttons shown
- ✅ Helpful message displayed
- ✅ No crashes or errors

---

### Test 5: Multi-Channel WITH Calibration Sensor
**Priority:** CRITICAL
**Time:** 10 minutes

#### Setup:
- Enable multi-channel in Device Selection
- Configure calibration sensor (e.g., Channel 0)
- Set reference channel (e.g., Channel 1)
- Save configuration

#### Steps:
1. Navigate to Series Settings → Pulse Series Config
2. Look for "Recording Mode" section

#### Expected Result:
```
### Recording Mode

Choose recording mode:
⚪ Standard (Room Response)
⚪ Calibration (Physical Impact)

[Help text displayed for each option]
```

#### Verify:
- ✅ Radio button selector appears
- ✅ Standard mode selected by default
- ✅ Help text available (hover or expand)
- ✅ Can click between modes
- ✅ Selection updates immediately

---

### Test 6: Calibration Mode Configuration Display
**Priority:** CRITICAL
**Time:** 10 minutes

#### Setup:
- Continue from Test 5
- Calibration mode available

#### Steps:
1. Select "Calibration (Physical Impact)" mode
2. Wait for UI to update

#### Expected Result:
```
🔨 Calibration Mode Configuration [Expanded]

[Sensor Setup]              [Processing Options]
🔨 Calibration: Ch 0        ✅ Normalization: Enabled
   [Channel Name]           (or ⚠️ Disabled)
🎤 Reference: Ch 1
   [Channel Name]

[Quality Validation]
Negative peak range: X.XX - X.XX
Correlation threshold: X.XX
```

#### Verify:
- ✅ Expander appears when calibration selected
- ✅ Sensor channels displayed correctly
- ✅ Channel names shown (if configured)
- ✅ Normalization status correct
- ✅ Validation thresholds displayed
- ✅ Layout is readable and organized

---

### Test 7: Mode Comparison Table
**Priority:** MEDIUM
**Time:** 5 minutes

#### Steps:
1. In Pulse Series Config tab
2. Look for "Recording Mode Comparison" expander
3. Click to expand

#### Expected Result:
- ℹ️ Expandable section with comparison table
- Table shows: Signal Source, Quality Validation, Normalization, etc.
- "When to use" guidance for both modes

#### Verify:
- ✅ Table renders correctly
- ✅ All rows present
- ✅ Icons display properly (✅ ❌)
- ✅ Text is readable
- ✅ "When to use" sections helpful

---

### Test 8: Mode Switching
**Priority:** HIGH
**Time:** 5 minutes

#### Steps:
1. Select "Standard (Room Response)"
2. Verify calibration config section disappears
3. Select "Calibration (Physical Impact)"
4. Verify calibration config section appears
5. Repeat 2-3 times

#### Expected Result:
- UI updates immediately when mode changed
- Calibration config section shows/hides correctly
- No delays or errors

#### Verify:
- ✅ Smooth UI updates
- ✅ No console errors
- ✅ Session state tracks selection
- ✅ Can switch freely between modes

---

### Test 9: Recording in Standard Mode
**Priority:** CRITICAL
**Time:** 10 minutes

#### Setup:
- Configure pulse series parameters
- Select "Standard (Room Response)" mode

#### Steps:
1. Go to "Recording & Analysis" tab
2. Click "Record Series" button
3. Wait for recording to complete

#### Expected Result:
```
🎵 Recording pulse series (standard mode)...
✅ Series recording OK — X.XXXs
ℹ️ Files saved: series_raw_xxx.wav, series_impulse_xxx.wav
```

#### Verify:
- ✅ Recording completes successfully
- ✅ Mode indicated in spinner message
- ✅ Files saved to TMP directory
- ✅ Analysis section shows: "📊 Last recording used Standard Mode"
- ✅ Standard analysis results displayed
- ✅ No calibration-specific output

---

### Test 10: Recording in Calibration Mode (Simulation)
**Priority:** CRITICAL
**Time:** 15 minutes

#### Setup:
- Multi-channel with calibration sensor
- Select "Calibration (Physical Impact)" mode
- Configure pulse series (e.g., 4-8 pulses)

#### Steps:
1. Go to "Recording & Analysis" tab
2. Click "Record Series" button
3. Wait for recording to complete
4. Observe output

#### Expected Result:
```
ℹ️ Recording with Calibration Mode (quality validation enabled)
🎵 Recording pulse series (calibration mode)...

✅ Calibration recording completed

[Total Cycles]  [Valid Cycles]  [Aligned Cycles]
      8              X (XX%)            X

📊 Per-Cycle Validation Results [Expand]
  Cycle 0: ✅ Valid
  Cycle 1: ✅ Valid
  ...
```

#### Verify:
- ✅ Mode indicator shown before recording
- ✅ Spinner shows "calibration mode"
- ✅ Success message displayed
- ✅ Metrics shown: Total, Valid %, Aligned
- ✅ Per-cycle results expandable
- ✅ Files saved (check TMP directory)
- ✅ Analysis section shows: "🔨 Last recording used Calibration Mode"

#### Notes:
- With synthetic pulses, validation may show low valid cycle count
- This is expected (calibration designed for physical impacts)
- Important: No crashes or errors during recording

---

### Test 11: Calibration Validation Results
**Priority:** HIGH
**Time:** 10 minutes

#### Steps:
1. After calibration mode recording (Test 10)
2. Expand "Per-Cycle Validation Results"
3. Review cycle-by-cycle status

#### Expected Result:
```
Cycle 0: ✅ Valid
Cycle 1: ❌ Invalid
  Reasons: Negative peak out of range (0.XX)
Cycle 2: ✅ Valid
...
```

#### Verify:
- ✅ Each cycle has status (Valid/Invalid)
- ✅ Invalid cycles show reasons
- ✅ Reasons are descriptive
- ✅ UI is readable

---

### Test 12: Session State Persistence
**Priority:** MEDIUM
**Time:** 5 minutes

#### Steps:
1. Select "Calibration (Physical Impact)" mode
2. Navigate to another panel (e.g., Scenarios)
3. Return to Series Settings
4. Check recording mode

#### Expected Result:
- Calibration mode still selected
- Calibration config still displayed
- Session state maintained

#### Verify:
- ✅ Mode selection persists
- ✅ No reset to default
- ✅ UI state consistent

---

### Test 13: Configuration File Loading
**Priority:** LOW (Phase 3 feature)
**Time:** 5 minutes

#### Steps:
1. Check `recorderConfig.json` for `default_recording_mode`
2. If not present, add: `"default_recording_mode": "calibration"`
3. Restart application
4. Navigate to Series Settings

#### Expected Result (Phase 1):
- Mode resets to 'standard' (default)
- Config value not used yet (Phase 3 feature)

#### Expected Result (Phase 3):
- Mode loads from config file
- Calibration selected if configured

#### Verify:
- ✅ Application loads without errors
- ✅ No crashes reading config

---

### Test 14: Edge Cases
**Priority:** MEDIUM
**Time:** 10 minutes

#### Test 14a: Rapid Mode Switching
1. Click between modes rapidly (10+ times)
2. Verify no errors or UI glitches

#### Test 14b: Recording During Mode Change
1. Select Standard mode
2. Start recording
3. While recording, try to change mode
4. Verify behavior (should not change during recording)

#### Test 14c: Invalid Calibration Configuration
1. Set calibration channel = reference channel
2. Try to use calibration mode
3. Verify error handling or warning

#### Verify:
- ✅ No crashes under rapid interaction
- ✅ Graceful handling of edge cases
- ✅ Clear error messages if problems occur

---

### Test 15: Backward Compatibility
**Priority:** CRITICAL
**Time:** 10 minutes

#### Steps:
1. Use single-channel configuration (existing setup)
2. Navigate to Series Settings
3. Record series (standard mode)
4. Verify existing workflow unchanged

#### Expected Result:
- Everything works as before Phase 1
- Standard mode is default and only option
- No new errors or issues
- Files saved correctly

#### Verify:
- ✅ Existing functionality preserved
- ✅ No breaking changes
- ✅ Legacy configurations work

---

## Test Results Checklist

### Critical Tests (Must Pass):
- [ ] Test 1: Basic Syntax and Import
- [ ] Test 2: Application Launch
- [ ] Test 5: Multi-Channel WITH Calibration Sensor
- [ ] Test 6: Calibration Mode Configuration Display
- [ ] Test 9: Recording in Standard Mode
- [ ] Test 10: Recording in Calibration Mode
- [ ] Test 15: Backward Compatibility

### High Priority Tests:
- [ ] Test 3: Series Settings Panel - Single Channel Mode
- [ ] Test 4: Multi-Channel Without Calibration Sensor
- [ ] Test 8: Mode Switching
- [ ] Test 11: Calibration Validation Results

### Medium Priority Tests:
- [ ] Test 7: Mode Comparison Table
- [ ] Test 12: Session State Persistence
- [ ] Test 14: Edge Cases

### Low Priority Tests:
- [ ] Test 13: Configuration File Loading (Phase 3)

---

## Common Issues and Troubleshooting

### Issue 1: No Radio Buttons Shown
**Symptom:** Only info message, no mode selector
**Cause:** Calibration sensor not configured
**Solution:**
1. Go to Device Selection tab
2. Enable multi-channel
3. Set calibration channel
4. Save configuration

### Issue 2: Calibration Recording Fails
**Symptom:** Error during calibration mode recording
**Cause:** Calibration validation failing (expected with synthetic signals)
**Solution:**
- This is expected behavior
- Calibration mode designed for physical impacts
- Test that error is handled gracefully
- Verify error message is clear

### Issue 3: UI Not Updating
**Symptom:** Calibration config doesn't appear when mode selected
**Cause:** Session state not updating
**Solution:**
1. Check browser console for errors
2. Try refreshing page (will lose session state)
3. Check `_render_calibration_mode_info()` logic

### Issue 4: Files Not Saved
**Symptom:** No files in TMP directory after recording
**Cause:** save_files parameter not passed correctly
**Solution:**
1. Check `_execute_series_recording()` line 559
2. Verify `save_files=True` is present
3. Check console for file I/O errors

### Issue 5: Validation Metrics Not Shown
**Symptom:** No cycle metrics after calibration recording
**Cause:** Result dict structure unexpected
**Solution:**
1. Check console for error messages
2. Verify `calibration_cycles` key in result dict
3. Add debugging print statements

---

## Performance Criteria

### Acceptable Performance:
- ✅ Mode switching: < 1 second UI update
- ✅ Recording start: < 2 seconds to begin
- ✅ Result display: < 1 second after recording
- ✅ No memory leaks during repeated recordings

### Unacceptable:
- ❌ UI freezing during mode switch
- ❌ Long delays (> 5 seconds) without feedback
- ❌ Application crashes
- ❌ Data loss or corruption

---

## Reporting Issues

### Issue Report Template:

```
**Test Case:** [Test number and name]
**Priority:** [Critical/High/Medium/Low]
**Status:** [Failed]

**Steps to Reproduce:**
1. [Step 1]
2. [Step 2]
3. [Step 3]

**Expected Result:**
[What should happen]

**Actual Result:**
[What actually happened]

**Console Output:**
[Paste any error messages]

**Screenshots:**
[Attach if relevant]

**Configuration:**
- Multi-channel: [Yes/No]
- Calibration sensor: [Channel number or None]
- Number of channels: [X]
- Sample rate: [XXXXX Hz]

**Additional Notes:**
[Any other relevant information]
```

---

## Success Criteria

### Phase 1 Testing PASSES if:
- ✅ All Critical tests pass
- ✅ At least 80% of High Priority tests pass
- ✅ No application crashes
- ✅ Backward compatibility maintained
- ✅ Mode selection works correctly
- ✅ Calibration config displays properly
- ✅ Recording works in both modes

### Phase 1 Testing FAILS if:
- ❌ Any Critical test fails
- ❌ Application crashes during normal use
- ❌ Backward compatibility broken
- ❌ Mode selection doesn't work
- ❌ Recording fails in either mode

---

## Next Steps After Testing

### If All Tests Pass:
1. ✅ Mark Phase 1 as COMPLETE
2. ✅ Document any minor issues for future fixes
3. ✅ Proceed to Phase 2 (Status Display)
4. ✅ Update CALIBRATION_MODE_INTEGRATION_PLAN.md

### If Critical Tests Fail:
1. ❌ Stop and fix issues immediately
2. ❌ Re-run failed tests
3. ❌ Document fixes made
4. ❌ Full re-test before proceeding

### If Only Minor Issues:
1. ⚠️ Document issues
2. ⚠️ Assess impact on Phase 2-5
3. ⚠️ Decide: Fix now or defer
4. ⚠️ Proceed with caution

---

## Conclusion

This testing guide provides comprehensive coverage of Phase 1 functionality. Follow the test cases in order, document results, and report any issues using the template provided.

**Estimated Total Testing Time:** 1-2 hours for full suite

**Ready to Test!** 🚀
