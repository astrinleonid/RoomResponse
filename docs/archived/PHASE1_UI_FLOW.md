# Phase 1 UI Flow - Calibration Mode Integration

**Visual Guide to New UI Elements**

---

## Series Settings Panel Structure

```
┌─────────────────────────────────────────────────────────────┐
│ Series Settings — Multi-pulse Configuration                 │
├─────────────────────────────────────────────────────────────┤
│ ✅ SDL Audio   ✅ Recorder   ✅ Visualizer   ✅ Devices     │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│  TABS:  [ Pulse Series Config ]  [ Recording & Analysis ]  │
│         [ Advanced Settings ]                                │
└─────────────────────────────────────────────────────────────┘
```

---

## Tab 1: Pulse Series Config (NEW ELEMENTS)

### Before Phase 1:
```
┌─────────────────────────────────────────────────────────────┐
│ Multi-pulse Series Configuration                            │
│ ℹ️ Settings loaded from recorderConfig.json                │
├─────────────────────────────────────────────────────────────┤
│ [Pulse Properties]  [Timing & Volume]  [Pulse Form]        │
│ ...                                                          │
└─────────────────────────────────────────────────────────────┘
```

### After Phase 1:
```
┌─────────────────────────────────────────────────────────────┐
│ Multi-pulse Series Configuration                            │
│ ℹ️ Settings loaded from recorderConfig.json                │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│ ┌──────────────────────────────────────────────────────┐   │
│ │ ### Recording Mode                        ← NEW!     │   │
│ │                                                       │   │
│ │ Choose recording mode:                                │   │
│ │ ⚪ Standard (Room Response)                          │   │
│ │ ⚪ Calibration (Physical Impact)                     │   │
│ │                                                       │   │
│ │ [?] Help: Standard Mode / Calibration Mode info...   │   │
│ └──────────────────────────────────────────────────────┘   │
│                                                              │
│ ┌──────────────────────────────────────────────────────┐   │
│ │ 🔨 Calibration Mode Configuration     ← NEW!        │   │
│ │                                                       │   │
│ │ [Sensor Setup]         [Processing Options]          │   │
│ │ 🔨 Calibration: Ch 2   ✅ Normalization: Enabled    │   │
│ │ 🎤 Reference: Ch 5     Responses by impact magnitude │   │
│ │                                                       │   │
│ │ [Quality Validation]                                  │   │
│ │ Negative peak: 0.10 - 0.95                           │   │
│ │ Correlation threshold: 0.70                           │   │
│ └──────────────────────────────────────────────────────┘   │
│                                                              │
│ ┌──────────────────────────────────────────────────────┐   │
│ │ ℹ️ Recording Mode Comparison          ← NEW!        │   │
│ │                                                       │   │
│ │ [Click to expand comparison table]                   │   │
│ └──────────────────────────────────────────────────────┘   │
│                                                              │
├─────────────────────────────────────────────────────────────┤
│ [Pulse Properties]  [Timing & Volume]  [Pulse Form]        │
│ ...                                                          │
└─────────────────────────────────────────────────────────────┘
```

---

## Mode Selection: Two Scenarios

### Scenario A: Calibration Sensor NOT Configured
```
┌─────────────────────────────────────────────────────────────┐
│ ### Recording Mode                                           │
│                                                              │
│ ℹ️ Standard Mode (Room Response)                           │
│ Calibration mode requires multi-channel setup with          │
│ calibration sensor. Configure in Device Selection tab.      │
└─────────────────────────────────────────────────────────────┘
```

### Scenario B: Calibration Sensor Configured
```
┌─────────────────────────────────────────────────────────────┐
│ ### Recording Mode                                           │
│                                                              │
│ Choose recording mode:                                       │
│ ⚪ Standard (Room Response)                                 │
│ ⚪ Calibration (Physical Impact)                            │
│                                                              │
│ [?] Standard Mode:                                          │
│     - Room acoustic responses using synthetic pulse train   │
│     - Audio output from speaker, captured by microphones    │
│     - Best for: Room impulse response measurements          │
│                                                              │
│ [?] Calibration Mode:                                       │
│     - Physical impact responses (e.g., hammer strikes)      │
│     - Requires calibration sensor (force/impact sensor)     │
│     - Per-cycle quality validation                          │
│     - Automatic alignment and optional normalization        │
│     - Best for: Piano hammer impact studies                 │
└─────────────────────────────────────────────────────────────┘
```

---

## Calibration Mode Info (Only Shows When Calibration Selected)

```
┌─────────────────────────────────────────────────────────────┐
│ 🔨 Calibration Mode Configuration                          │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  [Sensor Setup]                [Processing Options]         │
│  ┌─────────────────────────┐  ┌──────────────────────────┐ │
│  │ 🔨 Calibration Sensor:  │  │ ✅ Normalization:       │ │
│  │    Ch 2 - Hammer Sensor │  │    Enabled               │ │
│  │                          │  │ Responses normalized by  │ │
│  │ 🎤 Reference Channel:   │  │ impact magnitude         │ │
│  │    Ch 5 - Front Mic     │  │                          │ │
│  └─────────────────────────┘  └──────────────────────────┘ │
│                                                              │
│  [Quality Validation]                                        │
│  ┌────────────────────────────────────────────────────────┐ │
│  │ Negative peak range: 0.10 - 0.95                       │ │
│  │ Correlation threshold: 0.70                             │ │
│  └────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

---

## Mode Comparison Table (Expandable)

```
┌─────────────────────────────────────────────────────────────┐
│ ℹ️ Recording Mode Comparison                ▼ [Expanded]   │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│ | Feature               | Standard | Calibration |          │
│ |----------------------|----------|-------------|           │
│ | Signal Source        | Synthetic| Physical    |           │
│ | Quality Validation   | ❌ No    | ✅ Yes      |           │
│ | Cycle Alignment      | Basic    | ✅ Advanced |           │
│ | Normalization        | ❌ No    | ✅ Optional |           │
│ | Calibration Sensor   | Not req. | ✅ Required |           │
│                                                              │
│ When to use Calibration Mode:                               │
│ ✅ Recording physical impacts (hammer strikes)              │
│ ✅ Need per-event quality validation                        │
│ ✅ Varying impact magnitudes need normalization             │
│                                                              │
│ When to use Standard Mode:                                  │
│ ✅ Synthetic audio signals (pulse trains)                   │
│ ✅ Room acoustic measurements                               │
│ ✅ Basic impulse response extraction                        │
└─────────────────────────────────────────────────────────────┘
```

---

## Tab 2: Recording & Analysis (NEW INDICATOR)

### Before Recording:
```
┌─────────────────────────────────────────────────────────────┐
│ Series Recording Analysis                                    │
├─────────────────────────────────────────────────────────────┤
│ No series recording yet. Use Record Series.                 │
└─────────────────────────────────────────────────────────────┘
```

### After Recording (Standard Mode):
```
┌─────────────────────────────────────────────────────────────┐
│ Series Recording Analysis                                    │
├─────────────────────────────────────────────────────────────┤
│ 📊 Last recording used Standard Mode (room response)        │
│                                                              │
│ Recorded at: 14:32:15                                       │
│ [Analysis results...]                                        │
└─────────────────────────────────────────────────────────────┘
```

### After Recording (Calibration Mode):
```
┌─────────────────────────────────────────────────────────────┐
│ Series Recording Analysis                                    │
├─────────────────────────────────────────────────────────────┤
│ 🔨 Last recording used Calibration Mode (quality validation)│
│                                                              │
│ Recorded at: 14:32:15                                       │
│ [Analysis results...]                                        │
└─────────────────────────────────────────────────────────────┘
```

---

## Recording Flow

### Standard Mode:
```
User clicks "Record Series"
        ↓
🎵 Recording pulse series (standard mode)...
        ↓
Recording completed
        ↓
✅ Series recording OK — 2.400s
ℹ️ Files saved: series_raw_xxx.wav, series_impulse_xxx.wav
```

### Calibration Mode:
```
User clicks "Record Series"
        ↓
ℹ️ Recording with Calibration Mode (quality validation enabled)
        ↓
🎵 Recording pulse series (calibration mode)...
        ↓
Recording completed
        ↓
✅ Calibration recording completed

┌──────────────────────────────────────────┐
│  [Total Cycles] [Valid Cycles] [Aligned] │
│       8            7 (87.5%)       7     │
└──────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│ 📊 Per-Cycle Validation Results          ▼ [Expand]    │
├─────────────────────────────────────────────────────────┤
│ Cycle 0: ✅ Valid                                       │
│ Cycle 1: ✅ Valid                                       │
│ Cycle 2: ❌ Invalid                                     │
│   Reasons: Negative peak out of range (0.05)           │
│ Cycle 3: ✅ Valid                                       │
│ ...                                                      │
└─────────────────────────────────────────────────────────┘

✅ Series recording OK — 2.400s
ℹ️ Files saved: series_raw_xxx.wav, series_impulse_xxx.wav
```

---

## Session State Keys

### New Keys Added:
```python
'series_recording_mode'       # Current selected mode ('standard' or 'calibration')
'series_recording_mode_used'  # Mode used in last recording
```

### Used By:
- `_render_recording_mode_selection()` - Read/Write
- `_render_calibration_mode_info()` - Read
- `_execute_series_recording()` - Read/Write
- `_render_recording_analysis()` - Read

---

## User Interaction Flow

```
1. User opens Audio Settings → Series Settings
        ↓
2. [IF calibration sensor configured]
   User sees mode selector
        ↓
3. User selects "Calibration (Physical Impact)"
        ↓
4. Calibration config info appears automatically
        ↓
5. User can expand mode comparison table for guidance
        ↓
6. User configures pulse series parameters
        ↓
7. User switches to "Recording & Analysis" tab
        ↓
8. User clicks "Record Series"
        ↓
9. Recording runs with calibration mode
        ↓
10. Validation metrics displayed immediately
        ↓
11. User can expand per-cycle validation details
        ↓
12. Analysis shows "🔨 Calibration Mode" indicator
```

---

## Visual Indicators Summary

| Element | Icon | Color | Meaning |
|---------|------|-------|---------|
| Standard Mode | 📊 | Blue (info) | Room response recording |
| Calibration Mode | 🔨 | Blue (info) | Physical impact recording |
| Calibration Sensor | 🔨 | Green (success) | Impact sensor channel |
| Reference Channel | 🎤 | Blue (info) | Alignment reference |
| Normalization ON | ✅ | Green (success) | Enabled |
| Normalization OFF | ⚠️ | Yellow (warning) | Disabled |
| Valid Cycle | ✅ | Green | Passed validation |
| Invalid Cycle | ❌ | Red | Failed validation |

---

## Comparison: Before vs After

### Before Phase 1:
- ❌ No way to select recording mode in GUI
- ❌ No calibration configuration display
- ❌ Always uses standard mode
- ❌ No validation metrics shown

### After Phase 1:
- ✅ Clear mode selector with help text
- ✅ Calibration configuration display
- ✅ Mode-specific recording flow
- ✅ Validation metrics and per-cycle results
- ✅ Visual indicators throughout UI
- ✅ Backward compatible with existing workflows

---

## Conclusion

Phase 1 provides a comprehensive, user-friendly interface for selecting and using calibration mode directly from the Series Settings panel. The UI clearly communicates available options, provides helpful guidance, and gives immediate feedback on recording quality.
