# Calibration Mode Dataset Collection - Collect Panel Implementation Plan

**Document Version:** 1.0
**Created:** 2025-11-02
**Status:** 📋 Planning Phase
**Target:** Enable Calibration Mode dataset collection in Collect Panel

---

## Executive Summary

This document provides a comprehensive implementation plan for adding **Calibration Mode dataset collection** to the Collect Panel. Currently, calibration mode is only used in the Audio Settings → Calibration Impulse panel for testing/validation and **does not save data to disk**. This plan enables systematic multi-measurement data collection using calibration mode for physical impact studies.

---

## Understanding: Two Distinct Recording Modes

### Mode 1: STANDARD Mode (Room Acoustic Response)

**Physical Setup:**
- **Purpose:** Measure room acoustic response
- **Signal Source:** Synthetic audio signal (pulse train) from output device/speaker
- **Sensors:** Multiple microphones at different spatial positions
- **Recording:** All mics simultaneously capture room's response to the synthetic signal
- **Output:** Raw recordings + extracted impulse responses
- **Files Saved:** ✅ Yes (multi-channel WAV files)
- **Use in Collect Panel:** ✅ **Currently Used**

**Example Configuration:**
```json
{
  "multichannel": {
    "enabled": true,
    "num_channels": 3,
    "channel_names": ["Front Mic", "Rear Mic", "Side Mic"],
    "reference_channel": 0
  }
}
```

**Data Flow:**
```
Synthetic Pulse Train (Speaker)
        ↓
    [Room Environment]
        ↓
    ┌───────┬───────┬───────┐
    │ Mic 0 │ Mic 1 │ Mic 2 │
    └───────┴───────┴───────┘
        ↓       ↓       ↓
    Record Multi-Channel
        ↓
    Save WAV Files
        ↓
    Extract Impulse Responses
```

---

### Mode 2: CALIBRATION Mode (Physical Impact Measurement)

**Physical Setup:**
- **Purpose:** Measure physical impact responses (e.g., piano hammer strikes)
- **Signal Source:** PHYSICAL IMPACT (hammer striking piano string/surface)
- **Sensors:**
  - **Calibration Sensor (Ch 0):** Force/impact sensor attached to hammer
  - **Response Sensors (Ch 1-N):** Microphones/accelerometers measuring acoustic/vibrational response
- **Recording:** All sensors simultaneously capture the physical impact event
- **Output:** Cycle-level validated data with quality metrics
- **Files Saved:** ❌ **Currently NO** (analysis only) → ✅ **Target: YES**
- **Use in Collect Panel:** ❌ **Not Currently Available** → ✅ **Target: Available**

**Example Configuration:**
```json
{
  "multichannel": {
    "enabled": true,
    "num_channels": 4,
    "channel_names": ["Hammer Sensor", "Front Mic", "Rear Mic", "Side Mic"],
    "calibration_channel": 0,
    "reference_channel": 1,
    "response_channels": [1, 2, 3],
    "normalize_by_calibration": true
  }
}
```

**Data Flow:**
```
Physical Impact (Hammer Strike)
        ↓
    ┌─────────────────────────────┐
    │  Hammer Force Sensor (Ch 0) │  ← Measures impact magnitude
    └─────────────────────────────┘
        ↓
    [Piano String Vibration]
        ↓
    ┌───────┬───────┬───────┐
    │ Mic 1 │ Mic 2 │ Mic 3 │  ← Measure acoustic response
    └───────┴───────┴───────┘
        ↓       ↓       ↓
    Record Multi-Channel
        ↓
    Extract Cycles from Calibration Channel
        ↓
    Validate Each Cycle (impact quality)
        ↓
    Align All Channels by Impact Onset
        ↓
    Normalize Response by Impact Magnitude
        ↓
    [NEW] Save Cycle Data to Files
```

---

## Key Differences

| Aspect | Standard Mode | Calibration Mode |
|--------|---------------|------------------|
| **Signal Source** | Synthetic (speaker) | Physical (hammer/impact) |
| **Calibration Channel** | None | Yes (impact sensor) |
| **Purpose** | Dataset collection | Dataset collection + Quality validation |
| **File Output** | ✅ Saves WAV files | ❌ Currently NO → ✅ Target YES |
| **Processing** | Extract impulse response | Validate cycles + align + normalize |
| **Quality Validation** | No | Yes (per-cycle quality metrics) |
| **Normalization** | No | Yes (by impact magnitude) |
| **Use Case** | Room response research | Physical impact studies |
| **Current GUI** | Collect Panel | Audio Settings → Calibration Impulse (test only) |

---

## Problem Statement

**Current State:**
- ❌ Calibration mode **does NOT save any files** (only returns analysis data)
- ❌ Only used in Audio Settings panel for **testing/validation**
- ❌ Cannot be used in Collect Panel for **systematic data collection**
- ❌ No support for **Series mode** with calibration

**User Need:**
- ✅ Collect datasets of physical impact measurements (e.g., piano hammer strikes)
- ✅ Systematic multi-measurement collection with quality validation
- ✅ Save aligned and normalized cycle data for later analysis
- ✅ Use both Single Scenario and Series modes

**Gap:**
The `_take_record_calibration_mode()` method in `RoomResponseRecorder.py` does NOT save files. It only returns analysis data designed for display in the GUI.

---

## Solution Architecture

### Strategy: Add File-Saving Path to Calibration Mode

**Three-pronged approach:**

1. **Extend `take_record()` API** to save calibration data when requested
2. **Add Calibration Mode support** to Collect Panel UI and workflow
3. **Maintain backward compatibility** with existing Audio Settings usage

---

## Implementation Plan

### Phase 1: Extend RoomResponseRecorder for Calibration Dataset Collection ⭐⭐⭐⭐⭐

**File:** `RoomResponseRecorder.py`

#### 1.1: Add `save_calibration_data` Parameter

**Modify `take_record()` signature:**

```python
def take_record(self,
                output_file: str,
                impulse_file: str,
                method: int = 2,
                mode: str = 'standard',
                return_processed: bool = False,
                save_calibration_data: bool = False):  # NEW PARAMETER
    """
    Main API method to record room response

    Args:
        output_file: Filename for raw recording
        impulse_file: Filename for impulse response
        method: Recording method (1=manual, 2=auto, 3=specific devices)
        mode: Recording mode - 'standard' or 'calibration'
        return_processed: If True, return dict with processed data
        save_calibration_data: If True and mode='calibration', save cycle data to files

    Returns:
        Standard mode:
            Single-channel: np.ndarray (raw audio)
            Multi-channel: Dict[int, np.ndarray] (raw audio per channel)

        Calibration mode:
            Dict[str, Any] with calibration cycle data
    """
    # Validate mode parameter
    if mode not in ['standard', 'calibration']:
        raise ValueError(f"Invalid mode: {mode}. Must be 'standard' or 'calibration'")

    # Handle calibration mode
    if mode == 'calibration':
        result = self._take_record_calibration_mode()

        # NEW: Save calibration data if requested
        if save_calibration_data:
            self._save_calibration_dataset(result, output_file, impulse_file)
            print(f"✓ Calibration data saved to disk")

        return result

    # Standard mode continues unchanged...
```

**Effort:** 15 minutes

---

#### 1.2: Implement `_save_calibration_dataset()` Method

**New method to save calibration cycle data:**

```python
def _save_calibration_dataset(self,
                               calibration_result: Dict[str, Any],
                               output_file: str,
                               impulse_file: str) -> None:
    """
    Save calibration mode cycle data as multi-channel dataset.

    Saves all channels' cycle data separately for later analysis.

    Args:
        calibration_result: Dict from _take_record_calibration_mode()
        output_file: Base filename for raw data
        impulse_file: Base filename for aligned/normalized cycles

    File Structure Created:
        {output_file}_TIMESTAMP_ch{N}.wav           - Raw recording (full recording)
        {impulse_file}_aligned_TIMESTAMP_ch{N}.wav  - Aligned cycles (all valid cycles)
        {impulse_file}_normalized_TIMESTAMP_ch{N}.wav - Normalized cycles (if enabled)
        {output_file}_metadata_TIMESTAMP.json      - Validation & alignment metadata

    Each WAV file contains:
        - For raw: Full multi-pulse recording
        - For aligned: Concatenated aligned cycles
        - For normalized: Concatenated normalized cycles
    """
    import scipy.io.wavfile as wavfile
    import json
    from datetime import datetime

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Extract data from calibration result
    aligned_cycles_dict = calibration_result['aligned_multichannel_cycles']
    normalized_cycles_dict = calibration_result.get('normalized_multichannel_cycles', {})
    validation_results = calibration_result['validation_results']
    alignment_metadata = calibration_result['alignment_metadata']

    num_channels = self.multichannel_config['num_channels']
    channel_names = self.multichannel_config['channel_names']

    print(f"\n{'=' * 60}")
    print(f"Saving Calibration Dataset")
    print(f"{'=' * 60}")

    # Save each channel's data
    for ch_idx in range(num_channels):
        ch_name = channel_names[ch_idx] if ch_idx < len(channel_names) else f"Ch{ch_idx}"

        if ch_idx not in aligned_cycles_dict:
            print(f"  ⚠️  Channel {ch_idx} ({ch_name}): No data")
            continue

        # Get aligned cycles for this channel
        aligned_cycles = aligned_cycles_dict[ch_idx]  # Shape: [num_cycles, samples_per_cycle]

        # Save aligned cycles (concatenate all cycles)
        aligned_filename = f"{impulse_file}_aligned_{timestamp}_ch{ch_idx}.wav"
        aligned_flat = aligned_cycles.reshape(-1)  # Flatten to 1D
        wavfile.write(aligned_filename, self.sample_rate, aligned_flat.astype(np.float32))
        print(f"  ✓ Channel {ch_idx} ({ch_name}): Aligned cycles → {aligned_filename}")

        # Save normalized cycles if available
        if ch_idx in normalized_cycles_dict:
            normalized_cycles = normalized_cycles_dict[ch_idx]
            normalized_filename = f"{impulse_file}_normalized_{timestamp}_ch{ch_idx}.wav"
            normalized_flat = normalized_cycles.reshape(-1)
            wavfile.write(normalized_filename, self.sample_rate, normalized_flat.astype(np.float32))
            print(f"  ✓ Channel {ch_idx} ({ch_name}): Normalized cycles → {normalized_filename}")

    # Save metadata JSON
    metadata = {
        'timestamp': timestamp,
        'recording_mode': 'calibration',
        'sample_rate': int(self.sample_rate),
        'num_channels': num_channels,
        'channel_names': channel_names,
        'calibration_channel': self.multichannel_config['calibration_channel'],
        'reference_channel': self.multichannel_config['reference_channel'],
        'num_pulses': self.num_pulses,
        'cycle_duration': float(self.cycle_duration),
        'cycle_samples': int(self.cycle_samples),
        'num_total_cycles': len(validation_results),
        'num_valid_cycles': calibration_result['num_valid_cycles'],
        'num_aligned_cycles': calibration_result['num_aligned_cycles'],
        'normalization_enabled': calibration_result['metadata']['normalize_by_calibration'],
        'validation_results': validation_results,
        'alignment_metadata': {
            'correlation_threshold': alignment_metadata['correlation_threshold'],
            'aligned_onset_position': int(alignment_metadata['aligned_onset_position']),
            'reference_cycle_idx': int(alignment_metadata['reference_cycle_idx']),
            'num_cycles_aligned': len(alignment_metadata['valid_cycle_indices']),
        }
    }

    # Add normalization factors if available
    if 'normalization_factors' in calibration_result:
        metadata['normalization_factors'] = [float(f) for f in calibration_result['normalization_factors']]
        metadata['normalization_stats'] = {
            'min': float(min(calibration_result['normalization_factors'])),
            'max': float(max(calibration_result['normalization_factors'])),
            'mean': float(np.mean(calibration_result['normalization_factors'])),
            'std': float(np.std(calibration_result['normalization_factors']))
        }

    metadata_filename = f"{output_file}_metadata_{timestamp}.json"
    with open(metadata_filename, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"  ✓ Metadata → {metadata_filename}")
    print(f"{'=' * 60}\n")
```

**Effort:** 2-3 hours

---

### Phase 2: Add Recording Mode Selection to Collect Panel ⭐⭐⭐⭐⭐

**File:** `gui_collect_panel.py`

#### 2.1: Add Recording Mode Selector

**New method to render mode selection:**

```python
def _render_recording_mode_selection(self) -> str:
    """
    Render recording mode selection (Standard vs Calibration).

    Only shown if multi-channel with calibration sensor is configured.

    Returns:
        Selected mode: 'standard' or 'calibration'
    """
    mc_config = getattr(self.recorder, 'multichannel_config', {})
    mc_enabled = mc_config.get('enabled', False)
    has_calibration = mc_config.get('calibration_channel') is not None

    # If no calibration setup, always use standard mode
    if not mc_enabled or not has_calibration:
        # Store in session state
        st.session_state['collect_recording_mode'] = 'standard'
        return 'standard'

    st.markdown("### Recording Mode")

    # Get current mode from session state
    current_mode = st.session_state.get('collect_recording_mode', 'standard')
    default_index = 0 if current_mode == 'standard' else 1

    mode_selection = st.radio(
        "Choose recording mode:",
        options=["Standard (Room Response)", "Calibration (Physical Impact)"],
        index=default_index,
        key="recording_mode_radio",
        help="""
        **Standard Mode (Room Response):**
        - Record room acoustic responses using synthetic pulse train signals
        - Output from speaker, captured by microphones
        - Typical use: Room impulse response measurements

        **Calibration Mode (Physical Impact):**
        - Record physical impact responses (e.g., hammer strikes)
        - Requires calibration sensor (force/impact sensor)
        - Per-cycle quality validation with configurable thresholds
        - Automatic cycle alignment by impact onset
        - Optional normalization by impact magnitude
        - Typical use: Piano hammer impact studies, sensor calibration
        """
    )

    selected_mode = 'calibration' if 'Calibration' in mode_selection else 'standard'
    st.session_state['collect_recording_mode'] = selected_mode

    return selected_mode
```

**Effort:** 1 hour

---

#### 2.2: Display Mode-Specific Configuration Info

**Add to `_render_recorder_status()` or create new method:**

```python
def _render_calibration_mode_info(self) -> None:
    """
    Display calibration mode configuration when calibration mode is selected.
    """
    current_mode = st.session_state.get('collect_recording_mode', 'standard')

    if current_mode != 'calibration':
        return

    mc_config = getattr(self.recorder, 'multichannel_config', {})

    with st.expander("🔨 Calibration Mode Configuration", expanded=True):
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Sensor Setup**")
            cal_ch = mc_config.get('calibration_channel')
            cal_name = mc_config.get('channel_names', [])[cal_ch] if cal_ch < len(mc_config.get('channel_names', [])) else f"Channel {cal_ch}"
            st.success(f"✓ Calibration Sensor: Ch {cal_ch}")
            st.caption(f"   {cal_name}")

            ref_ch = mc_config.get('reference_channel', 0)
            ref_name = mc_config.get('channel_names', [])[ref_ch] if ref_ch < len(mc_config.get('channel_names', [])) else f"Channel {ref_ch}"
            st.info(f"🎤 Reference Channel: Ch {ref_ch}")
            st.caption(f"   {ref_name}")

        with col2:
            st.markdown("**Processing Options**")
            normalize_enabled = mc_config.get('normalize_by_calibration', False)
            if normalize_enabled:
                st.success("✓ Normalization: Enabled")
                st.caption("Responses normalized by impact magnitude")
            else:
                st.info("○ Normalization: Disabled")
                st.caption("Enable in Audio Settings → Device Selection")

        st.markdown("---")
        st.markdown("**What Gets Saved:**")
        st.markdown("""
        - ✅ Aligned cycles (all channels, multi-cycle WAV files)
        - ✅ Normalized cycles (if normalization enabled)
        - ✅ Validation metadata (per-cycle quality metrics)
        - ✅ Alignment metadata (onset positions, correlations)
        - ✅ Configuration snapshot (channels, thresholds, settings)
        """)

        st.markdown("**Data Quality:**")
        st.markdown("""
        - ✅ Per-cycle validation against quality thresholds
        - ✅ Only valid cycles included in aligned output
        - ✅ Detailed validation reasons logged in metadata
        - ✅ Alignment correlation metrics saved
        """)
```

**Effort:** 1 hour

---

### Phase 3: Integrate Calibration Mode into Collection Workflow ⭐⭐⭐⭐⭐

**File:** `gui_collect_panel.py`

#### 3.1: Update `render()` Method

**Add mode selection before collection configuration:**

```python
def render(self) -> None:
    st.header("Collect - Data Collection")

    # Display recorder status including multi-channel configuration
    self._render_recorder_status()

    if not self._check_dependencies():
        return

    if self.recorder is None:
        st.error("Shared RoomResponseRecorder is not available...")
        return

    root = st.session_state.get(SK_DATASET_ROOT, os.getcwd())
    if not os.isdir(root):
        st.error("❌ Please provide a valid dataset root directory to continue.", icon="📂")
        return

    config_data = self._load_configuration(root)

    # NEW: Add recording mode selection
    recording_mode = self._render_recording_mode_selection()

    # NEW: Show calibration-specific info if in calibration mode
    if recording_mode == 'calibration':
        self._render_calibration_mode_info()

    mode = self._render_mode_selection()  # Single Scenario vs Series
    common_cfg = self._render_common_configuration(config_data)

    # Pass recording_mode to collection methods
    if mode == "Single Scenario":
        self._render_single_scenario_mode(common_cfg, recording_mode)
    else:
        self._render_series_mode(common_cfg, recording_mode)

    self._render_post_collection_actions(config_data["config_file"])
```

**Effort:** 30 minutes

---

#### 3.2: Update Single Scenario Mode

**Modify `_render_single_scenario_mode()` signature:**

```python
def _render_single_scenario_mode(self, common_cfg: Dict[str, Any], recording_mode: str = 'standard') -> None:
    st.markdown("### Single Scenario Configuration")
    c1, c2 = st.columns([1, 1])
    with c1:
        scenario_number = st.text_input("Scenario number", value="1")
    with c2:
        description = st.text_input("Description", value=f"Room response measurement scenario {scenario_number}")

    # Add mode indicator to scenario name
    if recording_mode == 'calibration':
        scenario_suffix = "_CAL"
    else:
        scenario_suffix = ""

    if common_cfg["computer_name"] and common_cfg["room_name"] and scenario_number:
        scenario_name = f"{common_cfg['computer_name']}-Scenario{scenario_number}-{common_cfg['room_name']}{scenario_suffix}"
        scenario_path = os.path.join(common_cfg["output_dir"], scenario_name)
        st.info(f"📁 Scenario will be saved as: `{scenario_name}`")
        st.caption(f"📂 Full path: `{scenario_path}`")

    st.markdown("### Execute Collection")

    # Show mode-specific info
    if recording_mode == 'calibration':
        st.info("🔨 Calibration Mode: Each measurement will validate impact quality and save cycle data")

    if st.button("🎤 Start Single Scenario Collection", type="primary", use_container_width=True):
        print("/n/n++++++++++++ Debug output of the recorder parameters 4 ++++++++++++++")
        self.recorder.print_signal_analysis()

        SingleScenarioExecutor(self.scenario_manager, recorder=self.recorder).execute(
            common_config=common_cfg,
            scenario_number=scenario_number,
            description=description,
            recording_mode=recording_mode  # NEW
        )
```

**Effort:** 30 minutes

---

#### 3.3: Update `SingleScenarioExecutor`

**Modify `execute()` method:**

```python
class SingleScenarioExecutor:
    def __init__(self, scenario_manager, recorder: Optional["RoomResponseRecorder"]=None):
        self.scenario_manager = scenario_manager
        self.recorder = recorder

    def execute(self, common_config: Dict[str, Any], scenario_number: str,
                description: str, recording_mode: str = 'standard') -> None:  # NEW PARAM

        if not self._validate_inputs(common_config, scenario_number):
            return

        scenario_name = f"{common_config['computer_name']}-Scenario{scenario_number}-{common_config['room_name']}"

        # Add mode suffix to scenario name
        if recording_mode == 'calibration':
            scenario_name += "_CAL"

        self._show_collection_info(scenario_name, common_config, recording_mode)

        try:
            params = {
                "scenario_number": scenario_number.strip(),
                "description": description.strip(),
                "computer_name": common_config["computer_name"].strip(),
                "room_name": common_config["room_name"].strip(),
                "num_measurements": int(common_config["num_measurements"]),
                "measurement_interval": float(common_config["measurement_interval"]),
                "recording_mode": recording_mode,  # NEW
            }

            collector = SingleScenarioCollector(
                base_output_dir=common_config["output_dir"],
                recorder_config=common_config["config_file"],
                scenario_config=params,
                merge_mode="append",
                allow_config_mismatch=False,
                resume=True,
                recorder=self.recorder,
                recording_mode=recording_mode  # NEW
            )

            mode_str = "calibration" if recording_mode == 'calibration' else "standard"
            st.info(f"🎵 Collection started ({mode_str} mode, blocking). Monitor the console for progress.")

            collector.collect_scenario(interactive_devices=common_config["interactive_devices"], confirm_start=False)

            st.success(f"🎉 Successfully collected scenario: {scenario_name}")
            st.info(f"📂 Data saved to: {collector.scenario_dir}")

        except Exception as e:
            st.error(f"❌ Collection failed: {e}")
            st.info("Check the terminal/console for detailed error information.")

    def _show_collection_info(self, scenario_name: str, common_config: Dict[str, Any], recording_mode: str = 'standard') -> None:
        st.markdown("### 🎤 Starting Single Scenario Collection")
        st.text(f"Scenario: {scenario_name}")
        st.text(f"Mode: {'Calibration (Physical Impact)' if recording_mode == 'calibration' else 'Standard (Room Response)'}")
        st.text(f"Measurements: {common_config['num_measurements']}")
        st.text(f"Interval: {common_config['measurement_interval']}s")
        st.text(f"Interactive devices: {common_config['interactive_devices']}")
        st.text("Initializing...")
```

**Effort:** 1 hour

---

### Phase 4: Update DatasetCollector for Calibration Mode ⭐⭐⭐⭐⭐

**File:** `DatasetCollector.py`

#### 4.1: Add `recording_mode` to `SingleScenarioCollector.__init__()`

```python
class SingleScenarioCollector:
    def __init__(self,
                 base_output_dir: str,
                 recorder_config: str,
                 scenario_config: dict,
                 merge_mode: str = "append",
                 allow_config_mismatch: bool = False,
                 resume: bool = False,
                 recorder: Optional["RoomResponseRecorder"] = None,
                 recording_mode: str = 'standard'):  # NEW PARAMETER

        # ... existing init code ...

        self.recording_mode = recording_mode

        # ... rest of init ...
```

**Effort:** 5 minutes

---

#### 4.2: Update `collect_scenario_measurements()` to Handle Calibration Mode

**Modify measurement loop:**

```python
def collect_scenario_measurements(self) -> List[MeasurementMetadata]:
    if not self.scenario:
        raise ValueError("Scenario not configured")

    # ... existing setup code ...

    for i in range(num_measurements):
        # ... existing measurement setup ...

        raw_filename = f"raw_{i:03d}.wav"
        impulse_filename = f"impulse_{i:03d}.wav"
        raw_path = self.scenario_dir / "raw_recordings" / raw_filename
        impulse_path = self.scenario_dir / "impulse_responses" / impulse_filename

        try:
            if self.recording_mode == 'calibration':
                # CALIBRATION MODE: Use calibration recording with file saving
                print(f"\n{'='*60}")
                print(f"Calibration Mode Recording - Measurement {i+1}/{num_measurements}")
                print(f"{'='*60}")

                calibration_result = self.recorder.take_record(
                    str(raw_path),
                    str(impulse_path),
                    mode='calibration',
                    save_calibration_data=True  # ← ENABLE FILE SAVING
                )

                # Store calibration metadata in measurement metadata
                calibration_metadata = {
                    'mode': 'calibration',
                    'num_total_cycles': len(calibration_result['validation_results']),
                    'num_valid_cycles': calibration_result['num_valid_cycles'],
                    'num_aligned_cycles': calibration_result['num_aligned_cycles'],
                    'normalization_enabled': calibration_result['metadata']['normalize_by_calibration'],
                    'calibration_channel': calibration_result['metadata']['calibration_channel'],
                }

                # Add to measurement metadata
                meta.custom_metadata['calibration'] = calibration_metadata

                print(f"✓ Calibration measurement {i+1} completed")
                print(f"  Valid cycles: {calibration_result['num_valid_cycles']}/{len(calibration_result['validation_results'])}")
                print(f"  Aligned cycles: {calibration_result['num_aligned_cycles']}")

            else:
                # STANDARD MODE: Existing behavior
                audio_data = self.recorder.take_record(
                    str(raw_path),
                    str(impulse_path)
                )

            # ... rest of existing measurement handling ...

        except Exception as e:
            print(f"  ❌ Measurement {i+1} failed: {e}")
            # ... existing error handling ...

    # ... rest of existing code ...
    return measurements
```

**Effort:** 1-2 hours

---

### Phase 5: Add Series Mode Support ⭐⭐⭐⭐

**File:** `gui_collect_panel.py`

#### 5.1: Update Series Mode UI

**Modify `_render_series_mode()` signature:**

```python
def _render_series_mode(self, common_cfg: Dict[str, Any], recording_mode: str = 'standard') -> None:
    st.markdown("### Series Configuration")
    # ... existing series configuration UI ...

    # Show mode indicator
    if recording_mode == 'calibration':
        st.info("🔨 Calibration Mode: All measurements will use calibration recording with validation")

    parsed = self._preview_series(series_scenarios, common_cfg)
    # ... rest of existing code ...

    if start_clicked:
        if not parsed:
            st.error("❌ No valid scenarios to collect.")
        else:
            evt_q: queue.Queue = queue.Queue()
            cmd_q: queue.Queue = queue.Queue()
            worker = SeriesWorker(
                # ... existing parameters ...
                recording_mode=recording_mode,  # NEW
            )
            # ... rest of existing code ...
```

**Effort:** 30 minutes

---

#### 5.2: Update SeriesWorker

**File:** `gui_series_worker.py`

**Add `recording_mode` parameter:**

```python
class SeriesWorker(threading.Thread):
    def __init__(self,
                 scenario_numbers: List[str],
                 base_output_dir: str,
                 config_file: str,
                 # ... existing parameters ...
                 recording_mode: str = 'standard'):  # NEW

        super().__init__(daemon=True)
        # ... existing init ...
        self.recording_mode = recording_mode

    def run(self):
        # ... existing code ...

        for scenario_number in self.scenario_numbers:
            # ... existing scenario setup ...

            collector = SingleScenarioCollector(
                # ... existing parameters ...
                recording_mode=self.recording_mode  # NEW
            )

            # ... rest of existing code ...
```

**Effort:** 30 minutes

---

### Phase 6: Testing & Validation ⭐⭐⭐

#### 6.1: Test Cases

**Test 1: Single Scenario - Calibration Mode**
- Configure multi-channel with calibration sensor
- Select Calibration mode in Collect panel
- Run single scenario with 5 measurements
- Verify files created:
  - `impulse_NNN_aligned_TIMESTAMP_chN.wav` for each channel
  - `impulse_NNN_normalized_TIMESTAMP_chN.wav` (if normalization enabled)
  - `raw_NNN_metadata_TIMESTAMP.json` with validation results
- Verify metadata contains validation and alignment info

**Test 2: Single Scenario - Standard Mode**
- Same configuration
- Select Standard mode
- Verify standard mode files created (existing behavior)
- Ensure no calibration metadata in files

**Test 3: Series Mode - Calibration Mode**
- Configure series with 3 scenarios, 3 measurements each
- Use calibration mode
- Verify all files created correctly across scenarios
- Check SeriesWorker handles calibration mode properly

**Test 4: Backward Compatibility**
- Test Audio Settings → Calibration Impulse panel
- Verify existing calibration test still works
- Ensure no files are saved (existing behavior preserved)

**Test 5: Error Handling**
- Try calibration mode without calibration sensor configured
- Verify appropriate error message
- Ensure graceful fallback to standard mode

**Effort:** 2-3 hours

---

## Implementation Summary

### Total Estimated Effort: **12-16 hours**

| Phase | Description | Priority | Effort |
|-------|-------------|----------|--------|
| 1 | Extend RoomResponseRecorder | ⭐⭐⭐⭐⭐ Critical | 2-3h |
| 2 | Add mode selection to Collect Panel UI | ⭐⭐⭐⭐⭐ Critical | 2h |
| 3 | Integrate into collection workflow | ⭐⭐⭐⭐⭐ Critical | 2h |
| 4 | Update DatasetCollector | ⭐⭐⭐⭐⭐ Critical | 2h |
| 5 | Add Series mode support | ⭐⭐⭐⭐ High | 1h |
| 6 | Testing & validation | ⭐⭐⭐ Medium | 2-3h |

### Recommended Implementation Order:

1. **Phase 1** → Core recording capability with file saving
2. **Phase 4** → Data collection integration in DatasetCollector
3. **Phase 2** → UI for mode selection in Collect Panel
4. **Phase 3** → Connect UI to backend workflow
5. **Phase 5** → Series mode support
6. **Phase 6** → Comprehensive testing

---

## File Structure Changes

### New Files Created Per Measurement (Calibration Mode):

```
scenario_directory/
├── impulse_responses/
│   ├── impulse_000_aligned_20251102_143022_ch0.wav
│   ├── impulse_000_aligned_20251102_143022_ch1.wav
│   ├── impulse_000_aligned_20251102_143022_ch2.wav
│   ├── impulse_000_aligned_20251102_143022_ch3.wav
│   ├── impulse_000_normalized_20251102_143022_ch0.wav  (if normalization enabled)
│   ├── impulse_000_normalized_20251102_143022_ch1.wav
│   ├── impulse_000_normalized_20251102_143022_ch2.wav
│   └── impulse_000_normalized_20251102_143022_ch3.wav
├── raw_recordings/
│   └── raw_000_metadata_20251102_143022.json
└── metadata/
    └── measurement_000.json  (includes calibration metadata)
```

### Metadata Structure (JSON):

```json
{
  "timestamp": "20251102_143022",
  "recording_mode": "calibration",
  "sample_rate": 48000,
  "num_channels": 4,
  "channel_names": ["Hammer", "Front Mic", "Rear Mic", "Side Mic"],
  "calibration_channel": 0,
  "reference_channel": 1,
  "num_pulses": 10,
  "cycle_duration": 2.0,
  "cycle_samples": 96000,
  "num_total_cycles": 10,
  "num_valid_cycles": 8,
  "num_aligned_cycles": 8,
  "normalization_enabled": true,
  "validation_results": [
    {
      "cycle_index": 0,
      "is_valid": true,
      "calibration_metrics": {
        "negative_peak": 0.3542,
        "positive_peak": 0.1234,
        "aftershock": 0.0234
      }
    }
  ],
  "alignment_metadata": {
    "correlation_threshold": 0.7,
    "aligned_onset_position": 1024,
    "reference_cycle_idx": 0,
    "num_cycles_aligned": 8
  },
  "normalization_factors": [0.3542, 0.3451, 0.3623, ...],
  "normalization_stats": {
    "min": 0.3451,
    "max": 0.3623,
    "mean": 0.3539,
    "std": 0.0056
  }
}
```

---

## Key Benefits

✅ **Unified Collection Interface** - One panel for both Standard and Calibration datasets
✅ **Quality Assurance** - Per-cycle validation built into collection
✅ **Rich Metadata** - Validation results saved with each measurement
✅ **Normalization Support** - Impact-normalized data for quantitative analysis
✅ **Backward Compatible** - Existing Audio Settings calibration test unchanged
✅ **Series Support** - Large-scale calibration dataset collection
✅ **Flexible File Structure** - Aligned + normalized cycles saved separately
✅ **Traceability** - Complete configuration and validation history saved

---

## Backward Compatibility

### Audio Settings → Calibration Impulse Panel

**Preserved Behavior:**
- Calibration test continues to work exactly as before
- Does NOT save files (analysis/display only)
- Used for testing, threshold tuning, and quality validation
- Multi-Channel Response Review continues to work

**Implementation Detail:**
- Audio Settings panel calls `take_record_calibration()` which internally calls `take_record(mode='calibration', save_calibration_data=False)`
- `save_calibration_data=False` is the default, so no files are saved
- Collect Panel explicitly sets `save_calibration_data=True` to enable file saving

---

## Future Enhancements

### Phase 7: Advanced Features (Future)
- **Real-time Quality Metrics** - Display validation stats during collection
- **Adaptive Thresholds** - Auto-adjust quality thresholds based on initial measurements
- **Cycle Selection UI** - Allow user to review and exclude bad cycles during collection
- **Multi-Rate Recording** - Support different sample rates for different channels
- **Compression Options** - Optional compression for large datasets

### Phase 8: Analysis Tools (Future)
- **Dataset Viewer** - GUI panel for viewing calibration datasets
- **Cycle Comparison** - Compare cycles across measurements
- **Quality Trends** - Track validation metrics over time
- **Batch Processing** - Re-process existing calibration datasets with new thresholds

---

## Conclusion

This implementation plan enables full **Calibration Mode dataset collection** in the Collect Panel while maintaining complete backward compatibility with existing functionality. The phased approach ensures incremental progress with testable milestones at each stage.

**Critical Success Factors:**
1. File saving in calibration mode (Phase 1)
2. DatasetCollector integration (Phase 4)
3. UI mode selection (Phase 2-3)
4. Comprehensive testing (Phase 6)

**Next Steps:**
1. Review and approve plan
2. Begin Phase 1 implementation
3. Test each phase incrementally
4. Update documentation as features are completed
