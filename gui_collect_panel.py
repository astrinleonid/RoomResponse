#!/usr/bin/env python3
"""
GUI Collection Panel — DROP-IN REPLACEMENT (v6)

v6 changes:
- Removed dependency on collect_dataset
- Duplicated _parse_series_expr and _load_defaults_from_config as local methods
- Clean architecture with no cross-module dependencies
"""
from __future__ import annotations
import os
import time
import json
import queue
from typing import List, Dict, Any
import streamlit as st

# Back-end deps
try:
    from DatasetCollector import SingleScenarioCollector
except ImportError:
    SingleScenarioCollector = None  # type: ignore

try:
    from gui_series_worker import SeriesWorker, WorkerCommand, WorkerEvent
except Exception:
    SeriesWorker = None  # type: ignore
    WorkerCommand = None  # type: ignore
    WorkerEvent = None  # type: ignore

# near the imports (optional; type-only import so it won't create a hard dep)
from typing import Optional
try:
    from RoomResponseRecorder import RoomResponseRecorder  # type: ignore
except Exception:
    RoomResponseRecorder = None  # type: ignore

# Session keys
SK_DATASET_ROOT = "dataset_root"
SK_SERIES_EVT_Q = "series_event_q"
SK_SERIES_CMD_Q = "series_cmd_q"
SK_SERIES_THREAD = "series_thread"
SK_SERIES_LAST = "series_last_event"
SK_SERIES_STARTED_AT = "series_started_at"
SK_COLLECTION_OUTPUT_OVERRIDE = "collection_output_override"

class CollectionPanel:
    def __init__(self, scenario_manager, recorder: Optional["RoomResponseRecorder"]=None):
        self.scenario_manager = scenario_manager
        self.recorder: Optional[RoomResponseRecorder] = recorder
        if self.recorder:
            print("\n\n++++++++++++ Debug output of the recorder parameters 3 ++++++++++++++")
            self.recorder.print_signal_analysis()
        else:
            print("Warning: CollectionPanel initialized without recorder")

    def _render_recorder_status(self) -> None:
        """Display recorder configuration status, including multi-channel setup."""
        if self.recorder is None:
            return

        mc_config = getattr(self.recorder, 'multichannel_config', {})
        mc_enabled = mc_config.get('enabled', False)

        # Create expandable status section
        with st.expander("📊 Recorder Configuration", expanded=False):
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**Recording Mode**")
                if mc_enabled:
                    num_channels = mc_config.get('num_channels', 1)
                    st.success(f"✓ Multi-Channel ({num_channels} channels)")

                    # Show channel configuration
                    cal_ch = mc_config.get('calibration_channel')
                    ref_ch = mc_config.get('reference_channel', 0)

                    if cal_ch is not None:
                        st.write(f"🔨 Calibration: Ch {cal_ch}")
                    else:
                        st.write("🔨 Calibration: None")

                    st.write(f"🎤 Reference: Ch {ref_ch}")

                else:
                    st.info("Single-Channel Mode")

            with col2:
                st.markdown("**Recording Parameters**")
                st.write(f"Sample Rate: {getattr(self.recorder, 'sample_rate', 'N/A')} Hz")
                st.write(f"Pulses: {getattr(self.recorder, 'num_pulses', 'N/A')}")
                st.write(f"Cycle Duration: {getattr(self.recorder, 'cycle_duration', 'N/A')} s")

            # Show detailed channel info if multi-channel enabled
            if mc_enabled:
                st.markdown("---")
                st.markdown("**Channel Configuration**")

                channel_names = mc_config.get('channel_names', [])
                response_channels = mc_config.get('response_channels', [])

                # Create a table of channels
                channel_info = []
                for ch in range(num_channels):
                    name = channel_names[ch] if ch < len(channel_names) else f"Channel {ch}"

                    # Determine role
                    roles = []
                    if ch == cal_ch:
                        roles.append("Calibration")
                    if ch == ref_ch:
                        roles.append("Reference")
                    if ch in response_channels:
                        roles.append("Response")

                    role_str = ", ".join(roles) if roles else "—"

                    # Icon based on primary role
                    if ch == cal_ch:
                        icon = "🔨"
                    elif ch == ref_ch:
                        icon = "🎤"
                    else:
                        icon = "🔊"

                    channel_info.append(f"{icon} Ch {ch}: **{name}** ({role_str})")

                for info in channel_info:
                    st.write(info)

                # Link to configuration
                st.caption("💡 Configure channels in Audio Settings → Device Selection & Testing → Multi-Channel Configuration")

    def render(self) -> None:
        st.header("Collect - Data Collection")

        # Display recorder status including multi-channel configuration
        self._render_recorder_status()

        if not self._check_dependencies():
            return

        if self.recorder is None:
            st.error("Shared RoomResponseRecorder is not available. "
                         "Ensure piano_response.py instantiates a single recorder "
                         "and passes it into CollectionPanel(recorder=...).")
            return
        root = st.session_state.get(SK_DATASET_ROOT, os.getcwd())
        if not os.path.isdir(root):
            st.error("❌ Please provide a valid dataset root directory to continue.", icon="📂")
            return
        config_data = self._load_configuration(root)
        mode = self._render_mode_selection()
        common_cfg = self._render_common_configuration(config_data)
        if mode == "Single Scenario":
            self._render_single_scenario_mode(common_cfg)
        else:
            self._render_series_mode(common_cfg)
        self._render_post_collection_actions(config_data["config_file"])

    def _check_dependencies(self) -> bool:
        ok = True
        if SingleScenarioCollector is None:
            st.error("❌ DatasetCollector.SingleScenarioCollector not found.", icon="🔧"); ok = False
        if SeriesWorker is None:
            st.error("❌ gui_series_worker.SeriesWorker not available. Add gui_series_worker.py.", icon="🔧"); ok = False
        if not ok:
            st.info("Ensure DatasetCollector.py and gui_series_worker.py are importable.")
        return ok

    def _parse_series_expr(self, expr: str) -> List[str]:
        """
        Parse series string like "0.1,0.2,1-3,7a".
        - Numeric ranges (e.g., 1-3) expand to ['1','2','3']
        - Non-numeric tokens pass through as-is (e.g., '0.1','7a')
        """
        out: List[str] = []
        if not expr:
            return out
        for token in [t.strip() for t in expr.split(',') if t.strip()]:
            if '-' in token:
                a, b = token.split('-', 1)
                if a.replace('.', '', 1).isdigit() and b.replace('.', '', 1).isdigit():
                    # numeric range only
                    if '.' in a or '.' in b:
                        # float ranges are ambiguous; keep literal token
                        out.append(token)
                    else:
                        ai, bi = int(a), int(b)
                        step = 1 if bi >= ai else -1
                        out.extend([str(i) for i in range(ai, bi + step, step)])
                else:
                    out.append(token)
            else:
                out.append(token)
        # keep order, remove dups preserving first occurrence
        seen = set()
        uniq = []
        for v in out:
            if v not in seen:
                seen.add(v)
                uniq.append(v)
        return uniq

    def _load_defaults_from_config(self, cfg_path: str) -> dict:
        """Load computer, room name, and num_measurements defaults from recorder config file."""
        defaults = {"computer": "Unknown_Computer", "room": "Unknown_Room", "num_measurements": 0}
        try:
            with open(cfg_path, 'r', encoding='utf-8') as f:
                file_config = json.load(f)
            for k in defaults:
                if k in file_config:
                    defaults[k] = file_config[k]
            return defaults
        except Exception:
            return defaults

    def _save_config_values(self, cfg_path: str, computer: str, room: str, num_measurements: int) -> None:
        """Save computer, room, and num_measurements back to config file."""
        try:
            with open(cfg_path, 'r', encoding='utf-8') as f:
                config = json.load(f)

            config['computer'] = computer
            config['room'] = room
            config['num_measurements'] = num_measurements

            with open(cfg_path, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2)
        except Exception as e:
            st.warning(f"Could not save config: {e}")

    def _load_configuration(self, root: str) -> Dict[str, Any]:
        cfg_path = os.path.join(root, "recorderConfig.json")
        if not os.path.exists(cfg_path):
            cfg_path = "recorderConfig.json"
        defaults = self._load_defaults_from_config(cfg_path)
        return {"defaults": defaults, "config_file": cfg_path}

    def _render_mode_selection(self) -> str:
        st.markdown("### Collection Mode")

        # Check if mode was set by navigation
        nav_mode = st.session_state.get('collect_mode', None)
        if nav_mode:
            # Use the navigation-specified mode as default
            default_index = 0 if nav_mode == "Single Scenario" else 1
        else:
            default_index = 0

        # Callback to update mode immediately
        def on_mode_change():
            """Update collect_mode when user changes selection."""
            st.session_state['collect_mode'] = st.session_state.collect_mode_radio

        mode = st.radio(
            "Choose collection mode:",
            options=["Single Scenario", "Series"],
            index=default_index,
            key="collect_mode_radio",
            on_change=on_mode_change,
            help="Single: blocking. Series: background worker.",
        )

        return mode

    def _render_common_configuration(self, config_data: Dict[str, Any]) -> Dict[str, Any]:
        st.markdown("### Common Configuration")

        # Recording mode selection
        st.markdown("**Recording Mode**")
        mc_config = self.recorder.multichannel_config if self.recorder else {}
        mc_enabled = mc_config.get('enabled', False)
        cal_channel = mc_config.get('calibration_channel')

        recording_mode = st.radio(
            "Select recording mode:",
            options=["Standard", "Calibration"],
            index=1,  # Default to Calibration mode
            help="Standard: Simple averaged response. Calibration: Advanced per-cycle alignment with quality filtering.",
            horizontal=True
        )

        # Show warning if calibration selected without proper setup
        if recording_mode == "Calibration":
            if not mc_enabled:
                st.error("⚠️ Calibration mode requires multi-channel recording to be enabled.")
                st.info("Go to Audio Settings → Device Selection & Testing → Enable multi-channel recording")
            elif cal_channel is None:
                st.error("⚠️ Calibration mode requires a calibration channel to be configured.")
                st.info("Go to Audio Settings → Device Selection & Testing → Multi-Channel Configuration → Set calibration channel")
            else:
                st.success(f"✓ Calibration mode ready (Calibration channel: Ch {cal_channel})")

        st.markdown("---")

        c1, c2 = st.columns([1, 1])
        with c1:
            # Use callbacks to save config on change
            computer_name = st.text_input(
                "Computer name",
                value=config_data["defaults"].get("computer", "Unknown_Computer"),
                key="computer_name_input"
            )
            room_name = st.text_input(
                "Room name",
                value=config_data["defaults"].get("room", "Unknown_Room"),
                key="room_name_input"
            )
            num_measurements = st.number_input(
                "Number of measurements",
                min_value=0,
                max_value=1000,
                value=config_data["defaults"].get("num_measurements", 0),
                step=1,
                key="num_measurements_input"
            )

            # Save to config if any values changed
            if (computer_name != config_data["defaults"].get("computer") or
                room_name != config_data["defaults"].get("room") or
                num_measurements != config_data["defaults"].get("num_measurements")):
                self._save_config_values(config_data["config_file"], computer_name, room_name, num_measurements)
        with c2:
            measurement_interval = st.number_input("Measurement interval (seconds)", 0.1, 60.0, 2.0, 0.1)
            interactive_devices = st.checkbox(
                "Interactive device selection (single scenario only)", value=False,
                help="Series uses fixed devices in a background worker.")
            config_file = st.text_input("Config file path", value=config_data["config_file"])

        st.markdown("### Output Directory")
        oc1, oc2 = st.columns([3, 1])
        with oc1:
            default_output = st.session_state.get(SK_DATASET_ROOT, os.getcwd())
            output_dir = st.text_input(
                "Collection output directory",
                value=st.session_state.get(SK_COLLECTION_OUTPUT_OVERRIDE, default_output),
                help="Can differ from dataset root.")
            output_dir = (output_dir or default_output).strip()
            try:
                resolved = os.path.abspath(output_dir)
                if os.path.isdir(resolved):
                    st.success(f"✓ Output directory: {resolved}", icon="📂")
                elif os.path.exists(resolved):
                    st.error("⚠️ Path exists but is not a directory", icon="📂")
                else:
                    st.warning(f"⚠️ Directory will be created: {resolved}", icon="📂")
            except Exception as e:
                st.error(f"Invalid path: {e}"); resolved = default_output
        with oc2:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("📂 Use Dataset Root"):
                st.session_state[SK_COLLECTION_OUTPUT_OVERRIDE] = st.session_state.get(SK_DATASET_ROOT, os.getcwd()); st.rerun()
        st.session_state[SK_COLLECTION_OUTPUT_OVERRIDE] = resolved

        # Convert recording mode to lowercase for API compatibility
        mode_param = recording_mode.lower()  # "Standard" -> "standard", "Calibration" -> "calibration"

        return {
            "computer_name": computer_name,
            "room_name": room_name,
            "num_measurements": int(num_measurements),
            "measurement_interval": float(measurement_interval),
            "interactive_devices": bool(interactive_devices),
            "config_file": config_file,
            "output_dir": resolved,
            "recording_mode": mode_param,
        }

    def _render_single_scenario_mode(self, common_cfg: Dict[str, Any]) -> None:
        st.markdown("### Single Scenario Configuration")
        c1, c2 = st.columns([1, 1])
        with c1:
            scenario_number = st.text_input("Scenario number", value="1")
        with c2:
            description = st.text_input("Description", value=f"Room response measurement scenario {scenario_number}")
        if common_cfg["computer_name"] and common_cfg["room_name"] and scenario_number:
            scenario_name = f"{common_cfg['computer_name']}-Scenario{scenario_number}-{common_cfg['room_name']}"
            scenario_path = os.path.join(common_cfg["output_dir"], scenario_name)
            st.info(f"📁 Scenario will be saved as: `{scenario_name}`"); st.caption(f"📂 Full path: `{scenario_path}`")
        st.markdown("### Execute Collection")

        # Check if collection is running
        is_running = st.session_state.get("single_thread") is not None and st.session_state.get("single_thread").is_alive()

        # Start button (disabled if already running)
        if st.button("🎤 Start Single Scenario Collection", type="primary", use_container_width=True, disabled=is_running):

            print("/n/n++++++++++++ Debug output of the recorder parameters 4 ++++++++++++++")
            self.recorder.print_signal_analysis()

            SingleScenarioExecutor(self.scenario_manager, recorder=self.recorder).execute(
                                    common_config = common_cfg,
                                    scenario_number = scenario_number,
                                    description = description
            )
            # Force a rerun to show the controls
            st.rerun()

        # Control buttons (only visible when collection is running)
        if is_running:
            st.markdown("---")
            st.markdown("### Collection Controls")
            col1, col2, col3 = st.columns(3)

            with col1:
                if st.button("⏸️ Pause", use_container_width=True):
                    if st.session_state.get("single_cmd_q"):
                        st.session_state["single_cmd_q"].put(WorkerCommand("pause"))

            with col2:
                if st.button("▶️ Resume", use_container_width=True):
                    if st.session_state.get("single_cmd_q"):
                        st.session_state["single_cmd_q"].put(WorkerCommand("resume"))

            with col3:
                if st.button("⏹️ Stop", type="secondary", use_container_width=True):
                    if st.session_state.get("single_cmd_q"):
                        st.session_state["single_cmd_q"].put(WorkerCommand("stop"))

            # Render status display
            self._render_single_scenario_status()
    def _render_series_mode(self, common_cfg: Dict[str, Any]) -> None:
        st.markdown("### Series Configuration")
        c1, c2 = st.columns([1, 1])
        with c1:
            series_scenarios = st.text_input("Scenario numbers", value="1,2,3",
                help="Comma list; supports ranges like 1-3; tokens like 0.1 or 7a are passed as-is.")
            pre_delay = st.number_input("Pre-delay (seconds)", 0.0, 600.0, 60.0, 5.0)
            inter_delay = st.number_input("Inter-scenario delay (seconds)", 0.0, 600.0, 60.0, 5.0)
        with c2:
            description_template = st.text_input("Description template", value="Room response measurement scenario {n}")
            st.caption("Note: Keyboard 'p' is not captured in the browser. Use the Pause/Resume buttons or the PAUSE file.")
            with st.expander("Beeps"):
                enable_beeps = st.checkbox("Enable beeps (one per scenario, double at end)", value=True)
                b1, b2 = st.columns(2)
                with b1:
                    beep_freq = st.number_input("Beep frequency (Hz)", 200, 4000, 880, 10)
                    beep_volume = st.slider("Beep volume", 0.0, 1.0, 0.2, 0.05)
                with b2:
                    beep_dur = st.number_input("Beep duration (ms)", 50, 1000, 200, 10)
            with st.expander("Advanced timing"):
                default_max_rec = max(10.0, 4.0 * float(common_cfg["measurement_interval"]))
                max_record_time = st.number_input("Max record time per measurement (s)", 2.0, 120.0, float(default_max_rec), 0.5,
                                                  help="Watchdog timeout to recover from a stuck recording. Lower = faster Stop/Pause responsiveness.")
                interval_mode_label = st.selectbox(
                    "Interval mode",
                    ["End→Start (minimum rest)", "Start→Start (fixed cadence)"],
                    index=0,
                    help="End→Start ensures cooldown ≥ interval after each measurement. Start→Start keeps cadence when possible.")
                interval_mode = "end_to_start" if interval_mode_label.startswith("End") else "start_to_start"
                warmup_n = st.number_input("Warm‑up measurements", 0, 10, 0, 1,
                                           help="Runs before the first real scenario to prime devices. Default 0. If >0, choose spacing below.")
                warmup_spacing_label = st.selectbox("Warm‑up spacing", ["Burst (fast)", "Respect interval"], index=0,
                                                    help="Burst = very fast (audible burst). Respect interval = waits the configured interval between warm‑ups.")
                warmup_spacing = "burst" if warmup_spacing_label.startswith("Burst") else "respect_interval"

        parsed = self._preview_series(series_scenarios, common_cfg)
        if parsed:
            st.caption(f"📁 All scenarios will be saved to: `{common_cfg['output_dir']}`")

        st.markdown("### Execute Series (Background)")
        btn_cols = st.columns([2, 2, 2, 2])
        with btn_cols[0]:
            start_clicked = st.button("🎤 Start Series", type="primary", use_container_width=True)
        with btn_cols[1]:
            pause_clicked = st.button("⏸️ Pause", use_container_width=True)
        with btn_cols[2]:
            resume_clicked = st.button("▶️ Resume", use_container_width=True)
        with btn_cols[3]:
            stop_clicked = st.button("🛑 Stop", use_container_width=True)

        if start_clicked:
            if not parsed:
                st.error("❌ No valid scenarios to collect.")
            else:
                evt_q: queue.Queue = queue.Queue(); cmd_q: queue.Queue = queue.Queue()
                worker = SeriesWorker(
                    scenario_numbers=parsed,
                    base_output_dir=common_cfg["output_dir"],
                    config_file=common_cfg["config_file"],
                    base_computer=common_cfg["computer_name"].strip(),
                    base_room=common_cfg["room_name"].strip(),
                    num_measurements=common_cfg["num_measurements"],
                    measurement_interval=common_cfg["measurement_interval"],
                    description_template=description_template,
                    warm_up_measurements=int(warmup_n),
                    warmup_spacing=warmup_spacing,
                    inter_delay=float(inter_delay),
                    pre_delay=float(pre_delay),
                    event_q=evt_q,
                    cmd_q=cmd_q,
                    enable_beeps=enable_beeps,
                    beep_volume=float(beep_volume),
                    beep_freq=int(beep_freq),
                    beep_dur_ms=int(beep_dur),
                    record_timeout_s=float(max_record_time),
                    interval_mode=interval_mode,
                    recording_mode=common_cfg["recording_mode"],
                )
                st.session_state[SK_SERIES_EVT_Q] = evt_q
                st.session_state[SK_SERIES_CMD_Q] = cmd_q
                st.session_state[SK_SERIES_THREAD] = worker
                st.session_state[SK_SERIES_STARTED_AT] = time.time()
                worker.start(); st.rerun()

        if pause_clicked and st.session_state.get(SK_SERIES_CMD_Q):
            st.session_state[SK_SERIES_CMD_Q].put(WorkerCommand("pause"))
        if resume_clicked and st.session_state.get(SK_SERIES_CMD_Q):
            st.session_state[SK_SERIES_CMD_Q].put(WorkerCommand("resume"))
        if stop_clicked and st.session_state.get(SK_SERIES_CMD_Q):
            st.session_state[SK_SERIES_CMD_Q].put(WorkerCommand("stop"))

        self._render_series_status()

    def _preview_series(self, expr: str, common_cfg: Dict[str, Any]) -> List[str]:
        if not expr:
            return []
        try:
            parsed = self._parse_series_expr(expr)
        except Exception as e:
            st.error(f"❌ Error parsing series: {e}"); return []
        if parsed:
            total_meas = len(parsed) * int(common_cfg["num_measurements"])
            total_time_s = total_meas * float(common_cfg["measurement_interval"])  # + delays shown separately
            st.info(f"📋 Series will collect {len(parsed)} scenarios: {', '.join(parsed)}")
            st.caption(f"⏱️ Measurement-only time (no delays): {total_time_s/60.0:.1f} minutes; total measurements: {total_meas}")
        else:
            st.warning("⚠️ No valid scenarios parsed from the input.")
        return parsed

    def _render_series_status(self) -> None:
        worker = st.session_state.get(SK_SERIES_THREAD)
        if worker is None:
            return
        st.markdown("---"); st.subheader("Series Status")
        evt_q: queue.Queue = st.session_state.get(SK_SERIES_EVT_Q)
        last_event = st.session_state.get(SK_SERIES_LAST)
        try:
            while True:
                ev = evt_q.get_nowait(); last_event = ev
        except queue.Empty:
            pass
        st.session_state[SK_SERIES_LAST] = last_event
        if last_event:
            st.write(f"**State:** {last_event.payload.get('state')} | **Event:** {last_event.kind}")
            msg = last_event.payload.get("message");
            if msg: st.caption(msg)
            if last_event.kind == "progress":
                st.write(
                    f"Scenario: {last_event.payload.get('scenario')} | "
                    f"Meas: {last_event.payload.get('local_index')}/"
                    f"{last_event.payload.get('total_per_scenario')} "
                    f"(abs {last_event.payload.get('absolute_index')})"
                )
                st.caption(
                    f"Record: {last_event.payload.get('record_ms')} ms | "
                    f"I/O: {last_event.payload.get('io_ms')} ms"
                )
            if last_event.kind == "error":
                st.error(last_event.payload.get("message"))
            if last_event.kind == "done":
                ok = last_event.payload.get("ok", False)
                st.success("Series complete" if ok else f"Series ended: {last_event.payload.get('reason')}")
        # Auto-refresh (1 Hz)
        try:
            if worker.is_alive():
                last_refresh = st.session_state.get("_series_last_refresh_ts", 0.0)
                now = time.time()
                if now - last_refresh > 1.0:
                    st.session_state["_series_last_refresh_ts"] = now
                    st.rerun()
        except Exception:
            st.caption("(auto-refresh skipped)")

    def _render_single_scenario_status(self) -> None:
        """Render status display for single scenario collection."""
        thread = st.session_state.get("single_thread")
        if thread is None:
            return

        st.markdown("---")
        st.markdown("### Collection Status")

        # Show last update time to verify refreshes are happening
        import datetime
        st.caption(f"Last update: {datetime.datetime.now().strftime('%H:%M:%S.%f')[:-3]}")

        evt_q: queue.Queue = st.session_state.get("single_evt_q")

        # Track separate last events by type
        last_progress = st.session_state.get("single_last_progress")
        last_status = st.session_state.get("single_last_status")
        last_error = st.session_state.get("single_last_error")
        last_done = st.session_state.get("single_last_done")

        # Drain all events from queue and categorize them
        if evt_q:
            try:
                while True:
                    ev = evt_q.get_nowait()
                    if ev.kind == "progress":
                        last_progress = ev
                    elif ev.kind == "status":
                        last_status = ev
                    elif ev.kind == "error":
                        last_error = ev
                    elif ev.kind == "done":
                        last_done = ev
            except queue.Empty:
                pass

        # Store categorized events
        st.session_state["single_last_progress"] = last_progress
        st.session_state["single_last_status"] = last_status
        st.session_state["single_last_error"] = last_error
        st.session_state["single_last_done"] = last_done

        # DEBUG: Print event details
        if last_progress:
            print(f"DEBUG UI: Progress event - payload keys={list(last_progress.payload.keys())}")
            print(f"DEBUG UI: Progress payload: {last_progress.payload}")
            st.caption(f"DEBUG: Got progress event with keys: {list(last_progress.payload.keys())}")
        if last_status:
            print(f"DEBUG UI: Status event - message={last_status.payload.get('message')}")

        # DEBUG: Show what events we have
        st.caption(f"DEBUG: Events - Progress: {last_progress is not None}, Status: {last_status is not None}, Error: {last_error is not None}, Done: {last_done is not None}")

        # Display progress information (always show if available)
        if last_progress:
            # Display progress information
            scenario = last_progress.payload.get('scenario', 'Unknown')
            local_idx = last_progress.payload.get('local_index', 0)
            total = last_progress.payload.get('total_measurements', 0)
            successful = last_progress.payload.get('successful_measurements', 0)
            failed = last_progress.payload.get('failed_measurements', 0)

            st.write(f"**Scenario:** {scenario}")
            st.write(f"**Progress:** {local_idx}/{total} measurements")
            st.write(f"**Successful:** {successful} | **Failed:** {failed}")

            # Show calibration info if available
            if 'valid_cycles' in last_progress.payload:
                valid = last_progress.payload.get('valid_cycles', 0)
                total_cycles = last_progress.payload.get('total_cycles', 0)
                aligned = last_progress.payload.get('aligned_cycles', 0)

                # Show per-measurement stats
                st.write(f"**Last Measurement:** {valid}/{total_cycles} valid cycles, {aligned} aligned")

                # Show cumulative totals if available
                if 'cumulative_valid_cycles' in last_progress.payload:
                    cum_valid = last_progress.payload.get('cumulative_valid_cycles', 0)
                    cum_total = last_progress.payload.get('cumulative_total_cycles', 0)
                    st.write(f"**Total (All Measurements):** {cum_valid}/{cum_total} valid cycles")

        # Display status message (if available)
        if last_status:
            msg = last_status.payload.get("message", "")
            st.info(f"📊 Status: {msg}")

        # Display error (if available)
        if last_error:
            msg = last_error.payload.get("message", "Unknown error")
            fatal = last_error.payload.get("fatal", False)
            if fatal:
                st.error(f"❌ Fatal Error: {msg}")
            else:
                st.warning(f"⚠️ Error: {msg}")

        # Display done message (if available)
        if last_done:
            ok = last_done.payload.get("ok", False)
            if ok:
                scenario = last_done.payload.get("scenario", "Unknown")
                st.success(f"🎉 Collection complete: {scenario}")
            else:
                reason = last_done.payload.get("reason", "Unknown reason")
                st.warning(f"⚠️ Collection ended: {reason}")

        # Auto-refresh while thread is alive (1 Hz)
        try:
            is_alive = thread.is_alive()
            print(f"DEBUG: Thread alive check: {is_alive}")
            st.caption(f"DEBUG: Thread is_alive = {is_alive}")

            if is_alive:
                last_refresh = st.session_state.get("_single_last_refresh_ts", 0.0)
                now = time.time()
                elapsed = now - last_refresh
                print(f"DEBUG: Auto-refresh check - elapsed={elapsed:.2f}s, threshold=1.0s")
                st.caption(f"DEBUG: Time since last refresh: {elapsed:.2f}s")

                if elapsed > 1.0:
                    st.session_state["_single_last_refresh_ts"] = now
                    print(f"DEBUG: *** TRIGGERING RERUN at {now} ***")
                    st.rerun()
                else:
                    print(f"DEBUG: Not yet time to refresh (need {1.0 - elapsed:.2f}s more)")
            else:
                print(f"DEBUG: Thread not alive, skipping auto-refresh")
                st.caption("DEBUG: Thread not alive")
        except Exception as e:
            st.caption(f"(auto-refresh skipped: {e})")
            print(f"DEBUG: Auto-refresh exception: {e}")
            import traceback
            traceback.print_exc()

    def _render_post_collection_actions(self, config_file: str) -> None:
        st.markdown("### Post-Collection")
        c1, c2 = st.columns([1, 1])
        with c1:
            if st.button("🔄 Refresh Scenarios List"):
                try: self.scenario_manager.clear_cache()
                except Exception: pass
                st.rerun()
        with c2:
            if st.button("💾 Save Names as Defaults"):
                st.info("Use the Collect form values; defaults are loaded from recorderConfig.json.")

class SingleScenarioExecutor:
    def __init__(self, scenario_manager, recorder: Optional["RoomResponseRecorder"]=None):
        self.scenario_manager = scenario_manager
        self.recorder = recorder

    def execute(self, common_config: Dict[str, Any], scenario_number: str, description: str) -> None:
        if not self._validate_inputs(common_config, scenario_number):
            return
        scenario_name = f"{common_config['computer_name']}-Scenario{scenario_number}-{common_config['room_name']}"
        self._show_collection_info(scenario_name, common_config)
        try:
            params = {
                "scenario_number": scenario_number.strip(),
                "description": description.strip(),
                "computer_name": common_config["computer_name"].strip(),
                "room_name": common_config["room_name"].strip(),
                "num_measurements": int(common_config["num_measurements"]),
                "measurement_interval": float(common_config["measurement_interval"]),
            }

            # Create event and command queues
            import queue
            evt_q = queue.Queue()
            cmd_q = queue.Queue()

            # Store queues in session state
            st.session_state["single_evt_q"] = evt_q
            st.session_state["single_cmd_q"] = cmd_q

            collector = SingleScenarioCollector(
                base_output_dir=common_config["output_dir"],
                recorder_config=common_config["config_file"],
                scenario_config=params,
                merge_mode="append", allow_config_mismatch=False, resume=True,
                recorder=self.recorder,
                recording_mode=common_config["recording_mode"],
                event_q=evt_q,
                cmd_q=cmd_q
            )

            # Run in background thread
            import threading
            def _run_collection():
                try:
                    collector.collect_scenario(interactive_devices=common_config["interactive_devices"], confirm_start=False)
                    evt_q.put_nowait(WorkerEvent("done", {"ok": True, "scenario": scenario_name}))
                except Exception as e:
                    evt_q.put_nowait(WorkerEvent("error", {"message": f"Collection failed: {e}", "fatal": True}))
                    evt_q.put_nowait(WorkerEvent("done", {"ok": False, "reason": str(e)}))

            thread = threading.Thread(target=_run_collection, daemon=True)
            st.session_state["single_thread"] = thread
            thread.start()

            st.info("🎵 Collection started. Use the controls below to monitor progress.")

        except Exception as e:
            st.error(f"❌ Failed to start collection: {e}")
            st.info("Check the terminal/console for detailed error information.")
    def _validate_inputs(self, common_config: Dict[str, Any], scenario_number: str) -> bool:
        if not common_config["computer_name"].strip() or not common_config["room_name"].strip() or not scenario_number.strip():
            st.error("❌ Computer name, room name, and scenario number are required."); return False
        return True
    def _show_collection_info(self, scenario_name: str, common_config: Dict[str, Any]) -> None:
        st.markdown("### 🎤 Starting Single Scenario Collection")
        st.text(f"Scenario: {scenario_name}")
        st.text(f"Measurements: {common_config['num_measurements']}")
        st.text(f"Interval: {common_config['measurement_interval']}s")
        st.text(f"Interactive devices: {common_config['interactive_devices']}")
        st.text("Initializing...")