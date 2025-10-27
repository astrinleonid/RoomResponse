#!/usr/bin/env python3
"""
Audio Settings Panel (System Info + Device Selection + Series Settings)

- No direct imports of sdl_audio_core.
- SDL readiness is inferred from recorder.get_sdl_core_info() (or can be passed in).
- Device selection uses the modular selector and writes into the shared recorder.
- Series Settings tab is included (delegates to SeriesSettingsPanel).
"""

from __future__ import annotations

import os
import sys
from typing import Dict, Optional, Any

# Streamlit imports
try:
    import streamlit as st
except ImportError:
    # Lightweight stub for environments without Streamlit (unit tests, etc.)
    class _Stub:
        def __getattr__(self, name):
            def _f(*a, **k):
                print(f"[STUB streamlit.{name}] args={a} kwargs={k}")
            return _f
    st = _Stub()

# Recorder import
try:
    from RoomResponseRecorder import RoomResponseRecorder
    RECORDER_AVAILABLE = True
except ImportError:
    RoomResponseRecorder = None  # type: ignore
    RECORDER_AVAILABLE = False

# Device selector (modular)
try:
    from gui_audio_device_selector import AudioDeviceSelector
    DEVICE_SELECTOR_AVAILABLE = True
except ImportError:
    AudioDeviceSelector = None  # type: ignore
    DEVICE_SELECTOR_AVAILABLE = False

# Series settings panel (modular)
try:
    from gui_series_settings_panel import SeriesSettingsPanel
    SERIES_SETTINGS_AVAILABLE = True
except ImportError:
    SeriesSettingsPanel = None  # type: ignore
    SERIES_SETTINGS_AVAILABLE = False


class AudioSettingsPanel:
    """Audio configuration panel with system-wide recorder management (SDL-free)."""

    def __init__(
        self,
        scenario_manager=None,
        recorder: Optional[RoomResponseRecorder] = None,
        sdl_ready: Optional[bool] = None,
    ):
        """
        Initialize with optional shared recorder instance.

        Args:
            scenario_manager: Optional scenario manager instance
            recorder: Optional shared RoomResponseRecorder instance
            sdl_ready: Optional explicit SDL readiness flag. If None, inferred from recorder.get_sdl_core_info()
        """
        self.scenario_manager = scenario_manager
        self.recorder: Optional[RoomResponseRecorder] = (
            recorder if recorder is not None else (RoomResponseRecorder() if RECORDER_AVAILABLE else None)
        )

        # Derive SDL readiness from the recorder snapshot unless explicitly provided
        if sdl_ready is not None:
            self.sdl_ready = bool(sdl_ready)
        else:
            self.sdl_ready = False
            if self.recorder and hasattr(self.recorder, "get_sdl_core_info"):
                try:
                    core = self.recorder.get_sdl_core_info()
                    self.sdl_ready = bool(core.get("sdl_available", False))
                except Exception:
                    self.sdl_ready = False

        # Child components
        self._device_selector = AudioDeviceSelector(self.recorder) if DEVICE_SELECTOR_AVAILABLE else None
        self._series_settings_panel = SeriesSettingsPanel(self.recorder) if SERIES_SETTINGS_AVAILABLE else None

        # Local state for diagnostics (e.g., last device scan summary)
        self._diagnostics: Dict[str, Any] = {}

    # -------------------------
    # Public entry point
    # -------------------------

    def render(self):
        st.header("Audio Settings")

        # Top status (recorder + SDL readiness)
        self._render_recorder_status()

        if not self.recorder:
            st.warning("Audio functionality will be limited without RoomResponseRecorder.")
            return

        # Initialize session state and sync with recorder
        self._init_session_state()
        self._sync_session_state_with_recorder()

        # Quick status bar (IDs + derived latency from UI buffer)
        self._render_audio_status_bar()

        # Tabs: System Info + Device Selection + Calibration Impulse (+ Series Settings when available)
        if SERIES_SETTINGS_AVAILABLE and self._series_settings_panel:
            tab1, tab2, tab3, tab4 = st.tabs([
                "System Info",
                "Device Selection",
                "Calibration Impulse",
                "Series Settings"
            ])
        else:
            tab1, tab2, tab3 = st.tabs([
                "System Info",
                "Device Selection",
                "Calibration Impulse"
            ])
            tab4 = None

        with tab1:
            self._render_system_info()
            self._render_paths_and_modules()

        with tab2:
            self._render_device_selection_tab()

        with tab3:
            self._render_calibration_impulse_tab()

        if tab4:
            with tab4:
                self._render_series_settings_tab()

    # -------------------------
    # Internal helpers
    # -------------------------

    def _render_recorder_status(self):
        """Top status panel for recorder and SDL."""
        cols = st.columns(3)

        with cols[0]:
            st.markdown("**Recorder**")
            if self.recorder:
                st.success("RoomResponseRecorder ready")
            elif RECORDER_AVAILABLE:
                st.warning("Recorder module available but instance not created")
            else:
                st.error("Recorder module missing")

        with cols[1]:
            st.markdown("**SDL Core**")
            if self.sdl_ready:
                st.success("SDL Ready")
            else:
                st.error("SDL Missing")

        with cols[2]:
            st.markdown("**Drivers**")
            if self.recorder:
                try:
                    core = self.recorder.get_sdl_core_info()
                    drivers = core.get("drivers", []) or []
                    if drivers:
                        st.info(f"{len(drivers)} drivers")
                    else:
                        st.warning("No drivers found")
                except Exception as e:
                    st.warning(f"Drivers unavailable: {e}")
            else:
                st.info("N/A")

        # Recorder creation helper if missing
        if not self.recorder and RECORDER_AVAILABLE:
            if st.button("Create Recorder Instance"):
                self.recorder = RoomResponseRecorder()
                if DEVICE_SELECTOR_AVAILABLE:
                    self._device_selector = AudioDeviceSelector(self.recorder)
                if SERIES_SETTINGS_AVAILABLE:
                    self._series_settings_panel = SeriesSettingsPanel(self.recorder)
                st.success("Recorder instance created")

        if not RECORDER_AVAILABLE:
            st.error("❌ RoomResponseRecorder module not available")
            with st.expander("Installation Required"):
                st.markdown("""
                **Missing RoomResponseRecorder module**

                Please ensure RoomResponseRecorder.py is present in your project directory.
                This module is required for audio recording functionality.
                """)
        elif self.recorder is None:
            st.warning("Recorder not created")

    def _init_session_state(self):
        """Initialize session state variables from recorder if available."""
        defaults = {
            'sample_rate': 48000,
            'buffer_size': 512,  # UI-only, not stored in recorder
            'input_device': -1,
            'output_device': -1,
        }

        if self.recorder:
            defaults.update({
                'sample_rate': int(getattr(self.recorder, 'sample_rate', defaults['sample_rate'])),
                'input_device': int(getattr(self.recorder, 'input_device', defaults['input_device'])),
                'output_device': int(getattr(self.recorder, 'output_device', defaults['output_device'])),
            })

        session_defaults = {
            'audio_selected_input_device': 'System Default',
            'audio_selected_output_device': 'System Default',
            'audio_sample_rate': defaults['sample_rate'],
            'audio_buffer_size': defaults['buffer_size'],
        }

        for k, v in session_defaults.items():
            if k not in st.session_state:
                st.session_state[k] = v

        if 'audio_device_cache' not in st.session_state:
            st.session_state['audio_device_cache'] = {
                'input': [],
                'output': [],
            }

    def _sync_session_state_with_recorder(self):
        """Push/prioritize settings from recorder into session state where necessary."""
        if not self.recorder:
            return
        st.session_state['audio_sample_rate'] = int(getattr(self.recorder, 'sample_rate', 48000))

    def _render_audio_status_bar(self):
        """Render a status bar with key parameters (no direct SDL)."""
        col1, col2, col3 = st.columns([1, 1, 1])

        with col1:
            if self.sdl_ready:
                st.success("SDL Ready")
            else:
                st.error("SDL Missing")

        with col2:
            if self.recorder:
                inp = int(getattr(self.recorder, 'input_device', -1))
                out = int(getattr(self.recorder, 'output_device', -1))
                st.info(f"In: {'Default' if inp == -1 else f'ID {inp}'} | Out: {'Default' if out == -1 else f'ID {out}'}")
            else:
                st.info("In/Out: N/A")

        with col3:
            sr = int(st.session_state.get('audio_sample_rate', 48000))
            bs = int(st.session_state.get('audio_buffer_size', 512))
            latency_ms = (bs / max(1, sr)) * 1000.0
            st.info(f"{sr//1000}kHz / ~{latency_ms:.0f}ms")

    # -------------------------
    # Tabs
    # -------------------------

    def _render_system_info(self):
        """Display system information via recorder snapshot (no SDL calls here)."""
        st.subheader("System Information")

        if not self.recorder:
            st.warning("Recorder not available")
            return

        try:
            core = self.recorder.get_sdl_core_info()
        except Exception as e:
            st.error(f"Failed to query recorder for SDL info: {e}")
            return

        # SDL/Core versions
        col1, col2 = st.columns([1, 1])
        with col1:
            st.markdown("**SDL Core**")
            mod_ver = core.get("module_version")
            sdl_ver = core.get("sdl_version")

            if mod_ver:
                st.success(f"Module Version: {mod_ver}")
            else:
                st.warning("Module Version: unavailable")

            if sdl_ver:
                st.info(f"SDL Version: {sdl_ver}")
            else:
                st.warning("SDL Version: unavailable")

        with col2:
            st.markdown("**Binding Info**")
            st.caption("Centralized via RoomResponseRecorder.get_sdl_core_info()")

        # Driver information
        st.markdown("---")
        st.markdown("**Audio Drivers**")
        drivers = core.get("drivers", []) or []
        st.success(f"Available drivers: {len(drivers)}")
        if drivers:
            st.caption(", ".join(drivers))

        # Device counts
        st.markdown("---")
        st.markdown("**Audio Devices**")
        counts = core.get("device_counts", {"input": 0, "output": 0, "total": 0})
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Input Devices", counts.get("input", 0))
        with col2:
            st.metric("Output Devices", counts.get("output", 0))
        with col3:
            st.metric("Total", counts.get("total", 0))

        # Optional engine stats if present
        stats = core.get("engine_stats")
        if stats:
            st.markdown("---")
            st.markdown("**Engine Stats**")
            for k, v in stats.items():
                st.write(f"**{k}:** {v}")

        # Diagnostics when SDL is not ready
        if not self.sdl_ready:
            errors = core.get("errors") or {}
            if errors:
                with st.expander("Diagnostics", expanded=False):
                    for k, v in errors.items():
                        st.write(f"**{k}:** {v}")

    def _render_paths_and_modules(self):
        """Debug helpers: module path, presence, and import tests."""
        st.subheader("Paths & Modules")
        with st.expander("Python and Modules"):
            st.code(f"Python: {sys.version}")
            st.code(f"cwd: {os.getcwd()}")

            # RoomResponseRecorder presence
            st.write("**RoomResponseRecorder module**")
            if RECORDER_AVAILABLE:
                st.success("Module import OK")
            else:
                st.error("Module import FAILED")

            if st.button("Inspect Recorder Class"):
                try:
                    if RoomResponseRecorder:
                        st.write(str(RoomResponseRecorder))
                        if hasattr(RoomResponseRecorder, '__init__'):
                            st.code(RoomResponseRecorder.__init__.__doc__)
                    else:
                        st.warning("Recorder class unavailable")
                except Exception as e:
                    st.error(f"Introspection error: {e}")

            # Utilities
            with st.expander("Utilities"):
                # Test import manually
                if st.button("Test RoomResponseRecorder Import"):
                    try:
                        if 'RoomResponseRecorder' in sys.modules:
                            del sys.modules['RoomResponseRecorder']
                        from RoomResponseRecorder import RoomResponseRecorder as TestRecorder  # reimport
                        test_recorder = TestRecorder()
                        st.success("Manual import successful!")
                        st.write(f"Test recorder created: {type(test_recorder)}")
                    except Exception as e:
                        st.error(f"Manual import failed: {e}")
                        st.code(str(e))

                # Show Python path
                if st.button("Show Python Path"):
                    st.code('\n'.join(sys.path))

                # Show current directory contents
                if st.button("Show Current Directory"):
                    files = [f for f in os.listdir('.') if f.endswith('.py')]
                    st.write("Python files in current directory:")
                    st.code('\n'.join(sorted(files)))

    def _render_device_selection_tab(self):
        """Render device selection using the modular component with shared recorder."""
        st.subheader("Device Selection")

        if DEVICE_SELECTOR_AVAILABLE and self._device_selector:
            self._device_selector.render()

            with st.expander("Current Selection Summary", expanded=False):
                if self.recorder:
                    input_id = int(getattr(self.recorder, 'input_device', -1))
                    output_id = int(getattr(self.recorder, 'output_device', -1))
                    st.write(f"**Input Device ID:** {input_id}")
                    st.write(f"**Output Device ID:** {output_id}")
                    if input_id == -1 or output_id == -1:
                        st.info("Using system default for any device set to -1")
                else:
                    st.warning("Recorder not available")
        else:
            st.warning("AudioDeviceSelector component not available")

        # Multi-channel configuration section
        st.markdown("---")
        self._render_multichannel_configuration()

    def _render_multichannel_configuration(self):
        """Render multi-channel configuration UI."""
        st.subheader("Multi-Channel Configuration")
        st.markdown("Configure multi-channel recording settings.")

        if not self.recorder:
            st.warning("Recorder not available")
            return

        # Get current configuration
        mc_config = self.recorder.multichannel_config
        current_enabled = mc_config.get('enabled', False)
        current_num_channels = mc_config.get('num_channels', 1)
        current_channel_names = mc_config.get('channel_names', [f"Channel {i}" for i in range(current_num_channels)])
        current_ref_channel = mc_config.get('reference_channel', 0)
        current_cal_channel = mc_config.get('calibration_channel')

        # 1. Enable/disable toggle
        multichannel_enabled = st.checkbox(
            "Enable multi-channel recording",
            value=current_enabled,
            key="multichannel_enabled_checkbox",
            help="Record from multiple input channels simultaneously"
        )

        # Handle enable/disable state change immediately
        if multichannel_enabled != current_enabled:
            if not multichannel_enabled:
                # User wants to disable - do it immediately
                self.recorder.multichannel_config['enabled'] = False
                self.recorder.multichannel_config['num_channels'] = 1
                st.success("✓ Multi-channel recording disabled.")
                st.rerun()

        if multichannel_enabled:
            # 2. Number of channels input
            col1, col2 = st.columns(2)

            with col1:
                num_channels = st.number_input(
                    "Number of channels",
                    min_value=1,
                    max_value=32,
                    value=current_num_channels,
                    help="Total number of input channels to record"
                )

                # 4. Reference channel selection
                reference_channel = st.selectbox(
                    "Reference channel",
                    options=list(range(num_channels)),
                    index=min(current_ref_channel, num_channels - 1),
                    help="Channel used for onset detection and alignment"
                )

                # Calibration channel selection
                cal_options = ["None (Disabled)"] + [f"Ch {i}" for i in range(num_channels)]
                if current_cal_channel is None:
                    cal_default_idx = 0
                else:
                    cal_default_idx = current_cal_channel + 1 if current_cal_channel < num_channels else 0

                calibration_channel_str = st.selectbox(
                    "Calibration channel",
                    options=cal_options,
                    index=cal_default_idx,
                    help="Channel with calibration impulse (e.g., hammer accelerometer). Use None for no calibration."
                )

                # Parse calibration channel
                if calibration_channel_str == "None (Disabled)":
                    calibration_channel = None
                else:
                    calibration_channel = int(calibration_channel_str.replace("Ch ", ""))

            with col2:
                # 3. Channel naming
                st.markdown("**Channel Names**")
                channel_names = []
                for ch in range(num_channels):
                    default_name = current_channel_names[ch] if ch < len(current_channel_names) else f"Channel {ch}"
                    name = st.text_input(
                        f"Ch {ch}",
                        value=default_name,
                        key=f"ch_name_{ch}"
                    )
                    channel_names.append(name)

            # 5. Save configuration button
            if st.button("Save Multi-Channel Configuration", type="primary"):
                try:
                    # Update configuration
                    self.recorder.multichannel_config['enabled'] = True
                    self.recorder.multichannel_config['num_channels'] = num_channels
                    self.recorder.multichannel_config['channel_names'] = channel_names
                    self.recorder.multichannel_config['reference_channel'] = reference_channel
                    self.recorder.multichannel_config['calibration_channel'] = calibration_channel

                    # Ensure response_channels list is updated
                    if calibration_channel is not None:
                        # If calibration is enabled, response channels exclude calibration channel
                        self.recorder.multichannel_config['response_channels'] = [
                            ch for ch in range(num_channels) if ch != calibration_channel
                        ]
                    else:
                        # No calibration, all channels are response channels
                        self.recorder.multichannel_config['response_channels'] = list(range(num_channels))

                    # Validate the configuration
                    self.recorder._validate_multichannel_config()

                    cal_msg = f" | Calibration: Ch {calibration_channel}" if calibration_channel is not None else " | No calibration"
                    st.success(f"✓ Multi-channel configuration saved: {num_channels} channels{cal_msg}")
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed to save configuration: {e}")

            # Show current configuration summary
            st.markdown("---")
            st.markdown("**Current Configuration**")

            # Build summary message
            cal_summary = f" | **Calibration:** Ch {calibration_channel}" if calibration_channel is not None else ""
            st.info(f"**Enabled:** {num_channels} channels | **Reference:** Ch {reference_channel} ({channel_names[reference_channel]}){cal_summary}")

            with st.expander("Channel Details"):
                for ch_idx, ch_name in enumerate(channel_names):
                    # Determine icon based on channel role
                    if ch_idx == calibration_channel:
                        icon = "🔨"  # Calibration channel (hammer)
                        role = "(Calibration)"
                    elif ch_idx == reference_channel:
                        icon = "🎤"  # Reference channel
                        role = "(Reference)"
                    else:
                        icon = "🔊"  # Response channel
                        role = ""
                    st.write(f"{icon} **Ch {ch_idx}:** {ch_name} {role}")

        else:
            # Disable multi-channel mode
            if st.button("Save Configuration (Disable Multi-Channel)"):
                try:
                    self.recorder.multichannel_config['enabled'] = False
                    self.recorder.multichannel_config['num_channels'] = 1
                    st.success("✓ Multi-channel recording disabled. Using single-channel mode.")
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed to save configuration: {e}")

            st.info("Multi-channel recording is disabled. Enable it to record from multiple channels simultaneously.")

    def _display_current_device_channels(self):
        """Display current device with channel information."""
        try:
            devices_info = self.recorder.get_device_info_with_channels()
            current_id = int(getattr(self.recorder, 'input_device', -1))

            if current_id == -1:
                st.info("Using system default input device")
                max_ch = max((d['max_channels'] for d in devices_info['input_devices']), default=1)
                st.metric("Max Available Channels", max_ch)
            else:
                for dev in devices_info['input_devices']:
                    if dev['device_id'] == current_id:
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Device ID", dev['device_id'])
                        with col2:
                            st.metric("Device Name", dev['name'])
                        with col3:
                            st.metric("Channels", dev['max_channels'])
                        break
        except Exception as e:
            st.error(f"Error getting device info: {e}")

    def _render_calibration_impulse_tab(self):
        """Render calibration impulse configuration and testing tab."""
        st.subheader("Calibration Impulse")
        st.markdown("Configure and test calibration channel for multi-channel impulse response recording.")

        if not self.recorder:
            st.error("Recorder not initialized")
            return

        # Check if multi-channel is enabled
        mc_config = self.recorder.multichannel_config
        if not mc_config.get('enabled', False):
            st.warning("Multi-channel recording is not enabled. Calibration is only available in multi-channel mode.")
            st.info("Enable multi-channel recording in the Device Selection tab first.")
            return

        num_channels = mc_config.get('num_channels', 1)
        if num_channels < 2:
            st.warning("At least 2 channels required for calibration. Current configuration has only 1 channel.")
            return

        # Get calibration channel (configured in Device Selection tab)
        selected_cal_ch = mc_config.get('calibration_channel')
        channel_names = mc_config.get('channel_names', [f"Channel {i}" for i in range(num_channels)])

        # Show current calibration channel configuration
        st.markdown("### Current Configuration")
        if selected_cal_ch is not None:
            st.success(f"✓ Calibration channel: Ch {selected_cal_ch} ({channel_names[selected_cal_ch]})")
        else:
            st.warning("⚠️ No calibration channel selected. Configure it in the Device Selection tab.")
            st.info("Go to Device Selection tab → Multi-Channel Configuration → Calibration channel")
            return

        st.markdown("---")

        # Section 1: Calibration Quality Parameters
        st.markdown("### 1. Calibration Quality Parameters")
        st.markdown("Configure thresholds for validating calibration impulse quality.")

        # Get current quality config from recorder
        if hasattr(self.recorder, 'calibration_quality_config'):
            qual_config = self.recorder.calibration_quality_config
        else:
            # Default values
            qual_config = {
                'cal_min_amplitude': 0.1,
                'cal_max_amplitude': 0.95,
                'cal_min_duration_ms': 2.0,
                'cal_max_duration_ms': 20.0,
                'cal_duration_threshold': 0.3,
                'cal_double_hit_window_ms': [10, 50],
                'cal_double_hit_threshold': 0.3,
                'cal_tail_start_ms': 30.0,
                'cal_tail_max_rms_ratio': 0.15,
                'min_valid_cycles': 3
            }

        with st.expander("Quality Parameter Settings", expanded=False):
            st.markdown("#### Amplitude Validation")
            col1, col2 = st.columns(2)
            with col1:
                min_amp = st.number_input(
                    "Minimum Amplitude",
                    min_value=0.01,
                    max_value=1.0,
                    value=float(qual_config.get('cal_min_amplitude', 0.1)),
                    step=0.01,
                    help="Minimum acceptable peak amplitude (normalized 0-1)"
                )
            with col2:
                max_amp = st.number_input(
                    "Maximum Amplitude",
                    min_value=0.01,
                    max_value=1.0,
                    value=float(qual_config.get('cal_max_amplitude', 0.95)),
                    step=0.01,
                    help="Maximum acceptable peak amplitude (prevents clipping)"
                )

            st.markdown("#### Duration Validation")
            col1, col2, col3 = st.columns(3)
            with col1:
                min_dur = st.number_input(
                    "Min Duration (ms)",
                    min_value=0.5,
                    max_value=50.0,
                    value=float(qual_config.get('cal_min_duration_ms', 2.0)),
                    step=0.5,
                    help="Minimum impulse duration in milliseconds"
                )
            with col2:
                max_dur = st.number_input(
                    "Max Duration (ms)",
                    min_value=1.0,
                    max_value=100.0,
                    value=float(qual_config.get('cal_max_duration_ms', 20.0)),
                    step=1.0,
                    help="Maximum impulse duration in milliseconds"
                )
            with col3:
                dur_thresh = st.number_input(
                    "Duration Threshold",
                    min_value=0.1,
                    max_value=0.9,
                    value=float(qual_config.get('cal_duration_threshold', 0.3)),
                    step=0.05,
                    help="Threshold for measuring duration (fraction of peak)"
                )

            st.markdown("#### Double Hit Detection")
            col1, col2 = st.columns(2)
            with col1:
                dh_window = st.text_input(
                    "Search Window [start, end] (ms)",
                    value=str(qual_config.get('cal_double_hit_window_ms', [10, 50])),
                    help="Time window to search for secondary impacts, e.g., [10, 50]"
                )
            with col2:
                dh_thresh = st.number_input(
                    "Double Hit Threshold",
                    min_value=0.1,
                    max_value=0.9,
                    value=float(qual_config.get('cal_double_hit_threshold', 0.3)),
                    step=0.05,
                    help="Threshold for detecting double hits (fraction of main peak)"
                )

            st.markdown("#### Tail Noise Validation")
            col1, col2 = st.columns(2)
            with col1:
                tail_start = st.number_input(
                    "Tail Start (ms)",
                    min_value=10.0,
                    max_value=200.0,
                    value=float(qual_config.get('cal_tail_start_ms', 30.0)),
                    step=5.0,
                    help="Where tail region begins after impulse (milliseconds)"
                )
            with col2:
                tail_max_rms = st.number_input(
                    "Max Tail RMS Ratio",
                    min_value=0.01,
                    max_value=1.0,
                    value=float(qual_config.get('cal_tail_max_rms_ratio', 0.15)),
                    step=0.01,
                    help="Maximum acceptable tail noise (fraction of impulse RMS)"
                )

            st.markdown("#### General Settings")
            min_valid = st.number_input(
                "Minimum Valid Cycles",
                min_value=1,
                max_value=20,
                value=int(qual_config.get('min_valid_cycles', 3)),
                help="Minimum number of valid calibration cycles required"
            )

            # Save button for quality parameters
            if st.button("Save Quality Parameters", type="primary"):
                try:
                    # Parse double hit window
                    import ast
                    dh_window_parsed = ast.literal_eval(dh_window)

                    new_config = {
                        'cal_min_amplitude': min_amp,
                        'cal_max_amplitude': max_amp,
                        'cal_min_duration_ms': min_dur,
                        'cal_max_duration_ms': max_dur,
                        'cal_duration_threshold': dur_thresh,
                        'cal_double_hit_window_ms': dh_window_parsed,
                        'cal_double_hit_threshold': dh_thresh,
                        'cal_tail_start_ms': tail_start,
                        'cal_tail_max_rms_ratio': tail_max_rms,
                        'min_valid_cycles': min_valid
                    }

                    # Save to recorder
                    if hasattr(self.recorder, 'calibration_quality_config'):
                        self.recorder.calibration_quality_config = new_config
                    else:
                        # Store in a custom attribute if not available
                        self.recorder.calibration_quality_config = new_config

                    st.success("✓ Quality parameters saved successfully!")

                except Exception as e:
                    st.error(f"Failed to save quality parameters: {e}")

        st.markdown("---")

        # Section 2: Test Calibration Impulse
        st.markdown("### 2. Test Calibration Impulse")
        st.markdown("Emit a train of impulses and check calibration quality for each cycle.")

        col1, col2 = st.columns([1, 1])

        with col1:
            if st.button("Run Calibration Test", type="primary"):
                st.session_state['cal_test_running'] = True

        with col2:
            if st.button("Clear Results"):
                if 'cal_test_results' in st.session_state:
                    del st.session_state['cal_test_results']
                st.session_state['cal_test_running'] = False

        # Run calibration test
        if st.session_state.get('cal_test_running', False):
            with st.spinner("Recording calibration impulses..."):
                try:
                    results = self._perform_calibration_test()
                    st.session_state['cal_test_results'] = results
                    st.session_state['cal_test_running'] = False
                    st.success("✓ Calibration test completed!")
                except Exception as e:
                    st.error(f"Calibration test failed: {e}")
                    st.session_state['cal_test_running'] = False

        # Display results
        if 'cal_test_results' in st.session_state:
            self._render_calibration_test_results(st.session_state['cal_test_results'])

    def _perform_calibration_test(self) -> Dict:
        """
        Perform a calibration test by recording impulses and validating quality.

        Returns:
            Dictionary with test results including per-cycle metrics
        """
        # Record multi-channel audio using internal method
        recorded_audio = self.recorder._record_method_2()

        if recorded_audio is None:
            raise ValueError("Recording failed - no data captured")

        # Process the recorded signal
        result = self.recorder._process_recorded_signal(recorded_audio)

        if result is None or 'impulse' not in result:
            raise ValueError("Processing failed or no impulse data returned")

        # Extract calibration channel data
        cal_ch = self.recorder.multichannel_config.get('calibration_channel')
        if cal_ch is None:
            raise ValueError("Calibration channel not configured")

        # Get metadata about validation
        metadata = result.get('metadata', {})

        return {
            'success': True,
            'num_cycles': self.recorder.num_pulses,
            'calibration_channel': cal_ch,
            'metadata': metadata,
            'result': result
        }

    def _render_calibration_test_results(self, results: Dict):
        """Render calibration test results with per-cycle quality metrics."""
        st.markdown("#### Calibration Test Results")

        metadata = results.get('metadata', {})
        num_cycles = results.get('num_cycles', 0)
        cal_ch = results.get('calibration_channel', 0)

        # Overall summary
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Cycles", num_cycles)
        with col2:
            valid_cycles = metadata.get('valid_cycles_after_calibration', [])
            st.metric("Valid Cycles", len(valid_cycles) if valid_cycles else "N/A")
        with col3:
            st.metric("Calibration Channel", f"Ch {cal_ch}")

        # Per-cycle quality metrics
        st.markdown("#### Per-Cycle Quality Metrics")

        if 'calibration_results' in metadata:
            cal_results = metadata['calibration_results']

            # Create a table of results
            import pandas as pd

            table_data = []
            for cycle_result in cal_results:
                cycle_idx = cycle_result.get('cycle_index', 0)
                valid = cycle_result.get('calibration_valid', False)
                metrics = cycle_result.get('calibration_metrics', {})
                failures = cycle_result.get('calibration_failures', [])

                row = {
                    'Cycle': cycle_idx,
                    'Valid': '✓' if valid else '✗',
                    'Peak Amp': f"{metrics.get('peak_amplitude', 0):.3f}",
                    'Duration (ms)': f"{metrics.get('duration_ms', 0):.1f}",
                    'Secondary Peak': f"{metrics.get('secondary_peak_ratio', 0):.2f}",
                    'Tail RMS': f"{metrics.get('tail_rms_ratio', 0):.3f}",
                    'Issues': ', '.join(failures) if failures else 'None'
                }
                table_data.append(row)

            if table_data:
                df = pd.DataFrame(table_data)
                st.dataframe(df, use_container_width=True, hide_index=True)
            else:
                st.info("No calibration validation data available in results.")
        else:
            st.info("Calibration validation not performed. This may be because calibration channel is not enabled or recording failed.")

        # Show any additional warnings or information
        if metadata.get('calibration_failed', False):
            st.error("⚠️ Calibration validation failed!")
            st.write("**Reason:**", metadata.get('calibration_error', 'Unknown error'))

    def _render_series_settings_tab(self):
        """Render the series settings if the component is available."""
        st.subheader("Series Settings")
        if SERIES_SETTINGS_AVAILABLE and self._series_settings_panel:
            self._series_settings_panel.render()
        else:
            st.warning("SeriesSettingsPanel component not available")

    # Accessors (optional)
    def get_recorder(self) -> Optional[RoomResponseRecorder]:
        return self.recorder

    def get_device_selector(self):
        return self._device_selector
