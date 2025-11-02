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

# Calibration impulse panel (modular)
try:
    from gui_calibration_impulse_panel import CalibrationImpulsePanel
    CALIBRATION_IMPULSE_AVAILABLE = True
except ImportError:
    CalibrationImpulsePanel = None  # type: ignore
    CALIBRATION_IMPULSE_AVAILABLE = False


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
        self._calibration_impulse_panel = CalibrationImpulsePanel(self.recorder, self) if CALIBRATION_IMPULSE_AVAILABLE else None

        # Local state for diagnostics (e.g., last device scan summary)
        self._diagnostics: Dict[str, Any] = {}

    # -------------------------
    # Public entry point
    # -------------------------

    def render(self):
        st.header("Audio Settings")

        # Top status (recorder + SDL readiness + critical system info)
        self._render_recorder_status()

        if not self.recorder:
            st.warning("Audio functionality will be limited without RoomResponseRecorder.")
            return

        # Initialize session state and sync with recorder
        self._init_session_state()
        self._sync_session_state_with_recorder()

        # Quick status bar (IDs + derived latency from UI buffer)
        self._render_audio_status_bar()

        # Determine which focus area was requested from navigation
        focus = st.session_state.get('audio_settings_focus', '')

        # Build tab structure
        if SERIES_SETTINGS_AVAILABLE and self._series_settings_panel:
            tab_names = [
                "Device Selection & Testing",
                "Calibration Impulse",
                "Series Settings"
            ]
            # Map focus to tab index and section
            focus_map = {
                'device_selection': (0, None),
                'multichannel': (0, 'multichannel'),  # Tab 0, multichannel section
                'calibration': (1, None),
                'series_settings': (2, None)
            }
        else:
            tab_names = [
                "Device Selection & Testing",
                "Calibration Impulse"
            ]
            # Map focus to tab index and section (no Series Settings tab)
            focus_map = {
                'device_selection': (0, None),
                'multichannel': (0, 'multichannel'),  # Tab 0, multichannel section
                'calibration': (1, None),
                'series_settings': (None, None)  # Not available
            }

        # Determine which tab to show based on focus
        if focus and focus in focus_map:
            tab_idx, section = focus_map[focus]
            if tab_idx is not None:
                default_tab = tab_idx
            else:
                default_tab = 0
        else:
            default_tab = 0

        # Initialize or get current tab selection from session state
        if 'audio_settings_current_tab' not in st.session_state:
            st.session_state['audio_settings_current_tab'] = default_tab
        elif focus:  # If focus was set by navigation, override current tab
            st.session_state['audio_settings_current_tab'] = default_tab

        # Tab selector (radio buttons for programmatic control)
        st.markdown("---")

        # Use the session state value as the current tab index
        current_tab_idx = st.session_state['audio_settings_current_tab']

        # Callback to update tab selection immediately
        def on_tab_change():
            """Update current tab index when user selects a new tab."""
            selected_tab_name = st.session_state.audio_settings_tab_selector
            selected_idx = tab_names.index(selected_tab_name)
            st.session_state['audio_settings_current_tab'] = selected_idx

        selected_tab_name = st.radio(
            "Section:",
            options=tab_names,
            index=current_tab_idx,
            key="audio_settings_tab_selector",
            on_change=on_tab_change,
            horizontal=True,
            label_visibility="collapsed"
        )

        # Use the current index from session state (updated by callback)
        current_tab_idx = st.session_state['audio_settings_current_tab']

        st.markdown("---")

        # Show subsection indicator if navigating to multichannel
        subsection = None
        if focus == 'multichannel' and current_tab_idx == 0:
            st.info(f"📍 **Multi-Channel Configuration** section")
            subsection = 'multichannel'

        # Render the selected tab content
        if current_tab_idx == 0:
            self._render_device_selection_tab(focus_subsection=subsection)
        elif current_tab_idx == 1:
            self._render_calibration_impulse_tab()
        elif current_tab_idx == 2 and len(tab_names) > 2:
            self._render_series_settings_tab()

        # Clear focus after rendering so it doesn't persist
        if 'audio_settings_focus' in st.session_state:
            del st.session_state['audio_settings_focus']

    # -------------------------
    # Internal helpers
    # -------------------------

    def _render_recorder_status(self):
        """Top status panel for recorder, SDL, and critical system info."""
        # Row 1: Core status
        cols = st.columns(5)

        with cols[0]:
            st.markdown("**Recorder**")
            if self.recorder:
                st.success("Ready")
            elif RECORDER_AVAILABLE:
                st.warning("Not created")
            else:
                st.error("Missing")

        with cols[1]:
            st.markdown("**SDL Core**")
            if self.sdl_ready:
                st.success("Ready")
            else:
                st.error("Missing")

        with cols[2]:
            st.markdown("**Drivers**")
            if self.recorder:
                try:
                    core = self.recorder.get_sdl_core_info()
                    drivers = core.get("drivers", []) or []
                    if drivers:
                        st.info(f"{len(drivers)} avail")
                    else:
                        st.warning("None")
                except Exception as e:
                    st.warning("N/A")
            else:
                st.info("N/A")

        with cols[3]:
            st.markdown("**Devices**")
            if self.recorder:
                try:
                    core = self.recorder.get_sdl_core_info()
                    counts = core.get("device_counts", {"input": 0, "output": 0})
                    st.info(f"{counts.get('input', 0)} in / {counts.get('output', 0)} out")
                except Exception:
                    st.info("N/A")
            else:
                st.info("N/A")

        with cols[4]:
            st.markdown("**Version**")
            if self.recorder:
                try:
                    core = self.recorder.get_sdl_core_info()
                    mod_ver = core.get("module_version", "?")
                    st.info(f"v{mod_ver}")
                except Exception:
                    st.info("N/A")
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
                if CALIBRATION_IMPULSE_AVAILABLE:
                    self._calibration_impulse_panel = CalibrationImpulsePanel(self.recorder, self)
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

    def _load_config_from_file(self):
        """Load configuration using centralized config manager.

        Returns:
            Dictionary with configuration values, or empty dict if file doesn't exist
        """
        from config_manager import config_manager
        return config_manager.load_config()

    def _save_config_to_file(self) -> bool:
        """Save current audio and multichannel settings using centralized config manager.

        Returns:
            True if successful, False otherwise
        """
        try:
            from config_manager import config_manager

            # Load existing config or create new one
            config = config_manager.load_config()

            # Update device selection
            if self.recorder:
                config['input_device'] = int(getattr(self.recorder, 'input_device', -1))
                config['output_device'] = int(getattr(self.recorder, 'output_device', -1))

                # Update multichannel configuration
                mc_config = self.recorder.multichannel_config
                config['multichannel_config'] = {
                    'enabled': bool(mc_config.get('enabled', False)),
                    'num_channels': int(mc_config.get('num_channels', 1)),
                    'channel_names': list(mc_config.get('channel_names', ['Channel 0'])),
                    'calibration_channel': mc_config.get('calibration_channel'),  # None or int
                    'reference_channel': int(mc_config.get('reference_channel', 0)),
                    'response_channels': list(mc_config.get('response_channels', [0])),
                    'normalize_by_calibration': bool(mc_config.get('normalize_by_calibration', False))
                }

                # Update calibration quality configuration
                if hasattr(self.recorder, 'calibration_quality_config'):
                    config['calibration_quality_config'] = dict(self.recorder.calibration_quality_config)

            # Save using config manager with error reporting
            success, error_msg = config_manager.save_config_with_error(config, updated_by="Audio Settings Panel")
            if not success:
                st.error(f"Config save failed: {error_msg}")
                st.code(error_msg)

            return success
        except Exception as e:
            st.error(f"Failed to save configuration: {e}")
            import traceback
            st.code(traceback.format_exc())
            return False

    def _init_session_state(self):
        """Initialize session state variables from configuration file and recorder."""
        # Only load configuration from file on first run (not every render)
        if 'audio_config_loaded' not in st.session_state:
            config = self._load_config_from_file()

            defaults = {
                'sample_rate': 48000,
                'buffer_size': 512,  # UI-only, not stored in recorder
                'input_device': config.get('input_device', -1),
                'output_device': config.get('output_device', -1),
            }

            if self.recorder:
                # Apply loaded device config to recorder (only on first load)
                self.recorder.input_device = defaults['input_device']
                self.recorder.output_device = defaults['output_device']

                # Load and apply multichannel config (only on first load)
                mc_config_file = config.get('multichannel_config', {})
                if mc_config_file:
                    self.recorder.multichannel_config['enabled'] = bool(mc_config_file.get('enabled', False))
                    self.recorder.multichannel_config['num_channels'] = int(mc_config_file.get('num_channels', 1))
                    self.recorder.multichannel_config['channel_names'] = list(mc_config_file.get('channel_names', ['Channel 0']))
                    self.recorder.multichannel_config['calibration_channel'] = mc_config_file.get('calibration_channel')
                    self.recorder.multichannel_config['reference_channel'] = int(mc_config_file.get('reference_channel', 0))
                    self.recorder.multichannel_config['response_channels'] = list(mc_config_file.get('response_channels', [0]))
                    self.recorder.multichannel_config['normalize_by_calibration'] = bool(mc_config_file.get('normalize_by_calibration', False))

                # Load and apply calibration quality config (only on first load)
                cal_quality_config_file = config.get('calibration_quality_config', {})
                if cal_quality_config_file and hasattr(self.recorder, 'calibration_quality_config'):
                    self.recorder.calibration_quality_config.update(cal_quality_config_file)

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

            # Mark that configuration has been loaded
            st.session_state['audio_config_loaded'] = True
        else:
            # Config already loaded, just ensure session defaults exist
            session_defaults = {
                'audio_selected_input_device': 'System Default',
                'audio_selected_output_device': 'System Default',
                'audio_sample_rate': 48000,
                'audio_buffer_size': 512,
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

    def _render_device_selection_tab(self, focus_subsection=None):
        """Render device selection using the modular component with shared recorder.

        Args:
            focus_subsection: Optional subsection to highlight ('multichannel' for multi-channel config)
        """
        st.subheader("Device Selection")

        if DEVICE_SELECTOR_AVAILABLE and self._device_selector:
            self._device_selector.render()


            # Save device selection button
            st.markdown("---")
            if st.button("💾 Save Device Selection to Config File", type="primary", key="save_device_selection"):
                if self._save_config_to_file():
                    st.success("✓ Device selection saved to recorderConfig.json")
                    st.info("Device settings will be loaded automatically on next session")
                else:
                    st.error("Failed to save configuration file")
        else:
            st.warning("AudioDeviceSelector component not available")

        # Multi-channel configuration section
        st.markdown("---")
        self._render_multichannel_configuration(focus_subsection=focus_subsection)

    def _render_multichannel_configuration(self, focus_subsection=None):
        """Render multi-channel configuration UI.

        Args:
            focus_subsection: Optional subsection focus indicator
        """
        # Highlight this section if it's the focused subsection
        if focus_subsection == 'multichannel':
            st.success("📍 **Multi-Channel Configuration** ← You are here")

        st.subheader("Multi-Channel Configuration")
        st.markdown("Configure multi-channel recording settings.")

        # Show config file location
        import os
        from config_manager import config_manager
        config_path = config_manager.get_config_path()
        with st.expander("📁 Configuration File Location", expanded=False):
            st.code(f"Config file: {config_path}")
            st.caption(f"Exists: {config_path.exists()}")
            st.caption(f"Working directory: {os.getcwd()}")

        if not self.recorder:
            st.warning("Recorder not available")
            return

        # Get device channel capabilities and details
        max_device_channels = 1
        selected_device_name = "Unknown"
        selected_device_id = -1

        try:
            devices_info = self.recorder.get_device_info_with_channels()
            current_device_id = int(getattr(self.recorder, 'input_device', -1))
            selected_device_id = current_device_id

            if current_device_id == -1:
                # System default - find max channels from all devices
                max_device_channels = max((d['max_channels'] for d in devices_info['input_devices']), default=1)
                selected_device_name = "System Default"
            else:
                # Find specific device
                for dev in devices_info['input_devices']:
                    if dev['device_id'] == current_device_id:
                        max_device_channels = dev['max_channels']
                        selected_device_name = dev.get('name', f"Device {current_device_id}")
                        break
        except Exception as e:
            st.warning(f"Could not detect device capabilities: {e}")
            max_device_channels = 2  # Safe default


        # Show device capability info
        if max_device_channels == 1:
            st.info(f"ℹ️ Selected input device supports **{max_device_channels} channel** (mono only)")
        else:
            st.info(f"ℹ️ Selected input device supports up to **{max_device_channels} channels**")

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
                    max_value=min(32, max_device_channels),
                    value=min(current_num_channels, max_device_channels),
                    help=f"Total number of input channels to record (max {max_device_channels} for your device)"
                )

                # Warning if trying to use more channels than device supports
                if num_channels > max_device_channels:
                    st.error(f"⚠️ Your device only supports {max_device_channels} channels! Please reduce the number of channels.")

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

                # Normalize by calibration checkbox (only if calibration is enabled)
                current_normalize = mc_config.get('normalize_by_calibration', False)
                if calibration_channel is not None:
                    normalize_by_calibration = st.checkbox(
                        "Normalize by calibration",
                        value=current_normalize,
                        key="normalize_by_calibration_checkbox",
                        help="Divide response channels by calibration impulse magnitude (negative peak) to normalize amplitude"
                    )
                else:
                    normalize_by_calibration = False
                    if current_normalize:
                        st.info("ℹ️ Normalization disabled (no calibration channel selected)")

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
            col_save1, col_save2 = st.columns(2)
            with col_save1:
                if st.button("Apply Multi-Channel Configuration", type="primary"):
                    try:
                        # Validate channel count against device capabilities
                        if num_channels > max_device_channels:
                            st.error(f"❌ Cannot save: Your device only supports {max_device_channels} channels, but you configured {num_channels} channels.")
                            st.info(f"Please reduce the number of channels to {max_device_channels} or less.")
                            return

                        # Update configuration
                        self.recorder.multichannel_config['enabled'] = True
                        self.recorder.multichannel_config['num_channels'] = num_channels
                        self.recorder.multichannel_config['channel_names'] = channel_names
                        self.recorder.multichannel_config['reference_channel'] = reference_channel
                        self.recorder.multichannel_config['calibration_channel'] = calibration_channel
                        self.recorder.multichannel_config['normalize_by_calibration'] = normalize_by_calibration

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

                        st.success("✓ Configuration applied to recorder!")

                        cal_msg = f" | Calibration: Ch {calibration_channel}" if calibration_channel is not None else " | No calibration"
                        st.info(f"Multi-channel configuration: {num_channels} channels{cal_msg}")
                        st.info("💡 Click 'Save to Config File' to persist settings across sessions")

                    except Exception as e:
                        st.error(f"Failed to apply configuration: {e}")

            with col_save2:
                if st.button("💾 Save to Config File", type="secondary"):
                    if self._save_config_to_file():
                        st.success(f"✓ Configuration saved to recorderConfig.json")
                        st.info("Settings will be loaded automatically on next session")
                    else:
                        st.error("Failed to save configuration file")

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
            col_dis1, col_dis2 = st.columns(2)
            with col_dis1:
                if st.button("Apply (Disable Multi-Channel)", type="primary"):
                    try:
                        self.recorder.multichannel_config['enabled'] = False
                        self.recorder.multichannel_config['num_channels'] = 1
                        st.success("✓ Multi-channel recording disabled. Using single-channel mode.")
                        st.info("💡 Click 'Save to Config File' to persist this setting")
                    except Exception as e:
                        st.error(f"Failed to apply configuration: {e}")

            with col_dis2:
                if st.button("💾 Save to Config File", type="secondary", key="save_disabled_mc"):
                    if self._save_config_to_file():
                        st.success("✓ Configuration saved to recorderConfig.json")
                        st.info("Multi-channel disabled setting will be loaded on next session")

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
        """Render calibration impulse using the modular component with shared recorder."""
        if CALIBRATION_IMPULSE_AVAILABLE and self._calibration_impulse_panel:
            self._calibration_impulse_panel.render()
        else:
            st.subheader("Calibration Impulse")
            st.warning("CalibrationImpulsePanel component not available")


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
