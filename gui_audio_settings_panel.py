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
import numpy as np
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

# Audio visualizer (modular)
try:
    from gui_audio_visualizer import AudioVisualizer
    AUDIO_VISUALIZER_AVAILABLE = True
except ImportError:
    AudioVisualizer = None  # type: ignore
    AUDIO_VISUALIZER_AVAILABLE = False


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
                    'response_channels': list(mc_config.get('response_channels', [0]))
                }

                # Update calibration quality configuration
                if hasattr(self.recorder, 'calibration_quality_config'):
                    config['calibration_quality_config'] = dict(self.recorder.calibration_quality_config)

            # Save using config manager
            success = config_manager.save_config(config, updated_by="Audio Settings Panel")

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
        """Render calibration impulse configuration and testing tab."""
        st.subheader("Calibration Impulse")
        st.markdown("Configure and test calibration channel for multi-channel impulse response recording.")

        if not self.recorder:
            st.error("Recorder not initialized")
            return

        # Check if multi-channel is enabled
        mc_config = self.recorder.multichannel_config

        if not mc_config.get('enabled', False):
            st.warning("⚠️ Multi-channel recording is not enabled. Calibration is only available in multi-channel mode.")
            st.info("📍 **To enable multi-channel recording:**")
            st.markdown("""
            1. Go to **Device Selection & Testing** tab
            2. Scroll down to **Multi-Channel Configuration** section
            3. Check **"Enable multi-channel recording"**
            4. Configure your channels
            5. Click **"Apply Multi-Channel Configuration"**
            6. Click **"💾 Save to Config File"** to persist settings
            """)
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

        # Section 1: Calibration Quality Parameters (Collapsible)
        with st.expander("### 1. Calibration Quality Parameters", expanded=False):
            st.markdown("Configure thresholds for validating calibration impulse quality.")

            # Get current quality config from recorder (V2 format)
            if hasattr(self.recorder, 'calibration_quality_config'):
                qual_config = self.recorder.calibration_quality_config
            else:
                # Default values (V2 Refactored - min/max ranges)
                qual_config = {
                    'min_negative_peak': 0.1,
                    'max_negative_peak': 0.95,
                    'min_positive_peak': 0.0,
                    'max_positive_peak': 0.6,
                    'min_aftershock': 0.0,
                    'max_aftershock': 0.3,
                    'aftershock_window_ms': 10.0,
                    'aftershock_skip_ms': 2.0,
                    'min_valid_cycles': 3
                }

            # Tool 1: Manual threshold editing in tabular form
            st.markdown("#### 🔧 Tool 1: Manual Threshold Configuration")
            st.info("Edit quality thresholds directly. Use Tool 2 below to auto-calculate from good cycles.")

            # Create compact tabular layout for thresholds
            col1, col2, col3 = st.columns([2, 1, 1])
            with col1:
                st.markdown("**Quality Metric**")
            with col2:
                st.markdown("**Minimum**")
            with col3:
                st.markdown("**Maximum**")

            # Negative Peak
            col1, col2, col3 = st.columns([2, 1, 1])
            with col1:
                st.markdown("**Negative Peak** (absolute)")
            with col2:
                min_neg_peak = st.number_input(
                    "Min Neg Peak",
                    min_value=0.0,
                    max_value=1.0,
                    value=float(qual_config.get('min_negative_peak', 0.1)),
                    step=0.01,
                    key="min_neg_peak",
                    label_visibility="collapsed"
                )
            with col3:
                max_neg_peak = st.number_input(
                    "Max Neg Peak",
                    min_value=0.0,
                    max_value=1.0,
                    value=float(qual_config.get('max_negative_peak', 0.95)),
                    step=0.01,
                    key="max_neg_peak",
                    label_visibility="collapsed"
                )

            # Positive Peak
            col1, col2, col3 = st.columns([2, 1, 1])
            with col1:
                st.markdown("**Positive Peak** (absolute)")
            with col2:
                min_pos_peak = st.number_input(
                    "Min Pos Peak",
                    min_value=0.0,
                    max_value=1.0,
                    value=float(qual_config.get('min_positive_peak', 0.0)),
                    step=0.01,
                    key="min_pos_peak",
                    label_visibility="collapsed"
                )
            with col3:
                max_pos_peak = st.number_input(
                    "Max Pos Peak",
                    min_value=0.0,
                    max_value=1.0,
                    value=float(qual_config.get('max_positive_peak', 0.6)),
                    step=0.01,
                    key="max_pos_peak",
                    label_visibility="collapsed"
                )

            # Aftershock
            col1, col2, col3 = st.columns([2, 1, 1])
            with col1:
                st.markdown("**Aftershock** (absolute)")
            with col2:
                min_aftershock = st.number_input(
                    "Min Aftershock",
                    min_value=0.0,
                    max_value=1.0,
                    value=float(qual_config.get('min_aftershock', 0.0)),
                    step=0.01,
                    key="min_aftershock",
                    label_visibility="collapsed"
                )
            with col3:
                max_aftershock = st.number_input(
                    "Max Aftershock",
                    min_value=0.0,
                    max_value=1.0,
                    value=float(qual_config.get('max_aftershock', 0.3)),
                    step=0.01,
                    key="max_aftershock",
                    label_visibility="collapsed"
                )

            st.markdown("---")
            st.markdown("**Configuration Parameters**")
            col1, col2, col3 = st.columns(3)
            with col1:
                aftershock_window = st.number_input(
                    "Aftershock Window (ms)",
                    min_value=5.0,
                    max_value=50.0,
                    value=float(qual_config.get('aftershock_window_ms', 10.0)),
                    step=1.0,
                    help="Time window after peak to check for aftershocks"
                )
            with col2:
                aftershock_skip = st.number_input(
                    "Aftershock Skip (ms)",
                    min_value=0.0,
                    max_value=10.0,
                    value=float(qual_config.get('aftershock_skip_ms', 2.0)),
                    step=0.5,
                    help="Skip first N ms after peak"
                )
            with col3:
                min_valid = st.number_input(
                    "Min Valid Cycles",
                    min_value=1,
                    max_value=20,
                    value=int(qual_config.get('min_valid_cycles', 3)),
                    help="Minimum number of valid calibration cycles required"
                )

            st.markdown("---")

            # Tool 2: Automatic threshold learning from marked cycles
            st.markdown("#### 🎯 Tool 2: Automatic Threshold Learning")
            st.markdown("Select good quality cycles in Section 2's Quality Metrics Summary table to automatically calculate optimal thresholds.")

            # Check if calibration test results are available
            if 'cal_test_results' in st.session_state:
                results = st.session_state['cal_test_results']
                num_cycles = results.get('num_cycles', 0)
                calibration_cycles = results.get('all_calibration_cycles')
                sample_rate = results.get('sample_rate', 48000)

                if calibration_cycles is not None:
                    # Get cycles selected in Section 2's table
                    marked_good = st.session_state.get('cal_test_selected_cycles', [])

                    if len(marked_good) > 0:
                        st.success(f"✓ {len(marked_good)} cycle(s) selected in Section 2: {', '.join(map(str, marked_good))}")

                        # Button to calculate thresholds from selected cycles
                        if st.button("🎯 Calculate Thresholds from Selected Cycles", type="secondary"):
                            try:
                                from calibration_validator_v2 import calculate_thresholds_from_marked_cycles

                                # Calculate thresholds
                                learned_thresholds = calculate_thresholds_from_marked_cycles(
                                    calibration_cycles,
                                    marked_good,
                                    sample_rate,
                                    margin=0.05  # 5% margin on both sides
                                )

                                # Store in session state and update the input fields
                                st.session_state['cal_test_learned_thresholds'] = learned_thresholds

                                # Update the recorder config with learned thresholds
                                self.recorder.calibration_quality_config.update(learned_thresholds.to_dict())

                                st.success(f"✓ Thresholds calculated from {len(marked_good)} selected cycles and loaded into configuration!")
                                st.info("💡 Review the updated thresholds in Tool 1 above, then click 'Save Configuration' below.")

                                # Show summary of calculated thresholds
                                with st.expander("📊 Calculated Thresholds Summary", expanded=True):
                                    col1, col2, col3 = st.columns(3)
                                    with col1:
                                        st.metric("Negative Peak Range",
                                                 f"{learned_thresholds.min_negative_peak:.3f} - {learned_thresholds.max_negative_peak:.3f}")
                                    with col2:
                                        st.metric("Positive Peak Range",
                                                 f"{learned_thresholds.min_positive_peak:.3f} - {learned_thresholds.max_positive_peak:.3f}")
                                    with col3:
                                        st.metric("Aftershock Range",
                                                 f"{learned_thresholds.min_aftershock:.3f} - {learned_thresholds.max_aftershock:.3f}")

                            except Exception as e:
                                st.error(f"Failed to calculate thresholds: {e}")
                                import traceback
                                st.code(traceback.format_exc())
                    else:
                        st.info("👇 Go to Section 2 below and select cycles in the Quality Metrics Summary table")
                else:
                    st.info("⚠️ No calibration cycles available. Run a calibration test in Section 2 below.")
            else:
                st.info("⚠️ No calibration test results available. Run a calibration test in Section 2 below first.")

            st.markdown("---")

            # Unified save button for both manual and automatic configuration
            if st.button("💾 Save Configuration", type="primary", key="save_quality_config"):
                try:
                    # Build V2 refactored config (min/max ranges)
                    # Use values from session state if learned thresholds exist, otherwise use manual inputs
                    if 'cal_test_learned_thresholds' in st.session_state:
                        learned = st.session_state['cal_test_learned_thresholds']
                        new_config = learned.to_dict()
                        # Preserve configuration parameters that aren't part of learned thresholds
                        new_config['min_valid_cycles'] = min_valid
                    else:
                        new_config = {
                            'min_negative_peak': min_neg_peak,
                            'max_negative_peak': max_neg_peak,
                            'min_positive_peak': min_pos_peak,
                            'max_positive_peak': max_pos_peak,
                            'min_aftershock': min_aftershock,
                            'max_aftershock': max_aftershock,
                            'aftershock_window_ms': aftershock_window,
                            'aftershock_skip_ms': aftershock_skip,
                            'min_valid_cycles': min_valid
                        }

                    # Save to recorder
                    if hasattr(self.recorder, 'calibration_quality_config'):
                        self.recorder.calibration_quality_config = new_config
                    else:
                        self.recorder.calibration_quality_config = new_config

                    # Save to config file
                    if self._save_config_to_file():
                        st.success("✓ Quality configuration saved successfully!")
                        st.info("Settings will be loaded automatically on next session. Clear results and re-run calibration test to apply new thresholds.")
                    else:
                        st.warning("⚠️ Configuration saved to recorder but failed to save to config file")

                except Exception as e:
                    st.error(f"Failed to save configuration: {e}")

        st.markdown("---")

        # Section 2: Test Calibration Impulse (Collapsible)
        with st.expander("### 2. Test Calibration Impulse", expanded=True):
            st.markdown("Emit a train of impulses and check calibration quality for each cycle.")

            col1, col2 = st.columns([1, 1])

            with col1:
                if st.button("Run Calibration Test", type="primary"):
                    st.session_state['cal_test_running'] = True

            with col2:
                if st.button("Clear Results"):
                    if 'cal_test_results' in st.session_state:
                        del st.session_state['cal_test_results']
                    if 'cal_test_selected_cycles' in st.session_state:
                        del st.session_state['cal_test_selected_cycles']
                    if 'cal_test_learned_thresholds' in st.session_state:
                        del st.session_state['cal_test_learned_thresholds']
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

        This method now records ALL impulses regardless of quality, extracts
        per-cycle waveforms and metrics, and returns them for user exploration.

        Returns:
            Dictionary with test results including:
            - all_calibration_cycles: Raw waveforms for each cycle (np.ndarray)
            - validation_results: Quality metrics for each cycle
            - sample_rate: Sample rate for waveform playback
        """
        # Validate device capabilities before recording
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

        # Record multi-channel audio using internal method
        recorded_audio = self.recorder._record_method_2()

        if recorded_audio is None:
            raise ValueError("Recording failed - no data captured. Check your audio device connections.")

        # Extract calibration channel raw data
        cal_ch = self.recorder.multichannel_config.get('calibration_channel')
        if cal_ch is None:
            raise ValueError("Calibration channel not configured")

        # For multi-channel, recorded_audio is a dict
        if isinstance(recorded_audio, dict):
            cal_raw = recorded_audio.get(cal_ch)
        else:
            # Single channel fallback
            cal_raw = recorded_audio

        if cal_raw is None:
            raise ValueError(f"Calibration channel {cal_ch} not found in recorded data")

        # STEP 1-4: Initial cycle extraction using simple reshape (AS-IS from document)
        expected_samples = self.recorder.cycle_samples * self.recorder.num_pulses

        # Pad or trim to expected length
        if len(cal_raw) < expected_samples:
            padded = np.zeros(expected_samples)
            padded[:len(cal_raw)] = cal_raw
            cal_raw = padded
        else:
            cal_raw = cal_raw[:expected_samples]

        # Simple reshape into individual cycles (NO alignment yet)
        initial_cycles = cal_raw.reshape(self.recorder.num_pulses, self.recorder.cycle_samples)

        # Run validation on initial cycles (Step 4 from document)
        from calibration_validator_v2 import CalibrationValidatorV2, QualityThresholds

        thresholds = QualityThresholds.from_config(self.recorder.calibration_quality_config)
        validator = CalibrationValidatorV2(thresholds, self.recorder.sample_rate)

        initial_validation_results = []
        for cycle_idx in range(self.recorder.num_pulses):
            validation = validator.validate_cycle(
                initial_cycles[cycle_idx],
                cycle_idx
            )
            validation_dict = {
                'cycle_index': cycle_idx,
                'calibration_valid': validation.calibration_valid,
                'calibration_metrics': validation.calibration_metrics,
                'calibration_failures': validation.calibration_failures,
                'is_user_marked_good': False
            }
            initial_validation_results.append(validation_dict)

        # STEP 5: Align cycles by onset (negative peak detection)
        # This filters to valid cycles only and aligns them
        alignment_result = self.recorder.align_cycles_by_onset(
            initial_cycles,
            initial_validation_results,
            correlation_threshold=0.7
        )

        aligned_cycles = alignment_result['aligned_cycles']
        valid_cycle_indices = alignment_result['valid_cycle_indices']

        # Build validation results for aligned cycles
        aligned_validation_results = []
        for i, original_idx in enumerate(valid_cycle_indices):
            # Use validation from original cycle
            if original_idx < len(initial_validation_results):
                aligned_validation_results.append(initial_validation_results[original_idx])

        # STEP 6: Apply SAME alignment to ALL channels (multi-channel support)
        aligned_multichannel_cycles = {}
        if isinstance(recorded_audio, dict):
            # Multi-channel recording - apply alignment to each channel
            for channel_name, channel_data in recorded_audio.items():
                aligned_channel_cycles = self.recorder.apply_alignment_to_channel(
                    channel_data,
                    alignment_result
                )
                aligned_multichannel_cycles[channel_name] = aligned_channel_cycles
        else:
            # Single channel - just use the calibration cycles
            aligned_multichannel_cycles[cal_ch] = aligned_cycles

        return {
            'success': True,
            'num_cycles': self.recorder.num_pulses,
            'calibration_channel': cal_ch,
            'sample_rate': self.recorder.sample_rate,
            # Initial extraction (Steps 1-4) - FOR EXISTING UI
            'all_calibration_cycles': initial_cycles,  # ALL cycles for existing quality metrics table
            'validation_results': initial_validation_results,  # Validation for ALL cycles
            # Alignment (Step 5-6) - FOR NEW ALIGNMENT SECTION AND DOWNSTREAM USE
            'alignment_metadata': alignment_result,
            'aligned_cycles': aligned_cycles,  # Calibration channel aligned cycles
            'aligned_multichannel_cycles': aligned_multichannel_cycles,  # ALL channels aligned uniformly
            'aligned_validation_results': aligned_validation_results,  # Validation for aligned cycles
            'cycle_duration_s': self.recorder.cycle_samples / self.recorder.sample_rate
        }

    def _render_calibration_test_results(self, results: Dict):
        """
        Render calibration test results with Quality Metrics Summary and Per-Cycle Analysis.

        The Quality Metrics Summary table allows clicking on cycles to view detailed analysis.
        """
        st.markdown("#### Calibration Test Results")

        # Extract data
        num_cycles = results.get('num_cycles', 0)
        cal_ch = results.get('calibration_channel', 0)
        sample_rate = results.get('sample_rate', 48000)
        calibration_cycles = results.get('all_calibration_cycles')  # Shape: (num_cycles, cycle_samples)
        validation_results = results.get('validation_results', [])
        cycle_duration_s = results.get('cycle_duration_s', 0.1)

        if calibration_cycles is None or len(validation_results) == 0:
            st.warning("No calibration data available. Please run the calibration test.")
            return

        # Overall summary
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Cycles Recorded", num_cycles)
        with col2:
            valid_count = sum(1 for v in validation_results if v.get('calibration_valid', False))
            st.metric("Valid Cycles", valid_count)
        with col3:
            st.metric("Calibration Channel", f"Ch {cal_ch}")

        st.markdown("---")

        # Quality Metrics Summary Table with Checkboxes
        st.markdown("#### Quality Metrics Summary")
        st.info("💡 Check the boxes to select cycles for analysis, comparison, or threshold learning")
        import pandas as pd

        # Initialize selection state if not exists
        if 'cal_test_selected_cycles' not in st.session_state:
            st.session_state['cal_test_selected_cycles'] = []

        # Display table with checkboxes for each row
        selected_cycles = []

        # Create column headers
        cols = st.columns([0.5, 0.8, 0.8, 1.2, 1.2, 1.2, 2.5])
        with cols[0]:
            st.markdown("**Select**")
        with cols[1]:
            st.markdown("**Cycle**")
        with cols[2]:
            st.markdown("**Valid**")
        with cols[3]:
            st.markdown("**Neg Peak**")
        with cols[4]:
            st.markdown("**Pos Peak**")
        with cols[5]:
            st.markdown("**Aftershock**")
        with cols[6]:
            st.markdown("**Issues**")

        st.markdown("---")

        # Render each row with a checkbox
        for v_result in validation_results:
            cycle_idx = v_result.get('cycle_index', 0)
            valid = v_result.get('calibration_valid', False)
            metrics = v_result.get('calibration_metrics', {})
            failures = v_result.get('calibration_failures', [])

            cols = st.columns([0.5, 0.8, 0.8, 1.2, 1.2, 1.2, 2.5])

            with cols[0]:
                # Checkbox for this cycle
                is_checked = st.checkbox(
                    "",
                    value=cycle_idx in st.session_state['cal_test_selected_cycles'],
                    key=f"cycle_checkbox_{cycle_idx}",
                    label_visibility="collapsed"
                )
                if is_checked:
                    if cycle_idx not in selected_cycles:
                        selected_cycles.append(cycle_idx)

            with cols[1]:
                st.markdown(f"{cycle_idx}")
            with cols[2]:
                st.markdown('✓' if valid else '✗')
            with cols[3]:
                st.markdown(f"{metrics.get('negative_peak', 0):.3f}")
            with cols[4]:
                st.markdown(f"{metrics.get('positive_peak', 0):.3f}")
            with cols[5]:
                st.markdown(f"{metrics.get('aftershock', 0):.3f}")
            with cols[6]:
                st.markdown(', '.join(failures) if failures else 'None')

        # Update session state with current selections
        st.session_state['cal_test_selected_cycles'] = sorted(selected_cycles)

        st.markdown("---")

        # Display selection info
        if selected_cycles:
            st.success(f"✓ Selected {len(selected_cycles)} cycle(s): {', '.join(map(str, selected_cycles))}")
            st.info("💡 These cycles will be used for: detailed analysis, comparison overlay, and threshold learning (Section 1)")
        else:
            st.info("👆 Check the boxes above to select cycles")

        st.markdown("---")

        # Waveform Visualization (unified component for single or multiple cycles)
        st.markdown("#### Waveform Analysis")

        if selected_cycles:
            if AUDIO_VISUALIZER_AVAILABLE and AudioVisualizer:
                # Use the same unified component for both single and multiple waveforms
                signals = [calibration_cycles[i] for i in selected_cycles]
                labels = [f"Cycle {i} {'✓' if validation_results[i].get('calibration_valid', False) else '✗'}"
                         for i in selected_cycles]

                # Generate appropriate title
                if len(selected_cycles) == 1:
                    title = f"Calibration Impulse - Cycle {selected_cycles[0]}"
                else:
                    title = f"Calibration Impulse - {len(selected_cycles)} Cycles Overlay"

                AudioVisualizer.render_multi_waveform_with_zoom(
                    audio_signals=signals,
                    sample_rate=sample_rate,
                    labels=labels,
                    title=title,
                    component_id="cal_waveform_viz",  # Same ID for both single and multiple
                    height=400,
                    normalize=False,
                    show_analysis=True
                )
            else:
                st.warning("AudioVisualizer not available - cannot display waveform")
                st.info("Install gui_audio_visualizer.py to enable waveform visualization")
        else:
            st.info("👆 Select one or more cycles in the Quality Metrics Summary table above to view waveforms")

        st.markdown("---")

        # User guidance
        with st.expander("💡 How to Use the Calibration Impulse Tool", expanded=False):
            st.markdown("""
            **Purpose:** Configure and validate quality criteria for calibration impulses in multi-channel recording.

            **Quick Workflow:**

            1. **Run Test** → Section 2: Click "Run Calibration Test"
            2. **Select Cycles** → Section 2: Check boxes next to good quality cycles in the table
            3. **Auto-Calculate** → Section 1: Click "Calculate Thresholds from Selected Cycles"
            4. **Save** → Section 1: Click "Save Configuration"

            ---

            **Section 1: Calibration Quality Parameters** (Collapsible)

            - **Tool 1 - Manual Configuration:**
              - Edit thresholds directly in the tabular form
              - Adjust min/max ranges for each quality metric

            - **Tool 2 - Automatic Threshold Learning:**
              - Select cycles in Section 2's Quality Metrics Summary table
              - The selected cycles appear here automatically
              - Click "Calculate Thresholds from Selected Cycles" to auto-compute optimal ranges
              - Review calculated thresholds displayed in Tool 1
              - Click "Save Configuration" at the bottom to persist settings

            **Section 2: Test Calibration Impulse** (Collapsible)

            1. **Run Test:** Click "Run Calibration Test" to record calibration impulses

            2. **Select Cycles in Table:**
               - Check the boxes next to cycles in the Quality Metrics Summary table
               - Check multiple boxes to select multiple cycles
               - Selected cycles are used for:
                 - **Waveform Visualization** (single or overlay depending on selection)
                 - **Threshold Learning** (Section 1, Tool 2)

            3. **Waveform Analysis:**
               - **Unified component:** Same visualization and controls for single or multiple cycles
               - **View modes:** Toggle between waveform and spectrum views
               - **Zoom controls:** Use sliders in "Zoom Controls" expander to zoom to any time range
               - **Persistent zoom:** Zoom settings preserved when adding/removing cycles
               - **Reset Zoom:** Button to return to full view
               - All quality metrics are visible in the table above
               - Cycles shown are **aligned** via cross-correlation for accurate comparison

            ---

            **Quality Criteria:**
            - **Negative Peak:** Strong negative pulse from hammer impact (absolute amplitude)
            - **Positive Peak:** Minimal positive component (absolute amplitude)
            - **Aftershock:** No significant rebounds within 10ms of main pulse (absolute amplitude)

            **What Makes a Good Calibration Impulse:**
            - Single strong negative pulse (hammer impact signature)
            - Sharp, clean waveform with quick decay
            - No aftershocks or bounces immediately after impact
            - Minimal positive component
            - Consistent amplitude across cycles

            **Tips:**
            - Select at least 3-5 good cycles for reliable automatic threshold calculation
            - Use the comparison overlay to verify selected cycles are similar
            - The system adds 5% safety margin to calculated thresholds
            - After saving new thresholds, clear results and re-run to validate
            - Invalid cycles aren't bad - they help you understand quality variations
            """)

        # ====================================================================
        # Alignment Results Review Section (at end of panel)
        # ====================================================================
        alignment_metadata = results.get('alignment_metadata')
        all_cycles = results.get('all_calibration_cycles')  # ALL initial cycles
        aligned_cycles_data = results.get('aligned_cycles')  # Only filtered, aligned cycles
        aligned_validation = results.get('aligned_validation_results', [])

        if alignment_metadata and all_cycles is not None:
            st.markdown("---")
            st.markdown("#### Alignment Results Review")
            st.markdown("""
            **Onset-Based Cycle Alignment:** This section shows valid cycles aligned by their negative peak (hammer impact onset).
            - Invalid cycles are filtered out (only valid cycles shown)
            - Negative peak (onset) detected in each valid cycle
            - All cycles shifted so onsets align at common position
            - Cycles with poor correlation after alignment are filtered out
            - Result: All displayed cycles should overlay precisely
            """)

            # Extract alignment data
            valid_cycle_indices = alignment_metadata.get('valid_cycle_indices', [])
            onset_positions = alignment_metadata.get('onset_positions', [])
            aligned_onset_position = alignment_metadata.get('aligned_onset_position', 0)
            correlations = alignment_metadata.get('correlations', [])
            reference_idx = alignment_metadata.get('reference_cycle_idx', 0)
            correlation_threshold = alignment_metadata.get('correlation_threshold', 0.7)

            num_initial = len(all_cycles)
            num_aligned = len(aligned_cycles_data) if aligned_cycles_data is not None else 0

            # Summary metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Initial Cycles", num_initial)
            with col2:
                st.metric("Valid & Aligned", num_aligned)
            with col3:
                mean_corr = np.mean(correlations) if correlations else 0
                st.metric("Mean Correlation", f"{mean_corr:.3f}")
            with col4:
                st.metric("Aligned Onset Pos", f"{aligned_onset_position} samples")

            st.markdown("---")

            # Alignment Table with Checkboxes (same pattern as Quality Metrics table)
            st.markdown("#### Aligned Cycles Table")
            st.info("💡 Check the boxes to select cycles for overlay visualization - aligned cycles should overlap exactly")

            # Initialize alignment selection state
            if 'alignment_selected_cycles' not in st.session_state:
                st.session_state['alignment_selected_cycles'] = []

            alignment_selected_cycles = []

            # Create column headers
            cols = st.columns([0.5, 0.8, 1.0, 1.0, 1.0, 0.8, 1.0, 0.7])
            with cols[0]:
                st.markdown("**Select**")
            with cols[1]:
                st.markdown("**Cycle #**")
            with cols[2]:
                st.markdown("**Original Onset**")
            with cols[3]:
                st.markdown("**Aligned Onset**")
            with cols[4]:
                st.markdown("**Correlation**")
            with cols[5]:
                st.markdown("**Valid**")
            with cols[6]:
                st.markdown("**Neg. Peak**")
            with cols[7]:
                st.markdown("**Note**")

            st.markdown("---")

            # Render each row with checkbox - only for aligned cycles
            for aligned_idx, original_idx in enumerate(valid_cycle_indices):
                # All cycles here passed validation
                is_ref = "REF" if aligned_idx == reference_idx else ""

                # Get metrics for aligned cycle
                aligned_peak = aligned_validation[aligned_idx].get('calibration_metrics', {}).get('negative_peak', 0) if aligned_idx < len(aligned_validation) else 0

                # Get onset positions
                original_onset = onset_positions[aligned_idx] if aligned_idx < len(onset_positions) else 0

                cols = st.columns([0.5, 0.8, 1.0, 1.0, 1.0, 0.8, 1.0, 0.7])

                with cols[0]:
                    is_checked = st.checkbox(
                        "",
                        value=aligned_idx in st.session_state['alignment_selected_cycles'],
                        key=f"alignment_checkbox_{aligned_idx}",
                        label_visibility="collapsed"
                    )
                    if is_checked:
                        if aligned_idx not in alignment_selected_cycles:
                            alignment_selected_cycles.append(aligned_idx)

                with cols[1]:
                    st.markdown(f"{original_idx}")
                with cols[2]:
                    st.markdown(f"{original_onset} samples")
                with cols[3]:
                    st.markdown(f"{aligned_onset_position} samples")
                with cols[4]:
                    st.markdown(f"{correlations[aligned_idx]:.3f}" if aligned_idx < len(correlations) else "N/A")
                with cols[5]:
                    st.markdown('✓')  # All shown cycles are valid
                with cols[6]:
                    st.markdown(f"{aligned_peak:.3f}")
                with cols[7]:
                    st.markdown(is_ref)

            # Update session state
            st.session_state['alignment_selected_cycles'] = sorted(alignment_selected_cycles)

            st.caption(f"Showing {num_aligned} valid, aligned cycles (invalid cycles filtered out)")
            st.markdown("---")

            # Display selection info
            if alignment_selected_cycles:
                original_cycle_numbers = [valid_cycle_indices[idx] for idx in alignment_selected_cycles if idx < len(valid_cycle_indices)]
                st.success(f"✓ Selected {len(alignment_selected_cycles)} cycle(s) - Original cycle #: {', '.join(map(str, original_cycle_numbers))}")
                st.info("💡 These aligned cycles will be overlaid in the visualization below - they should overlap exactly at the onset")
            else:
                st.info("👆 Check the boxes above to select cycles for visualization")

            st.markdown("---")

            # Visualization: Aligned cycles overlay
            st.markdown("#### Aligned Cycles Overlay")

            if alignment_selected_cycles:
                if AUDIO_VISUALIZER_AVAILABLE and AudioVisualizer:
                    # Prepare signals for visualization
                    # Show all selected aligned cycles overlaid
                    signals = []
                    labels = []

                    for aligned_idx in alignment_selected_cycles:
                        if aligned_idx < len(aligned_cycles_data):
                            original_idx = valid_cycle_indices[aligned_idx] if aligned_idx < len(valid_cycle_indices) else aligned_idx
                            # Add aligned waveform
                            signals.append(aligned_cycles_data[aligned_idx])
                            labels.append(f"Cycle {original_idx} (aligned)")

                    # Generate title
                    if len(alignment_selected_cycles) == 1:
                        original_idx = valid_cycle_indices[alignment_selected_cycles[0]] if alignment_selected_cycles[0] < len(valid_cycle_indices) else alignment_selected_cycles[0]
                        title = f"Aligned Cycle {original_idx} - Onset at {aligned_onset_position} samples"
                    else:
                        title = f"Aligned Cycles Overlay ({len(alignment_selected_cycles)} cycles) - All onsets aligned at {aligned_onset_position} samples"

                    AudioVisualizer.render_multi_waveform_with_zoom(
                        audio_signals=signals,
                        sample_rate=sample_rate,
                        labels=labels,
                        title=title,
                        component_id="alignment_overlay_viz",
                        height=400,
                        normalize=False,
                        show_analysis=True
                    )

                    st.info("💡 All displayed cycles have been aligned by their negative peak (onset). They should overlap precisely.")

                    # Show detailed metrics for selected cycles
                    st.markdown("**Selected Cycles Details:**")

                    for aligned_idx in alignment_selected_cycles:
                        if aligned_idx < len(valid_cycle_indices):
                            original_idx = valid_cycle_indices[aligned_idx]
                            with st.expander(f"Cycle {original_idx} Details", expanded=False):
                                col1, col2 = st.columns(2)

                                with col1:
                                    st.markdown("**Aligned Cycle Metrics:**")
                                    if aligned_idx < len(aligned_validation):
                                        metrics = aligned_validation[aligned_idx].get('calibration_metrics', {})
                                        st.write(f"Negative Peak: {metrics.get('negative_peak', 0):.3f}")
                                        st.write(f"Positive Peak: {metrics.get('positive_peak', 0):.3f}")
                                        st.write(f"Aftershock: {metrics.get('aftershock', 0):.3f}")
                                        st.write(f"Valid: ✓")

                                with col2:
                                    st.markdown("**Alignment Info:**")
                                    if aligned_idx < len(onset_positions):
                                        st.write(f"Original Onset Position: {onset_positions[aligned_idx]} samples")
                                    st.write(f"Aligned Onset Position: {aligned_onset_position} samples")
                                    if aligned_idx < len(correlations):
                                        st.write(f"Correlation: {correlations[aligned_idx]:.3f}")
                                    if aligned_idx == reference_idx:
                                        st.write("**[REFERENCE CYCLE]**")

                else:
                    st.warning("AudioVisualizer not available - cannot display overlay")
                    st.info("Install gui_audio_visualizer.py to enable waveform visualization")
            else:
                st.info("👆 Select one or more cycles in the Aligned Cycles Table above to view overlay")

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
