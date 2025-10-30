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

            # Save using config manager
            success = config_manager.save_config(config, updated_by="Audio Settings Panel")

            # Debug: Verify what was written
            if success:
                st.caption(f"✓ Wrote to: {config_manager.get_config_path()}")

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

            # DEBUG: Show all available devices
            with st.expander("🔍 DEBUG: All Available Devices", expanded=False):
                if self.recorder:
                    try:
                        devices_info = self.recorder.get_device_info_with_channels()

                        st.markdown("**Input Devices:**")
                        for dev in devices_info.get('input_devices', []):
                            st.write(f"- **ID {dev['device_id']}:** {dev.get('name', 'Unknown')} ({dev['max_channels']} channels)")

                        st.markdown("**Output Devices:**")
                        for dev in devices_info.get('output_devices', []):
                            st.write(f"- **ID {dev['device_id']}:** {dev.get('name', 'Unknown')} ({dev['max_channels']} channels)")
                    except Exception as e:
                        st.error(f"Failed to query devices: {e}")

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

        # DEBUG: Show current device selection
        with st.expander("🔍 DEBUG: Current Device Selection", expanded=True):
            st.markdown("**Selected Input Device:**")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Device ID", selected_device_id)
            with col2:
                st.metric("Max Channels", max_device_channels)
            with col3:
                st.code(selected_device_name if len(selected_device_name) < 20 else selected_device_name[:17] + "...")

            st.caption(f"Full device name: {selected_device_name}")

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

        # DEBUG: Show loaded configuration
        with st.expander("🔍 DEBUG: Loaded Multi-Channel Configuration", expanded=False):
            st.json({
                "enabled": current_enabled,
                "num_channels": current_num_channels,
                "reference_channel": current_ref_channel,
                "calibration_channel": current_cal_channel,
                "channel_names": current_channel_names,
                "response_channels": mc_config.get('response_channels', [])
            })

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

                        # DEBUG: Show what was just saved
                        st.success("✓ Configuration applied to recorder!")
                        with st.expander("🔍 DEBUG: Configuration Applied to recorder.multichannel_config", expanded=True):
                            st.markdown(f"**Target Device:** {selected_device_name} (ID: {selected_device_id}, Max Channels: {max_device_channels})")
                            st.json({
                                "enabled": True,
                                "num_channels": num_channels,
                                "reference_channel": reference_channel,
                                "calibration_channel": calibration_channel,
                                "channel_names": channel_names,
                                "response_channels": self.recorder.multichannel_config['response_channels']
                            })
                            st.caption("This configuration is applied to the recorder session.")

                        cal_msg = f" | Calibration: Ch {calibration_channel}" if calibration_channel is not None else " | No calibration"
                        st.info(f"Multi-channel configuration: {num_channels} channels{cal_msg}")
                        st.info("💡 Click 'Save to Config File' to persist settings across sessions")

                    except Exception as e:
                        st.error(f"Failed to apply configuration: {e}")

            with col_save2:
                if st.button("💾 Save to Config File", type="secondary"):
                    import os
                    config_path = os.path.abspath("recorderConfig.json")
                    if self._save_config_to_file():
                        st.success(f"✓ Configuration saved to recorderConfig.json")
                        with st.expander("🔍 DEBUG: Save Details", expanded=True):
                            st.code(f"File path: {config_path}")
                            st.markdown("**Saved multichannel_config:**")
                            st.json(self.recorder.multichannel_config)
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

        # DEBUG: Show current multichannel state
        with st.expander("🔍 DEBUG: Current Multichannel Configuration", expanded=False):
            st.json({
                "enabled": mc_config.get('enabled', False),
                "num_channels": mc_config.get('num_channels', 1),
                "channel_names": mc_config.get('channel_names', []),
                "calibration_channel": mc_config.get('calibration_channel'),
                "reference_channel": mc_config.get('reference_channel', 0)
            })

            # Show what's in the config file
            config_file_data = self._load_config_from_file()
            st.markdown("**From recorderConfig.json:**")
            st.json(config_file_data.get('multichannel_config', {}))

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

        This method now records ALL impulses regardless of quality, extracts
        per-cycle waveforms and metrics, and returns them for user exploration.

        Returns:
            Dictionary with test results including:
            - all_calibration_cycles: Raw waveforms for each cycle (np.ndarray)
            - validation_results: Quality metrics for each cycle
            - sample_rate: Sample rate for waveform playback
        """
        # DEBUG: Show recorder settings before recording
        st.info("🔍 DEBUG: Recorder settings at calibration test start:")
        debug_info = {
            "sample_rate": self.recorder.sample_rate,
            "pulse_duration": self.recorder.pulse_duration,
            "cycle_duration": self.recorder.cycle_duration,
            "num_pulses": self.recorder.num_pulses,
            "pulse_frequency": self.recorder.pulse_frequency,
            "impulse_form": self.recorder.impulse_form,
            "volume": self.recorder.volume,
            "multichannel_enabled": self.recorder.multichannel_config.get('enabled', False),
            "num_channels_config": self.recorder.multichannel_config.get('num_channels', 1)
        }
        st.json(debug_info)

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

        # Reshape calibration data into cycles
        expected_samples = self.recorder.cycle_samples * self.recorder.num_pulses

        # Pad or trim to expected length
        if len(cal_raw) < expected_samples:
            padded = np.zeros(expected_samples)
            padded[:len(cal_raw)] = cal_raw
            cal_raw = padded
        else:
            cal_raw = cal_raw[:expected_samples]

        # Reshape into individual cycles
        calibration_cycles = cal_raw.reshape(self.recorder.num_pulses, self.recorder.cycle_samples)

        # Run validation on each cycle (but don't filter them out)
        # Use V2 validator with negative pulse detection
        from calibration_validator_v2 import CalibrationValidatorV2, QualityThresholds

        # Create thresholds from config
        thresholds = QualityThresholds.from_config(self.recorder.calibration_quality_config)
        validator = CalibrationValidatorV2(thresholds, self.recorder.sample_rate)

        validation_results = []
        for cycle_idx in range(self.recorder.num_pulses):
            validation = validator.validate_cycle(
                calibration_cycles[cycle_idx],
                cycle_idx
            )
            # Convert to dict for easier handling in UI
            validation_dict = {
                'cycle_index': cycle_idx,
                'calibration_valid': validation.calibration_valid,
                'calibration_metrics': validation.calibration_metrics,
                'calibration_failures': validation.calibration_failures,
                'is_user_marked_good': False  # Will be set by user in UI
            }
            validation_results.append(validation_dict)

        return {
            'success': True,
            'num_cycles': self.recorder.num_pulses,
            'calibration_channel': cal_ch,
            'sample_rate': self.recorder.sample_rate,
            'all_calibration_cycles': calibration_cycles,  # All raw cycles (num_pulses x cycle_samples)
            'validation_results': validation_results,  # Per-cycle quality metrics
            'cycle_duration_s': self.recorder.cycle_samples / self.recorder.sample_rate
        }

    def _render_calibration_test_results(self, results: Dict):
        """
        Render calibration test results with per-cycle waveform visualization.

        Uses AudioVisualizer to display each calibration impulse cycle with
        quality metrics overlaid, allowing the user to explore and decide
        upon quality criteria.
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

        # User marking interface for automatic threshold learning
        st.markdown("#### Mark Good Cycles for Automatic Threshold Learning")
        st.markdown("Select cycles that represent **good quality** calibration impulses. "
                   "The system will calculate quality thresholds based on your selection.")

        # Initialize session state for user markings if not exists
        if 'cal_test_marked_good' not in st.session_state:
            st.session_state['cal_test_marked_good'] = []

        # Multi-select for marking good cycles
        marked_good = st.multiselect(
            "Mark Good Cycles (will be used to calculate thresholds)",
            options=list(range(num_cycles)),
            default=st.session_state.get('cal_test_marked_good', []),
            format_func=lambda x: f"Cycle {x}",
            key="cal_test_marked_good_selector",
            help="Select cycles that represent ideal calibration impulses"
        )

        # Update session state
        st.session_state['cal_test_marked_good'] = marked_good

        # Button to calculate thresholds
        if len(marked_good) > 0:
            if st.button("🎯 Calculate Thresholds from Marked Cycles", type="primary"):
                try:
                    import pandas as pd
                    from calibration_validator_v2 import calculate_thresholds_from_marked_cycles

                    # Calculate thresholds
                    learned_thresholds = calculate_thresholds_from_marked_cycles(
                        calibration_cycles,
                        marked_good,
                        sample_rate,
                        safety_margin=0.2  # 20% margin
                    )

                    # Store in session state for use in visualization
                    st.session_state['cal_test_learned_thresholds'] = learned_thresholds

                    # Display calculated thresholds
                    st.success(f"✓ Thresholds calculated from {len(marked_good)} marked cycles!")

                    # Show detailed analysis of marked cycles
                    st.markdown("**Analysis of Marked Cycles:**")

                    # Calculate statistics from marked cycles
                    marked_cycles_data = []
                    for idx in marked_good:
                        cycle = calibration_cycles[idx]
                        neg_peak_idx = np.argmin(cycle)
                        neg_peak = abs(cycle[neg_peak_idx])
                        pos_peak = np.max(cycle)

                        # Calculate aftershock
                        decay_skip_samples = int(2.0 * sample_rate / 1000)
                        window_start = neg_peak_idx + decay_skip_samples
                        window_end = min(len(cycle), neg_peak_idx + int(10.0 * sample_rate / 1000))
                        aftershock = 0.0
                        if window_end > window_start:
                            aftershock = np.max(np.abs(cycle[window_start:window_end]))

                        marked_cycles_data.append({
                            'Cycle': idx,
                            'Neg Peak': neg_peak,
                            'Pos Peak': pos_peak,
                            'Pos/Neg': pos_peak / neg_peak if neg_peak > 0 else 0,
                            'Aftershock': aftershock / neg_peak if neg_peak > 0 else 0
                        })

                    df_marked = pd.DataFrame(marked_cycles_data)
                    st.dataframe(df_marked.style.format({
                        'Neg Peak': '{:.3f}',
                        'Pos Peak': '{:.3f}',
                        'Pos/Neg': '{:.3f}',
                        'Aftershock': '{:.3f}'
                    }), use_container_width=True, hide_index=True)

                    # Show calculated thresholds in a clear table
                    st.markdown("**Calculated Quality Thresholds:**")

                    threshold_data = {
                        'Parameter': [
                            'Min Negative Peak',
                            'Max Negative Peak',
                            'Max Aftershock Ratio',
                            'Max Positive/Negative Ratio'
                        ],
                        'Value': [
                            f"{learned_thresholds.min_negative_peak:.3f}",
                            f"{learned_thresholds.max_negative_peak:.3f}",
                            f"{learned_thresholds.max_aftershock_ratio:.3f}",
                            f"{learned_thresholds.max_positive_peak_ratio:.3f}"
                        ],
                        'Interpretation': [
                            f"Based on min of marked cycles ({df_marked['Neg Peak'].min():.3f}) with 20% margin",
                            f"Based on max of marked cycles ({df_marked['Neg Peak'].max():.3f}) with 20% margin",
                            f"Based on max aftershock in marked cycles ({df_marked['Aftershock'].max():.3f}) with 20% margin",
                            f"Based on max pos/neg ratio in marked cycles ({df_marked['Pos/Neg'].max():.3f}) with 20% margin"
                        ]
                    }

                    df_thresholds = pd.DataFrame(threshold_data)
                    st.dataframe(df_thresholds, use_container_width=True, hide_index=True)

                    # Apply button
                    if st.button("✅ Apply These Thresholds to Configuration", type="secondary"):
                        # Update recorder config
                        self.recorder.calibration_quality_config.update(learned_thresholds.to_dict())
                        st.success("✓ Thresholds applied to configuration!")
                        st.info("Re-run the calibration test to validate with new thresholds")

                except Exception as e:
                    st.error(f"Failed to calculate thresholds: {e}")
                    import traceback
                    st.code(traceback.format_exc())
        else:
            st.info("👆 Select at least one good cycle to calculate thresholds")

        st.markdown("---")

        # Create a summary table
        st.markdown("#### Quality Metrics Summary")
        import pandas as pd

        table_data = []
        for v_result in validation_results:
            cycle_idx = v_result.get('cycle_index', 0)
            valid = v_result.get('calibration_valid', False)
            metrics = v_result.get('calibration_metrics', {})
            failures = v_result.get('calibration_failures', [])
            is_marked = cycle_idx in marked_good

            row = {
                'Cycle': cycle_idx,
                'Marked': '⭐' if is_marked else '',
                'Valid': '✓' if valid else '✗',
                'Neg Peak': f"{metrics.get('negative_peak', metrics.get('peak_amplitude', 0)):.3f}",
                'Aftershock': f"{metrics.get('aftershock_ratio', metrics.get('secondary_peak_ratio', 0)):.2f}",
                'Pos/Neg Ratio': f"{metrics.get('positive_peak_ratio', 0):.2f}",
                'Issues': ', '.join(failures) if failures else 'None'
            }
            table_data.append(row)

        if table_data:
            df = pd.DataFrame(table_data)
            st.dataframe(df, use_container_width=True, hide_index=True)

        st.markdown("---")

        # Per-cycle visualization section
        st.markdown("#### Per-Cycle Waveform Analysis")
        st.markdown("Explore individual calibration impulse cycles to assess quality and adjust thresholds.")

        # Cycle selector
        selected_cycle = st.selectbox(
            "Select Cycle to Visualize",
            options=list(range(num_cycles)),
            format_func=lambda x: f"Cycle {x} {'✓ Valid' if validation_results[x].get('calibration_valid', False) else '✗ Invalid'}",
            key="cal_test_cycle_selector"
        )

        # Get data for selected cycle
        cycle_waveform = calibration_cycles[selected_cycle]
        cycle_validation = validation_results[selected_cycle]
        cycle_metrics = cycle_validation.get('calibration_metrics', {})
        cycle_failures = cycle_validation.get('calibration_failures', [])
        is_valid = cycle_validation.get('calibration_valid', False)

        # Display cycle info
        col1, col2 = st.columns([2, 1])

        with col1:
            if is_valid:
                st.success(f"✓ Cycle {selected_cycle}: **VALID**")
            else:
                st.error(f"✗ Cycle {selected_cycle}: **INVALID** - {', '.join(cycle_failures)}")

        with col2:
            st.info(f"Duration: {cycle_duration_s * 1000:.1f} ms")

        # Display metrics in columns
        st.markdown("**Quality Metrics:**")
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            neg_peak = cycle_metrics.get('negative_peak', cycle_metrics.get('peak_amplitude', 0))
            st.metric("Negative Peak", f"{neg_peak:.3f}")
        with col2:
            aftershock = cycle_metrics.get('aftershock_ratio', cycle_metrics.get('secondary_peak_ratio', 0))
            st.metric("Aftershock Ratio", f"{aftershock:.3f}")
        with col3:
            pos_ratio = cycle_metrics.get('positive_peak_ratio', 0)
            st.metric("Positive/Negative", f"{pos_ratio:.3f}")
        with col4:
            # Show if cycle is user-marked
            is_marked = selected_cycle in st.session_state.get('cal_test_marked_good', [])
            if is_marked:
                st.metric("User Marked", "⭐ Good")
            else:
                st.metric("User Marked", "Not marked")

        # Visualize the waveform using AudioVisualizer
        if AUDIO_VISUALIZER_AVAILABLE and AudioVisualizer:
            st.markdown("**Waveform:**")

            # Primary visualization - AudioVisualizer (this was working correctly)
            visualizer = AudioVisualizer(component_id=f"cal_cycle_{selected_cycle}")
            visualizer.render(
                audio_data=cycle_waveform,
                sample_rate=sample_rate,
                title=f"Calibration Impulse - Cycle {selected_cycle}",
                show_controls=True,
                show_analysis=True,
                height=300
            )

            # Optional: Quality criteria overlay (if thresholds calculated)
            learned_thresholds = st.session_state.get('cal_test_learned_thresholds')
            if learned_thresholds:
                with st.expander("📊 Quality Criteria Overlay", expanded=False):
                    import matplotlib.pyplot as plt

                    fig, ax = plt.subplots(figsize=(14, 6))

                    # Time axis
                    time_ms = np.arange(len(cycle_waveform)) / sample_rate * 1000

                    # Plot waveform
                    ax.plot(time_ms, cycle_waveform, 'b-', linewidth=1.5, label='Calibration Impulse', alpha=0.8)
                    ax.axhline(y=0, color='k', linestyle='--', linewidth=0.5, alpha=0.3)
                    ax.grid(True, alpha=0.2)

                    # Mark negative peak
                    neg_peak_idx = cycle_metrics.get('negative_peak_idx', np.argmin(cycle_waveform))
                    neg_peak_time_ms = neg_peak_idx / sample_rate * 1000
                    neg_peak_val = cycle_waveform[neg_peak_idx]
                    ax.plot(neg_peak_time_ms, neg_peak_val, 'ro', markersize=10, label=f'Negative Peak: {abs(neg_peak_val):.3f}')

                    # Mark aftershock window (2-10ms after peak)
                    aftershock_start_ms = neg_peak_time_ms + 2.0
                    aftershock_end_ms = neg_peak_time_ms + 10.0
                    ax.axvspan(aftershock_start_ms, aftershock_end_ms, alpha=0.2, color='orange',
                              label=f'Aftershock Window (2-10ms)')

                    # Min/max negative peak thresholds
                    ax.axhline(y=-learned_thresholds.min_negative_peak, color='g', linestyle='--',
                              linewidth=2, alpha=0.6, label=f'Min Neg Peak: {learned_thresholds.min_negative_peak:.3f}')
                    ax.axhline(y=-learned_thresholds.max_negative_peak, color='r', linestyle='--',
                              linewidth=2, alpha=0.6, label=f'Max Neg Peak: {learned_thresholds.max_negative_peak:.3f}')

                    # Aftershock threshold
                    aftershock_threshold_val = learned_thresholds.max_aftershock_ratio * abs(neg_peak_val)
                    ax.axhline(y=aftershock_threshold_val, color='orange', linestyle=':',
                              linewidth=2, alpha=0.6, label=f'Aftershock Limit: {aftershock_threshold_val:.3f}')
                    ax.axhline(y=-aftershock_threshold_val, color='orange', linestyle=':',
                              linewidth=2, alpha=0.6)

                    # Labels and title
                    ax.set_xlabel('Time (ms)', fontsize=12)
                    ax.set_ylabel('Amplitude', fontsize=12)
                    ax.set_title(f'Cycle {selected_cycle} - Quality Criteria Overlay',
                                fontsize=14, fontweight='bold')
                    ax.legend(loc='upper right', fontsize=9)

                    # Set y-axis limits with some padding
                    y_min = min(cycle_waveform.min() * 1.2, -0.1)
                    y_max = max(cycle_waveform.max() * 1.2, 0.1)
                    ax.set_ylim(y_min, y_max)

                    # Zoom to relevant portion (first 20ms typically contains the impulse)
                    ax.set_xlim(0, min(20, time_ms[-1]))

                    plt.tight_layout()
                    st.pyplot(fig, use_container_width=True)
                    plt.close(fig)

                    st.info("🟢 Green line: Min acceptable peak | 🔴 Red line: Max acceptable peak | 🟠 Orange region: Aftershock window (2-10ms)")
            else:
                st.info("💡 Calculate thresholds above to enable quality criteria overlay visualization")

        else:
            st.warning("AudioVisualizer not available - cannot display waveform")
            st.info("Install gui_audio_visualizer.py to enable waveform visualization")

        st.markdown("---")

        # Comparison view - overlay multiple cycles
        st.markdown("#### Compare Multiple Cycles")
        st.markdown("Overlay multiple cycles to compare their waveforms and identify outliers.")

        # Multi-select for cycles to compare
        cycles_to_compare = st.multiselect(
            "Select cycles to overlay",
            options=list(range(num_cycles)),
            default=[0] if num_cycles > 0 else [],
            format_func=lambda x: f"Cycle {x} {'✓' if validation_results[x].get('calibration_valid', False) else '✗'}",
            key="cal_test_cycles_compare"
        )

        if len(cycles_to_compare) > 0:
            # Use AudioVisualizer static method for overlay plotting
            if AUDIO_VISUALIZER_AVAILABLE and AudioVisualizer:
                signals = [calibration_cycles[i] for i in cycles_to_compare]
                labels = [f"Cycle {i} {'✓' if validation_results[i].get('calibration_valid', False) else '✗'}"
                         for i in cycles_to_compare]

                fig = AudioVisualizer.render_overlay_plot(
                    audio_signals=signals,
                    sample_rate=sample_rate,
                    title="Calibration Impulse Comparison",
                    labels=labels,
                    normalize=True,
                    max_signals=8,
                    show_legend=True,
                    figsize=(12, 5),
                    alpha=0.7,
                    linewidth=1.2
                )

                st.pyplot(fig, use_container_width=True)
            else:
                st.info("Select AudioVisualizer component to enable overlay comparison")
        else:
            st.info("Select one or more cycles above to compare their waveforms")

        st.markdown("---")

        # User guidance
        with st.expander("💡 How to Use This Tool", expanded=False):
            st.markdown("""
            **Purpose:** This tool helps you establish quality criteria for calibration impulses based on actual measurements.

            **New Workflow (Automatic Threshold Learning):**
            1. **Run Test:** Click "Run Calibration Test" to record calibration impulses
            2. **Review Waveforms:** Examine each cycle using the visualizer and overlay comparison
            3. **Mark Good Cycles:** Select cycles that represent ideal calibration impulses (consistent, clean hammer strikes)
            4. **Calculate Thresholds:** Click "Calculate Thresholds from Marked Cycles" - the system automatically determines quality criteria
            5. **Apply & Validate:** Apply the calculated thresholds and re-run the test to verify
            6. **Iterate:** Refine your marking if needed and recalculate

            **Quality Criteria (Automatic Detection):**
            - **Negative Peak:** Strong negative pulse from hammer impact (NOT normalized - preserves amplitude)
            - **Aftershock:** No significant rebounds within 10ms of main pulse (should be < 50% of peak)
            - **Positive/Negative Ratio:** Predominantly negative pulse (positive component should be small)

            **What Makes a Good Calibration Impulse:**
            - Single strong negative pulse (hammer impact signature)
            - Sharp, clean waveform with quick decay
            - No aftershocks or bounces immediately after impact
            - Minimal positive component
            - Consistent amplitude across marked cycles

            **Tips:**
            - Mark at least 3-5 good cycles for reliable threshold calculation
            - Good cycles should be similar to each other (use overlay plot to verify)
            - The system adds 20% safety margin to calculated thresholds
            - Re-test after applying thresholds to ensure they work correctly
            - Invalid cycles don't mean bad data - they help identify quality variations
            """)

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
