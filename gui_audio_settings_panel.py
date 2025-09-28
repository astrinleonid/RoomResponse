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

        # Tabs: System Info + Device Selection (+ Series Settings when available)
        if SERIES_SETTINGS_AVAILABLE and self._series_settings_panel:
            tab1, tab2, tab3 = st.tabs(["System Info", "Device Selection", "Series Settings"])
        else:
            tab1, tab2 = st.tabs(["System Info", "Device Selection"])
            tab3 = None  # keep a name for type clarity

        with tab1:
            self._render_system_info()
            self._render_paths_and_modules()

        with tab2:
            self._render_device_selection_tab()

        if tab3:
            with tab3:
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
