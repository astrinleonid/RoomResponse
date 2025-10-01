#!/usr/bin/env python3
"""
Room Response GUI - Simplified Audio Collection Interface

Focused on audio sample collection and impulse response extraction.
Removes machine learning pipeline components (Process, Classify, Predict, Visualize).
Streamlined for robust audio measurement workflows.
"""

import os
import sys
import string
import streamlit as st

# Import core audio components
try:
    from RoomResponseRecorder import RoomResponseRecorder
except ImportError:
    RoomResponseRecorder = None

try:
    from ScenarioManager import ScenarioManager
except ImportError:
    ScenarioManager = None

try:
    from gui_collect_panel import CollectionPanel
except ImportError:
    CollectionPanel = None

# Optional: Audio analysis panel (if you want basic audio file inspection)
try:
    from gui_audio_panel import AudioAnalysisPanel
except ImportError:
    AudioAnalysisPanel = None

try:
    from gui_audio_settings_panel import AudioSettingsPanel
except ImportError:
    AudioSettingsPanel = None

try:
    from gui_scenarios_panel import ScenariosPanel
except ImportError:
    ScenariosPanel = None

# ---------------------------- Session Keys ----------------------------
SK_DATASET_ROOT = "dataset_root"
SK_DEFAULT_DATASET_ROOT = "piano"

# Dataset picker keys
SK_DATASET_NAME = "dataset_folder_name"
SK_DATASET_NAME_PENDING = "dataset_folder_name_pending"
SK_BROWSER_OPEN = "dataset_browser_open"
SK_BROWSER_CWD = "dataset_browser_cwd"
SK_BROWSER_FILTER = "dataset_browser_filter"


class AudioCollectionGUI:
    """Simplified GUI focused on audio collection and impulse response measurement."""

    def __init__(self):
        self.scenario_manager = None
        self.scenarios_panel = None
        self.collection_panel = None
        self.audio_panel = None
        self.audio_settings_panel = None
        self.recorder = None
        self._initialize_components()

    def _initialize_components(self):
        """Initialize only the audio-focused components."""
        if RoomResponseRecorder is not None:
            try:
                # Persist a single recorder instance across Streamlit reruns.
                if "recorder" not in st.session_state or st.session_state["recorder"] is None:
                    st.session_state["recorder"] = RoomResponseRecorder()
                self.recorder = st.session_state["recorder"]
            except Exception:
                self.recorder = None
        if ScenarioManager is not None:
            self.scenario_manager = ScenarioManager()
        if ScenariosPanel and self.scenario_manager:
            self.scenarios_panel = ScenariosPanel(self.scenario_manager)
        if CollectionPanel and self.scenario_manager:
            self.collection_panel = CollectionPanel(self.scenario_manager, recorder=self.recorder)
        if AudioAnalysisPanel and self.scenario_manager:
            self.audio_panel = AudioAnalysisPanel(self.scenario_manager)
        if AudioSettingsPanel:
            self.audio_settings_panel = AudioSettingsPanel(self.scenario_manager, recorder=self.recorder)

    def run(self):
        """Main application entry point."""
        st.set_page_config(
            page_title="Audio Collection & Impulse Response Tool",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        self._ensure_initial_state()
        self._ensure_dataset_root_ui()
        panel = self._render_sidebar_navigation()
        self._render_panel(panel)

    def _ensure_initial_state(self):
        """Initialize session state with audio-focused defaults."""
        st.session_state.setdefault(SK_BROWSER_OPEN, False)
        st.session_state.setdefault(SK_BROWSER_FILTER, "")

        # Initialize dataset defaults
        default_root = st.session_state.get(
            SK_DATASET_ROOT,
            os.path.join(os.getcwd(), SK_DEFAULT_DATASET_ROOT)
        )
        default_root = os.path.abspath(default_root)
        st.session_state.setdefault(SK_DATASET_ROOT, default_root)

        # Handle pending folder name updates
        pending = st.session_state.pop(SK_DATASET_NAME_PENDING, None)
        if pending is not None:
            st.session_state[SK_DATASET_NAME] = pending
        else:
            current_name = os.path.basename(st.session_state[SK_DATASET_ROOT])
            st.session_state.setdefault(SK_DATASET_NAME, current_name or SK_DEFAULT_DATASET_ROOT)

    def _ensure_dataset_root_ui(self) -> str:
        """Dataset root selection interface."""
        st.sidebar.markdown("### Dataset Location")

        current_root = st.session_state.get(SK_DATASET_ROOT,
                                          os.path.join(os.getcwd(), SK_DEFAULT_DATASET_ROOT))
        default_name = os.path.basename(current_root) if os.path.basename(current_root) else SK_DEFAULT_DATASET_ROOT

        st.session_state.setdefault(SK_DATASET_NAME, default_name)

        # Compact folder input
        st.sidebar.text_input(
            "Dataset folder (name or path)",
            key=SK_DATASET_NAME,
            value=st.session_state[SK_DATASET_NAME],
            help="Enter folder name (creates subfolder in working directory) or full path"
        )

        typed = (st.session_state.get(SK_DATASET_NAME) or "").strip()

        # Resolve path
        if not typed:
            resolved = os.path.abspath(current_root)
        elif self._looks_like_path(typed):
            resolved = os.path.abspath(os.path.expanduser(typed))
        else:
            resolved = os.path.abspath(os.path.join(os.getcwd(), typed))

        # Validation and status
        if self.scenario_manager is not None:
            ok, msg = self.scenario_manager.validate_dataset_root(resolved)
            icon = "OK" if ok else "ERROR"
            (st.sidebar.success if ok else st.sidebar.error)(f"{icon} {msg}")

        # Show resolved path
        st.sidebar.caption("Resolved path:")
        st.sidebar.code(resolved, language=None)

        # Handle root changes
        if self.scenario_manager is not None:
            last_root = st.session_state.get(self.scenario_manager.SK_LAST_DATASET_ROOT)
            if last_root != resolved:
                st.session_state[self.scenario_manager.SK_LAST_DATASET_ROOT] = resolved
                self.scenario_manager.clear_cache()

        st.session_state[SK_DATASET_ROOT] = resolved
        return resolved

    def _looks_like_path(self, s: str) -> bool:
        """Check if string looks like a file path rather than just a folder name."""
        return (os.path.isabs(s) or
                s.startswith("~") or
                ("/" in s) or
                ("\\" in s) or
                (len(s) >= 2 and s[1] == ":"))  # Windows drive letter

    def _render_sidebar_navigation(self) -> str:
        """Render navigation menu focused on audio workflow."""
        st.sidebar.markdown("---")
        st.sidebar.markdown("### Audio Tools")

        # Check SDL audio core status
        try:
            import sdl_audio_core as sdl
            st.sidebar.success("Audio engine ready")
            # Show basic audio info
            with st.sidebar.expander("Audio System Info"):
                    try:
                        drivers = sdl.AudioEngine.get_audio_drivers()  # was sdl.get_audio_drivers()
                        st.write(f"**Available drivers:** {', '.join(drivers)}")

                        devices = sdl.list_all_devices()
                        st.write(
                            f"**Total devices:** "
                            f"{len(devices.get('input_devices', []))} input, "
                            f"{len(devices.get('output_devices', []))} output"
                        )
                    except Exception as e:
                        st.write(f"Info unavailable: {e}")
        except ImportError:
            st.sidebar.error("SDL audio core not available")
            st.sidebar.caption("Run build script to compile audio engine")

        # Main panel selection
        options = ["Scenarios", "Collect", "Audio Settings", "Audio Analysis"]
        selected = st.sidebar.radio(
            "Select panel",
            options=options,
            index=0,
            help="Choose your workflow step"
        )

        return selected

    def _render_panel(self, panel: str):
        """Render the selected panel."""
        if panel == "Scenarios":
            if self.scenarios_panel:
                self.scenarios_panel.render()
            else:
                st.error("Scenarios panel not available")
                st.info("Check that gui_scenarios_panel.py is present and properly imported")
        elif panel == "Collect":
            if self.collection_panel:
                self.collection_panel.render()
            else:
                st.error("Collection panel not available")
                st.info("Check that gui_collect_panel.py is present and properly imported")
        elif panel == "Audio Settings":
            if self.audio_settings_panel:
                self.audio_settings_panel.render()
            else:
                st.error("Audio Settings panel not available")
                st.info("Check that gui_audio_settings_panel.py is present and properly imported")
        elif panel == "Audio Analysis":
            if self.audio_panel:
                self.audio_panel.render()
            else:
                self._render_basic_audio_analysis()
        else:
            st.info(f"Panel '{panel}' not implemented.")

    def _render_basic_audio_analysis(self):
        """Basic audio file analysis when dedicated panel isn't available."""
        st.header("Audio Analysis")

        root = st.session_state.get(SK_DATASET_ROOT)
        if not os.path.isdir(root):
            st.error("Please set a valid dataset directory first")
            return

        st.info("Basic audio analysis panel - upload audio files for quick inspection")

        uploaded_file = st.file_uploader(
            "Upload audio file for analysis",
            type=['wav', 'mp3', 'flac', 'ogg']
        )

        if uploaded_file:
            st.audio(uploaded_file, format='audio/wav')
            st.success(f"File: {uploaded_file.name}")
            st.info("For advanced analysis features, implement gui_audio_panel.py")


def main():
    """Application entry point."""
    try:
        app = AudioCollectionGUI()
        app.run()
    except Exception as e:
        st.error(f"Application error: {e}")
        st.exception(e)


if __name__ == "__main__":
    main()
