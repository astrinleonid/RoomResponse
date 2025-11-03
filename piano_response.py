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

try:
    from gui_config_profiles import ConfigProfileManager
except ImportError:
    ConfigProfileManager = None

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
        self.config_profile_manager = None
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
        if ConfigProfileManager is not None:
            self.config_profile_manager = ConfigProfileManager(recorder=self.recorder)

    def run(self):
        """Main application entry point."""
        st.set_page_config(
            page_title="Audio Collection & Impulse Response Tool",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        self._ensure_initial_state()
        self._ensure_dataset_root_ui()
        if self.config_profile_manager is not None:
            self.config_profile_manager.render_sidebar_ui()
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
        st.sidebar.markdown("### Dataset")

        current_root = st.session_state.get(SK_DATASET_ROOT,
                                          os.path.join(os.getcwd(), SK_DEFAULT_DATASET_ROOT))
        default_name = os.path.basename(current_root) if os.path.basename(current_root) else SK_DEFAULT_DATASET_ROOT

        st.session_state.setdefault(SK_DATASET_NAME, default_name)

        # Compact folder input
        st.sidebar.text_input(
            "Folder",
            key=SK_DATASET_NAME,
            help="Folder name or full path"
        )

        typed = (st.session_state.get(SK_DATASET_NAME) or "").strip()

        # Resolve path
        if not typed:
            resolved = os.path.abspath(current_root)
        elif self._looks_like_path(typed):
            resolved = os.path.abspath(os.path.expanduser(typed))
        else:
            resolved = os.path.abspath(os.path.join(os.getcwd(), typed))

        # Validation and status (compact)
        if self.scenario_manager is not None:
            ok, msg = self.scenario_manager.validate_dataset_root(resolved)
            if ok:
                st.sidebar.success("✓ Valid")
            else:
                st.sidebar.error(f"✗ {msg}")

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
        """Render hierarchical navigation menu using radio buttons."""
        st.sidebar.markdown("---")
        st.sidebar.markdown("### Navigation")

        # Initialize selected_panel if not set
        if 'selected_panel' not in st.session_state:
            st.session_state['selected_panel'] = "Collect"
            st.session_state['selected_section'] = "Single Scenario"

        # Build hierarchical navigation options
        nav_options = [
            ("📁 Collect", [
                ("  → Single Scenario", "Collect", "Single Scenario"),
                ("  → Series", "Collect", "Series"),
            ]),
            ("⚙️ Audio Settings", [
                ("  → Device Selection & Testing", "Audio Settings", "device_selection"),
                ("  → Multi-Channel Configuration", "Audio Settings", "multichannel"),
                ("  → Calibration Impulse", "Audio Settings", "calibration"),
                ("  → Series Settings", "Audio Settings", "series_settings"),
            ]),
            ("🎬 Scenarios", []),
            ("📊 Audio Analysis", []),
        ]

        # Flatten options for radio button
        radio_options = []
        option_map = {}  # Map display text to (panel, section)

        for group_name, subitems in nav_options:
            if subitems:
                # Group header (non-selectable, shown with bold)
                radio_options.append(f"**{group_name}**")
                option_map[f"**{group_name}**"] = (None, None)  # Header, not selectable

                # Sub-items
                for display, panel, section in subitems:
                    radio_options.append(display)
                    option_map[display] = (panel, section)
            else:
                # Top-level item (no sub-items)
                radio_options.append(group_name)
                # Extract panel name from group_name (remove emoji)
                panel_name = group_name.split(" ", 1)[1] if " " in group_name else group_name
                option_map[group_name] = (panel_name, None)

        # Determine current selection for radio button default
        current_panel = st.session_state.get('selected_panel', "Collect")
        current_section = st.session_state.get('selected_section', "Single Scenario")

        # Find matching radio option
        default_idx = 0
        for idx, opt in enumerate(radio_options):
            if opt in option_map:
                panel, section = option_map[opt]
                if panel == current_panel and section == current_section:
                    default_idx = idx
                    break
                elif panel == current_panel and section is None:
                    default_idx = idx

        # Render radio button navigation with callback
        def on_nav_change():
            """Callback to handle navigation changes immediately."""
            selected = st.session_state.nav_radio_selection

            if selected in option_map:
                panel, section = option_map[selected]

                # Skip if it's a header
                if panel is None:
                    return

                # Update session state based on selection
                st.session_state['selected_panel'] = panel
                st.session_state['selected_section'] = section

                # Handle specific panel/section routing
                if panel == "Collect":
                    st.session_state['collect_mode'] = section
                elif panel == "Audio Settings":
                    st.session_state['audio_settings_focus'] = section
                else:
                    # Clear any focus for other panels
                    if 'audio_settings_focus' in st.session_state:
                        del st.session_state['audio_settings_focus']
                    if 'collect_mode' in st.session_state:
                        del st.session_state['collect_mode']

        selected = st.sidebar.radio(
            "Select section:",
            options=radio_options,
            index=default_idx,
            key="nav_radio_selection",
            on_change=on_nav_change,
            label_visibility="collapsed"
        )

        return st.session_state['selected_panel']

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
