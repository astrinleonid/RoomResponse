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
import json
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
            ("🎼 ESPRIT Aggregation", []),
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
        elif panel == "ESPRIT Aggregation":
            self._render_esprit_aggregation()
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

    def _render_esprit_aggregation(self):
        """ESPRIT aggregation panel for combining results from multiple scenarios."""
        st.header("🎼 ESPRIT Modal Analysis Aggregation")

        root = st.session_state.get(SK_DATASET_ROOT)
        if not os.path.isdir(root):
            st.error("Please set a valid dataset directory first")
            return

        try:
            from pathlib import Path
            from esprit_helper import ESPRITAggregator, ESPRITScenarioProcessor
            import json
        except ImportError as e:
            st.error(f"ESPRIT modules not available: {e}")
            return

        st.markdown("""
        This panel allows you to:
        1. Select scenarios from a collection (identified by Computer+Room name)
        2. Aggregate ESPRIT results across selected scenarios
        3. View stabilization diagram and mode shapes
        """)

        # Step 1: Filter by Computer + Room
        st.markdown("### 1. Select Collection")

        col1, col2 = st.columns(2)

        with col1:
            computer_name = st.text_input("Computer Name", value="", key="esprit_agg_computer")

        with col2:
            room_name = st.text_input("Room Name", value="", key="esprit_agg_room")

        if not computer_name or not room_name:
            st.info("👆 Enter Computer and Room names to find matching scenarios")
            return

        # Find matching scenarios
        scenarios = ESPRITAggregator.find_collection_scenarios(
            base_dir=Path(root),
            computer_name=computer_name,
            room_name=room_name
        )

        if not scenarios:
            st.warning(f"No scenarios found for Computer: {computer_name}, Room: {room_name}")
            return

        st.success(f"✓ Found {len(scenarios)} scenarios in this collection")

        # Step 2: Select scenarios to aggregate
        st.markdown("### 2. Select Scenarios to Aggregate")

        # Separate scenarios by ESPRIT status and signal data availability
        scenarios_with_esprit = []
        scenarios_with_signals = []
        scenarios_without_esprit = []

        for scenario_dir, scenario_num in scenarios:
            # Check for multi-band results (preferred) or single-band (legacy)
            esprit_multiband = scenario_dir / "analysis" / "esprit_all_bands_summary.json"
            esprit_single = scenario_dir / "analysis" / "esprit_single_point.json"
            # Check for signal data (needed for mode shapes)
            signal_file = scenario_dir / "analysis" / "esprit_band0_40-500Hz_signals.npy"

            if esprit_multiband.exists() or esprit_single.exists():
                scenarios_with_esprit.append((scenario_dir, scenario_num))
                if signal_file.exists():
                    scenarios_with_signals.append((scenario_dir, scenario_num))
            else:
                scenarios_without_esprit.append((scenario_dir, scenario_num))

        # Show detailed status
        st.info(f"📊 {len(scenarios_with_esprit)} scenarios with ESPRIT results, "
                f"{len(scenarios_without_esprit)} without")

        if scenarios_with_esprit:
            if len(scenarios_with_signals) == len(scenarios_with_esprit):
                st.success(f"✓ All {len(scenarios_with_signals)} ESPRIT scenarios have signal data for mode shapes")
            elif len(scenarios_with_signals) > 0:
                st.warning(f"⚠️ Only {len(scenarios_with_signals)}/{len(scenarios_with_esprit)} scenarios have signal data for mode shapes. "
                          f"Reprocess the others with 'Force reprocess' to enable mode shape visualization.")
            else:
                st.warning(f"⚠️ No scenarios have signal data for mode shapes. "
                          f"Reprocess with 'Force reprocess' to enable mode shape visualization.")

        # Option to process scenarios without ESPRIT results
        if scenarios_without_esprit or scenarios_with_esprit:
            with st.expander("🔄 Process/Reprocess Scenarios with ESPRIT", expanded=len(scenarios_with_esprit) == 0):
                st.markdown("Run ESPRIT analysis on scenarios that don't have results yet, or reprocess existing ones.")

                # Build lists for different categories
                scenarios_no_esprit = [(path, num) for path, num in scenarios_without_esprit]
                scenarios_no_signals = [(path, num) for path, num in scenarios_with_esprit
                                       if (path, num) not in scenarios_with_signals]
                scenarios_complete = [(path, num) for path, num in scenarios_with_esprit
                                     if (path, num) in scenarios_with_signals]

                # Quick selection buttons
                st.markdown("**Quick Selection:**")
                btn_col1, btn_col2, btn_col3, btn_col4 = st.columns(4)

                with btn_col1:
                    if st.button(f"📋 Unprocessed ({len(scenarios_no_esprit)})",
                                disabled=len(scenarios_no_esprit) == 0,
                                help="Select scenarios without any ESPRIT results"):
                        st.session_state['esprit_quick_select'] = 'no_esprit'
                        st.rerun()

                with btn_col2:
                    if st.button(f"📋 No Signals ({len(scenarios_no_signals)})",
                                disabled=len(scenarios_no_signals) == 0,
                                help="Select scenarios that need reprocessing for mode shapes"):
                        st.session_state['esprit_quick_select'] = 'no_signals'
                        st.rerun()

                with btn_col3:
                    if st.button(f"📋 All Incomplete ({len(scenarios_no_esprit) + len(scenarios_no_signals)})",
                                disabled=len(scenarios_no_esprit) + len(scenarios_no_signals) == 0,
                                help="Select all scenarios that need processing"):
                        st.session_state['esprit_quick_select'] = 'all_incomplete'
                        st.rerun()

                with btn_col4:
                    if st.button("📋 Select All", help="Select all scenarios"):
                        st.session_state['esprit_quick_select'] = 'all'
                        st.rerun()

                # Build options dict with status labels
                process_options = {}
                for path, num in scenarios_no_esprit:
                    process_options[f"Scenario {num} (no ESPRIT)"] = (path, num)
                for path, num in scenarios_no_signals:
                    process_options[f"Scenario {num} (no signals)"] = (path, num)
                for path, num in scenarios_complete:
                    process_options[f"Scenario {num} (complete)"] = (path, num)

                # Determine default selection based on quick select or smart default
                quick_select = st.session_state.get('esprit_quick_select', None)
                if quick_select == 'no_esprit':
                    default_process = [f"Scenario {num} (no ESPRIT)" for _, num in scenarios_no_esprit]
                elif quick_select == 'no_signals':
                    default_process = [f"Scenario {num} (no signals)" for _, num in scenarios_no_signals]
                elif quick_select == 'all_incomplete':
                    default_process = ([f"Scenario {num} (no ESPRIT)" for _, num in scenarios_no_esprit] +
                                      [f"Scenario {num} (no signals)" for _, num in scenarios_no_signals])
                elif quick_select == 'all':
                    default_process = list(process_options.keys())
                else:
                    # Smart default: prioritize unprocessed, then no signals
                    if scenarios_no_esprit:
                        default_process = [f"Scenario {num} (no ESPRIT)" for _, num in scenarios_no_esprit]
                    elif scenarios_no_signals:
                        default_process = [f"Scenario {num} (no signals)" for _, num in scenarios_no_signals]
                    else:
                        default_process = []

                # Clear quick select after use
                if 'esprit_quick_select' in st.session_state:
                    del st.session_state['esprit_quick_select']

                process_labels = st.multiselect(
                    "Select scenarios to process:",
                    options=list(process_options.keys()),
                    default=default_process,
                    help="Select scenarios to run ESPRIT analysis on"
                )

                if process_labels:
                    col1, col2 = st.columns(2)
                    with col1:
                        process_K = st.number_input("Model Order (K)", min_value=10, max_value=50, value=30, key="process_K")
                    with col2:
                        process_L = st.slider("Hankel Window", min_value=0.3, max_value=0.7, value=0.5, key="process_L")

                    # Auto-enable force reprocess if any selected scenarios have existing results
                    has_existing = any("no signals" in label or "complete" in label for label in process_labels)
                    force_default = has_existing and any("no signals" in label for label in process_labels)

                    force_reprocess = st.checkbox(
                        "Force reprocess (overwrite existing results)",
                        value=force_default,
                        help="Required when reprocessing scenarios that already have ESPRIT results"
                    )

                    if has_existing and not force_reprocess:
                        st.warning("⚠️ Some selected scenarios already have ESPRIT results. Enable 'Force reprocess' to update them.")

                    run_button = st.button("🎼 Run ESPRIT on Selected Scenarios", type="secondary", key="run_esprit_btn")

                    if run_button:
                        scenarios_to_process = [process_options[label] for label in process_labels]
                        st.info(f"Starting ESPRIT processing for {len(scenarios_to_process)} scenarios...")
                        print(f"\n{'='*70}")
                        print(f"ESPRIT BATCH PROCESSING: {len(scenarios_to_process)} scenarios")
                        print(f"{'='*70}")

                        progress_bar = st.progress(0)
                        status_text = st.empty()

                        processed_count = 0
                        failed_count = 0

                        # Build ESPRIT config once (outside loop)
                        esprit_config = {
                            'enabled': True,
                            'M_out': 7,
                            'N_use': 28800,
                            'fs': 48000,
                            'L_fraction': process_L,
                            'K': process_K,
                            'skip_m': 0,
                            'selected_channels': [2, 3, 4, 5, 6]
                        }

                        # Load config from recorderConfig.json if available
                        config_file = Path(root) / "recorderConfig.json"
                        if config_file.exists():
                            print(f"Loading config from: {config_file}")
                            with open(config_file, 'r') as f:
                                recorder_config = json.load(f)
                            mc_config = recorder_config.get('multichannel_config', {})
                            esprit_saved = recorder_config.get('esprit_config', {})

                            esprit_config['M_out'] = mc_config.get('num_channels', 7)
                            esprit_config['skip_m'] = mc_config.get('calibration_channel', 0)
                            esprit_config['fs'] = recorder_config.get('sample_rate', 48000)

                            if esprit_saved.get('selected_channels'):
                                esprit_config['selected_channels'] = esprit_saved['selected_channels']

                            truncate_config = recorder_config.get('truncate_config', {})
                            if truncate_config.get('enabled'):
                                ir_length_ms = truncate_config.get('ir_working_length_ms', 600)
                                esprit_config['N_use'] = int(esprit_config['fs'] * ir_length_ms / 1000)

                        print(f"ESPRIT Config: K={esprit_config['K']}, L={esprit_config['L_fraction']}, channels={esprit_config['selected_channels']}")

                        for i, (scenario_dir, scenario_num) in enumerate(scenarios_to_process):
                            status_text.text(f"Processing Scenario {scenario_num} ({i+1}/{len(scenarios_to_process)})...")
                            progress_bar.progress((i) / len(scenarios_to_process))
                            print(f"\n--- Processing Scenario {scenario_num} ---")

                            try:
                                result = ESPRITScenarioProcessor.process_scenario(
                                    scenario_dir=scenario_dir,
                                    esprit_config=esprit_config.copy(),
                                    force_reprocess=force_reprocess
                                )

                                if result:
                                    processed_count += 1
                                    print(f"OK Scenario {scenario_num} processed successfully")
                                else:
                                    failed_count += 1
                                    print(f"FAILED Scenario {scenario_num} returned None")

                            except Exception as e:
                                st.error(f"Failed to process Scenario {scenario_num}: {e}")
                                print(f"ERROR Scenario {scenario_num}: {e}")
                                import traceback
                                traceback.print_exc()
                                failed_count += 1

                        progress_bar.progress(1.0)
                        status_text.text("Processing complete!")

                        print(f"\n{'='*70}")
                        print(f"BATCH COMPLETE: {processed_count} success, {failed_count} failed")
                        print(f"{'='*70}\n")

                        if processed_count > 0:
                            st.success(f"✓ Successfully processed {processed_count} scenarios")
                        if failed_count > 0:
                            st.warning(f"⚠️ {failed_count} scenarios failed to process")

                        st.balloons()
                        # Don't rerun immediately - let user see results
                        if st.button("🔄 Refresh to see updated results"):
                            st.rerun()

        if not scenarios_with_esprit:
            st.warning("No scenarios have ESPRIT results yet. Use the section above to process them.")
            return

        # Multi-select for scenarios
        scenario_options = {f"Scenario {num}": (path, num)
                          for path, num in scenarios_with_esprit}

        selected_labels = st.multiselect(
            "Select scenarios to include in aggregation:",
            options=list(scenario_options.keys()),
            default=list(scenario_options.keys())  # Select all by default
        )

        if not selected_labels:
            st.warning("No scenarios selected")
            return

        selected_scenarios = [scenario_options[label] for label in selected_labels]

        st.caption(f"Selected: {len(selected_scenarios)} scenarios")

        # Step 3: Run aggregation
        st.markdown("### 3. Aggregate Results")

        # Band selection
        band_options = {
            "Band 0: 40-500 Hz (Piano fundamentals)": 0,
            "Band 1: 500-1000 Hz (Low harmonics)": 1,
            "Band 2: 1000-2000 Hz (Mid harmonics)": 2,
            "Band 3: 2000-4000 Hz (High harmonics)": 3,
        }

        selected_band_label = st.selectbox(
            "Frequency Band to Aggregate:",
            options=list(band_options.keys()),
            index=0,
            help="Select which frequency band to aggregate across scenarios"
        )
        selected_band_index = band_options[selected_band_label]

        # Mode detection criteria
        with st.expander("Mode Detection Criteria", expanded=False):
            st.markdown("""
            **How modes are identified as "common":**
            - Modes from different scenarios are grouped if their frequencies are within the tolerance
            - A mode group is kept only if it appears in enough scenarios (minimum occurrence)
            """)

            crit_col1, crit_col2 = st.columns(2)
            with crit_col1:
                min_occurrence_pct = st.slider(
                    "Minimum Occurrence (%)",
                    min_value=10,
                    max_value=100,
                    value=30,
                    step=5,
                    help="Mode must appear in at least this % of scenarios to be considered common"
                )
            with crit_col2:
                freq_tolerance_pct = st.slider(
                    "Frequency Tolerance (%)",
                    min_value=0.5,
                    max_value=10.0,
                    value=2.0,
                    step=0.5,
                    help="Modes within this % of each other are considered the same mode"
                )

            st.caption(f"With {len(selected_scenarios)} scenarios and {min_occurrence_pct}% threshold, "
                      f"a mode must appear in at least {max(1, int(len(selected_scenarios) * min_occurrence_pct / 100))} scenarios")

        col1, col2 = st.columns(2)

        with col1:
            collection_name = st.text_input(
                "Collection Name",
                value=f"{computer_name}_{room_name}",
                help="Name for this aggregated analysis"
            )

        with col2:
            band_suffix = f"_band{selected_band_index}"
            output_filename = st.text_input(
                "Output Filename",
                value=f"esprit_aggregated_{computer_name}_{room_name}{band_suffix}.json",
                help="Filename for aggregated results"
            )

        if st.button("🎼 Run Aggregation", type="primary", use_container_width=True):
            with st.spinner(f"Aggregating ESPRIT results for {selected_band_label}..."):
                selected_dirs = [path for path, num in selected_scenarios]
                output_file = Path(root) / output_filename

                results = ESPRITAggregator.aggregate_scenarios(
                    scenario_dirs=selected_dirs,
                    output_file=output_file,
                    collection_name=collection_name,
                    band_index=selected_band_index,
                    min_occurrence_pct=float(min_occurrence_pct),
                    freq_tolerance_pct=float(freq_tolerance_pct)
                )

                if results:
                    st.success(f"✓ Aggregation complete for {results.get('band_name', 'Band ' + str(selected_band_index))}!")
                    st.session_state["esprit_agg_last_results"] = results
                    st.session_state["esprit_agg_output_file"] = str(output_file)
                else:
                    st.error("Aggregation failed. Check console for details.")

        # Step 4: Display results
        if "esprit_agg_last_results" in st.session_state:
            st.markdown("### 4. Aggregated Results")

            results = st.session_state["esprit_agg_last_results"]
            output_file = st.session_state.get("esprit_agg_output_file", "")

            common_f = results.get('common_f', [])
            common_z = results.get('common_z', [])
            num_scenarios = results.get('num_scenarios', 0)
            band_name = results.get('band_name', 'Unknown')

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Frequency Band", band_name)
            with col2:
                st.metric("Common Modes", len(common_f))
            with col3:
                st.metric("Scenarios", num_scenarios)

            if output_file:
                st.info(f"📁 Results saved to: `{output_file}`")

            # Display mode table
            if common_f:
                import pandas as pd

                mode_occurrences = results.get('mode_occurrences', [num_scenarios] * len(common_f))

                df = pd.DataFrame({
                    'Mode': range(len(common_f)),
                    'Frequency (Hz)': [f"{f:.2f}" for f in common_f],
                    'Damping Ratio': [f"{z:.4f}" for z in common_z],
                    'Damping (%)': [f"{z*100:.2f}" for z in common_z],
                    'Q Factor': [f"{1/(2*z):.1f}" if z > 0 else "∞" for z in common_z],
                    'Occurrence': [f"{occ}/{num_scenarios} ({occ/num_scenarios*100:.0f}%)"
                                  for occ in mode_occurrences]
                })

                st.dataframe(df, use_container_width=True)

                # Plot frequency spectrum
                import matplotlib.pyplot as plt

                fig, ax = plt.subplots(figsize=(12, 6))
                ax.stem(range(len(common_f)), common_f, basefmt=' ')
                ax.set_xlabel('Mode Index')
                ax.set_ylabel('Frequency (Hz)')
                ax.set_title(f'Modal Frequencies - {collection_name}')
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)

                # Plot mode shapes if available
                signed_shapes = results.get('signed_shapes', [])
                scenario_names = results.get('scenario_names', [])

                if signed_shapes and len(signed_shapes) > 0:
                    st.markdown("#### Mode Shapes")

                    mode_idx = st.slider(
                        "Select mode to view:",
                        min_value=0,
                        max_value=len(common_f) - 1,
                        value=0
                    )

                    fig, ax = plt.subplots(figsize=(12, 6))
                    shape = signed_shapes[mode_idx]
                    ax.plot(range(len(shape)), shape, 'o-', linewidth=2, markersize=8)
                    ax.set_xlabel('Scenario Index')
                    ax.set_ylabel('Signed Amplitude (Normalized)')
                    ax.set_title(f'Mode Shape {mode_idx}: f={common_f[mode_idx]:.2f} Hz, ζ={common_z[mode_idx]:.4f}')
                    ax.set_xticks(range(len(shape)))
                    ax.set_xticklabels(scenario_names, rotation=45, ha='right')
                    ax.grid(True, alpha=0.3)
                    ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
                    plt.tight_layout()
                    st.pyplot(fig)


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
