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

# ---------------------------- Session Keys ----------------------------
SK_DATASET_ROOT = "dataset_root"
SK_DEFAULT_DATASET_ROOT = "room_response_dataset"
SK_SCN_SELECTIONS = "scenarios_selected_set"
SK_SCN_EXPLORE = "scenarios_explore_path"
SK_FILTER_TEXT = "filter_text"
SK_FILTER_COMPUTER = "filter_computer"
SK_FILTER_ROOM = "filter_room"
SK_SAVED_LABELS = "saved_labels_cache"

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
        self.collection_panel = None
        self.audio_panel = None
        self.audio_settings_panel = None
        self._initialize_components()

    def _initialize_components(self):
        """Initialize only the audio-focused components."""
        if ScenarioManager is not None:
            self.scenario_manager = ScenarioManager()
        if CollectionPanel and self.scenario_manager:
            self.collection_panel = CollectionPanel(self.scenario_manager)
        if AudioAnalysisPanel and self.scenario_manager:
            self.audio_panel = AudioAnalysisPanel(self.scenario_manager)
        if AudioSettingsPanel:
            self.audio_settings_panel = AudioSettingsPanel(self.scenario_manager)

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
        st.session_state.setdefault(SK_SCN_SELECTIONS, set())
        st.session_state.setdefault(SK_SAVED_LABELS, {})
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
                st.session_state[SK_SCN_SELECTIONS] = set()
                st.session_state[SK_SCN_EXPLORE] = None

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
                    drivers = sdl.get_audio_drivers()
                    st.write(f"**Available drivers:** {', '.join(drivers)}")
                    devices = sdl.list_all_devices()
                    st.write(f"**Total devices:** {len(devices.get('input', []))} input, {len(devices.get('output', []))} output")
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
            self._render_scenarios_panel()
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

    def _render_scenarios_panel(self):
        """Render scenarios overview panel."""
        if self.scenario_manager is None:
            st.error("ScenarioManager not available.")
            return

        st.header("Scenarios Overview")
        root = st.session_state.get(SK_DATASET_ROOT, os.getcwd())
        
        if not os.path.isdir(root):
            st.error("Please provide a valid dataset root directory.")
            return

        # Filters
        self._render_scenario_filters()

        # Load scenarios
        with st.spinner("Loading scenarios..."):
            df = self.scenario_manager.build_scenarios_df(root)
            
        if df.empty:
            st.info("No scenarios found. Use the Collect panel to create recordings.")
            st.markdown("**Quick Start:**")
            st.markdown("1. Go to **Collect** panel")
            st.markdown("2. Configure your audio settings")
            st.markdown("3. Start recording scenarios")
            return

        # Apply filters
        dfv = self.scenario_manager.apply_filters(
            df,
            st.session_state.get(SK_FILTER_TEXT, ""),
            st.session_state.get(SK_FILTER_COMPUTER, ""),
            st.session_state.get(SK_FILTER_ROOM, ""),
        )
        dfv = self.scenario_manager.sort_scenarios_df(dfv)

        # Bulk operations (simplified for audio workflow)
        self._render_scenario_bulk_operations(df, dfv)
        
        # Selection controls
        self._render_scenario_selection_controls(dfv, len(df))
        
        # Scenarios table
        self._render_scenarios_table(dfv)
        
        # Summary and explorer
        self._render_scenario_summary(df, dfv)
        self._render_scenario_explorer()

    def _render_scenario_filters(self):
        """Render scenario filtering controls."""
        col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
        
        with col1:
            st.text_input(
                "Filter scenarios (regex supported)",
                key=SK_FILTER_TEXT,
                placeholder="e.g., ^1\\., quiet, room.*",
                help="Filter by scenario number or description using regex"
            )
        with col2:
            st.text_input("Computer", key=SK_FILTER_COMPUTER)
        with col3:
            st.text_input("Room", key=SK_FILTER_ROOM)
        with col4:
            if st.button("Refresh", help="Re-scan dataset directory"):
                self.scenario_manager.clear_cache()
                st.rerun()

    def _render_scenario_bulk_operations(self, df, dfv):
        """Simplified bulk operations for audio workflow."""
        st.markdown("### Bulk Label Management")
        
        existing_labels = sorted(self.scenario_manager.get_unique_labels(df))
        if existing_labels:
            st.caption(f"Existing labels: {', '.join(existing_labels[:10])}")

        with st.form("bulk_label_form"):
            col1, col2 = st.columns([3, 1])
            
            with col1:
                label_text = st.text_input(
                    "Label to apply", 
                    placeholder="quiet, baseline, test-setup-1",
                    help="Comma-separated labels"
                )
                append_mode = st.checkbox("Append to existing labels", value=False)
            
            with col2:
                apply_btn = st.form_submit_button(f"Apply to {len(dfv)} scenarios")
                clear_btn = st.form_submit_button(f"Clear all labels")

        if apply_btn and label_text.strip():
            updated = self._bulk_apply_labels(dfv, label_text.strip(), append_mode)
            if updated > 0:
                st.success(f"Updated {updated} scenarios")
                self.scenario_manager.clear_cache()
                st.rerun()

        if clear_btn:
            updated = self.scenario_manager.bulk_apply_label(dfv, None)
            if updated > 0:
                st.success(f"Cleared labels from {updated} scenarios")
                self.scenario_manager.clear_cache()
                st.rerun()

    def _bulk_apply_labels(self, dfv, label_text: str, append_mode: bool) -> int:
        """Apply labels to multiple scenarios."""
        updated = 0
        new_labels = [s.strip() for s in label_text.split(",") if s.strip()]
        
        for _, row in dfv.iterrows():
            path = row["path"]
            if append_mode:
                current = (row.get("label") or "").strip()
                current_set = [s.strip() for s in current.split(",") if s.strip()] if current else []
                merged = current_set[:]
                for label in new_labels:
                    if label not in merged:
                        merged.append(label)
                final_label = ", ".join(merged) if merged else None
            else:
                final_label = label_text
            
            if self.scenario_manager.write_label(path, final_label):
                updated += 1
        
        return updated

    def _render_scenario_selection_controls(self, dfv, total_count):
        """Render scenario selection controls."""
        col1, col2, col3, col4 = st.columns([1, 1, 1, 2])
        
        with col1:
            if st.button("Select All Filtered"):
                for _, row in dfv.iterrows():
                    st.session_state[SK_SCN_SELECTIONS].add(row["path"])
                st.rerun()
        
        with col2:
            if st.button("Clear Selection"):
                st.session_state[SK_SCN_SELECTIONS].clear()
                st.rerun()
        
        with col3:
            if st.button("Invert Selection"):
                for _, row in dfv.iterrows():
                    path = row["path"]
                    selections = st.session_state[SK_SCN_SELECTIONS]
                    if path in selections:
                        selections.discard(path)
                    else:
                        selections.add(path)
                st.rerun()
        
        with col4:
            selected_count = len(st.session_state[SK_SCN_SELECTIONS])
            st.caption(f"Selected: {selected_count} | Filtered: {len(dfv)} | Total: {total_count}")

    def _render_scenarios_table(self, dfv):
        """Render simplified scenarios table for audio workflow."""
        st.markdown("---")
        st.markdown("### Scenarios")
        
        # Table headers
        col1, col2, col3, col4, col5 = st.columns([0.08, 0.15, 0.25, 0.35, 0.17])
        with col1: st.markdown("**Select**")
        with col2: st.markdown("**Scenario**")
        with col3: st.markdown("**Labels**")
        with col4: st.markdown("**Description & Files**")
        with col5: st.markdown("**Actions**")
        
        # Table rows
        for idx, row in dfv.iterrows():
            self._render_scenario_row(idx, row)

    def _render_scenario_row(self, idx, row):
        """Render individual scenario row."""
        scn_path = row["path"]
        number_str = row.get("number_str", "")
        label_val = row.get("label", "")
        description_val = row.get("description", "")
        sample_count = row.get("sample_count", 0)

        col1, col2, col3, col4, col5 = st.columns([0.08, 0.15, 0.25, 0.35, 0.17])

        # Selection checkbox
        with col1:
            key_sel = f"sel_{idx}_{hash(scn_path)}"
            is_selected = scn_path in st.session_state[SK_SCN_SELECTIONS]
            new_selection = st.checkbox("", value=is_selected, key=key_sel)
            if new_selection != is_selected:
                if new_selection:
                    st.session_state[SK_SCN_SELECTIONS].add(scn_path)
                else:
                    st.session_state[SK_SCN_SELECTIONS].discard(scn_path)

        # Scenario info
        with col2:
            st.write(f"**{number_str}**" if number_str else "—")
            st.caption(f"{sample_count} samples")

        # Labels (editable)
        with col3:
            key_lbl = f"lbl_{idx}_{hash(scn_path)}"
            new_label = st.text_input(
                "Labels", 
                value=label_val, 
                key=key_lbl, 
                label_visibility="collapsed",
                placeholder="Add labels..."
            )
            if new_label != label_val:
                save_label = new_label.strip() if new_label.strip() else None
                if self.scenario_manager.write_label(scn_path, save_label):
                    self.scenario_manager.clear_cache()
                    st.rerun()

        # Description and file count
        with col4:
            if description_val:
                display_desc = description_val if len(description_val) <= 40 else description_val[:37] + "..."
                st.write(display_desc)
            else:
                st.caption("No description")
            
            # Show file types available
            files_info = self._get_scenario_files_info(scn_path)
            if files_info:
                st.caption(files_info)

        # Actions
        with col5:
            if st.button("Explore", key=f"explore_{idx}_{hash(scn_path)}", help="Explore files"):
                st.session_state[SK_SCN_EXPLORE] = scn_path
                st.rerun()

    def _get_scenario_files_info(self, scenario_path: str) -> str:
        """Get brief info about files in scenario."""
        if not os.path.exists(scenario_path):
            return ""
        
        info_parts = []
        
        # Check for different file types
        raw_dir = os.path.join(scenario_path, "raw_recordings")
        impulse_dir = os.path.join(scenario_path, "impulse_responses") 
        room_dir = os.path.join(scenario_path, "room_responses")
        
        if os.path.exists(raw_dir):
            raw_count = len([f for f in os.listdir(raw_dir) if f.endswith('.wav')])
            if raw_count > 0:
                info_parts.append(f"Raw: {raw_count}")
        
        if os.path.exists(impulse_dir):
            impulse_count = len([f for f in os.listdir(impulse_dir) if f.endswith('.wav')])
            if impulse_count > 0:
                info_parts.append(f"Impulse: {impulse_count}")
        
        if os.path.exists(room_dir):
            room_count = len([f for f in os.listdir(room_dir) if f.endswith('.wav')])
            if room_count > 0:
                info_parts.append(f"Room: {room_count}")
        
        return " | ".join(info_parts) if info_parts else "No audio files"

    def _render_scenario_summary(self, df, dfv):
        """Render scenario summary."""
        st.markdown("---")
        unique_labels = self.scenario_manager.get_unique_labels(df)
        selected_count = len(st.session_state[SK_SCN_SELECTIONS])
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown(f"""
            **Summary:**
            - **Total scenarios:** {len(df)}
            - **Filtered scenarios:** {len(dfv)}
            - **Selected scenarios:** {selected_count}
            - **Unique labels:** {len(unique_labels)}
            """)
            
            if unique_labels:
                labels_text = ', '.join(sorted(unique_labels)[:5])
                if len(unique_labels) > 5:
                    labels_text += f" (and {len(unique_labels) - 5} more)"
                st.caption(f"Labels: {labels_text}")
        
        with col2:
            if selected_count > 0:
                st.info(f"Target: {selected_count} scenarios selected")
                if st.button("Show Selected Details"):
                    self._show_selection_details()

    def _show_selection_details(self):
        """Show details about selected scenarios."""
        selected_paths = st.session_state[SK_SCN_SELECTIONS]
        if not selected_paths:
            return
        
        st.markdown("### Selected Scenarios Details")
        for path in sorted(selected_paths):
            if os.path.exists(path):
                scenario_name = os.path.basename(path)
                files_info = self._get_scenario_files_info(path)
                st.write(f"- **{scenario_name}:** {files_info}")

    def _render_scenario_explorer(self):
        """Render scenario file explorer."""
        exp_path = st.session_state.get(SK_SCN_EXPLORE)
        if not exp_path or not os.path.exists(exp_path):
            return

        st.markdown("---")
        st.subheader(f"Exploring: {os.path.basename(exp_path)}")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Audio files
            wavs = self.scenario_manager.list_wavs(exp_path) if self.scenario_manager else []
            if wavs:
                st.markdown(f"**Audio Files ({len(wavs)} found):**")
                
                # Show first few files
                preview_files = wavs[:5]
                selected_file = st.selectbox(
                    "Select file to play", 
                    preview_files,
                    format_func=lambda x: os.path.basename(x)
                )
                
                if selected_file:
                    st.audio(selected_file, format="audio/wav")
                    
                    # File info
                    file_size = os.path.getsize(selected_file) / (1024 * 1024)  # MB
                    st.caption(f"Size: {file_size:.2f} MB | Path: {selected_file}")
                
                if len(wavs) > 5:
                    st.caption(f"Showing first 5 of {len(wavs)} audio files")
            else:
                st.info("No audio files found in this scenario")
        
        with col2:
            # File structure
            st.markdown("**Folder Structure:**")
            subdirs = ["raw_recordings", "impulse_responses", "room_responses", "metadata"]
            for subdir in subdirs:
                full_path = os.path.join(exp_path, subdir)
                if os.path.exists(full_path):
                    file_count = len([f for f in os.listdir(full_path) if not f.startswith('.')])
                    st.write(f"OK {subdir} ({file_count} files)")
                else:
                    st.write(f"MISSING {subdir}")
            
            # Close explorer
            if st.button("Close Explorer"):
                st.session_state[SK_SCN_EXPLORE] = None
                st.rerun()


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