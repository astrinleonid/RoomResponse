#!/usr/bin/env python3
"""
Room Response GUI - Main Application

Streamlit-based GUI for the Room Response data collection and analysis pipeline.
Provides interfaces for scenario management, data collection, feature processing,
classification, prediction, and visualization.
"""

import os
import streamlit as st

# Import managers and panels
try:
    from ScenarioManager import ScenarioManager
except ImportError:
    ScenarioManager = None

try:
    from gui_collect_panel import CollectionPanel
except ImportError:
    CollectionPanel = None

try:
    from gui_process_panel import ProcessingPanel
except ImportError:
    ProcessingPanel = None

try:
    from gui_classify_panel import ClassificationPanel
except ImportError:
    ClassificationPanel = None

# ----------------------------
# Session Keys and Constants
# ----------------------------
SK_DATASET_ROOT = "dataset_root"
SK_DEFAULT_DATASET_ROOT = "room_response_dataset"
SK_SCN_SELECTIONS = "scenarios_selected_set"
SK_SCN_EXPLORE = "scenarios_explore_path"
SK_FILTER_TEXT = "filter_text"
SK_FILTER_COMPUTER = "filter_computer"
SK_FILTER_ROOM = "filter_room"
SK_SAVED_LABELS = "saved_labels_cache"

# Classification session keys
SK_CLASSIFICATION_ARTIFACTS = "classification_artifacts"
SK_LAST_MODEL_INFO = "last_model_info"


class RoomResponseGUI:
    """Main GUI application class."""
    
    def __init__(self):
        """Initialize the GUI application."""
        self.scenario_manager = None
        self.collection_panel = None
        self.processing_panel = None
        self.classification_panel = None
        
        # Initialize components
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize GUI components."""
        if ScenarioManager is not None:
            self.scenario_manager = ScenarioManager()
        
        if CollectionPanel is not None and self.scenario_manager is not None:
            self.collection_panel = CollectionPanel(self.scenario_manager)
        
        if ProcessingPanel is not None and self.scenario_manager is not None:
            self.processing_panel = ProcessingPanel(self.scenario_manager)
        
        if ClassificationPanel is not None and self.scenario_manager is not None:
            self.classification_panel = ClassificationPanel(self.scenario_manager)
    
    def run(self):
        """Run the main GUI application."""
        st.set_page_config(page_title="Room Response GUI", layout="wide")
        
        # Ensure dataset root is configured
        self._ensure_dataset_root_ui()
        
        # Sidebar navigation
        panel = self._render_sidebar_navigation()
        
        # Render selected panel
        self._render_panel(panel)
    
    def _ensure_dataset_root_ui(self) -> str:
        """Dataset root selection in sidebar with validation."""
        st.sidebar.markdown("### Dataset")
        default_root = st.session_state.get(SK_DATASET_ROOT, 
                                           os.path.join(os.getcwd(), SK_DEFAULT_DATASET_ROOT))
        
        root = st.sidebar.text_input(
            "Dataset root folder",
            value=default_root,
            help="Folder containing scenario subfolders (e.g., Computer-Scenario1-Room).",
        )
        
        # Validate and normalize path
        root = root.strip() if root else default_root
        if root:
            root = os.path.abspath(root)
            if self.scenario_manager is not None:
                is_valid, message = self.scenario_manager.validate_dataset_root(root)
                if is_valid:
                    st.sidebar.success(f"✓ {message}", icon="📁")
                else:
                    st.sidebar.error(f"⚠️ {message}", icon="📁")
        
        # Check if dataset root changed to clear cache
        if self.scenario_manager is not None:
            last_root = st.session_state.get(self.scenario_manager.SK_LAST_DATASET_ROOT)
            if last_root != root:
                st.session_state[self.scenario_manager.SK_LAST_DATASET_ROOT] = root
                # Clear relevant caches when dataset root changes
                self.scenario_manager.clear_cache()
                st.session_state[SK_SCN_SELECTIONS] = set()
                st.session_state[SK_SCN_EXPLORE] = None
        
        st.session_state[SK_DATASET_ROOT] = root
        return root
    
    def _render_sidebar_navigation(self) -> str:
        """Render sidebar navigation panel selection."""
        st.sidebar.markdown("### Panels")
        
        # Check if we have a trained model for enabling Predict/Visualize
        has_model = bool(st.session_state.get(SK_CLASSIFICATION_ARTIFACTS) and 
                        st.session_state.get(SK_LAST_MODEL_INFO))
        
        # Panel options with conditional enabling
        panel_options = ["Scenarios", "Collect", "Process", "Classify", "Predict", "Visualize"]
        
        # Create radio button
        selected_panel = st.sidebar.radio(
            "Select a panel",
            options=panel_options,
            index=0,
            help="Panels are mutually exclusive to avoid overlapping actions."
        )
        
        # Show status for disabled panels
        if selected_panel == "Predict" and not has_model:
            st.sidebar.warning("⚠️ Predict panel requires a trained model from Classify panel")
            selected_panel = "Classify"  # Redirect to classify
        
        if selected_panel == "Visualize" and not has_model:
            st.sidebar.warning("⚠️ Visualize panel requires a trained model from Classify panel")
            selected_panel = "Classify"  # Redirect to classify
        
        # Show model status in sidebar
        if has_model:
            model_info = st.session_state.get(SK_LAST_MODEL_INFO)
            st.sidebar.success("✅ Model available")
            st.sidebar.caption(f"Type: {model_info.get('model_type', 'Unknown').upper()}")
            st.sidebar.caption(f"Features: {model_info.get('feature_type', 'Unknown')}")
        else:
            st.sidebar.info("ℹ️ No model trained yet")
        
        return selected_panel
    
    def _render_panel(self, panel: str):
        """Render the selected panel."""
        if panel == "Scenarios":
            self._render_scenarios_panel()
        elif panel == "Collect":
            self._render_collect_panel()
        elif panel == "Process":
            self._render_process_panel()
        elif panel == "Classify":
            self._render_classify_panel()
        else:
            self._render_placeholder_panel(panel)
    
    def _render_scenarios_panel(self):
        """Render the scenarios management panel."""
        if self.scenario_manager is None:
            st.error("❌ ScenarioManager not available. Please ensure ScenarioManager.py is present.")
            return
        
        st.header("Scenarios")
        
        # Initialize session state
        st.session_state.setdefault(SK_SCN_SELECTIONS, set())
        st.session_state.setdefault(SK_SAVED_LABELS, {})
        
        root = st.session_state.get(SK_DATASET_ROOT, os.getcwd())
        if not os.path.isdir(root):
            st.error("❌ Please provide a valid dataset root directory to continue.", icon="📁")
            st.info("The dataset root should contain scenario folders with names like 'Computer-Scenario1-Room'.")
            return
        
        # Filters and controls
        self._render_scenario_filters()
        
        # Build and filter data
        with st.spinner("Loading scenarios..."):
            df = self.scenario_manager.build_scenarios_df(root)
        
        if df.empty:
            st.info("No scenarios found in the dataset root.", icon="📂")
            return
        
        # Apply filters
        dfv = self.scenario_manager.apply_filters(
            df,
            st.session_state.get(SK_FILTER_TEXT, ""),
            st.session_state.get(SK_FILTER_COMPUTER, ""),
            st.session_state.get(SK_FILTER_ROOM, "")
        )
        
        # Sort scenarios
        dfv = self.scenario_manager.sort_scenarios_df(dfv)
        
        # Render scenario interface
        self._render_scenario_bulk_operations(df, dfv)
        self._render_scenario_selection_controls(dfv, len(df))
        self._render_scenario_table(dfv)
        self._render_scenario_summary(df, dfv)
        self._render_scenario_explorer()
    
    def _render_scenario_filters(self):
        """Render scenario filtering controls."""
        col1, col2, col3, col4 = st.columns([2, 1, 1, 1])

        with col1:
            filter_text = st.text_input(
                "Scenario name filter (regex supported)",
                key=SK_FILTER_TEXT,  # stable key
                label_visibility="collapsed",
                placeholder=r"Filter by scenario number (regex): ^6\., 0.*  |  name:Studio"
            )

        with col2:
            filter_computer = st.text_input(
                "Computer filter", key=SK_FILTER_COMPUTER,
                label_visibility="collapsed", placeholder="Filter by computer"
            )

        with col3:
            filter_room = st.text_input(
                "Room filter", key=SK_FILTER_ROOM,
                label_visibility="collapsed", placeholder="Filter by room"
            )

        with col4:
            if st.button("Re-analyze", help="Refresh scenario list from filesystem"):
                self.scenario_manager.clear_cache()
                st.rerun()

    
    def _render_scenario_bulk_operations(self, df, dfv):
        """Render simple, reliable bulk label ops for all *filtered* scenarios."""
        st.markdown("**Bulk operations for filtered scenarios**")

        # Optional: show a hint of existing labels to guide users
        existing = sorted(self.scenario_manager.get_unique_labels(df))
        if existing:
            st.caption("Existing labels: " + ", ".join(existing[:12]) + ("…" if len(existing) > 12 else ""))

        # Form ensures atomic submit (prevents mid-rerun desync between widgets)
        with st.form("bulk_ops_form", clear_on_submit=False):
            c1, c2, c3 = st.columns([2, 1, 1])

            with c1:
                label_text = st.text_input(
                    "Label(s) to apply",
                    key="bulk_label_input",
                    placeholder="e.g., quiet,baseline  (comma-separated allowed)"
                )
                append_mode = st.checkbox(
                    "Append instead of replace",
                    value=False,
                    help="If checked, adds the label(s) to any existing labels; otherwise replaces them."
                )

            with c2:
                apply_btn = st.form_submit_button(
                    f"Apply to {len(dfv)}",
                    help="Apply the label(s) above to all *filtered* scenarios."
                )

            with c3:
                clear_btn = st.form_submit_button(
                    f"Clear labels ({len(dfv)})",
                    help="Remove all labels from all *filtered* scenarios."
                )

        # Handle submits outside the form block (values persist in session_state)
        if apply_btn:
            new_label = (label_text or "").strip()
            if not new_label:
                st.warning("Please enter label text to apply.")
                return

            # If your ScenarioManager.bulk_apply_label only accepts (dfv, label) and *replaces*,
            # you can emulate append behavior here by reading & writing merged labels.
            updated = 0
            try:
                # Prefer a native append-aware API if you have it:
                # updated = self.scenario_manager.bulk_apply_label(dfv, new_label, append=append_mode)

                # Fallback shim if your API only replaces:
                if append_mode:
                    # Merge with existing, dedupe, keep order stable-ish
                    for _, row in dfv.iterrows():
                        path = row["path"]
                        current = (row.get("label") or "").strip()
                        if current:
                            current_set = [s.strip() for s in current.split(",") if s.strip()]
                        else:
                            current_set = []

                        to_add = [s.strip() for s in new_label.split(",") if s.strip()]
                        merged = current_set[:]
                        for t in to_add:
                            if t not in merged:
                                merged.append(t)

                        merged_str = ", ".join(merged)
                        if self.scenario_manager.write_label(path, merged_str):
                            updated += 1
                else:
                    # Replace mode: one call that updates all filtered rows at once
                    updated = self.scenario_manager.bulk_apply_label(dfv, new_label)
            except TypeError:
                # If your bulk_apply_label doesn't accept append kwarg, this catches it gracefully.
                updated = self.scenario_manager.bulk_apply_label(dfv, new_label)

            if updated:
                st.success(f"✅ Applied label(s) to {updated} scenario(s).")
                self.scenario_manager.clear_cache()
                st.rerun()
            else:
                st.error("❌ Failed to apply label(s).")

        if clear_btn:
            updated = self.scenario_manager.bulk_apply_label(dfv, None)
            if updated:
                st.success(f"✅ Cleared labels for {updated} scenario(s).")
                self.scenario_manager.clear_cache()
                st.rerun()
            else:
                st.error("❌ Failed to clear labels.")

    
    def _render_scenario_selection_controls(self, dfv, total_count):
        """Render scenario selection controls."""
        col1, col2, col3, col4 = st.columns([1, 1, 1, 2])
        
        with col1:
            if st.button("Select All Filtered", help="Select all currently filtered scenarios"):
                for _, row in dfv.iterrows():
                    st.session_state[SK_SCN_SELECTIONS].add(row["path"])
                st.rerun()
        
        with col2:
            if st.button("Clear Selection", help="Clear all selected scenarios"):
                st.session_state[SK_SCN_SELECTIONS].clear()
                st.rerun()
        
        with col3:
            if st.button("Invert Selection", help="Invert selection for filtered scenarios"):
                for _, row in dfv.iterrows():
                    path = row["path"]
                    if path in st.session_state[SK_SCN_SELECTIONS]:
                        st.session_state[SK_SCN_SELECTIONS].remove(path)
                    else:
                        st.session_state[SK_SCN_SELECTIONS].add(path)
                st.rerun()
        
        with col4:
            selected_count = len(st.session_state[SK_SCN_SELECTIONS])
            filtered_count = len(dfv)
            st.caption(f"Selected: {selected_count} | Filtered: {filtered_count} | Total: {total_count}")
    
    def _render_scenario_table(self, dfv):
        """Render the main scenarios table."""
        st.markdown("---")
        
        # Table header
        col1, col2, col3, col4, col5, col6 = st.columns([0.08, 0.15, 0.25, 0.12, 0.25, 0.15])
        with col1: st.markdown("**☑**")
        with col2: st.markdown("**Scenario #**")
        with col3: st.markdown("**Group Label**")
        with col4: st.markdown("**Features**")
        with col5: st.markdown("**Description**")
        with col6: st.markdown("**Actions**")
        
        # Scenario rows
        for idx, row in dfv.iterrows():
            self._render_scenario_row(idx, row)
    
    def _render_scenario_row(self, idx, row):
        """Render a single scenario row."""
        scn_path = row["path"]
        scn_name = row["scenario"]
        number_str = row.get("number_str") or ""
        label_val = row.get("label") or ""
        description_val = row.get("description") or ""
        sample_count = row.get("sample_count", 0)
        features_avail = row.get("features_available", {})
        
        col1, col2, col3, col4, col5, col6 = st.columns([0.08, 0.15, 0.25, 0.12, 0.25, 0.15])
        
        # Selection checkbox
        with col1:
            key_sel = f"sel_{idx}_{scn_path}"
            is_selected = scn_path in st.session_state[SK_SCN_SELECTIONS]
            new_selection = st.checkbox(
                "Select",
                value=is_selected, 
                key=key_sel, 
                label_visibility="collapsed",
                help=f"Select {scn_name}"
            )
            
            if new_selection != is_selected:
                if new_selection:
                    st.session_state[SK_SCN_SELECTIONS].add(scn_path)
                else:
                    st.session_state[SK_SCN_SELECTIONS].discard(scn_path)
        
        # Scenario number
        with col2:
            st.write(f"**{number_str}**" if number_str else "—")
            st.caption(f"{sample_count} samples")
        
        # Group label editor
        with col3:
            key_lbl = f"lbl_{idx}_{scn_path}"
            new_label = st.text_input(
                "Group label",
                value=label_val, 
                key=key_lbl, 
                label_visibility="collapsed",
                placeholder="Enter group label(s)...",
                help="Comma-separated for multiple labels"
            )
            
            # Auto-save label changes with proper cache management
            if new_label != label_val:
                label_to_save = new_label.strip() if new_label.strip() else None
                if self.scenario_manager.write_label(scn_path, label_to_save):
                    # Clear cache to force refresh
                    self.scenario_manager.clear_cache()
                    # Store success state but don't show persistent message
                    st.session_state[SK_SAVED_LABELS][scn_path] = new_label.strip()
                    # Trigger rerun to refresh the UI with new data
                    st.rerun()
        
        # Features status
        with col4:
            features_str = self.scenario_manager.format_features_status(features_avail)
            st.write(features_str)
            if features_str != "—":
                st.caption("S=Spectrum, M=MFCC, A=Audio")
        
        # Description
        with col5:
            if description_val:
                display_desc = description_val if len(description_val) <= 50 else description_val[:47] + "..."
                st.write(display_desc)
                if len(description_val) > 50:
                    st.caption(f"Full: {description_val}")
            else:
                st.caption("No description")
        
        # Actions
        with col6:
            action_col1, action_col2 = st.columns([1, 1])
            
            with action_col1:
                if st.button("⚙️", key=f"menu_{idx}_{scn_path}", help="Quick actions"):
                    action = st.radio(
                        f"Actions for {number_str}",
                        ["Add to group", "Select", "Deselect"],
                        key=f"action_{idx}_{scn_path}",
                        horizontal=True
                    )
                    
                    if action == "Select":
                        st.session_state[SK_SCN_SELECTIONS].add(scn_path)
                    elif action == "Deselect":
                        st.session_state[SK_SCN_SELECTIONS].discard(scn_path)
            
            with action_col2:
                if st.button("🔍", key=f"exp_{idx}_{scn_path}", help="Explore scenario"):
                    st.session_state[SK_SCN_EXPLORE] = scn_path
                    st.rerun()
    
    def _render_scenario_summary(self, df, dfv):
        """Render scenario summary information."""
        st.markdown("---")
        unique_labels = self.scenario_manager.get_unique_labels(df)
        
        summary_info = f"""
        **Summary:**
        - Total scenarios: {len(df)}
        - Filtered scenarios: {len(dfv)}
        - Selected scenarios: {len(st.session_state[SK_SCN_SELECTIONS])}
        - Unique labels: {len(unique_labels)} ({', '.join(sorted(unique_labels)) if unique_labels else 'none'})
        """
        st.markdown(summary_info)
    
    def _render_scenario_explorer(self):
        """Render inline scenario explorer."""
        exp_path = st.session_state.get(SK_SCN_EXPLORE)
        if exp_path and os.path.exists(exp_path):
            st.markdown("---")
            st.subheader(f"📁 Exploring: {os.path.basename(exp_path)}")
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                wavs = self.scenario_manager.list_wavs(exp_path)
                if wavs:
                    st.markdown(f"**Audio files ({len(wavs)} found):**")
                    preview_files = wavs[:5] if len(wavs) > 5 else wavs
                    selected_wav = st.selectbox(
                        "Select audio file to preview", 
                        preview_files, 
                        format_func=lambda x: os.path.basename(x),
                        label_visibility="collapsed"
                    )
                    if selected_wav:
                        st.audio(selected_wav, format="audio/wav")
                        
                    if len(wavs) > 5:
                        st.caption(f"Showing first 5 of {len(wavs)} audio files")
                else:
                    st.info("No audio files found")
            
            with col2:
                features_info = self.scenario_manager.check_features_available(exp_path)
                st.markdown("**Features:**")
                for feature_type, available in features_info.items():
                    status = "✅" if available else "❌"
                    st.write(f"{status} {feature_type.upper()}")
                
                feat_path = os.path.join(exp_path, "features.csv")
                if os.path.exists(feat_path):
                    try:
                        import pandas as pd
                        feat_df = pd.read_csv(feat_path)
                        with st.expander("MFCC Features Preview"):
                            st.dataframe(feat_df.head(10), use_container_width=True)
                            st.caption(f"Shape: {feat_df.shape}")
                    except Exception as e:
                        st.error(f"Error reading features.csv: {e}")
                
                if st.button("❌ Close Explorer"):
                    st.session_state[SK_SCN_EXPLORE] = None
                    st.rerun()
    
    def _render_collect_panel(self):
        """Render the collection panel."""
        if self.collection_panel is not None:
            self.collection_panel.render()
        else:
            st.error("❌ Collection panel not available. Please ensure gui_collect_panel.py is present.")
    
    def _render_process_panel(self):
        """Render the processing panel."""
        if self.processing_panel is not None:
            self.processing_panel.render()
        else:
            st.error("❌ Processing panel not available. Please ensure gui_process_panel.py is present.")
    
    def _render_classify_panel(self):
        """Render the classification panel."""
        if self.classification_panel is not None:
            self.classification_panel.render()
        else:
            st.error("❌ Classification panel not available. Please ensure gui_classify_panel.py is present.")
            st.info("The gui_classify_panel.py file should be in the same directory as this GUI application.")
    
    def _render_placeholder_panel(self, name: str):
        """Render placeholder for unimplemented panels."""
        st.header(name)
        st.info(f"The {name} panel will be implemented in the next development phase.", icon="🚧")


def main():
    """Main application entry point."""
    app = RoomResponseGUI()
    app.run()


if __name__ == "__main__":
    main()