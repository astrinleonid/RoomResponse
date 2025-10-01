#!/usr/bin/env python3
"""
Scenarios Panel - Dataset browsing, filtering, and management

Extracted from piano_response.py for consistency with other GUI panels.
Provides:
- Scenario listing with filtering
- Label management (individual and bulk)
- Selection controls
- File explorer for scenario inspection with audio visualization
"""

import os
import numpy as np
import streamlit as st
from typing import Optional

# Optional: Audio visualizer
try:
    from gui_audio_visualizer import AudioVisualizer
    VISUALIZER_AVAILABLE = True
except ImportError:
    AudioVisualizer = None
    VISUALIZER_AVAILABLE = False

# Optional: Signal alignment
try:
    from signal_alignment import SignalAligner, align_impulse_responses, average_signals
    ALIGNMENT_AVAILABLE = True
except ImportError:
    SignalAligner = None
    align_impulse_responses = None
    average_signals = None
    ALIGNMENT_AVAILABLE = False

# Session keys
SK_SCN_SELECTIONS = "scenarios_selected_set"
SK_SCN_EXPLORE = "scenarios_explore_path"
SK_FILTER_TEXT = "filter_text"
SK_FILTER_COMPUTER = "filter_computer"
SK_FILTER_ROOM = "filter_room"


class ScenariosPanel:
    """Panel for browsing and managing scenario datasets."""

    def __init__(self, scenario_manager):
        """
        Initialize the scenarios panel.

        Args:
            scenario_manager: ScenarioManager instance for data operations
        """
        self.scenario_manager = scenario_manager

    def render(self) -> None:
        """Main render method for the Scenarios panel."""
        if self.scenario_manager is None:
            st.error("ScenarioManager not available.")
            return

        st.header("Scenarios Overview")
        root = st.session_state.get("dataset_root", os.getcwd())

        if not os.path.isdir(root):
            st.error("Please provide a valid dataset root directory.")
            return

        # Initialize session state
        self._init_session_state()

        # Filters
        self._render_scenario_filters()

        # Load scenarios
        with st.spinner("Loading scenarios..."):
            df = self.scenario_manager.build_scenarios_df(root)

        if df.empty:
            self._render_empty_state()
            return

        # Apply filters
        dfv = self.scenario_manager.apply_filters(
            df,
            st.session_state.get(SK_FILTER_TEXT, ""),
            st.session_state.get(SK_FILTER_COMPUTER, ""),
            st.session_state.get(SK_FILTER_ROOM, ""),
        )
        dfv = self.scenario_manager.sort_scenarios_df(dfv)

        # Bulk operations
        self._render_scenario_bulk_operations(df, dfv)

        # Selection controls
        self._render_scenario_selection_controls(dfv, len(df))

        # Scenarios table
        self._render_scenarios_table(dfv)

        # Summary and explorer
        self._render_scenario_summary(df, dfv)
        self._render_scenario_explorer()

    def _init_session_state(self) -> None:
        """Initialize session state defaults."""
        st.session_state.setdefault(SK_SCN_SELECTIONS, set())
        st.session_state.setdefault(SK_FILTER_TEXT, "")
        st.session_state.setdefault(SK_FILTER_COMPUTER, "")
        st.session_state.setdefault(SK_FILTER_ROOM, "")

    def _render_empty_state(self) -> None:
        """Render UI when no scenarios are found."""
        st.info("No scenarios found. Use the Collect panel to create recordings.")
        st.markdown("**Quick Start:**")
        st.markdown("1. Go to **Collect** panel")
        st.markdown("2. Configure your audio settings")
        st.markdown("3. Start recording scenarios")

    def _render_scenario_filters(self) -> None:
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

    def _render_scenario_bulk_operations(self, df, dfv) -> None:
        """Render bulk label management controls (collapsible)."""
        with st.expander("Bulk Label Management", expanded=False):
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

    def _render_scenario_selection_controls(self, dfv, total_count) -> None:
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

    def _render_scenarios_table(self, dfv) -> None:
        """Render scenarios table."""
        st.markdown("---")
        st.markdown("### Scenarios")

        # Table headers
        col1, col2, col3, col4, col5, col6 = st.columns([0.06, 0.1, 0.15, 0.15, 0.35, 0.14])
        with col1:
            st.markdown("**Select**")
        with col2:
            st.markdown("**#**")
        with col3:
            st.markdown("**Computer**")
        with col4:
            st.markdown("**Room**")
        with col5:
            st.markdown("**Labels & Files**")
        with col6:
            st.markdown("**Actions**")

        # Table rows
        for idx, row in dfv.iterrows():
            self._render_scenario_row(idx, row)

    def _render_scenario_row(self, idx, row) -> None:
        """Render individual scenario row."""
        scn_path = row["path"]
        number_str = row.get("number_str", "")
        computer = row.get("computer", "")
        room = row.get("room", "")
        label_val = row.get("label", "")
        description_val = row.get("description", "")
        sample_count = row.get("sample_count", 0)

        col1, col2, col3, col4, col5, col6 = st.columns([0.06, 0.1, 0.15, 0.15, 0.35, 0.14])

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

        # Scenario number
        with col2:
            st.write(f"**{number_str}**" if number_str else "—")
            st.caption(f"{sample_count} files")

        # Computer name
        with col3:
            st.write(computer if computer else "—")

        # Room name
        with col4:
            st.write(room if room else "—")

        # Labels and file info
        with col5:
            # Labels (editable)
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

            # File types available
            files_info = self._get_scenario_files_info(scn_path)
            if files_info:
                st.caption(files_info)

        # Actions
        with col6:
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

    def _render_scenario_summary(self, df, dfv) -> None:
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

    def _show_selection_details(self) -> None:
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

    def _render_scenario_explorer(self) -> None:
        """Render scenario file explorer with audio visualization."""
        exp_path = st.session_state.get(SK_SCN_EXPLORE)
        if not exp_path or not os.path.exists(exp_path):
            return

        st.markdown("---")
        st.subheader(f"Exploring: {os.path.basename(exp_path)}")

        # Close button at the top
        if st.button("✕ Close Explorer", key="close_explorer_top"):
            st.session_state[SK_SCN_EXPLORE] = None
            st.rerun()

        # Get audio files grouped by type
        audio_files = self._get_audio_files_by_type(exp_path)

        if not any(audio_files.values()):
            st.info("No audio files found in this scenario")
            self._render_folder_structure(exp_path)
            return

        # Audio type selection
        available_types = [k for k, v in audio_files.items() if v]
        if not available_types:
            st.info("No audio files found")
            return

        audio_type = st.selectbox(
            "Audio type",
            available_types,
            format_func=lambda x: f"{x.replace('_', ' ').title()} ({len(audio_files[x])} files)"
        )

        files_of_type = audio_files[audio_type]

        # View mode tabs: Single File or Overlay
        if len(files_of_type) > 1:
            view_mode = st.radio(
                "View mode",
                ["Single File", "Overlay All"],
                horizontal=True,
                help="Single File: View one file at a time. Overlay All: Compare all files overlaid."
            )
        else:
            view_mode = "Single File"

        # Render based on view mode
        if view_mode == "Overlay All":
            self._render_overlay_view(files_of_type, audio_type, exp_path)
        else:
            # Single file view
            col1, col2 = st.columns([3, 1])

            with col1:
                # File selection within type
                if len(files_of_type) > 10:
                    # Show dropdown for many files
                    selected_file = st.selectbox(
                        "Select file",
                        files_of_type,
                        format_func=lambda x: os.path.basename(x)
                    )
                else:
                    # Show radio buttons for few files
                    selected_file = st.radio(
                        "Select file",
                        files_of_type,
                        format_func=lambda x: os.path.basename(x)
                    )

            with col2:
                # File structure sidebar
                self._render_folder_structure(exp_path)

            # Visualize selected file
            if selected_file and os.path.exists(selected_file):
                self._render_audio_file(selected_file)

    def _get_audio_files_by_type(self, scenario_path: str) -> dict:
        """Get audio files grouped by type."""
        files_by_type = {
            "impulse_responses": [],
            "impulse_responses_aligned": [],
            "room_responses": [],
            "raw_recordings": [],
        }

        for file_type in files_by_type.keys():
            type_dir = os.path.join(scenario_path, file_type)
            if os.path.exists(type_dir):
                try:
                    wav_files = [
                        os.path.join(type_dir, f)
                        for f in os.listdir(type_dir)
                        if f.lower().endswith('.wav')
                    ]
                    files_by_type[file_type] = sorted(wav_files)
                except (OSError, PermissionError):
                    pass

        return files_by_type

    def _render_folder_structure(self, exp_path: str) -> None:
        """Render folder structure info."""
        st.markdown("**Folder Structure:**")
        subdirs = ["raw_recordings", "impulse_responses", "room_responses", "metadata"]
        for subdir in subdirs:
            full_path = os.path.join(exp_path, subdir)
            if os.path.exists(full_path):
                try:
                    file_count = len([f for f in os.listdir(full_path) if not f.startswith('.')])
                    st.write(f"✓ {subdir} ({file_count})")
                except (OSError, PermissionError):
                    st.write(f"✓ {subdir} (error)")
            else:
                st.write(f"✗ {subdir}")

    def _render_overlay_view(self, file_paths: list, audio_type: str, scenario_path: str) -> None:
        """Render overlay view of multiple audio files."""
        st.markdown("---")
        st.markdown(f"### Overlay View: {audio_type.replace('_', ' ').title()}")

        # Debug info
        st.caption(f"🔍 Debug: Viewing {len(file_paths)} files of type '{audio_type}' | Alignment available: {ALIGNMENT_AVAILABLE}")

        if not VISUALIZER_AVAILABLE:
            st.warning("AudioVisualizer not available - overlay view requires gui_audio_visualizer.py")
            return

        # Overlay controls
        col1, col2 = st.columns([3, 1])

        with col1:
            max_files = st.slider(
                "Number of files to overlay",
                min_value=2,
                max_value=min(len(file_paths), 50),
                value=min(len(file_paths), 10),
                help="Limit the number of overlaid signals for better visualization"
            )

        with col2:
            normalize = st.checkbox(
                "Normalize each signal",
                value=True,
                help="Normalize each signal to max amplitude of 1"
            )
            self._render_folder_structure(scenario_path)

        # Load audio files
        with st.spinner(f"Loading {max_files} audio files..."):
            audio_signals = []
            labels = []
            sample_rate = None

            for i, file_path in enumerate(file_paths[:max_files]):
                try:
                    audio_data, sr = AudioVisualizer.load_wav_file(file_path)
                    if audio_data is not None and len(audio_data) > 0:
                        audio_signals.append(audio_data)
                        # Extract file index from filename (e.g., "impulse_..._42_..." -> "42")
                        basename = os.path.basename(file_path)
                        # Try to find index number in filename
                        import re
                        match = re.search(r'_(\d+)_', basename)
                        if match:
                            labels.append(f"#{match.group(1)}")
                        else:
                            labels.append(f"File {i+1}")

                        if sample_rate is None:
                            sample_rate = sr
                        elif sample_rate != sr:
                            st.warning(f"Sample rate mismatch: {basename} has {sr} Hz (expected {sample_rate} Hz)")
                except Exception as e:
                    st.warning(f"Failed to load {os.path.basename(file_path)}: {e}")

        # Show loading results
        if not audio_signals:
            st.error("Failed to load any audio files")
            return

        st.info(f"✓ Loaded {len(audio_signals)} of {len(file_paths)} files")

        # Render overlay plot using AudioVisualizer
        try:
            fig = AudioVisualizer.render_overlay_plot(
                audio_signals=audio_signals,
                sample_rate=sample_rate,
                title=f"{audio_type.replace('_', ' ').title()} - Overlay Comparison",
                labels=labels,
                normalize=normalize,
                show_legend=(len(audio_signals) <= 15),
                figsize=(12, 5),
                alpha=0.5,
                linewidth=0.8
            )
            st.pyplot(fig, use_container_width=True)

            # Statistics
            st.markdown("---")
            st.markdown("**Overlay Statistics:**")

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                durations = [len(sig) / sample_rate for sig in audio_signals]
                st.metric("Avg Duration", f"{np.mean(durations):.3f}s")

            with col2:
                lengths = [len(sig) for sig in audio_signals]
                st.metric("Avg Length", f"{int(np.mean(lengths))} samples")

            with col3:
                if not normalize:
                    max_amps = [np.max(np.abs(sig)) for sig in audio_signals]
                    st.metric("Avg Max Amp", f"{np.mean(max_amps):.3f}")
                else:
                    st.metric("Normalization", "Enabled")

            with col4:
                rms_vals = [np.sqrt(np.mean(sig**2)) for sig in audio_signals]
                st.metric("Avg RMS", f"{np.mean(rms_vals):.4f}")

            # Show variability info
            with st.expander("Variability Analysis"):
                st.caption("Analyze consistency across signals")

                # Length variability
                if len(set(lengths)) > 1:
                    st.warning(f"⚠️ Length varies: {min(lengths)} to {max(lengths)} samples")
                else:
                    st.success(f"✓ All signals same length: {lengths[0]} samples")

                # Amplitude variability (if not normalized)
                if not normalize:
                    max_amps = [np.max(np.abs(sig)) for sig in audio_signals]
                    amp_std = np.std(max_amps)
                    amp_mean = np.mean(max_amps)
                    if amp_mean > 0:
                        amp_cv = (amp_std / amp_mean) * 100  # Coefficient of variation
                        st.write(f"**Amplitude variation:** {amp_cv:.1f}% (CV)")
                        if amp_cv > 20:
                            st.warning(f"⚠️ High amplitude variation detected")

                # RMS consistency
                rms_std = np.std(rms_vals)
                rms_mean = np.mean(rms_vals)
                if rms_mean > 0:
                    rms_cv = (rms_std / rms_mean) * 100
                    st.write(f"**RMS variation:** {rms_cv:.1f}% (CV)")

        except Exception as e:
            st.error(f"Error rendering overlay plot: {e}")
            import traceback
            st.code(traceback.format_exc())

        # Signal Alignment Tool (outside the try-except for visualization)
        # Debug: Show status
        if not ALIGNMENT_AVAILABLE:
            st.warning("⚠️ Signal alignment module not available. Install signal_alignment.py to enable this feature.")
        elif audio_type != "impulse_responses":
            st.info(f"ℹ️ Signal alignment is only available for impulse_responses (current type: {audio_type})")

        if ALIGNMENT_AVAILABLE and audio_type == "impulse_responses":
            st.markdown("---")
            with st.expander("🔧 Signal Alignment Tool", expanded=False):
                st.markdown("**Synchronize impulse responses using cross-correlation**")
                st.caption("Aligns all signals to a reference by finding optimal time shifts")

                col_a, col_b = st.columns([2, 1])

                with col_a:
                    ref_idx = st.number_input(
                        "Reference signal index (0-based)",
                        min_value=0,
                        max_value=len(file_paths) - 1,
                        value=0,
                        help="Index of the signal to use as reference (0 = first file)"
                    )

                with col_b:
                    threshold = st.slider(
                        "Noise threshold",
                        min_value=0.0,
                        max_value=1.0,
                        value=0.3,
                        step=0.05,
                        help="Amplitude threshold for removing noise (0.3 recommended)"
                    )

                if st.button("🔄 Align All Signals", type="primary", use_container_width=True):
                    self._run_signal_alignment(
                        file_paths,
                        scenario_path,
                        ref_idx,
                        threshold
                    )

        # Signal Averaging Tool
        if ALIGNMENT_AVAILABLE:
            st.markdown("---")
            with st.expander("📊 Signal Averaging Tool", expanded=False):
                st.markdown("**Average multiple signals into a single representative signal**")
                st.caption("Optionally align signals before averaging for better results")

                col_a, col_b = st.columns([2, 1])

                with col_a:
                    align_before_avg = st.checkbox(
                        "Align signals before averaging",
                        value=True,
                        help="Recommended: Align signals first for better averaging results"
                    )

                    if align_before_avg:
                        avg_ref_idx = st.number_input(
                            "Reference signal for alignment (0-based)",
                            min_value=0,
                            max_value=len(file_paths) - 1,
                            value=0,
                            key="avg_ref_idx",
                            help="Index of the signal to use as reference for alignment"
                        )

                        avg_threshold = st.slider(
                            "Alignment threshold",
                            min_value=0.0,
                            max_value=1.0,
                            value=0.3,
                            step=0.05,
                            key="avg_threshold",
                            help="Amplitude threshold for noise removal during alignment"
                        )
                    else:
                        avg_ref_idx = 0
                        avg_threshold = 0.3

                with col_b:
                    output_filename = st.text_input(
                        "Output filename",
                        value=f"averaged_{audio_type}.wav",
                        help="Name for the averaged signal file"
                    )

                if st.button("📈 Average All Signals", type="primary", use_container_width=True):
                    self._run_signal_averaging(
                        file_paths,
                        scenario_path,
                        audio_type,
                        output_filename,
                        align_before_avg,
                        avg_ref_idx,
                        avg_threshold
                    )

    def _render_audio_file(self, file_path: str) -> None:
        """Render audio file with visualization using AudioVisualizer API."""
        st.markdown("---")

        # File info
        file_size = os.path.getsize(file_path) / (1024 * 1024)
        st.caption(f"📁 {os.path.basename(file_path)} ({file_size:.2f} MB)")

        # Use AudioVisualizer if available
        if VISUALIZER_AVAILABLE:
            try:
                # Load audio data using AudioVisualizer utility
                audio_data, sample_rate = AudioVisualizer.load_wav_file(file_path)

                if audio_data is not None:
                    # Create unique ID for this visualizer
                    viz_id = f"explorer_{hash(file_path)}"
                    visualizer = AudioVisualizer(viz_id)

                    # Render with full features
                    visualizer.render(
                        audio_data=audio_data,
                        sample_rate=sample_rate,
                        title=f"Waveform: {os.path.basename(file_path)}",
                        show_controls=True,
                        show_analysis=True,
                        height=350
                    )
                else:
                    st.error("Failed to load audio file")
                    self._render_basic_audio_player(file_path)
            except Exception as e:
                st.warning(f"Visualizer error: {e}")
                self._render_basic_audio_player(file_path)
        else:
            # Fallback to basic audio player
            st.info("AudioVisualizer not available - using basic player")
            self._render_basic_audio_player(file_path)

    def _render_basic_audio_player(self, file_path: str) -> None:
        """Render basic audio player without visualization."""
        st.audio(file_path, format="audio/wav")

    def _run_signal_alignment(
        self,
        file_paths: list,
        scenario_path: str,
        reference_index: int,
        threshold: float
    ) -> None:
        """
        Run signal alignment and save results.

        Args:
            file_paths: List of file paths to align
            scenario_path: Path to scenario directory
            reference_index: Index of reference signal
            threshold: Noise threshold value
        """
        if not ALIGNMENT_AVAILABLE:
            st.error("Signal alignment module not available")
            return

        # Create output directory
        output_dir = os.path.join(scenario_path, "impulse_responses_aligned")

        try:
            # Progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()

            def progress_callback(current, total, message):
                progress = current / total if total > 0 else 0
                progress_bar.progress(progress)
                status_text.text(f"{message} ({current}/{total})")

            # Run alignment
            with st.spinner("Aligning signals..."):
                report = align_impulse_responses(
                    file_paths=file_paths,
                    output_dir=output_dir,
                    reference_index=reference_index,
                    threshold=threshold,
                    progress_callback=progress_callback
                )

            # Clear progress indicators
            progress_bar.empty()
            status_text.empty()

            # Show success message
            st.success(f"✅ Successfully aligned {report['num_signals']} signals!")
            st.info(f"📁 Saved to: `{output_dir}`")

            # Show alignment statistics
            st.markdown("### Alignment Report")

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric(
                    "Signals Aligned",
                    report['num_signals']
                )

            with col2:
                st.metric(
                    "Sample Rate",
                    f"{report['sample_rate']} Hz"
                )

            with col3:
                st.metric(
                    "Max Shift",
                    f"{report['shifts_ms']['max']:.1f} ms"
                )

            with col4:
                st.metric(
                    "Avg Correlation",
                    f"{report['correlations']['mean']:.3f}"
                )

            # Detailed statistics
            with st.expander("Detailed Statistics"):
                st.markdown("**Time Shifts (milliseconds):**")
                st.json({
                    "Min": f"{report['shifts_ms']['min']:.2f} ms",
                    "Max": f"{report['shifts_ms']['max']:.2f} ms",
                    "Mean": f"{report['shifts_ms']['mean']:.2f} ms",
                    "Std Dev": f"{report['shifts_ms']['std']:.2f} ms"
                })

                st.markdown("**Time Shifts (samples):**")
                st.json({
                    "Min": report['shifts_samples']['min'],
                    "Max": report['shifts_samples']['max'],
                    "Mean": f"{report['shifts_samples']['mean']:.1f}",
                    "Std Dev": f"{report['shifts_samples']['std']:.1f}"
                })

                st.markdown("**Cross-Correlation Quality:**")
                st.json({
                    "Min": f"{report['correlations']['min']:.4f}",
                    "Max": f"{report['correlations']['max']:.4f}",
                    "Mean": f"{report['correlations']['mean']:.4f}",
                    "Std Dev": f"{report['correlations']['std']:.4f}"
                })

            # Tips
            st.info(
                "💡 **Tip:** Low correlation values (<0.5) may indicate poor alignment. "
                "Try adjusting the threshold or selecting a different reference signal."
            )

            # Offer to refresh and view aligned files
            if st.button("📂 View Aligned Files"):
                st.session_state[SK_SCN_EXPLORE] = scenario_path
                st.rerun()

        except Exception as e:
            st.error(f"Alignment failed: {e}")
            import traceback
            st.code(traceback.format_exc())

    def _run_signal_averaging(
        self,
        file_paths: list,
        scenario_path: str,
        audio_type: str,
        output_filename: str,
        align_first: bool,
        reference_index: int,
        threshold: float
    ) -> None:
        """
        Run signal averaging and save result.

        Args:
            file_paths: List of file paths to average
            scenario_path: Path to scenario directory
            audio_type: Type of audio files (for output directory)
            output_filename: Name for output file
            align_first: Whether to align before averaging
            reference_index: Index of reference signal for alignment
            threshold: Noise threshold value
        """
        if not ALIGNMENT_AVAILABLE:
            st.error("Signal averaging module not available")
            return

        # Determine output directory and file path
        output_dir = os.path.join(scenario_path, f"{audio_type}_averaged")
        output_file = os.path.join(output_dir, output_filename)

        try:
            # Progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()

            def progress_callback(current, total, message):
                progress = current / total if total > 0 else 0
                progress_bar.progress(progress)
                status_text.text(f"{message} ({current}/{total})")

            # Run averaging
            with st.spinner("Averaging signals..."):
                report = average_signals(
                    file_paths=file_paths,
                    output_file=output_file,
                    align_first=align_first,
                    reference_index=reference_index,
                    threshold=threshold,
                    progress_callback=progress_callback
                )

            # Clear progress indicators
            progress_bar.empty()
            status_text.empty()

            # Show success message
            st.success(f"✅ Successfully averaged {report['num_signals']} signals!")
            st.info(f"📁 Saved to: `{output_file}`")

            # Show averaging statistics
            st.markdown("### Averaging Report")

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric(
                    "Signals Averaged",
                    report['num_signals']
                )

            with col2:
                st.metric(
                    "Sample Rate",
                    f"{report['sample_rate']} Hz"
                )

            with col3:
                st.metric(
                    "Duration",
                    f"{report['averaged_duration_s']:.3f} s"
                )

            with col4:
                st.metric(
                    "Peak Amplitude",
                    f"{report['averaged_peak']:.3f}"
                )

            # Additional metrics
            col5, col6, col7 = st.columns(3)

            with col5:
                st.metric(
                    "RMS Level",
                    f"{report['averaged_rms']:.4f}"
                )

            with col6:
                st.metric(
                    "Aligned First",
                    "Yes" if report['aligned'] else "No"
                )

            with col7:
                st.metric(
                    "Length (samples)",
                    f"{report['averaged_length']:,}"
                )

            # Detailed statistics
            with st.expander("Individual Signal Statistics"):
                st.markdown("**Length Statistics:**")
                st.json({
                    "Min": f"{report['individual_stats']['lengths']['min']:,} samples",
                    "Max": f"{report['individual_stats']['lengths']['max']:,} samples",
                    "Mean": f"{report['individual_stats']['lengths']['mean']:.1f} samples",
                })

                st.markdown("**RMS Statistics:**")
                st.json({
                    "Min": f"{report['individual_stats']['rms']['min']:.4f}",
                    "Max": f"{report['individual_stats']['rms']['max']:.4f}",
                    "Mean": f"{report['individual_stats']['rms']['mean']:.4f}",
                })

                st.markdown("**Peak Amplitude Statistics:**")
                st.json({
                    "Min": f"{report['individual_stats']['peak']['min']:.4f}",
                    "Max": f"{report['individual_stats']['peak']['max']:.4f}",
                    "Mean": f"{report['individual_stats']['peak']['mean']:.4f}",
                })

                # Show alignment report if aligned
                if report['aligned'] and 'alignment' in report:
                    st.markdown("---")
                    st.markdown("**Alignment Details:**")
                    alignment = report['alignment']
                    st.json({
                        "Max Shift": f"{alignment['shifts_ms']['max']:.2f} ms",
                        "Mean Shift": f"{alignment['shifts_ms']['mean']:.2f} ms",
                        "Avg Correlation": f"{alignment['correlations']['mean']:.3f}",
                    })

            # Visualization (automatically shown)
            st.markdown("---")
            st.markdown("### Averaged Signal Visualization")

            if VISUALIZER_AVAILABLE:
                try:
                    audio_data, sample_rate_loaded = AudioVisualizer.load_wav_file(output_file)
                    if audio_data is not None:
                        viz = AudioVisualizer("averaged_signal")
                        viz.render(
                            audio_data=audio_data,
                            sample_rate=sample_rate_loaded,
                            title="Averaged Signal",
                            show_controls=True,
                            show_analysis=True,
                            height=400
                        )
                    else:
                        st.error("Failed to load averaged signal for visualization")
                except Exception as e:
                    st.error(f"Visualization error: {e}")
                    import traceback
                    st.code(traceback.format_exc())
            else:
                st.warning("AudioVisualizer not available - averaged signal saved but cannot be visualized")
                st.info(f"You can find the file at: {output_file}")

            # Offer to refresh and view in explorer
            st.markdown("---")
            if st.button("📂 Refresh Explorer"):
                st.session_state[SK_SCN_EXPLORE] = scenario_path
                st.rerun()

        except Exception as e:
            st.error(f"Averaging failed: {e}")
            import traceback
            st.code(traceback.format_exc())
