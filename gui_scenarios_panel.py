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

# Optional: FIR filter file I/O
try:
    from FirFilterFileIO import save_fir_filters, load_fir_filters, get_fir_file_info
    FIR_FILTER_IO_AVAILABLE = True
except ImportError:
    save_fir_filters = None
    load_fir_filters = None
    get_fir_file_info = None
    FIR_FILTER_IO_AVAILABLE = False

# Optional: Multi-channel filename utilities
try:
    from multichannel_filename_utils import group_files_by_channel, parse_multichannel_filename
    MULTICHANNEL_UTILS_AVAILABLE = True
except ImportError:
    group_files_by_channel = None
    parse_multichannel_filename = None
    MULTICHANNEL_UTILS_AVAILABLE = False

# Optional: Recorder for channel names
try:
    from RoomResponseRecorder import RoomResponseRecorder
    RECORDER_AVAILABLE = True
except ImportError:
    RoomResponseRecorder = None
    RECORDER_AVAILABLE = False

# Session keys
SK_SCN_SELECTIONS = "scenarios_selected_set"
SK_SCN_EXPLORE = "scenarios_explore_path"
SK_SCN_FIR_EXPORT = "scenarios_fir_export_path"
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
        self._render_fir_export_dialog()

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
        col1, col2, col3, col4, col5, col6 = st.columns([0.06, 0.1, 0.15, 0.15, 0.33, 0.16])
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

        col1, col2, col3, col4, col5, col6 = st.columns([0.06, 0.1, 0.15, 0.15, 0.33, 0.16])

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
            col_a, col_b = st.columns(2)
            with col_a:
                if st.button("Explore", key=f"explore_{idx}_{hash(scn_path)}", help="Explore files", use_container_width=True):
                    st.session_state[SK_SCN_EXPLORE] = scn_path
                    st.rerun()
            with col_b:
                if FIR_FILTER_IO_AVAILABLE:
                    if st.button("Export FIR", key=f"fir_{idx}_{hash(scn_path)}", help="Export as FIR filter", use_container_width=True):
                        st.session_state[SK_SCN_FIR_EXPORT] = scn_path
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
            raw_count = len([f for f in os.listdir(raw_dir) if f.endswith(('.wav', '.npy'))])
            if raw_count > 0:
                info_parts.append(f"Raw: {raw_count}")

        if os.path.exists(impulse_dir):
            impulse_count = len([f for f in os.listdir(impulse_dir) if f.endswith(('.wav', '.npy'))])
            if impulse_count > 0:
                info_parts.append(f"Impulse: {impulse_count}")

        if os.path.exists(room_dir):
            room_count = len([f for f in os.listdir(room_dir) if f.endswith(('.wav', '.npy'))])
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

        # Check if multi-channel
        is_multichannel = self.scenario_manager.is_multichannel_scenario(exp_path)

        # Enhanced debug info
        num_channels = self.scenario_manager.detect_num_channels_in_scenario(exp_path)
        st.info(f"Debug: is_multichannel={is_multichannel} | num_channels={num_channels} | Utils available: {MULTICHANNEL_UTILS_AVAILABLE} | Files: {len(files_of_type)}")

        # View mode tabs: Single File, Overlay All, or Channel Exploration
        if len(files_of_type) > 1:
            if is_multichannel and MULTICHANNEL_UTILS_AVAILABLE:
                view_mode = st.radio(
                    "View mode",
                    ["Single File", "Overlay All", "By Channel", "By Measurement"],
                    horizontal=True,
                    help="Single File: View one file at a time. Overlay All: Compare all files. By Channel: Overlay measurements per channel. By Measurement: Compare channels per measurement."
                )
            else:
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
        elif view_mode == "By Channel":
            self._render_by_channel_view(exp_path, os.path.basename(exp_path))
        elif view_mode == "By Measurement":
            self._render_by_measurement_view(exp_path, os.path.basename(exp_path))
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
        """Get audio files grouped by type (supports both WAV and NumPy formats)."""
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
                    # Find both WAV and NumPy files
                    audio_files = [
                        os.path.join(type_dir, f)
                        for f in os.listdir(type_dir)
                        if f.lower().endswith(('.wav', '.npy'))
                    ]

                    # Remove duplicates: if both .wav and .npy exist for same base name,
                    # keep only one entry (the loader will prefer .npy automatically)
                    unique_files = {}
                    for filepath in audio_files:
                        base_name = filepath.rsplit('.', 1)[0]  # Remove extension
                        if base_name not in unique_files:
                            unique_files[base_name] = filepath

                    files_by_type[file_type] = sorted(unique_files.values())
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

        # Load audio files (supports both WAV and NumPy formats)
        with st.spinner(f"Loading {max_files} audio files..."):
            audio_signals = []
            labels = []
            sample_rate = None
            format_counts = {"npy": 0, "wav": 0, "error": 0}

            for i, file_path in enumerate(file_paths[:max_files]):
                try:
                    # Try loading with new method that supports both WAV and NumPy
                    audio_data, sr, format_type = AudioVisualizer.load_audio_file(file_path, default_sample_rate=48000)

                    if audio_data is not None and len(audio_data) > 0:
                        audio_signals.append(audio_data)
                        format_counts[format_type] += 1

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
                    else:
                        format_counts["error"] += 1
                except Exception as e:
                    format_counts["error"] += 1
                    st.warning(f"Failed to load {os.path.basename(file_path)}: {e}")

        # Show loading results
        if not audio_signals:
            st.error("Failed to load any audio files")
            return

        # Show format breakdown
        format_info = []
        if format_counts["npy"] > 0:
            format_info.append(f"{format_counts['npy']} NumPy (full resolution)")
        if format_counts["wav"] > 0:
            format_info.append(f"{format_counts['wav']} WAV")
        if format_counts["error"] > 0:
            format_info.append(f"{format_counts['error']} errors")

        format_summary = ", ".join(format_info) if format_info else "unknown format"
        st.info(f"✓ Loaded {len(audio_signals)} of {len(file_paths)} files ({format_summary})")

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

        # FIR Filter Export Tool
        if FIR_FILTER_IO_AVAILABLE:
            st.markdown("---")
            with st.expander("💾 Export as FIR Filter", expanded=False):
                st.markdown("**Save calculated response as FIR filter**")
                st.caption("Export single scenario response to .fir filter bank (create new or add to existing)")

                # Source selection: single file or averaged
                st.markdown("**1. Select Source:**")
                source_mode = st.radio(
                    "Response source",
                    ["Single File", "Average of All Files"],
                    horizontal=True,
                    help="Choose a single response file or average all files in this scenario"
                )

                selected_file = None
                if source_mode == "Single File":
                    if len(file_paths) > 10:
                        selected_file = st.selectbox(
                            "Select response file",
                            file_paths,
                            format_func=lambda x: os.path.basename(x),
                            key="fir_single_file"
                        )
                    else:
                        selected_file = st.radio(
                            "Select response file",
                            file_paths,
                            format_func=lambda x: os.path.basename(x),
                            key="fir_single_file_radio"
                        )

                # File mode: create new or add to existing
                st.markdown("---")
                st.markdown("**2. Target FIR File:**")

                file_mode = st.radio(
                    "File operation",
                    ["Create New File", "Add to Existing File"],
                    horizontal=True,
                    help="Create a new .fir file or add this filter to an existing one"
                )

                fir_filename = None
                filter_length = None
                existing_info = None

                if file_mode == "Create New File":
                    col_a, col_b = st.columns([2, 1])

                    with col_a:
                        fir_filename = st.text_input(
                            "New FIR filename",
                            value=f"{audio_type}_bank.fir",
                            help="Name for new .fir filter bank file"
                        )
                        if not fir_filename.endswith('.fir'):
                            fir_filename = fir_filename + '.fir'

                    with col_b:
                        filter_length = st.number_input(
                            "Filter length (taps)",
                            min_value=64,
                            max_value=131072,
                            value=8192,
                            step=512,
                            help="Number of taps (samples) for FIR filters"
                        )

                else:  # Add to Existing File
                    # Look for existing .fir files in root FIR folder
                    dataset_root = st.session_state.get("dataset_root", os.getcwd())
                    fir_dir = os.path.join(dataset_root, "FIR")
                    existing_files = []
                    if os.path.exists(fir_dir):
                        existing_files = [f for f in os.listdir(fir_dir) if f.endswith('.fir')]

                    if not existing_files:
                        st.warning("⚠️ No existing .fir files found. Please create a new file first.")
                    else:
                        fir_filename = st.selectbox(
                            "Select existing FIR file",
                            existing_files,
                            help="Choose an existing .fir file to add this filter to"
                        )

                        # Load and display existing file info
                        existing_file_path = os.path.join(fir_dir, fir_filename)
                        try:
                            existing_info = get_fir_file_info(existing_file_path)
                            filter_length = existing_info['filter_length']

                            st.info(
                                f"📋 Existing file: {existing_info['num_filters']} filters, "
                                f"{existing_info['num_inputs']}×{existing_info['num_outputs']} channels, "
                                f"{filter_length} taps, {existing_info['sample_rate']} Hz"
                            )
                        except Exception as e:
                            st.error(f"Failed to read existing file: {e}")
                            existing_info = None

                # Channel mapping
                st.markdown("---")
                st.markdown("**3. Channel Mapping:**")

                col_a, col_b = st.columns(2)

                with col_a:
                    input_channel = st.number_input(
                        "Input channel",
                        min_value=0,
                        max_value=31,
                        value=0,
                        help="Source input channel for this filter"
                    )

                with col_b:
                    output_channel = st.number_input(
                        "Output channel",
                        min_value=0,
                        max_value=31,
                        value=0,
                        help="Destination output channel for this filter"
                    )

                # Sample rate
                sample_rate_input = st.number_input(
                    "Sample rate (Hz)",
                    min_value=8000,
                    max_value=192000,
                    value=48000,
                    step=1000,
                    help="Sample rate of the impulse response"
                )

                # Export button
                st.markdown("---")
                can_export = (
                    (source_mode == "Average of All Files" or selected_file is not None) and
                    fir_filename is not None and
                    filter_length is not None
                )

                if not can_export:
                    st.warning("⚠️ Please complete all settings above")

                if st.button(
                    "💾 Export to FIR Filter",
                    type="primary",
                    use_container_width=True,
                    disabled=not can_export
                ):
                    self._export_to_fir_filter(
                        scenario_path=scenario_path,
                        file_mode=file_mode,
                        fir_filename=fir_filename,
                        source_mode=source_mode,
                        selected_file=selected_file,
                        all_files=file_paths,
                        filter_length=filter_length,
                        input_channel=input_channel,
                        output_channel=output_channel,
                        sample_rate=sample_rate_input,
                        existing_info=existing_info
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
                # Load audio data using AudioVisualizer utility (supports both WAV and NumPy)
                audio_data, sample_rate, format_type = AudioVisualizer.load_audio_file(file_path, default_sample_rate=48000)

                if audio_data is not None:
                    # Create unique ID for this visualizer
                    viz_id = f"explorer_{hash(file_path)}"
                    visualizer = AudioVisualizer(viz_id)

                    # Add format indicator to title
                    format_label = "NumPy (full resolution)" if format_type == "npy" else "WAV"
                    title = f"Waveform: {os.path.basename(file_path)} [{format_label}]"

                    # Render with full features
                    visualizer.render(
                        audio_data=audio_data,
                        sample_rate=sample_rate,
                        title=title,
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

    def _export_to_fir_filter(
        self,
        scenario_path: str,
        file_mode: str,
        fir_filename: str,
        source_mode: str,
        selected_file: Optional[str],
        all_files: list,
        filter_length: int,
        input_channel: int,
        output_channel: int,
        sample_rate: int,
        existing_info: Optional[dict]
    ) -> None:
        """
        Export impulse response as FIR filter (create new or add to existing).

        Args:
            scenario_path: Path to scenario directory (used for context only)
            file_mode: "Create New File" or "Add to Existing File"
            fir_filename: Name of .fir file
            source_mode: "Single File" or "Average of All Files"
            selected_file: Path to selected file (if single file mode)
            all_files: List of all available files (for averaging)
            filter_length: Number of taps for the filter
            input_channel: Input channel number
            output_channel: Output channel number
            sample_rate: Sample rate in Hz
            existing_info: Info dict from existing file (if adding to existing)
        """
        if not FIR_FILTER_IO_AVAILABLE:
            st.error("FIR filter I/O module not available")
            return

        # Save to centralized FIR folder at dataset root level
        dataset_root = st.session_state.get("dataset_root", os.getcwd())
        output_dir = os.path.join(dataset_root, "FIR")
        output_file = os.path.join(output_dir, fir_filename)

        try:
            # Create output directory
            os.makedirs(output_dir, exist_ok=True)

            # Progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()

            # Step 1: Load or compute the impulse response
            status_text.text("Loading impulse response...")
            progress_bar.progress(0.2)

            impulse_response = None
            source_description = ""

            if source_mode == "Single File":
                # Load single file
                source_description = os.path.basename(selected_file)
                status_text.text(f"Loading {source_description}...")

                if VISUALIZER_AVAILABLE:
                    impulse_response, sr = AudioVisualizer.load_wav_file(selected_file)
                else:
                    import scipy.io.wavfile as wavfile
                    sr, impulse_response = wavfile.read(selected_file)
                    if impulse_response.dtype == np.int16:
                        impulse_response = impulse_response.astype(np.float32) / 32768.0
                    elif impulse_response.dtype == np.int32:
                        impulse_response = impulse_response.astype(np.float32) / 2147483648.0

                if impulse_response is None or len(impulse_response) == 0:
                    st.error(f"Failed to load {source_description}")
                    progress_bar.empty()
                    status_text.empty()
                    return

            else:  # Average of All Files
                source_description = f"Average of {len(all_files)} files"
                status_text.text(f"Averaging {len(all_files)} files...")

                signals = []
                sr = None

                for i, file_path in enumerate(all_files):
                    progress = 0.2 + (0.3 * (i / len(all_files)))
                    progress_bar.progress(progress)

                    if VISUALIZER_AVAILABLE:
                        audio_data, file_sr = AudioVisualizer.load_wav_file(file_path)
                    else:
                        import scipy.io.wavfile as wavfile
                        file_sr, audio_data = wavfile.read(file_path)
                        if audio_data.dtype == np.int16:
                            audio_data = audio_data.astype(np.float32) / 32768.0
                        elif audio_data.dtype == np.int32:
                            audio_data = audio_data.astype(np.float32) / 2147483648.0

                    if audio_data is not None and len(audio_data) > 0:
                        signals.append(audio_data)
                        if sr is None:
                            sr = file_sr

                if not signals:
                    st.error("Failed to load any files for averaging")
                    progress_bar.empty()
                    status_text.empty()
                    return

                # Average the signals (pad to same length first)
                max_len = max(len(s) for s in signals)
                padded_signals = []
                for sig in signals:
                    if len(sig) < max_len:
                        sig = np.pad(sig, (0, max_len - len(sig)), mode='constant')
                    padded_signals.append(sig)

                impulse_response = np.mean(padded_signals, axis=0).astype(np.float32)

            # Verify sample rate
            if sr != sample_rate:
                st.warning(f"Sample rate mismatch: loaded {sr} Hz, expected {sample_rate} Hz")

            progress_bar.progress(0.6)

            # Step 2: Truncate or pad to filter_length
            status_text.text(f"Truncating/padding to {filter_length} taps...")

            if len(impulse_response) > filter_length:
                # Truncate
                impulse_response = impulse_response[:filter_length]
            elif len(impulse_response) < filter_length:
                # Pad with zeros
                padding = filter_length - len(impulse_response)
                impulse_response = np.pad(impulse_response, (0, padding), mode='constant')

            impulse_response = impulse_response.astype(np.float32)
            progress_bar.progress(0.7)

            # Step 3: Create or update FIR filter bank
            if file_mode == "Create New File":
                # Create new filter bank with this single filter
                status_text.text("Creating new FIR filter bank...")

                # Determine num_inputs and num_outputs to accommodate this channel mapping
                num_inputs = input_channel + 1
                num_outputs = output_channel + 1
                num_filters = num_inputs * num_outputs

                # Create empty filter bank
                filters = np.zeros((num_filters, filter_length), dtype=np.float32)

                # Insert the impulse response at the correct position
                # mapId = input_channel × num_outputs + output_channel
                map_id = input_channel * num_outputs + output_channel
                filters[map_id] = impulse_response

                progress_bar.progress(0.8)

                # Save new file
                status_text.text(f"Saving new file {fir_filename}...")
                save_fir_filters(
                    filepath=output_file,
                    filters=filters,
                    num_inputs=num_inputs,
                    num_outputs=num_outputs,
                    sample_rate=sample_rate
                )

                is_new_file = True
                final_num_inputs = num_inputs
                final_num_outputs = num_outputs
                final_num_filters = num_filters

            else:  # Add to Existing File
                # Load existing filter bank
                status_text.text(f"Loading existing file {fir_filename}...")
                existing_data = load_fir_filters(output_file)

                existing_filters = existing_data['filters']
                num_inputs = existing_data['num_inputs']
                num_outputs = existing_data['num_outputs']
                num_filters = existing_data['num_filters']

                progress_bar.progress(0.75)

                # Check if we need to expand the filter bank
                required_inputs = input_channel + 1
                required_outputs = output_channel + 1

                if required_inputs > num_inputs or required_outputs > num_outputs:
                    # Need to expand filter bank
                    status_text.text("Expanding filter bank dimensions...")

                    new_num_inputs = max(num_inputs, required_inputs)
                    new_num_outputs = max(num_outputs, required_outputs)
                    new_num_filters = new_num_inputs * new_num_outputs

                    # Create expanded filter bank
                    new_filters = np.zeros((new_num_filters, filter_length), dtype=np.float32)

                    # Copy existing filters to new positions
                    for old_map_id in range(num_filters):
                        old_in_ch = old_map_id // num_outputs
                        old_out_ch = old_map_id % num_outputs
                        new_map_id = old_in_ch * new_num_outputs + old_out_ch
                        new_filters[new_map_id] = existing_filters[old_map_id]

                    filters = new_filters
                    num_inputs = new_num_inputs
                    num_outputs = new_num_outputs
                    num_filters = new_num_filters
                else:
                    filters = existing_filters.copy()

                progress_bar.progress(0.85)

                # Insert the new impulse response
                map_id = input_channel * num_outputs + output_channel
                filters[map_id] = impulse_response

                # Check if we're overwriting an existing filter
                existing_filter_present = np.any(existing_filters[map_id] != 0) if map_id < len(existing_filters) else False
                if existing_filter_present:
                    st.warning(f"⚠️ Overwriting existing filter at Input {input_channel} → Output {output_channel}")

                # Save updated file
                status_text.text(f"Saving updated file {fir_filename}...")
                save_fir_filters(
                    filepath=output_file,
                    filters=filters,
                    num_inputs=num_inputs,
                    num_outputs=num_outputs,
                    sample_rate=sample_rate
                )

                is_new_file = False
                final_num_inputs = num_inputs
                final_num_outputs = num_outputs
                final_num_filters = num_filters

            progress_bar.progress(1.0)

            # Clear progress indicators
            progress_bar.empty()
            status_text.empty()

            # Show success message
            if is_new_file:
                st.success(f"✅ Created new FIR filter bank with 1 filter!")
            else:
                st.success(f"✅ Added filter to existing FIR filter bank!")

            st.info(f"📁 Saved to: `{output_file}`")

            # Show FIR filter export information
            st.markdown("### FIR Filter Export Report")

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Filter Bank Size", f"{final_num_inputs}×{final_num_outputs}")

            with col2:
                st.metric("Total Filters", final_num_filters)

            with col3:
                st.metric("Filter Length", f"{filter_length} taps")

            with col4:
                st.metric("Sample Rate", f"{sample_rate} Hz")

            # Additional info
            col5, col6, col7 = st.columns(3)

            file_size = os.path.getsize(output_file)
            file_size_mb = file_size / (1024 * 1024)

            with col5:
                st.metric("File Size", f"{file_size_mb:.2f} MB")

            with col6:
                duration_ms = (filter_length / sample_rate) * 1000
                st.metric("Filter Duration", f"{duration_ms:.1f} ms")

            with col7:
                # Count non-zero filters
                final_data = load_fir_filters(output_file)
                non_zero_filters = sum(1 for f in final_data['filters'] if np.any(f != 0))
                st.metric("Active Filters", non_zero_filters)

            # This filter's details
            st.markdown("---")
            st.markdown("**Added Filter Details:**")

            col_a, col_b, col_c = st.columns(3)

            with col_a:
                st.write(f"**Routing:** Input {input_channel} → Output {output_channel}")

            with col_b:
                st.write(f"**Source:** {source_description}")

            with col_c:
                peak_val = np.max(np.abs(impulse_response))
                st.write(f"**Peak Amplitude:** {peak_val:.4f}")

            # Detailed channel mapping
            with st.expander("Full Channel Mapping"):
                st.markdown("**All filters in bank:**")
                final_data = load_fir_filters(output_file)
                for map_id, (in_ch, out_ch) in enumerate(final_data['channel_mapping']):
                    is_active = np.any(final_data['filters'][map_id] != 0)
                    status = "✓ Active" if is_active else "○ Empty"
                    highlight = " **← Just added**" if map_id == (input_channel * final_num_outputs + output_channel) else ""
                    st.write(f"- Filter {map_id}: Input {in_ch} → Output {out_ch} | {status}{highlight}")

            # Tips
            st.info(
                "💡 **Tip:** You can add more filters to this .fir file by selecting "
                "\"Add to Existing File\" and specifying different input/output channels."
            )

        except Exception as e:
            st.error(f"FIR export failed: {e}")
            import traceback
            st.code(traceback.format_exc())
            if 'progress_bar' in locals():
                progress_bar.empty()
            if 'status_text' in locals():
                status_text.empty()

    def _render_fir_export_dialog(self) -> None:
        """Render FIR export dialog when a scenario is selected for export."""
        if not FIR_FILTER_IO_AVAILABLE:
            return

        fir_export_path = st.session_state.get(SK_SCN_FIR_EXPORT)
        if not fir_export_path or not os.path.exists(fir_export_path):
            return

        st.markdown("---")
        st.subheader(f"💾 Export FIR Filter: {os.path.basename(fir_export_path)}")

        # Close button at the top
        if st.button("✕ Close FIR Export", key="close_fir_export_top"):
            st.session_state[SK_SCN_FIR_EXPORT] = None
            st.rerun()

        # Get available audio files
        audio_files = self._get_audio_files_by_type(fir_export_path)
        available_types = [k for k, v in audio_files.items() if v]

        if not available_types:
            st.warning("No audio files found in this scenario")
            return

        # Step 1: Select audio type
        st.markdown("**1. Select Audio Type:**")
        audio_type = st.selectbox(
            "Audio type",
            available_types,
            format_func=lambda x: f"{x.replace('_', ' ').title()} ({len(audio_files[x])} files)",
            key="fir_export_audio_type"
        )

        files_of_type = audio_files[audio_type]

        # Step 2: Select source
        st.markdown("---")
        st.markdown("**2. Select Source:**")
        source_mode = st.radio(
            "Response source",
            ["Single File", "Average of All Files"],
            horizontal=True,
            help="Choose a single response file or average all files",
            key="fir_export_source_mode"
        )

        selected_file = None
        if source_mode == "Single File":
            if len(files_of_type) > 10:
                selected_file = st.selectbox(
                    "Select response file",
                    files_of_type,
                    format_func=lambda x: os.path.basename(x),
                    key="fir_export_single_file"
                )
            else:
                selected_file = st.radio(
                    "Select response file",
                    files_of_type,
                    format_func=lambda x: os.path.basename(x),
                    key="fir_export_single_file_radio"
                )

        # Step 3: File mode
        st.markdown("---")
        st.markdown("**3. Target FIR File:**")

        file_mode = st.radio(
            "File operation",
            ["Create New File", "Add to Existing File"],
            horizontal=True,
            help="Create a new .fir file or add this filter to an existing one",
            key="fir_export_file_mode"
        )

        fir_filename = None
        filter_length = None
        existing_info = None

        if file_mode == "Create New File":
            col_a, col_b = st.columns([2, 1])

            with col_a:
                fir_filename = st.text_input(
                    "New FIR filename",
                    value=f"{audio_type}_bank.fir",
                    help="Name for new .fir filter bank file",
                    key="fir_export_new_filename"
                )
                if not fir_filename.endswith('.fir'):
                    fir_filename = fir_filename + '.fir'

            with col_b:
                filter_length = st.number_input(
                    "Filter length (taps)",
                    min_value=64,
                    max_value=131072,
                    value=8192,
                    step=512,
                    help="Number of taps (samples) for FIR filters",
                    key="fir_export_filter_length"
                )

        else:  # Add to Existing File
            # Look for existing .fir files in root FIR folder
            dataset_root = st.session_state.get("dataset_root", os.getcwd())
            fir_dir = os.path.join(dataset_root, "FIR")
            existing_files = []
            if os.path.exists(fir_dir):
                existing_files = [f for f in os.listdir(fir_dir) if f.endswith('.fir')]

            if not existing_files:
                st.warning("⚠️ No existing .fir files found in FIR folder. Please create a new file first.")
            else:
                fir_filename = st.selectbox(
                    "Select existing FIR file",
                    existing_files,
                    help="Choose an existing .fir file to add this filter to",
                    key="fir_export_existing_filename"
                )

                # Load and display existing file info
                existing_file_path = os.path.join(fir_dir, fir_filename)
                try:
                    existing_info = get_fir_file_info(existing_file_path)
                    filter_length = existing_info['filter_length']

                    st.info(
                        f"📋 Existing file: {existing_info['num_filters']} filters, "
                        f"{existing_info['num_inputs']}×{existing_info['num_outputs']} channels, "
                        f"{filter_length} taps, {existing_info['sample_rate']} Hz"
                    )
                except Exception as e:
                    st.error(f"Failed to read existing file: {e}")
                    existing_info = None

        # Step 4: Channel mapping
        st.markdown("---")
        st.markdown("**4. Channel Mapping:**")

        col_a, col_b = st.columns(2)

        with col_a:
            input_channel = st.number_input(
                "Input channel",
                min_value=0,
                max_value=31,
                value=0,
                help="Source input channel for this filter",
                key="fir_export_input_channel"
            )

        with col_b:
            output_channel = st.number_input(
                "Output channel",
                min_value=0,
                max_value=31,
                value=0,
                help="Destination output channel for this filter",
                key="fir_export_output_channel"
            )

        # Step 5: Sample rate
        sample_rate_input = st.number_input(
            "Sample rate (Hz)",
            min_value=8000,
            max_value=192000,
            value=48000,
            step=1000,
            help="Sample rate of the impulse response",
            key="fir_export_sample_rate"
        )

        # Export button
        st.markdown("---")
        can_export = (
            (source_mode == "Average of All Files" or selected_file is not None) and
            fir_filename is not None and
            filter_length is not None
        )

        if not can_export:
            st.warning("⚠️ Please complete all settings above")

        col_export, col_close = st.columns([3, 1])

        with col_export:
            if st.button(
                "💾 Export to FIR Filter",
                type="primary",
                use_container_width=True,
                disabled=not can_export,
                key="fir_export_button"
            ):
                self._export_to_fir_filter(
                    scenario_path=fir_export_path,
                    file_mode=file_mode,
                    fir_filename=fir_filename,
                    source_mode=source_mode,
                    selected_file=selected_file,
                    all_files=files_of_type,
                    filter_length=filter_length,
                    input_channel=input_channel,
                    output_channel=output_channel,
                    sample_rate=sample_rate_input,
                    existing_info=existing_info
                )

        with col_close:
            if st.button("Cancel", use_container_width=True, key="fir_export_cancel"):
                st.session_state[SK_SCN_FIR_EXPORT] = None
                st.rerun()

    # ========================================================================
    # Channel Exploration and Visualization
    # ========================================================================

    def _render_by_channel_view(self, scenario_path: str, scenario_name: str) -> None:
        """Render channel-wise view: select one channel, overlay multiple measurements."""
        st.markdown("#### View measurements from the same channel")
        st.caption("Select a channel and visualize multiple measurements overlaid")

        # Get available channels
        impulse_folder = os.path.join(scenario_path, "impulse_responses")
        if not os.path.isdir(impulse_folder):
            st.error("Impulse responses folder not found")
            return

        # Get all WAV files
        wav_files = []
        try:
            for filename in os.listdir(impulse_folder):
                if filename.lower().endswith(('.wav', '.npy')):
                    wav_files.append(os.path.join(impulse_folder, filename))
        except (OSError, PermissionError):
            st.error("Cannot read impulse responses folder")
            return

        if not wav_files:
            st.info("No audio files found")
            return

        # Group by channel
        channel_groups = group_files_by_channel(wav_files)

        if not channel_groups:
            st.info("No multi-channel files found")
            return

        # Get channel info from recorder config (if available)
        channel_names = self._load_channel_names_from_metadata(scenario_path)

        # Channel selector
        available_channels = sorted(channel_groups.keys())
        channel_labels = [f"Ch {ch}: {channel_names.get(ch, f'Channel {ch}')}" for ch in available_channels]

        selected_channel_idx = st.selectbox(
            "Select Channel:",
            range(len(available_channels)),
            format_func=lambda i: channel_labels[i],
            key="explore_selected_channel"
        )

        selected_channel = available_channels[selected_channel_idx]
        channel_files = channel_groups[selected_channel]

        st.info(f"Found {len(channel_files)} measurements for channel {selected_channel}")

        # Measurement selection
        st.markdown("**Select Measurements to Overlay:**")

        # Parse measurement indices
        measurement_info = []
        for file_path in channel_files:
            parsed = parse_multichannel_filename(file_path)
            if parsed:
                measurement_info.append({
                    'index': parsed.index,
                    'file': file_path,
                    'timestamp': parsed.timestamp
                })

        measurement_info.sort(key=lambda x: x['index'])

        if not measurement_info:
            st.error("No valid measurements found")
            return

        # Multi-select for measurements
        all_indices = [m['index'] for m in measurement_info]
        selected_indices = st.multiselect(
            "Choose measurements to overlay:",
            all_indices,
            default=all_indices[:min(5, len(all_indices))],  # Default to first 5
            format_func=lambda idx: f"Measurement {idx:03d}",
            key=f"explore_measurements_ch{selected_channel}"
        )

        if not selected_indices:
            st.warning("Select at least one measurement")
            return

        # Session state key for loaded data
        loaded_key = f"loaded_ch{selected_channel}_{len(selected_indices)}"

        # Auto-load if selections changed
        if st.button("Load and Visualize", key=f"load_viz_ch{selected_channel}") or loaded_key not in st.session_state:
            with st.spinner("Loading audio files..."):
                audio_signals = []
                labels = []
                sample_rate = 48000  # Default

                for meas_idx in selected_indices:
                    # Find the file for this measurement
                    meas_file = next((m['file'] for m in measurement_info if m['index'] == meas_idx), None)
                    if meas_file:
                        # Load audio file (supports both .wav and .npy)
                        audio_data, sr, fmt = AudioVisualizer.load_audio_file(meas_file, default_sample_rate=48000)
                        if audio_data is not None:
                            audio_signals.append(audio_data)
                            labels.append(f"Meas {meas_idx:03d}")
                            sample_rate = sr

                if audio_signals:
                    st.session_state[loaded_key] = {
                        'signals': audio_signals,
                        'labels': labels,
                        'sample_rate': sample_rate,
                        'channel': selected_channel,
                        'scenario_name': scenario_name,
                        'channel_name': channel_names.get(selected_channel, f'Channel {selected_channel}')
                    }
                else:
                    st.error("Failed to load any audio files")
                    return

        # Display if data is loaded
        if loaded_key in st.session_state:
            data = st.session_state[loaded_key]
            st.success(f"Loaded {len(data['signals'])} measurements")

            # Visualization options
            normalize = st.checkbox(
                "Normalize signals",
                value=True,
                help="Normalize each signal to max amplitude of 1",
                key=f"normalize_ch{selected_channel}"
            )

            # Render overlay
            st.markdown(f"### Channel {data['channel']}: {data['channel_name']}")

            AudioVisualizer.render_multi_waveform_with_zoom(
                audio_signals=data['signals'],
                sample_rate=data['sample_rate'],
                labels=data['labels'],
                title=f"Overlay: Ch{data['channel']} - {data['scenario_name']}",
                component_id=f"overlay_ch{data['channel']}",
                normalize=normalize
            )

    def _render_by_measurement_view(self, scenario_path: str, scenario_name: str) -> None:
        """Render measurement-wise view: select one measurement, compare multiple channels side-by-side."""
        st.markdown("#### Compare channels within a single measurement")
        st.caption("Select a measurement and visualize multiple channels side-by-side")

        # Get measurements
        measurements_dict = self.scenario_manager.list_wavs_multichannel(scenario_path)

        if not measurements_dict:
            st.info("No measurements found")
            return

        # Measurement selector
        available_measurements = sorted(measurements_dict.keys())
        selected_measurement = st.selectbox(
            "Select Measurement:",
            available_measurements,
            format_func=lambda idx: f"Measurement {idx:03d}",
            key="explore_selected_measurement"
        )

        measurement_files = measurements_dict[selected_measurement]

        # Parse channel information
        channel_file_map = {}
        for file_path in measurement_files:
            parsed = parse_multichannel_filename(file_path)
            if parsed and parsed.is_multichannel:
                channel_file_map[parsed.channel] = file_path

        if not channel_file_map:
            st.info("No channel files found for this measurement")
            return

        # Get channel info from metadata
        channel_names = self._load_channel_names_from_metadata(scenario_path)

        st.info(f"Found {len(channel_file_map)} channels in measurement {selected_measurement:03d}")

        # Channel multi-select
        all_channels = sorted(channel_file_map.keys())
        channel_labels_dict = {ch: f"Ch {ch}: {channel_names.get(ch, f'Channel {ch}')}" for ch in all_channels}

        selected_channels = st.multiselect(
            "Choose channels to compare:",
            all_channels,
            default=all_channels[:min(4, len(all_channels))],  # Default to first 4
            format_func=lambda ch: channel_labels_dict[ch],
            key=f"explore_channels_meas{selected_measurement}"
        )

        if not selected_channels:
            st.warning("Select at least one channel")
            return

        # Layout mode
        layout_mode = st.radio(
            "Display Layout:",
            ["Overlaid (Single Plot)", "Stacked (Separate Plots)"],
            index=0,
            horizontal=True,
            key=f"layout_mode_meas{selected_measurement}"
        )

        # Session state key for loaded data
        loaded_key = f"loaded_meas{selected_measurement}_{len(selected_channels)}_{layout_mode}"

        # Load and display
        if st.button("Load and Visualize", key=f"load_viz_meas{selected_measurement}") or loaded_key not in st.session_state:
            with st.spinner("Loading audio files..."):
                audio_signals = []
                labels = []
                sample_rate = 48000  # Default

                for ch_idx in selected_channels:
                    file_path = channel_file_map[ch_idx]
                    # Load audio file
                    audio_data, sr, fmt = AudioVisualizer.load_audio_file(file_path, default_sample_rate=48000)
                    if audio_data is not None:
                        audio_signals.append(audio_data)
                        labels.append(channel_labels_dict[ch_idx])
                        sample_rate = sr

                if audio_signals:
                    st.session_state[loaded_key] = {
                        'signals': audio_signals,
                        'labels': labels,
                        'sample_rate': sample_rate,
                        'measurement': selected_measurement,
                        'channels': selected_channels,
                        'layout': layout_mode
                    }
                else:
                    st.error("Failed to load any audio files")
                    return

        # Display if data is loaded
        if loaded_key in st.session_state:
            data = st.session_state[loaded_key]
            st.success(f"Loaded {len(data['signals'])} channels")

            # Visualization options
            normalize = st.checkbox(
                "Normalize signals",
                value=False,
                help="Normalize each signal to max amplitude of 1",
                key=f"normalize_meas{selected_measurement}"
            )

            # Render comparison
            st.markdown(f"### Measurement {data['measurement']:03d} - Channel Comparison")

            if data['layout'] == "Overlaid (Single Plot)":
                # Single overlaid plot
                AudioVisualizer.render_multi_waveform_with_zoom(
                    audio_signals=data['signals'],
                    sample_rate=data['sample_rate'],
                    labels=data['labels'],
                    title=f"Meas {data['measurement']:03d} - Channels Overlaid",
                    component_id=f"compare_meas{data['measurement']}_overlay",
                    normalize=normalize
                )
            else:
                # Stacked plots (one per channel)
                st.markdown("**Stacked Channel View:**")
                for i, (audio_data, label) in enumerate(zip(data['signals'], data['labels'])):
                    with st.expander(f"📊 {label}", expanded=(i == 0)):
                        AudioVisualizer.render_multi_waveform_with_zoom(
                            audio_signals=[audio_data],
                            sample_rate=data['sample_rate'],
                            labels=[label],
                            title=label,
                            component_id=f"compare_meas{data['measurement']}_ch{data['channels'][i]}",
                            normalize=normalize,
                            height=300
                        )

    def _load_channel_names_from_metadata(self, scenario_path: str) -> dict:
        """Load channel names from scenario metadata."""
        channel_names = {}
        metadata_path = os.path.join(scenario_path, "metadata", "session_metadata.json")

        if os.path.exists(metadata_path):
            try:
                import json
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)

                # Try to get channel names from recorder config in metadata
                recorder_config = metadata.get('recorder_config', {})
                mc_config = recorder_config.get('multichannel_config', {})

                if 'channel_names' in mc_config:
                    for i, name in enumerate(mc_config['channel_names']):
                        channel_names[i] = name
            except Exception:
                pass

        return channel_names
