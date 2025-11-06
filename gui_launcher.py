#!/usr/bin/env python3
"""
Room Response GUI - Main Application (Refactored + Lightweight Folder Picker)

- Keeps a single ScenarioClassifier instance in session state (if you use it in Classify panel)
- Delegates model persistence/metadata to ScenarioClassifier
- Adds a robust, lazy folder browser for dataset selection:
    * Shows only folder name in the text field
    * Resolves short names as subfolders of the working directory
    * Lists only immediate subfolders (no recursion) to avoid hangs
    * Safely updates session_state without modifying widget keys post-creation
"""

import os
import sys
import string
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

try:
    from gui_visualize_panel import VisualizationPanel
except ImportError:
    VisualizationPanel = None

# Optional: PredictPanel if you have it
try:
    from gui_predict_panel import PredictionPanel
except ImportError:
    PredictionPanel = None

try:
    from gui_config_profiles import ConfigProfileManager
except ImportError:
    ConfigProfileManager = None

# ---------------------------- Session Keys ----------------------------
SK_DATASET_ROOT = "dataset_root"                  # full absolute path
SK_DEFAULT_DATASET_ROOT = "room_response_dataset"
SK_SCN_SELECTIONS = "scenarios_selected_set"
SK_SCN_EXPLORE = "scenarios_explore_path"
SK_FILTER_TEXT = "filter_text"
SK_FILTER_COMPUTER = "filter_computer"
SK_FILTER_ROOM = "filter_room"
SK_SAVED_LABELS = "saved_labels_cache"

# Dataset picker keys
SK_DATASET_NAME = "dataset_folder_name"           # only folder name shown in field
SK_DATASET_NAME_PENDING = "dataset_folder_name_pending"
SK_BROWSER_OPEN = "dataset_browser_open"
SK_BROWSER_CWD = "dataset_browser_cwd"
SK_BROWSER_FILTER = "dataset_browser_filter"

SK_DATASET_NAME = "dataset_folder_name"  # text box stores only the short folder name or a user-provided path

# Classifier obj key (if used by Classify panel)
SK_CLASSIFIER_OBJ = "classifier_obj"


class RoomResponseGUI:
    def __init__(self):
        self.scenario_manager = None
        self.collection_panel = None
        self.processing_panel = None
        self.classification_panel = None
        self.visualization_panel = None
        self.prediction_panel = None
        self.config_profile_manager = None
        self._initialize_components()

    # ---------------------------- Setup ----------------------------
    def _initialize_components(self):
        if ScenarioManager is not None:
            self.scenario_manager = ScenarioManager()
        if CollectionPanel and self.scenario_manager:
            self.collection_panel = CollectionPanel(self.scenario_manager)
        if ProcessingPanel and self.scenario_manager:
            self.processing_panel = ProcessingPanel(self.scenario_manager)
        if ClassificationPanel and self.scenario_manager:
            self.classification_panel = ClassificationPanel(self.scenario_manager)
        if VisualizationPanel and self.scenario_manager:
            self.visualization_panel = VisualizationPanel(self.scenario_manager)
        if PredictionPanel and self.scenario_manager:
            self.prediction_panel = PredictionPanel(self.scenario_manager)
        if ConfigProfileManager is not None:
            self.config_profile_manager = ConfigProfileManager()

    def run(self):
        st.set_page_config(page_title="Room Response GUI", layout="wide")
        self._ensure_initial_state()
        self._ensure_dataset_root_ui()
        if self.config_profile_manager is not None:
            self.config_profile_manager.render_sidebar_ui()
        panel = self._render_sidebar_navigation()
        self._render_panel(panel)

    def _ensure_initial_state(self):
        # Initialize persistent containers
        st.session_state.setdefault(SK_SCN_SELECTIONS, set())
        st.session_state.setdefault(SK_SAVED_LABELS, {})
        st.session_state.setdefault(SK_BROWSER_OPEN, False)
        st.session_state.setdefault(SK_BROWSER_FILTER, "")
        # Initialize dataset defaults
        default_root = st.session_state.get(
            SK_DATASET_ROOT, os.path.join(os.getcwd(), SK_DEFAULT_DATASET_ROOT)
        )
        default_root = os.path.abspath(default_root)
        st.session_state.setdefault(SK_DATASET_ROOT, default_root)
        # If a pending name was set by the browser, assign it BEFORE widget creation
        pending = st.session_state.pop(SK_DATASET_NAME_PENDING, None)
        if pending is not None:
            st.session_state[SK_DATASET_NAME] = pending
        else:
            # If no explicit name yet, derive from current root
            st.session_state.setdefault(SK_DATASET_NAME, os.path.basename(st.session_state[SK_DATASET_ROOT]) or SK_DEFAULT_DATASET_ROOT)

    # ---------------------------- Dataset Picker ----------------------------
    def _ensure_dataset_root_ui(self) -> str:
        """Dataset root selection (no browsing). Users type a folder *name* or any path.
        - Bare name -> treat as subfolder of CWD
        - Path (absolute, relative with separators, or ~) -> resolve accordingly
        - Show resolved absolute path under the field
        """
        import os
        st.sidebar.markdown("### Dataset")

        # Derive sane defaults
        current_root = st.session_state.get(SK_DATASET_ROOT, os.path.join(os.getcwd(), SK_DEFAULT_DATASET_ROOT))
        default_name = os.path.basename(current_root) if os.path.basename(current_root) else SK_DEFAULT_DATASET_ROOT

        # Important: set default BEFORE creating the widget; never assign to this key afterwards
        st.session_state.setdefault(SK_DATASET_NAME, default_name)

        # Short input: folder name or path (keeps sidebar UI compact)
        folder_hint = "Folder name (e.g. room_response_dataset) or path (e.g. D:\\data\\rr)"
        st.sidebar.text_input(
            "Dataset folder (name or path)",
            key=SK_DATASET_NAME,
            help=f"{folder_hint}. A bare name will be treated as a subfolder of the working directory.",
        )

        typed = (st.session_state.get(SK_DATASET_NAME) or "").strip()

        # Resolve to an absolute path
        def _looks_like_path(s: str) -> bool:
            # Treat as path if absolute, starts with ~, or contains a path separator
            return os.path.isabs(s) or s.startswith("~") or (("/" in s) or ("\\" in s))

        if not typed:
            # Fall back to previous root if user clears the field
            resolved = os.path.abspath(current_root)
        elif _looks_like_path(typed):
            resolved = os.path.abspath(os.path.expanduser(typed))
        else:
            # Bare folder name → subfolder of CWD
            resolved = os.path.abspath(os.path.join(os.getcwd(), typed))

        # Validate and show status
        if self.scenario_manager is not None:
            ok, msg = self.scenario_manager.validate_dataset_root(resolved)
            (st.sidebar.success if ok else st.sidebar.error)(msg, icon="📁")

        # Show resolved path read-only (so long paths don't break editing)
        st.sidebar.caption("Resolved path:")
        st.sidebar.code(resolved, language=None)

        # Handle root changes → clear caches & selection
        if self.scenario_manager is not None:
            last_root = st.session_state.get(self.scenario_manager.SK_LAST_DATASET_ROOT)
            if last_root != resolved:
                st.session_state[self.scenario_manager.SK_LAST_DATASET_ROOT] = resolved
                self.scenario_manager.clear_cache()
                st.session_state[SK_SCN_SELECTIONS] = set()
                st.session_state[SK_SCN_EXPLORE] = None

        st.session_state[SK_DATASET_ROOT] = resolved
        return resolved


    def _on_dataset_name_change(self):
        """Resolve typed name to absolute path; store to SK_DATASET_ROOT and invalidate caches if changed."""
        name = (st.session_state.get(SK_DATASET_NAME, "") or "").strip()
        if not name:
            return
        path = self._resolve_name_to_path(name)
        if os.path.abspath(path) != os.path.abspath(st.session_state.get(SK_DATASET_ROOT, "")):
            self._apply_new_dataset_root(path)

    def _apply_new_dataset_root(self, new_root: str):
        """Set new dataset root and invalidate caches safely."""
        new_root = os.path.abspath(new_root)
        st.session_state[SK_DATASET_ROOT] = new_root
        # Cache invalidation on root change
        if self.scenario_manager is not None:
            last_root = st.session_state.get(self.scenario_manager.SK_LAST_DATASET_ROOT)
            if last_root != new_root:
                st.session_state[self.scenario_manager.SK_LAST_DATASET_ROOT] = new_root
                self.scenario_manager.clear_cache()
                st.session_state[SK_SCN_SELECTIONS] = set()
                st.session_state[SK_SCN_EXPLORE] = None

    def _resolve_name_to_path(self, name: str) -> str:
        """Map a short folder name to CWD subfolder; leave absolute paths as-is."""
        if not name:
            # fallback to default in CWD
            return os.path.join(os.getcwd(), SK_DEFAULT_DATASET_ROOT)
        # Windows absolute path detection needs special-casing (e.g., C:\ or X:/)
        if os.path.isabs(name) or self._looks_like_windows_abs(name):
            return os.path.abspath(name)
        # Relative: treat as subfolder of working directory
        return os.path.abspath(os.path.join(os.getcwd(), name))

    @staticmethod
    def _looks_like_windows_abs(p: str) -> bool:
        if len(p) >= 2 and p[1] == ":":
            return True
        return False

    # ---- Folder Browser (lazy, one level, safe) ----
    def _render_folder_browser(self):
        st.sidebar.markdown("---")
        st.sidebar.markdown("**Choose dataset folder**")

        cwd = st.session_state.get(SK_BROWSER_CWD, os.getcwd())
        cwd = os.path.abspath(cwd)

        # Quick jump row (drives on Windows, root/home on POSIX)
        cj1, cj2 = st.sidebar.columns([1, 1])
        with cj1:
            if st.button("Home"):
                st.session_state[SK_BROWSER_CWD] = os.path.expanduser("~")
        with cj2:
            if os.name == "nt":
                if st.button("Drives"):
                    # present a small selector with available drives
                    st.session_state[SK_BROWSER_CWD] = self._first_available_drive() or "C:\\"
            else:
                if st.button("Root"):
                    st.session_state[SK_BROWSER_CWD] = os.path.abspath(os.sep)

        # Current path + controls
        st.sidebar.caption(f"Current: {cwd}")
        btns = st.sidebar.columns([1, 1, 1])
        with btns[0]:
            if st.button("⬆️ Up"):
                parent = os.path.dirname(cwd.rstrip("\\/"))
                if parent and os.path.isdir(parent):
                    st.session_state[SK_BROWSER_CWD] = parent
        with btns[1]:
            if st.button("Use this folder"):
                chosen = cwd
                # Set new dataset root + close browser
                st.session_state[SK_DATASET_ROOT] = chosen
                # IMPORTANT: don't mutate the text_input's key directly; set PENDING then rerun
                st.session_state[SK_DATASET_NAME_PENDING] = os.path.basename(chosen) or chosen
                # Invalidate caches
                if self.scenario_manager is not None:
                    st.session_state[self.scenario_manager.SK_LAST_DATASET_ROOT] = chosen
                    self.scenario_manager.clear_cache()
                    st.session_state[SK_SCN_SELECTIONS] = set()
                    st.session_state[SK_SCN_EXPLORE] = None
                st.session_state[SK_BROWSER_OPEN] = False
                st.rerun()
        with btns[2]:
            if st.button("Cancel"):
                st.session_state[SK_BROWSER_OPEN] = False
                return

        # Filter box (non-regex, fast substring)
        st.sidebar.text_input(
            "Filter folders",
            key=SK_BROWSER_FILTER,
            placeholder="Type to filter…",
            label_visibility="collapsed",
        )

        # List immediate subfolders (fast, non-recursive, capped)
        with st.spinner("Listing folders…"):
            subdirs = self._list_subdirs(cwd)
        filt = (st.session_state.get(SK_BROWSER_FILTER, "") or "").lower().strip()
        if filt:
            subdirs = [p for p in subdirs if filt in os.path.basename(p).lower()]

        if not subdirs:
            st.sidebar.info("No subfolders here.")
            return

        # Show as selectbox + Open button (no per-row heavy widgets)
        # Limit to first 1000 to keep UI snappy
        MAX_SHOW = 1000
        shown = subdirs[:MAX_SHOW]
        if len(subdirs) > MAX_SHOW:
            st.sidebar.caption(f"Showing first {MAX_SHOW} of {len(subdirs)}")

        choice = st.sidebar.selectbox(
            "Subfolders",
            options=shown,
            index=0,
            format_func=lambda p: "📁 " + os.path.basename(p),
            label_visibility="collapsed",
        )
        if st.sidebar.button("Open selected"):
            if choice and os.path.isdir(choice):
                st.session_state[SK_BROWSER_CWD] = choice

    def _list_subdirs(self, base: str):
        """Return a sorted list of immediate subdirectories, robust to errors."""
        try:
            with os.scandir(base) as it:
                dirs = [entry.path for entry in it if entry.is_dir(follow_symlinks=False)]
        except Exception:
            dirs = []
        # Alphabetical by name, case-insensitive
        dirs.sort(key=lambda p: os.path.basename(p).lower())
        return dirs

    def _first_available_drive(self) -> str | None:
        if os.name != "nt":
            return None
        for letter in string.ascii_uppercase:
            drive = f"{letter}:/"
            try:
                if os.path.isdir(drive):
                    return drive
            except Exception:
                continue
        return None

    # ---------------------------- Sidebar Nav ----------------------------
    def _render_sidebar_navigation(self) -> str:
        st.sidebar.markdown("### Panels")

        # If you keep a classifier object in session (used in Classify panel)
        clf = st.session_state.get(SK_CLASSIFIER_OBJ)
        has_model = bool(getattr(clf, "is_trained", lambda: False)())

        options = ["Scenarios", "Collect", "Process", "Classify", "Predict", "Visualize"]
        selected = st.sidebar.radio("Select a panel", options=options, index=0)

        if selected in ("Predict", "Visualize") and not has_model:
            st.sidebar.warning("⚠️ Requires a trained/loaded model from Classify panel.")
            selected = "Classify"

        if has_model:
            info = getattr(clf, "get_model_info", lambda: {})()
            st.sidebar.success("✅ Model available")
            st.sidebar.caption(f"Type: {info.get('model_type', '—').upper()}")
            st.sidebar.caption(f"Features: {info.get('feature_type', '—')}")
            if ds := info.get("dataset_root"):
                st.sidebar.caption(f"Dataset: {ds}")
        else:
            st.sidebar.info("ℹ️ No model trained yet")

        return selected

    # ---------------------------- Panels ----------------------------
    def _render_panel(self, panel: str):
        if panel == "Scenarios":
            self._render_scenarios_panel()
        elif panel == "Collect":
            self.collection_panel.render() if self.collection_panel else st.error("❌ Collection panel missing")
        elif panel == "Process":
            self.processing_panel.render() if self.processing_panel else st.error("❌ Processing panel missing")
        elif panel == "Classify":
            self.classification_panel.render() if self.classification_panel else st.error("❌ Classification panel missing")
        elif panel == "Predict":
            if self.prediction_panel:
                self.prediction_panel.render()
            else:
                st.info("Prediction panel not implemented yet.")
        elif panel == "Visualize":
            self.visualization_panel.render() if self.visualization_panel else st.info("Visualization panel not implemented yet.")
        else:
            st.info(f"Panel '{panel}' not implemented.")

    # ---------------------------- Scenarios Panel ----------------------------
    def _render_scenarios_panel(self):
        if self.scenario_manager is None:
            st.error("❌ ScenarioManager not available.")
            return

        st.header("Scenarios")
        root = st.session_state.get(SK_DATASET_ROOT, os.getcwd())
        if not os.path.isdir(root):
            st.error("❌ Provide a valid dataset root directory.", icon="📁")
            return

        self._render_scenario_filters()

        with st.spinner("Loading scenarios..."):
            df = self.scenario_manager.build_scenarios_df(root)
        if df.empty:
            st.info("No scenarios found.")
            return

        dfv = self.scenario_manager.apply_filters(
            df,
            st.session_state.get(SK_FILTER_TEXT, ""),
            st.session_state.get(SK_FILTER_COMPUTER, ""),
            st.session_state.get(SK_FILTER_ROOM, ""),
        )
        dfv = self.scenario_manager.sort_scenarios_df(dfv)

        self._render_scenario_bulk_operations(df, dfv)
        self._render_scenario_selection_controls(dfv, len(df))
        self._render_scenario_table(dfv)
        self._render_scenario_summary(df, dfv)
        self._render_scenario_explorer()

    def _render_scenario_filters(self):
        col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
        with col1:
            st.text_input(
                "Scenario name filter (regex supported)",
                key=SK_FILTER_TEXT,
                label_visibility="collapsed",
                placeholder=r"Filter by scenario number (regex): ^6\., 0.*  |  name:Studio",
            )
        with col2:
            st.text_input("Computer filter", key=SK_FILTER_COMPUTER, label_visibility="collapsed")
        with col3:
            st.text_input("Room filter", key=SK_FILTER_ROOM, label_visibility="collapsed")
        with col4:
            if st.button("Re-analyze"):
                self.scenario_manager.clear_cache()
                st.rerun()

    def _render_scenario_bulk_operations(self, df, dfv):
        st.markdown("**Bulk operations for filtered scenarios**")
        existing = sorted(self.scenario_manager.get_unique_labels(df))
        if existing:
            st.caption("Existing labels: " + ", ".join(existing[:12]) + ("…" if len(existing) > 12 else ""))

        with st.form("bulk_ops_form", clear_on_submit=False):
            c1, c2, c3 = st.columns([2, 1, 1])
            with c1:
                label_text = st.text_input("Label(s) to apply", key="bulk_label_input", placeholder="quiet,baseline")
                append_mode = st.checkbox("Append instead of replace", value=False)
            with c2:
                apply_btn = st.form_submit_button(f"Apply to {len(dfv)}")
            with c3:
                clear_btn = st.form_submit_button(f"Clear labels ({len(dfv)})")

        if apply_btn:
            new_label = (label_text or "").strip()
            if not new_label:
                st.warning("Enter label text to apply.")
                return
            updated = 0
            try:
                if append_mode:
                    for _, row in dfv.iterrows():
                        path = row["path"]
                        current = (row.get("label") or "").strip()
                        current_set = [s.strip() for s in current.split(",") if s.strip()] if current else []
                        to_add = [s.strip() for s in new_label.split(",") if s.strip()]
                        merged = current_set[:]
                        for t in to_add:
                            if t not in merged:
                                merged.append(t)
                        merged_str = ", ".join(merged)
                        if self.scenario_manager.write_label(path, merged_str):
                            updated += 1
                else:
                    updated = self.scenario_manager.bulk_apply_label(dfv, new_label)
            except TypeError:
                updated = self.scenario_manager.bulk_apply_label(dfv, new_label)

            if updated:
                st.success(f"Applied to {updated} scenario(s).")
                self.scenario_manager.clear_cache()
                st.rerun()
            else:
                st.error("Failed to apply label(s).")

        if clear_btn:
            updated = self.scenario_manager.bulk_apply_label(dfv, None)
            if updated:
                st.success(f"Cleared labels for {updated} scenario(s).")
                self.scenario_manager.clear_cache()
                st.rerun()
            else:
                st.error("Failed to clear labels.")

    def _render_scenario_selection_controls(self, dfv, total_count):
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
                    p = row["path"]
                    s = st.session_state[SK_SCN_SELECTIONS]
                    (s.discard if p in s else s.add)(p)
                st.rerun()
        with col4:
            st.caption(f"Selected: {len(st.session_state[SK_SCN_SELECTIONS])} | Filtered: {len(dfv)} | Total: {total_count}")

    def _render_scenario_table(self, dfv):
        st.markdown("---")
        col1, col2, col3, col4, col5, col6 = st.columns([0.08, 0.15, 0.25, 0.12, 0.25, 0.15])
        with col1: st.markdown("**☑**")
        with col2: st.markdown("**Scenario #**")
        with col3: st.markdown("**Group Label**")
        with col4: st.markdown("**Features**")
        with col5: st.markdown("**Description**")
        with col6: st.markdown("**Actions**")
        for idx, row in dfv.iterrows():
            self._render_scenario_row(idx, row)

    def _render_scenario_row(self, idx, row):
        scn_path = row["path"]
        number_str = row.get("number_str") or ""
        label_val = row.get("label") or ""
        description_val = row.get("description") or ""
        sample_count = row.get("sample_count", 0)
        features_avail = row.get("features_available", {})

        col1, col2, col3, col4, col5, col6 = st.columns([0.08, 0.15, 0.25, 0.12, 0.25, 0.15])

        with col1:
            key_sel = f"sel_{idx}_{scn_path}"
            is_selected = scn_path in st.session_state[SK_SCN_SELECTIONS]
            new_selection = st.checkbox("Select", value=is_selected, key=key_sel, label_visibility="collapsed")
            if new_selection != is_selected:
                if new_selection:
                    st.session_state[SK_SCN_SELECTIONS].add(scn_path)
                else:
                    st.session_state[SK_SCN_SELECTIONS].discard(scn_path)

        with col2:
            st.write(f"**{number_str}**" if number_str else "—")
            st.caption(f"{sample_count} samples")

        with col3:
            key_lbl = f"lbl_{idx}_{scn_path}"
            new_label = st.text_input("Group label", value=label_val, key=key_lbl, label_visibility="collapsed", placeholder="labels…")
            if new_label != label_val:
                to_save = new_label.strip() if new_label.strip() else None
                if self.scenario_manager.write_label(scn_path, to_save):
                    self.scenario_manager.clear_cache()
                    st.session_state[SK_SAVED_LABELS][scn_path] = new_label.strip()
                    st.rerun()

        with col4:
            features_str = self.scenario_manager.format_features_status(features_avail)
            st.write(features_str)
            st.caption("S=Spectrum, M=MFCC, A=Audio" if features_str != "—" else "")

        with col5:
            if description_val:
                disp = description_val if len(description_val) <= 50 else description_val[:47] + "…"
                st.write(disp)
                st.caption(f"Full: {description_val}" if len(description_val) > 50 else "")
            else:
                st.caption("No description")

        with col6:
            a1, a2 = st.columns([1, 1])
            with a1:
                if st.button("⚙️", key=f"menu_{idx}_{scn_path}"):
                    action = st.radio(
                        f"Actions for {number_str}",
                        ["Add to group", "Select", "Deselect"],
                        key=f"action_{idx}_{scn_path}",
                        horizontal=True,
                    )
                    if action == "Select":
                        st.session_state[SK_SCN_SELECTIONS].add(scn_path)
                    elif action == "Deselect":
                        st.session_state[SK_SCN_SELECTIONS].discard(scn_path)
            with a2:
                if st.button("🔍", key=f"exp_{idx}_{scn_path}"):
                    st.session_state[SK_SCN_EXPLORE] = scn_path
                    st.rerun()

    def _render_scenario_summary(self, df, dfv):
        st.markdown("---")
        unique_labels = self.scenario_manager.get_unique_labels(df)
        st.markdown(
            f"""
        **Summary:**
        - Total scenarios: {len(df)}
        - Filtered scenarios: {len(dfv)}
        - Selected scenarios: {len(st.session_state[SK_SCN_SELECTIONS])}
        - Unique labels: {len(unique_labels)} ({', '.join(sorted(unique_labels)) if unique_labels else 'none'})
        """
        )

    def _render_scenario_explorer(self):
        exp_path = st.session_state.get(SK_SCN_EXPLORE)
        if exp_path and os.path.exists(exp_path):
            st.markdown("---")
            st.subheader(f"📁 Exploring: {os.path.basename(exp_path)}")
            col1, col2 = st.columns([2, 1])
            with col1:
                wavs = self.scenario_manager.list_wavs(exp_path)
                if wavs:
                    st.markdown(f"**Audio files ({len(wavs)} found):**")
                    prev = wavs[:5] if len(wavs) > 5 else wavs
                    sel = st.selectbox("Select audio file", prev, format_func=lambda x: os.path.basename(x), label_visibility="collapsed")
                    if sel:
                        st.audio(sel, format="audio/wav")
                    if len(wavs) > 5:
                        st.caption(f"Showing first 5 of {len(wavs)} audio files")
                else:
                    st.info("No audio files found")
            with col2:
                feat = self.scenario_manager.check_features_available(exp_path)
                st.markdown("**Features:**")
                for k, v in feat.items():
                    st.write(("✅" if v else "❌") + f" {k.upper()}")
                feat_path = os.path.join(exp_path, "features.csv")
                if os.path.exists(feat_path):
                    try:
                        import pandas as pd
                        d = pd.read_csv(feat_path)
                        with st.expander("MFCC Features Preview"):
                            st.dataframe(d.head(10), use_container_width=True)
                            st.caption(f"Shape: {d.shape}")
                    except Exception as e:
                        st.error(f"Error reading features.csv: {e}")
                if st.button("❌ Close Explorer"):
                    st.session_state[SK_SCN_EXPLORE] = None
                    st.rerun()


def main():
    app = RoomResponseGUI()
    app.run()


if __name__ == "__main__":
    main()
