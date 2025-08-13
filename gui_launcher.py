# gui_launcher.py
# Streamlined, single-source-of-truth Scenarios block + sidebar workflow panels

import io
import os
import re
import json
import itertools
import contextlib
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

# --- Your modules (must exist) ---
from ScenarioSelector import ScenarioSelector
from ScenarioClassifier import ScenarioClassifier    # cleaned version with save/load + inference
from FeatureExtractor import AudioFeatureExtractor   # uses recorderConfig.json sample_rate, max freq etc.

sns.set_context("talk")

# ---------------------------- App config ----------------------------
st.set_page_config(page_title="Room Response — ML Studio", layout="wide")
APP_STATE_KEYS = [
    "analyzed", "selector", "scenarios", "df_all", "df_view",
    "selected_map", "labels_map",
    "single_last", "all_last", "group_last",
    "classification_artifacts",
    "panels"
]
for k in APP_STATE_KEYS:
    st.session_state.setdefault(k, None)

# ---------------------------- Cached dataset analysis ----------------------------
@st.cache_data(show_spinner=True)
def analyze_dataset_cached(dataset_path: str):
    """
    Returns:
      selector        - ScenarioSelector with valid scenarios
      scenarios_all   - dict of ALL scenarios (valid + incomplete)
      df_all          - compact DataFrame used by UI list
    """
    selector = ScenarioSelector(dataset_path)
    valid = selector.analyze_dataset() or {}  # only scenarios with features

    def parse_dir_name(name: str):
        m = re.match(r'^(.+?)-Scenario(.+?)-(.+)$', name)
        if m:
            return m.group(1).strip(), m.group(2).strip(), m.group(3).strip(), None
        m = re.match(r'^(.+?)-S(.+?)-(.+)$', name)
        if m:
            return m.group(1).strip(), m.group(2).strip(), m.group(3).strip(), None
        return None, None, None, name

    def load_meta_info(scen_dir: Path):
        meta_paths = [
            scen_dir / "metadata" / "session_metadata.json",
            scen_dir / "session_metadata.json",
        ]
        info = {}
        for p in meta_paths:
            if p.is_file():
                try:
                    md = json.load(open(p, "r", encoding="utf-8"))
                    si = md.get("scenario_info", {})
                    info["computer_name"] = si.get("computer_name")
                    info["room_name"] = si.get("room_name")
                    info["scenario_number"] = si.get("scenario_number")
                    info["description"] = si.get("description")
                    return info
                except Exception:
                    pass
        return info

    def has_features_files(scen_dir: Path):
        return (scen_dir / "spectrum.csv").is_file() or (scen_dir / "features.csv").is_file()

    def wav_count(scen_dir: Path):
        wav_dir = scen_dir / "impulse_responses"
        if not wav_dir.is_dir():
            return 0
        return sum(1 for f in os.listdir(wav_dir) if f.lower().endswith(".wav"))

    dataset_path_p = Path(dataset_path)
    scenarios_all: Dict[str, Dict] = {}
    if dataset_path_p.is_dir():
        for item in sorted(os.listdir(dataset_path)):
            scen_dir = dataset_path_p / item
            if not scen_dir.is_dir():
                continue
            if not re.search(r"scenario", item, re.IGNORECASE):
                continue

            if item in valid:
                scenarios_all[item] = valid[item].copy()
                scenarios_all[item]["full_path"] = str(scen_dir)
                # keep whatever selector put in there
                continue

            comp, scen_no, room, desc = parse_dir_name(item)
            meta = load_meta_info(scen_dir)
            comp = meta.get("computer_name") or comp
            scen_no = meta.get("scenario_number") or scen_no
            room = meta.get("room_name") or room
            desc = meta.get("description") or desc

            spectrum_exists = (scen_dir / "spectrum.csv").is_file()
            mfcc_exists = (scen_dir / "features.csv").is_file()

            sample_count = 0
            try:
                if spectrum_exists:
                    sample_count = max(sample_count, len(pd.read_csv(scen_dir / "spectrum.csv")))
                if mfcc_exists:
                    sample_count = max(sample_count, len(pd.read_csv(scen_dir / "features.csv")))
            except Exception:
                pass
            if sample_count == 0:
                sample_count = wav_count(scen_dir)

            scenarios_all[item] = {
                'directory_name': item,
                'full_path': str(scen_dir),
                'computer_name': comp,
                'room_name': room,
                'scenario_number': scen_no,
                'description': desc,
                'measurement_date': None,
                'measurement_time': None,
                'sample_count': sample_count,
                'features_available': {
                    'spectrum': spectrum_exists,
                    'mfcc': mfcc_exists,
                    'audio': wav_count(scen_dir) > 0
                },
                'feature_details': {},
                'metadata_available': bool(meta),
                'metadata_info': meta,
                'file_timestamps': {},
                'validity': {
                    'has_features': has_features_files(scen_dir),
                    'has_samples': sample_count > 0,
                    'parseable_name': comp is not None and scen_no is not None and room is not None,
                }
            }

    rows = []
    for key, s in scenarios_all.items():
        rows.append({
            "key": key,
            "scenario_number": s.get("scenario_number"),
            "description": s.get("description") or "",
            "computer": s.get("computer_name") or "",
            "room": s.get("room_name") or "",
            "samples": s.get("sample_count", 0),
            "spectrum": s.get("features_available", {}).get("spectrum", False),
            "mfcc": s.get("features_available", {}).get("mfcc", False),
            "full_path": s.get("full_path") or "",
        })
    df = pd.DataFrame(rows)
    return selector, scenarios_all, df


# ---------------------------- df_view single source of truth ----------------------------
import fnmatch

def _apply_name_patterns(series: pd.Series, pattern_text: str) -> pd.Series:
    if not isinstance(pattern_text, str) or not pattern_text.strip():
        return pd.Series([True] * len(series), index=series.index)
    parts = [p.strip() for p in re.split(r"[,\s]+", pattern_text) if p.strip()]
    mask = pd.Series([False] * len(series), index=series.index)
    s = series.astype(str).fillna("")
    for pat in parts:
        mask |= s.apply(lambda x: fnmatch.fnmatch(x, pat))
    return mask

def build_df_view(df_all: pd.DataFrame, *, computer: str = "All", room: str = "All", name_pattern: str = "") -> pd.DataFrame:
    if df_all is None or df_all.empty:
        return pd.DataFrame(columns=["key", "scenario_number", "description", "computer", "room", "samples", "spectrum", "mfcc", "full_path"])
    out = df_all.copy()
    if computer and computer != "All":
        out = out[out["computer"] == computer]
    if room and room != "All":
        out = out[out["room"] == room]
    name_col = "scenario_number" if "scenario_number" in out.columns else "key"
    mask = _apply_name_patterns(out[name_col], name_pattern or "")
    out = out[mask]
    if "scenario_number" in out.columns:
        out = out.sort_values(by=["computer", "room", "scenario_number"], na_position="last")
    else:
        out = out.sort_values(by=["computer", "room"], na_position="last")
    return out

def set_df_view_in_session(df_all: pd.DataFrame, computer: str, room: str, name_pattern: str):
    st.session_state["df_view"] = build_df_view(df_all, computer=computer, room=room, name_pattern=name_pattern)

def get_df_view(df_all: pd.DataFrame) -> pd.DataFrame:
    return st.session_state.get("df_view", df_all.copy() if df_all is not None else pd.DataFrame())


# ---------------------------- Helpers: selection & labels ----------------------------
def ensure_state_maps():
    if st.session_state.get("selected_map") is None:
        st.session_state["selected_map"] = {}  # key -> bool
    if st.session_state.get("labels_map") is None:
        st.session_state["labels_map"] = {}    # key -> comma-separated labels

def selected_keys_from_editor(edited_df: pd.DataFrame) -> List[str]:
    if "select" not in edited_df.columns:
        return []
    picked = edited_df[edited_df["select"] == True]
    return picked["key"].tolist() if "key" in picked.columns else []

def keys_with_label(label: str) -> List[str]:
    if not label:
        return []
    out = []
    for k, labstr in st.session_state["labels_map"].items():
        labs = [s.strip() for s in str(labstr).split(",") if s.strip()]
        if label in labs:
            out.append(k)
    return out


# ---------------------------- Sidebar panels ----------------------------
with st.sidebar:
    st.header("Dataset")
    dataset_path = st.text_input("Dataset directory", value="room_response_dataset", key="dataset_path")
    analyze_click = st.button("Analyze dataset", type="primary")

    st.markdown("---")
    # Panel toggles (keep persistent in session)
    if st.session_state.get("panels") is None:
        st.session_state["panels"] = {
            "collect": False,
            "process": False,
            "classify": False,
            "predict": False,
            "visualize": False,
        }

    # Buttons to toggle panels
    cols = st.columns(1)
    with cols[0]:
        if st.button("Collect"):
            st.session_state["panels"]["collect"] = not st.session_state["panels"]["collect"]
        if st.button("Process"):
            st.session_state["panels"]["process"] = not st.session_state["panels"]["process"]
        if st.button("Classify"):
            st.session_state["panels"]["classify"] = not st.session_state["panels"]["classify"]
        # Predict / Visualize are enabled only after classification
        disabled_after = st.session_state.get("classification_artifacts") is None
        st.button("Predict", disabled=disabled_after, key="btn_pred_toggle",
                  on_click=lambda: st.session_state["panels"].update(predict=not st.session_state["panels"]["predict"]))
        st.button("Visualize", disabled=disabled_after, key="btn_viz_toggle",
                  on_click=lambda: st.session_state["panels"].update(visualize=not st.session_state["panels"]["visualize"]))

# Initial (re)analysis
if analyze_click or not st.session_state.get("analyzed", False):
    selector, scenarios_all, df_all = analyze_dataset_cached(dataset_path)
    st.session_state.update({
        "selector": selector,
        "scenarios": scenarios_all,   # ALL scenarios (valid + incomplete)
        "df_all": df_all,
        "analyzed": True,
        "single_last": None,
        "all_last": None,
        "group_last": None,
        "classification_artifacts": None
    })
else:
    selector = st.session_state.get("selector")

scenarios = st.session_state.get("scenarios", {})
df_all = st.session_state.get("df_all", pd.DataFrame())
if df_all is None:
    df_all = pd.DataFrame()

if df_all.empty or not scenarios:
    st.warning("No scenarios found in this dataset directory.")
    st.stop()

ensure_state_maps()

# ---------------------------- Top metrics & filters ----------------------------
st.title("Room Response — ML Studio")

# Metrics
total_samples = sum(scenarios[k].get("sample_count", 0) for k in scenarios)
spec_count = sum(1 for s in scenarios.values() if s.get("features_available", {}).get("spectrum", False))
mfcc_count = sum(1 for s in scenarios.values() if s.get("features_available", {}).get("mfcc", False))
m1, m2, m3, m4 = st.columns(4)
m1.metric("Scenarios", f"{len(scenarios)}")
m2.metric("Samples", f"{total_samples:,}")
m3.metric("With Spectrum", f"{spec_count}")
m4.metric("With MFCC", f"{mfcc_count}")

st.divider()

# Filters (drive df_view building once)
computers = ["All"] + sorted([c for c in df_all["computer"].unique() if c])
rooms = ["All"] + sorted([r for r in df_all["room"].unique() if r])
fc1, fc2, fc3 = st.columns([1, 1, 1])
with fc1:
    computer_filter = st.selectbox("Filter by Computer", options=computers, index=0, key="flt_computer")
with fc2:
    room_filter = st.selectbox("Filter by Room", options=rooms, index=0, key="flt_room")
with fc3:
    name_pattern = st.text_input("Scenario # pattern (glob)", value="", placeholder="e.g. 4.* or 6.2*", key="flt_namepat")

set_df_view_in_session(df_all, computer_filter, room_filter, name_pattern)
df_view = get_df_view(df_all)

# ---------------------------- Scenarios block (single place) ----------------------------
st.subheader("Scenarios")

# Build editor dataframe with Select, Scenario #, Group label, Explore
# Keep selection/labels from session if present
select_vals = []
label_vals = []
keys = df_view["key"].tolist()
for k in keys:
    select_vals.append(bool(st.session_state["selected_map"].get(k, False)))
    label_vals.append(st.session_state["labels_map"].get(k, ""))

df_table = pd.DataFrame({
    "select": select_vals,
    "scenario_number": df_view["scenario_number"].tolist(),
    "labels": label_vals,
    "explore": ["🔎 Explore"] * len(keys),
    "key": keys,  # hidden/internal
})

column_config = {
    "select": st.column_config.CheckboxColumn("Select"),
    "scenario_number": st.column_config.TextColumn("Scenario #"),
    "labels": st.column_config.TextColumn("Group label(s)", help="Comma-separated labels, unlimited"),
    "explore": st.column_config.TextColumn("Explore", disabled=True),
    "key": st.column_config.TextColumn("key", disabled=True, width="small"),
}

edited = st.data_editor(
    df_table[["select", "scenario_number", "labels", "explore", "key"]],
    use_container_width=True,
    hide_index=True,
    column_config=column_config,
    disabled=["scenario_number", "explore", "key"],
    num_rows="fixed",
    key="scenarios_editor"
)

# Update session maps from editor
for _, row in edited.iterrows():
    k = row["key"]
    st.session_state["selected_map"][k] = bool(row["select"])
    st.session_state["labels_map"][k] = str(row["labels"] or "").strip()

# Row actions
c1, c2, c3 = st.columns([1, 1, 2])
with c1:
    bulk_label = st.text_input("Assign label to FILTERED", value="", placeholder="e.g. A", key="bulk_label")
with c2:
    if st.button("Apply to filtered"):
        if bulk_label.strip():
            for k in keys:
                cur = st.session_state["labels_map"].get(k, "")
                labs = [s.strip() for s in cur.split(",") if s.strip()]
                if bulk_label not in labs:
                    labs.append(bulk_label)
                st.session_state["labels_map"][k] = ",".join(sorted(set(labs)))
            st.success(f"Assigned label '{bulk_label}' to {len(keys)} scenario(s).")
            st.rerun()
with c3:
    # Explore currently highlighted (first selected if any; else prompt)
    explore_candidates = [k for k in keys if st.session_state["selected_map"].get(k)]
    sel_for_explore = explore_candidates[0] if explore_candidates else (keys[0] if keys else None)
    if st.button("Explore selected (opens explorer)"):
        if sel_for_explore:
            # Pass selection via query param; switch to separate explorer if available
            st.experimental_set_query_params(scenario=sel_for_explore, dataset=str(dataset_path))
            try:
                st.switch_page("scenario_explorer.py")
            except Exception:
                st.info("scenario_explorer.py not found. Create it to enable detailed exploration.")
        else:
            st.warning("No scenario selected to explore.")

st.caption(f"Selected: {sum(1 for v in st.session_state['selected_map'].values() if v)} scenario(s)")

st.divider()

# ---------------------------- PROCESS (Feature Extraction) panel ----------------------------
if st.session_state["panels"]["process"]:
    st.header("Process — Feature Extraction")
    pc1, pc2, pc3 = st.columns([1.2, 1, 1])
    with pc1:
        wav_subfolder = st.text_input("WAV subfolder", value="impulse_responses")
    with pc2:
        recording_type = st.selectbox("Recording type", options=["any", "average", "raw"], index=0)
    with pc3:
        max_freq = st.number_input("Max spectrum freq (Hz)", min_value=0, value=0, step=1000, help="0 = no limit")

    qc1, qc2, qc3 = st.columns([1.2, 1, 1])
    with qc1:
        mfcc_filename = st.text_input("MFCC filename", value="features.csv")
    with qc2:
        spectrum_filename = st.text_input("Spectrum filename", value="spectrum.csv")
    with qc3:
        exist_mode = st.radio(
            "If features already exist",
            options=["Skip scenario (both files present)", "Keep existing (write missing only)", "Overwrite both files"],
            horizontal=False,
        )

    def run_extraction(keys_arg: List[str]):
        if not keys_arg:
            st.warning("No scenarios selected.")
            return
        overwrite_flag = (exist_mode == "Overwrite both files")
        skip_if_both_exist = (exist_mode == "Skip scenario (both files present)")

        extractor = AudioFeatureExtractor(
            sample_rate=16000,
            n_mfcc=13,
            config_filename="recorderConfig.json",
            max_spectrum_freq=(None if max_freq <= 0 else float(max_freq)),
        )

        rows = []
        prog = st.progress(0)
        for i, k in enumerate(keys_arg, start=1):
            s = scenarios[k]
            folder = s.get("full_path") or str(Path(dataset_path) / k)

            if skip_if_both_exist:
                if (Path(folder) / mfcc_filename).exists() and (Path(folder) / spectrum_filename).exists():
                    rows.append({"scenario": k, "status": "skipped (both files present)"})
                    prog.progress(int(i / len(keys_arg) * 100))
                    continue

            try:
                ok = extractor.process_scenario_folder(
                    scenario_folder=folder,
                    wav_subfolder=wav_subfolder,
                    recording_type=recording_type,
                    mfcc_filename=mfcc_filename,
                    spectrum_filename=spectrum_filename,
                    dataset_path_for_config=dataset_path,
                    overwrite_existing_files=overwrite_flag
                )
                rows.append({"scenario": k, "status": "ok" if ok else "failed"})
            except Exception as e:
                rows.append({"scenario": k, "status": f"error: {e}"})

            prog.progress(int(i / len(keys_arg) * 100))

        st.success("Feature extraction finished.")
        st.dataframe(pd.DataFrame(rows), use_container_width=True)

        st.info("Re-analyze the dataset to refresh the list.")
        if st.button("Re-analyze now"):
            analyze_dataset_cached.clear()
            st.session_state["analyzed"] = False
            st.rerun()

    c1, c2 = st.columns([1, 1])
    with c1:
        sel_keys = [k for k, v in st.session_state["selected_map"].items() if v]
        st.button("Extract for SELECTED", disabled=(len(sel_keys) == 0), on_click=lambda: run_extraction(sel_keys))
    with c2:
        st.button("Extract for ALL", on_click=lambda: run_extraction(list(scenarios.keys())))

    st.divider()

# ---------------------------- CLASSIFY panel ----------------------------
if st.session_state["panels"]["classify"]:
    st.header("Classify")

    feature_type = st.selectbox("Feature type", options=["spectrum", "mfcc"], index=0, key="clf_feature_type")
    mode = st.radio("Run mode", options=["Single pair", "All pairs", "Groups"], horizontal=True, key="clf_mode")

    # Model & CV params
    pc1, pc2, pc3 = st.columns([1, 1, 1])
    with pc1:
        model_type = st.selectbox("Model", options=["svm", "logistic"], index=0, key="clf_model")
    with pc2:
        test_size = st.slider("Test size (holdout)", 0.05, 0.5, 0.3, step=0.05, key="clf_testsize")
    with pc3:
        cv_folds = st.slider("CV folds", 2, 10, 5, step=1, key="clf_cvf")

    def missing_feature_scenarios(keys_: List[str], feature_type_: str) -> List[str]:
        missing = []
        for k in keys_:
            s = scenarios.get(k, {})
            if not s.get("features_available", {}).get(feature_type_, False):
                name = s.get("description") or f"S{s.get('scenario_number')}" or k
                missing.append(name)
        return missing

    # ---------- Single pair ----------
    if mode.startswith("Single"):
        sel = [k for k, v in st.session_state["selected_map"].items() if v]
        can_run = len(sel) == 2 and not missing_feature_scenarios(sel, feature_type)

        if st.button("Run classification for selected pair", disabled=not can_run):
            k1, k2 = sel
            s1, s2 = scenarios[k1], scenarios[k2]
            lbl1 = s1.get("description") or f"S{s1.get('scenario_number')}" or k1
            lbl2 = s2.get("description") or f"S{s2.get('scenario_number')}" or k2
            if lbl1 == lbl2:
                lbl1, lbl2 = lbl1 + f"-{k1}", lbl2 + f"-{k2}"

            clf = ScenarioClassifier(model_type=model_type, feature_type=feature_type)

            p1 = s1["full_path"] or str(Path(dataset_path) / k1)
            p2 = s2["full_path"] or str(Path(dataset_path) / k2)

            try:
                X, y, feature_names, label_names = clf.prepare_dataset(p1, p2, lbl1, lbl2)
            except FileNotFoundError as e:
                st.error(str(e))
                st.stop()

            with st.spinner("Training & evaluating..."):
                results = clf.train_and_evaluate(X, y, test_size=test_size, cv_folds=cv_folds)
                buf = io.StringIO()
                with contextlib.redirect_stdout(buf):
                    clf.print_results(results)
                report_text = buf.getvalue()

            st.session_state["single_last"] = {
                "pair": (k1, k2),
                "labels": (lbl1, lbl2),
                "clf": clf,
                "results": results,
                "report_text": report_text,
            }
            # Save model blob for Predict/Visualize enablement
            st.session_state["classification_artifacts"] = {
                "model_bytes": clf.dumps_model_bytes(extra_meta={
                    "feature_type": feature_type,
                    "label_names": label_names,
                    "feature_names": feature_names
                }),
                "feature_type": feature_type,
                "label_names": label_names,
                "feature_names": feature_names
            }
            st.success("Classification complete. Predict / Visualize panels are now enabled.")

    # ---------- All pairs ----------
    elif mode.startswith("All"):
        sel = [k for k, v in st.session_state["selected_map"].items() if v]
        can_run_all = len(sel) >= 2 and not missing_feature_scenarios(sel, feature_type)

        if st.button("Run classification for ALL selected pairs", disabled=not can_run_all):
            pairs = list(itertools.combinations(sel, 2))
            prog = st.progress(0)
            run_results = []
            acc_map = {}  # (k1,k2) -> test acc

            for i, (k1, k2) in enumerate(pairs, start=1):
                s1, s2 = scenarios[k1], scenarios[k2]
                lbl1 = s1.get("description") or f"S{s1.get('scenario_number')}" or k1
                lbl2 = s2.get("description") or f"S{s2.get('scenario_number')}" or k2
                if lbl1 == lbl2:
                    lbl1, lbl2 = lbl1 + f"-{k1}", lbl2 + f"-{k2}"

                clf = ScenarioClassifier(model_type=model_type, feature_type=feature_type)
                p1 = s1["full_path"] or str(Path(dataset_path) / k1)
                p2 = s2["full_path"] or str(Path(dataset_path) / k2)

                try:
                    X, y, feature_names, label_names = clf.prepare_dataset(p1, p2, lbl1, lbl2)
                    results = clf.train_and_evaluate(X, y, test_size=test_size, cv_folds=cv_folds)
                    buf = io.StringIO()
                    with contextlib.redirect_stdout(buf):
                        clf.print_results(results)
                    run_results.append({
                        "pair": (k1, k2),
                        "labels": (lbl1, lbl2),
                        "clf": clf,
                        "results": results,
                        "report_text": buf.getvalue()
                    })
                    acc_map[(k1, k2)] = results["test_accuracy"]
                except Exception as e:
                    run_results.append({"pair": (k1, k2), "labels": (lbl1, lbl2), "error": str(e)})

                prog.progress(int(i / len(pairs) * 100))

            st.session_state["all_last"] = run_results

            # Accuracy matrix visualization
            st.subheader("All-Pairs Accuracy Matrix")
            idx_labels = [scenarios[k].get("scenario_number") or k for k in sel]
            mat = np.full((len(sel), len(sel)), np.nan, dtype=float)
            key_index = {k: i for i, k in enumerate(sel)}
            for (a, b), acc in acc_map.items():
                i, j = key_index[a], key_index[b]
                mat[i, j] = mat[j, i] = acc
            fig, ax = plt.subplots(figsize=(0.6*len(sel)+2, 0.6*len(sel)+2))
            sns.heatmap(mat, annot=True, fmt=".2f", cmap="viridis",
                        xticklabels=idx_labels, yticklabels=idx_labels, ax=ax)
            ax.set_title("Test Accuracy")
            fig.tight_layout()
            st.pyplot(fig)

    # ---------- Groups ----------
        
    else:
            st.subheader("Group vs Group")

            gc1, gc2 = st.columns([1, 1])
            with gc1:
                label_A = st.text_input("Group A label", value="A")
            with gc2:
                label_B = st.text_input("Group B label", value="B")

            groupA_keys = keys_with_label(label_A)
            groupB_keys = keys_with_label(label_B)

            st.caption(f"A: {len(groupA_keys)} scenarios | B: {len(groupB_keys)} scenarios")

            missing_groups = missing_feature_scenarios(groupA_keys + groupB_keys, feature_type)
            if missing_groups:
                st.error(
                    "Some assigned scenarios do not have the required features "
                    f"for **{feature_type}**:\n\n- " + "\n- ".join(missing_groups) +
                    "\n\nExtract features or adjust assignments."
                )

            can_run_groups = (len(groupA_keys) > 0 and len(groupB_keys) > 0 and not missing_groups)

            # ---- FIXED: build dataset without calling classifier for loading ----
            def prepare_group_dataset(groupA: List[str], groupB: List[str],
                                      scenarios: Dict[str, Dict], dataset_path: str,
                                      feature_type_: str, balance: bool = True):
                """
                Load per-scenario feature matrices from the saved CSVs:
                  - spectrum.csv -> columns starting with 'freq_' or 'bin_' (fallback: numeric)
                  - features.csv -> columns starting with 'mfcc' (fallback: numeric)
                Returns X, y, feature_names.
                """
                def load_features_from_folder_local(folder: str, feature_type_local: str):
                    folder_p = Path(folder)
                    csv_name = "spectrum.csv" if feature_type_local == "spectrum" else "features.csv"
                    csv_path = folder_p / csv_name
                    if not csv_path.exists():
                        raise FileNotFoundError(f"Missing {csv_name} in {folder}")

                    df = pd.read_csv(csv_path)

                    if feature_type_local == "spectrum":
                        feat_cols = [c for c in df.columns
                                    if str(c).startswith("freq_") or str(c).startswith("bin_") or re.fullmatch(r"\d+", str(c))]
                    else:
                        # MFCC
                        feat_cols = [c for c in df.columns if str(c).lower().startswith("mfcc")]

                    # Fallback: all numeric columns except a typical 'filename'
                    if not feat_cols:
                        feat_cols = [c for c in df.columns
                                     if c != "filename" and pd.api.types.is_numeric_dtype(df[c])]

                    if not feat_cols:
                        raise ValueError(f"No feature columns found in {csv_path.name}")

                    Xmat = df[feat_cols].to_numpy(dtype=float)
                    return Xmat, feat_cols

                X_blocks = []
                y_blocks = []
                feat_names_ref = None

                # Group A -> label 0
                for k in groupA:
                    folder = scenarios[k].get("full_path") or str(Path(dataset_path) / k)
                    Xa, fn = load_features_from_folder_local(folder, feature_type_)
                    if feat_names_ref is None:
                        feat_names_ref = fn
                    X_blocks.append(Xa)
                    y_blocks.append(np.zeros(Xa.shape[0], dtype=int))

                # Group B -> label 1
                for k in groupB:
                    folder = scenarios[k].get("full_path") or str(Path(dataset_path) / k)
                    Xb, fn = load_features_from_folder_local(folder, feature_type_)
                    if feat_names_ref is None:
                        feat_names_ref = fn
                    X_blocks.append(Xb)
                    y_blocks.append(np.ones(Xb.shape[0], dtype=int))

                Xc = np.vstack(X_blocks)
                yc = np.hstack(y_blocks)

                if balance:
                    # simple downsample to min class size
                    n0 = (yc == 0).sum()
                    n1 = (yc == 1).sum()
                    nmin = min(n0, n1)
                    idx0 = np.where(yc == 0)[0][:nmin]
                    idx1 = np.where(yc == 1)[0][:nmin]
                    idx = np.hstack([idx0, idx1])
                    Xc, yc = Xc[idx], yc[idx]

                return Xc, yc, feat_names_ref
            # ---- END FIX ----

            if st.button("Run classification for GROUPS", type="primary", disabled=not can_run_groups):
                try:
                    X, y, feat_names = prepare_group_dataset(
                        groupA_keys, groupB_keys, scenarios, dataset_path, feature_type, balance=True
                    )
                except Exception as e:
                    st.error(f"Failed to build dataset: {e}")
                    st.stop()

                clf = ScenarioClassifier(model_type=model_type, feature_type=feature_type)
                with st.spinner("Training & evaluating (group classifier)..."):
                    results = clf.train_and_evaluate(X, y, test_size=test_size, cv_folds=cv_folds)
                    buf = io.StringIO()
                    with contextlib.redirect_stdout(buf):
                        clf.print_results(results)
                    report_text = buf.getvalue()

                st.session_state["group_last"] = {
                    "groups": (groupA_keys, groupB_keys),
                    "labels": (label_A, label_B),
                    "clf": clf,
                    "results": results,
                    "feature_names": feat_names,
                    "feature_type": feature_type
                }
                st.session_state["classification_artifacts"] = {
                    "model_bytes": clf.dumps_model_bytes(extra_meta={
                        "feature_type": feature_type,
                        "label_names": [label_A, label_B],
                        "feature_names": feat_names
                    }),
                    "feature_type": feature_type,
                    "label_names": [label_A, label_B],
                    "feature_names": feat_names
                }
                st.success("Group classification complete. Predict / Visualize panels are now enabled.")

    st.divider()

# ---------------------------- PREDICT panel ----------------------------
if st.session_state["panels"]["predict"]:
    st.header("Predict — Single Sample")
    art = st.session_state.get("classification_artifacts")
    if not art:
        st.info("Train a classifier first.")
    else:
        clf = ScenarioClassifier.load_from_bytes(art["model_bytes"])
        st.write(f"Loaded model: **{clf.model_type}** on **{art['feature_type']}** features")

        # Let user pick a folder to record or pick a WAV; to keep self-contained, allow file upload:
        uploaded = st.file_uploader("Upload a single WAV to classify", type=["wav"])
        if uploaded is not None:
            # Save temp WAV and extract features via FeatureExtractor to a temp CSV-like vector
            tmp_dir = Path(".streamlit_tmp")
            tmp_dir.mkdir(exist_ok=True)
            wav_path = tmp_dir / f"temp_{datetime.now().strftime('%H%M%S%f')}.wav"
            with open(wav_path, "wb") as f:
                f.write(uploaded.getbuffer())

            # Use FeatureExtractor to compute feature vector in-memory
            extractor = AudioFeatureExtractor(config_filename="recorderConfig.json")
            feat_vec, feat_names = extractor.extract_single_sample_features(str(wav_path), feature_type=art["feature_type"])
            if feat_vec is None:
                st.error("Failed to extract features from uploaded file.")
            else:
                pred, proba = clf.predict_single(feat_vec)
                st.success(f"Predicted: **{art['label_names'][pred]}**  |  Confidence: {np.max(proba):.3f}")

# ---------------------------- VISUALIZE panel ----------------------------
if st.session_state["panels"]["visualize"]:
    st.header("Visualize — Last Results")
    # Show text reports compactly
    if st.session_state.get("single_last"):
        blk = st.session_state["single_last"]
        st.subheader(f"Single Pair: {blk['pair'][0]} vs {blk['pair'][1]}")
        st.code(blk["report_text"] or "(no text)", language="text")

    if st.session_state.get("all_last"):
        runs = st.session_state["all_last"]
        rows = []
        for r in runs:
            (k1, k2) = r["pair"]
            row = {"pair": f"{k1} vs {k2}"}
            if "results" in r:
                row.update({
                    "train_acc": r["results"]["train_accuracy"],
                    "test_acc": r["results"]["test_accuracy"],
                    "cv_mean": r["results"]["cv_mean"],
                    "cv_std": r["results"]["cv_std"],
                })
            else:
                row.update({"train_acc": None, "test_acc": None, "cv_mean": None, "cv_std": None})
            row["status"] = "ok" if "results" in r else ("error: " + r.get("error",""))
            rows.append(row)
        with st.expander("All Pairs Overview", expanded=True):
            st.dataframe(pd.DataFrame(rows), use_container_width=True)

    if st.session_state.get("group_last"):
        blk = st.session_state["group_last"]
        st.subheader(f"Groups: {blk['labels'][0]} vs {blk['labels'][1]}")
        # no heavy plots here; classifier report already saved in text
        st.text(f"Holdout Acc: {blk['results']['test_accuracy']:.3f} | CV: {blk['results']['cv_mean']:.3f} ± {blk['results']['cv_std']:.3f}")

# ---------------------------- COLLECT panel (optional light hooks) ----------------------------
if st.session_state["panels"]["collect"]:
    st.header("Collect — Series")
    st.caption("Run your series collection from the GUI. (Requires collect_dataset.py with collect_series function.)")
    try:
        from collect_dataset import collect_series
        ok_collect = True
    except Exception:
        ok_collect = False
        st.info("collect_dataset.py with function collect_series(...) not found; panel shown for completeness.")

    c1, c2 = st.columns([2, 1])
    with c1:
        scenario_numbers_str = st.text_input("Scenario numbers (comma/space separated, glob ok but will be literal here)",
                                             value="1 2 3")
    with c2:
        start_delay = st.number_input("Start delay (s)", min_value=0, value=60, step=10)
    c3, c4 = st.columns([1, 1])
    with c3:
        between_delay = st.number_input("Delay between scenarios (s)", min_value=0, value=60, step=10)
    with c4:
        interactive = st.checkbox("Interactive device selection", value=False)

    if st.button("Start series collection", disabled=not ok_collect):
        nums = [s.strip() for s in re.split(r"[,\s]+", scenario_numbers_str) if s.strip()]
        try:
            collect_series(
                scenario_numbers=nums,
                output_dir=dataset_path,
                config_file="recorderConfig.json",
                interactive=interactive,
                quiet=True,
                start_delay_sec=int(start_delay),
                between_delay_sec=int(between_delay),
                beeper="sdl"
            )
            st.success("Series collection finished.")
            analyze_dataset_cached.clear()
            st.session_state["analyzed"] = False
        except Exception as e:
            st.error(f"Series collection failed: {e}")
