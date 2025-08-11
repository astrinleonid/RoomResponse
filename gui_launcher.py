# streamlit_app.py
# Streamlined UI + Visualizations + Feature Extraction integration

import io
import os
import re
import json
import itertools
import contextlib
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

from ScenarioSelector import ScenarioSelector
from ScenarioClassifier import ScenarioClassifier
from FeatureExtractor import AudioFeatureExtractor

# ---------------------------- Page/UI setup ----------------------------
st.set_page_config(page_title="Room Response Scenario Selector", layout="wide")
sns.set_context("talk")
st.title("Room Response Scenario Selector")
st.caption("Analyze → filter → select → classify (single pair or all pairs) with visualizations")

# ---------------------------- Sidebar: dataset path ----------------------------
with st.sidebar:
    st.header("Dataset")
    dataset_path = st.text_input("Dataset directory", value="room_response_dataset")
    analyze_click = st.button("Analyze dataset")

# ---------------------------- Cached dataset analysis ----------------------------
@st.cache_data(show_spinner=True)
def analyze_dataset_cached(dataset_path: str):
    """
    Returns:
      selector        - ScenarioSelector (valid scenarios only)
      scenarios_all   - dict of ALL scenario folders (valid + incomplete)
      df_all          - DataFrame for UI table (includes scenarios without features)
    """
    selector = ScenarioSelector(dataset_path)
    valid = selector.analyze_dataset() or {}  # valid = has features & samples

    def parse_dir_name(name: str):
        m = re.match(r'^(.+?)-Scenario(.+?)-(.+)$', name)
        if m:
            return m.group(1).strip(), m.group(2).strip(), m.group(3).strip(), None
        m = re.match(r'^(.+?)-S(.+?)-(.+)$', name)
        if m:
            return m.group(1).strip(), m.group(2).strip(), m.group(3).strip(), None
        return None, None, None, name  # fallback: put whole name in description

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

    # Collect ALL scenario folders (match 'scenario' in name)
    scenarios_all: dict[str, dict] = {}
    dataset_path_p = Path(dataset_path)
    if dataset_path_p.is_dir():
        for item in sorted(os.listdir(dataset_path)):
            scen_dir = dataset_path_p / item
            if not scen_dir.is_dir():
                continue
            if not re.search(r"scenario", item, re.IGNORECASE):
                continue

            # If already valid, reuse the rich entry from selector and ensure full_path
            if item in valid:
                scenarios_all[item] = valid[item].copy()
                scenarios_all[item]["full_path"] = str(scen_dir)
                continue

            # Build minimal entry for incomplete scenarios (so they appear in UI)
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

    # Build UI table
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

# ---------------------------- First-run / analyze logic ----------------------------
if "analyzed" not in st.session_state:
    st.session_state["analyzed"] = False
if "last_dataset_path" not in st.session_state:
    st.session_state["last_dataset_path"] = None

# Auto-analyze on first load, when dataset path changes, or when button clicked
should_analyze = (
    analyze_click
    or not st.session_state["analyzed"]
    or (st.session_state["last_dataset_path"] != dataset_path)
)

if should_analyze:
    selector, scenarios_all, df_all = analyze_dataset_cached(dataset_path)
    st.session_state.update({
        "selector": selector,
        "scenarios": scenarios_all,   # ALL (valid + incomplete)
        "df_all": df_all,
        "analyzed": True,
        "last_dataset_path": dataset_path,
        "single_last": None,
        "all_last": None,
    })

# Local shortcuts after analysis
scenarios = st.session_state.get("scenarios", {})
df_all = st.session_state.get("df_all", pd.DataFrame())

# Guard (safe now—after we tried to analyze)
if not Path(dataset_path).is_dir():
    st.warning(f"Dataset directory not found: {dataset_path}")
    st.stop()
if df_all.empty:
    st.warning("No scenarios found in this dataset directory.")
    st.stop()

# ---------------------------- Helpers ----------------------------
def default_label_for_scenario(s: dict, key: str) -> str:
    for cand in [s.get("description"),
                 f"S{s.get('scenario_number')}" if s.get("scenario_number") not in (None, "") else None,
                 s.get("room_name"), s.get("computer_name"), key]:
        if cand and str(cand).strip():
            return str(cand)
    return key

def ensure_distinct(a: str, b: str, sa: dict, sb: dict, ka: str, kb: str) -> tuple[str, str]:
    if a != b:
        return a, b
    sufa = f"-S{sa.get('scenario_number')}" if sa.get("scenario_number") else f"-{ka}"
    sufb = f"-S{sb.get('scenario_number')}" if sb.get("scenario_number") else f"-{kb}"
    return a + sufa, b + sufb

def selected_keys_from_table(edited_df: pd.DataFrame) -> list[str]:
    if "select" not in edited_df.columns:
        return []
    picked = edited_df[edited_df["select"] == True]
    return picked["key"].tolist() if "key" in picked.columns else picked.index.tolist()

def filter_df(df: pd.DataFrame, computer: str, room: str, feature_filter: str) -> pd.DataFrame:
    out = df.copy()
    if computer != "All":
        out = out[out["computer"] == computer]
    if room != "All":
        out = out[out["room"] == room]
    if feature_filter == "spectrum":
        out = out[out["spectrum"] == True]
    elif feature_filter == "mfcc":
        out = out[out["mfcc"] == True]
    return out.sort_values(by=["computer", "room", "scenario_number"], na_position="last")

def missing_feature_scenarios(keys: list[str], scenarios: dict, feature_type: str) -> list[str]:
    missing = []
    for k in keys:
        s = scenarios.get(k, {})
        if not s.get("features_available", {}).get(feature_type, False):
            name = s.get("description") or f"S{s.get('scenario_number')}" or k
            missing.append(name)
    return missing

def read_features_meta(folder: str) -> dict | None:
    p = Path(folder) / "features_meta.json"
    if p.is_file():
        try:
            return json.load(open(p, "r", encoding="utf-8"))
        except Exception:
            return None
    return None

def _suffix_int(name: str) -> int | None:
    m = re.search(r'(\d+)$', name) or re.search(r'freq[_\-]?(\d+)', name)
    return int(m.group(1)) if m else None

def pair_bin_hz(k1: str, k2: str, scenarios: dict, dataset_path: str, warn: bool = True) -> float:
    vals = []
    for k in (k1, k2):
        folder = scenarios[k].get("full_path") or str(Path(dataset_path) / k)
        meta = read_features_meta(folder)
        if meta:
            if meta.get("bin_hz"):
                vals.append(float(meta["bin_hz"]))
            elif meta.get("sample_rate") and meta.get("fft_len"):
                vals.append(float(meta["sample_rate"]) / float(meta["fft_len"]))
    if len(vals) == 2 and abs(vals[0] - vals[1]) > 1e-6 and warn:
        st.warning(f"Frequency bin resolution differs between the two scenarios: {vals[0]:.3f} vs {vals[1]:.3f} Hz. Using the first.")
    if vals:
        return vals[0]
    return 30.0  # fallback

def humanize_feature_labels(feature_names: List[str], feature_type: str, bin_hz: float) -> List[str]:
    if feature_type != "spectrum":
        return [fn.replace("mfcc_", "MFCC ") if fn.startswith("mfcc_") else fn for fn in feature_names]
    labels = []
    for fn in feature_names:
        k = _suffix_int(fn)
        labels.append(f"{int(round(k * bin_hz))} Hz" if k is not None else fn)
    return labels

# ---------- Visualization builders ----------
def fig_confusion_matrix(cm: np.ndarray, class_names: list[str]) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(5.5, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    fig.tight_layout()
    return fig

def fig_cv_scores(cv_scores: np.ndarray, cv_mean: float) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.bar(range(1, len(cv_scores)+1), cv_scores, alpha=0.8)
    ax.axhline(y=cv_mean, linestyle="--", label=f"Mean: {cv_mean:.3f}")
    ax.set_title("Cross-Validation Accuracy by Fold")
    ax.set_xlabel("Fold")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1)
    ax.legend()
    fig.tight_layout()
    return fig

def compute_feature_importance(clf: ScenarioClassifier, results: dict) -> pd.DataFrame:
    model = clf.model
    if clf.model_type == 'logistic' and hasattr(model, 'coef_'):
        importance = np.abs(model.coef_[0])
    elif clf.model_type == 'svm' and hasattr(model, 'support_vectors_') and hasattr(model, 'dual_coef_'):
        sv = model.support_vectors_
        dual = model.dual_coef_[0]
        importance = np.abs((dual[:, None] * sv).sum(axis=0))
    else:
        Xc = np.vstack([results['X_train'], results['X_test']])
        yc = np.hstack([results['y_train'], results['y_test']])
        class0 = Xc[yc == 0]
        class1 = Xc[yc == 1]
        importance = np.abs(class1.mean(axis=0) - class0.mean(axis=0))

    names = clf.feature_names if getattr(clf, "feature_names", None) else [f"f{i}" for i in range(len(importance))]
    imp_df = pd.DataFrame({"feature": names, "importance": importance})
    imp_df.sort_values("importance", ascending=False, inplace=True)
    return imp_df.reset_index(drop=True)

def fig_feature_importance(imp_df: pd.DataFrame, feature_type: str, bin_hz: float, top_k: int = 30) -> plt.Figure:
    top = imp_df.head(min(top_k, len(imp_df))).copy()
    pretty = humanize_feature_labels(top["feature"].tolist(), feature_type, bin_hz)
    top["pretty"] = pretty

    fig, ax = plt.subplots(figsize=(9, 8 if len(top) > 20 else 6))
    vals = top["importance"].values[::-1]
    labs = top["pretty"].values[::-1]
    ax.barh(range(len(vals)), vals, alpha=0.9)
    ax.set_yticks(range(len(vals)))
    ax.set_yticklabels(labs, fontsize=9)
    ax.set_title("Top Feature Importance")
    ax.set_xlabel("Importance")
    fig.tight_layout()
    return fig

# ---------------------------- Summary metrics & Filters ----------------------------
total_samples = sum(scenarios[k].get("sample_count", 0) for k in scenarios)
spec_count = sum(1 for s in scenarios.values() if s.get("features_available", {}).get("spectrum", False))
mfcc_count = sum(1 for s in scenarios.values() if s.get("features_available", {}).get("mfcc", False))

m1, m2, m3, m4 = st.columns(4)
m1.metric("Scenarios", f"{len(scenarios)}")
m2.metric("Samples", f"{total_samples:,}")
m3.metric("With Spectrum", f"{spec_count}")
m4.metric("With MFCC", f"{mfcc_count}")
st.divider()

fc1, fc2, fc3 = st.columns([1, 1, 1])
computers = ["All"] + sorted([c for c in st.session_state["df_all"]["computer"].unique() if c])
rooms = ["All"] + sorted([r for r in st.session_state["df_all"]["room"].unique() if r])

with fc1:
    computer_filter = st.selectbox("Filter by Computer", options=computers, index=0)
with fc2:
    room_filter = st.selectbox("Filter by Room", options=rooms, index=0)
with fc3:
    feature_filter = st.selectbox("Filter by Feature Type (list view filter)", options=["All", "spectrum", "mfcc"], index=0)

df_view = filter_df(st.session_state["df_all"], computer_filter, room_filter, feature_filter)
if "select" not in df_view.columns:
    df_view.insert(0, "select", False)

column_config = {
    "select": st.column_config.CheckboxColumn("Select", help="Tick to include in classification"),
    "scenario_number": st.column_config.TextColumn("Scenario #"),
    "description": st.column_config.TextColumn("Description"),
    "computer": st.column_config.TextColumn("Computer"),
    "room": st.column_config.TextColumn("Room"),
    "samples": st.column_config.NumberColumn("Samples"),
    "spectrum": st.column_config.CheckboxColumn("Spectrum"),
    "mfcc": st.column_config.CheckboxColumn("MFCC"),
    "key": st.column_config.TextColumn("key", help="internal id", disabled=True, width="small"),
}

st.subheader("Scenarios")
edited = st.data_editor(
    df_view[["select","scenario_number","description","computer","room","samples","spectrum","mfcc","key"]],
    use_container_width=True, hide_index=True,
    column_config=column_config,
    disabled=["scenario_number","description","computer","room","samples","spectrum","mfcc","key"],
    num_rows="fixed"
)
selected_keys = selected_keys_from_table(edited)
st.caption(f"Selected: {len(selected_keys)} scenario(s)")
st.divider()

# ---------------------------- Feature Extraction ----------------------------
st.header("Feature Extraction")

ec1, ec2, ec3 = st.columns([1.2, 1, 1])
with ec1:
    wav_subfolder = st.text_input("WAV subfolder", value="impulse_responses")
with ec2:
    recording_type = st.selectbox("Recording type", options=["any", "average", "raw"], index=0)
with ec3:
    max_freq = st.number_input("Max spectrum freq (Hz)", min_value=0, value=0, step=1000,
                               help="0 = no limit")

fc1, fc2, fc3 = st.columns([1.2, 1, 1])
with fc1:
    mfcc_filename = st.text_input("MFCC filename", value="features.csv")
with fc2:
    spectrum_filename = st.text_input("Spectrum filename", value="spectrum.csv")
with fc3:
    exist_mode = st.radio(
        "If features already exist",
        options=["Skip scenario (both files present)", "Keep existing (write missing only)", "Overwrite both files"],
        horizontal=False,
    )

def _update_state_after_extraction(keys: list[str]):
    """Update in-memory scenarios/df_all so UI and classification reflect new files immediately."""
    scenarios_local = st.session_state.get("scenarios", {})
    df_all_local = st.session_state.get("df_all", pd.DataFrame()).copy()

    for k in keys:
        s = scenarios_local.get(k)
        if not s:
            continue
        folder = s.get("full_path") or str(Path(dataset_path) / k)

        spec_path = Path(folder) / spectrum_filename
        mfcc_path = Path(folder) / mfcc_filename

        spec_exists = spec_path.exists()
        mfcc_exists = mfcc_path.exists()

        # update features_available
        s.setdefault("features_available", {})
        s["features_available"]["spectrum"] = bool(spec_exists)
        s["features_available"]["mfcc"] = bool(mfcc_exists)

        # recompute sample count (prefer CSV rows; fallback keep old)
        samples = s.get("sample_count", 0)
        try:
            if spec_exists:
                samples = max(samples, len(pd.read_csv(spec_path)))
            if mfcc_exists:
                samples = max(samples, len(pd.read_csv(mfcc_path)))
        except Exception:
            pass
        s["sample_count"] = int(samples)
        scenarios_local[k] = s

        # reflect in df_all
        if not df_all_local.empty:
            idx = df_all_local["key"] == k
            if idx.any():
                df_all_local.loc[idx, "spectrum"] = spec_exists
                df_all_local.loc[idx, "mfcc"] = mfcc_exists
                df_all_local.loc[idx, "samples"] = int(samples)

    st.session_state["scenarios"] = scenarios_local
    st.session_state["df_all"] = df_all_local

def _run_extraction_for_keys(keys: list[str]):
    if not keys:
        st.warning("No scenarios selected.")
        return

    overwrite_flag = (exist_mode == "Overwrite both files")
    skip_if_both_exist = (exist_mode == "Skip scenario (both files present)")

    extractor = AudioFeatureExtractor(
        sample_rate=16000,            # fallback if config missing
        n_mfcc=13,
        config_filename="recorderConfig.json",
        max_spectrum_freq=(None if max_freq <= 0 else float(max_freq)),
    )

    rows = []
    prog = st.progress(0)
    for i, k in enumerate(keys, start=1):
        s = scenarios[k]
        folder = s.get("full_path") or str(Path(dataset_path) / k)

        if skip_if_both_exist:
            mfcc_path = Path(folder) / mfcc_filename
            spec_path = Path(folder) / spectrum_filename
            if mfcc_path.exists() and spec_path.exists():
                rows.append({"scenario": k, "status": "skipped (both files present)"})
                prog.progress(int(i / len(keys) * 100))
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

        prog.progress(int(i / len(keys) * 100))

    st.success("Feature extraction finished.")
    st.dataframe(pd.DataFrame(rows), use_container_width=True)

    # 🔄 Immediately refresh in-memory state (no manual button) and re-run the app
    _update_state_after_extraction(keys)
    analyze_dataset_cached.clear()       # drop cached 'old' analysis
    st.session_state["analyzed"] = False # force re-analysis path
    st.rerun()

c1, c2 = st.columns([1, 1])
with c1:
    run_sel = st.button("Extract for SELECTED scenarios", disabled=(len(selected_keys) == 0))
with c2:
    run_all = st.button("Extract for ALL scenarios in dataset")

if run_sel:
    _run_extraction_for_keys(selected_keys)
if run_all:
    _run_extraction_for_keys(list(scenarios.keys()))

st.divider()

# ---------------------------- Classification ----------------------------
st.header("Classification")

feature_type = st.selectbox("Feature type for classification", options=["spectrum", "mfcc"], index=0)
mode = st.radio("Run mode", options=["Single pair (exactly 2 selected)", "All pairs (across all selected)"], horizontal=True)

pc1, pc2, pc3 = st.columns([1, 1, 1])
with pc1:
    model_type = st.selectbox("Model", options=["svm", "logistic"], index=0)
with pc2:
    test_size = st.slider("Test size (holdout)", 0.05, 0.5, 0.3, step=0.05)
with pc3:
    cv_folds = st.slider("CV folds", 2, 10, 5, step=1)

missing = missing_feature_scenarios(selected_keys, scenarios, feature_type)
if missing:
    st.error(
        "The following selected scenarios do not have the required features "
        f"for **{feature_type}**:\n\n- " + "\n- ".join(missing) +
        "\n\nExtract features, change selection, or choose a different feature type."
    )

# Single pair
if mode.startswith("Single"):
    can_run = len(selected_keys) == 2 and not missing
    if st.button("Run classification for selected pair", disabled=not can_run):
        k1, k2 = selected_keys
        s1, s2 = scenarios[k1], scenarios[k2]
        lbl1 = default_label_for_scenario(s1, k1)
        lbl2 = default_label_for_scenario(s2, k2)
        lbl1, lbl2 = ensure_distinct(lbl1, lbl2, s1, s2, k1, k2)

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

# All pairs
else:
    can_run_all = len(selected_keys) >= 2 and not missing
    if st.button("Run classification for ALL selected pairs", disabled=not can_run_all):
        pairs = list(itertools.combinations(selected_keys, 2))
        prog = st.progress(0)
        run_results = []

        for i, (k1, k2) in enumerate(pairs, start=1):
            s1, s2 = scenarios[k1], scenarios[k2]
            lbl1 = default_label_for_scenario(s1, k1)
            lbl2 = default_label_for_scenario(s2, k2)
            lbl1, lbl2 = ensure_distinct(lbl1, lbl2, s1, s2, k1, k2)

            if not (s1.get("features_available", {}).get(feature_type, False) and
                    s2.get("features_available", {}).get(feature_type, False)):
                run_results.append({"pair": (k1, k2), "labels": (lbl1, lbl2), "error": "missing features"})
                prog.progress(int(i / len(pairs) * 100))
                continue

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
            except Exception as e:
                run_results.append({"pair": (k1, k2), "labels": (lbl1, lbl2), "error": str(e)})

            prog.progress(int(i / len(pairs) * 100))

        st.session_state["all_last"] = run_results

# ---------------------------- Visualizations ----------------------------
# Single pair block
if st.session_state.get("single_last"):
    block = st.session_state["single_last"]
    k1, k2 = block["pair"]; lbl1, lbl2 = block["labels"]
    clf = block["clf"]; results = block["results"]; report_text = block["report_text"]
    bh = pair_bin_hz(k1, k2, scenarios, dataset_path, warn=True)

    st.success(f"Classification complete: **{k1}** vs **{k2}**  |  Labels: **{lbl1}** vs **{lbl2}**")
    mc1, mc2, mc3 = st.columns(3)
    mc1.metric("Train Acc", f"{results['train_accuracy']:.3f}")
    mc2.metric("Test Acc", f"{results['test_accuracy']:.3f}")
    mc3.metric("CV Mean ± Std", f"{results['cv_mean']:.3f} ± {results['cv_std']:.3f}")

    with st.expander("Show Confusion Matrix", expanded=False):
        st.pyplot(fig_confusion_matrix(results["confusion_matrix"], list(clf.label_encoder.classes_)))
    with st.expander("Show Cross-Validation Scores", expanded=False):
        st.pyplot(fig_cv_scores(results["cv_scores"], results["cv_mean"]))
    with st.expander("Show Feature Importance", expanded=False):
        st.caption(f"Bin resolution: ~{bh:.3f} Hz")
        imp_df = compute_feature_importance(clf, results)
        top_k = st.slider("Top K features", 5, 50, 30, step=5, key=f"single_topk_{k1}_{k2}")
        st.pyplot(fig_feature_importance(imp_df, feature_type=feature_type, bin_hz=bh, top_k=top_k))
        st.download_button(
            "Download feature importance (CSV)",
            data=imp_df.to_csv(index=False).encode("utf-8"),
            file_name=f"feature_importance_{k1}_vs_{k2}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
        )
    with st.expander("Text report", expanded=False):
        st.code(report_text or "(no text output)", language="text")

# All pairs block
if st.session_state.get("all_last"):
    runs = st.session_state["all_last"]

    overview_rows = []
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
        overview_rows.append(row)

    with st.expander("All Pairs Overview", expanded=False):
        st.dataframe(pd.DataFrame(overview_rows), use_container_width=True)

    for r in runs:
        (k1, k2) = r["pair"]
        lbls = r["labels"]
        with st.expander(f"{k1} vs {k2}  |  Labels: {lbls[0]} vs {lbls[1]}", expanded=False):
            if "results" not in r:
                st.error(r.get("error", "Unknown error"))
                continue
            clf = r["clf"]; results = r["results"]

            mc1, mc2, mc3 = st.columns(3)
            mc1.metric("Train Acc", f"{results['train_accuracy']:.3f}")
            mc2.metric("Test Acc", f"{results['test_accuracy']:.3f}")
            mc3.metric("CV Mean ± Std", f"{results['cv_mean']:.3f} ± {results['cv_std']:.3f}")

            with st.expander("Show Confusion Matrix", expanded=False):
                st.pyplot(fig_confusion_matrix(results["confusion_matrix"], list(clf.label_encoder.classes_)))
            with st.expander("Show Cross-Validation Scores", expanded=False):
                st.pyplot(fig_cv_scores(results["cv_scores"], results["cv_mean"]))
            with st.expander("Show Feature Importance", expanded=False):
                bh = pair_bin_hz(k1, k2, scenarios, dataset_path, warn=False)
                st.caption(f"Bin resolution: ~{bh:.3f} Hz")
                imp_df = compute_feature_importance(clf, results)
                st.pyplot(fig_feature_importance(imp_df, feature_type=feature_type, bin_hz=bh, top_k=20))
                st.download_button(
                    "Download feature importance (CSV)",
                    data=imp_df.to_csv(index=False).encode("utf-8"),
                    file_name=f"feature_importance_{k1}_vs_{k2}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                )
            with st.expander("Text report", expanded=False):
                st.code(r.get("report_text",""), language="text")

st.divider()

# ---------------------------- Export selection ----------------------------
st.subheader("Export Selection")
payload = {
    "dataset_path": dataset_path,
    "selection_timestamp": datetime.now().isoformat(),
    "selected_keys": selected_keys,
    "selected_summaries": [
        {
            "key": k,
            "scenario_number": scenarios[k].get("scenario_number"),
            "description": scenarios[k].get("description"),
            "computer": scenarios[k].get("computer_name"),
            "room": scenarios[k].get("room_name"),
            "samples": scenarios[k].get("sample_count"),
            "features_available": scenarios[k].get("features_available"),
        }
        for k in selected_keys
    ]
}
st.download_button(
    "Download selection JSON",
    data=json.dumps(payload, indent=2, default=str),
    file_name=f"scenario_selection_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
    mime="application/json",
    disabled=len(selected_keys) == 0
)
