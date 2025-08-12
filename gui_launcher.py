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
import fnmatch

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

from ScenarioSelector import ScenarioSelector
from ScenarioClassifier import ScenarioClassifier
from FeatureExtractor import AudioFeatureExtractor
from DatasetCollector import SingleScenarioCollector

from collect_dataset import collect_scenarios_series

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

def filter_df(df: pd.DataFrame, computer: str, room: str, name_pattern: str) -> pd.DataFrame:
    out = df.copy()
    if computer != "All":
        out = out[out["computer"] == computer]
    if room != "All":
        out = out[out["room"] == room]

    pattern = (name_pattern or "").strip()
    if pattern:
        # support comma-separated patterns (e.g., "0.*,*Small")
        pats = [p.strip().lower() for p in pattern.split(",") if p.strip()]

        def match_row(row):
            s_num = str(row.get("scenario_number", "")).lower()
            key = str(row.get("key", "")).lower()
            desc = str(row.get("description", "")).lower()
            # match against number, key, or description
            return any(
                fnmatch.fnmatch(s_num, p) or fnmatch.fnmatch(key, p) or fnmatch.fnmatch(desc, p)
                for p in pats
            )

        out = out[out.apply(match_row, axis=1)]

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

def group_bin_hz(keys: list[str], scenarios: dict, dataset_path: str, warn: bool = True) -> float:
    """Get a representative bin_hz across many scenarios (warn if they differ)."""
    vals = []
    for k in keys:
        folder = scenarios[k].get("full_path") or str(Path(dataset_path) / k)
        meta = read_features_meta(folder)
        if meta:
            if meta.get("bin_hz"):
                vals.append(float(meta["bin_hz"]))
            elif meta.get("sample_rate") and meta.get("fft_len"):
                vals.append(float(meta["sample_rate"]) / float(meta["fft_len"]))
    vals = [v for v in vals if v is not None]
    if not vals:
        return 30.0
    v0 = vals[0]
    if warn and any(abs(v - v0) > 1e-6 for v in vals[1:]):
        st.warning(
            "Frequency bin resolution differs across selected scenarios. "
            f"Examples: {sorted(set(round(v,3) for v in vals))}. "
            "Proceeding with the first group's bin size; consider re-extraction for consistency."
        )
    return v0


def _feature_prefix(feature_type: str) -> str:
    return "freq_" if feature_type == "spectrum" else "mfcc_"


def _read_feature_cols(csv_path: Path, prefix: str) -> list[str]:
    """Read only the header to determine available feature columns."""
    df = pd.read_csv(csv_path, nrows=1)
    return [c for c in df.columns if c.startswith(prefix)]


def _load_features_matrix(folder: str, feature_type: str, limit_k: int | None = None) -> np.ndarray:
    """Load features from a scenario folder, optionally truncating to first K columns."""
    csv_name = "spectrum.csv" if feature_type == "spectrum" else "features.csv"
    path = Path(folder) / csv_name
    if not path.exists():
        raise FileNotFoundError(f"Feature file not found: {path}")
    df = pd.read_csv(path)
    prefix = _feature_prefix(feature_type)
    cols = [c for c in df.columns if c.startswith(prefix)]
    if not cols:
        raise ValueError(f"No {feature_type} columns found in {path}")
    # sort by numeric suffix just in case
    cols_sorted = sorted(cols, key=lambda c: _suffix_int(c) or 0)
    if limit_k is not None:
        cols_sorted = cols_sorted[:limit_k]
    return df[cols_sorted].to_numpy(), cols_sorted


def prepare_group_dataset(
    groupA_keys: list[str],
    groupB_keys: list[str],
    scenarios: dict,
    dataset_path: str,
    feature_type: str,
    balance: bool = False
):
    """
    Build (X, y, feature_names, labels) from many scenarios per group.
    Align columns by truncating to common first-K features across all folders.
    """
    if not groupA_keys or not groupB_keys:
        raise ValueError("Both groups must have at least one scenario.")

    # Determine common K across all selected scenarios by counting available columns
    prefix = _feature_prefix(feature_type)
    all_keys = groupA_keys + groupB_keys
    per_folder_cols = []
    min_k = None
    for k in all_keys:
        folder = scenarios[k].get("full_path") or str(Path(dataset_path) / k)
        csv_name = "spectrum.csv" if feature_type == "spectrum" else "features.csv"
        cols = _read_feature_cols(Path(folder) / csv_name, prefix)
        if not cols:
            raise ValueError(f"No {feature_type} columns in {folder}/{csv_name}")
        # sort by suffix and count
        cols_sorted = sorted(cols, key=lambda c: _suffix_int(c) or 0)
        per_folder_cols.append((k, folder, cols_sorted))
        count = len(cols_sorted)
        min_k = count if min_k is None else min(min_k, count)

    if min_k is None or min_k <= 0:
        raise ValueError("Cannot determine common feature length across scenarios.")

    # Load and stack matrices with the common K columns
    XA_list, XB_list = [], []
    for k, folder, _ in per_folder_cols:
        Xmat, cols_used = _load_features_matrix(folder, feature_type, limit_k=min_k)
        if k in groupA_keys:
            XA_list.append(Xmat)
        else:
            XB_list.append(Xmat)

    XA = np.vstack(XA_list) if XA_list else np.empty((0, min_k))
    XB = np.vstack(XB_list) if XB_list else np.empty((0, min_k))

    # Optional balancing by undersampling to min class size
    if balance and len(XA) > 0 and len(XB) > 0:
        n = min(len(XA), len(XB))
        rng = np.random.default_rng(42)
        if len(XA) > n:
            XA = XA[rng.choice(len(XA), size=n, replace=False)]
        if len(XB) > n:
            XB = XB[rng.choice(len(XB), size=n, replace=False)]

    X = np.vstack([XA, XB])
    y = np.array([0] * len(XA) + [1] * len(XB))
    # Feature names: freq_0..K-1 or mfcc_0..K-1
    feature_names = [f"{prefix}{i}" for i in range(min_k)]
    return X, y, feature_names



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

# ---------------------------- Filters + Scenarios table ----------------------------
# Make sure we have a base table
df_all = st.session_state.get("df_all", pd.DataFrame())
if df_all.empty:
    st.warning("No scenarios found in this dataset directory.")
    st.stop()

# Filters
fc1, fc2, fc3 = st.columns([1, 1, 1])
computers = ["All"] + sorted([c for c in df_all["computer"].unique() if c])
rooms = ["All"] + sorted([r for r in df_all["room"].unique() if r])

with fc1:
    computer_filter = st.selectbox("Filter by Computer", options=computers, index=0)
with fc2:
    room_filter = st.selectbox("Filter by Room", options=rooms, index=0)
with fc3:
    name_pattern = st.text_input(
        "Scenario pattern (glob)",
        value="",
        placeholder="e.g., 0.*  or  *Small  or  *Scenario0.*",
        help="Matches scenario number, folder name (key), or description. "
             "Supports *, ? and comma-separated patterns."
    )

# Build filtered view
df_view = filter_df(df_all, computer_filter, room_filter, name_pattern)

# Ensure 'select' exists and is boolean
df_view = df_view.copy()
if "select" not in df_view.columns:
    df_view.insert(0, "select", False)
else:
    df_view["select"] = df_view["select"].fillna(False).astype(bool)

# Render table
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
    num_rows="fixed",
)

selected_keys = selected_keys_from_table(edited)
st.caption(f"Selected: {len(selected_keys)} scenario(s)")
st.divider()



# ---------------------------- Feature Extraction ----------------------------
st.header("Feature Extraction")
with st.expander("Open feature extraction panel", expanded=False):
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

# ---------------------------- Data Collection (NEW) ----------------------------
st.header("Data Collection (Create a New Scenario)")

with st.expander("Open data collection panel", expanded=False):
    cc1, cc2, cc3 = st.columns([1, 1, 1])
    with cc1:
        col_computer = st.text_input("Computer name", value="unknownComp").strip()
    with cc2:
        col_room = st.text_input("Room name", value="unknownRoom").strip()
    with cc3:
        col_scen_no = st.text_input("Scenario number", value="1").strip()

    desc = st.text_input("Description", value="Room response measurement")

    mc1, mc2 = st.columns([1, 1])
    with mc1:
        num_meas = st.number_input("Number of measurements", min_value=1, value=30, step=1)
    with mc2:
        meas_interval = st.number_input("Interval between measurements (s)", min_value=0.1, value=2.0, step=0.1)

    out_dir = st.text_input("Output directory", value=dataset_path)
    config_file = st.text_input("Config file (recorderConfig.json)", value="recorderConfig.json")

    interactive_devices = st.checkbox(
        "Interactive device selection (console prompts)", value=False,
        help="Streamlit can't handle console prompts. If you need device selection, run the CLI: "
             "`python collect_dataset.py -i`."
    )
    if interactive_devices:
        st.warning("Interactive prompts may hang the app. Recommended to leave this off in Streamlit.")

    auto_extract = st.checkbox("Extract features after collection (uses extraction settings above)", value=True)

    def _slug(s: str) -> str:
        return (s or "").strip().replace(" ", "_")

    scenario_key_new = f"{_slug(col_computer)}-Scenario{col_scen_no}-{_slug(col_room)}"

    if st.button("Start collection", type="primary"):
        if not out_dir:
            st.error("Please provide an output directory.")
            st.stop()

        scenario_parameters = {
            "computer_name": _slug(col_computer) or "unknownComp",
            "room_name": _slug(col_room) or "unknownRoom",
            "scenario_number": col_scen_no or "1",
            "description": desc or f"Room response measurement scenario {col_scen_no}",
            # IMPORTANT: SingleScenarioCollector expects num_measurements (not num_measures)
            "num_measurements": int(num_meas),
            "measurement_interval": float(meas_interval),
        }

        buf = io.StringIO()
        try:
            with st.spinner("Collecting measurements... this may take a while"):
                with contextlib.redirect_stdout(buf):
                    collector = SingleScenarioCollector(
                        base_output_dir=out_dir,
                        recorder_config=config_file,
                        scenario_config=scenario_parameters
                    )
                    # Avoid console prompts inside Streamlit
                    collector.collect_scenario(interactive_devices=False)
            st.success(f"Collection complete: **{scenario_key_new}**")
        except Exception as e:
            st.error(f"Collection failed: {e}")
        finally:
            log_text = buf.getvalue()
            if log_text.strip():
                st.text_area("Collection log", value=log_text, height=240)

        # If we want to extract immediately, do it now, then refresh UI.
        if auto_extract:
            try:
                # Make sure the new scenario is visible in memory
                selector_ref, scenarios_all_ref, df_all_ref = analyze_dataset_cached(out_dir)
                st.session_state["selector"] = selector_ref
                st.session_state["scenarios"] = scenarios_all_ref
                st.session_state["df_all"] = df_all_ref
                st.session_state["analyzed"] = True

                if scenario_key_new in scenarios_all_ref:
                    st.info(f"Extracting features for: {scenario_key_new}")
                    _run_extraction_for_keys([scenario_key_new])  # this will update state & st.rerun()
                else:
                    # If for some reason not found, just force a full refresh
                    analyze_dataset_cached.clear()
                    st.session_state["analyzed"] = False
                    st.rerun()
            except Exception as ex:
                st.error(f"Auto-extraction failed: {ex}")
                analyze_dataset_cached.clear()
                st.session_state["analyzed"] = False
                st.rerun()
        else:
            # No auto-extraction: just refresh the dataset so the new scenario shows up
            analyze_dataset_cached.clear()
            st.session_state["analyzed"] = False
            st.rerun()

# ---------------------------- Series Collection (Multi-scenario) ----------------------------
st.header("Series Collection")

with st.expander("Collect multiple scenarios in a series", expanded=False):
    sc1, sc2 = st.columns([1, 1])
    with sc1:
        series_input = st.text_input(
            "Scenario numbers (comma/range syntax)",
            value="0.1,0.2,1-3",
            help="Comma-separated and/or numeric ranges. Examples: 0.1,0.2,1-3,7"
        )
        pre_delay = st.number_input("Delay BEFORE first scenario (sec)", min_value=0.0, value=60.0, step=5.0)
        inter_delay = st.number_input("Delay BETWEEN scenarios (sec)", min_value=0.0, value=60.0, step=5.0)
        interactive_devices = st.checkbox("Interactive device selection", value=False)
    with sc2:
        config_path = st.text_input("Recorder config file", value="recorderConfig.json")
        num_meas = st.number_input("Measurements per scenario", min_value=1, value=30, step=1)
        meas_interval = st.number_input("Interval between measurements (sec)", min_value=0.1, value=2.0, step=0.1)
        desc_tmpl = st.text_input("Description template (use {n})", value="Room response measurement scenario {n}")

    # Load defaults for computer/room from config (for convenience)
    def _load_defaults(cfg_path: str):
        try:
            with open(cfg_path, "r", encoding="utf-8") as f:
                js = json.load(f)
            return js.get("computer", "unknownComp"), js.get("room", "unknownRoom")
        except Exception:
            return "unknownComp", "unknownRoom"

    d1, d2, d3 = st.columns([1, 1, 1])
    with d1:
        default_comp, default_room = _load_defaults(config_path)
        base_computer = st.text_input("Computer name", value=default_comp)
    with d2:
        base_room = st.text_input("Room name", value=default_room)
    with d3:
        beep_on = st.checkbox("Beep cues", value=True)
        beep_freq = st.number_input("Beep freq (Hz)", min_value=100, max_value=4000, value=880, step=10)
        beep_vol = st.slider("Beep volume", 0.0, 1.0, 0.2, 0.05)

    # Parser mirror (keep in GUI to validate before kicking off work)
    def _parse_series_expr(expr: str) -> list[str]:
        out = []
        if not expr:
            return out
        for token in [t.strip() for t in expr.split(',') if t.strip()]:
            if '-' in token:
                a, b = token.split('-', 1)
                if a.replace('.', '', 1).isdigit() and b.replace('.', '', 1).isdigit() and ('.' not in a and '.' not in b):
                    ai, bi = int(a), int(b)
                    step = 1 if bi >= ai else -1
                    out.extend([str(i) for i in range(ai, bi + step, step)])
                else:
                    out.append(token)
            else:
                out.append(token)
        seen = set()
        uniq = []
        for v in out:
            if v not in seen:
                seen.add(v)
                uniq.append(v)
        return uniq

    start_series = st.button("Start SERIES collection", type="primary")
    if start_series:
        numbers = _parse_series_expr(series_input)
        if not numbers:
            st.error("Please provide at least one valid scenario number.")
        else:
            st.info(f"Planned scenarios: {numbers}")
            with st.spinner("Collecting series..."):
                # Run the series (pre-delay before first + inter-delay between)
                try:
                    collect_scenarios_series(
                        scenario_numbers=numbers,
                        base_output_dir=dataset_path,   # your current dataset directory
                        config_file=config_path,
                        base_computer=base_computer,
                        base_room=base_room,
                        description_template=desc_tmpl,
                        num_measurements=int(num_meas),
                        measurement_interval=float(meas_interval),
                        interactive_devices=interactive_devices,
                        pre_delay=float(pre_delay),
                        inter_delay=float(inter_delay),
                        enable_beeps=bool(beep_on),
                        beep_volume=float(beep_vol),
                        beep_freq=int(beep_freq),
                        beep_dur_ms=200,
                    )
                except Exception as e:
                    st.error(f"Series collection failed: {e}")
                else:
                    st.success("Series collection finished.")

            # Refresh dataset so new scenarios/samples show up immediately
            analyze_dataset_cached.clear()
            st.session_state["analyzed"] = False
            st.rerun()

# ---------------------------- Classification ----------------------------

st.header("Classification")

feature_type = st.selectbox("Feature type for classification", options=["spectrum", "mfcc"], index=0)
mode = st.radio(
    "Run mode",
    options=["Single pair (exactly 2 selected)", "All pairs (across all selected)", "Group vs Group"],
    horizontal=True
)

# Model & CV params
pc1, pc2, pc3, pc4 = st.columns([1, 1, 1, 1])
with pc1:
    model_type = st.selectbox("Model", options=["svm", "logistic"], index=0)
with pc2:
    test_size = st.slider("Test size (holdout)", 0.05, 0.5, 0.3, step=0.05)
with pc3:
    cv_folds = st.slider("CV folds", 2, 10, 5, step=1)
with pc4:
    balance_groups = st.checkbox("Balance groups (undersample to smallest)", value=False,
                                 help="For Group vs Group mode: randomly undersample the larger class.")

# ---------- Shared: guard for missing features among currently selected ----------
missing = missing_feature_scenarios(selected_keys, scenarios, feature_type)
if mode != "Group vs Group" and missing:
    st.error(
        "The following selected scenarios do not have the required features "
        f"for **{feature_type}**:\n\n- " + "\n- ".join(missing) +
        "\n\nPlease adjust selection, choose a different feature type, or extract features."
    )

# ---------- Single pair ----------
if mode.startswith("Single"):
    can_run = len(selected_keys) == 2 and not missing_feature_scenarios(selected_keys, st.session_state["scenarios"], feature_type)
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

# ---------- All pairs ----------
elif mode.startswith("All"):
    can_run_all = len(selected_keys) >= 2 and not missing_feature_scenarios(selected_keys, st.session_state["scenarios"], feature_type)
    if st.button("Run classification for ALL selected pairs", disabled=not can_run_all):
        pairs = list(itertools.combinations(selected_keys, 2))
        prog = st.progress(0)
        run_results = []

        for i, (k1, k2) in enumerate(pairs, start=1):
            s1, s2 = scenarios[k1], scenarios[k2]
            lbl1 = default_label_for_scenario(s1, k1)
            lbl2 = default_label_for_scenario(s2, k2)
            lbl1, lbl2 = ensure_distinct(lbl1, lbl2, s1, s2, k1, k2)

            # Skip incompatible pairs (defensive)
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

# ---------- Group vs Group ----------
else:
    st.subheader("Assign scenarios to groups")

    # Persistent assignment mapping across reruns
    if "group_assign" not in st.session_state:
        st.session_state["group_assign"] = {}
    assign_map: dict[str, str] = st.session_state["group_assign"]

    st.caption("Tip: narrow with the filters and glob pattern above, then bulk-assign the FILTERED rows.")

    # Bulk assign for the currently FILTERED view
    ba1, ba2, ba3 = st.columns([1, 1, 1])
    if ba1.button("Assign FILTERED to Group A"):
        for k in df_view["key"].tolist():
            assign_map[k] = "A"
    if ba2.button("Assign FILTERED to Group B"):
        for k in df_view["key"].tolist():
            assign_map[k] = "B"
    if ba3.button("Clear groups for FILTERED"):
        for k in df_view["key"].tolist():
            assign_map.pop(k, None)

    st.session_state["group_assign"] = assign_map  # persist

    # Build table for the current FILTERED view, seeded from the mapping
    df_assign = df_view.copy()
    if "group" not in df_assign.columns:
        df_assign.insert(1, "group", df_assign["key"].map(assign_map).fillna("-"))

    column_config_group = {
        "group": st.column_config.SelectboxColumn(
            "Group", options=["-", "A", "B"], help="Assign each scenario to a group"
        ),
        "scenario_number": st.column_config.TextColumn("Scenario #"),
        "description": st.column_config.TextColumn("Description"),
        "computer": st.column_config.TextColumn("Computer"),
        "room": st.column_config.TextColumn("Room"),
        "samples": st.column_config.NumberColumn("Samples"),
        "spectrum": st.column_config.CheckboxColumn("Spectrum"),
        "mfcc": st.column_config.CheckboxColumn("MFCC"),
        "key": st.column_config.TextColumn("key", disabled=True, width="small"),
    }

    edited_assign = st.data_editor(
        df_assign[["group","scenario_number","description","computer","room","samples","spectrum","mfcc","key"]],
        use_container_width=True, hide_index=True,
        column_config=column_config_group,
        disabled=["scenario_number","description","computer","room","samples","spectrum","mfcc","key"],
        num_rows="fixed"
    )

    # Update mapping from inline edits in the grid
    for _, row in edited_assign.iterrows():
        k = row["key"]
        g = row["group"]
        if g in ("A", "B"):
            assign_map[k] = g
        else:
            assign_map.pop(k, None)

    st.session_state["group_assign"] = assign_map  # persist edits

    # Final assigned keys from the persistent mapping (across any filters)
    groupA_keys = [k for k, g in assign_map.items() if g == "A"]
    groupB_keys = [k for k, g in assign_map.items() if g == "B"]

    st.caption(f"Assigned totals — Group A: {len(groupA_keys)} | Group B: {len(groupB_keys)}")

    # Custom labels for groups
    gc1, gc2 = st.columns([1, 1])
    with gc1:
        groupA_label = st.text_input("Label for Group A", value="Group A")
    with gc2:
        groupB_label = st.text_input("Label for Group B", value="Group B")

    # Check missing features among all assigned
    missing_groups = missing_feature_scenarios(groupA_keys + groupB_keys, scenarios, feature_type)
    if missing_groups:
        st.error(
            "Some assigned scenarios do not have the required features "
            f"for **{feature_type}**:\n\n- " + "\n- ".join(missing_groups) +
            "\n\nExtract features or adjust assignments."
        )

    # Run
    can_run_groups = (len(groupA_keys) > 0 and len(groupB_keys) > 0 and not missing_groups)
    if st.button("Run classification for GROUPS", type="primary", disabled=not can_run_groups):
        # Build dataset from many folders
        try:
            X, y, feat_names = prepare_group_dataset(
                groupA_keys, groupB_keys, scenarios, dataset_path, feature_type, balance=balance_groups
            )
        except Exception as e:
            st.error(f"Failed to build dataset: {e}")
            st.stop()

        # Train using ScenarioClassifier's pipeline
        clf = ScenarioClassifier(model_type=model_type, feature_type=feature_type)
        clf.feature_names = feat_names  # ensure feature names for importance plotting

        with st.spinner("Training & evaluating (group classifier)..."):
            results = clf.train_and_evaluate(X, y, test_size=test_size, cv_folds=cv_folds)
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                clf.print_results(results)
            report_text = buf.getvalue()

        # Save for visualization block
        st.session_state["group_last"] = {
            "groups": (groupA_keys, groupB_keys),
            "labels": (groupA_label, groupB_label),
            "clf": clf,
            "results": results,
            "feature_names": feat_names,
            "feature_type": feature_type
        }

        st.success(
            f"Group classification complete: **{groupA_label}** (n={np.sum(y==0)}) "
            f"vs **{groupB_label}** (n={np.sum(y==1)})"
        )


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

# ---------------------------- Group Visualizations ----------------------------
if st.session_state.get("group_last"):
    block = st.session_state["group_last"]
    (A_keys, B_keys) = block["groups"]
    (lblA, lblB) = block["labels"]
    clf = block["clf"]; results = block["results"]
    feat_type = block.get("feature_type", feature_type)

    # Determine a representative bin size for labels (spectrum)
    if feat_type == "spectrum":
        bh = group_bin_hz(A_keys + B_keys, scenarios, dataset_path, warn=True)
    else:
        bh = 0.0

    # Headline
    st.subheader(f"Group Results: {lblA} vs {lblB}")
    mc1, mc2, mc3 = st.columns(3)
    mc1.metric("Train Acc", f"{results['train_accuracy']:.3f}")
    mc2.metric("Test Acc", f"{results['test_accuracy']:.3f}")
    mc3.metric("CV Mean ± Std", f"{results['cv_mean']:.3f} ± {results['cv_std']:.3f}")

    # Charts on demand
    with st.expander("Show Confusion Matrix", expanded=False):
        st.pyplot(fig_confusion_matrix(results["confusion_matrix"], list(clf.label_encoder.classes_)))

    with st.expander("Show Cross-Validation Scores", expanded=False):
        st.pyplot(fig_cv_scores(results["cv_scores"], results["cv_mean"]))

    with st.expander("Show Feature Importance", expanded=False):
        if feat_type == "spectrum":
            st.caption(f"Bin resolution (representative): ~{bh:.3f} Hz")
        imp_df = compute_feature_importance(clf, results)
        top_k = st.slider("Top K features", 5, 50, 30, step=5, key=f"group_topk_{hash(lblA+lblB)%9999}")
        st.pyplot(fig_feature_importance(imp_df, feature_type=feat_type, bin_hz=bh, top_k=top_k))
        st.download_button(
            "Download feature importance (CSV)",
            data=imp_df.to_csv(index=False).encode("utf-8"),
            file_name=f"feature_importance_{lblA}_vs_{lblB}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
        )

    # List which scenarios were included
    with st.expander("Scenarios included"):
        def _fmt(keys):
            rows = []
            for k in keys:
                s = scenarios.get(k, {})
                rows.append({
                    "key": k,
                    "scenario_number": s.get("scenario_number"),
                    "description": s.get("description"),
                    "computer": s.get("computer_name"),
                    "room": s.get("room_name"),
                    "samples": s.get("sample_count"),
                    "spectrum": s.get("features_available", {}).get("spectrum", False),
                    "mfcc": s.get("features_available", {}).get("mfcc", False),
                })
            return pd.DataFrame(rows)
        st.write(f"**{lblA}**")
        st.dataframe(_fmt(A_keys), use_container_width=True)
        st.write(f"**{lblB}**")
        st.dataframe(_fmt(B_keys), use_container_width=True)


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
