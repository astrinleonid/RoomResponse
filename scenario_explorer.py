# scenario_explorer.py
# Minimal scenario dataset explorer module used by gui_launcher.py

from pathlib import Path
from typing import Dict
import io
import json

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

def _read_json(p: Path) -> dict:
    try:
        return json.load(open(p, "r", encoding="utf-8"))
    except Exception:
        return {}

def _write_json(p: Path, data: dict):
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

def render(scenario_key: str, scenarios: Dict[str, dict], dataset_path: str):
    s = scenarios[scenario_key]
    folder = Path(s.get("full_path") or Path(dataset_path) / scenario_key)

    st.header(f"Scenario Explorer — {scenario_key}")
    st.caption(str(folder))

    # 1) Metadata
    st.subheader("Metadata")
    # Load session-like metadata if present
    md_paths = [
        folder / "metadata" / "session_metadata.json",
        folder / "session_metadata.json",
        folder / "metadata" / f"{folder.name}_metadata.json",
    ]
    md = {}
    md_file = None
    for p in md_paths:
        if p.is_file():
            md = _read_json(p); md_file = p; break
    if not md:
        # create a simple structure if missing
        md = {
            "scenario_info": {
                "scenario_name": folder.name,
                "scenario_number": s.get("scenario_number"),
                "computer_name": s.get("computer_name"),
                "room_name": s.get("room_name"),
                "description": s.get("description") or "",
            }
        }
        md_file = folder / "metadata" / "session_metadata.json"

    si = md.get("scenario_info", {})
    c1, c2 = st.columns([1, 1])
    with c1:
        scen_num = st.text_input("Scenario #", value=str(si.get("scenario_number") or ""))
        computer = st.text_input("Computer", value=str(si.get("computer_name") or ""))
        room = st.text_input("Room", value=str(si.get("room_name") or ""))
    with c2:
        desc = st.text_input("Description", value=str(si.get("description") or ""))
        scen_name = st.text_input("Scenario Name", value=str(si.get("scenario_name") or folder.name), disabled=True)
        # Recorder config (read-only summary if present)
        rec_cfg = md.get("recorder_config", {})
        if rec_cfg:
            st.json(rec_cfg)

    if st.button("Save metadata"):
        md["scenario_info"] = {
            "scenario_name": scen_name,
            "scenario_number": scen_num,
            "computer_name": computer,
            "room_name": room,
            "description": desc
        }
        _write_json(md_file, md)
        st.success(f"Saved to {md_file}")

    st.divider()

    # 2) Audio files
    st.subheader("Audio Files")
    subfolders = ["raw_recordings", "impulse_responses", "room_responses"]
    a1, a2 = st.columns([1, 3])
    with a1:
        pick_sub = st.selectbox("Folder", options=subfolders, index=1)
    files = []
    sub_dir = folder / pick_sub
    if sub_dir.is_dir():
        files = sorted([p.name for p in sub_dir.glob("*.wav")])
    with a2:
        if files:
            pick_file = st.selectbox("File", options=files)
            wav_path = sub_dir / pick_file
            st.audio(str(wav_path))
        else:
            st.info("No WAV files found in this subfolder.")

    st.divider()

    # 3) Features quick look
    st.subheader("Features")
    left, right = st.columns(2)

    # Spectrum
    with left:
        spec_path = folder / "spectrum.csv"
        if spec_path.is_file():
            st.caption("Spectrum (average)")
            try:
                df = pd.read_csv(spec_path)
                freq_cols = [c for c in df.columns if c.startswith("freq_")]
                if freq_cols:
                    avg = df[freq_cols].mean(axis=0).values
                    fig, ax = plt.subplots(figsize=(6, 3))
                    ax.plot(avg)
                    ax.set_title("Average Spectrum (bins)")
                    ax.set_xlabel("Bin"); ax.set_ylabel("Norm Magnitude")
                    st.pyplot(fig)
                with st.expander("Show spectrum table"):
                    st.dataframe(df.head(200), use_container_width=True)
            except Exception as e:
                st.error(f"Failed to read spectrum: {e}")
        else:
            st.info("No spectrum.csv found.")

    # MFCC
    with right:
        mfcc_path = folder / "features.csv"
        if mfcc_path.is_file():
            st.caption("MFCC table (first rows)")
            try:
                dfm = pd.read_csv(mfcc_path)
                st.dataframe(dfm.head(200), use_container_width=True)
            except Exception as e:
                st.error(f"Failed to read MFCC features: {e}")
        else:
            st.info("No features.csv found.")

    st.divider()

    if st.button("Back to main"):
        st.session_state["active_panel"] = "analyze"
