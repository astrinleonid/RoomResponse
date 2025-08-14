
"""
Optional compatibility helpers for environments lacking ScenarioSelector.
Not required if your project already provides ScenarioSelector.analyze_dataset.
"""
from typing import List, Dict
import os

def analyze_dataset(dataset_root: str):
    """
    Very small analyzer that lists folders containing WAVs or features.csv.
    Returns a list of dicts to mimic a simple DataFrame-like structure.
    """
    rows: List[Dict] = []
    if not os.path.isdir(dataset_root):
        return rows
    for name in sorted(os.listdir(dataset_root)):
        p = os.path.join(dataset_root, name)
        if not os.path.isdir(p):
            continue
        has_feats = os.path.exists(os.path.join(p, "features.csv"))
        wavs = []
        for r, d, files in os.walk(p):
            for f in files:
                if f.lower().endswith(".wav"):
                    wavs.append(os.path.join(r, f))
        if len(wavs) == 0 and not has_feats:
            continue
        rows.append({"scenario": name, "path": p, "n_wavs": len(wavs), "has_features_csv": has_feats})
    try:
        import pandas as pd
        return pd.DataFrame(rows)
    except Exception:
        return rows
