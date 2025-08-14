#!/usr/bin/env python3
"""
ScenarioManager - Centralized scenario data management

Handles:
- Scenario folder parsing and validation
- Feature availability checking
- Dataset analysis and caching
- Label and description management
- Filtering and sorting
"""

import os
import re
import json
from pathlib import Path
from typing import List, Optional, Dict, Tuple, Set

import pandas as pd
import streamlit as st

# Optional ScenarioSelector integration
try:
    from ScenarioSelector import analyze_dataset as scenario_selector_analyze
except Exception:
    scenario_selector_analyze = None


class ScenarioManager:
    """Manages scenario data, parsing, and operations."""
    
    # Session state keys
    SK_SCENARIOS_DF = "scenarios_df_cache"
    SK_LAST_DATASET_ROOT = "last_dataset_root"
    
    # Regex patterns for parsing scenario names
    _NAME_FULL_RE = re.compile(
        r'^(?P<computer>.+?)-Scenario(?P<num>[A-Za-z0-9]+(?:\.[A-Za-z0-9]+)*)-(?P<room>.+)$',
        re.IGNORECASE,
    )
    
    _NAME_NUM_RE = re.compile(
        r'(?i)-Scenario(?P<num>[A-Za-z0-9]+(?:\.[A-Za-z0-9]+)*)-'
    )
    
    def __init__(self):
        """Initialize the scenario manager."""
        pass
    
    # ----------------------------
    # File and folder utilities
    # ----------------------------
    
    @staticmethod
    def list_wavs(folder: str) -> List[str]:
        """List all WAV files recursively in a folder."""
        if not os.path.isdir(folder):
            return []
        
        out = []
        try:
            for r, d, files in os.walk(folder):
                for f in files:
                    if f.lower().endswith(('.wav', '.wave')):
                        out.append(os.path.join(r, f))
        except (OSError, PermissionError):
            pass
        return sorted(out)
    
    @staticmethod
    def check_features_available(path: str) -> Dict[str, bool]:
        """Check which feature files are available."""
        return {
            'spectrum': os.path.exists(os.path.join(path, "spectrum.csv")),
            'mfcc': os.path.exists(os.path.join(path, "features.csv")),
            'audio': len(ScenarioManager.list_wavs(path)) > 0
        }
    
    @staticmethod
    def count_feature_samples(scenario_path: str, wav_subfolder: str = "impulse_responses", 
                             recording_type: str = "any") -> int:
        """
        Count samples that would actually be processed by FeatureExtractor.
        Matches FeatureExtractor.find_wav_files() logic exactly.
        """
        wav_folder_path = os.path.join(scenario_path, wav_subfolder)
        if not os.path.isdir(wav_folder_path):
            return 0
        
        count = 0
        try:
            for filename in os.listdir(wav_folder_path):  # Non-recursive, specific folder
                file_path = os.path.join(wav_folder_path, filename)
                if not (os.path.isfile(file_path) and filename.lower().endswith('.wav')):
                    continue
                
                fn_lower = filename.lower()
                if recording_type == "raw":
                    if fn_lower.endswith('raw_recording.wav') or fn_lower.startswith("raw_"):
                        count += 1
                elif recording_type == "average":
                    if ((fn_lower.endswith('recording.wav') and not fn_lower.endswith('raw_recording.wav')) 
                        or fn_lower.startswith("impulse_")):
                        count += 1
                else:  # "any"
                    count += 1
        except (OSError, PermissionError):
            pass
        
        return count

    # ----------------------------
    # Metadata operations
    # ----------------------------
    
    @staticmethod
    def read_label(path: str) -> Optional[str]:
        """Read label from scenario_meta.json."""
        meta = os.path.join(path, "scenario_meta.json")
        if os.path.exists(meta):
            try:
                with open(meta, "r", encoding="utf-8") as f:
                    d = json.load(f)
                    if isinstance(d, dict):
                        val = d.get("label")
                        if isinstance(val, str) and val.strip():
                            return val.strip()
            except Exception:
                pass
        return None
    
    @staticmethod
    def read_description(path: str) -> Optional[str]:
        """Read description from scenario_meta.json or session_metadata.json."""
        # Try scenario_meta.json first
        meta = os.path.join(path, "scenario_meta.json")
        if os.path.exists(meta):
            try:
                with open(meta, "r", encoding="utf-8") as f:
                    d = json.load(f)
                    if isinstance(d, dict):
                        val = d.get("description")
                        if isinstance(val, str) and val.strip():
                            return val.strip()
            except Exception:
                pass
        
        # Fallback to session_metadata.json
        session_meta = os.path.join(path, "metadata", "session_metadata.json")
        if os.path.exists(session_meta):
            try:
                with open(session_meta, "r", encoding="utf-8") as f:
                    d = json.load(f)
                    if isinstance(d, dict):
                        scenario_info = d.get("scenario_info", {})
                        if isinstance(scenario_info, dict):
                            val = scenario_info.get("description")
                            if isinstance(val, str) and val.strip():
                                return val.strip()
            except Exception:
                pass
        
        return None
    
    @staticmethod
    def write_label(path: str, label: Optional[str]) -> bool:
        """Write label to scenario_meta.json."""
        meta = os.path.join(path, "scenario_meta.json")
        data: Dict[str, object] = {}
        
        # Read existing data
        if os.path.exists(meta):
            try:
                with open(meta, "r", encoding="utf-8") as f:
                    data = json.load(f) or {}
            except Exception:
                data = {}
        
        # Update label
        if label is None or not str(label).strip():
            data.pop("label", None)
        else:
            data["label"] = str(label).strip()
        
        # Write back
        try:
            os.makedirs(os.path.dirname(meta), exist_ok=True)
            with open(meta, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            return True
        except Exception as e:
            if 'st' in globals():
                st.error(f"Failed to write label for {os.path.basename(path)}: {e}", icon="⚠️")
            return False
    
    # ----------------------------
    # Scenario name parsing
    # ----------------------------
    
    @classmethod
    def parse_scenario_folder_name(cls, folder_name: str) -> Tuple[str, str, str]:
        """
        Parse '<computer>-Scenario<scenario>-<room>' into a 3-tuple:
            (number_str, computer, room)
        - number_str is returned as the exact string found (e.g., '3.5', '5.34', '7a', '1.2.3').
        - computer and room are trimmed of leading/trailing separators: space . _ -
        - If the pattern doesn't fully match, we try a fallback that still extracts <scenario>
          and derives computer/room by slicing before/after the match.
        """
        name = folder_name or ""

        # 1) Try the full structured match first.
        m = cls._NAME_FULL_RE.match(name)
        if m:
            number_str = m.group("num")
            computer = m.group("computer").strip(" ._-")
            room = m.group("room").strip(" ._-")
            return number_str, computer, room

        # 2) Fallback: find just the '-Scenario<...>-' chunk and slice around it.
        m = cls._NAME_NUM_RE.search(name)
        if m:
            number_str = m.group("num")
            computer = name[:m.start()].rstrip(" ._-")
            room = name[m.end():].lstrip(" ._-")
            return number_str, computer, room

        # 3) No match at all: return empty scenario, treat the whole name as 'computer'.
        return "", name.strip(" ._-"), ""
    
    # ----------------------------
    # Dataset analysis
    # ----------------------------

    @classmethod
    def analyze_dataset_filesystem(cls, root: str) -> pd.DataFrame:
        """Analyze dataset by scanning filesystem directly with correct sample counting."""
        rows = []
        if not os.path.isdir(root):
            return pd.DataFrame(columns=["scenario", "path", "sample_count", "features_available"])
        
        try:
            entries = sorted(os.listdir(root))
        except (OSError, PermissionError):
            return pd.DataFrame(columns=["scenario", "path", "sample_count", "features_available"])
        
        for name in entries:
            p = os.path.join(root, name)
            if not os.path.isdir(p):
                continue
            
            # Check if this looks like a scenario folder (contains "scenario" case-insensitive)
            if "scenario" not in name.lower():
                continue
                
            # Count samples using FeatureExtractor-compatible logic
            sample_count = cls.count_feature_samples(p, "impulse_responses", "any")
            features_avail = cls.check_features_available(p)
            
            # Include scenario even if no files (can be processed later)
            rows.append({
                "scenario": name, 
                "path": p, 
                "sample_count": sample_count,  # Now matches FeatureExtractor
                "features_available": features_avail
            })
        
        return pd.DataFrame(rows)
    
    @classmethod
    def build_scenarios_df(cls, root: str, force_refresh: bool = False) -> pd.DataFrame:
        """
        Build enriched scenarios dataframe with caching.
        Adds columns: label, number_str, computer, room, description
        """
        # Check cache first
        if not force_refresh and cls.SK_SCENARIOS_DF in st.session_state:
            cached_df = st.session_state[cls.SK_SCENARIOS_DF]
            if isinstance(cached_df, pd.DataFrame):
                return cached_df
        
        # Try ScenarioSelector first, fallback to filesystem
        base = None
        if scenario_selector_analyze is not None:
            try:
                df = scenario_selector_analyze(root)
                if isinstance(df, pd.DataFrame) and not df.empty:
                    # Normalize column names
                    column_mapping = {}
                    for col in df.columns:
                        col_lower = col.lower()
                        if "scenario" in col_lower and "scenario" not in column_mapping:
                            column_mapping[col] = "scenario"
                        elif col_lower in ["path", "folder"] and "path" not in column_mapping:
                            column_mapping[col] = "path"
                        elif "sample" in col_lower and "count" in col_lower:
                            column_mapping[col] = "sample_count"
                    
                    df = df.rename(columns=column_mapping)
                    
                    # Ensure required columns exist
                    if "sample_count" not in df.columns:
                        df["sample_count"] = df["path"].apply(lambda p: len(cls.list_wavs(p)) if isinstance(p, str) and os.path.isdir(p) else 0)
                    if "features_available" not in df.columns:
                        df["features_available"] = df["path"].apply(lambda p: cls.check_features_available(p) if isinstance(p, str) else {})
                    
                    base = df[["scenario", "path", "sample_count", "features_available"]].copy()
            except Exception as e:
                if 'st' in globals():
                    st.warning(f"ScenarioSelector failed, using filesystem scan: {e}")
        
        # Fallback to filesystem scan
        if base is None or base.empty:
            base = cls.analyze_dataset_filesystem(root)
        
        if base.empty:
            st.session_state[cls.SK_SCENARIOS_DF] = base
            return base

        # Enrich with labels and parsed fields
        enriched = base.copy()
        
        # Attach labels and descriptions
        enriched["label"] = enriched["path"].apply(cls.read_label)
        enriched["description"] = enriched["path"].apply(cls.read_description)
        
        # Parse name fields
        parsed = enriched["scenario"].apply(cls.parse_scenario_folder_name)
        enriched["number_str"] = [p[0] for p in parsed]
        enriched["computer"] = [p[1] for p in parsed]
        enriched["room"] = [p[2] for p in parsed]
        
        # Cache the result
        st.session_state[cls.SK_SCENARIOS_DF] = enriched
        return enriched
    
    @classmethod
    def clear_cache(cls):
        """Clear the scenarios dataframe cache."""
        if cls.SK_SCENARIOS_DF in st.session_state:
            del st.session_state[cls.SK_SCENARIOS_DF]
    
    # ----------------------------
    # Filtering and sorting
    # ----------------------------
    
    @staticmethod
    def apply_filters(df: pd.DataFrame, text: str, computer: str, room: str) -> pd.DataFrame:
        """Apply filters with regex support. Primary filter targets scenario number first."""
        filt = df.copy()
        if filt.empty:
            return filt

        # --- Primary filter (number-first with smart fallback) ---
        if text:
            raw = text.strip()
            mode = "auto"
            if raw.lower().startswith("name:"):
                mode, patt = "name", raw[5:].strip()
            elif raw.lower().startswith("path:"):
                mode, patt = "path", raw[5:].strip()
            else:
                patt = raw

            try:
                pattern = re.compile(patt, re.IGNORECASE)
            except re.error:
                # Invalid regex -> treat as plain substring
                pattern = None

            def safe_series(s):
                return s.fillna("").astype(str)

            if mode == "name":
                mask = safe_series(filt["scenario"]).str.contains(pattern or re.escape(patt), regex=bool(pattern), case=False)
                filt = filt[mask]
            elif mode == "path":
                mask = safe_series(filt["path"]).str.contains(pattern or re.escape(patt), regex=bool(pattern), case=False)
                filt = filt[mask]
            else:
                # AUTO: try number_str with .match (start-anchored), then fallback to scenario/path contains
                num_mask = safe_series(filt.get("number_str", "")).str.match(pattern or re.compile(re.escape(patt), re.IGNORECASE))
                if num_mask.any():
                    filt = filt[num_mask]
                else:
                    name_mask = safe_series(filt["scenario"]).str.contains(pattern or re.escape(patt), regex=bool(pattern), case=False)
                    path_mask = safe_series(filt["path"]).str.contains(pattern or re.escape(patt), regex=bool(pattern), case=False)
                    filt = filt[name_mask | path_mask]

        # --- Computer filter (kept as simple contains) ---
        if computer:
            c = computer.strip().lower()
            mask = (
                df["computer"].fillna("").str.lower().str.contains(c, na=False) |
                df["scenario"].fillna("").str.lower().str.contains(c, na=False)
            )
            filt = filt[filt.index.isin(df[mask].index)]

        # --- Room filter (kept as simple contains) ---
        if room:
            r = room.strip().lower()
            mask = (
                df["room"].fillna("").str.lower().str.contains(r, na=False) |
                df["scenario"].fillna("").str.lower().str.contains(r, na=False)
            )
            filt = filt[filt.index.isin(df[mask].index)]

        return filt

    
    @staticmethod
    def sort_scenarios_df(df: pd.DataFrame) -> pd.DataFrame:
        """Sort scenarios by numeric scenario number when possible, else by name."""
        if df.empty:
            return df
        
        def sort_key(number_str):
            """Create sort key for scenario numbers."""
            if pd.isna(number_str) or not str(number_str).strip():
                return (2, 0)  # Empty numbers at end
            
            num_str = str(number_str).strip()
            
            # Try pure numeric
            try:
                return (0, float(num_str))
            except ValueError:
                pass
            
            # Try numeric with suffix (e.g., "1a", "2.5b")
            match = re.match(r'^(\d+(?:\.\d+)?)([a-zA-Z]*)$', num_str)
            if match:
                try:
                    numeric_part = float(match.group(1))
                    suffix = match.group(2).lower()
                    suffix_value = sum(ord(c) - ord('a') for c in suffix) / 100.0
                    return (0, numeric_part + suffix_value)
                except ValueError:
                    pass
            
            # Non-numeric: sort alphabetically
            return (1, num_str.lower())
        
        df_sorted = df.copy()
        df_sorted["_sort_key"] = df_sorted["number_str"].apply(sort_key)
        df_sorted = df_sorted.sort_values(["_sort_key", "scenario"]).drop(columns=["_sort_key"])
        return df_sorted.reset_index(drop=True)
    
    # ----------------------------
    # Label management
    # ----------------------------
    
    @staticmethod
    def get_unique_labels(df: pd.DataFrame) -> Set[str]:
        """Get all unique non-empty labels from the dataframe."""
        labels = set()
        for label_val in df["label"]:
            if label_val and isinstance(label_val, str):
                # Support comma-separated multi-labels
                for lbl in label_val.split(","):
                    lbl = lbl.strip()
                    if lbl:
                        labels.add(lbl)
        return labels
    
    # ----------------------------
    # Display utilities
    # ----------------------------
    
    @staticmethod
    def format_features_status(features_dict: Dict[str, bool]) -> str:
        """Format features availability as a compact string."""
        if not isinstance(features_dict, dict):
            return "—"
        
        status_chars = []
        if features_dict.get('spectrum', False):
            status_chars.append("S")
        if features_dict.get('mfcc', False):
            status_chars.append("M")
        if features_dict.get('audio', False):
            status_chars.append("A")
        
        return "".join(status_chars) if status_chars else "—"
    
    # ----------------------------
    # Validation utilities
    # ----------------------------
    
    @staticmethod
    def validate_dataset_root(root: str) -> Tuple[bool, str]:
        """
        Validate dataset root directory.
        Returns (is_valid, message)
        """
        if not root or not root.strip():
            return False, "Dataset root path is empty"
        
        root = root.strip()
        if not os.path.exists(root):
            return False, "Directory does not exist"
        
        if not os.path.isdir(root):
            return False, "Path is not a directory"
        
        try:
            # Try to list directory contents
            entries = os.listdir(root)
            scenario_folders = [e for e in entries if os.path.isdir(os.path.join(root, e)) and "scenario" in e.lower()]
            
            if not scenario_folders:
                return True, f"Directory exists but contains no scenario folders ({len(entries)} items total)"
            else:
                return True, f"Found {len(scenario_folders)} scenario folders ({len(entries)} items total)"
        
        except (OSError, PermissionError):
            return False, "Permission denied or I/O error"
    
    # ----------------------------
    # Bulk operations
    # ----------------------------
    
    @classmethod
    def bulk_apply_label(cls, df: pd.DataFrame, label: Optional[str]) -> int:
        """
        Apply a label to all scenarios in the dataframe.
        Returns the number of scenarios successfully updated.
        """
        updated_count = 0
        
        # Handle None or empty label (clear labels)
        label_to_apply = None if not label or not str(label).strip() else str(label).strip()
        
        for _, row in df.iterrows():
            if cls.write_label(row["path"], label_to_apply):
                updated_count += 1
        
        return updated_count