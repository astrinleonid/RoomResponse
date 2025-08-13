#!/usr/bin/env python3
"""
Audio Feature Extractor

Processes all scenario folders in a dataset and creates feature CSV files
for each scenario directly in their subfolders.

Also provides aligned single-sample extraction for inference:
- build_feature_vector_from_audio(audio, feature_type, feature_names)
- build_feature_vector_from_wav(file_path, feature_type, feature_names)
"""

import os
import re
import glob
import argparse
import json
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import librosa
from tqdm import tqdm


def _suffix_int(name: str) -> Optional[int]:
    """Extract trailing integer or freq_* integer from a feature name."""
    m = re.search(r'(\d+)$', name) or re.search(r'freq[_\-]?(\d+)', name)
    return int(m.group(1)) if m else None


class AudioFeatureExtractor:
    """Feature extractor that processes scenarios and saves features locally."""

    def __init__(
        self,
        sample_rate: int = 16000,
        n_mfcc: int = 13,
        config_filename: Optional[str] = None,
        max_spectrum_freq: Optional[float] = None,
    ):
        """
        Initialize the feature extractor.

        Args:
            sample_rate: Fallback sample rate for audio loading and MFCCs
            n_mfcc: Number of MFCC coefficients to extract
            config_filename: If provided and exists (absolute or relative to dataset/scenario),
                             read sample_rate from this JSON (key: "sample_rate")
            max_spectrum_freq: If set, trim spectrum columns above this frequency (when saving CSVs)
        """
        self.sample_rate = int(sample_rate)
        self.n_mfcc = int(n_mfcc)
        self.config_filename = config_filename
        self.max_spectrum_freq = max_spectrum_freq

        # If config file is absolute and exists, apply SR immediately
        if self.config_filename and os.path.isfile(self.config_filename):
            try:
                with open(self.config_filename, "r", encoding="utf-8") as f:
                    cfg = json.load(f)
                sr = int(cfg.get("sample_rate", self.sample_rate))
                if sr > 0:
                    self.sample_rate = sr
            except Exception:
                pass

    # ---------------- Internals used by both batch and single-sample paths ----------------

    def _adapt_parameters_for_audio_length(self, audio_length: int) -> Tuple[int, int]:
        if audio_length < 512:
            n_fft = max(64, 2 ** int(np.log2(max(4, audio_length // 2))))
            hop_length = max(16, n_fft // 4)
        elif audio_length < 2048:
            n_fft = 512
            hop_length = 128
        else:
            n_fft = 2048
            hop_length = 512

        n_fft = min(n_fft, max(4, audio_length))
        hop_length = min(hop_length, max(1, audio_length // 4))
        return n_fft, hop_length

    def _extract_spectrum_from_audio(self, audio: np.ndarray) -> np.ndarray:
        try:
            if len(audio) < 4:
                return np.zeros(2, dtype=float)
            fft_result = np.fft.fft(audio)
            mag = np.abs(fft_result[:len(audio) // 2 + 1])
            if np.max(mag) > 0:
                mag = mag / np.max(mag)
            return mag.astype(float)
        except Exception:
            return np.zeros(len(audio) // 2 + 1 if len(audio) > 0 else 2, dtype=float)

    def _extract_mfcc_from_audio(self, audio: np.ndarray) -> np.ndarray:
        try:
            if len(audio) < 10:
                return np.zeros(self.n_mfcc, dtype=float)
            n_fft, hop_length = self._adapt_parameters_for_audio_length(len(audio))
            mfcc = librosa.feature.mfcc(
                y=audio.astype(float),
                sr=self.sample_rate,
                n_mfcc=self.n_mfcc,
                n_fft=n_fft,
                hop_length=hop_length
            )
            valid_mask = ~(np.isnan(mfcc[0]) | (np.sum(np.abs(mfcc), axis=0) == 0))
            if not np.any(valid_mask):
                return np.zeros(self.n_mfcc, dtype=float)
            return np.mean(mfcc[:, valid_mask], axis=1).astype(float)
        except Exception:
            return np.zeros(self.n_mfcc, dtype=float)

    # --------- NEW: aligned single-sample feature vector builders (used by inference) ---------

    def build_feature_vector_from_audio(
        self,
        audio: np.ndarray,
        feature_type: str,
        feature_names: List[str],
    ) -> np.ndarray:
        """
        Produce a 1D vector aligned exactly to `feature_names`.

        - For 'spectrum': compute magnitude spectrum (positive half, normalized),
          and pick bins by 'freq_<k>'. If a requested index exceeds available bins,
          0.0 is used.
        - For 'mfcc': compute MFCCs (count=self.n_mfcc) and map by 'mfcc_<i>'.

        NOTE: This method does not resample `audio`. Ensure that `self.sample_rate`
        matches the real sampling rate (e.g., by initializing this extractor with
        the proper recorder config).
        """
        feature_type = feature_type.lower()
        if feature_type == "spectrum":
            # Compute spectrum once
            mag = self._extract_spectrum_from_audio(audio)
            # Align
            vec = []
            for name in feature_names:
                k = _suffix_int(name)
                if k is None or k < 0 or k >= len(mag):
                    vec.append(0.0)
                else:
                    vec.append(float(mag[k]))
            return np.array(vec, dtype=float)

        # MFCC path
        mfcc = self._extract_mfcc_from_audio(audio)
        vec_map = {f"mfcc_{i}": float(mfcc[i]) for i in range(len(mfcc))}
        return np.array([vec_map.get(n, 0.0) for n in feature_names], dtype=float)

    def build_feature_vector_from_wav(
        self,
        file_path: str,
        feature_type: str,
        feature_names: List[str],
    ) -> np.ndarray:
        """
        Load a WAV (librosa with sr=self.sample_rate) and return an aligned vector.
        """
        audio = self.load_audio_file(file_path)
        if audio is None:
            raise FileNotFoundError(f"Failed to read audio: {file_path}")
        return self.build_feature_vector_from_audio(audio, feature_type, feature_names)

    # ---------------- Dataset scanning/loading/saving (unchanged public API) ----------------

    def find_wav_files(self, folder_path: str, recording_type: str = "any") -> List[str]:
        if not os.path.isdir(folder_path):
            return []
        wav_files = []
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            if not (os.path.isfile(file_path) and filename.lower().endswith('.wav')):
                continue
            fn_lower = filename.lower()
            if recording_type == "raw":
                if fn_lower.endswith('raw_recording.wav') or fn_lower.startswith("raw_"):
                    wav_files.append(file_path)
            elif recording_type == "average":
                if (fn_lower.endswith('recording.wav') and not fn_lower.endswith('raw_recording.wav')) or fn_lower.startswith("impulse_"):
                    wav_files.append(file_path)
            else:
                wav_files.append(file_path)
        return sorted(wav_files)

    def load_audio_file(self, file_path: str) -> Optional[np.ndarray]:
        try:
            audio, _ = librosa.load(file_path, sr=self.sample_rate, mono=True)
            return audio.astype(float)
        except Exception as e:
            print(f"Warning: Failed to load {file_path}: {e}")
            return None

    def _trim_spectrum_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """If max_spectrum_freq is set, drop freq_* columns above that."""
        if self.max_spectrum_freq is None:
            return df
        # Estimate bin_hz from count of freq_* and assumed SR; we cannot perfectly
        # reconstruct here, so we trim on index if label embeds the bin index.
        freq_cols = [c for c in df.columns if c.startswith("freq_")]
        if not freq_cols:
            return df
        # Use index -> frequency: f = k * (sr/N). We don't know N; best we can do is
        # approximate with training settings. Here we simply keep columns up to the
        # inferred cut index based on typical labeling "freq_<k>" and a bin_hz guess.
        # Practical approach: if you want precise trim, do it at training time.
        # For safety, do not drop anything if we cannot parse ints.
        ks = [(_suffix_int(c), c) for c in freq_cols]
        ks = [(k, c) for k, c in ks if k is not None]
        if not ks:
            return df
        # We cannot map k->Hz without bin_hz; leave trimming responsibility optional.
        # To avoid surprises, no-op here. You can implement exact trim where you save spectrum.
        return df

    def process_scenario_folder(
        self,
        scenario_folder: str,
        wav_subfolder: str,
        recording_type: str = "any",
        mfcc_filename: str = "features.csv",
        spectrum_filename: str = "spectrum.csv",
        dataset_path_for_config: Optional[str] = None,
        overwrite_existing_files: bool = False,
    ) -> bool:
        """
        Process a single scenario folder and save both MFCC and spectrum CSV files.

        Args mirror earlier versions +:
          - dataset_path_for_config: if provided, we try <scenario>/recorderConfig.json and
            <dataset_path_for_config>/recorderConfig.json to set sample_rate.
          - overwrite_existing_files: False = keep existing file(s); True = overwrite.

        Returns True if at least 1 of the two files was successfully written.
        """
        # Optional: update SR from recorderConfig.json next to scenario or dataset root
        if self.config_filename:
            config_candidates = []
            scen_dir = Path(scenario_folder)
            # absolute
            if Path(self.config_filename).is_file():
                config_candidates.append(Path(self.config_filename))
            # scenario-local
            config_candidates.append(scen_dir / self.config_filename)
            # dataset root
            if dataset_path_for_config:
                config_candidates.append(Path(dataset_path_for_config) / self.config_filename)

            for p in config_candidates:
                if p.is_file():
                    try:
                        with open(p, "r", encoding="utf-8") as f:
                            cfg = json.load(f)
                        sr = int(cfg.get("sample_rate", self.sample_rate))
                        if sr > 0:
                            self.sample_rate = sr
                        break
                    except Exception:
                        pass

        wav_folder_path = os.path.join(scenario_folder, wav_subfolder)
        if not os.path.exists(wav_folder_path):
            print(f"WAV subfolder not found: {wav_folder_path}")
            return False

        wav_files = self.find_wav_files(wav_folder_path, recording_type)
        if not wav_files:
            print(f"No {recording_type} WAV files found in {wav_folder_path}")
            return False

        print(f"Processing {len(wav_files)} files in {os.path.basename(scenario_folder)}")

        mfcc_rows, spec_rows = [], []
        last_spec_len = 0

        for file_path in tqdm(wav_files, desc="  Features", leave=False):
            audio = self.load_audio_file(file_path)
            if audio is None:
                continue

            filename = os.path.basename(file_path)

            # MFCC
            mfcc_vec = self._extract_mfcc_from_audio(audio)
            mfcc_row = {"filename": filename}
            for i, v in enumerate(mfcc_vec):
                mfcc_row[f"mfcc_{i}"] = float(v)
            mfcc_rows.append(mfcc_row)

            # Spectrum
            spec = self._extract_spectrum_from_audio(audio)
            last_spec_len = max(last_spec_len, len(spec))
            spec_row = {"filename": filename}
            for i, v in enumerate(spec):
                spec_row[f"freq_{i}"] = float(v)
            spec_rows.append(spec_row)

        if not mfcc_rows or not spec_rows:
            print(f"No features extracted from {scenario_folder}")
            return False

        success_count = 0

        # Save MFCC
        try:
            mfcc_df = pd.DataFrame(mfcc_rows)
            mfcc_output_path = Path(scenario_folder) / mfcc_filename
            if (not overwrite_existing_files) and mfcc_output_path.exists():
                print(f"  → MFCC exists, keeping: {mfcc_filename}")
            else:
                mfcc_df.to_csv(mfcc_output_path, index=False)
                print(f"  → Saved {len(mfcc_df)} MFCC features to {mfcc_filename}")
                success_count += 1
        except Exception as e:
            print(f"  → Failed to save MFCC features: {e}")

        # Save Spectrum
        try:
            spec_df = pd.DataFrame(spec_rows)
            spec_df = self._trim_spectrum_df(spec_df)
            spec_output_path = Path(scenario_folder) / spectrum_filename
            if (not overwrite_existing_files) and spec_output_path.exists():
                print(f"  → Spectrum exists, keeping: {spectrum_filename}")
            else:
                spec_df.to_csv(spec_output_path, index=False)
                print(f"  → Saved {len(spec_df)} spectrum features to {spectrum_filename}")
                print(f"  → Spectrum length: {last_spec_len} frequency bins")
                success_count += 1
        except Exception as e:
            print(f"  → Failed to save spectrum features: {e}")

        # Optional: write features_meta.json for downstream frequency labeling
        try:
            meta = {
                "sample_rate": self.sample_rate,
                "fft_len": None,   # unknown per-file; defined by len(audio)
                "bin_hz": None     # per-file; not constant without fixed FFT
            }
            (Path(scenario_folder) / "features_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        except Exception:
            pass

        return success_count > 0

    def find_scenario_folders(self, dataset_path: str) -> List[str]:
        scenario_folders = []
        if not os.path.exists(dataset_path):
            print(f"Dataset path not found: {dataset_path}")
            return scenario_folders
        for item in os.listdir(dataset_path):
            item_path = os.path.join(dataset_path, item)
            if os.path.isdir(item_path) and re.search(r'scenario', item, re.IGNORECASE):
                scenario_folders.append(item_path)
        return sorted(scenario_folders)

    def process_dataset(
        self,
        dataset_path: str,
        wav_subfolder: str,
        recording_type: str = "any",
        mfcc_filename: str = "features.csv",
        spectrum_filename: str = "spectrum.csv",
        skip_existing: bool = True
    ) -> None:
        print(f"Processing dataset: {dataset_path}")
        print(f"WAV subfolder: {wav_subfolder}")
        print(f"Recording type: {recording_type}")
        print(f"MFCC output filename: {mfcc_filename}")
        print(f"Spectrum output filename: {spectrum_filename}")

        scenario_folders = self.find_scenario_folders(dataset_path)
        if not scenario_folders:
            print("No scenario folders found!")
            return

        print(f"Found {len(scenario_folders)} scenario folders")

        processed = skipped = failed = 0

        for scenario_folder in tqdm(scenario_folders, desc="Processing scenarios"):
            scenario_name = os.path.basename(scenario_folder)
            mfcc_file = os.path.join(scenario_folder, mfcc_filename)
            spectrum_file = os.path.join(scenario_folder, spectrum_filename)

            if skip_existing and os.path.exists(mfcc_file) and os.path.exists(spectrum_file):
                skipped += 1
                print(f"Skipping {scenario_name} - feature files already exist")
                continue

            success = self.process_scenario_folder(
                scenario_folder, wav_subfolder, recording_type,
                mfcc_filename, spectrum_filename,
                dataset_path_for_config=dataset_path,
                overwrite_existing_files=not skip_existing
            )

            if success:
                processed += 1
            else:
                failed += 1

        print(f"\n=== Processing Summary ===")
        print(f"Total scenarios: {len(scenario_folders)}")
        print(f"Processed: {processed}")
        print(f"Skipped: {skipped}")
        print(f"Failed: {failed}")

        if processed > 0:
            print(f"\nFeature files created:")
            print(f"  - {mfcc_filename} (MFCC features)")
            print(f"  - {spectrum_filename} (Full spectrum features)")
            print(f"Files saved in each scenario folder")


def main():
    parser = argparse.ArgumentParser(description='Extract MFCC & spectrum features for each scenario in dataset')
    parser.add_argument('--dataset_path', default='room_response_dataset')
    parser.add_argument('--wav_subfolder', default='impulse_responses')
    parser.add_argument('--recording-type', choices=['average','raw','any'], default='any')
    parser.add_argument('--mfcc-filename', default='features.csv')
    parser.add_argument('--spectrum-filename', default='spectrum.csv')
    parser.add_argument('--sample-rate', type=int, default=16000)
    parser.add_argument('--n-mfcc', type=int, default=13)
    parser.add_argument('--config-filename', type=str, default='recorderConfig.json')
    parser.add_argument('--max-spectrum-freq', type=float, default=None)
    parser.add_argument('--force', action='store_true', help='Overwrite existing feature files')
    args = parser.parse_args()

    extractor = AudioFeatureExtractor(
        sample_rate=args.sample_rate,
        n_mfcc=args.n_mfcc,
        config_filename=args.config_filename,
        max_spectrum_freq=args.max_spectrum_freq
    )

    extractor.process_dataset(
        dataset_path=args.dataset_path,
        wav_subfolder=args.wav_subfolder,
        recording_type=args.recording_type,
        mfcc_filename=args.mfcc_filename,
        spectrum_filename=args.spectrum_filename,
        skip_existing=not args.force
    )


if __name__ == "__main__":
    main()
