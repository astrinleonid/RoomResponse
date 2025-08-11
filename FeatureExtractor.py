#!/usr/bin/env python3
"""
Audio Feature Extractor (uses sample_rate from recorderConfig.json)
+ optional upper-frequency limit for spectrum features
+ explicit keep/overwrite behavior for existing feature files.

Processes all scenario folders in a dataset and creates feature CSV files
for each scenario directly in their subfolders.
"""

import os
import re
import json
import argparse
import numpy as np
import pandas as pd
import librosa
from tqdm import tqdm
from pathlib import Path
from typing import List, Optional, Tuple


class AudioFeatureExtractor:
    """Feature extractor that processes scenarios and saves features locally."""

    def __init__(
        self,
        sample_rate: int = 16000,
        n_mfcc: int = 13,
        config_filename: str = "recorderConfig.json",
        max_spectrum_freq: Optional[float] = None,  # limit spectrum to ≤ this frequency (Hz)
    ):
        """
        Args:
            sample_rate: Default sample rate if config is missing
            n_mfcc: Number of MFCC coefficients to extract
            config_filename: JSON recorder config file to read sample_rate from
            max_spectrum_freq: If set (>0), crop spectrum to ≤ this frequency (Hz)
        """
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc
        self.config_filename = config_filename
        self.max_spectrum_freq = max_spectrum_freq

    # -------------------- Config & audio I/O --------------------

    def _read_sample_rate_from_config(self, scenario_folder: str, dataset_path: Optional[str] = None) -> int:
        candidates = [
            Path(scenario_folder) / self.config_filename,
            Path(scenario_folder) / "metadata" / self.config_filename,
        ]
        if dataset_path:
            candidates.append(Path(dataset_path) / self.config_filename)

        for cfg_path in candidates:
            try:
                if cfg_path.is_file():
                    with open(cfg_path, "r", encoding="utf-8") as f:
                        cfg = json.load(f)
                    sr = int(cfg.get("sample_rate", 0))
                    if sr > 0:
                        return sr
            except Exception:
                pass
        return int(self.sample_rate)

    def load_audio_file(self, file_path: str, target_sr: Optional[int]) -> Tuple[Optional[np.ndarray], Optional[int]]:
        try:
            y, sr_native = librosa.load(file_path, sr=None, mono=True)
        except Exception as e:
            print(f"Warning: Failed to load {file_path}: {e}")
            return None, None

        if target_sr and sr_native and sr_native != target_sr:
            try:
                y = librosa.resample(y=y, orig_sr=sr_native, target_sr=target_sr)
                sr_used = target_sr
            except Exception as e:
                print(f"Warning: Resample failed for {file_path} ({sr_native}→{target_sr}): {e}")
                sr_used = sr_native
        else:
            sr_used = sr_native if sr_native else target_sr

        return y, sr_used

    # -------------------- Feature extraction --------------------

    def _adapt_parameters_for_audio_length(self, audio_length: int) -> Tuple[int, int]:
        if audio_length < 512:
            n_fft = max(64, 2 ** int(np.log2(max(2, audio_length // 2)))) if audio_length >= 4 else 64
            hop_length = max(16, n_fft // 4)
        elif audio_length < 2048:
            n_fft = 512
            hop_length = 128
        else:
            n_fft = 2048
            hop_length = 512

        n_fft = min(n_fft, max(4, audio_length))
        hop_length = max(1, min(hop_length, max(1, audio_length // 4)))
        return n_fft, hop_length

    def _extract_spectrum_from_audio(self, audio: np.ndarray, sr_used: int) -> np.ndarray:
        try:
            if len(audio) < 4:
                return np.zeros(2)

            fft_result = np.fft.fft(audio)
            mag = np.abs(fft_result[:len(fft_result) // 2 + 1])

            # Optional upper-frequency crop
            if self.max_spectrum_freq and self.max_spectrum_freq > 0:
                nyquist = sr_used / 2.0
                effective_max = min(self.max_spectrum_freq, nyquist)
                bin_hz = sr_used / float(len(audio))
                cutoff_bin = int(np.floor(effective_max / bin_hz))
                cutoff_bin = max(0, min(cutoff_bin, len(mag) - 1))
                mag = mag[:cutoff_bin + 1]

            # Normalize after cropping
            maxv = np.max(mag) if mag.size else 0.0
            if maxv > 0:
                mag = mag / maxv

            return mag
        except Exception as e:
            print(f"Warning: Spectrum extraction failed: {e}")
            return np.zeros(len(audio) // 2 + 1 if len(audio) > 0 else 2)

    def _extract_mfcc_from_audio(self, audio: np.ndarray, sr_used: int) -> np.ndarray:
        try:
            if len(audio) < 10:
                return np.zeros(self.n_mfcc)

            n_fft, hop_length = self._adapt_parameters_for_audio_length(len(audio))

            mfcc = librosa.feature.mfcc(
                y=audio,
                sr=sr_used,
                n_mfcc=self.n_mfcc,
                n_fft=n_fft,
                hop_length=hop_length
            )

            valid_mask = ~(np.isnan(mfcc[0]) | (np.sum(np.abs(mfcc), axis=0) == 0))
            if not np.any(valid_mask):
                return np.zeros(self.n_mfcc)

            return np.mean(mfcc[:, valid_mask], axis=1)
        except Exception as e:
            print(f"Warning: MFCC extraction failed: {e}")
            return np.zeros(self.n_mfcc)

    # -------------------- Dataset scanning --------------------

    def find_wav_files(self, folder_path: str, recording_type: str = "any") -> List[str]:
        if not os.path.isdir(folder_path):
            return []
        wav_files = []
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            if not (os.path.isfile(file_path) and filename.lower().endswith('.wav')):
                continue
            name = filename.lower()
            if recording_type == "raw":
                if name.endswith('raw_recording.wav'):
                    wav_files.append(file_path)
            elif recording_type == "average":
                if (name.endswith('recording.wav') and not name.endswith('raw_recording.wav')):
                    wav_files.append(file_path)
            else:
                wav_files.append(file_path)
        return sorted(wav_files)

    # -------------------- Per-scenario processing --------------------

    def process_scenario_folder(self, scenario_folder: str, wav_subfolder: str,
                                recording_type: str = "any",
                                mfcc_filename: str = "features.csv",
                                spectrum_filename: str = "spectrum.csv",
                                dataset_path_for_config: Optional[str] = None,
                                overwrite_existing_files: bool = True) -> bool:
        """
        Process a single scenario folder and save both MFCC and spectrum CSV files.

        If overwrite_existing_files is False:
          - existing files are NOT overwritten;
          - any missing file is created.
        """
        wav_folder_path = os.path.join(scenario_folder, wav_subfolder)
        if not os.path.exists(wav_folder_path):
            print(f"WAV subfolder not found: {wav_folder_path}")
            return False

        scenario_sr = self._read_sample_rate_from_config(scenario_folder, dataset_path_for_config)
        if not scenario_sr:
            scenario_sr = self.sample_rate  # fallback

        wav_files = self.find_wav_files(wav_folder_path, recording_type)
        if not wav_files:
            print(f"No {recording_type} WAV files found in {wav_folder_path}")
            return False

        print(
            f"Processing {len(wav_files)} files in {os.path.basename(scenario_folder)} "
            f"(sr={scenario_sr}"
            + (f", max_freq={self.max_spectrum_freq} Hz" if self.max_spectrum_freq and self.max_spectrum_freq > 0 else "")
            + f", overwrite={'yes' if overwrite_existing_files else 'no'})"
        )

        mfcc_features_list = []
        spectrum_features_list = []
        first_fft_len = None
        first_bin_hz = None
        actual_max_freq = None

        for file_path in tqdm(wav_files, desc=f"  Features", leave=False):
            audio, sr_used = self.load_audio_file(file_path, target_sr=scenario_sr)
            if audio is None or sr_used is None:
                continue

            if first_fft_len is None:
                first_fft_len = len(audio)
                first_bin_hz = sr_used / float(first_fft_len)
                if self.max_spectrum_freq and self.max_spectrum_freq > 0:
                    actual_max_freq = min(self.max_spectrum_freq, sr_used / 2.0)

            filename = os.path.basename(file_path)

            # MFCC
            mfcc_vec = self._extract_mfcc_from_audio(audio, sr_used=sr_used)
            mfcc_row = {'filename': filename}
            for i, val in enumerate(mfcc_vec):
                mfcc_row[f'mfcc_{i}'] = float(val)
            mfcc_features_list.append(mfcc_row)

            # Spectrum
            spectrum_vec = self._extract_spectrum_from_audio(audio, sr_used=sr_used)
            spec_row = {'filename': filename}
            for i, val in enumerate(spectrum_vec):
                spec_row[f'freq_{i}'] = float(val)
            spectrum_features_list.append(spec_row)

        if not mfcc_features_list and not spectrum_features_list:
            print(f"No features extracted from {scenario_folder}")
            return False

        # Decide per-file writing based on existing files + overwrite flag
        mfcc_output_path = os.path.join(scenario_folder, mfcc_filename)
        spectrum_output_path = os.path.join(scenario_folder, spectrum_filename)
        mfcc_exists = os.path.exists(mfcc_output_path)
        spectrum_exists = os.path.exists(spectrum_output_path)

        ok = True

        # Save MFCC
        try:
            if mfcc_exists and not overwrite_existing_files:
                print(f"  → Keeping existing {mfcc_filename}")
            else:
                mfcc_df = pd.DataFrame(mfcc_features_list)
                mfcc_df.to_csv(mfcc_output_path, index=False)
                print(f"  → Saved {len(mfcc_df)} MFCC feature rows to {mfcc_filename}")
        except Exception as e:
            ok = False
            print(f"  → Failed to save MFCC features: {e}")

        # Save spectrum
        try:
            if spectrum_exists and not overwrite_existing_files:
                print(f"  → Keeping existing {spectrum_filename}")
            else:
                spectrum_df = pd.DataFrame(spectrum_features_list)
                spectrum_df.to_csv(spectrum_output_path, index=False)
                print(f"  → Saved {len(spectrum_df)} spectrum rows to {spectrum_filename}")
                if spectrum_features_list:
                    print(f"  → Spectrum length: {len(spectrum_features_list[0]) - 1} frequency bins")
        except Exception as e:
            ok = False
            print(f"  → Failed to save spectrum features: {e}")

        # Save per-scenario feature metadata for downstream labeling
        try:
            spectrum_vec_example_len = None
            if spectrum_features_list:
                # subtract one for 'filename' key when counting cols
                spectrum_vec_example_len = len(next(iter(spectrum_features_list[0].keys())))  # not robust
            meta = {
                "sample_rate": int(scenario_sr),
                "fft_len": int(first_fft_len) if first_fft_len else None,
                "bin_hz": (float(first_bin_hz) if first_bin_hz is not None else None),
                "max_freq_hz": (float(actual_max_freq) if actual_max_freq is not None else None),
                "n_bins": int(len([k for k in spectrum_features_list[0].keys() if k.startswith("freq_")])) if spectrum_features_list else None,
                "config_source": self.config_filename,
            }
            with open(os.path.join(scenario_folder, "features_meta.json"), "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2)
        except Exception as e:
            print(f"  → Warning: could not write features_meta.json: {e}")

        return ok

    # -------------------- Dataset-level processing --------------------

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

    def process_dataset(self, dataset_path: str, wav_subfolder: str,
                        recording_type: str = "any",
                        mfcc_filename: str = "features.csv",
                        spectrum_filename: str = "spectrum.csv",
                        skip_existing: bool = True,
                        overwrite_existing: Optional[bool] = None) -> None:
        """
        Process entire dataset and create both MFCC and spectrum CSV files for each scenario.

        Behavior:
          - If overwrite_existing is True  -> always (re)write both files.
          - If overwrite_existing is False -> never overwrite; create missing files only.
          - If overwrite_existing is None  -> use 'skip_existing' (legacy): skip scenarios when BOTH files exist.
        """
        print(f"Processing dataset: {dataset_path}")
        print(f"WAV subfolder: {wav_subfolder}")
        print(f"Recording type: {recording_type}")
        print(f"MFCC output filename: {mfcc_filename}")
        print(f"Spectrum output filename: {spectrum_filename}")
        print(f"Config filename: {self.config_filename}")
        if self.max_spectrum_freq and self.max_spectrum_freq > 0:
            print(f"Max spectrum frequency: {self.max_spectrum_freq} Hz")

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
            both_exist = os.path.exists(mfcc_file) and os.path.exists(spectrum_file)

            # Determine scenario-level action
            if overwrite_existing is True:
                # process & overwrite both
                pass
            elif overwrite_existing is False:
                # always process; will only write missing
                pass
            else:
                # legacy mode via skip_existing: if both exist, skip the entire scenario
                if skip_existing and both_exist:
                    skipped += 1
                    print(f"Skipping {scenario_name} - feature files already exist")
                    continue

            ok = self.process_scenario_folder(
                scenario_folder=scenario_folder,
                wav_subfolder=wav_subfolder,
                recording_type=recording_type,
                mfcc_filename=mfcc_filename,
                spectrum_filename=spectrum_filename,
                dataset_path_for_config=dataset_path,
                overwrite_existing_files=(True if overwrite_existing is True else False)
            )

            if ok:
                processed += 1
            else:
                failed += 1

        # Summary
        print(f"\n=== Processing Summary ===")
        print(f"Total scenarios: {len(scenario_folders)}")
        print(f"Processed: {processed}")
        print(f"Skipped: {skipped}")
        print(f"Failed: {failed}")

        if processed > 0:
            mode = "overwrite all" if overwrite_existing is True else ("keep existing (write missing only)" if overwrite_existing is False else ("skip existing (both present)"))
            print(f"\nMode: {mode}")
            print(f"Feature files:")
            print(f"  - {mfcc_filename} (MFCC features)")
            print(f"  - {spectrum_filename} (Spectrum features{' ≤ ' + str(self.max_spectrum_freq) + ' Hz' if self.max_spectrum_freq else ''})")
            print(f"Per-scenario metadata saved to features_meta.json")


def main():
    """
    Main entry point for feature extraction.
    """
    parser = argparse.ArgumentParser(
        description='Extract MFCC and spectrum features for each scenario in dataset (uses recorderConfig.json sample_rate)'
    )

    # Paths
    parser.add_argument('--dataset_path', default='room_response_dataset',
                        help='Path to dataset directory containing scenario folders')
    parser.add_argument('--wav_subfolder', default='impulse_responses',
                        help='Subfolder containing WAV files (e.g., "impulse_responses")')

    # Options
    parser.add_argument('--recording-type', choices=['average', 'raw', 'any'],
                        default='any', help='Type of recording to process (default: any)')
    parser.add_argument('--mfcc-filename', default='features.csv',
                        help='Name of MFCC feature CSV per scenario (default: features.csv)')
    parser.add_argument('--spectrum-filename', default='spectrum.csv',
                        help='Name of spectrum feature CSV per scenario (default: spectrum.csv)')
    parser.add_argument('--sample-rate', type=int, default=16000,
                        help='Fallback sample rate if config is missing (default: 16000)')
    parser.add_argument('--n-mfcc', type=int, default=13,
                        help='Number of MFCC coefficients (default: 13)')
    parser.add_argument('--config-filename', default='recorderConfig.json',
                        help='Recorder config filename to read sample_rate from (default: recorderConfig.json)')

    # Spectrum limiter
    parser.add_argument('--max-freq', type=float, default=None,
                        help='Upper frequency limit (Hz) for spectrum features; if omitted or ≤0, no limit')

    # Existing files handling
    parser.add_argument('--overwrite-existing', action='store_true',
                        help='Overwrite existing feature files if present')
    parser.add_argument('--force', action='store_true',
                        help='[Deprecated alias] Overwrite existing feature files')

    # Legacy mode: skip scenarios that already have both files
    parser.add_argument('--keep-existing', action='store_true',
                        help='Keep existing files; create missing ones only (ignores --overwrite-existing/--force)')

    args = parser.parse_args()

    # Resolve behavior flags (priority: keep-existing > overwrite-existing/force > legacy skip)
    if args.keep_existing:
        overwrite_existing = False
        skip_existing = False
    elif args.overwrite_existing or args.force:
        overwrite_existing = True
        skip_existing = False
    else:
        overwrite_existing = None  # use legacy skip_existing
        skip_existing = True

    extractor = AudioFeatureExtractor(
        sample_rate=args.sample_rate,
        n_mfcc=args.n_mfcc,
        config_filename=args.config_filename,
        max_spectrum_freq=(args.max_freq if (args.max_freq and args.max_freq > 0) else None),
    )

    extractor.process_dataset(
        dataset_path=args.dataset_path,
        wav_subfolder=args.wav_subfolder,
        recording_type=args.recording_type,
        mfcc_filename=args.mfcc_filename,
        spectrum_filename=args.spectrum_filename,
        skip_existing=skip_existing,
        overwrite_existing=overwrite_existing
    )


if __name__ == "__main__":
    main()
