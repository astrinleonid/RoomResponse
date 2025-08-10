#!/usr/bin/env python3
"""
Audio Feature Extractor

Processes all scenario folders in a dataset and creates feature CSV files
for each scenario directly in their subfolders.
"""

import os
import re
import glob
import argparse
import numpy as np
import pandas as pd
import librosa
from tqdm import tqdm
from pathlib import Path
from typing import List, Optional, Tuple


class AudioFeatureExtractor:
    """Feature extractor that processes scenarios and saves features locally."""

    def __init__(self, sample_rate: int = 16000, n_mfcc: int = 13):
        """
        Initialize the feature extractor.

        Args:
            sample_rate (int): Target sample rate for audio loading
            n_mfcc (int): Number of MFCC coefficients to extract
        """
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc

    def _adapt_parameters_for_audio_length(self, audio_length: int) -> Tuple[int, int]:
        """
        Adapt FFT parameters based on audio length to avoid warnings.

        Args:
            audio_length (int): Length of audio signal in samples

        Returns:
            Tuple[int, int]: (n_fft, hop_length) adapted for the audio length
        """
        if audio_length < 512:
            n_fft = max(64, 2 ** int(np.log2(audio_length // 2))) if audio_length >= 4 else 64
            hop_length = max(16, n_fft // 4)
        elif audio_length < 2048:
            n_fft = 512
            hop_length = 128
        else:
            n_fft = 2048
            hop_length = 512

        # Ensure parameters don't exceed audio length
        n_fft = min(n_fft, audio_length)
        hop_length = min(hop_length, audio_length // 4) if audio_length > 4 else 1

        return n_fft, hop_length

    def _extract_spectrum_from_audio(self, audio: np.ndarray) -> np.ndarray:
        """
        Extract spectrum features using FFT on the full audio length.

        Args:
            audio (np.ndarray): Audio signal

        Returns:
            np.ndarray: Magnitude spectrum (positive frequencies only)
        """
        try:
            # Skip very short audio
            if len(audio) < 4:
                return np.zeros(2)  # Minimum meaningful spectrum

            # Apply FFT to full audio length
            fft_result = np.fft.fft(audio)

            # Take magnitude and keep only positive frequencies (first half)
            magnitude_spectrum = np.abs(fft_result[:len(fft_result) // 2 + 1])

            # Normalize to prevent very large values
            if np.max(magnitude_spectrum) > 0:
                magnitude_spectrum = magnitude_spectrum / np.max(magnitude_spectrum)

            return magnitude_spectrum

        except Exception as e:
            print(f"Warning: Spectrum extraction failed: {e}")
            return np.zeros(len(audio) // 2 + 1 if len(audio) > 0 else 2)

    def _extract_mfcc_from_audio(self, audio: np.ndarray) -> np.ndarray:
        """
        Extract MFCC features from a single audio signal.

        Args:
            audio (np.ndarray): Audio signal

        Returns:
            np.ndarray: MFCC coefficients averaged across time
        """
        try:
            # Skip very short audio
            if len(audio) < 10:
                return np.zeros(self.n_mfcc)

            # Adapt parameters for audio length
            n_fft, hop_length = self._adapt_parameters_for_audio_length(len(audio))

            # Extract MFCC
            mfcc = librosa.feature.mfcc(
                y=audio,
                sr=self.sample_rate,
                n_mfcc=self.n_mfcc,
                n_fft=n_fft,
                hop_length=hop_length
            )

            # Remove invalid frames and average across time
            valid_mask = ~(np.isnan(mfcc[0]) | (np.sum(np.abs(mfcc), axis=0) == 0))

            if not np.any(valid_mask):
                return np.zeros(self.n_mfcc)

            # Average valid MFCC coefficients across time
            valid_mfcc = mfcc[:, valid_mask]
            mfcc_averaged = np.mean(valid_mfcc, axis=1)

            return mfcc_averaged

        except Exception as e:
            print(f"Warning: MFCC extraction failed: {e}")
            return np.zeros(self.n_mfcc)

    def find_wav_files(self, folder_path: str, recording_type: str = "any") -> List[str]:
        """
        Find WAV files in a folder based on recording type.

        Args:
            folder_path (str): Path to search for WAV files
            recording_type (str): Type of recording - "average", "raw", or "any"

        Returns:
            List[str]: List of WAV file paths
        """
        if not os.path.isdir(folder_path):
            return []

        wav_files = []
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)

            # Check if it's a WAV file
            if not (os.path.isfile(file_path) and filename.lower().endswith('.wav')):
                continue

            # Filter based on recording type
            filename_lower = filename.lower()

            if recording_type == "raw":
                if filename_lower.endswith('raw_recording.wav'):
                    wav_files.append(file_path)
            elif recording_type == "average":
                if (filename_lower.endswith('recording.wav') and
                        not filename_lower.endswith('raw_recording.wav')):
                    wav_files.append(file_path)
            else:  # recording_type == "any"
                wav_files.append(file_path)

        return sorted(wav_files)

    def load_audio_file(self, file_path: str) -> Optional[np.ndarray]:
        """
        Load a single audio file.

        Args:
            file_path (str): Path to the audio file

        Returns:
            Optional[np.ndarray]: Audio signal or None if loading failed
        """
        try:
            audio, _ = librosa.load(file_path, sr=self.sample_rate, mono=True)
            return audio
        except Exception as e:
            print(f"Warning: Failed to load {file_path}: {e}")
            return None

    def process_scenario_folder(self, scenario_folder: str, wav_subfolder: str,
                                recording_type: str = "any",
                                mfcc_filename: str = "features.csv",
                                spectrum_filename: str = "spectrum.csv") -> bool:
        """
        Process a single scenario folder and save both MFCC and spectrum CSV files.

        Args:
            scenario_folder (str): Path to scenario folder
            wav_subfolder (str): Name of subfolder containing WAV files
            recording_type (str): Type of recording to process
            mfcc_filename (str): Name of MFCC features CSV file
            spectrum_filename (str): Name of spectrum features CSV file

        Returns:
            bool: True if successful, False otherwise
        """
        # Construct path to WAV files
        wav_folder_path = os.path.join(scenario_folder, wav_subfolder)

        if not os.path.exists(wav_folder_path):
            print(f"WAV subfolder not found: {wav_folder_path}")
            return False

        # Find WAV files
        wav_files = self.find_wav_files(wav_folder_path, recording_type)

        if not wav_files:
            print(f"No {recording_type} WAV files found in {wav_folder_path}")
            return False

        print(f"Processing {len(wav_files)} files in {os.path.basename(scenario_folder)}")

        # Extract features from each file
        mfcc_features_list = []
        spectrum_features_list = []

        for file_path in tqdm(wav_files, desc=f"  Features", leave=False):
            # Load audio
            audio = self.load_audio_file(file_path)
            if audio is None:
                continue

            filename = os.path.basename(file_path)

            # Extract MFCC features
            mfcc_features = self._extract_mfcc_from_audio(audio)
            mfcc_row = {'filename': filename}
            for i, mfcc_val in enumerate(mfcc_features):
                mfcc_row[f'mfcc_{i}'] = float(mfcc_val)
            mfcc_features_list.append(mfcc_row)

            # Extract spectrum features
            spectrum_features = self._extract_spectrum_from_audio(audio)
            spectrum_row = {'filename': filename}
            for i, spec_val in enumerate(spectrum_features):
                spectrum_row[f'freq_{i}'] = float(spec_val)
            spectrum_features_list.append(spectrum_row)

        if not mfcc_features_list or not spectrum_features_list:
            print(f"No features extracted from {scenario_folder}")
            return False

        # Create DataFrames and save both files
        success_count = 0

        # Save MFCC features
        try:
            mfcc_df = pd.DataFrame(mfcc_features_list)
            mfcc_output_path = os.path.join(scenario_folder, mfcc_filename)
            mfcc_df.to_csv(mfcc_output_path, index=False)
            print(f"  → Saved {len(mfcc_df)} MFCC features to {mfcc_filename}")
            success_count += 1
        except Exception as e:
            print(f"  → Failed to save MFCC features: {e}")

        # Save spectrum features
        try:
            spectrum_df = pd.DataFrame(spectrum_features_list)
            spectrum_output_path = os.path.join(scenario_folder, spectrum_filename)
            spectrum_df.to_csv(spectrum_output_path, index=False)
            print(f"  → Saved {len(spectrum_df)} spectrum features to {spectrum_filename}")
            print(f"  → Spectrum length: {len(spectrum_features)} frequency bins")
            success_count += 1
        except Exception as e:
            print(f"  → Failed to save spectrum features: {e}")

        return success_count > 0

    def find_scenario_folders(self, dataset_path: str) -> List[str]:
        """
        Find all scenario folders in the dataset.

        Args:
            dataset_path (str): Path to dataset directory

        Returns:
            List[str]: List of scenario folder paths
        """
        scenario_folders = []

        if not os.path.exists(dataset_path):
            print(f"Dataset path not found: {dataset_path}")
            return scenario_folders

        # Look for folders that match scenario pattern
        for item in os.listdir(dataset_path):
            item_path = os.path.join(dataset_path, item)
            if os.path.isdir(item_path):
                # Check if folder name contains "scenario" (case insensitive)
                if re.search(r'scenario', item, re.IGNORECASE):
                    scenario_folders.append(item_path)

        return sorted(scenario_folders)

    def process_dataset(self, dataset_path: str, wav_subfolder: str,
                        recording_type: str = "any",
                        mfcc_filename: str = "features.csv",
                        spectrum_filename: str = "spectrum.csv",
                        skip_existing: bool = True) -> None:
        """
        Process entire dataset and create both MFCC and spectrum CSV files for each scenario.

        Args:
            dataset_path (str): Path to dataset directory
            wav_subfolder (str): Name of subfolder containing WAV files
            recording_type (str): Type of recording to process
            mfcc_filename (str): Name of MFCC features CSV file for each scenario
            spectrum_filename (str): Name of spectrum features CSV file for each scenario
            skip_existing (bool): Skip scenarios that already have feature files
        """
        print(f"Processing dataset: {dataset_path}")
        print(f"WAV subfolder: {wav_subfolder}")
        print(f"Recording type: {recording_type}")
        print(f"MFCC output filename: {mfcc_filename}")
        print(f"Spectrum output filename: {spectrum_filename}")

        # Find scenario folders
        scenario_folders = self.find_scenario_folders(dataset_path)

        if not scenario_folders:
            print("No scenario folders found!")
            return

        print(f"Found {len(scenario_folders)} scenario folders")

        # Process each scenario
        processed = 0
        skipped = 0
        failed = 0

        for scenario_folder in tqdm(scenario_folders, desc="Processing scenarios"):
            scenario_name = os.path.basename(scenario_folder)

            # Check if feature files already exist
            mfcc_file = os.path.join(scenario_folder, mfcc_filename)
            spectrum_file = os.path.join(scenario_folder, spectrum_filename)

            if skip_existing and os.path.exists(mfcc_file) and os.path.exists(spectrum_file):
                skipped += 1
                print(f"Skipping {scenario_name} - feature files already exist")
                continue

            # Process the scenario
            success = self.process_scenario_folder(
                scenario_folder, wav_subfolder, recording_type,
                mfcc_filename, spectrum_filename
            )

            if success:
                processed += 1
            else:
                failed += 1

        # Print summary
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
    """
    Main entry point for feature extraction.
    """
    parser = argparse.ArgumentParser(
        description='Extract MFCC features for each scenario in dataset'
    )

    # Required arguments
    parser.add_argument('dataset_path',
                        help='Path to dataset directory containing scenario folders')
    parser.add_argument('wav_subfolder',
                        help='Name of subfolder containing WAV files (e.g., "impulse_responses")')

    # Optional arguments
    parser.add_argument('--recording-type', choices=['average', 'raw', 'any'],
                        default='any', help='Type of recording to process (default: any)')
    parser.add_argument('--mfcc-filename', default='features.csv',
                        help='Name of MFCC feature CSV file to create in each scenario (default: features.csv)')
    parser.add_argument('--spectrum-filename', default='spectrum.csv',
                        help='Name of spectrum feature CSV file to create in each scenario (default: spectrum.csv)')
    parser.add_argument('--sample-rate', type=int, default=16000,
                        help='Audio sample rate (default: 16000)')
    parser.add_argument('--n-mfcc', type=int, default=13,
                        help='Number of MFCC coefficients (default: 13)')
    parser.add_argument('--force', action='store_true',
                        help='Overwrite existing feature files')

    args = parser.parse_args()

    # Create feature extractor
    extractor = AudioFeatureExtractor(
        sample_rate=args.sample_rate,
        n_mfcc=args.n_mfcc
    )

    # Process dataset
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