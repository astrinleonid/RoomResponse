import os
import shutil
import re
import glob
import argparse
import json
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass, asdict
from pathlib import Path
import numpy as np
import pandas as pd
import librosa
from tqdm import tqdm
from abc import ABC, abstractmethod

# Configuration constants
DEFAULT_SAMPLE_RATE = 16000
DEFAULT_MFCC_COEFFICIENTS = 13
AUDIO_PADDING_VALUE = 0.0
METADATA_COLUMNS = ['computer_name', 'scenario_id', 'room_name', 'signal_shape', 'filename']


@dataclass
class FeatureConfig:
    """Configuration for feature extraction parameters."""
    # Common audio parameters
    sample_rate: int = DEFAULT_SAMPLE_RATE

    # MFCC parameters
    n_mfcc: int = DEFAULT_MFCC_COEFFICIENTS
    n_fft: int = 2048
    hop_length: int = 512

    # Spectral features parameters
    n_chroma: int = 12
    n_mel: int = 128

    # Temporal features parameters
    frame_length: int = 2048

    # Zero crossing rate parameters
    zcr_frame_length: int = 2048
    zcr_hop_length: int = 512

    # Spectral contrast parameters
    n_bands: int = 6
    fmin: float = 200.0

    # Tonnetz parameters (harmony features)
    tonnetz_method: str = 'chroma_cqt'

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'FeatureConfig':
        """Create FeatureConfig from dictionary."""
        return cls(**{k: v for k, v in config_dict.items() if k in cls.__annotations__})

    def to_dict(self) -> Dict[str, Any]:
        """Convert FeatureConfig to dictionary."""
        return asdict(self)


@dataclass
class AudioMetadata:
    """Metadata for audio files."""
    filename: str
    computer_name: Optional[str] = None
    scenario_id: Optional[str] = None
    room_name: Optional[str] = None
    signal_shape: Optional[str] = None
    file_path: Optional[str] = None


class FeatureExtractor(ABC):
    """Abstract base class for feature extractors."""

    @abstractmethod
    def extract(self, audio: np.ndarray, config: FeatureConfig) -> Dict[str, float]:
        """Extract features from audio signal."""
        pass

    @abstractmethod
    def get_feature_names(self, config: FeatureConfig) -> List[str]:
        """Get list of feature names this extractor produces."""
        pass


class MFCCExtractor(FeatureExtractor):
    """MFCC (Mel-Frequency Cepstral Coefficients) feature extractor."""

    def extract(self, audio: np.ndarray, config: FeatureConfig) -> Dict[str, float]:
        """Extract MFCC features."""
        try:
            mfccs = librosa.feature.mfcc(
                y=audio,
                sr=config.sample_rate,
                n_mfcc=config.n_mfcc,
                n_fft=config.n_fft,
                hop_length=config.hop_length
            )

            # Remove invalid frames and average across time
            valid_mask = ~(np.isnan(mfccs[0]) | (np.sum(np.abs(mfccs), axis=0) == 0))
            if not np.any(valid_mask):
                return {f'mfcc_{i}': 0.0 for i in range(config.n_mfcc)}

            mfcc_averaged = np.mean(mfccs[:, valid_mask], axis=1)
            return {f'mfcc_{i}': float(mfcc_averaged[i]) for i in range(len(mfcc_averaged))}

        except Exception as e:
            print(f"Warning: MFCC extraction failed: {e}")
            return {f'mfcc_{i}': 0.0 for i in range(config.n_mfcc)}

    def get_feature_names(self, config: FeatureConfig) -> List[str]:
        """Get MFCC feature names."""
        return [f'mfcc_{i}' for i in range(config.n_mfcc)]


class SpectralFeaturesExtractor(FeatureExtractor):
    """Spectral features extractor (centroid, bandwidth, rolloff, flatness)."""

    def extract(self, audio: np.ndarray, config: FeatureConfig) -> Dict[str, float]:
        """Extract spectral features."""
        features = {}

        try:
            # Spectral centroid
            spectral_centroids = librosa.feature.spectral_centroid(
                y=audio, sr=config.sample_rate, hop_length=config.hop_length
            )[0]
            features['spectral_centroid_mean'] = float(np.mean(spectral_centroids))
            features['spectral_centroid_std'] = float(np.std(spectral_centroids))

            # Spectral bandwidth
            spectral_bandwidth = librosa.feature.spectral_bandwidth(
                y=audio, sr=config.sample_rate, hop_length=config.hop_length
            )[0]
            features['spectral_bandwidth_mean'] = float(np.mean(spectral_bandwidth))
            features['spectral_bandwidth_std'] = float(np.std(spectral_bandwidth))

            # Spectral rolloff
            spectral_rolloff = librosa.feature.spectral_rolloff(
                y=audio, sr=config.sample_rate, hop_length=config.hop_length
            )[0]
            features['spectral_rolloff_mean'] = float(np.mean(spectral_rolloff))
            features['spectral_rolloff_std'] = float(np.std(spectral_rolloff))

            # Spectral flatness
            spectral_flatness = librosa.feature.spectral_flatness(
                y=audio, hop_length=config.hop_length
            )[0]
            features['spectral_flatness_mean'] = float(np.mean(spectral_flatness))
            features['spectral_flatness_std'] = float(np.std(spectral_flatness))

        except Exception as e:
            print(f"Warning: Spectral features extraction failed: {e}")
            for name in self.get_feature_names(config):
                features[name] = 0.0

        return features

    def get_feature_names(self, config: FeatureConfig) -> List[str]:
        """Get spectral feature names."""
        return [
            'spectral_centroid_mean', 'spectral_centroid_std',
            'spectral_bandwidth_mean', 'spectral_bandwidth_std',
            'spectral_rolloff_mean', 'spectral_rolloff_std',
            'spectral_flatness_mean', 'spectral_flatness_std'
        ]


class ChromaFeaturesExtractor(FeatureExtractor):
    """Chroma features extractor (pitch class profiles)."""

    def extract(self, audio: np.ndarray, config: FeatureConfig) -> Dict[str, float]:
        """Extract chroma features."""
        try:
            chroma = librosa.feature.chroma_stft(
                y=audio,
                sr=config.sample_rate,
                n_chroma=config.n_chroma,
                hop_length=config.hop_length
            )

            # Average across time
            chroma_mean = np.mean(chroma, axis=1)
            chroma_std = np.std(chroma, axis=1)

            features = {}
            for i in range(config.n_chroma):
                features[f'chroma_{i}_mean'] = float(chroma_mean[i])
                features[f'chroma_{i}_std'] = float(chroma_std[i])

            return features

        except Exception as e:
            print(f"Warning: Chroma features extraction failed: {e}")
            return {name: 0.0 for name in self.get_feature_names(config)}

    def get_feature_names(self, config: FeatureConfig) -> List[str]:
        """Get chroma feature names."""
        names = []
        for i in range(12):  # Standard 12 chroma bins
            names.extend([f'chroma_{i}_mean', f'chroma_{i}_std'])
        return names


class MelSpectrogramExtractor(FeatureExtractor):
    """Mel-spectrogram features extractor."""

    def extract(self, audio: np.ndarray, config: FeatureConfig) -> Dict[str, float]:
        """Extract mel-spectrogram features."""
        try:
            mel_spec = librosa.feature.melspectrogram(
                y=audio,
                sr=config.sample_rate,
                n_mels=config.n_mel,
                hop_length=config.hop_length
            )

            # Convert to log scale
            log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)

            # Statistical features across time
            mel_mean = np.mean(log_mel_spec, axis=1)
            mel_std = np.std(log_mel_spec, axis=1)

            features = {}
            # Use summary statistics instead of all mel bins
            features['mel_spec_mean'] = float(np.mean(mel_mean))
            features['mel_spec_std'] = float(np.mean(mel_std))
            features['mel_spec_min'] = float(np.min(mel_mean))
            features['mel_spec_max'] = float(np.max(mel_mean))
            features['mel_spec_range'] = features['mel_spec_max'] - features['mel_spec_min']

            return features

        except Exception as e:
            print(f"Warning: Mel-spectrogram extraction failed: {e}")
            return {name: 0.0 for name in self.get_feature_names(config)}

    def get_feature_names(self, config: FeatureConfig) -> List[str]:
        """Get mel-spectrogram feature names."""
        return ['mel_spec_mean', 'mel_spec_std', 'mel_spec_min', 'mel_spec_max', 'mel_spec_range']


class TemporalFeaturesExtractor(FeatureExtractor):
    """Temporal features extractor (ZCR, RMS energy, tempo)."""

    def extract(self, audio: np.ndarray, config: FeatureConfig) -> Dict[str, float]:
        """Extract temporal features."""
        features = {}

        try:
            # Zero Crossing Rate
            zcr = librosa.feature.zero_crossing_rate(
                audio,
                frame_length=config.zcr_frame_length,
                hop_length=config.zcr_hop_length
            )[0]
            features['zcr_mean'] = float(np.mean(zcr))
            features['zcr_std'] = float(np.std(zcr))

            # RMS Energy
            rms = librosa.feature.rms(
                y=audio,
                frame_length=config.frame_length,
                hop_length=config.hop_length
            )[0]
            features['rms_mean'] = float(np.mean(rms))
            features['rms_std'] = float(np.std(rms))

            # Tempo estimation
            try:
                tempo, _ = librosa.beat.beat_track(y=audio, sr=config.sample_rate)
                features['tempo'] = float(tempo)
            except:
                features['tempo'] = 0.0

        except Exception as e:
            print(f"Warning: Temporal features extraction failed: {e}")
            for name in self.get_feature_names(config):
                features[name] = 0.0

        return features

    def get_feature_names(self, config: FeatureConfig) -> List[str]:
        """Get temporal feature names."""
        return ['zcr_mean', 'zcr_std', 'rms_mean', 'rms_std', 'tempo']


class SpectralContrastExtractor(FeatureExtractor):
    """Spectral contrast features extractor."""

    def extract(self, audio: np.ndarray, config: FeatureConfig) -> Dict[str, float]:
        """Extract spectral contrast features."""
        try:
            contrast = librosa.feature.spectral_contrast(
                y=audio,
                sr=config.sample_rate,
                n_bands=config.n_bands,
                fmin=config.fmin,
                hop_length=config.hop_length
            )

            # Average across time
            contrast_mean = np.mean(contrast, axis=1)

            features = {}
            for i in range(len(contrast_mean)):
                features[f'spectral_contrast_{i}'] = float(contrast_mean[i])

            return features

        except Exception as e:
            print(f"Warning: Spectral contrast extraction failed: {e}")
            return {name: 0.0 for name in self.get_feature_names(config)}

    def get_feature_names(self, config: FeatureConfig) -> List[str]:
        """Get spectral contrast feature names."""
        return [f'spectral_contrast_{i}' for i in range(7)]  # 6 bands + 1 overall


class TonnetzExtractor(FeatureExtractor):
    """Tonnetz (tonal centroid) features extractor."""

    def extract(self, audio: np.ndarray, config: FeatureConfig) -> Dict[str, float]:
        """Extract tonnetz features."""
        try:
            tonnetz = librosa.feature.tonnetz(
                y=audio,
                sr=config.sample_rate
            )

            # Average across time
            tonnetz_mean = np.mean(tonnetz, axis=1)
            tonnetz_std = np.std(tonnetz, axis=1)

            features = {}
            for i in range(6):  # Tonnetz has 6 dimensions
                features[f'tonnetz_{i}_mean'] = float(tonnetz_mean[i])
                features[f'tonnetz_{i}_std'] = float(tonnetz_std[i])

            return features

        except Exception as e:
            print(f"Warning: Tonnetz extraction failed: {e}")
            return {name: 0.0 for name in self.get_feature_names(config)}

    def get_feature_names(self, config: FeatureConfig) -> List[str]:
        """Get tonnetz feature names."""
        names = []
        for i in range(6):
            names.extend([f'tonnetz_{i}_mean', f'tonnetz_{i}_std'])
        return names


class AudioFeatureExtractor:
    """Main class for extracting multiple types of audio features."""

    def __init__(self, config: Optional[Union[FeatureConfig, Dict[str, Any], str]] = None):
        """
        Initialize the feature extractor.

        Args:
            config: Feature configuration (FeatureConfig object, dict, or path to JSON file)
        """
        self.config = self._load_config(config)

        # Initialize all feature extractors
        self.extractors: Dict[str, FeatureExtractor] = {
            'mfcc': MFCCExtractor(),
            'spectral': SpectralFeaturesExtractor(),
            'chroma': ChromaFeaturesExtractor(),
            'mel': MelSpectrogramExtractor(),
            'temporal': TemporalFeaturesExtractor(),
            'contrast': SpectralContrastExtractor(),
            'tonnetz': TonnetzExtractor()
        }

        self.enabled_extractors = set(self.extractors.keys())

    def _load_config(self, config: Optional[Union[FeatureConfig, Dict[str, Any], str]]) -> FeatureConfig:
        """Load configuration from various sources."""
        if config is None:
            return FeatureConfig()
        elif isinstance(config, FeatureConfig):
            return config
        elif isinstance(config, dict):
            return FeatureConfig.from_dict(config)
        elif isinstance(config, str):
            # Load from JSON file
            with open(config, 'r') as f:
                config_dict = json.load(f)
            return FeatureConfig.from_dict(config_dict)
        else:
            raise ValueError(f"Invalid config type: {type(config)}")

    def enable_extractors(self, extractors: List[str]):
        """Enable specific feature extractors."""
        invalid = set(extractors) - set(self.extractors.keys())
        if invalid:
            raise ValueError(f"Invalid extractors: {invalid}. Available: {list(self.extractors.keys())}")
        self.enabled_extractors = set(extractors)

    def disable_extractors(self, extractors: List[str]):
        """Disable specific feature extractors."""
        self.enabled_extractors -= set(extractors)

    def get_all_feature_names(self) -> List[str]:
        """Get all feature names from enabled extractors."""
        all_names = []
        for name, extractor in self.extractors.items():
            if name in self.enabled_extractors:
                all_names.extend(extractor.get_feature_names(self.config))
        return all_names

    def extract_features_from_audio(self, audio: np.ndarray) -> Dict[str, float]:
        """Extract all enabled features from a single audio signal."""
        all_features = {}

        for name, extractor in self.extractors.items():
            if name in self.enabled_extractors:
                try:
                    features = extractor.extract(audio, self.config)
                    all_features.update(features)
                except Exception as e:
                    print(f"Warning: Failed to extract {name} features: {e}")
                    # Add zero values for failed features
                    feature_names = extractor.get_feature_names(self.config)
                    for feature_name in feature_names:
                        all_features[feature_name] = 0.0

        return all_features

    def load_wav_files(self, file_paths: List[str]) -> np.ndarray:
        """Load multiple WAV files and return as a padded numpy array."""
        print(f"Loading {len(file_paths)} WAV files...")

        audio_data = []
        max_length = 0

        for file_path in tqdm(file_paths, desc="Loading audio"):
            try:
                audio, _ = librosa.load(file_path, sr=self.config.sample_rate, mono=True)
                audio_data.append(audio)
                max_length = max(max_length, len(audio))
            except Exception as e:
                print(f"Warning: Failed to load {file_path}: {e}")
                audio_data.append(np.array([]))

        # Create padded array
        num_files = len(file_paths)
        padded_audio = np.full((num_files, max_length), AUDIO_PADDING_VALUE, dtype=np.float32)

        for i, audio in enumerate(audio_data):
            if len(audio) > 0:
                padded_audio[i, :len(audio)] = audio

        print(f"Loaded audio: {num_files} files, max length: {max_length} samples")
        return padded_audio

    def extract_features_from_files(self, file_paths: List[str],
                                    file_metadata: Dict[str, AudioMetadata]) -> List[Dict[str, Any]]:
        """Extract features from multiple audio files."""
        if not file_paths:
            return []

        # Load all audio files
        audio_array = self.load_wav_files(file_paths)

        # Extract features from each audio signal
        print("Extracting features...")
        dataset_rows = []

        for i, file_path in enumerate(tqdm(file_paths, desc="Processing files")):
            audio = audio_array[i]

            # Skip empty audio
            if len(audio) == 0 or np.all(audio == 0):
                print(f"Warning: Skipping empty audio file: {os.path.basename(file_path)}")
                continue

            # Extract features
            features = self.extract_features_from_audio(audio)

            # Create dataset row
            metadata = file_metadata[file_path]
            row = {
                'filename': metadata.filename,
                'computer_name': metadata.computer_name,
                'scenario_id': metadata.scenario_id,
                'room_name': metadata.room_name,
                'signal_shape': metadata.signal_shape,
                'path': file_path
            }

            # Add all extracted features
            row.update(features)
            dataset_rows.append(row)

        print(f"Successfully processed {len(dataset_rows)} files")
        return dataset_rows


class DatasetManager:
    """Manages dataset creation, loading, and manipulation."""

    def __init__(self, feature_extractor: AudioFeatureExtractor):
        """Initialize with a feature extractor."""
        self.feature_extractor = feature_extractor

    def find_wav_files(self, folder_path: str, recording_type: str = "any") -> List[str]:
        """Find WAV files in a folder based on recording type."""
        if not os.path.isdir(folder_path):
            print(f"Warning: Directory not found: {folder_path}")
            return []

        wav_files = []
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)

            if not (os.path.isfile(file_path) and filename.lower().endswith('.wav')):
                continue

            filename_lower = filename.lower()

            if recording_type == "raw":
                if filename_lower.endswith('raw_recording.wav'):
                    wav_files.append(file_path)
            elif recording_type == "average":
                if (filename_lower.endswith('recording.wav') and
                        not filename_lower.endswith('raw_recording.wav')):
                    wav_files.append(file_path)
            else:  # any
                if filename_lower.endswith('.wav'):
                    wav_files.append(file_path)

        return wav_files

    def parse_folder_metadata(self, folder_path: str) -> Optional[AudioMetadata]:
        """Extract metadata from folder name."""
        folder_name = os.path.basename(folder_path)
        parts = folder_name.split('-')

        if len(parts) < 2:
            print(f"Warning: Invalid folder name format: {folder_name}")
            return None

        # Find scenario part
        scenario_pattern = re.compile(r'[Ss]c.*?rio(\d+(?:\.\d+)?)')
        scenario_index = -1
        scenario_id = None

        for i, part in enumerate(parts):
            match = scenario_pattern.match(part)
            if match:
                scenario_index = i
                scenario_id = match.group(1)
                break

        if scenario_index == -1:
            print(f"Warning: No scenario found in folder: {folder_name}")
            return None

        # Extract metadata
        computer_name = parts[0].split('_')[-1] if scenario_index > 0 else None
        remaining_parts = parts[scenario_index + 1:]

        if not remaining_parts:
            print(f"Warning: No room name found in folder: {folder_name}")
            return None

        room_name = remaining_parts[0]
        signal_shape = remaining_parts[1] if len(remaining_parts) > 1 else None

        return AudioMetadata(
            filename="",  # Will be set per file
            computer_name=computer_name,
            scenario_id=scenario_id,
            room_name=room_name,
            signal_shape=signal_shape
        )

    def collect_files_to_process(self, root_folder: str, recording_type: str,
                                 existing_entries: set, replace_duplicates: bool) -> Tuple[
        List[str], Dict[str, AudioMetadata]]:
        """Collect files that need processing."""
        print(f"Scanning folders for {recording_type} WAV files...")

        scenario_folders = [f for f in glob.glob(os.path.join(root_folder, "*")) if os.path.isdir(f)]
        print(f"Found {len(scenario_folders)} scenario folders")

        files_to_process = []
        file_metadata = {}
        total_files_found = 0

        for folder_path in scenario_folders:
            metadata_template = self.parse_folder_metadata(folder_path)
            if metadata_template is None:
                continue

            # Look for audio files
            audio_folders = ["diagnostics", "impulse_responses"]
            wav_files_path = None

            for audio_folder in audio_folders:
                candidate_path = os.path.join(folder_path, audio_folder)
                if os.path.exists(candidate_path):
                    wav_files_path = candidate_path
                    break

            if wav_files_path is None:
                print(f"Warning: No audio folder found in {folder_path}")
                continue

            wav_files = self.find_wav_files(wav_files_path, recording_type)
            total_files_found += len(wav_files)

            if not wav_files:
                print(f"Warning: No {recording_type} WAV files found in {wav_files_path}")
                continue

            print(f"Found {len(wav_files)} {recording_type} WAV files in {os.path.basename(folder_path)}")

            for file_path in wav_files:
                filename = os.path.basename(file_path)

                # Create metadata for this file
                file_metadata_obj = AudioMetadata(
                    filename=filename,
                    computer_name=metadata_template.computer_name,
                    scenario_id=metadata_template.scenario_id,
                    room_name=metadata_template.room_name,
                    signal_shape=metadata_template.signal_shape,
                    file_path=file_path
                )

                # Check for duplicates
                entry_key = tuple([
                    str(file_metadata_obj.computer_name),
                    str(file_metadata_obj.scenario_id),
                    str(file_metadata_obj.room_name),
                    str(file_metadata_obj.signal_shape),
                    filename
                ])

                if not replace_duplicates and entry_key in existing_entries:
                    print(f"Skipping {filename} - already in dataset")
                    continue

                files_to_process.append(file_path)
                file_metadata[file_path] = file_metadata_obj

        print(f"Total {recording_type} files found: {total_files_found}")
        print(f"Files to process: {len(files_to_process)}")

        return files_to_process, file_metadata

    def load_existing_dataset(self, output_file: str) -> Tuple[Optional[pd.DataFrame], set]:
        """Load existing dataset if it exists."""
        if not os.path.exists(output_file):
            return None, set()

        try:
            existing_df = pd.read_csv(output_file)
            print(f"Loaded existing dataset: {len(existing_df)} entries")

            existing_entries = set()
            for _, row in existing_df.iterrows():
                entry_key = tuple(str(row.get(col, '')) for col in METADATA_COLUMNS)
                existing_entries.add(entry_key)

            return existing_df, existing_entries

        except Exception as e:
            print(f"Warning: Failed to load existing dataset: {e}")
            return None, set()

    def combine_with_existing_data(self, new_rows: List[Dict[str, Any]],
                                   existing_df: Optional[pd.DataFrame],
                                   replace_duplicates: bool) -> pd.DataFrame:
        """Combine new data with existing dataset."""
        new_df = pd.DataFrame(new_rows) if new_rows else pd.DataFrame()

        if existing_df is None or existing_df.empty:
            return new_df

        if new_df.empty:
            return existing_df

        if replace_duplicates:
            print("Replacing duplicate entries...")
            for _, new_row in new_df.iterrows():
                conditions = []
                for col in METADATA_COLUMNS:
                    if col in existing_df.columns:
                        conditions.append(existing_df[col] == new_row[col])

                if conditions:
                    combined_condition = conditions[0]
                    for condition in conditions[1:]:
                        combined_condition = combined_condition & condition
                    existing_df = existing_df[~combined_condition]

        final_df = pd.concat([existing_df, new_df], ignore_index=True)
        print(f"Combined dataset: {len(final_df)} total entries")

        return final_df

    def create_dataset(self, root_folder: str, recording_type: str = "any",
                       output_file: str = "audio_features_dataset.csv",
                       append: bool = True, replace_duplicates: bool = False) -> pd.DataFrame:
        """Create a comprehensive audio features dataset."""
        print(f"=== Starting Audio Features Dataset Creation for {recording_type.upper()} recordings ===")

        # Load existing dataset if appending
        existing_df, existing_entries = self.load_existing_dataset(output_file) if append else (None, set())

        # Collect files to process
        files_to_process, file_metadata = self.collect_files_to_process(
            root_folder, recording_type, existing_entries, replace_duplicates
        )

        # Extract features
        new_rows = self.feature_extractor.extract_features_from_files(files_to_process, file_metadata)

        # Combine with existing data
        final_dataset = self.combine_with_existing_data(new_rows, existing_df, replace_duplicates)

        # Save dataset
        if not final_dataset.empty:
            final_dataset.to_csv(output_file, index=False)
            print(f"Dataset saved to: {output_file}")
            print(f"Total entries: {len(final_dataset)}")
        else:
            print("No data to save")

        return final_dataset

    def print_dataset_summary(self, dataset: pd.DataFrame):
        """Print a summary of the dataset contents."""
        if dataset.empty:
            print("Dataset is empty")
            return

        print("\n=== Dataset Summary ===")
        print(f"Total samples: {len(dataset)}")
        print(f"Unique computers: {dataset['computer_name'].nunique()}")
        print(f"  -> {list(dataset['computer_name'].unique())}")
        print(f"Unique scenarios: {dataset['scenario_id'].nunique()}")
        print(f"  -> {list(dataset['scenario_id'].unique())}")
        print(f"Unique rooms: {dataset['room_name'].nunique()}")
        print(f"  -> {list(dataset['room_name'].unique())}")
        print(f"Unique signal shapes: {dataset['signal_shape'].nunique()}")
        print(f"  -> {list(dataset['signal_shape'].unique())}")

        # Feature statistics
        feature_columns = [col for col in dataset.columns if col not in METADATA_COLUMNS + ['path']]
        print(f"Total features extracted: {len(feature_columns)}")

        print("\nFeature types:")
        feature_types = {}
        for col in feature_columns:
            feature_type = col.split('_')[0]
            if feature_type not in feature_types:
                feature_types[feature_type] = 0
            feature_types[feature_type] += 1

        for feature_type, count in sorted(feature_types.items()):
            print(f"  {feature_type}: {count} features")

        print("\nFirst few rows:")
        print(dataset.head())


def main():
    """Main entry point with enhanced command line interface."""
    parser = argparse.ArgumentParser(
        description='Extract comprehensive audio features and create labeled dataset',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with all features
  python audio_feature_extractor.py --source-folder room_response_dataset

  # Use specific features only
  python audio_feature_extractor.py --features mfcc spectral temporal

  # Use custom configuration
  python audio_feature_extractor.py --config my_config.json

  # Process only raw recordings
  python audio_feature_extractor.py --recording-type raw
        """
    )

    # Input/Output arguments
    parser.add_argument('--source-folder', '--download_folder',
                        default='room_response_dataset',
                        help='Source folder containing audio data')
    parser.add_argument('--output-file', '--dataset_file',
                        default='comprehensive_audio_features.csv',
                        help='Output CSV file name')
    parser.add_argument('--subfolder-name',
                        default='impulse_responses',
                        help='Name of subfolder containing WAV files')

    # Recording type
    parser.add_argument('--recording-type',
                        choices=['average', 'raw', 'any'],
                        default='any',
                        help='Type of recording to process')

    # Feature selection
    parser.add_argument('--features',
                        nargs='*',
                        choices=['mfcc', 'spectral', 'chroma', 'mel', 'temporal', 'contrast', 'tonnetz'],
                        help='Specific features to extract (default: all)')

    # Configuration
    parser.add_argument('--config',
                        help='Path to JSON configuration file for feature extraction')
    parser.add_argument('--sample-rate',
                        type=int,
                        default=DEFAULT_SAMPLE_RATE,
                        help=f'Audio sample rate (default: {DEFAULT_SAMPLE_RATE})')
    parser.add_argument('--n-mfcc',
                        type=int,
                        default=DEFAULT_MFCC_COEFFICIENTS,
                        help=f'Number of MFCC coefficients (default: {DEFAULT_MFCC_COEFFICIENTS})')

    # Processing options
    parser.add_argument('--append',
                        action='store_true',
                        default=True,
                        help='Append to existing dataset (default: True)')
    parser.add_argument('--replace-duplicates',
                        action='store_true',
                        help='Replace duplicate entries in dataset')
    parser.add_argument('--zip', '-z',
                        action='store_true',
                        help='Process zip files instead of directories')

    # Output options
    parser.add_argument('--verbose', '-v',
                        action='store_true',
                        help='Verbose output')
    parser.add_argument('--save-config',
                        help='Save current configuration to JSON file')

    args = parser.parse_args()

    # Create feature configuration
    config_dict = {
        'sample_rate': args.sample_rate,
        'n_mfcc': args.n_mfcc
    }

    # Load additional config from file if provided
    if args.config:
        try:
            with open(args.config, 'r') as f:
                file_config = json.load(f)
            config_dict.update(file_config)
        except Exception as e:
            print(f"Warning: Could not load config file {args.config}: {e}")

    feature_config = FeatureConfig.from_dict(config_dict)

    # Save configuration if requested
    if args.save_config:
        with open(args.save_config, 'w') as f:
            json.dump(feature_config.to_dict(), f, indent=2)
        print(f"Configuration saved to: {args.save_config}")

    # Initialize feature extractor
    feature_extractor = AudioFeatureExtractor(feature_config)

    # Enable specific features if requested
    if args.features:
        feature_extractor.enable_extractors(args.features)
        print(f"Enabled features: {args.features}")
    else:
        print("Using all available features")

    if args.verbose:
        print(f"Feature configuration: {feature_config.to_dict()}")
        print(f"Enabled extractors: {feature_extractor.enabled_extractors}")
        print(f"Total features to extract: {len(feature_extractor.get_all_feature_names())}")

    # Set up paths
    source_folder = args.source_folder
    project_folder = os.path.dirname(os.path.abspath(__file__))
    temp_folder = os.path.join(project_folder, "room_data_temp")

    print(f"Source folder: {source_folder}")
    print(f"Temporary folder: {temp_folder}")
    print(f"Recording type: {args.recording_type}")
    print(f"Output file: {args.output_file}")

    # Copy relevant subfolders if processing from zip or need temp copy
    if args.zip or not os.path.exists(source_folder):
        try:
            from copyFolder import copy_specific_subfolders
            print("Copying relevant folders...")
            copy_specific_subfolders(source_folder, temp_folder, args.subfolder_name, args.zip)
            source_folder = temp_folder

            if args.zip:
                print("Finished extracting and copying from zip files")
            else:
                print("Finished copying folders")
        except ImportError:
            print("Warning: copyFolder module not available, using source folder directly")

    # Create dataset manager and process files
    dataset_manager = DatasetManager(feature_extractor)

    try:
        dataset = dataset_manager.create_dataset(
            root_folder=source_folder,
            recording_type=args.recording_type,
            output_file=args.output_file,
            append=args.append,
            replace_duplicates=args.replace_duplicates
        )

        # Print results
        dataset_manager.print_dataset_summary(dataset)

        if args.verbose and not dataset.empty:
            print("\nFeature statistics:")
            feature_columns = [col for col in dataset.columns if col not in METADATA_COLUMNS + ['path']]
            feature_stats = dataset[feature_columns].describe()
            print(feature_stats)

    finally:
        # Clean up temporary folder if created
        if source_folder == temp_folder and os.path.exists(temp_folder):
            try:
                shutil.rmtree(temp_folder)
                print(f"Cleaned up temporary folder: {temp_folder}")
            except Exception as e:
                print(f"Warning: Could not remove temporary folder: {e}")

    print(f"\n=== Process Complete for {args.recording_type.upper()} recordings ===")


if __name__ == "__main__":
    main()