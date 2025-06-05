import os
import shutil
import re
import glob
import argparse
import numpy as np
import pandas as pd
import librosa
from tqdm import tqdm
from copyFolder import copy_specific_subfolders
from extract_features import extract_mfcc_from_array

# Configuration constants
DEFAULT_SAMPLE_RATE = 16000
DEFAULT_MFCC_COEFFICIENTS = 13
AUDIO_PADDING_VALUE = 0.0
METADATA_COLUMNS = ['computer_name', 'scenario_id', 'room_name', 'signal_shape', 'filename']


def load_wav_files(file_paths, sample_rate=DEFAULT_SAMPLE_RATE):
    """
    Load multiple WAV files and return as a padded numpy array.
    
    Args:
        file_paths (list): List of WAV file paths to load
        sample_rate (int): Target sample rate for all files
        
    Returns:
        numpy.ndarray: 2D array where each row is audio from one file
    """
    print(f"Loading {len(file_paths)} WAV files...")
    
    audio_data = []
    max_length = 0
    
    # Load each WAV file
    for file_path in tqdm(file_paths, desc="Loading audio"):
        try:
            # Load audio file (librosa handles resampling and mono conversion)
            audio, _ = librosa.load(file_path, sr=sample_rate, mono=True)
            audio_data.append(audio)
            max_length = max(max_length, len(audio))
            
        except Exception as e:
            print(f"Warning: Failed to load {file_path}: {e}")
            audio_data.append(np.array([]))  # Empty array for failed loads
    
    # Create padded array where all audio has same length
    num_files = len(file_paths)
    padded_audio = np.full((num_files, max_length), AUDIO_PADDING_VALUE, dtype=np.float32)
    
    # Fill with actual audio data
    for i, audio in enumerate(audio_data):
        if len(audio) > 0:
            padded_audio[i, :len(audio)] = audio
    
    print(f"Loaded audio: {num_files} files, max length: {max_length} samples")
    return padded_audio


def find_wav_files(folder_path, recording_type="average"):
    """
    Find all WAV files in a folder that match the recording type.
    
    Args:
        folder_path (str): Path to search in
        recording_type (str): Type of recording - "average" or "raw"
                             "average": files ending with "recording.wav" (but not "raw_recording.wav")
                             "raw": files ending with "raw_recording.wav"
        
    Returns:
        list: List of matching WAV file paths
    """
    if not os.path.isdir(folder_path):
        print(f"Warning: Directory not found: {folder_path}")
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
            # For raw recordings: must end with "raw_recording.wav"
            if filename_lower.endswith('raw_recording.wav'):
                wav_files.append(file_path)
        else:  # recording_type == "average" (default)
            # For average recordings: must end with "recording.wav" but NOT "raw_recording.wav"
            if filename_lower.endswith('recording.wav') and not filename_lower.endswith('raw_recording.wav'):
                wav_files.append(file_path)
    
    return wav_files


def parse_folder_metadata(folder_path):
    """
    Extract metadata from folder name using pattern: <computer>-Scenario<id>-<room>-<shape>
    
    Args:
        folder_path (str): Path to the folder
        
    Returns:
        dict or None: Metadata dictionary or None if parsing fails
    """
    folder_name = os.path.basename(folder_path)
    parts = folder_name.split('-')
    
    if len(parts) < 2:
        print(f"Warning: Invalid folder name format: {folder_name}")
        return None
    
    # Find scenario part using regex
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
    
    # Extract computer name (before scenario)
    computer_name = parts[0].split('_')[-1] if scenario_index > 0 else None
    
    # Extract room and signal shape (after scenario)
    remaining_parts = parts[scenario_index + 1:]
    if not remaining_parts:
        print(f"Warning: No room name found in folder: {folder_name}")
        return None
    
    room_name = remaining_parts[0]
    signal_shape = remaining_parts[1] if len(remaining_parts) > 1 else None
    
    return {
        "computer_name": computer_name,
        "scenario_id": scenario_id,
        "room_name": room_name,
        "signal_shape": signal_shape
    }


def load_existing_dataset(output_file):
    """
    Load existing dataset if it exists.
    
    Args:
        output_file (str): Path to existing CSV file
        
    Returns:
        tuple: (DataFrame or None, set of existing entries)
    """
    if not os.path.exists(output_file):
        return None, set()
    
    try:
        existing_df = pd.read_csv(output_file)
        print(f"Loaded existing dataset: {len(existing_df)} entries")
        
        # Create set of existing entries for duplicate checking
        existing_entries = set()
        for _, row in existing_df.iterrows():
            entry_key = tuple(str(row.get(col, '')) for col in METADATA_COLUMNS)
            existing_entries.add(entry_key)
        
        return existing_df, existing_entries
        
    except Exception as e:
        print(f"Warning: Failed to load existing dataset: {e}")
        return None, set()


def collect_files_to_process(root_folder, recording_type, existing_entries, replace_duplicates):
    """
    Scan folders and collect WAV files that need processing.
    
    Args:
        root_folder (str): Root directory to scan
        recording_type (str): Type of recording - "average" or "raw"
        existing_entries (set): Set of existing dataset entries
        replace_duplicates (bool): Whether to replace existing entries
        
    Returns:
        tuple: (list of file paths, dict mapping file paths to metadata)
    """
    print(f"Scanning folders for {recording_type} WAV files...")
    
    # Find all scenario folders
    scenario_folders = [f for f in glob.glob(os.path.join(root_folder, "*")) if os.path.isdir(f)]
    print(f"Found {len(scenario_folders)} scenario folders")
    
    files_to_process = []
    file_metadata = {}
    total_files_found = 0
    
    for folder_path in scenario_folders:
        # Parse metadata from folder name
        metadata = parse_folder_metadata(folder_path)
        if metadata is None:
            continue
        
        # Look for diagnostics subfolder
        diagnostics_path = os.path.join(folder_path, "diagnostics")
        if not os.path.exists(diagnostics_path):
            print(f"Warning: No diagnostics folder in {folder_path}")
            continue
        
        # Find WAV files in diagnostics folder based on recording type
        wav_files = find_wav_files(diagnostics_path, recording_type)
        total_files_found += len(wav_files)
        
        if not wav_files:
            print(f"Warning: No {recording_type} WAV files found in {diagnostics_path}")
            continue
        
        print(f"Found {len(wav_files)} {recording_type} WAV files in {os.path.basename(folder_path)}")
        
        # Check each file against existing entries
        for file_path in wav_files:
            filename = os.path.basename(file_path)
            entry_key = tuple([
                str(metadata['computer_name']),
                str(metadata['scenario_id']),
                str(metadata['room_name']),
                str(metadata['signal_shape']),
                filename
            ])
            
            # Skip if already exists and not replacing
            if not replace_duplicates and entry_key in existing_entries:
                print(f"Skipping {filename} - already in dataset")
                continue
            
            files_to_process.append(file_path)
            file_metadata[file_path] = metadata
    
    print(f"Total {recording_type} files found: {total_files_found}")
    print(f"Files to process: {len(files_to_process)}")
    
    return files_to_process, file_metadata


def extract_features_from_files(file_paths, file_metadata, sample_rate, n_mfcc):
    """
    Extract MFCC features from WAV files and create dataset rows.
    
    Args:
        file_paths (list): List of WAV file paths
        file_metadata (dict): Mapping of file paths to metadata
        sample_rate (int): Audio sample rate
        n_mfcc (int): Number of MFCC coefficients
        
    Returns:
        list: List of dataset rows (dictionaries)
    """
    if not file_paths:
        return []
    
    # Load all audio files
    audio_array = load_wav_files(file_paths, sample_rate)
    
    # Extract MFCC features
    print("Extracting MFCC features...")
    mfcc_array = extract_mfcc_from_array(audio_array, sample_rate=sample_rate, n_mfcc=n_mfcc)
    
    # Process each file's features
    dataset_rows = []
    for i, file_path in enumerate(file_paths):
        metadata = file_metadata[file_path]
        mfcc_features = mfcc_array[i]
        
        # Remove invalid frames (NaN or all zeros)
        valid_mask = ~(np.isnan(mfcc_features[0]) | (np.sum(np.abs(mfcc_features), axis=0) == 0))
        
        if not np.any(valid_mask):
            print(f"Warning: No valid MFCC data for {os.path.basename(file_path)}")
            continue
        
        # Average valid MFCC coefficients across time
        valid_mfcc = mfcc_features[:, valid_mask]
        mfcc_averaged = np.mean(valid_mfcc, axis=1)
        
        # Create dataset row
        row = {
            'filename': os.path.basename(file_path),
            'computer_name': metadata['computer_name'],
            'scenario_id': metadata['scenario_id'],
            'room_name': metadata['room_name'],
            'signal_shape': metadata['signal_shape'],
            'path': file_path
        }
        
        # Add MFCC coefficients as separate columns
        for j, mfcc_value in enumerate(mfcc_averaged):
            row[f'mfcc_{j}'] = mfcc_value
        
        dataset_rows.append(row)
    
    print(f"Successfully processed {len(dataset_rows)} files")
    return dataset_rows


def combine_with_existing_data(new_rows, existing_df, replace_duplicates):
    """
    Combine new data with existing dataset.
    
    Args:
        new_rows (list): List of new dataset rows
        existing_df (DataFrame or None): Existing dataset
        replace_duplicates (bool): Whether to replace duplicates
        
    Returns:
        DataFrame: Combined dataset
    """
    new_df = pd.DataFrame(new_rows) if new_rows else pd.DataFrame()
    
    # If no existing data, return new data
    if existing_df is None or existing_df.empty:
        return new_df
    
    # If no new data, return existing data
    if new_df.empty:
        return existing_df
    
    if replace_duplicates:
        print("Replacing duplicate entries...")
        # Remove duplicates from existing data based on metadata columns
        for _, new_row in new_df.iterrows():
            # Create condition to match all metadata columns
            conditions = []
            for col in METADATA_COLUMNS:
                if col in existing_df.columns:
                    conditions.append(existing_df[col] == new_row[col])
            
            if conditions:
                # Remove matching rows
                combined_condition = conditions[0]
                for condition in conditions[1:]:
                    combined_condition = combined_condition & condition
                existing_df = existing_df[~combined_condition]
    
    # Combine dataframes
    final_df = pd.concat([existing_df, new_df], ignore_index=True)
    print(f"Combined dataset: {len(final_df)} total entries")
    
    return final_df


def create_mfcc_dataset(root_folder, recording_type="average", sample_rate=DEFAULT_SAMPLE_RATE, 
                       n_mfcc=DEFAULT_MFCC_COEFFICIENTS, output_file="mfcc_dataset.csv", 
                       append=True, replace_duplicates=False):
    """
    Main function to create MFCC dataset from WAV files.
    
    Args:
        root_folder (str): Root directory containing scenario folders
        recording_type (str): Type of recording - "average" or "raw"
        sample_rate (int): Audio sample rate
        n_mfcc (int): Number of MFCC coefficients
        output_file (str): Output CSV file path
        append (bool): Whether to append to existing file
        replace_duplicates (bool): Whether to replace duplicate entries
        
    Returns:
        DataFrame: The created dataset
    """
    print(f"=== Starting MFCC Dataset Creation for {recording_type.upper()} recordings ===")
    
    # Load existing dataset if appending
    existing_df, existing_entries = load_existing_dataset(output_file) if append else (None, set())
    
    # Collect files that need processing
    files_to_process, file_metadata = collect_files_to_process(
        root_folder, recording_type, existing_entries, replace_duplicates
    )
    
    # Extract features from files
    new_rows = extract_features_from_files(files_to_process, file_metadata, sample_rate, n_mfcc)
    
    # Combine with existing data
    final_dataset = combine_with_existing_data(new_rows, existing_df, replace_duplicates)
    
    # Save dataset
    if not final_dataset.empty:
        final_dataset.to_csv(output_file, index=False)
        print(f"Dataset saved to: {output_file}")
        print(f"Total entries: {len(final_dataset)}")
    else:
        print("No data to save")
    
    return final_dataset


def print_dataset_summary(dataset):
    """
    Print a summary of the dataset contents.
    
    Args:
        dataset (DataFrame): The dataset to summarize
    """
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
    
    print("\nFirst few rows:")
    print(dataset.head())


def main():
    """
    Main entry point - handles command line arguments and orchestrates the process.
    """
    parser = argparse.ArgumentParser(
        description='Process WAV files from folder structure and create MFCC dataset'
    )
    parser.add_argument('--download_folder', 
                       default='/Users/leonidastrin/Downloads',
                       help='Source folder containing downloaded data')
    parser.add_argument('--subfolder-name', 
                       default='diagnostics',
                       help='Name of subfolder containing WAV files')
    parser.add_argument('--dataset_file', 
                       default='mfcc_dataset_with_metadata.csv',
                       help='Output CSV file name')
    parser.add_argument('--recording_type', 
                       choices=['average', 'raw'],
                       default='average',
                       help='Type of recording to process: "average" for recording.wav files, "raw" for raw_recording.wav files (default: average)')
    parser.add_argument('-z', '--zip', 
                       action='store_true',
                       help='Process zip files instead of directories')
    
    args = parser.parse_args()
    
    # Set up paths
    source_folder = args.download_folder
    project_folder = os.path.dirname(os.path.abspath(__file__))
    temp_folder = os.path.join(project_folder, "room_data_temp")
    
    print(f"Source folder: {source_folder}")
    print(f"Temporary folder: {temp_folder}")
    print(f"Recording type: {args.recording_type}")
    
    # Copy relevant subfolders to temporary location
    print("Copying relevant folders...")
    copy_specific_subfolders(source_folder, temp_folder, args.subfolder_name, args.zip)
    
    if args.zip:
        print("Finished extracting and copying from zip files")
    else:
        print("Finished copying folders")
    
    # Create MFCC dataset from copied folders
    dataset = create_mfcc_dataset(
        root_folder=temp_folder,
        recording_type=args.recording_type,
        sample_rate=DEFAULT_SAMPLE_RATE,
        n_mfcc=DEFAULT_MFCC_COEFFICIENTS,
        output_file=args.dataset_file,
        append=True,
        replace_duplicates=False
    )
    
    # Clean up temporary folder
    try:
        shutil.rmtree(temp_folder)
        print(f"Cleaned up temporary folder: {temp_folder}")
    except Exception as e:
        print(f"Warning: Could not remove temporary folder: {e}")
    
    # Print results
    print_dataset_summary(dataset)
    print(f"\n=== Process Complete for {args.recording_type.upper()} recordings ===")


if __name__ == "__main__":
    main()