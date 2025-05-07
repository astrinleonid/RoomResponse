
import os
import shutil
import re
import glob
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from copyFolder import copy_specific_subfolders
from extract_features import extract_mfcc_from_array

METADATA = ['computer_name','scenario_id','room_name','signal_shape']


def combine_files_to_numpy_array(file_list, pad_value=np.nan):
    """
    Reads multiple text files containing float values (one per line) and
    combines them into a single NumPy array where each row represents data from one file.
    Pads shorter files with a specified pad_value to match the length of the longest file.

    Args:
        file_list (list): List of paths to text files
        pad_value: Value to use for padding (default: np.nan)

    Returns:
        numpy.ndarray: A 2D NumPy array where each row contains values from one file
    """
    # First, determine the maximum length by reading all files
    file_data = []
    max_length = 0

    values_array = []

    print("Extracting data from files ...")
    for file_path in tqdm(file_list):
        try:
            with open(file_path, 'r') as file:
                # Read lines from the file and convert to float
                values = []
                for line in file:
                    if line.startswith('#'):
                        continue
                    try:
                        value = float(line.split(',')[1].strip())
                        values.append(value)
                    except:
                        print("Warning: line with no value")
                max_length = max(max_length, len(values))
                values_array.append(values)
        except FileNotFoundError:
            print(f"Warning: File not found: {file_path}")
            file_data.append([])  # Add empty list to maintain corresponding indices
        except ValueError as e:
            print(f"Warning: Error parsing values in {file_path}: {e}")
            file_data.append([])  # Add empty list to maintain corresponding indices

    # Create a 2D array with proper padding
    result_array = np.full((len(file_list), max_length), pad_value)

    # Fill the array with actual data
    for i, values in enumerate(values_array):
        if values:  # Only process non-empty lists
            result_array[i, :len(values)] = values

    return result_array

def extract_files_with_substring(folder_path, substring):
    """
    Extract all files from a folder that contain a specific substring in their names.

    Args:
        folder_path (str): Path to the folder to search in
        substring (str): Substring to look for in file names

    Returns:
        list: List of file paths that contain the substring
    """
    matching_files = []

    # Check if the folder exists
    if not os.path.isdir(folder_path):
        print(f"Error: The folder '{folder_path}' does not exist.")
        return matching_files

    # Walk through all files in the folder
    for file_name in os.listdir(folder_path):
        # Get the full file path
        file_path = os.path.join(folder_path, file_name)

        # Check if it's a file (not a directory) and contains the substring
        if os.path.isfile(file_path) and substring.lower() in file_name.lower():
            matching_files.append(file_path)

    return matching_files


def extract_metadata_from_path(folder_path):
    """
    Extract metadata from folder name in the format: <computer name>-Scenario<scenario ID>-<Room name>

    Args:
        folder_path (str): Path to the folder

    Returns:
        dict: Dictionary containing computer_name, scenario_id, and room_name
    """
    # Get the folder name from the path

    parts = folder_path.split('-')

    if len(parts) < 2:
        print(f"Not enough parts in folder name: {folder_path}")
        return None

    # Find which part contains "scenario"
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
        print(f"No scenario pattern found in: {folder_path}")
        return None

    # Determine computer name (if any)
    computer_name = None
    if scenario_index > 0:
        computer_name = parts[0].split('_')[-1]

    # Determine room name and signal shape
    remaining_parts = parts[scenario_index + 1:]

    if not remaining_parts:
        print(f"No room name found in: {folder_path}")
        return None

    room_name = remaining_parts[0]

    # Signal shape is optional
    signal_shape = None
    if len(remaining_parts) > 1:
        signal_shape = remaining_parts[1]

    return {
        "computer_name": computer_name,
        "scenario_id": scenario_id,
        "room_name": room_name,
        "signal_shape": signal_shape
    }

def create_mfcc_dataset_hierarchical(root_folder, substring='recording',
                                     sample_rate=22050, n_mfcc=13,
                                     output_file='mfcc_dataset_with_metadata.csv',
                                     append=True, replace_duplicates=False):
    """
    Creates a dataset of MFCC features from a hierarchical folder structure
    with metadata extracted from folder names. Can append to existing dataset.
    Optimized to avoid unnecessary MFCC extraction for existing entries.

    Args:
        root_folder (str): Path to the root folder containing subfolders
        substring (str): Substring to filter files by name
        sample_rate (int): Sample rate of audio files
        n_mfcc (int): Number of MFCC coefficients to extract
        output_file (str): Path to save the CSV dataset
        append (bool): If True, append to existing file; if False, create new file
        replace_duplicates (bool): If True, replace existing entries with same metadata; if False, keep both

    Returns:
        pandas.DataFrame: The created or updated dataset
    """
    existing_df = None
    existing_entries = set()

    # Check if output file exists and load it if needed
    if append and os.path.exists(output_file):
        print(f"Loading existing dataset from {output_file}")
        existing_df = pd.read_csv(output_file)
        print(f"Loaded {len(existing_df)} existing entries")

        # Create a set of existing entries based on metadata
        if not replace_duplicates:
            # Only create this set if we're not replacing entries
            for _, row in existing_df.iterrows():
                entry_key = (
                    str(row.get('computer_name', '')),
                    str(row.get('scenario_id', '')),
                    str(row.get('room_name', '')),
                    str(row.get('signal_shape', '')),
                    str(row.get('filename', ''))
                )
                existing_entries.add(entry_key)

    # Find all subfolders that might contain scenario data
    scenario_folders = [f for f in glob.glob(os.path.join(root_folder, "*"))
                        if os.path.isdir(f)]

    total_files_found = 0
    total_files_processed = 0
    new_data = []

    # First pass: collect metadata and determine which files need processing
    files_to_process = []
    file_metadata = {}

    for scenario_folder in scenario_folders:
        # Extract metadata from folder name
        metadata = extract_metadata_from_path(scenario_folder)

        if metadata is None:
            print(f"Warning: Could not extract metadata from {scenario_folder}. Skipping.")
            continue

        # Check if there's a diagnostics subfolder
        diagnostics_folder = os.path.join(scenario_folder, "diagnostics")
        if not os.path.exists(diagnostics_folder):
            print(f"Warning: No diagnostics folder found in {scenario_folder}")
            continue

        # Get recording files from diagnostics folder
        recording_files = extract_files_with_substring(diagnostics_folder, substring)

        if not recording_files:
            print(f"Warning: No recording files found in {diagnostics_folder}")
            continue

        print(f"Found {len(recording_files)} recording files in {diagnostics_folder}")
        total_files_found += len(recording_files)

        # Check each file against existing entries
        for file_path in recording_files:
            filename = os.path.basename(file_path)
            entry_key = (
                str(metadata['computer_name']),
                str(metadata['scenario_id']),
                str(metadata['room_name']),
                str(metadata['signal_shape']),
                filename
            )

            # Skip if entry exists and we're not replacing
            if not replace_duplicates and entry_key in existing_entries:
                print(f"Skipping {filename} - already exists in dataset")
                continue

            # If we're replacing or entry doesn't exist, add to processing list
            files_to_process.append(file_path)
            file_metadata[file_path] = metadata

    print(f"Found {total_files_found} total files, {len(files_to_process)} need processing")

    # Second pass: process only the necessary files
    if files_to_process:
        # Read audio data only for files that need processing
        audio_array = combine_files_to_numpy_array(files_to_process)

        # Extract MFCC features
        print("Extracting MFCC coefficients ...")
        mfcc_array = extract_mfcc_from_array(audio_array, sample_rate=sample_rate, n_mfcc=n_mfcc)

        # Process MFCC features and add to dataset
        for i, file_path in enumerate(files_to_process):
            metadata = file_metadata[file_path]
            mfcc = mfcc_array[i]

            # Remove NaN frames
            mask = ~np.isnan(mfcc[0])
            if not np.any(mask):
                print(f"Warning: No valid MFCC data for file {os.path.basename(file_path)}. Skipping.")
                continue

            valid_mfcc = mfcc[:, mask]

            # Average across time frames to get a fixed-length feature vector
            mfcc_avg = np.mean(valid_mfcc, axis=1)

            # Create a row with metadata, filename, and features
            row = {
                'filename': os.path.basename(file_path),
                'computer_name': metadata['computer_name'],
                'scenario_id': metadata['scenario_id'],
                'room_name': metadata['room_name'],
                'signal_shape': metadata['signal_shape'],
                'path': file_path  # Store full path for reference
            }

            # Add each MFCC coefficient as a separate column
            for j in range(len(mfcc_avg)):
                row[f'mfcc_{j}'] = mfcc_avg[j]

            new_data.append(row)
            total_files_processed += 1

    # Create DataFrame from new data
    new_df = pd.DataFrame(new_data) if new_data else pd.DataFrame()

    # Combine with existing data if appending
    if append and existing_df is not None and not existing_df.empty:
        if replace_duplicates and not new_df.empty:
            # Define metadata columns to check for duplicates
            metadata_cols = ['computer_name', 'scenario_id', 'room_name', 'signal_shape', 'filename']

            print("Replacing duplicates based on metadata...")
            # Remove entries from existing_df that match new data
            for _, new_row in new_df.iterrows():
                match_condition = True
                for col in metadata_cols:
                    if col in new_row and col in existing_df:
                        match_condition = match_condition & (existing_df[col] == new_row[col])

                # Drop matching rows from existing_df
                if not match_condition.empty:
                    existing_df = existing_df.loc[~match_condition]

            # Concatenate remaining existing data with new data
            final_df = pd.concat([existing_df, new_df], ignore_index=True)

            # Report how many were replaced
            num_replaced = len(existing_df) - len(final_df) + len(new_df)
            print(f"Replaced {num_replaced} duplicate entries")
        else:
            print("Appending new entries to existing dataset...")
            # Simply concatenate the dataframes
            final_df = pd.concat([existing_df, new_df], ignore_index=True)
    else:
        # Just use the new data or empty DataFrame if no new data
        final_df = new_df if not new_df.empty else pd.DataFrame()

    print(f"Total files found: {total_files_found}")
    print(f"Total files processed: {total_files_processed}")
    print(f"Total samples in dataset: {len(final_df)}")

    # Save to CSV
    if not final_df.empty:
        final_df.to_csv(output_file, index=False)
        print(f"Dataset saved to {output_file}")
    else:
        print("No data to save.")

    return final_df


def main():
    parser = argparse.ArgumentParser(description='Copy specific sub-subfolders from a directory structure')
    parser.add_argument('--download_folder', default='/Users/leonidastrin/Downloads',
                        help='Folder where downloaded data is stored')
    parser.add_argument('--subfolder-name', default='diagnostics',
                        help='Name of the sub-subfolder to copy (default: diagnostics)')
    parser.add_argument('--dataset_file', default='mfcc_dataset_with_metadata.csv',
                        help='Name of the dataset file (default: mfcc_dataset_with_metadata.csv)')
    parser.add_argument('-z', '--zip', action='store_true',
                        help='Process zip files instead of directories')


    args = parser.parse_args()

    source = args.download_folder
    project_folder =  os.path.dirname(os.path.abspath(__file__))
    destination = os.path.join(project_folder, "room_data_temp")
    print(f"Copying data from {source} to {destination}")
    copy_specific_subfolders(source, destination, args.subfolder_name, args.zip)

    if args.zip:
        print(f"Finished extracting and copying all matching '{args.subfolder_name}' folders")
    else:
        print(f"Finished copying all matching '{args.subfolder_name}' folders")

    dataset = create_mfcc_dataset_hierarchical(
        root_folder=destination,
        substring='recording',
        sample_rate=16000,  # Adjust based on your audio files
        n_mfcc=13,
        output_file=args.dataset_file
    )
    try:
        shutil.rmtree(destination)
        print(f"Cleaned up temporary directory: {destination}")
    except Exception as e:
        print(f"Warning: Could not remove temporary directory: {e}")

    print(dataset.head())
    print("\nDataset Summary:")
    print(f"Unique computers: {dataset['computer_name'].nunique()} \n {dataset['computer_name'].unique()}")
    print(f"Unique signals: {dataset['signal_shape'].nunique()} \n {dataset['signal_shape'].unique()}")
    print(f"Unique scenarios: {dataset['scenario_id'].nunique()} \n {dataset['scenario_id'].unique()}")
    print(f"Unique rooms: {dataset['room_name'].nunique()} \n {dataset['room_name'].unique()}")


if __name__ == "__main__":
    main()