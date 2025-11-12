#!/usr/bin/env python3
"""
Export Piano Point Response Files

Generates text files for each measurement point containing averaged impulse responses
from all 6 channels in the format:
# point=XX  channels=6  N_REF=XXXXX
# columns: ch0	ch1	ch2	ch3	ch4	ch5
<data rows with tab-separated values>

This script will:
1. Check all scenarios in /piano directory
2. Generate missing averaged responses for scenarios that don't have them
3. Export all scenarios to text files
"""

import numpy as np
from pathlib import Path
from typing import Dict, List
from collections import defaultdict

def group_files_by_channel(file_paths: List[Path]) -> Dict[int, List[Path]]:
    """
    Group impulse response files by channel number.

    Returns:
        Dict mapping channel number to list of file paths for that channel
    """
    channels = defaultdict(list)

    for filepath in file_paths:
        filename = filepath.name
        # Extract channel from filename: impulse_..._chN.npy
        if "_ch" in filename:
            try:
                ch_part = filename.split("_ch")[1]
                ch_num = int(ch_part.split(".")[0])
                channels[ch_num].append(filepath)
            except (ValueError, IndexError):
                print(f"      Warning: Could not extract channel from {filename}")

    return channels

def average_channel_signals(file_paths: List[Path]) -> np.ndarray:
    """
    Average multiple signals for a single channel.
    All signals are padded to the same length before averaging.

    Args:
        file_paths: List of paths to .npy files for the same channel

    Returns:
        Averaged signal as numpy array
    """
    signals = []

    # Load all signals
    for filepath in file_paths:
        signal = np.load(filepath)
        signals.append(signal)

    if not signals:
        raise ValueError("No signals to average")

    # Find maximum length
    max_len = max(len(s) for s in signals)

    # Pad all signals to same length with zeros
    padded_signals = []
    for signal in signals:
        if len(signal) < max_len:
            padded = np.pad(signal, (0, max_len - len(signal)), mode='constant')
        else:
            padded = signal
        padded_signals.append(padded)

    # Stack and average
    stacked = np.array(padded_signals)
    averaged = np.mean(stacked, axis=0)

    return averaged

def generate_averaged_responses_for_scenario(scenario_path: Path) -> bool:
    """
    Generate averaged impulse responses for a scenario if they don't exist.

    Args:
        scenario_path: Path to scenario folder

    Returns:
        True if successful or if averages already exist, False otherwise
    """
    scenario_name = scenario_path.name
    impulse_dir = scenario_path / "impulse_responses"
    output_dir = scenario_path / "averaged_responses"

    # Check if averages already exist
    if output_dir.exists():
        avg_files = list(output_dir.glob("average_ch*.npy"))
        if len(avg_files) == 6:
            return True  # Already has all 6 averaged files

    print(f"  Generating averaged responses for {scenario_name}...")

    if not impulse_dir.exists():
        print(f"    ERROR: impulse_responses folder not found!")
        return False

    # Get all .npy files
    npy_files = list(impulse_dir.glob("*.npy"))

    if not npy_files:
        print(f"    ERROR: No .npy files found in impulse_responses!")
        return False

    # Group by channel
    channels = group_files_by_channel(npy_files)

    if not channels:
        print(f"    ERROR: No channel data found!")
        return False

    if len(channels) != 6:
        print(f"    WARNING: Expected 6 channels, found {len(channels)}")

    # Create output directory
    output_dir.mkdir(exist_ok=True)

    # Process each channel
    for ch_num in sorted(channels.keys()):
        ch_files = channels[ch_num]

        try:
            # Average signals for this channel
            averaged_signal = average_channel_signals(ch_files)

            # Save averaged signal
            output_file = output_dir / f"average_ch{ch_num}.npy"
            np.save(output_file, averaged_signal)

        except Exception as e:
            print(f"    ERROR processing channel {ch_num}: {e}")
            return False

    print(f"    Generated {len(channels)} averaged response files")
    return True

def load_averaged_responses(scenario_path: Path, num_channels: int = 6) -> Dict[int, np.ndarray]:
    """
    Load all averaged response files for a scenario.

    Args:
        scenario_path: Path to scenario folder
        num_channels: Number of channels to load (default: 6)

    Returns:
        Dict mapping channel number to averaged signal array
    """
    avg_dir = scenario_path / "averaged_responses"

    if not avg_dir.exists():
        raise FileNotFoundError(f"No averaged_responses folder found in {scenario_path}")

    channels = {}
    for ch in range(num_channels):
        avg_file = avg_dir / f"average_ch{ch}.npy"
        if not avg_file.exists():
            raise FileNotFoundError(f"Missing averaged response file: {avg_file}")

        channels[ch] = np.load(avg_file)

    return channels

def export_point_response_file(
    point_number: int,
    channels: Dict[int, np.ndarray],
    output_file: str
):
    """
    Export point response file in the specified text format.

    Args:
        point_number: Measurement point number
        channels: Dict mapping channel number to signal array
        output_file: Output file path
    """
    num_channels = len(channels)
    channel_nums = sorted(channels.keys())

    # Verify all channels have same length
    lengths = [len(channels[ch]) for ch in channel_nums]
    if len(set(lengths)) != 1:
        raise ValueError(f"Channel lengths don't match: {lengths}")

    N_REF = lengths[0]

    print(f"  Exporting point {point_number}:")
    print(f"    Channels: {channel_nums}")
    print(f"    Length: {N_REF} samples")

    # Write file
    with open(output_file, 'w') as f:
        # Write header
        f.write(f"# point={point_number}  channels={num_channels}  N_REF={N_REF}\n")

        # Write column labels
        col_labels = "\t".join([f"ch{ch}" for ch in channel_nums])
        f.write(f"# columns: {col_labels}\n")

        # Write data rows
        for i in range(N_REF):
            # Get sample from each channel
            row_values = [channels[ch][i] for ch in channel_nums]

            # Format as scientific notation with tab separation
            row_str = "\t".join([f"{val:e}" for val in row_values])
            f.write(row_str + "\n")

    print(f"    Saved: {output_file}")

def main():
    """Main entry point"""
    piano_dir = Path("piano")
    output_dir = Path("piano_point_responses")

    # Create output directory
    output_dir.mkdir(exist_ok=True)

    # Find all scenarios
    scenarios = sorted(piano_dir.glob("Neumann-Scenario*"))

    if not scenarios:
        print("No scenarios found in piano/ folder!")
        return

    print("=" * 80)
    print("EXPORT PIANO POINT RESPONSE FILES")
    print("=" * 80)
    print(f"Found {len(scenarios)} measurement points in piano/ folder")
    print(f"Output directory: {output_dir}")
    print()

    # Step 1: Generate missing averaged responses
    print("=" * 80)
    print("STEP 1: GENERATING MISSING AVERAGED RESPONSES")
    print("=" * 80)

    missing_count = 0
    generated_count = 0

    for scenario_path in scenarios:
        avg_dir = scenario_path / "averaged_responses"

        # Check if averages exist and are complete
        needs_generation = True
        if avg_dir.exists():
            avg_files = list(avg_dir.glob("average_ch*.npy"))
            if len(avg_files) == 6:
                needs_generation = False

        if needs_generation:
            missing_count += 1
            if generate_averaged_responses_for_scenario(scenario_path):
                generated_count += 1

    if missing_count == 0:
        print("All scenarios already have averaged responses!")
    else:
        print(f"\nGenerated averaged responses: {generated_count}/{missing_count} scenarios")

    print()

    # Step 2: Export all scenarios
    print("=" * 80)
    print("STEP 2: EXPORTING POINT RESPONSE FILES")
    print("=" * 80)

    success_count = 0
    failed = []

    for scenario_path in scenarios:
        scenario_name = scenario_path.name

        # Extract point number from scenario name
        # Format: Neumann-Scenario57-Take1 -> point 57
        try:
            point_number = int(scenario_name.split("-Scenario")[1].split("-")[0])
        except (ValueError, IndexError):
            print(f"Warning: Could not extract point number from {scenario_name}")
            continue

        # Output file name
        output_file = output_dir / f"{point_number:02d}.txt"

        print(f"Processing point {point_number} ({scenario_name})...")

        try:
            # Load averaged responses
            channels = load_averaged_responses(scenario_path, num_channels=6)

            # Export to text file
            export_point_response_file(point_number, channels, str(output_file))

            success_count += 1

        except Exception as e:
            print(f"  ERROR: {e}")
            failed.append((point_number, str(e)))

        print()

    # Summary
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Successfully exported: {success_count}/{len(scenarios)} point response files")

    if failed:
        print(f"\nFailed exports:")
        for point, error in failed:
            print(f"  Point {point}: {error}")
    else:
        print(f"\nAll point response files exported successfully!")
        print(f"Location: {output_dir.absolute()}")

    print("=" * 80)

    # Show file sizes
    if success_count > 0:
        print("\nGenerated files:")
        for txt_file in sorted(output_dir.glob("*.txt")):
            size_mb = txt_file.stat().st_size / (1024 * 1024)
            print(f"  {txt_file.name}: {size_mb:.2f} MB")

if __name__ == "__main__":
    main()
