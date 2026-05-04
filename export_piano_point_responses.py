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

    # Check if averages already exist (we'll regenerate if needed)
    if output_dir.exists():
        avg_files = list(output_dir.glob("average_ch*.npy"))
        if avg_files:
            return True  # Already has averaged files

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

def load_averaged_responses(scenario_path: Path) -> Dict[int, np.ndarray]:
    """
    Load all averaged response files for a scenario.

    Args:
        scenario_path: Path to scenario folder

    Returns:
        Dict mapping channel number to averaged signal array
    """
    avg_dir = scenario_path / "averaged_responses"

    if not avg_dir.exists():
        raise FileNotFoundError(f"No averaged_responses folder found in {scenario_path}")

    # Find all available averaged channel files
    avg_files = list(avg_dir.glob("average_ch*.npy"))
    if not avg_files:
        raise FileNotFoundError(f"No averaged response files found in {avg_dir}")

    channels = {}
    for avg_file in avg_files:
        # Extract channel number from filename: average_chN.npy
        try:
            ch_num = int(avg_file.stem.replace("average_ch", ""))
            channels[ch_num] = np.load(avg_file)
        except ValueError:
            print(f"Warning: Could not parse channel number from {avg_file.name}")

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
    # Remap channels: source -> destination
    # 0 (calibration) -> 2, 2 -> 0, 3 -> 1, 4 -> 3, 5 -> 4, 6 -> 5
    channel_map = {0: 2, 2: 0, 3: 1, 4: 3, 5: 4, 6: 5}

    remapped = {}
    for src, dst in channel_map.items():
        if src in channels:
            remapped[dst] = channels[src]

    num_channels = len(remapped)
    channel_nums = sorted(remapped.keys())

    # Verify all channels have same length
    lengths = [len(remapped[ch]) for ch in channel_nums]
    if len(set(lengths)) != 1:
        raise ValueError(f"Channel lengths don't match: {lengths}")

    N_REF = lengths[0]

    print(f"  Exporting point {point_number}:")
    print(f"    Remapped channels: {channel_nums} (from {sorted(channels.keys())})")
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
            row_values = [remapped[ch][i] for ch in channel_nums]

            # Format as scientific notation with tab separation
            row_str = "\t".join([f"{val:e}" for val in row_values])
            f.write(row_str + "\n")

    print(f"    Saved: {output_file}")

def export_point_responses(
    scenario_paths: List[Path],
    output_dir: Path,
    progress_callback=None
) -> dict:
    """
    Export point responses for a list of scenarios.

    Args:
        scenario_paths: List of scenario directory paths to export
        output_dir: Directory to save exported text files
        progress_callback: Optional callback function(current, total, message)

    Returns:
        Dict with export results and statistics
    """
    output_dir = Path(output_dir)

    # Create output directory
    output_dir.mkdir(exist_ok=True)

    scenarios = [Path(p) for p in scenario_paths]

    if not scenarios:
        return {
            'success': False,
            'error': 'No scenarios provided',
            'exported_count': 0,
            'failed_count': 0,
            'failed': []
        }

    # Step 1: Generate missing averaged responses
    missing_count = 0
    generated_count = 0

    for i, scenario_path in enumerate(scenarios):
        if progress_callback:
            progress_callback(i, len(scenarios) * 2, f"Checking averaged responses for {scenario_path.name}")

        avg_dir = scenario_path / "averaged_responses"

        # Check if averages exist
        needs_generation = True
        if avg_dir.exists():
            avg_files = list(avg_dir.glob("average_ch*.npy"))
            if avg_files:
                needs_generation = False

        if needs_generation:
            missing_count += 1
            if generate_averaged_responses_for_scenario(scenario_path):
                generated_count += 1

    # Step 2: Export all scenarios
    success_count = 0
    failed = []

    for i, scenario_path in enumerate(scenarios):
        scenario_name = scenario_path.name

        if progress_callback:
            progress_callback(len(scenarios) + i, len(scenarios) * 2, f"Exporting {scenario_name}")

        # Extract point number from scenario name
        try:
            point_number = int(scenario_name.split("-Scenario")[1].split("-")[0])
        except (ValueError, IndexError):
            failed.append((scenario_name, "Could not extract point number"))
            continue

        # Output file name
        output_file = output_dir / f"{point_number:02d}.txt"

        try:
            # Load averaged responses
            channels = load_averaged_responses(scenario_path)

            # Export to text file
            export_point_response_file(point_number, channels, str(output_file))

            success_count += 1

        except Exception as e:
            failed.append((scenario_name, str(e)))

    # Return summary
    return {
        'success': success_count > 0,
        'exported_count': success_count,
        'failed_count': len(failed),
        'failed': failed,
        'generated_avg_count': generated_count,
        'missing_avg_count': missing_count,
        'output_dir': str(output_dir.absolute())
    }


def main():
    """Main entry point"""
    piano_dir = Path("piano")
    output_dir = Path("belarus_responses")

    # Create output directory
    output_dir.mkdir(exist_ok=True)

    # Find all scenarios
    scenarios = sorted(piano_dir.glob("Belarus-Scenario*"))

    if not scenarios:
        print("No scenarios found in piano/ folder!")
        return

    print("=" * 80)
    print("EXPORT PIANO POINT RESPONSE FILES")
    print("=" * 80)
    print(f"Found {len(scenarios)} measurement points in piano/ folder")
    print(f"Output directory: {output_dir}")
    print()

    # Export using the reusable function
    def progress_print(current, total, message):
        print(f"[{current}/{total}] {message}")

    result = export_point_responses(scenarios, output_dir, progress_callback=progress_print)

    # Summary
    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Successfully exported: {result['exported_count']}/{len(scenarios)} point response files")

    if result['failed']:
        print(f"\nFailed exports:")
        for point, error in result['failed']:
            print(f"  {point}: {error}")
    else:
        print(f"\nAll point response files exported successfully!")
        print(f"Location: {output_dir.absolute()}")

    print("=" * 80)

    # Show file sizes
    if result['exported_count'] > 0:
        print("\nGenerated files:")
        for txt_file in sorted(output_dir.glob("*.txt")):
            size_mb = txt_file.stat().st_size / (1024 * 1024)
            print(f"  {txt_file.name}: {size_mb:.2f} MB")

if __name__ == "__main__":
    main()
