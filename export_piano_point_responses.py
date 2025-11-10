#!/usr/bin/env python3
"""
Export Piano Point Response Files

Generates text files for each measurement point containing averaged impulse responses
from all 6 channels in the format:
# point=XX  channels=6  N_REF=XXXXX
# columns: ch0	ch1	ch2	ch3	ch4	ch5
<data rows with tab-separated values>
"""

import numpy as np
from pathlib import Path
from typing import Dict, List

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
    print(f"Found {len(scenarios)} measurement points")
    print(f"Output directory: {output_dir}")
    print()

    # Process each scenario
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
        output_file = output_dir / f"point_{point_number:02d}_response.txt"

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
        for txt_file in sorted(output_dir.glob("point_*_response.txt")):
            size_mb = txt_file.stat().st_size / (1024 * 1024)
            print(f"  {txt_file.name}: {size_mb:.2f} MB")

if __name__ == "__main__":
    main()
