#!/usr/bin/env python3
"""
Piano Soundboard Measurement Analysis
Analyzes collected multi-channel impulse response data from piano scenarios
"""

import os
import json
from pathlib import Path
from collections import defaultdict

def analyze_piano_scenarios(piano_dir="piano"):
    """Analyze all piano measurement scenarios"""

    piano_path = Path(piano_dir)
    scenarios = []

    # Find all scenario folders
    for scenario_folder in sorted(piano_path.glob("Neumann-Scenario*")):
        scenario_name = scenario_folder.name

        # Extract scenario number
        scenario_num = scenario_name.split("-Scenario")[1].split("-")[0]

        # Count impulse response files
        impulse_dir = scenario_folder / "impulse_responses"
        if impulse_dir.exists():
            npy_files = list(impulse_dir.glob("*.npy"))
            total_files = len(npy_files)

            # Group by measurement index to count channels
            measurements = defaultdict(list)
            for npy_file in npy_files:
                # Extract measurement index from filename
                # Format: impulse_Neumann-Scenario57-Take1_000_20251105_181455_319_ch0.npy
                parts = npy_file.stem.split("_")
                try:
                    # Find the measurement index (3 digits before timestamp)
                    for i, part in enumerate(parts):
                        if part.isdigit() and len(part) == 3:
                            meas_idx = int(part)
                            measurements[meas_idx].append(npy_file.name)
                            break
                except (ValueError, IndexError):
                    continue

            num_measurements = len(measurements)
            channels_per_measurement = len(measurements[list(measurements.keys())[0]]) if measurements else 0

            # Load metadata
            metadata_file = scenario_folder / "metadata" / "session_metadata.json"
            metadata = {}
            if metadata_file.exists():
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)

            scenarios.append({
                'scenario_number': int(scenario_num),
                'folder_name': scenario_name,
                'total_files': total_files,
                'num_measurements': num_measurements,
                'num_channels': channels_per_measurement,
                'metadata': metadata
            })

    return scenarios

def print_summary(scenarios):
    """Print summary of piano scenarios"""

    print("=" * 80)
    print("PIANO SOUNDBOARD MEASUREMENT SUMMARY")
    print("=" * 80)
    print()

    if not scenarios:
        print("No scenarios found in piano/ folder")
        return

    print(f"Total measurement points on piano soundboard: {len(scenarios)}")
    print()

    print("MEASUREMENT POINTS (Scenario numbers correspond to physical locations):")
    print("-" * 80)
    print(f"{'Point':<8} {'Measurements':<15} {'Channels':<10} {'Total Files':<12} {'Folder'}")
    print("-" * 80)

    total_measurements = 0
    total_files = 0

    for scenario in scenarios:
        print(f"{scenario['scenario_number']:<8} "
              f"{scenario['num_measurements']:<15} "
              f"{scenario['num_channels']:<10} "
              f"{scenario['total_files']:<12} "
              f"{scenario['folder_name']}")
        total_measurements += scenario['num_measurements']
        total_files += scenario['total_files']

    print("-" * 80)
    print(f"{'TOTAL':<8} {total_measurements:<15} {'':<10} {total_files:<12}")
    print()

    # Print detailed info for first scenario
    if scenarios and scenarios[0].get('metadata'):
        print("=" * 80)
        print("EXAMPLE RECORDING CONFIGURATION (Point {})".format(scenarios[0]['scenario_number']))
        print("=" * 80)
        metadata = scenarios[0]['metadata']

        if 'recorder_config' in metadata:
            config = metadata['recorder_config']
            print(f"Sample Rate: {config.get('sample_rate', 'N/A')} Hz")
            print(f"Pulse Duration: {config.get('pulse_duration', 'N/A')} seconds")
            print(f"Pulse Form: {config.get('impulse_form', 'N/A')}")
            print(f"Cycle Duration: {config.get('cycle_duration', 'N/A')} seconds")
            print(f"Number of Pulses: {config.get('num_pulses', 'N/A')}")
            print(f"Volume: {config.get('volume', 'N/A')}")
            print()

        if 'multichannel_config' in metadata.get('recorder_config', {}):
            mc_config = metadata['recorder_config']['multichannel_config']
            print("Multi-Channel Configuration:")
            print(f"  Enabled: {mc_config.get('enabled', False)}")
            print(f"  Number of Channels: {mc_config.get('num_channels', 'N/A')}")
            print(f"  Calibration Channel: {mc_config.get('calibration_channel', 'N/A')}")
            print(f"  Reference Channel: {mc_config.get('reference_channel', 'N/A')}")
            if 'channel_names' in mc_config:
                print("  Channel Names:")
                for i, name in enumerate(mc_config['channel_names']):
                    print(f"    ch{i}: {name}")
            print()

    print("=" * 80)
    print("DATA FORMAT:")
    print("=" * 80)
    print("Files: NumPy .npy format (binary arrays)")
    print("Naming: impulse_{scenario}_{index}_{timestamp}_ch{N}.npy")
    print("Location: piano/{scenario}/impulse_responses/")
    print()
    print("Each measurement point contains:")
    print("  - Multiple measurements (repetitions at same location)")
    print("  - Multiple channels per measurement (6 microphone positions)")
    print("  - Synchronized multi-channel impulse responses")
    print("=" * 80)

if __name__ == "__main__":
    scenarios = analyze_piano_scenarios("piano")
    print_summary(scenarios)
