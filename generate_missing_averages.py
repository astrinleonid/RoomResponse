#!/usr/bin/env python3
"""
Generate Missing Averaged Responses for Piano Scenarios

This script creates averaged impulse responses for scenarios that don't have them yet.
It processes multi-channel data by averaging each channel separately.
"""

import os
import numpy as np
from pathlib import Path
from collections import defaultdict
from typing import Dict, List

def load_npy_file(filepath: str) -> np.ndarray:
    """Load a .npy file"""
    return np.load(filepath)

def save_npy_file(filepath: str, data: np.ndarray):
    """Save data to .npy file"""
    np.save(filepath, data)

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
                print(f"Warning: Could not extract channel from {filename}")

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
        signal = load_npy_file(filepath)
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

def generate_averaged_responses_for_scenario(scenario_path: Path, dry_run: bool = False):
    """
    Generate averaged impulse responses for a scenario.

    Args:
        scenario_path: Path to scenario folder
        dry_run: If True, only show what would be done without actually creating files
    """
    scenario_name = scenario_path.name
    impulse_dir = scenario_path / "impulse_responses"
    output_dir = scenario_path / "averaged_responses"

    print(f"\nProcessing: {scenario_name}")
    print(f"  Input: {impulse_dir}")

    if not impulse_dir.exists():
        print(f"  ERROR: impulse_responses folder not found!")
        return False

    # Get all .npy files
    npy_files = list(impulse_dir.glob("*.npy"))

    if not npy_files:
        print(f"  ERROR: No .npy files found in impulse_responses!")
        return False

    print(f"  Found {len(npy_files)} impulse response files")

    # Group by channel
    channels = group_files_by_channel(npy_files)

    if not channels:
        print(f"  ERROR: No channel data found!")
        return False

    print(f"  Detected {len(channels)} channels: {sorted(channels.keys())}")

    # Show file counts per channel
    for ch_num in sorted(channels.keys()):
        print(f"    Channel {ch_num}: {len(channels[ch_num])} measurements")

    if dry_run:
        print(f"  [DRY RUN] Would create averaged_responses folder")
        for ch_num in sorted(channels.keys()):
            print(f"  [DRY RUN] Would create: average_ch{ch_num}.npy")
        return True

    # Create output directory
    output_dir.mkdir(exist_ok=True)
    print(f"  Output: {output_dir}")

    # Process each channel
    for ch_num in sorted(channels.keys()):
        ch_files = channels[ch_num]
        print(f"  Processing channel {ch_num}...")

        try:
            # Average signals for this channel
            averaged_signal = average_channel_signals(ch_files)

            # Save averaged signal
            output_file = output_dir / f"average_ch{ch_num}.npy"
            save_npy_file(str(output_file), averaged_signal)

            print(f"    Saved: {output_file.name} (length: {len(averaged_signal)} samples)")

            # Show statistics
            peak = np.max(np.abs(averaged_signal))
            rms = np.sqrt(np.mean(averaged_signal ** 2))
            print(f"    Stats: peak={peak:.4f}, rms={rms:.6f}")

        except Exception as e:
            print(f"    ERROR processing channel {ch_num}: {e}")
            return False

    print(f"  SUCCESS: Generated {len(channels)} averaged response files")
    return True

def main():
    """Main entry point"""
    piano_dir = Path("piano")

    # Scenarios that need averaged responses
    missing_scenarios = [
        "Neumann-Scenario57-Take1",
        "Neumann-Scenario65-Take1",
        "Neumann-Scenario82-Take1"
    ]

    print("=" * 80)
    print("GENERATE MISSING AVERAGED RESPONSES FOR PIANO SCENARIOS")
    print("=" * 80)

    # Check if scenarios exist
    scenarios_to_process = []
    for scenario_name in missing_scenarios:
        scenario_path = piano_dir / scenario_name
        if scenario_path.exists():
            scenarios_to_process.append(scenario_path)
            print(f"Found: {scenario_name}")
        else:
            print(f"WARNING: {scenario_name} not found!")

    if not scenarios_to_process:
        print("\nNo scenarios to process!")
        return

    print(f"\nWill process {len(scenarios_to_process)} scenarios")

    # Ask for confirmation
    response = input("\nProceed with generating averaged responses? (yes/no): ").strip().lower()
    if response not in ['yes', 'y']:
        print("Cancelled.")
        return

    # Process each scenario
    print("\n" + "=" * 80)
    print("PROCESSING SCENARIOS")
    print("=" * 80)

    success_count = 0
    for scenario_path in scenarios_to_process:
        if generate_averaged_responses_for_scenario(scenario_path):
            success_count += 1

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Successfully processed: {success_count}/{len(scenarios_to_process)} scenarios")
    print("=" * 80)

if __name__ == "__main__":
    main()
