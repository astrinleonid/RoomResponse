"""
test_multipoint.py
Test multi-point stabilization across multiple excitation locations.
This matches the esprit.py approach of spatial stabilization.
"""
import sys
import numpy as np
from pathlib import Path
from glob import glob

sys.path.insert(0, str(Path(__file__).parent))

from preprocessing_minimal import (
    load_measurement_file,
    preprocess_measurement,
    MinimalPreprocessingConfig
)
from esprit_core import esprit_modal_identification
from stabilization import multipoint_stabilization, StableMode


def test_multipoint_stabilization(data_dir: str, pattern: str = "*.txt",
                                  max_files: int = 10):
    """
    Test multi-point stabilization across multiple measurements.

    This simulates the esprit.py workflow where measurements from different
    excitation points are processed and modes are clustered spatially.
    """

    print("="*80)
    print("Multi-Point Stabilization Test")
    print("="*80)
    print()

    # Find measurement files
    search_path = Path(data_dir) / pattern
    measurement_files = sorted(glob(str(search_path)))[:max_files]

    if len(measurement_files) == 0:
        print(f"ERROR: No files found matching {search_path}")
        sys.exit(1)

    print(f"Found {len(measurement_files)} measurement files")
    print(f"Processing first {max_files} files:\n")

    # Parameters
    fs = 48000
    model_order = 30
    freq_range = (0, 500)

    # Load and preprocess all measurements
    print("Loading and preprocessing measurements...")
    measurement_data = []
    valid_files = []

    for i, filepath in enumerate(measurement_files):
        try:
            filename = Path(filepath).name
            print(f"  [{i+1}/{len(measurement_files)}] {filename}...", end=" ")

            # Load
            force, responses = load_measurement_file(filepath, skip_channel=2)

            # Preprocess
            config = MinimalPreprocessingConfig(use_highpass=True, remove_contact=True)
            processed, metadata = preprocess_measurement(force, responses, fs, config)

            print(f"OK ({metadata['duration_processed_s']:.3f}s, {metadata['n_channels']} ch)")

            measurement_data.append(processed)
            valid_files.append(filename)

        except Exception as e:
            print(f"FAILED: {e}")
            continue

    print(f"\nSuccessfully loaded {len(measurement_data)} measurements\n")

    if len(measurement_data) == 0:
        print("ERROR: No valid measurements loaded")
        sys.exit(1)

    # ESPRIT parameters
    esprit_params = {
        'model_order': model_order,
        'freq_range': freq_range,
        'use_tls': True,
        'use_conjugate_pairing': True,
        'use_multichannel': False,  # Single channel for speed
        'min_freq': 30.0
    }

    # Run multi-point stabilization
    print("="*80)
    print("Running Multi-Point Stabilization")
    print("="*80)
    print(f"ESPRIT parameters:")
    print(f"  Model order: {model_order}")
    print(f"  Frequency range: {freq_range[0]}-{freq_range[1]} Hz")
    print(f"  TLS-ESPRIT: {esprit_params['use_tls']}")
    print(f"  Conjugate pairing: {esprit_params['use_conjugate_pairing']}")
    print(f"  Minimum frequency: {esprit_params['min_freq']} Hz")
    print()

    print("Processing each measurement point...")

    stable_modes = multipoint_stabilization(
        measurement_data,
        fs,
        esprit_modal_identification,
        esprit_params,
        freq_tol_hz=2.0,
        damping_tol=0.05,
        min_detections=max(3, len(measurement_data) // 3)  # Require ~1/3 of points
    )

    # Results
    print("\n" + "="*80)
    print("RESULTS: Stable Modes Across Multiple Excitation Points")
    print("="*80)
    print(f"Total measurement points: {len(measurement_data)}")
    print(f"Minimum detections required: {max(3, len(measurement_data) // 3)}")
    print(f"Stable modes identified: {len(stable_modes)}\n")

    if len(stable_modes) > 0:
        print("Mode | Frequency (Hz) |  ±Std  | Damping (%) |  ±Std  | Detections | Detection Rate")
        print("-----|----------------|--------|-------------|--------|------------|---------------")
        for i, mode in enumerate(stable_modes):
            detection_rate = mode.n_detections / len(measurement_data) * 100
            print(f"{i:4d} | {mode.frequency:14.2f} | {mode.std_frequency:6.2f} | "
                  f"{mode.damping*100:11.2f} | {mode.std_damping*100:6.2f} | "
                  f"{mode.n_detections:10d} | {detection_rate:13.1f}%")

        print()
        print("Interpretation:")
        print("  - Detection Rate: % of measurement points where this mode was identified")
        print("  - High detection rate (>70%): Strong, stable physical mode")
        print("  - Medium detection rate (30-70%): Moderate mode, may be location-dependent")
        print("  - Low std: Consistent frequency/damping across points (good)")
        print()

        # Show which files contributed to top 3 modes
        print("-"*80)
        print("Top 3 Modes - Source Analysis")
        print("-"*80)
        for i, mode in enumerate(stable_modes[:3]):
            print(f"\nMode {i}: {mode.frequency:.2f} Hz (detected in {mode.n_detections} points)")
            print(f"  Source files: ", end="")
            source_files = [valid_files[sid] for sid in mode.source_ids]
            print(", ".join(source_files))

    else:
        print("No stable modes found!")
        print()
        print("Possible reasons:")
        print("  - min_detections threshold too high")
        print("  - Measurements have very different modal content")
        print("  - ESPRIT parameters not optimal for this data")

    print("\n" + "="*80)


if __name__ == "__main__":
    # Default: test on piano point response data
    data_dir = "piano_point_responses"
    max_files = 10

    if len(sys.argv) > 1:
        data_dir = sys.argv[1]

    if len(sys.argv) > 2:
        max_files = int(sys.argv[2])

    test_multipoint_stabilization(data_dir, pattern="*.txt", max_files=max_files)
