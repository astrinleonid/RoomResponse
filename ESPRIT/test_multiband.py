"""
test_multiband.py
Test multi-band processing and compare with single-band approach.
"""
import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from preprocessing_minimal import (
    load_measurement_file,
    preprocess_measurement,
    MinimalPreprocessingConfig
)
from esprit_core import esprit_modal_identification
from band_processing import STANDARD_BANDS, process_band, select_bands_for_range
from stabilization import ModeCandidate, stabilize_modes


def test_multiband_single_measurement(filepath: str):
    """Test multi-band processing on a single measurement."""

    print("="*80)
    print("Multi-Band ESPRIT Test")
    print("="*80)
    print(f"File: {filepath}\n")

    # Parameters
    fs = 48000
    model_order = 30
    freq_range = (0, 500)

    # Load and preprocess
    print("Loading and preprocessing...")
    force, responses = load_measurement_file(filepath, skip_channel=2)
    config = MinimalPreprocessingConfig(use_highpass=True, remove_contact=True)
    processed, metadata = preprocess_measurement(force, responses, fs, config)

    print(f"  Duration: {metadata['duration_processed_s']:.3f} s")
    print(f"  Channels: {metadata['n_channels']}\n")

    # Select relevant bands for our frequency range
    selected_bands = select_bands_for_range(freq_range, STANDARD_BANDS)
    print(f"Selected {len(selected_bands)} frequency bands:")
    for band in selected_bands:
        print(f"  {band.name}: {band.f_min}-{band.f_max} Hz, "
              f"decimation={band.decimation}, order={band.filter_order}")
    print()

    # Process each band
    all_candidates = []

    for band_idx, band in enumerate(selected_bands):
        print("-"*80)
        print(f"Band {band_idx+1}/{len(selected_bands)}: {band.name} ({band.f_min}-{band.f_max} Hz)")
        print("-"*80)

        try:
            # Process band
            band_data, fs_band, band_meta = process_band(
                processed, fs, band, apply_preemphasis=True
            )

            print(f"  fs_band: {fs_band:.0f} Hz")
            print(f"  Samples: {band_meta['n_samples_original']} -> {band_meta['n_samples_decimated']}")
            print(f"  Preemphasis: {band_meta['preemphasis_applied']} (factor={band.exp_factor})")

            # Run ESPRIT on this band
            result = esprit_modal_identification(
                band_data,
                fs=fs_band,
                model_order=model_order,
                freq_range=(band.f_min, band.f_max),
                use_tls=True,
                use_conjugate_pairing=True,
                use_multichannel=False,
                min_freq=max(band.f_min, 30.0)
            )

            print(f"  Modes found: {len(result.frequencies)}")

            if len(result.frequencies) > 0:
                print(f"\n  Mode | Frequency (Hz) | Damping (%)")
                print(f"  -----|----------------|------------")
                for i, (f, z) in enumerate(zip(result.frequencies, result.damping_ratios)):
                    print(f"  {i:4d} | {f:14.2f} | {z*100:10.2f}")

                    # Add to candidates for cross-band stabilization
                    candidate = ModeCandidate(
                        frequency=f,
                        damping=z,
                        pole=result.poles[i],
                        mode_shape=result.mode_shapes[i] if result.mode_shapes is not None else None,
                        quality=1.0,
                        source_id=band_idx,
                        band_name=band.name
                    )
                    all_candidates.append(candidate)

            print()

        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
            print()

    # Cross-band stabilization
    print("="*80)
    print("Cross-Band Stabilization")
    print("="*80)
    print(f"Total candidates from all bands: {len(all_candidates)}\n")

    if len(all_candidates) > 0:
        # Stabilize with lenient requirements (min_detections=1 for single measurement)
        stable_modes = stabilize_modes(
            all_candidates,
            freq_tol_hz=2.0,
            damping_tol=0.05,
            min_detections=1  # Single measurement, so require detection in at least 1 band
        )

        print(f"Stable modes identified: {len(stable_modes)}\n")

        if len(stable_modes) > 0:
            print("Mode | Frequency (Hz) | Damping (%) | Std Freq | Std Damp | Detections | Bands")
            print("-----|----------------|-------------|----------|----------|------------|------")
            for i, mode in enumerate(stable_modes):
                bands_str = ",".join(set([all_candidates[sid].band_name
                                         for sid in mode.source_ids]))
                print(f"{i:4d} | {mode.frequency:14.2f} | {mode.damping*100:11.2f} | "
                      f"{mode.std_frequency:8.2f} | {mode.std_damping*100:8.2f} | "
                      f"{mode.n_detections:10d} | {bands_str}")

    print("\n" + "="*80)
    print("Analysis Complete")
    print("="*80)


if __name__ == "__main__":
    test_file = "piano_point_responses/70.txt"

    if len(sys.argv) > 1:
        test_file = sys.argv[1]

    if not Path(test_file).exists():
        print(f"ERROR: File not found: {test_file}")
        sys.exit(1)

    test_multiband_single_measurement(test_file)
