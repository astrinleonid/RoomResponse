"""
test_single_vs_multiband.py
Quick comparison: single-band vs multi-band ESPRIT processing.
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
from band_processing import STANDARD_BANDS, process_band


def main(filepath: str):
    print("="*80)
    print("Single-Band vs Multi-Band Comparison")
    print("="*80)
    print(f"File: {filepath}\n")

    # Load and preprocess
    fs = 48000
    force, responses = load_measurement_file(filepath, skip_channel=2)
    config = MinimalPreprocessingConfig(use_highpass=True, remove_contact=True)
    processed, metadata = preprocess_measurement(force, responses, fs, config)

    print(f"Duration: {metadata['duration_processed_s']:.3f} s")
    print(f"Channels: {metadata['n_channels']}\n")

    # Test 1: Single-band (full range, no preprocessing)
    print("-"*80)
    print("Test 1: Single-Band TLS-ESPRIT (0-500 Hz, no band processing)")
    print("-"*80)

    result_single = esprit_modal_identification(
        processed,
        fs=fs,
        model_order=30,
        freq_range=(0, 500),
        use_tls=True,
        use_conjugate_pairing=True,
        use_multichannel=False,
        min_freq=30.0
    )

    print(f"Modes: {len(result_single.frequencies)}")
    if len(result_single.frequencies) > 0:
        for i, (f, z) in enumerate(zip(result_single.frequencies, result_single.damping_ratios)):
            print(f"  {i}: {f:7.2f} Hz, damping={z*100:5.2f}%")
    print()

    # Test 2: Multi-band with Low band (30-200 Hz)
    print("-"*80)
    print("Test 2: Multi-Band - Low Frequency Band (30-200 Hz)")
    print("-"*80)

    band_low = STANDARD_BANDS[0]  # 30-200 Hz, decimation=4, exp=0.3
    print(f"Band: {band_low.name}")
    print(f"  Range: {band_low.f_min}-{band_low.f_max} Hz")
    print(f"  Decimation: {band_low.decimation}x")
    print(f"  Exp factor: {band_low.exp_factor}")
    print(f"  Filter order: {band_low.filter_order}")

    processed_low, fs_low, meta_low = process_band(processed, fs, band_low, apply_preemphasis=True)
    print(f"  fs_band: {fs_low:.0f} Hz")
    print(f"  Samples: {meta_low['n_samples_decimated']}")

    result_low = esprit_modal_identification(
        processed_low,
        fs=fs_low,
        model_order=30,
        freq_range=(band_low.f_min, band_low.f_max),
        use_tls=True,
        use_conjugate_pairing=True,
        use_multichannel=False,
        min_freq=band_low.f_min
    )

    print(f"\nModes: {len(result_low.frequencies)}")
    if len(result_low.frequencies) > 0:
        for i, (f, z) in enumerate(zip(result_low.frequencies, result_low.damping_ratios)):
            print(f"  {i}: {f:7.2f} Hz, damping={z*100:5.2f}%")
    print()

    # Test 3: Multi-band with Mid-Low band (150-500 Hz)
    print("-"*80)
    print("Test 3: Multi-Band - Mid-Low Frequency Band (150-500 Hz)")
    print("-"*80)

    band_mid = STANDARD_BANDS[1]  # 150-500 Hz, decimation=2, exp=0.2
    print(f"Band: {band_mid.name}")
    print(f"  Range: {band_mid.f_min}-{band_mid.f_max} Hz")
    print(f"  Decimation: {band_mid.decimation}x")

    processed_mid, fs_mid, meta_mid = process_band(processed, fs, band_mid, apply_preemphasis=True)
    print(f"  fs_band: {fs_mid:.0f} Hz")

    result_mid = esprit_modal_identification(
        processed_mid,
        fs=fs_mid,
        model_order=30,
        freq_range=(band_mid.f_min, band_mid.f_max),
        use_tls=True,
        use_conjugate_pairing=True,
        use_multichannel=False,
        min_freq=band_mid.f_min
    )

    print(f"\nModes: {len(result_mid.frequencies)}")
    if len(result_mid.frequencies) > 0:
        for i, (f, z) in enumerate(zip(result_mid.frequencies, result_mid.damping_ratios)):
            print(f"  {i}: {f:7.2f} Hz, damping={z*100:5.2f}%")
    print()

    # Summary
    print("="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Single-band (0-500 Hz, no processing):  {len(result_single.frequencies)} modes")
    print(f"Multi-band Low (30-200 Hz, dec=4):     {len(result_low.frequencies)} modes")
    print(f"Multi-band Mid (150-500 Hz, dec=2):    {len(result_mid.frequencies)} modes")
    print(f"Total from multi-band:                 {len(result_low.frequencies) + len(result_mid.frequencies)} modes")
    print()
    print("Note: Multi-band should find more modes due to:")
    print("  - Bandpass filtering reduces noise from outside band")
    print("  - Exponential pre-emphasis compensates for signal decay")
    print("  - Decimation allows longer effective window in lower bands")
    print("="*80)


if __name__ == "__main__":
    test_file = "piano_point_responses/70.txt"
    if len(sys.argv) > 1:
        test_file = sys.argv[1]

    if not Path(test_file).exists():
        print(f"ERROR: File not found: {test_file}")
        sys.exit(1)

    main(test_file)
