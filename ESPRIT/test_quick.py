"""
test_quick.py
Quick test of TLS-ESPRIT improvements (single-channel only, faster).
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


def quick_test(filepath: str):
    """Quick comparison test."""

    print("="*80)
    print("Quick ESPRIT Test: LS vs TLS")
    print("="*80)
    print(f"File: {filepath}\n")

    # Parameters
    fs = 48000
    model_order = 30
    freq_range = (0, 500)

    # Load and preprocess
    force, responses = load_measurement_file(filepath, skip_channel=2)
    config = MinimalPreprocessingConfig(
        use_highpass=True,
        hp_cut_hz=1.0,
        remove_contact=True
    )
    processed, metadata = preprocess_measurement(force, responses, fs, config)

    print(f"Duration: {metadata['duration_processed_s']:.3f} s")
    print(f"Channels: {metadata['n_channels']}\n")

    # Test 1: LS-ESPRIT (old)
    print("-"*80)
    print("LS-ESPRIT (Old Method)")
    print("-"*80)

    params_ls = esprit_modal_identification(
        processed, fs=fs, model_order=model_order,
        freq_range=freq_range,
        use_tls=False,  # Old LS method
        use_conjugate_pairing=False,
        min_freq=0.0
    )

    print(f"Modes found: {len(params_ls.frequencies)}")
    for i, (f, z) in enumerate(zip(params_ls.frequencies, params_ls.damping_ratios)):
        print(f"  {i}: {f:7.2f} Hz, ζ={z*100:5.2f}%")

    # Test 2: TLS-ESPRIT (new)
    print("\n" + "-"*80)
    print("TLS-ESPRIT (New Method)")
    print("-"*80)

    params_tls = esprit_modal_identification(
        processed, fs=fs, model_order=model_order,
        freq_range=freq_range,
        use_tls=True,  # New TLS method
        use_conjugate_pairing=True,
        min_freq=30.0
    )

    print(f"Modes found: {len(params_tls.frequencies)}")
    for i, (f, z) in enumerate(zip(params_tls.frequencies, params_tls.damping_ratios)):
        print(f"  {i}: {f:7.2f} Hz, ζ={z*100:5.2f}%")

    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"LS-ESPRIT:  {len(params_ls.frequencies)} modes")
    print(f"TLS-ESPRIT: {len(params_tls.frequencies)} modes")

    if len(params_ls.frequencies) > 0:
        change = ((len(params_tls.frequencies) - len(params_ls.frequencies)) /
                 len(params_ls.frequencies) * 100)
        print(f"Change: {change:+.1f}%")

    print("="*80)


if __name__ == "__main__":
    test_file = "piano_point_responses/70.txt"
    if len(sys.argv) > 1:
        test_file = sys.argv[1]

    if not Path(test_file).exists():
        print(f"File not found: {test_file}")
        sys.exit(1)

    quick_test(test_file)
