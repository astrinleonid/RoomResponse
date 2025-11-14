"""
test_tls_improvements.py
Test the improved TLS-ESPRIT implementation with conjugate pairing and radius filtering.

This script compares:
1. Original LS-ESPRIT (use_tls=False, use_conjugate_pairing=False)
2. New TLS-ESPRIT with all improvements (use_tls=True, use_conjugate_pairing=True)
"""
import sys
import numpy as np
from pathlib import Path

# Add ESPRIT folder to path if needed
sys.path.insert(0, str(Path(__file__).parent))

from preprocessing_minimal import (
    load_measurement_file,
    preprocess_measurement,
    MinimalPreprocessingConfig
)
from esprit_core import esprit_modal_identification


def test_single_measurement(filepath: str, output_dir: str = None):
    """Test improvements on a single measurement file."""

    print("="*80)
    print("ESPRIT Algorithm Comparison Test")
    print("="*80)
    print(f"Input file: {filepath}")
    print()

    # Parameters
    fs = 48000
    model_order = 30
    freq_range = (0, 500)

    # Load data
    print("Loading measurement data...")
    force, responses = load_measurement_file(filepath, skip_channel=2)
    print(f"  Force shape: {force.shape}")
    print(f"  Responses shape: {responses.shape}")
    print(f"  Duration: {len(force)/fs:.3f} s")
    print()

    # Preprocess (minimal - no exponential windowing!)
    print("Preprocessing (minimal)...")
    config = MinimalPreprocessingConfig(
        use_highpass=True,
        hp_cut_hz=1.0,
        remove_contact=True,
        contact_tail_fraction=0.03
    )
    processed, metadata = preprocess_measurement(force, responses, fs, config)
    print(f"  Contact removed: {metadata['contact_removed']}")
    if metadata['contact_removed']:
        print(f"  Contact duration: {metadata['contact_duration_ms']:.3f} ms")
    print(f"  Processed samples: {metadata['n_samples_processed']}")
    print(f"  Duration: {metadata['duration_processed_s']:.4f} s")
    print()

    # Test 1: Original LS-ESPRIT (old behavior)
    print("-"*80)
    print("Test 1: Original LS-ESPRIT (baseline)")
    print("-"*80)
    try:
        modal_params_ls = esprit_modal_identification(
            processed,
            fs=fs,
            model_order=model_order,
            use_gpu=False,
            max_damping=0.2,
            freq_range=freq_range,
            use_tls=False,                    # Use LS-ESPRIT
            use_conjugate_pairing=False,       # No pairing
            use_multichannel=False,
            min_freq=0.0                      # No min frequency
        )

        print(f"Modes identified: {len(modal_params_ls.frequencies)}")
        if len(modal_params_ls.frequencies) > 0:
            print("\nIdentified modes:")
            for k, (f, zeta) in enumerate(zip(modal_params_ls.frequencies,
                                             modal_params_ls.damping_ratios)):
                print(f"  Mode {k}: f = {f:8.2f} Hz, zeta = {zeta*100:5.2f} %")
        else:
            print("  No modes found!")
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        modal_params_ls = None

    print()

    # Test 2: New TLS-ESPRIT with all improvements
    print("-"*80)
    print("Test 2: TLS-ESPRIT with Conjugate Pairing + Radius Filtering")
    print("-"*80)
    try:
        modal_params_tls = esprit_modal_identification(
            processed,
            fs=fs,
            model_order=model_order,
            use_gpu=False,
            max_damping=0.2,
            freq_range=freq_range,
            use_tls=True,                     # Use TLS-ESPRIT (robust)
            use_conjugate_pairing=True,       # Validate conjugate pairs
            use_multichannel=False,
            min_freq=30.0                     # Min 30 Hz (reject DC artifacts)
        )

        print(f"Modes identified: {len(modal_params_tls.frequencies)}")
        if len(modal_params_tls.frequencies) > 0:
            print("\nIdentified modes:")
            for k, (f, zeta) in enumerate(zip(modal_params_tls.frequencies,
                                             modal_params_tls.damping_ratios)):
                print(f"  Mode {k}: f = {f:8.2f} Hz, zeta = {zeta*100:5.2f} %")
        else:
            print("  No modes found!")
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        modal_params_tls = None

    print()

    # Test 3: TLS-ESPRIT with multi-channel Hankel
    print("-"*80)
    print("Test 3: TLS-ESPRIT with Multi-Channel Hankel Stacking")
    print("-"*80)
    try:
        modal_params_mc = esprit_modal_identification(
            processed,
            fs=fs,
            model_order=model_order,
            use_gpu=False,
            max_damping=0.2,
            freq_range=freq_range,
            use_tls=True,
            use_conjugate_pairing=True,
            use_multichannel=True,            # Stack all channels
            min_freq=30.0
        )

        print(f"Modes identified: {len(modal_params_mc.frequencies)}")
        if len(modal_params_mc.frequencies) > 0:
            print("\nIdentified modes:")
            for k, (f, zeta) in enumerate(zip(modal_params_mc.frequencies,
                                             modal_params_mc.damping_ratios)):
                print(f"  Mode {k}: f = {f:8.2f} Hz, zeta = {zeta*100:5.2f} %")
        else:
            print("  No modes found!")
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        modal_params_mc = None

    print()

    # Summary comparison
    print("="*80)
    print("SUMMARY")
    print("="*80)
    n_ls = len(modal_params_ls.frequencies) if modal_params_ls else 0
    n_tls = len(modal_params_tls.frequencies) if modal_params_tls else 0
    n_mc = len(modal_params_mc.frequencies) if modal_params_mc else 0

    print(f"Original LS-ESPRIT:                {n_ls} modes")
    print(f"TLS-ESPRIT + Conjugate Pairing:    {n_tls} modes")
    print(f"TLS-ESPRIT + Multi-Channel:        {n_mc} modes")
    print()

    if n_tls > 0 and n_ls > 0:
        improvement = ((n_tls - n_ls) / n_ls) * 100 if n_ls > 0 else 0
        print(f"Improvement: {improvement:+.1f}% change in mode count")

    print("="*80)

    return modal_params_ls, modal_params_tls, modal_params_mc


if __name__ == "__main__":
    # Test on point_70 (as mentioned in the documentation)
    test_file = "piano_point_responses/point_70_response.txt"

    if len(sys.argv) > 1:
        test_file = sys.argv[1]

    if not Path(test_file).exists():
        print(f"ERROR: File not found: {test_file}")
        print()
        print("Usage: python test_tls_improvements.py [measurement_file]")
        print("Default: piano_point_responses/point_70_response.txt")
        sys.exit(1)

    results = test_single_measurement(test_file)
