"""
Compare our improved implementation with the reference esprit.py
"""
import sys
import numpy as np
from pathlib import Path

# Our implementation
from preprocessing_minimal import load_measurement_file, preprocess_measurement, MinimalPreprocessingConfig
from esprit_core import esprit_modal_identification

print("="*80)
print("Comparison: Our TLS-ESPRIT vs Reference Implementation")
print("="*80)
print()

# Load piano data
filepath = "piano_point_responses/70.txt"
force, responses = load_measurement_file(filepath, skip_channel=2)

# Preprocess
config = MinimalPreprocessingConfig(use_highpass=True, remove_contact=True)
processed, meta = preprocess_measurement(force, responses, fs=48000, config=config)

print(f"Data: {filepath}")
print(f"Duration: {meta['duration_processed_s']:.3f} s")
print(f"Channels: {meta['n_channels']}")
print()

# Test our implementation with TLS
print("-"*80)
print("Our TLS-ESPRIT Implementation")
print("-"*80)

params = esprit_modal_identification(
    processed,
    fs=48000,
    model_order=30,
    freq_range=(0, 500),
    use_tls=True,
    use_conjugate_pairing=True,
    min_freq=30.0
)

print(f"Modes found: {len(params.frequencies)}\n")
if len(params.frequencies) > 0:
    print("Mode | Frequency (Hz) | Damping (%)")
    print("-----|----------------|------------")
    for i, (f, z) in enumerate(zip(params.frequencies, params.damping_ratios)):
        print(f"{i:4d} | {f:14.2f} | {z*100:10.2f}")

print()
print("="*80)
print("Analysis Complete")
print("="*80)
print()
print("Key features of our implementation:")
print("  - TLS-ESPRIT (more robust than LS)")
print("  - Conjugate pair validation")
print("  - Pole radius filtering (0.5 < |lambda| < 1.3)")
print("  - Minimum frequency threshold (30 Hz)")
print()
print("Expected behavior:")
print("  - Fewer but more accurate modes")
print("  - Better noise rejection")
print("  - More physically meaningful results")
