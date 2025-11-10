"""Test stabilized ESPRIT implementation."""
import sys
sys.path.insert(0, str(__file__).rsplit('\\', 1)[0])

import numpy as np
from preprocessing_minimal import load_measurement_file, preprocess_measurement
from esprit_core import esprit_modal_identification

# Test on point_70
force, responses = load_measurement_file('piano_point_responses/point_70_response.txt', skip_channel=2)
processed, meta = preprocess_measurement(force, responses, fs=48000)

print("=" * 70)
print("COMPARISON: Standard vs Stabilized ESPRIT")
print("=" * 70)
print()

# Test 1: Standard (non-stabilized)
print("TEST 1: Standard ESPRIT (single M=30, L=2000)")
print("-" * 70)
modal1 = esprit_modal_identification(
    processed,
    fs=48000,
    model_order=30,
    window_length=2000,
    use_gpu=False,
    max_damping=0.2,
    freq_range=(0, 500),
    ref_sensor=0,
    use_stabilization=False
)
print(f"Modes identified: {len(modal1.frequencies)}")
if len(modal1.frequencies) > 0:
    for i, (f, zeta) in enumerate(zip(modal1.frequencies, modal1.damping_ratios)):
        print(f"  Mode {i+1}: {f:7.2f} Hz, {zeta*100:5.2f}%")
print()

# Test 2: Stabilized
print("TEST 2: Stabilized ESPRIT (grid: L=[1024,1536,2000], M=[20,30,40])")
print("-" * 70)
modal2 = esprit_modal_identification(
    processed,
    fs=48000,
    model_order=30,
    window_length=2000,
    use_gpu=False,
    max_damping=0.2,
    freq_range=(0, 500),
    ref_sensor=0,
    use_stabilization=True
)
print(f"Modes identified: {len(modal2.frequencies)}")
if len(modal2.frequencies) > 0:
    for i, (f, zeta) in enumerate(zip(modal2.frequencies, modal2.damping_ratios)):
        print(f"  Mode {i+1}: {f:7.2f} Hz, {zeta*100:5.2f}%")
print()

# Comparison
print("COMPARISON:")
print("-" * 70)
print(f"Standard:    {len(modal1.frequencies)} modes")
print(f"Stabilized:  {len(modal2.frequencies)} modes")
print()

if len(modal1.frequencies) > 0 and len(modal2.frequencies) > 0:
    print("Frequency differences (stabilized - standard):")
    # Match modes by frequency
    for f2, z2 in zip(modal2.frequencies, modal2.damping_ratios):
        # Find closest mode in standard
        diffs = np.abs(modal1.frequencies - f2)
        if np.min(diffs) < 10:  # Within 10 Hz
            idx = np.argmin(diffs)
            f1 = modal1.frequencies[idx]
            z1 = modal1.damping_ratios[idx]
            print(f"  {f2:7.2f} Hz: Δf={f2-f1:+6.2f} Hz, Δzeta={100*(z2-z1):+6.2f}%")
        else:
            print(f"  {f2:7.2f} Hz: NEW (not in standard)")

print()
print("=" * 70)
