"""Test the improved ESPRIT implementation."""
import sys
sys.path.insert(0, str(__file__).rsplit('\\', 1)[0])

import numpy as np
from preprocessing_minimal import load_measurement_file, preprocess_measurement
from esprit_core import esprit_modal_identification

# Test on point_70
force, responses = load_measurement_file('piano_point_responses/point_70_response.txt', skip_channel=2)
processed, meta = preprocess_measurement(force, responses, fs=48000)

print("=" * 70)
print("IMPROVED ESPRIT TEST - Point 70")
print("=" * 70)
print(f"Processed shape: {processed.shape}")
print(f"Contact removed: {meta['contact_duration_ms']:.3f} ms\n")

# Test with window_length=2000 (known to work)
window_length = 2000
print(f"Window length: {window_length}")
print(f"Model order: 30")
print()

modal_params = esprit_modal_identification(
    processed,
    fs=48000,
    model_order=30,
    window_length=window_length,
    use_gpu=False,
    max_damping=0.2,
    freq_range=(0, 500),
    ref_sensor=0  # Will be auto-selected
)

print(f"Modes identified: {len(modal_params.frequencies)}")
print(f"Actual model order used: {modal_params.model_order}")
print()

if len(modal_params.frequencies) > 0:
    print("Modal Parameters:")
    print(f"{'Mode':>4} {'Freq (Hz)':>10} {'Damping (%)':>12} {'|Pole|':>10}")
    print("-" * 40)
    for i, (f, zeta, pole) in enumerate(zip(
        modal_params.frequencies,
        modal_params.damping_ratios,
        modal_params.poles
    )):
        pole_mag = np.abs(pole)
        print(f"{i+1:4d} {f:10.2f} {zeta*100:12.2f} {pole_mag:10.2f}")

    print()
    print(f"Frequency range: {modal_params.frequencies.min():.1f} - {modal_params.frequencies.max():.1f} Hz")
    print(f"Mean damping: {modal_params.damping_ratios.mean()*100:.2f}%")
    print(f"Damping range: {modal_params.damping_ratios.min()*100:.2f} - {modal_params.damping_ratios.max()*100:.2f}%")

    # Check singular values for subspace quality
    print()
    print("Singular values (first 10):")
    for i, sv in enumerate(modal_params.singular_values[:10]):
        print(f"  σ_{i+1:2d} = {sv:10.2f}")
else:
    print("WARNING: No modes identified!")

print()
print("=" * 70)
