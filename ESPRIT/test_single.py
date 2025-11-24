"""Quick test of ESPRIT on single file."""
import sys
sys.path.insert(0, str(__file__).rsplit('\\', 1)[0])

from preprocessing_minimal import load_measurement_file, preprocess_measurement
from esprit_core import esprit_modal_identification

# Load and preprocess
force, responses = load_measurement_file('piano_point_responses/point_70_response.txt', skip_channel=2)
processed, meta = preprocess_measurement(force, responses, fs=48000)

print(f"Processed shape: {processed.shape}")
print(f"Contact end: {meta['contact_end_sample']} samples ({meta['contact_duration_ms']:.3f} ms)")

# Run ESPRIT with fixed window length
window_length = 2000
print(f"Window length: {window_length}")

modal_params = esprit_modal_identification(
    processed,
    fs=48000,
    model_order=30,
    window_length=window_length,
    use_gpu=False,
    max_damping=0.2,
    freq_range=(0, 500),
    ref_sensor=0
)

print(f"\nModes identified: {len(modal_params.frequencies)}")
if len(modal_params.frequencies) > 0:
    print("Frequencies (Hz):")
    for i, (f, zeta) in enumerate(zip(modal_params.frequencies, modal_params.damping_ratios)):
        print(f"  Mode {i+1}: {f:7.2f} Hz, damping {zeta*100:5.2f}%")
