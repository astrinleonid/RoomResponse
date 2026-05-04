#!/usr/bin/env python3
"""
Test ESPRIT processing directly from .npy impulse response files.
"""

import numpy as np
from pathlib import Path
from ESPRIT.esprit_streaming import StreamingESPRITProcessor

# Scenario directory
scenario_dir = Path("piano/Ivers&Pond-Scenario88-Take1")
impulse_dir = scenario_dir / "impulse_responses"

print(f"Testing ESPRIT on: {scenario_dir}")
print("=" * 70)

# Get all .npy files
npy_files = sorted(impulse_dir.glob("impulse_*.npy"))
print(f"Found {len(npy_files)} .npy files")

# Group by channel
channels_dict = {}  # {channel_idx: [file1, file2, ...]}
for file_path in npy_files:
    # Extract channel from filename: ..._ch0.npy
    ch_idx = int(file_path.stem.split("_ch")[-1])
    channels_dict.setdefault(ch_idx, []).append(file_path)

print(f"Detected {len(channels_dict)} channels")
for ch_idx, files in sorted(channels_dict.items()):
    print(f"  Channel {ch_idx}: {len(files)} measurements")

# Average each channel
print("\nAveraging measurements per channel...")
averaged_channels = {}
for channel_idx, channel_files in sorted(channels_dict.items()):
    signals = []

    for file_path in channel_files:
        data = np.load(file_path)
        signals.append(data)

    if not signals:
        continue

    # Find minimum length and truncate all to match
    min_length = min(len(s) for s in signals)
    truncated_signals = [s[:min_length] for s in signals]

    # Average across all measurements
    averaged_signal = np.mean(truncated_signals, axis=0)
    averaged_channels[channel_idx] = averaged_signal

    print(f"  Channel {channel_idx}: averaged {len(signals)} measurements -> {len(averaged_signal)} samples")

# Stack channels into (M_out, N_use) format
channel_arrays = [averaged_channels[i] for i in sorted(averaged_channels.keys())]
y_raw = np.vstack(channel_arrays)  # Shape: (num_channels, samples)

print(f"\nAveraged response shape: {y_raw.shape}")
print(f"Sample rate: 48000 Hz")
print(f"Duration: {y_raw.shape[1] / 48000:.3f} seconds")

# ESPRIT configuration
M_out = y_raw.shape[0]
N_use = y_raw.shape[1]
fs = 48000

print(f"\n{'='*70}")
print("Processing with ESPRIT (all 4 bands)...")
print(f"{'='*70}")

# Process all 4 bands
band_names = [
    "band0_40-500Hz",
    "band1_500-1000Hz",
    "band2_1000-2000Hz",
    "band3_2000-4000Hz"
]

all_results = {}
total_modes = 0

for band_idx, band_name in enumerate(band_names):
    print(f"\n--- Processing Band {band_idx} ({band_name}) ---")

    try:
        processor = StreamingESPRITProcessor(
            M_out=M_out,
            N_use=N_use,
            fs=fs,
            band_index=band_idx,
            L_fraction=0.5,
            K=30,
            skip_m=2
        )

        result = processor.process_measurement(r_index=0, y_raw=y_raw)

        num_modes = len(result.get('frequencies', []))
        all_results[band_name] = {
            'num_modes': num_modes,
            'frequencies': result.get('frequencies', []),
            'damping_ratios': result.get('damping_ratios', []),
            'Q_factors': result.get('Q_factors', [])
        }

        total_modes += num_modes

        print(f"  OK {num_modes} modes detected")
        if num_modes > 0:
            freqs = result['frequencies'][:5]
            print(f"    Top frequencies: {[f'{f:.1f}' for f in freqs]} Hz")

    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()

print(f"\n{'='*70}")
print(f"ESPRIT PROCESSING COMPLETE")
print(f"{'='*70}")
print(f"Total modes across all bands: {total_modes}")
print("\nBreakdown by band:")
for band_name, result in all_results.items():
    print(f"  {band_name}: {result['num_modes']} modes")

# Save results
import json
analysis_dir = scenario_dir / "analysis"
analysis_dir.mkdir(exist_ok=True)

for band_name, result in all_results.items():
    output_file = analysis_dir / f"esprit_{band_name}.json"

    save_data = {
        'scenario_name': scenario_dir.name,
        'band_name': band_name,
        'num_modes': result['num_modes'],
        'frequencies': result['frequencies'].tolist() if hasattr(result['frequencies'], 'tolist') else result['frequencies'],
        'damping_ratios': result['damping_ratios'].tolist() if hasattr(result['damping_ratios'], 'tolist') else result['damping_ratios'],
        'Q_factors': result['Q_factors'].tolist() if hasattr(result['Q_factors'], 'tolist') else result['Q_factors']
    }

    with open(output_file, 'w') as f:
        json.dump(save_data, f, indent=2)

    print(f"  Saved: {output_file.name}")

# Save summary
summary_file = analysis_dir / "esprit_all_bands_summary.json"
summary = {
    'scenario_name': scenario_dir.name,
    'processing_mode': 'multi-band',
    'total_modes_all_bands': total_modes,
    'modes_per_band': {band: result['num_modes'] for band, result in all_results.items()}
}

with open(summary_file, 'w') as f:
    json.dump(summary, f, indent=2)

print(f"  Saved: {summary_file.name}")

print(f"\n{'='*70}")
print("SUCCESS! All results saved to analysis/ directory")
print(f"{'='*70}")
