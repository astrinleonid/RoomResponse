"""
Analyze consistency of modes detected by stabilization across measurement points.

This script processes the comprehensive comparison results to see if the additional
modes found by stabilization (LS+stab) are consistently detected across different
measurement locations or if they're spurious/location-specific.
"""
import numpy as np
import json
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt

# Load all measurement files
DATA_DIR = Path("piano_point_responses")
mp_files = sorted(DATA_DIR.glob("point_*_response.json"))

print("="*80)
print("STABILIZATION MODE CONSISTENCY ANALYSIS")
print("="*80)
print(f"Analyzing {len(mp_files)} measurement points")
print()

# Import necessary functions
import sys
sys.path.insert(0, 'ESPRIT')
from esprit_core import esprit_modal_identification

# Parameters
fs = 5000.0
dt = 1.0 / fs
L = 140
M = 12
freq_band = (40.0, 500.0)

# Storage for all detected modes
all_modes_nostab = []  # TLS without stabilization
all_modes_stab = []    # LS with stabilization
point_names = []

print("Processing measurement points...")
print("-" * 80)

for mp_file in mp_files:
    point_name = mp_file.stem
    point_names.append(point_name)

    # Load measurement data
    with open(mp_file, 'r') as f:
        mp_data = json.load(f)

    signals = np.array(mp_data['data'], dtype=np.float64)
    if signals.ndim == 1:
        signals = signals.reshape(-1, 1)

    # Apply frequency band filter
    from scipy.signal import butter, filtfilt
    b, a = butter(4, [freq_band[0], freq_band[1]], btype='band', fs=fs)
    signals_filtered = filtfilt(b, a, signals, axis=0)

    # Test 1: TLS without stabilization
    result_nostab = esprit_modal_identification(
        signals=signals_filtered,
        fs=fs,
        window_length=L,
        model_order=M,
        use_stabilization=False,
        use_tls=True,
        use_gpu=False,
        use_conjugate_pairing=True,
        max_damping=0.2,
        freq_range=freq_band
    )

    # Test 2: LS with stabilization
    result_stab = esprit_modal_identification(
        signals=signals_filtered,
        fs=fs,
        window_length=L,
        model_order=M,
        use_stabilization=True,
        use_tls=False,  # LS for stabilization
        use_gpu=False,
        use_conjugate_pairing=True,
        max_damping=0.2,
        freq_range=freq_band
    )

    all_modes_nostab.append(result_nostab.frequencies)
    all_modes_stab.append(result_stab.frequencies)

    n_nostab = len(result_nostab.frequencies)
    n_stab = len(result_stab.frequencies)
    n_extra = n_stab - n_nostab

    print(f"{point_name:<25}: TLS={n_nostab:2d} modes, LS+stab={n_stab:2d} modes, extra={n_extra:+3d}")

print()
print("="*80)
print("FREQUENCY HISTOGRAM ANALYSIS")
print("="*80)

# Create frequency bins (10 Hz resolution)
freq_bins = np.arange(freq_band[0], freq_band[1] + 10, 10)
freq_centers = (freq_bins[:-1] + freq_bins[1:]) / 2

# Count how many points detect a mode in each frequency bin
hist_nostab = np.zeros(len(freq_bins) - 1)
hist_stab = np.zeros(len(freq_bins) - 1)

for freqs in all_modes_nostab:
    counts, _ = np.histogram(freqs, bins=freq_bins)
    hist_nostab += counts

for freqs in all_modes_stab:
    counts, _ = np.histogram(freqs, bins=freq_bins)
    hist_stab += counts

# Additional modes histogram (only in stab, not in nostab)
hist_extra = hist_stab - hist_nostab

print(f"\nFrequency bins with extra modes from stabilization:")
print(f"{'Freq (Hz)':<12} {'TLS count':<12} {'LS+stab count':<15} {'Extra':<10} {'Consistency'}")
print("-" * 80)

for i, (fc, cnt_nostab, cnt_stab, cnt_extra) in enumerate(zip(freq_centers, hist_nostab, hist_stab, hist_extra)):
    if cnt_extra > 0:
        consistency = cnt_extra / len(mp_files) * 100
        print(f"{fc:>6.0f}-{fc+10:>6.0f}   {cnt_nostab:>6.0f}       {cnt_stab:>6.0f}          {cnt_extra:>6.0f}     {consistency:>5.1f}%")

print()
print("="*80)
print("MODE CLUSTERING ANALYSIS")
print("="*80)

# Cluster all modes across all points (5 Hz tolerance)
def cluster_frequencies(freq_list, tol=5.0):
    """Cluster frequencies across multiple measurements."""
    all_freqs = np.concatenate(freq_list)

    if len(all_freqs) == 0:
        return [], []

    # Sort frequencies
    sorted_idx = np.argsort(all_freqs)
    sorted_freqs = all_freqs[sorted_idx]

    # Cluster by tolerance
    clusters = []
    current_cluster = [sorted_freqs[0]]

    for f in sorted_freqs[1:]:
        if f - current_cluster[-1] <= tol:
            current_cluster.append(f)
        else:
            clusters.append(current_cluster)
            current_cluster = [f]

    clusters.append(current_cluster)

    # Compute cluster centers and counts
    cluster_centers = [np.mean(c) for c in clusters]
    cluster_counts = [len(c) for c in clusters]

    return cluster_centers, cluster_counts

# Cluster modes
centers_nostab, counts_nostab = cluster_frequencies(all_modes_nostab, tol=5.0)
centers_stab, counts_stab = cluster_frequencies(all_modes_stab, tol=5.0)

print(f"\nConsistent modes (detected in ≥80% of measurements):")
print(f"{'Algorithm':<20} {'Freq (Hz)':<12} {'Count':<10} {'Consistency'}")
print("-" * 80)

threshold = 0.8 * len(mp_files)

print("\nTLS (no stabilization):")
for fc, cnt in zip(centers_nostab, counts_nostab):
    if cnt >= threshold:
        consistency = cnt / len(mp_files) * 100
        print(f"  {fc:>8.1f} Hz       {cnt:>4d}/{len(mp_files):<4d}   {consistency:>5.1f}%")

print("\nLS + stabilization:")
for fc, cnt in zip(centers_stab, counts_stab):
    if cnt >= threshold:
        consistency = cnt / len(mp_files) * 100
        print(f"  {fc:>8.1f} Hz       {cnt:>4d}/{len(mp_files):<4d}   {consistency:>5.1f}%")

# Identify modes ONLY found by stabilization
print()
print("="*80)
print("MODES UNIQUE TO STABILIZATION")
print("="*80)

unique_to_stab = []
for fc_stab, cnt_stab in zip(centers_stab, counts_stab):
    # Check if this frequency is NOT in nostab clusters
    found_in_nostab = any(abs(fc_stab - fc_nostab) < 5.0 for fc_nostab in centers_nostab)

    if not found_in_nostab:
        unique_to_stab.append((fc_stab, cnt_stab))

if unique_to_stab:
    print(f"\nModes found ONLY by stabilization (not in TLS):")
    print(f"{'Freq (Hz)':<12} {'Count':<10} {'Consistency'}")
    print("-" * 60)

    for fc, cnt in unique_to_stab:
        consistency = cnt / len(mp_files) * 100
        marker = "***" if consistency >= 80 else "**" if consistency >= 50 else "*" if consistency >= 30 else ""
        print(f"{fc:>8.1f} Hz     {cnt:>4d}/{len(mp_files):<4d}   {consistency:>5.1f}%  {marker}")

    print("\n*** = Highly consistent (≥80%)")
    print("**  = Moderately consistent (≥50%)")
    print("*   = Somewhat consistent (≥30%)")
else:
    print("\nNo unique modes found - stabilization only confirms TLS modes")

# Visualization
print()
print("="*80)
print("Generating visualization...")

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

# Plot 1: Mode count per measurement point
x = np.arange(len(point_names))
width = 0.35

ax1.bar(x - width/2, [len(f) for f in all_modes_nostab], width,
        label='TLS (no stab)', alpha=0.7, color='steelblue')
ax1.bar(x + width/2, [len(f) for f in all_modes_stab], width,
        label='LS + stab', alpha=0.7, color='coral')

ax1.set_xlabel('Measurement Point')
ax1.set_ylabel('Number of Modes')
ax1.set_title('Mode Count: TLS vs LS+Stabilization')
ax1.set_xticks(x)
ax1.set_xticklabels([p.replace('point_', '').replace('_response', '')
                      for p in point_names], rotation=45, ha='right')
ax1.legend()
ax1.grid(axis='y', alpha=0.3)

# Plot 2: Frequency histogram comparison
ax2.bar(freq_centers - 2.5, hist_nostab, width=5,
        label='TLS (no stab)', alpha=0.7, color='steelblue')
ax2.bar(freq_centers + 2.5, hist_stab, width=5,
        label='LS + stab', alpha=0.7, color='coral')

ax2.set_xlabel('Frequency (Hz)')
ax2.set_ylabel('Total Mode Count (across all points)')
ax2.set_title('Frequency Distribution of Detected Modes')
ax2.legend()
ax2.grid(axis='y', alpha=0.3)
ax2.set_xlim(freq_band)

plt.tight_layout()
output_path = Path("ESPRIT/comparison_results/stabilization_consistency.png")
output_path.parent.mkdir(exist_ok=True)
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"Saved visualization to {output_path}")

print()
print("="*80)
print("SUMMARY")
print("="*80)

avg_nostab = np.mean([len(f) for f in all_modes_nostab])
avg_stab = np.mean([len(f) for f in all_modes_stab])
avg_extra = avg_stab - avg_nostab

print(f"\nAverage modes per measurement:")
print(f"  TLS (no stabilization): {avg_nostab:.1f} modes")
print(f"  LS + stabilization:     {avg_stab:.1f} modes")
print(f"  Additional modes:       {avg_extra:+.1f} modes ({avg_extra/avg_nostab*100:+.1f}%)")

n_consistent_nostab = sum(1 for cnt in counts_nostab if cnt >= threshold)
n_consistent_stab = sum(1 for cnt in counts_stab if cnt >= threshold)

print(f"\nConsistent modes (≥80% of points):")
print(f"  TLS (no stabilization): {n_consistent_nostab} modes")
print(f"  LS + stabilization:     {n_consistent_stab} modes")

n_unique_stab = len([1 for fc, cnt in unique_to_stab if cnt >= threshold])
print(f"\nUnique consistent modes in stabilization: {n_unique_stab}")

if n_unique_stab > 0:
    print("\n✓ Stabilization finds additional CONSISTENT modes")
else:
    print("\n✗ Stabilization's extra modes are NOT consistent across measurements")
    print("  (likely spurious/location-specific artifacts)")
