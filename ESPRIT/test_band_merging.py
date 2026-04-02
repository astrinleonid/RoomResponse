"""
test_band_merging.py
Test MAC-validated band merging on Belarus Scenario 40 data.

Runs ESPRIT with EXTENDED_BANDS, then compares raw (unmerged) vs merged mode counts.
Reports which modes were merged, MAC values, and frequency comparison against
STANDARD_BANDS baseline.

Usage:
    cd D:/repos/PianoidInstall/PianoidCore && .venv/Scripts/python D:/repos/RoomResponse/ESPRIT/test_band_merging.py
"""
import sys
import numpy as np
from pathlib import Path
import time

sys.path.insert(0, str(Path(__file__).parent))

from esprit_core import esprit_modal_identification, ModalParameters
from band_processing import (
    STANDARD_BANDS, EXTENDED_BANDS, process_band, merge_multiband_results,
)
from band_merging import merge_multiband_modes, band_center_weight, compute_mac


def load_belarus_scenario40():
    """Load and combine 6 channels from Belarus Scenario 40."""
    data_dir = Path("D:/repos/RoomResponse/piano/Belarus-Scenario40-Measuement1/averaged_responses")
    channels = []
    for i in range(6):
        ch = np.load(data_dir / f"average_ch{i}.npy")
        channels.append(ch)
    return np.column_stack(channels)


def run_esprit_per_band(signals, fs, bands, use_gpu=True, window_length=2000):
    """Run ESPRIT on each band, return per-band ModalParameters and used bands."""
    per_band_results = []
    used_bands = []

    for band in bands:
        band_data, fs_band, _ = process_band(signals, fs, band, apply_preemphasis=True)
        mo = band.model_order if band.model_order is not None else 30
        wl = min(window_length, len(band_data) // 2)

        try:
            result = esprit_modal_identification(
                band_data, fs=fs_band, model_order=mo, window_length=wl,
                freq_range=(band.f_min, band.f_max),
                use_gpu=use_gpu, use_tls=True, use_conjugate_pairing=True,
                use_multichannel=False, max_damping=0.2, min_freq=band.f_min,
            )
            per_band_results.append(result)
            used_bands.append(band)
            print(f"  {band.name:12s} [{band.f_min:5.0f}-{band.f_max:5.0f} Hz] "
                  f"M={mo:3d}  -> {len(result.frequencies):3d} modes")
        except Exception as e:
            print(f"  {band.name:12s} [{band.f_min:5.0f}-{band.f_max:5.0f} Hz] "
                  f"M={mo:3d}  -> FAILED: {e}")

    return per_band_results, used_bands


def main():
    fs = 48000.0

    print("Loading Belarus Scenario 40 data (6 channels, full length)...")
    signals = load_belarus_scenario40()
    print(f"  Shape: {signals.shape} ({signals.shape[0]} samples, "
          f"{signals.shape[1]} channels, {signals.shape[0]/fs:.3f} s)")

    # ---- Step 1: Run ESPRIT with EXTENDED_BANDS ----
    print(f"\n{'='*80}")
    print("STEP 1: Run ESPRIT per band (EXTENDED_BANDS)")
    print(f"{'='*80}")
    t0 = time.time()
    per_band_results, used_bands = run_esprit_per_band(signals, fs, EXTENDED_BANDS)
    t_esprit = time.time() - t0
    print(f"  ESPRIT time: {t_esprit:.2f} s")

    # Count raw modes
    n_raw = sum(len(r.frequencies) for r in per_band_results)
    print(f"\n  Total raw modes across all bands: {n_raw}")

    # ---- Step 2: Merge with MAC + band-center weighting ----
    print(f"\n{'='*80}")
    print("STEP 2: Merge with MAC validation + band-center weighting")
    print(f"{'='*80}")

    t0 = time.time()
    merged = merge_multiband_modes(
        per_band_results, used_bands,
        mac_threshold=0.9, freq_tol_pct=0.01,
    )
    t_merge = time.time() - t0

    n_unique = len(merged['frequencies'])
    n_merged = merged['n_merged']

    print(f"  Merge time: {t_merge*1000:.1f} ms")
    print(f"  Raw modes:    {n_raw}")
    print(f"  Merged (dup): {n_merged}")
    print(f"  Unique modes: {n_unique}")
    print(f"  Reduction:    {n_merged/n_raw*100:.1f}%" if n_raw > 0 else "  Reduction: N/A")

    # ---- Step 3: Merge log details ----
    if merged['merge_log']:
        print(f"\n{'='*80}")
        print("STEP 3: Merge decisions (duplicates removed)")
        print(f"{'='*80}")
        print(f"  {'Kept freq':>10s} {'Kept band':>12s} {'Kept wt':>8s}  "
              f"{'Disc freq':>10s} {'Disc band':>12s} {'Disc wt':>8s}  {'MAC':>6s}")
        print("  " + "-" * 80)
        for entry in merged['merge_log']:
            mac_str = f"{entry['mac']:.4f}" if entry['mac'] is not None else "  N/A "
            print(f"  {entry['kept_freq']:>10.2f} {entry['kept_band']:>12s} "
                  f"{entry['kept_weight']:>8.4f}  "
                  f"{entry['discarded_freq']:>10.2f} {entry['discarded_band']:>12s} "
                  f"{entry['discarded_weight']:>8.4f}  {mac_str}")
    else:
        print("\n  No merges occurred (no cross-band duplicates found).")

    # ---- Step 4: Frequency distribution ----
    print(f"\n{'='*80}")
    print("STEP 4: Frequency distribution (merged)")
    print(f"{'='*80}")
    freqs = merged['frequencies']
    ranges = [(30, 100), (100, 200), (200, 500), (500, 1000),
              (1000, 2000), (2000, 4000), (4000, 6000)]
    print(f"  {'Range':>15s}  {'Count':>6s}")
    print("  " + "-" * 30)
    for lo, hi in ranges:
        n = np.sum((freqs >= lo) & (freqs < hi))
        print(f"  {lo:5d}-{hi:5d} Hz  {n:>6d}")

    # ---- Step 5: Compare against STANDARD_BANDS baseline ----
    print(f"\n{'='*80}")
    print("STEP 5: Compare merged EXTENDED result vs STANDARD_BANDS baseline")
    print(f"{'='*80}")
    print("\nRunning STANDARD_BANDS baseline...")
    baseline_results, baseline_bands = run_esprit_per_band(signals, fs, STANDARD_BANDS)
    baseline_freqs = np.concatenate([r.frequencies for r in baseline_results]) if baseline_results else np.array([])

    print(f"\n  {'Metric':<40s} {'Baseline':>10s} {'Merged':>10s}")
    print("  " + "-" * 65)
    print(f"  {'Total modes':<40s} {len(baseline_freqs):>10d} {n_unique:>10d}")
    if len(baseline_freqs) > 0 and n_unique > 0:
        print(f"  {'Min freq (Hz)':<40s} {baseline_freqs.min():>10.1f} {freqs.min():>10.1f}")
        print(f"  {'Max freq (Hz)':<40s} {baseline_freqs.max():>10.1f} {freqs.max():>10.1f}")

        # Count overlapping modes (within 2 Hz)
        overlap_b = sum(1 for f in baseline_freqs if np.any(np.abs(freqs - f) < 2.0))
        overlap_e = sum(1 for f in freqs if np.any(np.abs(baseline_freqs - f) < 2.0))
        print(f"  {'Baseline modes matched in merged (2Hz)':<40s} {overlap_b:>10d}")
        print(f"  {'Merged modes matched in baseline (2Hz)':<40s} {overlap_e:>10d}")
        print(f"  {'Unique to merged (new discoveries)':<40s} {n_unique - overlap_e:>10d}")

    # ---- Step 6: Also test the high-level wrapper ----
    print(f"\n{'='*80}")
    print("STEP 6: Test merge_multiband_results() wrapper")
    print(f"{'='*80}")
    t0 = time.time()
    wrapper_result = merge_multiband_results(
        signals, fs, bands=EXTENDED_BANDS,
        esprit_params={
            'use_gpu': True, 'use_tls': True, 'use_conjugate_pairing': True,
            'use_multichannel': False, 'max_damping': 0.2, 'window_length': 2000,
        },
        mac_threshold=0.9, freq_tol_pct=0.01,
    )
    t_wrapper = time.time() - t0
    print(f"  Wrapper time: {t_wrapper:.2f} s")
    print(f"  Raw: {wrapper_result['n_raw']}, Merged: {wrapper_result['n_merged']}, "
          f"Unique: {len(wrapper_result['frequencies'])}")

    print(f"\n{'='*80}")
    print("DONE")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
