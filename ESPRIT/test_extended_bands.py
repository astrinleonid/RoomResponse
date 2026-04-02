"""
test_extended_bands.py
Compare STANDARD_BANDS vs EXTENDED_BANDS on Belarus Scenario 40 data.

Usage:
    cd D:/repos/PianoidInstall/PianoidCore && .venv/Scripts/python D:/repos/RoomResponse/ESPRIT/test_extended_bands.py
"""
import sys
import numpy as np
from pathlib import Path
import time

sys.path.insert(0, str(Path(__file__).parent))

from esprit_core import esprit_modal_identification
from band_processing import (
    STANDARD_BANDS, EXTENDED_BANDS, process_band, select_bands_for_range
)


def load_belarus_scenario40():
    """Load and combine 6 channels from Belarus Scenario 40."""
    data_dir = Path("D:/repos/RoomResponse/piano/Belarus-Scenario40-Measuement1/averaged_responses")
    channels = []
    for i in range(6):
        ch = np.load(data_dir / f"average_ch{i}.npy")
        channels.append(ch)
    # Stack as (T, n_channels)
    signals = np.column_stack(channels)
    return signals


def run_esprit_multiband(signals, fs, bands, global_model_order, use_gpu=True,
                         window_length=2000):
    """Run ESPRIT per band and collect all modes."""
    all_freqs = []
    all_damping = []
    all_poles = []
    band_labels = []

    for band in bands:
        # Process band (filter, preemphasis, decimation)
        band_data, fs_band, metadata = process_band(signals, fs, band, apply_preemphasis=True)

        # Determine model order: per-band or global fallback
        mo = band.model_order if band.model_order is not None else global_model_order

        # Clamp window_length to signal length
        wl = min(window_length, len(band_data) // 2)

        try:
            result = esprit_modal_identification(
                band_data,
                fs=fs_band,
                model_order=mo,
                window_length=wl,
                freq_range=(band.f_min, band.f_max),
                use_gpu=use_gpu,
                use_tls=True,
                use_conjugate_pairing=True,
                use_multichannel=False,
                max_damping=0.2,
                min_freq=band.f_min,
            )
            n = len(result.frequencies)
            all_freqs.extend(result.frequencies.tolist())
            all_damping.extend(result.damping_ratios.tolist())
            all_poles.extend(result.poles.tolist())
            band_labels.extend([band.name] * n)
            print(f"  {band.name:12s} [{band.f_min:5.0f}-{band.f_max:5.0f} Hz] "
                  f"M={mo:3d}  -> {n:3d} modes")
        except Exception as e:
            print(f"  {band.name:12s} [{band.f_min:5.0f}-{band.f_max:5.0f} Hz] "
                  f"M={mo:3d}  -> FAILED: {e}")

    return {
        'frequencies': np.array(all_freqs),
        'damping': np.array(all_damping),
        'poles': np.array(all_poles),
        'band_labels': band_labels,
    }


def print_comparison(baseline, extended):
    """Print comparison table."""
    bf = baseline['frequencies']
    ef = extended['frequencies']
    bd = baseline['damping']
    ed = extended['damping']

    print("\n" + "=" * 80)
    print("COMPARISON: STANDARD_BANDS (baseline) vs EXTENDED_BANDS (new)")
    print("=" * 80)

    # Total modes
    print(f"\n{'Metric':<35s} {'Baseline':>12s} {'Extended':>12s}")
    print("-" * 60)
    print(f"{'Total modes extracted':<35s} {len(bf):>12d} {len(ef):>12d}")

    # Frequency coverage
    if len(bf) > 0 and len(ef) > 0:
        print(f"{'Min frequency (Hz)':<35s} {bf.min():>12.1f} {ef.min():>12.1f}")
        print(f"{'Max frequency (Hz)':<35s} {bf.max():>12.1f} {ef.max():>12.1f}")
        print(f"{'Mean frequency (Hz)':<35s} {bf.mean():>12.1f} {ef.mean():>12.1f}")
    if len(ef) > 0 and len(bf) > 0:
        print(f"{'Mean damping ratio':<35s} {bd.mean():>12.5f} {ed.mean():>12.5f}")
        print(f"{'Median damping ratio':<35s} {np.median(bd):>12.5f} {np.median(ed):>12.5f}")
        print(f"{'Max damping ratio':<35s} {bd.max():>12.5f} {ed.max():>12.5f}")

    # Frequency distribution by range
    print(f"\n{'Frequency range':<25s} {'Baseline':>10s} {'Extended':>10s}")
    print("-" * 50)
    ranges = [(30, 100), (100, 200), (200, 500), (500, 1000),
              (1000, 2000), (2000, 4000), (4000, 6000)]
    for lo, hi in ranges:
        nb = np.sum((bf >= lo) & (bf < hi)) if len(bf) > 0 else 0
        ne = np.sum((ef >= lo) & (ef < hi)) if len(ef) > 0 else 0
        print(f"  {lo:5d}-{hi:5d} Hz          {nb:>10d} {ne:>10d}")

    # Overlap: modes within 2 Hz tolerance
    if len(bf) > 0 and len(ef) > 0:
        overlap_count = 0
        for f in bf:
            if np.any(np.abs(ef - f) < 2.0):
                overlap_count += 1
        print(f"\n{'Overlapping modes (2 Hz tol)':<35s} {overlap_count:>12d}")
        print(f"{'Unique to baseline':<35s} {len(bf) - overlap_count:>12d}")
        # reverse
        overlap_ext = 0
        for f in ef:
            if np.any(np.abs(bf - f) < 2.0):
                overlap_ext += 1
        print(f"{'Unique to extended':<35s} {len(ef) - overlap_ext:>12d}")

    # Per-band breakdown for extended
    print(f"\nExtended bands breakdown:")
    print(f"  {'Band':<12s} {'Count':>6s} {'Freq range':>20s}")
    print("  " + "-" * 42)
    for band_name in dict.fromkeys(extended['band_labels']):
        mask = [l == band_name for l in extended['band_labels']]
        freqs = ef[mask]
        if len(freqs) > 0:
            print(f"  {band_name:<12s} {len(freqs):>6d} "
                  f"{freqs.min():>8.1f} - {freqs.max():<8.1f} Hz")


def main():
    fs = 48000.0

    print("Loading Belarus Scenario 40 data...")
    signals = load_belarus_scenario40()
    print(f"  Shape: {signals.shape} ({signals.shape[0]} samples, {signals.shape[1]} channels)")
    print(f"  Duration: {signals.shape[0]/fs:.3f} s")
    print(f"  Using full-length signal (no truncation)")

    # --- Baseline: STANDARD_BANDS ---
    print(f"\n{'='*80}")
    print("BASELINE: STANDARD_BANDS (model_order=30)")
    print(f"{'='*80}")
    t0 = time.time()
    baseline = run_esprit_multiband(signals, fs, STANDARD_BANDS,
                                    global_model_order=30, use_gpu=True)
    t_baseline = time.time() - t0
    print(f"  Total time: {t_baseline:.2f} s")

    # --- Extended: EXTENDED_BANDS ---
    print(f"\n{'='*80}")
    print("EXTENDED: EXTENDED_BANDS (per-band model orders)")
    print(f"{'='*80}")
    t0 = time.time()
    extended = run_esprit_multiband(signals, fs, EXTENDED_BANDS,
                                   global_model_order=30, use_gpu=True)
    t_extended = time.time() - t0
    print(f"  Total time: {t_extended:.2f} s")

    # --- Comparison ---
    print_comparison(baseline, extended)

    print(f"\n{'Timing':<35s} {'Baseline':>12s} {'Extended':>12s}")
    print("-" * 60)
    print(f"{'Total time (s)':<35s} {t_baseline:>12.2f} {t_extended:>12.2f}")
    print(f"{'Speedup':<35s} {'':>12s} {t_baseline/t_extended if t_extended > 0 else 0:>11.2f}x")


if __name__ == "__main__":
    main()
