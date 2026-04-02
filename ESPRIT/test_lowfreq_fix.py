"""
test_lowfreq_fix.py
Compare low-frequency mode extraction before and after the window_length fix.

Before: EXTENDED_BANDS used global window_length=2000 for all bands (42ms at 48kHz).
After:  Ultra-Low/Low bands use window_length=12000/9600 (250ms/200ms), plus
        reduced exp_factor and increased model_order.

Usage:
    cd D:/repos/PianoidInstall/PianoidCore && .venv/Scripts/python D:/repos/RoomResponse/ESPRIT/test_lowfreq_fix.py
"""
import sys
import os
import time
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

os.environ['PYTHONIOENCODING'] = 'utf-8'
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

sys.path.insert(0, str(Path(__file__).parent))

from esprit_core import esprit_modal_identification
from band_processing import FrequencyBand, merge_multiband_results, EXTENDED_BANDS


# Old bands (before fix) - same structure but with old parameters
OLD_EXTENDED_BANDS = [
    FrequencyBand(f_min=30,   f_max=100,  filter_order=4, decimation=1, exp_factor=0.30, name="Ultra-Low",  model_order=10),
    FrequencyBand(f_min=80,   f_max=200,  filter_order=4, decimation=1, exp_factor=0.25, name="Low",        model_order=15),
    FrequencyBand(f_min=180,  f_max=400,  filter_order=5, decimation=1, exp_factor=0.20, name="Low-Mid",    model_order=25),
    FrequencyBand(f_min=350,  f_max=700,  filter_order=5, decimation=1, exp_factor=0.15, name="Mid",        model_order=35),
    FrequencyBand(f_min=600,  f_max=1200, filter_order=6, decimation=1, exp_factor=0.10, name="Mid-High",   model_order=45),
    FrequencyBand(f_min=1000, f_max=2500, filter_order=6, decimation=1, exp_factor=0.08, name="High",       model_order=50),
    FrequencyBand(f_min=2000, f_max=4500, filter_order=8, decimation=1, exp_factor=0.05, name="Upper",      model_order=50),
    FrequencyBand(f_min=4000, f_max=6000, filter_order=8, decimation=1, exp_factor=0.03, name="Top",        model_order=50),
]


def load_scenario_channels(scenario_dir: Path) -> np.ndarray:
    """Load 6 channels from averaged_responses."""
    avg_dir = scenario_dir / "averaged_responses"
    channels = []
    for ch in range(6):
        data = np.load(avg_dir / f"average_ch{ch}.npy")
        channels.append(data.astype(np.float64))
    return np.column_stack(channels)


def run_esprit_for_bands(signals, fs, bands, label):
    """Run ESPRIT with given bands and return merged result."""
    merged = merge_multiband_results(
        signals, fs,
        bands=bands,
        esprit_function=esprit_modal_identification,
        esprit_params={
            'use_gpu': True,
            'use_tls': True,
            'use_conjugate_pairing': True,
            'max_damping': 0.2,
            'window_length': 2000,  # global default (overridden by per-band for new bands)
        },
        apply_preemphasis=True,
        mac_threshold=0.9,
        freq_tol_pct=0.01,
    )
    return merged


def main():
    piano_dir = Path("D:/repos/RoomResponse/piano")
    fs = 48000.0
    low_freq_cutoff = 200.0

    scenario_ids = [3, 20, 40, 60, 80]
    scenario_dirs = []
    for sid in scenario_ids:
        d = piano_dir / f"Belarus-Scenario{sid}-Measuement1"
        if d.exists():
            scenario_dirs.append((sid, d))
        else:
            print(f"WARNING: Scenario {sid} not found at {d}")

    print("=" * 80)
    print("LOW-FREQUENCY MODE EXTRACTION: BEFORE vs AFTER FIX")
    print(f"Cutoff: modes below {low_freq_cutoff} Hz")
    print(f"Scenarios: {[s[0] for s in scenario_dirs]}")
    print("=" * 80)

    old_all_lowfreq = {}
    new_all_lowfreq = {}

    for sid, scenario_dir in scenario_dirs:
        print(f"\n--- Scenario {sid} ---")
        signals = load_scenario_channels(scenario_dir)
        print(f"  Signal shape: {signals.shape} ({signals.shape[0]/fs*1000:.0f} ms)")

        # BEFORE fix
        t0 = time.time()
        old_result = run_esprit_for_bands(signals, fs, OLD_EXTENDED_BANDS, "OLD")
        t_old = time.time() - t0
        old_freqs = old_result['frequencies']
        old_low = old_freqs[old_freqs < low_freq_cutoff]
        old_all_lowfreq[sid] = sorted(old_low.tolist())

        # AFTER fix
        t0 = time.time()
        new_result = run_esprit_for_bands(signals, fs, EXTENDED_BANDS, "NEW")
        t_new = time.time() - t0
        new_freqs = new_result['frequencies']
        new_low = new_freqs[new_freqs < low_freq_cutoff]
        new_all_lowfreq[sid] = sorted(new_low.tolist())

        print(f"  OLD: {len(old_low)} modes < {low_freq_cutoff} Hz "
              f"({len(old_freqs)} total) [{t_old:.1f}s]")
        if len(old_low) > 0:
            print(f"       freqs: {[f'{f:.1f}' for f in sorted(old_low)]}")

        print(f"  NEW: {len(new_low)} modes < {low_freq_cutoff} Hz "
              f"({len(new_freqs)} total) [{t_new:.1f}s]")
        if len(new_low) > 0:
            print(f"       freqs: {[f'{f:.1f}' for f in sorted(new_low)]}")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    total_old = sum(len(v) for v in old_all_lowfreq.values())
    total_new = sum(len(v) for v in new_all_lowfreq.values())

    print(f"\nTotal low-freq modes (<{low_freq_cutoff} Hz) across {len(scenario_dirs)} scenarios:")
    print(f"  BEFORE: {total_old}")
    print(f"  AFTER:  {total_new}")
    print(f"  Improvement: {total_new - total_old:+d} modes")

    # Consistency: find frequencies that appear in multiple scenarios
    def find_consistent_modes(all_lowfreq, tolerance_hz=5.0):
        """Find modes that appear in at least 2 scenarios within tolerance."""
        all_freqs = []
        for sid, freqs in all_lowfreq.items():
            for f in freqs:
                all_freqs.append((f, sid))
        all_freqs.sort()

        clusters = []
        used = set()
        for i, (f1, s1) in enumerate(all_freqs):
            if i in used:
                continue
            cluster = [(f1, s1)]
            used.add(i)
            for j, (f2, s2) in enumerate(all_freqs):
                if j in used:
                    continue
                if abs(f2 - f1) < tolerance_hz and s2 != s1:
                    cluster.append((f2, s2))
                    used.add(j)
            if len(set(s for _, s in cluster)) >= 2:
                mean_f = np.mean([f for f, _ in cluster])
                n_scenarios = len(set(s for _, s in cluster))
                clusters.append((mean_f, n_scenarios, cluster))

        return clusters

    old_consistent = find_consistent_modes(old_all_lowfreq)
    new_consistent = find_consistent_modes(new_all_lowfreq)

    print(f"\nConsistent modes (appear in >= 2 scenarios, +/- 5 Hz):")
    print(f"  BEFORE: {len(old_consistent)} consistent modes")
    for mean_f, n_sc, cluster in old_consistent:
        print(f"    {mean_f:.1f} Hz - in {n_sc}/{len(scenario_dirs)} scenarios")

    print(f"  AFTER:  {len(new_consistent)} consistent modes")
    for mean_f, n_sc, cluster in new_consistent:
        print(f"    {mean_f:.1f} Hz - in {n_sc}/{len(scenario_dirs)} scenarios")


if __name__ == "__main__":
    main()
