"""
run_full_belarus.py
Run ESPRIT with EXTENDED_BANDS (per-band window_length for low-freq) + MAC-based band merging
on the full Belarus dataset, then run spatial mode tracking.

v2: saves damping ratios, mode shape info, and runs spatial tracking automatically.

Usage:
    cd D:/repos/PianoidInstall/PianoidCore && .venv/Scripts/python D:/repos/RoomResponse/ESPRIT/run_full_belarus.py
"""
import sys
import os
import time
import json
import gc
import faulthandler
import numpy as np
from pathlib import Path
from collections import defaultdict

# Enable faulthandler for segfault diagnostics
faulthandler.enable()

# Fix Windows console encoding for Unicode characters
os.environ['PYTHONIOENCODING'] = 'utf-8'
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# Add ESPRIT directory to path
sys.path.insert(0, str(Path(__file__).parent))

from esprit_core import esprit_modal_identification
from band_processing import EXTENDED_BANDS, merge_multiband_results
from mode_tracking import track_modes_along_bridge, build_per_scenario_data, extract_scenario_number


def load_scenario_channels(scenario_dir: Path) -> np.ndarray:
    """Load 6 channels from averaged_responses and stack into (T, 6) array."""
    avg_dir = scenario_dir / "averaged_responses"
    channels = []
    for ch in range(6):
        data = np.load(avg_dir / f"average_ch{ch}.npy")
        channels.append(data.astype(np.float64))
    return np.column_stack(channels)


def format_time(seconds: float) -> str:
    """Format seconds into human-readable string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds / 60:.1f}m"
    else:
        return f"{seconds / 3600:.1f}h"


def print_progress_summary(i, n_scenarios, results, errors, total_start):
    """Print a Telegram-friendly progress summary."""
    total_elapsed = time.time() - total_start
    avg_per = total_elapsed / (i + 1)
    remaining = avg_per * (n_scenarios - i - 1)
    ok_results = [r for r in results if r['status'] == 'ok']
    mode_counts = [r['n_modes'] for r in ok_results]
    avg_modes = np.mean(mode_counts) if mode_counts else 0
    min_modes = min(mode_counts) if mode_counts else 0
    max_modes = max(mode_counts) if mode_counts else 0

    # Count low-freq modes
    low_freq_counts = []
    for r in ok_results:
        n_low = sum(1 for f in r['frequencies'] if f < 200)
        low_freq_counts.append(n_low)
    avg_low = np.mean(low_freq_counts) if low_freq_counts else 0

    print(f"\n{'='*60}")
    print(f"  PROGRESS: {i+1}/{n_scenarios} scenarios complete")
    print(f"{'='*60}")
    print(f"  Elapsed:   {format_time(total_elapsed)}")
    print(f"  Avg/scene: {avg_per:.1f}s")
    print(f"  ETA:       {format_time(remaining)}")
    print(f"  Modes:     avg={avg_modes:.1f}, min={min_modes}, max={max_modes}")
    print(f"  <200Hz:    avg={avg_low:.1f}")
    print(f"  Errors:    {len(errors)}")
    print(f"{'='*60}\n")
    sys.stdout.flush()


def run_extraction():
    """Run ESPRIT extraction on all Belarus scenarios."""
    piano_dir = Path("D:/repos/RoomResponse/piano")
    output_path = Path("D:/repos/RoomResponse/ESPRIT/belarus_full_results_v2.json")

    # Scan all Belarus scenario folders
    scenario_dirs = sorted([
        d for d in piano_dir.iterdir()
        if d.is_dir() and d.name.startswith("Belarus-Scenario")
    ], key=lambda d: d.name)

    n_scenarios = len(scenario_dirs)

    # Print band config
    print(f"Found {n_scenarios} Belarus scenarios")
    print(f"Using EXTENDED_BANDS ({len(EXTENDED_BANDS)} bands) with MAC-based merging")
    print(f"Band configuration:")
    for b in EXTENDED_BANDS:
        wl_info = f", window={b.window_length}" if b.window_length else ""
        mo_info = f", order={b.model_order}" if b.model_order else ""
        print(f"  {b.name:12s}: {b.f_min:5.0f}-{b.f_max:5.0f} Hz{mo_info}{wl_info}")
    print(f"GPU: True, TLS: True, full signal length (28800 samples)")
    print("=" * 80)
    sys.stdout.flush()

    fs = 48000.0
    results = []
    errors = []
    total_start = time.time()

    for i, scenario_dir in enumerate(scenario_dirs):
        scenario_name = scenario_dir.name
        scenario_start = time.time()

        try:
            # Load channels
            signals = load_scenario_channels(scenario_dir)
            n_channels = signals.shape[1]

            # Run ESPRIT with EXTENDED_BANDS + band merging
            merged = merge_multiband_results(
                signals, fs,
                bands=EXTENDED_BANDS,
                esprit_function=esprit_modal_identification,
                esprit_params={
                    'use_gpu': True,
                    'use_tls': True,
                    'use_conjugate_pairing': True,
                    'max_damping': 0.2,
                    'window_length': 2000,
                },
                apply_preemphasis=True,
                mac_threshold=0.9,
                freq_tol_pct=0.01,
            )

            elapsed = time.time() - scenario_start
            freqs = merged['frequencies']
            dampings = merged['damping_ratios']
            n_modes = len(freqs)
            f_min_found = float(freqs.min()) if n_modes > 0 else 0.0
            f_max_found = float(freqs.max()) if n_modes > 0 else 0.0
            n_raw = merged.get('n_raw', n_modes)
            n_merged = merged.get('n_merged', n_modes)
            has_shapes = merged.get('mode_shapes') is not None
            n_low = sum(1 for f in freqs if f < 200)

            result = {
                'scenario': scenario_name,
                'n_modes': n_modes,
                'n_raw': n_raw,
                'n_merged': n_merged,
                'n_channels': n_channels,
                'has_mode_shapes': has_shapes,
                'f_min': round(f_min_found, 2),
                'f_max': round(f_max_found, 2),
                'n_below_200hz': n_low,
                'frequencies': [round(float(f), 2) for f in sorted(freqs)],
                'damping_ratios': [round(float(d), 6) for d in dampings[np.argsort(freqs)]],
                'time_s': round(elapsed, 2),
                'status': 'ok',
            }
            results.append(result)

            print(f"[{i+1:2d}/{n_scenarios}] {scenario_name} - "
                  f"{n_modes} modes (raw {n_raw} -> merged {n_merged}), "
                  f"<200Hz: {n_low}, "
                  f"freq [{f_min_found:.1f} - {f_max_found:.1f}] Hz, "
                  f"shapes: {'yes' if has_shapes else 'no'}, "
                  f"{elapsed:.1f}s")

        except Exception as e:
            import traceback
            elapsed = time.time() - scenario_start
            error_msg = f"{type(e).__name__}: {e}"
            errors.append({'scenario': scenario_name, 'error': error_msg})
            results.append({
                'scenario': scenario_name,
                'n_modes': 0,
                'n_raw': 0,
                'n_merged': 0,
                'n_channels': 0,
                'has_mode_shapes': False,
                'f_min': 0,
                'f_max': 0,
                'n_below_200hz': 0,
                'frequencies': [],
                'damping_ratios': [],
                'time_s': round(elapsed, 2),
                'status': 'error',
                'error': error_msg,
            })
            print(f"[{i+1:2d}/{n_scenarios}] {scenario_name} - "
                  f"ERROR: {error_msg} ({elapsed:.1f}s)")
            traceback.print_exc()

        sys.stdout.flush()

        # Free GPU memory pool to prevent VMS accumulation on Windows WDDM
        # (CuPy keeps ~12 GB in its pool per scenario; on 16 GB RAM this OOMs)
        signals = None  # noqa: F841
        gc.collect()
        try:
            import cupy as cp
            cp.get_default_memory_pool().free_all_blocks()
            cp.get_default_pinned_memory_pool().free_all_blocks()
        except (ImportError, Exception):
            pass

        # Every 10 scenarios, print Telegram-friendly summary
        if (i + 1) % 10 == 0:
            print_progress_summary(i, n_scenarios, results, errors, total_start)

    # Final extraction summary
    total_elapsed = time.time() - total_start
    ok_results = [r for r in results if r['status'] == 'ok']
    all_modes = [r['n_modes'] for r in ok_results]
    all_low = [r['n_below_200hz'] for r in ok_results]

    print("\n" + "=" * 80)
    print("ESPRIT EXTRACTION COMPLETE")
    print("=" * 80)
    print(f"Total scenarios: {n_scenarios}")
    print(f"Successful: {len(ok_results)}")
    print(f"Errors: {len(errors)}")
    print(f"Total time: {format_time(total_elapsed)} ({total_elapsed:.1f}s)")
    print(f"Avg time/scenario: {total_elapsed / n_scenarios:.1f}s")
    print()

    if all_modes:
        print(f"Total modes (merged): {sum(all_modes)}")
        print(f"Modes/scenario: min={min(all_modes)}, max={max(all_modes)}, "
              f"avg={np.mean(all_modes):.1f}, median={np.median(all_modes):.1f}")
        print(f"Modes <200Hz/scenario: min={min(all_low)}, max={max(all_low)}, "
              f"avg={np.mean(all_low):.1f}")
        print()

    if errors:
        print("ERRORS:")
        for e in errors:
            print(f"  {e['scenario']}: {e['error']}")
        print()
    sys.stdout.flush()

    # Save extraction results
    output = {
        'metadata': {
            'version': 'v2',
            'bands': 'EXTENDED_BANDS',
            'band_config': [
                {
                    'name': b.name,
                    'f_min': b.f_min,
                    'f_max': b.f_max,
                    'filter_order': b.filter_order,
                    'decimation': b.decimation,
                    'exp_factor': b.exp_factor,
                    'model_order': b.model_order,
                    'window_length': b.window_length,
                }
                for b in EXTENDED_BANDS
            ],
            'n_bands': len(EXTENDED_BANDS),
            'fs': fs,
            'signal_length': 28800,
            'use_gpu': True,
            'use_tls': True,
            'mac_threshold': 0.9,
            'freq_tol_pct': 0.01,
            'total_time_s': round(total_elapsed, 2),
            'n_scenarios': n_scenarios,
            'n_successful': len(ok_results),
            'n_errors': len(errors),
        },
        'summary': {
            'total_modes': sum(all_modes) if all_modes else 0,
            'min_modes': min(all_modes) if all_modes else 0,
            'max_modes': max(all_modes) if all_modes else 0,
            'avg_modes': round(float(np.mean(all_modes)), 1) if all_modes else 0,
            'median_modes': round(float(np.median(all_modes)), 1) if all_modes else 0,
            'low_freq_modes': {
                'below_200hz_min': min(all_low) if all_low else 0,
                'below_200hz_max': max(all_low) if all_low else 0,
                'below_200hz_avg': round(float(np.mean(all_low)), 1) if all_low else 0,
            },
        },
        'results': results,
        'errors': errors,
    }

    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"Extraction results saved to {output_path}")
    sys.stdout.flush()

    return results, output_path


def run_tracking(results):
    """Run spatial mode tracking on extraction results."""
    tracking_output_path = Path("D:/repos/RoomResponse/ESPRIT/belarus_tracked_modes_v2.json")

    print("\n" + "=" * 80)
    print("SPATIAL MODE TRACKING")
    print("=" * 80)
    sys.stdout.flush()

    tracking_start = time.time()

    # Build per-scenario frequency data
    ok_results = [r for r in results if r['status'] == 'ok']
    per_scenario = build_per_scenario_data(ok_results)

    # Build damping lookup: {scenario_idx: {freq: damping}} for post-hoc attachment
    damping_lookup = {}
    for r in ok_results:
        idx = extract_scenario_number(r["scenario"])
        freq_damp = {}
        for f, d in zip(r['frequencies'], r['damping_ratios']):
            freq_damp[round(f, 2)] = d
        damping_lookup[idx] = freq_damp

    print(f"Tracking {len(per_scenario)} scenarios, "
          f"scenario range: {min(per_scenario.keys())}-{max(per_scenario.keys())}")
    sys.stdout.flush()

    # Run tracking
    chains = track_modes_along_bridge(
        per_scenario,
        freq_tol_pct=0.02,
        max_gap=3,
    )

    # Post-hoc: attach damping values to chain detections
    for chain in chains:
        for sc_idx, det in chain.detections.items():
            sc_dampings = damping_lookup.get(sc_idx, {})
            # Find closest frequency match
            closest_freq = min(sc_dampings.keys(),
                             key=lambda f: abs(f - det.frequency),
                             default=None)
            if closest_freq is not None and abs(closest_freq - det.frequency) < 1.0:
                det.damping_ratio = sc_dampings[closest_freq]
        # Re-finalize to update damping_mean
        chain.finalize(len(per_scenario))

    tracking_elapsed = time.time() - tracking_start

    # Classify chains
    stability_counts = defaultdict(int)
    for chain in chains:
        stability_counts[chain.stability] += 1

    # Separate by frequency range
    chains_below_200 = [c for c in chains if c.frequency_mean < 200]
    chains_200_1000 = [c for c in chains if 200 <= c.frequency_mean < 1000]
    chains_above_1000 = [c for c in chains if c.frequency_mean >= 1000]

    print(f"\nTracking completed in {tracking_elapsed:.1f}s")
    print(f"Total chains: {len(chains)}")
    print(f"  Stable (>=50%):      {stability_counts.get('stable', 0)}")
    print(f"  Semi-stable (25-50%): {stability_counts.get('semi-stable', 0)}")
    print(f"  Weak (10-25%):       {stability_counts.get('weak', 0)}")
    print(f"  Spurious (<10%):     {stability_counts.get('spurious', 0)}")
    print()
    print(f"By frequency range:")
    print(f"  <200 Hz:    {len(chains_below_200)} chains "
          f"({sum(1 for c in chains_below_200 if c.stability == 'stable')} stable)")
    print(f"  200-1000 Hz: {len(chains_200_1000)} chains "
          f"({sum(1 for c in chains_200_1000 if c.stability == 'stable')} stable)")
    print(f"  >1000 Hz:  {len(chains_above_1000)} chains "
          f"({sum(1 for c in chains_above_1000 if c.stability == 'stable')} stable)")
    print()

    # Print stable chains
    stable_chains = [c for c in chains if c.stability in ('stable', 'semi-stable')]
    print(f"Stable + semi-stable chains ({len(stable_chains)}):")
    for c in stable_chains:
        print(f"  Chain {c.chain_id:3d}: {c.frequency_mean:7.1f} Hz, "
              f"drift={c.frequency_drift:+6.1f} Hz, "
              f"damping={c.damping_mean:.4f}, "
              f"coverage={c.coverage:.0%} ({c.detection_count}/{len(per_scenario)}), "
              f"{c.stability}")

    # Compare with v1
    v1_path = Path("D:/repos/RoomResponse/ESPRIT/belarus_full_results.json")
    if v1_path.exists():
        print("\n" + "-" * 60)
        print("COMPARISON WITH v1")
        print("-" * 60)
        with open(v1_path) as f:
            v1 = json.load(f)
        v1_summary = v1['summary']
        v2_ok = [r for r in results if r['status'] == 'ok']
        v2_total = sum(r['n_modes'] for r in v2_ok)
        v2_modes = [r['n_modes'] for r in v2_ok]
        v1_low_counts = []
        for r in v1.get('results', []):
            if r.get('status') == 'ok':
                n_low = sum(1 for f in r.get('frequencies', []) if f < 200)
                v1_low_counts.append(n_low)
        v2_low_counts = [r['n_below_200hz'] for r in v2_ok]

        print(f"  {'Metric':<25s} {'v1':>10s} {'v2':>10s} {'Change':>10s}")
        print(f"  {'-'*55}")
        print(f"  {'Total modes':<25s} {v1_summary['total_modes']:>10d} {v2_total:>10d} "
              f"{v2_total - v1_summary['total_modes']:>+10d}")
        print(f"  {'Avg modes/scenario':<25s} {v1_summary['avg_modes']:>10.1f} "
              f"{np.mean(v2_modes):>10.1f} {np.mean(v2_modes) - v1_summary['avg_modes']:>+10.1f}")
        print(f"  {'Min modes/scenario':<25s} {v1_summary['min_modes']:>10d} "
              f"{min(v2_modes):>10d} {min(v2_modes) - v1_summary['min_modes']:>+10d}")
        print(f"  {'Max modes/scenario':<25s} {v1_summary['max_modes']:>10d} "
              f"{max(v2_modes):>10d} {max(v2_modes) - v1_summary['max_modes']:>+10d}")
        if v1_low_counts and v2_low_counts:
            print(f"  {'Avg <200Hz modes':<25s} {np.mean(v1_low_counts):>10.1f} "
                  f"{np.mean(v2_low_counts):>10.1f} "
                  f"{np.mean(v2_low_counts) - np.mean(v1_low_counts):>+10.1f}")
        print(f"  {'Total time':<25s} {v1['metadata']['total_time_s']:>10.1f}s "
              f"{'':>10s}")
        print()
    sys.stdout.flush()

    # Save tracking results
    tracking_output = {
        'metadata': {
            'version': 'v2',
            'freq_tol_pct': 0.02,
            'max_gap': 3,
            'n_scenarios': len(per_scenario),
            'scenario_range': [min(per_scenario.keys()), max(per_scenario.keys())],
            'tracking_time_s': round(tracking_elapsed, 2),
        },
        'summary': {
            'total_chains': len(chains),
            'stable': stability_counts.get('stable', 0),
            'semi_stable': stability_counts.get('semi-stable', 0),
            'weak': stability_counts.get('weak', 0),
            'spurious': stability_counts.get('spurious', 0),
            'chains_below_200hz': len(chains_below_200),
            'chains_200_1000hz': len(chains_200_1000),
            'chains_above_1000hz': len(chains_above_1000),
        },
        'chains': [
            {
                'chain_id': c.chain_id,
                'frequency_mean': round(c.frequency_mean, 2),
                'frequency_range': [round(c.frequency_range[0], 2),
                                    round(c.frequency_range[1], 2)],
                'frequency_drift': round(c.frequency_drift, 2),
                'damping_mean': round(c.damping_mean, 6),
                'detection_count': c.detection_count,
                'coverage': round(c.coverage, 4),
                'stability': c.stability,
                'detections': {
                    str(k): {
                        'frequency': round(d.frequency, 2),
                        'damping_ratio': round(d.damping_ratio, 6) if d.damping_ratio else None,
                    }
                    for k, d in sorted(c.detections.items())
                },
            }
            for c in chains
        ],
    }

    with open(tracking_output_path, 'w') as f:
        json.dump(tracking_output, f, indent=2)
    print(f"Tracking results saved to {tracking_output_path}")
    sys.stdout.flush()


def main():
    results, output_path = run_extraction()
    run_tracking(results)

    print("\n" + "=" * 80)
    print("ALL DONE")
    print("=" * 80)
    sys.stdout.flush()


if __name__ == '__main__':
    main()
