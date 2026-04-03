"""
Cross-scenario mode stabilization for Belarus ESPRIT results.

Clusters modes across 78 scenarios by frequency proximity (1% relative tolerance),
classifies by detection consistency, and outputs the global mode set.
"""

import json
import sys
from pathlib import Path

INPUT_PATH = Path(r"D:\repos\RoomResponse\ESPRIT\belarus_full_results.json")
OUTPUT_PATH = Path(r"D:\repos\RoomResponse\ESPRIT\belarus_global_modes.json")

FREQ_TOL_PCT = 0.01  # 1% relative tolerance

# Stability thresholds (fraction of total scenarios)
STABLE_THRESH = 0.50
SEMI_THRESH = 0.25
WEAK_THRESH = 0.10


def load_data(path):
    with open(path) as f:
        data = json.load(f)
    n_scenarios = data["metadata"]["n_scenarios"]
    results = data["results"]
    print(f"Loaded {n_scenarios} scenarios, {data['summary']['total_modes']} total modes")
    return n_scenarios, results


def collect_all_modes(results):
    """Collect all (frequency, scenario_index) pairs, sorted by frequency."""
    modes = []
    for i, r in enumerate(results):
        for freq in r["frequencies"]:
            modes.append((freq, i))
    modes.sort(key=lambda x: x[0])
    print(f"Collected {len(modes)} modes, freq range [{modes[0][0]:.1f}, {modes[-1][0]:.1f}] Hz")
    return modes


def cluster_modes(modes, tol_pct):
    """
    Sequential clustering: iterate sorted modes, assign to nearest existing
    cluster if within tol_pct of cluster mean, else create new cluster.
    Each cluster tracks: list of frequencies, set of scenario indices.
    """
    clusters = []  # list of {"freq_sum", "count", "freqs", "scenarios"}

    for freq, scenario_idx in modes:
        assigned = False
        # Check existing clusters - since modes are sorted, only need to check
        # clusters whose mean is close. We check from the end (most recent/highest freq).
        for c in reversed(clusters):
            mean = c["freq_sum"] / c["count"]
            if abs(freq - mean) / mean <= tol_pct:
                c["freq_sum"] += freq
                c["count"] += 1
                c["freqs"].append(freq)
                c["scenarios"].add(scenario_idx)
                assigned = True
                break
            # If we've gone past possible matches (freq too high), stop
            if freq > mean * (1 + tol_pct * 2):
                break

        if not assigned:
            clusters.append({
                "freq_sum": freq,
                "count": 1,
                "freqs": [freq],
                "scenarios": {scenario_idx},
            })

    print(f"Formed {len(clusters)} clusters")
    return clusters


def classify_clusters(clusters, n_scenarios):
    """Compute stats and classify each cluster by stability."""
    import statistics

    global_modes = []
    for i, c in enumerate(clusters):
        freqs = c["freqs"]
        det_count = len(c["scenarios"])
        det_frac = det_count / n_scenarios

        freq_mean = sum(freqs) / len(freqs)
        freq_std = statistics.stdev(freqs) if len(freqs) > 1 else 0.0

        if det_frac >= STABLE_THRESH:
            stability = "stable"
        elif det_frac >= SEMI_THRESH:
            stability = "semi-stable"
        elif det_frac >= WEAK_THRESH:
            stability = "weak"
        else:
            stability = "spurious"

        global_modes.append({
            "id": i,
            "frequency_mean": round(freq_mean, 2),
            "frequency_std": round(freq_std, 3),
            "detection_count": det_count,
            "detection_fraction": round(det_frac, 3),
            "stability": stability,
            "per_scenario_frequencies": [round(f, 2) for f in sorted(freqs)],
        })

    # Sort by frequency
    global_modes.sort(key=lambda m: m["frequency_mean"])
    # Reassign IDs after sorting
    for i, m in enumerate(global_modes):
        m["id"] = i

    return global_modes


def print_report(global_modes, n_scenarios):
    """Print analysis report."""
    counts = {"stable": 0, "semi-stable": 0, "weak": 0, "spurious": 0}
    for m in global_modes:
        counts[m["stability"]] += 1

    print("\n" + "=" * 70)
    print("CROSS-SCENARIO MODE STABILIZATION REPORT")
    print("=" * 70)
    print(f"\nTotal scenarios: {n_scenarios}")
    print(f"Total unique global modes: {len(global_modes)}")
    print(f"\nStability classification:")
    print(f"  Stable (>={int(STABLE_THRESH*100)}% scenarios, >={int(STABLE_THRESH*n_scenarios)}):  {counts['stable']}")
    print(f"  Semi-stable (25-50%, 20-38):    {counts['semi-stable']}")
    print(f"  Weak (10-25%, 8-19):            {counts['weak']}")
    print(f"  Spurious (<10%, <8):            {counts['spurious']}")

    # Stable modes table
    stable = [m for m in global_modes if m["stability"] == "stable"]
    print(f"\n{'=' * 70}")
    print(f"STABLE MODES ({len(stable)} modes, detected in >={int(STABLE_THRESH*n_scenarios)} of {n_scenarios} scenarios)")
    print(f"{'=' * 70}")
    print(f"{'ID':>4}  {'Freq Mean':>10}  {'Freq Std':>9}  {'Det Count':>9}  {'Det %':>6}")
    print(f"{'-'*4}  {'-'*10}  {'-'*9}  {'-'*9}  {'-'*6}")
    for m in stable:
        print(f"{m['id']:4d}  {m['frequency_mean']:10.2f}  {m['frequency_std']:9.3f}  "
              f"{m['detection_count']:9d}  {m['detection_fraction']*100:5.1f}%")

    # Frequency distribution of stable modes
    print(f"\n{'=' * 70}")
    print("FREQUENCY DISTRIBUTION OF STABLE MODES")
    print(f"{'=' * 70}")
    bands = [
        (0, 100, "0-100 Hz"),
        (100, 200, "100-200 Hz"),
        (200, 500, "200-500 Hz"),
        (500, 1000, "500-1000 Hz"),
        (1000, 2000, "1-2 kHz"),
        (2000, 4000, "2-4 kHz"),
        (4000, 8000, "4-8 kHz"),
    ]
    for lo, hi, label in bands:
        n_stable = sum(1 for m in global_modes if m["stability"] == "stable" and lo <= m["frequency_mean"] < hi)
        n_semi = sum(1 for m in global_modes if m["stability"] == "semi-stable" and lo <= m["frequency_mean"] < hi)
        n_weak = sum(1 for m in global_modes if m["stability"] == "weak" and lo <= m["frequency_mean"] < hi)
        n_spur = sum(1 for m in global_modes if m["stability"] == "spurious" and lo <= m["frequency_mean"] < hi)
        n_total = n_stable + n_semi + n_weak + n_spur
        bar_s = "#" * n_stable
        bar_ss = "+" * n_semi
        bar_w = "." * n_weak
        print(f"  {label:>12s}  stable={n_stable:3d}  semi={n_semi:3d}  weak={n_weak:3d}  spur={n_spur:3d}  total={n_total:3d}  {bar_s}{bar_ss}{bar_w}")

    # Semi-stable modes summary
    semi = [m for m in global_modes if m["stability"] == "semi-stable"]
    if semi:
        print(f"\n{'=' * 70}")
        print(f"SEMI-STABLE MODES ({len(semi)} modes, detected in 20-38 scenarios)")
        print(f"{'=' * 70}")
        print(f"{'ID':>4}  {'Freq Mean':>10}  {'Freq Std':>9}  {'Det Count':>9}  {'Det %':>6}")
        print(f"{'-'*4}  {'-'*10}  {'-'*9}  {'-'*9}  {'-'*6}")
        for m in semi:
            print(f"{m['id']:4d}  {m['frequency_mean']:10.2f}  {m['frequency_std']:9.3f}  "
                  f"{m['detection_count']:9d}  {m['detection_fraction']*100:5.1f}%")


def save_output(global_modes, n_scenarios, output_path):
    output = {
        "total_scenarios": n_scenarios,
        "clustering": {
            "freq_tol_pct": FREQ_TOL_PCT,
            "stable_threshold": STABLE_THRESH,
        },
        "summary": {
            "total_global_modes": len(global_modes),
            "stable": sum(1 for m in global_modes if m["stability"] == "stable"),
            "semi_stable": sum(1 for m in global_modes if m["stability"] == "semi-stable"),
            "weak": sum(1 for m in global_modes if m["stability"] == "weak"),
            "spurious": sum(1 for m in global_modes if m["stability"] == "spurious"),
        },
        "global_modes": global_modes,
    }
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved {len(global_modes)} global modes to {output_path}")


def main():
    n_scenarios, results = load_data(INPUT_PATH)
    modes = collect_all_modes(results)
    clusters = cluster_modes(modes, FREQ_TOL_PCT)
    global_modes = classify_clusters(clusters, n_scenarios)
    print_report(global_modes, n_scenarios)
    save_output(global_modes, n_scenarios, OUTPUT_PATH)


if __name__ == "__main__":
    main()
