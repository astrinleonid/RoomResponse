"""
Test spatial mode tracking vs global clustering on Belarus dataset.
"""

import json
import sys
from pathlib import Path
from collections import Counter

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from mode_tracking import (
    build_per_scenario_data,
    track_modes_along_bridge,
)

DATA_DIR = Path(r"D:\repos\RoomResponse\ESPRIT")
FULL_RESULTS = DATA_DIR / "belarus_full_results.json"
GLOBAL_MODES = DATA_DIR / "belarus_global_modes.json"


def load_data():
    with open(FULL_RESULTS) as f:
        full = json.load(f)
    with open(GLOBAL_MODES) as f:
        global_modes = json.load(f)
    return full, global_modes


def main():
    print("=" * 70)
    print("SPATIAL MODE TRACKING vs GLOBAL CLUSTERING — Belarus Dataset")
    print("=" * 70)

    full_data, global_data = load_data()

    # Build per-scenario data
    per_scenario = build_per_scenario_data(full_data["results"])
    scenario_order = sorted(per_scenario.keys())
    print(f"\nScenarios: {len(scenario_order)} "
          f"(bridge positions {scenario_order[0]} to {scenario_order[-1]})")
    total_modes_input = sum(len(v) for v in per_scenario.values())
    print(f"Total mode detections across all scenarios: {total_modes_input}")

    # Run spatial tracking
    print("\n--- Running spatial mode tracking (freq_tol=2%, max_gap=3) ---")
    chains = track_modes_along_bridge(
        per_scenario,
        scenario_order=scenario_order,
        freq_tol_pct=0.02,
        max_gap=3,
    )
    print(f"Total mode chains found: {len(chains)}")

    # Stability breakdown
    stability_counts = Counter(c.stability for c in chains)
    print(f"\nStability breakdown:")
    for cls in ["stable", "semi-stable", "weak", "spurious"]:
        print(f"  {cls:12s}: {stability_counts.get(cls, 0)}")

    # Modes below 200 Hz
    low_chains = [c for c in chains if c.frequency_mean < 200]
    low_stable = [c for c in low_chains if c.stability == "stable"]
    print(f"\nModes below 200 Hz:")
    print(f"  Total chains: {len(low_chains)}")
    print(f"  Stable:       {len(low_stable)}")

    # Coverage stats
    coverages = [c.coverage for c in chains]
    avg_coverage = sum(coverages) / len(coverages) if coverages else 0
    print(f"\nAverage chain coverage: {avg_coverage:.3f} "
          f"({avg_coverage*100:.1f}% of scenarios)")

    # Drift stats
    drifts = [abs(c.frequency_drift) for c in chains]
    max_drift_chain = max(chains, key=lambda c: abs(c.frequency_drift))
    print(f"\nMaximum frequency drift: {abs(max_drift_chain.frequency_drift):.2f} Hz "
          f"(chain {max_drift_chain.chain_id}, "
          f"mean={max_drift_chain.frequency_mean:.1f} Hz, "
          f"range={max_drift_chain.frequency_range[0]:.1f}-"
          f"{max_drift_chain.frequency_range[1]:.1f} Hz, "
          f"coverage={max_drift_chain.coverage:.1%})")

    # Relative drift
    rel_drifts = [abs(c.frequency_drift) / c.frequency_mean * 100
                  for c in chains if c.frequency_mean > 0 and c.detection_count > 1]
    if rel_drifts:
        print(f"Maximum relative drift: {max(rel_drifts):.2f}%")
        avg_rel_drift = sum(rel_drifts) / len(rel_drifts)
        print(f"Average relative drift: {avg_rel_drift:.2f}%")

    # --- Comparison with global clustering ---
    print("\n" + "=" * 70)
    print("COMPARISON WITH GLOBAL CLUSTERING")
    print("=" * 70)

    global_modes_list = global_data["global_modes"]
    global_summary = global_data["summary"]

    print(f"\n{'Metric':<40s} {'Global':>10s} {'Spatial':>10s}")
    print("-" * 62)
    print(f"{'Total modes/chains':<40s} "
          f"{global_summary['total_global_modes']:>10d} {len(chains):>10d}")
    print(f"{'Stable':<40s} "
          f"{global_summary['stable']:>10d} {stability_counts.get('stable', 0):>10d}")
    print(f"{'Semi-stable':<40s} "
          f"{global_summary['semi_stable']:>10d} {stability_counts.get('semi-stable', 0):>10d}")
    print(f"{'Weak':<40s} "
          f"{global_summary['weak']:>10d} {stability_counts.get('weak', 0):>10d}")
    print(f"{'Spurious':<40s} "
          f"{global_summary['spurious']:>10d} {stability_counts.get('spurious', 0):>10d}")

    # Match global modes to spatial chains by frequency proximity
    # For each global mode, find chains whose mean freq is within 1%
    global_to_chains = {}  # global_id -> list of chain_ids
    chain_to_globals = {}  # chain_id -> list of global_ids

    for gm in global_modes_list:
        gid = gm["id"]
        gfreq = gm["frequency_mean"]
        matching_chains = []
        for c in chains:
            if c.frequency_mean > 0 and abs(c.frequency_mean - gfreq) / gfreq <= 0.015:
                matching_chains.append(c.chain_id)
        global_to_chains[gid] = matching_chains
        for cid in matching_chains:
            chain_to_globals.setdefault(cid, []).append(gid)

    # How many global modes split into multiple chains?
    splits = {gid: cids for gid, cids in global_to_chains.items() if len(cids) > 1}
    unmatched_global = {gid: cids for gid, cids in global_to_chains.items() if len(cids) == 0}
    one_to_one = {gid: cids for gid, cids in global_to_chains.items() if len(cids) == 1}

    print(f"\n{'Global modes with 1:1 chain match':<40s} {len(one_to_one):>10d}")
    print(f"{'Global modes split into multiple chains':<40s} {len(splits):>10d}")
    print(f"{'Global modes with no chain match':<40s} {len(unmatched_global):>10d}")

    # Chains not matching any global mode (new discoveries)
    matched_chain_ids = set()
    for cids in global_to_chains.values():
        matched_chain_ids.update(cids)
    new_chains = [c for c in chains if c.chain_id not in matched_chain_ids]
    new_stable = [c for c in new_chains if c.stability in ("stable", "semi-stable")]
    print(f"{'Chains not matching any global mode':<40s} {len(new_chains):>10d}")
    print(f"{'  of which stable/semi-stable':<40s} {len(new_stable):>10d}")

    # Stability upgrades: spurious in global -> stable/semi-stable in spatial
    upgrades = 0
    downgrades = 0
    for gm in global_modes_list:
        gid = gm["id"]
        g_stability = gm["stability"]
        matching_cids = global_to_chains.get(gid, [])
        if not matching_cids:
            continue
        # Use the chain with best coverage as representative
        best_chain = max(
            [c for c in chains if c.chain_id in matching_cids],
            key=lambda c: c.coverage
        )
        if g_stability == "spurious" and best_chain.stability in ("stable", "semi-stable"):
            upgrades += 1
        elif g_stability in ("stable", "semi-stable") and best_chain.stability == "spurious":
            downgrades += 1

    print(f"\n{'Stability upgrades (spurious -> stable)':<40s} {upgrades:>10d}")
    print(f"{'Stability downgrades (stable -> spurious)':<40s} {downgrades:>10d}")

    # Show some interesting examples of high-drift chains
    print(f"\n--- Top 10 chains by absolute frequency drift ---")
    drift_sorted = sorted(chains, key=lambda c: abs(c.frequency_drift), reverse=True)
    print(f"{'Chain':>6s} {'Mean Hz':>8s} {'Range':>16s} {'Drift Hz':>9s} "
          f"{'Drift%':>7s} {'Cov':>6s} {'Stability':>12s}")
    for c in drift_sorted[:10]:
        rel = abs(c.frequency_drift) / c.frequency_mean * 100 if c.frequency_mean > 0 else 0
        print(f"{c.chain_id:>6d} {c.frequency_mean:>8.1f} "
              f"{c.frequency_range[0]:>7.1f}-{c.frequency_range[1]:<7.1f} "
              f"{c.frequency_drift:>+9.2f} {rel:>6.2f}% "
              f"{c.coverage:>5.1%} {c.stability:>12s}")

    # Summary of stable chains by frequency band
    print(f"\n--- Stable chains by frequency band ---")
    bands = [(0, 100), (100, 200), (200, 500), (500, 1000),
             (1000, 2000), (2000, 5000), (5000, 20000)]
    stable_chains = [c for c in chains if c.stability == "stable"]
    for lo, hi in bands:
        band_chains = [c for c in stable_chains if lo <= c.frequency_mean < hi]
        if band_chains:
            avg_cov = sum(c.coverage for c in band_chains) / len(band_chains)
            avg_drift = sum(abs(c.frequency_drift) for c in band_chains) / len(band_chains)
            print(f"  {lo:>5d}-{hi:<5d} Hz: {len(band_chains):>3d} chains, "
                  f"avg coverage {avg_cov:.1%}, avg drift {avg_drift:.2f} Hz")

    print("\nDone.")


if __name__ == "__main__":
    main()
