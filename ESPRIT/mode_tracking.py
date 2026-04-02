"""
Spatial mode tracking along the piano bridge.

Instead of global frequency clustering (which treats all scenarios equally),
this module tracks modes along the bridge using spatial continuity. Each step
only needs to match its neighbor, allowing modes that drift significantly
across the full bridge span to be tracked correctly.

Designed for the Belarus dataset: 78 scenarios ordered by bridge position
(scenario 3 = bass end, scenario 87 = treble end).
"""

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class ModeDetection:
    """A single mode detection at one scenario."""
    scenario_index: int
    frequency: float
    damping_ratio: Optional[float] = None


@dataclass
class ModeChain:
    """A mode tracked across multiple scenarios along the bridge."""
    chain_id: int
    detections: Dict[int, ModeDetection] = field(default_factory=dict)
    frequency_mean: float = 0.0
    frequency_range: Tuple[float, float] = (0.0, 0.0)
    frequency_drift: float = 0.0
    damping_mean: float = 0.0
    detection_count: int = 0
    coverage: float = 0.0
    stability: str = "spurious"
    _gap_counter: int = field(default=0, repr=False)
    _closed: bool = field(default=False, repr=False)
    _last_freq: float = field(default=0.0, repr=False)

    def add_detection(self, det: ModeDetection):
        self.detections[det.scenario_index] = det
        self._gap_counter = 0
        self._last_freq = det.frequency

    def increment_gap(self):
        self._gap_counter += 1

    @property
    def gap(self):
        return self._gap_counter

    def finalize(self, total_scenarios: int):
        if not self.detections:
            return
        freqs = [d.frequency for d in self.detections.values()]
        self.frequency_mean = sum(freqs) / len(freqs)
        self.frequency_range = (min(freqs), max(freqs))
        self.detection_count = len(self.detections)
        self.coverage = self.detection_count / total_scenarios

        # Drift: frequency change from first to last detection in bridge order
        ordered = sorted(self.detections.values(), key=lambda d: d.scenario_index)
        self.frequency_drift = ordered[-1].frequency - ordered[0].frequency

        # Damping
        dampings = [d.damping_ratio for d in self.detections.values()
                    if d.damping_ratio is not None]
        self.damping_mean = sum(dampings) / len(dampings) if dampings else 0.0

        # Stability classification (same thresholds as global clustering)
        if self.coverage >= 0.50:
            self.stability = "stable"
        elif self.coverage >= 0.25:
            self.stability = "semi-stable"
        elif self.coverage >= 0.10:
            self.stability = "weak"
        else:
            self.stability = "spurious"


def extract_scenario_number(scenario_name: str) -> int:
    """Extract numeric scenario index from scenario name string."""
    m = re.search(r'Scenario(\d+)', scenario_name)
    if m:
        return int(m.group(1))
    raise ValueError(f"Cannot extract scenario number from: {scenario_name}")


def build_per_scenario_data(results: list) -> Dict[int, List[float]]:
    """
    Convert raw results list to {scenario_number: [frequencies]} dict.
    """
    per_scenario = {}
    for r in results:
        idx = extract_scenario_number(r["scenario"])
        per_scenario[idx] = sorted(r["frequencies"])
    return per_scenario


def track_modes_along_bridge(
    per_scenario_freqs: Dict[int, List[float]],
    scenario_order: Optional[List[int]] = None,
    freq_tol_pct: float = 0.02,
    max_gap: int = 3,
) -> List[ModeChain]:
    """
    Track modes along the bridge using spatial continuity.

    Args:
        per_scenario_freqs: dict mapping scenario_index to list of frequencies
        scenario_order: list of scenario indices in bridge order.
            If None, sorted keys of per_scenario_freqs are used.
        freq_tol_pct: max relative frequency change between adjacent scenarios
        max_gap: max consecutive missing scenarios before chain closes

    Returns:
        list of ModeChain objects, sorted by frequency_mean
    """
    if scenario_order is None:
        scenario_order = sorted(per_scenario_freqs.keys())

    total_scenarios = len(scenario_order)
    next_chain_id = 0
    active_chains: List[ModeChain] = []
    finished_chains: List[ModeChain] = []

    # Step 1: Initialize from first scenario
    first_idx = scenario_order[0]
    first_freqs = per_scenario_freqs.get(first_idx, [])
    for freq in first_freqs:
        chain = ModeChain(chain_id=next_chain_id)
        next_chain_id += 1
        det = ModeDetection(scenario_index=first_idx, frequency=freq)
        chain.add_detection(det)
        active_chains.append(chain)

    # Step 2: Process subsequent scenarios in bridge order
    for sc_idx in scenario_order[1:]:
        freqs = per_scenario_freqs.get(sc_idx, [])

        # Track which chains got matched this step
        matched_chains = set()
        # Track which frequencies got assigned
        assigned_freqs = set()

        # Build candidate pairs: (frequency_index, chain_index, freq_distance)
        candidates = []
        for fi, freq in enumerate(freqs):
            for ci, chain in enumerate(active_chains):
                if chain._closed:
                    continue
                ref_freq = chain._last_freq
                if ref_freq == 0:
                    continue
                rel_diff = abs(freq - ref_freq) / ref_freq
                if rel_diff <= freq_tol_pct:
                    candidates.append((fi, ci, rel_diff))

        # Sort by distance (greedy best-first assignment)
        candidates.sort(key=lambda x: x[2])

        for fi, ci, dist in candidates:
            if fi in assigned_freqs or ci in matched_chains:
                continue
            det = ModeDetection(scenario_index=sc_idx, frequency=freqs[fi])
            active_chains[ci].add_detection(det)
            matched_chains.add(ci)
            assigned_freqs.add(fi)

        # Unmatched frequencies start new chains
        for fi, freq in enumerate(freqs):
            if fi not in assigned_freqs:
                chain = ModeChain(chain_id=next_chain_id)
                next_chain_id += 1
                det = ModeDetection(scenario_index=sc_idx, frequency=freq)
                chain.add_detection(det)
                active_chains.append(chain)

        # Unmatched chains: increment gap, close if exceeded
        new_active = []
        for ci, chain in enumerate(active_chains):
            if ci not in matched_chains and not chain._closed:
                # Only increment gap for chains that existed before this step
                # (new chains just added won't be in matched_chains but shouldn't
                # get a gap increment since they were just created this step)
                if chain.detections and max(chain.detections.keys()) < sc_idx:
                    chain.increment_gap()
                    if chain.gap > max_gap:
                        chain._closed = True
                        chain.finalize(total_scenarios)
                        finished_chains.append(chain)
                        continue
            new_active.append(chain)
        active_chains = new_active

    # Finalize remaining active chains
    for chain in active_chains:
        chain.finalize(total_scenarios)
        finished_chains.append(chain)

    # Post-processing: merge chains that likely represent the same mode
    # re-detected after a gap > max_gap
    finished_chains = _merge_split_chains(finished_chains, freq_tol_pct,
                                          total_scenarios)

    # Sort by mean frequency
    finished_chains.sort(key=lambda c: c.frequency_mean)

    # Re-number chain IDs
    for i, chain in enumerate(finished_chains):
        chain.chain_id = i

    return finished_chains


def _merge_split_chains(
    chains: List[ModeChain],
    freq_tol_pct: float,
    total_scenarios: int,
) -> List[ModeChain]:
    """
    Merge chains that represent the same mode but were split by a gap > max_gap.

    Two chains are merged if:
    - Their frequency ranges are close (overlap or within tolerance)
    - They don't overlap in scenario coverage (one ends before the other starts)
    """
    if not chains:
        return chains

    # Sort by mean frequency for efficient comparison
    chains.sort(key=lambda c: c.frequency_mean)
    merged = [False] * len(chains)
    result = []

    for i in range(len(chains)):
        if merged[i]:
            continue

        current = chains[i]

        # Look for chains to merge with
        for j in range(i + 1, len(chains)):
            if merged[j]:
                continue
            other = chains[j]

            # Frequency proximity check
            freq_diff = abs(current.frequency_mean - other.frequency_mean)
            ref_freq = min(current.frequency_mean, other.frequency_mean)
            if ref_freq > 0 and freq_diff / ref_freq > freq_tol_pct * 2:
                # Too far in frequency, and list is sorted, so stop
                break

            # Check non-overlapping scenario coverage
            current_scenarios = set(current.detections.keys())
            other_scenarios = set(other.detections.keys())
            overlap = current_scenarios & other_scenarios
            if len(overlap) > 0:
                continue  # They overlap, can't merge

            # Check bridge-position continuity: gap between last of one
            # and first of other should be reasonable
            c_max = max(current_scenarios)
            c_min = min(current_scenarios)
            o_max = max(other_scenarios)
            o_min = min(other_scenarios)

            # One should end before the other starts (approximately)
            if c_max < o_min or o_max < c_min:
                # Check frequency continuity at the junction
                if c_max < o_min:
                    end_freq = current.detections[c_max].frequency
                    start_freq = other.detections[o_min].frequency
                else:
                    end_freq = other.detections[o_max].frequency
                    start_freq = current.detections[c_min].frequency

                junction_diff = abs(end_freq - start_freq) / min(end_freq, start_freq)
                if junction_diff <= freq_tol_pct * 3:  # Slightly relaxed
                    # Merge: absorb other into current
                    for sc_idx, det in other.detections.items():
                        current.detections[sc_idx] = det
                    current.finalize(total_scenarios)
                    merged[j] = True

        result.append(current)

    return result
