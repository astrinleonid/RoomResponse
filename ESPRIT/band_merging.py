"""
band_merging.py
Multi-band mode merging with MAC validation and band-center-weighted frequency estimation.

When ESPRIT runs multi-band analysis, the same physical mode can appear in overlapping
bands at slightly different frequencies (5-20 Hz shift) due to bandpass filter edge
effects. This module deduplicates those cross-band detections using:

1. Band-center confidence weighting: modes near band center are more trustworthy
2. Modal Assurance Criterion (MAC): validates spatial consistency of mode shapes
3. Frequency proximity + damping similarity fallback when shapes are unavailable
"""
from __future__ import annotations
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass, field


@dataclass
class BandMode:
    """A mode detected in a specific frequency band, with band metadata."""
    frequency: float                  # Natural frequency (Hz)
    damping: float                    # Damping ratio (dimensionless)
    pole: complex                     # Continuous-time pole
    mode_shape: Optional[np.ndarray]  # Complex mode shape vector, shape (n_channels,)
    band_index: int                   # Which band produced this mode
    band_f_min: float                 # Band lower bound (Hz)
    band_f_max: float                 # Band upper bound (Hz)
    center_weight: float = 0.0        # Band-center confidence weight (0-1)
    band_name: str = ""               # Band name for reporting


def band_center_weight(freq: float, f_min: float, f_max: float) -> float:
    """
    Compute how well-centered a detected mode is within its band.

    Returns a weight from 0.0 (at band edge) to 1.0 (at band center).
    Modes near the center of a band are less affected by filter roll-off
    and are therefore more reliable estimates.

    Args:
        freq: Detected mode frequency (Hz)
        f_min: Band lower bound (Hz)
        f_max: Band upper bound (Hz)

    Returns:
        Weight in [0, 1]. 1.0 at band center, 0.0 at or beyond edges.
    """
    center = (f_min + f_max) / 2
    half_width = (f_max - f_min) / 2
    if half_width <= 0:
        return 0.0
    distance_from_center = abs(freq - center)
    weight = max(0.0, 1.0 - (distance_from_center / half_width))
    return weight


def compute_mac(shape1: np.ndarray, shape2: np.ndarray) -> float:
    """
    Modal Assurance Criterion between two complex mode shape vectors.

    MAC quantifies the spatial consistency between two mode shapes.
    MAC = 1.0 means identical spatial patterns; MAC = 0.0 means orthogonal.

    Args:
        shape1: Complex mode shape vector, shape (n_channels,)
        shape2: Complex mode shape vector, shape (n_channels,)

    Returns:
        MAC value in [0, 1].
    """
    numerator = abs(np.dot(shape1.conj(), shape2)) ** 2
    denom1 = np.dot(shape1.conj(), shape1).real
    denom2 = np.dot(shape2.conj(), shape2).real
    denominator = denom1 * denom2
    if denominator < 1e-30:
        return 0.0
    return float(numerator / denominator)


def _modes_match_with_shapes(
    m1: BandMode, m2: BandMode,
    freq_tol_pct: float, mac_threshold: float
) -> Tuple[bool, float]:
    """
    Check if two modes match using frequency proximity + MAC.

    Returns:
        (is_match, mac_value)
    """
    # Frequency proximity check (relative tolerance)
    f_ref = max(m1.frequency, 1.0)
    if abs(m1.frequency - m2.frequency) / f_ref > freq_tol_pct:
        return False, 0.0

    # MAC check
    if m1.mode_shape is None or m2.mode_shape is None:
        return False, 0.0
    if len(m1.mode_shape) != len(m2.mode_shape):
        return False, 0.0

    mac_val = compute_mac(m1.mode_shape, m2.mode_shape)
    return mac_val >= mac_threshold, mac_val


def _modes_match_no_shapes(
    m1: BandMode, m2: BandMode,
    freq_tol_pct: float, damping_rel_tol: float = 0.5
) -> bool:
    """
    Fallback matching when mode shapes are not available.
    Uses frequency proximity + damping similarity.

    Args:
        m1, m2: Modes to compare
        freq_tol_pct: Frequency tolerance as fraction of frequency
        damping_rel_tol: Relative damping tolerance (default 50%)

    Returns:
        True if modes are considered the same physical mode
    """
    # Frequency proximity
    f_ref = max(m1.frequency, 1.0)
    if abs(m1.frequency - m2.frequency) / f_ref > freq_tol_pct:
        return False

    # Damping similarity (within relative tolerance)
    d_ref = max(abs(m1.damping), abs(m2.damping), 1e-6)
    if abs(m1.damping - m2.damping) / d_ref > damping_rel_tol:
        return False

    return True


def merge_multiband_modes(
    band_results: list,
    bands: list,
    mac_threshold: float = 0.9,
    freq_tol_pct: float = 0.01,
    damping_rel_tol: float = 0.5,
) -> Dict[str, Any]:
    """
    Merge modes from overlapping frequency bands using MAC + band-center weighting.

    When multiple bands detect the same physical mode, keeps the estimate from
    the band where it is best-centered (highest band_center_weight), since
    modes near band center are least affected by filter edge distortion.

    Args:
        band_results: List of ModalParameters objects, one per band. Each must have
                      .frequencies, .damping_ratios, .poles, .mode_shapes attributes.
        bands: List of FrequencyBand configs (must have .f_min, .f_max, .name).
        mac_threshold: Minimum MAC for two modes to be considered the same (default 0.9).
        freq_tol_pct: Frequency tolerance as fraction of frequency (default 1%).
        damping_rel_tol: Relative damping tolerance for shape-free fallback (default 50%).

    Returns:
        dict with keys:
            'frequencies': np.ndarray of merged frequencies
            'damping_ratios': np.ndarray of merged damping ratios
            'poles': np.ndarray of merged poles
            'mode_shapes': np.ndarray of merged mode shapes (or None)
            'band_names': list of band names for each kept mode
            'center_weights': np.ndarray of center weights
            'n_raw': total raw modes before merging
            'n_merged': number of duplicate pairs removed
            'merge_log': list of dicts describing each merge decision
    """
    # Step 1: Collect all modes with band metadata
    all_modes: List[BandMode] = []

    for band_idx, (result, band) in enumerate(zip(band_results, bands)):
        if result is None or len(result.frequencies) == 0:
            continue

        has_shapes = (
            result.mode_shapes is not None
            and len(result.mode_shapes) > 0
            and result.mode_shapes.ndim == 2
            and result.mode_shapes.shape[0] == len(result.frequencies)
        )

        for i in range(len(result.frequencies)):
            shape = result.mode_shapes[i] if has_shapes else None
            freq = result.frequencies[i]
            cw = band_center_weight(freq, band.f_min, band.f_max)

            all_modes.append(BandMode(
                frequency=freq,
                damping=result.damping_ratios[i],
                pole=result.poles[i],
                mode_shape=shape,
                band_index=band_idx,
                band_f_min=band.f_min,
                band_f_max=band.f_max,
                center_weight=cw,
                band_name=getattr(band, 'name', f"Band{band_idx}"),
            ))

    n_raw = len(all_modes)

    if n_raw == 0:
        return {
            'frequencies': np.array([]),
            'damping_ratios': np.array([]),
            'poles': np.array([], dtype=complex),
            'mode_shapes': None,
            'band_names': [],
            'center_weights': np.array([]),
            'n_raw': 0,
            'n_merged': 0,
            'merge_log': [],
        }

    # Step 2: Sort by frequency
    all_modes.sort(key=lambda m: m.frequency)

    # Step 3: Greedy deduplication
    # Determine if we can use MAC (need shapes with consistent channel count)
    shape_lengths = [len(m.mode_shape) for m in all_modes if m.mode_shape is not None]
    use_mac = len(shape_lengths) > 0 and len(set(shape_lengths)) == 1

    accepted: List[BandMode] = []
    discarded_indices = set()
    merge_log: List[Dict] = []

    for i, mode_i in enumerate(all_modes):
        if i in discarded_indices:
            continue

        # Check against already-accepted modes
        merged = False
        for j, mode_j in enumerate(accepted):
            if use_mac:
                is_match, mac_val = _modes_match_with_shapes(
                    mode_i, mode_j, freq_tol_pct, mac_threshold
                )
            else:
                is_match = _modes_match_no_shapes(
                    mode_i, mode_j, freq_tol_pct, damping_rel_tol
                )
                mac_val = None

            if is_match:
                # Keep the one with higher band-center weight
                if mode_i.center_weight > mode_j.center_weight:
                    # Replace accepted mode with this better-centered one
                    log_entry = {
                        'kept_freq': mode_i.frequency,
                        'kept_band': mode_i.band_name,
                        'kept_weight': mode_i.center_weight,
                        'discarded_freq': mode_j.frequency,
                        'discarded_band': mode_j.band_name,
                        'discarded_weight': mode_j.center_weight,
                        'mac': mac_val,
                    }
                    accepted[j] = mode_i
                else:
                    log_entry = {
                        'kept_freq': mode_j.frequency,
                        'kept_band': mode_j.band_name,
                        'kept_weight': mode_j.center_weight,
                        'discarded_freq': mode_i.frequency,
                        'discarded_band': mode_i.band_name,
                        'discarded_weight': mode_i.center_weight,
                        'mac': mac_val,
                    }
                merge_log.append(log_entry)
                merged = True
                break

        if not merged:
            accepted.append(mode_i)

    # Step 4: Build output arrays
    n_merged_count = n_raw - len(accepted)

    frequencies = np.array([m.frequency for m in accepted])
    damping_ratios = np.array([m.damping for m in accepted])
    poles = np.array([m.pole for m in accepted])
    center_weights = np.array([m.center_weight for m in accepted])
    band_names = [m.band_name for m in accepted]

    # Mode shapes: only if all accepted modes have them
    if all(m.mode_shape is not None for m in accepted) and len(accepted) > 0:
        mode_shapes = np.array([m.mode_shape for m in accepted])
    else:
        mode_shapes = None

    # Sort by frequency
    sort_idx = np.argsort(frequencies)
    frequencies = frequencies[sort_idx]
    damping_ratios = damping_ratios[sort_idx]
    poles = poles[sort_idx]
    center_weights = center_weights[sort_idx]
    band_names = [band_names[i] for i in sort_idx]
    if mode_shapes is not None:
        mode_shapes = mode_shapes[sort_idx]

    return {
        'frequencies': frequencies,
        'damping_ratios': damping_ratios,
        'poles': poles,
        'mode_shapes': mode_shapes,
        'band_names': band_names,
        'center_weights': center_weights,
        'n_raw': n_raw,
        'n_merged': n_merged_count,
        'merge_log': merge_log,
    }
