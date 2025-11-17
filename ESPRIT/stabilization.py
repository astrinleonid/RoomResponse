"""
stabilization.py
Multi-point stabilization diagram for ESPRIT modal identification.

This module implements spatial stabilization: modes are extracted from multiple
excitation points (or measurement locations) and clustered to identify stable
physical modes that appear consistently across different spatial positions.
"""
from __future__ import annotations
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist


@dataclass
class ModeCandidate:
    """A mode candidate from one analysis run."""
    frequency: float       # Natural frequency (Hz)
    damping: float        # Damping ratio (dimensionless)
    pole: complex         # Continuous-time pole (s)
    mode_shape: Optional[np.ndarray] = None  # Mode shape vector
    quality: float = 1.0  # Quality metric (0-1)
    source_id: int = 0    # Source identifier (excitation point, parameter combo, etc.)
    band_name: str = ""   # Frequency band name


@dataclass
class StableMode:
    """A stable physical mode identified through clustering."""
    frequency: float          # Average frequency (Hz)
    damping: float           # Average damping ratio
    pole: complex            # Representative pole
    mode_shape: np.ndarray   # Average mode shape (if available)
    std_frequency: float     # Standard deviation of frequency
    std_damping: float       # Standard deviation of damping
    n_detections: int        # Number of times detected
    quality: float           # Average quality
    source_ids: List[int]    # All source IDs where detected
    band_name: str = ""      # Frequency band


def cluster_modes_hierarchical(candidates: List[ModeCandidate],
                               freq_tol_hz: float = 2.0,
                               damping_tol: float = 0.05) -> List[List[int]]:
    """
    Cluster mode candidates using hierarchical clustering.

    Modes are clustered based on frequency and damping proximity. Two modes
    belong to the same cluster if:
    - |f1 - f2| < freq_tol_hz
    - |zeta1 - zeta2| < damping_tol

    Args:
        candidates: List of mode candidates
        freq_tol_hz: Frequency tolerance (Hz)
        damping_tol: Damping ratio tolerance (dimensionless)

    Returns:
        clusters: List of lists, each containing indices of candidates in a cluster
    """
    if len(candidates) == 0:
        return []

    if len(candidates) == 1:
        return [[0]]

    # Build feature matrix: [frequency, damping]
    # Normalize to make tolerances comparable
    features = np.array([
        [c.frequency / freq_tol_hz, c.damping / damping_tol]
        for c in candidates
    ])

    # Hierarchical clustering with single linkage
    # Distance threshold = sqrt(1^2 + 1^2) = sqrt(2) for normalized features
    try:
        condensed_dist = pdist(features, metric='euclidean')
        Z = linkage(condensed_dist, method='single')
        cluster_labels = fcluster(Z, t=np.sqrt(2), criterion='distance')
    except:
        # Fallback: each candidate is its own cluster
        return [[i] for i in range(len(candidates))]

    # Group indices by cluster label
    clusters = {}
    for idx, label in enumerate(cluster_labels):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(idx)

    return list(clusters.values())


def average_mode_shapes(shapes: List[np.ndarray],
                       correlation_threshold: float = 0.7) -> Optional[np.ndarray]:
    """
    Average mode shapes with sign/phase alignment.

    Mode shapes from different analyses may have arbitrary sign/phase.
    This function aligns them using correlation before averaging.

    Args:
        shapes: List of mode shape vectors (same length)
        correlation_threshold: Minimum correlation to accept (default: 0.7)

    Returns:
        averaged_shape: Averaged and normalized mode shape, or None if inconsistent
    """
    if len(shapes) == 0:
        return None

    if len(shapes) == 1:
        shape = shapes[0].copy()
        return shape / np.max(np.abs(shape))  # Normalize

    # Use first shape as reference
    ref_shape = shapes[0]
    aligned_shapes = [ref_shape]

    for shape in shapes[1:]:
        # Try both positive and negative alignment
        corr_pos = np.abs(np.dot(shape.conj(), ref_shape))
        corr_neg = np.abs(np.dot(-shape.conj(), ref_shape))

        if corr_pos > corr_neg:
            aligned = shape
            corr = corr_pos
        else:
            aligned = -shape
            corr = corr_neg

        # Normalize correlation by magnitudes
        corr /= (np.linalg.norm(shape) * np.linalg.norm(ref_shape) + 1e-10)

        # Only accept if correlation is high enough
        if corr > correlation_threshold:
            aligned_shapes.append(aligned)

    if len(aligned_shapes) < len(shapes) / 2:
        # Less than half passed correlation test - inconsistent
        return None

    # Average aligned shapes
    avg_shape = np.mean(aligned_shapes, axis=0)

    # Normalize
    avg_shape /= (np.max(np.abs(avg_shape)) + 1e-10)

    return avg_shape


def stabilize_modes(candidates: List[ModeCandidate],
                   freq_tol_hz: float = 2.0,
                   damping_tol: float = 0.05,
                   min_detections: int = 2,
                   shape_correlation_threshold: float = 0.7) -> List[StableMode]:
    """
    Identify stable modes from a list of candidates through clustering.

    A mode is considered "stable" if it appears consistently across multiple
    analyses (e.g., different excitation points, different parameter settings).

    Args:
        candidates: List of mode candidates from various analyses
        freq_tol_hz: Frequency clustering tolerance (Hz)
        damping_tol: Damping clustering tolerance
        min_detections: Minimum number of detections to be considered stable
        shape_correlation_threshold: Minimum correlation for mode shape averaging

    Returns:
        stable_modes: List of stable modes with averaged properties
    """
    if len(candidates) == 0:
        return []

    # Cluster candidates
    clusters = cluster_modes_hierarchical(candidates, freq_tol_hz, damping_tol)

    stable_modes = []

    for cluster_indices in clusters:
        cluster_modes = [candidates[i] for i in cluster_indices]
        n_detections = len(cluster_modes)

        # Filter by minimum detections
        if n_detections < min_detections:
            continue

        # Extract properties
        frequencies = np.array([m.frequency for m in cluster_modes])
        dampings = np.array([m.damping for m in cluster_modes])
        poles = np.array([m.pole for m in cluster_modes])
        qualities = np.array([m.quality for m in cluster_modes])
        source_ids = [m.source_id for m in cluster_modes]

        # Compute averages and standard deviations
        avg_freq = np.mean(frequencies)
        std_freq = np.std(frequencies)
        avg_damp = np.mean(dampings)
        std_damp = np.std(dampings)
        avg_quality = np.mean(qualities)

        # Average pole (complex)
        avg_pole = np.mean(poles)

        # Average mode shapes if available
        shapes = [m.mode_shape for m in cluster_modes if m.mode_shape is not None]
        if len(shapes) > 0:
            avg_shape = average_mode_shapes(shapes, shape_correlation_threshold)
        else:
            avg_shape = None

        # Get band name (use most common)
        band_names = [m.band_name for m in cluster_modes if m.band_name]
        if band_names:
            band_name = max(set(band_names), key=band_names.count)
        else:
            band_name = ""

        stable_mode = StableMode(
            frequency=avg_freq,
            damping=avg_damp,
            pole=avg_pole,
            mode_shape=avg_shape,
            std_frequency=std_freq,
            std_damping=std_damp,
            n_detections=n_detections,
            quality=avg_quality,
            source_ids=source_ids,
            band_name=band_name
        )

        stable_modes.append(stable_mode)

    # Sort by frequency
    stable_modes.sort(key=lambda m: m.frequency)

    return stable_modes


def multipoint_stabilization(measurement_data: List[np.ndarray],
                            fs: float,
                            esprit_function,
                            esprit_params: Dict[str, Any],
                            freq_tol_hz: float = 2.0,
                            damping_tol: float = 0.05,
                            min_detections: int = 3) -> List[StableMode]:
    """
    Perform multi-point stabilization across multiple measurements.

    This is the main interface for spatial stabilization. It processes multiple
    measurements (e.g., from different excitation points) and identifies modes
    that appear consistently.

    Args:
        measurement_data: List of measurement arrays, each shape (T, n_channels)
        fs: Sampling frequency (Hz)
        esprit_function: ESPRIT analysis function (e.g., esprit_modal_identification)
        esprit_params: Parameters to pass to ESPRIT function (dict)
        freq_tol_hz: Frequency clustering tolerance (Hz)
        damping_tol: Damping clustering tolerance
        min_detections: Minimum detections to be stable (default: 3)

    Returns:
        stable_modes: List of stable modes

    Example:
        >>> from esprit_core import esprit_modal_identification
        >>> measurements = [data1, data2, data3]  # Different excitation points
        >>> params = {'model_order': 30, 'use_tls': True, 'use_conjugate_pairing': True}
        >>> stable = multipoint_stabilization(measurements, 48000, esprit_modal_identification, params)
    """
    all_candidates = []

    for source_id, measurement in enumerate(measurement_data):
        try:
            # Run ESPRIT on this measurement
            result = esprit_function(measurement, fs, **esprit_params)

            # Convert to ModeCandidate objects
            for i in range(len(result.frequencies)):
                candidate = ModeCandidate(
                    frequency=result.frequencies[i],
                    damping=result.damping_ratios[i],
                    pole=result.poles[i],
                    mode_shape=result.mode_shapes[i] if result.mode_shapes is not None else None,
                    quality=1.0,  # Could compute from singular values
                    source_id=source_id,
                    band_name=""
                )
                all_candidates.append(candidate)

        except Exception as e:
            # Skip failed measurements
            print(f"Warning: Measurement {source_id} failed: {e}")
            continue

    # Stabilize across all candidates
    stable_modes = stabilize_modes(
        all_candidates,
        freq_tol_hz=freq_tol_hz,
        damping_tol=damping_tol,
        min_detections=min_detections
    )

    return stable_modes


def multiband_multipoint_stabilization(measurement_data: List[np.ndarray],
                                      fs: float,
                                      esprit_function,
                                      esprit_params: Dict[str, Any],
                                      bands: List[Any],
                                      band_processor,
                                      freq_tol_hz: float = 2.0,
                                      damping_tol: float = 0.05,
                                      min_detections: int = 2) -> List[StableMode]:
    """
    Combined multi-band and multi-point stabilization.

    This is the most comprehensive analysis: processes multiple measurements
    across multiple frequency bands and identifies stable modes.

    Args:
        measurement_data: List of measurement arrays, each (T, n_channels)
        fs: Sampling frequency (Hz)
        esprit_function: ESPRIT analysis function
        esprit_params: Parameters for ESPRIT function
        bands: List of FrequencyBand objects
        band_processor: Function to process a band (from band_processing module)
        freq_tol_hz: Frequency clustering tolerance
        damping_tol: Damping clustering tolerance
        min_detections: Minimum detections across all bands+points

    Returns:
        stable_modes: Stable modes across all bands and points
    """
    all_candidates = []

    # Process each measurement
    for source_id, measurement in enumerate(measurement_data):
        # Process each frequency band
        for band in bands:
            try:
                # Band processing
                processed, fs_band, _ = band_processor(measurement, fs, band)

                # ESPRIT analysis on this band
                # Update freq_range to match band
                params_band = esprit_params.copy()
                params_band['freq_range'] = (band.f_min, band.f_max)

                result = esprit_function(processed, fs_band, **params_band)

                # Convert to candidates
                for i in range(len(result.frequencies)):
                    # Check if frequency is within band range
                    if band.f_min <= result.frequencies[i] <= band.f_max:
                        candidate = ModeCandidate(
                            frequency=result.frequencies[i],
                            damping=result.damping_ratios[i],
                            pole=result.poles[i],
                            mode_shape=result.mode_shapes[i] if result.mode_shapes is not None else None,
                            quality=1.0,
                            source_id=source_id,
                            band_name=band.name
                        )
                        all_candidates.append(candidate)

            except Exception as e:
                # Skip failed band processing
                print(f"Warning: Measurement {source_id}, band {band.name} failed: {e}")
                continue

    # Stabilize across all candidates
    stable_modes = stabilize_modes(
        all_candidates,
        freq_tol_hz=freq_tol_hz,
        damping_tol=damping_tol,
        min_detections=min_detections
    )

    return stable_modes
