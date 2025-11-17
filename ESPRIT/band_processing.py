"""
band_processing.py
Multi-band signal processing for ESPRIT modal identification.

This module implements the frequency band decomposition strategy used in the
reference esprit.py implementation. Different frequency ranges are processed
with band-specific filtering, decimation, and exponential pre-emphasis.
"""
from __future__ import annotations
import numpy as np
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
from scipy import signal as sp_signal


@dataclass
class FrequencyBand:
    """Configuration for a single frequency band."""
    f_min: float           # Minimum frequency (Hz)
    f_max: float           # Maximum frequency (Hz)
    filter_order: int      # Butterworth filter order
    decimation: int        # Decimation factor
    exp_factor: float      # Exponential pre-emphasis factor (positive, mild)
    name: str = ""         # Optional band name

    def __post_init__(self):
        if not self.name:
            self.name = f"{self.f_min:.0f}-{self.f_max:.0f}Hz"


# Standard band presets matching esprit.py
STANDARD_BANDS = [
    FrequencyBand(f_min=30, f_max=200, filter_order=4, decimation=4, exp_factor=0.3, name="Low"),
    FrequencyBand(f_min=150, f_max=500, filter_order=5, decimation=2, exp_factor=0.2, name="Mid-Low"),
    FrequencyBand(f_min=400, f_max=1500, filter_order=6, decimation=1, exp_factor=0.1, name="Mid-High"),
    FrequencyBand(f_min=1200, f_max=5000, filter_order=8, decimation=1, exp_factor=0.05, name="High"),
]


def design_bandpass_filter(f_min: float, f_max: float, fs: float,
                           order: int = 4) -> np.ndarray:
    """
    Design Butterworth bandpass filter.

    Args:
        f_min: Lower cutoff frequency (Hz)
        f_max: Upper cutoff frequency (Hz)
        fs: Sampling frequency (Hz)
        order: Filter order (default: 4)

    Returns:
        sos: Second-order sections for filtering
    """
    nyq = fs / 2

    # Ensure frequencies are valid
    f_min_norm = max(f_min, 1.0) / nyq
    f_max_norm = min(f_max, nyq * 0.99) / nyq

    if f_min_norm >= f_max_norm:
        raise ValueError(f"Invalid frequency range: {f_min}-{f_max} Hz at fs={fs} Hz")

    # Design Butterworth bandpass
    sos = sp_signal.butter(order, [f_min_norm, f_max_norm],
                          btype='bandpass', output='sos')

    return sos


def apply_exponential_preemphasis(signal: np.ndarray, exp_factor: float,
                                  fs: float) -> np.ndarray:
    """
    Apply mild exponential pre-emphasis to enhance later portions of signal.

    This is the OPPOSITE of exponential windowing for damping extraction.
    Used here to compensate for natural decay before ESPRIT processing,
    improving SNR for later modes.

    Args:
        signal: Input signal, shape (T,) or (T, n_channels)
        exp_factor: Positive exponential factor (typical: 0.05-0.3)
        fs: Sampling frequency (Hz)

    Returns:
        emphasized: Signal with exponential pre-emphasis

    Note:
        - exp_factor is POSITIVE (enhances tail)
        - Much milder than the -70dB windowing we removed
        - Applied BEFORE ESPRIT, not during preprocessing
    """
    T = len(signal)
    t = np.arange(T) / fs

    # Exponential ramp: exp(+alpha * t)
    # Small positive alpha slightly boosts decaying tail
    window = np.exp(exp_factor * t)

    if signal.ndim == 1:
        return signal * window
    else:
        return signal * window[:, np.newaxis]


def process_band(signals: np.ndarray, fs: float, band: FrequencyBand,
                 apply_preemphasis: bool = True) -> Tuple[np.ndarray, float, Dict]:
    """
    Process signals for a specific frequency band.

    Pipeline:
    1. Bandpass filter to band range
    2. Optional exponential pre-emphasis
    3. Decimation to reduce computational cost

    Args:
        signals: Multi-channel signals, shape (T, n_channels)
        fs: Sampling frequency (Hz)
        band: Band configuration
        apply_preemphasis: Apply exponential pre-emphasis (default: True)

    Returns:
        processed: Band-processed signals, shape (T_dec, n_channels)
        fs_band: Effective sampling frequency after decimation
        metadata: Processing information
    """
    T, n_channels = signals.shape

    # Step 1: Bandpass filter
    sos = design_bandpass_filter(band.f_min, band.f_max, fs, band.filter_order)
    filtered = sp_signal.sosfiltfilt(sos, signals, axis=0)

    # Step 2: Exponential pre-emphasis (optional, mild)
    if apply_preemphasis and band.exp_factor > 0:
        emphasized = apply_exponential_preemphasis(filtered, band.exp_factor, fs)
    else:
        emphasized = filtered

    # Step 3: Decimation
    if band.decimation > 1:
        # Use scipy's decimate for anti-aliasing
        decimated = sp_signal.decimate(emphasized, band.decimation, axis=0,
                                       ftype='iir', zero_phase=True)
    else:
        decimated = emphasized

    fs_band = fs / band.decimation

    metadata = {
        'band_name': band.name,
        'f_range': (band.f_min, band.f_max),
        'fs_original': fs,
        'fs_decimated': fs_band,
        'decimation_factor': band.decimation,
        'filter_order': band.filter_order,
        'exp_factor': band.exp_factor,
        'n_samples_original': T,
        'n_samples_decimated': len(decimated),
        'preemphasis_applied': apply_preemphasis and band.exp_factor > 0
    }

    return decimated, fs_band, metadata


def select_bands_for_range(freq_range: Tuple[float, float],
                           bands: List[FrequencyBand] = None) -> List[FrequencyBand]:
    """
    Select relevant frequency bands that overlap with target range.

    Args:
        freq_range: Target frequency range (f_min, f_max) in Hz
        bands: List of band configurations (default: STANDARD_BANDS)

    Returns:
        selected_bands: Bands that overlap with freq_range
    """
    if bands is None:
        bands = STANDARD_BANDS

    f_target_min, f_target_max = freq_range

    selected = []
    for band in bands:
        # Check for overlap: band overlaps if its max > target_min AND its min < target_max
        if band.f_max > f_target_min and band.f_min < f_target_max:
            selected.append(band)

    return selected


def process_multiband(signals: np.ndarray, fs: float,
                     freq_range: Tuple[float, float] = (0, np.inf),
                     bands: Optional[List[FrequencyBand]] = None,
                     apply_preemphasis: bool = True) -> List[Tuple[np.ndarray, float, Dict]]:
    """
    Process signals across multiple frequency bands.

    Args:
        signals: Multi-channel signals, shape (T, n_channels)
        fs: Sampling frequency (Hz)
        freq_range: Target frequency range (default: full range)
        bands: Band configurations (default: STANDARD_BANDS)
        apply_preemphasis: Apply exponential pre-emphasis (default: True)

    Returns:
        band_results: List of (processed_signals, fs_band, metadata) for each band
    """
    if bands is None:
        bands = STANDARD_BANDS

    # Select bands that overlap with target frequency range
    selected_bands = select_bands_for_range(freq_range, bands)

    if len(selected_bands) == 0:
        # No bands selected - create single custom band
        f_min = max(freq_range[0], 10.0)
        f_max = min(freq_range[1], fs / 2 * 0.99)
        custom_band = FrequencyBand(
            f_min=f_min, f_max=f_max,
            filter_order=6, decimation=1, exp_factor=0.1,
            name=f"Custom_{f_min:.0f}-{f_max:.0f}Hz"
        )
        selected_bands = [custom_band]

    # Process each band
    band_results = []
    for band in selected_bands:
        processed, fs_band, metadata = process_band(
            signals, fs, band, apply_preemphasis=apply_preemphasis
        )
        band_results.append((processed, fs_band, metadata))

    return band_results


def validate_band_coverage(bands: List[FrequencyBand],
                          target_range: Tuple[float, float]) -> Dict:
    """
    Validate that bands adequately cover the target frequency range.

    Args:
        bands: List of band configurations
        target_range: Target (f_min, f_max) in Hz

    Returns:
        validation_report: Coverage statistics
    """
    if len(bands) == 0:
        return {
            'is_valid': False,
            'coverage': 0.0,
            'gaps': [(target_range[0], target_range[1])],
            'overlaps': []
        }

    f_target_min, f_target_max = target_range
    target_span = f_target_max - f_target_min

    # Sort bands by f_min
    sorted_bands = sorted(bands, key=lambda b: b.f_min)

    # Find gaps and overlaps
    gaps = []
    overlaps = []

    # Check coverage from target_min
    if sorted_bands[0].f_min > f_target_min:
        gaps.append((f_target_min, sorted_bands[0].f_min))

    # Check gaps between consecutive bands
    for i in range(len(sorted_bands) - 1):
        b1 = sorted_bands[i]
        b2 = sorted_bands[i + 1]

        if b1.f_max < b2.f_min:
            # Gap
            gaps.append((b1.f_max, b2.f_min))
        elif b1.f_max > b2.f_min:
            # Overlap
            overlaps.append((b2.f_min, b1.f_max))

    # Check coverage to target_max
    if sorted_bands[-1].f_max < f_target_max:
        gaps.append((sorted_bands[-1].f_max, f_target_max))

    # Calculate coverage
    covered_span = sum(min(b.f_max, f_target_max) - max(b.f_min, f_target_min)
                      for b in sorted_bands)
    coverage = covered_span / target_span if target_span > 0 else 0.0

    return {
        'is_valid': coverage > 0.9,  # At least 90% coverage
        'coverage': coverage,
        'gaps': gaps,
        'overlaps': overlaps,
        'n_bands': len(bands)
    }
