"""
preprocessing_minimal.py
Minimal preprocessing for ESPRIT modal identification on pre-processed measurement data.

This module is designed for data that has ALREADY been processed by RoomResponseRecorder:
- Signals are aligned by force channel maximum (see signal_processor.py:align_cycles_by_onset)
- Quality filtering is applied (see CalibrationValidatorV2)
- Hann fade-out is applied at signal end (see signal_processor.py:truncate_with_fadeout)

Therefore, this preprocessing is MINIMAL and only performs:
1. Loading data from file
2. Optional high-pass filtering for DC removal
3. Simple segmentation (no windowing, no onset detection)

WARNING: Do NOT add exponential windowing here - it will corrupt damping estimates!
The existing Hann fade is sufficient and appropriate for the data.
"""
from __future__ import annotations
import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass


@dataclass
class MinimalPreprocessingConfig:
    """Minimal configuration for pre-processed data."""
    hp_cut_hz: float = 1.0  # High-pass filter cutoff (for DC removal only)
    use_highpass: bool = True  # Enable high-pass filtering
    remove_contact: bool = True  # Remove hammer contact period
    contact_tail_fraction: float = 0.03  # Fraction of peak for contact end detection
    contact_delay_samples: int = 0  # Additional delay after contact end


def load_measurement_file(filepath: str, skip_channel: int = 2) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load measurement data from TSV file.

    The data format is:
    - Column skip_channel (default: 2) = force/calibration channel
    - All other columns = response channels

    Args:
        filepath: Path to measurement file
        skip_channel: Channel to treat as force/calibration (default: 2)

    Returns:
        force: Force signal, shape (T,)
        responses: Response signals, shape (T, n_channels)
    """
    # Load data (comments starting with # are ignored)
    data = np.loadtxt(filepath, comments='#')

    # Extract force channel
    force = data[:, skip_channel]

    # Extract response channels (all except force)
    channels = [i for i in range(data.shape[1]) if i != skip_channel]
    responses = data[:, channels]

    return force, responses


def detect_contact_end_from_force(force: np.ndarray, tail_fraction: float = 0.03) -> int:
    """
    Detect when hammer-structure contact ends (dataset-level operation).

    This function is run ONCE per dataset to determine the contact end sample.
    The result is then applied universally to all channels.

    Physical meaning:
    - During contact: Force is transmitted through hammer tip (nonlinear Hertzian contact)
    - After contact: Structure vibrates freely (linear modal behavior)
    - ESPRIT requires free vibration, so contact period must be excluded

    Algorithm:
    1. Assumes force is aligned with peak (negative) at sample 0
    2. Finds where |force| drops below tail_fraction * |peak|
    3. This indicates hammer has separated from structure

    Args:
        force: Force signal with peak at sample 0, shape (T,)
        tail_fraction: Fraction of peak force for threshold (default: 0.03 = 3%)

    Returns:
        contact_end_idx: Sample index where contact ends

    Example:
        force_peak = -0.626 N at sample 0
        threshold = 0.03 * 0.626 = 0.019 N
        force[6] = 0.016 N < threshold → contact ends at sample 6
        Duration: 6 samples @ 48kHz = 0.125 ms
    """
    # Peak should be at sample 0 (data is pre-aligned)
    peak_force = abs(force[0])
    threshold = tail_fraction * peak_force

    # Search forward from peak to find where force drops below threshold
    for i in range(len(force)):
        if abs(force[i]) < threshold:
            return i

    # Fallback: if force never drops below threshold (shouldn't happen)
    # Assume typical contact duration of ~10 samples (~0.2 ms @ 48kHz)
    return min(10, len(force) - 1)


def extract_free_decay(responses: np.ndarray, force: np.ndarray,
                       contact_end_idx: Optional[int] = None,
                       tail_fraction: float = 0.03,
                       delay_samples: int = 0) -> Tuple[np.ndarray, int]:
    """
    Extract free vibration portion after hammer contact ends.

    This removes the forced response period (during contact) and returns
    only the free decay suitable for ESPRIT analysis.

    Args:
        responses: Response signals, shape (T, n_channels)
        force: Force signal, shape (T,)
        contact_end_idx: Pre-computed contact end index (if None, will detect)
        tail_fraction: Fraction of peak for detection (if contact_end_idx is None)
        delay_samples: Additional delay after contact end (default: 0)

    Returns:
        free_decay: Free vibration signals, shape (T_free, n_channels)
        contact_end_idx: Sample where contact ended

    Note:
        For batch processing, detect contact_end_idx ONCE using the force
        signal, then pass it to all subsequent calls for consistent cutting.
    """
    # Detect contact end if not provided
    if contact_end_idx is None:
        contact_end_idx = detect_contact_end_from_force(force, tail_fraction)

    # Starting index for free decay
    start_idx = contact_end_idx + delay_samples

    # Ensure we don't go past the end of the signal
    if start_idx >= len(responses):
        start_idx = len(responses) - 1

    # Extract free decay portion
    free_decay = responses[start_idx:, :]

    return free_decay, contact_end_idx


def highpass_filter(signal: np.ndarray, fs: float, cutoff: float = 1.0) -> np.ndarray:
    """
    Apply simple high-pass filter to remove DC drift.

    Uses 2nd-order Butterworth filter with zero-phase filtering.

    Args:
        signal: Input signal (1D or 2D with channels as columns)
        fs: Sampling frequency in Hz
        cutoff: Cutoff frequency in Hz

    Returns:
        filtered: High-pass filtered signal
    """
    from scipy import signal as sp_signal

    # Design Butterworth high-pass filter
    sos = sp_signal.butter(2, cutoff, btype='highpass', fs=fs, output='sos')

    # Apply zero-phase filter
    filtered = sp_signal.sosfiltfilt(sos, signal, axis=0)

    return filtered


def preprocess_measurement(force: np.ndarray, responses: np.ndarray, fs: float,
                           config: Optional[MinimalPreprocessingConfig] = None,
                           contact_end_idx: Optional[int] = None) -> Tuple[np.ndarray, dict]:
    """
    Minimal preprocessing for pre-processed measurement data.

    IMPORTANT: This assumes data has ALREADY been processed by RoomResponseRecorder:
    - Aligned by force channel maximum (peak at sample 0)
    - Quality filtered
    - Hann fade applied at end

    This function performs:
    1. Contact period removal (optional, configurable)
    2. High-pass filtering for DC removal (optional, configurable)
    3. Basic metadata collection

    NO EXPONENTIAL WINDOWING - would corrupt damping estimates!

    Args:
        force: Force signal, shape (T,)
        responses: Multi-channel responses, shape (T, n_channels)
        fs: Sampling frequency in Hz
        config: Minimal preprocessing configuration
        contact_end_idx: Pre-computed contact end index (if None, will detect if needed)

    Returns:
        processed_responses: Processed response signals, shape (T_free, n_channels)
        metadata: Dictionary with processing information

    Usage:
        # Single measurement
        processed, meta = preprocess_measurement(force, responses, fs)

        # Batch processing (consistent cutting)
        contact_end = detect_contact_end_from_force(force_ref)
        for force, responses in measurements:
            processed, meta = preprocess_measurement(
                force, responses, fs, contact_end_idx=contact_end
            )
    """
    if config is None:
        config = MinimalPreprocessingConfig()

    T_original = len(responses)
    n_channels = responses.shape[1]

    # Step 1: Remove contact period if enabled
    if config.remove_contact:
        responses_segmented, detected_contact_end = extract_free_decay(
            responses, force,
            contact_end_idx=contact_end_idx,
            tail_fraction=config.contact_tail_fraction,
            delay_samples=config.contact_delay_samples
        )
        contact_removed = True
        contact_end_sample = detected_contact_end
        contact_duration_ms = (detected_contact_end / fs) * 1000
    else:
        responses_segmented = responses
        contact_removed = False
        contact_end_sample = None
        contact_duration_ms = None

    # Step 2: Optional high-pass filtering (DC removal)
    if config.use_highpass:
        processed_responses = highpass_filter(responses_segmented, fs, config.hp_cut_hz)
    else:
        processed_responses = responses_segmented.copy()

    # Step 3: Metadata
    T_processed = len(processed_responses)
    metadata = {
        'fs_hz': fs,
        'n_samples_original': T_original,
        'n_samples_processed': T_processed,
        'n_channels': n_channels,
        'duration_original_s': T_original / fs,
        'duration_processed_s': T_processed / fs,
        'contact_removed': contact_removed,
        'contact_end_sample': contact_end_sample,
        'contact_duration_ms': contact_duration_ms,
        'contact_tail_fraction': config.contact_tail_fraction if contact_removed else None,
        'highpass_applied': config.use_highpass,
        'highpass_cutoff_hz': config.hp_cut_hz if config.use_highpass else None,
        'note': 'Data pre-processed by RoomResponseRecorder: aligned, filtered, faded'
    }

    return processed_responses, metadata


def check_signal_basic(force: np.ndarray, responses: np.ndarray) -> dict:
    """
    Basic signal quality checks (lightweight version).

    NOTE: Comprehensive quality checks are done during measurement by
    CalibrationValidatorV2. This is just for quick diagnostics.

    Args:
        force: Force signal
        responses: Response signals

    Returns:
        quality_report: Basic quality indicators
    """
    quality = {}

    # Check dimensions
    quality['force_samples'] = len(force)
    quality['response_samples'] = len(responses)
    quality['n_channels'] = responses.shape[1] if responses.ndim > 1 else 1

    # Force signal stats
    quality['force_max'] = float(np.max(np.abs(force)))
    quality['force_rms'] = float(np.sqrt(np.mean(force**2)))

    # Response signal stats per channel
    if responses.ndim > 1:
        quality['response_max'] = [float(np.max(np.abs(responses[:, ch])))
                                   for ch in range(responses.shape[1])]
        quality['response_rms'] = [float(np.sqrt(np.mean(responses[:, ch]**2)))
                                   for ch in range(responses.shape[1])]
    else:
        quality['response_max'] = [float(np.max(np.abs(responses)))]
        quality['response_rms'] = [float(np.sqrt(np.mean(responses**2)))]

    # Simple SNR estimate (first 10% vs full signal)
    n_noise = len(responses) // 10
    if responses.ndim > 1:
        noise_rms = np.std(responses[:n_noise, :], axis=0)
        signal_rms = np.std(responses, axis=0)
    else:
        noise_rms = np.std(responses[:n_noise])
        signal_rms = np.std(responses)

    quality['snr_db'] = 20 * np.log10(signal_rms / (noise_rms + 1e-12))
    if isinstance(quality['snr_db'], np.ndarray):
        quality['snr_db'] = quality['snr_db'].tolist()

    return quality
