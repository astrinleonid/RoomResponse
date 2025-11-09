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
                           config: Optional[MinimalPreprocessingConfig] = None) -> Tuple[np.ndarray, dict]:
    """
    Minimal preprocessing for pre-processed measurement data.

    IMPORTANT: This assumes data has ALREADY been processed by RoomResponseRecorder:
    - Aligned by force channel maximum
    - Quality filtered
    - Hann fade applied

    This function only:
    1. Optionally applies high-pass filter for DC removal
    2. Returns the full response signals
    3. Provides basic metadata

    NO WINDOWING, NO ONSET DETECTION, NO SEGMENTATION.

    Args:
        force: Force signal, shape (T,)
        responses: Multi-channel responses, shape (T, n_channels)
        fs: Sampling frequency in Hz
        config: Minimal preprocessing configuration

    Returns:
        processed_responses: Processed response signals, shape (T, n_channels)
        metadata: Dictionary with basic information
    """
    if config is None:
        config = MinimalPreprocessingConfig()

    T = len(responses)
    n_channels = responses.shape[1]

    # Optional high-pass filtering (DC removal only)
    if config.use_highpass:
        processed_responses = highpass_filter(responses, fs, config.hp_cut_hz)
    else:
        processed_responses = responses.copy()

    # Basic metadata
    metadata = {
        'fs_hz': fs,
        'n_samples': T,
        'n_channels': n_channels,
        'duration_s': T / fs,
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
