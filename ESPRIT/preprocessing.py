"""
preprocessing.py
Signal preprocessing utilities for ESPRIT modal identification.

Implements onset detection, windowing, and signal conditioning
per project specification.
"""
from __future__ import annotations
import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass


@dataclass
class PreprocessingConfig:
    """Configuration parameters for preprocessing."""
    force_threshold_abs: float = 5e-3  # Absolute force threshold in N
    force_threshold_deriv: Optional[float] = None  # Derivative threshold (computed if None)
    contact_tail_frac: float = 0.03  # Fraction of peak for contact end detection
    delta_after_contact: float = 1e-3  # Delay after contact in seconds
    target_tail_db: float = -70  # Target attenuation at window end in dB
    hp_cut_hz: float = 1.0  # High-pass filter cutoff in Hz


def detect_onset(force: np.ndarray, threshold_abs: float = 5e-3,
                 threshold_deriv: Optional[float] = None) -> int:
    """
    Detect impact onset from force signal.

    Args:
        force: Force signal in N, shape (T,)
        threshold_abs: Absolute force threshold
        threshold_deriv: Derivative threshold (if None, computed as 3*std(dF))

    Returns:
        n0: Onset index
    """
    # Absolute threshold crossing
    abs_crossings = np.where(force > threshold_abs)[0]

    if len(abs_crossings) == 0:
        # Fallback: use first significant value
        return np.argmax(np.abs(force))

    # Derivative threshold
    dforce = np.diff(force)
    if threshold_deriv is None:
        threshold_deriv = 3 * np.std(dforce)

    deriv_crossings = np.where(dforce > threshold_deriv)[0]

    # Take earliest detection
    if len(deriv_crossings) > 0:
        n0 = min(abs_crossings[0], deriv_crossings[0])
    else:
        n0 = abs_crossings[0]

    return int(n0)


def detect_contact_end(force: np.ndarray, onset: int, tail_frac: float = 0.03) -> int:
    """
    Detect end of hammer-structure contact.

    Args:
        force: Force signal in N
        onset: Onset index
        tail_frac: Fraction of peak force for tail detection

    Returns:
        nC: Contact end index
    """
    # Find peak after onset
    force_after_onset = force[onset:]
    peak_idx = np.argmax(np.abs(force_after_onset))
    peak_value = np.abs(force_after_onset[peak_idx])

    # Find where force drops below threshold after peak
    threshold = tail_frac * peak_value
    tail_region = force_after_onset[peak_idx:]

    below_threshold = np.where(np.abs(tail_region) < threshold)[0]

    if len(below_threshold) > 0:
        contact_end = onset + peak_idx + below_threshold[0]
    else:
        # Fallback: use 2x peak index
        contact_end = onset + 2 * peak_idx

    return int(contact_end)


def create_exponential_window(length: int, target_db: float = -70) -> np.ndarray:
    """
    Create exponential decay window for response signals.

    The window achieves target_db attenuation at the end.

    Args:
        length: Window length in samples
        target_db: Target attenuation in dB at window end

    Returns:
        window: Exponential window, shape (length,)
    """
    # A_end / A_0 = 10^(target_db/20)
    attenuation_ratio = 10 ** (target_db / 20)

    # α = ln(A_0 / A_end) / T
    alpha = -np.log(attenuation_ratio) / length

    # w[n] = exp(-α * n)
    n = np.arange(length)
    window = np.exp(-alpha * n)

    return window


def extract_decay_segment(response: np.ndarray, contact_end: int,
                          delta_samples: int, window_length: int,
                          fs: float, target_db: float = -70) -> np.ndarray:
    """
    Extract windowed free decay segment from response signal.

    Args:
        response: Response signal (single channel)
        contact_end: Contact end index
        delta_samples: Delay after contact in samples
        window_length: Desired window length in samples
        fs: Sampling frequency
        target_db: Target window attenuation in dB

    Returns:
        segment: Windowed decay segment
    """
    start_idx = contact_end + delta_samples
    end_idx = start_idx + window_length

    # Check bounds
    if end_idx > len(response):
        end_idx = len(response)
        window_length = end_idx - start_idx

    # Extract segment
    segment = response[start_idx:end_idx]

    # Apply exponential window
    window = create_exponential_window(window_length, target_db)
    windowed_segment = segment * window

    return windowed_segment


def highpass_filter(signal: np.ndarray, fs: float, cutoff: float = 1.0) -> np.ndarray:
    """
    Apply simple high-pass filter to remove DC drift.

    Uses a 2nd-order Butterworth filter.

    Args:
        signal: Input signal
        fs: Sampling frequency in Hz
        cutoff: Cutoff frequency in Hz

    Returns:
        filtered: High-pass filtered signal
    """
    from scipy import signal as sp_signal

    # Design Butterworth high-pass filter
    sos = sp_signal.butter(2, cutoff, btype='highpass', fs=fs, output='sos')

    # Apply filter
    filtered = sp_signal.sosfiltfilt(sos, signal, axis=0)

    return filtered


def preprocess_measurement(force: np.ndarray, responses: np.ndarray, fs: float,
                           config: Optional[PreprocessingConfig] = None) -> Tuple[np.ndarray, dict]:
    """
    Complete preprocessing pipeline for measurement data.

    Args:
        force: Force signal, shape (T,)
        responses: Multi-channel response signals, shape (T, n_channels)
        fs: Sampling frequency in Hz
        config: Preprocessing configuration

    Returns:
        decay_segments: Windowed decay segments, shape (T_win, n_channels)
        metadata: Dictionary with preprocessing information
    """
    if config is None:
        config = PreprocessingConfig()

    # Detect onset
    n0 = detect_onset(force, config.force_threshold_abs, config.force_threshold_deriv)

    # Detect contact end
    nC = detect_contact_end(force, n0, config.contact_tail_frac)

    # Compute delta in samples
    delta_samples = int(config.delta_after_contact * fs)

    # Determine window length (use remaining signal after contact + delta)
    max_length = len(responses) - (nC + delta_samples)
    window_length = max_length  # Use full available decay

    # Apply high-pass filter to responses
    responses_filtered = highpass_filter(responses, fs, config.hp_cut_hz)

    # Extract decay segments for all channels
    n_channels = responses.shape[1]
    decay_segments = np.zeros((window_length, n_channels))

    for ch in range(n_channels):
        decay_segments[:, ch] = extract_decay_segment(
            responses_filtered[:, ch], nC, delta_samples,
            window_length, fs, config.target_tail_db
        )

    metadata = {
        'onset_index': n0,
        'contact_end_index': nC,
        'onset_time_s': n0 / fs,
        'contact_end_time_s': nC / fs,
        'decay_start_index': nC + delta_samples,
        'decay_length': window_length,
        'decay_duration_s': window_length / fs,
        'fs_hz': fs,
    }

    return decay_segments, metadata


def load_measurement_file(filepath: str, skip_channel: int = 2) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load measurement data from TSV file.

    Args:
        filepath: Path to measurement file
        skip_channel: Channel to skip (calibration channel)

    Returns:
        force: Force signal (from skip_channel), shape (T,)
        responses: Response signals (other channels), shape (T, n_channels)
    """
    # Load data
    data = np.loadtxt(filepath, comments='#')

    # Extract force (calibration channel)
    force = data[:, skip_channel]

    # Extract response channels (all except calibration)
    channels = [i for i in range(data.shape[1]) if i != skip_channel]
    responses = data[:, channels]

    return force, responses


def check_signal_quality(force: np.ndarray, responses: np.ndarray,
                         clip_threshold: float = 0.95) -> dict:
    """
    Perform sanity checks on signal quality.

    Args:
        force: Force signal
        responses: Response signals
        clip_threshold: Threshold for clipping detection (as fraction of max)

    Returns:
        quality_report: Dictionary with quality indicators
    """
    quality = {}

    # Check for clipping in force
    force_max = np.max(np.abs(force))
    force_near_max = np.sum(np.abs(force) > clip_threshold * force_max)
    quality['force_clipped'] = force_near_max > 10  # More than 10 samples near max

    # Check for clipping in responses
    resp_max = np.max(np.abs(responses), axis=0)
    resp_clipping = []
    for ch in range(responses.shape[1]):
        near_max = np.sum(np.abs(responses[:, ch]) > clip_threshold * resp_max[ch])
        resp_clipping.append(near_max > 10)
    quality['response_clipped'] = resp_clipping

    # Check for double hits (multiple peaks in force)
    from scipy import signal as sp_signal
    peaks, _ = sp_signal.find_peaks(force, height=0.1*force_max, distance=int(0.01*len(force)))
    quality['n_force_peaks'] = len(peaks)
    quality['double_hit'] = len(peaks) > 1

    # Signal-to-noise estimate (RMS of first 10% vs last 10%)
    noise_estimate = np.std(responses[:len(responses)//10, :], axis=0)
    signal_rms = np.std(responses, axis=0)
    quality['snr_db'] = 20 * np.log10(signal_rms / (noise_estimate + 1e-12))

    return quality
