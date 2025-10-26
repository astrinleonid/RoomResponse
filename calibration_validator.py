"""
Calibration validator for piano impulse response measurements.

This module validates calibration channel cycles against quality criteria to ensure
reliable impulse response measurements. Used in multi-channel recording pipeline.
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np


@dataclass
class CycleValidation:
    """Validation result for a single cycle"""
    cycle_index: int
    calibration_valid: bool
    calibration_metrics: Dict[str, float]
    calibration_failures: List[str]
    xcorr_valid: bool = True
    xcorr_mean: float = 1.0


class CalibrationValidator:
    """Validates calibration channel cycles against quality criteria"""

    def __init__(self, config: Dict, sample_rate: int):
        """
        Initialize calibration validator.

        Args:
            config: Calibration quality configuration dict with keys:
                - cal_min_amplitude: Minimum acceptable peak amplitude
                - cal_max_amplitude: Maximum acceptable peak amplitude
                - cal_min_duration_ms: Minimum impulse duration in milliseconds
                - cal_max_duration_ms: Maximum impulse duration in milliseconds
                - cal_duration_threshold: Threshold for measuring duration (fraction of peak)
                - cal_double_hit_window_ms: Time window to search for secondary impacts [start_ms, end_ms]
                - cal_double_hit_threshold: Threshold for detecting double hits (fraction of main peak)
                - cal_tail_start_ms: Where tail region begins after impulse (milliseconds)
                - cal_tail_max_rms_ratio: Maximum acceptable tail noise (fraction of impulse RMS)
            sample_rate: Audio sample rate in Hz
        """
        self.config = config
        self.sample_rate = sample_rate

    def validate_magnitude(self, cycle: np.ndarray) -> Tuple[bool, Dict]:
        """
        Check if peak amplitude is within acceptable range.

        Args:
            cycle: Single cycle audio data

        Returns:
            (passed, metrics) where metrics contains 'peak_amplitude'
        """
        peak = np.max(np.abs(cycle))
        min_amp = self.config['cal_min_amplitude']
        max_amp = self.config['cal_max_amplitude']

        passed = min_amp <= peak <= max_amp
        metrics = {'peak_amplitude': float(peak)}

        return passed, metrics

    def validate_duration(self, cycle: np.ndarray) -> Tuple[bool, Dict]:
        """
        Check if impulse duration is within acceptable range.

        Args:
            cycle: Single cycle audio data

        Returns:
            (passed, metrics) where metrics contains 'duration_ms'
        """
        peak = np.max(np.abs(cycle))
        threshold = self.config['cal_duration_threshold'] * peak

        above_threshold = np.abs(cycle) > threshold
        duration_samples = np.sum(above_threshold)
        duration_ms = (duration_samples / self.sample_rate) * 1000.0

        passed = (self.config['cal_min_duration_ms'] <= duration_ms <=
                  self.config['cal_max_duration_ms'])
        metrics = {'duration_ms': float(duration_ms)}

        return passed, metrics

    def validate_double_hit(self, cycle: np.ndarray) -> Tuple[bool, Dict]:
        """
        Check for secondary impulses (double hits).

        Args:
            cycle: Single cycle audio data

        Returns:
            (passed, metrics) where metrics contains 'secondary_peak_ratio'
        """
        peak_idx = np.argmax(np.abs(cycle))
        peak = np.abs(cycle[peak_idx])

        # Define search window
        window_start_ms, window_end_ms = self.config['cal_double_hit_window_ms']
        window_start = peak_idx + int(window_start_ms * self.sample_rate / 1000)
        window_end = peak_idx + int(window_end_ms * self.sample_rate / 1000)

        if window_end > len(cycle):
            window_end = len(cycle)

        search_window = cycle[window_start:window_end]
        secondary_peak = np.max(np.abs(search_window)) if len(search_window) > 0 else 0.0

        threshold = self.config['cal_double_hit_threshold'] * peak
        passed = secondary_peak < threshold

        metrics = {
            'secondary_peak_ratio': float(secondary_peak / peak) if peak > 0 else 0.0
        }

        return passed, metrics

    def validate_tail_noise(self, cycle: np.ndarray) -> Tuple[bool, Dict]:
        """
        Check tail noise level after impulse.

        Args:
            cycle: Single cycle audio data

        Returns:
            (passed, metrics) where metrics contains 'tail_rms_ratio'
        """
        peak_idx = np.argmax(np.abs(cycle))
        tail_start = peak_idx + int(self.config['cal_tail_start_ms'] * self.sample_rate / 1000)

        if tail_start >= len(cycle):
            return True, {'tail_rms_ratio': 0.0}

        tail = cycle[tail_start:]
        impulse_region = cycle[max(0, peak_idx-50):min(len(cycle), peak_idx+50)]

        tail_rms = np.sqrt(np.mean(tail**2))
        impulse_rms = np.sqrt(np.mean(impulse_region**2))

        tail_ratio = (tail_rms / impulse_rms) if impulse_rms > 0 else 0.0
        passed = tail_ratio <= self.config['cal_tail_max_rms_ratio']

        metrics = {'tail_rms_ratio': float(tail_ratio)}

        return passed, metrics

    def validate_cycle(self, cycle: np.ndarray, cycle_index: int) -> CycleValidation:
        """
        Validate a single calibration cycle against all criteria.

        Args:
            cycle: Single cycle audio data
            cycle_index: Index of the cycle being validated

        Returns:
            CycleValidation object with results
        """
        failures = []
        all_metrics = {}

        # Run all validation checks
        mag_pass, mag_metrics = self.validate_magnitude(cycle)
        all_metrics.update(mag_metrics)
        if not mag_pass:
            failures.append(f"Magnitude out of range: {mag_metrics['peak_amplitude']:.3f}")

        dur_pass, dur_metrics = self.validate_duration(cycle)
        all_metrics.update(dur_metrics)
        if not dur_pass:
            failures.append(f"Duration out of range: {dur_metrics['duration_ms']:.1f}ms")

        hit_pass, hit_metrics = self.validate_double_hit(cycle)
        all_metrics.update(hit_metrics)
        if not hit_pass:
            failures.append(f"Double hit detected: {hit_metrics['secondary_peak_ratio']:.2f}")

        noise_pass, noise_metrics = self.validate_tail_noise(cycle)
        all_metrics.update(noise_metrics)
        if not noise_pass:
            failures.append(f"Tail noise too high: {noise_metrics['tail_rms_ratio']:.3f}")

        overall_valid = len(failures) == 0

        return CycleValidation(
            cycle_index=cycle_index,
            calibration_valid=overall_valid,
            calibration_metrics=all_metrics,
            calibration_failures=failures
        )
