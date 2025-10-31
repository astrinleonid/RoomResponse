"""
Calibration validator V2 for piano impulse response measurements.

This module validates calibration channel cycles based on:
1. Strong negative pulse detection (no normalization)
2. Aftershock detection within 10ms window (>0.5 of max magnitude)
3. User-driven automatic threshold learning

Key changes from V1:
- NO normalization of calibration impulses
- Focus on negative pulse pattern (hammer impact signature)
- Aftershock detection (not double-hit, but immediate rebounds)
- Automatic threshold calculation from user-marked "good" cycles
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import numpy as np


@dataclass
class CycleValidation:
    """Validation result for a single cycle"""
    cycle_index: int
    calibration_valid: bool
    calibration_metrics: Dict[str, float]
    calibration_failures: List[str]
    is_user_marked_good: bool = False  # NEW: User marking for threshold learning


@dataclass
class QualityThresholds:
    """Quality thresholds (can be learned from user-marked cycles)"""
    min_negative_peak: float  # Minimum negative peak magnitude (absolute value)
    max_negative_peak: float  # Maximum negative peak to avoid clipping
    max_aftershock_ratio: float  # Maximum aftershock as fraction of main peak (default 0.5)
    aftershock_window_ms: float  # Time window to check for aftershocks (default 10ms)
    max_positive_peak_ratio: float  # Max positive peak relative to negative (should be small)

    @classmethod
    def from_config(cls, config: Dict) -> 'QualityThresholds':
        """
        Create thresholds from configuration dict.

        Supports both old format (min/max ranges) and new format (ratios):
        - Old: min_positive_peak, max_positive_peak, min_aftershock, max_aftershock
        - New: max_positive_peak_ratio, max_aftershock_ratio
        """
        # Handle max_positive_peak_ratio: try new format first, fallback to old format
        if 'max_positive_peak_ratio' in config:
            max_positive_peak_ratio = config['max_positive_peak_ratio']
        elif 'max_positive_peak' in config:
            # Old format: convert max_positive_peak to ratio (assuming it's already relative)
            max_positive_peak_ratio = config['max_positive_peak']
        else:
            max_positive_peak_ratio = 0.3

        # Handle max_aftershock_ratio: try new format first, fallback to old format
        if 'max_aftershock_ratio' in config:
            max_aftershock_ratio = config['max_aftershock_ratio']
        elif 'max_aftershock' in config:
            # Old format: convert max_aftershock to ratio (assuming it's already relative)
            max_aftershock_ratio = config['max_aftershock']
        else:
            max_aftershock_ratio = 0.5

        return cls(
            min_negative_peak=config.get('min_negative_peak', 0.1),
            max_negative_peak=config.get('max_negative_peak', 0.95),
            max_aftershock_ratio=max_aftershock_ratio,
            aftershock_window_ms=config.get('aftershock_window_ms', 10.0),
            max_positive_peak_ratio=max_positive_peak_ratio
        )

    @classmethod
    def from_user_marked_cycles(cls,
                                 cycles: np.ndarray,
                                 marked_good: List[int],
                                 sample_rate: int,
                                 safety_margin: float = 0.2) -> 'QualityThresholds':
        """
        Automatically calculate thresholds from user-marked "good" cycles.

        Args:
            cycles: All calibration cycles (num_cycles, cycle_samples)
            marked_good: List of cycle indices that user marked as good
            sample_rate: Sample rate in Hz
            safety_margin: Add this margin to calculated thresholds (0.2 = 20%)

        Returns:
            QualityThresholds calculated from marked cycles
        """
        if len(marked_good) == 0:
            raise ValueError("At least one cycle must be marked as good")

        good_cycles = cycles[marked_good]

        # Calculate negative peaks for all good cycles
        negative_peaks = []
        positive_peaks = []
        aftershock_ratios = []

        for cycle in good_cycles:
            # Find negative peak (minimum value)
            neg_peak_idx = np.argmin(cycle)
            neg_peak_val = abs(cycle[neg_peak_idx])  # Convert to positive for comparison
            negative_peaks.append(neg_peak_val)

            # Find positive peak
            pos_peak_val = np.max(cycle)
            positive_peaks.append(pos_peak_val / neg_peak_val if neg_peak_val > 0 else 0)

            # Calculate aftershock in 10ms window
            aftershock_window_samples = int(10.0 * sample_rate / 1000)
            window_start = neg_peak_idx + 1
            window_end = min(len(cycle), neg_peak_idx + aftershock_window_samples)

            if window_end > window_start:
                window = cycle[window_start:window_end]
                max_aftershock = np.max(np.abs(window))
                aftershock_ratios.append(max_aftershock / neg_peak_val if neg_peak_val > 0 else 0)
            else:
                aftershock_ratios.append(0.0)

        # Calculate thresholds with safety margin
        min_neg = np.min(negative_peaks) * (1 - safety_margin)
        max_neg = np.max(negative_peaks) * (1 + safety_margin)
        max_aftershock = np.max(aftershock_ratios) * (1 + safety_margin)
        max_pos_ratio = np.max(positive_peaks) * (1 + safety_margin)

        # Clamp to reasonable ranges
        min_neg = max(0.05, min_neg)  # At least 5% amplitude
        max_neg = min(0.98, max_neg)  # Leave headroom for clipping
        max_aftershock = min(0.95, max_aftershock)  # Cap at 95%
        max_pos_ratio = min(0.95, max_pos_ratio)  # Cap at 95% (allow actual measured values)

        return cls(
            min_negative_peak=min_neg,
            max_negative_peak=max_neg,
            max_aftershock_ratio=max_aftershock,
            aftershock_window_ms=10.0,
            max_positive_peak_ratio=max_pos_ratio
        )

    def to_dict(self) -> Dict:
        """Convert to dictionary for saving/display"""
        return {
            'min_negative_peak': self.min_negative_peak,
            'max_negative_peak': self.max_negative_peak,
            'max_aftershock_ratio': self.max_aftershock_ratio,
            'aftershock_window_ms': self.aftershock_window_ms,
            'max_positive_peak_ratio': self.max_positive_peak_ratio
        }


class CalibrationValidatorV2:
    """
    Validates calibration channel cycles based on negative pulse pattern.

    Key features:
    - NO normalization (preserves raw amplitude information)
    - Detects strong negative pulse (hammer impact signature)
    - Checks for aftershocks within 10ms (immediate rebounds)
    - Supports automatic threshold learning from user markings
    """

    def __init__(self, thresholds: QualityThresholds, sample_rate: int):
        """
        Initialize calibration validator.

        Args:
            thresholds: Quality thresholds (from config or learned from data)
            sample_rate: Audio sample rate in Hz
        """
        self.thresholds = thresholds
        self.sample_rate = sample_rate

    def validate_negative_pulse(self, cycle: np.ndarray) -> Tuple[bool, Dict]:
        """
        Check for strong negative pulse within acceptable range.

        A good calibration impulse should have:
        - Strong negative peak (hammer impact)
        - Peak amplitude within acceptable range (not too weak, not clipped)

        Args:
            cycle: Single cycle audio data (NOT normalized)

        Returns:
            (passed, metrics) where metrics contains:
            - 'negative_peak': absolute value of minimum
            - 'negative_peak_idx': sample index of negative peak
        """
        # Find negative peak (minimum value in waveform)
        neg_peak_idx = np.argmin(cycle)
        neg_peak_val = cycle[neg_peak_idx]  # This will be negative
        neg_peak_abs = abs(neg_peak_val)  # Absolute value for comparison

        # Check if within acceptable range
        passed = (self.thresholds.min_negative_peak <= neg_peak_abs <=
                 self.thresholds.max_negative_peak)

        metrics = {
            'negative_peak': float(neg_peak_abs),
            'negative_peak_idx': int(neg_peak_idx),
            'negative_peak_raw': float(neg_peak_val)  # Keep raw value too
        }

        return passed, metrics

    def validate_aftershock(self, cycle: np.ndarray, neg_peak_idx: int,
                           neg_peak_abs: float) -> Tuple[bool, Dict]:
        """
        Check for aftershocks within 10ms of main negative pulse.

        Aftershocks are immediate rebounds after the hammer impact that
        indicate poor contact or secondary bounces. This checks for
        significant peaks AFTER the initial decay.

        Args:
            cycle: Single cycle audio data
            neg_peak_idx: Index of negative peak
            neg_peak_abs: Absolute value of negative peak

        Returns:
            (passed, metrics) where metrics contains:
            - 'max_aftershock': maximum absolute value in aftershock window
            - 'aftershock_ratio': aftershock / negative_peak
        """
        # REVISED: Start aftershock window 2ms after peak to skip main pulse decay
        # This prevents the natural decay tail from being flagged as aftershock
        decay_skip_ms = 2.0  # Skip first 2ms to let main pulse decay
        decay_skip_samples = int(decay_skip_ms * self.sample_rate / 1000)

        # Define aftershock search window (from 2ms to 10ms after peak)
        window_start = neg_peak_idx + decay_skip_samples
        window_end_offset = int(self.thresholds.aftershock_window_ms * self.sample_rate / 1000)
        window_end = min(len(cycle), neg_peak_idx + window_end_offset)

        if window_end <= window_start:
            # No window to check
            return True, {'max_aftershock': 0.0, 'aftershock_ratio': 0.0}

        # Extract window and find maximum absolute value
        window = cycle[window_start:window_end]
        max_aftershock = np.max(np.abs(window))

        # Calculate ratio relative to main peak
        aftershock_ratio = max_aftershock / neg_peak_abs if neg_peak_abs > 0 else 0.0

        # Check if aftershock is acceptable (should be < 0.5 of main peak)
        passed = aftershock_ratio <= self.thresholds.max_aftershock_ratio

        metrics = {
            'max_aftershock': float(max_aftershock),
            'aftershock_ratio': float(aftershock_ratio),
            'aftershock_window_start_ms': float(decay_skip_ms),
            'aftershock_window_end_ms': float(self.thresholds.aftershock_window_ms)
        }

        return passed, metrics

    def validate_positive_peak(self, cycle: np.ndarray, neg_peak_abs: float) -> Tuple[bool, Dict]:
        """
        Check that positive peak is small relative to negative peak.

        A good hammer impact should be predominantly negative. If there's a large
        positive component, it may indicate issues with the sensor or setup.

        Args:
            cycle: Single cycle audio data
            neg_peak_abs: Absolute value of negative peak

        Returns:
            (passed, metrics) where metrics contains:
            - 'positive_peak': maximum positive value
            - 'positive_peak_ratio': positive / negative
        """
        pos_peak = np.max(cycle)
        pos_ratio = pos_peak / neg_peak_abs if neg_peak_abs > 0 else 0.0

        # Positive peak should be small compared to negative
        passed = pos_ratio <= self.thresholds.max_positive_peak_ratio

        metrics = {
            'positive_peak': float(pos_peak),
            'positive_peak_ratio': float(pos_ratio)
        }

        return passed, metrics

    def validate_cycle(self, cycle: np.ndarray, cycle_index: int,
                      is_user_marked_good: bool = False) -> CycleValidation:
        """
        Validate a single calibration cycle against all criteria.

        Args:
            cycle: Single cycle audio data (NOT normalized)
            cycle_index: Index of the cycle being validated
            is_user_marked_good: Whether user marked this as a good cycle

        Returns:
            CycleValidation object with results
        """
        failures = []
        all_metrics = {}

        # 1. Check for strong negative pulse
        neg_pass, neg_metrics = self.validate_negative_pulse(cycle)
        all_metrics.update(neg_metrics)

        if not neg_pass:
            neg_peak = neg_metrics['negative_peak']
            if neg_peak < self.thresholds.min_negative_peak:
                failures.append(f"Weak negative pulse: {neg_peak:.3f} < {self.thresholds.min_negative_peak:.3f}")
            else:
                failures.append(f"Excessive negative pulse (clipping?): {neg_peak:.3f} > {self.thresholds.max_negative_peak:.3f}")

        # 2. Check for aftershocks (use detected negative peak)
        neg_peak_idx = neg_metrics['negative_peak_idx']
        neg_peak_abs = neg_metrics['negative_peak']

        aftershock_pass, aftershock_metrics = self.validate_aftershock(
            cycle, neg_peak_idx, neg_peak_abs
        )
        all_metrics.update(aftershock_metrics)

        if not aftershock_pass:
            ratio = aftershock_metrics['aftershock_ratio']
            failures.append(f"Aftershock detected: {ratio:.2f} > {self.thresholds.max_aftershock_ratio:.2f}")

        # 3. Check positive peak ratio
        pos_pass, pos_metrics = self.validate_positive_peak(cycle, neg_peak_abs)
        all_metrics.update(pos_metrics)

        if not pos_pass:
            ratio = pos_metrics['positive_peak_ratio']
            failures.append(f"Excessive positive peak: {ratio:.2f} > {self.thresholds.max_positive_peak_ratio:.2f}")

        # Overall validation
        overall_valid = len(failures) == 0

        return CycleValidation(
            cycle_index=cycle_index,
            calibration_valid=overall_valid,
            calibration_metrics=all_metrics,
            calibration_failures=failures,
            is_user_marked_good=is_user_marked_good
        )


def calculate_thresholds_from_marked_cycles(
    cycles: np.ndarray,
    marked_good_indices: List[int],
    sample_rate: int,
    safety_margin: float = 0.2
) -> QualityThresholds:
    """
    Convenience function to calculate quality thresholds from user-marked cycles.

    Args:
        cycles: All calibration cycles (num_cycles, cycle_samples)
        marked_good_indices: List of cycle indices marked as "good" by user
        sample_rate: Sample rate in Hz
        safety_margin: Safety margin to add to thresholds (default 20%)

    Returns:
        QualityThresholds object

    Example:
        >>> cycles = recorded_calibration_cycles  # Shape: (8, 4800)
        >>> user_marked = [0, 2, 4, 6]  # User says these are good
        >>> thresholds = calculate_thresholds_from_marked_cycles(
        ...     cycles, user_marked, 48000
        ... )
        >>> validator = CalibrationValidatorV2(thresholds, 48000)
    """
    return QualityThresholds.from_user_marked_cycles(
        cycles, marked_good_indices, sample_rate, safety_margin
    )
