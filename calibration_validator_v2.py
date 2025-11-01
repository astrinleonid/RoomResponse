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
    """
    Comprehensive quality thresholds for calibration impulse validation.

    All ratios are relative to the main negative peak amplitude.
    """
    # 1. Negative Peak Range (absolute amplitude)
    min_negative_peak: float  # Minimum negative peak magnitude (absolute value)
    max_negative_peak: float  # Maximum negative peak to avoid clipping

    # 2. Precursor (peaks before negative peak)
    max_precursor_ratio: float  # Maximum pre-impact peak / negative peak

    # 3. Negative Peak Width
    min_negative_peak_width_ms: float  # Minimum width at 50% amplitude
    max_negative_peak_width_ms: float  # Maximum width at 50% amplitude

    # 4. First Positive Peak After Negative
    max_first_positive_ratio: float  # Max first positive peak / negative peak

    # 5. First Positive Peak Timing
    min_first_positive_time_ms: float  # Min time from negative to first positive peak
    max_first_positive_time_ms: float  # Max time from negative to first positive peak

    # 6. Highest Positive Peak After Negative
    max_highest_positive_ratio: float  # Max highest positive peak / negative peak

    # 7. Secondary Negative Peak (replaces aftershock)
    max_secondary_negative_ratio: float  # Max secondary negative / main negative peak
    secondary_negative_window_ms: float  # Time window to check for secondary negative

    @classmethod
    def from_config(cls, config: Dict) -> 'QualityThresholds':
        """
        Create thresholds from configuration dict.

        Supports legacy formats for backward compatibility:
        - V1: min_positive_peak, max_positive_peak, min_aftershock, max_aftershock
        - V2: max_positive_peak_ratio, max_aftershock_ratio
        - V3 (current): Full comprehensive criteria
        """
        # 1. Negative Peak Range - always required
        min_negative_peak = config.get('min_negative_peak', 0.1)
        max_negative_peak = config.get('max_negative_peak', 0.95)

        # 2. Precursor - new in V3
        max_precursor_ratio = config.get('max_precursor_ratio', 0.2)

        # 3. Negative Peak Width - new in V3
        min_negative_peak_width_ms = config.get('min_negative_peak_width_ms', 0.3)
        max_negative_peak_width_ms = config.get('max_negative_peak_width_ms', 3.0)

        # 4. First Positive Peak - new in V3
        max_first_positive_ratio = config.get('max_first_positive_ratio', 0.3)

        # 5. First Positive Peak Timing - new in V3
        min_first_positive_time_ms = config.get('min_first_positive_time_ms', 0.1)
        max_first_positive_time_ms = config.get('max_first_positive_time_ms', 5.0)

        # 6. Highest Positive Peak - new in V3, fallback to legacy max_positive_peak_ratio
        if 'max_highest_positive_ratio' in config:
            max_highest_positive_ratio = config['max_highest_positive_ratio']
        elif 'max_positive_peak_ratio' in config:
            max_highest_positive_ratio = config['max_positive_peak_ratio']
        elif 'max_positive_peak' in config:
            max_highest_positive_ratio = config['max_positive_peak']
        else:
            max_highest_positive_ratio = 0.5

        # 7. Secondary Negative Peak - new in V3, fallback to legacy aftershock
        if 'max_secondary_negative_ratio' in config:
            max_secondary_negative_ratio = config['max_secondary_negative_ratio']
        elif 'max_aftershock_ratio' in config:
            max_secondary_negative_ratio = config['max_aftershock_ratio']
        elif 'max_aftershock' in config:
            max_secondary_negative_ratio = config['max_aftershock']
        else:
            max_secondary_negative_ratio = 0.3

        secondary_negative_window_ms = config.get('secondary_negative_window_ms',
                                                   config.get('aftershock_window_ms', 10.0))

        return cls(
            min_negative_peak=min_negative_peak,
            max_negative_peak=max_negative_peak,
            max_precursor_ratio=max_precursor_ratio,
            min_negative_peak_width_ms=min_negative_peak_width_ms,
            max_negative_peak_width_ms=max_negative_peak_width_ms,
            max_first_positive_ratio=max_first_positive_ratio,
            min_first_positive_time_ms=min_first_positive_time_ms,
            max_first_positive_time_ms=max_first_positive_time_ms,
            max_highest_positive_ratio=max_highest_positive_ratio,
            max_secondary_negative_ratio=max_secondary_negative_ratio,
            secondary_negative_window_ms=secondary_negative_window_ms
        )

    @classmethod
    def from_user_marked_cycles(cls,
                                 cycles: np.ndarray,
                                 marked_good: List[int],
                                 sample_rate: int,
                                 safety_margin: float = 0.05) -> 'QualityThresholds':
        """
        Automatically calculate comprehensive thresholds from user-marked "good" cycles.

        Analyzes all 7 quality criteria and calculates appropriate thresholds
        based on the characteristics of the marked cycles.

        Args:
            cycles: All calibration cycles (num_cycles, cycle_samples)
            marked_good: List of cycle indices that user marked as good
            sample_rate: Sample rate in Hz
            safety_margin: Safety margin to add to thresholds (default 0.05 = 5%)

        Returns:
            QualityThresholds calculated from marked cycles
        """
        if len(marked_good) == 0:
            raise ValueError("At least one cycle must be marked as good")

        good_cycles = cycles[marked_good]

        # Initialize lists for all metrics
        negative_peaks = []
        precursor_ratios = []
        neg_peak_widths_ms = []
        first_pos_ratios = []
        first_pos_times_ms = []
        highest_pos_ratios = []
        secondary_neg_ratios = []

        for cycle in good_cycles:
            # Find negative peak
            neg_peak_idx = np.argmin(cycle)
            neg_peak_val = abs(cycle[neg_peak_idx])
            negative_peaks.append(neg_peak_val)

            # 1. Precursor ratio (max peak before negative peak, excluding rise time)
            # Exclude 1ms before negative peak to avoid detecting the rise time
            rise_time_samples = int(1.0 * sample_rate / 1000)
            precursor_end_idx = max(0, neg_peak_idx - rise_time_samples)

            if precursor_end_idx > 0:
                precursor_region = cycle[:precursor_end_idx]
                max_precursor = np.max(np.abs(precursor_region))
                precursor_ratios.append(max_precursor / neg_peak_val if neg_peak_val > 0 else 0)
            else:
                precursor_ratios.append(0.0)

            # 2. Negative peak width at 50%
            threshold = neg_peak_val * 0.5
            left_idx = neg_peak_idx
            for i in range(neg_peak_idx, -1, -1):
                if abs(cycle[i]) < threshold:
                    left_idx = i
                    break
            right_idx = neg_peak_idx
            for i in range(neg_peak_idx, len(cycle)):
                if abs(cycle[i]) < threshold:
                    right_idx = i
                    break
            width_samples = right_idx - left_idx
            width_ms = (width_samples / sample_rate) * 1000.0
            neg_peak_widths_ms.append(width_ms)

            # 3. First positive peak after negative
            search_start = neg_peak_idx + 1
            first_pos_val = 0.0
            first_pos_time = 0.0

            if search_start < len(cycle):
                for i in range(search_start, len(cycle)):
                    if cycle[i] > 0:
                        search_end = min(i + int(0.005 * sample_rate), len(cycle))
                        peak_region = cycle[i:search_end]
                        if len(peak_region) > 0:
                            local_peak_idx = np.argmax(peak_region)
                            first_pos_idx = i + local_peak_idx
                            first_pos_val = cycle[first_pos_idx]
                            first_pos_time = ((first_pos_idx - neg_peak_idx) / sample_rate) * 1000.0
                        break

            first_pos_ratios.append(first_pos_val / neg_peak_val if neg_peak_val > 0 else 0)
            first_pos_times_ms.append(first_pos_time)

            # 4. Highest positive peak after negative
            after_neg = cycle[neg_peak_idx + 1:] if neg_peak_idx + 1 < len(cycle) else np.array([])
            if len(after_neg) > 0:
                highest_pos_val = np.max(after_neg)
                highest_pos_ratios.append(highest_pos_val / neg_peak_val if neg_peak_val > 0 else 0)
            else:
                highest_pos_ratios.append(0.0)

            # 5. Secondary negative peak (replaces aftershock)
            decay_skip_samples = int(2.0 * sample_rate / 1000)
            window_start = neg_peak_idx + decay_skip_samples
            window_end = min(len(cycle), neg_peak_idx + int(10.0 * sample_rate / 1000))

            if window_end > window_start:
                window = cycle[window_start:window_end]
                min_val = np.min(window)
                max_secondary_negative = abs(min_val)
                secondary_neg_ratios.append(max_secondary_negative / neg_peak_val if neg_peak_val > 0 else 0)
            else:
                secondary_neg_ratios.append(0.0)

        # Calculate thresholds with safety margin
        # 1. Negative peak range
        min_neg = np.min(negative_peaks) * (1 - safety_margin)
        max_neg = np.max(negative_peaks) * (1 + safety_margin)

        # 2. Precursor (upper limit only)
        max_precursor = np.max(precursor_ratios) * (1 + safety_margin)

        # 3. Negative peak width (min/max range)
        min_width = np.min(neg_peak_widths_ms) * (1 - safety_margin)
        max_width = np.max(neg_peak_widths_ms) * (1 + safety_margin)

        # 4. First positive ratio (upper limit only)
        max_first_pos = np.max(first_pos_ratios) * (1 + safety_margin)

        # 5. First positive timing (min/max range)
        valid_times = [t for t in first_pos_times_ms if t > 0]
        if valid_times:
            min_first_pos_time = np.min(valid_times) * (1 - safety_margin)
            max_first_pos_time = np.max(valid_times) * (1 + safety_margin)
        else:
            min_first_pos_time = 0.1
            max_first_pos_time = 5.0

        # 6. Highest positive (upper limit only)
        max_highest_pos = np.max(highest_pos_ratios) * (1 + safety_margin)

        # 7. Secondary negative (upper limit only)
        max_secondary_neg = np.max(secondary_neg_ratios) * (1 + safety_margin)

        # Clamp to reasonable ranges
        min_neg = max(0.05, min_neg)
        max_neg = min(0.98, max_neg)
        max_precursor = min(0.95, max_precursor)
        min_width = max(0.1, min_width)
        max_width = min(10.0, max_width)
        max_first_pos = min(0.95, max_first_pos)
        min_first_pos_time = max(0.05, min_first_pos_time)
        max_first_pos_time = min(10.0, max_first_pos_time)
        max_highest_pos = min(0.95, max_highest_pos)
        max_secondary_neg = min(0.95, max_secondary_neg)

        return cls(
            min_negative_peak=min_neg,
            max_negative_peak=max_neg,
            max_precursor_ratio=max_precursor,
            min_negative_peak_width_ms=min_width,
            max_negative_peak_width_ms=max_width,
            max_first_positive_ratio=max_first_pos,
            min_first_positive_time_ms=min_first_pos_time,
            max_first_positive_time_ms=max_first_pos_time,
            max_highest_positive_ratio=max_highest_pos,
            max_secondary_negative_ratio=max_secondary_neg,
            secondary_negative_window_ms=10.0
        )

    def to_dict(self) -> Dict:
        """Convert to dictionary for saving/display (V3 comprehensive format)"""
        return {
            'min_negative_peak': self.min_negative_peak,
            'max_negative_peak': self.max_negative_peak,
            'max_precursor_ratio': self.max_precursor_ratio,
            'min_negative_peak_width_ms': self.min_negative_peak_width_ms,
            'max_negative_peak_width_ms': self.max_negative_peak_width_ms,
            'max_first_positive_ratio': self.max_first_positive_ratio,
            'min_first_positive_time_ms': self.min_first_positive_time_ms,
            'max_first_positive_time_ms': self.max_first_positive_time_ms,
            'max_highest_positive_ratio': self.max_highest_positive_ratio,
            'max_secondary_negative_ratio': self.max_secondary_negative_ratio,
            'secondary_negative_window_ms': self.secondary_negative_window_ms
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

    def validate_precursor(self, cycle: np.ndarray, neg_peak_idx: int,
                          neg_peak_abs: float) -> Tuple[bool, Dict]:
        """
        Check for precursor peaks BEFORE the main negative peak.

        Precursors indicate vibrations or impacts before the main hammer strike,
        which should be minimal for a clean calibration impulse.

        IMPORTANT: Excludes the rise time of the negative peak itself (1ms before peak)
        to avoid detecting the rising edge as a precursor.

        Args:
            cycle: Single cycle audio data
            neg_peak_idx: Index of negative peak
            neg_peak_abs: Absolute value of negative peak

        Returns:
            (passed, metrics) with max_precursor and precursor_ratio
        """
        # Exclude 1ms before negative peak to avoid detecting the rise time
        rise_time_ms = 1.0
        rise_time_samples = int(rise_time_ms * self.sample_rate / 1000)
        precursor_end_idx = max(0, neg_peak_idx - rise_time_samples)

        if precursor_end_idx == 0:
            # No precursor region (peak is at start or within rise time)
            return True, {'max_precursor': 0.0, 'precursor_ratio': 0.0}

        # Find maximum absolute value BEFORE negative peak (excluding rise time)
        precursor_region = cycle[:precursor_end_idx]
        max_precursor = np.max(np.abs(precursor_region))

        # Calculate ratio relative to main peak
        precursor_ratio = max_precursor / neg_peak_abs if neg_peak_abs > 0 else 0.0

        # Check if precursor is acceptable
        passed = precursor_ratio <= self.thresholds.max_precursor_ratio

        metrics = {
            'max_precursor': float(max_precursor),
            'precursor_ratio': float(precursor_ratio),
            'precursor_exclude_ms': float(rise_time_ms)
        }

        return passed, metrics

    def validate_negative_peak_width(self, cycle: np.ndarray, neg_peak_idx: int,
                                    neg_peak_abs: float) -> Tuple[bool, Dict]:
        """
        Measure negative peak width at 50% of maximum amplitude.

        Peak width indicates the impulse duration - too narrow may indicate noise,
        too wide may indicate a slow impact or sensor issues.

        Args:
            cycle: Single cycle audio data
            neg_peak_idx: Index of negative peak
            neg_peak_abs: Absolute value of negative peak

        Returns:
            (passed, metrics) with peak_width_ms
        """
        # Find 50% threshold (halfway between peak and zero)
        threshold = neg_peak_abs * 0.5

        # Search backwards from peak to find left edge
        left_idx = neg_peak_idx
        for i in range(neg_peak_idx, -1, -1):
            if abs(cycle[i]) < threshold:
                left_idx = i
                break

        # Search forwards from peak to find right edge
        right_idx = neg_peak_idx
        for i in range(neg_peak_idx, len(cycle)):
            if abs(cycle[i]) < threshold:
                right_idx = i
                break

        # Calculate width in milliseconds
        width_samples = right_idx - left_idx
        width_ms = (width_samples / self.sample_rate) * 1000.0

        # Check if within acceptable range
        passed = (self.thresholds.min_negative_peak_width_ms <= width_ms <=
                 self.thresholds.max_negative_peak_width_ms)

        metrics = {
            'negative_peak_width_ms': float(width_ms),
            'negative_peak_width_samples': int(width_samples)
        }

        return passed, metrics

    def validate_first_positive_peak(self, cycle: np.ndarray, neg_peak_idx: int,
                                    neg_peak_abs: float) -> Tuple[bool, Dict]:
        """
        Check the FIRST positive peak AFTER the negative peak.

        This represents the initial rebound or sensor response after the impact.

        Args:
            cycle: Single cycle audio data
            neg_peak_idx: Index of negative peak
            neg_peak_abs: Absolute value of negative peak

        Returns:
            (passed, metrics) with first_positive_peak, ratio, and timing
        """
        # Search for first positive peak after negative peak
        # Look in region starting just after negative peak
        search_start = neg_peak_idx + 1
        if search_start >= len(cycle):
            return True, {
                'first_positive_peak': 0.0,
                'first_positive_ratio': 0.0,
                'first_positive_time_ms': 0.0
            }

        # Find first local maximum (positive peak) after negative peak
        # Look for first value above zero, then find the peak
        first_pos_idx = search_start
        first_pos_val = 0.0

        # Find first crossing above zero
        for i in range(search_start, len(cycle)):
            if cycle[i] > 0:
                # Found positive region, now find peak in next few samples
                search_end = min(i + int(0.005 * self.sample_rate), len(cycle))  # 5ms window
                peak_region = cycle[i:search_end]
                if len(peak_region) > 0:
                    local_peak_idx = np.argmax(peak_region)
                    first_pos_idx = i + local_peak_idx
                    first_pos_val = cycle[first_pos_idx]
                break

        # Calculate ratio and timing
        first_pos_ratio = first_pos_val / neg_peak_abs if neg_peak_abs > 0 else 0.0
        time_from_neg_ms = ((first_pos_idx - neg_peak_idx) / self.sample_rate) * 1000.0

        # Check magnitude criterion
        magnitude_passed = first_pos_ratio <= self.thresholds.max_first_positive_ratio

        # Check timing criterion
        timing_passed = (self.thresholds.min_first_positive_time_ms <= time_from_neg_ms <=
                        self.thresholds.max_first_positive_time_ms)

        passed = magnitude_passed and timing_passed

        metrics = {
            'first_positive_peak': float(first_pos_val),
            'first_positive_ratio': float(first_pos_ratio),
            'first_positive_time_ms': float(time_from_neg_ms),
            'first_positive_idx': int(first_pos_idx)
        }

        return passed, metrics

    def validate_highest_positive_peak(self, cycle: np.ndarray, neg_peak_idx: int,
                                      neg_peak_abs: float) -> Tuple[bool, Dict]:
        """
        Check the HIGHEST positive peak AFTER the negative peak.

        This represents the maximum positive excursion in the decay/response.

        Args:
            cycle: Single cycle audio data
            neg_peak_idx: Index of negative peak
            neg_peak_abs: Absolute value of negative peak

        Returns:
            (passed, metrics) with highest_positive_peak and ratio
        """
        # Search for highest positive value after negative peak
        after_neg = cycle[neg_peak_idx + 1:] if neg_peak_idx + 1 < len(cycle) else np.array([])

        if len(after_neg) == 0:
            return True, {'highest_positive_peak': 0.0, 'highest_positive_ratio': 0.0}

        highest_pos_val = np.max(after_neg)
        highest_pos_ratio = highest_pos_val / neg_peak_abs if neg_peak_abs > 0 else 0.0

        # Check if acceptable
        passed = highest_pos_ratio <= self.thresholds.max_highest_positive_ratio

        metrics = {
            'highest_positive_peak': float(highest_pos_val),
            'highest_positive_ratio': float(highest_pos_ratio)
        }

        return passed, metrics

    def validate_secondary_negative_peak(self, cycle: np.ndarray, neg_peak_idx: int,
                                        neg_peak_abs: float) -> Tuple[bool, Dict]:
        """
        Check for secondary negative peaks AFTER the main negative peak.

        Secondary negative peaks indicate hammer bounces or rebounds,
        which should be minimal for a clean single-impact calibration.

        This replaces the old "aftershock" criterion with a more specific
        focus on negative peaks (bounces) rather than any peaks.

        Args:
            cycle: Single cycle audio data
            neg_peak_idx: Index of negative peak
            neg_peak_abs: Absolute value of negative peak

        Returns:
            (passed, metrics) with max_secondary_negative and ratio
        """
        # Define search window (skip first 2ms to avoid main pulse decay)
        decay_skip_ms = 2.0
        decay_skip_samples = int(decay_skip_ms * self.sample_rate / 1000)

        window_start = neg_peak_idx + decay_skip_samples
        window_end_offset = int(self.thresholds.secondary_negative_window_ms * self.sample_rate / 1000)
        window_end = min(len(cycle), neg_peak_idx + window_end_offset)

        if window_end <= window_start:
            return True, {'max_secondary_negative': 0.0, 'secondary_negative_ratio': 0.0}

        # Extract window and find most negative value (minimum)
        window = cycle[window_start:window_end]
        min_val = np.min(window)  # This will be negative if there's a secondary negative peak
        max_secondary_negative = abs(min_val)  # Convert to absolute for comparison

        # Calculate ratio relative to main peak
        secondary_ratio = max_secondary_negative / neg_peak_abs if neg_peak_abs > 0 else 0.0

        # Check if acceptable
        passed = secondary_ratio <= self.thresholds.max_secondary_negative_ratio

        metrics = {
            'max_secondary_negative': float(max_secondary_negative),
            'secondary_negative_ratio': float(secondary_ratio),
            'secondary_negative_window_start_ms': float(decay_skip_ms),
            'secondary_negative_window_end_ms': float(self.thresholds.secondary_negative_window_ms)
        }

        return passed, metrics

    def validate_cycle(self, cycle: np.ndarray, cycle_index: int,
                      is_user_marked_good: bool = False) -> CycleValidation:
        """
        Validate a single calibration cycle against all 7 comprehensive criteria.

        Args:
            cycle: Single cycle audio data (NOT normalized)
            cycle_index: Index of the cycle being validated
            is_user_marked_good: Whether user marked this as a good cycle

        Returns:
            CycleValidation object with results
        """
        failures = []
        all_metrics = {}

        # 1. Check for strong negative pulse (required for all other checks)
        neg_pass, neg_metrics = self.validate_negative_pulse(cycle)
        all_metrics.update(neg_metrics)

        if not neg_pass:
            neg_peak = neg_metrics['negative_peak']
            if neg_peak < self.thresholds.min_negative_peak:
                failures.append(f"Weak negative pulse: {neg_peak:.3f} < {self.thresholds.min_negative_peak:.3f}")
            else:
                failures.append(f"Excessive negative pulse (clipping?): {neg_peak:.3f} > {self.thresholds.max_negative_peak:.3f}")

        # Extract negative peak info for subsequent checks
        neg_peak_idx = neg_metrics['negative_peak_idx']
        neg_peak_abs = neg_metrics['negative_peak']

        # 2. Check for precursor (peaks before main impact)
        precursor_pass, precursor_metrics = self.validate_precursor(
            cycle, neg_peak_idx, neg_peak_abs
        )
        all_metrics.update(precursor_metrics)

        if not precursor_pass:
            ratio = precursor_metrics['precursor_ratio']
            failures.append(f"Precursor detected: {ratio:.3f} > {self.thresholds.max_precursor_ratio:.3f}")

        # 3. Check negative peak width
        width_pass, width_metrics = self.validate_negative_peak_width(
            cycle, neg_peak_idx, neg_peak_abs
        )
        all_metrics.update(width_metrics)

        if not width_pass:
            width = width_metrics['negative_peak_width_ms']
            if width < self.thresholds.min_negative_peak_width_ms:
                failures.append(f"Negative peak too narrow: {width:.2f}ms < {self.thresholds.min_negative_peak_width_ms:.2f}ms")
            else:
                failures.append(f"Negative peak too wide: {width:.2f}ms > {self.thresholds.max_negative_peak_width_ms:.2f}ms")

        # 4. Check first positive peak (magnitude and timing)
        first_pos_pass, first_pos_metrics = self.validate_first_positive_peak(
            cycle, neg_peak_idx, neg_peak_abs
        )
        all_metrics.update(first_pos_metrics)

        if not first_pos_pass:
            ratio = first_pos_metrics['first_positive_ratio']
            time_ms = first_pos_metrics['first_positive_time_ms']
            if ratio > self.thresholds.max_first_positive_ratio:
                failures.append(f"First positive peak too large: {ratio:.3f} > {self.thresholds.max_first_positive_ratio:.3f}")
            if time_ms < self.thresholds.min_first_positive_time_ms:
                failures.append(f"First positive peak too early: {time_ms:.2f}ms < {self.thresholds.min_first_positive_time_ms:.2f}ms")
            elif time_ms > self.thresholds.max_first_positive_time_ms:
                failures.append(f"First positive peak too late: {time_ms:.2f}ms > {self.thresholds.max_first_positive_time_ms:.2f}ms")

        # 5. Check highest positive peak
        highest_pos_pass, highest_pos_metrics = self.validate_highest_positive_peak(
            cycle, neg_peak_idx, neg_peak_abs
        )
        all_metrics.update(highest_pos_metrics)

        if not highest_pos_pass:
            ratio = highest_pos_metrics['highest_positive_ratio']
            failures.append(f"Highest positive peak too large: {ratio:.3f} > {self.thresholds.max_highest_positive_ratio:.3f}")

        # 6. Check for secondary negative peaks (hammer bounces)
        secondary_neg_pass, secondary_neg_metrics = self.validate_secondary_negative_peak(
            cycle, neg_peak_idx, neg_peak_abs
        )
        all_metrics.update(secondary_neg_metrics)

        if not secondary_neg_pass:
            ratio = secondary_neg_metrics['secondary_negative_ratio']
            failures.append(f"Secondary negative peak detected: {ratio:.3f} > {self.thresholds.max_secondary_negative_ratio:.3f}")

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
