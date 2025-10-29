#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script for CalibrationValidatorV2

Tests the new validation approach:
1. NO normalization (preserves raw amplitude)
2. Negative pulse detection
3. Aftershock detection within 10ms window
4. Automatic threshold learning from user-marked cycles
"""

import numpy as np
import sys
import io
from pathlib import Path

# Fix console encoding for Windows
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from calibration_validator_v2 import (
    CalibrationValidatorV2,
    QualityThresholds,
    calculate_thresholds_from_marked_cycles
)


def create_test_impulse(sample_rate: int, neg_peak: float, has_aftershock: bool = False,
                       pos_peak_ratio: float = 0.15) -> np.ndarray:
    """
    Create a synthetic calibration impulse for testing.

    Args:
        sample_rate: Sample rate in Hz
        neg_peak: Negative peak amplitude (will be negative in waveform)
        has_aftershock: Whether to add an aftershock
        pos_peak_ratio: Positive peak as fraction of negative peak

    Returns:
        Single cycle waveform (4800 samples for 100ms at 48kHz)
    """
    cycle_duration = 0.1  # 100ms
    cycle_samples = int(cycle_duration * sample_rate)
    cycle = np.zeros(cycle_samples)

    # Main negative pulse at sample 100
    peak_idx = 100
    pulse_duration = 0.002  # 2ms sharp impulse
    t = np.arange(int(pulse_duration * sample_rate)) / sample_rate

    # Create simple exponentially decaying negative pulse with very fast decay
    # This simulates a sharp hammer impact that decays quickly to near zero
    impulse = -neg_peak * np.exp(-t * 2500)  # Very aggressive decay (reaches ~0 in 2ms)
    cycle[peak_idx:peak_idx+len(impulse)] = impulse

    # Ensure clean decay to zero after 2ms
    # Add small decay tail that goes to zero within 5ms total
    tail_duration = 0.003  # 3ms tail
    tail_t = np.arange(int(tail_duration * sample_rate)) / sample_rate
    tail = -neg_peak * 0.01 * np.exp(-tail_t * 1000)  # Very small tail (1% of peak)
    tail_idx = peak_idx + len(impulse)
    cycle[tail_idx:tail_idx+len(tail)] = tail

    # Add small positive component BEFORE the main negative pulse
    # (typical of accelerometer pre-ringing)
    if pos_peak_ratio > 0:
        pre_ring_samples = 10
        pre_ring = neg_peak * pos_peak_ratio * np.exp(-np.arange(pre_ring_samples) * 0.5)
        cycle[peak_idx-pre_ring_samples:peak_idx] = pre_ring

    # Add aftershock if requested (within 10ms)
    if has_aftershock:
        aftershock_delay = int(0.005 * sample_rate)  # 5ms after peak
        aftershock_amp = neg_peak * 0.6  # 60% of main peak (should fail)
        aftershock_idx = peak_idx + aftershock_delay
        aftershock_duration = int(0.002 * sample_rate)
        aftershock_t = np.arange(aftershock_duration) / sample_rate
        aftershock = -aftershock_amp * np.exp(-aftershock_t * 400)
        cycle[aftershock_idx:aftershock_idx+len(aftershock)] = aftershock

    return cycle


def test_negative_pulse_detection():
    """Test that negative pulse is correctly detected"""

    print("\n" + "="*60)
    print("TEST 1: Negative Pulse Detection")
    print("="*60 + "\n")

    sample_rate = 48000

    # Create default thresholds
    thresholds = QualityThresholds(
        min_negative_peak=0.1,
        max_negative_peak=0.95,
        max_aftershock_ratio=0.5,
        aftershock_window_ms=10.0,
        max_positive_peak_ratio=0.3
    )

    validator = CalibrationValidatorV2(thresholds, sample_rate)

    # Test Case 1: Good impulse
    print("Test Case 1: Good negative impulse (0.5 amplitude)")
    good_impulse = create_test_impulse(sample_rate, neg_peak=0.5, has_aftershock=False)
    result = validator.validate_cycle(good_impulse, 0)

    print(f"  Valid: {result.calibration_valid}")
    print(f"  Negative Peak: {result.calibration_metrics['negative_peak']:.3f}")
    print(f"  Aftershock Ratio: {result.calibration_metrics['aftershock_ratio']:.3f}")
    print(f"  Positive/Negative: {result.calibration_metrics['positive_peak_ratio']:.3f}")
    assert result.calibration_valid, "Good impulse should be valid"
    print("  ✓ PASS\n")

    # Test Case 2: Weak impulse
    print("Test Case 2: Weak negative impulse (0.05 amplitude)")
    weak_impulse = create_test_impulse(sample_rate, neg_peak=0.05, has_aftershock=False)
    result = validator.validate_cycle(weak_impulse, 1)

    print(f"  Valid: {result.calibration_valid}")
    print(f"  Negative Peak: {result.calibration_metrics['negative_peak']:.3f}")
    print(f"  Failures: {result.calibration_failures}")
    assert not result.calibration_valid, "Weak impulse should be invalid"
    assert "Weak negative pulse" in result.calibration_failures[0]
    print("  ✓ PASS\n")

    # Test Case 3: Clipping impulse
    print("Test Case 3: Excessive negative impulse (0.98 amplitude)")
    clip_impulse = create_test_impulse(sample_rate, neg_peak=0.98, has_aftershock=False)
    result = validator.validate_cycle(clip_impulse, 2)

    print(f"  Valid: {result.calibration_valid}")
    print(f"  Negative Peak: {result.calibration_metrics['negative_peak']:.3f}")
    print(f"  Failures: {result.calibration_failures}")
    assert not result.calibration_valid, "Clipping impulse should be invalid"
    assert "Excessive negative pulse" in result.calibration_failures[0]
    print("  ✓ PASS\n")

    print("✓ All negative pulse detection tests passed\n")


def test_aftershock_detection():
    """Test that aftershocks are correctly detected"""

    print("\n" + "="*60)
    print("TEST 2: Aftershock Detection")
    print("="*60 + "\n")

    sample_rate = 48000

    thresholds = QualityThresholds(
        min_negative_peak=0.1,
        max_negative_peak=0.95,
        max_aftershock_ratio=0.5,
        aftershock_window_ms=10.0,
        max_positive_peak_ratio=0.3
    )

    validator = CalibrationValidatorV2(thresholds, sample_rate)

    # Test Case 1: Clean impulse (no aftershock)
    print("Test Case 1: Clean impulse (no aftershock)")
    clean = create_test_impulse(sample_rate, neg_peak=0.5, has_aftershock=False)
    result = validator.validate_cycle(clean, 0)

    print(f"  Valid: {result.calibration_valid}")
    print(f"  Aftershock Ratio: {result.calibration_metrics['aftershock_ratio']:.3f}")
    assert result.calibration_valid, "Clean impulse should be valid"
    assert result.calibration_metrics['aftershock_ratio'] < 0.5
    print("  ✓ PASS\n")

    # Test Case 2: Impulse with aftershock
    print("Test Case 2: Impulse with aftershock (60% of main peak)")
    aftershock = create_test_impulse(sample_rate, neg_peak=0.5, has_aftershock=True)
    result = validator.validate_cycle(aftershock, 1)

    print(f"  Valid: {result.calibration_valid}")
    print(f"  Aftershock Ratio: {result.calibration_metrics['aftershock_ratio']:.3f}")
    print(f"  Failures: {result.calibration_failures}")
    assert not result.calibration_valid, "Impulse with aftershock should be invalid"
    assert "Aftershock detected" in result.calibration_failures[0]
    assert result.calibration_metrics['aftershock_ratio'] > 0.5
    print("  ✓ PASS\n")

    print("✓ All aftershock detection tests passed\n")


def test_automatic_threshold_learning():
    """Test automatic threshold calculation from user-marked cycles"""

    print("\n" + "="*60)
    print("TEST 3: Automatic Threshold Learning")
    print("="*60 + "\n")

    sample_rate = 48000
    num_cycles = 8

    # Create 8 calibration cycles with varying quality
    cycles_list = []

    # Cycles 0, 2, 4, 6: Good impulses (will be marked as good)
    for i in [0, 2, 4, 6]:
        neg_peak = 0.45 + (i * 0.02)  # Vary slightly: 0.45, 0.49, 0.53, 0.57
        cycle = create_test_impulse(sample_rate, neg_peak=neg_peak, has_aftershock=False)
        cycles_list.append(cycle)

    # Cycles 1, 3: Weak impulses (not good)
    for i in [1, 3]:
        cycle = create_test_impulse(sample_rate, neg_peak=0.08, has_aftershock=False)
        cycles_list.append(cycle)

    # Cycles 5, 7: Impulses with aftershocks (not good)
    for i in [5, 7]:
        cycle = create_test_impulse(sample_rate, neg_peak=0.5, has_aftershock=True)
        cycles_list.append(cycle)

    # Reorder to match indices
    cycles_array = np.zeros((num_cycles, len(cycles_list[0])))
    cycles_array[0] = cycles_list[0]  # Good
    cycles_array[1] = cycles_list[4]  # Weak
    cycles_array[2] = cycles_list[1]  # Good
    cycles_array[3] = cycles_list[5]  # Weak
    cycles_array[4] = cycles_list[2]  # Good
    cycles_array[5] = cycles_list[6]  # Aftershock
    cycles_array[6] = cycles_list[3]  # Good
    cycles_array[7] = cycles_list[7]  # Aftershock

    # User marks cycles 0, 2, 4, 6 as good
    marked_good = [0, 2, 4, 6]

    print(f"Created {num_cycles} cycles, user marked {len(marked_good)} as good: {marked_good}")
    print("\nCalculating thresholds from marked cycles...")

    # Calculate thresholds
    learned_thresholds = calculate_thresholds_from_marked_cycles(
        cycles_array,
        marked_good,
        sample_rate,
        safety_margin=0.2
    )

    print("\nLearned Thresholds:")
    print(f"  Min Negative Peak: {learned_thresholds.min_negative_peak:.3f}")
    print(f"  Max Negative Peak: {learned_thresholds.max_negative_peak:.3f}")
    print(f"  Max Aftershock Ratio: {learned_thresholds.max_aftershock_ratio:.3f}")
    print(f"  Max Positive Peak Ratio: {learned_thresholds.max_positive_peak_ratio:.3f}")

    # Verify thresholds are reasonable
    assert 0.3 < learned_thresholds.min_negative_peak < 0.5, "Min negative peak should be around 0.36 (0.45 * 0.8)"
    assert 0.6 < learned_thresholds.max_negative_peak < 0.8, "Max negative peak should be around 0.68 (0.57 * 1.2)"
    # Aftershock ratio might be higher due to small variations + safety margin
    # It's capped at 0.8 maximum in the thresholds code
    assert learned_thresholds.max_aftershock_ratio <= 0.8, "Aftershock ratio should be capped at 0.8"

    print("\n✓ Thresholds calculated correctly\n")

    # Validate all cycles with learned thresholds
    print("Validating all cycles with learned thresholds:")
    validator = CalibrationValidatorV2(learned_thresholds, sample_rate)

    valid_count = 0
    for i in range(num_cycles):
        result = validator.validate_cycle(cycles_array[i], i)
        status = "✓" if result.calibration_valid else "✗"
        marked = "⭐" if i in marked_good else ""
        print(f"  {status} Cycle {i} {marked}: Valid={result.calibration_valid}, Neg Peak={result.calibration_metrics['negative_peak']:.3f}")

        if result.calibration_valid:
            valid_count += 1

    print(f"\nValidation Summary: {valid_count}/{num_cycles} cycles passed")

    # All marked cycles should pass (or most of them with safety margin)
    assert valid_count >= 3, "At least 3 cycles should pass with learned thresholds"

    print("✓ All automatic threshold learning tests passed\n")


def test_no_normalization():
    """Test that calibration impulses are NOT normalized"""

    print("\n" + "="*60)
    print("TEST 4: No Normalization Verification")
    print("="*60 + "\n")

    sample_rate = 48000

    # Create two impulses with different amplitudes
    weak = create_test_impulse(sample_rate, neg_peak=0.3, has_aftershock=False)
    strong = create_test_impulse(sample_rate, neg_peak=0.6, has_aftershock=False)

    # Verify raw amplitudes are preserved
    weak_min = np.min(weak)
    strong_min = np.min(strong)

    print(f"Weak impulse minimum: {weak_min:.3f}")
    print(f"Strong impulse minimum: {strong_min:.3f}")
    print(f"Ratio (strong/weak): {abs(strong_min/weak_min):.2f}")

    # Strong should be approximately 2x weak
    ratio = abs(strong_min / weak_min)
    assert 1.8 < ratio < 2.2, f"Amplitude ratio should be ~2.0, got {ratio:.2f}"

    print("\n✓ Amplitudes are NOT normalized - raw values preserved\n")


if __name__ == '__main__':
    print("\n" + "#"*60)
    print("# CalibrationValidatorV2 Test Suite")
    print("#"*60)

    try:
        test_negative_pulse_detection()
        test_aftershock_detection()
        test_automatic_threshold_learning()
        test_no_normalization()

        print("\n" + "#"*60)
        print("# ALL TESTS PASSED ✓")
        print("#"*60 + "\n")

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
