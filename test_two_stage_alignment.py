#!/usr/bin/env python3
"""
Test script for onset-based cycle alignment

Tests the implementation:
- Steps 1-4: Simple reshape extraction and validation
- Step 5: Align cycles by detecting onset (negative peak) in each cycle
  - Filter invalid cycles
  - Find negative peak position
  - Shift all cycles to align peaks at common position
  - Filter by correlation after alignment
"""

import numpy as np
from RoomResponseRecorder import RoomResponseRecorder


def generate_test_signal_with_offset(sample_rate=48000, num_pulses=10, cycle_duration=0.5,
                                     pulse_duration=0.008, start_offset_ms=50.0):
    """
    Generate impulse train with a known start offset to test alignment.

    Args:
        sample_rate: Sample rate in Hz
        num_pulses: Number of impulses
        cycle_duration: Time between impulse starts (seconds)
        pulse_duration: Duration of each pulse (seconds)
        start_offset_ms: Offset before first impulse (milliseconds)

    Returns:
        Generated audio signal
    """
    cycle_samples = int(cycle_duration * sample_rate)
    pulse_samples = int(pulse_duration * sample_rate)
    start_offset_samples = int(start_offset_ms * sample_rate / 1000)

    # Total samples = offset + cycles
    total_samples = start_offset_samples + (num_pulses * cycle_samples)
    signal = np.zeros(total_samples)

    np.random.seed(42)

    for i in range(num_pulses):
        # Position with offset
        pulse_start = start_offset_samples + (i * cycle_samples)
        pulse_center = pulse_start + pulse_samples // 2

        if pulse_center < len(signal):
            # Negative impulse
            signal[pulse_center] = -0.8

            # Add some ringing
            for j in range(1, pulse_samples // 4):
                pos = pulse_center + j
                if pos < len(signal):
                    signal[pos] = 0.3 * np.exp(-j / 100) * np.sin(j * 0.5)

    # Add noise (reduced for better correlation in test)
    signal += np.random.normal(0, 0.005, len(signal))  # Less noise

    return signal


def test_two_stage_pipeline():
    """Test the two-stage alignment pipeline."""
    print("=" * 70)
    print("Testing Two-Stage Cycle Alignment Pipeline")
    print("=" * 70)

    # Create recorder instance
    recorder = RoomResponseRecorder()

    # Generate test signal with small jitter to test alignment
    print("\n1. Generating test signal with small timing jitter...")
    test_signal = generate_test_signal_with_offset(
        sample_rate=recorder.sample_rate,
        num_pulses=recorder.num_pulses,
        cycle_duration=recorder.cycle_duration,
        pulse_duration=recorder.pulse_duration,
        start_offset_ms=0.0  # No offset so all impulses fit properly
    )
    print(f"   - Signal length: {len(test_signal)} samples")
    print(f"   - Duration: {len(test_signal) / recorder.sample_rate:.3f} seconds")

    # STEPS 1-4: Initial extraction with simple reshape
    print("\n2. STEPS 1-4: Initial extraction (simple reshape)...")
    expected_samples = recorder.cycle_samples * recorder.num_pulses

    # Trim to expected length (this will cut off the offset!)
    if len(test_signal) > expected_samples:
        test_signal_trimmed = test_signal[:expected_samples]
    else:
        test_signal_trimmed = np.pad(test_signal, (0, expected_samples - len(test_signal)))

    # Simple reshape
    initial_cycles = test_signal_trimmed.reshape(recorder.num_pulses, recorder.cycle_samples)
    print(f"   - Extracted {len(initial_cycles)} cycles using simple reshape")
    print(f"   - Each cycle: {recorder.cycle_samples} samples")

    # Validate cycles using actual CalibrationValidatorV2
    from calibration_validator_v2 import CalibrationValidatorV2, QualityThresholds

    # Use lenient thresholds for test
    thresholds = QualityThresholds(
        min_negative_peak=0.3,  # Lenient
        max_negative_peak=1.0,
        min_positive_peak=0.0,
        max_positive_peak=0.5,
        min_aftershock=0.0,
        max_aftershock=0.5,
        aftershock_window_ms=10.0,
        aftershock_skip_ms=2.0
    )

    validator = CalibrationValidatorV2(thresholds, recorder.sample_rate)

    validation_results = []
    print(f"\n   Validating {len(initial_cycles)} initial cycles...")
    for i, cycle in enumerate(initial_cycles):
        validation = validator.validate_cycle(cycle, i)
        validation_dict = {
            'cycle_index': i,
            'calibration_valid': validation.calibration_valid,
            'calibration_metrics': validation.calibration_metrics,
            'calibration_failures': validation.calibration_failures
        }
        validation_results.append(validation_dict)

        # Show validation status
        status = "VALID" if validation.calibration_valid else "INVALID"
        neg_peak = validation.calibration_metrics.get('negative_peak', 0)
        print(f"   Cycle {i}: {status} (neg_peak={neg_peak:.3f})")

    # STEP 5: Align cycles by onset detection
    print("\n3. STEP 5: Align cycles by onset (negative peak detection)...")
    alignment_result = recorder.align_cycles_by_onset(
        initial_cycles,
        validation_results,
        correlation_threshold=0.3  # Lower threshold for noisy test signal
    )

    aligned_cycles = alignment_result['aligned_cycles']
    valid_cycle_indices = alignment_result['valid_cycle_indices']
    onset_positions = alignment_result['onset_positions']
    aligned_onset_position = alignment_result['aligned_onset_position']
    correlations = alignment_result['correlations']
    reference_idx = alignment_result['reference_cycle_idx']

    print(f"   - Valid cycles kept: {len(valid_cycle_indices)}/{len(initial_cycles)}")
    print(f"   - Aligned onset position: {aligned_onset_position} samples")
    print(f"   - Reference cycle: {reference_idx} (in valid set)")
    print(f"   - Correlation threshold: {alignment_result['correlation_threshold']}")

    print("\n   Onset positions in ALL valid cycles (before filtering by correlation):")
    all_valid_indices = [i for i, v in enumerate(validation_results) if v.get('calibration_valid', False)]
    for i in all_valid_indices:
        if i < len(initial_cycles):
            onset = int(np.argmin(initial_cycles[i]))
            print(f"   Cycle {i}: onset={onset:4d} samples")

    print("\n   Per-cycle onset positions and correlations (after alignment and filtering):")
    for i, original_idx in enumerate(valid_cycle_indices):
        onset_pos = onset_positions[i] if i < len(onset_positions) else 0
        corr = correlations[i] if i < len(correlations) else 0
        ref_marker = " <- REFERENCE" if i == reference_idx else ""
        print(f"   Original Cycle {original_idx}: onset={onset_pos:4d} samples, correlation={corr:.3f}{ref_marker}")

    # Verify alignment: check if onsets are actually at aligned position
    print("\n4. Verifying Alignment Quality...")
    print(f"   Checking if negative peaks are at position {aligned_onset_position}...")

    alignment_errors = []
    for i, cycle in enumerate(aligned_cycles):
        actual_onset = int(np.argmin(cycle))
        error = abs(actual_onset - aligned_onset_position)
        alignment_errors.append(error)
        status = "OK" if error <= 2 else "WARN"
        print(f"   Cycle {valid_cycle_indices[i]}: onset at {actual_onset} samples (error={error}) [{status}]")

    max_error = max(alignment_errors) if alignment_errors else 0
    print(f"   Maximum alignment error: {max_error} samples")

    # Compare initial vs aligned cycles
    print("\n5. Comparison: Initial vs Aligned Cycles")
    print(f"   {'Original#':<10} {'Initial Peak':<15} {'Aligned Peak':<15} {'Change'}")
    print(f"   {'-' * 10} {'-' * 15} {'-' * 15} {'-' * 15}")

    for i, original_idx in enumerate(valid_cycle_indices):
        initial_peak = np.max(np.abs(initial_cycles[original_idx]))
        aligned_peak = np.max(np.abs(aligned_cycles[i]))
        change = (aligned_peak - initial_peak) / initial_peak * 100 if initial_peak > 0 else 0
        print(f"   {original_idx:<10} {initial_peak:<15.3f} {aligned_peak:<15.3f} {change:+.1f}%")

    # Summary statistics
    print("\n6. Summary Statistics")
    if len(valid_cycle_indices) > 0:
        initial_peaks = [np.max(np.abs(initial_cycles[i])) for i in valid_cycle_indices]
        aligned_peaks = [np.max(np.abs(c)) for c in aligned_cycles]
        print(f"   Initial cycles - Mean peak: {np.mean(initial_peaks):.3f}")
        print(f"   Aligned cycles - Mean peak: {np.mean(aligned_peaks):.3f}")

    mean_corr = np.mean(correlations) if correlations else 0
    print(f"   Mean correlation: {mean_corr:.3f}")
    print(f"   Alignment accuracy: max error = {max_error} samples")

    print("\n" + "=" * 70)
    print("Test Complete!")
    print("=" * 70)
    print("\nInterpretation:")
    print("- All aligned cycles should have negative peaks at the same position")
    print("- Maximum alignment error should be 0-2 samples (perfect alignment)")
    print("- Correlations should be high (>0.7) for good cycles")
    print("- All displayed cycles passed validation and correlation filters")


if __name__ == "__main__":
    test_two_stage_pipeline()
