"""
Test script for impulse response truncation feature.

This script tests the truncate_with_fadeout method in SignalProcessor
with various parameters to verify correct behavior.
"""

import numpy as np
from signal_processor import SignalProcessor, SignalProcessingConfig


def test_truncation_basic():
    """Test basic truncation with fade-out"""
    print("=" * 60)
    print("TEST 1: Basic Truncation")
    print("=" * 60)

    # Create config
    config = SignalProcessingConfig(
        num_pulses=8,
        cycle_samples=4800,
        sample_rate=48000,
        multichannel_config={}
    )

    # Create processor
    processor = SignalProcessor(config)

    # Create test signal (1 second at 48kHz)
    signal = np.sin(2 * np.pi * 440 * np.arange(48000) / 48000)

    # Truncate to 500ms with 50ms fade
    truncated = processor.truncate_with_fadeout(
        signal,
        working_length_ms=500.0,
        fade_length_ms=50.0
    )

    expected_samples = int(500 * 48000 / 1000)  # 24000 samples

    print(f"Original length: {len(signal)} samples ({len(signal) / 48000 * 1000:.1f} ms)")
    print(f"Truncated length: {len(truncated)} samples ({len(truncated) / 48000 * 1000:.1f} ms)")
    print(f"Expected length: {expected_samples} samples")

    assert len(truncated) == expected_samples, f"Expected {expected_samples}, got {len(truncated)}"
    print("[PASS] Length correct")

    # Check fade-out is applied (last samples should be near zero)
    fade_samples = int(50 * 48000 / 1000)  # 2400 samples
    fade_region = truncated[-fade_samples:]
    assert np.abs(fade_region[-1]) < 0.01, "Last sample should be near zero"
    print(f"[PASS] Fade-out applied (last sample: {fade_region[-1]:.6f})")

    print("[PASS] TEST 1 PASSED\n")


def test_truncation_no_op():
    """Test truncation when signal is already shorter than target"""
    print("=" * 60)
    print("TEST 2: No-op Truncation (signal already shorter)")
    print("=" * 60)

    config = SignalProcessingConfig(
        num_pulses=8,
        cycle_samples=4800,
        sample_rate=48000,
        multichannel_config={}
    )

    processor = SignalProcessor(config)

    # Create short signal (100ms)
    signal = np.sin(2 * np.pi * 440 * np.arange(4800) / 48000)

    # Try to truncate to 500ms (longer than signal)
    truncated = processor.truncate_with_fadeout(
        signal,
        working_length_ms=500.0,
        fade_length_ms=50.0
    )

    print(f"Original length: {len(signal)} samples")
    print(f"Truncated length: {len(truncated)} samples")

    assert len(truncated) == len(signal), "Signal should be unchanged"
    assert np.array_equal(truncated, signal), "Signal content should be unchanged"
    print("[PASS] Signal unchanged (no truncation needed)")

    print("[PASS] TEST 2 PASSED\n")


def test_truncation_edge_cases():
    """Test edge cases and validation"""
    print("=" * 60)
    print("TEST 3: Edge Cases and Validation")
    print("=" * 60)

    config = SignalProcessingConfig(
        num_pulses=8,
        cycle_samples=4800,
        sample_rate=48000,
        multichannel_config={}
    )

    processor = SignalProcessor(config)
    signal = np.sin(2 * np.pi * 440 * np.arange(48000) / 48000)

    # Test 1: Fade length >= working length (should raise error)
    print("Testing: Fade length >= working length...")
    try:
        processor.truncate_with_fadeout(signal, 100.0, 100.0)
        print("[FAIL] Should have raised ValueError")
        assert False
    except ValueError as e:
        print(f"[PASS] Correctly raised ValueError: {e}")

    # Test 2: Negative fade length (should raise error)
    print("\nTesting: Negative fade length...")
    try:
        processor.truncate_with_fadeout(signal, 100.0, -10.0)
        print("[FAIL] Should have raised ValueError")
        assert False
    except ValueError as e:
        print(f"[PASS] Correctly raised ValueError: {e}")

    # Test 3: Very small working length (10ms)
    print("\nTesting: Very small working length (10ms)...")
    truncated = processor.truncate_with_fadeout(signal, 10.0, 2.0)
    expected_samples = int(10 * 48000 / 1000)
    assert len(truncated) == expected_samples
    print(f"[PASS] Small truncation works ({len(truncated)} samples)")

    # Test 4: Very large working length (5000ms)
    print("\nTesting: Very large working length (5000ms)...")
    truncated = processor.truncate_with_fadeout(signal, 5000.0, 50.0)
    # Signal is only 1000ms, so should be unchanged
    assert len(truncated) == len(signal)
    print(f"[PASS] Large working length works (no truncation, {len(truncated)} samples)")

    print("\n[PASS] TEST 3 PASSED\n")


def test_fade_smoothness():
    """Test that fade-out is smooth (using Hann window)"""
    print("=" * 60)
    print("TEST 4: Fade Smoothness (Hann window)")
    print("=" * 60)

    config = SignalProcessingConfig(
        num_pulses=8,
        cycle_samples=4800,
        sample_rate=48000,
        multichannel_config={}
    )

    processor = SignalProcessor(config)

    # Create constant signal (easier to visualize fade)
    signal = np.ones(48000)

    # Truncate with fade
    truncated = processor.truncate_with_fadeout(signal, 500.0, 50.0)

    fade_samples = int(50 * 48000 / 1000)
    fade_start = len(truncated) - fade_samples

    # Check that fade region is monotonically decreasing
    fade_region = truncated[fade_start:]
    differences = np.diff(fade_region)
    assert np.all(differences <= 0), "Fade should be monotonically decreasing"
    print("[PASS] Fade is monotonically decreasing")

    # Check that fade starts at ~1.0 and ends at ~0.0
    assert 0.9 < fade_region[0] <= 1.0, f"Fade should start near 1.0, got {fade_region[0]}"
    assert fade_region[-1] < 0.01, f"Fade should end near 0.0, got {fade_region[-1]}"
    print(f"[PASS] Fade range correct: {fade_region[0]:.4f} -> {fade_region[-1]:.6f}")

    # Check that non-fade region is unchanged
    non_fade_region = truncated[:fade_start]
    assert np.allclose(non_fade_region, 1.0), "Non-fade region should be unchanged"
    print(f"[PASS] Non-fade region unchanged ({len(non_fade_region)} samples = 1.0)")

    print("[PASS] TEST 4 PASSED\n")


def test_integration_with_calibration():
    """Test truncation in context similar to calibration mode"""
    print("=" * 60)
    print("TEST 5: Integration Test (Calibration Mode Context)")
    print("=" * 60)

    config = SignalProcessingConfig(
        num_pulses=10,
        cycle_samples=48000,  # 1 second cycles
        sample_rate=48000,
        multichannel_config={
            'enabled': True,
            'num_channels': 8
        }
    )

    processor = SignalProcessor(config)

    # Simulate aligned cycles (5 valid cycles of 1 second each)
    num_valid_cycles = 5
    cycles = np.random.randn(num_valid_cycles, 48000) * 0.1

    # Average cycles
    averaged = processor.average_cycles(cycles, start_cycle=0)
    print(f"Averaged response: {len(averaged)} samples ({len(averaged) / 48000:.1f} s)")

    # Apply truncation (as would happen in calibration mode)
    truncated = processor.truncate_with_fadeout(
        averaged,
        working_length_ms=300.0,  # Keep 300ms
        fade_length_ms=30.0       # Fade over 30ms
    )

    expected_samples = int(300 * 48000 / 1000)
    print(f"Truncated response: {len(truncated)} samples ({len(truncated) / 48000 * 1000:.1f} ms)")
    print(f"File size reduction: {(1 - len(truncated) / len(averaged)) * 100:.1f}%")

    assert len(truncated) == expected_samples
    print("[PASS] Truncation applied correctly after averaging")

    print("[PASS] TEST 5 PASSED\n")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("IMPULSE RESPONSE TRUNCATION TEST SUITE")
    print("=" * 60 + "\n")

    try:
        test_truncation_basic()
        test_truncation_no_op()
        test_truncation_edge_cases()
        test_fade_smoothness()
        test_integration_with_calibration()

        print("=" * 60)
        print("ALL TESTS PASSED [PASS]")
        print("=" * 60)
        print("\nTruncation feature is working correctly!")
        print("Ready for use in both standard and calibration modes.")

    except AssertionError as e:
        print(f"\n[FAIL] TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
    except Exception as e:
        print(f"\n[FAIL] ERROR: {e}")
        import traceback
        traceback.print_exc()
