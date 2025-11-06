"""
Basic tests for SignalProcessor (no pytest required)

Tests key functionality to ensure the refactoring works correctly.
"""

import numpy as np
from signal_processor import SignalProcessor, SignalProcessingConfig


def test_config_creation():
    """Test basic config creation"""
    config = SignalProcessingConfig(
        num_pulses=8,
        cycle_samples=4800,
        sample_rate=48000
    )
    assert config.num_pulses == 8
    assert config.cycle_samples == 4800
    assert config.sample_rate == 48000
    print("[PASS] Config creation test")


def test_extract_cycles():
    """Test cycle extraction"""
    config = SignalProcessingConfig(
        num_pulses=4,
        cycle_samples=1000,
        sample_rate=48000
    )
    processor = SignalProcessor(config)

    audio = np.random.randn(4000).astype(np.float32)
    cycles = processor.extract_cycles(audio)

    assert cycles.shape == (4, 1000)
    print("[PASS] Extract cycles test")


def test_average_cycles():
    """Test cycle averaging"""
    config = SignalProcessingConfig(
        num_pulses=4,
        cycle_samples=1000,
        sample_rate=48000
    )
    processor = SignalProcessor(config)

    cycles = np.array([
        np.ones(1000) * 1.0,
        np.ones(1000) * 2.0,
        np.ones(1000) * 3.0,
        np.ones(1000) * 4.0
    ])

    avg = processor.average_cycles(cycles, start_cycle=1)
    assert avg.shape == (1000,)
    assert np.allclose(avg, 3.0)  # (2 + 3 + 4) / 3 = 3.0
    print("[PASS] Average cycles test")


def test_spectral_analysis():
    """Test spectral analysis"""
    config = SignalProcessingConfig(
        num_pulses=4,
        cycle_samples=1000,
        sample_rate=48000
    )
    processor = SignalProcessor(config)

    t = np.arange(1000) / 48000
    signal = np.sin(2 * np.pi * 1000 * t)
    responses = {0: signal}

    result = processor.compute_spectral_analysis(responses)

    assert 'frequencies' in result
    assert 'magnitudes' in result
    assert 'magnitude_db' in result
    print("[PASS] Spectral analysis test")


def test_find_sound_onset():
    """Test onset detection"""
    config = SignalProcessingConfig(
        num_pulses=4,
        cycle_samples=1000,
        sample_rate=48000
    )
    processor = SignalProcessor(config)

    # Create signal with onset at sample 100
    audio = np.zeros(1000)
    audio[100:] = np.random.randn(900) * 0.5 + 0.5

    onset = processor.find_sound_onset(audio)

    # Should detect onset around sample 100
    assert 80 <= onset <= 150
    print(f"[PASS] Find sound onset test (detected at sample {onset})")


def test_align_cycles_basic():
    """Test basic cycle alignment"""
    config = SignalProcessingConfig(
        num_pulses=3,
        cycle_samples=1000,
        sample_rate=48000,
        multichannel_config={'alignment_target_onset_position': 0}
    )
    processor = SignalProcessor(config)

    # Create 3 cycles with negative peaks at different positions
    cycle1 = np.zeros(1000)
    cycle1[100] = -1.0

    cycle2 = np.zeros(1000)
    cycle2[150] = -1.0

    cycle3 = np.zeros(1000)
    cycle3[120] = -1.0

    cycles = np.array([cycle1, cycle2, cycle3])

    validation_results = [
        {'calibration_valid': True},
        {'calibration_valid': True},
        {'calibration_valid': True}
    ]

    result = processor.align_cycles_by_onset(cycles, validation_results)

    assert len(result['aligned_cycles']) > 0
    assert 'aligned_onset_position' in result
    assert 'correlations' in result
    print(f"[PASS] Align cycles test ({len(result['aligned_cycles'])} cycles aligned)")


def test_normalize_by_calibration():
    """Test calibration normalization"""
    config = SignalProcessingConfig(
        num_pulses=1,
        cycle_samples=1000,
        sample_rate=48000
    )
    processor = SignalProcessor(config)

    aligned_cycles = {
        0: np.array([np.ones(1000) * 2.0]),  # Calibration
        1: np.array([np.ones(1000) * 4.0])   # Response
    }

    validation_results = [
        {'calibration_metrics': {'negative_peak': 0.5}}
    ]

    normalized, factors = processor.normalize_by_calibration(
        aligned_cycles,
        validation_results,
        calibration_channel=0,
        valid_cycle_indices=[0]
    )

    assert np.allclose(normalized[0], 2.0)  # Calibration unchanged
    assert np.allclose(normalized[1], 8.0)  # Response normalized
    print("[PASS] Normalize by calibration test")


def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("SignalProcessor Basic Tests")
    print("="*60 + "\n")

    tests = [
        test_config_creation,
        test_extract_cycles,
        test_average_cycles,
        test_spectral_analysis,
        test_find_sound_onset,
        test_align_cycles_basic,
        test_normalize_by_calibration,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"[FAIL] {test.__name__}: {e}")
            failed += 1
        except Exception as e:
            print(f"[ERROR] {test.__name__}: {e}")
            failed += 1

    print("\n" + "="*60)
    print(f"Results: {passed} passed, {failed} failed")
    print("="*60 + "\n")

    return failed == 0


if __name__ == '__main__':
    success = main()
    exit(0 if success else 1)
