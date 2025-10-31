"""
Level 2 Integration Tests - Code Path Verification
Tests that code paths execute correctly (no actual hardware needed)
"""

import sys
import os
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from RoomResponseRecorder import RoomResponseRecorder

# Fix Windows console encoding
if os.name == 'nt':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def test_standard_mode_code_path():
    """Verify standard mode executes existing code path"""
    print("\n" + "="*60)
    print("TEST: Standard mode code path")
    print("="*60)

    try:
        recorder = RoomResponseRecorder("recorderConfig.json")

        # Mock the recording method to avoid needing hardware
        mock_audio = np.random.randn(recorder.cycle_samples * recorder.num_pulses).astype(np.float32)

        with patch.object(recorder, '_record_method_2', return_value=mock_audio):
            with patch.object(recorder, '_save_single_channel_files'):
                # Call with default parameters (should use standard mode)
                result = recorder.take_record("test_raw.wav", "test_impulse.wav")

                # Should return raw audio (backward compatible)
                assert isinstance(result, np.ndarray), f"Expected ndarray, got {type(result)}"
                assert len(result) > 0, "Result should not be empty"

                print("✅ Standard mode returns raw audio (backward compatible)")
                print("✅ PASSED: Standard mode code path works")
                return True

    except Exception as e:
        print(f"❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_return_processed_flag():
    """Verify return_processed flag changes return type"""
    print("\n" + "="*60)
    print("TEST: return_processed flag")
    print("="*60)

    try:
        recorder = RoomResponseRecorder("recorderConfig.json")

        # Mock the recording method
        mock_audio = np.random.randn(recorder.cycle_samples * recorder.num_pulses).astype(np.float32)

        with patch.object(recorder, '_record_method_2', return_value=mock_audio):
            with patch.object(recorder, '_save_single_channel_files'):
                # Call with return_processed=True
                result = recorder.take_record("test_raw.wav", "test_impulse.wav", return_processed=True)

                # Should return dict with processed data
                assert isinstance(result, dict), f"Expected dict, got {type(result)}"
                assert 'raw' in result, "Result should contain 'raw'"
                assert 'impulse' in result, "Result should contain 'impulse'"
                assert 'room_response' in result, "Result should contain 'room_response'"

                print("✅ return_processed=True returns dict with processed data")
                print("✅ PASSED: return_processed flag works")
                return True

    except Exception as e:
        print(f"❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_multichannel_standard_mode():
    """Verify multi-channel standard mode works"""
    print("\n" + "="*60)
    print("TEST: Multi-channel standard mode")
    print("="*60)

    try:
        recorder = RoomResponseRecorder("recorderConfig.json")

        # Enable multi-channel
        recorder.multichannel_config['enabled'] = True
        recorder.multichannel_config['num_channels'] = 2
        recorder.multichannel_config['reference_channel'] = 0  # Fix reference channel

        # Mock multi-channel recording
        mock_audio = {
            0: np.random.randn(recorder.cycle_samples * recorder.num_pulses).astype(np.float32),
            1: np.random.randn(recorder.cycle_samples * recorder.num_pulses).astype(np.float32)
        }

        with patch.object(recorder, '_record_method_2', return_value=mock_audio):
            with patch.object(recorder, '_save_multichannel_files'):
                # Call with default parameters
                result = recorder.take_record("test_raw.wav", "test_impulse.wav")

                # Should return dict of raw audio
                assert isinstance(result, dict), f"Expected dict, got {type(result)}"
                assert 0 in result, "Result should contain channel 0"
                assert 1 in result, "Result should contain channel 1"
                assert isinstance(result[0], np.ndarray), "Channel 0 should be ndarray"

                print("✅ Multi-channel standard mode returns dict of raw audio")
                print("✅ PASSED: Multi-channel standard mode works")
                return True

    except Exception as e:
        print(f"❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_calibration_mode_structure():
    """Verify calibration mode returns correct structure"""
    print("\n" + "="*60)
    print("TEST: Calibration mode structure")
    print("="*60)

    try:
        recorder = RoomResponseRecorder("recorderConfig.json")

        # Enable multi-channel with calibration channel
        recorder.multichannel_config['enabled'] = True
        recorder.multichannel_config['num_channels'] = 2
        recorder.multichannel_config['calibration_channel'] = 1

        # Mock multi-channel recording
        mock_audio = {
            0: np.random.randn(recorder.cycle_samples * recorder.num_pulses).astype(np.float32),
            1: np.random.randn(recorder.cycle_samples * recorder.num_pulses).astype(np.float32)
        }

        # Mock validation and alignment methods
        with patch.object(recorder, '_record_method_2', return_value=mock_audio):
            # Mock align_cycles_by_onset to return proper structure
            mock_alignment = {
                'aligned_cycles': np.random.randn(recorder.num_pulses, recorder.cycle_samples).astype(np.float32),
                'num_aligned': recorder.num_pulses,
                'shifts': [0] * recorder.num_pulses,
                'valid_cycle_indices': list(range(recorder.num_pulses))
            }
            with patch.object(recorder, 'align_cycles_by_onset', return_value=mock_alignment):
                with patch.object(recorder, 'apply_alignment_to_channel', return_value=np.random.randn(recorder.num_pulses, recorder.cycle_samples).astype(np.float32)):
                    # Call calibration mode
                    result = recorder.take_record("", "", mode='calibration')

                    # Verify structure
                    assert isinstance(result, dict), f"Expected dict, got {type(result)}"
                    assert 'calibration_cycles' in result, "Missing calibration_cycles"
                    assert 'validation_results' in result, "Missing validation_results"
                    assert 'aligned_multichannel_cycles' in result, "Missing aligned_multichannel_cycles"
                    assert 'alignment_metadata' in result, "Missing alignment_metadata"
                    assert 'num_valid_cycles' in result, "Missing num_valid_cycles"
                    assert 'num_aligned_cycles' in result, "Missing num_aligned_cycles"
                    assert 'metadata' in result, "Missing metadata"

                    print("✅ Calibration mode returns complete structure")
                    print("✅ All required keys present")
                    print("✅ PASSED: Calibration mode structure correct")
                    return True

    except Exception as e:
        print(f"❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_convenience_method_delegates():
    """Verify convenience method delegates correctly"""
    print("\n" + "="*60)
    print("TEST: Convenience method delegation")
    print("="*60)

    try:
        recorder = RoomResponseRecorder("recorderConfig.json")

        # Enable multi-channel with calibration channel
        recorder.multichannel_config['enabled'] = True
        recorder.multichannel_config['calibration_channel'] = 0

        # Mock recording and alignment
        mock_audio = {0: np.random.randn(recorder.cycle_samples * recorder.num_pulses).astype(np.float32)}
        mock_alignment = {
            'aligned_cycles': np.random.randn(recorder.num_pulses, recorder.cycle_samples).astype(np.float32),
            'num_aligned': recorder.num_pulses,
            'shifts': [0] * recorder.num_pulses,
            'valid_cycle_indices': list(range(recorder.num_pulses))
        }

        with patch.object(recorder, '_record_method_2', return_value=mock_audio):
            with patch.object(recorder, 'align_cycles_by_onset', return_value=mock_alignment):
                with patch.object(recorder, 'apply_alignment_to_channel', return_value=np.random.randn(recorder.num_pulses, recorder.cycle_samples).astype(np.float32)):
                    # Call convenience method
                    result = recorder.take_record_calibration()

                    # Should have same structure as mode='calibration'
                    assert isinstance(result, dict), "Should return dict"
                    assert 'calibration_cycles' in result, "Should have calibration_cycles"

                    print("✅ Convenience method delegates to calibration mode")
                    print("✅ PASSED: Convenience method works")
                    return True

    except Exception as e:
        print(f"❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_all_tests():
    """Run all Level 2 integration tests"""
    print("\n" + "="*60)
    print("LEVEL 2 INTEGRATION TESTS - CODE PATH VERIFICATION")
    print("="*60)

    tests = [
        ("Standard mode code path", test_standard_mode_code_path),
        ("return_processed flag", test_return_processed_flag),
        ("Multi-channel standard mode", test_multichannel_standard_mode),
        ("Calibration mode structure", test_calibration_mode_structure),
        ("Convenience method delegation", test_convenience_method_delegates),
    ]

    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n❌ TEST FAILED WITH EXCEPTION: {name}")
            print(f"   Error: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))

    # Print summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{status}: {name}")

    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 ALL LEVEL 2 TESTS PASSED!")
        print("   Proceeding to Level 3 (hardware) tests is SAFE")
        return True
    else:
        print(f"\n❌ {total - passed} TEST(S) FAILED")
        print("   DO NOT PROCEED - Fix failures first")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
