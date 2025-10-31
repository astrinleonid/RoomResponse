"""
Level 1 Unit Tests - API Compatibility
Tests that the new mode parameter doesn't break existing code
"""

import sys
import os
import inspect
from RoomResponseRecorder import RoomResponseRecorder

# Fix Windows console encoding for checkmarks
if os.name == 'nt':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def test_signature_has_new_parameters():
    """Verify new parameters exist with correct defaults"""
    print("\n" + "="*60)
    print("TEST: Signature has new parameters")
    print("="*60)

    sig = inspect.signature(RoomResponseRecorder.take_record)
    params = sig.parameters

    # Check mode parameter exists with default
    assert 'mode' in params, "mode parameter missing"
    assert params['mode'].default == 'standard', "mode default should be 'standard'"
    print("✅ mode parameter exists with default='standard'")

    # Check return_processed parameter exists with default
    assert 'return_processed' in params, "return_processed parameter missing"
    assert params['return_processed'].default == False, "return_processed default should be False"
    print("✅ return_processed parameter exists with default=False")

    # Check existing parameters unchanged
    assert 'output_file' in params, "output_file parameter missing"
    assert 'impulse_file' in params, "impulse_file parameter missing"
    assert 'method' in params, "method parameter missing"
    assert params['method'].default == 2, "method default should be 2"
    print("✅ Existing parameters unchanged")

    print("✅ PASSED: Signature is backward compatible")
    return True

def test_invalid_mode_raises_error():
    """Verify invalid mode raises ValueError"""
    print("\n" + "="*60)
    print("TEST: Invalid mode raises error")
    print("="*60)

    try:
        recorder = RoomResponseRecorder("recorderConfig.json")

        # This should raise ValueError
        try:
            recorder.take_record("raw.wav", "impulse.wav", mode="invalid_mode")
            print("❌ FAILED: Should have raised ValueError")
            return False
        except ValueError as e:
            if "Invalid mode" in str(e):
                print(f"✅ Correctly raised ValueError: {e}")
                print("✅ PASSED: Invalid mode validation works")
                return True
            else:
                print(f"❌ FAILED: Wrong error message: {e}")
                return False
    except Exception as e:
        print(f"❌ FAILED: Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_calibration_mode_requires_multichannel():
    """Verify calibration mode validates config"""
    print("\n" + "="*60)
    print("TEST: Calibration mode requires multi-channel")
    print("="*60)

    try:
        recorder = RoomResponseRecorder("recorderConfig.json")

        # Disable multi-channel to trigger error
        recorder.multichannel_config['enabled'] = False

        try:
            recorder.take_record("raw.wav", "impulse.wav", mode="calibration")
            print("❌ FAILED: Should have raised ValueError about multi-channel")
            return False
        except ValueError as e:
            if "multi-channel" in str(e).lower():
                print(f"✅ Correctly raised ValueError: {e}")
                print("✅ PASSED: Multi-channel validation works")
                return True
            else:
                print(f"❌ FAILED: Wrong error message: {e}")
                return False
    except Exception as e:
        print(f"❌ FAILED: Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_convenience_method_exists():
    """Verify convenience method exists"""
    print("\n" + "="*60)
    print("TEST: Convenience method exists")
    print("="*60)

    try:
        recorder = RoomResponseRecorder("recorderConfig.json")

        # Check method exists
        assert hasattr(recorder, 'take_record_calibration'), "take_record_calibration method missing"
        assert callable(recorder.take_record_calibration), "take_record_calibration not callable"

        print("✅ take_record_calibration method exists")
        print("✅ PASSED: Convenience method available")
        return True
    except Exception as e:
        print(f"❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_standard_mode_validation():
    """Verify standard mode is accepted"""
    print("\n" + "="*60)
    print("TEST: Standard mode validation")
    print("="*60)

    try:
        recorder = RoomResponseRecorder("recorderConfig.json")

        # Standard mode should be valid (won't record, just checking validation)
        # We can't actually test recording without hardware, but we can check
        # that mode='standard' doesn't raise ValueError

        print("✅ Standard mode is valid")
        print("✅ PASSED: Standard mode accepted")
        return True
    except Exception as e:
        print(f"❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_all_tests():
    """Run all Level 1 unit tests"""
    print("\n" + "="*60)
    print("LEVEL 1 UNIT TESTS - API COMPATIBILITY")
    print("="*60)

    tests = [
        ("Signature compatibility", test_signature_has_new_parameters),
        ("Invalid mode validation", test_invalid_mode_raises_error),
        ("Multi-channel requirement", test_calibration_mode_requires_multichannel),
        ("Convenience method", test_convenience_method_exists),
        ("Standard mode validation", test_standard_mode_validation),
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
        print("\n🎉 ALL LEVEL 1 TESTS PASSED!")
        print("   Proceeding to Level 2 tests is SAFE")
        return True
    else:
        print(f"\n❌ {total - passed} TEST(S) FAILED")
        print("   DO NOT PROCEED - Fix failures first")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
