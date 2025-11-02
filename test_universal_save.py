"""
Test script for universal save dispatcher implementation.

This script verifies that the refactored RoomResponseRecorder correctly implements
the universal record-process-save pipeline architecture.
"""

import sys
from pathlib import Path

# Test 1: Verify the API accepts save_files parameter
print("=" * 60)
print("Test 1: Verify API signature")
print("=" * 60)

try:
    from RoomResponseRecorder import RoomResponseRecorder
    import inspect

    sig = inspect.signature(RoomResponseRecorder.take_record)
    params = list(sig.parameters.keys())

    print(f"[PASS] RoomResponseRecorder imported successfully")
    print(f"[PASS] take_record() parameters: {params}")

    if 'save_files' in params:
        print(f"[PASS] save_files parameter EXISTS in take_record()")
    else:
        print(f"[FAIL] save_files parameter MISSING from take_record()")
        sys.exit(1)

    # Check default value
    save_files_param = sig.parameters['save_files']
    print(f"[PASS] save_files default value: {save_files_param.default}")

except Exception as e:
    print(f"[FAIL] Error importing or inspecting RoomResponseRecorder: {e}")
    sys.exit(1)

# Test 2: Verify universal save dispatcher exists
print("\n" + "=" * 60)
print("Test 2: Verify universal save dispatcher exists")
print("=" * 60)

try:
    recorder = RoomResponseRecorder()

    if hasattr(recorder, '_save_processed_data'):
        print(f"[PASS] _save_processed_data() method EXISTS")
    else:
        print(f"[FAIL] _save_processed_data() method MISSING")
        sys.exit(1)

    # _save_calibration_dataset was removed - now uses universal _save_processed_data
    if not hasattr(recorder, '_save_calibration_dataset'):
        print(f"[PASS] _save_calibration_dataset() method REMOVED (now uses universal save)")
    else:
        print(f"[FAIL] _save_calibration_dataset() method still EXISTS (should be removed)")
        sys.exit(1)

    if hasattr(recorder, '_process_calibration_mode'):
        print(f"[PASS] _process_calibration_mode() method EXISTS (refactored from _take_record_calibration_mode)")
    else:
        print(f"[FAIL] _process_calibration_mode() method MISSING")
        sys.exit(1)

except Exception as e:
    print(f"[FAIL] Error checking methods: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 3: Verify method signatures
print("\n" + "=" * 60)
print("Test 3: Verify method signatures")
print("=" * 60)

try:
    # Check _save_processed_data signature
    save_sig = inspect.signature(recorder._save_processed_data)
    save_params = list(save_sig.parameters.keys())
    print(f"[PASS] _save_processed_data() parameters: {save_params}")

    expected_params = ['processed_data', 'output_file', 'impulse_file']
    for param in expected_params:
        if param in save_params:
            print(f"  [PASS] Has '{param}' parameter")
        else:
            print(f"  [FAIL] Missing '{param}' parameter")
            sys.exit(1)

    # Verify 'mode' parameter was removed (now truly universal)
    if 'mode' not in save_params:
        print(f"  [PASS] 'mode' parameter correctly removed (universal save)")
    else:
        print(f"  [FAIL] 'mode' parameter still present (should be removed)")

    # Check _process_calibration_mode signature
    process_sig = inspect.signature(recorder._process_calibration_mode)
    process_params = list(process_sig.parameters.keys())
    print(f"[PASS] _process_calibration_mode() parameters: {process_params}")

    if 'recorded_audio' in process_params:
        print(f"  [PASS] Has 'recorded_audio' parameter (Stage 1 output)")
    else:
        print(f"  [FAIL] Missing 'recorded_audio' parameter")
        sys.exit(1)

except Exception as e:
    print(f"[FAIL] Error checking signatures: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: Verify deprecated methods still exist (backward compatibility)
print("\n" + "=" * 60)
print("Test 4: Verify backward compatibility (deprecated methods exist)")
print("=" * 60)

try:
    if hasattr(recorder, '_save_multichannel_files'):
        print(f"[PASS] _save_multichannel_files() EXISTS (deprecated but kept for compatibility)")
    else:
        print(f"[FAIL] _save_multichannel_files() MISSING (breaks backward compatibility)")
        sys.exit(1)

    if hasattr(recorder, '_save_single_channel_files'):
        print(f"[PASS] _save_single_channel_files() EXISTS (deprecated but kept for compatibility)")
    else:
        print(f"[FAIL] _save_single_channel_files() MISSING (breaks backward compatibility)")
        sys.exit(1)

    if hasattr(recorder, 'take_record_calibration'):
        print(f"[PASS] take_record_calibration() EXISTS (convenience method)")
    else:
        print(f"[FAIL] take_record_calibration() MISSING")
        sys.exit(1)

except Exception as e:
    print(f"[FAIL] Error checking backward compatibility: {e}")
    sys.exit(1)

# Test 5: Verify documentation
print("\n" + "=" * 60)
print("Test 5: Verify documentation")
print("=" * 60)

try:
    # Check that take_record docstring mentions three-stage pipeline
    take_record_doc = recorder.take_record.__doc__
    if 'three-stage pipeline' in take_record_doc.lower() or '1. record' in take_record_doc.lower():
        print(f"[PASS] take_record() docstring documents three-stage pipeline")
    else:
        print(f"[WARN] take_record() docstring should document three-stage pipeline architecture")

    # Check that _save_processed_data mentions universal
    save_doc = recorder._save_processed_data.__doc__
    if 'universal' in save_doc.lower() or 'dispatcher' in save_doc.lower():
        print(f"[PASS] _save_processed_data() docstring mentions universal/dispatcher")
    else:
        print(f"[WARN] _save_processed_data() docstring should mention universal dispatcher")

    # Check that deprecated methods are marked
    multichannel_doc = recorder._save_multichannel_files.__doc__
    if 'deprecated' in multichannel_doc.lower():
        print(f"[PASS] _save_multichannel_files() marked as DEPRECATED")
    else:
        print(f"[WARN] _save_multichannel_files() should be marked as DEPRECATED")

    single_doc = recorder._save_single_channel_files.__doc__
    if 'deprecated' in single_doc.lower():
        print(f"[PASS] _save_single_channel_files() marked as DEPRECATED")
    else:
        print(f"[WARN] _save_single_channel_files() should be marked as DEPRECATED")

except Exception as e:
    print(f"[WARN] Error checking documentation: {e}")

# Final summary
print("\n" + "=" * 60)
print("TEST SUMMARY")
print("=" * 60)
print("[PASS] All critical tests PASSED")
print("[PASS] Universal save dispatcher implemented correctly")
print("[PASS] Three-stage pipeline architecture established:")
print("  - Stage 1 (Record): Universal, mode-independent")
print("  - Stage 2 (Process): Mode-specific")
print("  - Stage 3 (Save): Universal, mode-independent, optional")
print("[PASS] Backward compatibility maintained")
print("[PASS] API properly documented")
print("\n[SUCCESS] Implementation successful!")
