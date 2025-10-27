#!/usr/bin/env python3
"""
Test UMC1820 multi-channel recording with various device configurations.
This script replicates the error conditions seen in the GUI.
"""

import sys
import io
import numpy as np

# Fix Windows console encoding
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

try:
    import sdl_audio_core
except ImportError:
    print("ERROR: sdl_audio_core not available")
    sys.exit(1)


def print_separator():
    print("\n" + "=" * 70)


def list_devices():
    """List all available audio devices."""
    print_separator()
    print("AVAILABLE AUDIO DEVICES")
    print_separator()

    devices = sdl_audio_core.list_all_devices()

    print("\nINPUT DEVICES:")
    for dev in devices.get('input_devices', []):
        print(f"  ID {dev.device_id}: {dev.name}")
        print(f"    Max channels: {dev.max_channels}")

    print("\nOUTPUT DEVICES:")
    for dev in devices.get('output_devices', []):
        print(f"  ID {dev.device_id}: {dev.name}")
        print(f"    Max channels: {dev.max_channels}")

    return devices


def create_test_signal(duration=0.1, sample_rate=48000, freq=1000):
    """Create a simple sine wave test signal."""
    t = np.arange(int(duration * sample_rate)) / sample_rate
    signal = (0.3 * np.sin(2 * np.pi * freq * t)).tolist()
    return signal


def test_device_combination(input_dev_id, output_dev_id, num_channels, test_name):
    """Test a specific device combination."""
    print_separator()
    print(f"TEST: {test_name}")
    print(f"  Input Device: {input_dev_id}")
    print(f"  Output Device: {output_dev_id}")
    print(f"  Input Channels: {num_channels}")
    print_separator()

    test_signal = create_test_signal()
    print(f"Test signal: {len(test_signal)} samples")

    try:
        result = sdl_audio_core.measure_room_response_auto_multichannel(
            test_signal,
            volume=0.3,
            input_device=input_dev_id,
            output_device=output_dev_id,
            input_channels=num_channels
        )

        if result['success']:
            print(f"✓ SUCCESS")
            print(f"  Channels recorded: {result['num_channels']}")
            print(f"  Samples per channel: {result['samples_per_channel']}")

            # Check signal levels
            for ch_idx, ch_data in enumerate(result['multichannel_data']):
                ch_np = np.array(ch_data)
                max_amp = np.max(np.abs(ch_np))
                rms = np.sqrt(np.mean(ch_np ** 2))
                print(f"  Channel {ch_idx}: max={max_amp:.4f}, rms={rms:.4f}")

            return True
        else:
            print(f"✗ FAILED")
            print(f"  Error: {result.get('error_message', 'Unknown error')}")
            return False

    except Exception as e:
        print(f"✗ EXCEPTION: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("UMC1820 Multi-Channel Test Suite")
    print(f"SDL Version: {sdl_audio_core.get_version()}")

    # List all devices
    devices = list_devices()

    # Find UMC1820 and other devices
    input_devs = devices.get('input_devices', [])
    output_devs = devices.get('output_devices', [])

    umc_input = None
    umc_output = None
    other_input = None
    other_output = None

    for dev in input_devs:
        if 'UMC' in dev.name or 'U-PHORIA' in dev.name.upper():
            umc_input = dev
        elif other_input is None:
            other_input = dev

    for dev in output_devs:
        if 'UMC' in dev.name or 'U-PHORIA' in dev.name.upper():
            umc_output = dev
        elif other_output is None:
            other_output = dev

    print_separator()
    print("DEVICE IDENTIFICATION")
    print_separator()
    print(f"UMC Input: {umc_input.name if umc_input else 'NOT FOUND'} (ID {umc_input.device_id if umc_input else 'N/A'})")
    print(f"UMC Output: {umc_output.name if umc_output else 'NOT FOUND'} (ID {umc_output.device_id if umc_output else 'N/A'})")
    print(f"Other Input: {other_input.name if other_input else 'NOT FOUND'} (ID {other_input.device_id if other_input else 'N/A'})")
    print(f"Other Output: {other_output.name if other_output else 'NOT FOUND'} (ID {other_output.device_id if other_output else 'N/A'})")

    if not umc_input:
        print("\nERROR: UMC1820 input device not found!")
        return

    # Test suite
    results = []

    # TEST 1: Default devices (in=-1, out=-1) with 1 channel
    # This should replicate the GUI error when UMC is system default
    results.append((
        "Default devices with 1 channel",
        test_device_combination(-1, -1, 1, "Default devices (in=-1, out=-1) with 1 channel")
    ))

    # TEST 2: UMC input + default output with 1 channel
    if umc_input:
        results.append((
            "UMC input + default output, 1 channel",
            test_device_combination(umc_input.device_id, -1, 1, "UMC input + default output, 1 channel")
        ))

    # TEST 3: UMC input + UMC output with 1 channel
    # This is what the user reported failing
    if umc_input and umc_output:
        results.append((
            "UMC input + UMC output, 1 channel",
            test_device_combination(umc_input.device_id, umc_output.device_id, 1, "UMC input + UMC output, 1 channel")
        ))

    # TEST 4: UMC input + UMC output with 2 channels
    if umc_input and umc_output:
        results.append((
            "UMC input + UMC output, 2 channels",
            test_device_combination(umc_input.device_id, umc_output.device_id, 2, "UMC input + UMC output, 2 channels")
        ))

    # TEST 5: UMC input + UMC output with 8 channels
    if umc_input and umc_output:
        results.append((
            "UMC input + UMC output, 8 channels",
            test_device_combination(umc_input.device_id, umc_output.device_id, 8, "UMC input + UMC output, 8 channels")
        ))

    # TEST 6: UMC input + UMC output with max channels
    if umc_input and umc_output:
        max_ch = umc_input.max_channels
        results.append((
            f"UMC input + UMC output, {max_ch} channels (max)",
            test_device_combination(umc_input.device_id, umc_output.device_id, max_ch, f"UMC input + UMC output, {max_ch} channels (max)")
        ))

    # TEST 7: UMC input + other output with 8 channels
    if umc_input and other_output:
        results.append((
            "UMC input + other output, 8 channels",
            test_device_combination(umc_input.device_id, other_output.device_id, 8, "UMC input + other output, 8 channels")
        ))

    # TEST 8: Other input + UMC output with 1 channel
    if other_input and umc_output:
        results.append((
            "Other input + UMC output, 1 channel",
            test_device_combination(other_input.device_id, umc_output.device_id, 1, "Other input + UMC output, 1 channel")
        ))

    # TEST 9: Other input + other output with 1 channel (baseline)
    if other_input and other_output:
        results.append((
            "Other input + other output, 1 channel (baseline)",
            test_device_combination(other_input.device_id, other_output.device_id, 1, "Other input + other output, 1 channel (baseline)")
        ))

    # Summary
    print_separator()
    print("TEST SUMMARY")
    print_separator()

    for test_name, success in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{status}: {test_name}")

    total = len(results)
    passed = sum(1 for _, success in results if success)
    print(f"\nTotal: {passed}/{total} tests passed")

    if passed < total:
        print("\nFailed tests indicate device/driver compatibility issues.")
        print("Check the error messages above for details.")


if __name__ == '__main__':
    main()
