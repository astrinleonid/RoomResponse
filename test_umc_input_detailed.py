#!/usr/bin/env python3
"""
Detailed test of UMC1820 INPUT device with SDL directly.
Try to understand why it's failing to open.
"""

import sys
import io

# Fix Windows console encoding
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

try:
    import sdl_audio_core
except ImportError:
    print("ERROR: sdl_audio_core not available")
    sys.exit(1)


def test_umc_input_channel_counts():
    """Test UMC input with various channel counts to find which ones work."""

    devices = sdl_audio_core.list_all_devices()

    umc_input = None
    for dev in devices.get('input_devices', []):
        if 'UMC' in dev.name or 'U-PHORIA' in dev.name.upper():
            umc_input = dev
            break

    if not umc_input:
        print("ERROR: UMC1820 input not found")
        return

    print(f"Testing UMC1820 Input Device")
    print(f"Device ID: {umc_input.device_id}")
    print(f"Device Name: {umc_input.name}")
    print(f"Max Channels: {umc_input.max_channels}")
    print()

    # Use Realtek for output (we know this works)
    output_dev = None
    for dev in devices.get('output_devices', []):
        if 'Realtek' in dev.name:
            output_dev = dev
            break

    if not output_dev:
        output_dev_id = 0  # fallback to device 0
    else:
        output_dev_id = output_dev.device_id

    print(f"Using output device ID: {output_dev_id}")
    print()

    # Test signal
    import numpy as np
    duration = 0.1
    sample_rate = 48000
    t = np.arange(int(duration * sample_rate)) / sample_rate
    test_signal = (0.3 * np.sin(2 * np.pi * 1000 * t)).tolist()

    # Try various channel counts
    test_configs = [
        1, 2, 4, 6, 8, 10,  # Standard counts
        # 12, 16,  # Higher counts
    ]

    results = []

    for num_ch in test_configs:
        print(f"Testing with {num_ch} channels...")

        try:
            result = sdl_audio_core.measure_room_response_auto_multichannel(
                test_signal,
                volume=0.2,
                input_device=umc_input.device_id,
                output_device=output_dev_id,
                input_channels=num_ch
            )

            if result['success']:
                print(f"  ✓ SUCCESS - got {result['num_channels']} channels")
                results.append((num_ch, True, result['num_channels']))
            else:
                print(f"  ✗ FAILED - {result.get('error_message', 'Unknown error')}")
                results.append((num_ch, False, None))

        except Exception as e:
            print(f"  ✗ EXCEPTION - {e}")
            results.append((num_ch, False, None))

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    for requested, success, actual in results:
        if success:
            print(f"✓ {requested} channels requested → {actual} channels obtained")
        else:
            print(f"✗ {requested} channels requested → FAILED")

    # Check if ANY configuration worked
    if not any(success for _, success, _ in results):
        print("\n⚠ WARNING: UMC1820 input failed with ALL channel configurations!")
        print("This indicates a driver/SDL compatibility issue with the device.")
        print("\nPossible causes:")
        print("1. UMC1820 Windows driver doesn't properly support SDL/WASAPI")
        print("2. Device requires exclusive mode access")
        print("3. Device is being used by another application")
        print("4. SDL version incompatibility with UMC1820 driver")
        print("\nRecommendations:")
        print("- Check if UMC1820 is set to 'exclusive mode' in Windows sound settings")
        print("- Ensure no other apps are using the UMC1820")
        print("- Try updating UMC1820 drivers")
        print("- Test with ASIO driver instead of SDL (if available)")


if __name__ == '__main__':
    test_umc_input_channel_counts()
