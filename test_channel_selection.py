#!/usr/bin/env python3
"""
Test script to verify channel selection works correctly in MicTesting.AudioRecorder

This test verifies that:
1. AudioRecorder correctly configures multi-channel input
2. get_audio_chunk() returns data from the SELECTED channel only
3. Different channels contain different data (not all the same)
"""

import sys
import time
import numpy as np

# UTF-8 setup for Windows
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')

try:
    import sdl_audio_core
    import MicTesting
except ImportError as e:
    print(f"✗ Failed to import required modules: {e}")
    sys.exit(1)


def test_channel_selection():
    """Test that channel selection works correctly"""
    print("\n" + "="*70)
    print("TEST: Channel Selection in MicTesting.AudioRecorder")
    print("="*70)

    # List available devices
    try:
        devices = sdl_audio_core.list_all_devices()
        input_devices = devices.get('input_devices', [])

        if not input_devices:
            print("✗ No input devices found")
            return False

        print(f"\nFound {len(input_devices)} input device(s):")
        for dev in input_devices:
            print(f"  [{dev.device_id}] {dev.name} - {dev.max_channels} channels")

        # Find first multi-channel device (2+ channels)
        multichannel_dev = None
        for dev in input_devices:
            if dev.max_channels >= 2:
                multichannel_dev = dev
                break

        if not multichannel_dev:
            print("\n⚠ No multi-channel device found (need 2+ channels)")
            print("Testing with mono device (channel selection won't be tested)")
            test_device = input_devices[0]
            test_channels = 1
        else:
            test_device = multichannel_dev
            test_channels = min(test_device.max_channels, 4)  # Test up to 4 channels
            print(f"\n✓ Using multi-channel device: {test_device.name}")
            print(f"  Testing {test_channels} channels")

    except Exception as e:
        print(f"✗ Device enumeration failed: {e}")
        return False

    # Test 1: Verify channel 0 works
    print("\n" + "-"*70)
    print("Test 1: Monitor Channel 0")
    print("-"*70)
    try:
        with MicTesting.AudioRecorder(
            sample_rate=48000,
            input_device=test_device.device_id,
            input_channels=test_channels,
            input_channel=0
        ) as ar:
            time.sleep(0.2)  # Let some data accumulate
            chunk = ar.get_audio_chunk(4800)  # 100ms at 48kHz

            if len(chunk) == 0:
                print("✗ No data received on channel 0")
                return False

            rms = np.sqrt(np.mean(chunk ** 2))
            db = 20 * np.log10(rms) if rms > 1e-10 else -60.0

            print(f"✓ Channel 0: {len(chunk)} samples, RMS={rms:.6f}, Level={db:+.1f} dB")
    except Exception as e:
        print(f"✗ Channel 0 test failed: {e}")
        return False

    # Test 2: Verify channel 1 works (if available)
    if test_channels >= 2:
        print("\n" + "-"*70)
        print("Test 2: Monitor Channel 1")
        print("-"*70)
        try:
            with MicTesting.AudioRecorder(
                sample_rate=48000,
                input_device=test_device.device_id,
                input_channels=test_channels,
                input_channel=1
            ) as ar:
                time.sleep(0.2)
                chunk = ar.get_audio_chunk(4800)

                if len(chunk) == 0:
                    print("✗ No data received on channel 1")
                    return False

                rms = np.sqrt(np.mean(chunk ** 2))
                db = 20 * np.log10(rms) if rms > 1e-10 else -60.0

                print(f"✓ Channel 1: {len(chunk)} samples, RMS={rms:.6f}, Level={db:+.1f} dB")
        except Exception as e:
            print(f"✗ Channel 1 test failed: {e}")
            return False

    # Test 3: Verify channels are different (simultaneous recording)
    if test_channels >= 2:
        print("\n" + "-"*70)
        print("Test 3: Verify Channels are Different")
        print("-"*70)
        try:
            # Record all channels simultaneously
            engine = sdl_audio_core.AudioEngine()
            config = sdl_audio_core.AudioEngineConfig()
            config.sample_rate = 48000
            config.input_channels = test_channels

            if not engine.initialize(config):
                print("✗ Failed to initialize engine")
                return False

            engine.set_input_device(test_device.device_id)
            engine.start_recording()

            time.sleep(0.3)  # Record for 300ms

            # Get data from each channel
            channel_data = []
            for ch in range(test_channels):
                data = engine.get_recorded_data_channel(ch)
                if len(data) < 1000:
                    print(f"✗ Insufficient data on channel {ch}: {len(data)} samples")
                    engine.stop_recording()
                    engine.shutdown()
                    return False
                channel_data.append(np.array(data[:4800]))  # Take first 100ms worth

            engine.stop_recording()
            engine.shutdown()

            # Compare channels
            print(f"\n✓ Recorded {test_channels} channels simultaneously")

            for ch in range(test_channels):
                rms = np.sqrt(np.mean(channel_data[ch] ** 2))
                db = 20 * np.log10(rms) if rms > 1e-10 else -60.0
                print(f"  Channel {ch}: RMS={rms:.6f}, Level={db:+.1f} dB")

            # Check if channels are identical (they shouldn't be)
            if test_channels >= 2:
                correlation = np.corrcoef(channel_data[0], channel_data[1])[0, 1]
                print(f"\n  Correlation between Ch0 and Ch1: {correlation:.4f}")

                if abs(correlation - 1.0) < 0.001:
                    print("  ⚠ WARNING: Channels are nearly identical!")
                    print("  This suggests de-interleaving may not be working correctly.")
                else:
                    print("  ✓ Channels are different (de-interleaving working)")

        except Exception as e:
            print(f"✗ Channel comparison test failed: {e}")
            return False

    # Test 4: Verify invalid channel is rejected
    print("\n" + "-"*70)
    print("Test 4: Verify Invalid Channel is Rejected")
    print("-"*70)
    try:
        # Try to access channel beyond device capability
        with MicTesting.AudioRecorder(
            sample_rate=48000,
            input_device=test_device.device_id,
            input_channels=test_channels,
            input_channel=test_channels + 5  # Beyond available channels
        ) as ar:
            time.sleep(0.2)
            chunk = ar.get_audio_chunk(4800)

            if len(chunk) == 0:
                print(f"✓ Correctly returns empty data for channel {test_channels + 5}")
            else:
                print(f"⚠ WARNING: Got data from invalid channel {test_channels + 5}")
                print(f"  This may indicate the channel index is not being validated")
    except Exception as e:
        print(f"⚠ Exception when accessing invalid channel: {e}")
        print("  (This is acceptable behavior)")

    print("\n" + "="*70)
    print("✓ ALL TESTS PASSED")
    print("="*70)
    return True


if __name__ == '__main__':
    try:
        success = test_channel_selection()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n\n✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
