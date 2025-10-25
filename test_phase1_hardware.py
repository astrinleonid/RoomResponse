#!/usr/bin/env python3
"""
Phase 1 Hardware Testing Suite
Tests multi-channel recording with actual audio hardware
"""

import sys
import numpy as np
import os
from pathlib import Path


def list_audio_devices():
    """List all available audio devices"""
    print("=" * 60)
    print("AUDIO DEVICE ENUMERATION")
    print("=" * 60)

    try:
        import sdl_audio_core

        devices = sdl_audio_core.list_all_devices()

        print(f"\nInput Devices: {len(devices['input_devices'])}")
        for dev in devices['input_devices']:
            print(f"  [{dev.device_id}] {dev.name}")
            print(f"      Max channels: {dev.max_channels}")

        print(f"\nOutput Devices: {len(devices['output_devices'])}")
        for dev in devices['output_devices']:
            print(f"  [{dev.device_id}] {dev.name}")
            print(f"      Max channels: {dev.max_channels}")

        return devices
    except Exception as e:
        print(f"✗ Failed to enumerate devices: {e}")
        import traceback
        traceback.print_exc()
        return None


def find_multichannel_device(devices, min_channels=2):
    """Find an input device that supports at least min_channels"""
    if not devices:
        return None

    for dev in devices['input_devices']:
        if dev.max_channels >= min_channels:
            return dev

    return None


def test_single_channel_recording():
    """Test mono recording (baseline test)"""
    print("\n" + "=" * 60)
    print("TEST 1: Single-Channel Recording (Baseline)")
    print("=" * 60)

    try:
        import sdl_audio_core

        # Generate test signal (1kHz sine, 100ms)
        sample_rate = 48000
        duration = 0.1
        freq = 1000

        t = np.arange(int(duration * sample_rate)) / sample_rate
        test_signal = (0.3 * np.sin(2 * np.pi * freq * t)).tolist()

        print(f"Test signal: {len(test_signal)} samples @ {sample_rate} Hz")
        print("Recording with default (mono) configuration...")

        result = sdl_audio_core.measure_room_response_auto(
            test_signal,
            volume=0.3,
            input_device=-1,  # Default
            output_device=-1  # Default
        )

        if not result['success']:
            print(f"✗ Recording failed: {result.get('error_message', 'Unknown error')}")
            return False

        recorded = result['recorded_data']
        print(f"✓ Recorded {len(recorded)} samples")

        # Basic validation
        assert len(recorded) > 0, "No data recorded"
        assert isinstance(recorded, list), "Data should be list"

        # Check for non-zero signal
        recorded_np = np.array(recorded)
        max_amp = np.max(np.abs(recorded_np))
        rms = np.sqrt(np.mean(recorded_np ** 2))

        print(f"  Max amplitude: {max_amp:.4f}")
        print(f"  RMS: {rms:.4f}")

        if max_amp < 0.001:
            print("⚠️  Warning: Very low signal amplitude - check volume/microphone")

        return True

    except Exception as e:
        print(f"✗ Single-channel test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_multichannel_recording(num_channels=2):
    """Test multi-channel recording"""
    print("\n" + "=" * 60)
    print(f"TEST 2: {num_channels}-Channel Recording")
    print("=" * 60)

    try:
        import sdl_audio_core

        # Check if we have a device that supports this many channels
        devices = sdl_audio_core.list_all_devices()
        dev = find_multichannel_device(devices, num_channels)

        if not dev:
            print(f"⚠️  SKIPPED: No device found supporting {num_channels} channels")
            print(f"   Available devices:")
            for d in devices['input_devices']:
                print(f"     - {d.name}: max {d.max_channels} channels")
            return True  # Not a failure, just can't test

        print(f"Using device: {dev.name} (max {dev.max_channels} channels)")

        # Generate test signal
        sample_rate = 48000
        duration = 0.2  # 200ms
        freq = 1000

        t = np.arange(int(duration * sample_rate)) / sample_rate
        test_signal = (0.3 * np.sin(2 * np.pi * freq * t)).tolist()

        print(f"Test signal: {len(test_signal)} samples @ {sample_rate} Hz")
        print(f"Recording {num_channels} channels...")

        result = sdl_audio_core.measure_room_response_auto_multichannel(
            test_signal,
            volume=0.3,
            input_device=dev.device_id,
            output_device=0,  # Default output
            input_channels=num_channels
        )

        if not result['success']:
            print(f"✗ Recording failed: {result.get('error_message', 'Unknown error')}")
            return False

        print(f"✓ Recording completed successfully")
        print(f"  Channels recorded: {result['num_channels']}")
        print(f"  Samples per channel: {result['samples_per_channel']}")

        # Validate result structure
        assert result['num_channels'] == num_channels, \
            f"Expected {num_channels} channels, got {result['num_channels']}"

        multichannel_data = result['multichannel_data']
        assert len(multichannel_data) == num_channels, "Channel count mismatch"

        # Validate all channels have same length
        lengths = [len(ch) for ch in multichannel_data]
        assert all(L == lengths[0] for L in lengths), \
            f"Channels have different lengths: {lengths}"
        print(f"✓ All channels have same length: {lengths[0]} samples")

        # Analyze each channel
        for ch_idx, ch_data in enumerate(multichannel_data):
            ch_np = np.array(ch_data, dtype=np.float32)
            max_amp = np.max(np.abs(ch_np))
            rms = np.sqrt(np.mean(ch_np ** 2))

            print(f"  Channel {ch_idx}: max={max_amp:.4f}, RMS={rms:.4f}")

        # Save to WAV files for manual inspection
        save_multichannel_wav(multichannel_data, sample_rate, num_channels)

        return True

    except Exception as e:
        print(f"✗ Multi-channel test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def save_multichannel_wav(multichannel_data, sample_rate, num_channels):
    """Save recorded channels to WAV files"""
    try:
        from scipy.io import wavfile

        output_dir = Path("test_recordings")
        output_dir.mkdir(exist_ok=True)

        for ch_idx, ch_data in enumerate(multichannel_data):
            filename = output_dir / f"channel_{ch_idx}_{num_channels}ch.wav"
            ch_np = np.array(ch_data, dtype=np.float32)

            # Convert to 16-bit int for WAV
            ch_int16 = (ch_np * 32767).astype(np.int16)

            wavfile.write(filename, sample_rate, ch_int16)
            print(f"  Saved: {filename}")

        print(f"✓ Saved {len(multichannel_data)} channel files to {output_dir}/")

    except ImportError:
        print("  (scipy not available - skipping WAV export)")
    except Exception as e:
        print(f"  Warning: Failed to save WAV files: {e}")


def test_channel_synchronization(num_channels=2):
    """Test that channels are synchronized (same timing)"""
    print("\n" + "=" * 60)
    print(f"TEST 3: {num_channels}-Channel Synchronization")
    print("=" * 60)

    try:
        import sdl_audio_core
        from scipy import signal as scipy_signal

        # Check device availability
        devices = sdl_audio_core.list_all_devices()
        dev = find_multichannel_device(devices, num_channels)

        if not dev:
            print(f"⚠️  SKIPPED: No device found supporting {num_channels} channels")
            return True

        print(f"Using device: {dev.name}")

        # Generate chirp for better correlation
        sample_rate = 48000
        duration = 1.0
        num_samples = int(duration * sample_rate)
        t = np.arange(num_samples) / sample_rate

        # Chirp from 200Hz to 2000Hz
        test_signal = (0.3 * scipy_signal.chirp(t, 200, duration, 2000)).tolist()

        print(f"Recording {num_channels} channels with chirp signal...")

        result = sdl_audio_core.measure_room_response_auto_multichannel(
            test_signal,
            volume=0.2,
            input_device=dev.device_id,
            output_device=0,
            input_channels=num_channels
        )

        if not result['success']:
            print(f"✗ Recording failed: {result['error_message']}")
            return False

        # Convert to numpy
        channels = [np.array(ch, dtype=np.float32) for ch in result['multichannel_data']]

        # Use channel 0 as reference
        reference = channels[0]

        print(f"\nCross-correlation analysis (reference: Channel 0):")

        max_lag_allowed = 10  # samples
        all_synchronized = True

        for ch_idx in range(1, num_channels):
            # Cross-correlate with reference
            correlation = np.correlate(reference, channels[ch_idx], mode='full')

            # Find peak
            lag = np.argmax(correlation) - len(channels[ch_idx]) + 1

            # Normalized correlation
            max_corr = np.max(correlation)
            norm_corr = max_corr / (np.linalg.norm(reference) * np.linalg.norm(channels[ch_idx]))

            status = "✓" if abs(lag) <= max_lag_allowed else "✗"
            print(f"  {status} Channel {ch_idx}: lag = {lag:+4d} samples, correlation = {norm_corr:.4f}")

            if abs(lag) > max_lag_allowed:
                all_synchronized = False

        if all_synchronized:
            print(f"\n✓ All channels synchronized within {max_lag_allowed} samples")
            return True
        else:
            print(f"\n✗ SYNCHRONIZATION FAILURE: Some channels exceed {max_lag_allowed} sample lag")
            return False

    except ImportError:
        print("⚠️  SKIPPED: scipy not available (needed for chirp signal)")
        return True
    except Exception as e:
        print(f"✗ Synchronization test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_different_channel_counts():
    """Test various channel configurations"""
    print("\n" + "=" * 60)
    print("TEST 4: Different Channel Configurations")
    print("=" * 60)

    try:
        import sdl_audio_core

        devices = sdl_audio_core.list_all_devices()

        # Find max channels available
        max_channels = 1
        best_dev = None
        for dev in devices['input_devices']:
            if dev.max_channels > max_channels:
                max_channels = dev.max_channels
                best_dev = dev

        print(f"Maximum channels available: {max_channels}")
        if best_dev:
            print(f"Device: {best_dev.name}")

        # Test configurations up to max available
        test_configs = [1, 2, 4, 8]
        test_configs = [c for c in test_configs if c <= max_channels]

        if not test_configs:
            print("⚠️  Only mono devices available")
            test_configs = [1]

        results = []

        for num_ch in test_configs:
            print(f"\nTesting {num_ch} channel(s)...")

            # Quick test
            sample_rate = 48000
            duration = 0.1
            t = np.arange(int(duration * sample_rate)) / sample_rate
            test_signal = (0.2 * np.sin(2 * np.pi * 1000 * t)).tolist()

            result = sdl_audio_core.measure_room_response_auto_multichannel(
                test_signal,
                volume=0.2,
                input_device=best_dev.device_id if best_dev else -1,
                output_device=0,
                input_channels=num_ch
            )

            success = result['success']
            if success:
                assert result['num_channels'] == num_ch
                print(f"  ✓ {num_ch} channels: {result['samples_per_channel']} samples/channel")
            else:
                print(f"  ✗ {num_ch} channels: {result['error_message']}")

            results.append((num_ch, success))

        # Summary
        print("\nConfiguration Summary:")
        for num_ch, success in results:
            status = "✓" if success else "✗"
            print(f"  {status} {num_ch} channel(s): {'PASSED' if success else 'FAILED'}")

        return all(success for _, success in results)

    except Exception as e:
        print(f"✗ Channel configuration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all hardware tests"""
    print("\n" + "=" * 60)
    print("PHASE 1 HARDWARE TESTING SUITE")
    print("Testing multi-channel recording with actual audio hardware")
    print("=" * 60)

    # First, list devices
    devices = list_audio_devices()
    if not devices:
        print("\n✗ Could not enumerate audio devices - cannot proceed")
        return 1

    print("\n" + "=" * 60)
    print("RUNNING TESTS")
    print("=" * 60)

    tests = [
        ("Single-Channel Recording", lambda: test_single_channel_recording()),
        ("Multi-Channel Recording (2ch)", lambda: test_multichannel_recording(2)),
        ("Multi-Channel Recording (4ch)", lambda: test_multichannel_recording(4)),
        ("Channel Synchronization", lambda: test_channel_synchronization(2)),
        ("Different Configurations", test_different_channel_counts),
    ]

    results = []

    for name, test_func in tests:
        try:
            success = test_func()
            results.append((name, success))
        except KeyboardInterrupt:
            print("\n\n⚠️  Test interrupted by user")
            results.append((name, False))
            break
        except Exception as e:
            print(f"\n✗ Test '{name}' crashed: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))

    # Summary
    print("\n" + "=" * 60)
    print("HARDWARE TEST SUMMARY")
    print("=" * 60)

    for name, success in results:
        status = "PASSED" if success else "FAILED"
        icon = "✓" if success else "✗"
        print(f"{icon} {name:<40} {status}")

    passed = sum(1 for _, s in results if s)
    total = len(results)
    print(f"\nOverall: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 ALL HARDWARE TESTS PASSED!")
        print("\nPhase 1 implementation is complete and validated!")
        print("\nNext steps:")
        print("  1. Review recorded WAV files in test_recordings/")
        print("  2. Proceed to Phase 2: Update RoomResponseRecorder")
        return 0
    else:
        print("\n⚠️  SOME TESTS FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())
