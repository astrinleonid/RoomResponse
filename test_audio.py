#!/usr/bin/env python3
"""
SDL Audio Core - Test Script
Tests the SDL audio module functionality
"""

import numpy as np
import time
import matplotlib.pyplot as plt
from datetime import datetime

try:
    import sdl_audio_core

    print(f"✓ SDL Audio Core loaded successfully")
    print(f"  Version: {sdl_audio_core.__version__}")
    print(f"  SDL Version: {sdl_audio_core.SDL_VERSION}")
    print(f"  Available classes: {[x for x in dir(sdl_audio_core) if not x.startswith('_')]}")
except ImportError as e:
    print(f"✗ Failed to import SDL Audio Core: {e}")
    print("\nTroubleshooting:")
    print("1. Make sure you've built the module with build_sdl_audio.bat")
    print("2. Check that you're in the correct virtual environment")
    print("3. Try: python -c \"import sdl_audio_core\"")
    exit(1)


def test_device_enumeration():
    """Test device enumeration functionality"""
    print("\n" + "=" * 50)
    print("DEVICE ENUMERATION TEST")
    print("=" * 50)

    try:
        # Test standalone device listing
        devices = sdl_audio_core.list_all_devices()

        print(f"\nFound {len(devices['input_devices'])} input devices:")
        for i, device in enumerate(devices['input_devices']):
            print(f"  [{device.device_id}] {device.name}")

        print(f"\nFound {len(devices['output_devices'])} output devices:")
        for i, device in enumerate(devices['output_devices']):
            print(f"  [{device.device_id}] {device.name}")

        return True

    except Exception as e:
        print(f"✗ Device enumeration failed: {e}")
        return False


def test_audio_engine():
    """Test basic AudioEngine functionality"""
    print("\n" + "=" * 50)
    print("AUDIO ENGINE TEST")
    print("=" * 50)

    try:
        # Create audio engine
        engine = sdl_audio_core.AudioEngine()
        print("✓ AudioEngine created")

        # Initialize with default settings
        config = sdl_audio_core.AudioEngineConfig()
        config.sample_rate = 48000
        config.buffer_size = 1024
        config.enable_logging = True

        if not engine.initialize(config):
            print("✗ Failed to initialize AudioEngine")
            return False
        print("✓ AudioEngine initialized")

        # Get device lists
        input_devices = engine.get_input_devices()
        output_devices = engine.get_output_devices()

        print(f"✓ Found {len(input_devices)} input and {len(output_devices)} output devices")

        # Test device combination if we have devices
        if input_devices and output_devices:
            input_id = input_devices[0].device_id
            output_id = output_devices[0].device_id

            print(f"Testing device combination: input {input_id}, output {output_id}")
            can_combine = engine.test_device_combination(input_id, output_id)
            print(f"✓ Device combination test: {'PASS' if can_combine else 'FAIL'}")

        # Get statistics
        stats = engine.get_stats()
        print(f"✓ Engine stats: {stats.input_samples_processed} input samples processed")

        # Shutdown
        engine.shutdown()
        print("✓ AudioEngine shutdown complete")

        return True

    except Exception as e:
        print(f"✗ AudioEngine test failed: {e}")
        return False


def test_device_testing():
    """Test device pair testing functionality"""
    print("\n" + "=" * 50)
    print("DEVICE PAIR TESTING")
    print("=" * 50)

    try:
        # Create room response measurer
        measurer = sdl_audio_core.RoomResponseMeasurer()

        config = sdl_audio_core.RoomResponseConfig()
        config.sample_rate = 48000
        config.num_pulses = 3  # Short test
        config.volume = 0.1  # Low volume for testing

        if not measurer.initialize(config):
            print("✗ Failed to initialize RoomResponseMeasurer")
            return False
        print("✓ RoomResponseMeasurer initialized")

        # Get devices
        input_devices = measurer.get_input_devices()
        output_devices = measurer.get_output_devices()

        if not input_devices or not output_devices:
            print("✗ No audio devices available for testing")
            return False

        # Test the first pair
        input_id = input_devices[0].device_id
        output_id = output_devices[0].device_id

        print(f"\nTesting device pair:")
        print(f"  Input:  [{input_id}] {input_devices[0].name}")
        print(f"  Output: [{output_id}] {output_devices[0].name}")

        # Quick device test
        result = measurer.test_device_pair(input_id, output_id)

        print(f"\nTest results:")
        print(f"  Can open devices: {'✓' if result.can_open_devices else '✗'}")
        print(f"  Acoustic coupling: {'✓' if result.has_acoustic_coupling else '✗'}")
        print(f"  Coupling strength: {result.coupling_strength:.6f}")

        if result.error_message:
            print(f"  Error: {result.error_message}")

        # Find best device pair
        print(f"\nSearching for best device pair...")
        best_pair = measurer.find_best_device_pair()

        if best_pair.can_open_devices:
            print(f"Best pair found:")
            print(f"  Input:  [{best_pair.input_device_id}] {best_pair.input_device_name}")
            print(f"  Output: [{best_pair.output_device_id}] {best_pair.output_device_name}")
            print(f"  Coupling: {'✓' if best_pair.has_acoustic_coupling else '✗'} ({best_pair.coupling_strength:.6f})")
        else:
            print("✗ No working device pairs found")

        measurer.shutdown()
        print("✓ Device testing complete")

        return True

    except Exception as e:
        print(f"✗ Device testing failed: {e}")
        return False


def test_signal_processing():
    """Test signal processing utilities"""
    print("\n" + "=" * 50)
    print("SIGNAL PROCESSING TEST")
    print("=" * 50)

    try:
        # Generate test signals
        sample_rate = 48000
        duration = 1.0
        frequency = 1000.0

        # Generate sine wave
        sine_wave = sdl_audio_core.signal_processing.generate_sine_wave(
            frequency, duration, sample_rate, 0.5)
        print(f"✓ Generated sine wave: {len(sine_wave)} samples")

        # Generate white noise
        noise = sdl_audio_core.signal_processing.generate_white_noise(
            duration, sample_rate, 0.1)
        print(f"✓ Generated white noise: {len(noise)} samples")

        # Apply window
        windowed = sdl_audio_core.signal_processing.apply_window(sine_wave, "hann")
        print(f"✓ Applied Hann window")

        # Calculate RMS
        rms = sdl_audio_core.signal_processing.calculate_rms(sine_wave)
        print(f"✓ RMS of sine wave: {rms:.6f}")

        # Calculate peak
        peak = sdl_audio_core.signal_processing.calculate_peak(sine_wave)
        print(f"✓ Peak of sine wave: {peak:.6f}")

        # Cross-correlation
        correlation = sdl_audio_core.signal_processing.cross_correlate(sine_wave[:1000], sine_wave[:1000])
        max_lag = sdl_audio_core.signal_processing.find_max_correlation_lag(correlation)
        print(f"✓ Cross-correlation max lag: {max_lag}")

        # Test normalization
        normalized = sdl_audio_core.RoomResponseMeasurer.normalize_signal(sine_wave)
        print(f"✓ Normalized signal, max value: {np.max(np.abs(normalized)):.6f}")

        return True

    except Exception as e:
        print(f"✗ Signal processing test failed: {e}")
        return False


def test_room_response_measurement():
    """Test actual room response measurement (if devices support it)"""
    print("\n" + "=" * 50)
    print("ROOM RESPONSE MEASUREMENT TEST")
    print("=" * 50)

    try:
        # Create measurer
        measurer = sdl_audio_core.RoomResponseMeasurer()

        # Configure for quick test
        config = sdl_audio_core.RoomResponseConfig()
        config.sample_rate = 22050  # Lower sample rate for speed
        config.pulse_duration_ms = 20
        config.cycle_duration_ms = 200
        config.num_pulses = 3
        config.volume = 0.2
        config.averaging_start_pulse = 1

        if not measurer.initialize(config):
            print("✗ Failed to initialize measurer")
            return False

        # Find best device pair
        best_pair = measurer.find_best_device_pair()

        if not best_pair.can_open_devices:
            print("✗ No suitable device pair found for measurement")
            print("  This is normal if you don't have devices with acoustic coupling")
            return True  # Not a failure, just no suitable devices

        if not best_pair.has_acoustic_coupling:
            print("⚠ Device pair can open but has no acoustic coupling")
            print("  This means the microphone can't hear the speakers")
            print("  For real measurements, you need external speakers/microphone")
            return True

        print(f"Using device pair with acoustic coupling:")
        print(f"  Input:  {best_pair.input_device_name}")
        print(f"  Output: {best_pair.output_device_name}")
        print(f"  Coupling strength: {best_pair.coupling_strength:.6f}")

        # Ask user permission for actual measurement
        print(f"\n⚠ About to play audio for room response measurement!")
        print(f"Make sure your volume is at a reasonable level.")
        response = input("Proceed with measurement? (y/N): ").lower()

        if response != 'y':
            print("Measurement skipped by user")
            return True

        # Perform measurement
        print("Starting measurement...")
        start_time = time.time()

        result = measurer.measure_room_response_with_devices(
            best_pair.input_device_id, best_pair.output_device_id)

        elapsed_time = time.time() - start_time

        if not result.success:
            print(f"✗ Measurement failed: {result.error_message}")
            return False

        print(f"✓ Measurement completed in {elapsed_time:.1f} seconds")
        print(f"  Sample rate: {result.sample_rate} Hz")
        print(f"  Pulses recorded: {result.actual_pulses_recorded}")
        print(f"  Max amplitude: {result.max_amplitude:.6f}")
        print(f"  RMS level: {result.rms_level:.6f}")
        print(f"  SNR: {result.signal_to_noise_ratio:.1f} dB")
        print(f"  Onset sample: {result.onset_sample}")

        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        raw_filename = f"raw_recording_{timestamp}.wav"
        sdl_audio_core.RoomResponseMeasurer.save_wav_file(
            result.raw_recording, result.sample_rate, raw_filename)
        print(f"✓ Raw recording saved to: {raw_filename}")

        if len(result.room_response) > 0:
            response_filename = f"room_response_{timestamp}.wav"
            sdl_audio_core.RoomResponseMeasurer.save_wav_file(
                result.room_response, result.sample_rate, response_filename)
            print(f"✓ Room response saved to: {response_filename}")

        if len(result.impulse_response) > 0:
            impulse_filename = f"impulse_response_{timestamp}.wav"
            sdl_audio_core.RoomResponseMeasurer.save_wav_file(
                result.impulse_response, result.sample_rate, impulse_filename)
            print(f"✓ Impulse response saved to: {impulse_filename}")

        # Plot results if matplotlib is available
        try:
            plt.figure(figsize=(12, 8))

            # Plot raw recording
            plt.subplot(3, 1, 1)
            time_axis = np.arange(len(result.raw_recording)) / result.sample_rate
            plt.plot(time_axis, result.raw_recording)
            plt.title('Raw Recording')
            plt.xlabel('Time (s)')
            plt.ylabel('Amplitude')

            # Plot room response
            if len(result.room_response) > 0:
                plt.subplot(3, 1, 2)
                time_axis = np.arange(len(result.room_response)) / result.sample_rate * 1000
                plt.plot(time_axis, result.room_response)
                plt.title('Averaged Room Response')
                plt.xlabel('Time (ms)')
                plt.ylabel('Amplitude')

            # Plot impulse response
            if len(result.impulse_response) > 0:
                plt.subplot(3, 1, 3)
                time_axis = np.arange(len(result.impulse_response)) / result.sample_rate * 1000
                plt.plot(time_axis, result.impulse_response)
                plt.title('Time-Aligned Impulse Response')
                plt.xlabel('Time (ms)')
                plt.ylabel('Amplitude')
                if result.onset_sample >= 0:
                    plt.axvline(x=0, color='r', linestyle='--', label='Onset')
                    plt.legend()

            plt.tight_layout()
            plot_filename = f"room_response_plot_{timestamp}.png"
            plt.savefig(plot_filename)
            print(f"✓ Plot saved to: {plot_filename}")

            # Show plot if running interactively
            if hasattr(plt, 'show'):
                plt.show()

        except Exception as plot_error:
            print(f"Note: Could not create plot: {plot_error}")

        measurer.shutdown()
        return True

    except Exception as e:
        print(f"✗ Room response measurement failed: {e}")
        return False


def main():
    """Run all tests"""
    print("SDL Audio Core - Comprehensive Test Suite")
    print("=" * 60)

    tests = [
        ("Device Enumeration", test_device_enumeration),
        ("Audio Engine", test_audio_engine),
        ("Device Testing", test_device_testing),
        ("Signal Processing", test_signal_processing),
        ("Room Response Measurement", test_room_response_measurement)
    ]

    results = []

    for test_name, test_func in tests:
        print(f"\nRunning {test_name}...")
        try:
            success = test_func()
            results.append((test_name, success))
            print(f"{'✓' if success else '✗'} {test_name}: {'PASSED' if success else 'FAILED'}")
        except Exception as e:
            print(f"✗ {test_name}: EXCEPTION - {e}")
            results.append((test_name, False))

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    passed = sum(1 for _, success in results if success)
    total = len(results)

    for test_name, success in results:
        status = "PASSED" if success else "FAILED"
        icon = "✓" if success else "✗"
        print(f"{icon} {test_name:<30} {status}")

    print(f"\nOverall: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All tests passed! SDL Audio Core is working correctly.")
    else:
        print("⚠ Some tests failed. Check the output above for details.")

    return passed == total


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)