#!/usr/bin/env python3
"""
Test multi-channel initialization with UMC1820
"""

import sys
import io

# Fix Windows console encoding
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import sdl_audio_core

print("Testing multi-channel device initialization")
print("=" * 80)

devices = sdl_audio_core.list_all_devices()
input_devices = devices.get('input_devices', [])

print(f"\nAvailable input devices: {len(input_devices)}")
for dev in input_devices:
    print(f"  ID {dev.device_id}: {dev.name} ({dev.max_channels} channels)")

print("\n" + "=" * 80)

# Test 1: Open Device 0 (webcam) with 1 channel
print("\n[TEST 1] Device 0 with 1 channel:")
try:
    engine = sdl_audio_core.AudioEngine()
    config = sdl_audio_core.AudioEngineConfig()
    config.sample_rate = 48000
    config.input_channels = 1
    config.output_channels = 1
    config.input_device_id = 0
    config.output_device_id = 0
    config.enable_logging = True

    if engine.initialize(config):
        print("[SUCCESS]")
        engine.shutdown()
    else:
        print("[FAILED]")
except Exception as e:
    print(f"[EXCEPTION] {e}")

# Test 2: Open Device 1 (UMC) with 1 channel
if len(input_devices) > 1:
    print("\n[TEST 2] Device 1 (UMC) with 1 channel:")
    try:
        engine = sdl_audio_core.AudioEngine()
        config = sdl_audio_core.AudioEngineConfig()
        config.sample_rate = 48000
        config.input_channels = 1
        config.output_channels = 1
        config.input_device_id = 1
        config.output_device_id = 0
        config.enable_logging = True

        if engine.initialize(config):
            print("[SUCCESS]")
            engine.shutdown()
        else:
            print("[FAILED]")
    except Exception as e:
        print(f"[EXCEPTION] {e}")

    # Test 3: Open Device 1 (UMC) with 8 channels
    print("\n[TEST 3] Device 1 (UMC) with 8 channels:")
    try:
        engine = sdl_audio_core.AudioEngine()
        config = sdl_audio_core.AudioEngineConfig()
        config.sample_rate = 48000
        config.input_channels = 8
        config.output_channels = 1
        config.input_device_id = 1
        config.output_device_id = 0
        config.enable_logging = True

        if engine.initialize(config):
            print("[SUCCESS]")
            engine.shutdown()
        else:
            print("[FAILED]")
    except Exception as e:
        print(f"[EXCEPTION] {e}")

    # Test 4: Open Device 1 (UMC) with 10 channels (max)
    print("\n[TEST 4] Device 1 (UMC) with 10 channels (max):")
    try:
        engine = sdl_audio_core.AudioEngine()
        config = sdl_audio_core.AudioEngineConfig()
        config.sample_rate = 48000
        config.input_channels = 10
        config.output_channels = 1
        config.input_device_id = 1
        config.output_device_id = 0
        config.enable_logging = True

        if engine.initialize(config):
            print("[SUCCESS]")
            engine.shutdown()
        else:
            print("[FAILED]")
    except Exception as e:
        print(f"[EXCEPTION] {e}")

    # Test 5: CRITICAL - Open Device 0 (webcam) with 8 channels (should fail gracefully)
    print("\n[TEST 5] Device 0 (webcam, 1ch max) with 8 channels (EXPECT FAILURE):")
    try:
        engine = sdl_audio_core.AudioEngine()
        config = sdl_audio_core.AudioEngineConfig()
        config.sample_rate = 48000
        config.input_channels = 8
        config.output_channels = 1
        config.input_device_id = 0
        config.output_device_id = 0
        config.enable_logging = True

        if engine.initialize(config):
            print("[UNEXPECTED SUCCESS - This shouldn't work!]")
            engine.shutdown()
        else:
            print("[EXPECTED FAILURE - Webcam doesn't support 8 channels]")
    except Exception as e:
        print(f"[EXPECTED EXCEPTION] {e}")

print("\n" + "=" * 80)
print("Multi-channel testing complete!")
