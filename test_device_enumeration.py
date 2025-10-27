#!/usr/bin/env python3
"""
Diagnostic tool to check SDL audio device enumeration.
Run this with and without UMC1820 plugged in to compare.
"""

import sys
import io

# Fix Windows console encoding issues
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

try:
    import sdl_audio_core
    print("[OK] SDL Audio Core loaded successfully")
    print(f"Version: {sdl_audio_core.get_version()}")
    print(f"SDL Version: {sdl_audio_core.SDL_VERSION}")
    print()
except ImportError as e:
    print(f"[ERROR] Failed to import sdl_audio_core: {e}")
    sys.exit(1)

try:
    # Get all devices using list_all_devices
    devices = sdl_audio_core.list_all_devices()

    print("=" * 80)
    print("ALL DEVICES")
    print("=" * 80)

    input_devices = devices.get('input_devices', [])
    output_devices = devices.get('output_devices', [])

    print(f"\nFound {len(input_devices)} input devices and {len(output_devices)} output devices\n")

    # List all input devices
    print("[INPUT DEVICES]")
    print("-" * 80)
    for dev in input_devices:
        print(f"\nDevice ID: {dev.device_id}")
        print(f"  Name: {dev.name}")
        print(f"  Is Input: {dev.is_input}")
        print(f"  Max Channels: {dev.max_channels}")

    # List all output devices
    print("\n\n[OUTPUT DEVICES]")
    print("-" * 80)
    for dev in output_devices:
        print(f"\nDevice ID: {dev.device_id}")
        print(f"  Name: {dev.name}")
        print(f"  Is Input: {dev.is_input}")
        print(f"  Max Channels: {dev.max_channels}")

    print("\n" + "=" * 80)
    print("\n[TEST 1] Try opening DEFAULT device with 1 channel")
    print("-" * 80)

    try:
        engine = sdl_audio_core.AudioEngine()
        config = sdl_audio_core.AudioEngineConfig()
        config.sample_rate = 48000
        config.input_channels = 1
        config.output_channels = 1
        config.enable_logging = True
        # Leave device IDs as default (-1)

        print("Attempting to initialize with default devices (ID -1)...")
        if engine.initialize(config):
            print("[SUCCESS] Engine initialized with default devices")
            engine.shutdown()
        else:
            print("[FAILED] Could not initialize engine with default devices")
    except Exception as e:
        print(f"[EXCEPTION] {e}")

    if len(input_devices) > 0:
        print("\n" + "=" * 80)
        print(f"\n[TEST 2] Try opening FIRST input device (ID {input_devices[0].device_id}) with 1 channel")
        print("-" * 80)

        try:
            engine = sdl_audio_core.AudioEngine()
            config = sdl_audio_core.AudioEngineConfig()
            config.sample_rate = 48000
            config.input_channels = 1
            config.output_channels = 1
            config.input_device_id = input_devices[0].device_id
            if len(output_devices) > 0:
                config.output_device_id = output_devices[0].device_id
            config.enable_logging = True

            print(f"Attempting to initialize with Input Device ID {input_devices[0].device_id}...")
            if engine.initialize(config):
                print(f"[SUCCESS] Engine initialized with Device {input_devices[0].device_id}")
                engine.shutdown()
            else:
                print(f"[FAILED] Could not initialize engine with Device {input_devices[0].device_id}")
        except Exception as e:
            print(f"[EXCEPTION] {e}")

    if len(input_devices) > 1:
        print("\n" + "=" * 80)
        print(f"\n[TEST 3] Try opening SECOND input device (ID {input_devices[1].device_id}) with 1 channel")
        print("-" * 80)

        try:
            engine = sdl_audio_core.AudioEngine()
            config = sdl_audio_core.AudioEngineConfig()
            config.sample_rate = 48000
            config.input_channels = 1
            config.output_channels = 1
            config.input_device_id = input_devices[1].device_id
            if len(output_devices) > 0:
                config.output_device_id = output_devices[0].device_id
            config.enable_logging = True

            print(f"Attempting to initialize with Input Device ID {input_devices[1].device_id}...")
            if engine.initialize(config):
                print(f"[SUCCESS] Engine initialized with Device {input_devices[1].device_id}")
                engine.shutdown()
            else:
                print(f"[FAILED] Could not initialize engine with Device {input_devices[1].device_id}")
        except Exception as e:
            print(f"[EXCEPTION] {e}")

except Exception as e:
    print(f"[FATAL ERROR] {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 80)
print("Diagnostic complete!")
print("=" * 80)
