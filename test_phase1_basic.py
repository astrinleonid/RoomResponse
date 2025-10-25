#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 1 Basic Testing Suite
Tests multi-channel API without hardware (import and API validation)
"""

import sys
import os

# Fix Unicode output on Windows
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

def test_import():
    """Test that sdl_audio_core imports successfully"""
    print("=" * 60)
    print("TEST 1: Import Module")
    print("=" * 60)

    try:
        import sdl_audio_core
        print("✓ Module imported successfully")
        print(f"  Version: {sdl_audio_core.__version__}")
        return True
    except ImportError as e:
        print(f"✗ Failed to import: {e}")
        return False


def test_config_api():
    """Test that Config has new multi-channel fields"""
    print("\n" + "=" * 60)
    print("TEST 2: Config API")
    print("=" * 60)

    try:
        import sdl_audio_core

        config = sdl_audio_core.AudioEngineConfig()

        # Check default values
        assert hasattr(config, 'input_channels'), "Missing input_channels attribute"
        assert hasattr(config, 'output_channels'), "Missing output_channels attribute"
        print("✓ Config has input_channels and output_channels attributes")

        assert config.input_channels == 1, f"Default input_channels should be 1, got {config.input_channels}"
        assert config.output_channels == 1, f"Default output_channels should be 1, got {config.output_channels}"
        print("✓ Default channel counts are 1 (backward compatible)")

        # Test setting values
        config.input_channels = 4
        config.output_channels = 2
        assert config.input_channels == 4
        assert config.output_channels == 2
        print("✓ Can set channel counts to custom values")

        # Test repr
        repr_str = repr(config)
        assert 'input_channels=4' in repr_str, f"repr should show input_channels: {repr_str}"
        assert 'output_channels=2' in repr_str, f"repr should show output_channels: {repr_str}"
        print(f"✓ Config repr shows channel counts: {repr_str}")

        return True
    except Exception as e:
        print(f"✗ Config API test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_stats_api():
    """Test that Stats has new multi-channel fields"""
    print("\n" + "=" * 60)
    print("TEST 3: Stats API")
    print("=" * 60)

    try:
        import sdl_audio_core

        engine = sdl_audio_core.AudioEngine()
        config = sdl_audio_core.AudioEngineConfig()
        config.input_channels = 2

        if not engine.initialize(config):
            print("✗ Failed to initialize engine")
            return False

        stats = engine.get_stats()

        # Check new fields exist
        assert hasattr(stats, 'num_input_channels'), "Missing num_input_channels"
        assert hasattr(stats, 'num_output_channels'), "Missing num_output_channels"
        assert hasattr(stats, 'channel_buffer_sizes'), "Missing channel_buffer_sizes"
        print("✓ Stats has multi-channel fields")

        # Check values
        assert stats.num_input_channels == 2, f"Expected 2 input channels, got {stats.num_input_channels}"
        assert stats.num_output_channels == 1, f"Expected 1 output channel, got {stats.num_output_channels}"
        print(f"✓ Stats reports correct channel counts: {stats.num_input_channels} in, {stats.num_output_channels} out")

        assert isinstance(stats.channel_buffer_sizes, list), "channel_buffer_sizes should be a list"
        assert len(stats.channel_buffer_sizes) == 2, f"Expected 2 channel buffer sizes, got {len(stats.channel_buffer_sizes)}"
        print(f"✓ Stats.channel_buffer_sizes is list with {len(stats.channel_buffer_sizes)} elements")

        engine.shutdown()
        return True
    except Exception as e:
        print(f"✗ Stats API test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_engine_methods():
    """Test that AudioEngine has new multi-channel methods"""
    print("\n" + "=" * 60)
    print("TEST 4: AudioEngine Methods")
    print("=" * 60)

    try:
        import sdl_audio_core

        engine = sdl_audio_core.AudioEngine()

        # Check methods exist
        assert hasattr(engine, 'get_recorded_data_multichannel'), "Missing get_recorded_data_multichannel"
        assert hasattr(engine, 'get_recorded_data_channel'), "Missing get_recorded_data_channel"
        assert hasattr(engine, 'get_num_input_channels'), "Missing get_num_input_channels"
        assert hasattr(engine, 'get_num_output_channels'), "Missing get_num_output_channels"
        print("✓ AudioEngine has all new multi-channel methods")

        # Test get_num_channels before initialization
        config = sdl_audio_core.AudioEngineConfig()
        config.input_channels = 4
        config.output_channels = 1

        engine.initialize(config)

        assert engine.get_num_input_channels() == 4, "get_num_input_channels() should return 4"
        assert engine.get_num_output_channels() == 1, "get_num_output_channels() should return 1"
        print("✓ get_num_input/output_channels() return correct values")

        # Test get_recorded_data_multichannel (should return empty list of lists)
        multichannel_data = engine.get_recorded_data_multichannel()
        assert isinstance(multichannel_data, list), "get_recorded_data_multichannel should return list"
        assert len(multichannel_data) == 4, f"Should have 4 channels, got {len(multichannel_data)}"
        print(f"✓ get_recorded_data_multichannel() returns list with {len(multichannel_data)} channels")

        # Test get_recorded_data_channel
        for ch in range(4):
            channel_data = engine.get_recorded_data_channel(ch)
            assert isinstance(channel_data, list), f"Channel {ch} data should be list"
            assert len(channel_data) == 0, f"Channel {ch} should be empty initially"
        print("✓ get_recorded_data_channel() works for all channels")

        # Test backward compatibility: get_recorded_data() should still work
        data = engine.get_recorded_data()
        assert isinstance(data, list), "get_recorded_data() should still return list"
        print("✓ get_recorded_data() still works (backward compatible)")

        engine.shutdown()
        return True
    except Exception as e:
        print(f"✗ Engine methods test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_convenience_function():
    """Test that measure_room_response_auto_multichannel exists"""
    print("\n" + "=" * 60)
    print("TEST 5: Convenience Function")
    print("=" * 60)

    try:
        import sdl_audio_core

        # Check function exists
        assert hasattr(sdl_audio_core, 'measure_room_response_auto_multichannel'), \
            "Missing measure_room_response_auto_multichannel function"
        print("✓ measure_room_response_auto_multichannel() exists")

        # Check function is callable
        assert callable(sdl_audio_core.measure_room_response_auto_multichannel), \
            "Function should be callable"
        print("✓ Function is callable")

        # Note: pybind11 functions don't expose Python-style signatures via inspect
        # This is expected behavior and not a bug
        print("✓ Function ready for use (pybind11 binding)")

        return True
    except Exception as e:
        print(f"✗ Convenience function test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_channel_validation():
    """Test that invalid channel counts are rejected"""
    print("\n" + "=" * 60)
    print("TEST 6: Channel Count Validation")
    print("=" * 60)

    try:
        import sdl_audio_core

        # Test invalid channel counts
        invalid_counts = [0, -1, 33, 100]

        for count in invalid_counts:
            engine = sdl_audio_core.AudioEngine()
            config = sdl_audio_core.AudioEngineConfig()
            config.input_channels = count

            result = engine.initialize(config)
            assert not result, f"Should reject input_channels={count}"
            engine.shutdown()

        print(f"✓ Correctly rejected invalid channel counts: {invalid_counts}")

        # Test valid channel counts
        valid_counts = [1, 2, 4, 8, 16, 32]

        for count in valid_counts:
            engine = sdl_audio_core.AudioEngine()
            config = sdl_audio_core.AudioEngineConfig()
            config.input_channels = count

            result = engine.initialize(config)
            assert result, f"Should accept input_channels={count}"
            assert engine.get_num_input_channels() == count
            engine.shutdown()

        print(f"✓ Correctly accepted valid channel counts: {valid_counts}")

        return True
    except Exception as e:
        print(f"✗ Channel validation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_backward_compatibility():
    """Test that existing single-channel code still works"""
    print("\n" + "=" * 60)
    print("TEST 7: Backward Compatibility")
    print("=" * 60)

    try:
        import sdl_audio_core

        # Old-style usage: don't specify input_channels
        engine = sdl_audio_core.AudioEngine()
        config = sdl_audio_core.AudioEngineConfig()
        # Don't set input_channels - should default to 1

        assert engine.initialize(config), "Should initialize with default config"

        stats = engine.get_stats()
        assert stats.num_input_channels == 1, "Should default to 1 input channel"
        print("✓ Defaults to mono when input_channels not specified")

        # Old-style method should still work
        data = engine.get_recorded_data()
        assert isinstance(data, list), "get_recorded_data() should return list"
        print("✓ get_recorded_data() returns list (not dict)")

        engine.shutdown()

        # Test old convenience function still works
        assert hasattr(sdl_audio_core, 'measure_room_response_auto'), \
            "Old measure_room_response_auto should still exist"
        print("✓ measure_room_response_auto() still exists")

        return True
    except Exception as e:
        print(f"✗ Backward compatibility test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all basic tests"""
    print("\n" + "=" * 60)
    print("PHASE 1 BASIC TESTING SUITE")
    print("Testing multi-channel API without hardware")
    print("=" * 60)

    tests = [
        ("Import Module", test_import),
        ("Config API", test_config_api),
        ("Stats API", test_stats_api),
        ("AudioEngine Methods", test_engine_methods),
        ("Convenience Function", test_convenience_function),
        ("Channel Validation", test_channel_validation),
        ("Backward Compatibility", test_backward_compatibility),
    ]

    results = []

    for name, test_func in tests:
        try:
            success = test_func()
            results.append((name, success))
        except Exception as e:
            print(f"\n✗ Test '{name}' crashed: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    for name, success in results:
        status = "PASSED" if success else "FAILED"
        icon = "✓" if success else "✗"
        print(f"{icon} {name:<30} {status}")

    passed = sum(1 for _, s in results if s)
    total = len(results)
    print(f"\nOverall: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 ALL BASIC TESTS PASSED!")
        print("\nNext step: Run hardware tests with actual audio devices")
        print("  python test_phase1_hardware.py")
        return 0
    else:
        print("\n⚠️  SOME TESTS FAILED")
        print("\nPlease fix the failing tests before proceeding to hardware tests")
        return 1


if __name__ == "__main__":
    sys.exit(main())
