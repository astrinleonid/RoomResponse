"""
SDL Audio Core - Low-level audio module for simultaneous playback and recording

This package provides C++ SDL2-based audio functionality with Python bindings
for precise audio device control, room impulse response measurement, and
professional audio applications.

Example usage:
    import sdl_audio_core

    # List all audio devices
    devices = sdl_audio_core.list_all_devices()

    # Test device compatibility
    result = sdl_audio_core.quick_device_test(input_id=0, output_id=0)

    # Measure room response
    measurer = sdl_audio_core.RoomResponseMeasurer()
    config = sdl_audio_core.RoomResponseConfig()
    measurer.initialize(config)

    # Find best device pair automatically
    best_pair = measurer.find_best_device_pair()

    # Perform measurement
    result = measurer.measure_room_response_with_devices(
        best_pair.input_device_id, best_pair.output_device_id
    )
"""

# Import the compiled module
try:
    from .sdl_audio_core import *

    _module_loaded = True
except ImportError as e:
    _module_loaded = False
    _import_error = str(e)

# Version information
__version__ = "0.1.0"
__author__ = "SDL Audio Core Team"
__license__ = "MIT"


def get_build_info():
    """Get build and system information"""
    if not _module_loaded:
        return {
            "module_loaded": False,
            "import_error": _import_error,
            "python_version": None,
            "sdl_version": None
        }

    import sys

    info = {
        "module_loaded": True,
        "version": __version__,
        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "sdl_version": SDL_VERSION if 'SDL_VERSION' in globals() else "Unknown"
    }

    return info


def check_installation():
    """Check if the module is properly installed and working"""
    if not _module_loaded:
        print(f"❌ SDL Audio Core not properly installed")
        print(f"Import error: {_import_error}")
        print(f"\nTroubleshooting steps:")
        print(f"1. Make sure you've run build_sdl_audio.bat successfully")
        print(f"2. Check that SDL2 is properly installed")
        print(f"3. Verify you're in the correct virtual environment")
        print(f"4. Try rebuilding with: python -m pip install -e . --force-reinstall")
        return False

    try:
        # Test basic functionality
        devices = list_all_devices()

        print(f"✅ SDL Audio Core {__version__} installed correctly")
        print(f"SDL Version: {SDL_VERSION}")
        print(f"Found {len(devices['input_devices'])} input and {len(devices['output_devices'])} output devices")

        return True

    except Exception as e:
        print(f"❌ SDL Audio Core installed but not working properly")
        print(f"Error: {e}")
        return False


# Module-level convenience functions
def quick_room_response_test(volume=0.1, duration_ms=500):
    """
    Quick room response test using best available device pair

    Args:
        volume: Playback volume (0.0 - 1.0)
        duration_ms: Test duration in milliseconds

    Returns:
        MeasurementResult or None if no suitable devices
    """
    if not _module_loaded:
        raise ImportError("SDL Audio Core not properly loaded")

    measurer = RoomResponseMeasurer()

    # Quick test configuration
    config = RoomResponseConfig()
    config.volume = volume
    config.num_pulses = max(1, duration_ms // 100)
    config.cycle_duration_ms = 100
    config.pulse_duration_ms = 10

    if not measurer.initialize(config):
        return None

    # Find best device pair
    best_pair = measurer.find_best_device_pair()

    if not best_pair.has_acoustic_coupling:
        measurer.shutdown()
        return None

    # Perform measurement
    result = measurer.measure_room_response_with_devices(
        best_pair.input_device_id, best_pair.output_device_id
    )

    measurer.shutdown()
    return result


def get_recommended_device_pairs(max_pairs=3):
    """
    Get recommended device pairs for room response measurement

    Args:
        max_pairs: Maximum number of pairs to return

    Returns:
        List of DeviceTestResult objects, sorted by quality
    """
    if not _module_loaded:
        raise ImportError("SDL Audio Core not properly loaded")

    measurer = RoomResponseMeasurer()
    if not measurer.initialize():
        return []

    all_results = measurer.test_all_device_combinations()

    # Filter to working pairs with coupling
    good_pairs = [r for r in all_results if r.can_open_devices and r.has_acoustic_coupling]

    measurer.shutdown()

    return good_pairs[:max_pairs]


# Export convenience functions
__all__ = [
    # Core classes (imported from C++ module)
    'AudioEngine',
    'AudioEngineConfig',
    'AudioEngineState',
    'AudioEngineStats',
    'AudioDevice',
    'RoomResponseMeasurer',
    'RoomResponseConfig',
    'MeasurementResult',
    'DeviceTestResult',

    # Module functions
    'list_all_devices',
    'quick_device_test',
    'get_version',

    # Package functions
    'get_build_info',
    'check_installation',
    'quick_room_response_test',
    'get_recommended_device_pairs',

    # Signal processing submodule
    'signal_processing',

    # Version info
    '__version__'
]

# Only add C++ module exports if successfully imported
if _module_loaded:
    # The actual exports are handled by the C++ module
    pass
else:
    # If module failed to load, provide stub functions
    def _not_loaded(*args, **kwargs):
        raise ImportError(f"SDL Audio Core not loaded: {_import_error}")


    # Replace functions with error stubs
    list_all_devices = _not_loaded
    quick_device_test = _not_loaded
    AudioEngine = _not_loaded
    RoomResponseMeasurer = _not_loaded