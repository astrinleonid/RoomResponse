import sdl_audio_core
import inspect


def explore_classes():
    """Explore the SDL classes in detail"""
    print("SDL Audio Core API Explorer")
    print("=" * 60)

    classes_to_explore = ['AudioDevice', 'AudioEngine', 'AudioEngineConfig', 'AudioEngineState']

    for class_name in classes_to_explore:
        if hasattr(sdl_audio_core, class_name):
            cls = getattr(sdl_audio_core, class_name)
            print(f"\n{class_name}:")
            print("-" * 40)

            # Try to create an instance to see what methods are available
            try:
                if class_name == 'AudioEngineState':
                    # This appears to be an enum, list its members
                    print("  Enum members:")
                    for attr in dir(cls):
                        if not attr.startswith('_'):
                            value = getattr(cls, attr)
                            print(f"    {attr}: {value}")
                else:
                    # Try to instantiate
                    instance = cls()
                    print(f"  Successfully created {class_name} instance")

                    # List methods and properties
                    print("  Methods and properties:")
                    for attr in dir(instance):
                        if not attr.startswith('_'):
                            try:
                                value = getattr(instance, attr)
                                if callable(value):
                                    print(f"    {attr}() - method")
                                else:
                                    print(f"    {attr}: {type(value).__name__} = {value}")
                            except Exception as e:
                                print(f"    {attr}: Error accessing - {e}")

            except Exception as e:
                print(f"  Could not create instance: {e}")


def test_quick_response():
    """Test the quick_room_response_test function"""
    print(f"\n{'=' * 60}")
    print("Testing quick_room_response_test function")
    print(f"{'=' * 60}")

    try:
        # Test with minimal parameters
        print("Testing with volume=0.1, duration_ms=200...")
        result = sdl_audio_core.quick_room_response_test(volume=0.1, duration_ms=200)

        if result is None:
            print("Result is None - no suitable devices or measurement failed")
            return None

        print(f"Success! Result type: {type(result)}")
        print("Result contents:")

        # Explore the result object
        for attr in dir(result):
            if not attr.startswith('_'):
                try:
                    value = getattr(result, attr)
                    if callable(value):
                        print(f"  {attr}() - method")
                    else:
                        value_str = str(value)
                        if len(value_str) > 100:
                            value_str = value_str[:100] + "..."
                        print(f"  {attr}: {type(value).__name__} = {value_str}")
                except Exception as e:
                    print(f"  {attr}: Error accessing - {e}")

        return result

    except Exception as e:
        print(f"Error in quick_room_response_test: {e}")
        return None


def test_devices():
    """Test device listing and recommendations"""
    print(f"\n{'=' * 60}")
    print("Testing device functions")
    print(f"{'=' * 60}")

    # Test list_all_devices
    try:
        devices = sdl_audio_core.list_all_devices()
        print(f"list_all_devices() result type: {type(devices)}")
        if isinstance(devices, dict):
            for key, value in devices.items():
                print(f"  {key}: {len(value) if isinstance(value, list) else type(value)} = {value}")
        else:
            print(f"  Content: {devices}")
    except Exception as e:
        print(f"Error in list_all_devices: {e}")

    # Test get_recommended_device_pairs
    try:
        pairs = sdl_audio_core.get_recommended_device_pairs(max_pairs=2)
        print(f"\nget_recommended_device_pairs() result type: {type(pairs)}")
        print(f"Number of pairs: {len(pairs) if hasattr(pairs, '__len__') else 'N/A'}")

        if pairs and len(pairs) > 0:
            print("First pair details:")
            first_pair = pairs[0]
            for attr in dir(first_pair):
                if not attr.startswith('_'):
                    try:
                        value = getattr(first_pair, attr)
                        if not callable(value):
                            print(f"    {attr}: {type(value).__name__} = {value}")
                    except Exception as e:
                        print(f"    {attr}: Error accessing - {e}")

        return pairs

    except Exception as e:
        print(f"Error in get_recommended_device_pairs: {e}")
        return []


def test_audio_engine():
    """Test AudioEngine class"""
    print(f"\n{'=' * 60}")
    print("Testing AudioEngine")
    print(f"{'=' * 60}")

    try:
        # Create engine and config
        engine = sdl_audio_core.AudioEngine()
        config = sdl_audio_core.AudioEngineConfig()

        print("AudioEngine and AudioEngineConfig created successfully")

        # Try to see what config options are available
        print("\nAudioEngineConfig properties:")
        for attr in dir(config):
            if not attr.startswith('_'):
                try:
                    value = getattr(config, attr)
                    if not callable(value):
                        print(f"  {attr}: {type(value).__name__} = {value}")
                except Exception as e:
                    print(f"  {attr}: Error accessing - {e}")

        # Try to see what engine methods are available
        print("\nAudioEngine methods:")
        for attr in dir(engine):
            if not attr.startswith('_'):
                try:
                    value = getattr(engine, attr)
                    if callable(value):
                        print(f"  {attr}() - method")
                    else:
                        print(f"  {attr}: {type(value).__name__} = {value}")
                except Exception as e:
                    print(f"  {attr}: Error accessing - {e}")

        return engine, config

    except Exception as e:
        print(f"Error testing AudioEngine: {e}")
        return None, None


if __name__ == "__main__":
    print("Starting SDL Audio Core API exploration...")

    # Check installation first
    try:
        check_result = sdl_audio_core.check_installation()
        print(f"Installation check: {check_result}")
    except Exception as e:
        print(f"Installation check error: {e}")

    # Explore classes
    explore_classes()

    # Test device functions
    test_devices()

    # Test quick response
    test_quick_response()

    # Test audio engine
    test_audio_engine()

    print(f"\n{'=' * 60}")
    print("API exploration completed")
    print(f"{'=' * 60}")