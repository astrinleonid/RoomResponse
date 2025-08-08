import sdl_audio_core
import inspect


def inspect_sdl_module():
    """Inspect the sdl_audio_core module to see available functions and their signatures"""

    print("SDL Audio Core Module Inspection")
    print("=" * 50)

    # Get all attributes of the module
    all_attrs = dir(sdl_audio_core)

    print(f"Module location: {sdl_audio_core.__file__ if hasattr(sdl_audio_core, '__file__') else 'Built-in module'}")
    print(f"Module doc: {sdl_audio_core.__doc__ if sdl_audio_core.__doc__ else 'No documentation'}")

    print(f"\nAll attributes ({len(all_attrs)}):")
    for attr in all_attrs:
        if not attr.startswith('_'):  # Skip private attributes
            print(f"  {attr}")

    print(f"\nCallable functions:")
    functions = []

    for attr_name in all_attrs:
        if not attr_name.startswith('_'):  # Skip private attributes
            attr = getattr(sdl_audio_core, attr_name)
            if callable(attr):
                functions.append(attr_name)
                print(f"\n  {attr_name}:")

                # Try to get function signature
                try:
                    sig = inspect.signature(attr)
                    print(f"    Signature: {attr_name}{sig}")
                except (ValueError, TypeError):
                    print(f"    Signature: Cannot determine (likely C function)")

                # Try to get docstring
                if attr.__doc__:
                    print(f"    Doc: {attr.__doc__}")
                else:
                    print(f"    Doc: No documentation")

    print(f"\nSummary:")
    print(f"  Total attributes: {len([a for a in all_attrs if not a.startswith('_')])}")
    print(f"  Callable functions: {len(functions)}")
    print(f"  Function names: {', '.join(functions)}")

    return functions


if __name__ == "__main__":
    try:
        available_functions = inspect_sdl_module()
    except Exception as e:
        print(f"Error inspecting module: {e}")
