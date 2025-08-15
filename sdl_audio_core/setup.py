#sdl_audio_core setup.py

import json
import os
import platform
from pathlib import Path
from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup, Extension
import pybind11


def load_build_config():
    """Load build configuration from JSON file"""
    config_path = Path(__file__).parent / "build_config.json"

    if not config_path.exists():
        raise FileNotFoundError(
            f"build_config.json not found at {config_path}. "
            "Please run detect_paths.py first to generate the configuration."
        )

    with open(config_path, 'r') as f:
        config = json.load(f)

    print(f"Loaded build configuration from {config_path}")
    for key, value in config.items():
        print(f"  {key}: {value}")

    return config


def get_extension_modules():
    """Configure and return the extension modules"""
    config = load_build_config()

    # Source files
    sources = [
        "src/python_bindings.cpp",
        "src/audio_engine.cpp",
        "src/device_manager.cpp",
        # "src/room_response.cpp"  # Temporarily disabled due to compilation issues
    ]

    # Include directories
    include_dirs = [
        pybind11.get_cmake_dir() + "/../include",
        "include",
        "src"
    ]

    # Libraries to link
    libraries = []
    library_dirs = []

    # Platform-specific configuration
    if platform.system() == "Windows":
        # Windows SDL2 configuration
        if "sdl2_include" in config:
            include_dirs.append(config["sdl2_include"])
        if "sdl2_lib" in config:
            library_dirs.append(config["sdl2_lib"])

        libraries.extend(["SDL2", "SDL2main"])

        # Windows-specific libraries
        libraries.extend(["winmm", "dsound", "dxguid", "ole32", "oleaut32", "uuid"])

    elif platform.system() == "Darwin":
        # macOS SDL2 configuration
        if "sdl2_include" in config:
            include_dirs.append(config["sdl2_include"])
        if "sdl2_lib" in config:
            library_dirs.append(config["sdl2_lib"])

        libraries.extend(["SDL2", "SDL2main"])

        # macOS frameworks
        extra_link_args = [
            "-framework", "CoreAudio",
            "-framework", "AudioUnit",
            "-framework", "AudioToolbox",
            "-framework", "CoreFoundation"
        ]

    else:  # Linux
        # Linux SDL2 configuration (usually via pkg-config)
        libraries.extend(["SDL2", "SDL2main", "asound", "pulse"])
        extra_link_args = []

    # Compiler flags
    extra_compile_args = []
    if platform.system() == "Windows":
        extra_compile_args.extend([
            "/O2",  # Optimization
            "/std:c++17",  # C++17 standard
            "/EHsc",  # Exception handling
            "/DWIN32",
            "/D_WINDOWS",
            "/DSDL_MAIN_HANDLED"  # Let SDL handle main
        ])
    else:
        extra_compile_args.extend([
            "-O3",
            "-std=c++17",
            "-DSDL_MAIN_HANDLED"
        ])

    # Create extension
    ext_modules = [
        Pybind11Extension(
            "sdl_audio_core",
            sources=sources,
            include_dirs=include_dirs,
            libraries=libraries,
            library_dirs=library_dirs,
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args if 'extra_link_args' in locals() else [],
            cxx_std=17,
        )
    ]

    return ext_modules


# Custom build command that provides better error messages
class CustomBuildExt(build_ext):
    def build_extensions(self):
        try:
            super().build_extensions()
        except Exception as e:
            print(f"\n{'=' * 60}")
            print("BUILD ERROR OCCURRED")
            print(f"{'=' * 60}")
            print(f"Error: {e}")
            print(f"\nTroubleshooting steps:")
            print(f"1. Ensure SDL2 development libraries are installed")
            print(f"2. Check that detect_paths.py found all required components")
            print(f"3. Verify your compiler is properly configured")

            if platform.system() == "Windows":
                print(f"4. Make sure Visual Studio C++ tools are installed")
                print(f"5. Check that SDL2 paths in build_config.json are correct")
            elif platform.system() == "Darwin":
                print(f"4. Install SDL2 with: brew install sdl2")
                print(f"5. Make sure Xcode command line tools are installed")
            else:
                print(f"4. Install SDL2 with: sudo apt-get install libsdl2-dev")
                print(f"5. Install ALSA development: sudo apt-get install libasound2-dev")

            raise


if __name__ == "__main__":
    setup(
        name="sdl_audio_core",
        version="0.1.0",
        author="Audio Researcher",
        description="Low-level SDL2 audio module for simultaneous playback and recording",
        long_description="A C++ module using SDL2 for precise audio device control, "
                         "designed for room impulse response measurement and audio research.",
        ext_modules=get_extension_modules(),
        cmdclass={"build_ext": CustomBuildExt},
        zip_safe=False,
        python_requires=">=3.7",
        install_requires=[
            "numpy>=1.19.0",
            "pybind11>=2.6.0"
        ],
        extras_require={
            "test": ["pytest", "scipy", "matplotlib"],
        },
        classifiers=[
            "Development Status :: 3 - Alpha",
            "Intended Audience :: Developers",
            "License :: OSI Approved :: MIT License",
            "Programming Language :: Python :: 3",
            "Programming Language :: Python :: 3.7",
            "Programming Language :: Python :: 3.8",
            "Programming Language :: Python :: 3.9",
            "Programming Language :: Python :: 3.10",
            "Programming Language :: Python :: 3.11",
            "Topic :: Multimedia :: Sound/Audio",
            "Topic :: Scientific/Engineering"
        ],
    )