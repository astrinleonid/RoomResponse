#!/usr/bin/env python3
"""
SDL Audio Core - System Detection Script
Automatically detects SDL2, compiler, and system configuration
"""

import json
import os
import platform
import subprocess
import sys
from pathlib import Path
import shutil


def run_command(cmd, capture_output=True, shell=True):
    """Run a command and return success status and output"""
    try:
        if isinstance(cmd, str):
            result = subprocess.run(cmd, shell=shell, capture_output=capture_output,
                                    text=True, timeout=30)
        else:
            result = subprocess.run(cmd, capture_output=capture_output,
                                    text=True, timeout=30)
        return result.returncode == 0, result.stdout.strip(), result.stderr.strip()
    except subprocess.TimeoutExpired:
        return False, "", "Command timed out"
    except Exception as e:
        return False, "", str(e)


def detect_windows_sdl2():
    """Detect SDL2 on Windows"""
    print("Detecting SDL2 on Windows...")

    # Common SDL2 installation paths
    search_paths = [
        r"C:\SDL2",
        r"C:\SDL2-*",
        r"C:\dev\SDL2",
        r"C:\lib\SDL2",
        r"C:\libs\SDL2",
        r"C:\Program Files\SDL2",
        r"C:\Program Files (x86)\SDL2",
        r"C:\Program Files\SDL2-*",           # Add this line
        r"C:\Program Files\SDL2-2.32.8",
        r".\SDL2",
        r"..\SDL2",
        r"..\..\SDL2"
    ]

    # Add vcpkg paths
    vcpkg_paths = [
        r"C:\vcpkg\installed\x64-windows",
        r"C:\tools\vcpkg\installed\x64-windows",
        r".\vcpkg\installed\x64-windows"
    ]

    sdl2_config = {}

    # Check vcpkg first (preferred method)
    for vcpkg_path in vcpkg_paths:
        vcpkg_path = Path(vcpkg_path)
        if vcpkg_path.exists():
            include_dir = vcpkg_path / "include"
            lib_dir = vcpkg_path / "lib"

            if (include_dir / "SDL2" / "SDL.h").exists() and lib_dir.exists():
                print(f"✓ Found SDL2 via vcpkg at: {vcpkg_path}")
                sdl2_config["sdl2_include"] = str(include_dir)
                sdl2_config["sdl2_lib"] = str(lib_dir)
                sdl2_config["installation_method"] = "vcpkg"
                return sdl2_config

    # Check manual installations
    import glob
    for search_pattern in search_paths:
        for path in glob.glob(search_pattern):
            path = Path(path)
            if path.exists():
                # Look for include and lib directories
                include_candidates = [
                    path / "include",
                    path / "include" / "SDL2",
                    path / "SDL2" / "include",
                ]

                lib_candidates = [
                    path / "lib",
                    path / "lib" / "x64",
                    path / "SDL2" / "lib" / "x64",
                    path / "lib" / "win32"
                ]

                include_dir = None
                lib_dir = None

                # Find include directory
                for candidate in include_candidates:
                    if (candidate / "SDL.h").exists() or (candidate / "SDL2" / "SDL.h").exists():
                        include_dir = candidate
                        break

                # Find lib directory
                for candidate in lib_candidates:
                    if candidate.exists() and any(candidate.glob("SDL2*.lib")):
                        lib_dir = candidate
                        break

                if include_dir and lib_dir:
                    print(f"✓ Found SDL2 manual installation at: {path}")
                    sdl2_config["sdl2_include"] = str(include_dir)
                    sdl2_config["sdl2_lib"] = str(lib_dir)
                    sdl2_config["installation_method"] = "manual"
                    return sdl2_config

    print("✗ SDL2 not found!")
    print("\nTo install SDL2 on Windows:")
    print("Option 1 (Recommended): Use vcpkg")
    print("  1. Install vcpkg: https://github.com/Microsoft/vcpkg")
    print("  2. Run: vcpkg install sdl2:x64-windows")
    print()
    print("Option 2: Manual installation")
    print("  1. Download SDL2 development libraries from https://www.libsdl.org/")
    print("  2. Extract to C:\\SDL2\\")
    print("  3. Make sure the structure is: C:\\SDL2\\include\\SDL2\\SDL.h")
    print("                                 C:\\SDL2\\lib\\x64\\SDL2.lib")

    return None


def detect_macos_sdl2():
    """Detect SDL2 on macOS"""
    print("Detecting SDL2 on macOS...")

    sdl2_config = {}

    # Check homebrew installation
    success, output, error = run_command("brew --prefix sdl2")
    if success and output:
        brew_path = Path(output)
        include_dir = brew_path / "include"
        lib_dir = brew_path / "lib"

        if (include_dir / "SDL2" / "SDL.h").exists() and lib_dir.exists():
            print(f"✓ Found SDL2 via Homebrew at: {brew_path}")
            sdl2_config["sdl2_include"] = str(include_dir)
            sdl2_config["sdl2_lib"] = str(lib_dir)
            sdl2_config["installation_method"] = "homebrew"
            return sdl2_config

    # Check system frameworks
    framework_path = Path("/Library/Frameworks/SDL2.framework")
    if framework_path.exists():
        headers_path = framework_path / "Headers"
        if (headers_path / "SDL.h").exists():
            print(f"✓ Found SDL2 Framework at: {framework_path}")
            sdl2_config["sdl2_include"] = str(headers_path)
            sdl2_config["sdl2_framework"] = str(framework_path)
            sdl2_config["installation_method"] = "framework"
            return sdl2_config

    # Check MacPorts
    macports_path = Path("/opt/local")
    if macports_path.exists():
        include_dir = macports_path / "include"
        lib_dir = macports_path / "lib"

        if (include_dir / "SDL2" / "SDL.h").exists() and lib_dir.exists():
            print(f"✓ Found SDL2 via MacPorts at: {macports_path}")
            sdl2_config["sdl2_include"] = str(include_dir)
            sdl2_config["sdl2_lib"] = str(lib_dir)
            sdl2_config["installation_method"] = "macports"
            return sdl2_config

    print("✗ SDL2 not found!")
    print("\nTo install SDL2 on macOS:")
    print("Option 1 (Recommended): Use Homebrew")
    print("  1. Install Homebrew: https://brew.sh/")
    print("  2. Run: brew install sdl2")
    print()
    print("Option 2: Use MacPorts")
    print("  1. Install MacPorts: https://www.macports.org/")
    print("  2. Run: sudo port install libsdl2")

    return None


def detect_linux_sdl2():
    """Detect SDL2 on Linux"""
    print("Detecting SDL2 on Linux...")

    sdl2_config = {}

    # Check pkg-config first
    success, output, error = run_command("pkg-config --cflags --libs sdl2")
    if success:
        print("✓ Found SDL2 via pkg-config")
        sdl2_config["installation_method"] = "pkg-config"
        # pkg-config handles paths automatically
        return sdl2_config

    # Check common system paths
    system_paths = [
        "/usr/include/SDL2",
        "/usr/local/include/SDL2",
        "/opt/local/include/SDL2"
    ]

    for include_path in system_paths:
        include_path = Path(include_path)
        if (include_path / "SDL.h").exists():
            print(f"✓ Found SDL2 headers at: {include_path}")
            sdl2_config["sdl2_include"] = str(include_path.parent)
            sdl2_config["installation_method"] = "system"
            return sdl2_config

    print("✗ SDL2 not found!")
    print("\nTo install SDL2 on Linux:")
    print("Ubuntu/Debian: sudo apt-get install libsdl2-dev")
    print("CentOS/RHEL:   sudo yum install SDL2-devel")
    print("Fedora:        sudo dnf install SDL2-devel")
    print("Arch:          sudo pacman -S sdl2")

    return None


def detect_compiler():
    """Detect available C++ compiler"""
    print("Detecting C++ compiler...")

    compilers_to_check = []

    if platform.system() == "Windows":
        # Check for Visual Studio
        vs_years = ["2022", "2019", "2017"]
        vs_editions = ["Enterprise", "Professional", "Community", "BuildTools"]

        for year in vs_years:
            for edition in vs_editions:
                vs_path = Path(f"C:/Program Files/Microsoft Visual Studio/{year}/{edition}")
                if vs_path.exists():
                    print(f"✓ Found Visual Studio {year} {edition}")
                    return {
                        "compiler": "msvc",
                        "version": year,
                        "edition": edition,
                        "path": str(vs_path)
                    }

        # Check for build tools
        build_tools_path = Path("C:/Program Files (x86)/Microsoft Visual Studio/2019/BuildTools")
        if build_tools_path.exists():
            print("✓ Found Visual Studio Build Tools")
            return {
                "compiler": "msvc",
                "version": "2019",
                "edition": "BuildTools"
            }

        compilers_to_check = ["cl", "clang++", "g++"]

    else:
        compilers_to_check = ["clang++", "g++", "c++"]

    # Check for available compilers
    for compiler in compilers_to_check:
        success, output, error = run_command(f"{compiler} --version")
        if success:
            print(f"✓ Found compiler: {compiler}")
            return {
                "compiler": compiler,
                "version_output": output.split('\n')[0] if output else "Unknown"
            }

    print("✗ No suitable C++ compiler found!")
    if platform.system() == "Windows":
        print("Please install Visual Studio 2019 or later with C++ tools")
    else:
        print("Please install a C++ compiler (g++ or clang++)")

    return None


def detect_python_environment():
    """Detect Python environment details"""
    print("Detecting Python environment...")

    python_info = {
        "version": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "architecture": platform.architecture()[0]
    }

    # Check if in virtual environment
    in_venv = hasattr(sys, 'real_prefix') or (
            hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix
    )

    python_info["virtual_env"] = in_venv
    if in_venv:
        python_info["venv_path"] = sys.prefix

    print(f"✓ Python {sys.version.split()[0]} ({'64-bit' if '64bit' in platform.architecture()[0] else '32-bit'})")
    print(f"  Virtual environment: {'Yes' if in_venv else 'No'}")

    return python_info


def main():
    """Main detection function"""
    print("=" * 60)
    print("SDL Audio Core - System Detection")
    print("=" * 60)
    print()

    # Collect all configuration
    config = {
        "detection_timestamp": subprocess.check_output(
            ["python", "-c", "import datetime; print(datetime.datetime.now().isoformat())"],
            text=True
        ).strip(),
        "platform": platform.system(),
        "architecture": platform.architecture()[0]
    }

    # Detect Python environment
    python_info = detect_python_environment()
    config["python"] = python_info

    # Detect compiler
    compiler_info = detect_compiler()
    if not compiler_info:
        print("\n❌ No suitable compiler found. Build will fail.")
        return False
    config["compiler"] = compiler_info

    # Detect SDL2 based on platform
    if platform.system() == "Windows":
        sdl2_info = detect_windows_sdl2()
    elif platform.system() == "Darwin":
        sdl2_info = detect_macos_sdl2()
    else:
        sdl2_info = detect_linux_sdl2()

    if not sdl2_info:
        print("\n❌ SDL2 not found. Build will fail.")
        return False

    config.update(sdl2_info)

    # Save configuration
    config_path = Path("build_config.json")
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

    print(f"\n✅ Configuration saved to {config_path}")
    print("\nDetection Summary:")
    print(f"  Platform: {config['platform']} ({config['architecture']})")
    print(f"  Compiler: {compiler_info.get('compiler', 'Unknown')}")
    print(f"  SDL2: {sdl2_info.get('installation_method', 'Found')}")
    print(f"  Python: {python_info['virtual_env'] and 'Virtual env' or 'System'}")

    print("\n🚀 Ready to build! Run build_sdl_audio.bat to compile the module.")
    return True


if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)