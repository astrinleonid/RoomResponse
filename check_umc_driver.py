#!/usr/bin/env python3
"""
Check current UMC1820 driver status and provide installation guidance.
"""

import sys
import io
import subprocess
import re

# Fix Windows console encoding
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')


def check_device_manager():
    """Check what audio devices are installed via Windows Management."""
    print("=" * 70)
    print("CHECKING DEVICE MANAGER")
    print("=" * 70)

    try:
        # Use PowerShell to query audio devices
        cmd = [
            'powershell',
            '-Command',
            "Get-PnpDevice -Class 'MEDIA' | Where-Object {$_.FriendlyName -like '*UMC*' -or $_.FriendlyName -like '*Audio*'} | Select-Object FriendlyName, InstanceId, Status | Format-List"
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)

        if result.stdout:
            print("\nAudio Devices Found:")
            print(result.stdout)

            # Check for Behringer driver
            if 'BEHRINGER UMC1820' in result.stdout.upper() or 'UMC1820' in result.stdout.upper():
                if 'USB\\VID_1397' in result.stdout.upper():
                    print("✓ UMC1820 device detected")

                    if 'BEHRINGER' in result.stdout:
                        print("✓ Using BEHRINGER native driver (GOOD!)")
                        return 'behringer'
                    else:
                        print("✗ Using generic USB Audio driver (NEEDS FIXING)")
                        return 'generic'
            else:
                print("⚠ UMC1820 not detected in Device Manager")
                print("  Make sure the device is plugged in and powered on")
                return 'not_found'
        else:
            print("Could not query Device Manager")
            return 'unknown'

    except Exception as e:
        print(f"Error checking Device Manager: {e}")
        return 'error'


def check_sdl_devices():
    """Check what SDL sees."""
    print("\n" + "=" * 70)
    print("CHECKING SDL AUDIO DEVICES")
    print("=" * 70)

    try:
        import sdl_audio_core

        devices = sdl_audio_core.list_all_devices()

        print(f"\nSDL Version: {sdl_audio_core.get_version()}")

        print("\nINPUT DEVICES:")
        umc_found = False
        for dev in devices.get('input_devices', []):
            is_umc = 'UMC' in dev.name.upper()
            marker = "  → UMC1820 ←" if is_umc else ""
            print(f"  ID {dev.device_id}: {dev.name} ({dev.max_channels} channels) {marker}")
            if is_umc:
                umc_found = True

        print("\nOUTPUT DEVICES:")
        for dev in devices.get('output_devices', []):
            is_umc = 'UMC' in dev.name.upper()
            marker = "  → UMC1820 ←" if is_umc else ""
            print(f"  ID {dev.device_id}: {dev.name} ({dev.max_channels} channels) {marker}")

        return umc_found

    except ImportError:
        print("✗ sdl_audio_core not available - cannot check SDL devices")
        return False
    except Exception as e:
        print(f"Error checking SDL devices: {e}")
        return False


def provide_guidance(driver_status):
    """Provide installation guidance based on current status."""
    print("\n" + "=" * 70)
    print("DIAGNOSIS & RECOMMENDATIONS")
    print("=" * 70)

    if driver_status == 'behringer':
        print("\n✓ STATUS: GOOD - Behringer native driver is installed")
        print("\nYour UMC1820 should work for multi-channel recording.")
        print("\nNext steps:")
        print("  1. Run: python test_umc_input_detailed.py")
        print("  2. If tests pass, you're ready to use multi-channel recording!")
        print("  3. If tests fail, try unplugging and replugging the UMC1820")

    elif driver_status == 'generic':
        print("\n✗ STATUS: PROBLEM - Using generic Windows USB Audio driver")
        print("\nThe generic driver LIMITS UMC1820 to 2 channels (stereo) only.")
        print("Multi-channel recording (10 channels) will NOT work.")
        print("\n" + "!" * 70)
        print("SOLUTION: Install Behringer native driver")
        print("!" * 70)
        print("\nFollow these steps:")
        print("  1. Read: install_behringer_driver.md")
        print("  2. Download driver from: https://www.behringer.com/downloads.html")
        print("  3. Uninstall current driver from Device Manager")
        print("  4. Install Behringer driver")
        print("  5. Restart computer")
        print("  6. Run this script again to verify")

    elif driver_status == 'not_found':
        print("\n⚠ STATUS: UMC1820 not detected")
        print("\nPossible reasons:")
        print("  1. Device is not plugged in")
        print("  2. Device is not powered on")
        print("  3. USB cable issue")
        print("  4. USB port issue")
        print("\nTroubleshooting:")
        print("  1. Check UMC1820 power LED is on")
        print("  2. Try different USB cable")
        print("  3. Try different USB port (preferably USB 3.0)")
        print("  4. Check Device Manager manually (Win+X → Device Manager)")

    else:
        print("\n⚠ STATUS: Could not determine driver status")
        print("\nManual check required:")
        print("  1. Press Win+X → Device Manager")
        print("  2. Expand 'Sound, video and game controllers'")
        print("  3. Look for 'BEHRINGER UMC1820' or 'USB Audio Device'")
        print("  4. If you see 'BEHRINGER UMC1820' → Driver is correct")
        print("  5. If you see 'USB Audio Device' → Need to install Behringer driver")


def main():
    print("UMC1820 Driver Status Checker")
    print("=" * 70)

    # Check Device Manager
    driver_status = check_device_manager()

    # Check SDL
    sdl_found_umc = check_sdl_devices()

    # Provide guidance
    provide_guidance(driver_status)

    print("\n" + "=" * 70)
    print("For detailed installation instructions, see:")
    print("  → install_behringer_driver.md")
    print("=" * 70)


if __name__ == '__main__':
    main()
