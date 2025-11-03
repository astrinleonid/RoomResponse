# Behringer UMC1820 Driver Installation Instructions

## Step-by-Step Installation Guide

### Step 1: Download the Driver

**Option A: Official Behringer Website (Recommended)**
1. Open your web browser
2. Go to: https://www.behringer.com/downloads.html
3. In the search box, type "UMC1820"
4. Click on "UMC1820" in the results
5. Look for "UMC Driver" or "Driver" download
6. Download the latest version (5.57.0 or 4.59.0)

**Option B: Direct Download Links**
If the official site is slow, you can try these verified sources:
- https://www.drvhub.net/devices/sound-cards/behringer/umc1820
- https://treexy.com/products/driver-fusion/database/sound-video-and-game-controllers/behringer/umc-1820/

**What you're looking for:**
- Filename: `UMC-Driver_4-59-0.zip` or `UMC-Driver_5-57-0.zip`
- Size: ~2-3 MB
- For Windows 10/11 64-bit

### Step 2: Prepare for Installation

1. **Close our application** if it's running
2. **Close any audio software** (DAWs, media players, etc.)
3. **Keep UMC1820 plugged in** for now

### Step 3: Uninstall Generic Driver (Important!)

1. Press `Win + X` and select **"Device Manager"**

2. Expand **"Sound, video and game controllers"**

3. Find your UMC1820 device. It might show as:
   - "USB Audio Device" (generic driver - this is what we need to replace)
   - "UMC1820"
   - "Audio Device on High Definition Audio Bus"

4. **Right-click** on the UMC1820 device → Select **"Uninstall device"**

5. **IMPORTANT**: Check the box that says:
   ☑ **"Delete the driver software for this device"**

6. Click **"Uninstall"**

7. **Unplug the UMC1820** from USB

8. **Wait 10 seconds**

### Step 4: Install Behringer Driver

1. **Extract the downloaded ZIP file**:
   - Right-click the downloaded `.zip` file
   - Select "Extract All..."
   - Choose a location (e.g., Desktop)
   - Click "Extract"

2. **Open the extracted folder**, you should see:
   - `Setup.exe` or `UMC_DriverSetup.exe`
   - Or possibly folders with different versions

3. **Run the installer**:
   - Right-click `Setup.exe` or `UMC_DriverSetup.exe`
   - Select **"Run as administrator"**
   - If prompted by Windows User Account Control, click **"Yes"**

4. **Follow the installation wizard**:
   - Click "Next" or "Install"
   - Accept the license agreement
   - Use default installation location
   - Wait for installation to complete

5. The installer will install **3 components**:
   - ✓ ASIO Driver
   - ✓ WDM Driver (this is what SDL uses)
   - ✓ MIDI Driver

6. When prompted, **Restart your computer** (very important!)

### Step 5: After Restart

1. **Plug in the UMC1820** to the same USB port

2. Windows will detect the device and install the Behringer driver automatically

3. **Wait for the "Device ready to use" notification**

### Step 6: Verify Installation

**Method 1: Device Manager Check**

1. Press `Win + X` → **"Device Manager"**
2. Expand **"Sound, video and game controllers"**
3. You should now see:
   - **"BEHRINGER UMC1820"** ✓ (Good - Behringer driver)
   - NOT "USB Audio Device" ✗ (Bad - generic driver)

**Method 2: Windows Sound Settings Check**

1. Right-click speaker icon in taskbar → **"Sounds"**
2. Go to **"Recording"** tab
3. You should see **"UMC1820"** listed
4. Right-click it → **"Properties"**
5. Go to **"Advanced"** tab
6. In the "Default Format" dropdown, you should see **multiple channel options**:
   - 2 channel, 16 bit, 48000 Hz
   - 2 channel, 24 bit, 48000 Hz
   - **8 channel, 24 bit, 48000 Hz** ← This confirms multi-channel support!
   - 10 channel options may also be available

### Step 7: Test with Our Application

1. Open PowerShell or Command Prompt in the RoomResponse directory:
   ```
   cd D:\repos\RoomResponse
   ```

2. Activate virtual environment:
   ```
   .venv\Scripts\activate
   ```

3. Run the UMC test:
   ```
   python test_umc_input_detailed.py
   ```

4. **Expected results** (after driver installation):
   ```
   Testing with 1 channels...
     ✓ SUCCESS - got 1 channels
   Testing with 2 channels...
     ✓ SUCCESS - got 2 channels
   Testing with 8 channels...
     ✓ SUCCESS - got 8 channels
   Testing with 10 channels...
     ✓ SUCCESS - got 10 channels
   ```

5. If all tests pass, run the full application:
   ```
   streamlit run piano_response.py
   ```

6. Go to **Audio Settings** → **Device Selection**
7. Select **UMC1820** as input device
8. Try **Calibration Impulse** test - should work now!

## Troubleshooting

### Problem: Windows still shows "USB Audio Device" instead of "BEHRINGER UMC1820"

**Solution 1: Manual Driver Update**
1. Device Manager → Right-click UMC1820
2. "Update driver"
3. "Browse my computer for drivers"
4. "Let me pick from a list of available drivers"
5. Select **"BEHRINGER UMC1820"** (not "USB Audio Device")
6. Click "Next"

**Solution 2: Reinstall**
1. Unplug UMC1820
2. Device Manager → View → Show hidden devices
3. Find and uninstall ANY "UMC1820" or "USB Audio Device" entries
4. Restart computer
5. Reinstall Behringer driver
6. Restart again
7. Plug in UMC1820

### Problem: Driver installation fails

1. Make sure you ran installer as Administrator
2. Disable antivirus temporarily
3. Try compatibility mode (Right-click setup.exe → Properties → Compatibility → Run as Windows 7)

### Problem: Tests still fail after driver installation

1. Check Device Manager - make sure it says "BEHRINGER UMC1820"
2. Try different USB port (preferably USB 3.0)
3. Restart computer again
4. Rebuild SDL audio core:
   ```
   cd sdl_audio_core
   ..\.venv\Scripts\python.exe -m pip install -e . --force-reinstall --no-deps
   ```

### Problem: Windows Update reinstalls generic driver

**Prevent Windows from reinstalling generic driver:**
1. Device Manager → BEHRINGER UMC1820 → Properties
2. "Details" tab → Property: Hardware Ids → Copy the VID and PID
3. Use "Show or hide updates" troubleshooter to block the generic driver update

## What Success Looks Like

After successful installation:

✓ Device Manager shows "BEHRINGER UMC1820"
✓ Multi-channel formats visible in Sound Settings
✓ `test_umc_input_detailed.py` passes all tests
✓ Calibration Impulse test works in the application
✓ Can select UMC1820 for multi-channel recording
✓ All 10 input channels accessible

## Additional Notes

- The Behringer driver includes ASIO support (bonus!)
- Lower latency than generic driver
- Better audio quality
- MIDI functionality enabled
- This is the CORRECT way to use UMC1820 on Windows

## Need Help?

If you encounter issues during installation:
1. Note the exact error message
2. Check which step failed
3. Check Device Manager to see what driver is actually installed
4. Run the test script to see specific error messages
