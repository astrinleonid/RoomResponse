# UMC1820 Troubleshooting Guide

## Problem
The Behringer UMC1820 **input** device fails to open with SDL audio, showing errors like:
- "Invalid source channels"
- "Unsupported number of audio channels"

## Test Results
Our automated tests show:
- ✓ **UMC1820 OUTPUT works perfectly** - Negotiates from mono to 12 channels successfully
- ✗ **UMC1820 INPUT completely fails** - Cannot open with ANY channel count (1, 2, 4, 6, 8, 10)

## Root Cause
The UMC1820's Windows audio driver (WDM/WASAPI mode) has compatibility issues with SDL 2.30.0. This is a known limitation with some professional audio interfaces when using standard Windows audio APIs instead of ASIO.

## Workarounds

### Option 1: Windows Sound Settings (Quick Try)

1. Open Windows Sound Settings:
   - Right-click speaker icon in taskbar
   - Select "Sounds" → "Recording" tab
   - Find "UMC1820" device → Right-click → "Properties"

2. Check "Advanced" tab:
   - Try different "Default Format" settings
   - Common working formats:
     - 2 channel, 16 bit, 48000 Hz
     - 2 channel, 24 bit, 48000 Hz
     - 8 channel, 24 bit, 48000 Hz

3. Check "Allow applications to take exclusive control"
   - Try both ENABLED and DISABLED
   - Some devices work better with exclusive mode OFF

4. Restart application and test again

### Option 2: Use Different Input Device (Current Workaround)

Since UMC1820 input doesn't work with SDL, but output does:
- Use webcam or other device for **input**
- Use UMC1820 for **output** (this works!)
- This allows testing the output/playback path

### Option 3: ASIO Driver (Professional Solution)

The UMC1820 is a professional interface designed for ASIO:

1. Download and install Behringer/ASIO4ALL drivers
2. Modify application to use ASIO instead of SDL
3. ASIO provides:
   - Lower latency
   - Better multi-channel support
   - More reliable operation with pro audio hardware

**Note:** This requires code changes to support ASIO API

### Option 4: Try Different SDL Version

SDL 2.30.0 may have specific issues with UMC1820. Trying SDL 2.28.x or 2.24.x might help, but this requires rebuilding the audio core.

## Current Application Status

**What Works:**
- ✓ Other audio devices (webcam, Realtek) work for both input and output
- ✓ Multi-channel output to UMC1820 (up to 12 channels)
- ✓ Channel negotiation for output devices
- ✓ Mono-to-multichannel replication for playback

**What Doesn't Work:**
- ✗ Recording from UMC1820 inputs (any channel count)
- ✗ Using UMC1820 as system default when it's plugged in

## Recommendations

**For Testing/Development:**
Use webcam or other simple device for input testing. This allows development to continue while investigating UMC1820 compatibility.

**For Production:**
Consider implementing ASIO support for professional multi-channel audio interfaces like the UMC1820.

## Technical Details

### SDL Error Messages
```
[AudioEngine] ERROR: Failed to open input device: Invalid source channels
[AudioEngine] ERROR: Failed to open input device: Unsupported number of audio channels
```

### Device Capabilities (as reported by SDL)
- Input: 10 channels max (but cannot actually open)
- Output: 12 channels max (works perfectly)

### Working Configurations
```
✓ Webcam (1ch input) + UMC1820 (12ch output) = WORKS
✓ Webcam (1ch input) + Realtek (2ch output) = WORKS
✗ UMC1820 (Nch input) + Any output = FAILS
```

## Files for Reference
- `test_umc_multichannel.py` - Comprehensive device combination tests
- `test_umc_input_detailed.py` - Focused UMC input channel count tests
- Both scripts demonstrate the issue programmatically
