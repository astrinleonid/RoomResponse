# Phase 1 Multi-Channel Audio Testing Guide

## Overview

This guide provides comprehensive testing procedures for the Phase 1 multi-channel audio implementation.

## Test Levels

### Level 1: Basic API Tests (No Hardware Required)
**File:** `test_phase1_basic.py`

Tests the multi-channel API without requiring audio hardware. Validates:
- Module imports successfully
- New configuration parameters exist and work
- New methods are available
- Backward compatibility is maintained
- Channel validation works correctly

**Run:**
```bash
python test_phase1_basic.py
```

**Expected Output:**
```
✓ Import Module                    PASSED
✓ Config API                       PASSED
✓ Stats API                        PASSED
✓ AudioEngine Methods              PASSED
✓ Convenience Function             PASSED
✓ Channel Validation               PASSED
✓ Backward Compatibility           PASSED

Overall: 7/7 tests passed
🎉 ALL BASIC TESTS PASSED!
```

### Level 2: Hardware Tests (Requires Audio Devices)
**File:** `test_phase1_hardware.py`

Tests actual multi-channel recording with real audio hardware. Validates:
- Single-channel recording (baseline)
- Multi-channel recording (2, 4, 8 channels)
- Channel synchronization via cross-correlation
- Different channel configurations
- Saves WAV files for manual inspection

**Prerequisites:**
- Audio input device (microphone, audio interface)
- Audio output device (speakers, headphones)
- `scipy` package for chirp signal and cross-correlation
- **For multi-channel testing (4+ channels):** Professional audio interface with native drivers installed
  - See [Multi-Channel Device Drivers](#multi-channel-device-drivers) section below

**Install scipy (if needed):**
```bash
pip install scipy
```

**Run:**
```bash
python test_phase1_hardware.py
```

**Expected Output:**
```
AUDIO DEVICE ENUMERATION
Input Devices: X
  [0] Microphone Array (...)
      Max channels: 2
  [1] MOTU 8A (...)
      Max channels: 8

✓ Single-Channel Recording         PASSED
✓ Multi-Channel Recording (2ch)    PASSED
✓ Multi-Channel Recording (4ch)    PASSED
✓ Channel Synchronization          PASSED
✓ Different Configurations         PASSED

Overall: 5/5 tests passed
🎉 ALL HARDWARE TESTS PASSED!
```

**Note:** Some tests may be skipped if your hardware doesn't support the required number of channels. This is normal and not a failure.

## Manual Testing

### Test 1: Quick Import Verification

```python
import sdl_audio_core

# Check version
print(sdl_audio_core.__version__)

# List new functions
print('New functions:')
print('  measure_room_response_auto_multichannel' in dir(sdl_audio_core))

# Check config
config = sdl_audio_core.AudioEngineConfig()
print(f'Default channels: {config.input_channels}')
```

### Test 2: Simple 2-Channel Recording

```python
import sdl_audio_core
import numpy as np

# Generate test signal (1kHz sine, 100ms)
sample_rate = 48000
t = np.arange(4800) / sample_rate
test_signal = (0.3 * np.sin(2 * np.pi * 1000 * t)).tolist()

# Record 2 channels
result = sdl_audio_core.measure_room_response_auto_multichannel(
    test_signal,
    volume=0.3,
    input_device=0,
    output_device=0,
    input_channels=2
)

if result['success']:
    print(f"Recorded {result['num_channels']} channels")
    print(f"Samples per channel: {result['samples_per_channel']}")

    # Access channel data
    ch0 = result['multichannel_data'][0]
    ch1 = result['multichannel_data'][1]

    print(f"Channel 0: {len(ch0)} samples")
    print(f"Channel 1: {len(ch1)} samples")
else:
    print(f"Failed: {result['error_message']}")
```

### Test 3: Enumerate Devices and Check Capabilities

```python
import sdl_audio_core

devices = sdl_audio_core.list_all_devices()

print("Input Devices:")
for dev in devices['input_devices']:
    print(f"  [{dev.device_id}] {dev.name}")
    print(f"      Max channels: {dev.max_channels}")
    print(f"      Supported rates: {dev.supported_sample_rates}")

print("\nOutput Devices:")
for dev in devices['output_devices']:
    print(f"  [{dev.device_id}] {dev.name}")
    print(f"      Max channels: {dev.max_channels}")
```

### Test 4: Direct AudioEngine Usage

```python
import sdl_audio_core

# Create engine with 4 channels
engine = sdl_audio_core.AudioEngine()
config = sdl_audio_core.AudioEngineConfig()
config.input_channels = 4
config.sample_rate = 48000

if engine.initialize(config):
    print("✓ Engine initialized")

    # Check channel count
    print(f"Input channels: {engine.get_num_input_channels()}")
    print(f"Output channels: {engine.get_num_output_channels()}")

    # Get stats
    stats = engine.get_stats()
    print(f"Stats: {stats.num_input_channels} in, {stats.num_output_channels} out")
    print(f"Channel buffer sizes: {stats.channel_buffer_sizes}")

    engine.shutdown()
else:
    print("✗ Failed to initialize")
```

## Multi-Channel Device Drivers

**Important:** Professional audio interfaces require **native drivers** for multi-channel operation.

### Why Native Drivers Are Required

Windows generic USB Audio Class 2.0 driver has limitations:
- Reports device capabilities (e.g., "10 channels available")
- But WDM/WASAPI interface is **hardcoded to stereo (2 channels)** only
- Multi-channel recording will fail with "Invalid source channels" error

Native manufacturer drivers provide:
- ✓ Full multi-channel access through WDM/WASAPI
- ✓ ASIO support for low-latency professional audio
- ✓ Proper channel configuration for Windows audio APIs
- ✓ Better audio quality and stability

### Common Professional Audio Interfaces

| Device | Native Driver Required | Max Channels | Driver Download |
|--------|------------------------|--------------|-----------------|
| Behringer UMC1820 | Yes | 18 input / 20 output | https://www.behringer.com/downloads.html |
| Focusrite Scarlett | Yes | 2-18 (model dependent) | https://focusrite.com/downloads |
| PreSonus AudioBox | Yes | 2-32 (model dependent) | https://www.presonus.com/products |
| MOTU Audio Express | Yes | 6 input / 6 output | https://motu.com/download |

### Installation and Verification

**For Behringer UMC1820:**

1. Check current driver status:
   ```bash
   python check_umc_driver.py
   ```

2. If using generic driver, follow installation guide:
   - See: [install_behringer_driver.md](install_behringer_driver.md)
   - Download from: https://www.behringer.com/downloads.html
   - Version 4.59.0 or 5.57.0 recommended

3. Verify installation:
   ```bash
   python test_umc_input_detailed.py
   ```

**Expected results after native driver installation:**
```
Testing with 1 channels...  ✓ SUCCESS
Testing with 2 channels...  ✓ SUCCESS
Testing with 8 channels...  ✓ SUCCESS
Testing with 10 channels... ✓ SUCCESS
```

**For other interfaces:** Check manufacturer's website for:
- ASIO drivers
- WDM/WASAPI drivers
- Device-specific control software

### Testing Without Native Drivers

If native drivers are not installed, tests will be limited to:
- ✓ 1-2 channel recording (stereo)
- ✗ Multi-channel recording will fail

This is expected behavior and not a bug in the software.

## Troubleshooting

### "No device found supporting N channels"

**Cause:** Your audio hardware doesn't support the requested number of channels.

**Solution:**
1. Check your device capabilities: Run the device enumeration test
2. Use a lower channel count that your hardware supports
3. For testing multi-channel with consumer hardware, use a USB audio interface

### "Recording failed: Device does not support X channels"

**Cause:** The SDL driver reports that the device doesn't support the requested channel count.

**Solution:**
1. Verify device capabilities in Windows Sound Settings or device manager
2. Some devices advertise max channels but require specific driver configurations
3. Try a different device or lower channel count

### "Invalid source channels" or Multi-Channel Devices Not Working

**Cause:** Professional audio interfaces (like Behringer UMC1820) require native drivers for multi-channel operation. Windows generic USB Audio driver limits devices to stereo (2 channels) only.

**Solution for Behringer UMC1820:**
1. Check current driver status:
   ```bash
   python check_umc_driver.py
   ```
2. If using generic driver, install Behringer native driver
3. See detailed instructions: [install_behringer_driver.md](install_behringer_driver.md)
4. Download driver from: https://www.behringer.com/downloads.html
5. After installation, all input channels will be accessible

**Solution for Other Professional Interfaces:**
- Check manufacturer's website for native ASIO/WDM drivers
- Focusrite: Focusrite Control software and ASIO driver
- PreSonus: Universal Control and ASIO driver
- MOTU: MOTU AVB/USB driver

**Quick Driver Check:**
```bash
# Check UMC1820 driver status and SDL detection
python check_umc_driver.py

# Test multi-channel device combinations
python test_umc_multichannel.py

# Test detailed channel counts
python test_umc_input_detailed.py
```

### "Very low signal amplitude - check volume/microphone"

**Cause:** Input signal is very quiet or microphone is muted.

**Solution:**
1. Check microphone/input is not muted in Windows Sound Settings
2. Increase input volume in Windows
3. Increase the `volume` parameter in the function call
4. Move closer to the speaker or increase speaker volume

### Build Errors

If you encounter build errors after pulling updates:

1. Clean the build:
```bash
cd sdl_audio_core
rm -rf build  # or rmdir /s /q build on Windows
```

2. Rebuild:
```bash
python setup.py build_ext --inplace
```

3. If still failing, check:
   - SDL2 is installed correctly
   - Visual Studio C++ tools are available
   - Python version matches (3.7+)

## Validation Checklist

Use this checklist to verify Phase 1 implementation:

- [ ] Basic tests pass (all 7 tests)
- [ ] Hardware tests pass (or skip gracefully if hardware unavailable)
- [ ] Can record mono (1 channel) - baseline
- [ ] Can record stereo (2 channels)
- [ ] Can record 4+ channels (if hardware available)
- [ ] Channels are synchronized (cross-correlation test)
- [ ] WAV files saved successfully
- [ ] Manual listening: WAV files sound correct
- [ ] Backward compatibility: old code still works
- [ ] Channel validation: rejects invalid counts (0, -1, 33+)
- [ ] Stats report correct channel counts

## Performance Benchmarks

Expected performance characteristics:

| Configuration | CPU Usage | Memory | Buffer Overruns |
|---------------|-----------|--------|-----------------|
| 1 channel @ 48kHz | < 1% | ~2 MB | 0 |
| 2 channels @ 48kHz | < 2% | ~4 MB | 0 |
| 4 channels @ 48kHz | < 3% | ~8 MB | 0 |
| 8 channels @ 48kHz | < 5% | ~16 MB | 0 |

**Note:** Actual performance depends on CPU, driver, and system load.

## Next Steps After Testing

Once all tests pass:

1. **Review recorded WAV files** in `test_recordings/` directory
   - Open in Audacity or similar
   - Verify each channel has distinct audio
   - Check for sync issues (channels should align)

2. **Integration testing**
   - Test with your actual piano recording setup
   - Verify with your specific hardware (audio interface)
   - Test realistic scenarios (longer recordings, realistic signals)

3. **Proceed to Phase 2**
   - Update `RoomResponseRecorder` to use multi-channel API
   - Implement piano-specific multi-channel workflows
   - See `PIANO_MULTICHANNEL_PLAN.md` for Phase 2 details

## Reporting Issues

If you find bugs or issues:

1. Note which test failed
2. Capture the full error output
3. Include your hardware info:
   - Audio interface model
   - Driver version
   - Number of channels requested/available
4. Attach `test_recordings/*.wav` files if relevant

## Additional Resources

- **Implementation Plan:** `PHASE1_IMPLEMENTATION_PLAN.md`
- **Code Changes:** `PHASE1_IMPLEMENTATION_SUMMARY.md`
- **Overall Strategy:** `PIANO_MULTICHANNEL_PLAN.md`
- **Technical Details:** `TECHNICAL_DOCUMENTATION.md`
