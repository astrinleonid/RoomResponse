# UMC1820 WASAPI Multi-Channel Solution

## Root Cause Analysis

After deep investigation, the UMC1820 input failure is caused by a **limitation in SDL2's WASAPI backend implementation**, not a bug in our application code.

### What We Found

1. **SDL2 WASAPI Backend Behavior**:
   - SDL's WASAPI code uses `AUDCLNT_STREAMFLAGS_AUTOCONVERTPCM` flag
   - This flag is ONLY applied when **sample rate** differs
   - It is NOT applied when **channel count** differs
   - Source: SDL2 `src/audio/wasapi/SDL_wasapi.c` lines 740-750

2. **WASAPI GetMixFormat Issue**:
   - `IAudioClient::GetMixFormat()` returns Windows' preferred format
   - For some devices (especially USB audio interfaces), this format may be incompatible
   - The channel count returned can be wrong or incompatible with the device's actual capabilities

3. **UMC1820 Specific Behavior**:
   - Reports 10 input channels via `SDL_GetAudioDeviceSpec()`
   - But `GetMixFormat()` likely returns a different configuration
   - SDL doesn't request `AUDCLNT_STREAMFLAGS_AUTOCONVERTPCM` for channel count conversion
   - Result: "Invalid source channels" / "Unsupported number of audio channels" errors

### Evidence

**From our tests:**
- ✓ UMC1820 OUTPUT works (negotiates to 12 channels successfully)
- ✗ UMC1820 INPUT fails with ALL channel counts (1, 2, 4, 6, 8, 10)
- ✓ Other devices (webcam, Realtek) work fine

**From research:**
- This is a known issue across multiple audio libraries (PortAudio, NAudio, JUCE)
- Microsoft docs: `AUDCLNT_STREAMFLAGS_AUTOCONVERTPCM` enables "channel matrixer and sample rate converter"
- Other libraries (miniaudio, portaudio) successfully use this flag for channel conversion

## Solutions

### Solution 1: Upgrade SDL2 (Recommended)

**Current version**: SDL 2.30.0

**Action**: Try SDL 2.28.x or check if SDL3 has fixes

**SDL3 Changes**: According to search results, SDL3 has "WASAPI: Reworked for new SDL3 audio API" which may include better channel handling.

**Steps**:
1. Download SDL 2.28.x or SDL3 (beta)
2. Rebuild `sdl_audio_core` with new SDL version
3. Test with UMC1820

### Solution 2: Patch SDL2 WASAPI Backend (Advanced)

Modify SDL's WASAPI code to always use `AUDCLNT_STREAMFLAGS_AUTOCONVERTPCM`:

**File**: `SDL2/src/audio/wasapi/SDL_wasapi.c`

**Current code** (around line 740-750):
```c
DWORD streamflags = 0;
if ((DWORD)device->spec.freq != waveformat->nSamplesPerSec) {
    streamflags |= (AUDCLNT_STREAMFLAGS_AUTOCONVERTPCM |
                    AUDCLNT_STREAMFLAGS_SRC_DEFAULT_QUALITY);
    waveformat->nSamplesPerSec = device->spec.freq;
}
```

**Proposed fix**:
```c
DWORD streamflags = 0;
// Always use AUTOCONVERTPCM for both sample rate AND channel count conversion
if ((DWORD)device->spec.freq != waveformat->nSamplesPerSec ||
    device->spec.channels != waveformat->nChannels) {
    streamflags |= (AUDCLNT_STREAMFLAGS_AUTOCONVERTPCM |
                    AUDCLNT_STREAMFLAGS_SRC_DEFAULT_QUALITY);
    waveformat->nSamplesPerSec = device->spec.freq;
    waveformat->nChannels = device->spec.channels;
}
```

**Implementation**:
1. Get SDL2 source code
2. Apply patch to `src/audio/wasapi/SDL_wasapi.c`
3. Rebuild SDL2
4. Rebuild our `sdl_audio_core` against patched SDL2

### Solution 3: Windows Audio Device Settings (Quick Try)

**Control Panel Method**:
1. Right-click speaker icon → "Sounds"
2. "Recording" tab → UMC1820 → "Properties"
3. "Advanced" tab → Try different "Default Format" settings:
   - 2 channel, 24 bit, 48000 Hz (most compatible)
   - 8 channel, 24 bit, 48000 Hz
   - Try both WITH and WITHOUT "exclusive mode"
4. Apply and test

**Why this might work**: Changes what `GetMixFormat()` returns, potentially matching what SDL requests.

### Solution 4: Use ASIO Instead of WASAPI (Professional)

The UMC1820 is designed for ASIO:

**Advantages**:
- Lower latency
- Better multi-channel support
- Direct driver access
- Industry standard for pro audio

**Disadvantages**:
- Requires code changes
- ASIO SDK license restrictions
- Windows-only

**Implementation** would require:
1. Replace SDL audio backend with ASIO
2. Use RtAudio or PortAudio with ASIO backend
3. Significant refactoring

### Solution 5: Force Shared Mode Format (Workaround)

Instead of using `GetMixFormat()`, force a specific format that we know works:

**Pseudocode**:
```c
// Instead of using GetMixFormat(), create our own format
WAVEFORMATEXTENSIBLE wfx = {};
wfx.Format.wFormatTag = WAVE_FORMAT_EXTENSIBLE;
wfx.Format.nChannels = requested_channels;
wfx.Format.nSamplesPerSec = 48000;
wfx.Format.wBitsPerSample = 32; // Float32
wfx.Format.nBlockAlign = (wfx.Format.nChannels * wfx.Format.wBitsPerSample) / 8;
wfx.Format.nAvgBytesPerSec = wfx.Format.nSamplesPerSec * wfx.Format.nBlockAlign;
wfx.Format.cbSize = 22;
wfx.Samples.wValidBitsPerSample = 32;
wfx.dwChannelMask = /* appropriate mask for channel count */;
wfx.SubFormat = KSDATAFORMAT_SUBTYPE_IEEE_FLOAT;

// Use this format with AUDCLNT_STREAMFLAGS_AUTOCONVERTPCM
IAudioClient::Initialize(
    AUDCLNT_SHAREMODE_SHARED,
    AUDCLNT_STREAMFLAGS_AUTOCONVERTPCM | AUDCLNT_STREAMFLAGS_SRC_DEFAULT_QUALITY,
    ...
    &wfx.Format,
    ...
);
```

This bypasses `GetMixFormat()` entirely.

## Recommended Approach

**Phase 1: Quick Test** (10 minutes)
- Try Solution 3: Change Windows audio settings for UMC1820
- Test different default formats in Control Panel

**Phase 2: SDL Upgrade** (1-2 hours)
- Download SDL 2.28.5 or try SDL3 preview
- Rebuild sdl_audio_core
- Test with UMC1820

**Phase 3: SDL Patch** (if Phase 2 fails) (2-4 hours)
- Get SDL2 source
- Apply WASAPI patch for channel conversion
- Build custom SDL2
- Integrate and test

**Phase 4: Alternative Backend** (if all else fails) (1-2 days)
- Implement RtAudio or PortAudio backend
- Use ASIO mode for professional interfaces
- Keep SDL for simple devices

## Technical References

1. **Microsoft WASAPI Documentation**:
   - `AUDCLNT_STREAMFLAGS_AUTOCONVERTPCM`: Enables automatic format conversion
   - https://learn.microsoft.com/en-us/windows/win32/coreaudio/audclnt-streamflags-xxx-constants

2. **SDL Issues**:
   - SDL #3234: "Improve WASAPI audio backend latency"
   - SDL #2814: "WASAPI: No sound on Windows 10"
   - SDL3: "WASAPI: Reworked for new SDL3 audio API"

3. **Similar Issues in Other Libraries**:
   - PortAudio #365: "Invalid max output channel count reported using WASAPI"
   - PortAudio #286: "WASAPI gets the channel count wrong for some devices"
   - NAudio #819: "WASAPI built-in resampler: Add AUDCLNT_STREAMFLAGS_AUTOCONVERTPCM flag"
   - FlexASIO #32: "Use WASAPI AUTOCONVERTPCM to automatically convert channel count"

4. **Our Test Results**:
   - `test_umc_multichannel.py`: Comprehensive device combination tests
   - `test_umc_input_detailed.py`: UMC input channel count tests
   - Both confirm UMC1820 input fails universally with SDL2 2.30.0

## Next Steps

1. **User action**: Try Windows Control Panel audio settings (Solution 3)
2. **Development**: Test with different SDL2 versions (Solution 1)
3. **If needed**: Implement SDL2 WASAPI patch (Solution 2)
4. **Long term**: Consider ASIO support for professional audio interfaces (Solution 4)
