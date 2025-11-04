# Full Resolution Audio Save Format

## Overview

The RoomResponse system now supports saving audio recordings in **full resolution** format alongside or instead of standard WAV files. This preserves the exact **calibration-normalized** values without additional max-value normalization or 16-bit quantization.

**Important Difference:**
- **WAV format**: Aligned → Averaged → **Max-normalized (0-95%)** → 16-bit quantized (for audio playback)
- **NumPy format**: Aligned → Averaged → **Calibration-normalized** → Full float64 precision (for measurement analysis)

The NumPy files preserve the **actual measurement values** after calibration normalization, while WAV files apply additional max-value normalization for audio software compatibility.

## Feature Summary

### What's New

- **NumPy Array Format (.npy)**: Save audio as full precision float64 NumPy arrays
- **Preserves Calibration Values**: NumPy format keeps actual measurement values (no max normalization)
- **Configurable Formats**: Choose WAV, NumPy, or both formats
- **GUI Configuration**: Easy-to-use interface in Audio Settings panel
- **No Quantization Loss**: Preserve exact calibration-normalized values without 16-bit rounding

## Formats Comparison

| Feature | WAV (16-bit) | NumPy (.npy) |
|---------|--------------|--------------|
| **Compatibility** | All audio software | Python/NumPy only |
| **Data Type** | 16-bit integer | 64-bit float |
| **Processing** | Aligned + Averaged | Aligned + Averaged |
| **Normalization** | **Max-value (0-95%)** | **Calibration only** |
| **Quantization** | Yes (65536 levels) | No (full precision) |
| **File Size** | ~94 KB/sec @ 48kHz | ~375 KB/sec @ 48kHz |
| **Measurement Values** | Scaled for playback | **Actual measured values** |
| **Use Case** | Audio playback, visualization | Scientific analysis, measurements |

### Key Benefits

**NumPy Format Advantages:**
1. **Preserves Actual Values**: No max-value normalization means actual measurement magnitudes preserved
2. **Calibration Normalized**: Values relative to calibration impulse (if enabled)
3. **No Quantization**: Full float64 precision without 16-bit rounding
4. **Scientific Analysis**: Perfect for comparing absolute magnitudes across measurements

**WAV Format Advantages:**
1. **Universal Playback**: Works in all audio software
2. **Optimized for Listening**: Max-normalization prevents clipping
3. **Smaller Files**: ~4x smaller than NumPy
4. **Standard Format**: Easy to share and visualize

## Configuration

### Method 1: Configuration File

Edit `recorderConfig.json`:

```json
{
  "save_format": {
    "save_wav": true,
    "save_npy": true
  }
}
```

### Method 2: GUI Panel

1. Open the application: `python piano_response.py`
2. Navigate to: **Audio Settings → Device Selection & Testing**
3. Find the **"Save Format Configuration"** section
4. Check/uncheck format options:
   - ☑ Save as WAV (16-bit) - Standard audio format
   - ☑ Save as NumPy (full resolution) - Full precision
5. Click **"Apply Save Format Settings"**

### Configuration Options

- `save_wav: true` - Save 16-bit WAV files (default: enabled)
- `save_npy: true` - Save float64 NumPy files (default: disabled)
- Both can be enabled simultaneously

**Note**: At least one format must be enabled.

## File Structure

When both formats are enabled, recordings produce:

```
raw_000_20251104123045_ch0.wav      # 16-bit WAV
raw_000_20251104123045_ch0.npy      # Full resolution NumPy

impulse_000_20251104123045_ch0.wav  # 16-bit WAV
impulse_000_20251104123045_ch0.npy  # Full resolution NumPy

room_000_room_20251104123045_ch0.wav  # 16-bit WAV
room_000_room_20251104123045_ch0.npy  # Full resolution NumPy
```

## Usage Examples

### Loading NumPy Files

```python
import numpy as np

# Load full resolution audio
audio_data = np.load('raw_000_20251104123045_ch0.npy')

print(f"Shape: {audio_data.shape}")
print(f"Data type: {audio_data.dtype}")  # float64
print(f"Min: {audio_data.min()}, Max: {audio_data.max()}")

# Process with full precision
# (no quantization, no normalization)
fft_result = np.fft.rfft(audio_data)
```

### Loading WAV Files (for comparison)

```python
import wave
import numpy as np

# Load 16-bit WAV
with wave.open('raw_000_20251104123045_ch0.wav', 'r') as wav_file:
    audio_int16 = np.frombuffer(wav_file.readframes(-1), dtype=np.int16)
    audio_float = audio_int16.astype(np.float64) / 32767.0

# Note: This data has been quantized and normalized
```

## Use Cases

### When to Use NumPy Format

1. **Absolute Magnitude Comparison**: Compare response levels across different measurements
2. **Calibration-Based Analysis**: Work with values relative to calibration impulse
3. **Scientific Measurements**: Need actual measured values, not playback-optimized
4. **High-Precision FFT**: Avoid quantization artifacts in frequency analysis
5. **Archival**: Preserve exact calibration-normalized values for future analysis
6. **Multi-Stage Processing**: Prevent cumulative quantization errors

### When to Use WAV Format

1. **Listening**: Standard playback in audio software
2. **Compatibility**: Share with non-Python tools
3. **File Size**: Storage space is limited
4. **Distribution**: Provide to users without Python/NumPy

### When to Use Both

- **Best of Both Worlds**: Compatibility + precision
- **Production Workflows**: WAV for review, NumPy for processing
- **Research Projects**: WAV for presentations, NumPy for analysis

## Implementation Details

### Changes Made

1. **RoomResponseRecorder.py**:
   - Modified `_save_wav()` method to support dual formats
   - Added configuration reading for save format settings
   - Automatic `.npy` file generation alongside `.wav` files

2. **gui_audio_settings_panel.py**:
   - Added `_render_save_format_configuration()` method
   - Checkbox controls for format selection
   - Validation to ensure at least one format enabled
   - Integrated with existing config save system

3. **recorderConfig.json**:
   - Added `save_format` configuration section
   - Default: WAV enabled, NumPy disabled

### Backward Compatibility

- **Fully backward compatible**: Existing code works unchanged
- **Default behavior**: Only WAV files saved (same as before)
- **Opt-in feature**: NumPy format requires explicit configuration

## Performance Considerations

### File Size

- NumPy files are **~4x larger** than 16-bit WAV files
- For 1 second @ 48 kHz:
  - WAV: ~94 KB
  - NumPy: ~375 KB

### Storage Requirements

For a typical multi-channel recording session (8 channels, 50 measurements, 1 second each):

- **WAV only**: ~37 MB (8 channels × 50 meas × 3 files × 94 KB)
- **NumPy only**: ~150 MB (8 channels × 50 meas × 3 files × 375 KB)
- **Both formats**: ~187 MB

### Write Performance

- Minimal performance impact
- NumPy save is very fast (direct memory write)
- Both formats write in parallel (no sequential delay)

## Testing

A comprehensive test suite is included: `test_save_formats.py`

Run tests:
```bash
python test_save_formats.py
```

The test verifies:
- WAV format saving and loading
- NumPy format saving and loading
- Precision comparison between formats
- Configuration file format

## Future Enhancements

Potential improvements:

1. **Compressed NumPy**: Use `.npz` with compression
2. **HDF5 Format**: Better for very large files
3. **Float32 Option**: Balance between precision and size
4. **Per-File Type Config**: Different formats for raw/impulse/room
5. **Metadata Storage**: Embed recording parameters in files

## Documentation Updates

This feature is documented in:
- `SAVE_FORMAT_FEATURE.md` (this file)
- `TECHNICAL_DOCUMENTATION.md` (to be updated)
- GUI tooltips and help text

## Questions & Support

For questions or issues:
- Check configuration in `recorderConfig.json`
- Review GUI settings in Audio Settings panel
- Run `test_save_formats.py` to verify functionality
- Consult `TECHNICAL_DOCUMENTATION.md` for system architecture

---

**Version**: 1.0
**Date**: 2025-11-04
**Status**: Implemented and Tested ✓
