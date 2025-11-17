# Multi-Band and Multi-Point ESPRIT Analysis

## Overview

This document describes the new multi-band and multi-point analysis capabilities that match the architecture of the reference `esprit.py` implementation.

## New Features

### 1. Multi-Band Processing ([band_processing.py](band_processing.py))

Process signals across multiple frequency bands with band-specific filtering, decimation, and pre-emphasis.

**Key Concepts:**
- **Band-specific filtering**: Different bands use different Butterworth filter orders
- **Decimation**: Lower frequency bands are decimated to reduce computational cost
- **Exponential pre-emphasis**: Mild positive exponential to compensate for signal decay

**Standard Band Configuration** (matching esprit.py):

| Band | Frequency Range | Filter Order | Decimation | Exp Factor |
|------|----------------|--------------|------------|------------|
| Low | 30-200 Hz | 4 | 4x | 0.3 |
| Mid-Low | 150-500 Hz | 5 | 2x | 0.2 |
| Mid-High | 400-1500 Hz | 6 | 1x | 0.1 |
| High | 1200-5000 Hz | 8 | 1x | 0.05 |

**Why Multi-Band?**
- **Improved frequency resolution**: Bandpass filtering removes out-of-band noise
- **Computational efficiency**: Decimation reduces data size for lower frequencies
- **Better SNR**: Exponential pre-emphasis compensates for natural signal decay
- **Targeted analysis**: Each band optimized for its frequency content

**Usage Example:**

```python
from band_processing import STANDARD_BANDS, process_band, select_bands_for_range

# Select bands for target frequency range
freq_range = (0, 500)  # Hz
selected_bands = select_bands_for_range(freq_range, STANDARD_BANDS)

# Process each band
for band in selected_bands:
    processed, fs_band, metadata = process_band(
        signals, fs=48000, band=band, apply_preemphasis=True
    )

    # Run ESPRIT on this band
    result = esprit_modal_identification(
        processed, fs=fs_band,
        freq_range=(band.f_min, band.f_max),
        ...
    )
```

### 2. Multi-Point Stabilization ([stabilization.py](stabilization.py))

Identify stable modes by clustering modal parameters across multiple measurements (excitation points).

**Key Concepts:**
- **Spatial stabilization**: Modes that appear at multiple excitation points are physical
- **Clustering**: Hierarchical clustering groups similar modes (by frequency and damping)
- **Mode shape averaging**: Aligns and averages mode shapes with phase correction
- **Detection rate**: Fraction of measurement points where mode was identified

**Stabilization Criteria:**
- **Frequency tolerance**: ±2 Hz (default)
- **Damping tolerance**: ±0.05 (default)
- **Minimum detections**: Mode must appear in ≥3 points (default)

**Usage Example:**

```python
from stabilization import multipoint_stabilization
from esprit_core import esprit_modal_identification

# Load multiple measurements
measurements = [data_point_00, data_point_10, data_point_20, ...]

# ESPRIT parameters
esprit_params = {
    'model_order': 30,
    'use_tls': True,
    'use_conjugate_pairing': True,
    'freq_range': (0, 500)
}

# Run multi-point stabilization
stable_modes = multipoint_stabilization(
    measurements,
    fs=48000,
    esprit_function=esprit_modal_identification,
    esprit_params=esprit_params,
    freq_tol_hz=2.0,
    min_detections=3
)

# Results
for mode in stable_modes:
    print(f"f={mode.frequency:.2f} Hz, "
          f"ζ={mode.damping:.4f}, "
          f"detected in {mode.n_detections} points")
```

### 3. Combined Multi-Band + Multi-Point Analysis

The most comprehensive approach: process multiple measurements across multiple frequency bands.

**Usage Example:**

```python
from stabilization import multiband_multipoint_stabilization
from band_processing import STANDARD_BANDS, process_band

stable_modes = multiband_multipoint_stabilization(
    measurement_data=measurements,
    fs=48000,
    esprit_function=esprit_modal_identification,
    esprit_params={'model_order': 30, 'use_tls': True, ...},
    bands=STANDARD_BANDS,
    band_processor=process_band,
    freq_tol_hz=2.0,
    min_detections=2  # Lower threshold since modes detected across bands too
)
```

## Comparison with esprit.py

### Similarities ✅

| Feature | esprit.py | Our Implementation |
|---------|-----------|-------------------|
| **TLS-ESPRIT** | Yes | ✅ Yes |
| **Conjugate pairing** | Yes | ✅ Yes |
| **Pole radius filtering** | Yes (0.5-1.3) | ✅ Yes (0.5-1.3) |
| **Multi-band processing** | 4 bands | ✅ Configurable bands |
| **Band decimation** | Yes | ✅ Yes |
| **Exponential pre-emphasis** | Mild (0.05-0.3) | ✅ Same |
| **Multi-point stabilization** | Yes | ✅ Yes |
| **Mode shape averaging** | Yes | ✅ Yes |

### Differences ⚠️

| Feature | esprit.py | Our Implementation | Notes |
|---------|-----------|-------------------|-------|
| **Data structure** | Cube (R, M, T) | List of (T, M) | esprit.py assumes pre-loaded cube |
| **Stabilization method** | Spatial (R points) | Spatial + parameter grid | Both valid |
| **Default bands** | Always 4 bands | Auto-select by freq_range | More flexible |
| **Multi-channel** | Always stacked | Opt-in | Performance trade-off |

## Test Scripts

### test_multiband.py

Tests multi-band processing on a single measurement.

**Usage:**
```bash
python ESPRIT/test_multiband.py piano_point_responses/70.txt
```

**What it tests:**
- Band selection for frequency range
- Band-specific preprocessing (filtering, decimation, pre-emphasis)
- ESPRIT analysis on each band
- Cross-band stabilization (modes appearing in multiple bands)

**Expected output:**
- Modes identified in each band
- Cross-band stable modes
- Comparison with single-band approach

### test_multipoint.py

Tests multi-point stabilization across multiple measurements.

**Usage:**
```bash
python ESPRIT/test_multipoint.py piano_point_responses 10
```

**What it tests:**
- Loading multiple measurement files
- Running ESPRIT on each measurement
- Spatial stabilization (clustering modes across points)
- Detection rate analysis

**Expected output:**
- Stable modes across multiple excitation points
- Detection rate (% of points where each mode appears)
- Frequency/damping standard deviation

### test_single_vs_multiband.py

Direct comparison between single-band and multi-band approaches.

**Usage:**
```bash
python ESPRIT/test_single_vs_multiband.py piano_point_responses/70.txt
```

**What it tests:**
- Single-band ESPRIT (full range, no preprocessing)
- Multi-band ESPRIT on Low band (30-200 Hz)
- Multi-band ESPRIT on Mid-Low band (150-500 Hz)
- Total modes from multi-band vs single-band

**Expected output:**
- Mode count comparison
- Demonstration of multi-band advantages

## When to Use Which Approach

### Single-Band ESPRIT (Original)
**Use when:**
- Single measurement, narrow frequency range
- Quick analysis needed
- Computational resources limited

**Command:**
```python
result = esprit_modal_identification(
    signals, fs=48000,
    freq_range=(0, 500),
    use_tls=True,
    use_conjugate_pairing=True
)
```

### Multi-Band ESPRIT
**Use when:**
- Wide frequency range (e.g., 30-5000 Hz)
- Need better frequency resolution
- Signal has strong low-frequency content that could be decimated

**Command:**
```python
# See test_multiband.py for full example
bands = select_bands_for_range(freq_range, STANDARD_BANDS)
for band in bands:
    processed, fs_band, _ = process_band(signals, fs, band)
    result = esprit_modal_identification(processed, fs_band, ...)
```

### Multi-Point Stabilization
**Use when:**
- Multiple measurements at different excitation points
- Need to identify global structural modes
- Want to reject location-specific artifacts

**Command:**
```python
stable_modes = multipoint_stabilization(
    measurements, fs=48000,
    esprit_function=esprit_modal_identification,
    esprit_params={...}
)
```

### Multi-Band + Multi-Point (esprit.py equivalent)
**Use when:**
- Multiple measurements, wide frequency range
- Maximum robustness required
- Production/research-quality analysis

**Command:**
```python
stable_modes = multiband_multipoint_stabilization(
    measurements, fs=48000,
    esprit_function=esprit_modal_identification,
    esprit_params={...},
    bands=STANDARD_BANDS,
    band_processor=process_band
)
```

## Implementation Notes

### Band Overlap

Standard bands have intentional overlap (e.g., Low: 30-200 Hz, Mid-Low: 150-500 Hz).

**Why overlap?**
- Modes near band edges detected in multiple bands → cross-validation
- Increases robustness of stabilization
- Ensures no "blind spots" between bands

**Handling duplicates:**
The stabilization clustering automatically merges modes detected in multiple bands
if they fall within frequency/damping tolerances.

### Exponential Pre-Emphasis vs Windowing

**⚠️ IMPORTANT DISTINCTION:**

**Exponential PRE-EMPHASIS (used here):**
```python
window = exp(+alpha * t)  # POSITIVE factor, mild (0.05-0.3)
signal_enhanced = signal * window
```
- Applied BEFORE ESPRIT
- Compensates for natural decay
- Improves SNR for later modes
- Does NOT corrupt damping estimates

**Exponential WINDOWING (removed from old code):**
```python
window = exp(-alpha * t)  # NEGATIVE factor, strong (-70dB)
signal_corrupted = signal * window
```
- Was applied during preprocessing
- Artificially increases damping by 30-50%
- CORRUPTS modal parameters
- **We removed this!**

### Numerical Stability

If you encounter SVD warnings (`init_gesdd failed`):

1. **Reduce window length** for problematic bands
2. **Check signal quality** (SNR, duration)
3. **Try LS-ESPRIT** (`use_tls=False`) for comparison
4. **Reduce model order** if needed

## Performance Considerations

### Computational Cost

**Single-band:**
- Fast (< 1 second per measurement)
- Linear in number of measurements

**Multi-band (4 bands):**
- 4x slower than single-band (per measurement)
- But: decimation reduces cost for low-frequency bands
- Net: ~3x slower than single-band

**Multi-point (N measurements):**
- N × single-band cost
- Plus clustering overhead (negligible)

**Multi-band + Multi-point (N measurements, 4 bands):**
- N × 4 × single-band cost
- Example: 10 measurements, 4 bands → ~40x single-band analysis time
- For TLS-ESPRIT: expect several minutes for comprehensive analysis

### Memory Usage

**Multi-channel Hankel with TLS:**
- TLS requires `full_matrices=True` in SVD
- Memory scales as O(L² × M²) where L=window length, M=n_channels
- For 5 channels, L=2000: requires ~38 GB (too large!)
- **Solution**: Reduce window length for multi-channel (use L=1000 or less)

## Future Enhancements

Potential improvements matching full esprit.py functionality:

1. **Automatic model order selection**: Grid search over model orders with stabilization
2. **Mode shape visualization**: Plot mode shapes in 2D/3D
3. **Participation factors**: Identify dominant modes for each excitation point
4. **GUI for stabilization diagram**: Interactive mode selection
5. **Batch processing utilities**: Process entire measurement directories
6. **Export to standard formats**: Universal File Format (UFF) for modal data

## References

1. Roy, R., & Kailath, T. (1989). "ESPRIT—estimation of signal parameters via rotational invariance techniques."
2. Working `esprit.py` implementation (reference)
3. Reynders, E. (2012). "System Identification Methods for (Operational) Modal Analysis"
4. [TLS_IMPROVEMENTS_IMPLEMENTED.md](TLS_IMPROVEMENTS_IMPLEMENTED.md) - Original TLS-ESPRIT improvements

---

**Date**: 2025-11-17
**Status**: Implemented and tested
**Compatibility**: Fully compatible with existing `esprit_core.py` API
