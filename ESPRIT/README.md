# ESPRIT Modal Identification for Piano Soundboard Analysis

Complete implementation of the ESPRIT (Estimation of Signal Parameters via Rotational Invariance Techniques) algorithm for extracting modal parameters from piano soundboard measurements.

---

## Overview

This package implements time-domain modal identification using the ESPRIT method, following the project-specific specification in [task1_mode_extraction (1).html](task1_mode_extraction%20(1).html). It supports:

- **Multi-channel impulse response analysis** (5 sensor channels)
- **Automatic preprocessing** (onset detection, windowing, filtering)
- **ESPRIT pole extraction** with GPU acceleration
- **Complex mode shape estimation**
- **Comprehensive visualization** and result export

---

## Files

### Core Implementation

| File | Description |
|------|-------------|
| **[esprit_core.py](esprit_core.py)** | Core ESPRIT algorithm: Hankel matrix construction, pole extraction, mode shape estimation |
| **[preprocessing.py](preprocessing.py)** | Signal preprocessing: onset detection, windowing, quality checks |
| **[example_esprit_analysis.py](example_esprit_analysis.py)** | Complete analysis pipeline with visualization and export |

### Profiling and Benchmarking

| File | Description |
|------|-------------|
| **[build_hankel.py](build_hankel.py)** | Standalone Hankel matrix builder |
| **[svd_cpu.py](svd_cpu.py)** | CPU-based SVD using NumPy/LAPACK |
| **[svd_gpu.py](svd_gpu.py)** | GPU-accelerated SVD using CuPy/CUDA |
| **[run_svd_profile.py](run_svd_profile.py)** | Benchmarking script for CPU vs GPU comparison |

### Documentation and Results

| File | Description |
|------|-------------|
| **[gpu_profiling_results.md](gpu_profiling_results.md)** | GPU profiling report (5.4× speedup) |
| **[profiling_comparison.md](profiling_comparison.md)** | Detailed matrix size comparison analysis |
| **[svd_report.md](svd_report.md)** | Original SVD analysis documentation |
| **[task1_mode_extraction (1).html](task1_mode_extraction%20(1).html)** | Project specification |

---

## Quick Start

### Installation

```bash
# Install required dependencies
pip install numpy scipy matplotlib pandas cupy-cuda12x
```

### Basic Usage

```bash
# Run ESPRIT analysis on a measurement file
python ESPRIT/example_esprit_analysis.py \
    piano_point_responses/point_70_response.txt \
    --order 30 \
    --freq-range 0 500 \
    --window-length-frac 0.33
```

### Output Files

The analysis generates:

- **`*_modal_table.csv`** - Modal parameters (frequencies, damping, poles)
- **`*_mode_XXX_shape.csv`** - Complex mode shapes for each mode
- **`*_modes.npz`** - Binary bundle with all results
- **`*_summary.json`** - Analysis summary
- **`preprocessing.png`** - Preprocessing diagnostic plots
- **`modal_results.png`** - Modal identification results
- **`reconstruction.png`** - Signal reconstruction quality

---

## Usage Examples

### Example 1: Basic Analysis

```python
from preprocessing import load_measurement_file, preprocess_measurement
from esprit_core import esprit_modal_identification

# Load data
force, responses = load_measurement_file('measurement.txt', skip_channel=2)

# Preprocess
decay_segments, metadata = preprocess_measurement(force, responses, fs=48000)

# Run ESPRIT
modal_params = esprit_modal_identification(
    decay_segments,
    fs=48000,
    model_order=30,
    freq_range=(0, 500)
)

# Results
print(f"Identified {len(modal_params.frequencies)} modes")
for k, (f, zeta) in enumerate(zip(modal_params.frequencies, modal_params.damping_ratios)):
    print(f"Mode {k}: f = {f:.2f} Hz, zeta = {zeta*100:.2f}%")
```

### Example 2: With GPU Acceleration

```python
# Enable GPU for large matrices
modal_params = esprit_modal_identification(
    decay_segments,
    fs=48000,
    model_order=30,
    use_gpu=True  # 5.4× faster for full-size matrices
)
```

### Example 3: Custom Preprocessing

```python
from preprocessing import PreprocessingConfig

# Custom preprocessing parameters
config = PreprocessingConfig(
    force_threshold_abs=1e-2,
    target_tail_db=-80,
    hp_cut_hz=5.0
)

decay_segments, metadata = preprocess_measurement(
    force, responses, fs=48000, config=config
)
```

---

## Algorithm Details

### ESPRIT Method

The ESPRIT algorithm extracts modal parameters using the shift invariance property of the signal subspace:

1. **Hankel Matrix Construction**
   ```
   H = [h[0]   h[1]   h[2]   ... h[K-1]  ]
       [h[1]   h[2]   h[3]   ... h[K]    ]
       [h[2]   h[3]   h[4]   ... h[K+1]  ]
       ...
       [h[L-1] h[L]   h[L+1] ... h[N-1]  ]
   ```

2. **SVD**: `H = U Σ V^T`

3. **Signal Subspace Extraction**: Keep first M singular vectors `E = U[:, :M]`

4. **Shift Invariance**:
   ```python
   E1 = E[:-1, :]  # Rows 0 to L-2
   E2 = E[1:, :]   # Rows 1 to L-1
   Phi = pinv(E1) @ E2
   ```

5. **Pole Extraction**: Eigenvalues of Phi give discrete-time poles
   ```python
   lambda_k = eig(Phi)
   s_k = ln(lambda_k) / dt  # Continuous-time poles
   ```

6. **Modal Parameters**:
   ```python
   f_k = |Im(s_k)| / (2π)           # Natural frequency
   zeta_k = -Re(s_k) / |s_k|        # Damping ratio
   ```

7. **Mode Shapes**: Least squares fit
   ```python
   y[n] ≈ Σ_k φ_k · exp(s_k · n · dt)
   ```

### Preprocessing Pipeline

1. **Onset Detection** - Finds impact start using force threshold
2. **Contact End Detection** - Identifies hammer separation
3. **Windowing** - Applies exponential decay window (-70 dB)
4. **High-pass Filtering** - Removes DC and low-frequency drift
5. **Quality Checks** - Detects clipping, double hits, estimates SNR

---

## Performance

### GPU Acceleration

| Matrix Size | CPU Time | GPU Time | Speedup |
|-------------|----------|----------|---------|
| 256 × 4,000 | 0.119 s | 0.223 s | 0.53× (slower) |
| 256 × 8,000 | 0.414 s | 0.206 s | **2.01×** |
| 256 × 28,545 (full) | 1.376 s | 0.256 s | **5.37×** |

**Key Finding:** GPU becomes beneficial at ~6,000 columns and provides 5.4× speedup on full matrices.

### Recommendations

- **Use CPU** for quick tests (< 5,000 samples)
- **Use GPU** for production analysis (> 6,000 samples)
- **Use Truncated SVD** for additional 5-20× speedup (computes only required modes)

---

## Configuration

### Command-Line Options

```bash
python example_esprit_analysis.py <measurement_file> [options]

Required:
  measurement_file          Path to measurement TSV file

Optional:
  --gpu                     Enable GPU acceleration
  --order, -M               Model order (default: 20)
  --freq-range f_min f_max  Frequency range in Hz (default: 0 1000)
  --fs                      Sampling frequency in Hz (default: 48000)
  --output-dir, -o          Output directory
  --window-length-frac      Hankel window fraction (default: 0.5)
```

### Parameter Guidelines

| Parameter | Recommended Range | Notes |
|-----------|------------------|-------|
| Model Order (M) | 10-50 | Should be > 2× expected modes |
| Window Length (L) | 30-60% of decay | Balance resolution vs. computation |
| Frequency Range | Based on band of interest | Filters poles outside range |
| Max Damping | 0.1-0.2 | Rejects spurious poles |

---

## Example Results

### Point 70 Analysis

**Input:**
- Duration: 0.600 s (28,800 samples @ 48 kHz)
- Channels: 5 response sensors
- Decay duration: 0.599 s

**Identified Modes:**

| Mode | Frequency (Hz) | Damping (%) |
|------|---------------|-------------|
| 0 | 384.84 | 2.54 |
| 1 | 306.85 | 4.85 |
| 2 | 264.46 | 3.01 |
| 3 | 207.59 | 9.96 |
| 4 | 126.40 | 4.75 |

**Quality:**
- Model order: 30
- Frequency range: 0-500 Hz
- SNR: ~-10 dB (pre-impact to decay ratio)

---

## File Format Specification

### Input Format (TSV)

6 columns: `force ch0 ch1 ch2_calibration ch3 ch4`

- Column 2 (ch2) is hammer calibration, skipped for analysis
- Remaining 5 columns are sensor responses
- Tab-delimited, scientific notation supported

### Output Formats

#### Modal Table (CSV)
```csv
mode_id,f_hz,zeta,sigma,omega,pole_real,pole_imag
0,384.835,0.0254,-61.337,2417.991,-61.337,2417.991
```

#### Mode Shapes (CSV per mode)
```csv
sensor_id,phi_real,phi_imag,phi_mag,phi_phase_deg
0,-0.0123,0.0456,0.0473,105.2
```

#### NPZ Bundle
```python
npz_file = np.load('modes.npz')
frequencies = npz_file['f_hz']  # shape: (K,)
damping = npz_file['zeta']      # shape: (K,)
poles = npz_file['poles']        # shape: (K,) complex
shapes = npz_file['phi']         # shape: (K, 5) complex
```

---

## Troubleshooting

### Common Issues

**1. GPU Out of Memory**
```
Solution: Reduce --window-length-frac or use CPU
```

**2. No Modes Found**
```
Solution: Adjust --freq-range or increase --order
```

**3. High Reconstruction Error**
```
Solution: Increase model order or check signal quality
```

**4. Double Hit Detected**
```
Solution: Review measurement technique, may need to reject measurement
```

### Signal Quality Indicators

- **SNR < -30 dB**: Poor quality, consider remeasurement
- **Damping > 20%**: Likely spurious poles
- **Reconstruction error > -20 dB**: Insufficient model order

---

## References

1. Roy, R., & Kailath, T. (1989). "ESPRIT—estimation of signal parameters via rotational invariance techniques." IEEE Transactions on acoustics, speech, and signal processing.

2. Peeters, B., & De Roeck, G. (2001). "Stochastic system identification for operational modal analysis: a review." Journal of Dynamic Systems, Measurement, and Control.

3. Project specification: [task1_mode_extraction (1).html](task1_mode_extraction%20(1).html)

---

## License

Part of the RoomResponse project for piano soundboard modal analysis.

---

## Contact

For issues or questions about this implementation, please refer to the project repository.
