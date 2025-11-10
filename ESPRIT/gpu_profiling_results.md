# GPU Profiling Results for ESPRIT Analysis

**Date:** 2025-11-08
**System:** RoomResponse - Piano Point Measurement Analysis
**Test Data:** piano_point_responses/point_70_response.txt

---

## Executive Summary

GPU acceleration provides **5.4× speedup** for full-size Hankel matrix SVD operations, making it highly beneficial for ESPRIT modal analysis.

---

## Test Configuration

### Input Data
- **Source file:** `piano_point_responses/point_70_response.txt`
- **Total channels:** 6 (1 calibration channel excluded)
- **Samples per channel:** 28,800
- **Channel used:** Channel 0 (after removing calibration channel 2)

### Hankel Matrix
- **Window length (L):** 256
- **Matrix dimensions:** 256 × 28,545
- **Matrix size:** 55.75 MB
- **Construction method:** Stride-based efficient construction

### Hardware
- **GPU:** CUDA-capable device
- **CUDA Version:** 12.x
- **CuPy Version:** 13.6.0
- **CPU:** NumPy with LAPACK backend

---

## Performance Results

### Full Matrix (256 × 28,545)

| Method | Time (seconds) | Speedup |
|--------|----------------|---------|
| **CPU (NumPy/LAPACK)** | 1.376 s | 1.0× (baseline) |
| **GPU (CuPy/CUDA)** | 0.256 s | **5.4×** |

**Key Finding:** GPU provides substantial speedup on full-size matrices.

### Small Matrix (256 × 4,000)

| Method | Time (seconds) | Speedup |
|--------|----------------|---------|
| **CPU (NumPy/LAPACK)** | 0.119 s | 1.0× (baseline) |
| **GPU (CuPy/CUDA)** | 0.223 s | **0.5×** (slower) |

**Key Finding:** For small matrices, CPU is faster due to GPU memory transfer overhead.

---

## Singular Value Validation

First 8 singular values from full matrix SVD:

```
σ₁ = 62.72
σ₂ = 55.16
σ₃ = 42.55
σ₄ = 38.39
σ₅ = 33.90
σ₆ = 33.48
σ₇ = 26.73
σ₈ = 25.04
```

These values are consistent between CPU and GPU implementations (verified numerically).

---

## Analysis and Recommendations

### When to Use GPU

✅ **Use GPU for:**
- Full-size Hankel matrices (256 × 28,545) → **5.4× faster**
- Production ESPRIT analysis on complete datasets
- Batch processing of multiple measurement points
- Real-time analysis requiring low latency

❌ **Don't use GPU for:**
- Small/capped matrices (< 5,000 columns)
- Quick exploratory analysis with truncated data
- Systems without CUDA support

### GPU Overhead Analysis

The crossover point where GPU becomes beneficial is approximately:
- **Matrix columns > 5,000** → Use GPU
- **Matrix columns < 5,000** → Use CPU

This is due to:
1. Host-to-device memory transfer time
2. Kernel launch overhead
3. Device-to-host result transfer time

### Further Optimization Opportunities

The report in [svd_report.md](ESPRIT/svd_report.md) recommends **Randomized/Truncated SVD** for additional speedup:

1. **Current approach:** Full SVD computes all 256 singular values
2. **Recommended:** Truncated SVD computing only M leading modes (M = 10-80)
3. **Expected speedup:** Additional 5-20× on top of GPU acceleration
4. **Combined potential:** Up to 100× faster than baseline CPU full SVD

---

## Files Generated

| File | Description | Size |
|------|-------------|------|
| `ESPRIT/Hankel_ch0_L256.npy` | Full Hankel matrix | 55.75 MB |
| `ESPRIT/Sigma.npy` | Singular values (256) | 2 KB |
| `ESPRIT/svd_timings.json` | Detailed profiling results | 1 KB |
| `ESPRIT/build_hankel.py` | Hankel matrix construction code | — |
| `ESPRIT/svd_cpu.py` | CPU SVD implementation | — |
| `ESPRIT/svd_gpu.py` | GPU SVD implementation | — |
| `ESPRIT/run_svd_profile.py` | Profiling script | — |

---

## Usage Instructions

### Build Hankel Matrix
```bash
python ESPRIT/build_hankel.py piano_point_responses/point_70_response.txt ESPRIT/Hankel_ch0_L256.npy --window_length 256 --channel 0
```

### Run Profiling
```bash
# Full matrix
python ESPRIT/run_svd_profile.py ESPRIT/Hankel_ch0_L256.npy --full

# Capped matrix (default: 4000 columns)
python ESPRIT/run_svd_profile.py ESPRIT/Hankel_ch0_L256.npy
```

### Use GPU SVD in Your Code
```python
from svd_gpu import svd_gpu
import numpy as np

# Load or create Hankel matrix
H = np.load("ESPRIT/Hankel_ch0_L256.npy")

# Compute SVD on GPU
U, s, Vt, elapsed_sec, gpu_used = svd_gpu(H)

if gpu_used:
    print(f"GPU SVD completed in {elapsed_sec:.3f} seconds")
else:
    print("GPU not available, falling back to CPU")
```

---

## Conclusion

For your ESPRIT modal analysis pipeline:
- **GPU acceleration is highly beneficial** with a **5.4× speedup** on full-size matrices
- The investment in GPU support is worthwhile for production analysis
- Consider implementing **Randomized SVD** for even greater performance gains
- For batch processing multiple piano points, GPU will provide significant time savings

**Next Step:** Implement truncated SVD to compute only the required modal orders (M = 10-80) instead of all 256 singular values.
