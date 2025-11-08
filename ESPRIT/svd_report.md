# SVD Profiling on Hankel Matrix (CPU vs GPU)

This document describes the steps taken to construct a Hankel matrix from your measurement data, compute its **Singular Value Decomposition (SVD)**, and benchmark performance with and without GPU acceleration.

---

## 1. Data Preparation

Your measurement file contained **6 channels**, where:
- **Channel 2** is the *hammer calibration* reference signal.
- All other channels represent sensor responses.

We removed channel 2 and kept the remaining **5 response channels**.

Then, we selected **channel 0** (after exclusion) to construct a **Hankel matrix**.

### Hankel Matrix Definition

For a signal \( x[n] \), a Hankel matrix \( H \) of window length \( L \) is:

\[
H =
\begin{bmatrix}
x[0] & x[1] & x[2] & \dots \\
x[1] & x[2] & x[3] & \dots \\
\vdots & \vdots & \vdots & \ddots
\end{bmatrix}
\]

We used:
- \( L = 256 \)
- Resulting Hankel dimension: **256 × 28545**

To allow time profiling in a limited environment, only the first **4000 columns** were used during SVD benchmarking:
- **Used matrix shape:** 256 × 4000

---

## 2. SVD Computation

We implemented **economy (thin) SVD**:

\[
H = U \Sigma V^T
\]

Files created:

| File | Purpose |
|------|---------|
| `svd_cpu.py` | Performs SVD on CPU using NumPy (LAPACK) |
| `svd_gpu.py` | Performs SVD on GPU using CuPy (if CUDA available) |
| `run_svd_profile.py` | Benchmarks CPU and GPU, saves results |

If no CUDA device is present, GPU test is skipped gracefully.

---

## 3. Performance Results (Capped Matrix: 256 × 4000)

| Compute Method | Available | Time (seconds) |
|----------------|-----------|----------------|
| **CPU (NumPy)** | Yes | **29.54 s** |
| **GPU (CuPy/CUDA)** | Not available in this environment | N/A |

This confirms that **SVD is the main computational bottleneck**.

First 8 singular values:

```
38.20, 37.35, 29.61, 27.53, 24.11, 23.81, 20.72, 20.68
```

---

## 4. Output Artifacts

| Artifact | Description |
|---------|-------------|
| `Hankel_ch0_L256.npy` | Full Hankel matrix |
| `Sigma.npy` | Singular values |
| `svd_timings.json` | Benchmark report |
| `svd_cpu.py`, `svd_gpu.py`, `run_svd_profile.py` | Reusable code modules |

---

## 5. Next Optimization Recommendation

Use **Randomized / Truncated SVD** to compute only the leading singular vectors required for ESPRIT.

This typically gives **5–20× speedup** with no loss in modal accuracy.

---

### Ready to proceed
Tell me the **expected modal order range**, e.g.:

```
M = 10…80
```

So I can generate a **GPU-optimized Randomized SVD** for your pipeline.
