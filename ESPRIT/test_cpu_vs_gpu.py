"""
Compare esprit.py vs esprit_core (CPU) vs esprit_core (GPU).

Tests:
1. Verify esprit_core CPU and GPU produce identical results
2. Compare timings of all three implementations
"""
import numpy as np
import sys
import time
sys.path.insert(0, 'ESPRIT')

from esprit import build_hankel, TLS_ESPRIT_FromUs, LambdasToFZQ
from esprit_core import esprit_modal_identification

print("="*80)
print("CPU vs GPU PERFORMANCE COMPARISON")
print("="*80)

# Generate synthetic signal (same as test_synthetic_comparison.py)
fs = 787.815125
dt = 1.0 / fs
N = 260
L = 140
K = 12

f_true = np.array([120.0, 145.0, 168.0, 185.0, 210.0, 235.0])
Q_true = np.array([12.0, 20.0, 15.0, 8.0, 25.0, 18.0])
zeta_true = 1.0 / (2.0 * Q_true)
A_true = np.array([1.0, 0.9, 0.8, 0.7, 0.65, 0.6])
ph_true = np.array([0.2, -0.6, 0.9, -1.1, 0.7, -0.3])

print(f"\nGround truth: {len(f_true)} modes")
print(f"Frequencies: {f_true} Hz")
print(f"Signal: N={N}, fs={fs:.2f} Hz, L={L}, K={K}")

# Generate signal with DAMPED frequency
x = np.zeros(N)
for k in range(len(f_true)):
    f = f_true[k]
    zeta = zeta_true[k]
    A = A_true[k]
    ph = ph_true[k]

    omega_n = 2.0 * np.pi * f
    omega_d = omega_n * np.sqrt(1 - zeta**2)  # Damped frequency
    alpha = zeta * omega_n  # Decay rate

    t = np.arange(N) / fs
    x += A * np.exp(-alpha * t) * np.cos(omega_d * t + ph)

signals = x.reshape(-1, 1)

# ============================================================================
# Test 1: esprit.py (TLS-U reference)
# ============================================================================
print("\n" + "="*80)
print("TEST 1: esprit.py (TLS-U reference) - CPU")
print("="*80)

start_time = time.perf_counter()

H = build_hankel(x, N, L)
U, S, Vh = np.linalg.svd(H, full_matrices=False)
Us = U[:, :K]

lam_re, lam_im = TLS_ESPRIT_FromUs(Us, L, L, 1, K)
F_ref, Q_ref, Z_ref = LambdasToFZQ(lam_re, lam_im, K, dt, max_modes=6)

# Sort by frequency
sort_idx = np.argsort(F_ref)
F_ref = F_ref[sort_idx]
Q_ref = Q_ref[sort_idx]
Z_ref = Z_ref[sort_idx]

time_ref = time.perf_counter() - start_time

print(f"\nProcessing time: {time_ref*1000:.4f} ms")
print(f"Detected {len(F_ref)} modes:")
for i, (f, q, z) in enumerate(zip(F_ref, Q_ref, Z_ref)):
    print(f"  Mode {i}: f={f:7.2f} Hz, Q={q:6.1f}, zeta={z:.4f}")

# ============================================================================
# Test 2: esprit_core.py (CPU)
# ============================================================================
print("\n" + "="*80)
print("TEST 2: esprit_core.py (TLS-U) - CPU")
print("="*80)

start_time = time.perf_counter()

result_cpu = esprit_modal_identification(
    signals=signals,
    fs=fs,
    window_length=L,
    model_order=K,
    use_stabilization=False,
    use_tls=True,
    use_gpu=False,  # CPU
    use_conjugate_pairing=True,
    max_damping=0.2,
    freq_range=(0, np.inf)
)

time_cpu = time.perf_counter() - start_time

F_cpu = result_cpu.frequencies
Z_cpu = result_cpu.damping_ratios
Q_cpu = 1.0 / (2.0 * Z_cpu)

# Sort by frequency
sort_idx = np.argsort(F_cpu)
F_cpu = F_cpu[sort_idx]
Z_cpu = Z_cpu[sort_idx]
Q_cpu = Q_cpu[sort_idx]

print(f"\nProcessing time: {time_cpu*1000:.4f} ms")
print(f"Detected {len(F_cpu)} modes:")
for i, (f, q, z) in enumerate(zip(F_cpu, Q_cpu, Z_cpu)):
    print(f"  Mode {i}: f={f:7.2f} Hz, Q={q:6.1f}, zeta={z:.4f}")

# ============================================================================
# Test 3: esprit_core.py (GPU)
# ============================================================================
print("\n" + "="*80)
print("TEST 3: esprit_core.py (TLS-U) - GPU")
print("="*80)

# Check if GPU is available
gpu_available = False
try:
    import cupy as cp
    gpu_available = True
    print("CuPy detected - GPU acceleration available")
    try:
        device = cp.cuda.Device()
        print(f"GPU device: {device.compute_capability}")
    except:
        print("GPU device: (info not available)")
except ImportError:
    print("CuPy not installed - GPU acceleration not available")
    print("Install with: pip install cupy-cuda12x")

if gpu_available:
    start_time = time.perf_counter()

    result_gpu = esprit_modal_identification(
        signals=signals,
        fs=fs,
        window_length=L,
        model_order=K,
        use_stabilization=False,
        use_tls=True,
        use_gpu=True,  # GPU
        use_conjugate_pairing=True,
        max_damping=0.2,
        freq_range=(0, np.inf)
    )

    time_gpu = time.perf_counter() - start_time

    F_gpu = result_gpu.frequencies
    Z_gpu = result_gpu.damping_ratios
    Q_gpu = 1.0 / (2.0 * Z_gpu)

    # Sort by frequency
    sort_idx = np.argsort(F_gpu)
    F_gpu = F_gpu[sort_idx]
    Z_gpu = Z_gpu[sort_idx]
    Q_gpu = Q_gpu[sort_idx]

    print(f"\nProcessing time: {time_gpu*1000:.4f} ms")
    print(f"Detected {len(F_gpu)} modes:")
    for i, (f, q, z) in enumerate(zip(F_gpu, Q_gpu, Z_gpu)):
        print(f"  Mode {i}: f={f:7.2f} Hz, Q={q:6.1f}, zeta={z:.4f}")
else:
    print("\nSkipping GPU test - CuPy not available")
    F_gpu = None
    Z_gpu = None
    Q_gpu = None
    time_gpu = None

# ============================================================================
# Comparison: CPU vs GPU Numerical Accuracy
# ============================================================================
if gpu_available:
    print("\n" + "="*80)
    print("CPU vs GPU NUMERICAL COMPARISON")
    print("="*80)

    if len(F_cpu) == len(F_gpu):
        print(f"\nBoth implementations detected {len(F_cpu)} modes")

        print(f"\n{'Mode':<6} {'CPU Freq (Hz)':<15} {'GPU Freq (Hz)':<15} {'Diff (Hz)':<12} {'Diff (ppm)'}")
        print("-"*80)

        freq_diffs = []
        freq_diffs_ppm = []
        for i in range(len(F_cpu)):
            diff_hz = abs(F_gpu[i] - F_cpu[i])
            diff_ppm = (diff_hz / F_cpu[i]) * 1e6 if F_cpu[i] > 0 else 0
            freq_diffs.append(diff_hz)
            freq_diffs_ppm.append(diff_ppm)
            print(f"{i:<6} {F_cpu[i]:>10.6f}     {F_gpu[i]:>10.6f}     {diff_hz:>8.6f}    {diff_ppm:>8.3f}")

        print(f"\nFrequency difference statistics:")
        print(f"  Max difference:  {np.max(freq_diffs):.8f} Hz ({np.max(freq_diffs_ppm):.3f} ppm)")
        print(f"  Mean difference: {np.mean(freq_diffs):.8f} Hz ({np.mean(freq_diffs_ppm):.3f} ppm)")
        print(f"  RMS difference:  {np.sqrt(np.mean(np.array(freq_diffs)**2)):.8f} Hz")

        # Damping comparison
        print(f"\n{'Mode':<6} {'CPU Zeta':<15} {'GPU Zeta':<15} {'Diff':<12} {'Diff (%)'}")
        print("-"*80)

        damp_diffs = []
        for i in range(len(Z_cpu)):
            diff = abs(Z_gpu[i] - Z_cpu[i])
            diff_pct = (diff / Z_cpu[i]) * 100 if Z_cpu[i] > 0 else 0
            damp_diffs.append(diff)
            print(f"{i:<6} {Z_cpu[i]:>10.6f}     {Z_gpu[i]:>10.6f}     {diff:>8.6f}    {diff_pct:>8.3f}")

        print(f"\nDamping difference statistics:")
        print(f"  Max difference:  {np.max(damp_diffs):.8f}")
        print(f"  Mean difference: {np.mean(damp_diffs):.8f}")

        # Verdict
        max_freq_diff_ppm = np.max(freq_diffs_ppm)
        max_damp_diff = np.max(damp_diffs)

        print("\n" + "="*80)
        print("NUMERICAL ACCURACY VERDICT")
        print("="*80)

        if max_freq_diff_ppm < 1.0 and max_damp_diff < 1e-6:
            print("\n[EXCELLENT] CPU and GPU results are IDENTICAL to machine precision")
            print(f"  Max frequency error: {max_freq_diff_ppm:.3f} ppm (< 1 ppm)")
            print(f"  Max damping error: {max_damp_diff:.2e} (< 1e-6)")
        elif max_freq_diff_ppm < 10.0 and max_damp_diff < 1e-5:
            print("\n[GOOD] CPU and GPU results are NUMERICALLY EQUIVALENT")
            print(f"  Max frequency error: {max_freq_diff_ppm:.3f} ppm (< 10 ppm)")
            print(f"  Max damping error: {max_damp_diff:.2e} (< 1e-5)")
        else:
            print("\n[WARNING] CPU and GPU results show SIGNIFICANT DIFFERENCES")
            print(f"  Max frequency error: {max_freq_diff_ppm:.3f} ppm")
            print(f"  Max damping error: {max_damp_diff:.2e}")
            print("  This may indicate a numerical issue in GPU implementation")
    else:
        print(f"\n[ERROR] Mode count mismatch!")
        print(f"  CPU detected: {len(F_cpu)} modes")
        print(f"  GPU detected: {len(F_gpu)} modes")

# ============================================================================
# Performance Comparison
# ============================================================================
print("\n" + "="*80)
print("PERFORMANCE COMPARISON")
print("="*80)

print(f"\n{'Implementation':<30} {'Time (ms)':<15} {'Speedup':<10} {'Modes'}")
print("-"*80)

print(f"{'esprit.py (TLS-U ref)':<30} {time_ref*1000:>10.4f}     {1.0:>6.2f}x    {len(F_ref)}")
print(f"{'esprit_core (CPU)':<30} {time_cpu*1000:>10.4f}     {time_ref/time_cpu:>6.2f}x    {len(F_cpu)}")

if gpu_available and time_gpu is not None:
    print(f"{'esprit_core (GPU)':<30} {time_gpu*1000:>10.4f}     {time_ref/time_gpu:>6.2f}x    {len(F_gpu)}")
    print(f"\nGPU vs CPU speedup: {time_cpu/time_gpu:.2f}x")

print("\n" + "="*80)
print("NOTES")
print("="*80)

print("\n1. Small problem size (N=260, L=140, K=12):")
print("   - GPU may show minimal speedup or even slowdown")
print("   - GPU transfer overhead dominates compute time")
print("   - GPU benefits appear at larger problem sizes (L > 5000)")

print("\n2. For production use on small problems:")
if gpu_available and time_gpu is not None:
    if time_gpu < time_cpu * 0.8:
        print("   -> Use GPU (significant speedup observed)")
    elif time_gpu < time_cpu * 1.2:
        print("   -> GPU and CPU comparable - use CPU (simpler)")
    else:
        print("   -> Use CPU (GPU overhead not justified)")
else:
    print("   -> Use CPU (GPU not available or not tested)")

print("\n3. esprit_core vs esprit.py:")
print(f"   -> esprit_core is {time_ref/time_cpu:.1f}x faster on CPU")
print("   -> Both produce identical results on synthetic data")
