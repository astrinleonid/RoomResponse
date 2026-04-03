"""
CPU vs GPU ESPRIT comparison on real Belarus data (Scenario 40).
Uses model_order=15, truncated to 4800 samples.
"""
import sys
import time
import numpy as np

sys.path.insert(0, 'D:/repos/RoomResponse/ESPRIT')

from esprit_core import (
    esprit_modal_identification,
    build_hankel_matrix,
    build_multichannel_hankel,
    esprit_poles,
)

# ── Load Belarus Scenario 40 ──────────────────────────────────────────────
data_dir = "D:/repos/RoomResponse/piano/Belarus-Scenario40-Measuement1/averaged_responses"
n_channels = 6
channels = []
for ch in range(n_channels):
    arr = np.load(f"{data_dir}/average_ch{ch}.npy").astype(np.float64)
    channels.append(arr)

signals_full = np.column_stack(channels)  # (28800, 6)
N_trunc = 4800
signals = signals_full[:N_trunc, :]

fs = 48000.0
dt = 1.0 / fs
model_order = 15

print("=" * 80)
print("CPU vs GPU ESPRIT — Belarus Scenario 40")
print("=" * 80)
print(f"Signal: {signals.shape[0]} samples x {signals.shape[1]} channels, fs={fs:.0f} Hz")
print(f"Model order: {model_order}")

# ── Warm-up GPU (CuPy JIT + CUDA context) ────────────────────────────────
try:
    import cupy as cp
    print(f"\nCuPy {cp.__version__} detected")
    dev = cp.cuda.Device(0)
    print(f"GPU: compute capability {dev.compute_capability}")
    # Warm-up: small SVD to initialize CUDA context
    _dummy = cp.linalg.svd(cp.eye(32, dtype=cp.float64), full_matrices=False)
    cp.cuda.Stream.null.synchronize()
    print("GPU warm-up done")
    gpu_available = True
except Exception as e:
    print(f"\nGPU not available: {e}")
    gpu_available = False

# ── CPU run ───────────────────────────────────────────────────────────────
print("\n" + "-" * 40)
print("CPU run")
print("-" * 40)

t0 = time.perf_counter()
result_cpu = esprit_modal_identification(
    signals=signals,
    fs=fs,
    model_order=model_order,
    use_gpu=False,
    use_tls=True,
    use_conjugate_pairing=True,
    max_damping=0.2,
    min_freq=30.0,
)
time_cpu = time.perf_counter() - t0

idx_cpu = np.argsort(result_cpu.frequencies)
F_cpu = result_cpu.frequencies[idx_cpu]
Z_cpu = result_cpu.damping_ratios[idx_cpu]
P_cpu = result_cpu.poles[idx_cpu]
SV_cpu = result_cpu.singular_values

print(f"Time: {time_cpu*1000:.2f} ms")
print(f"Modes found: {len(F_cpu)}")
for i, (f, z) in enumerate(zip(F_cpu, Z_cpu)):
    print(f"  {i:2d}  f={f:8.2f} Hz   zeta={z:.6f}")

# ── GPU run ───────────────────────────────────────────────────────────────
if not gpu_available:
    print("\nSkipping GPU — CuPy not available")
    sys.exit(1)

print("\n" + "-" * 40)
print("GPU run")
print("-" * 40)

t0 = time.perf_counter()
result_gpu = esprit_modal_identification(
    signals=signals,
    fs=fs,
    model_order=model_order,
    use_gpu=True,
    use_tls=True,
    use_conjugate_pairing=True,
    max_damping=0.2,
    min_freq=30.0,
)
time_gpu = time.perf_counter() - t0

idx_gpu = np.argsort(result_gpu.frequencies)
F_gpu = result_gpu.frequencies[idx_gpu]
Z_gpu = result_gpu.damping_ratios[idx_gpu]
P_gpu = result_gpu.poles[idx_gpu]
SV_gpu = result_gpu.singular_values

print(f"Time: {time_gpu*1000:.2f} ms")
print(f"Modes found: {len(F_gpu)}")
for i, (f, z) in enumerate(zip(F_gpu, Z_gpu)):
    print(f"  {i:2d}  f={f:8.2f} Hz   zeta={z:.6f}")

# ── Comparison ────────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("COMPARISON")
print("=" * 80)

# Mode count
if len(F_cpu) != len(F_gpu):
    print(f"\n[MISMATCH] CPU found {len(F_cpu)} modes, GPU found {len(F_gpu)} modes")
    # Try to match by closest frequency
    n_common = min(len(F_cpu), len(F_gpu))
    print(f"Comparing first {n_common} modes by frequency proximity")
else:
    n_common = len(F_cpu)
    print(f"\nBoth found {n_common} modes")

# Frequency comparison
print(f"\n{'Mode':<6} {'CPU Freq':<14} {'GPU Freq':<14} {'Abs Diff':<12} {'Rel Diff (ppm)'}")
print("-" * 70)

freq_rel_errors = []
for i in range(n_common):
    diff = abs(F_gpu[i] - F_cpu[i])
    rel = diff / F_cpu[i] * 1e6 if F_cpu[i] > 0 else 0
    freq_rel_errors.append(rel)
    print(f"{i:<6} {F_cpu[i]:>10.4f} Hz  {F_gpu[i]:>10.4f} Hz  {diff:>10.6f}  {rel:>10.3f}")

# Damping comparison
print(f"\n{'Mode':<6} {'CPU Zeta':<14} {'GPU Zeta':<14} {'Abs Diff':<12} {'Rel Diff (%)'}")
print("-" * 70)

damp_rel_errors = []
for i in range(n_common):
    diff = abs(Z_gpu[i] - Z_cpu[i])
    rel = diff / abs(Z_cpu[i]) * 100 if abs(Z_cpu[i]) > 0 else 0
    damp_rel_errors.append(rel)
    print(f"{i:<6} {Z_cpu[i]:>10.8f}    {Z_gpu[i]:>10.8f}    {diff:>10.2e}  {rel:>10.6f}")

# Singular value comparison
sv_diff = np.max(np.abs(SV_cpu - SV_gpu)) / np.max(np.abs(SV_cpu)) if len(SV_cpu) == len(SV_gpu) else float('nan')
print(f"\nSingular values max relative diff: {sv_diff:.2e}")

# Poles comparison
print(f"\n{'Mode':<6} {'CPU pole (real)':<20} {'GPU pole (real)':<20} {'Pole diff (abs)'}")
print("-" * 70)
for i in range(n_common):
    diff = abs(P_gpu[i] - P_cpu[i])
    print(f"{i:<6} {P_cpu[i].real:>16.6f}    {P_gpu[i].real:>16.6f}    {diff:>12.2e}")

# Timing
print(f"\n{'='*80}")
print("TIMING")
print(f"{'='*80}")
print(f"CPU: {time_cpu*1000:.2f} ms")
print(f"GPU: {time_gpu*1000:.2f} ms")
speedup = time_cpu / time_gpu if time_gpu > 0 else float('inf')
print(f"Speedup: {speedup:.2f}x {'(GPU faster)' if speedup > 1 else '(CPU faster)'}")

# Verdict
print(f"\n{'='*80}")
print("VERDICT")
print(f"{'='*80}")

max_freq_ppm = max(freq_rel_errors) if freq_rel_errors else 0
max_damp_pct = max(damp_rel_errors) if damp_rel_errors else 0

if len(F_cpu) == len(F_gpu) and max_freq_ppm < 1.0 and max_damp_pct < 0.0001:
    print("[PASS] Results are IDENTICAL within floating-point tolerance")
elif len(F_cpu) == len(F_gpu) and max_freq_ppm < 100 and max_damp_pct < 0.01:
    print("[PASS] Results are numerically equivalent (minor FP differences)")
else:
    print("[FAIL] Results differ significantly")

print(f"  Max frequency relative error: {max_freq_ppm:.3f} ppm")
print(f"  Max damping relative error:   {max_damp_pct:.6f} %")
print(f"  Mode count match: {'YES' if len(F_cpu) == len(F_gpu) else 'NO'}")
