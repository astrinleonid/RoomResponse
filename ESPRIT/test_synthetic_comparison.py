"""
Side-by-side comparison of esprit.py and esprit_core.py on synthetic data.

This test generates a synthetic signal with 6 known modes and compares
the performance of both implementations.
"""
import numpy as np
import sys
sys.path.insert(0, 'ESPRIT')

from esprit_core import esprit_modal_identification
import time

# Import esprit.py functions
from esprit import build_hankel, TLS_ESPRIT_FromUs, LambdasToFZQ

print("="*80)
print("SIDE-BY-SIDE COMPARISON: esprit.py vs esprit_core.py")
print("Synthetic Signal with 6 Known Modes")
print("="*80)

# Ground truth parameters
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

print("\nGround Truth Parameters:")
print(f"  Sampling frequency: {fs:.2f} Hz")
print(f"  Signal length: {N} samples ({N/fs:.3f} s)")
print(f"  Number of modes: {len(f_true)}")
print("\n  Mode parameters:")
print("  #   Freq (Hz)  Q        Damping    Amplitude")
print("-" * 60)
for i in range(len(f_true)):
    print(f"  {i}   {f_true[i]:<10.1f} {Q_true[i]:<8.1f} {zeta_true[i]:<10.4f} {A_true[i]:<10.2f}")

# Generate synthetic signal using DAMPED frequency (physically correct)
print(f"\nGenerating synthetic signal...")
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

# ============================================================================
# Test 1: esprit.py (TLS-U reference implementation)
# ============================================================================
print("\n" + "="*80)
print("TEST 1: esprit.py (TLS-U Reference)")
print("="*80)

start_time = time.time()

H = build_hankel(x, N, L)
U, S, Vh = np.linalg.svd(H, full_matrices=False)
Us = U[:, :K]

# TLS-ESPRIT from U-subspace
lam_re, lam_im = TLS_ESPRIT_FromUs(Us, L, L, 1, K)
F_ref, Q_ref, Z_ref = LambdasToFZQ(lam_re, lam_im, K, dt, max_modes=6)

# Sort by frequency
sort_idx = np.argsort(F_ref)
F_ref = F_ref[sort_idx]
Q_ref = Q_ref[sort_idx]
Z_ref = Z_ref[sort_idx]

time_ref = time.time() - start_time

print(f"\nIdentified {len(F_ref)} modes (processing time: {time_ref:.4f}s)")
for i, (f, q, z) in enumerate(zip(F_ref, Q_ref, Z_ref)):
    print(f"  Mode {i}: f={f:7.2f} Hz, Q={q:6.1f}, zeta={z:.4f}")

# Check matches
matched_ref = 0
for f_est in F_ref:
    for f_t in f_true:
        if abs(f_est - f_t) < 2.0:  # 2 Hz tolerance
            matched_ref += 1
            break

match_rate_ref = matched_ref / len(f_true)
print(f"\nMatch rate: {matched_ref}/{len(f_true)} = {match_rate_ref*100:.1f}%")

# ============================================================================
# Test 2: esprit_core.py (TLS-U implementation)
# ============================================================================
print("\n" + "="*80)
print("TEST 2: esprit_core.py (TLS-U)")
print("="*80)

# Reshape for esprit_core (expects multi-channel)
signals = x.reshape(-1, 1)

start_time = time.time()

result = esprit_modal_identification(
    signals=signals,
    fs=fs,
    window_length=L,
    model_order=K,
    use_stabilization=False,
    use_tls=True,
    use_gpu=False,
    use_conjugate_pairing=True,
    max_damping=0.2,
    freq_range=(0, np.inf)
)

time_core = time.time() - start_time

F_core = result.frequencies
Z_core = result.damping_ratios
Q_core = 1.0 / (2.0 * Z_core)

# Sort by frequency
sort_idx = np.argsort(F_core)
F_core = F_core[sort_idx]
Z_core = Z_core[sort_idx]
Q_core = Q_core[sort_idx]

print(f"\nIdentified {len(F_core)} modes (processing time: {time_core:.4f}s)")
for i, (f, q, z) in enumerate(zip(F_core, Q_core, Z_core)):
    print(f"  Mode {i}: f={f:7.2f} Hz, Q={q:6.1f}, zeta={z:.4f}")

# Check matches
matched_core = 0
for f_est in F_core:
    for f_t in f_true:
        if abs(f_est - f_t) < 2.0:  # 2 Hz tolerance
            matched_core += 1
            break

match_rate_core = matched_core / len(f_true)
print(f"\nMatch rate: {matched_core}/{len(f_true)} = {match_rate_core*100:.1f}%")

# ============================================================================
# Comparison Summary
# ============================================================================
print("\n" + "="*80)
print("COMPARISON SUMMARY")
print("="*80)

print(f"\n{'Implementation':<30} {'Modes':<10} {'Match Rate':<15} {'Time (s)':<10}")
print("-" * 80)
print(f"{'esprit.py (TLS-U ref)':<30} {len(F_ref):<10} {match_rate_ref*100:>6.1f}%{'':<8} {time_ref:<10.4f}")
print(f"{'esprit_core.py (TLS-U)':<30} {len(F_core):<10} {match_rate_core*100:>6.1f}%{'':<8} {time_core:<10.4f}")

# Direct comparison of identified modes
if len(F_ref) > 0 and len(F_core) > 0:
    print(f"\n" + "="*80)
    print("DIRECT MODE COMPARISON")
    print("="*80)
    print(f"\n{'Mode':<6} {'esprit.py f (Hz)':<18} {'esprit_core f (Hz)':<18} {'Diff (Hz)':<12}")
    print("-" * 80)

    n_compare = min(len(F_ref), len(F_core))
    freq_diffs = []
    for i in range(n_compare):
        diff = abs(F_core[i] - F_ref[i])
        freq_diffs.append(diff)
        print(f"{i:<6} {F_ref[i]:>8.2f}{'':<10} {F_core[i]:>8.2f}{'':<10} {diff:>8.4f}")

    if freq_diffs:
        print(f"\nAverage frequency difference: {np.mean(freq_diffs):.4f} Hz")
        print(f"Max frequency difference: {np.max(freq_diffs):.4f} Hz")
        print(f"RMS frequency difference: {np.sqrt(np.mean(np.array(freq_diffs)**2)):.4f} Hz")

# Final verdict
print("\n" + "="*80)
print("VALIDATION RESULT")
print("="*80)

if match_rate_ref >= 0.8 and match_rate_core >= 0.8:
    print("\n*** BOTH IMPLEMENTATIONS PASSED! ***")
    print(f"esprit.py: {match_rate_ref*100:.1f}% match rate")
    print(f"esprit_core.py: {match_rate_core*100:.1f}% match rate")

    if len(F_ref) == len(F_core) and len(freq_diffs) > 0:
        if np.max(freq_diffs) < 0.5:
            print("\n*** EXCELLENT AGREEMENT! ***")
            print(f"Maximum frequency difference: {np.max(freq_diffs):.4f} Hz")
        else:
            print(f"\nResults match ground truth but differ slightly between implementations")
            print(f"Maximum frequency difference: {np.max(freq_diffs):.4f} Hz")
else:
    print("\n*** VALIDATION FAILED ***")
    if match_rate_ref < 0.8:
        print(f"esprit.py: {match_rate_ref*100:.1f}% < 80%")
    if match_rate_core < 0.8:
        print(f"esprit_core.py: {match_rate_core*100:.1f}% < 80%")
