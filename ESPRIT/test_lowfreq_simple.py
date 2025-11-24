"""
Simple test: Why does esprit_core miss low-frequency modes that esprit.py detects?

Generate synthetic signal with modes at 25, 35, 45 Hz and see which get detected.
"""
import numpy as np
import sys
sys.path.insert(0, 'ESPRIT')

from esprit import build_hankel, TLS_ESPRIT_FromUs, LambdasToFZQ
from esprit_core import esprit_modal_identification

# Generate synthetic signal with LOW frequency modes
fs = 787.815125
dt = 1.0 / fs
N = 260
L = 140
K = 12

# THREE low-frequency modes: 25, 35, 45 Hz (test filtering boundaries)
f_true = np.array([25.0, 35.0, 45.0, 120.0, 185.0, 235.0])
Q_true = np.array([15.0, 18.0, 20.0, 12.0, 8.0, 18.0])
zeta_true = 1.0 / (2.0 * Q_true)
A_true = np.array([1.0, 0.9, 0.8, 0.7, 0.6, 0.5])
ph_true = np.array([0.0, 0.2, -0.3, 0.5, -0.7, 0.9])

print("="*80)
print("LOW FREQUENCY TEST: Modes at 25, 35, 45 Hz")
print("="*80)
print(f"\nGround truth ({len(f_true)} modes):")
for i, (f, q, z) in enumerate(zip(f_true, Q_true, zeta_true)):
    print(f"  Mode {i}: f={f:5.1f} Hz, Q={q:5.1f}, zeta={z:.4f}")

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

# ============================================================================
# Test 1: esprit.py - Print ALL candidates before filtering
# ============================================================================
print("\n" + "="*80)
print("TEST 1: esprit.py (TLS-U)")
print("="*80)

H = build_hankel(x, N, L)
U, S, Vh = np.linalg.svd(H, full_matrices=False)
Us = U[:, :K]

lam_re, lam_im = TLS_ESPRIT_FromUs(Us, L, L, 1, K)

print("\nLambdasToFZQ filtering (prints [MODE raw] for ALL candidates):")
F_ref, Q_ref, Z_ref = LambdasToFZQ(lam_re, lam_im, K, dt, max_modes=10)

sort_idx = np.argsort(F_ref)
F_ref = F_ref[sort_idx]
Q_ref = Q_ref[sort_idx]
Z_ref = Z_ref[sort_idx]

print(f"\n{len(F_ref)} modes ACCEPTED by esprit.py:")
for i, (f, q, z) in enumerate(zip(F_ref, Q_ref, Z_ref)):
    # Find closest ground truth
    closest_idx = np.argmin(np.abs(f_true - f))
    error = f - f_true[closest_idx]
    print(f"  Mode {i}: f={f:7.2f} Hz (closest: {f_true[closest_idx]:.1f}, error: {error:+5.2f}), Q={q:6.1f}, zeta={z:.4f}")

# ============================================================================
# Test 2: esprit_core.py
# ============================================================================
print("\n" + "="*80)
print("TEST 2: esprit_core.py (TLS-U)")
print("="*80)

signals = x.reshape(-1, 1)

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
    freq_range=(0, np.inf),
    min_freq=30.0  # <-- DEFAULT VALUE
)

F_core = result.frequencies
Z_core = result.damping_ratios
Q_core = 1.0 / (2.0 * Z_core)

sort_idx = np.argsort(F_core)
F_core = F_core[sort_idx]
Z_core = Z_core[sort_idx]
Q_core = Q_core[sort_idx]

print(f"\n{len(F_core)} modes detected by esprit_core:")
for i, (f, q, z) in enumerate(zip(F_core, Q_core, Z_core)):
    closest_idx = np.argmin(np.abs(f_true - f))
    error = f - f_true[closest_idx]
    print(f"  Mode {i}: f={f:7.2f} Hz (closest: {f_true[closest_idx]:.1f}, error: {error:+5.2f}), Q={q:6.1f}, zeta={z:.4f}")

# ============================================================================
# Comparison
# ============================================================================
print("\n" + "="*80)
print("COMPARISON")
print("="*80)

print(f"\nGround truth:   6 modes at [25.0, 35.0, 45.0, 120.0, 185.0, 235.0] Hz")
print(f"esprit.py:      {len(F_ref)} modes detected")
if len(F_ref) > 0:
    print(f"                Frequency range: [{F_ref.min():.1f}, {F_ref.max():.1f}] Hz")

print(f"esprit_core:    {len(F_core)} modes detected (with min_freq=30.0)")
if len(F_core) > 0:
    print(f"                Frequency range: [{F_core.min():.1f}, {F_core.max():.1f}] Hz")

# Check which ground truth modes are missing
print("\nMode detection analysis:")
for f_t in f_true:
    found_ref = any(abs(f - f_t) < 3.0 for f in F_ref)
    found_core = any(abs(f - f_t) < 3.0 for f in F_core)

    status_ref = "✓" if found_ref else "✗"
    status_core = "✓" if found_core else "✗"

    print(f"  f={f_t:5.1f} Hz:  esprit.py {status_ref}   esprit_core {status_core}")

# ============================================================================
# Test 3: esprit_core with min_freq=0 (disable filter)
# ============================================================================
print("\n" + "="*80)
print("TEST 3: esprit_core (TLS-U) with min_freq=0.0")
print("="*80)

result_nofilter = esprit_modal_identification(
    signals=signals,
    fs=fs,
    window_length=L,
    model_order=K,
    use_stabilization=False,
    use_tls=True,
    use_gpu=False,
    use_conjugate_pairing=True,
    max_damping=0.2,
    freq_range=(0, np.inf),
    min_freq=0.0  # <-- DISABLED
)

F_nofilter = result_nofilter.frequencies
Z_nofilter = result_nofilter.damping_ratios
Q_nofilter = 1.0 / (2.0 * Z_nofilter)

sort_idx = np.argsort(F_nofilter)
F_nofilter = F_nofilter[sort_idx]
Z_nofilter = Z_nofilter[sort_idx]
Q_nofilter = Q_nofilter[sort_idx]

print(f"\n{len(F_nofilter)} modes detected (no min_freq filter):")
for i, (f, q, z) in enumerate(zip(F_nofilter, Q_nofilter, Z_nofilter)):
    closest_idx = np.argmin(np.abs(f_true - f))
    error = f - f_true[closest_idx]
    print(f"  Mode {i}: f={f:7.2f} Hz (closest: {f_true[closest_idx]:.1f}, error: {error:+5.2f}), Q={q:6.1f}, zeta={z:.4f}")

if len(F_nofilter) > 0:
    print(f"\nFrequency range: [{F_nofilter.min():.1f}, {F_nofilter.max():.1f}] Hz")

print("\nMode detection analysis (min_freq=0):")
for f_t in f_true:
    found = any(abs(f - f_t) < 3.0 for f in F_nofilter)
    status = "✓" if found else "✗"
    print(f"  f={f_t:5.1f} Hz: {status}")
