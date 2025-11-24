"""Debug conjugate pairing to see why only 4 modes are detected."""
import numpy as np
import sys
sys.path.insert(0, 'ESPRIT')

from esprit_core import (esprit_poles_tls_usubspace, build_hankel_matrix,
                          filter_poles_by_radius, validate_conjugate_pairs)

# Generate synthetic signal (matching test parameters)
fs = 787.815125
dt = 1.0 / fs
N = 260
L = 140

f_true = np.array([120.0, 145.0, 168.0, 185.0, 210.0, 235.0])
Q_true = np.array([12.0, 20.0, 15.0, 8.0, 25.0, 18.0])
zeta_true = 1.0 / (2.0 * Q_true)
A_true = np.array([1.0, 0.9, 0.8, 0.7, 0.65, 0.6])
ph_true = np.array([0.2, -0.6, 0.9, -1.1, 0.7, -0.3])

# Synthesize signal
x = np.zeros(N)
for k in range(len(f_true)):
    f = f_true[k]
    Q = Q_true[k]
    A = A_true[k]
    ph = ph_true[k]
    w = 2.0 * np.pi * f
    alpha = w / (2.0 * Q)
    t = np.arange(N) / fs
    x += A * np.exp(-alpha * t) * np.cos(w * t + ph)

# Build Hankel matrix
H = build_hankel_matrix(x, L)

# Extract poles using TLS-U algorithm
poles_discrete_all, _ = esprit_poles_tls_usubspace(H, 12, dt, use_gpu=False)

print(f"Step 1: Generated {len(poles_discrete_all)} raw discrete-time poles")

# Apply radius filter (matching esprit_modal_identification)
radius_mask = filter_poles_by_radius(poles_discrete_all, r_min=0.5, r_max=1.3)
poles_discrete_filtered = poles_discrete_all[radius_mask]

print(f"Step 2: After radius filter [{0.5}, {1.3}]: {len(poles_discrete_filtered)} poles")
for i, lam in enumerate(poles_discrete_filtered):
    r = np.abs(lam)
    print(f"  Pole {i}: r={r:.6f}, re={lam.real:+.6f}, im={lam.imag:+.6f}")

# Apply conjugate pairing
poles_ct, pair_quality = validate_conjugate_pairs(poles_discrete_filtered, dt)

print(f"\nStep 3: After conjugate pairing: {len(poles_ct)} continuous-time poles")
for i, pole in enumerate(poles_ct):
    f_est = np.abs(pole.imag) / (2 * np.pi)
    zeta_est = -pole.real / np.abs(pole)
    print(f"  Pole {i}: f={f_est:.2f} Hz, zeta={zeta_est:.4f}, quality={pair_quality[i]:.6f}")

print(f"\nExpected 6 modes at: {f_true} Hz")
print(f"Result: {len(poles_ct)} / 6 modes detected")
