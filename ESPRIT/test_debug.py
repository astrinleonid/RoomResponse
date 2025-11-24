"""Debug test to see raw poles before filtering."""
import numpy as np
import sys
sys.path.insert(0, 'ESPRIT')

from esprit_core import esprit_poles_tls_usubspace, build_hankel_matrix

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
K = H.shape[1]  # Number of columns in Hankel matrix

# Extract poles using TLS-U algorithm with model_order=12
poles_discrete, s_values = esprit_poles_tls_usubspace(H, 12, dt, use_gpu=False)

print(f"RAW POLES (discrete-time, {len(poles_discrete)} poles):")
print("="*70)
for i, lam in enumerate(poles_discrete):
    r = np.abs(lam)
    theta = np.angle(lam)
    f_est = theta * fs / (2 * np.pi)
    log_r = np.log(r)
    zeta_est = -log_r / np.sqrt(log_r**2 + theta**2) if theta != 0 else 0
    print(f"  Pole {i}: lam={lam.real:+.6f}{lam.imag:+.6f}j  r={r:.6f}  f={f_est:7.2f} Hz  zeta={zeta_est:.4f}")

print(f"\nExpected 6 modes at: {f_true} Hz")
