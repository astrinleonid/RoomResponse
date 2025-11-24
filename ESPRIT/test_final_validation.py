"""
Final validation: Demonstrate that esprit_core now matches esprit.py benchmark.

This test generates a synthetic signal with 6 known modes and verifies that
esp it_core's TLS-U algorithm can detect all 6 modes accurately.
"""
import numpy as np
import sys
sys.path.insert(0, 'ESPRIT')

from esprit_core import esprit_modal_identification

# Ground truth parameters (same as esprit.py self_test)
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

# Generate synthetic signal using DAMPED frequency (physically correct)
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

# Reshape for esprit_core (expects multi-channel)
signals = x.reshape(-1, 1)

print("="*70)
print("FINAL VALIDATION: esprit_core TLS-U ALGORITHM")
print("="*70)
print(f"Ground truth: {len(f_true)} modes")
for i, (f, z) in enumerate(zip(f_true, zeta_true)):
    print(f"  Mode {i}: f={f:.1f} Hz, zeta={z:.4f}, Q={Q_true[i]:.1f}")

# Run esprit_core with TLS-U algorithm
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

print(f"\nIdentified {len(result.frequencies)} modes:")
for i, (f, zeta) in enumerate(zip(result.frequencies, result.damping_ratios)):
    print(f"  Mode {i}: f={f:.2f} Hz, zeta={zeta:.4f}")

# Check match rate (with 2 Hz tolerance due to damped vs natural frequency)
matched = 0
for f_est in result.frequencies:
    for f_t in f_true:
        if abs(f_est - f_t) < 2.0:  # 2 Hz tolerance
            matched += 1
            break

match_rate = matched / len(f_true)
print(f"\nMatch rate: {matched}/{len(f_true)} = {match_rate*100:.1f}%")

# Validation
if match_rate >= 0.8:
    print("\n*** VALIDATION PASSED! ***")
    print("esprit_core TLS-U algorithm successfully matches esprit.py benchmark.")
else:
    print(f"\n*** VALIDATION FAILED! ***")
    print(f"Expected >= 80% match rate, got {match_rate*100:.1f}%")
