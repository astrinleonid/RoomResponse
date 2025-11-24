"""Quick single test without visualization."""
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from test_esprit_core_synthetic import run_single_test

# Ground truth parameters
fs = 787.815125
N = 260
L = 140
K = 12

f_true = np.array([120.0, 145.0, 168.0, 185.0, 210.0, 235.0])
Q_true = np.array([12.0, 20.0, 15.0, 8.0, 25.0, 18.0])
zeta_true = 1.0 / (2.0 * Q_true)
A_true = np.array([1.0, 0.9, 0.8, 0.7, 0.65, 0.6])
ph_true = np.array([0.2, -0.6, 0.9, -1.1, 0.7, -0.3])

print("######################################################################")
print("# SINGLE TEST: TLS-ESPRIT (no noise, no stabilization)")
print("######################################################################")

comparison = run_single_test(
    fs=fs,
    n_samples=N,
    frequencies=f_true,
    damping_ratios=zeta_true,
    amplitudes=A_true,
    phases=ph_true,
    window_length=L,
    model_order=K,
    noise_level=0.0,
    use_tls=True,
    use_stabilization=False,
    visualize=False
)

print(f"\nFinal result: {comparison['match_rate']*100:.1f}% match rate")
if comparison['match_rate'] >= 0.8:
    print("TEST PASSED!")
else:
    print("TEST FAILED!")
