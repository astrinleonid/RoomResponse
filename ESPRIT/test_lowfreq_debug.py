"""Debug low frequency detection differences between esprit.py and esprit_core.py"""
import numpy as np
import sys
sys.path.insert(0, 'ESPRIT')

from esprit import build_hankel, TLS_ESPRIT_FromUs, LambdasToFZQ
from esprit_core import esprit_modal_identification, build_hankel_matrix
from esprit_core import esprit_poles_tls_usubspace, filter_poles_by_radius, validate_conjugate_pairs

# Load one of the real measurement files (same as comprehensive_comparison)
import json
from pathlib import Path

mp_file = Path("d:/repos/RoomResponse/measurements_piano/241118 piano measurements/DigPiano2.json")
with open(mp_file, 'r') as f:
    mp_data = json.load(f)

fs = mp_data['fs']
dt = 1.0 / fs
signals = np.array(mp_data['data'], dtype=np.float64)
if signals.ndim == 1:
    signals = signals.reshape(-1, 1)

T = len(signals)
L = 140  # Same as test parameters
K = 12

print("="*80)
print("LOW FREQUENCY DEBUG: esprit.py vs esprit_core.py")
print("="*80)
print(f"Signal: {mp_file.name}")
print(f"fs = {fs:.2f} Hz, T = {T}, L = {L}, K = {K}")
print()

# ============================================================================
# Test 1: esprit.py - Show ALL raw modes before filtering
# ============================================================================
print("="*80)
print("TEST 1: esprit.py (TLS-U) - RAW MODES")
print("="*80)

x = signals[:, 0]
H = build_hankel(x, T, L)
U, S, Vh = np.linalg.svd(H, full_matrices=False)
Us = U[:, :K]

lam_re, lam_im = TLS_ESPRIT_FromUs(Us, L, L, 1, K)

# Manually replicate LambdasToFZQ filtering to show what's rejected
print("\nShowing filtering process in LambdasToFZQ:")
print("(This will print [MODE raw] lines showing ALL candidates)")
F_ref, Q_ref, Z_ref = LambdasToFZQ(lam_re, lam_im, K, dt, max_modes=100)

print(f"\nAfter filtering: {len(F_ref)} modes accepted")
for i, (f, q, z) in enumerate(zip(F_ref, Q_ref, Z_ref)):
    print(f"  Mode {i}: f={f:7.2f} Hz, Q={q:6.1f}, zeta={z:.4f}")

if len(F_ref) > 0:
    print(f"\nFrequency range: [{F_ref.min():.1f}, {F_ref.max():.1f}] Hz")

# ============================================================================
# Test 2: esprit_core.py - Show filtering stages
# ============================================================================
print("\n" + "="*80)
print("TEST 2: esprit_core.py (TLS-U) - FILTERING STAGES")
print("="*80)

# Step 1: Extract raw discrete-time poles
H_core = build_hankel_matrix(x, L)
lam_all, _ = esprit_poles_tls_usubspace(H_core, K, dt, use_gpu=False)

print(f"\nStep 1: Raw discrete-time poles: {len(lam_all)}")

# Step 2: Radius filter
radius_mask = filter_poles_by_radius(lam_all, r_min=0.5, r_max=1.3)
lam_filtered = lam_all[radius_mask]

print(f"Step 2: After radius filter [0.5, 1.3]: {len(lam_filtered)} poles")

# Step 3: Conjugate pairing and conversion to continuous-time
poles_ct, pair_quality = validate_conjugate_pairs(lam_filtered, dt)

print(f"Step 3: After conjugate pairing: {len(poles_ct)} continuous-time poles")

# Step 4: Convert to frequencies
from esprit_core import poles_to_modal_params
frequencies, damping_ratios = poles_to_modal_params(poles_ct, fs)

print(f"\nAll continuous-time poles (before min_freq filter):")
for i, (pole, f, zeta) in enumerate(zip(poles_ct, frequencies, damping_ratios)):
    print(f"  Pole {i}: f={f:7.2f} Hz, zeta={zeta:.4f}, quality={pair_quality[i]:.6f}")

# Step 5: Apply min_freq filter (this is where low frequencies get rejected)
print(f"\nApplying min_freq=30.0 filter...")
from esprit_core import filter_poles

mask = filter_poles(poles_ct, frequencies, damping_ratios,
                   max_damping=0.2,
                   min_freq=30.0,
                   max_freq=np.inf)

frequencies_filtered = frequencies[mask]
damping_ratios_filtered = damping_ratios[mask]

print(f"Step 5: After filter_poles: {len(frequencies_filtered)} modes")
for i, (f, zeta) in enumerate(zip(frequencies_filtered, damping_ratios_filtered)):
    print(f"  Mode {i}: f={f:7.2f} Hz, zeta={zeta:.4f}")

if len(frequencies_filtered) > 0:
    print(f"\nFrequency range: [{frequencies_filtered.min():.1f}, {frequencies_filtered.max():.1f}] Hz")

# ============================================================================
# Comparison
# ============================================================================
print("\n" + "="*80)
print("COMPARISON SUMMARY")
print("="*80)

if len(F_ref) > 0 and len(frequencies_filtered) > 0:
    print(f"\nesprit.py:      {len(F_ref)} modes, f=[{F_ref.min():6.1f}, {F_ref.max():6.1f}] Hz")
    print(f"esprit_core:    {len(frequencies_filtered)} modes, f=[{frequencies_filtered.min():6.1f}, {frequencies_filtered.max():6.1f}] Hz")
    print(f"\nLowest frequency difference: {frequencies_filtered.min() - F_ref.min():.1f} Hz")
