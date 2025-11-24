"""
Analyze stabilization consistency from comprehensive_comparison.py output.

This script directly analyzes the output that was already generated.
"""

# From the comprehensive comparison output:
results = {
    'point_57_response': {'tls': 15, 'ls': 15, 'tls_nostab': 15, 'ls_stab': 21},
    'point_60_response': {'tls': 15, 'ls': 15, 'tls_nostab': 15, 'ls_stab': 26},
    'point_65_response': {'tls': 15, 'ls': 15, 'tls_nostab': 15, 'ls_stab': 33},
    'point_70_response': {'tls': 15, 'ls': 15, 'tls_nostab': 15, 'ls_stab': 26},
    'point_74_response': {'tls': 15, 'ls': 15, 'tls_nostab': 15, 'ls_stab': 27},
    'point_79_response': {'tls': 15, 'ls': 14, 'tls_nostab': 15, 'ls_stab': 20},
    'point_80_response': {'tls': 15, 'ls': 15, 'tls_nostab': 15, 'ls_stab': 24},
    'point_81_response': {'tls': 15, 'ls': 15, 'tls_nostab': 15, 'ls_stab': 25},
    'point_82_response': {'tls': 15, 'ls': 15, 'tls_nostab': 15, 'ls_stab': 25},
    'point_83_response': {'tls': 15, 'ls': 15, 'tls_nostab': 15, 'ls_stab': 21},
    'point_84_response': {'tls': 15, 'ls': 15, 'tls_nostab': 15, 'ls_stab': 24},
}

import numpy as np
import matplotlib.pyplot as plt

print("="*80)
print("STABILIZATION MODE COUNT ANALYSIS")
print("="*80)

# Extract mode counts
points = list(results.keys())
n_points = len(points)

tls_counts = np.array([results[p]['tls'] for p in points])
ls_counts = np.array([results[p]['ls'] for p in points])
tls_nostab_counts = np.array([results[p]['tls_nostab'] for p in points])
ls_stab_counts = np.array([results[p]['ls_stab'] for p in points])

# Compute statistics
print(f"\nNumber of measurement points: {n_points}")
print()
print("Mode count statistics:")
print(f"{'Implementation':<30} {'Mean':<8} {'Std':<8} {'Min':<8} {'Max':<8}")
print("-"*80)

for name, counts in [
    ('esprit.py (TLS-U)', tls_counts),
    ('esprit_core (LS, no stab)', ls_counts),
    ('esprit_core (TLS, no stab)', tls_nostab_counts),
    ('esprit_core (LS + stab)', ls_stab_counts)
]:
    print(f"{name:<30} {np.mean(counts):>6.1f}  {np.std(counts):>6.1f}  "
          f"{np.min(counts):>6d}  {np.max(counts):>6d}")

# Analyze extra modes from stabilization
extra_modes = ls_stab_counts - tls_nostab_counts

print()
print("="*80)
print("EXTRA MODES FROM STABILIZATION")
print("="*80)

print(f"\nExtra modes per measurement point:")
print(f"{'Point':<25} {'TLS (no stab)':<15} {'LS + stab':<15} {'Extra':<10} {'% Increase'}")
print("-"*80)

for i, point in enumerate(points):
    pct_increase = (extra_modes[i] / tls_nostab_counts[i]) * 100
    print(f"{point:<25} {tls_nostab_counts[i]:>8d}       {ls_stab_counts[i]:>8d}       "
          f"{extra_modes[i]:>6d}     {pct_increase:>6.1f}%")

print()
print(f"Extra modes statistics:")
print(f"  Mean:   {np.mean(extra_modes):.1f} modes ({np.mean(extra_modes)/np.mean(tls_nostab_counts)*100:.1f}% increase)")
print(f"  Median: {np.median(extra_modes):.0f} modes")
print(f"  Std:    {np.std(extra_modes):.1f} modes")
print(f"  Range:  [{np.min(extra_modes)}, {np.max(extra_modes)}]")

# Consistency analysis
print()
print("="*80)
print("CONSISTENCY ANALYSIS")
print("="*80)

# Check variance across points
cv_tls = np.std(tls_nostab_counts) / np.mean(tls_nostab_counts) * 100
cv_stab = np.std(ls_stab_counts) / np.mean(ls_stab_counts) * 100
cv_extra = np.std(extra_modes) / np.mean(extra_modes) * 100

print(f"\nCoefficient of variation (lower = more consistent):")
print(f"  TLS (no stab):       {cv_tls:>6.1f}%")
print(f"  LS + stabilization:  {cv_stab:>6.1f}%")
print(f"  Extra modes:         {cv_extra:>6.1f}%")

print()
if cv_extra < 30:
    print("[OK] Extra modes are HIGHLY CONSISTENT across measurement points")
    print(f"     (CV = {cv_extra:.1f}% < 30%)")
elif cv_extra < 50:
    print("[MODERATE] Extra modes are MODERATELY CONSISTENT across measurement points")
    print(f"           (CV = {cv_extra:.1f}% in range 30-50%)")
else:
    print("[WARNING] Extra modes are INCONSISTENT across measurement points")
    print(f"          (CV = {cv_extra:.1f}% > 50%)")
    print("          This suggests stabilization finds location-specific or spurious modes")

# Visualization
print()
print("="*80)
print("Generating visualization...")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Mode count comparison
x = np.arange(n_points)
width = 0.35

bars1 = ax1.bar(x - width/2, tls_nostab_counts, width,
                label='TLS (no stab)', alpha=0.8, color='steelblue')
bars2 = ax1.bar(x + width/2, ls_stab_counts, width,
                label='LS + stab', alpha=0.8, color='coral')

# Add extra count labels on top of stabilization bars
for i, (bar, extra) in enumerate(zip(bars2, extra_modes)):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
             f'+{extra}',
             ha='center', va='bottom', fontsize=8, fontweight='bold',
             color='darkred')

ax1.set_xlabel('Measurement Point', fontweight='bold')
ax1.set_ylabel('Number of Modes', fontweight='bold')
ax1.set_title('Mode Count: TLS vs LS+Stabilization', fontweight='bold', fontsize=12)
ax1.set_xticks(x)
ax1.set_xticklabels([p.replace('point_', '').replace('_response', '')
                      for p in points], rotation=45, ha='right')
ax1.legend()
ax1.grid(axis='y', alpha=0.3)
ax1.set_ylim(0, max(ls_stab_counts) * 1.15)

# Plot 2: Extra modes distribution
ax2.bar(x, extra_modes, color='darkred', alpha=0.7)
ax2.axhline(np.mean(extra_modes), color='black', linestyle='--',
            linewidth=2, label=f'Mean = {np.mean(extra_modes):.1f}')
ax2.axhline(np.mean(extra_modes) + np.std(extra_modes), color='gray',
            linestyle=':', linewidth=1.5, alpha=0.7, label=f'±1 std')
ax2.axhline(np.mean(extra_modes) - np.std(extra_modes), color='gray',
            linestyle=':', linewidth=1.5, alpha=0.7)

ax2.set_xlabel('Measurement Point', fontweight='bold')
ax2.set_ylabel('Extra Modes from Stabilization', fontweight='bold')
ax2.set_title('Additional Modes Detected by Stabilization', fontweight='bold', fontsize=12)
ax2.set_xticks(x)
ax2.set_xticklabels([p.replace('point_', '').replace('_response', '')
                      for p in points], rotation=45, ha='right')
ax2.legend()
ax2.grid(axis='y', alpha=0.3)
ax2.set_ylim(0, max(extra_modes) * 1.2)

plt.tight_layout()
from pathlib import Path
output_path = Path("ESPRIT/comparison_results/stabilization_consistency.png")
output_path.parent.mkdir(exist_ok=True)
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"Saved to {output_path}")

# Summary
print()
print("="*80)
print("SUMMARY")
print("="*80)

print(f"\n1. Stabilization detects {np.mean(extra_modes):.1f} additional modes on average")
print(f"   ({np.mean(extra_modes)/np.mean(tls_nostab_counts)*100:.1f}% increase over TLS baseline)")

print(f"\n2. Consistency: CV = {cv_extra:.1f}%")
if cv_extra < 30:
    print("   -> Extra modes are CONSISTENT across measurements")
    print("   -> These are likely REAL physical modes missed by TLS")
elif cv_extra < 50:
    print("   -> Extra modes show MODERATE variability")
    print("   -> Mix of real modes and location-specific features")
else:
    print("   -> Extra modes are HIGHLY VARIABLE")
    print("   -> Likely spurious/computational artifacts")

print(f"\n3. Baseline (TLS) is very stable:")
print(f"   All 11 points detected exactly {tls_nostab_counts[0]} modes (CV = {cv_tls:.1f}%)")

print()
print("Recommendation:")
if cv_extra < 30:
    print("  [OK] Use LS+stabilization for production analysis")
    print("  [OK] The additional modes are physically meaningful")
elif cv_extra < 50:
    print("  [!] Use LS+stabilization with manual validation")
    print("  [!] Some modes may require frequency-domain confirmation")
else:
    print("  [X] Stick with TLS (no stabilization) for reliability")
    print("  [X] Stabilization's extra modes lack consistency")
