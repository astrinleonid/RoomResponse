"""
Test script to run esprit.py self_test_detailed() function.

This script runs the enhanced self-test with detailed comparison metrics,
compatible with test_esprit_core_synthetic.py for side-by-side comparison.
"""

import sys
sys.path.insert(0, 'ESPRIT')

from esprit import self_test_detailed

if __name__ == "__main__":
    print("######################################################################")
    print("# ESPRIT.PY ENHANCED SELF-TEST")
    print("######################################################################")

    # Run self-test with no noise
    print("\nTest 1: TLS-ESPRIT (U-subspace, no noise)")
    print("="*70)
    comparison = self_test_detailed(noise_level=0.0, visualize=True)

    print("\n\n######################################################################")
    print("# ESPRIT.PY ENHANCED SELF-TEST WITH NOISE")
    print("######################################################################")

    # Run self-test with moderate noise
    print("\nTest 2: TLS-ESPRIT (U-subspace, with noise ~30dB SNR)")
    print("="*70)
    comparison_noise = self_test_detailed(noise_level=0.01, visualize=True)
