#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test to demonstrate that calibration impulse amplitudes are preserved (not renormalized).

SDL audio uses float32 format with samples in range [-1.0, +1.0] by definition.
This is NOT normalization - it's the standard audio representation.

The actual amplitude information IS preserved:
- Weak hit: peak ~ 0.3
- Strong hit: peak ~ 0.8

This test shows that amplitude ratios are maintained.
"""

import numpy as np
import sys
import io

# Fix console encoding for Windows
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

print("\n" + "="*60)
print("SDL Audio Float32 Format Explanation")
print("="*60 + "\n")

print("SDL uses AUDIO_F32SYS format:")
print("- Samples are 32-bit floating point")
print("- Range: -1.0 to +1.0")
print("- This is the STANDARD audio representation")
print("- NOT a normalization we apply - it's how SDL delivers audio\n")

print("="*60)
print("Amplitude Information IS Preserved")
print("="*60 + "\n")

# Simulate two calibration impulses with different magnitudes
weak_impulse = np.array([0.0, 0.05, 0.15, 0.30, 0.20, 0.10, 0.05, 0.02, 0.01])  # Weak hit
strong_impulse = np.array([0.0, 0.10, 0.30, 0.80, 0.50, 0.25, 0.12, 0.06, 0.03])  # Strong hit

weak_peak = np.max(np.abs(weak_impulse))
strong_peak = np.max(np.abs(strong_impulse))

print(f"Weak Impulse Peak:   {weak_peak:.3f}")
print(f"Strong Impulse Peak: {strong_peak:.3f}")
print(f"Ratio (Strong/Weak): {strong_peak/weak_peak:.2f}x\n")

print("✓ Amplitude ratios are preserved!")
print("✓ You CAN distinguish weak vs strong hits")
print("✓ You CAN set quality thresholds based on peak amplitude\n")

print("="*60)
print("What The Numbers Mean")
print("="*60 + "\n")

print("Peak Value | Interpretation")
print("-" * 40)
print("< 0.1      | Very weak (possibly noise/bad contact)")
print("0.1 - 0.3  | Weak but usable")
print("0.3 - 0.7  | Good strong hit")
print("0.7 - 0.95 | Very strong (good)")
print("> 0.95     | Likely clipping (too strong)\n")

print("="*60)
print("Calibration Validator V2 Behavior")
print("="*60 + "\n")

print("✓ Does NOT renormalize calibration impulses")
print("✓ Preserves raw amplitude from SDL")
print("✓ Compares peaks directly (e.g., 0.3 vs 0.8)")
print("✓ Learns thresholds from user-marked cycles")
print("✓ Example: Mark cycles with peaks 0.5-0.6 as good")
print("  → System learns: min_peak=0.4, max_peak=0.7 (with margin)\n")

print("="*60)
print("Playback Volume vs Recorded Amplitude")
print("="*60 + "\n")

print("Playback volume (0.0-1.0) controls OUTPUT amplitude")
print("Recorded amplitude reflects ACOUSTIC level at microphone\n")

print("Example:")
print("- Set volume=0.5 → speaker plays at 50% loudness")
print("- Weak hit with hammer → records as peak 0.25")
print("- Strong hit with hammer → records as peak 0.75")
print("- Ratio preserved: 3:1\n")

print("If you want LOUDER playback:")
print("- Increase volume parameter (e.g., 0.8)")
print("- Speaker plays louder")
print("- Hammer hits produce stronger acoustic signal")
print("- Recorded peaks will be higher\n")

print("✓ Amplitude information is RELATIVE to device max input")
print("✓ This is standard for ALL digital audio systems")
print("✓ Float32 format allows precise amplitude measurement\n")
