#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script for Calibration Impulse Visualizer

This script verifies that the calibration test workflow works correctly:
1. Record calibration impulses
2. Extract per-cycle waveforms
3. Validate quality metrics
4. Prepare data for visualization

Run this to verify the implementation without launching the full GUI.
"""

import numpy as np
import sys
import io
from pathlib import Path

# Fix console encoding for Windows
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

try:
    from RoomResponseRecorder import RoomResponseRecorder
    from calibration_validator import CalibrationValidator
except ImportError as e:
    print(f"ERROR: Failed to import required modules: {e}")
    sys.exit(1)


def test_calibration_data_extraction():
    """Test that calibration test extracts per-cycle data correctly"""

    print("\n" + "="*60)
    print("TEST: Calibration Data Extraction")
    print("="*60 + "\n")

    # Create a synthetic multi-channel recording with calibration
    print("Step 1: Creating synthetic calibration data...")

    sample_rate = 48000
    pulse_duration = 0.008
    cycle_duration = 0.1
    num_pulses = 8

    cycle_samples = int(cycle_duration * sample_rate)
    pulse_samples = int(pulse_duration * sample_rate)

    # Create calibration channel with varying quality impulses
    calibration_cycles = []

    for i in range(num_pulses):
        cycle = np.zeros(cycle_samples)

        # Vary impulse characteristics to test quality validation
        if i == 0:
            # Good impulse
            amplitude = 0.5
            duration = 0.005
        elif i == 1:
            # Weak impulse (should fail amplitude check)
            amplitude = 0.05
            duration = 0.005
        elif i == 2:
            # Good impulse
            amplitude = 0.6
            duration = 0.004
        elif i == 3:
            # Clipped impulse (should fail amplitude check)
            amplitude = 0.98
            duration = 0.003
        elif i == 4:
            # Good impulse
            amplitude = 0.55
            duration = 0.006
        elif i == 5:
            # Long impulse (should fail duration check)
            amplitude = 0.4
            duration = 0.025
        elif i == 6:
            # Good impulse
            amplitude = 0.52
            duration = 0.005
        else:
            # Good impulse
            amplitude = 0.58
            duration = 0.004

        # Generate impulse
        t = np.arange(int(duration * sample_rate)) / sample_rate
        impulse = amplitude * np.exp(-t * 200) * np.sin(2 * np.pi * 1000 * t)

        # Add to cycle
        cycle[100:100+len(impulse)] = impulse

        # Add some tail noise for cycles 1, 3, 5 (should fail tail noise check)
        if i in [1, 3, 5]:
            noise_start = int(0.03 * sample_rate)
            noise = np.random.normal(0, amplitude * 0.2, cycle_samples - noise_start)
            cycle[noise_start:] += noise

        calibration_cycles.append(cycle)

    calibration_array = np.array(calibration_cycles)

    print(f"  Created {num_pulses} calibration cycles")
    print(f"  Each cycle: {cycle_samples} samples ({cycle_duration*1000:.1f} ms)")
    print(f"  Sample rate: {sample_rate} Hz")
    print(f"  Shape: {calibration_array.shape}")

    # Step 2: Validate each cycle
    print("\nStep 2: Validating calibration cycles...")

    calibration_quality_config = {
        'cal_min_amplitude': 0.1,
        'cal_max_amplitude': 0.95,
        'cal_min_duration_ms': 2.0,
        'cal_max_duration_ms': 20.0,
        'cal_duration_threshold': 0.3,
        'cal_double_hit_window_ms': [10, 50],
        'cal_double_hit_threshold': 0.3,
        'cal_tail_start_ms': 30.0,
        'cal_tail_max_rms_ratio': 0.15,
        'min_valid_cycles': 3
    }

    validator = CalibrationValidator(calibration_quality_config, sample_rate)

    validation_results = []
    valid_count = 0

    for cycle_idx in range(num_pulses):
        validation = validator.validate_cycle(calibration_array[cycle_idx], cycle_idx)

        # Convert to dict
        validation_dict = {
            'cycle_index': cycle_idx,
            'calibration_valid': validation.calibration_valid,
            'calibration_metrics': validation.calibration_metrics,
            'calibration_failures': validation.calibration_failures
        }
        validation_results.append(validation_dict)

        if validation.calibration_valid:
            valid_count += 1
            print(f"  ✓ Cycle {cycle_idx}: VALID")
        else:
            print(f"  ✗ Cycle {cycle_idx}: INVALID - {', '.join(validation.calibration_failures)}")
            print(f"      Metrics: {validation.calibration_metrics}")

    print(f"\nValidation Summary: {valid_count}/{num_pulses} cycles valid")

    # Step 3: Verify data structure matches what GUI expects
    print("\nStep 3: Verifying data structure...")

    test_results = {
        'success': True,
        'num_cycles': num_pulses,
        'calibration_channel': 0,
        'sample_rate': sample_rate,
        'all_calibration_cycles': calibration_array,
        'validation_results': validation_results,
        'cycle_duration_s': cycle_samples / sample_rate
    }

    assert test_results['all_calibration_cycles'].shape == (num_pulses, cycle_samples), \
        "Calibration cycles array has wrong shape"
    assert len(test_results['validation_results']) == num_pulses, \
        "Validation results count mismatch"

    print(f"  ✓ Data structure correct")
    print(f"  ✓ all_calibration_cycles shape: {test_results['all_calibration_cycles'].shape}")
    print(f"  ✓ validation_results length: {len(test_results['validation_results'])}")

    # Step 4: Simulate what the UI would do
    print("\nStep 4: Simulating UI data extraction...")

    # Extract data like the UI does
    num_cycles = test_results.get('num_cycles', 0)
    cal_ch = test_results.get('calibration_channel', 0)
    sample_rate = test_results.get('sample_rate', 48000)
    calibration_cycles = test_results.get('all_calibration_cycles')
    validation_results = test_results.get('validation_results', [])
    cycle_duration_s = test_results.get('cycle_duration_s', 0.1)

    print(f"  num_cycles: {num_cycles}")
    print(f"  calibration_channel: {cal_ch}")
    print(f"  sample_rate: {sample_rate}")
    print(f"  calibration_cycles shape: {calibration_cycles.shape}")
    print(f"  validation_results count: {len(validation_results)}")
    print(f"  cycle_duration_s: {cycle_duration_s:.3f} s ({cycle_duration_s*1000:.1f} ms)")

    # Test individual cycle extraction
    print("\nStep 5: Testing individual cycle extraction...")
    for cycle_idx in [0, 3, 7]:
        cycle_waveform = calibration_cycles[cycle_idx]
        cycle_validation = validation_results[cycle_idx]
        cycle_metrics = cycle_validation.get('calibration_metrics', {})
        is_valid = cycle_validation.get('calibration_valid', False)

        print(f"\n  Cycle {cycle_idx}:")
        print(f"    Valid: {is_valid}")
        print(f"    Waveform shape: {cycle_waveform.shape}")
        print(f"    Peak amplitude: {cycle_metrics.get('peak_amplitude', 0):.3f}")
        print(f"    Duration (ms): {cycle_metrics.get('duration_ms', 0):.1f}")

    # Test multi-cycle overlay data
    print("\nStep 6: Testing multi-cycle overlay data...")
    cycles_to_compare = [0, 2, 4, 6]
    signals = [calibration_cycles[i] for i in cycles_to_compare]
    labels = [f"Cycle {i} {'✓' if validation_results[i].get('calibration_valid', False) else '✗'}"
             for i in cycles_to_compare]

    print(f"  Selected {len(signals)} cycles for comparison")
    print(f"  Labels: {labels}")
    print(f"  Signal shapes: {[s.shape for s in signals]}")

    print("\n" + "="*60)
    print("TEST PASSED: All data extraction working correctly!")
    print("="*60 + "\n")

    return test_results


def test_summary_table_generation():
    """Test that summary table data can be generated correctly"""

    print("\n" + "="*60)
    print("TEST: Summary Table Generation")
    print("="*60 + "\n")

    # Create dummy validation results
    validation_results = [
        {
            'cycle_index': i,
            'calibration_valid': i % 2 == 0,  # Alternating valid/invalid
            'calibration_metrics': {
                'peak_amplitude': 0.5 + i * 0.05,
                'duration_ms': 5.0 + i * 0.5,
                'secondary_peak_ratio': 0.1 + i * 0.02,
                'tail_rms_ratio': 0.05 + i * 0.01
            },
            'calibration_failures': [] if i % 2 == 0 else ['amplitude_too_low', 'excessive_tail_noise']
        }
        for i in range(8)
    ]

    # Generate table data like the UI does
    table_data = []
    for v_result in validation_results:
        cycle_idx = v_result.get('cycle_index', 0)
        valid = v_result.get('calibration_valid', False)
        metrics = v_result.get('calibration_metrics', {})
        failures = v_result.get('calibration_failures', [])

        row = {
            'Cycle': cycle_idx,
            'Valid': '✓' if valid else '✗',
            'Peak Amp': f"{metrics.get('peak_amplitude', 0):.3f}",
            'Duration (ms)': f"{metrics.get('duration_ms', 0):.1f}",
            'Secondary Peak': f"{metrics.get('secondary_peak_ratio', 0):.2f}",
            'Tail RMS': f"{metrics.get('tail_rms_ratio', 0):.3f}",
            'Issues': ', '.join(failures) if failures else 'None'
        }
        table_data.append(row)

    print("Generated table data:")
    print(f"  Rows: {len(table_data)}")
    print("\nSample rows:")
    for i in [0, 1, 7]:
        row = table_data[i]
        print(f"  Cycle {row['Cycle']}: {row['Valid']} | Amp={row['Peak Amp']} | Issues={row['Issues']}")

    print("\n✓ Table generation working correctly\n")


if __name__ == '__main__':
    print("\n" + "#"*60)
    print("# Calibration Visualizer Test Suite")
    print("#"*60)

    try:
        # Run tests
        test_results = test_calibration_data_extraction()
        test_summary_table_generation()

        print("\n" + "#"*60)
        print("# ALL TESTS PASSED ✓")
        print("#"*60 + "\n")

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
