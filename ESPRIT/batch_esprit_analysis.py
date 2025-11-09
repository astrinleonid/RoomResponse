"""
batch_esprit_analysis.py
Run ESPRIT modal analysis on multiple measurement files and check consistency.

Usage:
    python batch_esprit_analysis.py piano_point_responses/*.txt --order 30 --freq-range 0 500
"""
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict
import json

import sys
sys.path.insert(0, str(Path(__file__).parent))

from preprocessing_minimal import (
    load_measurement_file,
    detect_contact_end_from_force,
    preprocess_measurement,
    MinimalPreprocessingConfig
)
from esprit_core import esprit_modal_identification


def analyze_single_file(filepath: str, args) -> Dict:
    """Run ESPRIT analysis on a single file."""
    print(f"\nProcessing: {Path(filepath).name}")
    print("=" * 70)

    # Load data
    force, responses = load_measurement_file(filepath, skip_channel=2)

    # Detect contact end
    contact_end = detect_contact_end_from_force(force, tail_fraction=0.03)
    print(f"  Contact end: {contact_end} samples ({contact_end/args.fs*1000:.3f} ms)")

    # Preprocess with contact removal
    config = MinimalPreprocessingConfig(
        remove_contact=True,
        contact_tail_fraction=0.03,
        use_highpass=True,
        hp_cut_hz=1.0
    )

    processed, metadata = preprocess_measurement(
        force, responses, fs=args.fs,
        config=config,
        contact_end_idx=contact_end
    )

    print(f"  Processed: {metadata['n_samples_processed']} samples")
    print(f"  Duration: {metadata['duration_processed_s']:.4f} s")

    # Hankel window length (fixed to avoid SVD issues with large matrices)
    window_length = min(2000, int(len(processed) * 0.10))
    print(f"  Hankel window: {window_length}")

    # Run ESPRIT
    try:
        modal_params = esprit_modal_identification(
            processed,
            fs=args.fs,
            model_order=args.order,
            window_length=window_length,
            use_gpu=args.gpu,
            max_damping=0.2,
            freq_range=tuple(args.freq_range),
            ref_sensor=0
        )

        print(f"  Identified modes: {len(modal_params.frequencies)}")

        # Return results
        return {
            'filename': Path(filepath).name,
            'point_id': int(Path(filepath).stem.split('_')[1]),
            'success': True,
            'n_modes': len(modal_params.frequencies),
            'frequencies': modal_params.frequencies.tolist(),
            'damping_ratios': modal_params.damping_ratios.tolist(),
            'poles': [(p.real, p.imag) for p in modal_params.poles],
            'contact_end_sample': contact_end,
            'contact_duration_ms': metadata['contact_duration_ms'],
            'n_samples_processed': metadata['n_samples_processed']
        }

    except Exception as e:
        print(f"  ERROR: {e}")
        return {
            'filename': Path(filepath).name,
            'point_id': int(Path(filepath).stem.split('_')[1]),
            'success': False,
            'error': str(e)
        }


def analyze_consistency(results: List[Dict], args) -> Dict:
    """Analyze consistency of modal parameters across measurements."""
    print("\n" + "=" * 70)
    print("CONSISTENCY ANALYSIS")
    print("=" * 70)

    # Filter successful results
    successful = [r for r in results if r.get('success', False)]
    print(f"\nSuccessful analyses: {len(successful)} / {len(results)}")

    if len(successful) == 0:
        return {'error': 'No successful analyses'}

    # Analyze contact durations
    contact_durations = [r['contact_duration_ms'] for r in successful]
    print(f"\nContact Duration:")
    print(f"  Mean: {np.mean(contact_durations):.3f} ms")
    print(f"  Std:  {np.std(contact_durations):.3f} ms")
    print(f"  Min:  {np.min(contact_durations):.3f} ms")
    print(f"  Max:  {np.max(contact_durations):.3f} ms")

    # Analyze number of modes
    n_modes = [r['n_modes'] for r in successful]
    print(f"\nNumber of Modes Identified:")
    print(f"  Mean: {np.mean(n_modes):.1f}")
    print(f"  Min:  {np.min(n_modes)}")
    print(f"  Max:  {np.max(n_modes)}")
    print(f"  Mode counts: {dict(zip(*np.unique(n_modes, return_counts=True)))}")

    # Analyze frequencies (collect all)
    all_frequencies = []
    for r in successful:
        all_frequencies.extend(r['frequencies'])

    if len(all_frequencies) > 0:
        print(f"\nFrequency Distribution:")
        print(f"  Total modes: {len(all_frequencies)}")
        print(f"  Range: [{np.min(all_frequencies):.1f}, {np.max(all_frequencies):.1f}] Hz")
        print(f"  Mean: {np.mean(all_frequencies):.1f} Hz")

        # Histogram of frequencies
        hist, bins = np.histogram(all_frequencies, bins=10)
        print(f"\n  Frequency histogram:")
        for i in range(len(hist)):
            if hist[i] > 0:
                print(f"    {bins[i]:6.1f}-{bins[i+1]:6.1f} Hz: {hist[i]:3d} modes")

    # Analyze damping ratios
    all_damping = []
    for r in successful:
        all_damping.extend(r['damping_ratios'])

    if len(all_damping) > 0:
        print(f"\nDamping Ratio Distribution:")
        print(f"  Mean: {np.mean(all_damping)*100:.2f} %")
        print(f"  Std:  {np.std(all_damping)*100:.2f} %")
        print(f"  Min:  {np.min(all_damping)*100:.2f} %")
        print(f"  Max:  {np.max(all_damping)*100:.2f} %")

    # Check for common frequencies (cluster analysis)
    print(f"\nFrequency Clustering (modes appearing in multiple measurements):")
    freq_array = np.array(all_frequencies)

    # Find clusters using simple binning (5 Hz tolerance)
    bins = np.arange(args.freq_range[0], args.freq_range[1], 5)
    hist, bin_edges = np.histogram(freq_array, bins=bins)

    # Find prominent clusters (appear in multiple measurements)
    prominent_threshold = max(2, len(successful) * 0.3)  # At least 30% of measurements
    prominent = np.where(hist >= prominent_threshold)[0]

    if len(prominent) > 0:
        print(f"  Found {len(prominent)} consistent frequency clusters:")
        for idx in prominent:
            center = (bin_edges[idx] + bin_edges[idx+1]) / 2
            count = hist[idx]
            print(f"    {center:6.1f} Hz: appears in {count} measurements")
    else:
        print("  No strongly consistent frequencies found (may need tighter clustering)")

    return {
        'n_successful': len(successful),
        'n_total': len(results),
        'contact_duration_ms': {
            'mean': float(np.mean(contact_durations)),
            'std': float(np.std(contact_durations)),
            'min': float(np.min(contact_durations)),
            'max': float(np.max(contact_durations))
        },
        'n_modes': {
            'mean': float(np.mean(n_modes)),
            'min': int(np.min(n_modes)),
            'max': int(np.max(n_modes))
        },
        'frequencies': {
            'mean': float(np.mean(all_frequencies)) if all_frequencies else None,
            'range': [float(np.min(all_frequencies)), float(np.max(all_frequencies))] if all_frequencies else None
        },
        'damping': {
            'mean': float(np.mean(all_damping)) if all_damping else None,
            'std': float(np.std(all_damping)) if all_damping else None
        }
    }


def main():
    parser = argparse.ArgumentParser(description='Batch ESPRIT modal analysis')
    parser.add_argument('files', nargs='+', help='Measurement files')
    parser.add_argument('--gpu', action='store_true', help='Use GPU')
    parser.add_argument('--order', '-M', type=int, default=30, help='Model order')
    parser.add_argument('--freq-range', nargs=2, type=float, default=[0, 500],
                       help='Frequency range [min max] Hz')
    parser.add_argument('--fs', type=float, default=48000, help='Sampling frequency')
    parser.add_argument('--window-length-frac', type=float, default=0.33,
                       help='Hankel window fraction')
    parser.add_argument('--output', '-o', type=str, default='batch_results.json',
                       help='Output JSON file')

    args = parser.parse_args()

    print("=" * 70)
    print("BATCH ESPRIT MODAL ANALYSIS")
    print("=" * 70)
    print(f"Files: {len(args.files)}")
    print(f"Model order: {args.order}")
    print(f"Frequency range: {args.freq_range[0]}-{args.freq_range[1]} Hz")
    print(f"GPU: {'Enabled' if args.gpu else 'Disabled'}")

    # Process all files
    results = []
    for filepath in args.files:
        result = analyze_single_file(filepath, args)
        results.append(result)

    # Analyze consistency
    consistency = analyze_consistency(results, args)

    # Save results
    output_data = {
        'parameters': {
            'model_order': args.order,
            'freq_range': args.freq_range,
            'window_length_frac': args.window_length_frac,
            'fs': args.fs
        },
        'results': results,
        'consistency': consistency
    }

    output_path = Path(args.output)
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"\n" + "=" * 70)
    print(f"Results saved to: {output_path}")
    print("=" * 70)

    # Create summary CSV
    csv_path = output_path.with_suffix('.csv')
    summary_data = []
    for r in results:
        if r.get('success'):
            for i, (f, zeta) in enumerate(zip(r['frequencies'], r['damping_ratios'])):
                summary_data.append({
                    'point_id': r['point_id'],
                    'filename': r['filename'],
                    'mode_idx': i,
                    'frequency_hz': f,
                    'damping_ratio': zeta,
                    'damping_percent': zeta * 100
                })

    if summary_data:
        df = pd.DataFrame(summary_data)
        df.to_csv(csv_path, index=False)
        print(f"Summary CSV saved to: {csv_path}")


if __name__ == "__main__":
    main()
