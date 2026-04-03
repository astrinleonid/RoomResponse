"""
Run ESPRIT modal analysis on piano impulse response data.

This script processes multi-channel impulse responses from piano measurements
and identifies modal parameters (frequencies, damping ratios, mode shapes).
"""
import numpy as np
from pathlib import Path
import sys
import matplotlib.pyplot as plt

# Add ESPRIT folder to path
sys.path.insert(0, str(Path(__file__).parent / 'ESPRIT'))

from esprit_core import esprit_modal_identification, ModalParameters


def load_multichannel_impulse(folder_path: Path, scenario: str, measurement_idx: int = 0) -> np.ndarray:
    """
    Load multi-channel impulse response data.

    Args:
        folder_path: Path to scenario folder
        scenario: Scenario name (e.g., 'Belarus-Scenario62-Measuement1')
        measurement_idx: Which measurement to load (default: 0)

    Returns:
        signals: Multi-channel impulse responses, shape (T, n_channels)
    """
    impulse_dir = folder_path / 'impulse_responses'

    # Find all channels for this measurement
    pattern = f"impulse_{scenario}_{measurement_idx:03d}_*_ch*.npy"
    files = sorted(impulse_dir.glob(pattern))

    if not files:
        raise FileNotFoundError(f"No impulse files found matching: {pattern}")

    # Load all channels
    channels = []
    for file in files:
        data = np.load(file)
        channels.append(data)

    # Stack into multi-channel array
    signals = np.stack(channels, axis=1)  # Shape: (T, n_channels)

    print(f"Loaded {len(channels)} channels, {len(signals)} samples")

    return signals


def plot_results(modal_params: ModalParameters, scenario: str, output_dir: Path):
    """Plot modal analysis results."""

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'ESPRIT Modal Analysis: {scenario}', fontsize=14, fontweight='bold')

    # 1. Singular values (decay)
    ax = axes[0, 0]
    ax.semilogy(modal_params.singular_values, 'b.-', linewidth=1)
    ax.axvline(modal_params.model_order, color='r', linestyle='--', label=f'Model Order M={modal_params.model_order}')
    ax.set_xlabel('Index')
    ax.set_ylabel('Singular Value')
    ax.set_title('Singular Values from Hankel SVD')
    ax.grid(True, alpha=0.3)
    ax.legend()

    # 2. Pole locations (s-plane)
    ax = axes[0, 1]
    ax.scatter(modal_params.poles.real, modal_params.poles.imag, c='blue', s=50, alpha=0.7)
    ax.axhline(0, color='k', linewidth=0.5)
    ax.axvline(0, color='k', linewidth=0.5)
    ax.set_xlabel('Real Part (σ)')
    ax.set_ylabel('Imaginary Part (ω)')
    ax.set_title(f'Pole Locations in s-plane (N={len(modal_params.poles)} modes)')
    ax.grid(True, alpha=0.3)

    # 3. Frequency vs Damping
    ax = axes[1, 0]
    ax.scatter(modal_params.frequencies, modal_params.damping_ratios * 100, c='green', s=60, alpha=0.7)
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Damping Ratio (%)')
    ax.set_title('Modal Parameters')
    ax.grid(True, alpha=0.3)

    # 4. Mode shapes (magnitude)
    ax = axes[1, 1]
    n_modes = len(modal_params.frequencies)
    n_channels = modal_params.mode_shapes.shape[1]

    # Plot first 10 modes (or all if fewer)
    for k in range(min(10, n_modes)):
        magnitudes = np.abs(modal_params.mode_shapes[k, :])
        ax.plot(range(n_channels), magnitudes, 'o-', label=f'{modal_params.frequencies[k]:.1f} Hz', alpha=0.7)

    ax.set_xlabel('Channel')
    ax.set_ylabel('Mode Shape Magnitude')
    ax.set_title('Mode Shapes (First 10 Modes)')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, ncol=2)

    plt.tight_layout()

    # Save figure
    output_file = output_dir / f'{scenario}_esprit_results.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Saved plot to: {output_file}")

    plt.show()


def print_modal_summary(modal_params: ModalParameters):
    """Print summary of identified modes."""
    print("\n" + "=" * 80)
    print("MODAL ANALYSIS RESULTS")
    print("=" * 80)
    print(f"\nModel Order: {modal_params.model_order}")
    print(f"Identified Modes: {len(modal_params.frequencies)}")

    if len(modal_params.frequencies) > 0:
        print(f"\nFrequency Range: [{modal_params.frequencies.min():.2f}, {modal_params.frequencies.max():.2f}] Hz")
        print(f"Damping Range: [{modal_params.damping_ratios.min()*100:.2f}, {modal_params.damping_ratios.max()*100:.2f}] %")

        print("\n" + "-" * 80)
        print(f"{'Mode':<6} {'Frequency':<12} {'Damping':<12} {'Re(pole)':<15} {'Im(pole)':<15}")
        print(f"{'#':<6} {'(Hz)':<12} {'(%)':<12} {'(sigma)':<15} {'(omega)':<15}")
        print("-" * 80)

        for i, (f, zeta, pole) in enumerate(zip(modal_params.frequencies,
                                                  modal_params.damping_ratios,
                                                  modal_params.poles)):
            print(f"{i+1:<6} {f:<12.3f} {zeta*100:<12.3f} {pole.real:<15.3f} {pole.imag:<15.3f}")

        print("-" * 80)


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Run ESPRIT on piano impulse data')
    parser.add_argument('--scenario', type=str, default='Belarus-Scenario62-Measuement1',
                       help='Scenario folder name')
    parser.add_argument('--measurement', type=int, default=0,
                       help='Measurement index (default: 0)')
    parser.add_argument('--model-order', '-M', type=int, default=40,
                       help='Model order (number of poles)')
    parser.add_argument('--window-length', '-L', type=int, default=None,
                       help='Hankel window length (default: auto)')
    parser.add_argument('--fs', type=float, default=48000,
                       help='Sampling frequency (Hz)')
    parser.add_argument('--max-damping', type=float, default=0.2,
                       help='Maximum damping ratio for filtering')
    parser.add_argument('--freq-range', nargs=2, type=float, default=[30, 2000],
                       help='Frequency range [min max] Hz')
    parser.add_argument('--use-gpu', action='store_true',
                       help='Use GPU acceleration')
    parser.add_argument('--use-stabilization', action='store_true',
                       help='Enable automatic (M,L) grid stabilization')
    parser.add_argument('--use-multichannel', action='store_true',
                       help='Stack Hankel matrices from all channels')
    parser.add_argument('--no-plot', action='store_true',
                       help='Skip plotting')

    args = parser.parse_args()

    # Setup paths
    piano_dir = Path('piano')
    scenario_path = piano_dir / args.scenario

    if not scenario_path.exists():
        print(f"Error: Scenario folder not found: {scenario_path}")
        return

    print("=" * 80)
    print("ESPRIT MODAL ANALYSIS - PIANO DATA")
    print("=" * 80)
    print(f"Scenario: {args.scenario}")
    print(f"Measurement: {args.measurement}")
    print(f"Model Order: {args.model_order}")
    print(f"Sampling Rate: {args.fs} Hz")
    print(f"Frequency Range: {args.freq_range[0]}-{args.freq_range[1]} Hz")
    print(f"Max Damping: {args.max_damping*100:.1f} %")
    print(f"GPU: {'Enabled' if args.use_gpu else 'Disabled'}")
    print(f"Stabilization: {'Enabled' if args.use_stabilization else 'Disabled'}")
    print(f"Multi-channel Hankel: {'Yes' if args.use_multichannel else 'No'}")
    print("=" * 80)

    # Load impulse response data
    print("\nLoading impulse response data...")
    signals = load_multichannel_impulse(scenario_path, args.scenario, args.measurement)

    # Run ESPRIT modal identification
    print("\nRunning ESPRIT modal identification...")
    modal_params = esprit_modal_identification(
        signals=signals,
        fs=args.fs,
        model_order=args.model_order,
        window_length=args.window_length,
        use_gpu=args.use_gpu,
        max_damping=args.max_damping,
        freq_range=tuple(args.freq_range),
        ref_sensor=0,
        use_stabilization=args.use_stabilization,
        use_tls=True,
        use_conjugate_pairing=True,
        use_multichannel=args.use_multichannel
    )

    # Print results
    print_modal_summary(modal_params)

    # Plot results
    if not args.no_plot:
        output_dir = scenario_path / 'analysis'
        output_dir.mkdir(exist_ok=True)
        plot_results(modal_params, args.scenario, output_dir)

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)


if __name__ == '__main__':
    main()
