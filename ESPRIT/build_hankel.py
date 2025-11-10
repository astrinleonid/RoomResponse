"""
build_hankel.py
Builds a Hankel matrix from measurement data for ESPRIT analysis.
Usage:
    python build_hankel.py <input_file> <output_file> [--window_length L] [--channel CH]
"""
from __future__ import annotations
import numpy as np
import argparse
from pathlib import Path


def load_measurement_data(filepath: str, skip_channel: int = 2):
    """
    Load measurement data and exclude the calibration channel.

    Args:
        filepath: Path to the measurement .txt file
        skip_channel: Channel number to skip (default: 2 for hammer calibration)

    Returns:
        data: numpy array of shape (n_samples, n_channels) with calibration channel removed
    """
    # Load data, skipping header lines that start with #
    data = np.loadtxt(filepath, comments='#')

    # Remove the calibration channel (channel 2)
    channels = list(range(data.shape[1]))
    channels.remove(skip_channel)
    data_filtered = data[:, channels]

    print(f"Loaded data shape: {data.shape}")
    print(f"After removing channel {skip_channel}: {data_filtered.shape}")
    print(f"Number of samples: {data_filtered.shape[0]}")
    print(f"Number of channels: {data_filtered.shape[1]}")

    return data_filtered


def build_hankel_matrix(signal: np.ndarray, window_length: int):
    """
    Construct a Hankel matrix from a 1D signal.

    For a signal x[n] of length N and window length L:
    H has shape (L, K) where K = N - L + 1

    H = [[x[0],   x[1],   x[2],   ..., x[K-1]  ],
         [x[1],   x[2],   x[3],   ..., x[K]    ],
         [x[2],   x[3],   x[4],   ..., x[K+1]  ],
         ...
         [x[L-1], x[L],   x[L+1], ..., x[N-1]  ]]

    Args:
        signal: 1D numpy array
        window_length: Number of rows in Hankel matrix (L)

    Returns:
        H: Hankel matrix of shape (L, K)
    """
    N = len(signal)
    L = window_length
    K = N - L + 1

    if K <= 0:
        raise ValueError(f"Signal length {N} too short for window length {L}")

    # Efficient construction using stride tricks
    from numpy.lib.stride_tricks import as_strided

    shape = (L, K)
    strides = (signal.strides[0], signal.strides[0])
    H = as_strided(signal, shape=shape, strides=strides)

    # Return a copy to avoid issues with stride tricks
    return H.copy()


def main():
    parser = argparse.ArgumentParser(description='Build Hankel matrix from measurement data')
    parser.add_argument('input_file', type=str, help='Input measurement file (.txt)')
    parser.add_argument('output_file', type=str, help='Output Hankel matrix file (.npy)')
    parser.add_argument('--window_length', '-L', type=int, default=256,
                        help='Window length for Hankel matrix (default: 256)')
    parser.add_argument('--channel', '-c', type=int, default=0,
                        help='Channel to use after removing calibration channel (default: 0)')
    parser.add_argument('--skip_channel', type=int, default=2,
                        help='Calibration channel to skip (default: 2)')

    args = parser.parse_args()

    # Load data
    print(f"Loading measurement data from: {args.input_file}")
    data = load_measurement_data(args.input_file, skip_channel=args.skip_channel)

    # Select channel
    if args.channel >= data.shape[1]:
        raise ValueError(f"Channel {args.channel} not available. Only {data.shape[1]} channels.")

    signal = data[:, args.channel]
    print(f"\nSelected channel {args.channel} (after exclusion)")
    print(f"Signal length: {len(signal)}")

    # Build Hankel matrix
    print(f"\nBuilding Hankel matrix with window length L={args.window_length}")
    H = build_hankel_matrix(signal, args.window_length)
    print(f"Hankel matrix shape: {H.shape}")
    print(f"Hankel matrix size: {H.nbytes / 1024 / 1024:.2f} MB")

    # Save
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, H)
    print(f"\nSaved Hankel matrix to: {args.output_file}")

    # Show first few singular values for validation
    print(f"\nComputing first 8 singular values for validation...")
    s = np.linalg.svd(H, compute_uv=False)
    print(f"First 8 singular values: {s[:8]}")


if __name__ == "__main__":
    main()
