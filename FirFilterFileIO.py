"""
FIR Filter File I/O Module

This module provides functionality to save and load FIR filter banks in a binary format.

=== FILE FORMAT SPECIFICATION ===

The .fir file format is a binary format designed for efficient storage and loading of
multi-channel FIR filter banks with complete metadata.

STRUCTURE:
----------

1. HEADER (48 bytes total)
   Offset | Size | Type   | Field          | Description
   -------|------|--------|----------------|------------------------------------------
   0x00   | 4    | char[] | magic          | Magic number "FIR\0" (0x46 0x49 0x52 0x00)
   0x04   | 4    | uint32 | version        | File format version (currently 1)
   0x08   | 4    | uint32 | num_inputs     | Number of input channels
   0x0C   | 4    | uint32 | num_outputs    | Number of output channels
   0x10   | 4    | uint32 | filter_length  | Length of each filter (number of taps)
   0x14   | 4    | uint32 | sample_rate    | Sample rate in Hz (e.g., 48000)
   0x18   | 4    | uint32 | num_filters    | Total number of filters (= num_inputs × num_outputs)
   0x1C   | 12   | byte[] | reserved       | Reserved for future use (set to 0x00)

2. CHANNEL MAPPING (8 bytes × num_filters)
   For each filter (mapId = 0 to num_filters-1):
   Offset | Size | Type   | Field          | Description
   -------|------|--------|----------------|------------------------------------------
   +0     | 4    | uint32 | input_channel  | Source input channel for this filter
   +4     | 4    | uint32 | output_channel | Destination output channel for this filter

   Mapping convention: mapId = input_channel × num_outputs + output_channel

   Example for 2 inputs × 2 outputs:
   mapId=0: (in=0, out=0)  - Input 0 → Output 0
   mapId=1: (in=0, out=1)  - Input 0 → Output 1
   mapId=2: (in=1, out=0)  - Input 1 → Output 0
   mapId=3: (in=1, out=1)  - Input 1 → Output 1

3. FILTER DATA (4 bytes × filter_length × num_filters)
   All filter coefficients stored sequentially as float32 (little-endian).

   Layout: [filter_0][filter_1]...[filter_{N-1}]

   Each filter consists of filter_length float32 values representing the
   impulse response coefficients. Filters are stored in the same order as
   the channel mapping (by mapId).

DATA TYPES:
-----------
- All integers: little-endian uint32 (4 bytes)
- All floats: little-endian float32 (IEEE 754, 4 bytes)
- All offsets: byte offsets from start of file

EXAMPLE:
--------
A file with:
- 2 input channels
- 2 output channels
- 1024 taps per filter
- 48000 Hz sample rate

Would have:
- Header: 48 bytes
- Mapping: 32 bytes (4 filters × 8 bytes)
- Data: 16384 bytes (4 filters × 1024 taps × 4 bytes)
- Total: 16464 bytes

USAGE:
------
# Save filters
filters = np.random.randn(4, 1024).astype(np.float32)
save_fir_filters('myfilters.fir', filters, num_inputs=2, num_outputs=2, sample_rate=48000)

# Load filters
data = load_fir_filters('myfilters.fir')
filters = data['filters']  # shape: (4, 1024)
num_inputs = data['num_inputs']  # 2
num_outputs = data['num_outputs']  # 2
"""

import numpy as np
import struct
from pathlib import Path
from typing import Dict, Tuple, List


def save_fir_filters(filepath: str,
                     filters: np.ndarray,
                     num_inputs: int,
                     num_outputs: int,
                     sample_rate: int = 48000) -> None:
    """
    Save FIR filter bank to binary file format.

    Args:
        filepath: Path to save file (will create parent directories if needed)
        filters: np.ndarray of shape (num_filters, filter_length)
                 where num_filters = num_inputs × num_outputs
        num_inputs: Number of input channels
        num_outputs: Number of output channels
        sample_rate: Sample rate in Hz (default: 48000)

    Raises:
        ValueError: If filters shape doesn't match num_inputs × num_outputs
        IOError: If file cannot be written

    Example:
        >>> filters = np.random.randn(4, 1024).astype(np.float32)
        >>> save_fir_filters('bank.fir', filters, 2, 2, 48000)
    """
    num_filters, filter_length = filters.shape

    if num_filters != num_inputs * num_outputs:
        raise ValueError(
            f"Number of filters ({num_filters}) != "
            f"num_inputs ({num_inputs}) × num_outputs ({num_outputs})"
        )

    # Ensure parent directory exists
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, 'wb') as f:
        # === HEADER (48 bytes) ===
        f.write(b'FIR\0')                          # Magic number (4 bytes)
        f.write(struct.pack('<I', 1))              # Version (4 bytes)
        f.write(struct.pack('<I', num_inputs))     # num_inputs (4 bytes)
        f.write(struct.pack('<I', num_outputs))    # num_outputs (4 bytes)
        f.write(struct.pack('<I', filter_length))  # filter_length (4 bytes)
        f.write(struct.pack('<I', sample_rate))    # sample_rate (4 bytes)
        f.write(struct.pack('<I', num_filters))    # num_filters (4 bytes)
        f.write(b'\0' * 12)                        # Reserved (12 bytes)

        # === CHANNEL MAPPING (8 bytes × num_filters) ===
        # Mapping: mapId = in_ch * num_outputs + out_ch
        for map_id in range(num_filters):
            in_ch = map_id // num_outputs
            out_ch = map_id % num_outputs
            f.write(struct.pack('<I', in_ch))      # input_channel (4 bytes)
            f.write(struct.pack('<I', out_ch))     # output_channel (4 bytes)

        # === FILTER DATA (4 bytes × filter_length × num_filters) ===
        filters.astype(np.float32).tofile(f)

    file_size = Path(filepath).stat().st_size
    print(f"Saved {num_filters} filters ({filter_length} taps each) to {filepath}")
    print(f"File size: {file_size:,} bytes ({file_size / 1024:.1f} KB)")


def load_fir_filters(filepath: str) -> Dict:
    """
    Load FIR filter bank from binary file format.

    Args:
        filepath: Path to .fir file

    Returns:
        Dictionary containing:
            'filters': np.ndarray of shape (num_filters, filter_length), dtype=float32
            'num_inputs': int - number of input channels
            'num_outputs': int - number of output channels
            'filter_length': int - number of taps per filter
            'sample_rate': int - sample rate in Hz
            'num_filters': int - total number of filters
            'channel_mapping': List[Tuple[int, int]] - [(in_ch, out_ch), ...]
            'version': int - file format version

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file format is invalid or corrupted

    Example:
        >>> data = load_fir_filters('bank.fir')
        >>> filters = data['filters']  # shape: (4, 1024)
        >>> print(f"{data['num_inputs']}×{data['num_outputs']} filter bank")
    """
    if not Path(filepath).exists():
        raise FileNotFoundError(f"Filter file not found: {filepath}")

    with open(filepath, 'rb') as f:
        # === READ HEADER ===
        magic = f.read(4)
        if magic != b'FIR\0':
            raise ValueError(
                f"Invalid FIR file: wrong magic number {magic!r} "
                f"(expected b'FIR\\0')"
            )

        version = struct.unpack('<I', f.read(4))[0]
        if version != 1:
            raise ValueError(
                f"Unsupported file version {version} (expected 1)"
            )

        num_inputs = struct.unpack('<I', f.read(4))[0]
        num_outputs = struct.unpack('<I', f.read(4))[0]
        filter_length = struct.unpack('<I', f.read(4))[0]
        sample_rate = struct.unpack('<I', f.read(4))[0]
        num_filters = struct.unpack('<I', f.read(4))[0]
        reserved = f.read(12)  # Skip reserved bytes

        # Validate header
        if num_filters != num_inputs * num_outputs:
            raise ValueError(
                f"Corrupted file: num_filters ({num_filters}) != "
                f"num_inputs ({num_inputs}) × num_outputs ({num_outputs})"
            )

        # === READ CHANNEL MAPPING ===
        channel_mapping = []
        for _ in range(num_filters):
            in_ch = struct.unpack('<I', f.read(4))[0]
            out_ch = struct.unpack('<I', f.read(4))[0]

            # Validate mapping
            if in_ch >= num_inputs or out_ch >= num_outputs:
                raise ValueError(
                    f"Invalid channel mapping: ({in_ch}, {out_ch}) "
                    f"out of range for {num_inputs}×{num_outputs} config"
                )

            channel_mapping.append((in_ch, out_ch))

        # === READ FILTER DATA ===
        expected_floats = num_filters * filter_length
        filters_flat = np.fromfile(f, dtype=np.float32, count=expected_floats)

        if filters_flat.size != expected_floats:
            raise ValueError(
                f"Corrupted file: expected {expected_floats} float32 values, "
                f"got {filters_flat.size}"
            )

        filters = filters_flat.reshape(num_filters, filter_length)

        # Check if there's unexpected data at the end
        remaining = f.read()
        if remaining:
            print(f"Warning: {len(remaining)} extra bytes at end of file (ignored)")

    print(f"Loaded {num_filters} filters ({filter_length} taps each) from {filepath}")
    print(f"Configuration: {num_inputs} inputs × {num_outputs} outputs @ {sample_rate} Hz")

    return {
        'filters': filters,
        'num_inputs': num_inputs,
        'num_outputs': num_outputs,
        'filter_length': filter_length,
        'sample_rate': sample_rate,
        'num_filters': num_filters,
        'channel_mapping': channel_mapping,
        'version': version
    }


def validate_fir_file(filepath: str) -> bool:
    """
    Validate FIR filter file without fully loading it.

    Args:
        filepath: Path to .fir file

    Returns:
        True if file is valid, False otherwise

    Example:
        >>> if validate_fir_file('bank.fir'):
        >>>     data = load_fir_filters('bank.fir')
    """
    try:
        load_fir_filters(filepath)
        return True
    except (FileNotFoundError, ValueError, IOError) as e:
        print(f"Validation failed: {e}")
        return False


def get_fir_file_info(filepath: str) -> Dict:
    """
    Get metadata from FIR file without loading filter data.

    Args:
        filepath: Path to .fir file

    Returns:
        Dictionary with header information (no filter data)

    Example:
        >>> info = get_fir_file_info('bank.fir')
        >>> print(f"File contains {info['num_filters']} filters")
    """
    with open(filepath, 'rb') as f:
        # Read header only
        magic = f.read(4)
        if magic != b'FIR\0':
            raise ValueError(f"Invalid FIR file")

        version = struct.unpack('<I', f.read(4))[0]
        num_inputs = struct.unpack('<I', f.read(4))[0]
        num_outputs = struct.unpack('<I', f.read(4))[0]
        filter_length = struct.unpack('<I', f.read(4))[0]
        sample_rate = struct.unpack('<I', f.read(4))[0]
        num_filters = struct.unpack('<I', f.read(4))[0]

    file_size = Path(filepath).stat().st_size
    expected_size = 48 + (8 * num_filters) + (4 * num_filters * filter_length)

    return {
        'version': version,
        'num_inputs': num_inputs,
        'num_outputs': num_outputs,
        'filter_length': filter_length,
        'sample_rate': sample_rate,
        'num_filters': num_filters,
        'file_size': file_size,
        'expected_size': expected_size,
        'valid_size': file_size == expected_size
    }


if __name__ == '__main__':
    # Self-test
    print("=== FIR Filter File I/O Self-Test ===\n")

    # Create test filters
    num_inputs, num_outputs = 2, 2
    filter_length = 1024
    num_filters = num_inputs * num_outputs

    print(f"Creating {num_inputs}×{num_outputs} filter bank ({filter_length} taps)...")
    filters = np.random.randn(num_filters, filter_length).astype(np.float32)

    # Save
    test_path = 'test_filters.fir'
    print(f"\nSaving to {test_path}...")
    save_fir_filters(test_path, filters, num_inputs, num_outputs, 48000)

    # Get info
    print(f"\nGetting file info...")
    info = get_fir_file_info(test_path)
    for key, val in info.items():
        print(f"  {key}: {val}")

    # Validate
    print(f"\nValidating...")
    is_valid = validate_fir_file(test_path)
    print(f"  Valid: {is_valid}")

    # Load
    print(f"\nLoading from {test_path}...")
    loaded = load_fir_filters(test_path)

    # Verify
    print(f"\nVerifying data integrity...")
    filters_match = np.allclose(filters, loaded['filters'])
    print(f"  Filters match: {filters_match}")
    print(f"  Max error: {np.max(np.abs(filters - loaded['filters']))}")

    # Cleanup
    Path(test_path).unlink()
    print(f"\nTest file deleted. Self-test complete!")
