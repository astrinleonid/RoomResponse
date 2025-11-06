"""
Multi-channel filename parsing utilities.

This module provides utilities for parsing and handling multi-channel WAV filenames
following the convention:
- Single-channel: {type}_{index}_{timestamp}.wav
- Multi-channel: {type}_{index}_{timestamp}_ch{N}.wav
"""

import re
from dataclasses import dataclass
from typing import Optional, Dict, List
from pathlib import Path


@dataclass
class MultiChannelFilename:
    """Parsed multi-channel filename components"""
    file_type: str              # "impulse", "room", "raw"
    index: int                  # Measurement index (000, 001, ...)
    timestamp: str              # ISO timestamp (YYYYMMDD_HHMMSS)
    channel: Optional[int]      # Channel index (None for single-channel)
    is_multichannel: bool       # True if filename has _chN suffix
    full_path: str              # Original full path


def parse_multichannel_filename(filename: str) -> Optional[MultiChannelFilename]:
    """
    Parse filename to extract components.

    Patterns:
        Single-channel: {type}_{index}_{timestamp}.wav
        Multi-channel:  {type}_{index}_{timestamp}_ch{N}.wav

    Args:
        filename: Full path or just filename to parse

    Returns:
        MultiChannelFilename object if successful, None if pattern doesn't match

    Examples:
        >>> parse_multichannel_filename("impulse_000_20251025_143022_ch0.wav")
        MultiChannelFilename(file_type='impulse', index=0, timestamp='20251025_143022',
                            channel=0, is_multichannel=True, ...)

        >>> parse_multichannel_filename("impulse_005_20251025_143022.wav")
        MultiChannelFilename(file_type='impulse', index=5, timestamp='20251025_143022',
                            channel=None, is_multichannel=False, ...)
    """
    # Get just the filename if full path provided
    filename_only = Path(filename).name

    # Multi-channel pattern: matches _ch{N} before extension
    # Flexible pattern to handle various naming schemes
    mc_pattern = r'_ch(\d+)\.(wav|npy)$'
    mc_match = re.search(mc_pattern, filename_only, re.IGNORECASE)

    if mc_match:
        # Extract channel number
        channel = int(mc_match.group(1))

        # Try to extract index from filename (looking for _NNN_ pattern)
        index_pattern = r'_(\d{3})_'
        index_match = re.search(index_pattern, filename_only)
        index = int(index_match.group(1)) if index_match else 0

        # Try to extract timestamp
        timestamp_pattern = r'_(\d{8}_\d{6})'
        timestamp_match = re.search(timestamp_pattern, filename_only)
        timestamp = timestamp_match.group(1) if timestamp_match else ""

        # Extract file type (first word before underscore)
        type_pattern = r'^(\w+)_'
        type_match = re.search(type_pattern, filename_only)
        file_type = type_match.group(1) if type_match else "unknown"

        return MultiChannelFilename(
            file_type=file_type,
            index=index,
            timestamp=timestamp,
            channel=channel,
            is_multichannel=True,
            full_path=filename
        )

    # Single-channel pattern: no _ch{N} suffix
    # Try old format first: {type}_{index}_{timestamp}.wav/.npy
    sc_pattern = r'^(\w+)_(\d+)_(\d{8}_\d{6})\.(wav|npy)$'
    sc_match = re.search(sc_pattern, filename_only, re.IGNORECASE)

    if sc_match:
        return MultiChannelFilename(
            file_type=sc_match.group(1),
            index=int(sc_match.group(2)),
            timestamp=sc_match.group(3),
            channel=None,
            is_multichannel=False,
            full_path=filename
        )

    return None


def group_files_by_measurement(file_paths: List[str]) -> Dict[int, List[str]]:
    """
    Group WAV files by measurement index.

    Args:
        file_paths: List of file paths to group

    Returns:
        Dict mapping measurement index to list of channel files

    Example:
        {
            0: ['impulse_000_20251025_ch0.wav', 'impulse_000_20251025_ch1.wav'],
            1: ['impulse_001_20251025_ch0.wav', 'impulse_001_20251025_ch1.wav']
        }
    """
    measurements = {}

    for file_path in file_paths:
        parsed = parse_multichannel_filename(file_path)
        if parsed:
            idx = parsed.index
            if idx not in measurements:
                measurements[idx] = []
            measurements[idx].append(file_path)

    # Sort channel files within each measurement
    for idx in measurements:
        measurements[idx].sort()

    return measurements


def group_files_by_channel(file_paths: List[str]) -> Dict[int, List[str]]:
    """
    Group multi-channel files by channel index.

    Args:
        file_paths: List of file paths to group

    Returns:
        Dict mapping channel index to list of files for that channel

    Example:
        {
            0: ['impulse_000_ch0.wav', 'impulse_001_ch0.wav', 'impulse_002_ch0.wav'],
            1: ['impulse_000_ch1.wav', 'impulse_001_ch1.wav', 'impulse_002_ch1.wav']
        }
    """
    channels = {}

    for file_path in file_paths:
        parsed = parse_multichannel_filename(file_path)
        if parsed and parsed.is_multichannel:
            ch_idx = parsed.channel
            if ch_idx not in channels:
                channels[ch_idx] = []
            channels[ch_idx].append(file_path)

    # Sort files within each channel
    for ch_idx in channels:
        channels[ch_idx].sort()

    return channels


def detect_num_channels(file_paths: List[str]) -> int:
    """
    Detect number of channels by examining file naming.

    Args:
        file_paths: List of file paths to analyze

    Returns:
        Number of channels (1 for single-channel, >1 for multi-channel)

    Example:
        >>> files = ['imp_000_20251025_ch0.wav', 'imp_000_20251025_ch1.wav']
        >>> detect_num_channels(files)
        2
    """
    if not file_paths:
        return 1  # Default to single-channel

    # Parse all files
    parsed_files = [parse_multichannel_filename(f) for f in file_paths]
    parsed_files = [pf for pf in parsed_files if pf is not None]

    if not parsed_files:
        return 1

    # Check if any file is multi-channel
    if not any(pf.is_multichannel for pf in parsed_files):
        return 1  # All single-channel

    # Count unique channel indices
    channels = set()
    for pf in parsed_files:
        if pf.is_multichannel and pf.channel is not None:
            channels.add(pf.channel)

    return len(channels) if channels else 1


def get_measurement_files(file_paths: List[str], measurement_index: int,
                          file_type: str = None) -> Dict[int, str]:
    """
    Get all channel files for a specific measurement.

    Args:
        file_paths: List of file paths to search
        measurement_index: Measurement index to retrieve
        file_type: Optional filter by file type (e.g., "impulse", "room", "raw")

    Returns:
        Dict mapping channel index to file path
        For single-channel: {0: "file.wav"}
        For multi-channel: {0: "file_ch0.wav", 1: "file_ch1.wav", ...}

    Example:
        >>> files = ['imp_000_ch0.wav', 'imp_000_ch1.wav', 'imp_001_ch0.wav']
        >>> get_measurement_files(files, 0)
        {0: 'imp_000_ch0.wav', 1: 'imp_000_ch1.wav'}
    """
    channel_map = {}

    for file_path in file_paths:
        parsed = parse_multichannel_filename(file_path)
        if not parsed:
            continue

        # Filter by measurement index
        if parsed.index != measurement_index:
            continue

        # Filter by file type if specified
        if file_type and parsed.file_type.lower() != file_type.lower():
            continue

        # Map channel index to file path
        ch_idx = parsed.channel if parsed.is_multichannel else 0
        channel_map[ch_idx] = parsed.full_path

    return channel_map


def is_multichannel_dataset(file_paths: List[str]) -> bool:
    """
    Determine if a dataset uses multi-channel format.

    Args:
        file_paths: List of file paths to analyze

    Returns:
        True if any files use multi-channel naming convention
    """
    for file_path in file_paths:
        parsed = parse_multichannel_filename(file_path)
        if parsed and parsed.is_multichannel:
            return True
    return False
