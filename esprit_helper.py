#!/usr/bin/env python3
"""
ESPRIT Helper Module for Single Scenario Integration

Provides utilities for:
1. Processing individual scenarios with ESPRIT
2. Aggregating selected scenarios from a collection
3. Incrementally adding new scenarios to existing aggregations
"""

import os
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

try:
    from ESPRIT.esprit_streaming import StreamingESPRITProcessor
except ImportError:
    StreamingESPRITProcessor = None

try:
    from gui_audio_visualizer import AudioVisualizer
except ImportError:
    AudioVisualizer = None


class ESPRITScenarioProcessor:
    """Processes individual scenarios with ESPRIT modal analysis."""

    DEFAULT_CONFIG = {
        'M_out': 6,
        'N_use': 28800,
        'fs': 48000,
        'band_index': 0,
        'L_fraction': 0.5,
        'K': 30,
        'skip_m': 2
    }

    @staticmethod
    def _make_json_serializable(obj):
        """Convert NumPy arrays and other non-serializable types to JSON-compatible types."""
        if isinstance(obj, dict):
            return {k: ESPRITScenarioProcessor._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [ESPRITScenarioProcessor._make_json_serializable(item) for item in obj]
        elif hasattr(obj, 'tolist'):  # NumPy array
            return obj.tolist()
        elif hasattr(obj, 'item'):  # NumPy scalar
            return obj.item()
        else:
            return obj

    @staticmethod
    def process_scenario(
        scenario_dir: Path,
        esprit_config: Optional[Dict[str, Any]] = None,
        force_reprocess: bool = False,
        process_all_bands: bool = True
    ) -> Optional[Dict[str, Any]]:
        """
        Process a single scenario with ESPRIT across all frequency bands.

        Args:
            scenario_dir: Path to scenario directory
            esprit_config: ESPRIT configuration dict (uses defaults if None)
            force_reprocess: Re-process even if results exist
            process_all_bands: If True, process all 4 bands separately (recommended)

        Returns:
            Dictionary with ESPRIT results (all bands) or None on failure
        """
        if StreamingESPRITProcessor is None:
            print("ERROR: esprit_streaming module not available")
            return None

        if AudioVisualizer is None:
            print("ERROR: gui_audio_visualizer module not available")
            return None

        scenario_dir = Path(scenario_dir)
        analysis_dir = scenario_dir / "analysis"

        # Merge config with defaults
        config = {**ESPRITScenarioProcessor.DEFAULT_CONFIG, **(esprit_config or {})}

        print(f"\n{'='*70}")
        print(f"ESPRIT Processing: {scenario_dir.name}")
        print(f"{'='*70}")

        try:
            # Step 1: Load and average room responses (do this once)
            averaged_response = ESPRITScenarioProcessor._average_room_responses(scenario_dir)
            if averaged_response is None:
                print("ERROR: Failed to compute averaged room response")
                return None

            print(f"Averaged response shape: {averaged_response.shape}")

            # Step 1.5: Filter channels if specified in config
            selected_channels = config.get('selected_channels', None)
            if selected_channels is not None and len(selected_channels) > 0:
                # Filter to only selected channels
                available_channels = list(range(averaged_response.shape[0]))
                valid_channels = [ch for ch in selected_channels if ch < averaged_response.shape[0]]

                if len(valid_channels) != len(selected_channels):
                    print(f"WARNING: Some selected channels are out of range. Using {len(valid_channels)} valid channels.")

                if valid_channels:
                    averaged_response = averaged_response[valid_channels, :]
                    print(f"Filtered to {len(valid_channels)} selected channels: {valid_channels}")
                    print(f"New shape: {averaged_response.shape}")

                    # Update M_out to match actual filtered channel count
                    config['M_out'] = averaged_response.shape[0]
                    print(f"Updated M_out to {config['M_out']} (filtered channel count)")
                else:
                    print("ERROR: No valid channels selected")
                    return None

            if process_all_bands:
                # Process all 4 bands separately
                band_names = [
                    "band0_40-500Hz",
                    "band1_500-1000Hz",
                    "band2_1000-2000Hz",
                    "band3_2000-4000Hz"
                ]

                all_results = {
                    'scenario_name': scenario_dir.name,
                    'scenario_dir': str(scenario_dir),
                    'processing_mode': 'multi-band',
                    'bands': {}
                }

                total_modes = 0

                for band_idx, band_name in enumerate(band_names):
                    output_file = analysis_dir / f"esprit_{band_name}.json"

                    # Check if already processed
                    if output_file.exists() and not force_reprocess:
                        print(f"Band {band_idx} results already exist: {output_file}")
                        with open(output_file, 'r') as f:
                            band_result = json.load(f)
                    else:
                        print(f"\nProcessing Band {band_idx} ({band_name})...")

                        # Update config for this band
                        band_config = {**config, 'band_index': band_idx}

                        # Initialize ESPRIT processor for this band
                        processor = StreamingESPRITProcessor(
                            M_out=band_config['M_out'],
                            N_use=band_config['N_use'],
                            fs=band_config['fs'],
                            band_index=band_idx,
                            L_fraction=band_config['L_fraction'],
                            K=band_config['K'],
                            skip_m=band_config['skip_m']
                        )

                        # Process this scenario for this band
                        band_result = processor.process_measurement(r_index=0, y_raw=averaged_response)

                        # Add metadata
                        band_result['scenario_name'] = scenario_dir.name
                        band_result['scenario_dir'] = str(scenario_dir)
                        band_result['config'] = band_config
                        band_result['band_index'] = band_idx
                        band_result['band_name'] = band_name
                        band_result['num_modes'] = len(band_result.get('frequencies', []))

                        # Save processed signal data for mode shape calculation
                        analysis_dir.mkdir(exist_ok=True)
                        signal_file = analysis_dir / f"esprit_{band_name}_signals.npy"
                        if processor.processed_measurements:
                            _, processed_signal = processor.processed_measurements[-1]
                            np.save(signal_file, processed_signal)
                            band_result['signal_file'] = signal_file.name
                            print(f"    Saved processed signals: {processed_signal.shape}")

                        # Save individual band results
                        # Convert NumPy arrays to lists for JSON serialization
                        band_result_serializable = ESPRITScenarioProcessor._make_json_serializable(band_result)
                        with open(output_file, 'w') as f:
                            json.dump(band_result_serializable, f, indent=2)

                        print(f"  OK Band {band_idx}: {band_result['num_modes']} modes detected")
                        if band_result['num_modes'] > 0:
                            freqs = band_result['frequencies'][:3]
                            print(f"    Top frequencies: {[f'{f:.1f}' for f in freqs]} Hz")

                    all_results['bands'][band_name] = band_result
                    total_modes += band_result.get('num_modes', 0)

                # Save combined summary
                summary_file = analysis_dir / "esprit_all_bands_summary.json"
                summary = {
                    'scenario_name': scenario_dir.name,
                    'scenario_dir': str(scenario_dir),
                    'processing_mode': 'multi-band',
                    'total_modes_all_bands': total_modes,
                    'modes_per_band': {
                        band_name: result.get('num_modes', 0)
                        for band_name, result in all_results['bands'].items()
                    },
                    'band_files': {
                        band_name: str(analysis_dir / f"esprit_{band_name}.json")
                        for band_name in band_names
                    }
                }

                with open(summary_file, 'w') as f:
                    json.dump(summary, f, indent=2)

                print(f"\n{'='*70}")
                print(f"OK ESPRIT Multi-band Complete: {total_modes} total modes across all bands")
                for band_name, result in all_results['bands'].items():
                    print(f"  {band_name}: {result.get('num_modes', 0)} modes")
                print(f"  Summary saved to: {summary_file}")
                print(f"{'='*70}")

                return summary

            else:
                # Single band processing (legacy mode)
                output_file = analysis_dir / "esprit_single_point.json"

                if output_file.exists() and not force_reprocess:
                    print(f"ESPRIT results already exist: {output_file}")
                    with open(output_file, 'r') as f:
                        return json.load(f)

                processor = StreamingESPRITProcessor(
                    M_out=config['M_out'],
                    N_use=config['N_use'],
                    fs=config['fs'],
                    band_index=config.get('band_index', 0),
                    L_fraction=config['L_fraction'],
                    K=config['K'],
                    skip_m=config['skip_m']
                )

                result = processor.process_measurement(r_index=0, y_raw=averaged_response)
                result['scenario_name'] = scenario_dir.name
                result['scenario_dir'] = str(scenario_dir)
                result['config'] = config
                result['num_modes'] = len(result.get('frequencies', []))

                analysis_dir.mkdir(exist_ok=True)
                # Convert NumPy arrays to lists for JSON serialization
                result_serializable = ESPRITScenarioProcessor._make_json_serializable(result)
                with open(output_file, 'w') as f:
                    json.dump(result_serializable, f, indent=2)

                print(f"\nOK ESPRIT complete: {result['num_modes']} modes detected")
                if result['num_modes'] > 0:
                    freqs = result['frequencies'][:5]
                    print(f"  Top frequencies: {[f'{f:.2f}' for f in freqs]} Hz")
                print(f"  Results saved to: {output_file}")

                return result

        except Exception as e:
            print(f"ERROR: ESPRIT processing failed: {e}")
            import traceback
            traceback.print_exc()
            return None

    @staticmethod
    def _average_room_responses(scenario_dir: Path) -> Optional[np.ndarray]:
        """
        Average all room response files in a scenario.

        Supports two formats:
        1. .npy files in impulse_responses/ (preferred, from collection)
        2. .wav files in room_responses/ (legacy)

        Returns:
            Multi-channel averaged response as np.ndarray (M_out, N_use)
        """
        # Try .npy format first (from impulse_responses/)
        impulse_dir = scenario_dir / "impulse_responses"
        if impulse_dir.exists():
            npy_files = sorted(impulse_dir.glob("impulse_*.npy"))
            if npy_files:
                print(f"Found {len(npy_files)} .npy impulse response files")
                return ESPRITScenarioProcessor._average_npy_files(npy_files)

        # Fall back to .wav format (from room_responses/)
        room_dir = scenario_dir / "room_responses"
        if not room_dir.exists():
            print(f"ERROR: Neither impulse_responses nor room_responses directory found")
            return None

        # Get all room response files
        room_files = sorted(room_dir.glob("room_*.wav"))
        if not room_files:
            print(f"ERROR: No room response files found in {room_dir}")
            return None

        print(f"Found {len(room_files)} room response .wav files")

        # Group files by channel (for multi-channel)
        channels_dict = {}  # {channel_idx: [file1, file2, ...]}

        for file_path in room_files:
            # Check if multi-channel naming (e.g., room_*_ch0.wav)
            if "_ch" in file_path.stem:
                ch_idx = int(file_path.stem.split("_ch")[-1])
                channels_dict.setdefault(ch_idx, []).append(file_path)
            else:
                # Single-channel or unnamed multi-channel
                channels_dict.setdefault(0, []).append(file_path)

        # Average each channel
        averaged_channels = {}
        for channel_idx, channel_files in sorted(channels_dict.items()):
            signals = []

            for file_path in channel_files:
                audio_data, sr, fmt = AudioVisualizer.load_audio_file(str(file_path))
                if audio_data is not None:
                    signals.append(audio_data)

            if not signals:
                continue

            # Find minimum length and truncate all to match
            min_length = min(len(s) for s in signals)
            truncated_signals = [s[:min_length] for s in signals]

            # Average across all measurements
            averaged_signal = np.mean(truncated_signals, axis=0)
            averaged_channels[channel_idx] = averaged_signal

            print(f"  Channel {channel_idx}: averaged {len(signals)} files -> {len(averaged_signal)} samples")

        if not averaged_channels:
            print("ERROR: No channels could be averaged")
            return None

        # Stack channels into (M_out, N_use) format
        channel_arrays = [averaged_channels[i] for i in sorted(averaged_channels.keys())]
        y_raw = np.vstack(channel_arrays)  # Shape: (num_channels, samples)

        return y_raw

    @staticmethod
    def _average_npy_files(npy_files: List[Path]) -> Optional[np.ndarray]:
        """
        Average .npy impulse response files grouped by channel.

        Args:
            npy_files: List of .npy file paths

        Returns:
            Multi-channel averaged response as np.ndarray (M_out, N_use)
        """
        # Group files by channel (e.g., impulse_*_ch0.npy)
        channels_dict = {}  # {channel_idx: [file1, file2, ...]}

        for file_path in npy_files:
            # Extract channel from filename: ..._ch0.npy
            if "_ch" in file_path.stem:
                ch_idx = int(file_path.stem.split("_ch")[-1])
                channels_dict.setdefault(ch_idx, []).append(file_path)
            else:
                # Single-channel or unnamed
                channels_dict.setdefault(0, []).append(file_path)

        print(f"  Detected {len(channels_dict)} channels")

        # Average each channel
        averaged_channels = {}
        for channel_idx, channel_files in sorted(channels_dict.items()):
            signals = []

            for file_path in channel_files:
                try:
                    data = np.load(file_path)
                    signals.append(data)
                except Exception as e:
                    print(f"WARNING: Failed to load {file_path}: {e}")
                    continue

            if not signals:
                continue

            # Find minimum length and truncate all to match
            min_length = min(len(s) for s in signals)
            truncated_signals = [s[:min_length] for s in signals]

            # Average across all measurements
            averaged_signal = np.mean(truncated_signals, axis=0)
            averaged_channels[channel_idx] = averaged_signal

            print(f"  Channel {channel_idx}: averaged {len(signals)} measurements -> {len(averaged_signal)} samples")

        if not averaged_channels:
            print("ERROR: No channels could be averaged")
            return None

        # Stack channels into (M_out, N_use) format
        channel_arrays = [averaged_channels[i] for i in sorted(averaged_channels.keys())]
        y_raw = np.vstack(channel_arrays)  # Shape: (num_channels, samples)

        return y_raw


class ESPRITAggregator:
    """Aggregates ESPRIT results from multiple scenarios."""

    # Band definitions for multi-band processing
    BAND_NAMES = [
        ("band0_40-500Hz", "40-500 Hz"),
        ("band1_500-1000Hz", "500-1000 Hz"),
        ("band2_1000-2000Hz", "1000-2000 Hz"),
        ("band3_2000-4000Hz", "2000-4000 Hz"),
    ]

    @staticmethod
    def aggregate_scenarios(
        scenario_dirs: List[Path],
        output_file: Optional[Path] = None,
        collection_name: Optional[str] = None,
        band_index: int = 0,
        min_occurrence_pct: float = 30.0,
        freq_tolerance_pct: float = 2.0
    ) -> Optional[Dict[str, Any]]:
        """
        Aggregate ESPRIT results from multiple scenarios.

        Args:
            scenario_dirs: List of paths to scenario directories
            output_file: Where to save aggregated results
            collection_name: Name for this collection
            band_index: Which frequency band to aggregate (0-3)
            min_occurrence_pct: Minimum % of scenarios a mode must appear in (default: 30%)
            freq_tolerance_pct: Frequency tolerance for grouping similar modes (default: 2%)

        Returns:
            Aggregated results dictionary or None on failure
        """
        if StreamingESPRITProcessor is None:
            print("ERROR: esprit_streaming module not available")
            return None

        band_file_name, band_display_name = ESPRITAggregator.BAND_NAMES[band_index]

        print(f"\n{'='*70}")
        print(f"ESPRIT Aggregation: {len(scenario_dirs)} scenarios")
        print(f"Band: {band_display_name} (index {band_index})")
        print(f"{'='*70}")

        # Load individual results
        individual_results = []
        scenario_names = []

        for scenario_dir in scenario_dirs:
            # Try loading single-band results (for backward compatibility)
            result_file_single = scenario_dir / "analysis" / "esprit_single_point.json"
            # Try loading specified band from multi-band results
            result_file_multiband = scenario_dir / "analysis" / f"esprit_{band_file_name}.json"

            result_file = None
            if result_file_single.exists() and band_index == 0:
                result_file = result_file_single
            elif result_file_multiband.exists():
                result_file = result_file_multiband

            if result_file is None:
                print(f"WARNING: No ESPRIT results found for {scenario_dir.name} band {band_index}, skipping")
                continue

            try:
                with open(result_file, 'r') as f:
                    result = json.load(f)

                # Try to load processed signal data for mode shapes
                signal_file_name = result.get('signal_file')
                if signal_file_name:
                    signal_file = scenario_dir / "analysis" / signal_file_name
                    if signal_file.exists():
                        result['processed_signal'] = np.load(signal_file)
                    else:
                        result['processed_signal'] = None
                else:
                    result['processed_signal'] = None

                individual_results.append(result)
                scenario_names.append(scenario_dir.name)
                has_signal = "with signals" if result['processed_signal'] is not None else "no signals"
                print(f"  OK Loaded {scenario_dir.name}: {result.get('num_modes', 0)} modes ({has_signal})")
            except Exception as e:
                print(f"WARNING: Failed to load {result_file}: {e}")

        if not individual_results:
            print("ERROR: No valid ESPRIT results to aggregate")
            return None

        # Get config from first result
        config = individual_results[0].get('config', ESPRITScenarioProcessor.DEFAULT_CONFIG)

        # Initialize processor for aggregation
        processor = StreamingESPRITProcessor(
            M_out=config['M_out'],
            N_use=config['N_use'],
            fs=config['fs'],
            band_index=config['band_index'],
            L_fraction=config['L_fraction'],
            K=config['K'],
            skip_m=config['skip_m']
        )

        # Prepare for mode shape calculation
        decimate_factor = processor.current_preset.decimate_factor
        N_band = config['N_use'] // decimate_factor
        M_eff = config['M_out'] - (1 if config.get('skip_m') is not None else 0)

        # Track how many scenarios have real signal data
        scenarios_with_signals = 0

        # Reconstruct internal state from individual results
        for r_index, result in enumerate(individual_results):
            # Add individual results to processor's accumulation lists
            freqs = result.get('frequencies', [])
            damping = result.get('damping_ratios', [])

            processor.all_f.extend(freqs)
            processor.all_z.extend(damping)
            processor.all_r.extend([r_index] * len(freqs))
            processor.r_indices.append(r_index)

            # Use real processed signal if available, otherwise use zeros
            processed_signal = result.get('processed_signal')
            if processed_signal is not None:
                processor.processed_measurements.append((r_index, processed_signal))
                scenarios_with_signals += 1
            else:
                # Fallback to dummy data
                dummy_measurement = np.zeros((M_eff, N_band))
                processor.processed_measurements.append((r_index, dummy_measurement))

        print(f"\nMode shape data: {scenarios_with_signals}/{len(individual_results)} scenarios have signal data")

        # Get aggregated results
        try:
            final_results = processor.get_final_results(
                names=scenario_names,
                min_occurrence_pct=min_occurrence_pct,
                freq_tolerance_pct=freq_tolerance_pct
            )

            # Add metadata
            final_results['collection_name'] = collection_name or "Unknown"
            final_results['num_scenarios'] = len(scenario_names)
            final_results['scenario_names'] = scenario_names
            final_results['config'] = config
            final_results['band_index'] = band_index
            final_results['band_name'] = band_display_name
            final_results['min_occurrence_pct'] = min_occurrence_pct
            final_results['freq_tolerance_pct'] = freq_tolerance_pct

            print(f"\nOK Aggregation complete:")
            print(f"  Band: {band_display_name}")
            print(f"  Scenarios: {len(scenario_names)}")
            print(f"  Common modes: {len(final_results.get('common_f', []))}")

            # Save if output file specified
            if output_file:
                output_file.parent.mkdir(parents=True, exist_ok=True)
                with open(output_file, 'w') as f:
                    json.dump(final_results, f, indent=2)
                print(f"  Saved to: {output_file}")

            return final_results

        except Exception as e:
            print(f"ERROR: Aggregation failed: {e}")
            import traceback
            traceback.print_exc()
            return None

    @staticmethod
    def find_collection_scenarios(
        base_dir: Path,
        computer_name: str,
        room_name: str
    ) -> List[Tuple[Path, str]]:
        """
        Find all scenarios matching Computer+Room in dataset.

        Args:
            base_dir: Base dataset directory
            computer_name: Computer name to match
            room_name: Room name to match

        Returns:
            List of (scenario_path, scenario_number) tuples
        """
        scenarios = []
        base_dir = Path(base_dir)

        if not base_dir.exists():
            return scenarios

        # Look for scenario directories matching pattern
        for scenario_dir in base_dir.iterdir():
            if not scenario_dir.is_dir():
                continue

            # Check metadata file (try multiple locations)
            metadata_file = scenario_dir / "scenario_metadata.json"
            if not metadata_file.exists():
                # Try metadata subdirectory
                metadata_dir = scenario_dir / "metadata"
                if metadata_dir.exists():
                    # Look for any *_metadata.json file
                    metadata_files = list(metadata_dir.glob("*_metadata.json"))
                    if metadata_files:
                        metadata_file = metadata_files[0]
                    else:
                        continue
                else:
                    continue

            try:
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)

                # Check if scenario_info exists (new format) or use root level (legacy)
                scenario_info = metadata.get('scenario_info', metadata)

                # Try both field name variations
                comp = scenario_info.get('computer_name') or scenario_info.get('computer')
                room = scenario_info.get('room_name') or scenario_info.get('room')
                scenario_num = scenario_info.get('scenario_number', scenario_dir.name)

                # Check if Computer+Room match
                if comp == computer_name and room == room_name:
                    scenarios.append((scenario_dir, scenario_num))

            except Exception as e:
                print(f"WARNING: Failed to read {metadata_file}: {e}")

        # Sort by scenario number
        scenarios.sort(key=lambda x: x[1])

        return scenarios
