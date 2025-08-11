#!/usr/bin/env python3
"""
ScenarioSelector - Dataset Analysis and Info Extraction

Analyzes the room response dataset directory and extracts comprehensive
information about all available scenarios including metadata, sample counts,
and feature availability.
"""

import os
import sys
import json
import argparse
import pandas as pd
from datetime import datetime
from pathlib import Path
import re


class ScenarioSelector:
    """
    Analyzes dataset directory and extracts scenario information.
    """

    def __init__(self, dataset_path="room_response_dataset"):
        """
        Initialize the ScenarioSelector.

        Args:
            dataset_path (str): Path to the dataset directory
        """
        self.dataset_path = dataset_path
        self.scenarios = {}

    def analyze_dataset(self):
        """
        Analyze the entire dataset and extract information for all scenarios.

        Returns:
            dict: Dictionary containing all scenario information
        """
        print(f"Analyzing dataset: {self.dataset_path}")
        print("=" * 60)

        if not os.path.exists(self.dataset_path):
            print(f"ERROR: Dataset directory not found: {self.dataset_path}")
            return {}

        # Find all scenario directories
        scenario_dirs = self._find_scenario_directories()

        if not scenario_dirs:
            print("No scenario directories found in dataset.")
            return {}

        print(f"Found {len(scenario_dirs)} potential scenario directories:")

        # Analyze each scenario directory
        for scenario_dir in scenario_dirs:
            print(f"\nAnalyzing: {os.path.basename(scenario_dir)}")
            scenario_info = self._analyze_scenario_directory(scenario_dir)

            if scenario_info:
                scenario_key = os.path.basename(scenario_dir)
                self.scenarios[scenario_key] = scenario_info
                print(f"  ✓ Valid scenario with {scenario_info['sample_count']} samples")
            else:
                print(f"  ✗ Invalid or incomplete scenario")

        print(f"\n" + "=" * 60)
        print(f"Analysis complete. Found {len(self.scenarios)} valid scenarios.")

        return self.scenarios

    def _find_scenario_directories(self):
        """
        Find all potential scenario directories in the dataset.

        Returns:
            list: List of directory paths that could be scenarios
        """
        scenario_dirs = []

        for item in os.listdir(self.dataset_path):
            item_path = os.path.join(self.dataset_path, item)

            # Check if it's a directory
            if os.path.isdir(item_path):
                scenario_dirs.append(item_path)

        return sorted(scenario_dirs)

    def _analyze_scenario_directory(self, scenario_dir):
        """
        Analyze a single scenario directory and extract all information.

        Args:
            scenario_dir (str): Path to the scenario directory

        Returns:
            dict: Scenario information or None if invalid
        """
        scenario_name = os.path.basename(scenario_dir)

        # Initialize scenario info
        scenario_info = {
            'directory_name': scenario_name,
            'full_path': scenario_dir,
            'computer_name': None,
            'room_name': None,
            'scenario_number': None,
            'description': None,
            'measurement_date': None,
            'measurement_time': None,
            'sample_count': 0,
            'features_available': {
                'spectrum': False,
                'mfcc': False,
                'audio': False
            },
            'feature_details': {},
            'metadata_available': False,
            'metadata_info': {},
            'file_timestamps': {},
            'validity': {
                'has_features': False,
                'has_samples': False,
                'parseable_name': False
            }
        }

        # Parse scenario name
        self._parse_scenario_name(scenario_info)

        # Check for feature files
        self._check_feature_files(scenario_info, scenario_dir)

        # Check for metadata
        self._check_metadata(scenario_info, scenario_dir)

        # Check for audio files
        self._check_audio_files(scenario_info, scenario_dir)

        # Get file timestamps
        self._get_file_timestamps(scenario_info, scenario_dir)

        # Determine overall validity
        scenario_info['validity']['has_features'] = (
                scenario_info['features_available']['spectrum'] or
                scenario_info['features_available']['mfcc']
        )
        scenario_info['validity']['has_samples'] = scenario_info['sample_count'] > 0

        # Only return valid scenarios
        if scenario_info['validity']['has_features'] and scenario_info['validity']['has_samples']:
            return scenario_info
        else:
            return None

    def _parse_scenario_name(self, scenario_info):
        """
        Parse the scenario directory name to extract computer, scenario, and room info.
        Expected format: <computer>-Scenario<number>-<room>

        Args:
            scenario_info (dict): Scenario info dictionary to update
        """
        name = scenario_info['directory_name']

        # Try to parse standard format: Computer-Scenario<N>-Room
        pattern = r'^(.+?)-Scenario(.+?)-(.+)$'
        match = re.match(pattern, name)

        if match:
            scenario_info['computer_name'] = match.group(1).strip()
            scenario_info['scenario_number'] = match.group(2).strip()
            scenario_info['room_name'] = match.group(3).strip()
            scenario_info['validity']['parseable_name'] = True
        else:
            # Try alternative patterns
            # Pattern: Computer-S<N>-Room
            pattern_alt1 = r'^(.+?)-S(.+?)-(.+)$'
            match_alt1 = re.match(pattern_alt1, name)

            if match_alt1:
                scenario_info['computer_name'] = match_alt1.group(1).strip()
                scenario_info['scenario_number'] = match_alt1.group(2).strip()
                scenario_info['room_name'] = match_alt1.group(3).strip()
                scenario_info['validity']['parseable_name'] = True
            else:
                # Fallback: treat entire name as description
                scenario_info['description'] = name
                scenario_info['validity']['parseable_name'] = False

    def _check_feature_files(self, scenario_info, scenario_dir):
        """
        Check for feature files and analyze their content.

        Args:
            scenario_info (dict): Scenario info dictionary to update
            scenario_dir (str): Path to scenario directory
        """
        # Check for spectrum features
        spectrum_file = os.path.join(scenario_dir, 'spectrum.csv')
        if os.path.exists(spectrum_file):
            spectrum_info = self._analyze_feature_file(spectrum_file, 'spectrum')
            if spectrum_info:
                scenario_info['features_available']['spectrum'] = True
                scenario_info['feature_details']['spectrum'] = spectrum_info
                scenario_info['sample_count'] = max(scenario_info['sample_count'],
                                                    spectrum_info['sample_count'])

        # Check for MFCC features
        mfcc_file = os.path.join(scenario_dir, 'features.csv')
        if os.path.exists(mfcc_file):
            mfcc_info = self._analyze_feature_file(mfcc_file, 'mfcc')
            if mfcc_info:
                scenario_info['features_available']['mfcc'] = True
                scenario_info['feature_details']['mfcc'] = mfcc_info
                scenario_info['sample_count'] = max(scenario_info['sample_count'],
                                                    mfcc_info['sample_count'])

    def _analyze_feature_file(self, file_path, feature_type):
        """
        Analyze a feature CSV file.

        Args:
            file_path (str): Path to the feature file
            feature_type (str): Type of features ('spectrum' or 'mfcc')

        Returns:
            dict: Feature file information or None if invalid
        """
        try:
            df = pd.read_csv(file_path)

            # Determine feature columns
            if feature_type == 'spectrum':
                feature_cols = [col for col in df.columns if col.startswith('freq_')]
            else:  # mfcc
                feature_cols = [col for col in df.columns if col.startswith('mfcc_')]

            if not feature_cols:
                return None

            # Get file info
            file_size = os.path.getsize(file_path)
            file_modified = os.path.getmtime(file_path)

            return {
                'sample_count': len(df),
                'feature_count': len(feature_cols),
                'file_size_mb': file_size / (1024 * 1024),
                'columns': list(df.columns),
                'feature_columns': feature_cols,
                'file_modified': datetime.fromtimestamp(file_modified).isoformat(),
                'has_missing_values': df[feature_cols].isnull().any().any(),
                'data_range': {
                    'min': float(df[feature_cols].min().min()),
                    'max': float(df[feature_cols].max().max()),
                    'mean': float(df[feature_cols].mean().mean())
                }
            }

        except Exception as e:
            print(f"    Warning: Could not analyze {file_path}: {e}")
            return None

    def _check_metadata(self, scenario_info, scenario_dir):
        """
        Check for metadata files and extract information.

        Args:
            scenario_info (dict): Scenario info dictionary to update
            scenario_dir (str): Path to scenario directory
        """
        metadata_dir = os.path.join(scenario_dir, 'metadata')

        if os.path.exists(metadata_dir):
            scenario_info['metadata_available'] = True

            # Check for session metadata
            session_metadata_file = os.path.join(metadata_dir, 'session_metadata.json')
            if os.path.exists(session_metadata_file):
                try:
                    with open(session_metadata_file, 'r') as f:
                        metadata = json.load(f)

                    scenario_info['metadata_info'] = metadata

                    # Extract key information from metadata
                    if 'scenario_info' in metadata:
                        scenario_data = metadata['scenario_info']

                        # Override parsed name info with metadata if available
                        if 'computer_name' in scenario_data:
                            scenario_info['computer_name'] = scenario_data['computer_name']
                        if 'room_name' in scenario_data:
                            scenario_info['room_name'] = scenario_data['room_name']
                        if 'scenario_number' in scenario_data:
                            scenario_info['scenario_number'] = scenario_data['scenario_number']
                        if 'description' in scenario_data:
                            scenario_info['description'] = scenario_data['description']
                        if 'collection_timestamp' in scenario_data:
                            timestamp = scenario_data['collection_timestamp']
                            try:
                                dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                                scenario_info['measurement_date'] = dt.strftime('%Y-%m-%d')
                                scenario_info['measurement_time'] = dt.strftime('%H:%M:%S')
                            except:
                                scenario_info['measurement_date'] = timestamp.split('T')[0]

                except Exception as e:
                    print(f"    Warning: Could not parse metadata: {e}")

    def _check_audio_files(self, scenario_info, scenario_dir):
        """
        Check for audio files in the impulse_responses directory.

        Args:
            scenario_info (dict): Scenario info dictionary to update
            scenario_dir (str): Path to scenario directory
        """
        audio_dir = os.path.join(scenario_dir, 'impulse_responses')

        if os.path.exists(audio_dir):
            try:
                audio_files = [f for f in os.listdir(audio_dir) if f.endswith('.wav')]
                if audio_files:
                    scenario_info['features_available']['audio'] = True
                    scenario_info['feature_details']['audio'] = {
                        'file_count': len(audio_files),
                        'files': audio_files[:5],  # Store first 5 filenames as sample
                        'total_size_mb': sum(
                            os.path.getsize(os.path.join(audio_dir, f))
                            for f in audio_files
                        ) / (1024 * 1024)
                    }

                    # Update sample count if not set from features
                    if scenario_info['sample_count'] == 0:
                        scenario_info['sample_count'] = len(audio_files)

            except Exception as e:
                print(f"    Warning: Could not analyze audio directory: {e}")

    def _get_file_timestamps(self, scenario_info, scenario_dir):
        """
        Get creation/modification timestamps for key files.

        Args:
            scenario_info (dict): Scenario info dictionary to update
            scenario_dir (str): Path to scenario directory
        """
        key_files = {
            'directory': scenario_dir,
            'spectrum.csv': os.path.join(scenario_dir, 'spectrum.csv'),
            'features.csv': os.path.join(scenario_dir, 'features.csv'),
            'metadata': os.path.join(scenario_dir, 'metadata')
        }

        for name, path in key_files.items():
            if os.path.exists(path):
                try:
                    stat = os.stat(path)
                    scenario_info['file_timestamps'][name] = {
                        'modified': datetime.fromtimestamp(stat.st_mtime).isoformat(),
                        'created': datetime.fromtimestamp(stat.st_ctime).isoformat()
                    }
                except:
                    pass

    def print_summary(self):
        """Print a comprehensive summary of all scenarios."""
        if not self.scenarios:
            print("No scenarios found.")
            return

        print("\n" + "=" * 80)
        print("DATASET SUMMARY")
        print("=" * 80)

        total_samples = sum(s['sample_count'] for s in self.scenarios.values())
        spectrum_count = sum(1 for s in self.scenarios.values() if s['features_available']['spectrum'])
        mfcc_count = sum(1 for s in self.scenarios.values() if s['features_available']['mfcc'])
        audio_count = sum(1 for s in self.scenarios.values() if s['features_available']['audio'])

        print(f"Dataset Path: {self.dataset_path}")
        print(f"Total Scenarios: {len(self.scenarios)}")
        print(f"Total Samples: {total_samples:,}")
        print(f"Scenarios with Spectrum Features: {spectrum_count}")
        print(f"Scenarios with MFCC Features: {mfcc_count}")
        print(f"Scenarios with Audio Files: {audio_count}")

        # Get unique computers and rooms
        computers = set()
        rooms = set()
        for scenario in self.scenarios.values():
            if scenario['computer_name']:
                computers.add(scenario['computer_name'])
            if scenario['room_name']:
                rooms.add(scenario['room_name'])

        if computers:
            print(f"Computers: {', '.join(sorted(computers))}")
        if rooms:
            print(f"Rooms: {', '.join(sorted(rooms))}")

    def print_detailed_info(self):
        """Print detailed information for each scenario."""
        if not self.scenarios:
            print("No scenarios found.")
            return

        print("\n" + "=" * 80)
        print("DETAILED SCENARIO INFORMATION")
        print("=" * 80)

        for i, (scenario_key, scenario) in enumerate(self.scenarios.items(), 1):
            print(f"\n{i}. {scenario_key}")
            print("-" * len(f"{i}. {scenario_key}"))

            # Basic info
            if scenario['computer_name']:
                print(f"   Computer: {scenario['computer_name']}")
            if scenario['room_name']:
                print(f"   Room: {scenario['room_name']}")
            if scenario['scenario_number']:
                print(f"   Scenario Number: {scenario['scenario_number']}")
            if scenario['description']:
                print(f"   Description: {scenario['description']}")

            # Measurement info
            if scenario['measurement_date']:
                print(f"   Date: {scenario['measurement_date']}")
            if scenario['measurement_time']:
                print(f"   Time: {scenario['measurement_time']}")

            print(f"   Samples: {scenario['sample_count']}")

            # Features info
            features = []
            if scenario['features_available']['spectrum']:
                spec_info = scenario['feature_details']['spectrum']
                features.append(f"Spectrum ({spec_info['feature_count']} features)")
            if scenario['features_available']['mfcc']:
                mfcc_info = scenario['feature_details']['mfcc']
                features.append(f"MFCC ({mfcc_info['feature_count']} features)")
            if scenario['features_available']['audio']:
                audio_info = scenario['feature_details']['audio']
                features.append(f"Audio ({audio_info['file_count']} files)")

            print(f"   Available Features: {', '.join(features) if features else 'None'}")

            # Metadata info
            if scenario['metadata_available']:
                print(f"   Metadata: ✓ Available")
            else:
                print(f"   Metadata: ✗ Not available")

    def export_to_json(self, filename=None):
        """
        Export scenario information to JSON file.

        Args:
            filename (str): Output filename (optional)
        """
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"scenario_analysis_{timestamp}.json"

        export_data = {
            'dataset_path': self.dataset_path,
            'analysis_timestamp': datetime.now().isoformat(),
            'total_scenarios': len(self.scenarios),
            'scenarios': self.scenarios
        }

        try:
            with open(filename, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
            print(f"\nScenario information exported to: {filename}")
        except Exception as e:
            print(f"Error exporting to JSON: {e}")

    def get_scenario_pairs(self, feature_type='spectrum'):
        """
        Get all possible pairs of scenarios that can be compared.

        Args:
            feature_type (str): Feature type to check compatibility ('spectrum' or 'mfcc')

        Returns:
            list: List of tuples containing compatible scenario pairs
        """
        compatible_scenarios = [
            (key, scenario) for key, scenario in self.scenarios.items()
            if scenario['features_available'][feature_type]
        ]

        pairs = []
        for i in range(len(compatible_scenarios)):
            for j in range(i + 1, len(compatible_scenarios)):
                scenario1_key, scenario1 = compatible_scenarios[i]
                scenario2_key, scenario2 = compatible_scenarios[j]

                pairs.append({
                    'scenario1': {
                        'key': scenario1_key,
                        'name': f"{scenario1['computer_name']}-{scenario1['room_name']}"
                        if scenario1['computer_name'] and scenario1['room_name']
                        else scenario1_key,
                        'path': scenario1['full_path'],
                        'samples': scenario1['sample_count']
                    },
                    'scenario2': {
                        'key': scenario2_key,
                        'name': f"{scenario2['computer_name']}-{scenario2['room_name']}"
                        if scenario2['computer_name'] and scenario2['room_name']
                        else scenario2_key,
                        'path': scenario2['full_path'],
                        'samples': scenario2['sample_count']
                    },
                    'feature_type': feature_type,
                    'balanced': abs(scenario1['sample_count'] - scenario2['sample_count']) /
                                max(scenario1['sample_count'], scenario2['sample_count']) < 0.3
                })

        return pairs


# Import Gooey if available
try:
    from gooey import Gooey, GooeyParser

    GOOEY_AVAILABLE = True
except ImportError:
    GOOEY_AVAILABLE = False


class ScenarioGUI:
    """GUI interface for scenario selection using Gooey."""

    def __init__(self, selector):
        """
        Initialize the GUI with a ScenarioSelector instance.

        Args:
            selector (ScenarioSelector): Initialized ScenarioSelector instance
        """
        self.selector = selector
        self.scenarios = selector.scenarios

    def get_available_computers(self):
        """Get list of unique computer names."""
        computers = set()
        for scenario in self.scenarios.values():
            if scenario['computer_name']:
                computers.add(scenario['computer_name'])
        return sorted(list(computers))

    def get_available_rooms(self):
        """Get list of unique room names."""
        rooms = set()
        for scenario in self.scenarios.values():
            if scenario['room_name']:
                rooms.add(scenario['room_name'])
        return sorted(list(rooms))

    def get_scenario_choices(self):
        """Get list of scenario choices for selection."""
        choices = []
        for key, scenario in self.scenarios.items():
            # Create descriptive label
            label_parts = []
            if scenario['computer_name']:
                label_parts.append(scenario['computer_name'])
            if scenario['scenario_number']:
                label_parts.append(f"S{scenario['scenario_number']}")
            if scenario['room_name']:
                label_parts.append(scenario['room_name'])

            if label_parts:
                label = "-".join(label_parts)
            else:
                label = key

            # Add sample count and features
            features = []
            if scenario['features_available']['spectrum']:
                features.append("Spectrum")
            if scenario['features_available']['mfcc']:
                features.append("MFCC")

            full_label = f"{label} ({scenario['sample_count']} samples, {'/'.join(features)})"
            choices.append(full_label)

        return choices

    def filter_scenarios(self, computer_filter=None, room_filter=None, feature_filter=None):
        """
        Filter scenarios based on criteria.

        Args:
            computer_filter (str): Computer name to filter by
            room_filter (str): Room name to filter by
            feature_filter (str): Feature type to filter by ('spectrum' or 'mfcc')

        Returns:
            dict: Filtered scenarios
        """
        filtered = {}

        for key, scenario in self.scenarios.items():
            # Apply filters
            if computer_filter and computer_filter != "All":
                if scenario['computer_name'] != computer_filter:
                    continue

            if room_filter and room_filter != "All":
                if scenario['room_name'] != room_filter:
                    continue

            if feature_filter and feature_filter != "All":
                if not scenario['features_available'].get(feature_filter, False):
                    continue

            filtered[key] = scenario

        return filtered

    def print_selection(self, selected_scenarios, filters_applied):
        """
        Print the final selection results.

        Args:
            selected_scenarios (list): List of selected scenario keys
            filters_applied (dict): Applied filters
        """
        print("\n" + "=" * 80)
        print("SCENARIO SELECTION RESULTS")
        print("=" * 80)

        # Print applied filters
        print("Applied Filters:")
        if filters_applied.get('computer') and filters_applied['computer'] != "All":
            print(f"  Computer: {filters_applied['computer']}")
        if filters_applied.get('room') and filters_applied['room'] != "All":
            print(f"  Room: {filters_applied['room']}")
        if filters_applied.get('feature') and filters_applied['feature'] != "All":
            print(f"  Feature Type: {filters_applied['feature']}")
        if not any(f and f != "All" for f in filters_applied.values()):
            print("  No filters applied")

        print(f"\nSelected Scenarios: {len(selected_scenarios)}")
        print("-" * 40)

        if not selected_scenarios:
            print("No scenarios selected.")
            return

        # Print detailed info for each selected scenario
        for i, scenario_label in enumerate(selected_scenarios, 1):
            # Find the actual scenario key from the label
            scenario_key = self._get_scenario_key_from_label(scenario_label)
            if scenario_key and scenario_key in self.scenarios:
                scenario = self.scenarios[scenario_key]

                print(f"\n{i}. {scenario_label}")
                print(f"   Directory: {scenario_key}")
                print(f"   Path: {scenario['full_path']}")

                if scenario['computer_name']:
                    print(f"   Computer: {scenario['computer_name']}")
                if scenario['room_name']:
                    print(f"   Room: {scenario['room_name']}")
                if scenario['scenario_number']:
                    print(f"   Scenario: {scenario['scenario_number']}")
                if scenario['description']:
                    print(f"   Description: {scenario['description']}")

                if scenario['measurement_date']:
                    date_time = scenario['measurement_date']
                    if scenario['measurement_time']:
                        date_time += f" {scenario['measurement_time']}"
                    print(f"   Measured: {date_time}")

                print(f"   Samples: {scenario['sample_count']}")

                # Features
                features = []
                if scenario['features_available']['spectrum']:
                    feat_info = scenario['feature_details']['spectrum']
                    features.append(f"Spectrum ({feat_info['feature_count']} features)")
                if scenario['features_available']['mfcc']:
                    feat_info = scenario['feature_details']['mfcc']
                    features.append(f"MFCC ({feat_info['feature_count']} features)")
                if scenario['features_available']['audio']:
                    feat_info = scenario['feature_details']['audio']
                    features.append(f"Audio ({feat_info['file_count']} files)")

                print(f"   Features: {', '.join(features)}")

        # Print classification recommendations
        if len(selected_scenarios) >= 2:
            print("\n" + "=" * 80)
            print("CLASSIFICATION RECOMMENDATIONS")
            print("=" * 80)

            # Check feature compatibility
            spectrum_compatible = all(
                self._scenario_has_feature(label, 'spectrum')
                for label in selected_scenarios
            )
            mfcc_compatible = all(
                self._scenario_has_feature(label, 'mfcc')
                for label in selected_scenarios
            )

            print("Feature Compatibility:")
            print(
                f"  Spectrum: {'✓ All scenarios compatible' if spectrum_compatible else '✗ Not all scenarios have spectrum features'}")
            print(
                f"  MFCC: {'✓ All scenarios compatible' if mfcc_compatible else '✗ Not all scenarios have MFCC features'}")

            if spectrum_compatible or mfcc_compatible:
                print(f"\n✓ Ready for classification!")
                if spectrum_compatible and mfcc_compatible:
                    print("  Recommended: Try both spectrum and MFCC features")
                elif spectrum_compatible:
                    print("  Recommended: Use spectrum features")
                else:
                    print("  Recommended: Use MFCC features")

                # Sample balance check
                scenario_keys = [self._get_scenario_key_from_label(label) for label in selected_scenarios]
                sample_counts = [
                    self.scenarios[key]['sample_count']
                    for key in scenario_keys if key in self.scenarios
                ]

                if sample_counts:
                    min_samples = min(sample_counts)
                    max_samples = max(sample_counts)
                    if max_samples > 0:
                        balance_ratio = min_samples / max_samples
                        if balance_ratio < 0.5:
                            print(f"  ⚠ Warning: Imbalanced samples ({min_samples}-{max_samples})")
                        else:
                            print(f"  ✓ Well-balanced samples ({min_samples}-{max_samples})")
            else:
                print("✗ Cannot classify: No common feature types")

        elif len(selected_scenarios) == 1:
            print("\n✓ Single scenario selected. Select one more for binary classification.")
        else:
            print("\n! No scenarios selected.")

    def _get_scenario_key_from_label(self, label):
        """Extract scenario key from display label."""
        # Try to match the label back to a scenario key
        for key, scenario in self.scenarios.items():
            label_parts = []
            if scenario['computer_name']:
                label_parts.append(scenario['computer_name'])
            if scenario['scenario_number']:
                label_parts.append(f"S{scenario['scenario_number']}")
            if scenario['room_name']:
                label_parts.append(scenario['room_name'])

            if label_parts:
                expected_label_start = "-".join(label_parts)
                if label.startswith(expected_label_start):
                    return key

        return None

    def _scenario_has_feature(self, label, feature_type):
        """Check if a scenario (by label) has a specific feature type."""
        key = self._get_scenario_key_from_label(label)
        if key and key in self.scenarios:
            return self.scenarios[key]['features_available'].get(feature_type, False)
        return False


def create_gooey_interface(dataset_path):
    """
    Create and run the Gooey interface.

    Args:
        dataset_path (str): Path to the dataset directory

    Returns:
        dict: Selection results
    """
    if not GOOEY_AVAILABLE:
        print("Gooey not available. Please install it with: pip install Gooey")
        return None

    # Initialize selector and analyze dataset
    print(f"Analyzing dataset: {dataset_path}")
    selector = ScenarioSelector(dataset_path)
    scenarios = selector.analyze_dataset()

    if not scenarios:
        print("No valid scenarios found.")
        return None

    # Create GUI instance
    gui = ScenarioGUI(selector)

    @Gooey(
        program_name="Room Response Scenario Selector",
        program_description="Browse and select scenarios for room response classification",
        default_size=(1000, 800),
        show_success_modal=False,
        show_failure_modal=False,
        programmatic=True
    )
    def gui_main():
        """Main GUI function with Gooey decorators."""
        parser = GooeyParser(description="Select scenarios for classification")

        # Single main group with everything on one page
        main_group = parser.add_argument_group(
            "Scenario Selection & Filtering",
            f"Dataset: {dataset_path} | {len(scenarios)} scenarios | {sum(s['sample_count'] for s in scenarios.values()):,} total samples",
            gooey_options={'columns': 2}
        )

        # Get filter options
        computers = ["All"] + gui.get_available_computers()
        rooms = ["All"] + gui.get_available_rooms()

        # Filters in left column
        main_group.add_argument(
            '--computer_filter',
            metavar='Filter by Computer',
            choices=computers,
            default="All",
            help="Show only scenarios from this computer",
            gooey_options={'columns': 1}
        )

        main_group.add_argument(
            '--room_filter',
            metavar='Filter by Room',
            choices=rooms,
            default="All",
            help="Show only scenarios from this room",
            gooey_options={'columns': 1}
        )

        main_group.add_argument(
            '--feature_filter',
            metavar='Filter by Feature Type',
            choices=["All", "spectrum", "mfcc"],
            default="All",
            help="Show only scenarios with this feature type",
            gooey_options={'columns': 1}
        )

        # Add some spacing
        main_group.add_argument(
            '--separator1',
            metavar='─── Scenario Selection ───',
            default="Select scenarios below:",
            help="Select one or more scenarios for analysis",
            widget='Textarea',
            gooey_options={'height': 60, 'columns': 2}
        )

        # Get scenario choices
        scenario_choices = gui.get_scenario_choices()

        # Add checkbox for each scenario in 2 columns
        for i, choice in enumerate(scenario_choices):
            # Create a safe argument name
            safe_name = f"scenario_{i}"
            main_group.add_argument(
                f'--{safe_name}',
                metavar=choice,
                action='store_true',
                help=f"Select this scenario for analysis",
                gooey_options={'columns': 1}
            )

        # Output options at the bottom
        main_group.add_argument(
            '--separator2',
            metavar='─── Output Options ───',
            default="Configure output settings:",
            help="Output configuration",
            widget='Textarea',
            gooey_options={'height': 40, 'columns': 2}
        )

        main_group.add_argument(
            '--print_detailed',
            metavar='Show Detailed Information',
            action='store_true',
            default=True,
            help="Print detailed information for selected scenarios",
            gooey_options={'columns': 1}
        )

        main_group.add_argument(
            '--export_selection',
            metavar='Export Selection to JSON',
            action='store_true',
            help="Export selection results to JSON file",
            gooey_options={'columns': 1}
        )

        args = parser.parse_args()

        # Process the selection
        selected_scenarios = []
        for i, choice in enumerate(scenario_choices):
            safe_name = f"scenario_{i}"
            if getattr(args, safe_name, False):
                selected_scenarios.append(choice)

        # Apply filters (for display purposes)
        filters_applied = {
            'computer': args.computer_filter,
            'room': args.room_filter,
            'feature': args.feature_filter
        }

        # Return results
        return {
            'selected_scenarios': selected_scenarios,
            'filters_applied': filters_applied,
            'dataset_path': dataset_path,
            'print_detailed': args.print_detailed,
            'export_selection': args.export_selection,
            'gui_instance': gui
        }

    return gui_main()


def main():
    """Main function - GUI is now the default interface."""
    parser = argparse.ArgumentParser(
        description='Room Response Scenario Selector - GUI by default',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python ScenarioSelector.py                          # Launch GUI (default)
  python ScenarioSelector.py --dataset my_dataset    # GUI with custom dataset
  python ScenarioSelector.py --cli                    # Use command line interface
  python ScenarioSelector.py --cli --detailed        # CLI with detailed output
  python ScenarioSelector.py --cli --pairs spectrum  # CLI showing compatible pairs
        """
    )

    parser.add_argument(
        '--dataset', '-d',
        default='room_response_dataset',
        help='Dataset directory path (default: room_response_dataset)'
    )

    parser.add_argument(
        '--cli', '-c',
        action='store_true',
        help='Use command line interface instead of GUI'
    )

    # CLI-only options (only used when --cli is specified)
    parser.add_argument(
        '--export', '-e',
        action='store_true',
        help='[CLI only] Export analysis results to JSON file'
    )

    parser.add_argument(
        '--pairs', '-p',
        choices=['spectrum', 'mfcc'],
        help='[CLI only] Show compatible scenario pairs for given feature type'
    )

    parser.add_argument(
        '--detailed', '-v',
        action='store_true',
        help='[CLI only] Show detailed information for each scenario'
    )

    sys.argv = [a for a in sys.argv if a != '--ignore-gooey']
    args, _unknown = parser.parse_known_args()

    # Default to GUI unless --cli is specified
    if not args.cli:
        # Launch GUI interface
        if not GOOEY_AVAILABLE:
            print("Gooey not available. Please install it with:")
            print("pip install Gooey")
            print("\nFalling back to command line interface...")
            args.cli = True
        else:
            result = create_gooey_interface(args.dataset)
            if result:
                # Print the selection results
                gui = result['gui_instance']
                gui.print_selection(
                    result['selected_scenarios'],
                    result['filters_applied']
                )

                # Export if requested
                if result['export_selection']:
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    export_data = {
                        'dataset_path': result['dataset_path'],
                        'selection_timestamp': datetime.now().isoformat(),
                        'filters_applied': result['filters_applied'],
                        'selected_scenarios': result['selected_scenarios'],
                        'scenario_details': {
                            gui._get_scenario_key_from_label(label): gui.scenarios[
                                gui._get_scenario_key_from_label(label)]
                            for label in result['selected_scenarios']
                            if gui._get_scenario_key_from_label(label) in gui.scenarios
                        }
                    }

                    filename = f"scenario_selection_{timestamp}.json"
                    try:
                        with open(filename, 'w') as f:
                            json.dump(export_data, f, indent=2, default=str)
                        print(f"\nSelection exported to: {filename}")
                    except Exception as e:
                        print(f"Error exporting selection: {e}")
            return

    # Command line interface
    print("Using command line interface...")
    selector = ScenarioSelector(args.dataset)
    scenarios = selector.analyze_dataset()

    if not scenarios:
        print("No valid scenarios found.")
        sys.exit(1)

    # Print summary
    selector.print_summary()

    # Print detailed info if requested
    if args.detailed:
        selector.print_detailed_info()

    # Show compatible pairs if requested
    if args.pairs:
        pairs = selector.get_scenario_pairs(args.pairs)
        print(f"\n" + "=" * 80)
        print(f"COMPATIBLE SCENARIO PAIRS ({args.pairs.upper()} features)")
        print("=" * 80)

        if pairs:
            for i, pair in enumerate(pairs, 1):
                print(f"\n{i}. {pair['scenario1']['name']} vs {pair['scenario2']['name']}")
                print(f"   Samples: {pair['scenario1']['samples']} vs {pair['scenario2']['samples']}")
                print(f"   Balanced: {'✓' if pair['balanced'] else '✗'}")
                print(f"   Paths:")
                print(f"     {pair['scenario1']['path']}")
                print(f"     {pair['scenario2']['path']}")
        else:
            print(f"No compatible scenario pairs found for {args.pairs} features.")

    # Export to JSON if requested
    if args.export:
        selector.export_to_json()


if __name__ == "__main__":
    main()