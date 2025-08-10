import os
import json
import time
import numpy as np
import sys
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Any
import wave
from pathlib import Path

# Import the refactored recorder
from RoomResponseRecorder import RoomResponseRecorder


@dataclass
class ScenarioConfig:
    """Configuration for a measurement scenario"""
    scenario_number: str
    description: str
    computer_name: str
    room_name: str
    num_measurements: int = 30
    measurement_interval: float = 2.0  # seconds between measurements
    warm_up_measurements: int = 3  # measurements to discard at start
    additional_metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.additional_metadata is None:
            self.additional_metadata = {}

    @property
    def scenario_name(self) -> str:
        """Generate scenario name from components"""
        return f"{self.computer_name}-Scenario{self.scenario_number}-{self.room_name}"


@dataclass
class MeasurementMetadata:
    """Metadata for a single measurement"""
    scenario_name: str
    measurement_index: int
    timestamp: str
    filename_raw: str
    filename_impulse: str
    filename_room_response: str
    signal_params: Dict[str, Any]
    recording_stats: Dict[str, Any] = None
    quality_metrics: Dict[str, Any] = None
    notes: str = ""

    def to_dict(self):
        return asdict(self)


class SingleScenarioCollector:
    """Single scenario data collector for room response measurements"""

    def __init__(self,
                 base_output_dir: str = "room_response_dataset",
                 recorder_config: str = '',
                 scenario_config = False):
        """
        Initialize the single scenario collector

        Args:
            base_output_dir: Base directory for storing all data
            recorder_config: Configuration for the SDL recorder
        """
        self.base_output_dir = Path(base_output_dir)

        if scenario_config:
            self.setup_scenario_from_dict(**scenario_config)
        else:
            # Setup from user input
            self.setup_scenario_from_input(interactive_devices = False)

        self.recorder_config = recorder_config
        self.recorder = None

        self.measurements: List[MeasurementMetadata] = []
        self.interactive_mode = False
        self.recording_method = 2  # Default to auto method

        # Quality assessment thresholds - relaxed for real-world conditions
        self.quality_thresholds = {
            'min_snr_db': 15.0,
            'max_clip_percentage': 2.0,
            'min_dynamic_range_db': 25.0
        }

    def setup_scenario_from_dict(self, **parameters):

        parameter_names = ['scenario_number',
            'description',
            'computer_name',
            'room_name',
            'num_measurements',
            'measurement_interval']

        for p in parameter_names:
            if p not in parameters:
                raise ValueError(f"Scenario parameter {p} is missing")
        for p in parameters:
            if p not in parameter_names:
                raise ValueError(f"Unexpected scenario parameter {p}")
        self.scenario = ScenarioConfig(**parameters)



    def setup_scenario_from_input(self, interactive_devices: bool = False):
        """Setup scenario configuration from user input"""
        self.interactive_mode = interactive_devices

        print(f"\n{'=' * 60}")
        print("ROOM RESPONSE DATA COLLECTION - SINGLE SCENARIO")
        print(f"{'=' * 60}")

        # Get scenario information
        computer_name = input("Enter computer name: ").strip().replace(" ", "_")
        if not computer_name:
            computer_name = "Unknown_Computer"

        room_name = input("Enter room name: ").strip().replace(" ", "_")
        if not room_name:
            room_name = "Unknown_Room"

        try:
            scenario_number = (input("Enter scenario number: ").strip())
        except ValueError:
            print("Invalid scenario number, using 1")
            scenario_number = "0.1"

        description = input("Enter scenario description: ").strip()
        if not description:
            description = f"Room response measurement scenario {scenario_number}"

        try:
            num_measurements = int(input(f"Number of measurements (default 30): ").strip() or "30")
        except ValueError:
            num_measurements = 30

        try:
            interval = float(input(f"Interval between measurements in seconds (default 2.0): ").strip() or "2.0")
        except ValueError:
            interval = 2.0

        # Create scenario configuration
        self.scenario = ScenarioConfig(
            scenario_number=scenario_number,
            description=description,
            computer_name=computer_name,
            room_name=room_name,
            num_measurements=num_measurements,
            measurement_interval=interval
        )

        print(f"\nScenario configured:")
        print(f"  Name: {self.scenario.scenario_name}")
        print(f"  Description: {self.scenario.description}")
        print(f"  Measurements: {self.scenario.num_measurements}")
        print(f"  Interval: {self.scenario.measurement_interval}s")

    def setup_directories(self):
        """Create directory structure for the scenario"""
        if not self.scenario:
            raise ValueError("Scenario not configured")

        # Create scenario-specific directory
        self.scenario_dir = self.base_output_dir / self.scenario.scenario_name
        self.scenario_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories
        (self.scenario_dir / "raw_recordings").mkdir(exist_ok=True)
        (self.scenario_dir / "impulse_responses").mkdir(exist_ok=True)
        (self.scenario_dir / "room_responses").mkdir(exist_ok=True)
        (self.scenario_dir / "metadata").mkdir(exist_ok=True)
        (self.scenario_dir / "analysis").mkdir(exist_ok=True)

        print(f"\nDataset directory: {self.scenario_dir}")

    def initialize_recorder(self):
        """Initialize the room response recorder"""
        # Set recording method based on interactive mode
        self.recording_method = 1 if self.interactive_mode else 2

        self.recorder = RoomResponseRecorder(self.recorder_config)

        # Test the recorder setup
        print(f"\nTesting audio device setup...")
        devices = self.recorder.list_devices()
        if not devices or not devices.get('input_devices') or not devices.get('output_devices'):
            raise RuntimeError("No suitable audio devices found")

        # Store device info for metadata
        self.device_info = {
            'input_device': devices['input_devices'][0].name if devices['input_devices'] else "Unknown",
            'output_device': devices['output_devices'][0].name if devices['output_devices'] else "Unknown",
            'method': self.recording_method,
            'interactive': self.interactive_mode
        }

        print(f"Audio setup complete:")
        print(f"  Input: {self.device_info['input_device']}")
        print(f"  Output: {self.device_info['output_device']}")
        print(f"  Recording method: {self.recording_method}")
        if self.interactive_mode:
            print(f"  Mode: Interactive device selection")

        # Print signal configuration
        self.recorder.print_signal_analysis()

    def calculate_quality_metrics(self, audio_data: np.ndarray) -> Dict[str, float]:
        """Calculate quality metrics for a recorded signal"""
        try:
            # Signal-to-noise ratio estimation
            signal_power = np.mean(audio_data ** 2)

            # Estimate noise from quieter portions (first and last 10%)
            quiet_samples = int(0.1 * len(audio_data))
            noise_samples = np.concatenate([audio_data[:quiet_samples], audio_data[-quiet_samples:]])
            noise_power = np.mean(noise_samples ** 2)

            snr_db = 10 * np.log10(signal_power / (noise_power + 1e-10))

            # Clipping detection
            max_amplitude = np.max(np.abs(audio_data))
            clipped_samples = np.sum(np.abs(audio_data) > 0.95)
            clip_percentage = (clipped_samples / len(audio_data)) * 100

            # Dynamic range
            rms_level = np.sqrt(np.mean(audio_data ** 2))
            dynamic_range_db = 20 * np.log10(max_amplitude / (rms_level + 1e-10))

            return {
                'snr_db': float(snr_db),
                'max_amplitude': float(max_amplitude),
                'rms_level': float(rms_level),
                'clip_percentage': float(clip_percentage),
                'dynamic_range_db': float(dynamic_range_db)
            }

        except Exception as e:
            print(f"Error calculating quality metrics: {e}")
            return {}

    def assess_measurement_quality(self, metrics: Dict[str, float]) -> Dict[str, Any]:
        """Assess if a measurement meets quality thresholds"""
        assessment = {
            'overall_quality': 'good',
            'issues': [],
            'passed_checks': []
        }

        # Check SNR
        if metrics.get('snr_db', 0) < self.quality_thresholds['min_snr_db']:
            assessment['issues'].append(f"Low SNR: {metrics.get('snr_db', 0):.1f}dB")
            assessment['overall_quality'] = 'poor'
        else:
            assessment['passed_checks'].append("SNR acceptable")

        # Check clipping
        if metrics.get('clip_percentage', 0) > self.quality_thresholds['max_clip_percentage']:
            assessment['issues'].append(f"Clipping detected: {metrics.get('clip_percentage', 0):.1f}%")
            assessment['overall_quality'] = 'poor'
        else:
            assessment['passed_checks'].append("No significant clipping")

        # Check dynamic range
        if metrics.get('dynamic_range_db', 0) < self.quality_thresholds['min_dynamic_range_db']:
            assessment['issues'].append(f"Low dynamic range: {metrics.get('dynamic_range_db', 0):.1f}dB")
            if assessment['overall_quality'] == 'good':
                assessment['overall_quality'] = 'fair'
        else:
            assessment['passed_checks'].append("Dynamic range acceptable")

        return assessment

    def collect_scenario_measurements(self) -> List[MeasurementMetadata]:
        """Collect all measurements for the configured scenario"""
        if not self.scenario:
            raise ValueError("Scenario not configured")

        print(f"\n{'=' * 60}")
        print(f"COLLECTING: {self.scenario.scenario_name}")
        print(f"Description: {self.scenario.description}")
        print(f"Measurements to collect: {self.scenario.num_measurements}")
        print(f"{'=' * 60}")

        scenario_measurements = []
        successful_measurements = 0
        failed_measurements = 0

        # Warm-up measurements (discarded)
        if self.scenario.warm_up_measurements > 0:
            print(f"\nPerforming {self.scenario.warm_up_measurements} warm-up measurements...")
            for i in range(self.scenario.warm_up_measurements):
                print(f"  Warm-up {i + 1}/{self.scenario.warm_up_measurements}")
                try:
                    # Use temporary files for warm-up
                    temp_raw = "temp_warmup.wav"
                    temp_impulse = "temp_warmup_impulse.wav"

                    audio_data = self.recorder.take_record(
                        temp_raw, temp_impulse,
                        method=self.recording_method,
                        interactive=self.interactive_mode
                    )

                    # Clean up temp files
                    for temp_file in [temp_raw, temp_impulse]:
                        if os.path.exists(temp_file):
                            os.remove(temp_file)

                    time.sleep(self.scenario.measurement_interval)
                except Exception as e:
                    print(f"    Warm-up measurement failed: {e}")

        # Main measurement loop
        print(f"\nStarting main measurements...")
        for measurement_idx in range(self.scenario.num_measurements):
            print(f"\nMeasurement {measurement_idx + 1}/{self.scenario.num_measurements}")

            # Generate filenames with scenario name
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # Include milliseconds
            base_filename = f"{self.scenario.scenario_name}_{measurement_idx:03d}_{timestamp}"

            raw_filename = f"raw_{base_filename}.wav"
            impulse_filename = f"impulse_{base_filename}.wav"
            room_filename = f"room_{base_filename}.wav"

            # Full paths
            raw_path = self.scenario_dir / "raw_recordings" / raw_filename
            impulse_path = self.scenario_dir / "impulse_responses" / impulse_filename
            room_path = self.scenario_dir / "room_responses" / room_filename

            try:
                # Perform measurement using the unified API
                print(f"  Recording...")
                audio_data = self.recorder.take_record(
                    str(raw_path),
                    str(impulse_path),
                    method=self.recording_method,
                    interactive=self.interactive_mode if measurement_idx == 0 else False
                    # Only interactive on first measurement
                )

                # Handle room response file - the recorder creates it with a specific naming pattern
                expected_room_file = str(raw_path.parent / f"room_{raw_path.stem}_room.wav")
                if os.path.exists(expected_room_file):
                    os.rename(expected_room_file, str(room_path))

                if audio_data is not None:
                    # Calculate quality metrics
                    quality_metrics = self.calculate_quality_metrics(audio_data)
                    quality_assessment = self.assess_measurement_quality(quality_metrics)

                    # Create measurement metadata
                    measurement = MeasurementMetadata(
                        scenario_name=self.scenario.scenario_name,
                        measurement_index=measurement_idx,
                        timestamp=timestamp,
                        filename_raw=raw_filename,
                        filename_impulse=impulse_filename,
                        filename_room_response=room_filename,
                        signal_params=self.recorder_config,
                        quality_metrics=quality_metrics,
                        recording_stats={'samples_recorded': len(audio_data)}
                    )

                    scenario_measurements.append(measurement)
                    self.measurements.append(measurement)
                    successful_measurements += 1

                    # Print quality summary
                    print(f"  Quality: {quality_assessment['overall_quality']} "
                          f"(SNR: {quality_metrics.get('snr_db', 0):.1f}dB)")

                    if quality_assessment['issues']:
                        print(f"  Issues: {', '.join(quality_assessment['issues'])}")

                else:
                    print(f"  ❌ Measurement failed - no data recorded")
                    failed_measurements += 1

            except Exception as e:
                print(f"  ❌ Measurement failed: {e}")
                failed_measurements += 1

            # Wait before next measurement (except for the last one)
            if measurement_idx < self.scenario.num_measurements - 1:
                print(f"  Waiting {self.scenario.measurement_interval:.1f}s...")
                time.sleep(self.scenario.measurement_interval)

        print(f"\nScenario '{self.scenario.scenario_name}' completed:")
        print(f"  Successful: {successful_measurements}")
        print(f"  Failed: {failed_measurements}")
        if successful_measurements + failed_measurements > 0:
            success_rate = successful_measurements / (successful_measurements + failed_measurements) * 100
            print(f"  Success rate: {success_rate:.1f}%")

        return scenario_measurements

    def save_scenario_metadata(self):
        """Save all metadata for the scenario"""
        if not self.scenario:
            raise ValueError("Scenario not configured")

        metadata = {
            'scenario_info': {
                'scenario_name': self.scenario.scenario_name,
                'scenario_number': self.scenario.scenario_number,
                'computer_name': self.scenario.computer_name,
                'room_name': self.scenario.room_name,
                'description': self.scenario.description,
                'collection_timestamp': datetime.now().isoformat()
            },
            'recorder_config': self.recorder_config,
            'device_info': getattr(self, 'device_info', {}),
            'quality_thresholds': self.quality_thresholds,
            'measurements': [measurement.to_dict() for measurement in self.measurements],
            'summary': {
                'total_measurements': len(self.measurements),
                'planned_measurements': self.scenario.num_measurements,
                'success_rate': len(
                    self.measurements) / self.scenario.num_measurements * 100 if self.scenario.num_measurements > 0 else 0
            }
        }

        # Save to JSON file
        metadata_file = self.scenario_dir / "metadata" / f"{self.scenario.scenario_name}_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"Scenario metadata saved to: {metadata_file}")

    def generate_summary_report(self):
        """Generate a summary report of the data collection"""
        if not self.scenario:
            return

        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("ROOM RESPONSE SCENARIO COLLECTION SUMMARY")
        report_lines.append("=" * 80)
        report_lines.append(f"Scenario: {self.scenario.scenario_name}")
        report_lines.append(f"Description: {self.scenario.description}")
        report_lines.append(f"Computer: {self.scenario.computer_name}")
        report_lines.append(f"Room: {self.scenario.room_name}")
        report_lines.append(f"Dataset location: {self.scenario_dir}")
        report_lines.append("")

        # Collection summary
        success_rate = len(
            self.measurements) / self.scenario.num_measurements * 100 if self.scenario.num_measurements > 0 else 0
        report_lines.append("COLLECTION RESULTS:")
        report_lines.append("-" * 40)
        report_lines.append(f"  Planned measurements: {self.scenario.num_measurements}")
        report_lines.append(f"  Successful measurements: {len(self.measurements)}")
        report_lines.append(f"  Success rate: {success_rate:.1f}%")
        report_lines.append("")

        # Quality summary
        if self.measurements:
            all_quality = [m.quality_metrics for m in self.measurements if m.quality_metrics]
            if all_quality:
                avg_snr = np.mean([q.get('snr_db', 0) for q in all_quality])
                avg_clip = np.mean([q.get('clip_percentage', 0) for q in all_quality])

                report_lines.append("QUALITY METRICS:")
                report_lines.append("-" * 40)
                report_lines.append(f"  Average SNR: {avg_snr:.1f} dB")
                report_lines.append(f"  Average clipping: {avg_clip:.2f}%")
                report_lines.append("")

        # File structure
        report_lines.append("DATASET FILES:")
        report_lines.append("-" * 40)
        report_lines.append(f"  Raw recordings: {len(list((self.scenario_dir / 'raw_recordings').glob('*.wav')))}")
        report_lines.append(
            f"  Impulse responses: {len(list((self.scenario_dir / 'impulse_responses').glob('*.wav')))}")
        report_lines.append(f"  Room responses: {len(list((self.scenario_dir / 'room_responses').glob('*.wav')))}")

        report_lines.append("")
        report_lines.append("=" * 80)

        # Save report
        report_file = self.scenario_dir / f"{self.scenario.scenario_name}_SUMMARY.txt"
        with open(report_file, 'w') as f:
            f.write('\n'.join(report_lines))

        # Print to console
        print('\n'.join(report_lines))

    def collect_scenario(self, interactive_devices: bool = False, confirm_start = False):
        """Run the complete single scenario collection process"""
        try:

            # Setup directories and recorder
            self.setup_directories()
            self.initialize_recorder()

            # Audio quality recommendations
            print(f"\n{'=' * 60}")
            print("AUDIO QUALITY RECOMMENDATIONS:")
            print(f"{'=' * 60}")
            print("For best results:")
            print("• Ensure microphone and speakers are working properly")
            print("• Set moderate volume levels (avoid very loud or very quiet)")
            print("• Minimize background noise during recording")
            print("• Keep microphone and speakers at reasonable distance")
            print("• Test one measurement first to check signal levels")
            print(f"{'=' * 60}")

            print(f"\nCollection plan:")
            print(f"  Scenario: {self.scenario.scenario_name}")
            print(f"  Measurements: {self.scenario.num_measurements}")
            print(f"  Interval: {self.scenario.measurement_interval}s")
            estimated_time = (self.scenario.num_measurements * self.scenario.measurement_interval) / 60
            print(f"  Estimated time: {estimated_time:.1f} minutes")

            # Confirm start
            if confirm_start:
                response = input(f"\nProceed with data collection?\n"
                                 f"After pressing y recording will strt in 5 seconds\n"
                                 f"(y/n): ").strip().lower()
                if response != 'y':
                    print("Data collection cancelled.")
                    return
                time.sleep(5)
                # Collect data

            start_time = time.time()

            scenario_measurements = self.collect_scenario_measurements()

            end_time = time.time()
            session_duration = (end_time - start_time) / 60

            # Save metadata and generate report
            self.save_scenario_metadata()
            self.generate_summary_report()

            print(f"\n🎉 Scenario collection completed!")
            print(f"Duration: {session_duration:.1f} minutes")
            print(f"Dataset saved to: {self.scenario_dir}")

        except KeyboardInterrupt:
            print("\n⚠️  Data collection interrupted by user")
            if hasattr(self, 'measurements') and self.measurements:
                print("Saving partial data...")
                self.save_scenario_metadata()
                self.generate_summary_report()

        except Exception as e:
            print(f"\n❌ Error during data collection: {e}")
            raise


def main():
    """Main function for single scenario dataset collection"""
    print("Room Response Single Scenario Data Collector")
    print("=" * 60)

    # Check for interactive device selection flag
    interactive_devices = '--interactive' in sys.argv or '-i' in sys.argv

    # Configuration
    collector = SingleScenarioCollector(
        base_output_dir="room_response_dataset",
        recorder_config={
            'sample_rate': 48000,
            'pulse_duration': 0.008,
            'pulse_fade': 0.0001,
            'cycle_duration': 0.1,
            'num_pulses': 8,
            'volume': 0.4,
            'impulse_form': 'square'
        }
    )

    if interactive_devices:
        print("\nInteractive device selection mode enabled")
    else:
        print("\nUsing default audio devices (use -i flag for device selection)")

    # Run collection
    collector.collect_scenario(interactive_devices=interactive_devices)


if __name__ == "__main__":
    main()