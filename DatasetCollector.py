# DatasetCollector.py
import os
import json
import time
import queue
import numpy as np
import sys
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Any
from pathlib import Path

# Optional non-blocking keyboard on Windows
try:
    import msvcrt  # noqa: F401
    _HAS_MSVCRT = True
except Exception:
    _HAS_MSVCRT = False

from RoomResponseRecorder import RoomResponseRecorder


# ==============================
# Data classes
# ==============================

@dataclass
class ScenarioConfig:
    """Configuration for a measurement scenario"""
    scenario_number: str
    description: str
    computer_name: str
    room_name: str
    num_measurements: int = 30
    measurement_interval: float = 2.0  # seconds between measurements
    warm_up_measurements: int = 0      # optional warm-ups to discard at start
    additional_metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.additional_metadata is None:
            self.additional_metadata = {}

    @property
    def scenario_name(self) -> str:
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



# ==============================
# Collector
# ==============================

class SingleScenarioCollector:
    """
    Single-scenario data collector (recorder-API aligned):
      - Reuses an injected shared RoomResponseRecorder (if provided).
      - No interactive device prompts or method switching.
      - Append/overwrite/abort behavior for existing folders.
      - Compatibility checks vs. prior session_metadata.json (optional).
      - Pause/resume/stop via sentinel files; non-blocking keys on Windows.
    """

    def __init__(
        self,
        base_output_dir: str = "room_response_dataset",
        recorder_config: str | Dict[str, Any] = "recorderConfig.json",
        scenario_config: Dict[str, Any] | None = None,
        merge_mode: str = "append",               # 'append' (default), 'overwrite', 'abort'
        allow_config_mismatch: bool = False,      # if True, warn instead of error
        resume: bool = True,                      # continue from next index if appending
        pause_file_name: str = "PAUSE",           # drop this file into the scenario root to pause
        stop_file_name: str = "STOP",             # drop this file into the scenario root to stop
        recorder: Optional[RoomResponseRecorder] = None,
        recording_mode: str = "standard",         # 'standard' (default) or 'calibration'
        event_q: Optional[queue.Queue] = None,    # Optional event queue for progress reporting
        cmd_q: Optional[queue.Queue] = None       # Optional command queue for pause/stop control
    ):
        self.base_output_dir = Path(base_output_dir)
        self.merge_mode = merge_mode.lower()
        self.allow_config_mismatch = allow_config_mismatch
        self.resume_enabled = bool(resume)
        self.pause_file_name = pause_file_name
        self.stop_file_name = stop_file_name
        self.recording_mode = recording_mode.lower()  # Store recording mode

        # Event and command queues for GUI integration
        self.event_q = event_q
        self.cmd_q = cmd_q
        self._paused = False

        if scenario_config:
            self.setup_scenario_from_dict(**scenario_config)
        else:
            self.setup_scenario_from_input()

        # Normalize config input so we can store a dict into metadata
        self.recorder_config_path, self.recorder_config_dict = self._normalize_recorder_config(recorder_config)

        # Recorder (shared if injected)
        self.recorder: Optional[RoomResponseRecorder] = recorder

        print("/n/n++++++++++++ Debug output of the recorder parameters 1 ++++++++++++++")
        self.recorder.print_signal_analysis()

        # Collected measurements (this run)
        self.measurements: List[MeasurementMetadata] = []

        # Quality thresholds (simple heuristics; tune as you wish)
        self.quality_thresholds = {
            'min_snr_db': 15.0,
            'max_clip_percentage': 2.0,
            'min_dynamic_range_db': 25.0
        }

        # Paths populated in setup_directories()
        self.scenario_dir: Optional[Path] = None
        self.meta_dir: Optional[Path] = None
        self.meta_file: Optional[Path] = None  # metadata/session_metadata.json

        # Cached existing metadata
        self._existing_metadata: Dict[str, Any] | None = None
        self._existing_measurements_count: int = 0

    # -------------------- event helpers --------------------

    def _emit_event(self, kind: str, **payload):
        """Emit an event to the event queue if available."""
        if self.event_q:
            try:
                from gui_series_worker import WorkerEvent
                self.event_q.put_nowait(WorkerEvent(kind, payload))
            except Exception:
                pass

    def _emit_progress(self, **payload):
        """Emit a progress event."""
        self._emit_event("progress", **payload)

    def _emit_status(self, msg: str, **extra):
        """Emit a status event."""
        self._emit_event("status", message=msg, **extra)

    def _emit_error(self, msg: str, fatal: bool = False, **extra):
        """Emit an error event."""
        self._emit_event("error", message=msg, fatal=fatal, **extra)

    def _drain_commands(self):
        """Process any pending commands from the command queue."""
        if not self.cmd_q:
            return

        try:
            while True:
                from gui_series_worker import WorkerCommand
                cmd = self.cmd_q.get_nowait()
                if cmd.kind == "pause":
                    self._paused = True
                    self._emit_status("Paused by user")
                elif cmd.kind == "resume":
                    self._paused = False
                    self._emit_status("Resumed by user")
                elif cmd.kind == "stop":
                    self._emit_status("Stopping...")
                    return "stop"
        except queue.Empty:
            pass

        return None

    # -------------------- scenario setup --------------------

    def setup_scenario_from_dict(self, **parameters):
        required = {
            'scenario_number', 'description', 'computer_name', 'room_name',
            'num_measurements', 'measurement_interval'
        }
        unknown = set(parameters) - required
        missing = required - set(parameters)
        if missing:
            raise ValueError(f"Missing scenario parameter(s): {', '.join(sorted(missing))}")
        if unknown:
            raise ValueError(f"Unexpected scenario parameter(s): {', '.join(sorted(unknown))}")
        self.scenario = ScenarioConfig(**parameters)

    def setup_scenario_from_input(self):
        print("\n" + "=" * 60)
        print("ROOM RESPONSE DATA COLLECTION - SINGLE SCENARIO")
        print("=" * 60)

        computer_name = input("Enter computer name: ").strip().replace(" ", "_") or "Unknown_Computer"
        room_name = input("Enter room name: ").strip().replace(" ", "_") or "Unknown_Room"
        scenario_number = input("Enter scenario number: ").strip() or "1"
        description = input("Enter scenario description: ").strip() or f"Room response measurement scenario {scenario_number}"

        def _get_int(prompt: str, default: int) -> int:
            try:
                return int(input(prompt).strip() or str(default))
            except ValueError:
                return default

        def _get_float(prompt: str, default: float) -> float:
            try:
                return float(input(prompt).strip() or str(default))
            except ValueError:
                return default

        num_measurements = _get_int("Number of measurements (default 30): ", 30)
        interval = _get_float("Interval between measurements in seconds (default 2.0): ", 2.0)
        warmups = _get_int("Warm-up measurements to run (default 0): ", 0)

        self.scenario = ScenarioConfig(
            scenario_number=scenario_number,
            description=description,
            computer_name=computer_name,
            room_name=room_name,
            num_measurements=num_measurements,
            measurement_interval=interval,
            warm_up_measurements=warmups
        )

    # -------------------- paths & metadata --------------------

    def get_scenario_dir(self) -> Path:
        return self.scenario_dir

    def get_meta_file(self) -> Path:
        return self.meta_file

    def _normalize_recorder_config(self, cfg: str | Dict[str, Any]) -> tuple[Optional[Path], Dict[str, Any]]:
        """
        Returns (config_path, config_dict). If a path is provided, we read it.
        We always keep a dict to store in metadata, and pass the path (if any) to the recorder constructor.
        """
        if isinstance(cfg, dict):
            return None, cfg
        cfg_path = Path(cfg)
        cfg_dict: Dict[str, Any] = {}
        if cfg_path.is_file():
            try:
                cfg_dict = json.loads(cfg_path.read_text(encoding="utf-8"))
            except Exception as e:
                print(f"Warning: failed to read config file '{cfg_path}': {e}")
        else:
            print(f"Warning: config file '{cfg_path}' not found; proceeding with empty config dict.")
        return cfg_path, cfg_dict

    def setup_directories(self):
        """Prepare scenario directory & metadata paths. Honor merge_mode."""
        if not self.scenario:
            raise ValueError("Scenario not configured")

        self.scenario_dir = self.base_output_dir / self.scenario.scenario_name
        self.scenario_dir.mkdir(parents=True, exist_ok=True)

        # Subdirs
        (self.scenario_dir / "raw_recordings").mkdir(exist_ok=True)
        (self.scenario_dir / "impulse_responses").mkdir(exist_ok=True)
        (self.scenario_dir / "room_responses").mkdir(exist_ok=True)
        (self.scenario_dir / "metadata").mkdir(exist_ok=True)
        (self.scenario_dir / "analysis").mkdir(exist_ok=True)

        self.meta_dir = self.scenario_dir / "metadata"
        self.meta_file = self.meta_dir / "session_metadata.json"

        # Merge behavior
        if any(self.scenario_dir.iterdir()):
            if self.merge_mode == "abort":
                raise FileNotFoundError(
                    f"Scenario folder already exists: {self.scenario_dir}. Use merge_mode='append' or 'overwrite'."
                )
            elif self.merge_mode == "overwrite":
                # Clear subfolders (leave metadata dir present but remove files)
                for sub in ["raw_recordings", "impulse_responses", "room_responses", "analysis"]:
                    p = self.scenario_dir / sub
                    for f in p.glob("*"):
                        try:
                            f.unlink()
                        except Exception:
                            pass
                if self.meta_file.exists():
                    try:
                        self.meta_file.unlink()
                    except Exception:
                        pass

        # Load existing metadata if present (for compatibility check & resume)
        self._existing_metadata = self._load_existing_metadata()
        self._existing_measurements_count = self._count_existing_measurements()

    def _load_existing_metadata(self) -> Optional[Dict[str, Any]]:
        if self.meta_file and self.meta_file.is_file():
            try:
                return json.loads(self.meta_file.read_text(encoding="utf-8"))
            except Exception as e:
                print(f"Warning: failed to read existing metadata: {e}")
        return None

    def _count_existing_measurements(self) -> int:
        # Prefer metadata
        if self._existing_metadata and "measurements" in self._existing_metadata:
            try:
                return int(len(self._existing_metadata["measurements"]))
            except Exception:
                pass
        # Fallback: count impulses
        try:
            return len(list((self.scenario_dir / "impulse_responses").glob("*.wav")))
        except Exception:
            return 0

    # -------------------- compatibility --------------------

    def _config_compatible(self, old_cfg: Dict[str, Any], new_cfg: Dict[str, Any]) -> bool:
        """Define what 'compatible' means for appends; compare key signal params."""
        keys = [
            "sample_rate", "pulse_duration", "pulse_fade", "cycle_duration",
            "num_pulses", "volume", "pulse_frequency", "impulse_form"
        ]
        for k in keys:
            if old_cfg.get(k) != new_cfg.get(k):
                return False
        return True

    def _assert_or_warn_config_compat(self):
        if not self._existing_metadata:
            return
        old = self._existing_metadata.get("recorder_config", {})
        new = self.recorder_config_dict or {}
        if not self._config_compatible(old, new):
            msg = "Recorder config mismatch with existing scenario metadata."
            if self.allow_config_mismatch:
                print(f"⚠️  {msg} Proceeding due to allow_config_mismatch=True.")
            else:
                raise ValueError(msg + " Use allow_config_mismatch=True to override, or choose overwrite/new scenario.")

    # -------------------- recorder --------------------

    def initialize_recorder(self):
        """
        Prepare the recorder for use. Reuse injected shared recorder if provided.
        No device prompting or method switching; the UI is the source of truth.
        """
        cfg_arg = str(self.recorder_config_path) if self.recorder_config_path else self.recorder_config_dict

        if self.recorder is None:
            # CLI/standalone path: create a recorder instance from config
            self.recorder = RoomResponseRecorder(cfg_arg)
            print("+++++++++ New instance of the recorder is created +++++++++++++++")
        else:
            print("+++++++++ Shared instance of the recorder is used , debug output 2 +++++++++++++++")
            self.recorder.print_signal_analysis()
            # Shared recorder path: do NOT alter devices/settings here.
            # Optionally, if your refactored recorder exposes a non-invasive loader, you could call it here.
            pass

        # Snapshot device/sdl info for metadata (best-effort; non-fatal)
        self.device_info = {}
        try:
            if hasattr(self.recorder, "get_sdl_core_info"):
                core = self.recorder.get_sdl_core_info()
                # Keep only lightweight bits for metadata
                self.device_info = {
                    "sdl_available": bool(core.get("sdl_available")),
                    "module_version": core.get("module_version"),
                    "sdl_version": core.get("sdl_version"),
                    "device_counts": core.get("device_counts"),
                    "recorder": core.get("recorder", {}),
                    "installation_ok": core.get("installation_ok"),
                }
        except Exception:
            # Non-fatal
            self.device_info = {}

        # Optional visibility: print current signal analysis (safe no-op if method missing)
        try:
            if hasattr(self.recorder, "print_signal_analysis"):
                self.recorder.print_signal_analysis()
        except Exception:
            pass

    # -------------------- quality helpers --------------------

    def calculate_quality_metrics(self, audio_data: np.ndarray) -> Dict[str, float]:
        try:
            signal_power = float(np.mean(audio_data ** 2))
            quiet = max(1, int(0.1 * len(audio_data)))
            noise_samples = np.concatenate([audio_data[:quiet], audio_data[-quiet:]])
            noise_power = float(np.mean(noise_samples ** 2))

            snr_db = 10 * np.log10((signal_power + 1e-12) / (noise_power + 1e-12))
            max_amplitude = float(np.max(np.abs(audio_data)))
            clipped_samples = int(np.sum(np.abs(audio_data) > 0.95))
            clip_percentage = (clipped_samples / max(len(audio_data), 1)) * 100.0
            rms_level = float(np.sqrt(np.mean(audio_data ** 2)))
            dynamic_range_db = 20 * np.log10((max_amplitude + 1e-12) / (rms_level + 1e-12))

            return {
                'snr_db': float(snr_db),
                'max_amplitude': max_amplitude,
                'rms_level': rms_level,
                'clip_percentage': float(clip_percentage),
                'dynamic_range_db': float(dynamic_range_db)
            }
        except Exception as e:
            print(f"Error calculating quality metrics: {e}")
            return {}

    def assess_measurement_quality(self, metrics: Dict[str, float]) -> Dict[str, Any]:
        assessment = {'overall_quality': 'good', 'issues': [], 'passed_checks': []}
        if metrics.get('snr_db', 0) < self.quality_thresholds['min_snr_db']:
            assessment['issues'].append(f"Low SNR: {metrics.get('snr_db', 0):.1f}dB")
            assessment['overall_quality'] = 'poor'
        else:
            assessment['passed_checks'].append("SNR acceptable")

        if metrics.get('clip_percentage', 0) > self.quality_thresholds['max_clip_percentage']:
            assessment['issues'].append(f"Clipping: {metrics.get('clip_percentage', 0):.1f}%")
            assessment['overall_quality'] = 'poor'
        else:
            assessment['passed_checks'].append("No significant clipping")

        if metrics.get('dynamic_range_db', 0) < self.quality_thresholds['min_dynamic_range_db']:
            assessment['issues'].append(f"Low dynamic range: {metrics.get('dynamic_range_db', 0):.1f}dB")
            if assessment['overall_quality'] == 'good':
                assessment['overall_quality'] = 'fair'
        else:
            assessment['passed_checks'].append("Dynamic range acceptable")

        return assessment

    # -------------------- metadata persist --------------------

    def _build_base_metadata(self) -> Dict[str, Any]:
        return {
            'scenario_info': {
                'scenario_name': self.scenario.scenario_name,
                'scenario_number': self.scenario.scenario_number,
                'computer_name': self.scenario.computer_name,
                'room_name': self.scenario.room_name,
                'description': self.scenario.description,
                'collection_timestamp': datetime.now().isoformat()
            },
            'recorder_config': self.recorder_config_dict,  # store full config dict every run
            'device_info': getattr(self, 'device_info', {}),
            'quality_thresholds': self.quality_thresholds,
            'measurements': [],
            'summary': {
                'total_measurements': 0,
                'planned_measurements': self.scenario.num_measurements,
                'success_rate': 0.0
            }
        }

    def _save_metadata(self, append: bool = True):
        """Persist metadata to metadata/session_metadata.json."""
        if not self.meta_file:
            raise RuntimeError("Metadata path not initialized")

        if append and self._existing_metadata:
            meta = self._existing_metadata
            # Update high-level fields & config each time
            meta['recorder_config'] = self.recorder_config_dict
            meta['device_info'] = getattr(self, 'device_info', {})
            meta['quality_thresholds'] = self.quality_thresholds

            # Merge measurements
            meta_meas = meta.get('measurements', [])
            meta_meas.extend([m.to_dict() for m in self.measurements])
            meta['measurements'] = meta_meas

            # Update summary
            total = len(meta_meas)
            planned = self.scenario.num_measurements
            meta['summary'] = {
                'total_measurements': total,
                'planned_measurements': planned,
                'success_rate': (total / planned * 100.0) if planned > 0 else 0.0
            }
        else:
            meta = self._build_base_metadata()
            meta['measurements'] = [m.to_dict() for m in self.measurements]
            total = len(meta['measurements'])
            planned = self.scenario.num_measurements
            meta['summary']['total_measurements'] = total
            meta['summary']['planned_measurements'] = planned
            meta['summary']['success_rate'] = (total / planned * 100.0) if planned > 0 else 0.0

        # Write session_metadata.json (primary)
        self.meta_dir.mkdir(parents=True, exist_ok=True)
        self.meta_file.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        self._existing_metadata = meta  # keep in memory

        # Optional: legacy mirror (if any external tools rely on it)
        legacy = self.meta_dir / f"{self.scenario.scenario_name}_metadata.json"
        try:
            legacy.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        except Exception:
            pass

    # -------------------- pause/stop helpers --------------------

    def _check_pause_stop(self):
        """Pause if PAUSE file exists. Stop if STOP file exists. Also support Windows 'p'/'q' keys and GUI commands."""
        # Check GUI command queue first
        cmd_result = self._drain_commands()
        if cmd_result == "stop":
            print("\n🛑 Stop requested via GUI.")
            return "stop"

        # Handle GUI pause state
        while self._paused:
            time.sleep(0.1)
            cmd_result = self._drain_commands()
            if cmd_result == "stop":
                print("\n🛑 Stop requested during pause.")
                return "stop"
            # Check if resumed
            if not self._paused:
                print("▶️  Resumed via GUI.")
                break

        if self.scenario_dir:
            pause_path = self.scenario_dir / self.pause_file_name
            stop_path = self.scenario_dir / self.stop_file_name

            # STOP sentinel file
            if stop_path.exists():
                print("\n🛑 STOP file detected. Ending collection gracefully.")
                return "stop"

            # PAUSE sentinel file
            if pause_path.exists():
                print("\n⏸️  PAUSE file detected. Pausing... (remove the file to resume)")
                self._emit_status("Paused by PAUSE file")
                while pause_path.exists():
                    time.sleep(0.5)
                    # Still check GUI commands during file-based pause
                    cmd_result = self._drain_commands()
                    if cmd_result == "stop":
                        print("\n🛑 Stop requested during file pause.")
                        return "stop"
                print("▶️  Resuming.")
                self._emit_status("Resumed (PAUSE file removed)")

        # Windows non-blocking keypress
        if _HAS_MSVCRT:
            if msvcrt.kbhit():
                ch = msvcrt.getwch().lower()
                if ch == 'p':
                    print("\n⏸️  Paused. Press 'p' again to resume, 'q' to stop.")
                    self._emit_status("Paused by keyboard")
                    # wait for p or q
                    while True:
                        time.sleep(0.1)
                        # Check GUI commands during keyboard pause
                        cmd_result = self._drain_commands()
                        if cmd_result == "stop":
                            print("\n🛑 Stop requested during keyboard pause.")
                            return "stop"
                        if msvcrt.kbhit():
                            ch2 = msvcrt.getwch().lower()
                            if ch2 == 'p':
                                print("▶️  Resuming.")
                                self._emit_status("Resumed by keyboard")
                                break
                            if ch2 == 'q':
                                print("🛑 Stopping on user request (q).")
                                return "stop"
                elif ch == 'q':
                    print("\n🛑 Stopping on user request (q).")
                    return "stop"

        return None

    # -------------------- main loops --------------------

    def collect_scenario_measurements(self) -> List[MeasurementMetadata]:
        if not self.scenario:
            raise ValueError("Scenario not configured")

        print("\n" + "=" * 60)
        print(f"COLLECTING: {self.scenario.scenario_name}")
        print(f"Description: {self.scenario.description}")
        print(f"Planned measurements this run: {self.scenario.num_measurements}")
        print("=" * 60)

        # Compatibility check (if appending and metadata exists)
        if self.merge_mode != "overwrite":
            self._assert_or_warn_config_compat()

        # Determine starting index when resuming
        start_index = 0
        if self.merge_mode == "append" and self.resume_enabled:
            start_index = self._existing_measurements_count

        scenario_measurements: List[MeasurementMetadata] = []
        successful_measurements = 0
        failed_measurements = 0

        # Track cumulative calibration stats across all measurements
        total_valid_cycles = 0
        total_cycles = 0

        # Optional warm-up(s) at start (when not resuming mid-run)
        if self.scenario.warm_up_measurements > 0 and start_index == 0:
            print(f"\nPerforming {self.scenario.warm_up_measurements} warm-up measurements...")
            for i in range(self.scenario.warm_up_measurements):
                print(f"  Warm-up {i + 1}/{self.scenario.warm_up_measurements}")
                try:
                    # Note: Warm-up files are temporary and will be deleted immediately
                    # In calibration mode, files are saved to temp location as well
                    _ = self.recorder.take_record("temp_warmup.wav", "temp_warmup_impulse.wav", mode=self.recording_mode)
                except Exception as e:
                    print(f"    Warm-up failed: {e}")

                # Clean up temporary files (including multi-channel files in calibration mode)
                for tmp in ["temp_warmup.wav", "temp_warmup_impulse.wav"]:
                    try:
                        Path(tmp).unlink()
                    except Exception:
                        pass

                # Also clean up any temp multi-channel files
                temp_dir = Path(".")
                for f in temp_dir.glob("temp_warmup*_ch*.wav"):
                    try:
                        f.unlink()
                    except Exception:
                        pass
                for f in temp_dir.glob("temp_warmup*_ch*.npy"):
                    try:
                        f.unlink()
                    except Exception:
                        pass
                for f in temp_dir.glob("room_temp_warmup*"):
                    try:
                        f.unlink()
                    except Exception:
                        pass

                time.sleep(self.scenario.measurement_interval)

        # Main loop
        print("\nStarting main measurements...")
        self._emit_status(f"Starting {self.scenario.num_measurements} measurements")

        for local_idx in range(self.scenario.num_measurements):
            # Pause/stop checks
            action = self._check_pause_stop()
            if action == "stop":
                break

            absolute_idx = start_index + local_idx
            print(f"\nMeasurement {local_idx + 1}/{self.scenario.num_measurements} (absolute index: {absolute_idx})")

            # Emit progress before measurement
            self._emit_progress(
                scenario=self.scenario.scenario_name,
                local_index=local_idx + 1,
                total_measurements=self.scenario.num_measurements,
                absolute_index=absolute_idx,
                successful_measurements=successful_measurements,
                failed_measurements=failed_measurements
            )

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            base = f"{self.scenario.scenario_name}_{absolute_idx:03d}_{timestamp}"

            raw_filename = f"raw_{base}.wav"
            impulse_filename = f"impulse_{base}.wav"
            room_filename = f"room_{base}.wav"

            raw_path = self.scenario_dir / "raw_recordings" / raw_filename
            impulse_path = self.scenario_dir / "impulse_responses" / impulse_filename
            room_path = self.scenario_dir / "room_responses" / room_filename

            try:
                audio_data = self.recorder.take_record(
                    str(raw_path),
                    str(impulse_path),
                    mode=self.recording_mode
                )

                # If the recorder writes the room response under a derived name, relocate it to our target name.
                # (Best-effort, safe no-op if the expected file doesn't exist.)
                derived_room = raw_path.parent / f"room_{raw_path.stem}_room.wav"
                if derived_room.exists() and not room_path.exists():
                    try:
                        derived_room.rename(room_path)
                    except Exception:
                        pass

                if audio_data is not None:
                    # Handle mode-specific quality metrics and metadata
                    if self.recording_mode == 'calibration' and isinstance(audio_data, dict):
                        # CALIBRATION MODE: Use validation results from processing pipeline
                        # Files are already saved by take_record() with correct alignment

                        metadata = audio_data.get('metadata', {})
                        validation_results = audio_data.get('validation_results', [])

                        # Extract quality info from calibration validation
                        num_total = len(validation_results)
                        num_valid = metadata.get('num_valid_cycles', 0)
                        num_aligned = metadata.get('num_aligned_cycles', 0)

                        print(f"  ✓ Calibration: {num_valid}/{num_total} valid cycles, {num_aligned} aligned")

                        # Update cumulative totals
                        total_valid_cycles += num_valid
                        total_cycles += num_total

                        # Emit progress with both per-measurement and cumulative stats
                        self._emit_progress(
                            scenario=self.scenario.scenario_name,
                            local_index=local_idx + 1,
                            total_measurements=self.scenario.num_measurements,
                            absolute_index=absolute_idx,
                            successful_measurements=successful_measurements,
                            failed_measurements=failed_measurements,
                            valid_cycles=num_valid,
                            total_cycles=num_total,
                            aligned_cycles=num_aligned,
                            cumulative_valid_cycles=total_valid_cycles,
                            cumulative_total_cycles=total_cycles
                        )

                        # Check for zero valid cycles - auto-stop if detected
                        if num_valid == 0:
                            error_msg = f"⚠️ Zero valid cycles detected in measurement {local_idx + 1}. Stopping collection."
                            print(f"\n{error_msg}")
                            self._emit_error(error_msg, fatal=True)
                            break

                        # Store calibration-specific quality metrics
                        q = {
                            'mode': 'calibration',
                            'total_cycles': num_total,
                            'valid_cycles': num_valid,
                            'aligned_cycles': num_aligned,
                            'validation_pass_rate': (num_valid / max(num_total, 1)) * 100.0,
                            'alignment_pass_rate': (num_aligned / max(num_valid, 1)) * 100.0 if num_valid > 0 else 0.0,
                            'normalize_by_calibration': metadata.get('normalize_by_calibration', False)
                        }

                        # Get sample count from raw audio for metadata
                        raw_audio_dict = audio_data.get('raw', {})
                        if raw_audio_dict:
                            first_ch = list(raw_audio_dict.values())[0]
                            samples_count = int(len(first_ch))
                        else:
                            samples_count = 0

                    else:
                        # STANDARD MODE: Use traditional quality metrics
                        # Extract raw audio for quality metrics
                        if isinstance(audio_data, dict):
                            # Multi-channel: get first channel
                            raw_for_metrics = list(audio_data.values())[0] if audio_data else None
                        else:
                            # Single-channel
                            raw_for_metrics = audio_data

                        # Calculate quality metrics on raw audio
                        if raw_for_metrics is not None:
                            q = self.calculate_quality_metrics(raw_for_metrics)
                            _ = self.assess_measurement_quality(q)  # currently informational
                            samples_count = int(len(raw_for_metrics))
                        else:
                            q = {}
                            samples_count = 0

                    mm = MeasurementMetadata(
                        scenario_name=self.scenario.scenario_name,
                        measurement_index=absolute_idx,
                        timestamp=timestamp,
                        filename_raw=raw_filename,
                        filename_impulse=impulse_filename,
                        filename_room_response=room_filename,
                        signal_params=self.recorder_config_dict,
                        quality_metrics=q,
                        recording_stats={'samples_recorded': samples_count}
                    )
                    scenario_measurements.append(mm)
                    self.measurements.append(mm)
                    successful_measurements += 1

                    # Persist after each measurement so we can resume safely
                    self._save_metadata(append=True)
                else:
                    print("  ❌ Measurement failed - no data recorded")
                    failed_measurements += 1

            except Exception as e:
                print(f"  ❌ Measurement failed: {e}")
                failed_measurements += 1

            if local_idx < self.scenario.num_measurements - 1:
                print(f"  Waiting {self.scenario.measurement_interval:.1f}s...")
                time.sleep(self.scenario.measurement_interval)

        print(f"\nScenario '{self.scenario.scenario_name}' run complete:")
        print(f"  Successful: {successful_measurements}")
        print(f"  Failed: {failed_measurements}")
        total_this_run = successful_measurements + failed_measurements
        if total_this_run > 0:
            success_rate = successful_measurements / total_this_run * 100.0
            print(f"  Success rate this run: {success_rate:.1f}%")

        return scenario_measurements

    def save_scenario_metadata(self):
        """Final write (metadata is also saved incrementally after each measurement)."""
        self._save_metadata(append=True)

    def generate_summary_report(self):
        if not self.scenario or not self.scenario_dir:
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

        # Summaries (from latest metadata)
        meta = self._existing_metadata or {}
        total = len(meta.get("measurements", []))
        planned = meta.get("summary", {}).get("planned_measurements", self.scenario.num_measurements)
        success_rate = (total / planned * 100.0) if planned else 0.0

        report_lines.append("COLLECTION RESULTS:")
        report_lines.append("-" * 40)
        report_lines.append(f"  Planned measurements (this run): {self.scenario.num_measurements}")
        report_lines.append(f"  Total measurements accumulated: {total}")
        report_lines.append(f"  Success rate vs planned (this run): {success_rate:.1f}%")
        report_lines.append("")

        # Quick file counts
        raw_cnt = len(list((self.scenario_dir / 'raw_recordings').glob('*.wav')))
        imp_cnt = len(list((self.scenario_dir / 'impulse_responses').glob('*.wav')))
        room_cnt = len(list((self.scenario_dir / 'room_responses').glob('*.wav')))
        report_lines.append("DATASET FILES:")
        report_lines.append("-" * 40)
        report_lines.append(f"  Raw recordings: {raw_cnt}")
        report_lines.append(f"  Impulse responses: {imp_cnt}")
        report_lines.append(f"  Room responses: {room_cnt}")
        report_lines.append("")
        report_lines.append("=" * 80)

        report_file = self.scenario_dir / f"{self.scenario.scenario_name}_SUMMARY.txt"
        report_file.write_text("\n".join(report_lines), encoding="utf-8")
        print("\n".join(report_lines))

    def collect_scenario(self, interactive_devices: bool = False, confirm_start: bool = False):
        """
        Run the complete collection with pause/resume & compatibility.

        NOTE: 'interactive_devices' is retained for backward compatibility with the UI,
        but it is ignored (no interactive recorder flow in the new API).
        """
        try:
            self.setup_directories()
            self.initialize_recorder()

            print("\n" + "=" * 60)
            print("AUDIO QUALITY RECOMMENDATIONS:")
            print("=" * 60)
            print("• Quiet room; moderate volume; stable mic/speaker placement.")
            print("• You can PAUSE by creating a file named 'PAUSE' in the scenario folder,")
            print("  or press 'p' on Windows console. Remove PAUSE file or press 'p' again to resume.")
            print("• Create 'STOP' file (or press 'q' on Windows) to end gracefully.")
            print("=" * 60)

            print("\nCollection plan:")
            print(f"  Scenario: {self.scenario.scenario_name}")
            print(f"  Measurements (this run): {self.scenario.num_measurements}")
            print(f"  Interval: {self.scenario.measurement_interval}s")

            if confirm_start:
                resp = input("\nProceed with data collection? (y/n): ").strip().lower()
                if resp != 'y':
                    print("Data collection cancelled.")
                    return
                time.sleep(0.5)

            start = time.time()
            _ = self.collect_scenario_measurements()
            dur_min = (time.time() - start) / 60.0

            # Finalize
            self.save_scenario_metadata()
            self.generate_summary_report()

            print("\n✅ Scenario collection completed.")
            print(f"Duration: {dur_min:.1f} minutes")
            print(f"Dataset saved to: {self.scenario_dir}")

        except KeyboardInterrupt:
            print("\n⚠️  Collection interrupted by user (Ctrl+C). Saving partial data...")
            self.save_scenario_metadata()
            self.generate_summary_report()
        except Exception as e:
            print(f"\n❌ Error during data collection: {e}")
            raise


# ==============================
# CLI entry point
# ==============================

def main():
    """Main function for single scenario dataset collection (standalone use)"""
    print("Room Response Single Scenario Data Collector")
    print("=" * 60)

    # Build a collector with inline defaults (CLI path)
    collector = SingleScenarioCollector(
        base_output_dir="room_response_dataset",
        recorder_config={
            'sample_rate': 48000,
            'pulse_duration': 0.008,
            'pulse_fade': 0.0001,
            'cycle_duration': 0.1,
            'num_pulses': 8,
            'volume': 0.4,
            'impulse_form': 'sine'
        }
    )

    # Run collection
    collector.collect_scenario(interactive_devices=False)


if __name__ == "__main__":
    main()
