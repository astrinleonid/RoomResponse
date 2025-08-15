# gui_series_worker.py — v5
from __future__ import annotations
import time, json, queue, threading, math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime

from DatasetCollector import SingleScenarioCollector, MeasurementMetadata
from RoomResponseRecorder import RoomResponseRecorder

try:
    import sdl_audio_core as sdl
except Exception:
    sdl = None

@dataclass
class WorkerCommand:
    kind: str  # 'start'|'pause'|'resume'|'stop'
    payload: Optional[Dict[str, Any]] = None

@dataclass
class WorkerEvent:
    kind: str  # 'status'|'progress'|'error'|'done'|'heartbeat'
    payload: Dict[str, Any]

class SeriesWorker(threading.Thread):
    INIT = "INIT"; PRECHECK = "PRECHECK"; RUNNING = "RUNNING"; PAUSED = "PAUSED"; STOPPING = "STOPPING"; DONE = "DONE"
    def __init__(self,
        scenario_numbers: List[str], base_output_dir: str, config_file: str,
        base_computer: str, base_room: str,
        num_measurements: int, measurement_interval: float,
        description_template: str = "Room response measurement scenario {n}",
        warm_up_measurements: int = 0,
        warmup_spacing: str = "burst",  # 'burst' or 'respect_interval'
        inter_delay: float = 60.0, pre_delay: float = 60.0,
        event_q: Optional[queue.Queue] = None, cmd_q: Optional[queue.Queue] = None,
        rename_wait_s: float = 2.0, metadata_flush_every: int = 10, metadata_flush_seconds: float = 30.0,
        pause_file_name: str = "PAUSE", stop_file_name: str = "STOP",
        record_timeout_factor: float = 6.0,
        record_timeout_s: Optional[float] = None,
        enable_beeps: bool = True, beep_volume: float = 0.2, beep_freq: int = 880, beep_dur_ms: int = 200,
        interval_mode: str = "end_to_start",
    ):
        super().__init__(daemon=True)
        self.scenario_numbers = list(scenario_numbers)
        self.base_output_dir = Path(base_output_dir)
        self.config_file = config_file
        self.base_computer = base_computer; self.base_room = base_room
        self.num_measurements = int(num_measurements); self.measurement_interval = float(measurement_interval)
        self.description_template = description_template
        self.warm_up_measurements = int(max(0, warm_up_measurements))
        self.warmup_spacing = warmup_spacing if warmup_spacing in ("burst", "respect_interval") else "burst"
        self.inter_delay = float(max(0.0, inter_delay)); self.pre_delay = float(max(0.0, pre_delay))
        self.rename_wait_s = float(max(0.0, rename_wait_s))
        self.metadata_flush_every = max(1, int(metadata_flush_every)); self.metadata_flush_seconds = float(max(5.0, metadata_flush_seconds))
        self.pause_file_name = pause_file_name; self.stop_file_name = stop_file_name
        self.record_timeout_factor = float(max(1.0, record_timeout_factor)); self.record_timeout_s = record_timeout_s
        self.enable_beeps = bool(enable_beeps); self.beep_volume = float(beep_volume); self.beep_freq = int(beep_freq); self.beep_dur_ms = int(beep_dur_ms)
        self.interval_mode = interval_mode if interval_mode in ("end_to_start", "start_to_start") else "end_to_start"
        self.event_q = event_q or queue.Queue(); self.cmd_q = cmd_q or queue.Queue()
        self.state = self.PRECHECK; self._stop_requested = False; self._paused = False; self._paused_by_file = False
        self._recorder: Optional[RoomResponseRecorder] = None
        self._total_measurements = 0; self._last_flush_ts = time.time(); self._last_heartbeat = 0.0
        self._beep_engine = None; self._sample_rate = 48000

    def _emit(self, kind: str, **payload):
        try: self.event_q.put_nowait(WorkerEvent(kind, payload))
        except Exception: pass
    def _emit_status(self, msg: str, **extra): self._emit("status", state=self.state, message=msg, **extra)
    def _emit_error(self, msg: str, fatal: bool = False, **extra): self._emit("error", state=self.state, fatal=fatal, message=msg, **extra)
    def _emit_progress(self, **payload): self._emit("progress", state=self.state, **payload)
    def _emit_heartbeat(self, **payload):
        now = time.time()
        if now - self._last_heartbeat >= 1.0:
            self._last_heartbeat = now; self._emit("heartbeat", state=self.state, **payload)

    def enqueue_command(self, cmd: WorkerCommand):
        try: self.cmd_q.put_nowait(cmd)
        except Exception: pass
    def stop(self): self.enqueue_command(WorkerCommand("stop"))

    def run(self):
        try: self._loop()
        except Exception as e:
            self._emit_error(f"Worker crashed: {e}", fatal=True); self.state = self.DONE; self._emit("done", ok=False)

    def _loop(self):
        if self.pre_delay > 0:
            self._emit_status("pre-delay", seconds=self.pre_delay); self._wait_with_ticks(self.pre_delay, allow_pause=True)
            if self._stop_requested: return self._finalize(ok=False, reason="stopped during pre-delay")
        self._emit_status("init recorder")
        try: self._init_recorder_once()
        except Exception as e: return self._finalize(ok=False, reason=f"recorder init failed: {e}")
        if self.enable_beeps: self._init_beeper_once()
        if self.warm_up_measurements > 0:
            self._emit_status("warmup", count=self.warm_up_measurements)
            try: self._run_warmup(self.warm_up_measurements)
            except Exception as e: self._emit_error(f"warmup failed: {e}")
        self.state = self.RUNNING; self._emit_status("running", interval_mode=self.interval_mode)
        for s_idx, scen_no in enumerate(self.scenario_numbers, start=1):
            if self._stop_requested: return self._finalize(ok=False, reason="stopped")
            ok = self._run_one_scenario(str(scen_no), s_idx, len(self.scenario_numbers))
            if not ok: return self._finalize(ok=False, reason=f"scenario {scen_no} failed")
            if self.enable_beeps: self._beep(count=1)
            if s_idx < len(self.scenario_numbers) and self.inter_delay > 0:
                self._emit_status("inter-delay", seconds=self.inter_delay); self._wait_with_ticks(self.inter_delay, allow_pause=True)
                if self._stop_requested: return self._finalize(ok=False, reason="stopped during inter-delay")
        if self.enable_beeps: self._beep(count=2)
        return self._finalize(ok=True)

    def _finalize(self, ok: bool, reason: str | None = None):
        self.state = self.STOPPING; self._emit_status("finalizing", ok=ok, reason=reason)
        try:
            if self._recorder is not None: self._recorder.shutdown()
            if self._beep_engine is not None:
                try: self._beep_engine.stop(); self._beep_engine.shutdown()
                except Exception: pass
        except Exception: pass
        self.state = self.DONE; self._emit("done", ok=ok, reason=reason)

    def _init_recorder_once(self):
        self._recorder = RoomResponseRecorder(self.config_file)
        devices = self._recorder.list_devices()
        if not devices or not devices.get('input_devices') or not devices.get('output_devices'):
            raise RuntimeError("no suitable audio devices")
        self._emit_status("devices ready",
            input=devices['input_devices'][0].name if devices['input_devices'] else "?",
            output=devices['output_devices'][0].name if devices['output_devices'] else "?")

    def _load_sample_rate(self) -> int:
        try:
            cfg = json.loads(Path(self.config_file).read_text(encoding="utf-8"))
            sr = int(cfg.get("sample_rate", 48000)); return sr if sr > 0 else 48000
        except Exception: return 48000

    def _init_beeper_once(self):
        if not self.enable_beeps or sdl is None: return
        try:
            self._sample_rate = self._load_sample_rate()
            eng = sdl.AudioEngine(); cfg = sdl.AudioEngineConfig()
            cfg.sample_rate = int(self._sample_rate); cfg.buffer_size = 1024
            cfg.input_device_id = -1; cfg.output_device_id = -1; cfg.enable_logging = False
            if not eng.initialize(cfg): raise RuntimeError("AudioEngine.initialize failed")
            if not eng.start(): raise RuntimeError("AudioEngine.start failed")
            self._beep_engine = eng; self._emit_status("beeper ready", sr=int(self._sample_rate))
        except Exception as e:
            self._emit_error(f"beeper init failed: {e}"); self._beep_engine = None

    def _beep(self, count: int = 1):
        if not self.enable_beeps or self._beep_engine is None: return
        try:
            sr = int(self._sample_rate); n = max(1, int(sr * (self.beep_dur_ms / 1000.0)))
            t = [i / sr for i in range(n)]
            wave = [self.beep_volume * math.sin(2.0 * math.pi * float(self.beep_freq) * tt) for tt in t]
            for _ in range(max(1, int(count))):
                if not self._beep_engine.start_playback(wave): break
                self._beep_engine.wait_for_playback_completion(self.beep_dur_ms + 150)
                time.sleep(0.05)
        except Exception as e:
            self._emit_error(f"beep failed: {e}")

    # ---------------- Warm‑up ----------------
    def _run_warmup(self, n: int):
        for i in range(n):
            if self._stop_requested: return
            # Allow pausing before each warm‑up shot
            self._pre_measurement_gate(None)
            try:
                _ = self._recorder.take_record("/dev/null", "/dev/null", method=2, interactive=False)
            except Exception:
                pass
            # spacing choice
            if self.warmup_spacing == "respect_interval" and (i < n - 1):
                self._wait_with_ticks(self.measurement_interval, allow_pause=True)
            elif self.warmup_spacing == "burst":
                time.sleep(0.05)

    # ---------------- Scenario loop ----------------
    def _run_one_scenario(self, scen_no: str, idx: int, total: int) -> bool:
        desc = self.description_template.replace("{n}", str(scen_no))
        sc = SingleScenarioCollector(
            base_output_dir=str(self.base_output_dir), recorder_config=self.config_file,
            scenario_config={
                "scenario_number": str(scen_no), "description": desc,
                "computer_name": self.base_computer, "room_name": self.base_room,
                "num_measurements": self.num_measurements, "measurement_interval": self.measurement_interval,
            },
            merge_mode="append", allow_config_mismatch=False, resume=True,
        )
        sc.setup_directories()
        self._emit_status("scenario-start", scenario=sc.scenario.scenario_name, index=idx, total=total)

        last_start = None  # used in start_to_start mode
        for local_idx in range(self.num_measurements):
            if self._stop_requested: return False

            # ---- PRE‑MEASUREMENT GATE (reacts to Pause/Stop before recording) ----
            self._pre_measurement_gate(sc.scenario_dir)
            if self._stop_requested: return False

            # ---- Wait strategy for start→start cadence ----
            if self.interval_mode == "start_to_start" and last_start is not None:
                target = last_start + self.measurement_interval
                wait = max(0.0, target - time.time())
                if wait > 0:
                    self._wait_with_ticks(wait, allow_pause=True, scenario_dir=sc.scenario_dir)
                    if self._stop_requested: return False

            start_ts = time.time(); last_start = start_ts

            # ---- Recording ----
            m_abs_idx = sc._existing_measurements_count + local_idx
            try:
                ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                base = f"{sc.scenario.scenario_name}_{m_abs_idx:03d}_{ts}"
                raw_filename = f"raw_{base}.wav"; impulse_filename = f"impulse_{base}.wav"; room_filename = f"room_{base}.wav"
                raw_path = sc.scenario_dir / "raw_recordings" / raw_filename
                impulse_path = sc.scenario_dir / "impulse_responses" / impulse_filename
                room_path = sc.scenario_dir / "room_responses" / room_filename

                t_rec_s = time.time()
                timeout_s = self._effective_record_timeout()
                audio_data = self._take_record_with_timeout(str(raw_path), str(impulse_path), timeout_s)
                t_rec_e = time.time()

                expected_room_file = raw_path.parent / f"room_{raw_path.stem}_room.wav"
                t_io_s = time.time(); self._bounded_wait_for_file(expected_room_file, self.rename_wait_s, scenario_dir=sc.scenario_dir)
                if expected_room_file.exists():
                    try: expected_room_file.rename(room_path)
                    except Exception: pass
                t_io_e = time.time()

                q = sc.calculate_quality_metrics(audio_data) if audio_data is not None else {}
                _ = sc.assess_measurement_quality(q) if q else {}
                mm = MeasurementMetadata(
                    scenario_name=sc.scenario.scenario_name, measurement_index=m_abs_idx, timestamp=ts,
                    filename_raw=raw_filename, filename_impulse=impulse_filename, filename_room_response=room_filename,
                    signal_params=sc.recorder_config_dict, quality_metrics=q,
                    recording_stats={'samples_recorded': int(len(audio_data)) if audio_data is not None else 0},
                )
                sc.measurements.append(mm)
                self._total_measurements += 1
                if (self._total_measurements % self.metadata_flush_every == 0) or (time.time() - self._last_flush_ts) >= self.metadata_flush_seconds:
                    sc._save_metadata(append=True); self._last_flush_ts = time.time()

                self._emit_progress(
                    scenario=sc.scenario.scenario_name, local_index=local_idx + 1,
                    total_per_scenario=self.num_measurements, absolute_index=m_abs_idx,
                    record_ms=int((t_rec_e - t_rec_s) * 1000), io_ms=int((t_io_e - t_io_s) * 1000),
                )
            except TimeoutError as te:
                self._emit_error(f"record timeout: {te}; reinitializing recorder and retrying once")
                if not self._reinit_recorder():
                    self._emit_error("recorder reinit failed after timeout", fatal=False); return False
                try:
                    audio_data = self._take_record_with_timeout(str(raw_path), str(impulse_path), self._effective_record_timeout())
                except Exception as e:
                    self._emit_error(f"retry failed after reinit: {e}"); return False
            except Exception as e:
                self._emit_error(f"measurement failed: {e}")

            # ---- Cooldown after each measurement (end_to_start mode) ----
            if self.interval_mode == "end_to_start" and (local_idx < self.num_measurements - 1):
                self._emit_status("cooldown", seconds=self.measurement_interval)
                self._wait_with_ticks(self.measurement_interval, allow_pause=True, scenario_dir=sc.scenario_dir)
                if self._stop_requested: return False

        try:
            sc._save_metadata(append=True); sc.generate_summary_report()
        except Exception as e:
            self._emit_error(f"metadata flush failed: {e}")
        self._emit_status("scenario-end", scenario=sc.scenario.scenario_name)
        return True

    def _effective_record_timeout(self) -> float:
        if self.record_timeout_s and self.record_timeout_s > 0:
            return float(self.record_timeout_s)
        return max(10.0, self.record_timeout_factor * self.measurement_interval)

    def _bounded_wait_for_file(self, path: Path, timeout_s: float, scenario_dir: Optional[Path] = None):
        if timeout_s <= 0: return
        t0 = time.time()
        while time.time() - t0 < timeout_s and not self._stop_requested:
            self._drain_commands()
            if scenario_dir is not None:
                if (scenario_dir / self.stop_file_name).exists():
                    self._stop_requested = True; break
                if (scenario_dir / self.pause_file_name).exists() and not self._paused:
                    self._paused = True; self._paused_by_file = True; self.state = self.PAUSED; self._emit_status("paused by PAUSE file")
            if self._paused:
                time.sleep(0.1)
                continue
            if path.exists():
                return
            time.sleep(0.02)

    def _take_record_with_timeout(self, raw_path: str, impulse_path: str, timeout_s: float):
        result: Dict[str, Any] = {}
        def _runner():
            try:
                result['audio'] = self._recorder.take_record(raw_path, impulse_path, method=2, interactive=False)
            except Exception as e:
                result['error'] = e
        th = threading.Thread(target=_runner, daemon=True); th.start(); th.join(timeout_s)
        if th.is_alive():
            raise TimeoutError(f"take_record exceeded {timeout_s:.1f}s")
        if 'error' in result: raise result['error']
        return result.get('audio')

    def _reinit_recorder(self) -> bool:
        try:
            if self._recorder is not None:
                try: self._recorder.shutdown()
                except Exception: pass
            self._init_recorder_once(); return True
        except Exception: return False

    # ---- Pause/Stop helpers ----
    def _pre_measurement_gate(self, scenario_dir: Optional[Path]):
        """Block here until not paused and no PAUSE file, reacting to Stop immediately."""
        while not self._stop_requested:
            self._drain_commands()
            if scenario_dir is not None:
                if (scenario_dir / self.stop_file_name).exists():
                    self._stop_requested = True; self._emit_status("stop-file detected"); break
                pause_exists = (scenario_dir / self.pause_file_name).exists()
                if pause_exists and not self._paused:
                    self._paused = True; self._paused_by_file = True; self.state = self.PAUSED; self._emit_status("paused by PAUSE file")
                if (not pause_exists) and self._paused and self._paused_by_file:
                    self._paused = False; self._paused_by_file = False; self.state = self.RUNNING; self._emit_status("resumed (PAUSE file removed)")
            if not self._paused:
                return
            self._emit_heartbeat()
            time.sleep(0.1)

    def _wait_with_ticks(self, seconds: float, allow_pause: bool, scenario_dir: Optional[Path] = None):
        remaining = float(seconds); tick = 0.1 if seconds > 1.0 else 0.02; t_start = time.time()
        while remaining > 0 and not self._stop_requested:
            self._drain_commands()
            if scenario_dir is not None:
                stop_exists = (scenario_dir / self.stop_file_name).exists()
                pause_exists = (scenario_dir / self.pause_file_name).exists()
                if stop_exists:
                    self._stop_requested = True; self._emit_status("stop-file detected"); break
                if allow_pause:
                    if pause_exists and not self._paused:
                        self._paused = True; self._paused_by_file = True; self.state = self.PAUSED; self._emit_status("paused by PAUSE file")
                    if (not pause_exists) and self._paused and self._paused_by_file:
                        self._paused = False; self._paused_by_file = False; self.state = self.RUNNING; self._emit_status("resumed (PAUSE file removed)")
            if allow_pause and self._paused and not self._paused_by_file:
                self.state = self.PAUSED
            self._emit_heartbeat(remaining=max(0.0, remaining))
            time.sleep(0.1 if self._paused else min(tick, max(0.0, remaining)))
            remaining = seconds - (time.time() - t_start)

    def _drain_commands(self):
        try:
            while True:
                cmd = self.cmd_q.get_nowait()
                if cmd.kind == "pause":
                    self._paused = True; self._paused_by_file = False; self.state = self.PAUSED; self._emit_status("paused by command")
                elif cmd.kind == "resume":
                    self._paused = False; self._paused_by_file = False; self.state = self.RUNNING; self._emit_status("resumed (command)")
                elif cmd.kind == "stop":
                    self._stop_requested = True; self._emit_status("stopping")
        except queue.Empty:
            pass