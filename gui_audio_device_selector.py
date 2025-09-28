#!/usr/bin/env python3
"""
gui_audio_device_selector.py

Recorder-centric device selector with live Mic monitor:
- Enumerates devices via recorder.get_sdl_core_info()
- Input: select device + channel, start/stop live Mic monitor (UI progress bar)
- Output: select device and test speakers (via RoomResponseRecorder)
- No direct sdl_audio_core usage here; MicTesting owns it internally.
"""

from __future__ import annotations

import threading
import queue
import time
import numpy as np
from typing import Any, Dict, List, Optional

# Streamlit
try:
    import streamlit as st
except ImportError:
    class _Stub:
        def __getattr__(self, name):
            def _f(*a, **k):
                print(f"[STUB streamlit.{name}] args={a} kwargs={k}")
            return _f
    st = _Stub()

# Recorder (type only)
try:
    from RoomResponseRecorder import RoomResponseRecorder
except Exception:
    RoomResponseRecorder = None  # type: ignore

# MicTesting (owns sdl bindings internally)
try:
    import MicTesting
    MICTESTING_AVAILABLE = True
except Exception:
    MicTesting = None  # type: ignore
    MICTESTING_AVAILABLE = False


class AudioDeviceSelector:
    """Streamlit device selection widget group, fully recorder-centric."""

    def __init__(self, recorder: Optional["RoomResponseRecorder"]) -> None:
        self.recorder = recorder

        # Shared UI state defaults
        ss = st.session_state
        ss.setdefault('audio_device_cache', {'input': [], 'output': []})
        ss.setdefault('audio_selected_input_device', 'System Default')
        ss.setdefault('audio_selected_output_device', 'System Default')
        ss.setdefault('audio_input_channel', 0)
        ss.setdefault('audio_buffer_size', 512)
        ss.setdefault('audio_sample_rate', int(getattr(recorder, 'sample_rate', 48000) if recorder else 48000))

        # Mic monitor state
        ss.setdefault('mic_running', False)
        ss.setdefault('mic_chunk_sec', 0.1)
        ss.setdefault('mic_level_db', -60.0)  # last measured dB
        ss.setdefault('mic_queue', None)      # queue.Queue[float]
        ss.setdefault('mic_thread', None)     # threading.Thread
        ss.setdefault('mic_stop_event', None) # threading.Event

    # -------------
    # UI Entrypoint
    # -------------

    def render(self) -> None:
        st.markdown("### Devices")

        # Refresh + scan
        self._render_scan_row()

        # Input / Output selectors
        in_col, out_col = st.columns(2)
        with in_col:
            self._render_input_selector()
        with out_col:
            self._render_output_selector()

        st.markdown("---")

        # Input channel + Mic monitor (live)
        st.markdown("#### Microphone (Input)")
        ch_col, mon_col = st.columns([1, 2])
        with ch_col:
            self._render_input_channel_picker()
        with mon_col:
            self._render_mic_monitor_controls_and_meter()

        st.markdown("---")

        # Speaker test (output)
        st.markdown("#### Speakers (Output)")
        self._render_speaker_test_controls()

    # --------------------
    # Device scan & helpers
    # --------------------

    def _render_scan_row(self) -> None:
        scan_cols = st.columns([1, 2, 2])
        with scan_cols[0]:
            if st.button("Refresh Devices"):
                self._scan_and_cache_devices()
        with scan_cols[1]:
            cache = st.session_state['audio_device_cache']
            st.caption(f"Inputs: {len(cache['input'])} | Outputs: {len(cache['output'])}")
        with scan_cols[2]:
            if self.recorder:
                inp = getattr(self.recorder, 'input_device', -1)
                out = getattr(self.recorder, 'output_device', -1)
                st.caption(f"Selected → In: {'Default' if inp == -1 else inp} | Out: {'Default' if out == -1 else out}")

        # Initial scan if cache empty
        cache = st.session_state['audio_device_cache']
        if not cache['input'] and not cache['output']:
            self._scan_and_cache_devices()

    def _scan_and_cache_devices(self) -> None:
        input_list: List[Dict[str, Any]] = []
        output_list: List[Dict[str, Any]] = []

        if not self.recorder or not hasattr(self.recorder, "get_sdl_core_info"):
            st.warning("Recorder not available to list devices.")
            st.session_state['audio_device_cache'] = {'input': [], 'output': []}
            return

        try:
            core = self.recorder.get_sdl_core_info()
            raw_inputs = core.get("devices", {}).get("input_devices", []) or []
            raw_outputs = core.get("devices", {}).get("output_devices", []) or []
            input_list = [self._normalize_device_entry(d) for d in raw_inputs]
            output_list = [self._normalize_device_entry(d) for d in raw_outputs]
            st.success(f"Found {len(input_list)} input / {len(output_list)} output devices.")
        except Exception as e:
            st.error(f"Device scan failed: {e}")

        st.session_state['audio_device_cache'] = {'input': input_list, 'output': output_list}

    @staticmethod
    def _normalize_device_entry(dev: Any) -> Dict[str, Any]:
        """Convert raw entry into uniform dict: { 'name': str, 'device_id': int, 'repr': str }"""
        name, did = None, None
        if isinstance(dev, dict):
            name = dev.get("name") or dev.get("device_name") or str(dev)
            did = dev.get("device_id")
        else:
            name = getattr(dev, "name", None) or getattr(dev, "device_name", None) or str(dev)
            did = getattr(dev, "device_id", None)

        try:
            did = int(did)
        except Exception:
            text = str(dev)
            if "id=" in text:
                try:
                    idx = text.index("id=") + 3
                    digits = ""
                    for ch in text[idx:]:
                        if ch.isdigit(): digits += ch
                        else: break
                    did = int(digits) if digits else -1
                except Exception:
                    did = -1
            else:
                did = -1

        label = f"{name} (ID: {did})" if name is not None else f"Device (ID: {did})"
        return {"name": name or "Device", "device_id": int(did), "repr": label}

    @staticmethod
    def _extract_device_id(selection: str | Dict[str, Any]) -> int:
        """Extract device ID from selection value; default to -1 on failure."""
        if isinstance(selection, dict):
            try:
                return int(selection.get("device_id", -1))
            except Exception:
                return -1

        text = str(selection).strip()
        if text.lower().startswith("system default"):
            return -1
        if "(ID:" in text:
            try:
                return int(text.split("(ID:")[1].split(")")[0].strip())
            except Exception:
                return -1
        if "id=" in text:
            try:
                idx = text.index("id=") + 3
                digits = ""
                for ch in text[idx:]:
                    if ch.isdigit(): digits += ch
                    else: break
                return int(digits) if digits else -1
            except Exception:
                return -1
        return -1

    # ----------------
    # Input selection
    # ----------------

    def _render_input_selector(self) -> None:
        st.markdown("**Input Device**")

        cache = st.session_state['audio_device_cache']
        options: List[Any] = ["System Default"] + [dev["repr"] for dev in cache['input']]

        # Determine current index from recorder
        current_id = -1
        if self.recorder:
            try:
                current_id = int(getattr(self.recorder, "input_device", -1))
            except Exception:
                current_id = -1

        def _index_for_id(dev_id: int) -> int:
            if dev_id == -1:
                return 0
            for i, dev in enumerate(cache['input']):
                if int(dev["device_id"]) == dev_id:
                    return i + 1  # +1 for "System Default"
            return 0

        idx = _index_for_id(current_id)
        selection = st.selectbox("Select Microphone", options, index=idx, key="selector_input_device")

        selected_id = self._extract_device_id(selection)
        if self.recorder and selected_id != current_id:
            try:
                self.recorder.set_audio_devices(input=selected_id, output=None)
                st.success(f"Input device set to ID {selected_id}" if selected_id >= 0 else "Input device set to System Default")
            except Exception as e:
                st.error(f"Failed to set input device: {e}")

    # -------------------------
    # Input channel + live monitor
    # -------------------------

    def _render_input_channel_picker(self) -> None:
        current_ch = int(st.session_state.get('audio_input_channel', 0))
        new_ch = st.number_input("Channel (0-based)", min_value=0, max_value=7, value=current_ch, step=1)
        if new_ch != current_ch:
            st.session_state['audio_input_channel'] = int(new_ch)
            if self.recorder and hasattr(self.recorder, "input_channel"):
                try:
                    self.recorder.input_channel = int(new_ch)
                except Exception:
                    pass

    def _render_mic_monitor_controls_and_meter(self) -> None:
        """Simple 5Hz microphone monitor with programmatic updates."""
        st.markdown("**Live Mic Monitor (5Hz Updates)**")

        if not MICTESTING_AVAILABLE:
            st.warning("MicTesting module not available.")
            return

        # Simple controls
        col1, col2 = st.columns(2)
        with col1:
            if not st.session_state.get('mic_running', False):
                if st.button("Start Monitor", type="primary"):
                    self._start_simple_mic_monitor()
            else:
                if st.button("Stop Monitor"):
                    self._stop_simple_mic_monitor()

        with col2:
            if st.session_state.get('mic_running', False):
                st.success("Monitoring active - updates 5x/sec")
            else:
                st.info("Monitor stopped")

        # Level display
        self._render_simple_mic_display()

        # Auto-refresh at 5Hz when running
        if st.session_state.get('mic_running', False):
            time.sleep(0.2)  # 200ms = 5Hz
            st.rerun()

    def _start_simple_mic_monitor(self) -> None:
        """Start simple microphone monitoring."""
        if st.session_state.get('mic_running', False):
            return
        if not self.recorder:
            st.warning("Recorder not available")
            return

        try:
            sr = int(getattr(self.recorder, 'sample_rate', 48000))
            inp = int(getattr(self.recorder, 'input_device', -1))

            # Simple shared state
            shared_state = {
                'running': True,
                'latest_db': -60.0,
                'latest_rms': 0.0,
                'update_count': 0,
                'last_update': time.time(),
                'error': None
            }

            def worker():
                """Simple worker that continuously updates the latest reading."""
                try:
                    chunk_sec = 0.1  # Fixed 100ms chunks
                    min_samples = int(sr * chunk_sec)

                    with MicTesting.AudioRecorder(sample_rate=sr, input_device=inp, enable_logging=False) as ar:
                        while shared_state['running']:
                            try:
                                audio_chunk = ar.get_audio_chunk(min_samples)

                                if len(audio_chunk) > 0:
                                    rms = MicTesting.AudioProcessor.calculate_rms(audio_chunk)
                                    level_db = MicTesting.AudioProcessor.rms_to_db(rms)

                                    # Simple atomic update
                                    shared_state['latest_db'] = float(level_db)
                                    shared_state['latest_rms'] = float(rms)
                                    shared_state['update_count'] += 1
                                    shared_state['last_update'] = time.time()

                                time.sleep(chunk_sec)

                            except Exception as e:
                                shared_state['error'] = f"Recording error: {e}"
                                break

                except Exception as e:
                    shared_state['error'] = f"Worker error: {e}"
                    shared_state['running'] = False

            # Start worker
            thread = threading.Thread(target=worker, daemon=True)
            thread.start()

            # Store state
            st.session_state['mic_running'] = True
            st.session_state['mic_shared_state'] = shared_state
            st.session_state['mic_thread'] = thread

            st.success("Monitor started - 5Hz updates")

        except Exception as e:
            st.error(f"Failed to start monitor: {e}")

    def _stop_simple_mic_monitor(self) -> None:
        """Stop simple microphone monitoring."""
        shared_state = st.session_state.get('mic_shared_state')
        if shared_state:
            shared_state['running'] = False

        thread = st.session_state.get('mic_thread')
        if thread and thread.is_alive():
            thread.join(timeout=1.0)

        st.session_state['mic_running'] = False
        st.session_state['mic_shared_state'] = None
        st.session_state['mic_thread'] = None

    def _render_simple_mic_display(self) -> None:
        """Render simple microphone display."""
        shared_state = st.session_state.get('mic_shared_state')

        if not st.session_state.get('mic_running', False):
            st.info("Click 'Start Monitor' to begin")
            return

        if not shared_state:
            st.warning("No monitoring data")
            return

        if shared_state.get('error'):
            st.error(f"Error: {shared_state['error']}")
            return

        # Get data
        level_db = float(shared_state.get('latest_db', -60.0))
        rms = float(shared_state.get('latest_rms', 0.0))
        update_count = shared_state.get('update_count', 0)
        last_update = shared_state.get('last_update', 0)

        # Check freshness
        age = time.time() - last_update
        if age > 1.0:
            st.warning(f"Stale data ({age:.1f}s old)")
            return

        # Progress bar
        rng_db = 60.0
        percent = float(max(0.0, min(1.0, (level_db + rng_db) / rng_db)))
        st.progress(percent, text=f"Level: {level_db:+.1f} dBFS")

        # Metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("dB", f"{level_db:+.1f}")
        with col2:
            st.metric("RMS", f"{rms:.1e}")
        with col3:
            st.metric("Updates", update_count)

        # Assessment
        if level_db > -6:
            st.error("TOO LOUD - Clipping risk")
        elif level_db > -20:
            st.success("Good level")
        elif level_db > -40:
            st.info("Moderate level")
        elif level_db > -50:
            st.warning("Low level")
        else:
            st.error("Very low - check mic")

        # Fixed status calculation - avoid division by zero
        start_time = st.session_state.get('mic_start_time')
        if start_time is None:
            st.session_state['mic_start_time'] = time.time()
            rate_text = "Rate: calculating..."
        else:
            elapsed = time.time() - start_time
            if elapsed > 0:
                rate = update_count / elapsed
                rate_text = f"Rate: ~{rate:.1f} Hz"
            else:
                rate_text = "Rate: calculating..."

        st.caption(f"Age: {age:.1f}s | {rate_text}")
    # Even simpler alternative using st.empty() placeholder
    def _render_mic_monitor_with_placeholder(self) -> None:
        """Alternative approach using st.empty() for smoother updates."""
        st.markdown("**Live Mic Monitor (Placeholder Method)**")

        if not MICTESTING_AVAILABLE:
            st.warning("MicTesting module not available.")
            return

        # Controls
        col1, col2 = st.columns(2)
        with col1:
            if not st.session_state.get('mic_running_placeholder', False):
                if st.button("Start Monitor", type="primary", key="start_placeholder"):
                    self._start_placeholder_monitor()
            else:
                if st.button("Stop Monitor", key="stop_placeholder"):
                    self._stop_placeholder_monitor()

        with col2:
            st.info("Using st.empty() for updates")

        # Create placeholder for level display
        level_placeholder = st.empty()

        # Update display if running
        if st.session_state.get('mic_running_placeholder', False):
            self._update_placeholder_display(level_placeholder)

    def _start_placeholder_monitor(self) -> None:
        """Start placeholder-based monitoring."""
        if not self.recorder:
            st.warning("Recorder not available")
            return

        st.session_state['mic_running_placeholder'] = True
        st.session_state['mic_placeholder_data'] = {
            'level_db': -60.0,
            'update_count': 0,
            'start_time': time.time()
        }

    def _stop_placeholder_monitor(self) -> None:
        """Stop placeholder-based monitoring."""
        st.session_state['mic_running_placeholder'] = False
        st.session_state['mic_placeholder_data'] = None

    def _update_placeholder_display(self, placeholder) -> None:
        """Update the placeholder with current mic data."""
        if not self.recorder or not st.session_state.get('mic_running_placeholder', False):
            return

        try:
            # Quick measurement
            sr = int(getattr(self.recorder, 'sample_rate', 48000))
            inp = int(getattr(self.recorder, 'input_device', -1))

            # Use the simple measurement approach
            with MicTesting.AudioRecorder(sample_rate=sr, input_device=inp, enable_logging=False) as ar:
                time.sleep(0.05)  # Brief wait
                audio_chunk = ar.get_audio_chunk(int(sr * 0.1))

                if len(audio_chunk) > 0:
                    rms = MicTesting.AudioProcessor.calculate_rms(audio_chunk)
                    level_db = MicTesting.AudioProcessor.rms_to_db(rms)

                    # Update data
                    data = st.session_state['mic_placeholder_data']
                    data['level_db'] = float(level_db)
                    data['update_count'] += 1

                    # Update placeholder
                    with placeholder.container():
                        rng_db = 60.0
                        percent = float(max(0.0, min(1.0, (level_db + rng_db) / rng_db)))
                        st.progress(percent, text=f"Level: {level_db:+.1f} dBFS")
                        st.metric("Updates", data['update_count'])

                        # Level assessment
                        if level_db > -20:
                            st.success("Good level")
                        elif level_db > -40:
                            st.info("Moderate level")
                        else:
                            st.warning("Low level")

            # Schedule next update
            time.sleep(0.2)  # 5Hz
            st.rerun()

        except Exception as e:
            placeholder.error(f"Monitor error: {e}")
            st.session_state['mic_running_placeholder'] = False
    # -----------------
    # Output selection
    # -----------------

    def _render_output_selector(self) -> None:
        st.markdown("**Output Device**")

        cache = st.session_state['audio_device_cache']
        options: List[Any] = ["System Default"] + [dev["repr"] for dev in cache['output']]

        # Determine current index from recorder
        current_id = -1
        if self.recorder:
            try:
                current_id = int(getattr(self.recorder, "output_device", -1))
            except Exception:
                current_id = -1

        def _index_for_id(dev_id: int) -> int:
            if dev_id == -1:
                return 0
            for i, dev in enumerate(cache['output']):
                if int(dev["device_id"]) == dev_id:
                    return i + 1
            return 0

        idx = _index_for_id(current_id)
        selection = st.selectbox("Select Speakers", options, index=idx, key="selector_output_device")

        selected_id = self._extract_device_id(selection)
        if self.recorder and selected_id != current_id:
            try:
                self.recorder.set_audio_devices(input=None, output=selected_id)
                st.success(f"Output device set to ID {selected_id}" if selected_id >= 0 else "Output device set to System Default")
            except Exception as e:
                st.error(f"Failed to set output device: {e}")

    # -------------------------
    # Speaker test (output path)
    # -------------------------

    def _render_speaker_test_controls(self) -> None:
        sp1, sp2, sp3 = st.columns([1, 1, 1])
        with sp1:
            freq = st.slider("Frequency (Hz)", min_value=100, max_value=4000, value=1000, step=50, key="sp_freq")
        with sp2:
            dur = st.slider("Duration (s)", min_value=0.2, max_value=3.0, value=1.0, step=0.1, key="sp_dur")
        with sp3:
            default_vol = float(getattr(self.recorder, 'volume', 0.3)) if self.recorder else 0.3
            vol = st.slider("Volume", min_value=0.05, max_value=1.0, value=default_vol, step=0.05, key="sp_vol")

        if st.button("Play Test Tone"):
            self._run_speaker_test(freq=int(freq), duration=float(dur), volume=float(vol))

    def _run_speaker_test(self, freq: int, duration: float, volume: float) -> None:
        if not self.recorder:
            st.warning("Recorder not available")
            return

        # Preferred: dedicated API
        if hasattr(self.recorder, "play_test_tone"):
            try:
                with st.spinner("Playing test tone (recorder)..."):
                    self.recorder.play_test_tone(frequency=freq, duration=duration, volume=volume)  # type: ignore[attr-defined]
                st.success("Tone played.")
                return
            except Exception as e:
                st.error(f"Recorder speaker test failed: {e}")

        # Fallback: use recorder internals to play via its auto path
        try:
            with st.spinner("Playing test tone via recorder auto path..."):
                orig_freq = getattr(self.recorder, 'pulse_frequency', None)
                orig_vol = getattr(self.recorder, 'volume', None)
                orig_play = getattr(self.recorder, 'playback_signal', None)

                if hasattr(self.recorder, 'pulse_frequency'):
                    self.recorder.pulse_frequency = float(freq)  # type: ignore[attr-defined]
                if hasattr(self.recorder, 'volume'):
                    self.recorder.volume = float(volume)

                if hasattr(self.recorder, '_generate_complete_signal'):
                    self.recorder.playback_signal = self.recorder._generate_complete_signal()  # type: ignore[attr-defined]

                if hasattr(self.recorder, "_record_method_2"):
                    _ = self.recorder._record_method_2()  # play + optional capture
                elif hasattr(self.recorder, "take_record"):
                    _ = None
                else:
                    st.warning("No suitable recorder method found to play audio.")
                    return

                # restore
                if orig_freq is not None and hasattr(self.recorder, 'pulse_frequency'):
                    self.recorder.pulse_frequency = orig_freq  # type: ignore[attr-defined]
                if orig_vol is not None and hasattr(self.recorder, 'volume'):
                    self.recorder.volume = orig_vol
                if orig_play is not None:
                    self.recorder.playback_signal = orig_play  # type: ignore[attr-defined]

                st.success("Tone played.")
        except Exception as e:
            st.error(f"Speaker test failed: {e}")
