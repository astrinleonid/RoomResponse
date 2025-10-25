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
    # Multi-channel helpers
    # --------------------

    def _get_selected_device_max_channels(self) -> int:
        """Get max channels for currently selected input device."""
        if not self.recorder:
            return 1

        try:
            devices_info = self.recorder.get_device_info_with_channels()
            current_id = int(getattr(self.recorder, 'input_device', -1))

            # Default device: check all devices and return max
            if current_id == -1:
                return max((d['max_channels'] for d in devices_info['input_devices']), default=1)

            # Specific device: return its max_channels
            for dev in devices_info['input_devices']:
                if dev['device_id'] == current_id:
                    return dev['max_channels']

            return 1  # Fallback
        except Exception:
            return 1

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

        # Get device info with channels
        devices_info = self.recorder.get_device_info_with_channels() if self.recorder else {}
        input_devices = devices_info.get('input_devices', [])

        # Build options with channel info
        cache = st.session_state['audio_device_cache']
        options: List[Any] = ["System Default"]

        # Add devices with channel info if available
        if input_devices:
            for dev in input_devices:
                ch_text = f" ({dev['max_channels']} ch)" if dev['max_channels'] > 1 else ""
                options.append(f"{dev['name']} (ID: {dev['device_id']}){ch_text}")
        else:
            # Fallback to cache without channel info
            options.extend([dev["repr"] for dev in cache['input']])

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
            # Try to find in devices_info first
            if input_devices:
                for i, dev in enumerate(input_devices):
                    if int(dev["device_id"]) == dev_id:
                        return i + 1
            # Fallback to cache
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

        # Display max channels info
        max_ch = self._get_selected_device_max_channels()
        if max_ch > 1:
            st.info(f"✓ Multi-channel device: {max_ch} channels available")
        else:
            st.caption("Mono device (1 channel)")

    # -------------------------
    # Input channel + live monitor
    # -------------------------

    def _render_input_channel_picker(self) -> None:
        # Get max channels for dynamic range
        max_channels = self._get_selected_device_max_channels()

        current_ch = int(st.session_state.get('audio_input_channel', 0))
        new_ch = st.number_input(
            "Channel (0-based)",
            min_value=0,
            max_value=max(0, max_channels - 1),
            value=min(current_ch, max_channels - 1),
            step=1,
            help=f"Device supports up to {max_channels} channels"
        )
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

    # -------------------------
    # Multi-channel monitor
    # -------------------------

    def _render_multichannel_monitor(self) -> None:
        """Multi-channel microphone monitor with per-channel meters."""
        st.markdown("#### Multi-Channel Microphone Monitor")

        # Get available channels
        max_channels = self._get_selected_device_max_channels()

        # Channel count selector
        col1, col2 = st.columns([1, 2])
        with col1:
            num_channels = st.number_input(
                "Monitor Channels",
                min_value=1,
                max_value=min(max_channels, 8),  # Cap display at 8
                value=min(2, max_channels),
                step=1,
                key="mc_mon_channels"
            )

        with col2:
            st.caption(f"Device supports up to {max_channels} channels")

        # Start/Stop controls
        if not st.session_state.get('multichannel_monitor_running', False):
            if st.button("Start Multi-Channel Monitor", type="primary", key="mc_start"):
                self._start_multichannel_monitor(int(num_channels))
        else:
            if st.button("Stop Monitor", key="mc_stop"):
                self._stop_multichannel_monitor()

        # Display per-channel meters
        if st.session_state.get('multichannel_monitor_running', False):
            self._render_multichannel_meters()

            # Auto-refresh at 5Hz
            time.sleep(0.2)
            st.rerun()

    def _start_multichannel_monitor(self, num_channels: int) -> None:
        """Start multi-channel monitoring thread."""
        if not self.recorder or not MICTESTING_AVAILABLE:
            st.warning("Recorder or MicTesting not available")
            return

        try:
            sr = int(getattr(self.recorder, 'sample_rate', 48000))
            inp = int(getattr(self.recorder, 'input_device', -1))

            # Shared state for all channels
            shared_state = {
                'running': True,
                'num_channels': num_channels,
                'latest_levels': [-60.0] * num_channels,  # dB per channel
                'latest_rms': [0.0] * num_channels,
                'update_count': 0,
                'last_update': time.time(),
                'error': None
            }

            def worker():
                """Multi-channel monitoring worker."""
                try:
                    import sdl_audio_core

                    # Create engine with multi-channel config
                    engine = sdl_audio_core.AudioEngine()
                    config = sdl_audio_core.AudioEngineConfig()
                    config.sample_rate = sr
                    config.input_channels = num_channels

                    if not engine.initialize(config):
                        shared_state['error'] = "Failed to initialize audio engine"
                        shared_state['running'] = False
                        return

                    engine.set_input_device(inp)
                    engine.start_recording()

                    while shared_state['running']:
                        try:
                            time.sleep(0.1)  # 100ms chunks

                            # Get per-channel data
                            for ch in range(num_channels):
                                ch_data = engine.get_recorded_data_channel(ch)

                                if len(ch_data) > 100:  # Need enough samples
                                    # Take last 100ms worth
                                    recent = ch_data[-int(sr * 0.1):]
                                    ch_np = np.array(recent)

                                    rms = np.sqrt(np.mean(ch_np ** 2))
                                    db = 20 * np.log10(rms) if rms > 1e-10 else -60.0

                                    shared_state['latest_levels'][ch] = float(db)
                                    shared_state['latest_rms'][ch] = float(rms)

                            shared_state['update_count'] += 1
                            shared_state['last_update'] = time.time()

                        except Exception as e:
                            shared_state['error'] = f"Recording error: {e}"
                            break

                    engine.stop_recording()
                    engine.shutdown()

                except Exception as e:
                    shared_state['error'] = f"Worker error: {e}"
                    shared_state['running'] = False

            # Start worker thread
            thread = threading.Thread(target=worker, daemon=True)
            thread.start()

            st.session_state['multichannel_monitor_running'] = True
            st.session_state['multichannel_shared_state'] = shared_state
            st.session_state['multichannel_thread'] = thread

            st.success(f"Multi-channel monitor started ({num_channels} channels)")

        except Exception as e:
            st.error(f"Failed to start multi-channel monitor: {e}")

    def _stop_multichannel_monitor(self) -> None:
        """Stop multi-channel monitoring."""
        shared_state = st.session_state.get('multichannel_shared_state')
        if shared_state:
            shared_state['running'] = False

        thread = st.session_state.get('multichannel_thread')
        if thread and thread.is_alive():
            thread.join(timeout=2.0)

        st.session_state['multichannel_monitor_running'] = False
        st.session_state['multichannel_shared_state'] = None
        st.session_state['multichannel_thread'] = None

    def _render_multichannel_meters(self) -> None:
        """Render per-channel level meters."""
        shared_state = st.session_state.get('multichannel_shared_state')

        if not shared_state:
            st.warning("No monitoring data")
            return

        if shared_state.get('error'):
            st.error(f"Error: {shared_state['error']}")
            return

        num_channels = shared_state['num_channels']
        levels_db = shared_state['latest_levels']
        rms_values = shared_state['latest_rms']
        update_count = shared_state.get('update_count', 0)
        last_update = shared_state.get('last_update', 0)

        # Check data freshness
        age = time.time() - last_update
        if age > 1.0:
            st.warning(f"Stale data ({age:.1f}s old)")
            return

        # Display channels in grid (2 columns)
        cols_per_row = 2
        for row_start in range(0, num_channels, cols_per_row):
            cols = st.columns(cols_per_row)

            for i in range(cols_per_row):
                ch = row_start + i
                if ch >= num_channels:
                    break

                with cols[i]:
                    self._render_single_channel_meter(ch, levels_db[ch], rms_values[ch])

        # Status bar
        st.caption(f"Updates: {update_count} | Age: {age:.1f}s | Rate: ~5 Hz")

    def _render_single_channel_meter(self, ch_idx: int, db: float, rms: float) -> None:
        """Render a single channel's meter."""
        st.markdown(f"**Channel {ch_idx}**")

        # Progress bar (map -60 to 0 dB → 0 to 100%)
        rng_db = 60.0
        percent = max(0.0, min(1.0, (db + rng_db) / rng_db))

        # Color coding
        if db > -6:
            bar_text = f"⚠️ {db:+.1f} dB"
        elif db > -20:
            bar_text = f"✓ {db:+.1f} dB"
        else:
            bar_text = f"{db:+.1f} dB"

        st.progress(percent, text=bar_text)

        # Mini metrics
        col1, col2 = st.columns(2)
        with col1:
            st.caption(f"RMS: {rms:.1e}")
        with col2:
            # Level assessment
            if db > -6:
                st.caption("🔴 LOUD")
            elif db > -20:
                st.caption("🟢 Good")
            elif db > -40:
                st.caption("🟡 Moderate")
            else:
                st.caption("🔵 Low")

    # -------------------------
    # Multi-channel test recording
    # -------------------------

    def _render_multichannel_test(self) -> None:
        """Multi-channel test recording UI."""
        st.markdown("#### Multi-Channel Test Recording")

        max_channels = self._get_selected_device_max_channels()

        col1, col2, col3 = st.columns(3)

        with col1:
            test_channels = st.number_input(
                "Test Channels",
                min_value=1,
                max_value=min(max_channels, 32),
                value=min(2, max_channels),
                step=1,
                key="test_channels"
            )

        with col2:
            test_duration = st.slider(
                "Duration (s)",
                min_value=0.5,
                max_value=5.0,
                value=2.0,
                step=0.5,
                key="test_duration"
            )

        with col3:
            if st.button("Run Test Recording", type="primary"):
                self._run_multichannel_test(int(test_channels), float(test_duration))

        # Display last test results
        if 'last_multichannel_test' in st.session_state:
            self._display_test_results(st.session_state['last_multichannel_test'])

    def _run_multichannel_test(self, num_channels: int, duration: float) -> None:
        """Execute multi-channel test recording."""
        if not self.recorder:
            st.warning("Recorder not available")
            return

        with st.spinner(f"Testing {num_channels} channel(s)..."):
            try:
                result = self.recorder.test_multichannel_recording(
                    duration=duration,
                    num_channels=num_channels
                )

                st.session_state['last_multichannel_test'] = result

                if result['success']:
                    st.success(f"✓ Test successful: {num_channels} channels recorded")
                else:
                    st.error(f"✗ Test failed: {result.get('error_message', 'Unknown error')}")

            except Exception as e:
                st.error(f"Test error: {e}")

    def _display_test_results(self, result: dict) -> None:
        """Display multi-channel test results."""
        if not result.get('success', False):
            st.error(f"Last test failed: {result.get('error_message')}")
            return

        st.markdown("**Last Test Results**")

        num_ch = result['num_channels']
        samples = result['samples_per_channel']

        st.info(f"Recorded {num_ch} channels × {samples} samples")

        # Per-channel stats table
        stats = result.get('channel_stats', [])
        if stats:
            try:
                import pandas as pd

                df = pd.DataFrame([
                    {
                        'Channel': i,
                        'Max Amplitude': f"{s['max']:.4f}",
                        'RMS': f"{s['rms']:.4f}",
                        'Level (dB)': f"{s['db']:+.1f}"
                    }
                    for i, s in enumerate(stats)
                ])

                st.dataframe(df, use_container_width=True)
            except ImportError:
                # Fallback if pandas not available
                for i, s in enumerate(stats):
                    st.text(f"Channel {i}: Max={s['max']:.4f}, RMS={s['rms']:.4f}, dB={s['db']:+.1f}")

        # Assessment
        low_channels = [i for i, s in enumerate(stats) if s['max'] < 0.01]
        if low_channels:
            st.warning(f"⚠️ Low signal on channels: {low_channels}")
        else:
            st.success("✓ All channels have good signal levels")

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
