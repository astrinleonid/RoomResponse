#!/usr/bin/env python3
"""
Series Settings Panel - Multi-pulse Recording Configuration and Analysis

- Uses a shared/global RoomResponseRecorder passed in by the parent.
- Series configuration changes are APPLIED PERMANENTLY to the shared recorder.
- Includes a "Cycle Consistency Overlay" plot (multiple cycles on the same axes).
"""

from __future__ import annotations

import shutil
import time
import json
from pathlib import Path
from typing import Optional, Dict, Any, List
import math  # <-- added

import numpy as np
import streamlit as st

# Optional visualizer
try:
    from gui_audio_visualizer import AudioVisualizer
    VISUALIZER_AVAILABLE = True
except ImportError:
    VISUALIZER_AVAILABLE = False
    AudioVisualizer = None  # type: ignore

# Recorder type (provided by parent; we never instantiate here)
try:
    from RoomResponseRecorder import RoomResponseRecorder  # type: ignore
    RECORDER_AVAILABLE = True
except Exception:
    RoomResponseRecorder = None  # type: ignore
    RECORDER_AVAILABLE = False

# SDL core (optional; only used for quick preview if available)
try:
    import sdl_audio_core as sdl
    SDL_AVAILABLE = True
except Exception:
    sdl = None  # type: ignore
    SDL_AVAILABLE = False


class SeriesSettingsPanel:
    """Panel for configuring and testing multi-pulse recording series (shared recorder)."""

    def __init__(self, recorder: Optional["RoomResponseRecorder"] = None, audio_settings_panel=None):
        """
        Args:
            recorder: Shared/global RoomResponseRecorder instance (required)
            audio_settings_panel: Optional parent panel reference
        """
        self.recorder = recorder
        self.audio_settings_panel = audio_settings_panel
        self.component_id = "series_settings"

    # ----------------------
    # Public render entrypoint
    # ----------------------
    def render(self) -> None:
        st.header("Series Settings — Multi-pulse Configuration")

        # Prereq/status row
        self._show_prerequisites()
        if not (RECORDER_AVAILABLE and self.recorder):
            st.error("No shared RoomResponseRecorder available; connect it in Audio Settings.")
            return

        self._init_session_state()

        tab1, tab2, tab3 = st.tabs([
            "Pulse Series Config",
            "Recording & Analysis",
            "Advanced Settings"
        ])
        with tab1:
            self._render_pulse_series_config()
        with tab2:
            self._render_recording_analysis()
        with tab3:
            self._render_advanced_settings()

    # ----------------------
    # Session defaults
    # ----------------------
    def _init_session_state(self) -> None:
        # Ensure correct base types (ints for counts, floats for durations/freqs)
        defaults = {
            'series_num_pulses': int(getattr(self.recorder, 'num_pulses', 8)) if self.recorder else 8,
            'series_pulse_duration': float(1000.0 * getattr(self.recorder, 'pulse_duration', 0.008)) if self.recorder else 8.0,     # ms
            'series_cycle_duration': float(1000.0 * getattr(self.recorder, 'cycle_duration', 0.1)) if self.recorder else 100.0,     # ms
            'series_pulse_frequency': float(getattr(self.recorder, 'pulse_frequency', 1000.0)) if self.recorder else 1000.0,        # Hz
            'series_pulse_volume': float(getattr(self.recorder, 'volume', 0.4)) if self.recorder else 0.4,
            'series_pulse_form': getattr(self.recorder, 'impulse_form', 'sine') if self.recorder else 'sine',
            'series_fade_duration': float(1000.0 * getattr(self.recorder, 'pulse_fade', 0.0001)) if self.recorder else 0.1,         # ms

            # Analysis parameters
            'series_record_extra_time': 200.0,
            'series_averaging_start_cycle': 2,
            'series_show_individual_cycles': True,
            'series_show_averaged_result': True,

            # Recording data cache
            'series_recorded_audio': None,
            'series_sample_rate': int(getattr(self.recorder, 'sample_rate', 48000)) if self.recorder else 48000,
            'series_timestamp': 0.0,
            'series_analysis_data': {},

            # Visualization options
            'series_cycle_overlay_mode': 'all',
            'series_analysis_window_start': 0.0,
            'series_analysis_window_end': 1.0,
        }
        for k, v in defaults.items():
            st.session_state.setdefault(k, v)

    # ----------------------
    # Status / prerequisites
    # ----------------------
    def _show_prerequisites(self) -> None:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.success("✅ SDL Audio" if SDL_AVAILABLE else "❌ SDL Audio")
        with col2:
            ok = RECORDER_AVAILABLE and (self.recorder is not None)
            st.success("✅ Recorder" if ok else "❌ Recorder")
        with col3:
            st.success("✅ Visualizer" if VISUALIZER_AVAILABLE else "❌ Visualizer")
        with col4:
            in_dev = st.session_state.get('audio_selected_input_device', 'None')
            out_dev = st.session_state.get('audio_selected_output_device', 'None')
            if in_dev != 'None' and out_dev != 'None':
                st.success("✅ Devices")
            else:
                st.warning("⚠️ Devices")
            st.caption(f"In: {str(in_dev)[:24]}")
            st.caption(f"Out: {str(out_dev)[:24]}")

    # ----------------------
    # Config UI (permanent apply to recorder)
    # ----------------------
    def _render_pulse_series_config(self) -> None:
        st.markdown("**Multi-pulse Series Configuration**")
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**Pulse Properties**")
            # ints across the board
            num_pulses = st.number_input(
                "Number of pulses",
                min_value=int(1), max_value=int(200),
                value=int(st.session_state['series_num_pulses']),
                step=int(1)
            )
            # floats across the board
            pulse_duration_ms = st.number_input(
                "Pulse duration (ms)",
                min_value=float(0.02), max_value=float(200.0),
                value=float(st.session_state['series_pulse_duration']),
                step=float(0.02)
            )
            pulse_freq = st.number_input(
                "Pulse frequency (Hz)",
                min_value=float(20.0), max_value=float(24000.0),
                value=float(st.session_state['series_pulse_frequency']),
                step=float(50.0)
            )

        with col2:
            st.markdown("**Timing & Volume**")
            cycle_duration_ms = st.number_input(
                "Cycle duration (ms)",
                min_value=float(5.0), max_value=float(3000.0),
                value=float(st.session_state['series_cycle_duration']),
                step=float(5.0)
            )
            pulse_vol = st.slider(
                "Pulse volume",
                min_value=float(0.0), max_value=float(1.0),
                value=float(st.session_state['series_pulse_volume']),
                step=float(0.05)
            )
            fade_ms = st.number_input(
                "Fade duration (ms)",
                min_value=float(0.05), max_value=float(20.0),
                value=float(st.session_state['series_fade_duration']),
                step=float(0.05)
            )

        with col3:
            st.markdown("**Waveform & Analysis**")
            pulse_form = st.selectbox(
                "Pulse waveform",
                ["sine", "square"],
                index=0 if st.session_state['series_pulse_form'] == 'sine' else 1
            )
            extra_ms = st.number_input(
                "Extra record time (ms)",
                min_value=float(0.0), max_value=float(5000.0),
                value=float(st.session_state['series_record_extra_time']),
                step=float(25.0)
            )
            # ints across the board; max depends on num_pulses
            avg_start = st.number_input(
                "Averaging start cycle",
                min_value=int(1), max_value=int(max(1, int(num_pulses))),
                value=int(min(int(st.session_state['series_averaging_start_cycle']), int(num_pulses))),
                step=int(1)
            )

        # Store in session (correct types)
        st.session_state['series_num_pulses'] = int(num_pulses)
        st.session_state['series_pulse_duration'] = float(pulse_duration_ms)
        st.session_state['series_cycle_duration'] = float(cycle_duration_ms)
        st.session_state['series_pulse_frequency'] = float(pulse_freq)
        st.session_state['series_pulse_volume'] = float(pulse_vol)
        st.session_state['series_fade_duration'] = float(fade_ms)
        st.session_state['series_pulse_form'] = str(pulse_form)
        st.session_state['series_record_extra_time'] = float(extra_ms)
        st.session_state['series_averaging_start_cycle'] = int(avg_start)

        # APPLY PERMANENTLY to the shared recorder (no restore)
        self._apply_series_settings_to_recorder(self.recorder)

        self._show_calculated_parameters()
        st.markdown("---")
        self._render_series_controls()

    def _show_calculated_parameters(self) -> None:
        st.markdown("**Calculated Parameters**")
        pulse = float(st.session_state['series_pulse_duration'])
        cycle = float(st.session_state['series_cycle_duration'])
        num = int(st.session_state['series_num_pulses'])
        extra = float(st.session_state['series_record_extra_time'])

        gap = cycle - pulse
        total_ms = (num * cycle) + extra

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Gap Duration", f"{gap:.1f} ms")
        with c2:
            st.metric("Series Duration", f"{total_ms:.0f} ms")
        with c3:
            duty = (pulse / cycle) * 100.0 if cycle > 0 else 0.0
            st.metric("Duty Cycle", f"{duty:.1f}%")
        with c4:
            st.metric("Pulse Rate", f"{(1000.0 / cycle) if cycle > 0 else 0.0:.1f} Hz")

        if gap <= 0:
            st.error("⚠️ Gap duration is negative — increase cycle or reduce pulse.")
        elif gap < 5.0:
            st.warning("⚠️ Very short gap may cause overlapping echoes.")
        if cycle > 0 and (pulse / cycle) > 0.5:
            st.warning("⚠️ High duty cycle may cause excessive acoustic energy.")

    def _render_series_controls(self) -> None:
        st.markdown("**Series Controls**")
        c1, c2 = st.columns(2)
        with c1:
            if st.button("🎵 Record Series",
                         disabled=not (SDL_AVAILABLE and RECORDER_AVAILABLE and self.recorder)):
                self._execute_series_recording()
        # with c2:
        #     if st.button("🔊 Preview Series", disabled=not (SDL_AVAILABLE and self.recorder)):
        #         self._preview_series()
        with c2:
            if st.button("⚙️ Export Series Config"):
                self._export_series_config()

    # ----------------------
    # Record / Preview (use shared recorder settings as-is)
    # ----------------------
    def _execute_series_recording(self) -> None:
        if not self.recorder:
            st.error("Recorder unavailable")
            return

        try:
            with st.spinner("Recording pulse series..."):
                tmp = Path("TMP"); tmp.mkdir(exist_ok=True)
                for item in tmp.iterdir():
                    if item.is_file():
                        item.unlink()
                    elif item.is_dir():
                        shutil.rmtree(item)
                ts = int(time.time())
                raw_path = tmp / f"series_raw_{ts}.wav"
                imp_path = tmp / f"series_impulse_{ts}.wav"

                recorded_audio = self.recorder.take_record(
                    output_file=str(raw_path),
                    impulse_file=str(imp_path),
                    method=2
                )

                if recorded_audio is None:
                    st.error("Recording failed — no audio captured")
                    return

                analysis = self._analyze_series_recording(recorded_audio, self.recorder)
                st.session_state['series_recorded_audio'] = recorded_audio
                st.session_state['series_sample_rate'] = int(self.recorder.sample_rate)
                st.session_state['series_timestamp'] = time.time()
                st.session_state['series_analysis_data'] = analysis

                st.success(f"Series recording OK — {len(recorded_audio)/self.recorder.sample_rate:.3f}s")
                st.info(f"Files saved: {raw_path.name}, {imp_path.name}")
                st.rerun()

        except Exception as e:
            st.error(f"Recording error: {e}")
            with st.expander("Details"):
                st.code(str(e))

    # ----------------------
    # APPLY settings permanently to the shared recorder
    # ----------------------
    def _apply_series_settings_to_recorder(self, r: Optional["RoomResponseRecorder"]) -> None:
        if not r:
            return

        r.sample_rate = int(st.session_state.get('audio_sample_rate', getattr(r, 'sample_rate', 48000)))
        r.pulse_frequency = float(st.session_state['series_pulse_frequency'])
        r.pulse_duration = float(st.session_state['series_pulse_duration']) / 1000.0
        r.pulse_fade = float(st.session_state['series_fade_duration']) / 1000.0
        r.cycle_duration = float(st.session_state['series_cycle_duration']) / 1000.0
        r.num_pulses = int(st.session_state['series_num_pulses'])
        r.volume = float(st.session_state['series_pulse_volume'])
        r.impulse_form = str(st.session_state['series_pulse_form'])

        extra = float(st.session_state['series_record_extra_time']) / 1000.0
        r.total_duration = (r.num_pulses * r.cycle_duration) + extra

        # Recompute derived fields on recorder
        r.pulse_samples = int(r.pulse_duration * r.sample_rate)
        r.fade_samples = int(r.pulse_fade * r.sample_rate)
        r.cycle_samples = int(r.cycle_duration * r.sample_rate)
        r.gap_samples = r.cycle_samples - r.pulse_samples

        r.playback_signal = r._generate_complete_signal()

    # ----------------------
    # Analysis / Visualization
    # ----------------------
    def _analyze_series_recording(self, audio_data: np.ndarray, recorder: "RoomResponseRecorder") -> Dict[str, Any]:
        analysis: Dict[str, Any] = {}
        try:
            num = max(1, int(getattr(recorder, 'num_pulses', 1)))
            cyc = int(getattr(recorder, 'cycle_samples', 0))
            sr = int(getattr(recorder, 'sample_rate', 48000))

            expected = num * cyc if cyc > 0 else len(audio_data)
            signal_data = audio_data[:expected] if expected and len(audio_data) >= expected else audio_data

            # Split cycles
            cycles: List[np.ndarray] = []
            if cyc > 0 and num > 0:
                for i in range(num):
                    s = i * cyc
                    e = s + cyc
                    if e <= len(signal_data):
                        cycles.append(signal_data[s:e])
                    else:
                        seg = signal_data[s:] if s < len(signal_data) else np.zeros(0, dtype=np.float32)
                        pad = np.zeros(cyc, dtype=np.float32)
                        if len(seg) > 0:
                            pad[:len(seg)] = seg
                        cycles.append(pad)

            analysis['individual_cycles'] = cycles
            analysis['num_cycles_extracted'] = len(cycles)

            # Averaging
            if cycles:
                start = max(0, min(int(st.session_state['series_averaging_start_cycle']) - 1, len(cycles) - 1))
                to_avg = cycles[start:]
                avg_cycle = np.mean(np.stack(to_avg, axis=0), axis=0) if to_avg else cycles[-1]
                analysis['averaged_cycle'] = avg_cycle
                analysis['cycles_used_for_averaging'] = len(to_avg)
                analysis['averaging_start_cycle'] = start + 1

                # --- NEW: spectral analysis of the averaged impulse response segment ---
                try:
                    win_start_frac = float(st.session_state.get('series_analysis_window_start', 0.0))
                    win_end_frac = float(st.session_state.get('series_analysis_window_end', 1.0))
                    win_start_frac = max(0.0, min(1.0, win_start_frac))
                    win_end_frac = max(0.0, min(1.0, win_end_frac))
                    if win_end_frac <= win_start_frac:
                        win_end_frac = min(1.0, win_start_frac + 0.05)  # small guard

                    N = len(avg_cycle)
                    s = int(math.floor(N * win_start_frac))
                    e = int(math.ceil(N * win_end_frac))
                    seg = np.asarray(avg_cycle[s:e], dtype=np.float32)

                    # Hann window to reduce leakage
                    if seg.size > 1:
                        w = np.hanning(seg.size).astype(np.float32)
                        seg_w = seg * w
                    else:
                        seg_w = seg

                    # rFFT → magnitude dB
                    eps = 1e-12
                    spec = np.fft.rfft(seg_w)
                    mag = np.abs(spec)
                    mag_db = 20.0 * np.log10(mag + eps)
                    freqs = np.fft.rfftfreq(seg_w.size, d=1.0/float(sr))

                    analysis['averaged_spectrum'] = {
                        'freqs': freqs.astype(np.float32),
                        'magnitude_db': mag_db.astype(np.float32),
                        'window': [float(win_start_frac), float(win_end_frac)],
                        'n_fft': int(seg_w.size)
                    }
                except Exception as _fft_err:
                    analysis['spectrum_error'] = str(_fft_err)

            # Consistency metric
            if len(cycles) > 1:
                diffs = []
                for i in range(1, len(cycles)):
                    d = cycles[i] - cycles[i - 1]
                    diffs.append(float(np.sqrt(np.mean(d * d))))
                analysis['cycle_consistency'] = {
                    'rms_differences': diffs,
                    'mean_rms_diff': float(np.mean(diffs)),
                    'std_rms_diff': float(np.std(diffs)),
                }

            analysis['full_recording_metrics'] = {
                'max_amplitude': float(np.max(np.abs(audio_data))) if len(audio_data) else 0.0,
                'rms_level': float(np.sqrt(np.mean(audio_data ** 2))) if len(audio_data) else 0.0,
                'total_samples': int(len(audio_data)),
                'duration_seconds': float(len(audio_data) / sr) if sr > 0 else 0.0
            }

        except Exception as e:
            analysis['error'] = str(e)
            st.error(f"Analysis error: {e}")

        return analysis

    def _render_recording_analysis(self) -> None:
        st.markdown("**Series Recording Analysis**")

        audio = st.session_state.get('series_recorded_audio')
        sr = int(st.session_state.get('series_sample_rate', getattr(self.recorder, 'sample_rate', 48000)))
        analysis = st.session_state.get('series_analysis_data', {})

        if audio is None:
            st.info("No series recording yet. Use **Record Series**.")
            if VISUALIZER_AVAILABLE and st.session_state.get('series_preview_audio') is not None:
                st.markdown("**Series Preview**")
                AudioVisualizer("series_preview").render(
                    audio_data=st.session_state['series_preview_audio'],
                    sample_rate=int(st.session_state.get('series_preview_sample_rate', sr)),
                    title="Generated Series Signal",
                    show_controls=True,
                    show_analysis=False,
                    height=300
                )
            return

        ts = st.session_state.get('series_timestamp', 0)
        if ts:
            st.caption(f"Recorded at: {time.strftime('%H:%M:%S', time.localtime(ts))}")

        if analysis:
            self._display_analysis_metrics(analysis)

        self._render_visualization_controls()

        if VISUALIZER_AVAILABLE:
            st.markdown("**Full Recording**")
            AudioVisualizer("series_full_recording").render(
                audio_data=audio,
                sample_rate=sr,
                title="Complete Series Recording",
                show_controls=True,
                show_analysis=True,
                height=400
            )

        if analysis.get('individual_cycles'):
            self._render_cycle_analysis(analysis, sr)
            self._render_cycle_consistency_overlay(analysis, sr)

        if analysis.get('averaged_cycle') is not None:
            self._render_averaged_analysis(analysis, sr)

    def _display_analysis_metrics(self, analysis: Dict[str, Any]) -> None:
        st.markdown("**Analysis Results**")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Cycles Extracted", analysis.get('num_cycles_extracted', 0))
        with c2:
            st.metric("Cycles Averaged", analysis.get('cycles_used_for_averaging', 0))
        full = analysis.get('full_recording_metrics', {})
        with c3:
            st.metric("Max Amplitude", f"{full.get('max_amplitude', 0):.4f}")
        with c4:
            st.metric("RMS Level", f"{full.get('rms_level', 0):.4f}")

        cons = analysis.get('cycle_consistency')
        if cons:
            st.markdown("**Cycle-to-Cycle Consistency**")
            d1, d2 = st.columns(2)
            with d1:
                st.metric("Mean RMS Diff", f"{cons.get('mean_rms_diff', 0):.5f}")
            with d2:
                st.metric("Std RMS Diff", f"{cons.get('std_rms_diff', 0):.5f}")

    def _render_visualization_controls(self) -> None:
        with st.expander("Visualization Options"):
            c1, c2 = st.columns(2)
            with c1:
                st.session_state['series_show_individual_cycles'] = st.checkbox(
                    "Show individual cycles", value=st.session_state['series_show_individual_cycles'])
                st.session_state['series_cycle_overlay_mode'] = st.selectbox(
                    "Cycle overlay mode", ["all", "first_few", "averaged_only"])
            with c2:
                st.session_state['series_show_averaged_result'] = st.checkbox(
                    "Show averaged result", value=st.session_state['series_show_averaged_result'])
                st.session_state['series_analysis_window_start'] = st.slider(
                    "Analysis window start",
                    min_value=float(0.0), max_value=float(1.0),
                    value=float(st.session_state['series_analysis_window_start']),
                    step=float(0.01)
                )
                # NOTE: end slider is optional; if not shown, end defaults to 1.0.

    def _render_cycle_analysis(self, analysis: Dict[str, Any], sample_rate: int) -> None:
        if not st.session_state.get('series_show_individual_cycles', True):
            return
        cycles = analysis.get('individual_cycles', [])
        if not cycles:
            return

        st.markdown("**Individual Cycles (Inspect One)**")
        idx = st.selectbox("Select cycle", range(len(cycles)), format_func=lambda i: f"Cycle {i+1}")
        if 0 <= idx < len(cycles) and VISUALIZER_AVAILABLE:
            AudioVisualizer(f"series_cycle_{idx}").render(
                audio_data=cycles[idx],
                sample_rate=sample_rate,
                title=f"Cycle {idx+1} — Individual Analysis",
                show_controls=True,
                show_analysis=True,
                height=350
            )

    def _render_cycle_consistency_overlay(self, analysis: Dict[str, Any], sample_rate: int) -> None:
        """Plot many individual cycles on the same axes for consistency inspection."""
        cycles: List[np.ndarray] = analysis.get('individual_cycles', [])
        if not cycles:
            return

        st.markdown("**Cycle Consistency Overlay**")

        col_a, col_b = st.columns([2, 1])
        with col_a:
            max_to_plot = st.slider(
                "Number of cycles to overlay",
                min_value=int(1), max_value=int(len(cycles)),
                value=int(min(len(cycles), 10)),
                step=int(1)
            )
        with col_b:
            norm = st.checkbox("Normalize each cycle (max=1)", value=False)

        # Use AudioVisualizer static method for overlay plot
        plot_cycles = cycles[:max_to_plot]
        labels = [f"C{i+1}" for i in range(len(plot_cycles))]

        fig = AudioVisualizer.render_overlay_plot(
            audio_signals=plot_cycles,
            sample_rate=sample_rate,
            title="Overlay of Individual Cycles",
            labels=labels,
            normalize=norm,
            show_legend=(max_to_plot <= 12),
            figsize=(6.5, 3.0),
            alpha=0.55,
            linewidth=1.0
        )

        st.pyplot(fig, use_container_width=True)

    def _render_averaged_analysis(self, analysis: Dict[str, Any], sample_rate: int) -> None:
        if not st.session_state.get('series_show_averaged_result', True):
            return
        avg = analysis.get('averaged_cycle')
        if avg is None or len(avg) == 0:
            return

        if VISUALIZER_AVAILABLE:
            st.markdown("**Averaged Cycle Analysis**")
            used = analysis.get('cycles_used_for_averaging', 0)
            start = analysis.get('averaging_start_cycle', 1)
            st.info(f"Averaged from {used} cycles (starting at cycle {start})")
            AudioVisualizer("series_averaged_cycle").render(
                audio_data=avg,
                sample_rate=sample_rate,
                title="Averaged Cycle — Final Result",
                show_controls=True,
                show_analysis=True,
                height=400
            )

        # --- Spectrum of the averaged impulse response (windowed segment) ---
        if avg is not None and len(avg) > 0:
            st.markdown("**Averaged Cycle — Magnitude Spectrum**")

            # Get window settings
            win_start_frac = float(st.session_state.get('series_analysis_window_start', 0.0))
            win_end_frac = float(st.session_state.get('series_analysis_window_end', 1.0))

            colx, coly = st.columns([2, 1])
            with colx:
                st.caption(
                    f"Window: {win_start_frac:.2f} – {win_end_frac:.2f} "
                    f"(fraction of averaged cycle)"
                )
            with coly:
                log_x = st.checkbox("Log frequency axis", value=True, key="series_spectrum_logx")

            # Use AudioVisualizer static method for spectrum plot
            fig = AudioVisualizer.render_spectrum_plot(
                audio_data=avg,
                sample_rate=sample_rate,
                title="Averaged Impulse Response — Spectrum",
                log_x=log_x,
                window_func="hanning",
                window_range=(win_start_frac, win_end_frac),
                figsize=(6.5, 3.0)
            )

            st.pyplot(fig, use_container_width=True)

    # ----------------------
    # Advanced settings
    # ----------------------
    def _render_advanced_settings(self) -> None:
        """Utilities operating on the shared recorder; no new instances created."""
        if not self.recorder:
            st.warning("Recorder unavailable")
            return

        st.markdown("**Recorder Snapshot**")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Sample Rate", int(getattr(self.recorder, 'sample_rate', 0)))
        with c2:
            st.metric("Num Pulses", int(getattr(self.recorder, 'num_pulses', 0)))
        with c3:
            st.metric("Cycle (ms)", f"{float(getattr(self.recorder, 'cycle_duration', 0.0))*1000:.1f}")
        with c4:
            st.metric("Volume", f"{float(getattr(self.recorder, 'volume', 0.0)):.2f}")

        st.markdown("---")
        st.markdown("**State Management**")
        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("Rebuild Playback Signal"):
                try:
                    if hasattr(self.recorder, "_generate_complete_signal"):
                        _ = self.recorder._generate_complete_signal()
                        st.success("Playback signal regenerated.")
                    else:
                        st.info("Recorder has no _generate_complete_signal()")
                except Exception as e:
                    st.error(f"Failed to rebuild signal: {e}")
        with col_b:
            if st.button("Show SDL Core Info"):
                try:
                    info = self.recorder.get_sdl_core_info()
                    st.json(info)
                except Exception as e:
                    st.error(f"Failed to get SDL info: {e}")

        with st.expander("Derived Parameters"):
            try:
                st.write(f"pulse_samples: {int(getattr(self.recorder, 'pulse_samples', 0))}")
                st.write(f"fade_samples: {int(getattr(self.recorder, 'fade_samples', 0))}")
                st.write(f"cycle_samples: {int(getattr(self.recorder, 'cycle_samples', 0))}")
                st.write(f"gap_samples: {int(getattr(self.recorder, 'gap_samples', 0))}")
                st.write(f"total_duration: {float(getattr(self.recorder, 'total_duration', 0.0))} s")
            except Exception:
                pass

    # ----------------------
    # Export / config I/O
    # ----------------------
    def _export_series_config(self) -> None:
        cfg = {
            "series_config": {
                "num_pulses": int(st.session_state['series_num_pulses']),
                "pulse_duration_ms": float(st.session_state['series_pulse_duration']),
                "cycle_duration_ms": float(st.session_state['series_cycle_duration']),
                "pulse_frequency": float(st.session_state['series_pulse_frequency']),
                "pulse_volume": float(st.session_state['series_pulse_volume']),
                "pulse_form": str(st.session_state['series_pulse_form']),
                "fade_duration_ms": float(st.session_state['series_fade_duration']),
                "record_extra_time_ms": float(st.session_state['series_record_extra_time']),
                "averaging_start_cycle": int(st.session_state['series_averaging_start_cycle']),
            },
            "audio_settings": {
                "sample_rate": int(st.session_state.get('audio_sample_rate',
                                                        getattr(self.recorder, 'sample_rate', 48000))),
            },
            "export_timestamp": float(time.time()),
            "export_version": "1.1",
        }
        js = json.dumps(cfg, indent=2)
        st.download_button("Download Series Configuration",
                           data=js,
                           file_name=f"series_config_{int(time.time())}.json",
                           mime="application/json")
        with st.expander("Configuration Preview"):
            st.code(js, language="json")
