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

        tab1, tab2 = st.tabs([
            "Series Controls",
            "Advanced Settings"
        ])
        with tab1:
            self._render_pulse_series_config()
            # Recording & Analysis now appears below Series Controls
            if st.session_state.get('series_recorded_audio') is not None:
                st.markdown("---")
                self._render_recording_analysis()
        with tab2:
            self._render_advanced_settings()

    # ----------------------
    # Session defaults
    # ----------------------
    def _init_session_state(self) -> None:
        # Load configuration from file if available
        config = self._load_config_from_file()

        # Ensure correct base types (ints for counts, floats for durations/freqs)
        # Priority: 1) Config file, 2) Recorder attributes, 3) Hard-coded defaults
        # NOTE: Config file stores durations in SECONDS, UI uses MILLISECONDS
        defaults = {
            'series_num_pulses': int(config.get('num_pulses', getattr(self.recorder, 'num_pulses', 8) if self.recorder else 8)),
            'series_pulse_duration': float(1000.0 * config.get('pulse_duration', getattr(self.recorder, 'pulse_duration', 0.008) if self.recorder else 0.008)),     # s -> ms
            'series_cycle_duration': float(1000.0 * config.get('cycle_duration', getattr(self.recorder, 'cycle_duration', 0.1) if self.recorder else 0.1)),     # s -> ms
            'series_pulse_frequency': float(config.get('pulse_frequency', getattr(self.recorder, 'pulse_frequency', 1000.0) if self.recorder else 1000.0)),        # Hz
            'series_pulse_volume': float(config.get('volume', getattr(self.recorder, 'volume', 0.4) if self.recorder else 0.4)),
            'series_pulse_form': str(config.get('impulse_form', getattr(self.recorder, 'impulse_form', 'sine') if self.recorder else 'sine')),
            'series_fade_duration': float(1000.0 * config.get('pulse_fade', getattr(self.recorder, 'pulse_fade', 0.0001) if self.recorder else 0.0001)),         # s -> ms

            # Analysis parameters from series_config section
            'series_record_extra_time': float(config.get('series_config', {}).get('record_extra_time_ms', 200.0)),
            'series_averaging_start_cycle': int(config.get('series_config', {}).get('averaging_start_cycle', 2)),
            'series_show_individual_cycles': True,
            'series_show_averaged_result': True,

            # Recording data cache
            'series_recorded_audio': None,
            'series_sample_rate': int(config.get('sample_rate', getattr(self.recorder, 'sample_rate', 48000) if self.recorder else 48000)),
            'series_timestamp': 0.0,
            'series_analysis_data': {},

            # Visualization options
            'series_cycle_overlay_mode': 'all',
            'series_analysis_window_start': 0.0,
            'series_analysis_window_end': 1.0,

            # NEW: Recording mode (Phase 1 - Calibration Mode Integration)
            'series_recording_mode': str(config.get('default_recording_mode', 'calibration')),
            'series_recording_mode_used': 'calibration',  # Track mode used in last recording
        }
        for k, v in defaults.items():
            st.session_state.setdefault(k, v)

    # ----------------------
    # Configuration file I/O
    # ----------------------
    def _load_config_from_file(self) -> Dict[str, Any]:
        """Load configuration using centralized config manager.

        Returns:
            Dictionary with configuration values, or empty dict if file doesn't exist
        """
        from config_manager import config_manager
        return config_manager.load_config()

    def _save_config_to_file(self) -> bool:
        """Save current series settings using centralized config manager.

        Returns:
            True if successful, False otherwise
        """
        try:
            from config_manager import config_manager

            # Load existing config or create new one
            config = config_manager.load_config()

            # Update with current session values (convert from ms to seconds where needed)
            config['sample_rate'] = int(st.session_state['series_sample_rate'])
            config['pulse_duration'] = float(st.session_state['series_pulse_duration']) / 1000.0  # ms -> s
            config['pulse_fade'] = float(st.session_state['series_fade_duration']) / 1000.0  # ms -> s
            config['cycle_duration'] = float(st.session_state['series_cycle_duration']) / 1000.0  # ms -> s
            config['num_pulses'] = int(st.session_state['series_num_pulses'])
            config['volume'] = float(st.session_state['series_pulse_volume'])
            config['pulse_frequency'] = float(st.session_state['series_pulse_frequency'])
            config['impulse_form'] = str(st.session_state['series_pulse_form'])

            # Update series_config section
            if 'series_config' not in config:
                config['series_config'] = {}

            config['series_config']['record_extra_time_ms'] = float(st.session_state['series_record_extra_time'])
            config['series_config']['averaging_start_cycle'] = int(st.session_state['series_averaging_start_cycle'])

            # Save using config manager
            return config_manager.save_config(config, updated_by="Series Settings Panel")

        except Exception as e:
            st.error(f"Failed to save configuration: {e}")
            return False

    # ----------------------
    # Recording Mode Selection (Phase 1 - Calibration Mode Integration)
    # ----------------------
    def _render_recording_mode_selection(self) -> str:
        """
        Render recording mode selection UI.

        Only shows Calibration option if multi-channel with calibration sensor configured.

        Returns:
            Selected mode: 'standard' or 'calibration'
        """
        st.markdown("### Recording Mode")

        # Check if calibration mode is available
        mc_config = getattr(self.recorder, 'multichannel_config', {})
        mc_enabled = mc_config.get('enabled', False)
        has_calibration = mc_config.get('calibration_channel') is not None

        if not mc_enabled or not has_calibration:
            st.info("ℹ️ **Standard Mode** (Room Response)")
            st.caption("Calibration mode requires multi-channel setup with calibration sensor. Configure in Device Selection tab.")
            st.session_state['series_recording_mode'] = 'standard'
            return 'standard'

        # Get current mode from session state
        current_mode = st.session_state.get('series_recording_mode', 'calibration')
        default_index = 0 if current_mode == 'standard' else 1

        mode_selection = st.radio(
            "Choose recording mode:",
            options=["Standard (Room Response)", "Calibration (Physical Impact)"],
            index=default_index,
            key="series_recording_mode_radio",
            help="""
            **Standard Mode:**
            - Record room acoustic responses using synthetic pulse train
            - Audio output from speaker, captured by microphones
            - Best for: Room impulse response measurements

            **Calibration Mode:**
            - Record physical impact responses (e.g., hammer strikes)
            - Requires calibration sensor (force/impact sensor)
            - Per-cycle quality validation
            - Automatic alignment and optional normalization
            - Best for: Piano hammer impact studies, sensor calibration
            """
        )

        selected_mode = 'calibration' if 'Calibration' in mode_selection else 'standard'
        st.session_state['series_recording_mode'] = selected_mode

        return selected_mode

    def _render_calibration_mode_info(self) -> None:
        """Display calibration mode configuration details when enabled."""
        current_mode = st.session_state.get('series_recording_mode', 'standard')

        if current_mode != 'calibration':
            return

        mc_config = getattr(self.recorder, 'multichannel_config', {})

        with st.expander("🔨 Calibration Mode Configuration", expanded=True):
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**Sensor Setup**")
                cal_ch = mc_config.get('calibration_channel')
                channel_names = mc_config.get('channel_names', [])
                cal_name = channel_names[cal_ch] if cal_ch is not None and cal_ch < len(channel_names) else f"Channel {cal_ch}"

                st.success(f"🔨 Calibration Sensor: Ch {cal_ch} - {cal_name}")

                ref_ch = mc_config.get('reference_channel', 0)
                ref_name = channel_names[ref_ch] if ref_ch < len(channel_names) else f"Channel {ref_ch}"
                st.info(f"🎤 Reference Channel: Ch {ref_ch} - {ref_name}")

            with col2:
                st.markdown("**Processing Options**")

                # Normalization toggle - default to True for calibration mode
                current_normalize = mc_config.get('normalize_by_calibration', True)
                normalize_enabled = st.checkbox(
                    "Enable Normalization",
                    value=current_normalize,
                    key="series_normalize_by_calibration",
                    help="Divide response amplitudes by impact magnitude for quantitative comparison"
                )

                # Update recorder config if changed
                if normalize_enabled != current_normalize:
                    self.recorder.multichannel_config['normalize_by_calibration'] = normalize_enabled
                    if normalize_enabled:
                        st.success("✅ Normalization enabled - responses will be normalized by impact magnitude")
                    else:
                        st.info("ℹ️ Normalization disabled - responses will show raw aligned amplitudes")

                # Show current status
                if normalize_enabled:
                    st.caption("✅ Responses normalized by impact magnitude")
                else:
                    st.caption("⚠️ Raw aligned responses (not normalized)")

            # Display quality thresholds summary
            cal_config = getattr(self.recorder, 'calibration_quality_config', {})
            if cal_config:
                st.markdown("**Quality Validation**")
                with st.container():
                    col_a, col_b = st.columns(2)
                    with col_a:
                        min_neg = cal_config.get('min_negative_peak', 0)
                        max_neg = cal_config.get('max_negative_peak', 1)
                        st.caption(f"Negative peak range: {min_neg:.2f} - {max_neg:.2f}")
                    with col_b:
                        corr_thresh = mc_config.get('alignment_correlation_threshold', 0.7)
                        st.caption(f"Correlation threshold: {corr_thresh:.2f}")

    def _render_mode_comparison_table(self):
        """Display comparison table of Standard vs Calibration modes."""
        with st.expander("ℹ️ Recording Mode Comparison"):
            st.markdown("""
            | Feature | Standard Mode | Calibration Mode |
            |---------|---------------|------------------|
            | **Signal Source** | Synthetic pulse (speaker) | Physical impact (hammer) |
            | **Best For** | Room acoustics, reverb | Impact studies, piano research |
            | **Quality Validation** | ❌ No | ✅ Yes (per-cycle) |
            | **Cycle Alignment** | Basic (assumes perfect timing) | ✅ Advanced (onset detection) |
            | **Normalization** | ❌ No | ✅ Optional (by impact magnitude) |
            | **Calibration Sensor** | Not required | ✅ Required |
            | **File Output** | Raw + Impulse + Room Response | Aligned + Normalized + Metadata |
            | **Typical Use** | Room impulse responses | Piano hammer characterization |
            """)

            st.markdown("**When to use Calibration Mode:**")
            st.markdown("""
            - ✅ Recording physical impacts (hammer strikes, taps)
            - ✅ Need per-event quality validation
            - ✅ Varying impact magnitudes that need normalization
            - ✅ Precise alignment required for time-domain analysis

            **When to use Standard Mode:**
            - ✅ Synthetic audio signals (pulse trains)
            - ✅ Room acoustic measurements
            - ✅ Controlled signal source with consistent timing
            - ✅ Basic impulse response extraction
            """)

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

        # Show configuration status using centralized config manager
        from config_manager import config_manager
        config_path = config_manager.get_config_path()
        if config_path.exists():
            st.info(f"ℹ️ Settings loaded from **{config_path.name}** — Changes are applied to recorder but not saved until you click 'Save Configuration'")
        else:
            st.warning(f"⚠️ No configuration file found at {config_path} — Using default values")

        st.markdown("---")

        # NEW: Recording mode selection (Phase 1 - Calibration Mode Integration)
        recording_mode = self._render_recording_mode_selection()

        # NEW: Show calibration config if in calibration mode
        if recording_mode == 'calibration':
            self._render_calibration_mode_info()

        # NEW: Mode comparison table
        self._render_mode_comparison_table()

        st.markdown("---")

        # DEBUG: Show what's in the recorder vs config file
        with st.expander("🔍 DEBUG: Recorder vs Config File Settings", expanded=False):
            if self.recorder:
                st.markdown("**Current Recorder Settings:**")
                st.json({
                    "sample_rate": self.recorder.sample_rate,
                    "pulse_duration": self.recorder.pulse_duration,
                    "cycle_duration": self.recorder.cycle_duration,
                    "num_pulses": self.recorder.num_pulses,
                    "pulse_frequency": self.recorder.pulse_frequency,
                    "impulse_form": self.recorder.impulse_form,
                    "volume": self.recorder.volume
                })

            config = config_manager.load_config()
            st.markdown("**Config File Settings:**")
            st.json({
                "sample_rate": config.get('sample_rate'),
                "pulse_duration": config.get('pulse_duration'),
                "cycle_duration": config.get('cycle_duration'),
                "num_pulses": config.get('num_pulses'),
                "pulse_frequency": config.get('pulse_frequency'),
                "impulse_form": config.get('impulse_form'),
                "volume": config.get('volume')
            })

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
            # Get sample rate to constrain pulse duration
            current_sample_rate = int(st.session_state['series_sample_rate'])

            # Minimum pulse duration = one sample period
            min_pulse_ms = (1.0 / current_sample_rate) * 1000.0  # Convert to ms

            # Maximum pulse duration = 100 ms
            max_pulse_ms = 100.0

            # floats across the board
            pulse_duration_ms = st.number_input(
                "Pulse duration (ms)",
                min_value=float(min_pulse_ms),
                max_value=float(max_pulse_ms),
                value=float(min(max(st.session_state['series_pulse_duration'], min_pulse_ms), max_pulse_ms)),
                step=float(min_pulse_ms),
                help=f"Min: 1 sample ({min_pulse_ms:.4f} ms @ {current_sample_rate}Hz) | Max: {max_pulse_ms} ms"
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
            waveform_options = ["sine", "square", "voice_coil"]
            current_form = st.session_state['series_pulse_form']
            if current_form in waveform_options:
                default_idx = waveform_options.index(current_form)
            else:
                default_idx = 0
            pulse_form = st.selectbox(
                "Pulse waveform",
                waveform_options,
                index=default_idx
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
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            if st.button("🎵 Record Series",
                         disabled=not (SDL_AVAILABLE and RECORDER_AVAILABLE and self.recorder)):
                self._execute_series_recording()
        with c2:
            if st.button("💾 Save Configuration", type="primary"):
                if self._save_config_to_file():
                    st.success("✓ Configuration saved to recorderConfig.json")
                    st.info("Settings will be used as defaults in future sessions")
                    st.rerun()
        with c3:
            if st.button("🔄 Reset to Saved"):
                config = self._load_config_from_file()
                if config:
                    # Clear session state to force reload from config
                    keys_to_clear = [k for k in st.session_state.keys() if k.startswith('series_')]
                    for k in keys_to_clear:
                        del st.session_state[k]
                    st.success("✓ Settings reset to saved configuration")
                    st.rerun()
                else:
                    st.warning("No saved configuration found")
        with c4:
            if st.button("⚙️ Export Config"):
                self._export_series_config()

    # ----------------------
    # Record / Preview (use shared recorder settings as-is)
    # ----------------------
    def _execute_series_recording(self) -> None:
        if not self.recorder:
            st.error("Recorder unavailable")
            return

        # Get selected recording mode
        recording_mode = st.session_state.get('series_recording_mode', 'standard')

        try:
            # Display mode indicator
            if recording_mode == 'calibration':
                st.info("🔨 Recording with Calibration Mode (quality validation enabled)")

            with st.spinner(f"Recording pulse series ({recording_mode} mode)..."):
                tmp = Path("TMP"); tmp.mkdir(exist_ok=True)
                for item in tmp.iterdir():
                    if item.is_file():
                        item.unlink()
                    elif item.is_dir():
                        shutil.rmtree(item)
                ts = int(time.time())
                raw_path = tmp / f"series_raw_{ts}.wav"
                imp_path = tmp / f"series_impulse_{ts}.wav"

                # Call recorder with mode parameter
                # REFACTORED: Use return_processed=True to get complete processed dict
                recorded_audio = self.recorder.take_record(
                    output_file=str(raw_path),
                    impulse_file=str(imp_path),
                    method=2,
                    mode=recording_mode,
                    return_processed=True,  # Get complete processed dict for both modes
                    save_files=True  # Explicitly save files for testing
                )

                if recorded_audio is None:
                    st.error("Recording failed — no audio captured")
                    return

                # Mode-specific result handling
                if recording_mode == 'calibration':
                    # Calibration mode returns a dict with cycle data
                    if isinstance(recorded_audio, dict) and 'calibration_cycles' in recorded_audio:
                        st.success(f"✅ Calibration recording completed")

                        # Display validation summary
                        metadata = recorded_audio.get('metadata', {})
                        num_total = len(recorded_audio.get('validation_results', []))
                        num_valid = metadata.get('num_valid_cycles', 0)
                        num_aligned = metadata.get('num_aligned_cycles', 0)
                        normalize_enabled = metadata.get('normalize_by_calibration', False)

                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Total Cycles", num_total)
                        with col2:
                            st.metric("Valid Cycles", f"{num_valid} ({100*num_valid/max(num_total,1):.1f}%)")
                        with col3:
                            st.metric("Aligned Cycles", num_aligned)
                        with col4:
                            if normalize_enabled:
                                st.metric("Normalization", "✅ Enabled")
                            else:
                                st.metric("Normalization", "❌ Disabled")

                        # Show validation details in expander
                        with st.expander("📊 Per-Cycle Validation Results"):
                            validation_results = recorded_audio.get('validation_results', [])
                            for i, result in enumerate(validation_results):
                                status = "✅ Valid" if result.get('is_valid', False) else "❌ Invalid"
                                st.write(f"**Cycle {i}:** {status}")
                                if not result.get('is_valid', False):
                                    reasons = result.get('reasons', [])
                                    if reasons:
                                        st.caption(f"  Reasons: {', '.join(reasons)}")

                        # For analysis, prepare multi-channel data from calibration result
                        # IMPORTANT: Use normalized cycles if normalization was enabled, otherwise use aligned cycles
                        normalize_enabled = recorded_audio.get('metadata', {}).get('normalize_by_calibration', False)

                        if normalize_enabled:
                            # Use normalized cycles (responses divided by impact magnitude)
                            cycles_to_use = recorded_audio.get('normalized_multichannel_cycles', {})
                            if not cycles_to_use:
                                # Fallback to aligned if normalized not available
                                st.warning("Normalization was enabled but normalized data not found. Using aligned cycles.")
                                cycles_to_use = recorded_audio.get('aligned_multichannel_cycles', {})
                        else:
                            # Use aligned cycles (no normalization)
                            cycles_to_use = recorded_audio.get('aligned_multichannel_cycles', {})

                        if cycles_to_use:
                            # Flatten each channel's cycles for visualization/analysis
                            # This creates a dict similar to standard mode multi-channel output
                            flattened_channels = {}
                            for ch_idx, cycles_array in cycles_to_use.items():
                                # cycles_array shape: [num_cycles, samples_per_cycle]
                                flattened_channels[ch_idx] = cycles_array.reshape(-1)

                            # For calibration mode: Use flattened cycles for visualization
                            # (these are already aligned/normalized as appropriate)
                            analysis_audio = flattened_channels

                            # Store the full recorded_audio dict for cycle statistics table
                            # (contains metadata, validation_results, alignment_metadata, etc.)
                            st.session_state['series_calibration_data'] = recorded_audio

                            # Also store processed cycles separately for cycle analysis
                            st.session_state['series_processed_cycles'] = flattened_channels

                            # Store raw audio separately for potential future use
                            raw_audio = recorded_audio.get('raw', {})
                            st.session_state['series_raw_audio'] = raw_audio

                            # Calculate duration from reference channel
                            ref_ch = self.recorder.multichannel_config.get('reference_channel', 0)
                            if ref_ch in analysis_audio:
                                duration = len(analysis_audio[ref_ch]) / self.recorder.sample_rate
                            else:
                                first_ch = list(analysis_audio.keys())[0]
                                duration = len(analysis_audio[first_ch]) / self.recorder.sample_rate
                        else:
                            st.warning("No cycle data available for analysis")
                            return

                    else:
                        st.error("Unexpected calibration result format")
                        return
                else:
                    # Standard mode - now returns processed dict (REFACTORED)
                    # recorded_audio is now a processed_data dict with structure:
                    # {'raw': {...}, 'individual_cycles': {...}, 'room_response': {...}, 'metadata': {...}}

                    if not isinstance(recorded_audio, dict) or 'metadata' not in recorded_audio:
                        st.error("Unexpected standard mode result format")
                        return

                    # Extract metadata for display
                    metadata = recorded_audio.get('metadata', {})
                    num_channels = metadata.get('num_channels', 1)

                    # Get raw audio for display/playback
                    raw_audio_dict = recorded_audio.get('raw', {})
                    ref_ch = metadata.get('reference_channel', 0)

                    if ref_ch in raw_audio_dict:
                        ref_audio = raw_audio_dict[ref_ch]
                    else:
                        ref_audio = list(raw_audio_dict.values())[0] if raw_audio_dict else np.array([])

                    duration = len(ref_audio) / self.recorder.sample_rate if len(ref_audio) > 0 else 0.0

                    # Store raw audio for visualization (backward compatible)
                    analysis_audio = raw_audio_dict if num_channels > 1 else ref_audio

                    st.success(f"✅ Standard mode recording completed ({num_channels} channel{'s' if num_channels > 1 else ''})")

                # Run analysis using new refactored method (UNIFIED for both modes)
                analysis = self._analyze_series_recording(recorded_audio, self.recorder)

                # Store COMPLETE processed_data dict for channel switching (REFACTORED)
                st.session_state['series_processed_data'] = recorded_audio

                # Store data for visualization (backward compatible)
                st.session_state['series_recorded_audio'] = analysis_audio
                st.session_state['series_sample_rate'] = int(self.recorder.sample_rate)
                st.session_state['series_timestamp'] = time.time()
                st.session_state['series_analysis_data'] = analysis
                st.session_state['series_recording_mode_used'] = recording_mode  # Store mode used

                # Store which channel the analysis was computed for
                metadata = recorded_audio.get('metadata', {})
                ref_ch = metadata.get('reference_channel', 0)
                st.session_state['series_analysis_channel'] = ref_ch

                st.success(f"Series recording OK — {duration:.3f}s")
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
    def _analyze_series_recording(self, processed_data: Dict[str, Any], recorder, channel: int = None) -> Dict[str, Any]:
        """
        Extract analysis data from backend-processed results.

        REFACTORED: This method now only extracts and formats data from processed_data dict.
        All signal processing (cycle extraction, averaging) is done in the backend.

        Args:
            processed_data: Complete processed data dict from RoomResponseRecorder
            recorder: Recorder instance (for config access only)
            channel: Optional channel index to analyze (default: reference channel)

        Returns:
            Dict with analysis results formatted for visualization
        """
        analysis: Dict[str, Any] = {}
        try:
            # Get metadata
            metadata = processed_data.get('metadata', {})
            sr = int(metadata.get('sample_rate', getattr(recorder, 'sample_rate', 48000)))

            # Get channel to analyze
            if channel is None:
                # Default to reference channel
                ref_ch = metadata.get('reference_channel', 0)
                if 'reference_channel' not in metadata:
                    # Single-channel or no explicit reference
                    ref_ch = 0
            else:
                ref_ch = channel

            # Extract pre-computed cycles from backend
            individual_cycles_dict = processed_data.get('individual_cycles', {})
            cycles_array = individual_cycles_dict.get(ref_ch, np.array([]))

            # Convert from [num_cycles, cycle_samples] to list of arrays
            if cycles_array.ndim == 2:
                cycles = [cycles_array[i] for i in range(cycles_array.shape[0])]
            else:
                cycles = []

            analysis['individual_cycles'] = cycles
            analysis['num_cycles_extracted'] = len(cycles)

            # Get pre-computed averaged response from backend
            room_response_dict = processed_data.get('room_response', {})
            avg_cycle = room_response_dict.get(ref_ch, np.array([]))

            if len(avg_cycle) > 0:
                analysis['averaged_cycle'] = avg_cycle
                analysis['cycles_used_for_averaging'] = metadata.get('cycles_used_for_averaging', len(cycles))
                analysis['averaging_start_cycle'] = metadata.get('averaging_start_cycle', 1)

                # Spectral analysis from backend (REFACTORED: Phase 4)
                spectral_data = processed_data.get('spectral_analysis', {})
                if spectral_data and 'frequencies' in spectral_data:
                    # Backend already computed spectrum - extract for this channel
                    freqs = spectral_data.get('frequencies', np.array([]))
                    mag_db_dict = spectral_data.get('magnitude_db', {})
                    mag_db = mag_db_dict.get(ref_ch, np.array([]))
                    window = spectral_data.get('window', [0.0, 1.0])
                    n_fft = spectral_data.get('n_fft', 0)

                    if len(freqs) > 0 and len(mag_db) > 0:
                        analysis['averaged_spectrum'] = {
                            'freqs': freqs,
                            'magnitude_db': mag_db,
                            'window': window,
                            'n_fft': n_fft
                        }
                else:
                    # Fallback: compute locally if backend data not available
                    # (for backward compatibility with old recordings)
                    try:
                        win_start_frac = float(st.session_state.get('series_analysis_window_start', 0.0))
                        win_end_frac = float(st.session_state.get('series_analysis_window_end', 1.0))
                        win_start_frac = max(0.0, min(1.0, win_start_frac))
                        win_end_frac = max(0.0, min(1.0, win_end_frac))
                        if win_end_frac <= win_start_frac:
                            win_end_frac = min(1.0, win_start_frac + 0.05)

                        N = len(avg_cycle)
                        s = int(math.floor(N * win_start_frac))
                        e = int(math.ceil(N * win_end_frac))
                        seg = np.asarray(avg_cycle[s:e], dtype=np.float32)

                        if seg.size > 1:
                            w = np.hanning(seg.size).astype(np.float32)
                            seg_w = seg * w
                        else:
                            seg_w = seg

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

            # Consistency metric (computed from pre-extracted cycles)
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

            # Full recording metrics (from raw audio)
            raw_audio_dict = processed_data.get('raw', {})
            audio_data = raw_audio_dict.get(ref_ch, np.array([]))

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

        # Show recording mode indicator if available
        recording_mode_used = st.session_state.get('series_recording_mode_used', 'standard')
        if recording_mode_used == 'calibration':
            st.info("🔨 Last recording used **Calibration Mode** (with quality validation)")
        else:
            st.info("📊 Last recording used **Standard Mode** (room response)")

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

        # Show cycle statistics table for calibration mode
        self._render_cycle_statistics_table()

        self._render_visualization_controls()

        # Collapse Full Recording section by default
        with st.expander("**Full Recording**", expanded=False):
            if VISUALIZER_AVAILABLE:
                # Handle multi-channel data - extract single channel for visualization
                if isinstance(audio, dict):
                    # Multi-channel: get reference channel or first available
                    ref_ch = self.recorder.multichannel_config.get('reference_channel', 0)
                    available_channels = list(audio.keys())

                    # Allow user to select which channel to visualize
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        selected_ch = st.selectbox(
                            "Visualize Channel",
                            available_channels,
                            index=available_channels.index(ref_ch) if ref_ch in available_channels else 0,
                            key="series_viz_channel"
                        )
                    with col2:
                        st.caption(f"{len(available_channels)} channels")

                    viz_audio = audio[selected_ch]
                    viz_title = f"Complete Series Recording - Channel {selected_ch}"

                    # Re-compute analysis for selected channel (if different from stored analysis)
                    # REFACTORED: Use processed_data dict with channel parameter
                    stored_analysis_channel = st.session_state.get('series_analysis_channel', ref_ch)
                    if selected_ch != stored_analysis_channel:
                        # Get stored processed_data dict (unified for all modes)
                        processed_data = st.session_state.get('series_processed_data')
                        if processed_data:
                            # Recompute analysis for the selected channel using unified method
                            analysis = self._analyze_series_recording(processed_data, self.recorder, channel=selected_ch)
                        # Don't overwrite session state - keep original reference channel analysis
                else:
                    # Single-channel
                    viz_audio = audio
                    viz_title = "Complete Series Recording"

                AudioVisualizer("series_full_recording").render(
                    audio_data=viz_audio,
                    sample_rate=sr,
                    title=viz_title,
                    show_controls=True,
                    show_analysis=True,
                    height=400
                )

        if analysis.get('individual_cycles'):
            self._render_cycle_analysis(analysis, sr)
            self._render_cycle_consistency_overlay(analysis, sr)

        # Only render averaged analysis for standard mode (calibration mode shows it in cycle statistics section)
        recording_mode = st.session_state.get('series_recording_mode_used', 'standard')
        if analysis.get('averaged_cycle') is not None and recording_mode == 'standard':
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

    def _render_cycle_statistics_table(self) -> None:
        """
        Render comprehensive cycle statistics table for calibration mode recordings.
        Shows validation status, peak amplitudes before/after normalization,
        alignment positions, and rejection reasons for all cycles.
        """
        recording_mode = st.session_state.get('series_recording_mode_used', 'standard')
        if recording_mode != 'calibration':
            return  # Only show for calibration mode

        # Get the full calibration data (not just flattened channels)
        recorded_audio = st.session_state.get('series_calibration_data')
        if recorded_audio is None or not isinstance(recorded_audio, dict):
            return

        # Extract metadata
        metadata = recorded_audio.get('metadata', {})
        validation_results = metadata.get('validation_results', [])
        alignment_metadata = metadata.get('alignment_metadata', {})
        normalize_enabled = metadata.get('normalize_by_calibration', False)

        if not validation_results:
            return  # No cycle data available

        # Get cycle data
        aligned_multichannel = recorded_audio.get('aligned_multichannel_cycles', {})
        normalized_multichannel = recorded_audio.get('normalized_multichannel_cycles', {})
        normalization_factors = recorded_audio.get('normalization_factors', [])
        valid_cycle_indices = alignment_metadata.get('valid_cycle_indices', [])
        onset_positions = alignment_metadata.get('onset_positions', [])
        aligned_onset_position = alignment_metadata.get('aligned_onset_position', 0)
        correlation_threshold = metadata.get('correlation_threshold', 0.7)

        # Get selected channel with UI selector
        available_channels = list(aligned_multichannel.keys()) if aligned_multichannel else [0]
        default_ch = st.session_state.get('series_analysis_channel',
                                         metadata.get('reference_channel', 0))

        # Channel selector for cycle statistics and overlay
        col_ch1, col_ch2 = st.columns([3, 1])
        with col_ch1:
            selected_ch = st.selectbox(
                "Analysis Channel",
                available_channels,
                index=available_channels.index(default_ch) if default_ch in available_channels else 0,
                key="series_cycle_analysis_channel",
                help="Select which channel to display in cycle statistics and overlay chart"
            )
        with col_ch2:
            st.caption(f"{len(available_channels)} channels")

        # Update session state with selected channel
        st.session_state['series_analysis_channel'] = selected_ch

        # Build statistics table
        st.markdown("---")
        st.markdown("**📊 Cycle Statistics (Quality Control)**")
        st.markdown("Select cycles to display in the overlay chart below")

        # Initialize session state for cycle selection
        session_key = 'series_selected_cycles_for_overlay'
        if session_key not in st.session_state:
            # Default: select all kept cycles
            st.session_state[session_key] = valid_cycle_indices.copy()

        selected_cycles = []

        # Render table with checkboxes
        # Column headers
        has_normalization = normalize_enabled
        if has_normalization:
            col_widths = [0.3, 0.4, 0.7, 0.9, 0.8, 0.8, 0.8, 0.8, 0.8]
            cols = st.columns(col_widths)
            headers = ["Select", "Cycle", "Status", "Rejection Reason", "Impact Mag", "Peak Before", "Peak After", "Onset Pos", "Aligned Pos"]
        else:
            col_widths = [0.3, 0.4, 0.7, 0.9, 0.8, 0.8, 0.8, 0.8]
            cols = st.columns(col_widths)
            headers = ["Select", "Cycle", "Status", "Rejection Reason", "Impact Mag", "Peak Before", "Onset Pos", "Aligned Pos"]

        for col, header in zip(cols, headers):
            col.markdown(f"**{header}**")

        st.markdown("---")

        # Render rows
        for v_result in validation_results:
            cycle_idx = v_result.get('cycle_index', 0)
            is_valid = v_result.get('calibration_valid', False)
            metrics = v_result.get('calibration_metrics', {})
            failures = v_result.get('calibration_failures', [])

            # Check if this cycle was kept after alignment
            kept_after_alignment = cycle_idx in valid_cycle_indices

            # Get rejection reason
            if not is_valid:
                rejection_reason = "Validation: " + ", ".join(failures) if failures else "Validation failed"
            elif not kept_after_alignment:
                # Cycle was valid but rejected by correlation filter
                rejection_reason = f"Correlation filter (< {correlation_threshold})"
            else:
                rejection_reason = "-"

            # Get peak amplitude before normalization (from aligned cycles)
            peak_before = 0.0
            if kept_after_alignment and selected_ch in aligned_multichannel:
                try:
                    aligned_idx = valid_cycle_indices.index(cycle_idx)
                    cycle_data = aligned_multichannel[selected_ch][aligned_idx]
                    peak_before = float(np.max(np.abs(cycle_data)))
                except (ValueError, IndexError):
                    pass

            # Get peak amplitude after normalization (if enabled)
            peak_after = 0.0
            if normalize_enabled and kept_after_alignment and selected_ch in normalized_multichannel:
                try:
                    aligned_idx = valid_cycle_indices.index(cycle_idx)
                    cycle_data = normalized_multichannel[selected_ch][aligned_idx]
                    peak_after = float(np.max(np.abs(cycle_data)))
                except (ValueError, IndexError):
                    pass

            # Get normalization factor (impact magnitude)
            norm_factor = 0.0
            if cycle_idx < len(normalization_factors):
                norm_factor = normalization_factors[cycle_idx]

            # Get onset position (before alignment)
            onset_pos = 0
            if kept_after_alignment:
                try:
                    aligned_idx = valid_cycle_indices.index(cycle_idx)
                    if aligned_idx < len(onset_positions):
                        onset_pos = onset_positions[aligned_idx]
                except (ValueError, IndexError):
                    pass

            # Get aligned position (after alignment - should be same for all kept cycles)
            aligned_pos = aligned_onset_position if kept_after_alignment else 0

            # Render row with checkbox
            if has_normalization:
                cols = st.columns(col_widths)
            else:
                cols = st.columns(col_widths)

            with cols[0]:
                # Only allow selection of kept cycles
                if kept_after_alignment:
                    is_checked = st.checkbox(
                        "",
                        value=cycle_idx in st.session_state.get(session_key, []),
                        key=f"series_calib_cycle_checkbox_{cycle_idx}",
                        label_visibility="collapsed"
                    )
                    if is_checked:
                        selected_cycles.append(cycle_idx)
                else:
                    st.markdown("—")

            with cols[1]:
                st.markdown(f"{cycle_idx}")

            with cols[2]:
                status_text = '✓ Kept' if kept_after_alignment else '✗ Rejected'
                if kept_after_alignment:
                    st.markdown(f":green[{status_text}]")
                else:
                    st.markdown(f":red[{status_text}]")

            with cols[3]:
                st.markdown(rejection_reason)

            with cols[4]:
                st.markdown(f"{norm_factor:.4f}" if norm_factor > 0 else '-')

            with cols[5]:
                st.markdown(f"{peak_before:.4f}" if peak_before > 0 else '-')

            if has_normalization:
                with cols[6]:
                    st.markdown(f"{peak_after:.4f}" if peak_after > 0 else 'N/A')

                with cols[7]:
                    st.markdown(str(onset_pos) if kept_after_alignment else '-')

                with cols[8]:
                    st.markdown(str(aligned_pos) if kept_after_alignment else '-')
            else:
                with cols[6]:
                    st.markdown(str(onset_pos) if kept_after_alignment else '-')

                with cols[7]:
                    st.markdown(str(aligned_pos) if kept_after_alignment else '-')

        # Update session state with selected cycles
        st.session_state[session_key] = selected_cycles

        # Display selection info
        if selected_cycles:
            st.success(f"✓ Selected {len(selected_cycles)} cycle(s) for overlay: {', '.join(map(str, selected_cycles))}")
        else:
            st.info("Select cycles above to display in the overlay chart")

        # Render cycle overlay visualization
        if selected_cycles:
            self._render_calibration_cycle_overlay(
                selected_cycles=selected_cycles,
                valid_cycle_indices=valid_cycle_indices,
                aligned_multichannel=aligned_multichannel,
                normalized_multichannel=normalized_multichannel,
                normalize_enabled=normalize_enabled,
                selected_ch=selected_ch,
                sample_rate=metadata.get('sample_rate', 48000)
            )

        # Add averaged impulse response statistics
        if kept_after_alignment and selected_ch in aligned_multichannel:
            st.markdown("**📈 Averaged Impulse Response Statistics**")

            # Get averaged response from analysis
            analysis = st.session_state.get('series_analysis_data', {})
            averaged_cycle = analysis.get('averaged_cycle')

            if averaged_cycle is not None:
                averaged_cycle = np.array(averaged_cycle)

                # Find maximum absolute value in averaged cycle
                avg_peak = float(np.max(np.abs(averaged_cycle)))
                avg_peak_pos = int(np.argmax(np.abs(averaged_cycle)))
                avg_rms = float(np.sqrt(np.mean(averaged_cycle ** 2)))
                num_cycles_averaged = analysis.get('cycles_used_for_averaging', 0)

                # Calculate mean of individual peak amplitudes (also using maximum)
                individual_peaks = []
                individual_peak_positions = []
                cycles_to_use = normalized_multichannel if normalize_enabled else aligned_multichannel
                if selected_ch in cycles_to_use:
                    for cycle_data in cycles_to_use[selected_ch]:
                        peak_val = float(np.max(np.abs(cycle_data)))
                        peak_pos = int(np.argmax(np.abs(cycle_data)))
                        individual_peaks.append(peak_val)
                        individual_peak_positions.append(peak_pos)

                mean_individual_peak = np.mean(individual_peaks) if individual_peaks else 0.0

                # Calculate ratio
                peak_ratio = (avg_peak / mean_individual_peak) if mean_individual_peak > 0 else 0.0

                # Verify peaks are near aligned position (for quality check)
                aligned_pos = aligned_onset_position
                peak_alignment_ok = True
                if individual_peak_positions:
                    mean_peak_pos = np.mean(individual_peak_positions)
                    # Check if peaks deviate significantly from aligned position
                    if abs(mean_peak_pos - aligned_pos) > 50:  # More than 50 samples off
                        peak_alignment_ok = False

                col1, col2, col3, col4, col5 = st.columns(5)
                with col1:
                    st.metric("Cycles Averaged", num_cycles_averaged)
                with col2:
                    st.metric("Avg Peak Amplitude", f"{avg_peak:.4f}")
                with col3:
                    st.metric("Mean Individual Peak", f"{mean_individual_peak:.4f}")
                with col4:
                    st.metric("Peak Ratio", f"{peak_ratio:.3f}",
                             help="Should be ~1.0 if normalization is correct")
                with col5:
                    st.metric("Avg Peak Position", f"{avg_peak_pos} samples")

                # Warning if ratio is off
                if normalize_enabled and (peak_ratio < 0.95 or peak_ratio > 1.05):
                    st.warning(f"⚠️ Peak ratio {peak_ratio:.3f} is outside expected range (0.95-1.05). "
                              "This may indicate an issue with normalization or averaging.")

                # Warning if peak positions are misaligned
                if not peak_alignment_ok:
                    st.warning(f"⚠️ Peak positions deviate significantly from aligned position {aligned_pos}. "
                              "This may indicate an alignment issue.")

            # Render averaged cycle visualization for selected channel
            st.markdown("---")
            st.markdown("**📈 Averaged Cycle — Final Result**")
            st.caption(f"Averaged response for Channel {selected_ch}")

            # Compute averaged cycle from aligned/normalized multichannel data
            cycles_to_average = normalized_multichannel if normalize_enabled else aligned_multichannel
            if selected_ch in cycles_to_average:
                averaged_cycle_for_channel = np.mean(cycles_to_average[selected_ch], axis=0)

                # Render using AudioVisualizer (VISUALIZER_AVAILABLE defined at module level)
                if VISUALIZER_AVAILABLE:
                    AudioVisualizer("series_averaged_cycle_selected_channel").render(
                        audio_data=averaged_cycle_for_channel,
                        sample_rate=metadata.get('sample_rate', 48000),
                        title=f"Averaged Cycle — Channel {selected_ch}",
                        show_controls=True,
                        show_analysis=True,
                        height=400
                    )

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

    def _display_cycle_overlay_statistics(self,
                                            aligned_multichannel: Dict[int, np.ndarray],
                                            normalized_multichannel: Dict[int, np.ndarray],
                                            normalize_enabled: bool,
                                            selected_ch: int,
                                            array_indices: List[int],
                                            display_mode: str):
        """
        Display statistics for the cycle overlay (peak amplitude range, width %, std dev).

        Args:
            aligned_multichannel: Dict[channel -> aligned cycles array]
            normalized_multichannel: Dict[channel -> normalized cycles array]
            normalize_enabled: Whether normalization is enabled
            selected_ch: Selected channel index
            array_indices: List of array indices for selected cycles
            display_mode: Current display mode
        """
        st.markdown("**📊 Cycle Overlay Statistics**")

        # Determine which data to analyze based on display mode
        if display_mode == "Both (Side-by-Side)" and normalize_enabled:
            # Show statistics for both
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("*Aligned (Raw)*")
                aligned_waveforms = [aligned_multichannel[selected_ch][idx] for idx in array_indices]
                self._show_peak_statistics(aligned_waveforms)

            with col2:
                st.markdown("*Normalized*")
                normalized_waveforms = [normalized_multichannel[selected_ch][idx] for idx in array_indices]
                self._show_peak_statistics(normalized_waveforms)

        elif display_mode == "Normalized" and normalize_enabled and selected_ch in normalized_multichannel:
            # Normalized only
            normalized_waveforms = [normalized_multichannel[selected_ch][idx] for idx in array_indices]
            self._show_peak_statistics(normalized_waveforms)

        else:
            # Aligned (Raw) only
            aligned_waveforms = [aligned_multichannel[selected_ch][idx] for idx in array_indices]
            self._show_peak_statistics(aligned_waveforms)

        st.markdown("---")

    def _show_peak_statistics(self, waveforms: List[np.ndarray]):
        """
        Calculate and display peak amplitude statistics for a set of waveforms.

        Args:
            waveforms: List of cycle waveforms
        """
        if not waveforms:
            return

        # Find main peak (max absolute value) in each cycle
        peak_amplitudes = []
        for waveform in waveforms:
            peak_amp = float(np.max(np.abs(waveform)))
            peak_amplitudes.append(peak_amp)

        peak_amplitudes = np.array(peak_amplitudes)

        # Calculate statistics
        min_peak = float(np.min(peak_amplitudes))
        max_peak = float(np.max(peak_amplitudes))
        range_value = max_peak - min_peak
        avg_peak = float(np.mean(peak_amplitudes))
        std_dev = float(np.std(peak_amplitudes))

        # Calculate range width as percentage of average
        if avg_peak > 0:
            range_width_pct = (range_value / avg_peak) * 100.0
        else:
            range_width_pct = 0.0

        # Display in columns
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Peak Range",
                f"{range_value:.4f}",
                help=f"Range of main peak amplitudes across cycles: {min_peak:.4f} to {max_peak:.4f}"
            )

        with col2:
            st.metric(
                "Range Width %",
                f"{range_width_pct:.2f}%",
                help=f"Range width as percentage of average peak amplitude ({avg_peak:.4f})"
            )

        with col3:
            st.metric(
                "Std Dev",
                f"{std_dev:.4f}",
                help="Standard deviation of main peak amplitude across cycles"
            )

    def _render_calibration_cycle_overlay(self,
                                           selected_cycles: List[int],
                                           valid_cycle_indices: List[int],
                                           aligned_multichannel: Dict[int, np.ndarray],
                                           normalized_multichannel: Dict[int, np.ndarray],
                                           normalize_enabled: bool,
                                           selected_ch: int,
                                           sample_rate: int):
        """
        Render cycle overlay for calibration mode recordings.

        Args:
            selected_cycles: List of cycle indices to display (original indices from all cycles)
            valid_cycle_indices: List of cycle indices that were kept after validation/alignment
            aligned_multichannel: Dict[channel -> aligned cycles array]
            normalized_multichannel: Dict[channel -> normalized cycles array]
            normalize_enabled: Whether normalization is enabled
            selected_ch: Selected channel index
            sample_rate: Sample rate in Hz
        """
        st.markdown("---")
        st.markdown("**Cycle Consistency Overlay**")

        # Display mode selector (like Calibration Impulse panel)
        if normalize_enabled and selected_ch in normalized_multichannel:
            display_mode = st.radio(
                "Display Mode:",
                options=["Aligned (Raw)", "Normalized", "Both (Side-by-Side)"],
                index=1,  # Default to Normalized
                key="series_overlay_display_mode",
                horizontal=True
            )
        else:
            display_mode = "Aligned (Raw)"
            st.caption("Display Mode: Aligned (Raw) - Normalization not enabled")

        # Map selected original cycle indices to array indices
        array_indices = []
        labels = []
        for orig_idx in selected_cycles:
            if orig_idx in valid_cycle_indices:
                array_idx = valid_cycle_indices.index(orig_idx)
                array_indices.append(array_idx)
                labels.append(f"Cycle {orig_idx}")

        if not array_indices:
            st.warning("None of the selected cycles are in the kept cycles")
            return

        # Compute and display cycle overlay statistics
        self._display_cycle_overlay_statistics(
            aligned_multichannel=aligned_multichannel,
            normalized_multichannel=normalized_multichannel,
            normalize_enabled=normalize_enabled,
            selected_ch=selected_ch,
            array_indices=array_indices,
            display_mode=display_mode
        )

        # Render based on display mode
        if display_mode == "Both (Side-by-Side)" and normalize_enabled:
            # Two-column layout
            col1, col2 = st.columns(2)

            with col1:
                st.markdown(f"**Aligned (Raw)** ({len(array_indices)} cycles)")
                aligned_waveforms = [aligned_multichannel[selected_ch][idx] for idx in array_indices]

                if VISUALIZER_AVAILABLE and AudioVisualizer:
                    AudioVisualizer.render_multi_waveform_with_zoom(
                        audio_signals=aligned_waveforms,
                        labels=labels,
                        sample_rate=sample_rate,
                        title="Aligned Cycles",
                        component_id="series_calib_overlay_aligned",
                        height=400,
                        normalize=False,
                        show_analysis=True
                    )

            with col2:
                st.markdown(f"**Normalized** ({len(array_indices)} cycles)")
                normalized_waveforms = [normalized_multichannel[selected_ch][idx] for idx in array_indices]

                if VISUALIZER_AVAILABLE and AudioVisualizer:
                    AudioVisualizer.render_multi_waveform_with_zoom(
                        audio_signals=normalized_waveforms,
                        labels=labels,
                        sample_rate=sample_rate,
                        title="Normalized Cycles",
                        component_id="series_calib_overlay_normalized",
                        height=400,
                        normalize=False,
                        show_analysis=True
                    )

        elif display_mode == "Normalized" and normalize_enabled and selected_ch in normalized_multichannel:
            # Normalized only
            st.markdown(f"**Waveform Overlay - Normalized** ({len(array_indices)} cycle{'s' if len(array_indices) != 1 else ''} selected)")
            st.caption(f"Channel {selected_ch}")
            waveforms = [normalized_multichannel[selected_ch][idx] for idx in array_indices]

            if VISUALIZER_AVAILABLE and AudioVisualizer:
                title = f"Cycle {selected_cycles[0]}" if len(array_indices) == 1 else f"Cycles Overlay ({len(array_indices)} cycles)"
                AudioVisualizer.render_multi_waveform_with_zoom(
                    audio_signals=waveforms,
                    labels=labels,
                    sample_rate=sample_rate,
                    title=title,
                    component_id="series_calib_cycle_overlay",
                    height=400,
                    normalize=False,
                    show_analysis=True
                )

        else:
            # Aligned (Raw) only
            st.markdown(f"**Waveform Overlay - Aligned (Raw)** ({len(array_indices)} cycle{'s' if len(array_indices) != 1 else ''} selected)")
            st.caption(f"Channel {selected_ch}")
            waveforms = [aligned_multichannel[selected_ch][idx] for idx in array_indices]

            if VISUALIZER_AVAILABLE and AudioVisualizer:
                title = f"Cycle {selected_cycles[0]}" if len(array_indices) == 1 else f"Cycles Overlay ({len(array_indices)} cycles)"
                AudioVisualizer.render_multi_waveform_with_zoom(
                    audio_signals=waveforms,
                    labels=labels,
                    sample_rate=sample_rate,
                    title=title,
                    component_id="series_calib_cycle_overlay",
                    height=400,
                    normalize=False,
                    show_analysis=True
                )

    def _render_cycle_consistency_overlay(self, analysis: Dict[str, Any], sample_rate: int) -> None:
        """
        Plot individual cycles overlay with checkbox selection and zoom controls.

        Uses the same component pattern as Calibration Impulse Waveform Analysis.
        For calibration mode, integrates with Cycle Statistics table.
        For standard mode, provides dedicated cycle selection table.
        """
        cycles: List[np.ndarray] = analysis.get('individual_cycles', [])
        if not cycles:
            return

        recording_mode = st.session_state.get('series_recording_mode_used', 'standard')

        st.markdown("---")
        st.markdown("**Cycle Consistency Overlay**")

        # Initialize session state for cycle selection
        session_key = 'series_selected_cycles_for_overlay'
        if session_key not in st.session_state:
            # Default: select first 5 cycles (or all if less than 5)
            st.session_state[session_key] = list(range(min(5, len(cycles))))

        # For standard mode, render cycle selection table
        # For calibration mode, use selection from Cycle Statistics table
        if recording_mode == 'standard':
            st.markdown("Select individual cycles to display in the overlay chart")
            selected_cycles = self._render_cycle_selection_table(cycles)
        else:
            # Calibration mode: get selection from Cycle Statistics table
            selected_cycles = st.session_state.get(session_key, [])
            if not selected_cycles:
                st.info("👆 Check the boxes in the Cycle Statistics table above to select cycles for visualization")
                return

        # Render overlay visualization if cycles are selected
        if selected_cycles:
            st.markdown(f"**Waveform Overlay** ({len(selected_cycles)} cycle{'s' if len(selected_cycles) != 1 else ''} selected)")

            # Display normalization option
            normalize = st.checkbox(
                "Normalize each cycle (max=1)",
                value=False,
                key="series_overlay_normalize"
            )

            self._plot_cycle_overlay(
                selected_cycles=selected_cycles,
                cycle_data=np.array(cycles),
                sample_rate=sample_rate,
                normalize=normalize,
                component_id="series_cycle_overlay"
            )
        else:
            if recording_mode == 'standard':
                st.info("👆 Check the boxes above to select cycles for visualization")

    def _render_cycle_selection_table(self, cycles: List[np.ndarray]) -> List[int]:
        """
        Render checkbox table for cycle selection with metrics.

        Args:
            cycles: List of cycle waveforms

        Returns:
            List of selected cycle indices
        """
        session_key = 'series_selected_cycles_for_overlay'
        selected_cycles = []

        # Column headers
        cols = st.columns([0.4, 0.6, 0.8, 0.8, 0.8, 0.8])
        headers = ["Select", "Cycle #", "Neg Peak", "Pos Peak", "RMS", "Max Abs"]

        for col, header in zip(cols, headers):
            col.markdown(f"**{header}**")

        st.markdown("---")

        # Render rows for each cycle
        for cycle_idx, cycle in enumerate(cycles):
            cols = st.columns([0.4, 0.6, 0.8, 0.8, 0.8, 0.8])

            # Compute metrics for this cycle
            metrics = self._compute_cycle_metrics(cycle)

            with cols[0]:
                is_checked = st.checkbox(
                    "",
                    value=cycle_idx in st.session_state.get(session_key, []),
                    key=f"series_cycle_checkbox_{cycle_idx}",
                    label_visibility="collapsed"
                )
                if is_checked:
                    selected_cycles.append(cycle_idx)

            with cols[1]:
                st.markdown(f"{cycle_idx}")

            with cols[2]:
                st.markdown(f"{metrics['negative_peak']:.3f}")

            with cols[3]:
                st.markdown(f"{metrics['positive_peak']:.3f}")

            with cols[4]:
                st.markdown(f"{metrics['rms']:.3f}")

            with cols[5]:
                st.markdown(f"{metrics['max_abs']:.3f}")

        # Update session state
        st.session_state[session_key] = selected_cycles

        # Display selection info
        if selected_cycles:
            st.success(f"✓ Selected {len(selected_cycles)} cycle(s): {', '.join(map(str, selected_cycles))}")

        return selected_cycles

    def _compute_cycle_metrics(self, cycle: np.ndarray) -> Dict[str, float]:
        """
        Compute metrics for a single cycle.

        Args:
            cycle: Cycle waveform array

        Returns:
            Dict with negative_peak, positive_peak, rms, max_abs
        """
        if len(cycle) == 0:
            return {
                'negative_peak': 0.0,
                'positive_peak': 0.0,
                'rms': 0.0,
                'max_abs': 0.0
            }

        return {
            'negative_peak': float(np.min(cycle)),
            'positive_peak': float(np.max(cycle)),
            'rms': float(np.sqrt(np.mean(cycle ** 2))),
            'max_abs': float(np.max(np.abs(cycle)))
        }

    def _plot_cycle_overlay(self,
                            selected_cycles: List[int],
                            cycle_data: np.ndarray,
                            sample_rate: int,
                            normalize: bool,
                            component_id: str):
        """
        Plot overlay of selected cycles using AudioVisualizer with zoom controls.

        Args:
            selected_cycles: List of cycle indices to plot
            cycle_data: Cycle data array (num_cycles, samples_per_cycle)
            sample_rate: Sample rate in Hz
            normalize: Whether to normalize each cycle
            component_id: Unique component ID for zoom state persistence
        """
        if not VISUALIZER_AVAILABLE or not AudioVisualizer:
            st.warning("AudioVisualizer not available - cannot display overlay")
            return

        # Prepare waveforms for overlay
        waveforms = []
        labels = []

        for cycle_idx in selected_cycles:
            if cycle_idx < len(cycle_data):
                waveforms.append(cycle_data[cycle_idx])
                labels.append(f"Cycle {cycle_idx}")

        if not waveforms:
            st.warning("No valid cycle data to plot")
            return

        # Generate title
        if len(selected_cycles) == 1:
            title = f"Cycle {selected_cycles[0]}"
        else:
            title = f"Cycles Overlay ({len(selected_cycles)} cycles)"

        # Render with zoom controls
        AudioVisualizer.render_multi_waveform_with_zoom(
            audio_signals=waveforms,
            labels=labels,
            sample_rate=sample_rate,
            title=title,
            component_id=component_id,
            height=400,
            normalize=normalize,
            show_analysis=True
        )

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
