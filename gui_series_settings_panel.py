#!/usr/bin/env python3
"""
Series Settings Panel - Multi-pulse Recording Configuration and Analysis

This panel manages the pulse series parameters (number of pulses, cycle timing, intervals)
and provides visualization tools to record and analyze multi-pulse sequences with averaging.

Save this file as: gui_series_settings_panel.py
"""

import numpy as np
import time
import tempfile
import json
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
import streamlit as st

# Import dependencies
try:
    from gui_audio_visualizer import AudioVisualizer
    VISUALIZER_AVAILABLE = True
except ImportError:
    VISUALIZER_AVAILABLE = False
    AudioVisualizer = None

try:
    from RoomResponseRecorder import RoomResponseRecorder
    RECORDER_AVAILABLE = True
except ImportError:
    RECORDER_AVAILABLE = False
    RoomResponseRecorder = None

try:
    import sdl_audio_core as sdl
    SDL_AVAILABLE = True
except ImportError:
    SDL_AVAILABLE = False
    sdl = None


class SeriesSettingsPanel:
    """Panel for configuring and testing multi-pulse recording series."""
    
    def __init__(self, audio_settings_panel=None):
        """
        Initialize the series settings panel.
        
        Args:
            audio_settings_panel: Reference to parent AudioSettingsPanel for device access
        """
        self.audio_settings_panel = audio_settings_panel
        self.component_id = "series_settings"
        
    def render(self):
        """Render the series settings interface."""
        st.header("Series Settings - Multi-pulse Configuration")
        
        # Show prerequisites
        self._show_prerequisites()
        
        if not (SDL_AVAILABLE and RECORDER_AVAILABLE):
            st.error("Required components not available. Please install SDL Audio Core and ensure RoomResponseRecorder is available.")
            return
        
        # Initialize session state
        self._init_session_state()
        
        # Main configuration tabs
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
    
    def _init_session_state(self):
        """Initialize session state for series settings."""
        defaults = {
            # Core series parameters
            'series_num_pulses': 8,
            'series_pulse_duration': 8.0,
            'series_cycle_duration': 100.0,
            'series_pulse_frequency': 1000.0,
            'series_pulse_volume': 0.4,
            'series_pulse_form': 'sine',
            'series_fade_duration': 0.1,
            
            # Analysis parameters
            'series_record_extra_time': 200.0,
            'series_averaging_start_cycle': 2,
            'series_show_individual_cycles': True,
            'series_show_averaged_result': True,
            
            # Recording data
            'series_recorded_audio': None,
            'series_sample_rate': 48000,
            'series_timestamp': 0,
            'series_analysis_data': {},
            
            # Visualization settings
            'series_cycle_overlay_mode': 'all',
            'series_analysis_window_start': 0.0,
            'series_analysis_window_end': 1.0
        }
        
        for key, value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = value
    
    def _show_prerequisites(self):
        """Show status of required components."""
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if SDL_AVAILABLE:
                st.success("✅ SDL Audio")
            else:
                st.error("❌ SDL Audio")
        
        with col2:
            if RECORDER_AVAILABLE:
                st.success("✅ Recorder")
            else:
                st.error("❌ Recorder")
        
        with col3:
            if VISUALIZER_AVAILABLE:
                st.success("✅ Visualizer")
            else:
                st.error("❌ Visualizer")
        
        with col4:
            # Show current device selection from parent panel
            if self.audio_settings_panel:
                input_dev = st.session_state.get('audio_selected_input_device', 'None')
                output_dev = st.session_state.get('audio_selected_output_device', 'None')
                
                if input_dev != 'None' and output_dev != 'None':
                    st.success("✅ Devices")
                else:
                    st.warning("⚠️ Devices")
                    
                st.caption(f"In: {str(input_dev)[:8]}...")
                st.caption(f"Out: {str(output_dev)[:8]}...")
    
    def _render_pulse_series_config(self):
        """Render the main pulse series configuration."""
        st.markdown("**Multi-pulse Series Configuration**")
        
        # Basic pulse parameters
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**Pulse Properties**")
            
            st.session_state['series_num_pulses'] = st.number_input(
                "Number of pulses",
                min_value=1,
                max_value=50,
                value=st.session_state.get('series_num_pulses', 8),
                step=1,
                help="Total number of pulses in the series"
            )
            
            st.session_state['series_pulse_duration'] = st.number_input(
                "Pulse duration (ms)",
                min_value=1.0,
                max_value=100.0,
                value=st.session_state.get('series_pulse_duration', 8.0),
                step=0.5,
                help="Duration of each individual pulse"
            )
            
            st.session_state['series_pulse_frequency'] = st.number_input(
                "Pulse frequency (Hz)",
                min_value=20.0,
                max_value=20000.0,
                value=st.session_state.get('series_pulse_frequency', 1000.0),
                step=50.0,
                help="Frequency content of each pulse"
            )
        
        with col2:
            st.markdown("**Timing & Volume**")
            
            st.session_state['series_cycle_duration'] = st.number_input(
                "Cycle duration (ms)",
                min_value=10.0,
                max_value=1000.0,
                value=st.session_state.get('series_cycle_duration', 100.0),
                step=5.0,
                help="Time between pulse starts (includes pulse + gap)"
            )
            
            st.session_state['series_pulse_volume'] = st.slider(
                "Pulse volume",
                min_value=0.0,
                max_value=1.0,
                value=st.session_state.get('series_pulse_volume', 0.4),
                step=0.05,
                help="Volume level for all pulses"
            )
            
            st.session_state['series_fade_duration'] = st.number_input(
                "Fade duration (ms)",
                min_value=0.1,
                max_value=10.0,
                value=st.session_state.get('series_fade_duration', 0.1),
                step=0.1,
                help="Fade in/out to prevent clicks"
            )
        
        with col3:
            st.markdown("**Waveform & Analysis**")
            
            st.session_state['series_pulse_form'] = st.selectbox(
                "Pulse waveform",
                ["sine", "square"],
                index=0 if st.session_state.get('series_pulse_form', 'sine') == 'sine' else 1,
                help="Waveform shape for each pulse"
            )
            
            st.session_state['series_record_extra_time'] = st.number_input(
                "Extra record time (ms)",
                min_value=50.0,
                max_value=2000.0,
                value=st.session_state.get('series_record_extra_time', 200.0),
                step=25.0,
                help="Additional recording time after last pulse for reverb analysis"
            )
            
            st.session_state['series_averaging_start_cycle'] = st.number_input(
                "Averaging start cycle",
                min_value=1,
                max_value=st.session_state.get('series_num_pulses', 8),
                value=min(st.session_state.get('series_averaging_start_cycle', 2), 
                         st.session_state.get('series_num_pulses', 8)),
                step=1,
                help="First cycle to include in averaging (skip initial cycles for settling)"
            )
        
        # Calculated parameters display
        self._show_calculated_parameters()
        
        # Series preview and control
        st.markdown("---")
        self._render_series_controls()
    
    def _show_calculated_parameters(self):
        """Display calculated timing parameters."""
        st.markdown("**Calculated Parameters**")
        
        pulse_duration = st.session_state.get('series_pulse_duration', 8.0)
        cycle_duration = st.session_state.get('series_cycle_duration', 100.0)
        num_pulses = st.session_state.get('series_num_pulses', 8)
        extra_time = st.session_state.get('series_record_extra_time', 200.0)
        
        gap_duration = cycle_duration - pulse_duration
        total_series_time = (num_pulses * cycle_duration) + extra_time
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Gap Duration", f"{gap_duration:.1f} ms")
        with col2:
            st.metric("Series Duration", f"{total_series_time:.0f} ms")
        with col3:
            duty_cycle = (pulse_duration / cycle_duration) * 100
            st.metric("Duty Cycle", f"{duty_cycle:.1f}%")
        with col4:
            pulse_rate = 1000.0 / cycle_duration  # pulses per second
            st.metric("Pulse Rate", f"{pulse_rate:.1f} Hz")
        
        # Warnings for problematic configurations
        if gap_duration <= 0:
            st.error("⚠️ Gap duration is negative! Increase cycle duration or reduce pulse duration.")
        elif gap_duration < 5.0:
            st.warning("⚠️ Very short gap duration may cause overlapping echoes.")
        
        if duty_cycle > 50:
            st.warning("⚠️ High duty cycle may cause excessive acoustic energy.")
    
    def _render_series_controls(self):
        """Render recording and preview controls for the series."""
        st.markdown("**Series Controls**")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🎵 Record Series", 
                        disabled=not (SDL_AVAILABLE and RECORDER_AVAILABLE),
                        help="Record the complete pulse series and analyze results"):
                self._execute_series_recording()
        
        with col2:
            if st.button("🔊 Preview Series", 
                        disabled=not SDL_AVAILABLE,
                        help="Play the pulse series without recording"):
                self._preview_series()
        
        with col3:
            if st.button("⚙️ Export Series Config",
                        help="Export current series configuration"):
                self._export_series_config()
    
    def _execute_series_recording(self):
        """Execute multi-pulse series recording and analysis."""
        try:
            with st.spinner("Recording pulse series..."):
                # Create TMP folder
                tmp_dir = Path("TMP")
                tmp_dir.mkdir(exist_ok=True)
                
                # Generate unique filenames
                timestamp = int(time.time())
                temp_raw_path = tmp_dir / f"series_raw_{timestamp}.wav"
                temp_impulse_path = tmp_dir / f"series_impulse_{timestamp}.wav"
                
                # Create and configure recorder
                recorder = RoomResponseRecorder()
                
                # Get current audio settings
                sample_rate = st.session_state.get('audio_sample_rate', 48000)
                
                # Configure recorder for series
                recorder.sample_rate = sample_rate
                recorder.pulse_frequency = st.session_state.get('series_pulse_frequency', 1000.0)
                recorder.pulse_duration = st.session_state.get('series_pulse_duration', 8.0) / 1000.0
                recorder.pulse_fade = st.session_state.get('series_fade_duration', 0.1) / 1000.0
                recorder.cycle_duration = st.session_state.get('series_cycle_duration', 100.0) / 1000.0
                recorder.num_pulses = st.session_state.get('series_num_pulses', 8)
                recorder.volume = st.session_state.get('series_pulse_volume', 0.4)
                recorder.impulse_form = st.session_state.get('series_pulse_form', 'sine')
                
                # Add extra recording time
                extra_time = st.session_state.get('series_record_extra_time', 200.0) / 1000.0
                recorder.total_duration = (recorder.num_pulses * recorder.cycle_duration) + extra_time
                
                # Recalculate derived parameters
                recorder.pulse_samples = int(recorder.pulse_duration * recorder.sample_rate)
                recorder.fade_samples = int(recorder.pulse_fade * recorder.sample_rate)
                recorder.cycle_samples = int(recorder.cycle_duration * recorder.sample_rate)
                recorder.gap_samples = recorder.cycle_samples - recorder.pulse_samples
                
                # Regenerate signal with new parameters
                recorder.playback_signal = recorder._generate_complete_signal()
                
                # Execute recording
                recorded_audio = recorder.take_record(
                    output_file=str(temp_raw_path),
                    impulse_file=str(temp_impulse_path),
                    method=2  # Auto device selection
                )
                
                if recorded_audio is not None:
                    # Analyze the series recording
                    analysis_data = self._analyze_series_recording(recorded_audio, recorder)
                    
                    # Store results in session state
                    st.session_state['series_recorded_audio'] = recorded_audio
                    st.session_state['series_sample_rate'] = recorder.sample_rate
                    st.session_state['series_timestamp'] = time.time()
                    st.session_state['series_analysis_data'] = analysis_data
                    
                    st.success(f"Series recording successful! Duration: {len(recorded_audio) / recorder.sample_rate:.3f}s")
                    st.info(f"Files saved: {temp_raw_path.name}, {temp_impulse_path.name}")
                    st.rerun()
                else:
                    st.error("Recording failed - no audio data captured")
                    
        except Exception as e:
            st.error(f"Recording error: {e}")
            with st.expander("Error Details"):
                st.code(str(e))
    
    def _analyze_series_recording(self, audio_data: np.ndarray, recorder: RoomResponseRecorder) -> Dict[str, Any]:
        """Analyze the recorded pulse series."""
        analysis = {}
        
        try:
            # Basic parameters
            num_pulses = recorder.num_pulses
            cycle_samples = recorder.cycle_samples
            sample_rate = recorder.sample_rate
            
            # Expected signal length (may be shorter than actual recording due to extra time)
            expected_samples = num_pulses * cycle_samples
            signal_data = audio_data[:expected_samples] if len(audio_data) >= expected_samples else audio_data
            
            # Extract individual cycles
            cycles = []
            if len(signal_data) >= expected_samples:
                for i in range(num_pulses):
                    start_idx = i * cycle_samples
                    end_idx = start_idx + cycle_samples
                    if end_idx <= len(signal_data):
                        cycles.append(signal_data[start_idx:end_idx])
                    else:
                        # Pad incomplete cycle
                        incomplete_cycle = signal_data[start_idx:]
                        padded_cycle = np.zeros(cycle_samples)
                        padded_cycle[:len(incomplete_cycle)] = incomplete_cycle
                        cycles.append(padded_cycle)
            
            analysis['individual_cycles'] = cycles
            analysis['num_cycles_extracted'] = len(cycles)
            
            # Calculate averaged response
            if cycles:
                start_cycle = st.session_state.get('series_averaging_start_cycle', 2) - 1  # Convert to 0-based
                start_cycle = max(0, min(start_cycle, len(cycles) - 1))
                
                if start_cycle < len(cycles):
                    cycles_to_average = cycles[start_cycle:]
                    averaged_cycle = np.mean(cycles_to_average, axis=0)
                    analysis['averaged_cycle'] = averaged_cycle
                    analysis['cycles_used_for_averaging'] = len(cycles_to_average)
                    analysis['averaging_start_cycle'] = start_cycle + 1  # Convert back to 1-based
                else:
                    analysis['averaged_cycle'] = cycles[0] if cycles else np.array([])
                    analysis['cycles_used_for_averaging'] = 1
                    analysis['averaging_start_cycle'] = 1
            
            # Calculate cycle-to-cycle consistency
            if len(cycles) > 1:
                # RMS differences between consecutive cycles
                rms_diffs = []
                for i in range(1, len(cycles)):
                    diff = cycles[i] - cycles[i-1]
                    rms_diff = np.sqrt(np.mean(diff**2))
                    rms_diffs.append(rms_diff)
                
                analysis['cycle_consistency'] = {
                    'rms_differences': rms_diffs,
                    'mean_rms_diff': np.mean(rms_diffs),
                    'std_rms_diff': np.std(rms_diffs)
                }
            
            # Basic quality metrics for the full recording
            analysis['full_recording_metrics'] = {
                'max_amplitude': float(np.max(np.abs(audio_data))),
                'rms_level': float(np.sqrt(np.mean(audio_data**2))),
                'total_samples': len(audio_data),
                'duration_seconds': len(audio_data) / sample_rate
            }
            
        except Exception as e:
            analysis['error'] = str(e)
            st.error(f"Analysis error: {e}")
        
        return analysis
    
    def _preview_series(self):
        """Preview the pulse series without recording."""
        try:
            with st.spinner("Generating series preview..."):
                # Create temporary recorder for signal generation
                recorder = RoomResponseRecorder()
                
                # Configure with current settings
                sample_rate = st.session_state.get('audio_sample_rate', 48000)
                recorder.sample_rate = sample_rate
                recorder.pulse_frequency = st.session_state.get('series_pulse_frequency', 1000.0)
                recorder.pulse_duration = st.session_state.get('series_pulse_duration', 8.0) / 1000.0
                recorder.pulse_fade = st.session_state.get('series_fade_duration', 0.1) / 1000.0
                recorder.cycle_duration = st.session_state.get('series_cycle_duration', 100.0) / 1000.0
                recorder.num_pulses = st.session_state.get('series_num_pulses', 8)
                recorder.volume = st.session_state.get('series_pulse_volume', 0.4)
                recorder.impulse_form = st.session_state.get('series_pulse_form', 'sine')
                
                # Calculate derived parameters
                recorder.pulse_samples = int(recorder.pulse_duration * recorder.sample_rate)
                recorder.fade_samples = int(recorder.pulse_fade * recorder.sample_rate)
                recorder.cycle_samples = int(recorder.cycle_duration * recorder.sample_rate)
                recorder.gap_samples = recorder.cycle_samples - recorder.pulse_samples
                recorder.total_duration = recorder.cycle_duration * recorder.num_pulses
                
                # Generate signal
                signal = recorder._generate_complete_signal()
                signal_array = np.array(signal, dtype=np.float32)
                
                # Try to play using SDL
                try:
                    input_id, output_id = self._get_device_ids()
                    result = sdl.quick_device_test(input_id, output_id, signal)
                    
                    if result['success']:
                        st.success("Series preview played successfully!")
                        # Store for visualization
                        st.session_state['series_preview_audio'] = signal_array
                        st.session_state['series_preview_sample_rate'] = sample_rate
                    else:
                        st.error(f"Preview failed: {result.get('error_message', 'Unknown error')}")
                        
                except Exception as e:
                    st.error(f"Preview playback error: {e}")
                    # Still store for visualization even if playback failed
                    st.session_state['series_preview_audio'] = signal_array
                    st.session_state['series_preview_sample_rate'] = sample_rate
                    
        except Exception as e:
            st.error(f"Series generation error: {e}")
    
    def _export_series_config(self):
        """Export current series configuration."""
        config = {
            "series_config": {
                "num_pulses": st.session_state.get('series_num_pulses', 8),
                "pulse_duration_ms": st.session_state.get('series_pulse_duration', 8.0),
                "cycle_duration_ms": st.session_state.get('series_cycle_duration', 100.0),
                "pulse_frequency": st.session_state.get('series_pulse_frequency', 1000.0),
                "pulse_volume": st.session_state.get('series_pulse_volume', 0.4),
                "pulse_form": st.session_state.get('series_pulse_form', 'sine'),
                "fade_duration_ms": st.session_state.get('series_fade_duration', 0.1),
                "record_extra_time_ms": st.session_state.get('series_record_extra_time', 200.0),
                "averaging_start_cycle": st.session_state.get('series_averaging_start_cycle', 2)
            },
            "audio_settings": {
                "sample_rate": st.session_state.get('audio_sample_rate', 48000),
                "input_channels": st.session_state.get('audio_input_channels', 1),
                "output_channels": st.session_state.get('audio_output_channels', 2),
                "buffer_size": st.session_state.get('audio_buffer_size', 512)
            },
            "export_timestamp": time.time(),
            "export_version": "1.0"
        }
        
        config_json = json.dumps(config, indent=2)
        
        st.download_button(
            "Download Series Configuration",
            data=config_json,
            file_name=f"series_config_{int(time.time())}.json",
            mime="application/json"
        )
        
        with st.expander("Configuration Preview"):
            st.code(config_json, language='json')
    
    def _get_device_ids(self):
        """Get device IDs for recording/playback."""
        # Try to get device IDs from parent panel
        if self.audio_settings_panel and hasattr(self.audio_settings_panel, '_get_device_id_from_selection'):
            try:
                input_id = self.audio_settings_panel._get_device_id_from_selection('input')
                output_id = self.audio_settings_panel._get_device_id_from_selection('output')
                return input_id, output_id
            except:
                pass
        
        # Fallback to default devices
        return -1, -1
    
    def _render_recording_analysis(self):
        """Render the recording and analysis interface."""
        st.markdown("**Series Recording Analysis**")
        
        # Check for recorded data
        recorded_audio = st.session_state.get('series_recorded_audio')
        sample_rate = st.session_state.get('series_sample_rate', 48000)
        analysis_data = st.session_state.get('series_analysis_data', {})
        
        if recorded_audio is not None and VISUALIZER_AVAILABLE:
            # Show recording metadata
            timestamp = st.session_state.get('series_timestamp', 0)
            if timestamp > 0:
                record_time = time.strftime('%H:%M:%S', time.localtime(timestamp))
                st.caption(f"Recorded at: {record_time}")
            
            # Show analysis metrics
            if analysis_data:
                self._display_analysis_metrics(analysis_data)
            
            # Visualization controls
            self._render_visualization_controls()
            
            # Main visualizer for full recording
            st.markdown("**Full Recording**")
            visualizer = AudioVisualizer("series_full_recording")
            visualizer.render(
                audio_data=recorded_audio,
                sample_rate=sample_rate,
                title="Complete Series Recording",
                show_controls=True,
                show_analysis=True,
                height=400
            )
            
            # Individual cycles visualization
            if analysis_data.get('individual_cycles'):
                self._render_cycle_analysis(analysis_data, sample_rate)
            
            # Averaged result visualization
            if analysis_data.get('averaged_cycle') is not None:
                self._render_averaged_analysis(analysis_data, sample_rate)
        
        elif recorded_audio is None:
            st.info("No series recording available. Use 'Record Series' to capture a multi-pulse sequence.")
            
            # Show preview if available
            preview_audio = st.session_state.get('series_preview_audio')
            preview_sample_rate = st.session_state.get('series_preview_sample_rate', 48000)
            
            if preview_audio is not None and VISUALIZER_AVAILABLE:
                st.markdown("**Series Preview**")
                preview_visualizer = AudioVisualizer("series_preview")
                preview_visualizer.render(
                    audio_data=preview_audio,
                    sample_rate=preview_sample_rate,
                    title="Generated Series Signal",
                    show_controls=True,
                    show_analysis=False,
                    height=300
                )
        
        else:
            st.error("Audio visualizer not available. Please install gui_audio_visualizer.py")
    
    def _display_analysis_metrics(self, analysis_data: Dict[str, Any]):
        """Display analysis metrics from the series recording."""
        st.markdown("**Analysis Results**")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            cycles_extracted = analysis_data.get('num_cycles_extracted', 0)
            st.metric("Cycles Extracted", cycles_extracted)
        
        with col2:
            cycles_averaged = analysis_data.get('cycles_used_for_averaging', 0)
            st.metric("Cycles Averaged", cycles_averaged)
        
        with col3:
            full_metrics = analysis_data.get('full_recording_metrics', {})
            max_amp = full_metrics.get('max_amplitude', 0)
            st.metric("Max Amplitude", f"{max_amp:.4f}")
        
        with col4:
            rms_level = full_metrics.get('rms_level', 0)
            st.metric("RMS Level", f"{rms_level:.4f}")
        
        # Cycle consistency metrics
        consistency = analysis_data.get('cycle_consistency')
        if consistency:
            st.markdown("**Cycle-to-Cycle Consistency**")
            col1, col2 = st.columns(2)
            
            with col1:
                mean_diff = consistency.get('mean_rms_diff', 0)
                st.metric("Mean RMS Difference", f"{mean_diff:.5f}")
            
            with col2:
                std_diff = consistency.get('std_rms_diff', 0)
                st.metric("Std RMS Difference", f"{std_diff:.5f}")
            
            # Consistency assessment
            if mean_diff < 0.001:
                st.success("Excellent cycle consistency")
            elif mean_diff < 0.01:
                st.info("Good cycle consistency")
            elif mean_diff < 0.1:
                st.warning("Moderate cycle consistency")
            else:
                st.error("Poor cycle consistency - check for system instability")
    
    def _render_visualization_controls(self):
        """Render controls for visualization options."""
        with st.expander("Visualization Options"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.session_state['series_show_individual_cycles'] = st.checkbox(
                    "Show individual cycles",
                    value=st.session_state.get('series_show_individual_cycles', True)
                )
                
                st.session_state['series_cycle_overlay_mode'] = st.selectbox(
                    "Cycle overlay mode",
                    ["all", "first_few", "averaged_only"],
                    index=0,
                    help="How to display multiple cycles"
                )
            
            with col2:
                st.session_state['series_show_averaged_result'] = st.checkbox(
                    "Show averaged result",
                    value=st.session_state.get('series_show_averaged_result', True)
                )
                
                # Analysis window (for future use)
                st.session_state['series_analysis_window_start'] = st.slider(
                    "Analysis window start",
                    0.0, 1.0, 
                    st.session_state.get('series_analysis_window_start', 0.0),
                    help="Start of analysis window (fraction of cycle)"
                )
    
    def _render_cycle_analysis(self, analysis_data: Dict[str, Any], sample_rate: int):
        """Render individual cycle analysis."""
        if not st.session_state.get('series_show_individual_cycles', True):
            return
        
        cycles = analysis_data.get('individual_cycles', [])
        if not cycles:
            return
        
        st.markdown("**Individual Cycles Analysis**")
        
        # Cycle selector
        cycle_idx = st.selectbox(
            "Select cycle to analyze",
            range(len(cycles)),
            format_func=lambda x: f"Cycle {x+1}",
            help="Choose individual cycle for detailed analysis"
        )
        
        if 0 <= cycle_idx < len(cycles):
            selected_cycle = cycles[cycle_idx]
            
            # Visualize selected cycle
            cycle_visualizer = AudioVisualizer(f"series_cycle_{cycle_idx}")
            cycle_visualizer.render(
                audio_data=selected_cycle,
                sample_rate=sample_rate,
                title=f"Cycle {cycle_idx+1} - Individual Analysis",
                show_controls=True,
                show_analysis=True,
                height=350
            )
            
            # Cycle comparison if multiple cycles exist
            if len(cycles) > 1:
                st.markdown("**Cycle Overlay Comparison**")
                self._render_cycle_overlay(cycles, sample_rate, analysis_data)
    
    def _render_cycle_overlay(self, cycles: list, sample_rate: int, analysis_data: Dict[str, Any]):
        """Render overlay comparison of multiple cycles."""
        import matplotlib.pyplot as plt
        
        try:
            # Determine which cycles to show based on overlay mode
            overlay_mode = st.session_state.get('series_cycle_overlay_mode', 'all')
            
            if overlay_mode == 'all':
                cycles_to_show = cycles
                labels = [f"Cycle {i+1}" for i in range(len(cycles))]
            elif overlay_mode == 'first_few':
                max_cycles = min(5, len(cycles))
                cycles_to_show = cycles[:max_cycles]
                labels = [f"Cycle {i+1}" for i in range(max_cycles)]
            else:  # averaged_only
                avg_cycle = analysis_data.get('averaged_cycle')
                if avg_cycle is not None:
                    cycles_to_show = [avg_cycle]
                    labels = ["Averaged"]
                else:
                    cycles_to_show = []
                    labels = []
            
            if cycles_to_show:
                fig, ax = plt.subplots(figsize=(12, 6))
                
                # Create time axis for one cycle
                if len(cycles_to_show[0]) > 0:
                    time_axis = np.arange(len(cycles_to_show[0])) / sample_rate * 1000  # Convert to ms
                    
                    # Plot each cycle
                    colors = plt.cm.tab10(np.linspace(0, 1, len(cycles_to_show)))
                    
                    for i, (cycle, label, color) in enumerate(zip(cycles_to_show, labels, colors)):
                        alpha = 0.7 if len(cycles_to_show) > 1 else 1.0
                        ax.plot(time_axis, cycle, label=label, alpha=alpha, color=color, linewidth=1.5)
                    
                    ax.set_xlabel('Time (ms)')
                    ax.set_ylabel('Amplitude')
                    ax.set_title('Cycle Overlay Comparison')
                    ax.grid(True, alpha=0.3)
                    
                    if len(cycles_to_show) > 1:
                        ax.legend()
                    
                    st.pyplot(fig, use_container_width=True)
                    plt.close(fig)
                
                # Statistics about cycle variations
                if len(cycles_to_show) > 1:
                    self._show_cycle_variation_stats(cycles_to_show)
            
        except Exception as e:
            st.error(f"Cycle overlay error: {e}")
    
    def _show_cycle_variation_stats(self, cycles: list):
        """Show statistics about variation between cycles."""
        try:
            # Calculate point-wise statistics across cycles
            cycles_array = np.array(cycles)
            
            # Point-wise mean and std
            mean_cycle = np.mean(cycles_array, axis=0)
            std_cycle = np.std(cycles_array, axis=0)
            
            # Overall variation metrics
            max_std = np.max(std_cycle)
            mean_std = np.mean(std_cycle)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Max Point Variation", f"{max_std:.5f}")
            with col2:
                st.metric("Mean Point Variation", f"{mean_std:.5f}")
            with col3:
                # Coefficient of variation
                mean_amplitude = np.mean(np.abs(mean_cycle))
                if mean_amplitude > 0:
                    cv = mean_std / mean_amplitude * 100
                    st.metric("Coefficient of Variation", f"{cv:.2f}%")
            
        except Exception as e:
            st.error(f"Variation statistics error: {e}")
    
    def _render_averaged_analysis(self, analysis_data: Dict[str, Any], sample_rate: int):
        """Render averaged cycle analysis."""
        if not st.session_state.get('series_show_averaged_result', True):
            return
        
        averaged_cycle = analysis_data.get('averaged_cycle')
        if averaged_cycle is None or len(averaged_cycle) == 0:
            return
        
        st.markdown("**Averaged Cycle Analysis**")
        
        # Show averaging information
        cycles_used = analysis_data.get('cycles_used_for_averaging', 0)
        start_cycle = analysis_data.get('averaging_start_cycle', 1)
        
        st.info(f"Averaged from {cycles_used} cycles (starting from cycle {start_cycle})")
        
        # Visualize averaged cycle
        avg_visualizer = AudioVisualizer("series_averaged_cycle")
        avg_visualizer.render(
            audio_data=averaged_cycle,
            sample_rate=sample_rate,
            title="Averaged Cycle - Final Result",
            show_controls=True,
            show_analysis=True,
            height=400
        )
        
        # Comparison with individual cycles
        if analysis_data.get('individual_cycles'):
            st.markdown("**Averaging Effectiveness**")
            self._analyze_averaging_effectiveness(analysis_data, sample_rate)
    
    def _analyze_averaging_effectiveness(self, analysis_data: Dict[str, Any], sample_rate: int):
        """Analyze how effective the averaging process was."""
        try:
            averaged_cycle = analysis_data.get('averaged_cycle')
            individual_cycles = analysis_data.get('individual_cycles', [])
            start_cycle = analysis_data.get('averaging_start_cycle', 1) - 1  # Convert to 0-based
            
            if averaged_cycle is None or not individual_cycles:
                return
            
            # Compare averaged result with individual cycles used for averaging
            cycles_used = individual_cycles[start_cycle:]
            
            if len(cycles_used) == 0:
                return
            
            # Calculate SNR improvement through averaging
            # Compare noise in individual cycles vs averaged result
            
            # Estimate noise by looking at high-frequency content
            from scipy import signal as scipy_signal
            
            # High-pass filter to estimate noise (>5kHz)
            nyquist = sample_rate / 2
            high_freq = min(5000, nyquist * 0.8)
            sos = scipy_signal.butter(4, high_freq / nyquist, btype='high', output='sos')
            
            # Noise levels in individual cycles
            individual_noise_levels = []
            for cycle in cycles_used:
                try:
                    noise_component = scipy_signal.sosfilt(sos, cycle)
                    noise_rms = np.sqrt(np.mean(noise_component**2))
                    individual_noise_levels.append(noise_rms)
                except:
                    individual_noise_levels.append(0)
            
            # Noise level in averaged cycle
            try:
                avg_noise_component = scipy_signal.sosfilt(sos, averaged_cycle)
                avg_noise_rms = np.sqrt(np.mean(avg_noise_component**2))
            except:
                avg_noise_rms = 0
            
            mean_individual_noise = np.mean(individual_noise_levels)
            
            # SNR improvement calculation
            if avg_noise_rms > 0 and mean_individual_noise > 0:
                snr_improvement_db = 20 * np.log10(mean_individual_noise / avg_noise_rms)
                theoretical_improvement = 10 * np.log10(len(cycles_used))
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("SNR Improvement", f"{snr_improvement_db:.1f} dB")
                with col2:
                    st.metric("Theoretical Max", f"{theoretical_improvement:.1f} dB")
                with col3:
                    efficiency = (snr_improvement_db / theoretical_improvement) * 100 if theoretical_improvement > 0 else 0
                    st.metric("Averaging Efficiency", f"{efficiency:.0f}%")
                
                # Assessment
                if efficiency > 80:
                    st.success("Excellent averaging effectiveness - cycles are highly consistent")
                elif efficiency > 60:
                    st.info("Good averaging effectiveness - some cycle-to-cycle variation")
                elif efficiency > 40:
                    st.warning("Moderate averaging effectiveness - significant cycle variation")
                else:
                    st.error("Poor averaging effectiveness - high inconsistency between cycles")
            
        except Exception as e:
            st.error(f"Averaging analysis error: {e}")
    
    def _render_advanced_settings(self):
        """Render advanced settings and analysis options."""
        st.markdown("**Advanced Series Settings**")
        
        # Analysis settings
        st.markdown("**Analysis Configuration**")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Averaging Settings**")
            
            max_pulses = st.session_state.get('series_num_pulses', 8)
            st.session_state['series_averaging_start_cycle'] = st.number_input(
                "Start averaging from cycle",
                min_value=1,
                max_value=max_pulses,
                value=min(st.session_state.get('series_averaging_start_cycle', 2), max_pulses),
                step=1,
                help="Skip initial cycles that may have system settling effects",
                key="advanced_averaging_start"
            )
            
            # Quality thresholds
            st.markdown("**Quality Thresholds**")
            series_min_snr = st.number_input(
                "Minimum SNR (dB)",
                min_value=0.0,
                max_value=60.0,
                value=15.0,
                step=1.0,
                help="Minimum signal-to-noise ratio for acceptable recording"
            )
            
            series_max_variation = st.slider(
                "Max cycle variation (%)",
                min_value=1.0,
                max_value=50.0,
                value=10.0,
                step=1.0,
                help="Maximum acceptable variation between cycles"
            )
        
        with col2:
            st.markdown("**Export Settings**")
            
            export_individual = st.checkbox(
                "Export individual cycles",
                value=True,
                help="Export each cycle as separate audio file"
            )
            
            export_averaged = st.checkbox(
                "Export averaged result",
                value=True,
                help="Export the averaged cycle result"
            )
            
            export_analysis = st.checkbox(
                "Export analysis data",
                value=True,
                help="Export analysis metrics as JSON"
            )
            
            if st.button("📁 Export Series Data"):
                self._export_series_data(export_individual, export_averaged, export_analysis)
        
        # Configuration import/export
        st.markdown("---")
        st.markdown("**Configuration Management**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            uploaded_config = st.file_uploader(
                "Import Series Config",
                type=['json'],
                help="Load series configuration from file"
            )
            
            if uploaded_config:
                try:
                    config = json.load(uploaded_config)
                    self._import_series_config(config)
                    st.success("Series configuration imported")
                    st.rerun()
                except Exception as e:
                    st.error(f"Import failed: {e}")
        
        with col2:
            if st.button("🔄 Reset to Defaults"):
                self._reset_series_defaults()
                st.success("Reset to default settings")
                st.rerun()
    
    def _export_series_data(self, export_individual: bool, export_averaged: bool, export_analysis: bool):
        """Export series recording data."""
        recorded_audio = st.session_state.get('series_recorded_audio')
        analysis_data = st.session_state.get('series_analysis_data', {})
        sample_rate = st.session_state.get('series_sample_rate', 48000)
        
        if recorded_audio is None:
            st.error("No series recording available to export")
            return
        
        try:
            # Create export data structure
            timestamp = int(time.time())
            
            if export_analysis and analysis_data:
                # Export analysis as JSON
                export_data = {
                    "timestamp": timestamp,
                    "series_config": {
                        "num_pulses": st.session_state.get('series_num_pulses', 8),
                        "pulse_duration_ms": st.session_state.get('series_pulse_duration', 8.0),
                        "cycle_duration_ms": st.session_state.get('series_cycle_duration', 100.0),
                        "pulse_frequency": st.session_state.get('series_pulse_frequency', 1000.0),
                    },
                    "analysis_results": analysis_data
                }
                
                analysis_json = json.dumps(export_data, indent=2, default=str)
                
                st.download_button(
                    "📊 Download Analysis Data",
                    data=analysis_json,
                    file_name=f"series_analysis_{timestamp}.json",
                    mime="application/json"
                )
            
            # For audio exports, we'd need to implement WAV file creation
            # This would require additional audio processing libraries
            if export_individual or export_averaged:
                st.info("Audio export functionality requires additional implementation with WAV file creation")
                
        except Exception as e:
            st.error(f"Export error: {e}")
    
    def _import_series_config(self, config: Dict[str, Any]):
        """Import series configuration from JSON."""
        try:
            series_config = config.get('series_config', {})
            
            # Map config values to session state
            config_mapping = {
                'num_pulses': 'series_num_pulses',
                'pulse_duration_ms': 'series_pulse_duration',
                'cycle_duration_ms': 'series_cycle_duration',
                'pulse_frequency': 'series_pulse_frequency',
                'pulse_volume': 'series_pulse_volume',
                'pulse_form': 'series_pulse_form',
                'fade_duration_ms': 'series_fade_duration',
                'record_extra_time_ms': 'series_record_extra_time',
                'averaging_start_cycle': 'series_averaging_start_cycle'
            }
            
            for config_key, session_key in config_mapping.items():
                if config_key in series_config:
                    st.session_state[session_key] = series_config[config_key]
                    
        except Exception as e:
            raise ValueError(f"Invalid configuration format: {e}")
    
    def _reset_series_defaults(self):
        """Reset series settings to default values."""
        defaults = {
            'series_num_pulses': 8,
            'series_pulse_duration': 8.0,
            'series_cycle_duration': 100.0,
            'series_pulse_frequency': 1000.0,
            'series_pulse_volume': 0.4,
            'series_pulse_form': 'sine',
            'series_fade_duration': 0.1,
            'series_record_extra_time': 200.0,
            'series_averaging_start_cycle': 2,
            'series_show_individual_cycles': True,
            'series_show_averaged_result': True,
            'series_cycle_overlay_mode': 'all'
        }
        
        for key, value in defaults.items():
            st.session_state[key] = value