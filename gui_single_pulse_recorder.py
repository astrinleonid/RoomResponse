#!/usr/bin/env python3
"""
Single Pulse Recorder Component

Separated component for single pulse recording functionality to keep
the main audio settings panel manageable.

Save this file as: gui_single_pulse_recorder.py
"""

import numpy as np
import time
import tempfile
from pathlib import Path
from typing import Optional, Dict, Any
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


class SinglePulseRecorder:
    """Component for single pulse recording and analysis."""
    
    def __init__(self, audio_settings_panel=None):
        """
        Initialize the single pulse recorder.
        
        Args:
            audio_settings_panel: Reference to parent AudioSettingsPanel for device access
        """
        self.audio_settings_panel = audio_settings_panel
        self.component_id = "single_pulse_recorder"
        
    def render(self):
        """Render the single pulse recorder interface."""
        st.subheader("Single Pulse Recording & Analysis")
        
        # Show prerequisites
        self._show_prerequisites()
        
        if not (SDL_AVAILABLE and RECORDER_AVAILABLE):
            st.error("Required components not available. Please install SDL Audio Core and ensure RoomResponseRecorder is available.")
            return
        
        # Initialize session state
        self._init_session_state()
        
        # Pulse configuration
        self._render_pulse_config()
        
        # Recording controls
        self._render_recording_controls()
        
        # Response visualization
        if VISUALIZER_AVAILABLE:
            self._render_response_visualizer()
    
    def _init_session_state(self):
        """Initialize session state for pulse recording."""
        defaults = {
            'pulse_frequency': 1000.0,
            'pulse_duration': 8.0,
            'pulse_volume': 0.4,
            'pulse_form': 'sine',
            'record_duration': 200.0,
            'fade_duration': 0.1,
            'single_pulse_recorded_audio': None,
            'single_pulse_sample_rate': 48000,
            'single_pulse_timestamp': 0,
            'single_pulse_params': {}
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
    
    def _render_pulse_config(self):
        """Render pulse configuration controls."""
        st.markdown("**Pulse Configuration**")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.session_state['pulse_frequency'] = st.number_input(
                "Frequency (Hz)",
                min_value=20.0,
                max_value=20000.0,
                value=st.session_state.get('pulse_frequency', 1000.0),
                step=50.0,
                help="Frequency of the test pulse"
            )
            
            st.session_state['pulse_duration'] = st.number_input(
                "Duration (ms)",
                min_value=1.0,
                max_value=100.0,
                value=st.session_state.get('pulse_duration', 8.0),
                step=0.5,
                help="Duration of the pulse in milliseconds"
            )
        
        with col2:
            st.session_state['pulse_volume'] = st.slider(
                "Volume",
                min_value=0.0,
                max_value=1.0,
                value=st.session_state.get('pulse_volume', 0.4),
                step=0.05,
                help="Volume level for the test pulse"
            )
            
            waveform_options = ["sine", "square", "voice_coil"]
            current_form = st.session_state.get('pulse_form', 'sine')
            if current_form in waveform_options:
                default_idx = waveform_options.index(current_form)
            else:
                default_idx = 0
            st.session_state['pulse_form'] = st.selectbox(
                "Waveform",
                waveform_options,
                index=default_idx,
                help="Waveform shape for the test pulse"
            )
        
        with col3:
            st.session_state['record_duration'] = st.number_input(
                "Record Duration (ms)",
                min_value=50.0,
                max_value=1000.0,
                value=st.session_state.get('record_duration', 200.0),
                step=10.0,
                help="How long to record after the pulse"
            )
            
            st.session_state['fade_duration'] = st.number_input(
                "Fade Duration (ms)",
                min_value=0.1,
                max_value=10.0,
                value=st.session_state.get('fade_duration', 0.1),
                step=0.1,
                help="Fade in/out duration to prevent clicks"
            )
    
    def _render_recording_controls(self):
        """Render recording control buttons."""
        st.markdown("**Recording Controls**")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🎵 Record Pulse Response", 
                        disabled=not (SDL_AVAILABLE and RECORDER_AVAILABLE),
                        help="Record a single pulse and analyze the response"):
                self._execute_recording()
        
        with col2:
            if st.button("🔊 Preview Pulse", 
                        disabled=not SDL_AVAILABLE,
                        help="Play the pulse signal without recording"):
                self._preview_pulse()
        
        with col3:
            if st.button("⚙️ Export Config",
                        help="Export recorder configuration"):
                self._export_config()
    
    def _execute_recording(self):
        """Execute single pulse recording."""
        try:
            with st.spinner("Recording single pulse response..."):
                # Create TMP folder
                tmp_dir = Path("TMP")
                tmp_dir.mkdir(exist_ok=True)
                
                # Generate unique filenames
                timestamp = int(time.time())
                temp_raw_path = tmp_dir / f"single_pulse_raw_{timestamp}.wav"
                temp_impulse_path = tmp_dir / f"single_pulse_impulse_{timestamp}.wav"
                
                # Create and configure recorder
                recorder = RoomResponseRecorder()
                
                # Get current audio settings from session state
                sample_rate = st.session_state.get('audio_sample_rate', 48000)
                
                # Override recorder parameters
                recorder.sample_rate = sample_rate
                recorder.pulse_frequency = st.session_state.get('pulse_frequency', 1000.0)
                recorder.pulse_duration = st.session_state.get('pulse_duration', 8.0) / 1000.0
                recorder.pulse_fade = st.session_state.get('fade_duration', 0.1) / 1000.0
                recorder.volume = st.session_state.get('pulse_volume', 0.4)
                recorder.impulse_form = st.session_state.get('pulse_form', 'sine')
                
                # Configure for single pulse
                recorder.num_pulses = 1
                record_duration = st.session_state.get('record_duration', 200.0) / 1000.0
                recorder.cycle_duration = max(recorder.pulse_duration + record_duration, 0.1)
                
                # Recalculate derived parameters
                recorder.pulse_samples = int(recorder.pulse_duration * recorder.sample_rate)
                recorder.fade_samples = int(recorder.pulse_fade * recorder.sample_rate)
                recorder.cycle_samples = int(recorder.cycle_duration * recorder.sample_rate)
                recorder.gap_samples = recorder.cycle_samples - recorder.pulse_samples
                recorder.total_duration = recorder.cycle_duration * recorder.num_pulses
                
                # Regenerate signal with new parameters
                recorder.playback_signal = recorder._generate_complete_signal()
                
                # Execute recording
                recorded_audio = recorder.take_record(
                    output_file=str(temp_raw_path),
                    impulse_file=str(temp_impulse_path),
                    method=2  # Auto device selection
                )
                
                if recorded_audio is not None:
                    # Store results in session state
                    st.session_state['single_pulse_recorded_audio'] = recorded_audio
                    st.session_state['single_pulse_sample_rate'] = recorder.sample_rate
                    st.session_state['single_pulse_timestamp'] = time.time()
                    st.session_state['single_pulse_params'] = {
                        'frequency': recorder.pulse_frequency,
                        'duration_ms': recorder.pulse_duration * 1000,
                        'volume': recorder.volume,
                        'form': recorder.impulse_form,
                        'record_duration_ms': record_duration * 1000
                    }
                    
                    st.success(f"Recording successful! Duration: {len(recorded_audio) / recorder.sample_rate:.3f}s")
                    st.info(f"Files saved: {temp_raw_path.name}, {temp_impulse_path.name}")
                    st.rerun()
                else:
                    st.error("Recording failed - no audio data captured")
                    
        except Exception as e:
            st.error(f"Recording error: {e}")
            with st.expander("Error Details"):
                st.code(str(e))
    
    def _preview_pulse(self):
        """Preview the pulse signal without recording."""
        try:
            with st.spinner("Generating pulse preview..."):
                # Generate pulse signal
                sample_rate = st.session_state.get('audio_sample_rate', 48000)
                pulse_frequency = st.session_state.get('pulse_frequency', 1000.0)
                pulse_duration = st.session_state.get('pulse_duration', 8.0) / 1000.0
                pulse_volume = st.session_state.get('pulse_volume', 0.4)
                pulse_form = st.session_state.get('pulse_form', 'sine')
                fade_duration = st.session_state.get('fade_duration', 0.1) / 1000.0
                
                # Create pulse
                pulse_samples = int(pulse_duration * sample_rate)
                fade_samples = int(fade_duration * sample_rate)
                
                t = np.linspace(0, pulse_duration, pulse_samples, endpoint=False)
                
                if pulse_form == "sine":
                    pulse = np.sin(2 * np.pi * pulse_frequency * t)
                else:
                    pulse = np.ones(pulse_samples)
                
                # Apply fade envelope
                if fade_samples > 0:
                    fade_in = np.linspace(0, 1, fade_samples)
                    fade_out = np.linspace(1, 0, fade_samples)
                    pulse[:fade_samples] *= fade_in
                    pulse[-fade_samples:] *= fade_out
                
                pulse *= pulse_volume
                
                # Try to play using SDL
                try:
                    # Get device IDs
                    input_id, output_id = self._get_device_ids()
                    
                    result = sdl.quick_device_test(input_id, output_id, pulse.tolist())
                    
                    if result['success']:
                        st.success("Pulse preview played successfully!")
                        # Store for visualization
                        st.session_state['pulse_preview_audio'] = pulse
                        st.session_state['pulse_preview_sample_rate'] = sample_rate
                    else:
                        st.error(f"Preview failed: {result.get('error_message', 'Unknown error')}")
                        
                except Exception as e:
                    st.error(f"Preview error: {e}")
                    
        except Exception as e:
            st.error(f"Signal generation error: {e}")
    
    def _export_config(self):
        """Export current configuration as JSON."""
        config = {
            "single_pulse_config": {
                "sample_rate": st.session_state.get('audio_sample_rate', 48000),
                "pulse_frequency": st.session_state.get('pulse_frequency', 1000.0),
                "pulse_duration_ms": st.session_state.get('pulse_duration', 8.0),
                "pulse_volume": st.session_state.get('pulse_volume', 0.4),
                "pulse_form": st.session_state.get('pulse_form', 'sine'),
                "record_duration_ms": st.session_state.get('record_duration', 200.0),
                "fade_duration_ms": st.session_state.get('fade_duration', 0.1)
            },
            "audio_settings": {
                "input_channels": st.session_state.get('audio_input_channels', 1),
                "output_channels": st.session_state.get('audio_output_channels', 2),
                "buffer_size": st.session_state.get('audio_buffer_size', 512),
                "selected_input": st.session_state.get('audio_selected_input_device', 'System Default'),
                "selected_output": st.session_state.get('audio_selected_output_device', 'System Default')
            }
        }
        
        import json
        config_json = json.dumps(config, indent=2)
        
        st.download_button(
            "Download Configuration",
            data=config_json,
            file_name=f"single_pulse_config_{int(time.time())}.json",
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
    
    def _render_response_visualizer(self):
        """Render the response visualizer."""
        st.markdown("---")
        st.markdown("**Recorded Response Analysis**")
        
        # Check for recorded audio
        recorded_audio = st.session_state.get('single_pulse_recorded_audio')
        sample_rate = st.session_state.get('single_pulse_sample_rate', 48000)
        pulse_params = st.session_state.get('single_pulse_params', {})
        
        if recorded_audio is not None:
            # Show recording metadata
            timestamp = st.session_state.get('single_pulse_timestamp', 0)
            if timestamp > 0:
                record_time = time.strftime('%H:%M:%S', time.localtime(timestamp))
                st.caption(f"Recorded at: {record_time}")
            
            # Show pulse parameters
            if pulse_params:
                param_col1, param_col2, param_col3, param_col4 = st.columns(4)
                with param_col1:
                    st.metric("Frequency", f"{pulse_params.get('frequency', 0):.0f} Hz")
                with param_col2:
                    st.metric("Duration", f"{pulse_params.get('duration_ms', 0):.1f} ms")
                with param_col3:
                    st.metric("Volume", f"{pulse_params.get('volume', 0):.2f}")
                with param_col4:
                    st.metric("Form", pulse_params.get('form', 'unknown'))
            
            # Render visualizer
            visualizer = AudioVisualizer("single_pulse_response")
            visualizer_result = visualizer.render(
                audio_data=recorded_audio,
                sample_rate=sample_rate,
                title="Single Pulse Response",
                show_controls=True,
                show_analysis=True,
                height=400
            )
            
            # Additional pulse-specific analysis
            if visualizer_result.get("status") == "ready":
                self._render_pulse_analysis(recorded_audio, sample_rate, pulse_params)
        
        else:
            st.info("No recorded response available. Use 'Record Pulse Response' to capture audio.")
            
            # Show preview if available
            preview_audio = st.session_state.get('pulse_preview_audio')
            preview_sample_rate = st.session_state.get('pulse_preview_sample_rate', 48000)
            
            if preview_audio is not None:
                st.markdown("**Pulse Preview**")
                preview_visualizer = AudioVisualizer("pulse_preview")
                preview_visualizer.render(
                    audio_data=preview_audio,
                    sample_rate=preview_sample_rate,
                    title="Generated Pulse Signal",
                    show_controls=True,
                    show_analysis=False,
                    height=300
                )
    
    def _render_pulse_analysis(self, audio_data: np.ndarray, sample_rate: int, pulse_params: Dict):
        """Render pulse-specific acoustic analysis."""
        st.markdown("**Pulse Response Analysis**")
        
        try:
            # Find pulse onset
            onset_idx = self._find_pulse_onset(audio_data)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Timing Analysis**")
                
                onset_time_ms = (onset_idx / sample_rate) * 1000
                st.metric("Pulse Onset", f"{onset_time_ms:.1f} ms")
                
                # Response duration analysis
                if onset_idx < len(audio_data):
                    response_segment = audio_data[onset_idx:]
                    peak_value = np.max(np.abs(response_segment))
                    
                    if peak_value > 0:
                        # Time to decay to 10% of peak
                        decay_threshold = peak_value * 0.1
                        decay_indices = np.where(np.abs(response_segment) < decay_threshold)[0]
                        
                        if len(decay_indices) > 0:
                            decay_idx = decay_indices[0]
                            decay_time_ms = (decay_idx / sample_rate) * 1000
                            st.metric("Response Duration", f"{decay_time_ms:.1f} ms")
                            
                            # RT60 estimate
                            rt60_estimate = decay_time_ms * 6
                            st.metric("RT60 Estimate", f"{rt60_estimate:.0f} ms")
            
            with col2:
                st.markdown("**Quality Metrics**")
                
                # Signal to noise ratio
                if onset_idx > 100:
                    noise_floor = np.std(audio_data[:onset_idx])
                    signal_rms = np.sqrt(np.mean(audio_data[onset_idx:]**2))
                    
                    if noise_floor > 0:
                        snr_db = 20 * np.log10(signal_rms / noise_floor)
                        st.metric("SNR Estimate", f"{snr_db:.1f} dB")
                
                # Peak response
                peak_response = np.max(np.abs(audio_data))
                st.metric("Peak Response", f"{peak_response:.4f}")
                
                # Clarity calculation (C50)
                if onset_idx < len(audio_data) - sample_rate // 10:
                    direct_window = int(sample_rate * 0.05)  # 50ms
                    reverb_window = int(sample_rate * 0.1)   # 100ms
                    
                    direct_end = min(onset_idx + direct_window, len(audio_data))
                    reverb_end = min(onset_idx + direct_window + reverb_window, len(audio_data))
                    
                    direct_energy = np.sum(audio_data[onset_idx:direct_end]**2)
                    reverb_energy = np.sum(audio_data[direct_end:reverb_end]**2)
                    
                    if reverb_energy > 0:
                        clarity_db = 10 * np.log10(direct_energy / reverb_energy)
                        st.metric("C50 (Clarity)", f"{clarity_db:.1f} dB")
            
            # Response envelope visualization
            self._plot_response_envelope(audio_data, sample_rate, onset_idx)
            
        except Exception as e:
            st.error(f"Analysis error: {e}")
    
    def _find_pulse_onset(self, audio_data: np.ndarray, threshold_factor: float = 3.0) -> int:
        """Find the onset of the pulse in recorded audio."""
        window_size = max(10, len(audio_data) // 100)
        
        rms_values = []
        for i in range(len(audio_data) - window_size):
            rms = np.sqrt(np.mean(audio_data[i:i+window_size]**2))
            rms_values.append(rms)
        
        rms_values = np.array(rms_values)
        
        # Estimate background noise from first 10%
        background_samples = len(rms_values) // 10
        if background_samples > 0:
            background_level = np.mean(rms_values[:background_samples])
            threshold = background_level * threshold_factor
            
            onset_candidates = np.where(rms_values > threshold)[0]
            if len(onset_candidates) > 0:
                return onset_candidates[0]
        
        # Fallback to maximum amplitude
        return np.argmax(np.abs(audio_data))
    
    def _plot_response_envelope(self, audio_data: np.ndarray, sample_rate: int, onset_idx: int):
        """Plot the response envelope."""
        st.markdown("**Response Envelope**")
        
        try:
            import matplotlib.pyplot as plt
            
            # Calculate envelope
            window_size = sample_rate // 100  # 10ms windows
            envelope = []
            for i in range(0, len(audio_data) - window_size, window_size // 4):
                window = audio_data[i:i+window_size]
                rms = np.sqrt(np.mean(window**2))
                envelope.append(rms)
            
            envelope = np.array(envelope)
            time_axis = np.arange(len(envelope)) * (window_size // 4) / sample_rate * 1000  # ms
            
            # Create plot
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(time_axis, envelope, 'r-', linewidth=2, label='Response Envelope')
            
            # Mark onset
            onset_time_ms = (onset_idx / sample_rate) * 1000
            ax.axvline(x=onset_time_ms, color='g', linestyle='--', alpha=0.7, label='Pulse Onset')
            
            ax.set_xlabel('Time (ms)')
            ax.set_ylabel('RMS Amplitude')
            ax.set_title('Pulse Response Envelope')
            ax.grid(True, alpha=0.3)
            ax.legend()
            
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
            
        except Exception as e:
            st.error(f"Envelope plot error: {e}")