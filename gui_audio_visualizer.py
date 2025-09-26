#!/usr/bin/env python3
"""
Audio Visualizer Component - Reusable audio waveform display and playback

This component provides:
- Waveform visualization using matplotlib
- Audio playback controls
- Spectral analysis display
- Export functionality
- Multiple view modes (waveform, spectrum, spectrogram)

Save this file as: gui_audio_visualizer.py
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.figure import Figure
import streamlit as st
import io
import wave
import base64
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, Union
import tempfile
import os


class AudioVisualizer:
    """Reusable audio visualization and playback component."""
    
    def __init__(self, component_id: str = "audio_viz"):
        """
        Initialize the audio visualizer.
        
        Args:
            component_id: Unique identifier for this visualizer instance
        """
        self.component_id = component_id
        self.session_key_prefix = f"audio_viz_{component_id}"
        
    def render(self, 
               audio_data: Optional[np.ndarray] = None,
               sample_rate: int = 48000,
               title: str = "Audio Visualization",
               show_controls: bool = True,
               show_analysis: bool = True,
               height: int = 400) -> Dict[str, Any]:
        """
        Render the audio visualizer component.
        
        Args:
            audio_data: Audio samples as numpy array
            sample_rate: Sample rate in Hz
            title: Title for the visualizer
            show_controls: Whether to show playback controls
            show_analysis: Whether to show analysis panels
            height: Height of the visualization in pixels
            
        Returns:
            Dict containing user interactions and component state
        """
        # Initialize session state for this component
        self._init_session_state()
        
        # Container for the entire component
        container = st.container()
        
        with container:
            st.subheader(title)
            
            # Handle audio data updates
            if audio_data is not None:
                self._update_audio_data(audio_data, sample_rate)
            
            # Get current audio data from session state
            current_audio = self._get_current_audio_data()
            current_sample_rate = self._get_current_sample_rate()
            
            if current_audio is None:
                st.info("No audio data loaded")
                return {"status": "no_data"}
            
            # Main visualization tabs
            if show_analysis:
                viz_tab, analysis_tab, export_tab = st.tabs([
                    "Waveform", "Analysis", "Export"
                ])
            else:
                viz_tab = st.container()
            
            with viz_tab:
                result = self._render_waveform_tab(
                    current_audio, current_sample_rate, height
                )
            
            if show_analysis:
                with analysis_tab:
                    self._render_analysis_tab(current_audio, current_sample_rate)
                
                with export_tab:
                    self._render_export_tab(current_audio, current_sample_rate)
            
            # Playback controls
            if show_controls:
                self._render_playback_controls(current_audio, current_sample_rate)
            
            return {
                "status": "ready",
                "audio_length": len(current_audio),
                "sample_rate": current_sample_rate,
                "duration": len(current_audio) / current_sample_rate,
                **result
            }
    
    def _init_session_state(self):
        """Initialize session state variables for this component."""
        defaults = {
            f"{self.session_key_prefix}_audio_data": None,
            f"{self.session_key_prefix}_sample_rate": 48000,
            f"{self.session_key_prefix}_view_mode": "waveform",
            f"{self.session_key_prefix}_zoom_start": 0.0,
            f"{self.session_key_prefix}_zoom_end": 1.0,
            f"{self.session_key_prefix}_show_markers": True,
        }
        
        for key, value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = value
    
    def _update_audio_data(self, audio_data: np.ndarray, sample_rate: int):
        """Update the audio data in session state."""
        st.session_state[f"{self.session_key_prefix}_audio_data"] = audio_data.copy()
        st.session_state[f"{self.session_key_prefix}_sample_rate"] = sample_rate
        # Reset zoom when new audio is loaded
        st.session_state[f"{self.session_key_prefix}_zoom_start"] = 0.0
        st.session_state[f"{self.session_key_prefix}_zoom_end"] = 1.0
    
    def _get_current_audio_data(self) -> Optional[np.ndarray]:
        """Get current audio data from session state."""
        return st.session_state.get(f"{self.session_key_prefix}_audio_data")
    
    def _get_current_sample_rate(self) -> int:
        """Get current sample rate from session state."""
        return st.session_state.get(f"{self.session_key_prefix}_sample_rate", 48000)
    
    def _render_waveform_tab(self, audio_data: np.ndarray, sample_rate: int, height: int) -> Dict[str, Any]:
        """Render the main waveform visualization."""
        # View mode selector
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            view_mode = st.selectbox(
                "View Mode",
                ["waveform", "spectrum", "spectrogram"],
                index=["waveform", "spectrum", "spectrogram"].index(
                    st.session_state.get(f"{self.session_key_prefix}_view_mode", "waveform")
                ),
                key=f"{self.session_key_prefix}_view_selector"
            )
            st.session_state[f"{self.session_key_prefix}_view_mode"] = view_mode
        
        with col2:
            show_markers = st.checkbox(
                "Show Markers",
                value=st.session_state.get(f"{self.session_key_prefix}_show_markers", True),
                key=f"{self.session_key_prefix}_markers_checkbox"
            )
            st.session_state[f"{self.session_key_prefix}_show_markers"] = show_markers
        
        with col3:
            if st.button("Reset Zoom", key=f"{self.session_key_prefix}_reset_zoom"):
                st.session_state[f"{self.session_key_prefix}_zoom_start"] = 0.0
                st.session_state[f"{self.session_key_prefix}_zoom_end"] = 1.0
                st.rerun()
        
        # Zoom controls
        zoom_start = st.session_state.get(f"{self.session_key_prefix}_zoom_start", 0.0)
        zoom_end = st.session_state.get(f"{self.session_key_prefix}_zoom_end", 1.0)
        
        duration = len(audio_data) / sample_rate
        
        with st.expander("Zoom Controls"):
            col1, col2 = st.columns(2)
            with col1:
                zoom_start = st.slider(
                    "Start Time (s)",
                    0.0, duration,
                    zoom_start * duration,
                    key=f"{self.session_key_prefix}_zoom_start_slider"
                ) / duration
            with col2:
                zoom_end = st.slider(
                    "End Time (s)",
                    0.0, duration,
                    zoom_end * duration,
                    key=f"{self.session_key_prefix}_zoom_end_slider"
                ) / duration
        
        st.session_state[f"{self.session_key_prefix}_zoom_start"] = zoom_start
        st.session_state[f"{self.session_key_prefix}_zoom_end"] = zoom_end
        
        # Generate the plot based on view mode
        fig = self._create_plot(audio_data, sample_rate, view_mode, zoom_start, zoom_end, show_markers)
        
        # Display the plot
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)  # Prevent memory leaks
        
        # Audio statistics
        self._show_audio_stats(audio_data, sample_rate, zoom_start, zoom_end)
        
        return {
            "view_mode": view_mode,
            "zoom_start": zoom_start,
            "zoom_end": zoom_end
        }
    
    def _create_plot(self, audio_data: np.ndarray, sample_rate: int, view_mode: str, 
                    zoom_start: float, zoom_end: float, show_markers: bool) -> Figure:
        """Create the matplotlib plot based on view mode."""
        # Calculate zoom indices
        start_idx = int(zoom_start * len(audio_data))
        end_idx = int(zoom_end * len(audio_data))
        zoomed_audio = audio_data[start_idx:end_idx]
        
        # Create time axis
        time_axis = np.linspace(zoom_start * len(audio_data) / sample_rate,
                               zoom_end * len(audio_data) / sample_rate,
                               len(zoomed_audio))
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 6))
        fig.patch.set_facecolor('white')
        
        if view_mode == "waveform":
            self._plot_waveform(ax, time_axis, zoomed_audio, show_markers)
        elif view_mode == "spectrum":
            self._plot_spectrum(ax, zoomed_audio, sample_rate)
        elif view_mode == "spectrogram":
            self._plot_spectrogram(ax, zoomed_audio, sample_rate, time_axis)
        
        plt.tight_layout()
        return fig
    
    def _plot_waveform(self, ax, time_axis: np.ndarray, audio_data: np.ndarray, show_markers: bool):
        """Plot the waveform."""
        ax.plot(time_axis, audio_data, 'b-', linewidth=1, alpha=0.8)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Amplitude')
        ax.set_title('Audio Waveform')
        ax.grid(True, alpha=0.3)
        
        # Add markers for peaks if requested
        if show_markers and len(audio_data) > 0:
            # Find peaks
            peak_threshold = np.max(np.abs(audio_data)) * 0.5
            peaks = np.where(np.abs(audio_data) > peak_threshold)[0]
            
            if len(peaks) > 0:
                ax.scatter(time_axis[peaks], audio_data[peaks], 
                          color='red', s=20, alpha=0.6, label='Peaks')
                ax.legend()
        
        # Add zero line
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    
    def _plot_spectrum(self, ax, audio_data: np.ndarray, sample_rate: int):
        """Plot the frequency spectrum."""
        # Calculate FFT
        fft = np.fft.fft(audio_data)
        freqs = np.fft.fftfreq(len(fft), 1/sample_rate)
        
        # Take only positive frequencies
        pos_mask = freqs >= 0
        freqs = freqs[pos_mask]
        magnitude = np.abs(fft[pos_mask])
        
        # Convert to dB
        magnitude_db = 20 * np.log10(magnitude + 1e-10)
        
        ax.plot(freqs, magnitude_db, 'b-', linewidth=1)
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Magnitude (dB)')
        ax.set_title('Frequency Spectrum')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, sample_rate/2)
    
    def _plot_spectrogram(self, ax, audio_data: np.ndarray, sample_rate: int, time_axis: np.ndarray):
        """Plot the spectrogram."""
        # Calculate spectrogram
        window_size = min(512, len(audio_data) // 4)
        overlap = window_size // 2
        
        f, t, Sxx = self._compute_spectrogram(audio_data, sample_rate, window_size, overlap)
        
        # Convert to dB
        Sxx_db = 10 * np.log10(Sxx + 1e-10)
        
        # Adjust time axis to match zoom
        t_adjusted = t + time_axis[0]
        
        im = ax.pcolormesh(t_adjusted, f, Sxx_db, shading='gouraud', cmap='viridis')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Frequency (Hz)')
        ax.set_title('Spectrogram')
        
        # Add colorbar
        plt.colorbar(im, ax=ax, label='Power (dB)')
    
    def _compute_spectrogram(self, audio_data: np.ndarray, sample_rate: int, 
                           window_size: int, overlap: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute spectrogram using sliding window FFT."""
        step = window_size - overlap
        n_windows = (len(audio_data) - overlap) // step
        
        freqs = np.fft.fftfreq(window_size, 1/sample_rate)[:window_size//2]
        times = np.arange(n_windows) * step / sample_rate
        
        spectrogram = np.zeros((len(freqs), n_windows))
        
        for i in range(n_windows):
            start = i * step
            end = start + window_size
            if end <= len(audio_data):
                window = audio_data[start:end] * np.hanning(window_size)
                fft = np.fft.fft(window)
                spectrogram[:, i] = np.abs(fft[:len(freqs)])**2
        
        return freqs, times, spectrogram
    
    def _show_audio_stats(self, audio_data: np.ndarray, sample_rate: int, 
                         zoom_start: float, zoom_end: float):
        """Show audio statistics."""
        # Calculate stats for zoomed region
        start_idx = int(zoom_start * len(audio_data))
        end_idx = int(zoom_end * len(audio_data))
        zoomed_audio = audio_data[start_idx:end_idx]
        
        duration = len(zoomed_audio) / sample_rate
        max_amp = np.max(np.abs(zoomed_audio))
        rms = np.sqrt(np.mean(zoomed_audio**2))
        
        # Display stats in columns
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Duration", f"{duration:.3f} s")
        with col2:
            st.metric("Max Amplitude", f"{max_amp:.4f}")
        with col3:
            st.metric("RMS Level", f"{rms:.4f}")
        with col4:
            st.metric("Samples", f"{len(zoomed_audio):,}")
    
    def _render_analysis_tab(self, audio_data: np.ndarray, sample_rate: int):
        """Render the analysis tab with detailed audio analysis."""
        st.markdown("**Detailed Audio Analysis**")
        
        # Signal characteristics
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Signal Characteristics**")
            
            # Basic stats
            max_val = np.max(audio_data)
            min_val = np.min(audio_data)
            mean_val = np.mean(audio_data)
            std_val = np.std(audio_data)
            rms_val = np.sqrt(np.mean(audio_data**2))
            
            st.write(f"**Maximum:** {max_val:.6f}")
            st.write(f"**Minimum:** {min_val:.6f}")
            st.write(f"**Mean:** {mean_val:.6f}")
            st.write(f"**Std Dev:** {std_val:.6f}")
            st.write(f"**RMS:** {rms_val:.6f}")
            
            # Dynamic range
            if min_val != 0:
                dynamic_range = 20 * np.log10(max_val / abs(min_val))
                st.write(f"**Dynamic Range:** {dynamic_range:.1f} dB")
        
        with col2:
            st.markdown("**Frequency Analysis**")
            
            # Find dominant frequency
            fft = np.fft.fft(audio_data)
            freqs = np.fft.fftfreq(len(fft), 1/sample_rate)
            magnitude = np.abs(fft)
            
            # Find peak frequency (exclude DC component)
            pos_freqs = freqs[1:len(freqs)//2]
            pos_magnitude = magnitude[1:len(magnitude)//2]
            
            if len(pos_magnitude) > 0:
                peak_idx = np.argmax(pos_magnitude)
                dominant_freq = pos_freqs[peak_idx]
                st.write(f"**Dominant Frequency:** {dominant_freq:.1f} Hz")
                
                # Spectral centroid
                spectral_centroid = np.sum(pos_freqs * pos_magnitude) / np.sum(pos_magnitude)
                st.write(f"**Spectral Centroid:** {spectral_centroid:.1f} Hz")
        
        # Signal quality assessment
        st.markdown("---")
        st.markdown("**Signal Quality Assessment**")
        
        # Check for clipping
        clipping_threshold = 0.95
        clipped_samples = np.sum(np.abs(audio_data) > clipping_threshold)
        clipping_percentage = (clipped_samples / len(audio_data)) * 100
        
        if clipping_percentage > 0.1:
            st.error(f"⚠️ Clipping detected: {clipping_percentage:.2f}% of samples")
        else:
            st.success("✅ No significant clipping detected")
        
        # Signal-to-noise ratio estimate
        if rms_val > 0:
            # Simple noise floor estimation (bottom 10% of signal)
            sorted_abs = np.sort(np.abs(audio_data))
            noise_floor = np.mean(sorted_abs[:len(sorted_abs)//10])
            snr_estimate = 20 * np.log10(rms_val / (noise_floor + 1e-10))
            
            st.info(f"Estimated SNR: {snr_estimate:.1f} dB")
    
    def _render_export_tab(self, audio_data: np.ndarray, sample_rate: int):
        """Render the export tab."""
        st.markdown("**Export Options**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Export Audio**")
            
            # Export current view
            if st.button("Export Current View", key=f"{self.session_key_prefix}_export_view"):
                zoom_start = st.session_state.get(f"{self.session_key_prefix}_zoom_start", 0.0)
                zoom_end = st.session_state.get(f"{self.session_key_prefix}_zoom_end", 1.0)
                
                start_idx = int(zoom_start * len(audio_data))
                end_idx = int(zoom_end * len(audio_data))
                zoomed_audio = audio_data[start_idx:end_idx]
                
                wav_data = self._create_wav_download(zoomed_audio, sample_rate)
                st.download_button(
                    "Download Current View",
                    data=wav_data,
                    file_name=f"audio_export_view_{int(zoom_start*1000):04d}-{int(zoom_end*1000):04d}ms.wav",
                    mime="audio/wav",
                    key=f"{self.session_key_prefix}_download_view"
                )
            
            # Export full audio
            if st.button("Export Full Audio", key=f"{self.session_key_prefix}_export_full"):
                wav_data = self._create_wav_download(audio_data, sample_rate)
                st.download_button(
                    "Download Full Audio",
                    data=wav_data,
                    file_name="audio_export_full.wav",
                    mime="audio/wav",
                    key=f"{self.session_key_prefix}_download_full"
                )
        
        with col2:
            st.markdown("**Export Data**")
            
            # Export as CSV
            if st.button("Export as CSV", key=f"{self.session_key_prefix}_export_csv"):
                csv_data = self._create_csv_download(audio_data, sample_rate)
                st.download_button(
                    "Download CSV",
                    data=csv_data,
                    file_name="audio_data.csv",
                    mime="text/csv",
                    key=f"{self.session_key_prefix}_download_csv"
                )
            
            # Export metadata
            if st.button("Export Metadata", key=f"{self.session_key_prefix}_export_meta"):
                metadata = self._create_metadata_json(audio_data, sample_rate)
                st.download_button(
                    "Download Metadata",
                    data=metadata,
                    file_name="audio_metadata.json",
                    mime="application/json",
                    key=f"{self.session_key_prefix}_download_meta"
                )
    
    def _render_playback_controls(self, audio_data: np.ndarray, sample_rate: int):
        """Render playback controls."""
        st.markdown("---")
        st.markdown("**Playback Controls**")
        
        # Create audio player using HTML audio element
        wav_data = self._create_wav_download(audio_data, sample_rate)
        b64_audio = base64.b64encode(wav_data).decode()
        
        # HTML audio player
        audio_html = f"""
        <audio controls style="width: 100%;">
            <source src="data:audio/wav;base64,{b64_audio}" type="audio/wav">
            Your browser does not support the audio element.
        </audio>
        """
        
        st.markdown(audio_html, unsafe_allow_html=True)
        
        # Playback info
        duration = len(audio_data) / sample_rate
        st.caption(f"Duration: {duration:.3f} seconds | Sample Rate: {sample_rate} Hz")
    
    def _create_wav_download(self, audio_data: np.ndarray, sample_rate: int) -> bytes:
        """Create WAV file data for download."""
        # Normalize and convert to int16
        if np.max(np.abs(audio_data)) > 0:
            normalized = audio_data / np.max(np.abs(audio_data)) * 0.95
        else:
            normalized = audio_data
        
        audio_int16 = (normalized * 32767).astype(np.int16)
        
        # Create WAV file in memory
        buffer = io.BytesIO()
        with wave.open(buffer, 'wb') as wav_file:
            wav_file.setnchannels(1)  # Mono
            wav_file.setsampwidth(2)  # 2 bytes per sample
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio_int16.tobytes())
        
        return buffer.getvalue()
    
    def _create_csv_download(self, audio_data: np.ndarray, sample_rate: int) -> str:
        """Create CSV data for download."""
        time_axis = np.arange(len(audio_data)) / sample_rate
        
        csv_lines = ["time,amplitude"]
        for t, amp in zip(time_axis, audio_data):
            csv_lines.append(f"{t:.6f},{amp:.6f}")
        
        return "\n".join(csv_lines)
    
    def _create_metadata_json(self, audio_data: np.ndarray, sample_rate: int) -> str:
        """Create metadata JSON for download."""
        import json
        
        metadata = {
            "sample_rate": sample_rate,
            "length_samples": len(audio_data),
            "duration_seconds": len(audio_data) / sample_rate,
            "max_amplitude": float(np.max(np.abs(audio_data))),
            "rms_level": float(np.sqrt(np.mean(audio_data**2))),
            "mean": float(np.mean(audio_data)),
            "std_dev": float(np.std(audio_data)),
            "dynamic_range_db": float(20 * np.log10(np.max(np.abs(audio_data)) / (np.std(audio_data) + 1e-10))),
            "component_id": self.component_id,
            "export_timestamp": str(np.datetime64('now'))
        }
        
        return json.dumps(metadata, indent=2)