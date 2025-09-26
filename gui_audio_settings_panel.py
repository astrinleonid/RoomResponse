#!/usr/bin/env python3
"""
Audio Settings Panel - Device Configuration and Diagnostics

This version is corrected to work with the actual SDL audio core bindings
as defined in python_bindings.cpp. Uses the correct function names and
return formats from your compiled SDL module.

Includes integration with SinglePulseRecorder component for extended functionality.

Save this file as: gui_audio_settings_panel.py
"""

import os
import sys
import time
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import streamlit as st

# Audio core imports
try:
    import sdl_audio_core as sdl
    SDL_AVAILABLE = True
except ImportError:
    SDL_AVAILABLE = False
    sdl = None

try:
    from RoomResponseRecorder import RoomResponseRecorder
    RECORDER_AVAILABLE = True
except ImportError:
    RECORDER_AVAILABLE = False
    RoomResponseRecorder = None

# Import the single pulse recorder component
try:
    from gui_single_pulse_recorder import SinglePulseRecorder
    PULSE_RECORDER_AVAILABLE = True
except ImportError:
    PULSE_RECORDER_AVAILABLE = False
    SinglePulseRecorder = None

try:
    from gui_series_settings_panel import SeriesSettingsPanel
    SERIES_SETTINGS_AVAILABLE = True
except ImportError:
    SERIES_SETTINGS_AVAILABLE = False
    SeriesSettingsPanel = None

class AudioSettingsPanel:
    """Audio device configuration and diagnostics panel."""
    
    def __init__(self, scenario_manager=None):
        self.scenario_manager = scenario_manager
        self.recorder = None
        self._audio_test_active = False
        self._test_results = {}
        
    def render(self):
        """Main panel rendering method."""
        st.header("Audio Settings & Diagnostics")
        
        if not SDL_AVAILABLE:
            st.error("SDL Audio Core not available. Please build and install sdl_audio_core.")
            self._render_build_instructions()
            return
        
        # Show debug info about available SDL functions
        with st.expander("SDL Debug Info"):
            self._render_sdl_debug_info()
            
        # Initialize session state
        self._init_session_state()
        
        # Show current audio status
        self._render_audio_status_bar()
        
        # Main tabs
        if SERIES_SETTINGS_AVAILABLE:
            tab1, tab2, tab3, tab4, tab5 = st.tabs([
                "System Info", 
                "Device Selection", 
                "Parameters",
                "Testing",
                "Series Settings"
            ])
        else:
            tab1, tab2, tab3, tab4 = st.tabs([
                "System Info", 
                "Device Selection", 
                "Parameters",
                "Testing"
            ])

        with tab1:
            self._render_system_info()
            
        with tab2:
            self._render_device_selection()
            
        with tab3:
            self._render_audio_parameters()
            
        with tab4:
            self._render_testing_panel()

        # Add the new Series Settings tab
        if SERIES_SETTINGS_AVAILABLE:
            with tab5:
                if not hasattr(self, '_series_settings_panel'):
                    self._series_settings_panel = SeriesSettingsPanel(audio_settings_panel=self)
                self._series_settings_panel.render()
    
    def _render_sdl_debug_info(self):
        """Show available SDL functions for debugging."""
        st.markdown("**Available SDL Functions:**")
        
        if SDL_AVAILABLE:
            all_funcs = [attr for attr in dir(sdl) if not attr.startswith('_')]
            st.code('\n'.join(all_funcs))
            
            # Test the actual functions from your bindings
            st.markdown("**Function Tests:**")
            
            # Test list_all_devices (this should work)
            try:
                devices = sdl.list_all_devices()
                st.success(f"list_all_devices(): Works - found {len(devices.get('input_devices', []))} input, {len(devices.get('output_devices', []))} output devices")
            except Exception as e:
                st.error(f"list_all_devices(): Error - {e}")
            
            # Test get_version (this should work)
            try:
                version = sdl.get_version()
                st.success(f"get_version(): Works - {version}")
            except Exception as e:
                st.error(f"get_version(): Error - {e}")
            
            # Test get_build_info (this should work)
            try:
                build_info = sdl.get_build_info()
                st.success(f"get_build_info(): Works - {type(build_info)}")
            except Exception as e:
                st.error(f"get_build_info(): Error - {e}")
            
            # Test AudioEngine static methods
            try:
                engine_class = getattr(sdl, 'AudioEngine')
                drivers = engine_class.get_audio_drivers()
                st.success(f"AudioEngine.get_audio_drivers(): Works - {len(drivers)} drivers")
            except Exception as e:
                st.error(f"AudioEngine.get_audio_drivers(): Error - {e}")
            
            try:
                engine_class = getattr(sdl, 'AudioEngine')
                sdl_version = engine_class.get_sdl_version()
                st.success(f"AudioEngine.get_sdl_version(): Works - {sdl_version}")
            except Exception as e:
                st.error(f"AudioEngine.get_sdl_version(): Error - {e}")
        else:
            st.error("SDL not available")
    
    def _init_session_state(self):
        """Initialize session state variables."""
        defaults = {
            'selected_input_device': 'System Default',
            'selected_output_device': 'System Default',
            'sample_rate': 48000,
            'buffer_size': 512,
            'input_channels': 1,
            'output_channels': 2,
            'test_frequency': 1000.0,
            'test_duration': 2.0,
            'test_volume': 0.3,
            'last_device_scan': 0,
            'device_cache': {}
        }
        
        for key, value in defaults.items():
            if f'audio_{key}' not in st.session_state:
                st.session_state[f'audio_{key}'] = value
    
    def _render_audio_status_bar(self):
        """Render quick status overview."""
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.success("SDL Ready" if SDL_AVAILABLE else "SDL Missing")
        
        with col2:
            input_dev = st.session_state.get('audio_selected_input_device', 'None')
            display_input = (input_dev[:15] + "...") if len(str(input_dev)) > 15 else str(input_dev)
            st.info(f"In: {display_input}")
        
        with col3:
            output_dev = st.session_state.get('audio_selected_output_device', 'None') 
            display_output = (output_dev[:15] + "...") if len(str(output_dev)) > 15 else str(output_dev)
            st.info(f"Out: {display_output}")
        
        with col4:
            sample_rate = st.session_state.get('audio_sample_rate', 48000)
            buffer_size = st.session_state.get('audio_buffer_size', 512)
            latency_ms = (buffer_size / sample_rate) * 1000
            st.info(f"{sample_rate//1000}kHz/{latency_ms:.0f}ms")
    
    def _render_build_instructions(self):
        """Show build instructions when SDL is not available."""
        st.markdown("""
        ### SDL Audio Core Required
        
        Run the build script to compile the audio module:
        
        ```bash
        # From project root
        build_sdl_audio.bat
        ```
        
        Or manually:
        ```bash
        cd sdl_audio_core
        python setup.py build_ext --inplace
        pip install -e .
        ```
        """)
    
    def _render_system_info(self):
        """Render audio system information."""
        st.subheader("SDL Audio Core Status")
        
        if not SDL_AVAILABLE:
            st.error("SDL Audio Core not loaded")
            return
        
        # Basic SDL info using actual functions
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**SDL Module Info**")
            st.success("SDL Audio Core: Available")
            
            # Get version using actual function
            try:
                version = sdl.get_version()
                st.info(f"Version: {version}")
            except Exception as e:
                st.warning(f"Version unavailable: {e}")
            
            # Get SDL version through AudioEngine
            try:
                sdl_version = sdl.AudioEngine.get_sdl_version()
                st.info(f"SDL Version: {sdl_version}")
            except Exception as e:
                st.warning(f"SDL version unavailable: {e}")
        
        with col2:
            st.markdown("**Build Information**")
            
            # Get build info using actual function
            try:
                build_info = sdl.get_build_info()
                if isinstance(build_info, dict):
                    for key, value in build_info.items():
                        if key != 'python_version':  # Skip long python version
                            st.write(f"**{key}:** {value}")
                else:
                    st.info(f"Build info: {build_info}")
            except Exception as e:
                st.warning(f"Build info unavailable: {e}")
        
        # Driver information
        st.markdown("---")
        st.markdown("**Audio Drivers**")
        
        try:
            drivers = sdl.AudioEngine.get_audio_drivers()
            st.success(f"Available drivers: {len(drivers)}")
            st.caption(", ".join(drivers))
        except Exception as e:
            st.warning(f"Driver info unavailable: {e}")
        
        # Device count using actual function
        st.markdown("---")
        st.markdown("**Audio Devices**")
        
        try:
            devices = sdl.list_all_devices()
            input_count = len(devices.get('input_devices', []))
            output_count = len(devices.get('output_devices', []))
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Input Devices", input_count)
            with col2:
                st.metric("Output Devices", output_count)
            with col3:
                st.metric("Total Devices", input_count + output_count)
                
        except Exception as e:
            st.error(f"Error getting devices: {e}")
        
        # Test installation
        st.markdown("---")
        st.markdown("**Installation Test**")
        
        if st.button("Run Installation Check"):
            try:
                result = sdl.check_installation()
                if result:
                    st.success("Installation check passed!")
                else:
                    st.error("Installation check failed")
            except Exception as e:
                st.error(f"Installation check error: {e}")
    
    def _render_device_selection(self):
        """Render device selection interface."""
        st.subheader("Audio Device Configuration")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            if st.button("Refresh Devices"):
                self._refresh_audio_devices()
        
        with col2:
            last_scan = st.session_state.get('audio_last_device_scan', 0)
            if last_scan > 0:
                scan_time = time.strftime('%H:%M:%S', time.localtime(last_scan))
                st.info(f"Last scan: {scan_time}")
        
        # Get devices using the correct SDL function
        devices = self._get_audio_devices()
        
        if devices and (devices.get('input') or devices.get('output')):
            # Input devices
            st.markdown("**Input Devices**")
            input_devices = devices.get('input', [])
            self._render_device_selector('input', input_devices)
            
            st.markdown("---")
            
            # Output devices  
            st.markdown("**Output Devices**")
            output_devices = devices.get('output', [])
            self._render_device_selector('output', output_devices)
        else:
            st.warning("No audio devices found")
            st.markdown("""
            **Troubleshooting:**
            - Check that audio drivers are installed
            - Restart the application  
            - Close other audio applications
            - Check Windows audio settings
            """)
    
    def _refresh_audio_devices(self):
        """Refresh the audio device list."""
        with st.spinner("Scanning audio devices..."):
            st.session_state['audio_device_cache'] = {}
            st.session_state['audio_last_device_scan'] = time.time()
            
            try:
                devices = sdl.list_all_devices()
                # Convert to expected format
                formatted_devices = {
                    'input': devices.get('input_devices', []),
                    'output': devices.get('output_devices', [])
                }
                st.session_state['audio_device_cache'] = formatted_devices
                
                input_count = len(formatted_devices['input'])
                output_count = len(formatted_devices['output'])
                st.success(f"Found {input_count} input and {output_count} output devices")
                
            except Exception as e:
                st.error(f"Device scan failed: {e}")
                
        st.rerun()
    
    def _get_audio_devices(self) -> Dict[str, List]:
        """Get cached or fresh audio device list."""
        cache_key = 'audio_device_cache'
        
        if not st.session_state.get(cache_key):
            try:
                # Use the actual function from your bindings
                devices = sdl.list_all_devices()
                
                # Your bindings return a dict with 'input_devices' and 'output_devices' keys
                if isinstance(devices, dict):
                    # Convert to expected format
                    formatted_devices = {
                        'input': devices.get('input_devices', []),
                        'output': devices.get('output_devices', [])
                    }
                    st.session_state[cache_key] = formatted_devices
                    st.session_state['audio_last_device_scan'] = time.time()
                    return formatted_devices
                
            except Exception as e:
                st.error(f"Failed to get devices: {e}")
        
        return st.session_state.get(cache_key, {'input': [], 'output': []})
    
    def _render_device_selector(self, device_type: str, devices: List):
        """Render device selector for input or output."""
        if not devices:
            st.warning(f"No {device_type} devices found")
            return
        
        # Create device options
        options = ["System Default"]
        
        for i, device in enumerate(devices):
            if isinstance(device, dict):
                name = device.get('name', f'Device {i}')
                device_id = device.get('device_id', i)
                options.append(f"{name} (ID: {device_id})")
            else:
                # Handle case where device is just a string or other format
                options.append(str(device))
        
        # Current selection
        session_key = f'audio_selected_{device_type}_device'
        current = st.session_state.get(session_key, 'System Default')
        
        try:
            current_idx = options.index(current) if current in options else 0
        except:
            current_idx = 0
        
        # Device selector
        selected = st.selectbox(
            f"Select {device_type.title()} Device",
            options,
            index=current_idx,
            key=f"{device_type}_device_selector"
        )
        
        st.session_state[session_key] = selected
        
        # Show device details if available
        if selected != "System Default" and len(devices) > 0:
            try:
                device_idx = options.index(selected) - 1
                if 0 <= device_idx < len(devices):
                    device = devices[device_idx]
                    if isinstance(device, dict):
                        with st.expander(f"{device_type.title()} Device Details"):
                            for key, value in device.items():
                                st.write(f"**{key.title()}:** {value}")
            except:
                pass
    
    def _render_audio_parameters(self):
        """Render audio parameter configuration with single pulse recording integration."""
        st.subheader("Audio Parameters")
        
        # Preset buttons
        st.markdown("**Quick Presets**")
        preset_col1, preset_col2, preset_col3 = st.columns(3)
        
        with preset_col1:
            if st.button("Low Latency"):
                self._apply_preset("low_latency")
        
        with preset_col2:
            if st.button("Balanced"):
                self._apply_preset("balanced")
        
        with preset_col3:
            if st.button("High Quality"):
                self._apply_preset("high_quality")
        
        st.markdown("---")
        
        # Core parameters in columns
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Core Settings**")
            
            # Sample Rate
            sample_rates = [8000, 16000, 22050, 44100, 48000, 88200, 96000]
            current_rate = st.session_state.get('audio_sample_rate', 48000)
            
            try:
                rate_idx = sample_rates.index(current_rate)
            except:
                rate_idx = 4  # 48000 Hz
            
            st.session_state['audio_sample_rate'] = st.selectbox(
                "Sample Rate (Hz)",
                sample_rates,
                index=rate_idx,
                help="Audio sample rate - higher is better quality"
            )
            
            # Buffer Size
            buffer_sizes = [64, 128, 256, 512, 1024, 2048]
            current_buffer = st.session_state.get('audio_buffer_size', 512)
            
            try:
                buffer_idx = buffer_sizes.index(current_buffer)
            except:
                buffer_idx = 3  # 512 samples
            
            st.session_state['audio_buffer_size'] = st.selectbox(
                "Buffer Size (samples)",
                buffer_sizes,
                index=buffer_idx,
                help="Audio buffer size - smaller = lower latency"
            )
            
            # Calculate and show latency
            sample_rate = st.session_state['audio_sample_rate']
            buffer_size = st.session_state['audio_buffer_size']
            latency_ms = (buffer_size / sample_rate) * 1000
            
            if latency_ms < 10:
                st.success(f"Latency: {latency_ms:.1f} ms (Excellent)")
            elif latency_ms < 20:
                st.info(f"Latency: {latency_ms:.1f} ms (Good)")
            elif latency_ms < 50:
                st.warning(f"Latency: {latency_ms:.1f} ms (Acceptable)")
            else:
                st.error(f"Latency: {latency_ms:.1f} ms (High)")
        
        with col2:
            st.markdown("**Channel Configuration**")
            
            st.session_state['audio_input_channels'] = st.selectbox(
                "Input Channels",
                [1, 2],
                index=0 if st.session_state.get('audio_input_channels', 1) == 1 else 1,
                format_func=lambda x: "Mono" if x == 1 else "Stereo"
            )
            
            st.session_state['audio_output_channels'] = st.selectbox(
                "Output Channels", 
                [1, 2],
                index=1 if st.session_state.get('audio_output_channels', 2) == 2 else 0,
                format_func=lambda x: "Mono" if x == 1 else "Stereo"
            )
        
        # Single Pulse Recording Section
        st.markdown("---")
        
        if PULSE_RECORDER_AVAILABLE:
            # Create and render the single pulse recorder component
            if not hasattr(self, '_pulse_recorder'):
                self._pulse_recorder = SinglePulseRecorder(audio_settings_panel=self)
            
            self._pulse_recorder.render()
        else:
            st.error("Single Pulse Recorder component not available. Please install gui_single_pulse_recorder.py")
        
        # Legacy test signal parameters (kept for compatibility with existing testing functions)
        st.markdown("---")
        st.markdown("**Legacy Test Parameters**")
        
        with st.expander("Advanced Test Settings", expanded=False):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.session_state['audio_test_frequency'] = st.number_input(
                    "Test Frequency (Hz)",
                    min_value=20.0,
                    max_value=20000.0,
                    value=st.session_state.get('audio_test_frequency', 1000.0),
                    step=50.0
                )
            
            with col2:
                st.session_state['audio_test_duration'] = st.number_input(
                    "Test Duration (sec)",
                    min_value=0.1,
                    max_value=10.0,
                    value=st.session_state.get('audio_test_duration', 2.0),
                    step=0.1
                )
            
            with col3:
                st.session_state['audio_test_volume'] = st.slider(
                    "Test Volume",
                    min_value=0.0,
                    max_value=1.0,
                    value=st.session_state.get('audio_test_volume', 0.3),
                    step=0.05
                )
        
        # Configuration management
        st.markdown("---")
        self._render_config_management()
    
    def _apply_preset(self, preset_type: str):
        """Apply audio parameter presets."""
        presets = {
            "low_latency": {
                'audio_sample_rate': 48000,
                'audio_buffer_size': 128,
            },
            "balanced": {
                'audio_sample_rate': 48000,
                'audio_buffer_size': 512,
            },
            "high_quality": {
                'audio_sample_rate': 96000,
                'audio_buffer_size': 1024,
            }
        }
        
        if preset_type in presets:
            for key, value in presets[preset_type].items():
                st.session_state[key] = value
            
            st.success(f"Applied {preset_type.replace('_', ' ').title()} preset")
            st.rerun()
    
    def _render_testing_panel(self):
        """Render audio testing interface using actual SDL functions."""
        st.subheader("Audio Testing")
        
        # Quick device test
        st.markdown("**Quick Device Test**")
        
        # Get current device selections
        input_dev = st.session_state.get('audio_selected_input_device', 'System Default')
        output_dev = st.session_state.get('audio_selected_output_device', 'System Default')
        
        st.info(f"Current selection: {input_dev} -> {output_dev}")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Test Device Combination"):
                self._run_device_combination_test()
        
        with col2:
            if st.button("Quick Audio Test"):
                self._run_quick_audio_test()
        
        # Room response measurement
        st.markdown("---")
        st.markdown("**Room Response Measurement**")
        
        st.info("Test room response measurement with current device settings")
        
        if st.button("Measure Room Response"):
            self._run_room_response_test()
        
        # Display test results
        if hasattr(self, '_test_results') and self._test_results:
            self._display_test_results()
    
    def _run_device_combination_test(self):
        """Test if current device combination works."""
        try:
            with st.spinner("Testing device combination..."):
                # Try to get device IDs from selections
                input_id = self._get_device_id_from_selection('input')
                output_id = self._get_device_id_from_selection('output')
                
                # Use the SDL function to test devices
                result = sdl.AudioEngine().test_device_combination(input_id, output_id)
                
                if result:
                    st.success("Device combination works!")
                    self._test_results['device_test'] = {
                        'success': True,
                        'input_id': input_id,
                        'output_id': output_id,
                        'timestamp': time.time()
                    }
                else:
                    st.error("Device combination failed")
                    self._test_results['device_test'] = {
                        'success': False,
                        'error': 'Device combination test failed',
                        'timestamp': time.time()
                    }
                    
        except Exception as e:
            st.error(f"Device test error: {e}")
            self._test_results['device_test'] = {
                'success': False,
                'error': str(e),
                'timestamp': time.time()
            }
    
    def _run_quick_audio_test(self):
        """Run a quick audio test."""
        try:
            with st.spinner("Running quick audio test..."):
                input_id = self._get_device_id_from_selection('input')
                output_id = self._get_device_id_from_selection('output')
                
                # Generate a simple test signal
                frequency = st.session_state.get('audio_test_frequency', 1000.0)
                duration = st.session_state.get('audio_test_duration', 2.0)
                sample_rate = st.session_state.get('audio_sample_rate', 48000)
                
                # Create test signal (simple sine wave)
                import math
                test_signal = []
                for i in range(int(sample_rate * duration)):
                    t = i / sample_rate
                    amplitude = st.session_state.get('audio_test_volume', 0.3)
                    sample = amplitude * math.sin(2 * math.pi * frequency * t)
                    test_signal.append(sample)
                
                # Use SDL quick test function
                result = sdl.quick_device_test(input_id, output_id, test_signal)
                
                if result['success']:
                    st.success(f"Audio test successful! Recorded {result['samples_recorded']} samples")
                    self._test_results['audio_test'] = result
                else:
                    st.error(f"Audio test failed: {result.get('error_message', 'Unknown error')}")
                    self._test_results['audio_test'] = result
                    
        except Exception as e:
            st.error(f"Audio test error: {e}")
            self._test_results['audio_test'] = {
                'success': False,
                'error_message': str(e),
                'timestamp': time.time()
            }
    
    def _run_room_response_test(self):
        """Test room response measurement."""
        try:
            with st.spinner("Measuring room response..."):
                # Generate test signal
                frequency = st.session_state.get('audio_test_frequency', 1000.0)
                duration = st.session_state.get('audio_test_duration', 2.0)
                sample_rate = st.session_state.get('audio_sample_rate', 48000)
                volume = st.session_state.get('audio_test_volume', 0.3)
                
                # Create test signal
                import math
                test_signal = []
                for i in range(int(sample_rate * duration)):
                    t = i / sample_rate
                    sample = volume * math.sin(2 * math.pi * frequency * t)
                    test_signal.append(sample)
                
                # Use SDL room response function
                result = sdl.measure_room_response_auto(test_signal, volume)
                
                if result['success']:
                    st.success(f"Room response measurement successful!")
                    st.info(f"Test signal: {result['test_signal_samples']} samples")
                    st.info(f"Recorded: {result['recorded_samples']} samples")
                    self._test_results['room_response_test'] = result
                else:
                    st.error(f"Room response failed: {result.get('error_message', 'Unknown error')}")
                    self._test_results['room_response_test'] = result
                    
        except Exception as e:
            st.error(f"Room response test error: {e}")
            self._test_results['room_response_test'] = {
                'success': False,
                'error_message': str(e),
                'timestamp': time.time()
            }
    
    def _get_device_id_from_selection(self, device_type: str) -> int:
        """Get device ID from current selection."""
        session_key = f'audio_selected_{device_type}_device'
        selection = st.session_state.get(session_key, 'System Default')
        
        if selection == 'System Default':
            return -1  # Use default device
        
        # Extract ID from selection string like "Device Name (ID: 1)"
        try:
            if '(ID: ' in selection and ')' in selection:
                id_part = selection.split('(ID: ')[1].split(')')[0]
                return int(id_part)
        except:
            pass
        
        return 0  # Fallback to device 0
    
    def _display_test_results(self):
        """Display test results."""
        st.markdown("---")
        st.markdown("**Test Results**")
        
        for test_name, result in self._test_results.items():
            with st.expander(f"{test_name.replace('_', ' ').title()} - {'Success' if result.get('success') else 'Failed'}"):
                
                if result.get('success'):
                    st.success("Test passed")
                    
                    # Show relevant metrics based on test type
                    if 'samples_recorded' in result:
                        st.metric("Samples Recorded", result['samples_recorded'])
                    if 'input_device_id' in result:
                        st.write(f"Input Device ID: {result['input_device_id']}")
                    if 'output_device_id' in result:
                        st.write(f"Output Device ID: {result['output_device_id']}")
                    if 'test_signal_samples' in result:
                        st.metric("Test Signal Length", result['test_signal_samples'])
                    if 'recorded_samples' in result:
                        st.metric("Recorded Length", result['recorded_samples'])
                        
                else:
                    st.error(f"Test failed: {result.get('error_message', result.get('error', 'Unknown error'))}")
                
                # Timestamp
                if 'timestamp' in result:
                    test_time = time.strftime('%H:%M:%S', time.localtime(result['timestamp']))
                    st.caption(f"Tested at: {test_time}")
    
    def _render_config_management(self):
        """Render configuration management."""
        st.markdown("**Configuration**")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("Export Config"):
                config = self.get_current_config()
                config_json = json.dumps(config, indent=2)
                
                st.download_button(
                    "Download Config",
                    data=config_json,
                    file_name=f"audio_config_{int(time.time())}.json",
                    mime="application/json"
                )
        
        with col2:
            uploaded = st.file_uploader(
                "Import Config",
                type=['json'],
                key="config_upload"
            )
            
            if uploaded:
                try:
                    config = json.load(uploaded)
                    self._apply_config(config)
                    st.success("Config imported")
                    st.rerun()
                except Exception as e:
                    st.error(f"Import failed: {e}")
        
        with col3:
            if st.button("Reset Defaults"):
                self._reset_to_defaults()
                st.success("Reset to defaults")
                st.rerun()
    
    def get_current_config(self) -> Dict[str, Any]:
        """Get current configuration as dictionary."""
        return {
            'audio_settings': {
                'sample_rate': st.session_state.get('audio_sample_rate', 48000),
                'buffer_size': st.session_state.get('audio_buffer_size', 512),
                'input_channels': st.session_state.get('audio_input_channels', 1),
                'output_channels': st.session_state.get('audio_output_channels', 2),
                'selected_input_device': st.session_state.get('audio_selected_input_device', 'System Default'),
                'selected_output_device': st.session_state.get('audio_selected_output_device', 'System Default')
            },
            'test_parameters': {
                'test_frequency': st.session_state.get('audio_test_frequency', 1000.0),
                'test_duration': st.session_state.get('audio_test_duration', 2.0),
                'test_volume': st.session_state.get('audio_test_volume', 0.3)
            },
            'pulse_parameters': {
                'pulse_frequency': st.session_state.get('pulse_frequency', 1000.0),
                'pulse_duration': st.session_state.get('pulse_duration', 8.0),
                'pulse_volume': st.session_state.get('pulse_volume', 0.4),
                'pulse_form': st.session_state.get('pulse_form', 'sine'),
                'record_duration': st.session_state.get('record_duration', 200.0),
                'fade_duration': st.session_state.get('fade_duration', 0.1)
            },
            'export_timestamp': time.time(),
            'export_version': '1.0'
        }
    
    def _apply_config(self, config: Dict[str, Any]):
        """Apply configuration from dictionary."""
        try:
            # Apply audio settings
            audio_settings = config.get('audio_settings', {})
            for key, value in audio_settings.items():
                st.session_state[f'audio_{key}'] = value
            
            # Apply test parameters
            test_params = config.get('test_parameters', {})
            for key, value in test_params.items():
                st.session_state[f'audio_{key}'] = value
            
            # Apply pulse parameters
            pulse_params = config.get('pulse_parameters', {})
            for key, value in pulse_params.items():
                st.session_state[key] = value
                
        except Exception as e:
            raise ValueError(f"Invalid configuration format: {e}")
    
    def _reset_to_defaults(self):
        """Reset all settings to default values."""
        defaults = {
            'audio_selected_input_device': 'System Default',
            'audio_selected_output_device': 'System Default',
            'audio_sample_rate': 48000,
            'audio_buffer_size': 512,
            'audio_input_channels': 1,
            'audio_output_channels': 2,
            'audio_test_frequency': 1000.0,
            'audio_test_duration': 2.0,
            'audio_test_volume': 0.3,
            'pulse_frequency': 1000.0,
            'pulse_duration': 8.0,
            'pulse_volume': 0.4,
            'pulse_form': 'sine',
            'record_duration': 200.0,
            'fade_duration': 0.1
        }
        
        for key, value in defaults.items():
            st.session_state[key] = value