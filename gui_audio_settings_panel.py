#!/usr/bin/env python3
"""
Audio Settings Panel - Device Configuration and Diagnostics (Corrected)

Updated to work with actual SDL audio core function names.
This version gracefully handles missing functions and uses available alternatives.
"""

import os
import sys
import time
import json
import threading
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import streamlit as st
import numpy as np

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


class AudioSettingsPanel:
    """Audio device configuration and diagnostics panel."""
    
    def __init__(self, scenario_manager=None):
        self.scenario_manager = scenario_manager
        self.recorder = None
        self._audio_test_active = False
        self._test_results = {}
        self._available_functions = self._detect_available_functions()
        
    def _detect_available_functions(self) -> Dict[str, str]:
        """Detect which SDL functions are actually available."""
        function_map = {}
        
        if not SDL_AVAILABLE:
            return function_map
        
        # Check for version info functions
        for version_func in ['get_sdl_version', 'get_version', 'SDL_VERSION', '__version__']:
            if hasattr(sdl, version_func):
                function_map['version'] = version_func
                break
        
        # Check for build info functions
        for build_func in ['get_build_info', 'build_info', 'get_info']:
            if hasattr(sdl, build_func):
                function_map['build_info'] = build_func
                break
        
        # Check for driver functions
        for driver_func in ['get_audio_drivers', 'get_drivers', 'list_drivers']:
            if hasattr(sdl, driver_func):
                function_map['drivers'] = driver_func
                break
        
        # Check for device listing functions
        for device_func in ['list_all_devices', 'get_devices', 'enumerate_devices']:
            if hasattr(sdl, device_func):
                function_map['devices'] = device_func
                break
        
        # Check for input device functions
        for input_func in ['get_input_devices', 'list_input_devices']:
            if hasattr(sdl, input_func):
                function_map['input_devices'] = input_func
                break
        
        # Check for output device functions
        for output_func in ['get_output_devices', 'list_output_devices']:
            if hasattr(sdl, output_func):
                function_map['output_devices'] = output_func
                break
        
        return function_map
        
    def render(self):
        """Main panel rendering method."""
        st.header("Audio Settings & Diagnostics")
        
        if not SDL_AVAILABLE:
            st.error("SDL Audio Core not available. Please build and install sdl_audio_core.")
            self._render_build_instructions()
            return
            
        # Show available functions for debugging
        if st.checkbox("Show Debug Info", value=False):
            self._render_debug_info()
            
        # Initialize session state
        self._init_session_state()
        
        # Show current audio status
        self._render_audio_status_bar()
        
        # Main tabs
        tab1, tab2, tab3, tab4 = st.tabs([
            "System Info", 
            "Device Selection", 
            "Parameters", 
            "Diagnostics"
        ])
        
        with tab1:
            self._render_system_info()
            
        with tab2:
            self._render_device_selection()
            
        with tab3:
            self._render_audio_parameters()
            
        with tab4:
            self._render_diagnostics_and_test()
    
    def _render_debug_info(self):
        """Show debug information about available SDL functions."""
        st.markdown("**SDL Module Debug Info:**")
        
        # Show all available attributes
        all_attrs = [attr for attr in dir(sdl) if not attr.startswith('_')]
        st.write(f"Available attributes: {', '.join(all_attrs)}")
        
        # Show detected function mappings
        st.write("Function mappings:", self._available_functions)
        
        # Try to call each function safely
        st.markdown("**Function Test Results:**")
        for func_type, func_name in self._available_functions.items():
            try:
                func = getattr(sdl, func_name)
                result = func()
                st.success(f"{func_type} ({func_name}): {str(result)[:100]}")
            except Exception as e:
                st.error(f"{func_type} ({func_name}): Error - {e}")
    
    def _init_session_state(self):
        """Initialize session state variables."""
        defaults = {
            'selected_input_device': None,
            'selected_output_device': None,
            'sample_rate': 48000,
            'buffer_size': 512,
            'input_channels': 1,
            'output_channels': 2,
            'test_frequency': 1000.0,
            'test_duration': 2.0,
            'test_volume': 0.3,
            'audio_driver': 'default',
            'last_device_scan': 0,
            'device_cache': {}
        }
        
        for key, value in defaults.items():
            if f'audio_{key}' not in st.session_state:
                st.session_state[f'audio_{key}'] = value
    
    def _render_audio_status_bar(self):
        """Render quick status overview."""
        col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
        
        with col1:
            if SDL_AVAILABLE:
                st.success("SDL Ready")
            else:
                st.error("SDL Missing")
        
        with col2:
            input_dev = st.session_state.get('audio_selected_input_device', 'None')
            display_input = (input_dev[:15] + "...") if input_dev and len(input_dev) > 15 else (input_dev or "None")
            st.info(f"Input: {display_input}")
        
        with col3:
            output_dev = st.session_state.get('audio_selected_output_device', 'None') 
            display_output = (output_dev[:15] + "...") if output_dev and len(output_dev) > 15 else (output_dev or "None")
            st.info(f"Output: {display_output}")
        
        with col4:
            sample_rate = st.session_state.get('audio_sample_rate', 48000)
            buffer_size = st.session_state.get('audio_buffer_size', 512)
            latency_ms = (buffer_size / sample_rate) * 1000
            st.info(f"{sample_rate//1000}kHz | {latency_ms:.1f}ms")
    
    def _render_build_instructions(self):
        """Show build instructions when SDL is not available."""
        st.markdown("""
        ### SDL Audio Core Required
        
        The audio functionality requires the compiled SDL audio core module.
        
        **Build Instructions:**
        
        1. **Windows (using build script):**
           ```bash
           # Run from project root directory
           build_sdl_audio.bat
           ```
        
        2. **Manual build:**
           ```bash
           cd sdl_audio_core
           python setup.py build_ext --inplace
           pip install -e .
           ```
        
        3. **Verify installation:**
           ```python
           import sdl_audio_core as sdl
           print("Available functions:", dir(sdl))
           ```
        """)
    
    def _safe_call(self, func_type: str, default_return=None):
        """Safely call an SDL function with error handling."""
        if func_type not in self._available_functions:
            return default_return
        
        try:
            func_name = self._available_functions[func_type]
            func = getattr(sdl, func_name)
            return func()
        except Exception as e:
            st.warning(f"Error calling {func_type}: {e}")
            return default_return
    
    def _render_system_info(self):
        """Render audio system information."""
        st.subheader("Audio System Information")
        
        try:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**SDL Audio Core**")
                
                # Try to get version info
                version_info = self._safe_call('version', 'Unknown')
                if version_info != 'Unknown':
                    st.success(f"SDL Version: {version_info}")
                else:
                    st.info("SDL Version: Not available")
                
                # Try to get build info
                build_info = self._safe_call('build_info', 'Unknown')
                if build_info != 'Unknown':
                    st.info(f"Build: {build_info}")
                else:
                    st.info("Build Info: Not available")
            
            with col2:
                st.markdown("**System Audio**")
                
                # Try to get drivers
                drivers = self._safe_call('drivers', [])
                if drivers:
                    st.write(f"Available drivers: {len(drivers)}")
                    st.caption(", ".join(drivers[:3]) + ("..." if len(drivers) > 3 else ""))
                else:
                    st.info("Driver info not available")
                
                current_driver = st.session_state.get('audio_audio_driver', 'default')
                st.write(f"Selected driver: {current_driver}")
            
            # Device summary
            st.markdown("---")
            self._render_device_summary()
            
        except Exception as e:
            st.error(f"Error retrieving system information: {e}")
    
    def _render_device_summary(self):
        """Show device count and quick stats."""
        st.markdown("**Audio Devices Overview**")
        
        try:
            devices = self._get_audio_devices()
            input_devices = devices.get('input', [])
            output_devices = devices.get('output', [])
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Input Devices", len(input_devices))
            
            with col2:
                st.metric("Output Devices", len(output_devices))
            
            with col3:
                total_devices = len(input_devices) + len(output_devices)
                st.metric("Total Devices", total_devices)
                
        except Exception as e:
            st.error(f"Error getting device summary: {e}")
    
    def _render_device_selection(self):
        """Render device selection interface."""
        st.subheader("Audio Device Selection")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            if st.button("Refresh Devices"):
                self._refresh_audio_devices()
        
        with col2:
            last_scan = st.session_state.get('audio_last_device_scan', 0)
            if last_scan > 0:
                scan_time = time.strftime('%H:%M:%S', time.localtime(last_scan))
                st.info(f"Last device scan: {scan_time}")
        
        try:
            devices = self._get_audio_devices()
            
            if not devices.get('input') and not devices.get('output'):
                st.warning("No audio devices found. Try refreshing or check your audio drivers.")
                return
            
            # Input device selection
            st.markdown("---")
            self._render_input_device_selection(devices)
            
            # Output device selection
            st.markdown("---") 
            self._render_output_device_selection(devices)
            
            # Driver selection (if drivers are available)
            drivers = self._safe_call('drivers', [])
            if drivers:
                st.markdown("---")
                self._render_driver_selection(drivers)
            
        except Exception as e:
            st.error(f"Error loading devices: {e}")
    
    def _refresh_audio_devices(self):
        """Refresh the audio device list."""
        with st.spinner("Scanning audio devices..."):
            st.session_state['audio_device_cache'] = {}
            st.session_state['audio_last_device_scan'] = time.time()
            
            try:
                devices = self._safe_call('devices', {'input': [], 'output': []})
                st.session_state['audio_device_cache'] = devices
                
                input_count = len(devices.get('input', []))
                output_count = len(devices.get('output', []))
                st.success(f"Found {input_count} input and {output_count} output devices")
                
            except Exception as e:
                st.error(f"Device scan failed: {e}")
                
        st.rerun()
    
    def _get_audio_devices(self) -> Dict[str, List[Dict]]:
        """Get cached or fresh audio device list."""
        cache_key = 'audio_device_cache'
        
        if not st.session_state.get(cache_key):
            try:
                devices = self._safe_call('devices', {'input': [], 'output': []})
                st.session_state[cache_key] = devices
                st.session_state['audio_last_device_scan'] = time.time()
            except Exception as e:
                st.error(f"Failed to get devices: {e}")
                return {'input': [], 'output': []}
        
        return st.session_state.get(cache_key, {'input': [], 'output': []})
    
    def _render_input_device_selection(self, devices: Dict[str, List]):
        """Render input device selection."""
        st.markdown("**Input Device (Microphone)**")
        
        input_devices = devices.get('input', [])
        
        if not input_devices:
            st.warning("No input devices found")
            return
        
        # Create device options
        device_options = ["System Default"]
        for dev in input_devices:
            name = dev.get('name', 'Unknown Device') if isinstance(dev, dict) else str(dev)
            device_id = dev.get('id', '?') if isinstance(dev, dict) else '?'
            option_text = f"{name} (ID: {device_id})"
            device_options.append(option_text)
        
        # Current selection
        current_selection = st.session_state.get('audio_selected_input_device')
        try:
            current_idx = device_options.index(current_selection) if current_selection in device_options else 0
        except:
            current_idx = 0
        
        selected = st.selectbox(
            "Select Input Device",
            device_options,
            index=current_idx,
            key="input_device_selector"
        )
        
        st.session_state['audio_selected_input_device'] = selected
    
    def _render_output_device_selection(self, devices: Dict[str, List]):
        """Render output device selection."""
        st.markdown("**Output Device (Speakers/Headphones)**")
        
        output_devices = devices.get('output', [])
        
        if not output_devices:
            st.warning("No output devices found")
            return
        
        # Create device options
        device_options = ["System Default"]
        for dev in output_devices:
            name = dev.get('name', 'Unknown Device') if isinstance(dev, dict) else str(dev)
            device_id = dev.get('id', '?') if isinstance(dev, dict) else '?'
            option_text = f"{name} (ID: {device_id})"
            device_options.append(option_text)
        
        # Current selection
        current_selection = st.session_state.get('audio_selected_output_device')
        try:
            current_idx = device_options.index(current_selection) if current_selection in device_options else 0
        except:
            current_idx = 0
        
        selected = st.selectbox(
            "Select Output Device",
            device_options,
            index=current_idx,
            key="output_device_selector"
        )
        
        st.session_state['audio_selected_output_device'] = selected
    
    def _render_driver_selection(self, drivers: List[str]):
        """Render audio driver selection."""
        st.markdown("**Audio Driver**")
        
        driver_options = ["default"] + drivers
        
        current_driver = st.session_state.get('audio_audio_driver', 'default')
        try:
            current_idx = driver_options.index(current_driver)
        except:
            current_idx = 0
        
        selected_driver = st.selectbox(
            "Select Audio Driver",
            driver_options,
            index=current_idx,
            help="Audio driver affects latency and compatibility"
        )
        
        st.session_state['audio_audio_driver'] = selected_driver
    
    def _render_audio_parameters(self):
        """Render audio parameter configuration."""
        st.subheader("Audio Parameters")
        
        # Core parameters
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Core Parameters**")
            
            # Sample Rate
            sample_rates = [8000, 16000, 22050, 44100, 48000, 88200, 96000]
            current_rate = st.session_state.get('audio_sample_rate', 48000)
            
            try:
                rate_idx = sample_rates.index(current_rate)
            except:
                rate_idx = sample_rates.index(48000)
            
            selected_rate = st.selectbox(
                "Sample Rate (Hz)",
                sample_rates,
                index=rate_idx
            )
            st.session_state['audio_sample_rate'] = selected_rate
            
            # Buffer Size
            buffer_sizes = [64, 128, 256, 512, 1024, 2048]
            current_buffer = st.session_state.get('audio_buffer_size', 512)
            
            try:
                buffer_idx = buffer_sizes.index(current_buffer)
            except:
                buffer_idx = buffer_sizes.index(512)
            
            selected_buffer = st.selectbox(
                "Buffer Size (samples)",
                buffer_sizes,
                index=buffer_idx
            )
            st.session_state['audio_buffer_size'] = selected_buffer
            
            # Show estimated latency
            latency_ms = (selected_buffer / selected_rate) * 1000
            st.info(f"Estimated latency: {latency_ms:.1f} ms")
        
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
        
        # Test parameters
        st.markdown("---")
        st.markdown("**Test Signal Parameters**")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.session_state['audio_test_frequency'] = st.number_input(
                "Test Frequency (Hz)",
                min_value=50.0,
                max_value=20000.0,
                value=st.session_state.get('audio_test_frequency', 1000.0),
                step=50.0
            )
        
        with col2:
            st.session_state['audio_test_duration'] = st.number_input(
                "Test Duration (seconds)",
                min_value=0.5,
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
    
    def _render_diagnostics_and_test(self):
        """Render audio diagnostics and testing interface."""
        st.subheader("Audio System Diagnostics")
        
        # System check
        if st.button("Run System Check"):
            self._run_system_check()
        
        st.markdown("---")
        
        # Simple audio tests (simulated for now)
        st.markdown("**Audio Tests**")
        st.info("Audio testing functionality would be implemented here.")
        st.info("This requires integration with the actual RoomResponseRecorder.")
        
        # Configuration management
        st.markdown("---")
        self._render_config_management()
    
    def _run_system_check(self):
        """Run a basic system diagnostic."""
        with st.spinner("Running system diagnostics..."):
            results = {}
            
            # Check SDL availability
            results['SDL Available'] = SDL_AVAILABLE
            
            # Check available functions
            results['Functions Available'] = len(self._available_functions)
            
            # Check device availability
            try:
                devices = self._get_audio_devices()
                results['Input Devices'] = len(devices.get('input', []))
                results['Output Devices'] = len(devices.get('output', []))
            except:
                results['Device Scan'] = 'Failed'
            
            # Display results
            st.markdown("**Diagnostic Results:**")
            for check, result in results.items():
                if isinstance(result, bool):
                    icon = "OK" if result else "FAIL"
                    st.write(f"{icon} {check}")
                else:
                    st.write(f"INFO {check}: {result}")
    
    def _render_config_management(self):
        """Render configuration management."""
        st.markdown("**Configuration Management**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Export Config"):
                config = self._build_config()
                config_json = json.dumps(config, indent=2)
                
                st.download_button(
                    "Download Configuration",
                    data=config_json,
                    file_name="audio_config.json",
                    mime="application/json"
                )
        
        with col2:
            if st.button("Reset to Defaults"):
                self._reset_to_defaults()
                st.success("Reset to defaults")
                st.rerun()
    
    def _build_config(self) -> Dict[str, Any]:
        """Build current configuration."""
        return {
            'selected_input_device': st.session_state.get('audio_selected_input_device'),
            'selected_output_device': st.session_state.get('audio_selected_output_device'),
            'sample_rate': st.session_state.get('audio_sample_rate', 48000),
            'buffer_size': st.session_state.get('audio_buffer_size', 512),
            'input_channels': st.session_state.get('audio_input_channels', 1),
            'output_channels': st.session_state.get('audio_output_channels', 2),
            'test_frequency': st.session_state.get('audio_test_frequency', 1000.0),
            'test_duration': st.session_state.get('audio_test_duration', 2.0),
            'test_volume': st.session_state.get('audio_test_volume', 0.3),
            'audio_driver': st.session_state.get('audio_audio_driver', 'default')
        }
    
    def _reset_to_defaults(self):
        """Reset all settings to defaults."""
        defaults = {
            'audio_selected_input_device': None,
            'audio_selected_output_device': None,
            'audio_sample_rate': 48000,
            'audio_buffer_size': 512,
            'audio_input_channels': 1,
            'audio_output_channels': 2,
            'audio_test_frequency': 1000.0,
            'audio_test_duration': 2.0,
            'audio_test_volume': 0.3,
            'audio_audio_driver': 'default'
        }
        
        for key, value in defaults.items():
            st.session_state[key] = value
        
        # Clear cache
        st.session_state['audio_device_cache'] = {}
        st.session_state['audio_last_device_scan'] = 0
    
    def get_current_config(self) -> Dict[str, Any]:
        """Get current configuration for other components."""
        return self._build_config()


# Make the panel available for import
__all__ = ['AudioSettingsPanel']