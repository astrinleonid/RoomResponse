import numpy as np
import wave
import json
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any
from MicTesting import _SDL_AVAILABLE, sdl_audio_core

# try:
#     import sdl_audio_core as sdl_core
#     _SDL_AVAILABLE = True
# except ImportError:
#     sdl = None
#     _SDL_AVAILABLE = False


class RoomResponseRecorder:
    """Clean, refactored room response recorder with unified API"""

    def __init__(self, config_file_path: str = None):
        """
        Initialize the room response recorder from a JSON configuration file

        Args:
            config_file_path: Path to JSON configuration file. If None, tries to load from
                            'recorderConfig.json' in current directory. If that doesn't exist,
                            uses default config.
        """
        # If no config path specified, try to load from default location
        if config_file_path is None:
            default_path = "recorderConfig.json"
            import os
            if os.path.exists(default_path):
                config_file_path = default_path
                print(f"Loading configuration from default location: {default_path}")

        # Default configuration
        default_config = {
            'sample_rate': 48000,
            'pulse_duration': 0.008,
            'pulse_fade': 0.0001,
            'cycle_duration': 0.1,
            'num_pulses': 8,
            'volume': 0.4,
            'pulse_frequency': 1000,
            'impulse_form': 'sine'
        }

        # Multi-channel support (backward compatible default)
        self.input_channels = 1

        # Multi-channel configuration defaults
        self.multichannel_config = {
            'enabled': False,
            'num_channels': 1,
            'channel_names': ['Channel 0'],
            'calibration_channel': None,
            'reference_channel': 0,
            'response_channels': [0],
            'channel_calibration': {}
        }

        # Calibration quality configuration defaults (V2 Refactored - min/max ranges)
        self.calibration_quality_config = {
            # Negative peak (absolute value)
            'min_negative_peak': 0.1,
            'max_negative_peak': 0.95,
            # Positive peak (absolute value)
            'min_positive_peak': 0.0,
            'max_positive_peak': 0.6,
            # Aftershock (absolute value)
            'min_aftershock': 0.0,
            'max_aftershock': 0.3,
            # Configuration
            'aftershock_window_ms': 10.0,
            'aftershock_skip_ms': 2.0,
            'min_valid_cycles': 3
        }

        # Load configuration from file if provided
        file_config = {}  # Initialize outside try block so it's available later
        if config_file_path:
            try:
                with open(config_file_path, 'r') as f:
                    file_config = json.load(f)

                # Extract recorder config if it's nested under 'recorder_config' key
                if 'recorder_config' in file_config:
                    loaded_config = file_config['recorder_config']
                else:
                    loaded_config = file_config

                # Update defaults with loaded values
                for key, value in loaded_config.items():
                    if key in default_config:
                        default_config[key] = value

                # Load multi-channel config (support both 'multichannel' and 'multichannel_config' keys)
                if 'multichannel_config' in file_config:
                    self.multichannel_config.update(file_config['multichannel_config'])
                elif 'multichannel' in file_config:
                    self.multichannel_config.update(file_config['multichannel'])

                # Load calibration quality config (support both 'calibration_quality' and 'calibration_quality_config' keys)
                if 'calibration_quality_config' in file_config:
                    loaded_cal_config = file_config['calibration_quality_config']
                    # Check if it's old V1 format and migrate to V2 if needed
                    if 'cal_min_amplitude' in loaded_cal_config:
                        # Old V1 format detected - migrate to V2
                        print("Info: Migrating calibration_quality_config from V1 to V2 format")
                        self.calibration_quality_config = self._migrate_calibration_config_v1_to_v2(loaded_cal_config)
                    else:
                        # V2 format - use as-is
                        self.calibration_quality_config.update(loaded_cal_config)
                elif 'calibration_quality' in file_config:
                    loaded_cal_config = file_config['calibration_quality']
                    # Check if it's old V1 format and migrate to V2 if needed
                    if 'cal_min_amplitude' in loaded_cal_config:
                        # Old V1 format detected - migrate to V2
                        print("Info: Migrating calibration_quality from V1 to V2 format")
                        self.calibration_quality_config = self._migrate_calibration_config_v1_to_v2(loaded_cal_config)
                    else:
                        # V2 format - use as-is
                        self.calibration_quality_config.update(loaded_cal_config)

            except FileNotFoundError:
                print(f"Warning: Config file '{config_file_path}' not found. Using default configuration.")
            except json.JSONDecodeError as e:
                print(f"Warning: Invalid JSON in config file '{config_file_path}': {e}. Using default configuration.")
            except Exception as e:
                print(f"Warning: Error loading config file '{config_file_path}': {e}. Using default configuration.")

        # Set instance variables from final configuration
        for param, value in default_config.items():
            setattr(self, param, value)

        # Calculated parameters
        self.pulse_samples = int(self.pulse_duration * self.sample_rate)
        self.fade_samples = int(self.pulse_fade * self.sample_rate)
        self.cycle_samples = int(self.cycle_duration * self.sample_rate)
        self.gap_samples = self.cycle_samples - self.pulse_samples
        self.total_duration = self.cycle_duration * self.num_pulses

        # Signal generation
        self.playback_signal = self._generate_complete_signal()

        # Validate configuration
        self._validate_config()

        # Validate multi-channel config if enabled
        if self.multichannel_config.get('enabled', False):
            self._validate_multichannel_config()

        # Load device IDs from config (if provided), otherwise use defaults
        if config_file_path and file_config:
            self.input_device = file_config.get('input_device', -1)
            self.output_device = file_config.get('output_device', -1)
        else:
            self.input_device = -1
            self.output_device = -1

    def get_sdl_core_info(self) -> dict:
        """
        Centralized snapshot of sdl_audio_core state and recorder-related audio settings.
        Surfaces errors per section so the UI can display what's wrong.
        """
        info = {
            "sdl_available": _SDL_AVAILABLE,
            "module_version": None,
            "sdl_version": None,
            "drivers": [],
            "devices": {"input_devices": [], "output_devices": []},
            "device_counts": {"input": 0, "output": 0, "total": 0},
            "engine_stats": None,
            "recorder": {
                "input_device": getattr(self, "input_device", -1),
                "output_device": getattr(self, "output_device", -1),
                "sample_rate": getattr(self, "sample_rate", 48000),
                "volume": getattr(self, "volume", 0.3),
            },
            "errors": {},
            "installation_ok": None,
        }

        if not _SDL_AVAILABLE:
            info["errors"]["import"] = "sdl_audio_core not importable"
            return info

        # Module version
        try:
            info["module_version"] = sdl_audio_core.get_version()  # exposed by binding
        except Exception as e:
            info["errors"]["module_version"] = str(e)

        # SDL version (static)
        try:
            info["sdl_version"] = sdl_audio_core.AudioEngine.get_sdl_version()
        except Exception as e:
            info["errors"]["sdl_version"] = str(e)

        # Drivers (static)
        try:
            info["drivers"] = sdl_audio_core.AudioEngine.get_audio_drivers() or []
        except Exception as e:
            info["errors"]["drivers"] = str(e)

        # Devices
        try:
            devs = sdl_audio_core.list_all_devices() or {}
            in_list = devs.get("input_devices", []) or []
            out_list = devs.get("output_devices", []) or []
            info["devices"]["input_devices"] = in_list
            info["devices"]["output_devices"] = out_list
            info["device_counts"] = {
                "input": len(in_list),
                "output": len(out_list),
                "total": len(in_list) + len(out_list),
            }
        except Exception as e:
            info["errors"]["list_all_devices"] = str(e)

        # Optional installation check (binding returns bool)
        try:
            ok = sdl_audio_core.check_installation()
            info["installation_ok"] = bool(ok)
        except Exception as e:
            info["errors"]["check_installation"] = str(e)

        # Optional: best-effort engine stats
        try:
            eng = sdl_audio_core.AudioEngine()
            if hasattr(eng, "get_stats"):
                stats = eng.get_stats()
                if isinstance(stats, dict):
                    info["engine_stats"] = stats
                else:
                    fields = [
                        "actual_input_sample_rate",
                        "actual_output_sample_rate",
                        "input_latency_ms",
                        "output_latency_ms",
                        "xruns",
                    ]
                    info["engine_stats"] = {f: getattr(stats, f) for f in fields if hasattr(stats, f)} or None
        except Exception as e:
            info["errors"]["engine_stats"] = str(e)

        return info

    def get_device_info_with_channels(self) -> dict:
        """
        Enhanced device info including max_channels per device.

        Returns:
            {
                'input_devices': [
                    {'device_id': 0, 'name': 'Mic', 'max_channels': 2},
                    {'device_id': 1, 'name': 'Interface', 'max_channels': 8},
                    ...
                ],
                'output_devices': [...]
            }
        """
        try:
            devices = sdl_audio_core.list_all_devices()
            # Extract max_channels from device objects
            input_list = []
            for dev in devices.get('input_devices', []):
                # Handle both dict and object formats
                if isinstance(dev, dict):
                    input_list.append({
                        'device_id': dev.get('device_id', -1),
                        'name': dev.get('name', 'Unknown'),
                        'max_channels': dev.get('max_channels', 1)
                    })
                else:
                    input_list.append({
                        'device_id': getattr(dev, 'device_id', -1),
                        'name': getattr(dev, 'name', 'Unknown'),
                        'max_channels': getattr(dev, 'max_channels', 1)
                    })

            output_list = []
            for dev in devices.get('output_devices', []):
                # Handle both dict and object formats
                if isinstance(dev, dict):
                    output_list.append({
                        'device_id': dev.get('device_id', -1),
                        'name': dev.get('name', 'Unknown'),
                        'max_channels': dev.get('max_channels', 1)
                    })
                else:
                    output_list.append({
                        'device_id': getattr(dev, 'device_id', -1),
                        'name': getattr(dev, 'name', 'Unknown'),
                        'max_channels': getattr(dev, 'max_channels', 1)
                    })

            return {'input_devices': input_list, 'output_devices': output_list}
        except Exception as e:
            print(f"Error getting device info: {e}")
            return {'input_devices': [], 'output_devices': []}

    def _migrate_calibration_config_v1_to_v2(self, v1_config: Dict) -> Dict:
        """
        Migrate old V1 calibration config format to V2 format.

        V1 format (CalibrationValidator):
        - cal_min_amplitude, cal_max_amplitude, cal_min_duration_ms,
          cal_max_duration_ms, cal_duration_threshold, cal_double_hit_window_ms,
          cal_double_hit_threshold, cal_tail_start_ms, cal_tail_max_rms_ratio

        V2 format (CalibrationValidatorV2):
        - min_negative_peak, max_negative_peak, max_aftershock_ratio,
          aftershock_window_ms, max_positive_peak_ratio

        Args:
            v1_config: Old V1 format config dict

        Returns:
            V2 format config dict with migrated values
        """
        # Map old V1 keys to new V2 refactored format (min/max ranges)
        v2_config = {
            'min_negative_peak': v1_config.get('cal_min_amplitude', 0.1),
            'max_negative_peak': v1_config.get('cal_max_amplitude', 0.95),
            'min_positive_peak': 0.0,  # New parameter
            'max_positive_peak': 0.6,  # New parameter
            'min_aftershock': 0.0,  # New parameter
            'max_aftershock': 0.3,  # New parameter
            'aftershock_window_ms': 10.0,
            'aftershock_skip_ms': 2.0,  # New parameter
            'min_valid_cycles': v1_config.get('min_valid_cycles', 3)
        }

        print(f"  Migrated V1 to V2 ranges: neg=[{v2_config['min_negative_peak']}, {v2_config['max_negative_peak']}]")

        return v2_config

    def _validate_config(self):
        """Validate the recorder configuration"""
        if self.pulse_samples <= 0:
            raise ValueError("Pulse duration too short")
        if self.gap_samples < 0:
            raise ValueError("Cycle duration shorter than pulse duration")
        # For voice_coil mode, fade can be longer (it's the decay time)
        if self.impulse_form != "voice_coil" and self.fade_samples >= self.pulse_samples // 2:
            raise ValueError("Fade duration too long for pulse duration (only allowed in voice_coil mode)")
        if not 0.0 <= self.volume <= 1.0:
            raise ValueError("Volume must be between 0.0 and 1.0")
        if self.impulse_form not in ["square", "sine", "voice_coil"]:
            raise ValueError("Impulse form must be 'square', 'sine', or 'voice_coil'")

    def _validate_multichannel_config(self):
        """Validate multi-channel configuration"""
        num_ch = self.multichannel_config['num_channels']
        ref_ch = self.multichannel_config['reference_channel']
        cal_ch = self.multichannel_config.get('calibration_channel')

        if num_ch < 1 or num_ch > 32:
            raise ValueError("num_channels must be between 1 and 32")

        if ref_ch < 0 or ref_ch >= num_ch:
            raise ValueError(f"reference_channel {ref_ch} out of range [0, {num_ch-1}]")

        if cal_ch is not None and (cal_ch < 0 or cal_ch >= num_ch):
            raise ValueError(f"calibration_channel {cal_ch} out of range [0, {num_ch-1}]")

        # Ensure channel names list matches num_channels
        if len(self.multichannel_config['channel_names']) != num_ch:
            # Auto-generate names if missing
            self.multichannel_config['channel_names'] = [
                f"Channel {i}" for i in range(num_ch)
            ]

    def _generate_single_pulse(self, exact_samples: int) -> np.ndarray:
        """Generate a single pulse with exact sample count and smooth envelope"""
        if self.impulse_form == "sine":
            t = np.linspace(0, exact_samples / self.sample_rate, exact_samples, endpoint=False)
            pulse = np.sin(2 * np.pi * self.pulse_frequency * t)

            # Apply fade in/out to prevent clicks
            if self.fade_samples > 0:
                fade_in = np.linspace(0, 1, self.fade_samples)
                fade_out = np.linspace(1, 0, self.fade_samples)
                pulse[:self.fade_samples] *= fade_in
                pulse[-self.fade_samples:] *= fade_out

        elif self.impulse_form == "voice_coil":
            # Voice coil actuator impulse: square pulse + negative pull-back
            # pulse_duration controls the main positive pulse
            # fade controls the pull-back negative signal duration

            # Main positive square pulse
            pulse = np.ones(exact_samples)

            # Add pull-back negative signal at the end
            if self.fade_samples > 0 and self.fade_samples < exact_samples:
                # Pull-back signal: negative square wave
                # Position it at the end of the pulse
                pullback_samples = np.zeros(self.fade_samples)
                pullback_impulse_start = self.fade_samples//3
                pullback_impulse_samples = self.fade_samples - pullback_impulse_start
                pullback_samples[pullback_impulse_start:] = np.linspace(-0.5, 0, pullback_impulse_samples)
                pullback_start = exact_samples - self.fade_samples
                pulse[pullback_start:] = pullback_samples
                #-0.5  # Negative pull-back at half amplitude




        else:  # square
            pulse = np.ones(exact_samples)

            # Apply fade in/out to prevent clicks
            if self.fade_samples > 0:
                fade_in = np.linspace(0, 1, self.fade_samples)
                fade_out = np.linspace(1, 0, self.fade_samples)
                pulse[:self.fade_samples] *= fade_in
                pulse[-self.fade_samples:] *= fade_out

        return pulse * self.volume

    def _generate_complete_signal(self) -> list:
        """Generate the complete test signal with all pulses"""
        self.pulse_samples = int(self.pulse_duration * self.sample_rate)
        total_samples = self.cycle_samples * self.num_pulses
        signal = np.zeros(total_samples, dtype=np.float32)
        single_pulse = self._generate_single_pulse(self.pulse_samples)

        # Place pulses at exact cycle intervals
        for i in range(self.num_pulses):
            start_sample = i * self.cycle_samples
            end_sample = start_sample + self.pulse_samples
            if end_sample <= total_samples:
                signal[start_sample:end_sample] = single_pulse

        return signal.tolist()  # Convert for C++ compatibility

    def set_audio_devices(self, input = None, output = None):
        if input is not None:
            self.input_device = input
        if output is not None:
            self.output_device = output

    def get_signal_info(self) -> Dict[str, Any]:
        """Get information about the generated signal"""
        return {
            'sample_rate': self.sample_rate,
            'pulse_duration_ms': self.pulse_duration * 1000,
            'cycle_duration_ms': self.cycle_duration * 1000,
            'gap_duration_ms': (self.cycle_duration - self.pulse_duration) * 1000,
            'total_duration_ms': self.total_duration * 1000,
            'num_pulses': self.num_pulses,
            'total_samples': len(self.playback_signal),
            'volume': self.volume,
            'impulse_form': self.impulse_form
        }

    def print_signal_analysis(self):
        """Print detailed signal analysis"""
        info = self.get_signal_info()
        print(f"\nSignal Analysis:")
        print(f"- Sample rate: {info['sample_rate']} Hz")
        print(f"- Pulse duration: {info['pulse_duration_ms']:.1f} ms ({self.pulse_samples} samples)")
        print(f"- Gap duration: {info['gap_duration_ms']:.1f} ms ({self.gap_samples} samples)")
        print(f"- Cycle duration: {info['cycle_duration_ms']:.1f} ms ({self.cycle_samples} samples)")
        print(f"- Total duration: {info['total_duration_ms']:.1f} ms ({info['total_samples']} samples)")
        print(f"- Number of pulses: {info['num_pulses']}")
        print(f"- Volume: {info['volume']}")
        print(f"- Impulse form: {info['impulse_form']}")
        print(f"- Audio devices: in {self.input_device} out {self.output_device}")

    def list_devices(self) -> Optional[Dict]:
        """List all available audio devices"""
        try:
            devices = sdl_audio_core.list_all_devices()
            print(f"\nAvailable Audio Devices:")
            print(f"{'=' * 60}")

            print(f"Input Devices ({len(devices['input_devices'])}):")
            for device in devices['input_devices']:
                print(f"  [{device.device_id}] {device.name}")

            print(f"\nOutput Devices ({len(devices['output_devices'])}):")
            for device in devices['output_devices']:
                print(f"  [{device.device_id}] {device.name}")

            return devices
        except Exception as e:
            print(f"Error listing devices: {e}")
            return None

    def test_multichannel_recording(self, duration: float = 2.0,
                                   num_channels: int = 2) -> dict:
        """
        Test multi-channel recording.

        Args:
            duration: Recording duration in seconds (not used - uses fixed test signal)
            num_channels: Number of input channels to test

        Returns:
            {
                'success': bool,
                'num_channels': int,
                'samples_per_channel': int,
                'multichannel_data': List[List[float]],  # [channel_idx][samples]
                'channel_stats': [
                    {'max': float, 'rms': float, 'db': float},
                    ...
                ],
                'error_message': str (if failed)
            }
        """
        try:
            # Generate test signal (use existing playback signal or create simple tone)
            test_duration = 0.1  # 100ms chirp
            t = np.arange(int(test_duration * self.sample_rate)) / self.sample_rate
            test_signal = (0.3 * np.sin(2 * np.pi * 1000 * t)).tolist()

            # Record with multi-channel API
            result = sdl_audio_core.measure_room_response_auto_multichannel(
                test_signal,
                volume=0.3,
                input_device=self.input_device,
                output_device=self.output_device,
                input_channels=num_channels
            )

            if not result['success']:
                return result

            # Calculate per-channel statistics
            channel_stats = []
            for ch_data in result['multichannel_data']:
                ch_np = np.array(ch_data)
                max_amp = np.max(np.abs(ch_np))
                rms = np.sqrt(np.mean(ch_np ** 2))
                db = 20 * np.log10(rms) if rms > 0 else -60.0

                channel_stats.append({
                    'max': float(max_amp),
                    'rms': float(rms),
                    'db': float(db)
                })

            result['channel_stats'] = channel_stats
            return result

        except Exception as e:
            return {
                'success': False,
                'error_message': f"Multi-channel test failed: {e}"
            }

    def _record_method_2(self):
        """
        Method 2: Auto device selection with multi-channel support

        Returns:
            If single-channel: np.ndarray
            If multi-channel: Dict[int, np.ndarray] mapping channel index to data
        """
        print("Recording Method 2: Auto Device Selection")

        # Debug output
        print(f"\n{'='*60}")
        print("DEBUG: Recording Configuration")
        print(f"{'='*60}")
        print(f"Device Configuration:")
        print(f"  input_device ID: {self.input_device}")
        print(f"  output_device ID: {self.output_device}")
        print(f"\nMulti-channel Configuration:")
        print(f"  enabled: {self.multichannel_config.get('enabled', False)}")
        print(f"  num_channels: {self.multichannel_config.get('num_channels', 1)}")
        print(f"  calibration_channel: {self.multichannel_config.get('calibration_channel')}")
        print(f"  reference_channel: {self.multichannel_config.get('reference_channel', 0)}")
        print(f"  channel_names: {self.multichannel_config.get('channel_names', [])}")
        print(f"{'='*60}\n")

        try:
            is_multichannel = self.multichannel_config.get('enabled', False)
            num_channels_needed = self.multichannel_config.get('num_channels', 1) if is_multichannel else 1

            print(f"Will attempt to record: {num_channels_needed} channels (multichannel mode: {is_multichannel})")

            if is_multichannel:
                # WORKAROUND: Windows/SDL audio drivers are very picky about channel counts.
                # Many USB interfaces report N channels but only work when opened with exactly N.
                # Strategy: Open device with its maximum reported channel count, then extract
                # the channels we need from the full recording.

                device_max_channels = num_channels_needed
                try:
                    devices_info = self.get_device_info_with_channels()
                    if self.input_device >= 0:
                        for dev in devices_info.get('input_devices', []):
                            if dev['device_id'] == self.input_device:
                                device_max_channels = dev['max_channels']
                                print(f"Device ID {self.input_device} reports max {device_max_channels} channels")
                                break
                except Exception as e:
                    print(f"Warning: Could not query device channels: {e}")

                # STRATEGY: Always try to open with device's maximum channel count first
                # This is most likely to succeed with Windows audio drivers
                actual_device_channels = device_max_channels

                print(f"Opening device with its max capacity: {actual_device_channels} channels")
                print(f"Will extract {num_channels_needed} channels for processing")

                # Use new multi-channel function
                result = sdl_audio_core.measure_room_response_auto_multichannel(
                    self.playback_signal,
                    volume=self.volume,
                    input_device=self.input_device,
                    output_device=self.output_device,
                    input_channels=actual_device_channels
                )
            else:
                # Legacy single-channel function
                result = sdl_audio_core.measure_room_response_auto(
                    self.playback_signal,
                    volume=self.volume,
                    input_device=self.input_device,
                    output_device=self.output_device
                )

            if not result['success']:
                print(f"Measurement failed: {result.get('error_message', 'Unknown error')}")
                return None

            # Process result based on mode
            if is_multichannel:
                # Convert list of lists to dict of numpy arrays
                multichannel_data = result['multichannel_data']
                actual_recorded_channels = len(multichannel_data)

                print(f"Recorded {actual_recorded_channels} channels from device")

                # Extract only the channels we need (0 to num_channels_needed-1)
                if actual_recorded_channels >= num_channels_needed:
                    channel_dict = {
                        ch: np.array(multichannel_data[ch], dtype=np.float32)
                        for ch in range(num_channels_needed)
                    }
                    print(f"Extracted {num_channels_needed} channels for processing")
                else:
                    # Device returned fewer channels than expected
                    print(f"WARNING: Device only returned {actual_recorded_channels} channels, expected at least {num_channels_needed}")
                    channel_dict = {
                        ch: np.array(multichannel_data[ch], dtype=np.float32)
                        for ch in range(actual_recorded_channels)
                    }

                print(f"Processing {len(channel_dict)} channels, {len(list(channel_dict.values())[0])} samples each")
                return channel_dict
            else:
                # Legacy single-channel return
                recorded_data = result['recorded_data']
                print(f"Recorded {result['recorded_samples']} samples")
                return np.array(recorded_data, dtype=np.float32) if recorded_data else None

        except Exception as e:
            print(f"Error in recording: {e}")
            return None

    def _process_recorded_signal(self, recorded_audio) -> Dict[str, Any]:
        """
        Process recorded signal - supports both single and multi-channel

        Args:
            recorded_audio: Either np.ndarray (single-channel) or Dict[int, np.ndarray] (multi-channel)

        Returns:
            Dict with processed data for all channels
        """
        print("Processing recorded signal...")

        is_multichannel = isinstance(recorded_audio, dict)

        if is_multichannel:
            # Multi-channel processing (calibration handled separately in GUI)
            return self._process_multichannel_signal(recorded_audio)
        else:
            return self._process_single_channel_signal(recorded_audio)

    def _extract_cycles(self, audio: np.ndarray) -> np.ndarray:
        """
        Extract cycles from raw audio using simple reshape.

        Pads or trims audio to expected length, then reshapes into cycles.

        Args:
            audio: Raw audio signal

        Returns:
            Cycles array [num_cycles, cycle_samples]
        """
        expected_samples = self.cycle_samples * self.num_pulses

        # Pad or trim to expected length
        if len(audio) < expected_samples:
            padded = np.zeros(expected_samples, dtype=audio.dtype)
            padded[:len(audio)] = audio
            audio = padded
        else:
            audio = audio[:expected_samples]

        # Reshape into cycles
        return audio.reshape(self.num_pulses, self.cycle_samples)

    def _average_cycles(self, cycles: np.ndarray, start_cycle: int = None) -> np.ndarray:
        """
        Average cycles starting from start_cycle.

        Skips initial cycles to allow system settling.

        Args:
            cycles: Cycles array [num_cycles, cycle_samples]
            start_cycle: Index to start averaging from (default: num_pulses // 4)

        Returns:
            Averaged signal [cycle_samples]
        """
        if start_cycle is None:
            start_cycle = max(1, self.num_pulses // 4)

        return np.mean(cycles[start_cycle:], axis=0)

    def _process_single_channel_signal(self, recorded_audio: np.ndarray) -> Dict[str, Any]:
        """Process single-channel standard recording using helper methods"""

        # Extract cycles using helper
        cycles = self._extract_cycles(recorded_audio)

        # Average cycles using helper
        start_cycle = max(1, self.num_pulses // 4)
        room_response = self._average_cycles(cycles, start_cycle)
        print(f"Averaged cycles {start_cycle} to {self.num_pulses - 1}")

        # Extract impulse response
        impulse_response = self._extract_impulse_response(room_response)

        return {
            'raw': recorded_audio,
            'room_response': room_response,
            'impulse': impulse_response
        }

    def _process_multichannel_signal(self, multichannel_audio: Dict[int, np.ndarray]) -> Dict[str, Any]:
        """
        Process multi-channel recording with synchronized alignment using helper methods

        CRITICAL: All channels are aligned using the SAME shift calculated from reference channel
        """
        num_channels = len(multichannel_audio)
        ref_channel = self.multichannel_config.get('reference_channel', 0)

        print(f"Processing {num_channels} channels (reference: {ref_channel})")

        # 1. Process reference channel first using helpers
        ref_cycles = self._extract_cycles(multichannel_audio[ref_channel])
        start_cycle = max(1, self.num_pulses // 4)
        ref_room_response = self._average_cycles(ref_cycles, start_cycle)

        # 2. Find onset in reference channel
        onset_sample = self._find_onset_in_room_response(ref_room_response)
        shift_amount = -onset_sample  # Negative to move onset to beginning
        print(f"Found onset at sample {onset_sample} in reference channel {ref_channel}")

        # 3. Apply cycle averaging to ALL channels and align with SAME shift
        result = {
            'raw': {},
            'room_response': {},
            'impulse': {},
            'metadata': {
                'num_channels': num_channels,
                'reference_channel': ref_channel,
                'onset_sample': onset_sample,
                'shift_applied': shift_amount
            }
        }

        for ch_idx, audio in multichannel_audio.items():
            # Extract and average cycles for this channel using helpers
            cycles = self._extract_cycles(audio)
            room_response = self._average_cycles(cycles, start_cycle)

            # Apply THE SAME shift to this channel (critical for synchronization)
            impulse_response = np.roll(room_response, shift_amount)

            result['raw'][ch_idx] = audio
            result['room_response'][ch_idx] = room_response
            result['impulse'][ch_idx] = impulse_response

            print(f"  Channel {ch_idx}: aligned with shift={shift_amount}")

        return result

    def _find_onset_in_room_response(self, room_response: np.ndarray) -> int:
        """
        Find onset position in a room response (extracted helper method)
        """
        max_index = np.argmax(np.abs(room_response))

        if max_index > 50:
            search_start = max(0, max_index - 100)
            search_window = room_response[search_start:max_index + 50]
            onset_in_window = self._find_sound_onset(search_window)
            onset = search_start + onset_in_window
        else:
            onset = 0

        return onset

    def _extract_impulse_response(self, room_response: np.ndarray) -> np.ndarray:
        """Extract impulse response by finding onset and rotating signal"""
        try:
            max_index = np.argmax(np.abs(room_response))

            if max_index > 50:  # Search for onset if peak not at beginning
                search_start = max(0, max_index - 100)
                search_window = room_response[search_start:max_index + 50]

                onset_in_window = self._find_sound_onset(search_window)
                onset = search_start + onset_in_window

                print(f"Found onset at sample {onset} (peak at {max_index})")

                # Rotate signal to put onset at beginning
                impulse_response = np.concatenate([room_response[onset:], room_response[:onset]])
            else:
                print("Peak near beginning, using room response as impulse response")
                impulse_response = room_response.copy()

            return impulse_response

        except Exception as e:
            print(f"Error extracting impulse response: {e}")
            return room_response.copy()

    def _find_sound_onset(self, audio: np.ndarray, window_size: int = 10,
                          threshold_factor: float = 2) -> int:
        """Find sound onset using moving average and derivative"""
        if len(audio) < window_size * 2:
            return 0

        # Calculate moving RMS
        def moving_rms(signal, window):
            padded = np.pad(signal, window // 2, mode='constant')
            return np.sqrt(np.convolve(padded ** 2, np.ones(window) / window, mode='valid'))

        rms = moving_rms(audio, window_size)
        rms_diff = np.diff(rms)

        # Find significant increase
        background_level = np.std(rms[:window_size]) if len(rms) > window_size else np.std(rms)
        threshold = threshold_factor * background_level

        onset_candidates = np.where(rms_diff > threshold)[0]
        return onset_candidates[0] if len(onset_candidates) > 0 else 0

    # ========================================================================
    # Cycle Alignment Methods (for Calibration Test) - Two-Stage Process
    # ========================================================================

    def align_cycles_by_onset(self, initial_cycles: np.ndarray, validation_results: list,
                             correlation_threshold: float = 0.7) -> dict:
        """
        STEP 5: Align cycles by detecting onset (negative peak) in each cycle.

        Process:
        1. Filter: Keep only VALID cycles (from validation)
        2. Find onset: Locate negative peak in each valid cycle
        3. Align: Shift all cycles so negative peaks align at same position
        4. Cross-correlation check: Verify aligned cycles correlate well
        5. Filter again: Remove cycles with poor correlation after alignment

        Args:
            initial_cycles: 2D array from simple reshape (num_cycles, cycle_samples)
            validation_results: List of validation dicts from CalibrationValidatorV2
            correlation_threshold: Minimum correlation after alignment (default 0.7)

        Returns:
            Dictionary containing:
                - 'aligned_cycles': 2D array of aligned, filtered cycles
                - 'valid_cycle_indices': Original indices of cycles kept
                - 'onset_positions': Onset position found in each original cycle
                - 'aligned_onset_position': Common onset position in aligned cycles
                - 'correlations': Cross-correlation values after alignment
                - 'reference_cycle_idx': Index of reference cycle (in valid set)
        """
        if len(initial_cycles) == 0:
            return {
                'aligned_cycles': np.array([]),
                'valid_cycle_indices': [],
                'onset_positions': [],
                'aligned_onset_position': 0,
                'correlations': [],
                'reference_cycle_idx': 0
            }

        # STEP 1: Filter - Keep only VALID cycles
        valid_indices = [i for i, v in enumerate(validation_results) if v.get('calibration_valid', False)]

        if len(valid_indices) == 0:
            # No valid cycles
            return {
                'aligned_cycles': np.array([]),
                'valid_cycle_indices': [],
                'onset_positions': [],
                'aligned_onset_position': 0,
                'correlations': [],
                'reference_cycle_idx': 0
            }

        valid_cycles = initial_cycles[valid_indices]

        # STEP 2: Find onset (negative peak) in each valid cycle
        onset_positions = []
        for cycle in valid_cycles:
            # Find index of negative peak (minimum value)
            onset_idx = int(np.argmin(cycle))
            onset_positions.append(onset_idx)

        # STEP 3: Determine common onset position
        # Use a position near the beginning (e.g., 100 samples) to ensure onset is at chart start
        # This leaves some space for pre-onset data but puts the peak near the beginning
        target_onset_position = 100  # Position onset at 100 samples (near beginning)
        aligned_onset_position = target_onset_position

        # STEP 4: Align all cycles by shifting to common onset position
        aligned_cycles_list = []
        for i, cycle in enumerate(valid_cycles):
            shift_needed = aligned_onset_position - onset_positions[i]

            # Apply circular shift
            aligned_cycle = np.roll(cycle, shift_needed)
            aligned_cycles_list.append(aligned_cycle)

        aligned_cycles = np.array(aligned_cycles_list)

        # STEP 5: Select reference (highest energy among aligned)
        energies = np.sqrt(np.mean(aligned_cycles ** 2, axis=1))
        reference_idx = int(np.argmax(energies))
        reference_cycle = aligned_cycles[reference_idx]

        # STEP 6: Calculate cross-correlation with reference
        correlations = []
        for i, cycle in enumerate(aligned_cycles):
            if i == reference_idx:
                correlations.append(1.0)
            else:
                # Compute normalized cross-correlation at zero lag
                ref_energy = np.sum(reference_cycle ** 2)
                cyc_energy = np.sum(cycle ** 2)
                cross_product = np.sum(reference_cycle * cycle)

                if ref_energy > 0 and cyc_energy > 0:
                    corr_value = float(cross_product / np.sqrt(ref_energy * cyc_energy))
                else:
                    corr_value = 0.0

                correlations.append(corr_value)

        # STEP 7: Filter by correlation threshold
        final_indices = [i for i, corr in enumerate(correlations) if corr >= correlation_threshold]

        if len(final_indices) == 0:
            # All cycles filtered out
            return {
                'aligned_cycles': np.array([]),
                'valid_cycle_indices': [],
                'onset_positions': onset_positions,
                'aligned_onset_position': aligned_onset_position,
                'correlations': correlations,
                'reference_cycle_idx': reference_idx
            }

        # Return only cycles that passed correlation threshold
        final_aligned_cycles = aligned_cycles[final_indices]
        final_valid_indices = [valid_indices[i] for i in final_indices]
        final_correlations = [correlations[i] for i in final_indices]
        final_onset_positions = [onset_positions[i] for i in final_indices]

        # Adjust reference index to final set
        if reference_idx in final_indices:
            final_reference_idx = final_indices.index(reference_idx)
        else:
            final_reference_idx = 0

        return {
            'aligned_cycles': final_aligned_cycles,
            'valid_cycle_indices': final_valid_indices,
            'onset_positions': final_onset_positions,  # Only for cycles that passed correlation
            'aligned_onset_position': aligned_onset_position,
            'correlations': final_correlations,
            'reference_cycle_idx': final_reference_idx,
            'correlation_threshold': correlation_threshold
        }

    def apply_alignment_to_channel(self, channel_raw: np.ndarray,
                                   alignment_metadata: dict) -> np.ndarray:
        """
        Apply alignment shifts (calculated from calibration channel) to any channel.

        This ensures all channels are aligned uniformly based on calibration channel timing.

        Args:
            channel_raw: Raw audio from any channel (1D array)
            alignment_metadata: Alignment metadata from align_cycles_by_onset()

        Returns:
            2D array of aligned cycles (num_valid_cycles, cycle_samples)
            Only returns cycles that passed validation and correlation filters.
        """
        # Extract alignment info
        valid_cycle_indices = alignment_metadata.get('valid_cycle_indices', [])
        onset_positions = alignment_metadata.get('onset_positions', [])
        aligned_onset_position = alignment_metadata.get('aligned_onset_position', 0)

        if len(valid_cycle_indices) == 0:
            return np.array([])

        # Pad or trim channel to expected length
        expected_samples = self.cycle_samples * self.num_pulses
        if len(channel_raw) < expected_samples:
            padded = np.zeros(expected_samples)
            padded[:len(channel_raw)] = channel_raw
            channel_raw = padded
        else:
            channel_raw = channel_raw[:expected_samples]

        # Extract initial cycles using simple reshape
        initial_cycles = channel_raw.reshape(self.num_pulses, self.cycle_samples)

        # Apply the SAME shifts to this channel's cycles
        aligned_cycles_list = []
        for i, original_idx in enumerate(valid_cycle_indices):
            if original_idx < len(initial_cycles):
                cycle = initial_cycles[original_idx]

                # Calculate shift (same logic as calibration channel)
                if i < len(onset_positions):
                    original_onset = onset_positions[i]
                    shift_needed = aligned_onset_position - original_onset

                    # Apply circular shift
                    aligned_cycle = np.roll(cycle, shift_needed)
                    aligned_cycles_list.append(aligned_cycle)

        return np.array(aligned_cycles_list)

    def _save_wav(self, audio_data: np.ndarray, filename: str):
        """Save audio data to WAV file"""
        wav_file = None
        try:
            # Normalize and convert to int16
            max_val = np.max(np.abs(audio_data))
            if max_val > 0:
                audio_data = audio_data / max_val * 0.95  # Leave headroom

            audio_int16 = (audio_data * 32767).astype(np.int16)

            # Ensure directory exists
            Path(filename).parent.mkdir(parents=True, exist_ok=True)

            wav_file = wave.open(filename, 'w')
            wav_file.setnchannels(1)  # Mono
            wav_file.setsampwidth(2)  # 2 bytes per sample
            wav_file.setframerate(self.sample_rate)
            wav_file.writeframes(audio_int16.tobytes())
            wav_file.close()

            print(f"Audio saved to {filename}")

        except Exception as e:
            print(f"Error saving {filename}: {e}")
        finally:
            # Ensure file is properly closed
            if wav_file is not None:
                try:
                    wav_file.close()
                except:
                    pass

    def take_record(self,
                    output_file: str,
                    impulse_file: str,
                    method: int = 2,
                    mode: str = 'standard',
                    return_processed: bool = False):
        """
        Main API method to record room response

        Args:
            output_file: Filename for raw recording
            impulse_file: Filename for impulse response
            method: Recording method (1=manual, 2=auto, 3=specific devices)
            mode: Recording mode - 'standard' (default) or 'calibration'
            return_processed: If True, return dict with processed data instead of raw audio
                             (used internally, not for external API)

        Returns:
            Standard mode (default):
                Single-channel: np.ndarray (raw audio) - BACKWARD COMPATIBLE
                Multi-channel: Dict[int, np.ndarray] (raw audio per channel) - BACKWARD COMPATIBLE

            Calibration mode:
                Dict[str, Any] with calibration cycle data

            If return_processed=True:
                Dict[str, Any] with all processed data
        """
        # Validate mode parameter
        if mode not in ['standard', 'calibration']:
            raise ValueError(f"Invalid mode: {mode}. Must be 'standard' or 'calibration'")

        # Handle calibration mode (completely separate path)
        if mode == 'calibration':
            return self._take_record_calibration_mode()

        # STANDARD MODE - Continue with existing code (UNCHANGED)
        print(f"\n{'=' * 60}")
        print(f"Room Response Recording")
        print(f"{'=' * 60}")

        try:
            recorded_audio = self._record_method_2()
            if recorded_audio is None:
                print("Recording failed - no data captured")
                return None

            # Process the recorded signal
            processed_data = self._process_recorded_signal(recorded_audio)

            is_multichannel = isinstance(recorded_audio, dict)

            if is_multichannel:
                self._save_multichannel_files(output_file, impulse_file, processed_data)
            else:
                self._save_single_channel_files(output_file, impulse_file, processed_data)

            # Print success summary
            print(f"\n🎉 Recording completed successfully!")

            # BACKWARD COMPATIBLE RETURN
            if return_processed:
                return processed_data  # Internal use only
            else:
                return recorded_audio  # EXISTING BEHAVIOR - returns raw audio

        except Exception as e:
            print(f"Error during recording: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _take_record_calibration_mode(self) -> Dict[str, Any]:
        """
        Calibration mode recording - completely separate implementation.

        Does NOT save files, returns cycle-level data for analysis.

        Returns:
            Dict with:
                - 'calibration_cycles': np.ndarray [N, samples]
                - 'validation_results': List[Dict]
                - 'aligned_multichannel_cycles': Dict[int, np.ndarray]
                - 'alignment_metadata': Dict
                - 'num_valid_cycles': int
                - 'num_aligned_cycles': int

        Raises:
            ValueError: If multi-channel not configured or calibration_channel missing
        """
        from calibration_validator_v2 import CalibrationValidatorV2, QualityThresholds

        print(f"\n{'=' * 60}")
        print(f"Calibration Mode Recording")
        print(f"{'=' * 60}")

        # Validate calibration setup
        if not self.multichannel_config.get('enabled', False):
            raise ValueError("Calibration mode requires multi-channel configuration")

        cal_ch = self.multichannel_config.get('calibration_channel')
        if cal_ch is None:
            raise ValueError("Calibration mode requires 'calibration_channel' in multichannel_config")

        try:
            # Record audio
            recorded_audio = self._record_method_2()
            if recorded_audio is None:
                raise RuntimeError("Recording failed - no data captured")

            if not isinstance(recorded_audio, dict):
                raise ValueError("Calibration mode requires multi-channel recording")

            if cal_ch not in recorded_audio:
                raise ValueError(f"Calibration channel {cal_ch} not found in recorded channels")

            print(f"Processing calibration data (channel {cal_ch})...")

            # Extract cycles from calibration channel using helper method
            cal_raw = recorded_audio[cal_ch]
            initial_cycles = self._extract_cycles(cal_raw)

            # Validate each cycle
            thresholds = QualityThresholds.from_config(self.calibration_quality_config)
            validator = CalibrationValidatorV2(thresholds, self.sample_rate)

            validation_results = []
            for i, cycle in enumerate(initial_cycles):
                validation = validator.validate_cycle(cycle, i)
                validation_dict = {
                    'cycle_index': i,
                    'is_valid': validation.calibration_valid,
                    'calibration_valid': validation.calibration_valid,
                    'calibration_metrics': validation.calibration_metrics,
                    'calibration_failures': validation.calibration_failures
                }
                validation_results.append(validation_dict)

            # Count valid cycles
            num_valid = sum(1 for v in validation_results if v['is_valid'])
            print(f"Calibration validation: {num_valid}/{len(validation_results)} valid cycles")

            # Align cycles by onset
            correlation_threshold = 0.7
            alignment_result = self.align_cycles_by_onset(
                initial_cycles,
                validation_results,
                correlation_threshold=correlation_threshold
            )

            # Apply alignment to all channels
            aligned_multichannel_cycles = {}
            for ch_idx, channel_data in recorded_audio.items():
                aligned_channel = self.apply_alignment_to_channel(
                    channel_data,
                    alignment_result
                )
                aligned_multichannel_cycles[ch_idx] = aligned_channel

            # Calculate number of aligned cycles
            num_aligned = len(alignment_result['valid_cycle_indices'])

            print(f"🎉 Calibration recording completed!")
            print(f"   Valid cycles: {num_valid}/{self.num_pulses}")
            print(f"   Aligned cycles: {num_aligned}")

            return {
                'calibration_cycles': initial_cycles,
                'validation_results': validation_results,
                'aligned_multichannel_cycles': aligned_multichannel_cycles,
                'alignment_metadata': alignment_result,
                'num_valid_cycles': num_valid,
                'num_aligned_cycles': num_aligned,
                'metadata': {
                    'mode': 'calibration',
                    'calibration_channel': cal_ch,
                    'num_channels': len(recorded_audio),
                    'num_cycles': self.num_pulses,
                    'cycle_samples': self.cycle_samples,
                    'correlation_threshold': correlation_threshold
                }
            }

        except Exception as e:
            print(f"Error during calibration recording: {e}")
            import traceback
            traceback.print_exc()
            raise

    def take_record_calibration(self) -> Dict[str, Any]:
        """
        Convenience method for calibration recording.

        Equivalent to: take_record("", "", mode='calibration')

        Returns:
            Dict with calibration cycle data (see _take_record_calibration_mode)
        """
        return self.take_record(
            output_file="",  # Not used in calibration mode
            impulse_file="",  # Not used in calibration mode
            mode='calibration'
        )

    def _save_multichannel_files(self, output_file: str, impulse_file: str, processed_data: Dict):
        """Save multi-channel measurement files"""
        # Get number of channels from processed data
        if 'raw' in processed_data and isinstance(processed_data['raw'], dict):
            num_channels = len(processed_data['raw'])
            channel_indices = sorted(processed_data['raw'].keys())
        else:
            num_channels = len(processed_data['impulse'])
            channel_indices = sorted(processed_data['impulse'].keys())

        print(f"\nSaving {num_channels} channel files...")

        for ch_idx in channel_indices:
            # Generate per-channel filenames
            raw_ch_file = self._make_channel_filename(output_file, ch_idx)
            impulse_ch_file = self._make_channel_filename(impulse_file, ch_idx)

            # Generate room response filename
            output_path = Path(output_file)
            room_base = str(output_path.parent / f"room_{output_path.stem}_room.wav")
            room_ch_file = self._make_channel_filename(room_base, ch_idx)

            # Save files for this channel
            if 'raw' in processed_data and ch_idx in processed_data['raw']:
                self._save_wav(processed_data['raw'][ch_idx], raw_ch_file)
            if 'impulse' in processed_data and ch_idx in processed_data['impulse']:
                self._save_wav(processed_data['impulse'][ch_idx], impulse_ch_file)
            if 'room_response' in processed_data and ch_idx in processed_data['room_response']:
                self._save_wav(processed_data['room_response'][ch_idx], room_ch_file)

            ch_name = self.multichannel_config['channel_names'][ch_idx]
            print(f"  Channel {ch_idx} ({ch_name}): saved 3 files")

    def _save_single_channel_files(self, output_file: str, impulse_file: str, processed_data: Dict):
        """Save single-channel measurement files (legacy)"""
        self._save_wav(processed_data['raw'], output_file)
        self._save_wav(processed_data['impulse'], impulse_file)

        output_path = Path(output_file)
        room_response_file = str(output_path.parent / f"room_{output_path.stem}_room.wav")
        self._save_wav(processed_data['room_response'], room_response_file)

        print(f"- Raw recording: {output_file}")
        print(f"- Impulse response: {impulse_file}")
        print(f"- Room response: {room_response_file}")

        # Signal quality info and diagnostics
        max_amplitude = np.max(np.abs(processed_data['raw']))
        rms_level = np.sqrt(np.mean(processed_data['raw'] ** 2))
        print(f"- Recorded samples: {len(processed_data['raw'])}")
        print(f"- Max amplitude: {max_amplitude:.4f}")
        print(f"- RMS level: {rms_level:.4f}")

        # Audio quality diagnostics
        if max_amplitude < 0.01:
            print("⚠️  WARNING: Very low signal level detected!")
            print("   Try: Increase microphone gain or move closer to speaker")
        elif max_amplitude > 0.95:
            print("⚠️  WARNING: Signal may be clipping!")
            print("   Try: Reduce volume or speaker level")

        if rms_level < 0.005:
            print("⚠️  WARNING: Low RMS level - check audio connections")

    def _make_channel_filename(self, base_filename: str, channel_index: int) -> str:
        """
        Generate filename with channel suffix

        Examples:
            _make_channel_filename("impulse_000_20251025.wav", 0)
            -> "impulse_000_20251025_ch0.wav"
        """
        path = Path(base_filename)
        stem = path.stem
        suffix = path.suffix
        parent = path.parent

        new_filename = f"{stem}_ch{channel_index}{suffix}"
        return str(parent / new_filename)

    # Updated RoomResponseRecorder methods
