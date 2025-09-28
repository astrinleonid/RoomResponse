import numpy as np
import wave
import time
import json
import threading
import queue
import time
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
from MicTesting import _SDL_AVAILABLE, AudioRecorder, AudioRecordingWorker, AudioProcessingWorker, AudioProcessor, sdl_audio_core

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
            config_file_path: Path to JSON configuration file. If None, uses default config.
        """
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

        # Load configuration from file if provided
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

    def _validate_config(self):
        """Validate the recorder configuration"""
        if self.pulse_samples <= 0:
            raise ValueError("Pulse duration too short")
        if self.gap_samples < 0:
            raise ValueError("Cycle duration shorter than pulse duration")
        if self.fade_samples >= self.pulse_samples // 2:
            raise ValueError("Fade duration too long for pulse duration")
        if not 0.0 <= self.volume <= 1.0:
            raise ValueError("Volume must be between 0.0 and 1.0")
        if self.impulse_form not in ["square", "sine"]:
            raise ValueError("Impulse form must be 'square' or 'sine'")

    def _generate_single_pulse(self, exact_samples: int) -> np.ndarray:
        """Generate a single pulse with exact sample count and smooth envelope"""
        if self.impulse_form == "sine":
            t = np.linspace(0, exact_samples / self.sample_rate, exact_samples, endpoint=False)
            pulse = np.sin(2 * np.pi * self.pulse_frequency * t)
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


    def _record_method_2(self) -> Optional[np.ndarray]:
        """Method 2: Auto device selection with convenience function"""
        print("Recording Method 2: Auto Device Selection")

        try:
            result = sdl_audio_core.measure_room_response_auto(
                self.playback_signal,
                volume=self.volume,
                input_device =self.input_device,
                output_device =self.output_device
            )

            if not result['success']:
                print(f"Measurement failed: {result.get('error_message', 'Unknown error')}")
                return None

            recorded_data = result['recorded_data']
            print(f"Recorded {result['recorded_samples']} samples")

            return np.array(recorded_data, dtype=np.float32) if recorded_data else None

        except Exception as e:
            print(f"Error in method 2: {e}")
            return None

    def _process_recorded_signal(self, recorded_audio: np.ndarray) -> Dict[str, np.ndarray]:
        """Process recorded signal to extract room response and impulse response"""
        print("Processing recorded signal...")

        expected_samples = self.cycle_samples * self.num_pulses

        # Pad or trim to expected length
        if len(recorded_audio) < expected_samples:
            print(f"Warning: Recorded {len(recorded_audio)} samples, expected {expected_samples}")
            padded_audio = np.zeros(expected_samples)
            padded_audio[:len(recorded_audio)] = recorded_audio
            recorded_audio = padded_audio

        # Extract room response by averaging cycles
        try:
            signal_data = recorded_audio[:expected_samples]
            reshaped = signal_data.reshape(self.num_pulses, self.cycle_samples)

            # Skip first few cycles for system settling
            start_cycle = max(1, self.num_pulses // 4)
            room_response = np.mean(reshaped[start_cycle:], axis=0)
            print(f"Averaged cycles {start_cycle} to {self.num_pulses - 1}")

        except Exception as e:
            print(f"Error reshaping data: {e}")
            room_response = recorded_audio[:self.cycle_samples] if len(
                recorded_audio) >= self.cycle_samples else recorded_audio

        # Extract impulse response
        impulse_response = self._extract_impulse_response(room_response)

        return {
            'raw': recorded_audio,
            'room_response': room_response,
            'impulse': impulse_response
        }

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
                    method: int = 2) -> Optional[np.ndarray]:
        """
        Main API method to record room response

        Args:
            output_file: Filename for raw recording
            impulse_file: Filename for impulse response
            method: Recording method (1=manual, 2=auto, 3=specific devices)
            interactive: Whether to use interactive device selection (method 1 only)


        Returns:
            Recorded audio data as numpy array, or None if failed
        """
        print(f"\n{'=' * 60}")
        print(f"Room Response Recording - Method {method}")
        print(f"{'=' * 60}")

        try:
            recorded_audio = self._record_method_2()
            if recorded_audio is None:
                print("Recording failed - no data captured")
                return None

            # Process the recorded signal
            processed_data = self._process_recorded_signal(recorded_audio)

            # Save files
            self._save_wav(processed_data['raw'], output_file)
            self._save_wav(processed_data['impulse'], impulse_file)

            # Also save room response - create proper filename
            output_path = Path(output_file)
            room_response_file = str(output_path.parent / f"room_{output_path.stem}_room.wav")
            self._save_wav(processed_data['room_response'], room_response_file)

            # Print success summary
            print(f"\n🎉 Recording completed successfully!")
            print(f"- Raw recording: {output_file}")
            print(f"- Impulse response: {impulse_file}")
            print(f"- Room response: {room_response_file}")

            # Signal quality info and diagnostics
            max_amplitude = np.max(np.abs(recorded_audio))
            rms_level = np.sqrt(np.mean(recorded_audio ** 2))
            print(f"- Recorded samples: {len(recorded_audio)}")
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

            return recorded_audio

        except Exception as e:
            print(f"Error during recording: {e}")
            return None

    # Updated RoomResponseRecorder methods
    def test_mic(self, duration: float = 10.0, chunk_duration: float = 0.1):
        """
        Simple microphone test with real-time RMS monitoring using reusable components

        Args:
            duration: Total test duration in seconds
            chunk_duration: Audio chunk duration for processing in seconds
        """
        print(f"\nTesting microphone for {duration} seconds...")
        print("Speak into the microphone to see RMS levels")
        print("Press Ctrl+C to stop early\n")

        audio_queue = queue.Queue()

        try:
            # Create recorder and workers
            with AudioRecorder(self.sample_rate, self.input_device, enable_logging=False) as recorder:
                recording_worker = AudioRecordingWorker(recorder, audio_queue, chunk_duration)
                processing_worker = AudioProcessingWorker(audio_queue)

                # Start workers
                recording_worker.start()
                processing_worker.start()

                print("Recording started...")

                try:
                    time.sleep(duration)
                except KeyboardInterrupt:
                    print("\nStopping test...")

                # Stop workers
                recording_worker.stop()
                processing_worker.stop()

        except Exception as e:
            print(f"Error during microphone test: {e}")

        print("\nMicrophone test completed.")
        print("\nTips:")
        print("- Normal speech: -40 to -20 dB")
        print("- Loud speech: -20 to -10 dB")
        print("- Too quiet: below -50 dB")
        print("- Too loud/clipping: above -6 dB")

if __name__ == "__main__":
    r = RoomResponseRecorder()

    r.list_devices()
    print(r.get_sdl_core_info())
    print("\n\n")

    input_d = int(input("Please enter input device number: "))
    output_d = int(input("Please enter output device number: "))
    r.set_audio_devices(input_d, output_d)
    r.test_mic(duration=30.0)
    r.take_record("test.wav", "test_impulse.wav")