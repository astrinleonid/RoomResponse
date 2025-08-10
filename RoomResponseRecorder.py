import numpy as np
import wave
import time
from datetime import datetime
from pathlib import Path
import sdl_audio_core
from typing import Optional, Tuple, Dict, Any


class RoomResponseRecorder:
    """Clean, refactored room response recorder with unified API"""

    def __init__(self,
                 sample_rate: int = 48000,
                 pulse_duration: float = 0.008,
                 pulse_fade: float = 0.0001,
                 cycle_duration: float = 0.1,
                 num_pulses: int = 8,
                 volume: float = 0.4,
                 impulse_form: str = "square"):
        """
        Initialize the room response recorder

        Args:
            sample_rate: Audio sample rate in Hz
            pulse_duration: Duration of each pulse in seconds
            pulse_fade: Fade in/out duration for pulses in seconds
            cycle_duration: Time between pulse starts in seconds
            num_pulses: Number of pulses in the test signal
            volume: Playback volume (0.0 to 1.0)
            impulse_form: Type of pulse - "square" or "sine"
        """
        self.sample_rate = sample_rate
        self.pulse_duration = pulse_duration
        self.pulse_fade = pulse_fade
        self.cycle_duration = cycle_duration
        self.num_pulses = num_pulses
        self.volume = volume
        self.impulse_form = impulse_form

        # Calculated parameters
        self.pulse_samples = int(pulse_duration * sample_rate)
        self.fade_samples = int(pulse_fade * sample_rate)
        self.cycle_samples = int(cycle_duration * sample_rate)
        self.gap_samples = self.cycle_samples - self.pulse_samples
        self.total_duration = cycle_duration * num_pulses

        # Signal generation
        self.pulse_frequency = 1000  # Hz for sine wave pulses
        self.playback_signal = self._generate_complete_signal()

        # Validate configuration
        self._validate_config()

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

    def _select_devices_interactive(self) -> Tuple[Optional[Any], Optional[Any]]:
        """Interactively select input and output devices"""
        devices = self.list_devices()
        if not devices:
            return None, None

        try:
            input_devices = devices['input_devices']
            output_devices = devices['output_devices']

            # Select input device
            print(f"\nSelect Input Device:")
            for i, device in enumerate(input_devices):
                print(f"  {i}: [{device.device_id}] {device.name}")
            input_choice = int(input(f"Enter choice (0-{len(input_devices) - 1}): "))
            selected_input = input_devices[input_choice]

            # Select output device
            print(f"\nSelect Output Device:")
            for i, device in enumerate(output_devices):
                print(f"  {i}: [{device.device_id}] {device.name}")
            output_choice = int(input(f"Enter choice (0-{len(output_devices) - 1}): "))
            selected_output = output_devices[output_choice]

            print(f"\nSelected devices:")
            print(f"  Input: {selected_input.name}")
            print(f"  Output: {selected_output.name}")

            return selected_input, selected_output

        except (ValueError, IndexError) as e:
            print(f"Invalid selection: {e}")
            return None, None

    def _auto_select_devices(self) -> Tuple[Optional[Any], Optional[Any]]:
        """Automatically select the first available devices"""
        devices = self.list_devices()
        if not devices:
            return None, None

        input_devices = devices['input_devices']
        output_devices = devices['output_devices']

        if input_devices and output_devices:
            selected_input = input_devices[0]
            selected_output = output_devices[0]
            print(f"\nAuto-selected devices:")
            print(f"  Input: {selected_input.name}")
            print(f"  Output: {selected_output.name}")
            return selected_input, selected_output
        else:
            print("No suitable devices found for auto-selection")
            return None, None

    def _record_method_1(self, interactive: bool = False) -> Optional[np.ndarray]:
        """Method 1: Manual AudioEngine setup with device selection"""
        print("Recording Method 1: Manual AudioEngine Setup")

        # Check installation
        if not sdl_audio_core.check_installation():
            return None

        # Select devices
        if interactive:
            input_device, output_device = self._select_devices_interactive()
        else:
            input_device, output_device = self._auto_select_devices()

        if not input_device or not output_device:
            print("Device selection failed")
            return None

        try:
            # Create and configure audio engine
            engine = sdl_audio_core.AudioEngine()
            config = sdl_audio_core.AudioEngineConfig()
            config.sample_rate = self.sample_rate
            config.buffer_size = 1024
            config.enable_logging = True

            # Initialize and start engine
            if not engine.initialize(config):
                print("Failed to initialize audio engine")
                return None

            if not engine.start():
                print("Failed to start audio engine")
                engine.shutdown()
                return None

            # Set devices
            if not engine.set_input_device(input_device.device_id):
                print(f"Failed to set input device: {input_device.name}")
                engine.shutdown()
                return None

            if not engine.set_output_device(output_device.device_id):
                print(f"Failed to set output device: {output_device.name}")
                engine.shutdown()
                return None

            # Start synchronized recording and playback
            max_recording_samples = len(self.playback_signal) * 2

            if not engine.start_synchronized_recording_and_playback(
                    self.playback_signal, max_recording_samples):
                print("Failed to start synchronized operation")
                engine.shutdown()
                return None

            # Wait for completion
            duration_seconds = len(self.playback_signal) / self.sample_rate
            timeout_ms = int(duration_seconds * 1000) + 2000

            if not engine.wait_for_playback_completion(timeout_ms):
                print("Playback did not complete within timeout")
                engine.stop_synchronized_and_get_data()
                engine.shutdown()
                return None

            # Wait for echo/reverb and get data
            time.sleep(0.2)
            recorded_data = engine.stop_synchronized_and_get_data()

            # Get statistics
            stats = engine.get_stats()
            print(f"Recording stats: {len(recorded_data)} samples, "
                  f"underruns: {stats.buffer_underruns}, overruns: {stats.buffer_overruns}")

            engine.shutdown()

            return np.array(recorded_data, dtype=np.float32) if recorded_data else None

        except Exception as e:
            print(f"Error in method 1: {e}")
            return None

    def _record_method_2(self) -> Optional[np.ndarray]:
        """Method 2: Auto device selection with convenience function"""
        print("Recording Method 2: Auto Device Selection")

        try:
            result = sdl_audio_core.measure_room_response_auto(
                self.playback_signal,
                volume=self.volume
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

    def _record_method_3(self, input_device_id: int, output_device_id: int) -> Optional[np.ndarray]:
        """Method 3: Specific device IDs"""
        print(f"Recording Method 3: Specific Devices (in:{input_device_id}, out:{output_device_id})")

        try:
            result = sdl_audio_core.quick_device_test(
                input_device_id,
                output_device_id,
                self.playback_signal
            )

            if not result['success']:
                print(f"Device test failed: {result.get('error_message', 'Unknown error')}")
                return None

            recorded_data = result['recorded_data']
            print(f"Recorded {result['samples_recorded']} samples")

            return np.array(recorded_data, dtype=np.float32) if recorded_data else None

        except Exception as e:
            print(f"Error in method 3: {e}")
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
                    method: int = 2,
                    interactive: bool = False,
                    input_device_id: Optional[int] = None,
                    output_device_id: Optional[int] = None) -> Optional[np.ndarray]:
        """
        Main API method to record room response

        Args:
            output_file: Filename for raw recording
            impulse_file: Filename for impulse response
            method: Recording method (1=manual, 2=auto, 3=specific devices)
            interactive: Whether to use interactive device selection (method 1 only)
            input_device_id: Input device ID (method 3 only)
            output_device_id: Output device ID (method 3 only)

        Returns:
            Recorded audio data as numpy array, or None if failed
        """
        print(f"\n{'=' * 60}")
        print(f"Room Response Recording - Method {method}")
        print(f"{'=' * 60}")

        try:
            # Record using specified method
            if method == 1:
                recorded_audio = self._record_method_1(interactive=interactive)
            elif method == 2:
                recorded_audio = self._record_method_2()
            elif method == 3:
                if input_device_id is None or output_device_id is None:
                    # Interactive device selection for method 3
                    devices = self.list_devices()
                    if devices:
                        try:
                            input_device_id = int(input("\nEnter input device ID: "))
                            output_device_id = int(input("Enter output device ID: "))
                        except ValueError:
                            print("Invalid device ID entered")
                            return None
                    else:
                        print("Could not list devices")
                        return None
                recorded_audio = self._record_method_3(input_device_id, output_device_id)
            else:
                print(f"Invalid method {method}. Use 1, 2, or 3.")
                return None

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


def main():
    """Example usage of the refactored recorder"""
    import sys

    # Check for interactive flag
    interactive_mode = '--interactive' in sys.argv or '-i' in sys.argv

    # Create recorder with default settings
    recorder = RoomResponseRecorder(
        sample_rate=48000,
        pulse_duration=0.008,
        cycle_duration=0.1,
        num_pulses=8,
        volume=0.4,
        impulse_form="square"
    )

    # Print signal information
    recorder.print_signal_analysis()

    # Generate filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if interactive_mode:
        # Interactive mode - let user choose method and devices
        print(f"\n{'=' * 60}")
        print("INTERACTIVE MODE - Recording Method Selection:")
        print("1. Manual device setup (with optional interactive selection)")
        print("2. Automatic device selection (recommended)")
        print("3. Specific device IDs")
        print(f"{'=' * 60}")

        try:
            choice = input("Select method (1-3) or press Enter for method 2: ").strip()
            method = int(choice) if choice else 2

            if method not in [1, 2, 3]:
                raise ValueError("Invalid method")

            # Generate output filenames
            output_file = f"response_{timestamp}_method{method}.wav"
            impulse_file = f"impulse_{timestamp}_method{method}.wav"

            # Record based on method
            if method == 1:
                interactive = input("Use interactive device selection? (y/n): ").strip().lower() == 'y'
                audio_data = recorder.take_record(output_file, impulse_file, method=1, interactive=interactive)
            else:
                audio_data = recorder.take_record(output_file, impulse_file, method=method)

        except (ValueError, KeyboardInterrupt) as e:
            print(f"\nSession cancelled: {e}")
            return

    else:
        # Default mode - use method 2 automatically
        print(f"\n{'=' * 60}")
        print("AUTOMATIC MODE - Using default audio devices")
        print(f"{'=' * 60}")
        print("(Use --interactive or -i flag for device selection)")

        output_file = f"response_{timestamp}_auto.wav"
        impulse_file = f"impulse_{timestamp}_auto.wav"

        # Record with method 2 (auto)
        audio_data = recorder.take_record(output_file, impulse_file, method=2)

    # Print final status
    if audio_data is not None:
        print(f"\n✅ Recording completed successfully!")
        print(f"Files saved: {output_file}, {impulse_file}")
    else:
        print(f"\n❌ Recording failed!")
        print("Troubleshooting:")
        print("1. Check audio device connections")
        print("2. Verify device permissions")
        print("3. Try interactive mode: python RoomResponseRecorder.py --interactive")

    print(f"\n{'=' * 60}")
    print("Recording session complete!")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()