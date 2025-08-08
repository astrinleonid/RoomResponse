import numpy as np
import wave
import time
from datetime import datetime
import sdl_audio_core


class FinalSDLRoomResponseRecorder:
    def __init__(self, sample_rate=48000, pulse_samples=480, duration=0.1, num_pulses=5, volume=0.5, impulse_form = "square"):
        self.sample_rate = sample_rate
        self.pulse_duration = duration
        self.cycle_samples = int(duration * sample_rate)
        self.recording_duration = self.pulse_duration * num_pulses
        self.pulse_interval = self.cycle_samples / sample_rate
        self.num_pulses = num_pulses
        self.volume = volume
        self.pulse_frequency = 1000

        self.pulse_samples = pulse_samples
        self.gap_samples = self.cycle_samples - pulse_samples
        self.impulse_form = impulse_form

        # Generate the complete playback signal
        self.playback_signal = self._generate_complete_signal()

    def _generate_single_pulse_exact(self, exact_samples):
        """Generate a single sine wave pulse with exact sample count and smooth envelope"""
        if self.impulse_form == "sine":
            t = np.linspace(0, exact_samples / self.sample_rate, exact_samples, endpoint=False)

            # Create sine wave
            pulse = np.sin(2 * np.pi * self.pulse_frequency * t)

            # Apply envelope to smooth the pulse edges (prevents clicks)
            fade_samples = int(0.001 * self.sample_rate)  # 1ms fade
            if fade_samples > 0 and fade_samples < exact_samples // 2:
                fade_in = np.linspace(0, 1, fade_samples)
                fade_out = np.linspace(1, 0, fade_samples)
                pulse[:fade_samples] *= fade_in
                pulse[-fade_samples:] *= fade_out
        elif self.impulse_form == "square":
            pulse = np.ones(exact_samples)

        return pulse * self.volume

    def _generate_complete_signal(self):
        """Generate the complete signal with all pulses at correct intervals"""
        # Verify our calculations
        assert self.pulse_samples + self.gap_samples == self.cycle_samples, "Pulse + gap should equal cycle"

        # Calculate total samples needed
        total_samples = self.cycle_samples * self.num_pulses

        # Initialize output signal with zeros
        signal = np.zeros(total_samples, dtype=np.float32)

        # Generate single pulse
        single_pulse = self._generate_single_pulse_exact(self.pulse_samples)

        print(f"Generating signal with {self.num_pulses} pulses:")
        print(f"- Pulse duration: {self.pulse_samples / self.sample_rate * 1000:.1f} ms ({self.pulse_samples} samples)")
        print(f"- Gap duration: {self.gap_samples / self.sample_rate * 1000:.1f} ms ({self.gap_samples} samples)")
        print(f"- Full cycle: {self.cycle_samples / self.sample_rate * 1000:.1f} ms ({self.cycle_samples} samples)")
        print(f"- Total duration: {self.recording_duration * 1000:.1f} ms ({total_samples} samples)")

        # Place pulses at exact cycle intervals
        for i in range(self.num_pulses):
            start_sample = i * self.cycle_samples
            end_sample = start_sample + self.pulse_samples

            if end_sample <= total_samples:
                signal[start_sample:end_sample] = single_pulse
                print(
                    f"  Pulse {i + 1}: samples {start_sample}-{end_sample - 1} (at {start_sample / self.sample_rate * 1000:.1f} ms)")
            else:
                print(f"  Warning: Pulse {i + 1} would exceed signal duration, skipping")

        return signal.tolist()  # Convert to list for C++ compatibility

    def list_devices(self):
        """List all available audio devices"""
        try:
            devices = sdl_audio_core.list_all_devices()
            print(f"\nAvailable Audio Devices:")
            print(f"{'=' * 60}")

            print(f"Input Devices ({len(devices['input_devices'])}):")
            for i, device in enumerate(devices['input_devices']):
                print(f"  [{device.device_id}] {device.name}")

            print(f"\nOutput Devices ({len(devices['output_devices'])}):")
            for i, device in enumerate(devices['output_devices']):
                print(f"  [{device.device_id}] {device.name}")

            return devices
        except Exception as e:
            print(f"Error listing devices: {e}")
            return None

    def select_devices_interactive(self):
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

    def auto_select_devices(self):
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

    def record_room_response_method1(self, output_filename="room_response.wav",
                                     response_filename="impulse_response.wav",
                                     interactive=False):
        """Method 1: Using AudioEngine with device setup"""

        print(f"\n{'=' * 60}")
        print("SDL Room Response Recording - Method 1 (Manual Setup)")
        print(f"{'=' * 60}")

        # Check installation
        if not sdl_audio_core.check_installation():
            return None

        # Select devices
        if interactive:
            input_device, output_device = self.select_devices_interactive()
            if not input_device or not output_device:
                print("Device selection failed")
                return None
        else:
            input_device, output_device = self.auto_select_devices()
            if not input_device or not output_device:
                print("Auto device selection failed")
                return None

        try:
            # Create and configure audio engine
            engine = sdl_audio_core.AudioEngine()
            config = sdl_audio_core.AudioEngineConfig()
            config.sample_rate = self.sample_rate
            config.buffer_size = 1024
            config.enable_logging = True

            print(f"Initializing audio engine...")
            if not engine.initialize(config):
                print("Failed to initialize audio engine")
                return None

            print(f"Starting audio engine...")
            if not engine.start():
                print("Failed to start audio engine")
                engine.shutdown()
                return None

            # Set devices
            print(f"Setting up devices...")
            if not engine.set_input_device(input_device.device_id):
                print(f"Failed to set input device: {input_device.name}")
                engine.shutdown()
                return None

            if not engine.set_output_device(output_device.device_id):
                print(f"Failed to set output device: {output_device.name}")
                engine.shutdown()
                return None

            print(f"Starting synchronized recording and playback...")
            print(f"- Signal duration: {len(self.playback_signal) / self.sample_rate:.3f} seconds")
            print(f"- Signal samples: {len(self.playback_signal)}")

            # Start synchronized recording and playback
            max_recording_samples = len(self.playback_signal) * 2  # Extra buffer

            if not engine.start_synchronized_recording_and_playback(
                    self.playback_signal, max_recording_samples):
                print("Failed to start synchronized operation")
                engine.shutdown()
                return None

            # Wait for completion
            duration_seconds = len(self.playback_signal) / self.sample_rate
            timeout_ms = int(duration_seconds * 1000) + 2000  # Add 2 second buffer

            print(f"Waiting for playback completion (timeout: {timeout_ms}ms)...")

            if not engine.wait_for_playback_completion(timeout_ms):
                print("Playback did not complete within timeout")
                engine.stop_synchronized_and_get_data()
                engine.shutdown()
                return None

            # Wait a bit more for any echo/reverb
            print("Waiting for echo/reverb capture...")
            time.sleep(0.2)

            # Get recorded data
            print("Retrieving recorded data...")
            recorded_data = engine.stop_synchronized_and_get_data()

            # Get final statistics
            stats = engine.get_stats()
            print(f"\nRecording Statistics:")
            print(f"- Input samples processed: {stats.input_samples_processed}")
            print(f"- Output samples processed: {stats.output_samples_processed}")
            print(f"- Buffer underruns: {stats.buffer_underruns}")
            print(f"- Buffer overruns: {stats.buffer_overruns}")
            print(f"- Recorded samples: {len(recorded_data)}")

            # Shutdown engine
            engine.shutdown()

            if not recorded_data:
                print("No data was recorded")
                return None

            # Convert to numpy array for processing
            recorded_audio = np.array(recorded_data, dtype=np.float32)

            # Process the recorded signal
            processed_data = self._process_recorded_signal(recorded_audio)

            # Save files
            self._save_wav(processed_data['raw'], output_filename)
            self._save_wav(processed_data['impulse'], response_filename)
            self._save_wav(processed_data['room_response'], "room_response.wav")

            print(f"\nRecording completed successfully!")
            print(f"- Raw recording: {output_filename}")
            print(f"- Impulse response: {response_filename}")
            print(f"- Room response: room_response.wav")

            return recorded_audio

        except Exception as e:
            print(f"Error during recording: {e}")
            return None

    def record_room_response_method2(self, output_filename="room_response.wav",
                                     response_filename="impulse_response.wav"):
        """Method 2: Using convenience function with auto device selection"""

        print(f"\n{'=' * 60}")
        print("SDL Room Response Recording - Method 2 (Auto)")
        print(f"{'=' * 60}")

        try:
            # Use the built-in convenience function
            result = sdl_audio_core.measure_room_response_auto(
                self.playback_signal,
                volume=self.volume
            )

            if not result['success']:
                print(f"Measurement failed: {result.get('error_message', 'Unknown error')}")
                return None

            recorded_data = result['recorded_data']

            print(f"Measurement completed:")
            print(f"- Test signal samples: {result['test_signal_samples']}")
            print(f"- Recorded samples: {result['recorded_samples']}")

            if not recorded_data:
                print("No data was recorded")
                return None

            # Convert to numpy array for processing
            recorded_audio = np.array(recorded_data, dtype=np.float32)

            # Process the recorded signal
            processed_data = self._process_recorded_signal(recorded_audio)

            # Save files
            self._save_wav(processed_data['raw'], output_filename)
            self._save_wav(processed_data['impulse'], response_filename)
            self._save_wav(processed_data['room_response'], "room_response.wav")

            print(f"\nRecording completed successfully!")
            print(f"- Raw recording: {output_filename}")
            print(f"- Impulse response: {response_filename}")
            print(f"- Room response: room_response.wav")

            return recorded_audio

        except Exception as e:
            print(f"Error during recording: {e}")
            return None

    def record_room_response_method3(self, input_device_id, output_device_id,
                                     output_filename="room_response.wav",
                                     response_filename="impulse_response.wav"):
        """Method 3: Using specific device IDs"""

        print(f"\n{'=' * 60}")
        print("SDL Room Response Recording - Method 3 (Specific Devices)")
        print(f"{'=' * 60}")

        try:
            # Use the quick device test function
            result = sdl_audio_core.quick_device_test(
                input_device_id,
                output_device_id,
                self.playback_signal
            )

            if not result['success']:
                print(f"Device test failed: {result.get('error_message', 'Unknown error')}")
                return None

            recorded_data = result['recorded_data']

            print(f"Device test completed:")
            print(f"- Input device: {result['input_device_id']}")
            print(f"- Output device: {result['output_device_id']}")
            print(f"- Recorded samples: {result['samples_recorded']}")

            if not recorded_data:
                print("No data was recorded")
                return None

            # Convert to numpy array for processing
            recorded_audio = np.array(recorded_data, dtype=np.float32)

            # Process the recorded signal
            processed_data = self._process_recorded_signal(recorded_audio)

            # Save files
            self._save_wav(processed_data['raw'], output_filename)
            self._save_wav(processed_data['impulse'], response_filename)
            self._save_wav(processed_data['room_response'], "room_response.wav")

            print(f"\nRecording completed successfully!")
            print(f"- Raw recording: {output_filename}")
            print(f"- Impulse response: {response_filename}")
            print(f"- Room response: room_response.wav")

            return recorded_audio

        except Exception as e:
            print(f"Error during recording: {e}")
            return None

    def _process_recorded_signal(self, recorded_audio):
        """Process the recorded signal to extract room response and impulse response"""
        print("Processing recorded signal...")

        # Calculate expected samples per cycle
        expected_samples = self.cycle_samples * self.num_pulses

        print(f"- Recorded samples: {len(recorded_audio)}")
        print(f"- Expected samples: {expected_samples}")

        if len(recorded_audio) < expected_samples:
            print(f"Warning: Recorded fewer samples than expected")
            # Pad with zeros if necessary
            padded_audio = np.zeros(expected_samples)
            padded_audio[:len(recorded_audio)] = recorded_audio
            recorded_audio = padded_audio

        # Try to reshape into cycles for averaging
        try:
            # Use only the expected number of samples
            signal_data = recorded_audio[:expected_samples]
            resp = signal_data.reshape(self.num_pulses, self.cycle_samples)

            # Skip first few cycles to allow system settling
            start_cycle = max(1, self.num_pulses // 4)
            room_response = np.mean(resp[start_cycle:], axis=0)

            print(f"- Used cycles {start_cycle} to {self.num_pulses - 1} for averaging")

        except Exception as e:
            print(f"Error reshaping data for averaging: {e}")
            # Fallback: use first cycle worth of data
            room_response = recorded_audio[:self.cycle_samples] if len(
                recorded_audio) >= self.cycle_samples else recorded_audio

        # Extract impulse response
        impulse_response = self._extract_impulse_response(room_response)

        return {
            'raw': recorded_audio,
            'room_response': room_response,
            'impulse': impulse_response
        }

    def _extract_impulse_response(self, room_response):
        """Extract impulse response by finding onset and rotating signal"""
        try:
            max_index = np.argmax(np.abs(room_response))

            if max_index > 50:  # Only search for onset if peak is not at beginning
                # Look for onset in a window before the peak
                search_start = max(0, max_index - 100)
                search_window = room_response[search_start:max_index + 50]

                onset_in_window = self.find_sound_onset_derivative(search_window)
                onset = search_start + onset_in_window

                print(f"- Found onset at sample {onset} (peak at {max_index})")

                # Rotate signal to put onset at beginning
                impulse_response = np.concatenate([room_response[onset:], room_response[:onset]])
            else:
                print("- Peak near beginning, using room response as impulse response")
                impulse_response = room_response.copy()

            return impulse_response

        except Exception as e:
            print(f"Error extracting impulse response: {e}")
            return room_response.copy()

    def find_sound_onset_derivative(self, audio, window_size=10, threshold_factor=2):
        """Find sound onset using moving average and derivative"""
        if len(audio) < window_size * 2:
            return 0

        # Calculate moving RMS
        def moving_rms(signal, window):
            padded = np.pad(signal, window // 2, mode='constant')
            return np.sqrt(np.convolve(padded ** 2, np.ones(window) / window, mode='valid'))

        rms = moving_rms(audio, window_size)

        # Calculate derivative of RMS
        rms_diff = np.diff(rms)

        # Find significant increase
        background_level = np.std(rms[:window_size]) if len(rms) > window_size else np.std(rms)
        threshold = threshold_factor * background_level

        onset_candidates = np.where(rms_diff > threshold)[0]

        return onset_candidates[0] if len(onset_candidates) > 0 else 0

    def _save_wav(self, audio_data, filename):
        """Save audio data to WAV file"""
        try:
            # Normalize and convert to int16 for WAV format
            max_val = np.max(np.abs(audio_data))
            if max_val > 0:
                audio_data = audio_data / max_val * 0.95  # Leave headroom

            audio_int16 = (audio_data * 32767).astype(np.int16)

            with wave.open(filename, 'w') as wav_file:
                wav_file.setnchannels(1)  # Mono
                wav_file.setsampwidth(2)  # 2 bytes per sample (int16)
                wav_file.setframerate(self.sample_rate)
                wav_file.writeframes(audio_int16.tobytes())

            print(f"- Audio saved to {filename}")

        except Exception as e:
            print(f"Error saving audio file {filename}: {e}")

    def analyze_signal_timing(self):
        """Analyze the generated signal timing"""
        signal_array = np.array(self.playback_signal)

        print(f"\nSignal analysis:")
        print(f"- Total signal length: {len(signal_array)} samples")
        print(f"- Duration: {len(signal_array) / self.sample_rate:.3f} seconds")

        # Find pulse positions
        threshold = 0.001
        in_pulse = False
        pulse_positions = []

        for i in range(len(signal_array)):
            if not in_pulse and abs(signal_array[i]) > threshold:
                pulse_positions.append(i)
                in_pulse = True
            elif in_pulse and abs(signal_array[i]) <= threshold:
                in_pulse = False

        # Calculate timing verification
        if len(pulse_positions) > 1:
            print(f"\nPulse timing verification:")
            for i in range(len(pulse_positions) - 1):
                cycle_duration = pulse_positions[i + 1] - pulse_positions[i]
                print(f"  Cycle {i + 1}: {cycle_duration} samples ({cycle_duration / self.sample_rate * 1000:.1f} ms)")


def main():
    """Main function demonstrating all three recording methods"""

    # Create recorder instance
    recorder = FinalSDLRoomResponseRecorder(
        sample_rate=48000,  # Match your AudioEngine default
        pulse_samples=480,  # 10ms at 48kHz
        duration=0.1,  # 100ms cycles
        num_pulses=8,  # 8 pulses for good averaging
        volume=0.4,  # Moderate volume to avoid clipping
        impulse_form="sine"
    )

    # Analyze the generated signal
    recorder.analyze_signal_timing()

    # Generate timestamped filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    try:
        print(f"\n{'=' * 60}")
        print("SDL Room Response Recording - Choose Method")
        print(f"{'=' * 60}")
        print("1. Manual device setup (recommended for control)")
        print("2. Automatic device selection (easiest)")
        print("3. Specific device IDs (for known configurations)")
        print("4. Test all methods")

        choice = input("\nSelect option (1-4) or press Enter for method 2: ").strip()

        if choice == "1":
            # Method 1: Manual device setup
            print("\n" + "=" * 60)
            print("Using Method 1: Manual Device Setup")
            print("=" * 60)

            interactive = input("Interactive device selection? (y/n): ").strip().lower() == 'y'

            output_file = f"sdl_method1_{timestamp}.wav"
            impulse_file = f"sdl_impulse1_{timestamp}.wav"

            audio_data = recorder.record_room_response_method1(
                output_file, impulse_file, interactive=interactive
            )

        elif choice == "3":
            # Method 3: Specific device IDs
            print("\n" + "=" * 60)
            print("Using Method 3: Specific Device IDs")
            print("=" * 60)

            # First show available devices
            recorder.list_devices()

            try:
                input_id = int(input("\nEnter input device ID: "))
                output_id = int(input("Enter output device ID: "))

                output_file = f"sdl_method3_{timestamp}.wav"
                impulse_file = f"sdl_impulse3_{timestamp}.wav"

                audio_data = recorder.record_room_response_method3(
                    input_id, output_id, output_file, impulse_file
                )

            except ValueError:
                print("Invalid device ID entered")
                audio_data = None

        elif choice == "4":
            # Test all methods
            print("\n" + "=" * 60)
            print("Testing All Methods")
            print("=" * 60)

            methods_results = {}

            # Method 2 (auto)
            print("\n" + "-" * 40)
            print("Testing Method 2: Auto")
            print("-" * 40)
            try:
                output_file = f"sdl_method2_{timestamp}.wav"
                impulse_file = f"sdl_impulse2_{timestamp}.wav"
                audio_data = recorder.record_room_response_method2(output_file, impulse_file)
                methods_results['method2'] = audio_data is not None
            except Exception as e:
                print(f"Method 2 failed: {e}")
                methods_results['method2'] = False

            # Method 1 (auto device selection)
            print("\n" + "-" * 40)
            print("Testing Method 1: Auto device selection")
            print("-" * 40)
            try:
                output_file = f"sdl_method1_{timestamp}.wav"
                impulse_file = f"sdl_impulse1_{timestamp}.wav"
                audio_data = recorder.record_room_response_method1(
                    output_file, impulse_file, interactive=False
                )
                methods_results['method1'] = audio_data is not None
            except Exception as e:
                print(f"Method 1 failed: {e}")
                methods_results['method1'] = False

            # Method 3 (with first available devices)
            print("\n" + "-" * 40)
            print("Testing Method 3: First available devices")
            print("-" * 40)
            try:
                devices = recorder.list_devices()
                if devices and devices['input_devices'] and devices['output_devices']:
                    input_id = devices['input_devices'][0].device_id
                    output_id = devices['output_devices'][0].device_id

                    output_file = f"sdl_method3_{timestamp}.wav"
                    impulse_file = f"sdl_impulse3_{timestamp}.wav"

                    audio_data = recorder.record_room_response_method3(
                        input_id, output_id, output_file, impulse_file
                    )
                    methods_results['method3'] = audio_data is not None
                else:
                    print("No devices available for Method 3")
                    methods_results['method3'] = False
            except Exception as e:
                print(f"Method 3 failed: {e}")
                methods_results['method3'] = False

            # Print summary
            print(f"\n{'=' * 60}")
            print("Method Test Results:")
            print(f"{'=' * 60}")
            for method, success in methods_results.items():
                status = "✅ SUCCESS" if success else "❌ FAILED"
                print(f"  {method}: {status}")

            successful_methods = [m for m, s in methods_results.items() if s]
            if successful_methods:
                print(f"\nRecommended method: {successful_methods[0]}")
            else:
                print("\n⚠️  No methods succeeded. Check your audio setup.")

            audio_data = True  # Mark as completed for summary

        else:
            # Method 2: Automatic (default)
            print("\n" + "=" * 60)
            print("Using Method 2: Automatic Device Selection")
            print("=" * 60)

            output_file = f"sdl_method2_{timestamp}.wav"
            impulse_file = f"sdl_impulse2_{timestamp}.wav"

            audio_data = recorder.record_room_response_method2(output_file, impulse_file)

        # Print final summary
        if audio_data is not None:
            print(f"\n{'=' * 60}")
            print("🎉 Recording Session Completed Successfully!")
            print(f"{'=' * 60}")
            if choice != "4":
                print(f"- Signal duration: {len(recorder.playback_signal) / recorder.sample_rate:.3f} seconds")
                print(f"- Sample rate: {recorder.sample_rate} Hz")
                print(f"- Pulses: {recorder.num_pulses}")
                print(f"- Volume: {recorder.volume}")

                if hasattr(audio_data, '__len__'):
                    print(f"- Recorded samples: {len(audio_data)}")
                    print(f"- Recording duration: {len(audio_data) / recorder.sample_rate:.3f} seconds")
                    max_amplitude = np.max(np.abs(audio_data))
                    rms_level = np.sqrt(np.mean(audio_data ** 2))
                    print(f"- Max amplitude: {max_amplitude:.4f}")
                    print(f"- RMS level: {rms_level:.4f}")
        else:
            print(f"\n{'=' * 60}")
            print("❌ Recording Session Failed!")
            print(f"{'=' * 60}")
            print("Troubleshooting suggestions:")
            print("1. Check that your microphone and speakers are working")
            print("2. Verify audio device permissions")
            print("3. Make sure no other applications are using audio devices")
            print("4. Try different devices with interactive selection")
            print("5. Check volume levels (not too high to avoid clipping)")

    except KeyboardInterrupt:
        print("\n\n⚠️  Recording interrupted by user")

    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        print("\nIf this error persists:")
        print("1. Check your SDL audio core compilation")
        print("2. Verify all dependencies are installed")
        print("3. Try running the installation check:")
        print("   python -c 'import sdl_audio_core; sdl_audio_core.check_installation()'")

    print(f"\n{'=' * 60}")
    print("Session completed. Thank you for using SDL Room Response Recorder!")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()