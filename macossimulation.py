import numpy as np
import sounddevice as sd
import wave
import threading
from datetime import datetime


class RoomResponseRecorder:
    def __init__(self, sample_rate=48000, pulse_samples=10, duration = 0.1, num_pulses = 5, volume = 1):
        self.sample_rate = sample_rate
        self.pulse_duration = duration
        self.cycle_samples = int(duration * sample_rate)
        self.recording_duration = self.pulse_duration * num_pulses  # Total recording time
        self.pulse_interval = self.cycle_samples / sample_rate  # 100ms between pulses (center to center)
        self.num_pulses = num_pulses
        self.volume = volume
        self.pulse_frequency = 1000

        self.pulse_samples = pulse_samples  # Exactly 10ms at 44.1kHz
        self.gap_samples = self.cycle_samples - pulse_samples  # Exactly 90ms at 44.1kHz

        # Generate the complete playback signal with all pulses
        self.playback_signal = self._generate_complete_signal()

    def _generate_single_pulse_exact(self, exact_samples):
        """Generate a single sine wave pulse with exact sample count and smooth envelope"""
        t = np.linspace(0, exact_samples / self.sample_rate, exact_samples, endpoint=False)

        # Create sine wave
        pulse = np.sin(2 * np.pi * self.pulse_frequency * t)

        # Apply envelope to smooth the pulse edges (prevents clicks)
        fade_samples = int(0.001 * self.sample_rate)  # 1ms fade (44 samples at 44.1kHz)
        if fade_samples > 0 and fade_samples < exact_samples // 2:
            fade_in = np.linspace(0, 1, fade_samples)
            fade_out = np.linspace(1, 0, fade_samples)
            pulse[:fade_samples] *= fade_in
            pulse[-fade_samples:] *= fade_out

        return np.ones(exact_samples) * self.volume
        return pulse

    def _generate_complete_signal(self):
        """Generate the complete signal with all 5 pulses at correct intervals"""
        # Fixed timing parameters for exact sample counts


        # Verify our calculations
        assert self.pulse_samples + self.gap_samples == self.cycle_samples, "Pulse + gap should equal cycle"

        # Calculate total samples needed
        total_samples = self.cycle_samples * self.num_pulses

        # Initialize output signal with zeros
        signal = np.zeros(total_samples)

        # Generate single pulse with exact sample count
        single_pulse = self._generate_single_pulse_exact(self.pulse_samples)

        print(f"Generating signal with {self.num_pulses} pulses:")
        print(f"- Pulse duration: 10.0 ms ({self.pulse_samples} samples)")
        print(f"- Gap duration: 90.0 ms ({self.gap_samples} samples)")
        print(f"- Full cycle: 100.0 ms ({self.cycle_samples} samples)")
        print(f"- Total duration: {self.recording_duration * 1000:.1f} ms ({total_samples} samples)")

        # Place pulses at exact cycle intervals
        for i in range(self.num_pulses):
            start_sample = i * self.cycle_samples
            end_sample = start_sample + self.pulse_samples

            # Check if pulse fits within the signal duration
            if end_sample <= total_samples:
                signal[start_sample:end_sample] = single_pulse
                print(
                    f"  Pulse {i + 1}: samples {start_sample}-{end_sample - 1} (at {start_sample / self.sample_rate * 1000:.1f} ms)")
            else:
                print(f"  Warning: Pulse {i + 1} would exceed signal duration, skipping")

        return signal

    def _find_devices(self):
        """Find and select audio devices, preferring MacBook Pro devices"""
        try:
            devices = sd.query_devices()
            print("Available audio devices:")

            input_devices = []
            output_devices = []
            selected_input = None
            selected_output = None

            for i, device in enumerate(devices):
                device_name = device['name']
                max_in = device['max_input_channels']
                max_out = device['max_output_channels']

                print(f"  {i}: {device_name}")
                print(f"     Input channels: {max_in}, Output channels: {max_out}")

                # Collect input devices
                if max_in > 0:
                    input_devices.append((i, device_name, device))
                    # Prefer MacBook Pro Microphone
                    if "MacBook Pro" in device_name and "Microphone" in device_name:
                        selected_input = (i, device_name, device)
                        print(f"     *** Selected as preferred input device ***")

                # Collect output devices
                if max_out > 0:
                    output_devices.append((i, device_name, device))
                    # Prefer MacBook Pro Speakers
                    if "MacBook Pro" in device_name and "Speakers" in device_name:
                        selected_output = (i, device_name, device)
                        print(f"     *** Selected as preferred output device ***")

            # If no MacBook Pro devices found, use defaults
            if selected_input is None and input_devices:
                selected_input = input_devices[0]
                print(f"\nNo MacBook Pro microphone found, using: {selected_input[1]}")

            if selected_output is None and output_devices:
                selected_output = output_devices[0]
                print(f"No MacBook Pro speakers found, using: {selected_output[1]}")

            return selected_input, selected_output, input_devices, output_devices

        except Exception as e:
            print(f"Error querying devices: {e}")
            return None, None, [], []

    def _select_devices_interactively(self, input_devices, output_devices):
        """Allow user to select devices interactively"""
        print(f"\nInput devices:")
        for i, (idx, name, device) in enumerate(input_devices):
            print(f"  {i}: {name}")

        print(f"\nOutput devices:")
        for i, (idx, name, device) in enumerate(output_devices):
            print(f"  {i}: {name}")

        # Get user selection
        try:
            input_choice = int(input(f"\nSelect input device (0-{len(input_devices) - 1}): "))
            output_choice = int(input(f"Select output device (0-{len(output_devices) - 1}): "))

            selected_input = input_devices[input_choice]
            selected_output = output_devices[output_choice]

            print(f"\nSelected input: {selected_input[1]}")
            print(f"Selected output: {selected_output[1]}")

            return selected_input, selected_output

        except (ValueError, IndexError):
            print("Invalid selection, using defaults")
            return input_devices[0] if input_devices else None, output_devices[0] if output_devices else None

    def record_room_response(self, output_filename="room_response.wav", response_filename = "impulse_response.wav", interactive=False):
        """Record room response while playing the complete pulse sequence"""
        print(f"\nChecking audio devices...")

        # Find devices
        selected_input, selected_output, input_devices, output_devices = self._find_devices()

        if not selected_input or not selected_output:
            print("Error: Could not find suitable input/output devices")
            return None

        # Allow interactive device selection
        if interactive:
            selected_input, selected_output = self._select_devices_interactively(input_devices, output_devices)

        input_idx, input_name, input_device = selected_input
        output_idx, output_name, output_device = selected_output

        print(f"\nUsing devices:")
        print(f"  Input: {input_name} (device {input_idx})")
        print(f"  Output: {output_name} (device {output_idx})")

        print(f"\nStarting synchronized recording and playback...")
        print(f"Output file: {output_filename}")

        try:
            # Method 1: Try playrec with device indices and explicit mapping
            frames = len(self.playback_signal)  # Number of frames to record

            # Set the devices before recording
            sd.default.device = (input_idx, output_idx)
            sd.default.channels = (1, 1)  # 1 input channel, 1 output channel

            recording = sd.playrec(
                self.playback_signal.reshape(-1, 1),  # Playback signal (mono)
                samplerate=self.sample_rate,
                dtype=np.float32,
                # frames=frames  # Explicitly specify frame count
            )

            # Wait for completion
            sd.wait()

            print("Recording and playback completed")

        except Exception as e4:
                        print(f"All methods failed. Last error: {e4}")
                        print("\nTroubleshooting suggestions:")
                        print("1. Try selecting different devices with interactive=True")
                        print("2. Check System Preferences > Sound > Input/Output")
                        print("3. Make sure no other apps are using the microphone")
                        print("4. Try restarting the Python process")
                        return None

        # Extract recorded audio (remove extra dimension if needed)
        if recording.ndim > 1:
            recorded_audio = recording.flatten()
        else:
            recorded_audio = recording
        print(f'Length of recorded audio {len(recorded_audio)} samples')
        resp = recorded_audio.reshape(self.num_pulses, self.cycle_samples)
        recorded_audio = recorded_audio[self.cycle_samples * 4 :]
        room_response = np.mean(resp[4:], axis=0)
        max_index = np.argmax(room_response)
        if max_index > 50:
            onset = self.find_sound_onset_derivative(room_response[max_index - 100:]) + max_index - 100
            print(f"Found onset at sample no {onset}")
            impulse_response = np.concatenate([room_response[onset:], room_response[:onset]])
        else:
            impulse_response = room_response


        # Save recording to WAV file
        self._save_wav(recorded_audio, output_filename)
        self._save_wav(impulse_response, response_filename)
        self._save_wav(room_response, "room_response.wav")

        return recorded_audio

    def _save_wav(self, audio_data, filename):
        """Save audio data to WAV file"""
        # Normalize and convert to int16 for WAV format
        # First, prevent clipping
        max_val = np.max(np.abs(audio_data))
        if max_val > 0:
            audio_data = audio_data / max_val * 0.95  # Leave some headroom

        audio_int16 = (audio_data * 32767).astype(np.int16)

        with wave.open(filename, 'w') as wav_file:
            wav_file.setnchannels(1)  # Mono
            wav_file.setsampwidth(2)  # 2 bytes per sample (int16)
            wav_file.setframerate(self.sample_rate)
            wav_file.writeframes(audio_int16.tobytes())

        print(f"Audio saved to {filename}")

    def analyze_signal_timing(self):
        """Helper method to analyze the generated signal timing"""
        print(f"\nSignal analysis:")
        print(f"- Total signal length: {len(self.playback_signal)} samples")
        print(f"- Duration: {len(self.playback_signal) / self.sample_rate:.3f} seconds")

        # Analyze exact pulse positions by finding non-zero regions
        pulse_positions = []
        gap_durations = []

        # Find start positions of pulses by looking for transitions from 0 to non-zero
        threshold = 0.001
        in_pulse = False
        pulse_start = 0

        for i in range(len(self.playback_signal)):
            if not in_pulse and abs(self.playback_signal[i]) > threshold:
                # Start of pulse
                pulse_start = i
                pulse_positions.append(i)
                in_pulse = True
            elif in_pulse and abs(self.playback_signal[i]) <= threshold:
                # End of pulse
                pulse_end = i
                pulse_duration = pulse_end - pulse_start
                print(
                    f"  Pulse at sample {pulse_start}: duration {pulse_duration} samples ({pulse_duration / self.sample_rate * 1000:.1f} ms)")
                in_pulse = False

        # Calculate gaps between pulses
        if len(pulse_positions) > 1:
            print(f"\nPulse timing verification:")
            for i in range(len(pulse_positions) - 1):
                cycle_duration = pulse_positions[i + 1] - pulse_positions[i]
                gap_duration = cycle_duration - 441  # 441 samples per pulse
                print(f"  Cycle {i + 1}: {cycle_duration} samples ({cycle_duration / self.sample_rate * 1000:.1f} ms)")
                print(f"  Gap {i + 1}: {gap_duration} samples ({gap_duration / self.sample_rate * 1000:.1f} ms)")

        print(f"\nExpected timing:")
        print(f"  - Pulse duration: 441 samples (10.0 ms)")
        print(f"  - Gap duration: 3969 samples (90.0 ms)")
        print(f"  - Full cycle: 4410 samples (100.0 ms)")


    def find_sound_onset_derivative(self, audio, window_size=10, threshold_factor=2):
        """
        Find sound onset using moving average and derivative
        """

        # Calculate moving RMS
        def moving_rms(signal, window):
            return np.sqrt(np.convolve(signal ** 2, np.ones(window) / window, mode='same'))

        rms = moving_rms(audio, window_size)

        # Calculate derivative of RMS
        rms_diff = np.diff(rms)

        # Find significant increase
        background_level = np.std(rms[:window_size])
        threshold = threshold_factor * background_level

        onset_candidates = np.where(rms_diff > threshold)[0]

        return onset_candidates[0] if len(onset_candidates) > 0 else len(audio)

def main():
    # Create recorder instance


    recorder = RoomResponseRecorder(
        sample_rate=22100,
        pulse_samples=10,  # 10ms pulses
        duration = 0.3,  # 1kHz tone
        num_pulses = 20,
        volume = 0.5
    )

    # Analyze the generated signal
    recorder.analyze_signal_timing()

    # Generate filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"room_response_{timestamp}.wav"
    impulse_reponse_file = f"impulse_response_{timestamp}.wav"

    try:
        # Record room response with synchronized playback
        audio_data = recorder.record_room_response(output_file, impulse_reponse_file, interactive=False)

        # If you want to choose devices manually, use:
        # audio_data = recorder.record_room_response(output_file, interactive=True)

        print(f"\nRecording summary:")
        print(f"- Recorded samples: {len(audio_data)}")
        print(f"- Duration: {len(audio_data) / recorder.sample_rate:.3f} seconds")
        print(f"- Sample rate: {recorder.sample_rate} Hz")
        print(f"- Pulses in sequence: {recorder.num_pulses}")
        print(f"- Pulse interval: {recorder.pulse_interval * 1000:.0f} ms")
        print(f"- Output file: {output_file}")

        # Basic analysis of recorded signal
        max_amplitude = np.max(np.abs(audio_data))
        rms_level = np.sqrt(np.mean(audio_data ** 2))
        print(f"- Max amplitude: {max_amplitude:.4f}")
        print(f"- RMS level: {rms_level:.4f}")

    except Exception as e:
        print(f"Error during recording: {e}")
        print("\nTroubleshooting tips:")
        print("1. Make sure your microphone permissions are enabled")
        print("2. Check that no other apps are using the microphone")
        print("3. Verify your audio device settings")
        print("4. Try running: python -c 'import sounddevice; print(sounddevice.query_devices())'")


if __name__ == "__main__":
    main()