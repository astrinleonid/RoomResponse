#!/usr/bin/env python3
"""
Test Data Generator

Creates synthetic audio datasets for testing the scenario classifier.
Generates 4 datasets: 2 pure noise, 2 with distinct frequencies + noise.
"""

import os
import numpy as np
import pandas as pd
import librosa
from scipy.io import wavfile
import argparse
from pathlib import Path
import shutil


class TestDataGenerator:
    """Generator for synthetic audio test data."""

    def __init__(self, sample_rate=16000, duration_samples=1600):
        """
        Initialize the test data generator.

        Args:
            sample_rate (int): Audio sample rate
            duration_samples (int): Length of each audio sample
        """
        self.sample_rate = sample_rate
        self.duration_samples = duration_samples
        self.duration_seconds = duration_samples / sample_rate

    def generate_pure_noise(self, num_samples=100, noise_level=0.1, seed=None):
        """
        Generate pure white noise samples.

        Args:
            num_samples (int): Number of audio samples to generate
            noise_level (float): Amplitude of noise (0.0 to 1.0)
            seed (int): Random seed for reproducibility

        Returns:
            list: List of audio arrays
        """
        if seed is not None:
            np.random.seed(seed)

        samples = []
        for i in range(num_samples):
            # Generate white noise
            noise = np.random.normal(0, noise_level, self.duration_samples)
            samples.append(noise.astype(np.float32))

        return samples

    def generate_tonal_with_noise(self, frequencies, num_samples=100,
                                  tone_amplitude=0.3, noise_level=0.05, seed=None):
        """
        Generate samples with specific frequencies plus noise.

        Args:
            frequencies (list): List of frequencies to include
            num_samples (int): Number of audio samples to generate
            tone_amplitude (float): Amplitude of tonal components
            noise_level (float): Amplitude of noise
            seed (int): Random seed for reproducibility

        Returns:
            list: List of audio arrays
        """
        if seed is not None:
            np.random.seed(seed)

        samples = []
        time = np.arange(self.duration_samples) / self.sample_rate

        for i in range(num_samples):
            # Start with noise
            audio = np.random.normal(0, noise_level, self.duration_samples)

            # Add tonal components
            for freq in frequencies:
                # Add some randomness to frequency and phase
                freq_variation = freq * (1 + np.random.normal(0, 0.02))  # ±2% variation
                phase = np.random.uniform(0, 2 * np.pi)

                # Generate sine wave
                tone = tone_amplitude * np.sin(2 * np.pi * freq_variation * time + phase)

                # Add envelope for more realistic response
                envelope = np.exp(-time * np.random.uniform(1, 5))  # Decay envelope
                tone *= envelope

                audio += tone

            # Normalize to prevent clipping
            if np.max(np.abs(audio)) > 0:
                audio = audio / np.max(np.abs(audio)) * 0.8

            samples.append(audio.astype(np.float32))

        return samples

    def save_wav_files(self, audio_samples, output_folder, prefix="sample"):
        """
        Save audio samples as WAV files using scipy.

        Args:
            audio_samples (list): List of audio arrays
            output_folder (str): Output folder path
            prefix (str): Filename prefix
        """
        os.makedirs(output_folder, exist_ok=True)

        for i, audio in enumerate(audio_samples):
            filename = f"{prefix}_{i:03d}.wav"
            filepath = os.path.join(output_folder, filename)

            # Scale to 16-bit integer range for WAV format
            audio_int = (audio * 32767).astype(np.int16)
            wavfile.write(filepath, self.sample_rate, audio_int)

    def extract_features_from_audio(self, audio, n_mfcc=13):
        """
        Extract both spectrum and MFCC features from audio.

        Args:
            audio (np.ndarray): Audio signal
            n_mfcc (int): Number of MFCC coefficients

        Returns:
            tuple: (spectrum_features, mfcc_features)
        """
        # Extract spectrum (FFT)
        fft_result = np.fft.fft(audio)
        magnitude_spectrum = np.abs(fft_result[:len(fft_result) // 2 + 1])

        # Normalize spectrum
        if np.max(magnitude_spectrum) > 0:
            magnitude_spectrum = magnitude_spectrum / np.max(magnitude_spectrum)

        # Extract MFCC with adaptive parameters
        audio_length = len(audio)
        if audio_length < 512:
            n_fft = max(64, 2 ** int(np.log2(audio_length // 2))) if audio_length >= 4 else 64
            hop_length = max(16, n_fft // 4)
        else:
            n_fft = 512
            hop_length = 128

        # Ensure parameters don't exceed audio length
        n_fft = min(n_fft, audio_length)
        hop_length = min(hop_length, audio_length // 4) if audio_length > 4 else 1

        try:
            mfcc = librosa.feature.mfcc(
                y=audio, sr=self.sample_rate, n_mfcc=n_mfcc,
                n_fft=n_fft, hop_length=hop_length
            )
            # Average across time
            mfcc_features = np.mean(mfcc, axis=1)
        except:
            mfcc_features = np.zeros(n_mfcc)

        return magnitude_spectrum, mfcc_features

    def create_feature_csv(self, audio_samples, output_folder, n_mfcc=13):
        """
        Create both spectrum.csv and features.csv files.

        Args:
            audio_samples (list): List of audio arrays
            output_folder (str): Output folder path
            n_mfcc (int): Number of MFCC coefficients
        """
        spectrum_data = []
        mfcc_data = []

        for i, audio in enumerate(audio_samples):
            filename = f"sample_{i:03d}.wav"

            # Extract features
            spectrum, mfcc = self.extract_features_from_audio(audio, n_mfcc)

            # Spectrum features
            spectrum_row = {'filename': filename}
            for j, spec_val in enumerate(spectrum):
                spectrum_row[f'freq_{j}'] = float(spec_val)
            spectrum_data.append(spectrum_row)

            # MFCC features
            mfcc_row = {'filename': filename}
            for j, mfcc_val in enumerate(mfcc):
                mfcc_row[f'mfcc_{j}'] = float(mfcc_val)
            mfcc_data.append(mfcc_row)

        # Save CSV files
        spectrum_df = pd.DataFrame(spectrum_data)
        spectrum_df.to_csv(os.path.join(output_folder, 'spectrum.csv'), index=False)

        mfcc_df = pd.DataFrame(mfcc_data)
        mfcc_df.to_csv(os.path.join(output_folder, 'features.csv'), index=False)

    def create_scenario_folder(self, audio_samples, scenario_name, base_output_dir, n_mfcc=13):
        """
        Create a complete scenario folder with audio files and features.

        Args:
            audio_samples (list): List of audio arrays
            scenario_name (str): Name of the scenario
            base_output_dir (str): Base output directory
            n_mfcc (int): Number of MFCC coefficients
        """
        scenario_folder = os.path.join(base_output_dir, scenario_name)
        audio_folder = os.path.join(scenario_folder, 'impulse_responses')

        # Create folders
        os.makedirs(audio_folder, exist_ok=True)

        # Save WAV files
        self.save_wav_files(audio_samples, audio_folder)

        # Create feature CSV files
        self.create_feature_csv(audio_samples, scenario_folder, n_mfcc)

        print(f"Created scenario: {scenario_name} with {len(audio_samples)} samples")

        return scenario_folder


def main():
    """Generate test datasets for classifier evaluation."""
    parser = argparse.ArgumentParser(description='Generate synthetic test data for scenario classification')

    parser.add_argument('--output-dir', default='test_dataset',
                        help='Output directory for test data')
    parser.add_argument('--num-samples', type=int, default=100,
                        help='Number of samples per scenario')
    parser.add_argument('--sample-rate', type=int, default=16000,
                        help='Audio sample rate')
    parser.add_argument('--duration-samples', type=int, default=1600,
                        help='Duration of each sample in samples')
    parser.add_argument('--n-mfcc', type=int, default=13,
                        help='Number of MFCC coefficients')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')

    args = parser.parse_args()

    # Initialize generator
    generator = TestDataGenerator(
        sample_rate=args.sample_rate,
        duration_samples=args.duration_samples
    )

    print(f"Generating test data...")
    print(f"Output directory: {args.output_dir}")
    print(f"Samples per scenario: {args.num_samples}")
    print(f"Sample rate: {args.sample_rate} Hz")
    print(f"Duration: {args.duration_samples} samples ({args.duration_samples / args.sample_rate:.3f}s)")

    # Clean output directory
    if os.path.exists(args.output_dir):
        shutil.rmtree(args.output_dir)
    os.makedirs(args.output_dir)

    # Generate datasets
    print("\nGenerating datasets...")

    # 1. Pure noise dataset 1
    print("1. Generating Noise Dataset 1...")
    noise1_samples = generator.generate_pure_noise(
        num_samples=args.num_samples,
        noise_level=0.1,
        seed=args.seed
    )
    generator.create_scenario_folder(
        noise1_samples,
        'TestPC-Scenario1-Noise1',
        args.output_dir,
        args.n_mfcc
    )

    # 2. Pure noise dataset 2 (different seed for variation)
    print("2. Generating Noise Dataset 2...")
    noise2_samples = generator.generate_pure_noise(
        num_samples=args.num_samples,
        noise_level=0.12,  # Slightly different noise level
        seed=args.seed + 1000
    )
    generator.create_scenario_folder(
        noise2_samples,
        'TestPC-Scenario2-Noise2',
        args.output_dir,
        args.n_mfcc
    )

    # 3. Low frequency tones + noise
    print("3. Generating Low Frequency Dataset...")
    low_freq_samples = generator.generate_tonal_with_noise(
        frequencies=[250, 500, 750],  # Low frequencies
        num_samples=args.num_samples,
        tone_amplitude=0.4,
        noise_level=0.05,
        seed=args.seed + 2000
    )
    generator.create_scenario_folder(
        low_freq_samples,
        'TestPC-Scenario3-LowFreq',
        args.output_dir,
        args.n_mfcc
    )

    # 4. High frequency tones + noise
    print("4. Generating High Frequency Dataset...")
    high_freq_samples = generator.generate_tonal_with_noise(
        frequencies=[2000, 3500, 5000],  # High frequencies
        num_samples=args.num_samples,
        tone_amplitude=0.4,
        noise_level=0.05,
        seed=args.seed + 3000
    )
    generator.create_scenario_folder(
        high_freq_samples,
        'TestPC-Scenario4-HighFreq',
        args.output_dir,
        args.n_mfcc
    )

    # Create summary file
    summary_text = f"""
Test Dataset Summary
{'=' * 30}

Generated: 4 synthetic audio scenarios
Samples per scenario: {args.num_samples}
Sample rate: {args.sample_rate} Hz
Duration: {args.duration_samples} samples ({args.duration_samples / args.sample_rate:.3f}s)
Random seed: {args.seed}

Scenarios:
1. TestPC-Scenario1-Noise1: Pure white noise (level=0.10)
2. TestPC-Scenario2-Noise2: Pure white noise (level=0.12)
3. TestPC-Scenario3-LowFreq: Tones at 250, 500, 750 Hz + noise
4. TestPC-Scenario4-HighFreq: Tones at 2000, 3500, 5000 Hz + noise

Expected Classification Results:
- Noise1 vs Noise2: Difficult (should be ~50% accuracy)
- LowFreq vs HighFreq: Easy (should be >95% accuracy)
- Noise vs Tonal: Medium-Easy (should be >80% accuracy)

Each scenario folder contains:
- impulse_responses/: WAV files
- spectrum.csv: FFT-based features
- features.csv: MFCC-based features
"""

    with open(os.path.join(args.output_dir, 'README.txt'), 'w') as f:
        f.write(summary_text)

    print(f"\n{'=' * 50}")
    print(f"TEST DATA GENERATION COMPLETE")
    print(f"{'=' * 50}")
    print(f"Created 4 scenarios with {args.num_samples} samples each")
    print(f"Total: {4 * args.num_samples} synthetic audio files")
    print(f"Output directory: {args.output_dir}")
    print(f"\nScenarios created:")
    print(f"  1. Noise1 (pure noise)")
    print(f"  2. Noise2 (pure noise, different level)")
    print(f"  3. LowFreq (250, 500, 750 Hz + noise)")
    print(f"  4. HighFreq (2000, 3500, 5000 Hz + noise)")
    print(f"\nYou can now test the classifier on these scenarios!")


if __name__ == "__main__":
    main()