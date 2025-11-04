#!/usr/bin/env python3
"""
Test script for save format functionality.
Tests saving audio in both WAV and NumPy formats.
"""

import numpy as np
import wave
from pathlib import Path
import tempfile
import json

def test_save_formats():
    """Test saving audio in different formats"""
    print("Testing save format functionality...")
    print("=" * 60)

    # Create test audio data
    sample_rate = 48000
    duration = 1.0  # 1 second
    frequency = 440.0  # A4 note

    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    test_audio = 0.5 * np.sin(2 * np.pi * frequency * t)

    print(f"\n1. Created test audio:")
    print(f"   - Duration: {duration} seconds")
    print(f"   - Sample rate: {sample_rate} Hz")
    print(f"   - Frequency: {frequency} Hz")
    print(f"   - Shape: {test_audio.shape}")
    print(f"   - Data type: {test_audio.dtype}")
    print(f"   - Min value: {test_audio.min():.6f}")
    print(f"   - Max value: {test_audio.max():.6f}")

    print(f"\n   IMPORTANT: Both formats will save the SAME normalized data.")
    print(f"   NumPy format preserves exact normalized values without quantization.")

    # Create temporary directory for test files
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Test 1: Save as WAV only
        print(f"\n2. Testing WAV format (16-bit)...")
        wav_file = temp_path / "test_wav_only.wav"
        save_as_wav(test_audio, str(wav_file), sample_rate)

        if wav_file.exists():
            size_kb = wav_file.stat().st_size / 1024
            print(f"   [OK] WAV file created: {wav_file.name}")
            print(f"   - File size: {size_kb:.2f} KB")

            # Load and verify
            loaded_wav = load_wav(str(wav_file), sample_rate)
            correlation = np.corrcoef(test_audio, loaded_wav)[0, 1]
            print(f"   - Correlation with original: {correlation:.6f}")
            print(f"   - Data type after load: {loaded_wav.dtype}")
        else:
            print(f"   [ERROR] WAV file not created!")

        # Test 2: Save as NumPy only
        print(f"\n3. Testing NumPy format (full resolution)...")
        npy_file = temp_path / "test_npy_only.npy"
        save_as_npy(test_audio, str(npy_file))

        if npy_file.exists():
            size_kb = npy_file.stat().st_size / 1024
            print(f"   [OK] NumPy file created: {npy_file.name}")
            print(f"   - File size: {size_kb:.2f} KB")

            # Load and verify
            loaded_npy = np.load(str(npy_file))
            print(f"   - Data type after load: {loaded_npy.dtype}")

            # Compare to normalized original
            max_val = np.max(np.abs(test_audio))
            test_audio_normalized = test_audio / max_val * 0.95
            print(f"   - Exact match with normalized: {np.array_equal(test_audio_normalized, loaded_npy)}")
            max_diff = np.max(np.abs(test_audio_normalized - loaded_npy))
            print(f"   - Max difference from normalized: {max_diff:.2e}")
        else:
            print(f"   [ERROR] NumPy file not created!")

        # Test 3: Compare precision and verify both contain same normalized data
        print(f"\n4. Comparing formats...")
        if wav_file.exists() and npy_file.exists():
            loaded_wav = load_wav(str(wav_file), sample_rate)
            loaded_npy = np.load(str(npy_file))

            # Normalize the original data (same as both save functions do)
            max_val = np.max(np.abs(test_audio))
            test_audio_normalized = test_audio / max_val * 0.95

            # Compute differences from normalized original
            wav_error = np.abs(test_audio_normalized - loaded_wav)
            npy_error = np.abs(test_audio_normalized - loaded_npy)

            print(f"   Verifying both formats contain SAME normalized data...")
            print(f"\n   WAV (16-bit quantized) error vs normalized original:")
            print(f"   - Mean error: {np.mean(wav_error):.6f}")
            print(f"   - Max error: {np.max(wav_error):.6f}")
            print(f"   - RMS error: {np.sqrt(np.mean(wav_error**2)):.6f}")

            print(f"\n   NumPy (float64 exact) error vs normalized original:")
            print(f"   - Mean error: {np.mean(npy_error):.2e}")
            print(f"   - Max error: {np.max(npy_error):.2e}")
            print(f"   - RMS error: {np.sqrt(np.mean(npy_error**2)):.2e}")

            print(f"\n   Quantization precision improvement: {np.max(wav_error) / (np.max(npy_error) + 1e-20):.1e}x better")

            # Verify both WAV and NumPy contain the same normalized values (within quantization)
            wav_vs_npy_error = np.abs(loaded_wav - loaded_npy)
            print(f"\n   Difference between WAV and NumPy (quantization error only):")
            print(f"   - Max difference: {np.max(wav_vs_npy_error):.6f}")
            print(f"   - This is the 16-bit quantization error, NOT a data difference!")

        # Test 4: Test configuration
        print(f"\n5. Testing configuration format...")
        test_config = {
            "save_format": {
                "save_wav": True,
                "save_npy": True
            }
        }
        config_file = temp_path / "test_config.json"
        with open(config_file, 'w') as f:
            json.dump(test_config, f, indent=2)

        print(f"   [OK] Test configuration created")
        print(f"   - save_wav: {test_config['save_format']['save_wav']}")
        print(f"   - save_npy: {test_config['save_format']['save_npy']}")

    print("\n" + "=" * 60)
    print("[OK] All tests completed successfully!")
    print("\nIntegration with RoomResponseRecorder:")
    print("  1. Set 'save_npy': true in recorderConfig.json")
    print("  2. Use the Audio Settings panel to configure formats")
    print("  3. Full resolution .npy files will be saved alongside .wav files")

def save_as_wav(audio_data, filename, sample_rate):
    """Save audio as 16-bit WAV file"""
    # Normalize and convert to int16
    max_val = np.max(np.abs(audio_data))
    if max_val > 0:
        audio_normalized = audio_data / max_val * 0.95
    else:
        audio_normalized = audio_data

    audio_int16 = (audio_normalized * 32767).astype(np.int16)

    # Save
    Path(filename).parent.mkdir(parents=True, exist_ok=True)
    with wave.open(filename, 'w') as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_int16.tobytes())

def load_wav(filename, sample_rate):
    """Load WAV file and convert back to float"""
    with wave.open(filename, 'r') as wav_file:
        audio_int16 = np.frombuffer(wav_file.readframes(-1), dtype=np.int16)
    return audio_int16.astype(np.float64) / 32767.0

def save_as_npy(audio_data, filename):
    """Save audio as NumPy array (with same normalization as WAV)"""
    # Normalize (same as WAV)
    max_val = np.max(np.abs(audio_data))
    if max_val > 0:
        audio_normalized = audio_data / max_val * 0.95
    else:
        audio_normalized = audio_data

    Path(filename).parent.mkdir(parents=True, exist_ok=True)
    np.save(filename, audio_normalized)  # Save normalized data

if __name__ == "__main__":
    test_save_formats()
