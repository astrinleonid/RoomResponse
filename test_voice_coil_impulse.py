#!/usr/bin/env python3
"""
Test script for voice coil impulse generation

This script validates the new voice_coil impulse type:
- Main pulse: square wave controlled by pulse_duration
- Pull-back: negative signal controlled by fade parameter
"""

import numpy as np
import matplotlib.pyplot as plt
from RoomResponseRecorder import RoomResponseRecorder
import json
from pathlib import Path

def create_test_config(impulse_type: str) -> str:
    """Create a temporary config file for testing"""
    config = {
        'recorder_config': {
            'sample_rate': 48000,
            'pulse_duration': 0.010,  # 10ms main pulse
            'pulse_fade': 0.002,      # 2ms pull-back
            'cycle_duration': 0.1,    # 100ms cycle
            'num_pulses': 3,
            'volume': 0.5,
            'pulse_frequency': 1000,
            'impulse_form': impulse_type
        }
    }

    config_path = Path("test_config_temp.json")
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

    return str(config_path)

def test_impulse_generation():
    """Test all three impulse types and visualize them"""

    print("Testing impulse generation for all waveform types\n")
    print("="*60)

    impulse_types = ["sine", "square", "voice_coil"]
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))

    for idx, impulse_type in enumerate(impulse_types):
        print(f"\nTesting {impulse_type} impulse:")
        print("-" * 40)

        # Create config and recorder
        config_path = create_test_config(impulse_type)

        try:
            recorder = RoomResponseRecorder(config_path)

            # Get signal info
            info = recorder.get_signal_info()
            print(f"  Sample rate: {info['sample_rate']} Hz")
            print(f"  Pulse duration: {info['pulse_duration_ms']:.2f} ms")
            print(f"  Cycle duration: {info['cycle_duration_ms']:.2f} ms")
            print(f"  Number of pulses: {info['num_pulses']}")
            print(f"  Impulse form: {info['impulse_form']}")

            # Generate a single pulse for visualization
            pulse_samples = int(recorder.pulse_duration * recorder.sample_rate)
            single_pulse = recorder._generate_single_pulse(pulse_samples)

            # Time axis in milliseconds
            time_ms = np.arange(len(single_pulse)) / recorder.sample_rate * 1000

            # Plot
            ax = axes[idx]
            ax.plot(time_ms, single_pulse, linewidth=2, label=f'{impulse_type} impulse')
            ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
            ax.grid(True, alpha=0.3)
            ax.set_xlabel('Time (ms)')
            ax.set_ylabel('Amplitude')
            ax.set_title(f'{impulse_type.capitalize()} Impulse Waveform')
            ax.legend()

            # For voice_coil, highlight the pull-back region
            if impulse_type == "voice_coil":
                pullback_start_ms = (pulse_samples - recorder.fade_samples) / recorder.sample_rate * 1000
                ax.axvline(x=pullback_start_ms, color='r', linestyle=':',
                          label='Pull-back start', alpha=0.7)
                ax.text(pullback_start_ms, 0.3, 'Pull-back\nregion',
                       ha='center', fontsize=9, color='red')

            # Statistics
            max_val = np.max(single_pulse)
            min_val = np.min(single_pulse)
            mean_val = np.mean(single_pulse)

            print(f"  Max amplitude: {max_val:.4f}")
            print(f"  Min amplitude: {min_val:.4f}")
            print(f"  Mean amplitude: {mean_val:.4f}")

            # Voice coil specific validation
            if impulse_type == "voice_coil":
                # Check that we have positive main pulse
                main_pulse_region = single_pulse[:pulse_samples - recorder.fade_samples]
                assert np.all(main_pulse_region > 0), "Main pulse should be positive"

                # Check that we have negative pull-back
                pullback_region = single_pulse[pulse_samples - recorder.fade_samples:]
                assert np.all(pullback_region < 0), "Pull-back should be negative"

                print(f"  ✓ Voice coil validation passed")
                print(f"    - Main pulse samples: {len(main_pulse_region)}")
                print(f"    - Pull-back samples: {len(pullback_region)}")
                print(f"    - Main pulse amplitude: {np.mean(main_pulse_region):.4f}")
                print(f"    - Pull-back amplitude: {np.mean(pullback_region):.4f}")

        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # Clean up config file
            Path(config_path).unlink(missing_ok=True)

    plt.tight_layout()
    plt.savefig('voice_coil_impulse_comparison.png', dpi=150)
    print("\n" + "="*60)
    print("Visualization saved to: voice_coil_impulse_comparison.png")
    print("="*60)
    plt.show()

def test_voice_coil_parameters():
    """Test voice coil with different parameter combinations"""

    print("\n\nTesting Voice Coil with Different Parameters")
    print("="*60)

    test_cases = [
        {'pulse_duration': 0.008, 'fade': 0.001, 'desc': 'Short pulse, short pull-back'},
        {'pulse_duration': 0.015, 'fade': 0.003, 'desc': 'Medium pulse, medium pull-back'},
        {'pulse_duration': 0.020, 'fade': 0.005, 'desc': 'Long pulse, long pull-back'},
    ]

    fig, axes = plt.subplots(len(test_cases), 1, figsize=(12, 10))

    for idx, test_case in enumerate(test_cases):
        print(f"\nTest case: {test_case['desc']}")
        print(f"  Pulse duration: {test_case['pulse_duration']*1000:.2f} ms")
        print(f"  Fade duration: {test_case['fade']*1000:.2f} ms")

        config = {
            'recorder_config': {
                'sample_rate': 48000,
                'pulse_duration': test_case['pulse_duration'],
                'pulse_fade': test_case['fade'],
                'cycle_duration': 0.1,
                'num_pulses': 3,
                'volume': 0.5,
                'pulse_frequency': 1000,
                'impulse_form': 'voice_coil'
            }
        }

        config_path = Path(f"test_config_temp_{idx}.json")
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)

        try:
            recorder = RoomResponseRecorder(str(config_path))
            pulse_samples = int(recorder.pulse_duration * recorder.sample_rate)
            single_pulse = recorder._generate_single_pulse(pulse_samples)
            time_ms = np.arange(len(single_pulse)) / recorder.sample_rate * 1000

            ax = axes[idx] if len(test_cases) > 1 else axes
            ax.plot(time_ms, single_pulse, linewidth=2)
            ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
            ax.grid(True, alpha=0.3)
            ax.set_xlabel('Time (ms)')
            ax.set_ylabel('Amplitude')
            ax.set_title(test_case['desc'])

            pullback_start_ms = (pulse_samples - recorder.fade_samples) / recorder.sample_rate * 1000
            ax.axvline(x=pullback_start_ms, color='r', linestyle=':', alpha=0.5)

            print(f"  ✓ Generated successfully")

        except Exception as e:
            print(f"  ERROR: {e}")
        finally:
            config_path.unlink(missing_ok=True)

    plt.tight_layout()
    plt.savefig('voice_coil_parameter_test.png', dpi=150)
    print("\nParameter test visualization saved to: voice_coil_parameter_test.png")
    plt.show()

if __name__ == "__main__":
    print("Voice Coil Impulse Generation Test")
    print("="*60)

    # Test 1: Compare all three impulse types
    test_impulse_generation()

    # Test 2: Test voice coil with different parameters
    test_voice_coil_parameters()

    print("\n" + "="*60)
    print("All tests completed!")
    print("="*60)
