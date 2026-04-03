# Testing

## Test Inventory

| Test File | Area | Hardware Required |
|-----------|------|-------------------|
| `test_signal_processor.py` | Signal processing pipeline | No |
| `test_signal_processor_basic.py` | Basic signal processing ops | No |
| `test_amplitude_preservation.py` | Amplitude consistency through pipeline | No |
| `test_two_stage_alignment.py` | Two-stage alignment algorithm | No |
| `test_calibration_validator_v2.py` | Calibration quality validation | No |
| `test_calibration_return.py` | Calibration return values | No |
| `test_calibration_visualizer.py` | Calibration visualization | No |
| `test_channel_selection.py` | Multi-channel selection logic | No |
| `test_save_formats.py` | WAV/NPY save format handling | No |
| `test_truncation.py` | IR truncation and fade | No |
| `test_universal_save.py` | Universal save utility | No |
| `test_data_generator.py` | Synthetic test data generation | No |
| `test_voice_coil_impulse.py` | Voice coil impulse shape | No |
| `test_multichannel.py` | Multi-channel recording | Yes (audio interface) |
| `test_audio.py` | Basic audio I/O | Yes (audio device) |
| `test_device_enumeration.py` | SDL device detection | Yes (SDL2) |
| `test_umc_multichannel.py` | UMC1820 multi-channel | Yes (UMC1820) |
| `test_umc_input_detailed.py` | UMC1820 detailed input test | Yes (UMC1820) |

## ESPRIT Tests

| Test File | Area |
|-----------|------|
| `ESPRIT/test_30_perfect_detection.py` | 30-mode synthetic detection accuracy |

## Running Tests

```bash
# Run a specific test
python test_signal_processor.py

# Run signal processing tests (no hardware)
python test_signal_processor_basic.py
python test_amplitude_preservation.py
python test_two_stage_alignment.py
python test_truncation.py

# Hardware-dependent tests
python test_device_enumeration.py
python test_umc_multichannel.py
```

No test runner framework (pytest/unittest) is configured project-wide. Tests are standalone scripts.

## Diagnostic Scripts

| Script | Purpose |
|--------|---------|
| `check_umc_driver.py` | Check Behringer UMC1820 driver status |
| `check_averaged_responses.py` | Validate averaged response files |
| `debug_piano_scenarios.py` | Debug scenario loading issues |
| `analyze_piano_scenarios.py` | Analyze piano scenario data |
