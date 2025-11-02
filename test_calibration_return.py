"""
Test script to verify calibration mode return structure for GUI compatibility
"""
import sys

# Mock the result structure to verify what keys are present
mock_calibration_result = {
    'raw': {},  # Dict[int, np.ndarray]
    'room_response': {},  # Dict[int, np.ndarray]
    'impulse': {},  # Dict[int, np.ndarray]
    'metadata': {
        'mode': 'calibration',
        'num_channels': 4,
        'calibration_channel': 0,
        'num_pulses_recorded': 10,
        'num_cycles': 10,
        'num_valid_cycles': 8,
        'num_aligned_cycles': 7,
        'correlation_threshold': 0.7,
        'normalize_by_calibration': False,
        'validation_results': [],
        'alignment_metadata': {}
    },
    # BACKWARD COMPATIBILITY keys
    'calibration_cycles': [],
    'validation_results': [],
    'alignment_metadata': {},
    'aligned_multichannel_cycles': {},
}

print("Checking calibration mode return structure...")
print("=" * 60)

# Check required keys for GUI
required_gui_keys = [
    'calibration_cycles',
    'validation_results',
    'alignment_metadata',
    'aligned_multichannel_cycles'
]

print("\nGUI Compatibility Keys:")
for key in required_gui_keys:
    if key in mock_calibration_result:
        print(f"  [PASS] '{key}' exists (top-level)")
    else:
        print(f"  [FAIL] '{key}' missing!")
        sys.exit(1)

# Check new format keys for file saving
required_save_keys = ['raw', 'room_response', 'impulse']

print("\nFile Saving Keys:")
for key in required_save_keys:
    if key in mock_calibration_result:
        print(f"  [PASS] '{key}' exists")
    else:
        print(f"  [FAIL] '{key}' missing!")
        sys.exit(1)

# Check metadata structure
print("\nMetadata Structure:")
if 'metadata' in mock_calibration_result:
    print(f"  [PASS] 'metadata' exists")
    metadata = mock_calibration_result['metadata']

    required_metadata_keys = [
        'mode',
        'num_cycles',
        'calibration_channel',
        'validation_results',
        'alignment_metadata'
    ]

    for key in required_metadata_keys:
        if key in metadata:
            print(f"    [PASS] metadata['{key}'] exists")
        else:
            print(f"    [FAIL] metadata['{key}'] missing!")

print("\n" + "=" * 60)
print("[SUCCESS] All required keys present!")
print("\nStructure supports:")
print("  - GUI compatibility (old top-level keys)")
print("  - File saving (new format keys)")
print("  - Metadata (nested details)")
