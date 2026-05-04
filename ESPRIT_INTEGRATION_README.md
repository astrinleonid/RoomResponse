# ESPRIT Integration for Single Scenario Collection

This document describes the integration of streaming ESPRIT modal analysis into the RoomResponse data collection pipeline.

## Overview

The ESPRIT integration enables **real-time modal analysis** during single scenario data collection, with three key capabilities:

1. **Per-Scenario Processing**: Run ESPRIT analysis immediately after each scenario completes
2. **Selective Aggregation**: Aggregate ESPRIT results from selected scenarios within a collection
3. **Incremental Updates**: Add new scenarios to existing aggregated results

## Architecture

### Data Flow

```
Single Scenario Collection:
  ├─ Collect N measurements for scenario
  ├─ Average measurements → averaged room response
  ├─ ESPRIT processes averaged response
  │  └─ Results saved to scenario/analysis/esprit_single_point.json
  └─ Display results in UI

Aggregation (across multiple scenarios):
  ├─ Filter scenarios by Computer+Room name
  ├─ Select scenarios to aggregate
  ├─ Load individual ESPRIT results
  ├─ Aggregate using k-means clustering
  └─ Save to esprit_aggregated_{Computer}_{Room}.json
```

### Key Components

**1. esprit_helper.py**
- `ESPRITScenarioProcessor`: Process individual scenarios
- `ESPRITAggregator`: Aggregate results from multiple scenarios

**2. gui_collect_panel.py**
- ESPRIT configuration UI in Single Scenario mode
- Post-collection processing hook
- Progress event handling

**3. piano_response.py**
- ESPRIT Aggregation panel in navigation
- UI for selecting and aggregating scenarios
- Visualization of results

## Usage

### Step 1: Enable ESPRIT for Single Scenario Collection

1. Navigate to **Collect → Single Scenario**
2. Expand the **🎼 ESPRIT Modal Analysis** section
3. Enable the checkbox: "Enable real-time ESPRIT modal analysis"
4. Configure parameters:
   - **Model Order (K)**: Number of poles to extract per band (default: 30)
   - **Hankel Window**: Fraction of signal for analysis (default: 0.5)

5. Start collection as normal
6. ESPRIT will automatically process **ALL 4 frequency bands** after collection completes:
   - Band 0: 40-500 Hz (Piano fundamentals)
   - Band 1: 500-1000 Hz (Low harmonics)
   - Band 2: 1000-2000 Hz (Mid harmonics)
   - Band 3: 2000-4000 Hz (High harmonics)

### Step 2: Monitor ESPRIT Progress

During and after collection, the status panel shows:

- **Collection progress**: Measurements completed, success/failure counts
- **ESPRIT status**: "Running ESPRIT analysis..." message
- **ESPRIT results**: Total modes across all bands and breakdown per band

Example output:
```
🎼 ESPRIT Multi-Band Analysis Complete: MyComputer-Scenario1-MyRoom
Total Modes (All Bands): 45

band0_40-500Hz: 12 modes
band1_500-1000Hz: 15 modes
band2_1000-2000Hz: 10 modes
band3_2000-4000Hz: 8 modes
```

### Step 3: Aggregate Results Across Scenarios

1. Collect multiple scenarios from the same collection (same Computer+Room)
2. Navigate to **🎼 ESPRIT Aggregation** panel
3. Enter **Computer Name** and **Room Name**
4. System finds all matching scenarios
5. Select scenarios to include (all selected by default)
6. Click **🎼 Run Aggregation**
7. View results:
   - Common modes table (frequency, damping, Q-factor)
   - Frequency spectrum plot
   - Mode shapes across scenarios

## Configuration Parameters

### ESPRIT Processing Parameters

| Parameter | Description | Default | Range |
|-----------|-------------|---------|-------|
| `M_out` | Number of channels | 6 | Auto-detected |
| `N_use` | Samples per measurement | 28800 | Auto-detected |
| `fs` | Sampling frequency (Hz) | 48000 | Auto-detected |
| `band_index` | Frequency band preset | 0 | 0-3 |
| `L_fraction` | Hankel window fraction | 0.5 | 0.3-0.7 |
| `K` | Model order | 30 | 10-50 |
| `skip_m` | Channel to skip | 2 | 0-(M_out-1) |

### Band Presets

| Index | Frequency Range | Decimation | Typical Use |
|-------|----------------|------------|-------------|
| 0 | 40-500 Hz | 4× | Piano fundamentals |
| 1 | 500-1000 Hz | 2× | Low harmonics |
| 2 | 1000-2000 Hz | None | Mid harmonics |
| 3 | 2000-4000 Hz | None | High harmonics |

## File Structure

```
dataset_root/
└── Computer-ScenarioN-Room/
    ├── room_responses/
    │   ├── room_0000.wav
    │   ├── room_0001.wav
    │   └── ...
    ├── analysis/
    │   ├── esprit_band0_40-500Hz.json      # Band 0 results
    │   ├── esprit_band1_500-1000Hz.json    # Band 1 results
    │   ├── esprit_band2_1000-2000Hz.json   # Band 2 results
    │   ├── esprit_band3_2000-4000Hz.json   # Band 3 results
    │   └── esprit_all_bands_summary.json   # Summary of all bands
    └── scenario_metadata.json

dataset_root/
└── esprit_aggregated_Computer_Room.json  # Aggregated results (after merging)
```

## ESPRIT Results Format

### Individual Band Results

File: `scenario/analysis/esprit_band0_40-500Hz.json` (example for Band 0)

```json
{
  "r_index": 0,
  "scenario_name": "Computer-Scenario1-Room",
  "scenario_dir": "/path/to/scenario",
  "band_index": 0,
  "band_name": "band0_40-500Hz",
  "num_modes": 12,
  "frequencies": [82.3, 164.7, 247.1, ...],
  "damping_ratios": [0.0023, 0.0019, 0.0021, ...],
  "Q_factors": [217.4, 263.2, 238.1, ...],
  "lambdas_re": [-0.145, -0.312, ...],
  "lambdas_im": [517.3, 1034.8, ...],
  "config": {
    "M_out": 6,
    "N_use": 28800,
    "fs": 48000,
    "band_index": 0,
    "L_fraction": 0.5,
    "K": 30,
    "skip_m": 2
  }
}
```

### Multi-Band Summary

File: `scenario/analysis/esprit_all_bands_summary.json`

```json
{
  "scenario_name": "Computer-Scenario1-Room",
  "scenario_dir": "/path/to/scenario",
  "processing_mode": "multi-band",
  "total_modes_all_bands": 45,
  "modes_per_band": {
    "band0_40-500Hz": 12,
    "band1_500-1000Hz": 15,
    "band2_1000-2000Hz": 10,
    "band3_2000-4000Hz": 8
  },
  "band_files": {
    "band0_40-500Hz": "analysis/esprit_band0_40-500Hz.json",
    "band1_500-1000Hz": "analysis/esprit_band1_500-1000Hz.json",
    "band2_1000-2000Hz": "analysis/esprit_band2_1000-2000Hz.json",
    "band3_2000-4000Hz": "analysis/esprit_band3_2000-4000Hz.json"
  }
}
```

### Aggregated Results

File: `esprit_aggregated_Computer_Room.json`

```json
{
  "collection_name": "Computer_Room",
  "num_scenarios": 30,
  "scenario_names": ["1", "2", "3", ...],
  "common_f": [82.5, 165.0, 247.5, ...],
  "common_z": [0.0022, 0.0020, 0.0021, ...],
  "signed_shapes": [
    [1.0, 0.95, 0.87, ...],    # Mode 0 shape
    [1.0, -0.92, 0.85, ...],   # Mode 1 shape
    ...
  ],
  "participation": [
    [0.85, 0.81, 0.74, ...],   # Mode 0 participation
    [0.92, 0.88, 0.79, ...],   # Mode 1 participation
    ...
  ],
  "amplitudes_in_m": [
    [{"real": 0.12, "imag": 0.05}, ...],  # Mode 0 receiver amplitudes
    ...
  ],
  "r_indices": [0, 1, 2, ...],
  "config": {...}
}
```

## Validation

### Verifying Results

The streaming implementation follows `esprit.py` exactly and produces identical results:

```python
# Compare with reference implementation
from ESPRIT.esprit import stabilization_diagram_with_shapes
from esprit_helper import ESPRITAggregator

# Reference: batch processing with esprit.py
batch_results = stabilization_diagram_with_shapes(
    "esprit_data/index.txt",
    "esprit_data/y_cube.bin",
    band_index=0, K=30
)

# Streaming: process scenarios individually then aggregate
aggregated_results = ESPRITAggregator.aggregate_scenarios(
    scenario_dirs=[...],
    output_file="streaming_results.json"
)

# Verify frequencies match (within numerical precision)
assert max(abs(batch_results['common_f'] - aggregated_results['common_f'])) < 1e-10
```

## Troubleshooting

### Issue: ESPRIT processing fails

**Symptoms**: Error message "ESPRIT processing failed (see console for details)"

**Common causes**:
1. Missing room response files
2. Insufficient valid measurements
3. Wrong number of channels configured
4. Model order K too large for signal length

**Solutions**:
- Check console output for detailed error
- Verify `room_responses/` directory contains WAV files
- Check multi-channel configuration matches ESPRIT config
- Reduce K parameter if needed

### Issue: No scenarios found for aggregation

**Symptoms**: "No scenarios found for Computer: X, Room: Y"

**Solutions**:
- Verify Computer and Room names match exactly (case-sensitive)
- Check scenarios exist in dataset directory
- Verify `scenario_metadata.json` files contain correct metadata

### Issue: Different results from esprit.py

**Causes**:
- Different band_index
- Different model order K
- Different L_fraction
- Different skip_m channel

**Solution**: Ensure all parameters match between implementations

## Performance

### Timing Estimates

For typical scenario (30 measurements, 6 channels, 28800 samples @ 48kHz):

| Operation | Duration |
|-----------|----------|
| Scenario collection | ~5-10 minutes |
| Averaging room responses | ~0.5 seconds |
| ESPRIT processing | ~0.5-1 second |
| Aggregation (30 scenarios) | ~1-2 seconds |

**Total per scenario**: Collection time + 1 second ESPRIT overhead

### Memory Usage

- **Per scenario**: ~50 MB (averaged responses stored)
- **Aggregation**: ~200 MB (loading all individual results)

## API Reference

### ESPRITScenarioProcessor

```python
from esprit_helper import ESPRITScenarioProcessor

# Process a single scenario
result = ESPRITScenarioProcessor.process_scenario(
    scenario_dir=Path("Computer-Scenario1-Room"),
    esprit_config={
        'M_out': 6,
        'N_use': 28800,
        'fs': 48000,
        'band_index': 0,
        'L_fraction': 0.5,
        'K': 30,
        'skip_m': 2
    },
    force_reprocess=False
)
```

### ESPRITAggregator

```python
from esprit_helper import ESPRITAggregator
from pathlib import Path

# Find scenarios in collection
scenarios = ESPRITAggregator.find_collection_scenarios(
    base_dir=Path("dataset_root"),
    computer_name="MyComputer",
    room_name="MyRoom"
)

# Aggregate selected scenarios
results = ESPRITAggregator.aggregate_scenarios(
    scenario_dirs=[path for path, num in scenarios],
    output_file=Path("aggregated_results.json"),
    collection_name="MyComputer_MyRoom"
)
```

## Advanced Features

### Custom Band Configuration

```python
from ESPRIT.esprit import BandPreset, band_presets

# Define custom band
custom_band = BandPreset(
    low_freq=100.0,
    high_freq=800.0,
    decimate_factor=2,
    N_band=4096,
    filter_order=6,
    exp_alpha=0.01
)

# Add to presets
band_presets.append(custom_band)

# Use in configuration
esprit_config = {
    'band_index': len(band_presets) - 1,  # Use custom band
    ...
}
```

### Programmatic Processing

```python
from pathlib import Path
from esprit_helper import ESPRITScenarioProcessor, ESPRITAggregator

# Process all scenarios in a collection
base_dir = Path("dataset_root")
computer = "MyComputer"
room = "MyRoom"

# Find scenarios
scenarios = ESPRITAggregator.find_collection_scenarios(
    base_dir, computer, room
)

# Process each scenario
for scenario_dir, scenario_num in scenarios:
    print(f"Processing scenario {scenario_num}...")
    result = ESPRITScenarioProcessor.process_scenario(
        scenario_dir=scenario_dir,
        force_reprocess=True
    )
    if result:
        print(f"  ✓ {result['num_modes']} modes detected")

# Aggregate all
results = ESPRITAggregator.aggregate_scenarios(
    scenario_dirs=[path for path, num in scenarios],
    output_file=base_dir / f"esprit_{computer}_{room}.json",
    collection_name=f"{computer}_{room}"
)
```

## References

- **ESPRIT Algorithm**: [ESPRIT_CORE_GUIDE.md](ESPRIT/ESPRIT_CORE_GUIDE.md)
- **Streaming Implementation**: [STREAMING_README.md](ESPRIT/STREAMING_README.md)
- **Reference Implementation**: [esprit.py](ESPRIT/esprit.py)
- **Example Usage**: [example_streaming_usage.py](ESPRIT/example_streaming_usage.py)

## License

Part of the RoomResponse project.
