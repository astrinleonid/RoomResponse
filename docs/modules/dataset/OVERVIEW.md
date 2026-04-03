# Dataset & Collection

## Dataset Structure

```
room_response_dataset/
└── <computer>-Scenario<number>-<room>/
    ├── raw_recordings/
    │   └── raw_<scenario>_IDX_TIMESTAMP[_chN].wav|.npy
    ├── impulse_responses/
    │   └── impulse_<scenario>_IDX_TIMESTAMP[_chN].wav|.npy
    ├── room_responses/
    │   └── room_<scenario>_IDX_TIMESTAMP[_chN].wav|.npy
    ├── metadata/
    │   └── session_metadata.json
    ├── analysis/
    ├── features.csv
    ├── spectrum.csv
    └── <scenario>_SUMMARY.txt
```

## Naming Convention

Scenario folder: `{computer}-Scenario{number}-{room}`

File naming: `{type}_{scenario}_{index}_{timestamp}[_ch{N}].{ext}`

- `type`: `raw`, `impulse`, `room`
- `ext`: `.wav` or `.npy` (controlled by `save_format` config)

Multi-channel files append `_chN` suffix.

## SingleScenarioCollector

`DatasetCollector.py` manages single-scenario data collection.

| Feature | Description |
|---------|-------------|
| Append/overwrite | Detects existing folder, offers append or overwrite |
| Compatibility check | Validates config matches existing data |
| Metadata persistence | Saves session metadata to JSON after each batch |
| Pause/Stop | Sentinel files (`PAUSE`, `STOP`) for unattended control |
| Resume support | Can resume interrupted collection |
| Quality metrics | Stores per-measurement quality data |

### Data Classes

| Class | Purpose |
|-------|---------|
| `ScenarioConfig` | Scenario parameters (number, description, computer, room, counts) |
| `MeasurementMetadata` | Per-measurement metadata (files, stats, quality) |

## ScenarioManager

`ScenarioManager.py` provides centralized scenario data management for the GUI.

| Feature | Description |
|---------|-------------|
| Folder parsing | Regex-based extraction of computer, scenario number, room |
| Feature availability | Checks for `features.csv` / `spectrum.csv` existence |
| Filtering/sorting | By computer, room, text filter |
| Caching | DataFrame cached in Streamlit session state |
| Multi-channel detection | Uses `multichannel_filename_utils` to detect channel count |

## Multi-Channel Filename Utilities

`multichannel_filename_utils.py` provides:

| Function | Description |
|----------|-------------|
| `parse_multichannel_filename(name)` | Extract channel number from filename |
| `group_files_by_measurement(files)` | Group files sharing same measurement index |
| `group_files_by_channel(files)` | Group files by channel number |
| `detect_num_channels(files)` | Count distinct channels in file set |
| `is_multichannel_dataset(folder)` | Check if folder contains multi-channel data |

## collect_dataset.py (CLI)

Command-line interface for single and series collection.

```bash
# Single
python collect_dataset.py --quiet --scenario-number 1

# Series
python collect_dataset.py --series "0.1,0.2,1-3" --pre-delay 60
```

Series expression parsing: comma-separated values and dash ranges (e.g., `"0.1,0.2,1-3"` expands to `["0.1", "0.2", "1", "2", "3"]`).
