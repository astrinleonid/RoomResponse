# Piano Soundboard Impulse Response Measurements - Summary

**Generated:** 2025-11-05
**Dataset Location:** `d:\repos\RoomResponse\piano\`

---

## Overview

This dataset contains multi-channel impulse response measurements from **11 measurement points** on a piano soundboard. Each measurement point (scenario) corresponds to a specific physical location where a voice coil actuator excited the soundboard, and 6 synchronized microphones captured the acoustic response.

---

## Measurement Points Summary

| Point | Scenario | Measurements | Channels | Raw Files | Averaged |
|-------|----------|--------------|----------|-----------|----------|
| 57 | Neumann-Scenario57-Take1 | 38 | 6 | 228 | ✓ (6 files) |
| 60 | Neumann-Scenario60-Take1 | 20 | 6 | 120 | ✓ (6 files) |
| 65 | Neumann-Scenario65-Take1 | 20 | 6 | 120 | ✓ (6 files) |
| 70 | Neumann-Scenario70-Take1 | 20 | 6 | 120 | ✓ (6 files) |
| 74 | Neumann-Scenario74-Take1 | 20 | 6 | 120 | ✓ (6 files) |
| 79 | Neumann-Scenario79-Take1 | 20 | 6 | 120 | ✓ (6 files) |
| 80 | Neumann-Scenario80-Take1 | 10 | 6 | 60 | ✓ (6 files) |
| 81 | Neumann-Scenario81-Take1 | 8 | 6 | 48 | ✓ (6 files) |
| 82 | Neumann-Scenario82-Take1 | 10 | 6 | 60 | ✓ (6 files) |
| 83 | Neumann-Scenario83-Take1 | 10 | 6 | 60 | ✓ (6 files) |
| 84 | Neumann-Scenario84-Take1 | 9 | 6 | 54 | ✓ (6 files) |
| **TOTAL** | **11 points** | **185** | **6** | **1,110** | **66 files** |

---

## Recording Configuration

**Audio Parameters:**
- Sample Rate: 48,000 Hz
- Pulse Duration: 19 ms (0.019 seconds)
- Pulse Form: Voice coil excitation
- Cycle Duration: 1.0 second
- Number of Pulses per Recording: 5
- Playback Volume: 0.6

**Multi-Channel Setup:**
- Total Channels: 6 synchronized microphones
- Calibration Channel: Channel 2
- Reference Channel: Channel 5
- Response Channels: 0, 1, 3, 4, 5

**Channel Naming:**
- ch0: Channel 0
- ch1: Channel 1
- ch2: Channel 2 (Calibration)
- ch3: Channel 3
- ch4: Channel 4
- ch5: Channel 5 (Reference)

---

## Data Structure

### Directory Layout

```
piano/
├── Neumann-Scenario57-Take1/
│   ├── impulse_responses/
│   │   ├── impulse_Neumann-Scenario57-Take1_000_20251105_181455_319_ch0.npy
│   │   ├── impulse_Neumann-Scenario57-Take1_000_20251105_181455_319_ch1.npy
│   │   ├── ... (228 files total: 38 measurements × 6 channels)
│   ├── averaged_responses/
│   │   ├── average_ch0.npy
│   │   ├── average_ch1.npy
│   │   ├── average_ch2.npy
│   │   ├── average_ch3.npy
│   │   ├── average_ch4.npy
│   │   └── average_ch5.npy
│   ├── raw_recordings/
│   ├── room_responses/
│   └── metadata/
│       └── session_metadata.json
├── Neumann-Scenario60-Take1/
│   └── ... (same structure)
├── ... (9 more scenarios)
```

### File Formats

**Individual Impulse Responses:**
- Format: NumPy binary (.npy)
- Naming: `impulse_{scenario}_{index}_{timestamp}_ch{N}.npy`
- Content: 1D numpy array of float64 values (typically 28,800 samples = 0.6 seconds @ 48kHz)
- Location: `{scenario}/impulse_responses/`

**Averaged Impulse Responses:**
- Format: NumPy binary (.npy)
- Naming: `average_ch{N}.npy`
- Content: 1D numpy array of float64 values (averaged across all measurements for that channel)
- Location: `{scenario}/averaged_responses/`
- Length: 28,800 samples (0.6 seconds @ 48kHz)

**Metadata:**
- Format: JSON
- Location: `{scenario}/metadata/session_metadata.json`
- Contents: Recording configuration, device info, quality metrics, measurement list

---

## Averaged Response Statistics

All averaged responses have been successfully generated for all 11 measurement points.

**Example Statistics (Scenario 57):**

| Channel | Peak Amplitude | RMS Level |
|---------|----------------|-----------|
| ch0 | 1.0646 | 0.051569 |
| ch1 | 1.0874 | 0.054683 |
| ch2 | 0.8662 | 0.016651 |
| ch3 | 0.7272 | 0.038615 |
| ch4 | 0.6802 | 0.032730 |
| ch5 | 0.5162 | 0.028545 |

---

## Data Access

### Loading Averaged Impulse Response (Python)

```python
import numpy as np
from pathlib import Path

# Load averaged response for scenario 57, channel 0
scenario_path = Path("piano/Neumann-Scenario57-Take1")
avg_file = scenario_path / "averaged_responses" / "average_ch0.npy"
impulse_response = np.load(avg_file)

# Properties
sample_rate = 48000
duration = len(impulse_response) / sample_rate  # 0.6 seconds
print(f"Loaded impulse response: {len(impulse_response)} samples, {duration:.3f} seconds")
```

### Loading All Measurements for a Channel

```python
import numpy as np
from pathlib import Path

scenario_path = Path("piano/Neumann-Scenario57-Take1")
impulse_dir = scenario_path / "impulse_responses"

# Get all files for channel 0
ch0_files = sorted(impulse_dir.glob("*_ch0.npy"))

# Load all measurements
measurements = []
for file in ch0_files:
    signal = np.load(file)
    measurements.append(signal)

print(f"Loaded {len(measurements)} measurements for channel 0")
```

### Loading Metadata

```python
import json
from pathlib import Path

scenario_path = Path("piano/Neumann-Scenario57-Take1")
metadata_file = scenario_path / "metadata" / "session_metadata.json"

with open(metadata_file, 'r') as f:
    metadata = json.load(f)

# Access configuration
config = metadata['recorder_config']
print(f"Sample Rate: {config['sample_rate']} Hz")
print(f"Pulse Form: {config['impulse_form']}")

# Access measurement data
measurements = metadata['measurements']
print(f"Total measurements: {len(measurements)}")
```

---

## Quality Metrics

Each measurement in `session_metadata.json` includes quality metrics:

- **SNR (Signal-to-Noise Ratio)**: Measured in dB
- **Max Amplitude**: Peak signal level (0.0 to 1.0)
- **RMS Level**: Root mean square energy level
- **Clip Percentage**: Percentage of samples exceeding 95% amplitude
- **Dynamic Range**: Peak-to-RMS ratio in dB

---

## Export Options

### Option 1: Use Averaged Responses Directly

The averaged responses are ready to use and located in:
```
piano/{scenario}/averaged_responses/average_ch{N}.npy
```

### Option 2: Export to Other Formats

Use the provided utility scripts:

**Export averaged responses to WAV:**
```python
# See export_averaged_to_wav.py (to be created)
```

**Export to CSV/HDF5/MAT:**
```python
# See export_piano_data.py (to be created)
```

**Export to FIR filter format:**
- Use GUI: Scenarios Panel → FIR Filter Export Tool
- Select "Average of All Files" mode
- Specify filter length and output .fir file

---

## Analysis Workflows

### 1. Compare Response Across Measurement Points

```python
import numpy as np
from pathlib import Path

# Load averaged responses from multiple points for same channel
points = [57, 60, 65, 70, 74, 79, 80, 81, 82, 83, 84]
channel = 0

responses = {}
for point in points:
    scenario_path = Path(f"piano/Neumann-Scenario{point}-Take1")
    avg_file = scenario_path / "averaged_responses" / f"average_ch{channel}.npy"
    if avg_file.exists():
        responses[point] = np.load(avg_file)

# Now analyze spatial variation
```

### 2. Multi-Channel Analysis at Single Point

```python
import numpy as np
from pathlib import Path

# Load all channels for scenario 57
scenario_path = Path("piano/Neumann-Scenario57-Take1")
avg_dir = scenario_path / "averaged_responses"

channels = {}
for ch in range(6):
    avg_file = avg_dir / f"average_ch{ch}.npy"
    channels[ch] = np.load(avg_file)

# Analyze inter-channel relationships
```

### 3. Frequency Response Analysis

```python
import numpy as np
import matplotlib.pyplot as plt

# Load impulse response
impulse = np.load("piano/Neumann-Scenario57-Take1/averaged_responses/average_ch0.npy")

# Compute frequency response
fft = np.fft.rfft(impulse)
magnitude = np.abs(fft)
phase = np.angle(fft)

# Frequency axis
sample_rate = 48000
freqs = np.fft.rfftfreq(len(impulse), 1/sample_rate)

# Plot
plt.figure(figsize=(12, 4))
plt.subplot(121)
plt.semilogx(freqs, 20*np.log10(magnitude))
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude (dB)')
plt.grid(True)

plt.subplot(122)
plt.semilogx(freqs, phase)
plt.xlabel('Frequency (Hz)')
plt.ylabel('Phase (radians)')
plt.grid(True)
plt.show()
```

---

## Next Steps

Based on your needs, you can:

1. **Export averaged responses to a specific format** (WAV, CSV, MATLAB .mat, etc.)
2. **Perform frequency domain analysis** (FFT, transfer functions)
3. **Extract acoustic features** (resonant frequencies, decay times, modal analysis)
4. **Create spatial maps** showing response variation across soundboard
5. **Generate FIR filters** for convolution reverb or equalization

**What specific export format or analysis would you like to do next?**

---

## Scripts Available

- `analyze_piano_scenarios.py` - Summary of all scenarios
- `check_averaged_responses.py` - Verify averaged responses exist
- `generate_missing_averages.py` - Create missing averaged responses (completed)

---

## References

- Recording System: RoomResponse multi-channel impulse response system
- Documentation: `TECHNICAL_DOCUMENTATION.md`
- Multi-channel implementation: `PIANO_MULTICHANNEL_PLAN.md`
