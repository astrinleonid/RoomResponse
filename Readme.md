# Room Response Recording & ML Analysis Project

A comprehensive Python-based acoustic room response measurement system with machine learning analysis capabilities. The system combines custom pulse train signal generation with advanced feature extraction and classification for precise audio analysis and room characterization.

## 🎯 Project Overview

This project enables automated collection and analysis of room acoustic response datasets for research, audio analysis, and machine learning applications. It uses carefully designed pulse train signals to measure how rooms respond to audio, capturing characteristics like reverberation, echo, and acoustic signatures under different conditions.

## 🏗️ Architecture

### Core Components
- **C++ SDL Audio Core**: Low-level audio engine with pybind11 Python bindings
- **Python Audio Processing**: High-level room response recorder with signal processing
- **Feature Extraction Engine**: Multi-modal feature extraction (MFCC, FFT spectrum)
- **ML Classification System**: SVM and Logistic Regression models for scenario discrimination
- **Visualization Tools**: HTML-based spectrum analyzer and result plotting

### System Architecture
- **Hybrid Audio Approach**: SDL for device management + direct audio I/O for synchronized recording/playbook
- **Modular Feature Pipeline**: Separate feature extraction and dataset building
- **Configuration-Based**: JSON configuration files for signal parameters and defaults
- **ML-Ready Output**: Direct integration with scikit-learn ecosystem

## 📁 Project Structure

```
RoomResponse/
├── sdl_audio_core/                    # C++ module with Python bindings
│   ├── src/
│   │   ├── audio_engine.h/cpp              # Core audio engine
│   │   ├── device_manager.h/cpp            # Device enumeration
│   │   └── python_bindings.cpp            # pybind11 interface
│   └── build system files
├── RoomResponseRecorder.py            # Main recorder class (config-based)
├── DatasetCollector.py                # Single scenario dataset collector
├── collect_dataset.py                # Command-line wrapper script
├── batch_collect.ps1                  # PowerShell batch collection script
├── recorderConfig.json                # Signal configuration file
│
├── FeatureExtractor.py         # Core feature extraction engine
├── ScenarioClassifier.py             # ML classification system
├── test_data_generator.py             # Synthetic test data generator
├── test_classifier.ps1                # Comprehensive ML testing suite
├── spectrum_visualizer.html           # Interactive spectrum visualization
│
├── room_response_dataset/             # Output directory (created automatically)
│   └── <computer>-Scenario<num>-<room>/
│       ├── impulse_responses/              # Original recorded audio
│       ├── features.csv                    # MFCC features
│       ├── spectrum.csv                    # FFT spectrum features
│       ├── metadata/                       # Session and measurement metadata
│       └── analysis/                       # Analysis outputs
│
└── README.md                          # This file
```

## 🚀 Quick Start

### Prerequisites

1. **Python 3.8+** with numpy, pandas, librosa, scikit-learn, matplotlib
2. **SDL2** audio library
3. **Compiled SDL audio core** (C++ module)
4. **Audio devices**: Working microphone and speakers

### Installation

```bash
pip install numpy pandas librosa scikit-learn matplotlib seaborn scipy
# For advanced visualization
pip install plotly soundfile
```

### Basic Workflow

1. **Record Audio Data**:
```bash
# Interactive mode
python collect_dataset.py

```

2. **Extract Features**:
```bash
# Extract MFCC and spectrum features for all scenarios
python FeatureExtractor.py room_response_dataset impulse_responses
```

3. **Train ML Models**:
```bash
# Compare two scenarios
python ScenarioClassifier.py scenario_folder_1 scenario_folder_2

# Test on synthetic data
python test_data_generator.py
.\test_classifier.ps1
```

4. **Visualize Results**:
```bash
# Open spectrum_visualizer.html in browser
# Load any spectrum.csv file for interactive analysis
```

## 🎵 Audio Recording System

### Configuration Setup

Create or edit `recorderConfig.json`:
```json
{
  "sample_rate": 16000,
  "pulse_duration": 0.008,
  "pulse_fade": 0.0001,
  "cycle_duration": 0.1,
  "num_pulses": 8,
  "volume": 0.4,
  "pulse_frequency": 1000,
  "impulse_form": "sine",
  "computer": "MyLaptop",
  "room": "LivingRoom"
}
```

### Recording Methods

#### Single Scenario Collection
```bash
# Interactive mode
python collect_dataset.py

# Automated mode
python collect_dataset.py --quiet --scenario-number "1" --num-measurements 50
```

#### Batch Collection

# PowerShell version (Windows)
.\batch_collect.ps1
```

### Audio Quality Guidelines

**Optimal Recording Conditions:**
- Quiet environment (minimal background noise)
- Moderate volume levels (avoid too loud/quiet)
- Stable microphone and speaker positions
- Good acoustic separation between mic and speakers

**Quality Thresholds:**
- SNR: ≥ 15dB (good), ≥ 20dB (excellent)
- Clipping: < 2% (acceptable), < 1% (good)
- Dynamic Range: ≥ 25dB (acceptable), ≥ 30dB (good)

## 🔬 Feature Extraction System

### Audio Feature Extractor

The feature extraction system processes recorded audio and generates comprehensive feature sets optimized for short audio signals (impulse responses).

#### Basic Usage
```bash
# Extract features for all scenarios
python FeatureExtractor.py dataset_path wav_subfolder

# Example
python FeatureExtractor.py room_response_dataset impulse_responses
```

#### Advanced Options
```bash
# Specific recording types
python FeatureExtractor.py dataset_path impulse_responses --recording-type average

# Custom output filenames
python FeatureExtractor.py dataset_path impulse_responses \
    --mfcc-filename custom_mfcc.csv \
    --spectrum-filename custom_spectrum.csv

# Force overwrite existing files
python FeatureExtractor.py dataset_path impulse_responses --force
```

### Feature Types

#### 1. MFCC Features (features.csv)
- **13 Mel-Frequency Cepstral Coefficients** (default)
- Adaptive FFT parameters for short audio
- Perceptual audio representation
- Robust to noise and variations

**Output format:**
```csv
filename,mfcc_0,mfcc_1,mfcc_2,...,mfcc_12
sample_001.wav,-1.234,0.567,2.345,...,-0.123
sample_002.wav,-2.456,1.234,1.678,...,0.456
```

#### 2. Spectrum Features (spectrum.csv)
- **Full-length FFT analysis** (no windowing)
- ~801 frequency bins for 1600-sample audio
- Normalized magnitude spectrum
- Complete spectral information

**Output format:**
```csv
filename,freq_0,freq_1,freq_2,...,freq_800
sample_001.wav,0.234,0.567,0.345,...,0.123
sample_002.wav,0.456,0.234,0.678,...,0.456
```

### Feature Extraction API

```python
from FeatureExtractor import AudioFeatureExtractor

# Initialize extractor
extractor = AudioFeatureExtractor(sample_rate=16000, n_mfcc=13)

# Process entire dataset
extractor.process_dataset(
    dataset_path="room_response_dataset",
    wav_subfolder="impulse_responses",
    recording_type="any",
    mfcc_filename="features.csv",
    spectrum_filename="spectrum.csv"
)

# Process single scenario
success = extractor.process_scenario_folder(
    scenario_folder="path/to/scenario",
    wav_subfolder="impulse_responses"
)
```

## 🤖 Machine Learning System

### Scenario Classification

The ML system provides binary classification between two audio scenarios using either MFCC or spectrum features.

#### Supported Models
- **SVM (default)**: RBF kernel with probability estimates
- **Logistic Regression**: With feature importance analysis

#### Basic Usage
```bash
# Basic classification (SVM + Spectrum)
python ScenarioClassifier.py scenario_folder_1 scenario_folder_2

# Custom model and features
python ScenarioClassifier.py scenario_folder_1 scenario_folder_2 \
    --model logistic \
    --features mfcc \
    --test-size 0.2 \
    --cv-folds 10
```

#### Advanced Options
```bash
# Custom labels and output
python ScenarioClassifier.py scenario_folder_1 scenario_folder_2 \
    --scenario1-label "Empty Room" \
    --scenario2-label "Furnished Room" \
    --output-dir classification_results \
    --no-plot
```

### Classification API

```python
from ScenarioClassifier import ScenarioClassifier

# Initialize classifier
classifier = ScenarioClassifier(model_type='svm', feature_type='spectrum')

# Prepare dataset
X, y, feature_names, label_names = classifier.prepare_dataset(
    scenario1_folder, scenario2_folder,
    scenario1_label="Empty", scenario2_label="Furnished"
)

# Train and evaluate
results = classifier.train_and_evaluate(X, y, test_size=0.3, cv_folds=5)

# Print results
classifier.print_results(results)

# Generate plots
classifier.plot_results(results, save_path="results.png")
```

### Performance Evaluation

The system provides comprehensive evaluation metrics:

- **Training/Test Accuracy**: Model performance on seen/unseen data
- **Cross-Validation**: Robust performance estimation
- **Confusion Matrix**: Classification breakdown
- **Feature Importance**: Most discriminative features/frequencies
- **Classification Report**: Precision, Recall, F1-score per class

### Expected Performance

| Scenario Comparison | Expected Accuracy | Difficulty |
|-------------------|------------------|------------|
| Random Noise vs Random Noise | ~50% | Hard (chance level) |
| Noise vs Tonal Signals | 80-90% | Medium |
| Different Frequency Bands | 95%+ | Easy |
| Room Characteristics | 70-95% | Varies |

## 🧪 Testing and Validation

### Synthetic Test Data Generation

Generate controlled test datasets for algorithm validation:

```bash
# Generate 4 test scenarios (100 samples each)
python test_data_generator.py --output-dir test_dataset --num-samples 100

# Scenarios created:
# 1. Pure white noise (level 0.10)
# 2. Pure white noise (level 0.12)
# 3. Low frequencies (250, 500, 750 Hz) + noise
# 4. High frequencies (2000, 3500, 5000 Hz) + noise
```

### Comprehensive Testing Suite

Test classifier on all possible scenario pairs:

```bash
# PowerShell comprehensive test (Windows)
.\test_classifier.ps1

# Test with different configurations
.\test_classifier.ps1 -Model logistic -Features mfcc

# Custom directories
.\test_classifier.ps1 -TestDataDir my_test_data -OutputDir my_results
```

#### Test Output
- **Real-time progress** with color-coded results
- **Performance ranking** by accuracy
- **Detailed reports** (CSV and text)
- **Expected vs actual** results comparison
- **Feature importance analysis**

### Test Data API

```python
from test_data_generator import TestDataGenerator

# Initialize generator
generator = TestDataGenerator(sample_rate=16000, duration_samples=1600)

# Generate pure noise
noise_samples = generator.generate_pure_noise(num_samples=100, noise_level=0.1)

# Generate tonal signals
tonal_samples = generator.generate_tonal_with_noise(
    frequencies=[1000, 2000, 3000],
    num_samples=100,
    tone_amplitude=0.3,
    noise_level=0.05
)

# Create complete scenario folder
generator.create_scenario_folder(
    audio_samples=tonal_samples,
    scenario_name="Test-Scenario1-Tonal",
    base_output_dir="test_data"
)
```

## 📊 Visualization Tools

### Interactive Spectrum Visualizer

Web-based tool for analyzing extracted spectrum features:

1. **Open** `spectrum_visualizer.html` in any modern browser
2. **Load** any `spectrum.csv` file from your scenarios
3. **Explore** with multiple visualization types:
   - **Line Plot**: Average spectrum with confidence intervals
   - **Heatmap**: All files vs frequency
   - **3D Surface**: Interactive 3D representation
   - **Individual Plots**: Grid view of separate spectra

#### Features
- **Real-time parameter adjustment**
- **Frequency range selection**
- **Statistical analysis**
- **High-resolution export**
- **Drag & drop file loading**

### Programmatic Visualization

```python
import matplotlib.pyplot as plt
import pandas as pd

# Load spectrum data
spectrum_df = pd.read_csv("scenario_folder/spectrum.csv")

# Extract frequency columns
freq_cols = [col for col in spectrum_df.columns if col.startswith('freq_')]
spectrum_data = spectrum_df[freq_cols].values

# Calculate frequencies
sample_rate = 16000
frequencies = np.linspace(0, sample_rate/2, len(freq_cols))

# Plot average spectrum
avg_spectrum = np.mean(spectrum_data, axis=0)
plt.plot(frequencies, avg_spectrum)
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.title('Average Room Response Spectrum')
plt.show()
```

## ⚙️ Configuration Examples

### High-Quality Research Configuration
```json
{
  "sample_rate": 96000,
  "pulse_duration": 0.005,
  "cycle_duration": 0.15,
  "num_pulses": 12,
  "volume": 0.3,
  "impulse_form": "sine"
}
```

### Fast Survey Configuration
```json
{
  "sample_rate": 44100,
  "pulse_duration": 0.01,
  "cycle_duration": 0.08,
  "num_pulses": 6,
  "volume": 0.5,
  "impulse_form": "square"
}
```

### Short Audio Optimized (Default)
```json
{
  "sample_rate": 16000,
  "pulse_duration": 0.008,
  "n_fft": 512,
  "hop_length": 128,
  "n_mfcc": 13
}
```

## 📈 Research Applications

### Acoustic Analysis
- **Room reverberation time measurement**
- **Acoustic signature classification**
- **Echo and reflection analysis**
- **Frequency response characterization**
- **Environmental acoustic monitoring**

### Machine Learning Applications
- **Room type classification**
- **Occupancy detection**
- **Audio source localization**
- **Acoustic scene analysis**
- **Audio anomaly detection**

### Audio Engineering
- **Room acoustic optimization**
- **Speaker placement analysis**
- **Acoustic treatment effectiveness**
- **Sound system calibration**
- **Audio quality assessment**

## 🔧 API Reference

### RoomResponseRecorder Class

```python
from RoomResponseRecorder import RoomResponseRecorder

recorder = RoomResponseRecorder("recorderConfig.json")
audio_data = recorder.take_record(
    output_file="recording.wav",
    impulse_file="impulse.wav",
    method=2
)
```

### AudioFeatureExtractor Class

```python
from FeatureExtractor import AudioFeatureExtractor

extractor = AudioFeatureExtractor(sample_rate=16000, n_mfcc=13)

# Process single folder
features_df = extractor.extract_features_from_folder(
    folder_path="path/to/wav/files",
    recording_type="any"
)

# Process entire dataset
extractor.process_dataset(
    dataset_path="room_response_dataset",
    wav_subfolder="impulse_responses"
)
```

### ScenarioClassifier Class

```python
from ScenarioClassifier import ScenarioClassifier

classifier = ScenarioClassifier(model_type='svm', feature_type='spectrum')

# Prepare and train
X, y, features, labels = classifier.prepare_dataset(folder1, folder2)
results = classifier.train_and_evaluate(X, y)

# Analyze results
classifier.print_results(results)
classifier.plot_results(results)
classifier.save_model_info(results, folder1, folder2, "output_dir")
```

## 📊 Dataset Output Structure

### Naming Convention
Datasets use the convention: `<computer>-Scenario<number>-<room>`

Examples:
- `MyLaptop-Scenario1-LivingRoom`
- `ResearchPC-Scenario0.1-Lab`
- `FieldDevice-Scenario5a-Auditorium`

### Generated Files

Each scenario folder contains:

```
<computer>-Scenario<num>-<room>/
├── impulse_responses/
│   ├── raw_<scenario>_001_TIMESTAMP.wav      # Original recordings
│   ├── raw_<scenario>_002_TIMESTAMP.wav
│   └── ...
├── features.csv                              # MFCC features (13 coefficients)
├── spectrum.csv                              # FFT spectrum (~801 frequency bins)
├── metadata/
│   ├── session_metadata.json                # Recording session info
│   └── measurement_log.csv                  # Per-measurement details
└── analysis/                                 # ML analysis results
    ├── classification_results.txt
    ├── feature_importance.csv
    └── classification_plots.png
```

### Metadata Structure

```json
{
  "scenario_info": {
    "scenario_name": "MyLaptop-Scenario1-LivingRoom",
    "scenario_number": "1",
    "computer_name": "MyLaptop",
    "room_name": "LivingRoom",
    "description": "Empty room measurement",
    "collection_timestamp": "2025-08-10T17:30:00"
  },
  "recorder_config": {...},
  "device_info": {...},
  "measurements": [...],
  "summary": {
    "total_measurements": 30,
    "success_rate": 100.0
  }
}
```

## 🔧 Troubleshooting

### Common Issues

#### Feature Extraction Problems
**Symptoms:** "No spectrum/mfcc features found" errors

**Solutions:**
- Ensure WAV files exist in the specified subfolder
- Check that `FeatureExtractor.py` completed successfully
- Verify CSV files were created in scenario folders

#### ML Classification Issues
**Symptoms:** Low accuracy, overfitting, or convergence problems

**Solutions:**
- Check class balance (should be roughly equal)
- Try different models (`--model logistic` vs `--model svm`)
- Switch feature types (`--features mfcc` vs `--features spectrum`)
- Adjust test size (`--test-size 0.2`)

#### Audio Loading Problems
**Symptoms:** "Failed to load audio file" warnings

**Solutions:**
- Install missing dependencies: `pip install soundfile scipy`
- Check audio file integrity
- Verify sample rate compatibility

### Performance Optimization

**For Large Datasets:**
- Use batch processing with the PowerShell script
- Process scenarios incrementally
- Consider using MFCC features for faster processing

**For Real-time Analysis:**
- Reduce number of features extracted
- Use lower sample rates
- Implement feature caching

## 🤝 Contributing

### Development Setup

1. Clone repository with SDL audio core
2. Build C++ audio engine with pybind11
3. Install Python dependencies
4. Create `recorderConfig.json` with your settings
5. Test with synthetic data generation

### Code Style

- Follow PEP 8 for Python code
- Use type hints for all public methods
- Comprehensive docstrings for classes and methods
- Error handling with informative messages
- JSON configuration for all parameters

### Testing

```bash
# Generate test data
python test_data_generator.py

# Run comprehensive tests
.\test_classifier.ps1

# Test individual components
python FeatureExtractor.py test_dataset impulse_responses
python ScenarioClassifier.py test_dataset/scenario1 test_dataset/scenario2
```

## 📄 License

[Specify your license here]

## 🆘 Support

For issues, questions, or contributions:

1. **Check troubleshooting section** above
2. **Review audio quality guidelines**
3. **Test with synthetic data** using `test_data_generator.py`
4. **Verify feature extraction** completed successfully
5. **Validate configuration file** syntax

### Performance Benchmarks

**Expected Processing Times:**
- Feature extraction: ~1-2 seconds per 100 audio files
- ML training: ~5-30 seconds depending on dataset size
- Visualization: Near real-time for datasets up to 1000 samples

**Memory Requirements:**
- Audio processing: ~100MB for 1000 short audio files
- Feature matrices: ~50MB for spectrum features, ~5MB for MFCC
- ML models: <10MB for trained classifiers

---

**Note:** This project requires properly configured audio hardware, compiled SDL audio core module, and Python dependencies. Ensure all components are installed and configured correctly before use. The system is optimized for short audio impulse responses but can be adapted for longer recordings with configuration adjustments.