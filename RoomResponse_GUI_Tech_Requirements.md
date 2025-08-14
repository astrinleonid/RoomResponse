# Room Response GUI — Technical Requirements
*Last updated: 2025-08-14*

## 1) Scope & Goals
A Streamlit-based app that supports the full pipeline:
1. **Scenarios** (dataset analysis & management)
2. **Collect** new data (single scenario or series)
3. **Process** (feature extraction)
4. **Classify** (single pair / all pairs / groups)
5. **Predict** (single-sample inference using saved model)
6. **Visualize** (results & reports)

A separate **Scenario Explorer** module is used for inspecting and editing individual scenarios.

---

## 2) Modular Architecture & Files

### **Main Application**
- **`gui_launcher.py`** — main Streamlit app with `RoomResponseGUI` class
- **`ScenarioManager.py`** — centralized scenario data management class

### **Panel Modules (Class-Based)**
- **`gui_collect_panel.py`** — `CollectionPanel` class with `SingleScenarioExecutor` and `SeriesExecutor`
- **`gui_process_panel.py`** — `ProcessingPanel` class with `ProcessingExecutor`
- **`gui_classify_panel.py`** — `ClassificationPanel` class *(to be implemented)*
- **`gui_predict_panel.py`** — `PredictionPanel` class *(to be implemented)*
- **`gui_visualize_panel.py`** — `VisualizationPanel` class *(to be implemented)*
- **`scenario_explorer.py`** — `ScenarioExplorer` class *(to be implemented)*

### **Back-end Modules (No Logic Duplication)**
- `FeatureExtractor.py` — all feature extraction (MFCC + spectrum), metadata about bin resolution
- `ScenarioClassifier.py` — train/eval, save/load, inference; no plotting
- `collect_dataset.py` / `DatasetCollector.SingleScenarioCollector` — collection with append behavior, config-compat checks, pause/resume, series collection
- `RoomResponseRecorder.py` & **`sdl_audio_core`** — audio I/O & notification beeps

---

## 3) Left Sidebar Navigation (Primary UX)
Radio buttons act as **panel selectors** (only one active at a time):
- **Scenarios**
- **Collect**
- **Process**
- **Classify** *(to be implemented)*
- **Predict** *(disabled until a model is trained in this session)*
- **Visualize** *(disabled until a classification has completed this session)*

**State gating**:
- Enable **Predict** and **Visualize** only after a successful **Classify** run stores model artifacts in the session.

---

## 4) Scenarios Panel (Dataset Management)

### **Core Features (Implemented)**
- **Dataset root configuration** with validation and auto-detection
- **Scenario filtering** with regex support for scenario names
- **Bulk operations** for group label management
- **Selection controls** (select all, clear, invert)
- **Inline scenario explorer** with audio preview and feature status

### **Data Display**
**Columns (exact order & compact)**:
1. **Select** (checkbox)
2. **Scenario number** (parsed with smart sorting)
3. **Group label** (editable, comma-separated multi-labels)
4. **Features** (S=Spectrum, M=MFCC, A=Audio status)
5. **Description** (from metadata, truncated if long)
6. **Actions** (quick actions menu & explore button)

### **Filtering & Selection**
- **Primary filter**: Scenario name with regex support (e.g. `^6\.`, `0.*`)
- **Secondary filters**: Computer and Room name filters
- **Bulk label operations**: Apply labels to all filtered scenarios
- **Selection persistence**: Maintains selected scenarios across sessions

### **ScenarioManager Integration**
- **Centralized data management**: All scenario operations through `ScenarioManager` class
- **Caching**: Smart caching with manual refresh capability
- **Validation**: Dataset root validation with helpful messages
- **Parsing**: Robust scenario name parsing with fallback strategies

---

## 5) Collect Panel (Data Collection)

### **Implementation Structure**
- **`CollectionPanel`** class with modular configuration rendering
- **`SingleScenarioExecutor`** for individual scenario collection
- **`SeriesExecutor`** for multi-scenario collection with beeps and delays

### **Collection Modes**
- **Single Scenario**: Individual scenario collection with full configuration
- **Series**: Multiple scenarios with automated timing and audio cues

### **Configuration Options**
**Common fields**:
- Computer name, Room name (defaults from `recorderConfig.json`)
- Number of measurements, measurement interval
- Interactive device selection
- **Output directory selection** (new feature - can differ from dataset root)

**Series-specific**:
- **Pre-delay before first** scenario (default 60s)
- **Inter-scenario delay** (default 60s)
- **Beep configuration**: volume, frequency, duration
- **Scenario number parsing**: supports ranges like "1-3,5,7"

### **Enhanced Features**
- **Output directory flexibility**: Users can specify any directory for collection
- **Path validation**: Real-time validation of output directories
- **Preview functionality**: Shows exactly where scenarios will be saved
- **Progress delegation**: Collection runs in terminal with GUI status updates

### **Collection Behavior**
- **Folder reuse**: Append new measurements by default
- **Compatibility checks**: Verify recorder configuration compatibility
- **Pause & resume**: CTRL+C saves partial metadata, resume supported
- **SDL audio integration**: All beeps via SDL audio core (1 beep per scenario, 2 beeps at series end)

---

## 6) Process Panel (Feature Extraction)

### **Implementation Structure**
- **`ProcessingPanel`** class with configuration and preview
- **`ProcessingExecutor`** for actual feature extraction with progress tracking

### **Configuration**
**Inputs**:
- `wav_subfolder` (default `impulse_responses`)
- `recording_type` (`any` / `average` / `raw`)
- Audio parameters: sample rate (default 48kHz), MFCC coefficients (default 13)
- `max_spectrum_freq` (Hz; 0 = no limit, default 10kHz)
- Output filenames: `features.csv` (MFCC), `spectrum.csv` (spectrum)
- Config file path for reading parameters

**File handling policies**:
- **Skip scenarios** (both files present)
- **Keep existing** (write missing only)
- **Overwrite** both files

### **Processing Features**
- **Scenario selection**: Process selected scenarios or all scenarios
- **Preview table**: Shows current feature status and planned actions
- **Real-time progress**: Progress bars and status updates during processing
- **Results summary**: Detailed processing results with success/failure metrics
- **FeatureExtractor integration**: No duplication of extraction logic

---

## 7) Classify Panel *(To Be Implemented)*

### **Planned Structure**
- **`gui_classify_panel.py`** with `ClassificationPanel` class
- **Executor classes**: `SinglePairExecutor`, `AllPairsExecutor`, `GroupVsGroupExecutor`

### **Global Parameters**
- Feature type: `spectrum` or `mfcc`
- Model: `svm` or `logistic`
- `test_size` (0.05–0.5), `cv_folds` (2–10)

### **Classification Modes**
1. **Single Pair** (exactly 2 selected scenarios)
2. **All Pairs** (pairwise classification across all selected)
3. **Group vs Group** (classification between group labels)

### **Outputs**
- Results stored in session state for Predict/Visualize panels
- **Accuracy matrix** for all-pairs mode
- **Model artifacts** via `ScenarioClassifier.dumps_model_bytes(...)`
- **Downloadable model files**

---

## 8) Predict Panel *(To Be Implemented)*

### **Planned Structure**
- **`gui_predict_panel.py`** with `PredictionPanel` class
- **`PredictionExecutor`** for single-sample inference

### **Features**
- **Model source**: Use last trained or upload model file
- **Audio input**: Record new sample or select WAV file
- **Feature extraction**: Automatic feature matching with trained model
- **Prediction display**: Label predictions with confidence scores

---

## 9) Visualize Panel *(To Be Implemented)*

### **Planned Structure**
- **`gui_visualize_panel.py`** with `VisualizationPanel` class
- **Visualization classes**: `ConfusionMatrixViz`, `AccuracyMatrixViz`, `FeatureImportanceViz`

### **Visualization Types**
- **Single Pair**: Confusion matrix, CV scores, feature importance
- **All Pairs**: Accuracy heatmap with expandable per-pair details
- **Groups**: Group-based visualizations with class labels

---

## 10) Scenario Explorer *(To Be Implemented)*

### **Planned Structure**
- **`scenario_explorer.py`** with `ScenarioExplorer` class

### **Features**
- **Metadata editing**: Description, tags/labels with validation
- **File browser**: Audio files with play/visualize/delete options
- **Feature viewer**: Quick plots and data export
- **Configuration display**: Read-only recorder config with mismatch warnings

---

## 11) Data & State Management

### **Session State Keys**
- **Dataset & scenarios**: `dataset_root`, `scenarios_selected_set`, `scenarios_explore_path`
- **Filtering**: `filter_text`, `filter_computer`, `filter_room`
- **Collection**: `collection_output_override`
- **Classification results**: `classification_artifacts`, `last_model_info`
- **UI state**: `saved_labels_cache`

### **ScenarioManager Caching**
- **Smart caching**: Uses `scenarios_df_cache` with dataset root change detection
- **Cache invalidation**: Automatic on dataset root changes, manual via "Re-analyze"
- **Performance**: Efficient pandas operations, avoids re-reading files unnecessarily

---

## 12) File I/O & Compatibility

### **Scenario Naming**
- Format: `<computer>-Scenario<number>-<room>`
- **Robust parsing**: Handles alphanumeric scenario numbers (e.g., "1", "2.5", "3a")
- **Smart sorting**: Numeric sorting when possible, alphabetical fallback

### **Metadata Structure**
**On collection**:
- `scenario_meta.json` (GUI-specific: labels, descriptions)
- `metadata/session_metadata.json` (collection metadata)
- Recorder config snapshots for compatibility checks

**On processing**:
- `features.csv` (MFCC features)
- `spectrum.csv` (spectrum features)
- `features_meta.json` (FFT parameters, bin resolution)

---

## 13) Audio Requirements
- **SDL audio core integration**: All beeps and playback via SDL
- **Beep specifications**: ~880 Hz sine wave, ~200ms duration
- **Series collection beeps**: 1 beep per scenario completion, 2 beeps at series end
- **No platform dependencies**: No `winsound` or platform-specific audio libraries

---

## 14) Error Handling & UX

### **Dependency Management**
- **Graceful degradation**: Missing modules show clear error messages
- **Import isolation**: Each panel checks its own dependencies
- **User guidance**: Specific instructions for resolving missing components

### **Validation & Feedback**
- **Real-time validation**: Path validation, configuration checks
- **Progress feedback**: Progress bars for long operations
- **Clear error messages**: Actionable error descriptions with suggested solutions
- **Status indicators**: Visual feedback for dataset validity, feature availability

---

## 15) Performance & Scalability

### **Target Performance**
- **Dataset size**: ~100–200 scenarios, ~100–300 samples each
- **Memory efficiency**: Pandas/numpy optimization, minimal data copying
- **Caching strategy**: Smart caching with selective invalidation
- **UI responsiveness**: Non-blocking operations with progress feedback

### **Implementation Patterns**
- **Lazy loading**: Load data only when needed
- **Executor pattern**: Separate UI from processing logic
- **Session state optimization**: Minimal session state footprint

---

## 16) Development Guidelines

### **Panel Implementation Pattern**
1. **Create panel file**: `gui_[panel_name]_panel.py`
2. **Implement panel class**: `[PanelName]Panel` with `render()` method
3. **Add executor classes**: Separate UI from business logic
4. **Import in main**: Add to `gui_launcher.py` with error handling
5. **Update navigation**: Add to sidebar radio options

### **Code Organization Principles**
- **Single responsibility**: Each class has one clear purpose
- **Dependency injection**: Pass `scenario_manager` to panels
- **Error isolation**: Panel failures don't crash the application
- **Consistent patterns**: All panels follow the same structure

---

## 17) Acceptance Criteria

### **Implemented Features** ✅
1. **Scenarios panel**: Pattern filtering, selection, group labels, inline explorer
2. **Collect panel**: Single/series modes, output directory selection, SDL beeps, progress delegation
3. **Process panel**: FeatureExtractor integration, file policies, real-time progress
4. **ScenarioManager**: Centralized data management with caching and validation
5. **Modular architecture**: Clean separation of concerns with class-based panels

### **To Be Implemented** 🚧
6. **Classify panel**: All three modes, model artifacts, accuracy matrices
7. **Predict panel**: Single-sample inference with model compatibility checks
8. **Visualize panel**: Comprehensive result visualization with interactive charts
9. **Scenario Explorer**: Standalone scenario inspection and editing interface

---

## 18) Technical Implementation Notes

### **UI Framework Considerations**
- **Streamlit limitations**: No native right-click menus (using ⚙️ action buttons)
- **Session state management**: Careful handling of state persistence and cache invalidation
- **Progress indication**: Terminal delegation for long-running operations

### **Integration Points**
- **Back-end modules**: All data processing stays in respective modules
- **Feature extraction**: GUI only invokes, never duplicates FeatureExtractor logic
- **Audio operations**: All audio through SDL core, no platform-specific code
- **Model artifacts**: Binary model storage in session state for cross-panel access

### **Future Extensibility**
- **Plugin architecture**: Easy addition of new panels following established patterns
- **Configuration system**: Centralized configuration with validation
- **Export capabilities**: Model download, data export, report generation
- **Advanced features**: Batch processing, remote data sources, cloud integration