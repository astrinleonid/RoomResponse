# Room Response GUI — Technical Requirements
*Last updated: 2025-08-13*

## 1) Scope & Goals
A Streamlit-based app that supports the full pipeline:
1. **Analyze** dataset
2. **Collect** new data (single scenario or series)
3. **Process** (feature extraction)
4. **Classify** (single pair / all pairs / groups)
5. **Predict** (single-sample inference using saved model)
6. **Visualize** (results & reports)

A separate **Scenario Explorer** page/module is used for inspecting and editing a single scenario.

---

## 2) Modules / Files
- **`gui_launcher.py`** — main Streamlit app/UI.
- **`scenario_explorer.py`** — separate page/panel launched from the main app.
- Back-end modules (no duplication of logic in the GUI):
  - `FeatureExtractor.py` — all feature extraction (MFCC + spectrum), metadata about bin resolution, etc.
  - `ScenarioClassifier.py` — clean version; contains train/eval, save/load, inference; no plotting.
  - `collect_dataset.py` / `DatasetCollector.SingleScenarioCollector` — collection with append behavior, config-compat checks, pause/resume, series collection with delays and beeps via SDL.
  - `RoomResponseRecorder.py` & **`sdl_audio_core`** — audio I/O & notification beeps (no `winsound`).

---

## 3) Left Sidebar Navigation (Primary UX)
Buttons act as **expandable panels** (only one expanded at a time):
- **Analyze dataset**
- **Collect**
- **Process**
- **Classify**
- **Predict** *(disabled until a model is trained in this session)*
- **Visualize** *(disabled until a classification has completed this session)*

**State gating**:
- Enable **Predict** and **Visualize** only after a successful **Classify** run stores model artifacts in the session.

---

## 4) Analyze Panel (Dataset Scan)
**Inputs**
- Dataset root path

**Behavior**
- Scans **all** scenario folders (names containing “scenario”, case-insensitive).
- Reads `metadata/session_metadata.json` if present.
- Lists scenarios **even without features** (so they can be processed).
- Produces:
  - `scenarios: dict`
  - `df_all: DataFrame` with: `key`, `scenario_number`, `description`, `computer`, `room`, `sample_count`, `features_available.spectrum/mfcc/audio`, `full_path`.
- Cache with `@st.cache_data`; provide **Re-analyze** button to clear cache & rebuild.

**Empty state**
- If none found: warning + stop further rendering.

---

## 5) Scenarios Block (Central List & Selection)
**Columns (exact order & compact)**:
1. **Select** (checkbox)
2. **Scenario number**
3. **Group label** (free-form, comma-separated multi-labels)
4. **Explore** (button to open Scenario Explorer)

**Filters**
- **Scenario name filter** (pattern/regex; e.g. `0.*`, `^6\.` is primary).
- (Optional) Secondary filters by **Computer** and **Room** may remain.

**Bulk actions**
- “Assign group label to filtered set” (adds a label to all currently filtered rows).

**Selection semantics**
- **Selected**: used for **Single Pair** and **All Pairs** classification.
- **Group labels**: used for **Group vs Group** classification.

**Contextual actions**
- Right-click on **Scenario number** → **Select**, **Add to group**, **Open Scenario Explorer**.
  - Note: Streamlit has no native right-click; implement via custom component or a small (⋮) actions button per row.

**Persistence**
- Maintain `labels_map` (`{key: set(labels)}`) and `selected_map` (`{key: bool}`) in `st.session_state` and re-apply to `df_all`.

---

## 6) Collect Panel (Data Collection)
**Modes**
- **Single Scenario**
- **Series** (list of scenario numbers)

**Common fields**
- Computer name, Room name (use defaults from `recorderConfig.json` when available)
- Scenario number(s)
- Description (optional)
- `num_measurements`, `measurement_interval` (seconds)
- Interactive devices (bool)

**Series options**
- **Pre-delay before the first** scenario (default 60s).
- **Inter-scenario delay** (default 60s).
- **Beep cues via SDL**:
  - One beep upon completion of **each scenario**.
  - Two beeps at the **end of the series**.

**Folder re-use**
- If scenario folder exists → **append** new measurements by default.
- Verify compatibility with saved recorder configuration; warn if mismatched and allow proceed/cancel.

**Pause & resume**
- CTRL+C / cancel saves partial metadata; re-run resumes safely.

**Post-collection**
- Save/merge metadata: `session_metadata.json`, measurement log, config snapshot.
- Offer **Re-analyze** to refresh scenario list.

---

## 7) Process Panel (Feature Extraction)
**Inputs**
- `wav_subfolder` (default `impulse_responses`)
- `recording_type` (`any` / `average` / `raw`)
- `max_spectrum_freq` (Hz; 0 → disabled)
- `mfcc_filename` (default `features.csv`)
- `spectrum_filename` (default `spectrum.csv`)
- Existing files policy:
  - **Skip** scenario (both files present)
  - **Keep existing (write missing only)**
  - **Overwrite** both files

**Behavior**
- Call `AudioFeatureExtractor.process_scenario_folder(...)` on **Selected** or **All** scenarios.
- Per-scenario progress and summary.
- Prompt **Re-analyze** on completion.

**Note**
- **No** extraction code duplicated in GUI; everything goes through `FeatureExtractor.py`.

---

## 8) Classify Panel
**Global parameters**
- Feature type: `spectrum` or `mfcc`
- Model: `svm` or `logistic`
- `test_size` (0.05–0.5), `cv_folds` (2–10)

**Modes**
1. **Single Pair** (exactly 2 selected)
2. **All Pairs** (across all selected; compute pairwise)
3. **Group vs Group**
   - Choose two group labels from existing ones
   - Balance option to downsample to min class size

**Pre-flight checks**
- Ensure required feature CSVs exist for chosen type; show missing list.

**Outputs (saved to session)**
- Results: train/test accuracy, CV scores, confusion matrix, classification report.
- **All Pairs**: **accuracy matrix** (heatmap of pairwise accuracy).
- Model artifacts bytes via `ScenarioClassifier.dumps_model_bytes(...)`, including:
  - Model pipeline
  - Label encoder / names
  - Feature names & feature type
  - Relevant metadata (e.g., bin resolution)

**Post-conditions**
- Enable **Predict** and **Visualize**.
- Provide **download** of model file.

---

## 9) Predict Panel (Inference)
**Availability**
- Enabled after any successful classification in session.

**Inputs**
- Model source: **Use last trained** (default) or **upload model** file.
- Record a single new sample (via recorder / SDL) or select a WAV file.
- Use `FeatureExtractor` to create features matching the model’s feature type and columns.

**Flow**
1. Record/acquire sample.
2. Extract features via `FeatureExtractor`.
3. Load model (bytes/file) via `ScenarioClassifier`.
4. Predict label & probabilities; display nicely and log.

**Constraints**
- Preprocessing must match training pipeline exactly (scaler, feature order/columns).
- Detect and report mismatches clearly.

---

## 10) Visualize Panel
**Single Pair**
- Confusion Matrix
- CV Scores per fold
- Feature Importance (Top-K, barh). For spectrum, show bin resolution (from `features_meta.json` if available).

**All Pairs**
- **Accuracy matrix** (heatmap).
- Per-pair expanders: metrics, confusion matrix, CV, importance.

**Groups**
- Analogous visualizations with group labels as class names.

**UX**
- Heavy charts inside **expanders** to keep the UI tidy.

---

## 11) Scenario Explorer (Separate Page/Module)
**Header**
- Scenario name, quick stats.

**Metadata Editor**
- Edit: description, tags/labels.
- Read-only: recorder config snapshot; warn if mismatched with current config.
- Save back to `metadata/session_metadata.json`.

**Files Browser**
- Lists: raw recordings, impulse responses, room responses.
- Per-file: **play** (SDL), **download**, (optional) **delete**.

**Feature Viewer**
- Load `spectrum.csv` / `features.csv`.
- Quick plots (avg spectrum, MFCC stats) and small tables.
- Export visible snippet to CSV.

**Navigation**
- Back to main, preserve state.

---

## 12) Data & State Management
**`st.session_state` keys**
- `"analyzed"`: bool
- `"scenarios"`: dict (all scenarios, valid + incomplete)
- `"df_all"`: DataFrame for scenarios
- `"labels_map"`: `{key: set([...])}`
- `"selected_map"`: `{key: bool}`
- `"single_last"`, `"all_last"`, `"group_last"`: result packs
- `"classification_artifacts"`: includes `model_bytes`, `feature_type`, `feature_names`, `label_names`
- `"last_model_info"`: summary for Predict

**Caching**
- `@st.cache_data` for `analyze_dataset_cached(path)`.
- **Re-analyze** clears cache and refreshes immediately.

---

## 13) File I/O & Compatibility
**Scenario naming**
- `<computer>-Scenario<number>-<room>`

**On collection**
- Create or reuse scenario folder (append by default).
- Write/merge metadata including:
  - `scenario_info` (names, number, description, timestamp)
  - `recorder_config` snapshot for compatibility checks
  - `device_info`, thresholds, measurement log, summary

**On processing**
- `features.csv` (MFCC), `spectrum.csv` (spectrum).
- `features_meta.json` records FFT/sample rate/bin resolution.

**On classification**
- Model bytes downloadable; stored in session for Predict/Visualize.

---

## 14) Audio Requirements
- All beeps & playback via **SDL audio core**.
- Provide a simple sine beep (~800–1000 Hz, 150–250 ms).
- **Series collection** rules:
  - 1 beep after each scenario
  - 2 beeps after the series

---

## 15) Error Handling & UX
- Guard on empty `df_all` or missing `scenarios`.
- Clear, actionable errors:
  - Missing features for selected rows
  - Model/feature type mismatch on prediction
  - CSV/IO problems
  - Recorder initialization failures
- Progress bars for long loops.
- No long-running background threads beyond Streamlit standard.

---

## 16) Performance & Limits
- Target: ~100–200 scenarios, each with up to ~100–300 samples.
- Efficient pandas/numpy usage; avoid re-reading CSVs in tight loops.
- For “All Pairs”: compute once, cache results for visualization.

---

## 17) Acceptance Criteria
1. **Analyze** finds scenarios incl. no-feature ones; re-analyze updates counts after collect/process.
2. **Scenarios list** supports pattern filter, selection, group labels, and **Explore**.
3. **Collect** supports single/series; pre-delay & inter-delay; SDL beeps; append behavior; metadata & config snapshot; pause/resume.
4. **Process** runs `FeatureExtractor` with overwrite/keep/skip; re-analyze offered after completion.
5. **Classify** works in all three modes; missing-feature checks; artifacts stored; accuracy matrix for all-pairs.
6. **Predict** records/loads a single sample, extracts features, loads model, predicts; mismatch errors clear.
7. **Scenario Explorer** edits metadata, plays audio, shows features and small plots.

---

## 18) Implementation Notes
- Right-click menu requires a custom component or a per-row (⋮) actions menu as fallback.
- Keep **all data processing** in their respective back-end modules.
- GUI only **loads** features; does not recompute them unless invoking `FeatureExtractor`.
- Spectrum bin labels use `features_meta.json` where available (fallback to derived bin size or 30 Hz).
