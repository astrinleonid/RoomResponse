# Work in Progress

## ESPRIT Modal Analysis Integration

**Status:** In Progress
**Branch:** `dev`
**Related:** [ESPRIT Module](../modules/esprit/OVERVIEW.md)

Piano soundboard modal identification using TLS-ESPRIT with multi-band processing and GPU acceleration.

| Scope | Status | Notes |
|-------|--------|-------|
| Core ESPRIT algorithm | Done | `esprit_core.py` with conjugate pairing |
| Multi-band analysis | Done | `band_processing.py` |
| GPU SVD backend | Done | CuPy-based, benchmarked |
| Stabilization diagrams | Done | Multi-order sweep |
| Batch analysis | Done | `batch_esprit_analysis.py` |
| Piano data export | In Progress | `export_piano_point_responses.py` |
| GUI integration | Pending | |

## Multi-Channel Recording Refinements

**Status:** In Progress
**Branch:** `dev`
**Related:** [Recorder Module](../modules/recorder/OVERVIEW.md)

| Scope | Status | Notes |
|-------|--------|-------|
| Basic multi-channel capture | Done | Up to 32 channels via SDL2 |
| Calibration channel alignment | Done | Onset-based |
| Calibration quality validation | Done | V2 refactored, min/max ranges |
| Belarus measurement campaign | In Progress | 6-channel piano measurements |
