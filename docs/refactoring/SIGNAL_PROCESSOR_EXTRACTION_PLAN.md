# SignalProcessor Extraction Plan

**Date:** 2025-11-03
**Goal:** Extract signal processing logic from `RoomResponseRecorder` into separate `SignalProcessor` class
**Status:** 📋 PLANNING PHASE

---

## Executive Summary

Extract all signal processing methods (cycle extraction, alignment, normalization, averaging, spectral analysis) from the `RoomResponseRecorder` class into a new `SignalProcessor` class. This will:

- **Separate concerns:** Recording vs. Processing
- **Improve testability:** Signal processing can be tested independently
- **Enable reusability:** SignalProcessor can be used in other contexts
- **Simplify RoomResponseRecorder:** Focus on recording and file I/O

---

## Current Architecture Analysis

### Signal Processing Methods in RoomResponseRecorder

Currently, these methods are embedded in `RoomResponseRecorder` (1,586 lines):

#### **Universal Methods** (used by all modes)
1. `_extract_cycles()` - Lines 694-717 (24 lines)
2. `_average_cycles()` - Lines 719-735 (17 lines)
3. `_compute_spectral_analysis()` - Lines 737-796 (60 lines)
4. `_find_sound_onset()` - Lines 963-982 (20 lines)

#### **Standard Mode Methods**
5. `_find_onset_in_room_response()` - Lines 906-925 (20 lines)
6. `_extract_impulse_response()` - Lines 937-961 (25 lines)

#### **Calibration Mode Methods**
7. `align_cycles_by_onset()` - Lines 1007-1140 (134 lines)
8. `apply_alignment_to_channel()` - Lines 1142-1192 (51 lines)
9. `_normalize_by_calibration()` - Lines 1194-1276 (83 lines)

**Total Signal Processing Code:** ~434 lines (~27% of RoomResponseRecorder)

### Dependencies

These signal processing methods depend on:

**From RoomResponseRecorder attributes:**
- `self.num_pulses` - Number of cycles to record
- `self.cycle_samples` - Samples per cycle
- `self.sample_rate` - Audio sample rate
- `self.multichannel_config` - Multi-channel configuration dict

**External:**
- `numpy` - Array operations
- No other external dependencies for signal processing itself

### Called By

These signal processing methods are called by:

1. `_process_single_channel_signal()` - Calls: extract_cycles, average_cycles, extract_impulse, compute_spectral
2. `_process_multichannel_signal()` - Calls: extract_cycles, average_cycles, find_onset, compute_spectral
3. `_process_calibration_mode()` - Calls: extract_cycles, align_cycles_by_onset, apply_alignment, normalize, average_cycles

---

## Target Architecture

### New Class: SignalProcessor

```python
# signal_processor.py

from typing import Dict, List, Tuple, Optional, Any
import numpy as np


class SignalProcessingConfig:
    """Configuration for signal processing operations"""

    def __init__(self,
                 num_pulses: int,
                 cycle_samples: int,
                 sample_rate: int,
                 multichannel_config: Dict[str, Any] = None):
        """
        Initialize signal processing configuration.

        Args:
            num_pulses: Number of cycles/pulses
            cycle_samples: Samples per cycle
            sample_rate: Audio sample rate (Hz)
            multichannel_config: Optional multi-channel configuration
        """
        self.num_pulses = num_pulses
        self.cycle_samples = cycle_samples
        self.sample_rate = sample_rate
        self.multichannel_config = multichannel_config or {}


class SignalProcessor:
    """
    Signal processing operations for impulse response measurements.

    Handles cycle extraction, alignment, normalization, averaging, and spectral analysis.
    Separated from recording logic for better modularity and testability.
    """

    def __init__(self, config: SignalProcessingConfig):
        """
        Initialize signal processor with configuration.

        Args:
            config: Signal processing configuration
        """
        self.config = config

    # ========================================================================
    # UNIVERSAL METHODS (used by all modes)
    # ========================================================================

    def extract_cycles(self, audio: np.ndarray) -> np.ndarray:
        """Extract cycles from raw audio using simple reshape."""
        pass

    def average_cycles(self, cycles: np.ndarray, start_cycle: int = None) -> np.ndarray:
        """Average cycles starting from start_cycle."""
        pass

    def compute_spectral_analysis(self,
                                   responses: Dict[int, np.ndarray],
                                   window_start: float = 0.0,
                                   window_end: float = 1.0) -> Dict[str, Any]:
        """Compute FFT spectral analysis of responses."""
        pass

    def find_sound_onset(self, audio: np.ndarray,
                        window_size: int = 10,
                        threshold_factor: float = 2) -> int:
        """Find sound onset using moving average and derivative."""
        pass

    # ========================================================================
    # STANDARD MODE METHODS
    # ========================================================================

    def find_onset_in_room_response(self, room_response: np.ndarray) -> int:
        """Find onset position in a room response (simple alignment)."""
        pass

    def extract_impulse_response(self, room_response: np.ndarray) -> np.ndarray:
        """Extract impulse response by finding onset and rotating signal."""
        pass

    # ========================================================================
    # CALIBRATION MODE METHODS
    # ========================================================================

    def align_cycles_by_onset(self,
                              initial_cycles: np.ndarray,
                              validation_results: List[Dict[str, Any]],
                              correlation_threshold: float = 0.7) -> Dict[str, Any]:
        """
        Align cycles by detecting onset (negative peak) in each cycle.
        Advanced per-cycle alignment with cross-correlation filtering.
        """
        pass

    def apply_alignment_to_channel(self,
                                    channel_raw: np.ndarray,
                                    alignment_metadata: Dict[str, Any]) -> np.ndarray:
        """Apply alignment shifts (calculated from calibration channel) to any channel."""
        pass

    def normalize_by_calibration(self,
                                  aligned_multichannel_cycles: Dict[int, np.ndarray],
                                  validation_results: List[Dict],
                                  calibration_channel: int,
                                  valid_cycle_indices: List[int]) -> Tuple[Dict[int, np.ndarray], List[float]]:
        """Normalize response channels by calibration signal magnitude."""
        pass
```

### Updated RoomResponseRecorder

```python
# RoomResponseRecorder.py

from signal_processor import SignalProcessor, SignalProcessingConfig


class RoomResponseRecorder:
    """
    Room response recorder with integrated signal processing.

    Responsibilities:
    - Audio recording (SDL)
    - File I/O (saving/loading)
    - Configuration management
    - Pipeline orchestration (delegates processing to SignalProcessor)
    """

    def __init__(self, config_file_path: str = None):
        # ... existing initialization ...

        # Create signal processor with current configuration
        self.signal_processor = self._create_signal_processor()

    def _create_signal_processor(self) -> SignalProcessor:
        """Create SignalProcessor instance with current configuration."""
        config = SignalProcessingConfig(
            num_pulses=self.num_pulses,
            cycle_samples=self.cycle_samples,
            sample_rate=self.sample_rate,
            multichannel_config=self.multichannel_config
        )
        return SignalProcessor(config)

    def _process_single_channel_signal(self, recorded_audio: np.ndarray) -> Dict[str, Any]:
        """Process single-channel standard recording (delegates to SignalProcessor)"""

        # Delegate to SignalProcessor
        cycles = self.signal_processor.extract_cycles(recorded_audio)
        start_cycle = max(1, self.num_pulses // 4)
        room_response = self.signal_processor.average_cycles(cycles, start_cycle)
        impulse_response = self.signal_processor.extract_impulse_response(room_response)

        # Compute spectral analysis
        spectral_analysis = self.signal_processor.compute_spectral_analysis(
            {0: room_response},
            window_start=0.0,
            window_end=1.0
        )

        return {
            'raw': {0: recorded_audio},
            'individual_cycles': {0: cycles},
            'room_response': {0: room_response},
            'impulse': {0: impulse_response},
            'spectral_analysis': spectral_analysis,
            'metadata': {
                'mode': 'standard',
                'num_channels': 1,
                'num_cycles': self.num_pulses,
                'cycles_used_for_averaging': self.num_pulses - start_cycle,
                'averaging_start_cycle': start_cycle,
                'cycle_samples': self.cycle_samples,
                'sample_rate': self.sample_rate,
            }
        }

    # Similar delegation pattern for other processing methods...
```

---

## Refactoring Plan

### Phase 1: Create SignalProcessor Module (2 hours)

**Goal:** Create new `signal_processor.py` file with class structure

**Tasks:**
1. Create `signal_processor.py` file
2. Define `SignalProcessingConfig` dataclass
3. Define `SignalProcessor` class skeleton
4. Add docstrings and type hints
5. Add comprehensive module documentation

**Deliverables:**
- `signal_processor.py` with complete class structure
- All method signatures defined
- Comprehensive documentation

### Phase 2: Extract Universal Methods (2 hours)

**Goal:** Move universal signal processing methods to SignalProcessor

**Methods to Extract:**
1. `_extract_cycles()` → `extract_cycles()`
2. `_average_cycles()` → `average_cycles()`
3. `_compute_spectral_analysis()` → `compute_spectral_analysis()`
4. `_find_sound_onset()` → `find_sound_onset()`

**Process:**
1. Copy method to SignalProcessor
2. Update to use `self.config` instead of `self.num_pulses`, etc.
3. Remove leading underscore (make public)
4. Add comprehensive docstrings
5. Keep original method as delegation wrapper (for backward compatibility)

**Example:**
```python
# In RoomResponseRecorder (backward compatible wrapper)
def _extract_cycles(self, audio: np.ndarray) -> np.ndarray:
    """Extract cycles (delegates to SignalProcessor)"""
    return self.signal_processor.extract_cycles(audio)

# In SignalProcessor (actual implementation)
def extract_cycles(self, audio: np.ndarray) -> np.ndarray:
    """Extract cycles from raw audio using simple reshape."""
    expected_samples = self.config.cycle_samples * self.config.num_pulses
    # ... implementation ...
```

### Phase 3: Extract Standard Mode Methods (1 hour)

**Goal:** Move standard mode alignment methods to SignalProcessor

**Methods to Extract:**
1. `_find_onset_in_room_response()` → `find_onset_in_room_response()`
2. `_extract_impulse_response()` → `extract_impulse_response()`

**Process:** Same as Phase 2

### Phase 4: Extract Calibration Mode Methods (3 hours)

**Goal:** Move calibration mode methods to SignalProcessor

**Methods to Extract:**
1. `align_cycles_by_onset()` → `align_cycles_by_onset()`
2. `apply_alignment_to_channel()` → `apply_alignment_to_channel()`
3. `_normalize_by_calibration()` → `normalize_by_calibration()`

**Special Considerations:**
- `align_cycles_by_onset()` uses `self.multichannel_config` - need to access via `self.config.multichannel_config`
- `apply_alignment_to_channel()` uses `self.num_pulses` and `self.cycle_samples` - access via config
- All three methods are already well-encapsulated and have minimal dependencies

### Phase 5: Update Processing Methods (2 hours)

**Goal:** Update `_process_*` methods to use SignalProcessor delegation

**Files to Update:**
- `_process_single_channel_signal()` - Lines 798-838
- `_process_multichannel_signal()` - Lines 840-904
- `_process_calibration_mode()` - Lines 1428-1593

**Pattern:**
```python
# OLD
cycles = self._extract_cycles(audio)

# NEW
cycles = self.signal_processor.extract_cycles(audio)
```

### Phase 6: Testing & Validation (3 hours)

**Goal:** Ensure all functionality works after extraction

**Tasks:**
1. Run existing test suite (`test_refactoring_phases_1_3.py`)
2. Create new SignalProcessor unit tests
3. Test standard mode recording
4. Test calibration mode recording
5. Test multi-channel recording
6. Verify backward compatibility

**New Tests:**
```python
# test_signal_processor.py

def test_signal_processor_creation():
    """Test SignalProcessor can be created with config"""
    config = SignalProcessingConfig(
        num_pulses=8,
        cycle_samples=4800,
        sample_rate=48000
    )
    processor = SignalProcessor(config)
    assert processor.config.num_pulses == 8

def test_extract_cycles():
    """Test cycle extraction"""
    config = SignalProcessingConfig(num_pulses=4, cycle_samples=1000, sample_rate=48000)
    processor = SignalProcessor(config)

    audio = np.random.randn(4000)
    cycles = processor.extract_cycles(audio)

    assert cycles.shape == (4, 1000)

# ... more tests ...
```

### Phase 7: Documentation Update (1 hour)

**Goal:** Update all documentation to reflect new architecture

**Files to Update:**
1. `CALIBRATION_MODE_PIPELINE_DETAILED.md` - Update code organization section
2. `ARCHITECTURE_REFACTORING_PLAN.md` - Add Phase 7: SignalProcessor extraction
3. `REFACTORING_COMPLETE_SUMMARY.md` - Update with new changes
4. Create `SIGNAL_PROCESSOR_API_REFERENCE.md` - Complete API documentation

### Phase 8: Optional Cleanup (1 hour)

**Goal:** Remove delegation wrappers if desired (breaking change)

**Decision Point:** Keep delegation wrappers for backward compatibility or remove them?

**Option A: Keep Wrappers (Recommended)**
- No breaking changes
- Existing code continues to work
- Gradual migration possible

**Option B: Remove Wrappers**
- Cleaner code
- Forces explicit use of SignalProcessor
- Breaking change for any external code

---

## Benefits of Extraction

### 1. Separation of Concerns ✅

**Before:**
```
RoomResponseRecorder (1,586 lines)
├─ Audio recording (SDL)
├─ Signal processing (434 lines)
├─ File I/O
└─ Configuration
```

**After:**
```
RoomResponseRecorder (~1,150 lines)
├─ Audio recording (SDL)
├─ File I/O
└─ Configuration

SignalProcessor (~450 lines)
├─ Cycle extraction
├─ Alignment
├─ Normalization
└─ Averaging
```

### 2. Improved Testability ✅

**Before:** Must create full RoomResponseRecorder with config file to test signal processing

**After:** Can test SignalProcessor independently with simple config object

```python
# Easy to test
config = SignalProcessingConfig(num_pulses=8, cycle_samples=1000, sample_rate=48000)
processor = SignalProcessor(config)
result = processor.average_cycles(test_cycles, start_cycle=2)
```

### 3. Reusability ✅

SignalProcessor can be used in:
- CLI tools for offline processing
- Web APIs
- Batch processing scripts
- Other audio analysis tools

### 4. Configuration Flexibility ✅

SignalProcessor can be reconfigured on the fly:
```python
# Process with different configurations
config1 = SignalProcessingConfig(num_pulses=8, cycle_samples=4800, sample_rate=48000)
config2 = SignalProcessingConfig(num_pulses=16, cycle_samples=9600, sample_rate=96000)

processor1 = SignalProcessor(config1)
processor2 = SignalProcessor(config2)
```

### 5. Clearer Responsibilities ✅

- **RoomResponseRecorder:** "I record and save audio"
- **SignalProcessor:** "I process audio signals"
- **CalibrationValidatorV2:** "I validate impulse quality"

---

## Risks & Mitigation

### Risk 1: Breaking Changes

**Likelihood:** Medium
**Impact:** High

**Mitigation:**
- Keep delegation wrappers in RoomResponseRecorder
- Use `@deprecated` decorator to warn about future changes
- Provide migration guide

```python
# In RoomResponseRecorder
def _extract_cycles(self, audio: np.ndarray) -> np.ndarray:
    """
    Extract cycles (delegates to SignalProcessor).

    DEPRECATED: This method is now a delegation wrapper.
    Consider using self.signal_processor.extract_cycles() directly.
    """
    return self.signal_processor.extract_cycles(audio)
```

### Risk 2: Configuration Synchronization

**Likelihood:** Low
**Impact:** Medium

**Problem:** If `num_pulses` changes, SignalProcessor config might be stale

**Mitigation:**
- Create new SignalProcessor instance when config changes
- Add method to update SignalProcessor config

```python
def update_signal_processor_config(self):
    """Update SignalProcessor with current configuration."""
    self.signal_processor = self._create_signal_processor()
```

Call this after loading new config or changing parameters.

### Risk 3: Performance Overhead

**Likelihood:** Very Low
**Impact:** Low

**Concern:** Extra method call indirection might slow processing

**Reality:** Negligible impact (delegation is essentially free in Python)

**Verification:** Profile before/after to confirm no regression

### Risk 4: Increased Complexity

**Likelihood:** Low
**Impact:** Low

**Concern:** Two classes instead of one might be confusing

**Mitigation:**
- Clear documentation
- Logical separation makes code easier to understand overall
- Each class has clear, focused responsibility

---

## Migration Strategy

### For Internal Code (RoomResponseRecorder methods)

**Immediate:** Update all `_process_*` methods to use `self.signal_processor.*`

This is internal, no external impact.

### For External Code (if any)

**Gradual Migration:**

1. **Phase 1 (Now):** Keep delegation wrappers, add deprecation warnings
2. **Phase 2 (Future):** Update documentation to encourage SignalProcessor use
3. **Phase 3 (Far Future):** Consider removing wrappers if no usage detected

**Example Migration Path:**

```python
# Old way (still works)
recorder = RoomResponseRecorder()
cycles = recorder._extract_cycles(audio)

# New way (recommended)
recorder = RoomResponseRecorder()
cycles = recorder.signal_processor.extract_cycles(audio)

# Standalone (best for new code)
from signal_processor import SignalProcessor, SignalProcessingConfig

config = SignalProcessingConfig(num_pulses=8, cycle_samples=4800, sample_rate=48000)
processor = SignalProcessor(config)
cycles = processor.extract_cycles(audio)
```

---

## Implementation Timeline

### Total Estimated Time: 14 hours

| Phase | Description | Time | Priority |
|-------|-------------|------|----------|
| Phase 1 | Create SignalProcessor module | 2 hours | HIGH |
| Phase 2 | Extract universal methods | 2 hours | HIGH |
| Phase 3 | Extract standard mode methods | 1 hour | HIGH |
| Phase 4 | Extract calibration mode methods | 3 hours | HIGH |
| Phase 5 | Update processing methods | 2 hours | HIGH |
| Phase 6 | Testing & validation | 3 hours | CRITICAL |
| Phase 7 | Documentation update | 1 hour | MEDIUM |
| Phase 8 | Optional cleanup | 1 hour | LOW |

**Week 1 (Phases 1-3):** Create module and extract universal/standard methods (5 hours)
**Week 2 (Phases 4-6):** Extract calibration methods and validate (8 hours)
**Week 3 (Phase 7-8):** Documentation and optional cleanup (2 hours)

---

## Success Criteria

### Functional Criteria ✅

- [ ] All existing tests pass (100% pass rate maintained)
- [ ] Standard mode recording works identically
- [ ] Calibration mode recording works identically
- [ ] Multi-channel recording works identically
- [ ] File saving produces identical outputs

### Code Quality Criteria ✅

- [ ] SignalProcessor is fully self-contained (no dependencies on RoomResponseRecorder internals)
- [ ] All methods have comprehensive docstrings
- [ ] Type hints are complete
- [ ] Code follows existing style conventions

### Testing Criteria ✅

- [ ] SignalProcessor has dedicated unit tests
- [ ] Integration tests verify RoomResponseRecorder still works
- [ ] Test coverage >80% for SignalProcessor
- [ ] No performance regression (<5% slower)

### Documentation Criteria ✅

- [ ] SignalProcessor API fully documented
- [ ] CALIBRATION_MODE_PIPELINE_DETAILED.md updated
- [ ] Migration guide created
- [ ] Example usage provided

---

## Alternative Approaches Considered

### Alternative 1: Keep Everything in RoomResponseRecorder

**Pros:**
- No refactoring needed
- Simpler class structure

**Cons:**
- Growing class size (1,586 lines already)
- Mixed responsibilities (recording + processing)
- Harder to test processing independently
- Less reusable

**Decision:** Rejected - complexity is growing, separation is needed

### Alternative 2: Extract Each Processing Type to Separate Classes

**Option:**
- `CycleExtractor`
- `AlignmentProcessor`
- `NormalizationProcessor`
- `AveragingProcessor`

**Pros:**
- Maximum modularity
- Very focused classes

**Cons:**
- Too many classes
- Increased complexity in orchestration
- Methods often work together as a pipeline

**Decision:** Rejected - one SignalProcessor is simpler and still provides good separation

### Alternative 3: Create Processing Pipeline Pattern

**Option:** Chain of processors

```python
pipeline = ProcessingPipeline()
pipeline.add(CycleExtractor())
pipeline.add(Aligner())
pipeline.add(Normalizer())
pipeline.add(Averager())
result = pipeline.process(audio)
```

**Pros:**
- Very flexible
- Easy to add/remove stages
- Clean separation

**Cons:**
- Over-engineering for current needs
- More complex to understand
- Each stage needs careful interface design

**Decision:** Rejected - too complex for current requirements. Could revisit if pipeline becomes more dynamic.

---

## Conclusion

Extracting signal processing into `SignalProcessor` class is a **natural evolution** of the codebase that:

1. ✅ **Improves code organization** - Clear separation of recording vs. processing
2. ✅ **Enhances testability** - SignalProcessor can be tested in isolation
3. ✅ **Enables reusability** - Processing logic available for other tools
4. ✅ **Maintains compatibility** - Delegation wrappers ensure no breaking changes
5. ✅ **Reduces complexity** - Each class has focused responsibility

**Recommendation:** Proceed with extraction in 8 phases over 2-3 weeks (14 hours total effort).

**Next Steps:**
1. Review and approve this plan
2. Begin Phase 1: Create SignalProcessor module
3. Progress through phases with testing at each step
4. Update documentation upon completion

---

**Status:** 📋 **PLAN READY FOR REVIEW & APPROVAL**
**Created:** 2025-11-03
**Estimated Effort:** 14 hours (HIGH priority: 11 hours, CRITICAL: 3 hours)
