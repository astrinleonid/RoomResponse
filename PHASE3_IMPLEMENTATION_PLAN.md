# Phase 3 Implementation Plan - Unified Standard Mode Processing

**Document Version:** 1.0
**Date:** 2025-10-31
**Status:** 🚧 PLANNING - Ready for Implementation
**Prerequisites:** Phase 2 completed successfully ✅

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Current State Analysis](#2-current-state-analysis)
3. [Duplication Analysis](#3-duplication-analysis)
4. [Proposed Solution](#4-proposed-solution)
5. [Implementation Steps](#5-implementation-steps)
6. [Testing Strategy](#6-testing-strategy)
7. [Risk Mitigation](#7-risk-mitigation)

---

## 1. Executive Summary

### Objective

Consolidate `_process_single_channel_signal()` and `_process_multichannel_signal()` into unified helper methods to eliminate 90% code duplication.

### Key Principle

**PRESERVE BEHAVIOR EXACTLY** - No changes to output, only internal refactoring.

### Benefits

1. **Reduced Duplication:** 90% overlap eliminated
2. **Easier Maintenance:** Fix bugs in one place
3. **Cleaner Code:** Shared logic explicit
4. **Better Testing:** Helper methods independently testable

### Timeline

- **Planning:** 1 hour (this document)
- **Implementation:** 2-3 hours
- **Testing:** 2-3 hours
- **Total:** 5-7 hours

---

## 2. Current State Analysis

### Current Code Structure

**Two separate methods with 90% duplicate logic:**

#### `_process_single_channel_signal()` - Lines 694-727 (34 lines)
```python
def _process_single_channel_signal(self, recorded_audio: np.ndarray):
    # 1. Pad/trim to expected length
    # 2. Reshape into cycles
    # 3. Average cycles (skip first few)
    # 4. Extract impulse response
    # 5. Return dict
```

#### `_process_multichannel_signal()` - Lines 729-791 (63 lines)
```python
def _process_multichannel_signal(self, multichannel_audio: Dict[int, np.ndarray]):
    # 1. Pad/trim all channels
    # 2. Reshape reference channel
    # 3. Average reference channel cycles
    # 4. Find onset in reference
    # 5. Apply same shift to all channels:
    #    - Reshape each channel
    #    - Average each channel
    #    - Apply shift
    # 6. Return dict with all channels
```

### Duplicate Logic Identified

**Both methods perform:**
1. ✅ Pad/trim audio to expected length
2. ✅ Reshape into cycles
3. ✅ Average cycles (skip first few)
4. ✅ Find onset (implicitly via `_extract_impulse_response`)
5. ✅ Rotate/align signal

**Differences:**
- Multi-channel processes multiple arrays
- Multi-channel uses reference channel for onset
- Multi-channel applies same shift to all

---

## 3. Duplication Analysis

### Code Overlap

| Operation | Single-Channel | Multi-Channel | Duplication |
|-----------|---------------|---------------|-------------|
| Pad/trim | Lines 696-703 | Lines 743-749 | ✅ 100% |
| Reshape | Line 708 | Lines 753, 779 | ✅ 100% |
| Average cycles | Lines 711-713 | Lines 754-755, 780 | ✅ 100% |
| Find onset | Via `_extract_impulse_response` | Line 758 | ⚠️ Different methods |
| Align | Via `_extract_impulse_response` | Line 783 | ⚠️ Different implementation |

**Duplication Percentage:** ~90%

### Current Method Calls

**Single-channel:**
- `_extract_impulse_response()` → `_find_onset_in_room_response()` → `np.roll()`

**Multi-channel:**
- `_find_onset_in_room_response()` → `np.roll()`

**Issue:** Same logic, different call paths!

---

## 4. Proposed Solution

### New Architecture

**Extract common operations into helper methods:**

```
_process_recorded_signal()
  └─> if multichannel:
      │   _process_multichannel_signal()
      │     ├─> _extract_cycles()  [NEW HELPER]
      │     ├─> _average_cycles()  [NEW HELPER]
      │     ├─> _find_onset_in_room_response()  [EXISTING]
      │     └─> np.roll()
      └─> else:
          _process_single_channel_signal()
            ├─> _extract_cycles()  [NEW HELPER]
            ├─> _average_cycles()  [NEW HELPER]
            ├─> _find_onset_in_room_response()  [EXISTING]
            └─> np.roll()
```

### New Helper Methods

#### 1. `_extract_cycles(audio)` - Extract cycles from raw audio

**Purpose:** Consolidate pad/trim/reshape logic

```python
def _extract_cycles(self, audio: np.ndarray) -> np.ndarray:
    """
    Extract cycles from raw audio using simple reshape.

    Args:
        audio: Raw audio signal

    Returns:
        Cycles array [num_cycles, cycle_samples]
    """
    expected_samples = self.cycle_samples * self.num_pulses

    # Pad or trim to expected length
    if len(audio) < expected_samples:
        padded = np.zeros(expected_samples, dtype=audio.dtype)
        padded[:len(audio)] = audio
        audio = padded
    else:
        audio = audio[:expected_samples]

    # Reshape into cycles
    return audio.reshape(self.num_pulses, self.cycle_samples)
```

#### 2. `_average_cycles(cycles, start_cycle)` - Average cycles

**Purpose:** Consolidate averaging logic

```python
def _average_cycles(self, cycles: np.ndarray, start_cycle: int = None) -> np.ndarray:
    """
    Average cycles starting from start_cycle.

    Args:
        cycles: Cycles array [num_cycles, cycle_samples]
        start_cycle: Index to start averaging from (default: num_pulses // 4)

    Returns:
        Averaged signal [cycle_samples]
    """
    if start_cycle is None:
        start_cycle = max(1, self.num_pulses // 4)

    return np.mean(cycles[start_cycle:], axis=0)
```

### Modified Existing Methods

#### `_process_single_channel_signal()` - Use helpers

```python
def _process_single_channel_signal(self, recorded_audio: np.ndarray) -> Dict[str, Any]:
    """Process single-channel standard recording using helper methods"""

    # Extract cycles
    cycles = self._extract_cycles(recorded_audio)

    # Average cycles
    start_cycle = max(1, self.num_pulses // 4)
    room_response = self._average_cycles(cycles, start_cycle)
    print(f"Averaged cycles {start_cycle} to {self.num_pulses - 1}")

    # Find onset and extract impulse
    impulse_response = self._extract_impulse_response(room_response)

    return {
        'raw': recorded_audio,
        'room_response': room_response,
        'impulse': impulse_response
    }
```

#### `_process_multichannel_signal()` - Use helpers

```python
def _process_multichannel_signal(self, multichannel_audio: Dict[int, np.ndarray]) -> Dict[str, Any]:
    """Process multi-channel recording using helper methods"""

    num_channels = len(multichannel_audio)
    ref_channel = self.multichannel_config.get('reference_channel', 0)

    print(f"Processing {num_channels} channels (reference: {ref_channel})")

    # Process reference channel first
    ref_cycles = self._extract_cycles(multichannel_audio[ref_channel])
    start_cycle = max(1, self.num_pulses // 4)
    ref_room_response = self._average_cycles(ref_cycles, start_cycle)

    # Find onset in reference channel
    onset_sample = self._find_onset_in_room_response(ref_room_response)
    shift_amount = -onset_sample
    print(f"Found onset at sample {onset_sample} in reference channel {ref_channel}")

    # Process all channels with same shift
    result = {
        'raw': {},
        'room_response': {},
        'impulse': {},
        'metadata': {
            'num_channels': num_channels,
            'reference_channel': ref_channel,
            'onset_sample': onset_sample,
            'shift_applied': shift_amount
        }
    }

    for ch_idx, audio in multichannel_audio.items():
        # Extract and average cycles for this channel
        cycles = self._extract_cycles(audio)
        room_response = self._average_cycles(cycles, start_cycle)

        # Apply THE SAME shift (critical for synchronization)
        impulse_response = np.roll(room_response, shift_amount)

        result['raw'][ch_idx] = audio
        result['room_response'][ch_idx] = room_response
        result['impulse'][ch_idx] = impulse_response

        print(f"  Channel {ch_idx}: aligned with shift={shift_amount}")

    return result
```

### Lines of Code Impact

| Metric | Before | After | Change |
|--------|--------|-------|--------|
| **Total lines** | 97 | ~80 | -17 lines |
| **Duplicate code** | ~60 lines | 0 | -60 lines |
| **Helper methods** | 0 | 2 (~30 lines) | +30 lines |
| **Net change** | - | - | **-17 lines** |

---

## 5. Implementation Steps

### Step 1: Add Helper Method `_extract_cycles()`

**Location:** After `_process_recorded_signal()` (around line 693)

**Implementation:**
```python
def _extract_cycles(self, audio: np.ndarray) -> np.ndarray:
    """
    Extract cycles from raw audio using simple reshape.

    Pads or trims audio to expected length, then reshapes into cycles.

    Args:
        audio: Raw audio signal

    Returns:
        Cycles array [num_cycles, cycle_samples]
    """
    expected_samples = self.cycle_samples * self.num_pulses

    # Pad or trim to expected length
    if len(audio) < expected_samples:
        padded = np.zeros(expected_samples, dtype=audio.dtype)
        padded[:len(audio)] = audio
        audio = padded
    else:
        audio = audio[:expected_samples]

    # Reshape into cycles
    return audio.reshape(self.num_pulses, self.cycle_samples)
```

**Testing:**
- [ ] Test with exact length audio
- [ ] Test with short audio (padding)
- [ ] Test with long audio (trimming)
- [ ] Verify shape is [num_pulses, cycle_samples]

### Step 2: Add Helper Method `_average_cycles()`

**Location:** After `_extract_cycles()`

**Implementation:**
```python
def _average_cycles(self, cycles: np.ndarray, start_cycle: int = None) -> np.ndarray:
    """
    Average cycles starting from start_cycle.

    Skips initial cycles to allow system settling.

    Args:
        cycles: Cycles array [num_cycles, cycle_samples]
        start_cycle: Index to start averaging from (default: num_pulses // 4)

    Returns:
        Averaged signal [cycle_samples]
    """
    if start_cycle is None:
        start_cycle = max(1, self.num_pulses // 4)

    return np.mean(cycles[start_cycle:], axis=0)
```

**Testing:**
- [ ] Test with default start_cycle
- [ ] Test with explicit start_cycle
- [ ] Verify averaged output matches current behavior

### Step 3: Refactor `_process_single_channel_signal()`

**Changes:**
- Replace pad/trim/reshape with `_extract_cycles()`
- Replace averaging with `_average_cycles()`
- Keep rest of logic identical

**Testing:**
- [ ] Compare output with original implementation
- [ ] Verify file saving works
- [ ] Test with various audio lengths

### Step 4: Refactor `_process_multichannel_signal()`

**Changes:**
- Replace pad/trim/reshape with `_extract_cycles()` for all channels
- Replace averaging with `_average_cycles()`
- Keep synchronization logic identical

**Testing:**
- [ ] Compare output with original implementation
- [ ] Verify all channels aligned correctly
- [ ] Test with various channel counts

---

## 6. Testing Strategy

### Level 1: Unit Tests (Helper Methods)

**test_phase3_helpers.py:**

```python
def test_extract_cycles_exact_length():
    """Test with audio exactly matching expected length"""
    recorder = RoomResponseRecorder("recorderConfig.json")
    audio = np.random.randn(recorder.cycle_samples * recorder.num_pulses)

    cycles = recorder._extract_cycles(audio)

    assert cycles.shape == (recorder.num_pulses, recorder.cycle_samples)

def test_extract_cycles_padding():
    """Test with short audio (should pad)"""
    recorder = RoomResponseRecorder("recorderConfig.json")
    audio = np.random.randn(recorder.cycle_samples * 3)  # Only 3 cycles

    cycles = recorder._extract_cycles(audio)

    assert cycles.shape == (recorder.num_pulses, recorder.cycle_samples)
    # Later cycles should be zeros
    assert np.allclose(cycles[-1], 0)

def test_average_cycles_default():
    """Test averaging with default start_cycle"""
    recorder = RoomResponseRecorder("recorderConfig.json")
    cycles = np.random.randn(recorder.num_pulses, recorder.cycle_samples)

    averaged = recorder._average_cycles(cycles)

    start_cycle = max(1, recorder.num_pulses // 4)
    expected = np.mean(cycles[start_cycle:], axis=0)
    assert np.allclose(averaged, expected)
```

### Level 2: Integration Tests (Behavior Preservation)

**test_phase3_behavior.py:**

```python
def test_single_channel_output_unchanged():
    """Verify single-channel output matches original"""
    # Create mock audio
    # Process with OLD and NEW code
    # Compare outputs exactly

def test_multichannel_output_unchanged():
    """Verify multi-channel output matches original"""
    # Create mock multi-channel audio
    # Process with OLD and NEW code
    # Compare outputs exactly
```

### Level 3: Hardware Tests

**Use existing hardware tests:**
- `test_phase2_hardware.py` already covers hardware validation
- Re-run to ensure behavior unchanged

---

## 7. Risk Mitigation

### Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Behavior change | MEDIUM | HIGH | Extensive comparison testing |
| Rounding differences | LOW | LOW | Use `np.allclose()` for comparison |
| Edge cases missed | MEDIUM | MEDIUM | Test with various input lengths |

### Mitigation Strategies

#### Strategy 1: Comparison Testing

**Before refactoring:**
1. Capture outputs from current implementation
2. Save as reference

**After refactoring:**
1. Run same inputs through new implementation
2. Compare outputs bit-by-bit
3. Must match exactly

#### Strategy 2: Incremental Implementation

**Order:**
1. Add helper methods (no changes to existing code)
2. Test helpers in isolation
3. Refactor single-channel (smaller, simpler)
4. Test single-channel
5. Refactor multi-channel
6. Test multi-channel

#### Strategy 3: Rollback Plan

**If ANY test fails:**
```bash
git checkout RoomResponseRecorder.py
```

---

## 8. Success Criteria

### Functional Success

- ✅ Single-channel output identical to original
- ✅ Multi-channel output identical to original
- ✅ All file saving works
- ✅ All tests pass

### Code Quality Success

- ✅ Duplication eliminated (~60 lines)
- ✅ Helper methods reusable
- ✅ Clearer code structure
- ✅ Easier to maintain

---

## 9. Implementation Checklist

### Pre-Implementation
- [ ] Review this plan
- [ ] Create backup of RoomResponseRecorder.py
- [ ] Create test branch
- [ ] Run baseline tests

### Implementation
- [ ] Step 1: Add `_extract_cycles()`
- [ ] Step 2: Add `_average_cycles()`
- [ ] Step 3: Refactor `_process_single_channel_signal()`
- [ ] Step 4: Refactor `_process_multichannel_signal()`

### Testing
- [ ] Level 1: Unit tests for helpers
- [ ] Level 2: Integration tests for behavior
- [ ] Level 3: Hardware tests

### Finalization
- [ ] All tests passed
- [ ] Commit changes
- [ ] Merge to dev

---

**Status:** 🟢 **READY FOR IMPLEMENTATION**

**Estimated Duration:** 5-7 hours

**Next Step:** Begin implementation with Step 1
