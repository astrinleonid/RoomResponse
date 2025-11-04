"""
Unit tests for SignalProcessor

Tests all 9 signal processing methods in isolation to ensure they work correctly.
"""

import numpy as np
import pytest
from signal_processor import SignalProcessor, SignalProcessingConfig


class TestSignalProcessingConfig:
    """Test SignalProcessingConfig creation"""

    def test_config_creation(self):
        """Test basic config creation"""
        config = SignalProcessingConfig(
            num_pulses=8,
            cycle_samples=4800,
            sample_rate=48000
        )
        assert config.num_pulses == 8
        assert config.cycle_samples == 4800
        assert config.sample_rate == 48000
        assert config.multichannel_config == {}

    def test_config_with_multichannel(self):
        """Test config with multichannel settings"""
        mc_config = {'enabled': True, 'num_channels': 2}
        config = SignalProcessingConfig(
            num_pulses=8,
            cycle_samples=4800,
            sample_rate=48000,
            multichannel_config=mc_config
        )
        assert config.multichannel_config['enabled'] == True
        assert config.multichannel_config['num_channels'] == 2


class TestSignalProcessor:
    """Test SignalProcessor methods"""

    @pytest.fixture
    def config(self):
        """Create standard test configuration"""
        return SignalProcessingConfig(
            num_pulses=4,
            cycle_samples=1000,
            sample_rate=48000
        )

    @pytest.fixture
    def processor(self, config):
        """Create SignalProcessor instance"""
        return SignalProcessor(config)

    # ========================================================================
    # Test Universal Methods
    # ========================================================================

    def test_extract_cycles(self, processor):
        """Test cycle extraction from audio"""
        # Create audio with 4 cycles of 1000 samples each
        audio = np.random.randn(4000).astype(np.float32)

        cycles = processor.extract_cycles(audio)

        assert cycles.shape == (4, 1000)
        assert cycles.dtype == np.float32

    def test_extract_cycles_padding(self, processor):
        """Test cycle extraction pads short audio"""
        # Audio shorter than expected
        audio = np.random.randn(3000).astype(np.float32)

        cycles = processor.extract_cycles(audio)

        assert cycles.shape == (4, 1000)
        # Last 1000 samples should be zeros (padding)
        assert np.allclose(cycles[-1, :], 0.0)

    def test_extract_cycles_trimming(self, processor):
        """Test cycle extraction trims long audio"""
        # Audio longer than expected
        audio = np.random.randn(5000).astype(np.float32)

        cycles = processor.extract_cycles(audio)

        assert cycles.shape == (4, 1000)

    def test_average_cycles(self, processor):
        """Test cycle averaging"""
        # Create 4 cycles with known values
        cycles = np.array([
            np.ones(1000) * 1.0,
            np.ones(1000) * 2.0,
            np.ones(1000) * 3.0,
            np.ones(1000) * 4.0
        ])

        # Average from cycle 1 onward (skip first cycle)
        avg = processor.average_cycles(cycles, start_cycle=1)

        assert avg.shape == (1000,)
        # Should average cycles 1, 2, 3 → (2 + 3 + 4) / 3 = 3.0
        assert np.allclose(avg, 3.0)

    def test_average_cycles_default_start(self, processor):
        """Test cycle averaging with default start_cycle"""
        cycles = np.array([
            np.ones(1000) * 1.0,
            np.ones(1000) * 2.0,
            np.ones(1000) * 3.0,
            np.ones(1000) * 4.0
        ])

        # Default should skip first cycle (num_pulses // 4 = 1)
        avg = processor.average_cycles(cycles)

        assert np.allclose(avg, 3.0)

    def test_compute_spectral_analysis(self, processor):
        """Test FFT spectral analysis"""
        # Create test signal: 1000 Hz sine wave
        t = np.arange(1000) / 48000
        signal = np.sin(2 * np.pi * 1000 * t)

        responses = {0: signal}

        result = processor.compute_spectral_analysis(responses)

        assert 'frequencies' in result
        assert 'magnitudes' in result
        assert 'magnitude_db' in result
        assert 'window' in result
        assert 'n_fft' in result

        assert 0 in result['magnitudes']
        assert len(result['frequencies']) == len(result['magnitudes'][0])

    def test_find_sound_onset(self, processor):
        """Test onset detection"""
        # Create signal with onset at sample 100
        audio = np.zeros(1000)
        audio[100:] = np.random.randn(900) * 0.5 + 0.5  # Signal starts at 100

        onset = processor.find_sound_onset(audio)

        # Should detect onset around sample 100 (within reasonable tolerance)
        assert 80 <= onset <= 120

    def test_find_sound_onset_immediate(self, processor):
        """Test onset detection when signal starts immediately"""
        audio = np.random.randn(1000) * 0.5 + 0.5

        onset = processor.find_sound_onset(audio)

        # Should detect onset near beginning
        assert onset < 50

    # ========================================================================
    # Test Standard Mode Methods
    # ========================================================================

    def test_find_onset_in_room_response(self, processor):
        """Test onset finding in room response"""
        # Create room response with peak at sample 200
        room_response = np.zeros(1000)
        room_response[200] = 1.0
        room_response[201:250] = np.linspace(1.0, 0.0, 49)

        onset = processor.find_onset_in_room_response(room_response)

        # Should find onset before the peak
        assert onset < 200

    def test_extract_impulse_response(self, processor):
        """Test impulse response extraction"""
        # Create room response with peak at sample 200
        room_response = np.zeros(1000)
        room_response[200] = 1.0
        room_response[201:250] = np.linspace(1.0, 0.0, 49)

        impulse = processor.extract_impulse_response(room_response)

        assert impulse.shape == room_response.shape
        # After rotation, peak should be near beginning
        peak_idx = np.argmax(np.abs(impulse))
        assert peak_idx < 50

    # ========================================================================
    # Test Calibration Mode Methods
    # ========================================================================

    def test_align_cycles_by_onset_empty(self, processor):
        """Test alignment with empty cycles"""
        cycles = np.array([])
        validation_results = []

        result = processor.align_cycles_by_onset(cycles, validation_results)

        assert len(result['aligned_cycles']) == 0
        assert len(result['valid_cycle_indices']) == 0

    def test_align_cycles_by_onset_basic(self, processor):
        """Test basic cycle alignment"""
        # Create 3 cycles with negative peaks at different positions
        cycle1 = np.zeros(1000)
        cycle1[100] = -1.0  # Peak at 100

        cycle2 = np.zeros(1000)
        cycle2[150] = -1.0  # Peak at 150

        cycle3 = np.zeros(1000)
        cycle3[120] = -1.0  # Peak at 120

        cycles = np.array([cycle1, cycle2, cycle3])

        # All cycles marked as valid
        validation_results = [
            {'calibration_valid': True},
            {'calibration_valid': True},
            {'calibration_valid': True}
        ]

        result = processor.align_cycles_by_onset(cycles, validation_results)

        assert len(result['aligned_cycles']) > 0
        assert len(result['valid_cycle_indices']) > 0
        assert 'aligned_onset_position' in result
        assert 'correlations' in result

    def test_align_cycles_by_onset_filtering(self, processor):
        """Test that invalid cycles are filtered out"""
        cycles = np.random.randn(4, 1000)

        # Only cycles 1 and 3 are valid
        validation_results = [
            {'calibration_valid': False},
            {'calibration_valid': True},
            {'calibration_valid': False},
            {'calibration_valid': True}
        ]

        result = processor.align_cycles_by_onset(cycles, validation_results)

        # Should only process valid cycles (1 and 3)
        assert len(result['valid_cycle_indices']) <= 2

    def test_apply_alignment_to_channel_empty(self, processor):
        """Test alignment application with empty metadata"""
        channel_raw = np.random.randn(4000)

        alignment_metadata = {
            'valid_cycle_indices': [],
            'onset_positions': [],
            'aligned_onset_position': 0
        }

        result = processor.apply_alignment_to_channel(channel_raw, alignment_metadata)

        assert len(result) == 0

    def test_apply_alignment_to_channel_basic(self, processor):
        """Test applying alignment to channel"""
        channel_raw = np.random.randn(4000)

        # Simulate alignment from 2 valid cycles
        alignment_metadata = {
            'valid_cycle_indices': [0, 2],
            'onset_positions': [100, 120],
            'aligned_onset_position': 0
        }

        result = processor.apply_alignment_to_channel(channel_raw, alignment_metadata)

        assert result.shape[0] == 2  # 2 aligned cycles
        assert result.shape[1] == 1000  # cycle_samples

    def test_normalize_by_calibration_empty(self, processor):
        """Test normalization with empty data"""
        aligned_cycles = {}
        validation_results = []

        result, factors = processor.normalize_by_calibration(
            aligned_cycles,
            validation_results,
            calibration_channel=0,
            valid_cycle_indices=[]
        )

        assert len(result) == 0
        assert len(factors) == 0

    def test_normalize_by_calibration_basic(self, processor):
        """Test basic normalization"""
        # Create aligned cycles for 2 channels
        aligned_cycles = {
            0: np.array([np.ones(1000) * 2.0]),  # Calibration channel
            1: np.array([np.ones(1000) * 4.0])   # Response channel
        }

        # Validation results with negative peaks
        validation_results = [
            {'calibration_metrics': {'negative_peak': 0.5}}
        ]

        valid_cycle_indices = [0]

        normalized, factors = processor.normalize_by_calibration(
            aligned_cycles,
            validation_results,
            calibration_channel=0,
            valid_cycle_indices=valid_cycle_indices
        )

        # Calibration channel should be unchanged
        assert np.allclose(normalized[0], 2.0)

        # Response channel should be normalized by 0.5
        assert np.allclose(normalized[1], 8.0)  # 4.0 / 0.5 = 8.0

        assert factors == [0.5]


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
