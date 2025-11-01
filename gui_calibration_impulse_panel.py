#!/usr/bin/env python3
"""
Calibration Impulse Panel - Multi-Channel Calibration Configuration and Testing

- Uses a shared/global RoomResponseRecorder passed in by the parent.
- Provides calibration quality parameter configuration with automatic threshold learning.
- Validates calibration impulses in multi-channel recording mode.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Dict, Any

import numpy as np
import streamlit as st

# Optional visualizer
try:
    from gui_audio_visualizer import AudioVisualizer
    VISUALIZER_AVAILABLE = True
except ImportError:
    VISUALIZER_AVAILABLE = False
    AudioVisualizer = None  # type: ignore

# Recorder type (provided by parent; we never instantiate here)
try:
    from RoomResponseRecorder import RoomResponseRecorder  # type: ignore
    RECORDER_AVAILABLE = True
except Exception:
    RoomResponseRecorder = None  # type: ignore
    RECORDER_AVAILABLE = False


class CalibrationImpulsePanel:
    """Panel for configuring and testing calibration impulse quality (shared recorder)."""

    def __init__(self, recorder: Optional["RoomResponseRecorder"] = None, audio_settings_panel=None):
        """
        Args:
            recorder: Shared/global RoomResponseRecorder instance (required)
            audio_settings_panel: Optional parent panel reference
        """
        self.recorder = recorder
        self.audio_settings_panel = audio_settings_panel
        self.component_id = "calibration_impulse"

    # ----------------------
    # Public render entrypoint
    # ----------------------
    def render(self) -> None:
        st.subheader("Calibration Impulse")
        st.markdown("Configure and test calibration channel for multi-channel impulse response recording.")

        if not self.recorder:
            st.error("Recorder not initialized")
            return

        # Check if multi-channel is enabled
        mc_config = self.recorder.multichannel_config

        if not mc_config.get('enabled', False):
            st.warning("⚠️ Multi-channel recording is not enabled. Calibration is only available in multi-channel mode.")
            st.info("📍 **To enable multi-channel recording:**")
            st.markdown("""
            1. Go to **Device Selection & Testing** tab
            2. Scroll down to **Multi-Channel Configuration** section
            3. Check **"Enable multi-channel recording"**
            4. Configure your channels
            5. Click **"Apply Multi-Channel Configuration"**
            6. Click **"💾 Save to Config File"** to persist settings
            """)
            return

        num_channels = mc_config.get('num_channels', 1)
        if num_channels < 2:
            st.warning("At least 2 channels required for calibration. Current configuration has only 1 channel.")
            return

        # Get calibration channel (configured in Device Selection tab)
        selected_cal_ch = mc_config.get('calibration_channel')
        channel_names = mc_config.get('channel_names', [f"Channel {i}" for i in range(num_channels)])

        # Show current calibration channel configuration
        st.markdown("### Current Configuration")
        if selected_cal_ch is not None:
            st.success(f"✓ Calibration channel: Ch {selected_cal_ch} ({channel_names[selected_cal_ch]})")
        else:
            st.warning("⚠️ No calibration channel selected. Configure it in the Device Selection tab.")
            st.info("Go to Device Selection tab → Multi-Channel Configuration → Calibration channel")
            return

        st.markdown("---")

        # Section 1: Calibration Quality Parameters (Collapsible)
        with st.expander("### 1. Calibration Quality Parameters", expanded=False):
            self._render_quality_parameters()

        st.markdown("---")

        # Section 2: Test Calibration Impulse (Collapsible)
        with st.expander("### 2. Test Calibration Impulse", expanded=True):
            self._render_calibration_test()

    # ----------------------
    # Configuration file I/O
    # ----------------------
    def _load_config_from_file(self) -> Dict[str, Any]:
        """Load configuration using centralized config manager.

        Returns:
            Dictionary with configuration values, or empty dict if file doesn't exist
        """
        from config_manager import config_manager
        return config_manager.load_config()

    def _save_config_to_file(self) -> bool:
        """Save current calibration settings using centralized config manager.

        Returns:
            True if successful, False otherwise
        """
        try:
            from config_manager import config_manager

            # Load existing config or create new one
            config = config_manager.load_config()

            # Update calibration quality configuration
            if hasattr(self.recorder, 'calibration_quality_config'):
                config['calibration_quality_config'] = dict(self.recorder.calibration_quality_config)

            # Save using config manager with error reporting
            success, error_msg = config_manager.save_config_with_error(config, updated_by="Calibration Impulse Panel")
            if not success:
                st.error(f"Config save failed: {error_msg}")
                st.code(error_msg)

            return success
        except Exception as e:
            st.error(f"Failed to save configuration: {e}")
            import traceback
            st.code(traceback.format_exc())
            return False

    # ----------------------
    # Quality Parameters UI
    # ----------------------
    def _render_quality_parameters(self) -> None:
        """Render calibration quality parameter configuration UI."""
        st.markdown("Configure thresholds for validating calibration impulse quality.")

        # Get current quality config from recorder (V3 comprehensive format)
        if hasattr(self.recorder, 'calibration_quality_config'):
            qual_config = self.recorder.calibration_quality_config
        else:
            # Default values (V3 comprehensive format - 11 parameters)
            qual_config = {
                'min_negative_peak': 0.1,
                'max_negative_peak': 0.95,
                'max_precursor_ratio': 0.2,
                'min_negative_peak_width_ms': 0.3,
                'max_negative_peak_width_ms': 3.0,
                'max_first_positive_ratio': 0.3,
                'min_first_positive_time_ms': 0.1,
                'max_first_positive_time_ms': 5.0,
                'max_highest_positive_ratio': 0.5,
                'max_secondary_negative_ratio': 0.3,
                'secondary_negative_window_ms': 10.0,
            }

        # Tool 1: Manual threshold editing in comprehensive form
        st.markdown("#### 🔧 Tool 1: Manual Threshold Configuration")
        if 'cal_test_learned_thresholds' in st.session_state:
            st.info("💡 Values below were auto-calculated. You can edit them manually - your changes will be saved when you click 'Save Configuration'.")
        else:
            st.info("Edit quality thresholds directly. Use Tool 2 below to auto-calculate from good cycles.")

        st.markdown("**Comprehensive Quality Criteria (7 criteria, 11 parameters)**")

        # === 1. NEGATIVE PEAK RANGE ===
        st.markdown("##### 1️⃣ Negative Peak Range")
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            st.markdown("Negative Peak (absolute amplitude)")
        with col2:
            min_neg_peak = st.number_input(
                "Min",
                min_value=0.0,
                max_value=1.0,
                value=float(qual_config.get('min_negative_peak', 0.1)),
                step=0.01,
                key="min_neg_peak",
                help="Minimum acceptable negative peak amplitude"
            )
        with col3:
            max_neg_peak = st.number_input(
                "Max",
                min_value=0.0,
                max_value=1.0,
                value=float(qual_config.get('max_negative_peak', 0.95)),
                step=0.01,
                key="max_neg_peak",
                help="Maximum negative peak to avoid clipping"
            )

        # === 2. PRECURSOR ===
        st.markdown("##### 2️⃣ Precursor (peaks before impact)")
        col1, col2 = st.columns([2, 2])
        with col1:
            st.markdown("Max Precursor Ratio")
        with col2:
            max_precursor_ratio = st.number_input(
                "Max Precursor/Negative",
                min_value=0.0,
                max_value=1.0,
                value=float(qual_config.get('max_precursor_ratio', 0.2)),
                step=0.01,
                key="max_precursor_ratio",
                help="Maximum peak before negative peak as fraction of negative peak"
            )

        # === 3. NEGATIVE PEAK WIDTH ===
        st.markdown("##### 3️⃣ Negative Peak Width")
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            st.markdown("Peak Width at 50% amplitude (ms)")
        with col2:
            min_neg_width = st.number_input(
                "Min Width",
                min_value=0.1,
                max_value=10.0,
                value=float(qual_config.get('min_negative_peak_width_ms', 0.3)),
                step=0.1,
                key="min_neg_width",
                help="Minimum acceptable peak width"
            )
        with col3:
            max_neg_width = st.number_input(
                "Max Width",
                min_value=0.1,
                max_value=10.0,
                value=float(qual_config.get('max_negative_peak_width_ms', 3.0)),
                step=0.1,
                key="max_neg_width",
                help="Maximum acceptable peak width"
            )

        # === 4. FIRST POSITIVE PEAK ===
        st.markdown("##### 4️⃣ First Positive Peak (after negative)")
        col1, col2 = st.columns([2, 2])
        with col1:
            st.markdown("Max First Positive Ratio")
        with col2:
            max_first_pos_ratio = st.number_input(
                "Max First Pos/Negative",
                min_value=0.0,
                max_value=1.0,
                value=float(qual_config.get('max_first_positive_ratio', 0.3)),
                step=0.01,
                key="max_first_pos_ratio",
                help="Maximum first positive peak as fraction of negative peak"
            )

        # === 5. FIRST POSITIVE TIMING ===
        st.markdown("##### 5️⃣ First Positive Peak Timing")
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            st.markdown("Time from negative to first positive (ms)")
        with col2:
            min_first_pos_time = st.number_input(
                "Min Time",
                min_value=0.0,
                max_value=20.0,
                value=float(qual_config.get('min_first_positive_time_ms', 0.1)),
                step=0.1,
                key="min_first_pos_time",
                help="Minimum time from negative to first positive peak"
            )
        with col3:
            max_first_pos_time = st.number_input(
                "Max Time",
                min_value=0.0,
                max_value=20.0,
                value=float(qual_config.get('max_first_positive_time_ms', 5.0)),
                step=0.1,
                key="max_first_pos_time",
                help="Maximum time from negative to first positive peak"
            )

        # === 6. HIGHEST POSITIVE PEAK ===
        st.markdown("##### 6️⃣ Highest Positive Peak (anywhere after negative)")
        col1, col2 = st.columns([2, 2])
        with col1:
            st.markdown("Max Highest Positive Ratio")
        with col2:
            max_highest_pos_ratio = st.number_input(
                "Max Highest Pos/Negative",
                min_value=0.0,
                max_value=1.0,
                value=float(qual_config.get('max_highest_positive_ratio', 0.5)),
                step=0.01,
                key="max_highest_pos_ratio",
                help="Maximum highest positive peak as fraction of negative peak"
            )

        # === 7. SECONDARY NEGATIVE PEAK ===
        st.markdown("##### 7️⃣ Secondary Negative Peak (hammer bounce)")
        col1, col2 = st.columns([2, 2])
        with col1:
            st.markdown("Max Secondary Negative Ratio")
        with col2:
            max_secondary_neg_ratio = st.number_input(
                "Max Secondary Neg/Main Neg",
                min_value=0.0,
                max_value=1.0,
                value=float(qual_config.get('max_secondary_negative_ratio', 0.3)),
                step=0.01,
                key="max_secondary_neg_ratio",
                help="Maximum secondary negative peak as fraction of main negative peak"
            )

        col1, col2 = st.columns([2, 2])
        with col1:
            st.markdown("Secondary Negative Window (ms)")
        with col2:
            secondary_neg_window = st.number_input(
                "Window",
                min_value=5.0,
                max_value=50.0,
                value=float(qual_config.get('secondary_negative_window_ms', 10.0)),
                step=1.0,
                key="secondary_neg_window",
                help="Time window to check for secondary negative peaks"
            )

        st.markdown("---")

        # Tool 2: Automatic threshold learning from marked cycles
        st.markdown("#### 🎯 Tool 2: Automatic Threshold Learning")
        st.markdown("Select good quality cycles in Section 2's Quality Metrics Summary table to automatically calculate optimal thresholds.")

        # Check if calibration test results are available
        if 'cal_test_results' in st.session_state:
            results = st.session_state['cal_test_results']
            num_cycles = results.get('num_cycles', 0)
            calibration_cycles = results.get('all_calibration_cycles')
            sample_rate = results.get('sample_rate', 48000)

            if calibration_cycles is not None:
                # Get cycles selected in Section 2's table
                marked_good = st.session_state.get('cal_test_selected_cycles', [])

                if len(marked_good) > 0:
                    st.success(f"✓ {len(marked_good)} cycle(s) selected in Section 2: {', '.join(map(str, marked_good))}")

                    # Button to calculate thresholds from selected cycles
                    col_btn, col_clear = st.columns([3, 1])
                    with col_btn:
                        calc_button = st.button("🎯 Calculate Thresholds from Selected Cycles", type="secondary")
                    with col_clear:
                        if 'cal_test_learned_thresholds' in st.session_state:
                            if st.button("🗑️ Clear Learned", key="clear_learned_thresholds"):
                                del st.session_state['cal_test_learned_thresholds']
                                if 'cal_test_num_cycles_used' in st.session_state:
                                    del st.session_state['cal_test_num_cycles_used']
                                st.rerun()

                    if calc_button:
                        try:
                            from calibration_validator_v2 import calculate_thresholds_from_marked_cycles

                            # Calculate thresholds
                            learned_thresholds = calculate_thresholds_from_marked_cycles(
                                calibration_cycles,
                                marked_good,
                                sample_rate,
                                safety_margin=0.05  # 5% margin on both sides
                            )

                            # Store in session state and update the input fields
                            st.session_state['cal_test_learned_thresholds'] = learned_thresholds
                            st.session_state['cal_test_num_cycles_used'] = len(marked_good)

                            # Update the recorder config with learned thresholds
                            self.recorder.calibration_quality_config.update(learned_thresholds.to_dict())

                            # Rerun to refresh the manual threshold inputs with new values
                            st.rerun()

                        except Exception as e:
                            st.error(f"Failed to calculate thresholds: {e}")
                            import traceback
                            st.code(traceback.format_exc())

                    # Display learned thresholds if they exist (persists after rerun)
                    if 'cal_test_learned_thresholds' in st.session_state:
                        learned_thresholds = st.session_state['cal_test_learned_thresholds']
                        num_cycles_used = st.session_state.get('cal_test_num_cycles_used', 0)

                        st.success(f"✓ Thresholds calculated from {num_cycles_used} selected cycles and loaded into configuration!")
                        st.info("💡 Review the updated thresholds in Tool 1 above, then click 'Save Configuration' below.")

                        # Show detailed calculated thresholds
                        with st.expander("📊 Calculated Thresholds - Detailed Values", expanded=True):
                            st.markdown("**Calculated Threshold Values:**")

                            # Table format for clarity
                            col1, col2, col3 = st.columns([2, 1, 2])
                            with col1:
                                st.markdown("**Parameter**")
                            with col2:
                                st.markdown("**Value**")
                            with col3:
                                st.markdown("**Description**")

                            st.markdown("---")

                            # 1. Negative Peak Range
                            col1, col2, col3 = st.columns([2, 1, 2])
                            with col1:
                                st.markdown("**1️⃣ Min Negative Peak**")
                            with col2:
                                st.code(f"{learned_thresholds.min_negative_peak:.4f}")
                            with col3:
                                st.markdown("Minimum acceptable negative peak")

                            col1, col2, col3 = st.columns([2, 1, 2])
                            with col1:
                                st.markdown("**Max Negative Peak**")
                            with col2:
                                st.code(f"{learned_thresholds.max_negative_peak:.4f}")
                            with col3:
                                st.markdown("Maximum negative peak (avoid clipping)")

                            # 2. Precursor
                            col1, col2, col3 = st.columns([2, 1, 2])
                            with col1:
                                st.markdown("**2️⃣ Max Precursor Ratio**")
                            with col2:
                                st.code(f"{learned_thresholds.max_precursor_ratio:.4f}")
                            with col3:
                                st.markdown("Max peak before impact / negative peak")

                            # 3. Negative Peak Width
                            col1, col2, col3 = st.columns([2, 1, 2])
                            with col1:
                                st.markdown("**3️⃣ Min Peak Width**")
                            with col2:
                                st.code(f"{learned_thresholds.min_negative_peak_width_ms:.2f} ms")
                            with col3:
                                st.markdown("Minimum peak width at 50% amplitude")

                            col1, col2, col3 = st.columns([2, 1, 2])
                            with col1:
                                st.markdown("**Max Peak Width**")
                            with col2:
                                st.code(f"{learned_thresholds.max_negative_peak_width_ms:.2f} ms")
                            with col3:
                                st.markdown("Maximum peak width at 50% amplitude")

                            # 4. First Positive Peak
                            col1, col2, col3 = st.columns([2, 1, 2])
                            with col1:
                                st.markdown("**4️⃣ Max First Positive Ratio**")
                            with col2:
                                st.code(f"{learned_thresholds.max_first_positive_ratio:.4f}")
                            with col3:
                                st.markdown("Max first positive / negative peak")

                            # 5. First Positive Timing
                            col1, col2, col3 = st.columns([2, 1, 2])
                            with col1:
                                st.markdown("**5️⃣ Min First Positive Time**")
                            with col2:
                                st.code(f"{learned_thresholds.min_first_positive_time_ms:.2f} ms")
                            with col3:
                                st.markdown("Minimum time from negative to first positive")

                            col1, col2, col3 = st.columns([2, 1, 2])
                            with col1:
                                st.markdown("**Max First Positive Time**")
                            with col2:
                                st.code(f"{learned_thresholds.max_first_positive_time_ms:.2f} ms")
                            with col3:
                                st.markdown("Maximum time from negative to first positive")

                            # 6. Highest Positive Peak
                            col1, col2, col3 = st.columns([2, 1, 2])
                            with col1:
                                st.markdown("**6️⃣ Max Highest Positive Ratio**")
                            with col2:
                                st.code(f"{learned_thresholds.max_highest_positive_ratio:.4f}")
                            with col3:
                                st.markdown("Max highest positive / negative peak")

                            # 7. Secondary Negative Peak
                            col1, col2, col3 = st.columns([2, 1, 2])
                            with col1:
                                st.markdown("**7️⃣ Max Secondary Negative Ratio**")
                            with col2:
                                st.code(f"{learned_thresholds.max_secondary_negative_ratio:.4f}")
                            with col3:
                                st.markdown("Max secondary negative / main negative")

                            col1, col2, col3 = st.columns([2, 1, 2])
                            with col1:
                                st.markdown("**Secondary Negative Window**")
                            with col2:
                                st.code(f"{learned_thresholds.secondary_negative_window_ms:.1f} ms")
                            with col3:
                                st.markdown("Time window to check for secondary negative")

                            st.markdown("---")
                            st.caption(f"📌 Calculated from {num_cycles_used} selected good cycles with 5% safety margin")

                            # Quick summary metrics
                            st.markdown("**Quick Summary:**")
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("Neg Peak Range",
                                         f"{learned_thresholds.min_negative_peak:.3f}-{learned_thresholds.max_negative_peak:.3f}")
                            with col2:
                                st.metric("Precursor",
                                         f"≤{learned_thresholds.max_precursor_ratio:.3f}")
                            with col3:
                                st.metric("Width (ms)",
                                         f"{learned_thresholds.min_negative_peak_width_ms:.2f}-{learned_thresholds.max_negative_peak_width_ms:.2f}")
                            with col4:
                                st.metric("Sec Negative",
                                         f"≤{learned_thresholds.max_secondary_negative_ratio:.3f}")

                else:
                    st.info("👇 Go to Section 2 below and select cycles in the Quality Metrics Summary table")
            else:
                st.info("⚠️ No calibration cycles available. Run a calibration test in Section 2 below.")
        else:
            st.info("⚠️ No calibration test results available. Run a calibration test in Section 2 below first.")

        st.markdown("---")

        # Unified save button for both manual and automatic configuration
        if st.button("💾 Save Configuration", type="primary", key="save_quality_config"):
            try:
                # Build V3 comprehensive config using current manual input values
                # Always use the manual inputs since they represent the user's final decision
                # (even if they were initially populated by automatic learning)
                new_config = {
                    'min_negative_peak': min_neg_peak,
                    'max_negative_peak': max_neg_peak,
                    'max_precursor_ratio': max_precursor_ratio,
                    'min_negative_peak_width_ms': min_neg_width,
                    'max_negative_peak_width_ms': max_neg_width,
                    'max_first_positive_ratio': max_first_pos_ratio,
                    'min_first_positive_time_ms': min_first_pos_time,
                    'max_first_positive_time_ms': max_first_pos_time,
                    'max_highest_positive_ratio': max_highest_pos_ratio,
                    'max_secondary_negative_ratio': max_secondary_neg_ratio,
                    'secondary_negative_window_ms': secondary_neg_window,
                }

                # Save to recorder
                if hasattr(self.recorder, 'calibration_quality_config'):
                    self.recorder.calibration_quality_config = new_config
                else:
                    self.recorder.calibration_quality_config = new_config

                # Save to config file
                if self._save_config_to_file():
                    st.success("✓ Quality configuration saved successfully!")
                    st.info("Settings will be loaded automatically on next session. Clear results and re-run calibration test to apply new thresholds.")
                else:
                    st.warning("⚠️ Configuration saved to recorder but failed to save to config file")

            except Exception as e:
                st.error(f"Failed to save configuration: {e}")

    # ----------------------
    # Calibration Test UI
    # ----------------------
    def _render_calibration_test(self) -> None:
        """Render calibration test controls and results."""
        st.markdown("Emit a train of impulses and check calibration quality for each cycle.")

        col1, col2 = st.columns([1, 1])

        with col1:
            if st.button("Run Calibration Test", type="primary"):
                st.session_state['cal_test_running'] = True

        with col2:
            if st.button("Clear Results"):
                if 'cal_test_results' in st.session_state:
                    del st.session_state['cal_test_results']
                if 'cal_test_selected_cycles' in st.session_state:
                    del st.session_state['cal_test_selected_cycles']
                if 'cal_test_learned_thresholds' in st.session_state:
                    del st.session_state['cal_test_learned_thresholds']
                st.session_state['cal_test_running'] = False

        # Run calibration test
        if st.session_state.get('cal_test_running', False):
            with st.spinner("Recording calibration impulses..."):
                try:
                    results = self._perform_calibration_test()
                    st.session_state['cal_test_results'] = results
                    st.session_state['cal_test_running'] = False
                    st.success("✓ Calibration test completed!")
                except Exception as e:
                    st.error(f"Calibration test failed: {e}")
                    st.session_state['cal_test_running'] = False

        # Display results
        if 'cal_test_results' in st.session_state:
            self._render_calibration_test_results(st.session_state['cal_test_results'])

    # ----------------------
    # Calibration test execution
    # ----------------------
    def _validate_device_capabilities(self):
        """
        Validate that the selected device supports the configured number of channels.

        Raises:
            ValueError: If device doesn't support required channels
        """
        num_channels = self.recorder.multichannel_config.get('num_channels', 1)
        try:
            devices_info = self.recorder.get_device_info_with_channels()
            current_device_id = int(getattr(self.recorder, 'input_device', -1))

            if current_device_id == -1:
                max_device_channels = max((d['max_channels'] for d in devices_info['input_devices']), default=1)
            else:
                max_device_channels = 1
                for dev in devices_info['input_devices']:
                    if dev['device_id'] == current_device_id:
                        max_device_channels = dev['max_channels']
                        break

            if num_channels > max_device_channels:
                raise ValueError(
                    f"Device capability mismatch: Your input device only supports {max_device_channels} channels, "
                    f"but multi-channel configuration is set to {num_channels} channels. "
                    f"Please reduce the number of channels in Device Selection tab."
                )
        except Exception as e:
            if "capability mismatch" in str(e):
                raise
            # Continue if we can't check (device info might not be available)
            pass

    def _format_calibration_result_for_gui(self, recorder_result: Dict) -> Dict:
        """
        Format recorder's calibration result for GUI compatibility.

        Args:
            recorder_result: Result from recorder.take_record_calibration()

        Returns:
            Dict formatted for GUI display
        """
        cal_ch = recorder_result['metadata']['calibration_channel']
        alignment_result = recorder_result['alignment_metadata']
        valid_cycle_indices = alignment_result['valid_cycle_indices']

        # Build validation results for aligned cycles
        aligned_validation_results = []
        for i, original_idx in enumerate(valid_cycle_indices):
            if original_idx < len(recorder_result['validation_results']):
                aligned_validation_results.append(recorder_result['validation_results'][original_idx])

        return {
            'success': True,
            'num_cycles': recorder_result['metadata']['num_cycles'],
            'calibration_channel': cal_ch,
            'sample_rate': self.recorder.sample_rate,
            # Initial extraction - FOR EXISTING UI
            'all_calibration_cycles': recorder_result['calibration_cycles'],
            'validation_results': recorder_result['validation_results'],
            # Alignment - FOR NEW ALIGNMENT SECTION AND DOWNSTREAM USE
            'alignment_metadata': alignment_result,
            'aligned_cycles': recorder_result['aligned_multichannel_cycles'].get(cal_ch),
            'aligned_multichannel_cycles': recorder_result['aligned_multichannel_cycles'],
            'aligned_validation_results': aligned_validation_results,
            'cycle_duration_s': self.recorder.cycle_samples / self.recorder.sample_rate
        }

    def _perform_calibration_test(self) -> Dict:
        """
        Perform a calibration test using recorder's calibration mode.

        This method validates device capabilities, then delegates to the recorder's
        calibration mode implementation to avoid duplication.

        Returns:
            Dictionary with test results including:
            - all_calibration_cycles: Raw waveforms for each cycle
            - validation_results: Quality metrics for each cycle
            - aligned_cycles: Aligned calibration channel cycles
            - aligned_multichannel_cycles: Aligned cycles for all channels
            - sample_rate: Sample rate for waveform playback
        """
        # Validate device capabilities (GUI-specific check)
        self._validate_device_capabilities()

        # Use recorder's calibration mode (eliminates duplication)
        try:
            recorder_result = self.recorder.take_record_calibration()
        except Exception as e:
            # Re-raise with user-friendly message
            if "no data captured" in str(e).lower():
                raise ValueError("Recording failed - no data captured. Check your audio device connections.")
            raise

        # Format result for GUI compatibility
        return self._format_calibration_result_for_gui(recorder_result)

    def _render_calibration_test_results(self, results: Dict):
        """
        Render calibration test results with Quality Metrics Summary and Per-Cycle Analysis.

        The Quality Metrics Summary table allows clicking on cycles to view detailed analysis.
        """
        st.markdown("#### Calibration Test Results")

        # Extract data
        num_cycles = results.get('num_cycles', 0)
        cal_ch = results.get('calibration_channel', 0)
        sample_rate = results.get('sample_rate', 48000)
        calibration_cycles = results.get('all_calibration_cycles')  # Shape: (num_cycles, cycle_samples)
        validation_results = results.get('validation_results', [])
        cycle_duration_s = results.get('cycle_duration_s', 0.1)

        if calibration_cycles is None or len(validation_results) == 0:
            st.warning("No calibration data available. Please run the calibration test.")
            return

        # Overall summary
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Cycles Recorded", num_cycles)
        with col2:
            valid_count = sum(1 for v in validation_results if v.get('calibration_valid', False))
            st.metric("Valid Cycles", valid_count)
        with col3:
            st.metric("Calibration Channel", f"Ch {cal_ch}")

        st.markdown("---")

        # Quality Metrics Summary Table with Checkboxes
        st.markdown("#### Quality Metrics Summary")
        st.info("💡 Check the boxes to select cycles for analysis, comparison, or threshold learning")
        import pandas as pd

        # Initialize selection state if not exists
        if 'cal_test_selected_cycles' not in st.session_state:
            st.session_state['cal_test_selected_cycles'] = []

        # Display table with checkboxes for each row
        selected_cycles = []

        # Create column headers
        cols = st.columns([0.4, 0.6, 0.6, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 2.0])
        with cols[0]:
            st.markdown("**Select**")
        with cols[1]:
            st.markdown("**Cycle**")
        with cols[2]:
            st.markdown("**Valid**")
        with cols[3]:
            st.markdown("**Neg Peak**")
        with cols[4]:
            st.markdown("**Precursor**")
        with cols[5]:
            st.markdown("**Width ms**")
        with cols[6]:
            st.markdown("**1st Pos**")
        with cols[7]:
            st.markdown("**1st Time**")
        with cols[8]:
            st.markdown("**Max Pos**")
        with cols[9]:
            st.markdown("**2nd Neg**")
        with cols[10]:
            st.markdown("**Issues**")

        st.markdown("---")

        # Render each row with a checkbox
        for v_result in validation_results:
            cycle_idx = v_result.get('cycle_index', 0)
            valid = v_result.get('calibration_valid', False)
            metrics = v_result.get('calibration_metrics', {})
            failures = v_result.get('calibration_failures', [])

            cols = st.columns([0.4, 0.6, 0.6, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 2.0])

            with cols[0]:
                # Checkbox for this cycle
                is_checked = st.checkbox(
                    "",
                    value=cycle_idx in st.session_state['cal_test_selected_cycles'],
                    key=f"cycle_checkbox_{cycle_idx}",
                    label_visibility="collapsed"
                )
                if is_checked:
                    if cycle_idx not in selected_cycles:
                        selected_cycles.append(cycle_idx)

            with cols[1]:
                st.markdown(f"{cycle_idx}")
            with cols[2]:
                st.markdown('✓' if valid else '✗')
            with cols[3]:
                st.markdown(f"{metrics.get('negative_peak', 0):.3f}")
            with cols[4]:
                st.markdown(f"{metrics.get('precursor_ratio', 0):.3f}")
            with cols[5]:
                st.markdown(f"{metrics.get('negative_peak_width_ms', 0):.2f}")
            with cols[6]:
                st.markdown(f"{metrics.get('first_positive_ratio', 0):.3f}")
            with cols[7]:
                st.markdown(f"{metrics.get('first_positive_time_ms', 0):.2f}")
            with cols[8]:
                st.markdown(f"{metrics.get('highest_positive_ratio', 0):.3f}")
            with cols[9]:
                st.markdown(f"{metrics.get('secondary_negative_ratio', 0):.3f}")
            with cols[10]:
                st.markdown(', '.join(failures) if failures else 'None')

        # Update session state with current selections
        st.session_state['cal_test_selected_cycles'] = sorted(selected_cycles)

        st.markdown("---")

        # Display selection info
        if selected_cycles:
            st.success(f"✓ Selected {len(selected_cycles)} cycle(s): {', '.join(map(str, selected_cycles))}")
            st.info("💡 These cycles will be used for: detailed analysis, comparison overlay, and threshold learning (Section 1)")
        else:
            st.info("👆 Check the boxes above to select cycles")

        st.markdown("---")

        # Waveform Visualization (unified component for single or multiple cycles)
        st.markdown("#### Waveform Analysis")

        if selected_cycles:
            if VISUALIZER_AVAILABLE and AudioVisualizer:
                # Use the same unified component for both single and multiple waveforms
                signals = [calibration_cycles[i] for i in selected_cycles]
                labels = [f"Cycle {i} {'✓' if validation_results[i].get('calibration_valid', False) else '✗'}"
                         for i in selected_cycles]

                # Generate appropriate title
                if len(selected_cycles) == 1:
                    title = f"Calibration Impulse - Cycle {selected_cycles[0]}"
                else:
                    title = f"Calibration Impulse - {len(selected_cycles)} Cycles Overlay"

                AudioVisualizer.render_multi_waveform_with_zoom(
                    audio_signals=signals,
                    sample_rate=sample_rate,
                    labels=labels,
                    title=title,
                    component_id="cal_waveform_viz",  # Same ID for both single and multiple
                    height=400,
                    normalize=False,
                    show_analysis=True
                )
            else:
                st.warning("AudioVisualizer not available - cannot display waveform")
                st.info("Install gui_audio_visualizer.py to enable waveform visualization")
        else:
            st.info("👆 Select one or more cycles in the Quality Metrics Summary table above to view waveforms")

        st.markdown("---")

        # User guidance
        with st.expander("💡 How to Use the Calibration Impulse Tool", expanded=False):
            st.markdown("""
            **Purpose:** Configure and validate quality criteria for calibration impulses in multi-channel recording.

            **Quick Workflow:**

            1. **Run Test** → Section 2: Click "Run Calibration Test"
            2. **Select Cycles** → Section 2: Check boxes next to good quality cycles in the table
            3. **Auto-Calculate** → Section 1: Click "Calculate Thresholds from Selected Cycles"
            4. **Save** → Section 1: Click "Save Configuration"

            ---

            **Section 1: Calibration Quality Parameters** (Collapsible)

            - **Tool 1 - Manual Configuration:**
              - Edit thresholds directly in the tabular form
              - Adjust min/max ranges for each quality metric

            - **Tool 2 - Automatic Threshold Learning:**
              - Select cycles in Section 2's Quality Metrics Summary table
              - The selected cycles appear here automatically
              - Click "Calculate Thresholds from Selected Cycles" to auto-compute optimal ranges
              - Review calculated thresholds displayed in Tool 1
              - Click "Save Configuration" at the bottom to persist settings

            **Section 2: Test Calibration Impulse** (Collapsible)

            1. **Run Test:** Click "Run Calibration Test" to record calibration impulses

            2. **Select Cycles in Table:**
               - Check the boxes next to cycles in the Quality Metrics Summary table
               - Check multiple boxes to select multiple cycles
               - Selected cycles are used for:
                 - **Waveform Visualization** (single or overlay depending on selection)
                 - **Threshold Learning** (Section 1, Tool 2)

            3. **Waveform Analysis:**
               - **Unified component:** Same visualization and controls for single or multiple cycles
               - **View modes:** Toggle between waveform and spectrum views
               - **Zoom controls:** Use sliders in "Zoom Controls" expander to zoom to any time range
               - **Persistent zoom:** Zoom settings preserved when adding/removing cycles
               - **Reset Zoom:** Button to return to full view
               - All quality metrics are visible in the table above
               - Cycles shown are **aligned** via cross-correlation for accurate comparison

            ---

            **Quality Criteria:**
            - **Negative Peak:** Strong negative pulse from hammer impact (absolute amplitude)
            - **Positive Peak:** Minimal positive component (absolute amplitude)
            - **Aftershock:** No significant rebounds within 10ms of main pulse (absolute amplitude)

            **What Makes a Good Calibration Impulse:**
            - Single strong negative pulse (hammer impact signature)
            - Sharp, clean waveform with quick decay
            - No aftershocks or bounces immediately after impact
            - Minimal positive component
            - Consistent amplitude across cycles

            **Tips:**
            - Select at least 3-5 good cycles for reliable automatic threshold calculation
            - Use the comparison overlay to verify selected cycles are similar
            - The system adds 5% safety margin to calculated thresholds
            - After saving new thresholds, clear results and re-run to validate
            - Invalid cycles aren't bad - they help you understand quality variations
            """)

        # ====================================================================
        # Alignment Results Review Section (at end of panel)
        # ====================================================================
        alignment_metadata = results.get('alignment_metadata')
        all_cycles = results.get('all_calibration_cycles')  # ALL initial cycles
        aligned_cycles_data = results.get('aligned_cycles')  # Only filtered, aligned cycles
        aligned_validation = results.get('aligned_validation_results', [])

        if alignment_metadata and all_cycles is not None:
            st.markdown("---")
            st.markdown("#### Alignment Results Review")
            st.markdown("""
            **Onset-Based Cycle Alignment:** This section shows valid cycles aligned by their negative peak (hammer impact onset).
            - Invalid cycles are filtered out (only valid cycles shown)
            - Negative peak (onset) detected in each valid cycle
            - All cycles shifted so onsets align at common position
            - Cycles with poor correlation after alignment are filtered out
            - Result: All displayed cycles should overlay precisely
            """)

            # Extract alignment data
            valid_cycle_indices = alignment_metadata.get('valid_cycle_indices', [])
            onset_positions = alignment_metadata.get('onset_positions', [])
            aligned_onset_position = alignment_metadata.get('aligned_onset_position', 0)
            correlations = alignment_metadata.get('correlations', [])
            reference_idx = alignment_metadata.get('reference_cycle_idx', 0)
            correlation_threshold = alignment_metadata.get('correlation_threshold', 0.7)

            num_initial = len(all_cycles)
            num_aligned = len(aligned_cycles_data) if aligned_cycles_data is not None else 0

            # Summary metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Initial Cycles", num_initial)
            with col2:
                st.metric("Valid & Aligned", num_aligned)
            with col3:
                mean_corr = np.mean(correlations) if correlations else 0
                st.metric("Mean Correlation", f"{mean_corr:.3f}")
            with col4:
                st.metric("Aligned Onset Pos", f"{aligned_onset_position} samples")

            st.markdown("---")

            # Alignment Table with Checkboxes (same pattern as Quality Metrics table)
            st.markdown("#### Aligned Cycles Table")
            st.info("💡 Check the boxes to select cycles for overlay visualization - aligned cycles should overlap exactly")

            # Initialize alignment selection state
            if 'alignment_selected_cycles' not in st.session_state:
                st.session_state['alignment_selected_cycles'] = []

            alignment_selected_cycles = []

            # Create column headers
            cols = st.columns([0.5, 0.8, 1.0, 1.0, 1.0, 0.8, 1.0, 0.7])
            with cols[0]:
                st.markdown("**Select**")
            with cols[1]:
                st.markdown("**Cycle #**")
            with cols[2]:
                st.markdown("**Original Onset**")
            with cols[3]:
                st.markdown("**Aligned Onset**")
            with cols[4]:
                st.markdown("**Correlation**")
            with cols[5]:
                st.markdown("**Valid**")
            with cols[6]:
                st.markdown("**Neg. Peak**")
            with cols[7]:
                st.markdown("**Note**")

            st.markdown("---")

            # Render each row with checkbox - only for aligned cycles
            for aligned_idx, original_idx in enumerate(valid_cycle_indices):
                # All cycles here passed validation
                is_ref = "REF" if aligned_idx == reference_idx else ""

                # Get metrics for aligned cycle
                aligned_peak = aligned_validation[aligned_idx].get('calibration_metrics', {}).get('negative_peak', 0) if aligned_idx < len(aligned_validation) else 0

                # Get onset positions
                original_onset = onset_positions[aligned_idx] if aligned_idx < len(onset_positions) else 0

                cols = st.columns([0.5, 0.8, 1.0, 1.0, 1.0, 0.8, 1.0, 0.7])

                with cols[0]:
                    is_checked = st.checkbox(
                        "",
                        value=aligned_idx in st.session_state['alignment_selected_cycles'],
                        key=f"alignment_checkbox_{aligned_idx}",
                        label_visibility="collapsed"
                    )
                    if is_checked:
                        if aligned_idx not in alignment_selected_cycles:
                            alignment_selected_cycles.append(aligned_idx)

                with cols[1]:
                    st.markdown(f"{original_idx}")
                with cols[2]:
                    st.markdown(f"{original_onset} samples")
                with cols[3]:
                    st.markdown(f"{aligned_onset_position} samples")
                with cols[4]:
                    st.markdown(f"{correlations[aligned_idx]:.3f}" if aligned_idx < len(correlations) else "N/A")
                with cols[5]:
                    st.markdown('✓')  # All shown cycles are valid
                with cols[6]:
                    st.markdown(f"{aligned_peak:.3f}")
                with cols[7]:
                    st.markdown(is_ref)

            # Update session state
            st.session_state['alignment_selected_cycles'] = sorted(alignment_selected_cycles)

            st.caption(f"Showing {num_aligned} valid, aligned cycles (invalid cycles filtered out)")
            st.markdown("---")

            # Display selection info
            if alignment_selected_cycles:
                original_cycle_numbers = [valid_cycle_indices[idx] for idx in alignment_selected_cycles if idx < len(valid_cycle_indices)]
                st.success(f"✓ Selected {len(alignment_selected_cycles)} cycle(s) - Original cycle #: {', '.join(map(str, original_cycle_numbers))}")
                st.info("💡 These aligned cycles will be overlaid in the visualization below - they should overlap exactly at the onset")
            else:
                st.info("👆 Check the boxes above to select cycles for visualization")

            st.markdown("---")

            # Visualization: Aligned cycles overlay
            st.markdown("#### Aligned Cycles Overlay")

            if alignment_selected_cycles:
                if VISUALIZER_AVAILABLE and AudioVisualizer:
                    # Prepare signals for visualization
                    # Show all selected aligned cycles overlaid
                    signals = []
                    labels = []

                    for aligned_idx in alignment_selected_cycles:
                        if aligned_idx < len(aligned_cycles_data):
                            original_idx = valid_cycle_indices[aligned_idx] if aligned_idx < len(valid_cycle_indices) else aligned_idx
                            # Add aligned waveform
                            signals.append(aligned_cycles_data[aligned_idx])
                            labels.append(f"Cycle {original_idx} (aligned)")

                    # Generate title
                    if len(alignment_selected_cycles) == 1:
                        original_idx = valid_cycle_indices[alignment_selected_cycles[0]] if alignment_selected_cycles[0] < len(valid_cycle_indices) else alignment_selected_cycles[0]
                        title = f"Aligned Cycle {original_idx} - Onset at {aligned_onset_position} samples"
                    else:
                        title = f"Aligned Cycles Overlay ({len(alignment_selected_cycles)} cycles) - All onsets aligned at {aligned_onset_position} samples"

                    AudioVisualizer.render_multi_waveform_with_zoom(
                        audio_signals=signals,
                        sample_rate=sample_rate,
                        labels=labels,
                        title=title,
                        component_id="alignment_overlay_viz",
                        height=400,
                        normalize=False,
                        show_analysis=True
                    )

                    st.info("💡 All displayed cycles have been aligned by their negative peak (onset). They should overlap precisely.")

                    # Show detailed metrics for selected cycles
                    st.markdown("**Selected Cycles Details:**")

                    for aligned_idx in alignment_selected_cycles:
                        if aligned_idx < len(valid_cycle_indices):
                            original_idx = valid_cycle_indices[aligned_idx]
                            with st.expander(f"Cycle {original_idx} Details", expanded=False):
                                col1, col2 = st.columns(2)

                                with col1:
                                    st.markdown("**Aligned Cycle Metrics:**")
                                    if aligned_idx < len(aligned_validation):
                                        metrics = aligned_validation[aligned_idx].get('calibration_metrics', {})
                                        st.write(f"Negative Peak: {metrics.get('negative_peak', 0):.3f}")
                                        st.write(f"Positive Peak: {metrics.get('positive_peak', 0):.3f}")
                                        st.write(f"Aftershock: {metrics.get('aftershock', 0):.3f}")
                                        st.write(f"Valid: ✓")

                                with col2:
                                    st.markdown("**Alignment Info:**")
                                    if aligned_idx < len(onset_positions):
                                        st.write(f"Original Onset Position: {onset_positions[aligned_idx]} samples")
                                    st.write(f"Aligned Onset Position: {aligned_onset_position} samples")
                                    if aligned_idx < len(correlations):
                                        st.write(f"Correlation: {correlations[aligned_idx]:.3f}")
                                    if aligned_idx == reference_idx:
                                        st.write("**[REFERENCE CYCLE]**")

                else:
                    st.warning("AudioVisualizer not available - cannot display overlay")
                    st.info("Install gui_audio_visualizer.py to enable waveform visualization")
            else:
                st.info("👆 Select one or more cycles in the Aligned Cycles Table above to view overlay")
