#!/usr/bin/env python3
"""
GUI Processing Panel

Handles feature extraction interface using the existing FeatureExtractor module.
"""

import os
from typing import Optional
import pandas as pd
import streamlit as st

# Feature extraction modules
try:
    from FeatureExtractor import AudioFeatureExtractor
except ImportError:
    AudioFeatureExtractor = None


class ProcessingPanel:
    """Processing panel implementation for feature extraction."""
    
    def __init__(self, scenario_manager):
        """Initialize with scenario manager for data access."""
        self.scenario_manager = scenario_manager
    
    def render(self):
        """Render the processing panel UI."""
        st.header("Process - Feature Extraction")
        
        # Check dependencies
        if not self._check_dependencies():
            return
        
        # Get dataset root
        root = st.session_state.get("dataset_root", os.getcwd())
        if not os.path.isdir(root):
            st.error("❌ Please provide a valid dataset root directory to continue.", icon="📁")
            return
        
        # Load scenarios
        scenarios_df = self._load_scenarios(root)
        if scenarios_df.empty:
            st.info("No scenarios found in the dataset.", icon="📂")
            return
        
        # Configuration
        config = self._render_configuration()
        
        # File handling policy
        file_policy = self._render_file_policy()
        
        # Scenario selection
        selected_scenarios = self._render_scenario_selection(scenarios_df)
        if selected_scenarios.empty:
            return
        
        # Processing preview
        self._render_processing_preview(selected_scenarios, file_policy)
        
        # Processing controls
        self._render_processing_controls(selected_scenarios, config, file_policy, root)
    
    def _check_dependencies(self) -> bool:
        """Check if FeatureExtractor is available."""
        if AudioFeatureExtractor is None:
            st.error("❌ FeatureExtractor module not found. Please ensure FeatureExtractor.py is available.", icon="🔧")
            st.info("The FeatureExtractor.py file should be in the same directory as this GUI application.")
            return False
        return True
    
    def _load_scenarios(self, root: str) -> pd.DataFrame:
        """Load scenarios dataframe."""
        with st.spinner("Loading scenarios..."):
            return self.scenario_manager.build_scenarios_df(root)
    
    def _render_configuration(self) -> dict:
        """Render configuration options."""
        st.markdown("### Configuration")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            wav_subfolder = st.text_input(
                "WAV subfolder",
                value="impulse_responses",
                help="Subfolder within each scenario containing WAV files"
            )
            
            recording_type = st.selectbox(
                "Recording type",
                options=["any", "average", "raw"],
                index=0,
                help="Filter WAV files by type: 'any' (all files), 'average' (processed recordings), 'raw' (raw recordings)"
            )
            
            sample_rate = st.number_input(
                "Sample rate (Hz)",
                min_value=8000,
                max_value=192000,
                value=48000,
                step=1000,
                help="Audio sample rate for feature extraction"
            )
            
            n_mfcc = st.number_input(
                "Number of MFCC coefficients",
                min_value=1,
                max_value=20,
                value=13,
                step=1,
                help="Number of Mel-frequency cepstral coefficients to extract"
            )
        
        with col2:
            max_spectrum_freq = st.number_input(
                "Max spectrum frequency (Hz)",
                min_value=0,
                max_value=100000,
                value=10000,
                step=1000,
                help="Maximum frequency for spectrum features (0 = no limit)"
            )
            if max_spectrum_freq == 0:
                max_spectrum_freq = None
            
            mfcc_filename = st.text_input(
                "MFCC filename",
                value="features.csv",
                help="Output filename for MFCC features"
            )
            
            spectrum_filename = st.text_input(
                "Spectrum filename",
                value="spectrum.csv",
                help="Output filename for spectrum features"
            )
            
            config_filename = st.text_input(
                "Config filename",
                value="recorderConfig.json",
                help="JSON config file to read sample rate from (optional)"
            )
        
        return {
            "wav_subfolder": wav_subfolder,
            "recording_type": recording_type,
            "sample_rate": sample_rate,
            "n_mfcc": n_mfcc,
            "max_spectrum_freq": max_spectrum_freq,
            "mfcc_filename": mfcc_filename,
            "spectrum_filename": spectrum_filename,
            "config_filename": config_filename
        }
    
    def _render_file_policy(self) -> str:
        """Render file handling policy selection."""
        st.markdown("### File Handling Policy")
        return st.radio(
            "What to do with existing feature files?",
            options=[
                "Skip scenarios (both files present)",
                "Keep existing (write missing only)",
                "Overwrite both files"
            ],
            index=0,
            help="Policy for handling scenarios that already have feature files"
        )
    
    def _render_scenario_selection(self, scenarios_df: pd.DataFrame) -> pd.DataFrame:
        """Render scenario selection interface."""
        st.markdown("### Scenario Selection")
        
        # Get selected scenarios from session state
        selected_paths = list(st.session_state.get("scenarios_selected_set", set()))
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            process_mode = st.radio(
                "Process which scenarios?",
                options=["Selected scenarios only", "All scenarios"],
                index=0 if selected_paths else 1,
                help="Choose which scenarios to process"
            )
        
        with col2:
            if process_mode == "Selected scenarios only":
                st.info(f"📋 {len(selected_paths)} scenarios selected")
            else:
                st.info(f"📋 {len(scenarios_df)} total scenarios")
        
        # Determine scenarios to process
        if process_mode == "Selected scenarios only":
            if not selected_paths:
                st.warning("⚠️ No scenarios selected. Please select scenarios in the Scenarios panel first.")
                return pd.DataFrame()
            return scenarios_df[scenarios_df["path"].isin(selected_paths)].copy()
        else:
            return scenarios_df.copy()
    
    def _render_processing_preview(self, scenarios_df: pd.DataFrame, file_policy: str):
        """Render processing preview table."""
        st.markdown("### Processing Preview")
        
        preview_data = []
        for _, row in scenarios_df.iterrows():
            path = row["path"]
            scenario = row["scenario"]
            features_avail = row.get("features_available", {})
            
            mfcc_exists = features_avail.get('mfcc', False)
            spectrum_exists = features_avail.get('spectrum', False)
            audio_count = row.get("sample_count", 0)
            
            # Determine action based on policy
            action = self._determine_action(file_policy, mfcc_exists, spectrum_exists)
            
            preview_data.append({
                "Scenario": scenario,
                "Audio Files": audio_count,
                "MFCC": "✅" if mfcc_exists else "❌",
                "Spectrum": "✅" if spectrum_exists else "❌",
                "Action": action
            })
        
        preview_df = pd.DataFrame(preview_data)
        st.dataframe(preview_df, use_container_width=True, hide_index=True)
        
        # Summary
        actions_summary = preview_df["Action"].value_counts()
        summary_text = " | ".join([f"{action}: {count}" for action, count in actions_summary.items()])
        st.caption(f"**Summary:** {summary_text}")
    
    def _determine_action(self, file_policy: str, mfcc_exists: bool, spectrum_exists: bool) -> str:
        """Determine processing action based on policy and existing files."""
        if file_policy == "Skip scenarios (both files present)":
            if mfcc_exists and spectrum_exists:
                return "Skip (both exist)"
            else:
                return "Process"
        elif file_policy == "Keep existing (write missing only)":
            missing = []
            if not mfcc_exists:
                missing.append("MFCC")
            if not spectrum_exists:
                missing.append("Spectrum")
            return f"Write {', '.join(missing)}" if missing else "Skip (both exist)"
        else:  # Overwrite
            return "Overwrite all"
    
    def _render_processing_controls(self, scenarios_df: pd.DataFrame, config: dict, file_policy: str, root: str):
        """Render processing control buttons."""
        st.markdown("### Execute Processing")
        
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            if st.button("🚀 Start Processing", type="primary", use_container_width=True):
                ProcessingExecutor(self.scenario_manager).execute(
                    scenarios_df=scenarios_df,
                    config=config,
                    file_policy=file_policy,
                    dataset_root=root
                )
        
        with col2:
            if st.button("🔄 Refresh Preview", use_container_width=True):
                self.scenario_manager.clear_cache()
                st.rerun()
        
        with col3:
            st.caption("💡 Tip: Select scenarios in the Scenarios panel first")


class ProcessingExecutor:
    """Handles feature extraction execution."""
    
    def __init__(self, scenario_manager):
        self.scenario_manager = scenario_manager
    
    def execute(self, scenarios_df: pd.DataFrame, config: dict, file_policy: str, dataset_root: str):
        """Execute the feature extraction process."""
        
        # Initialize extractor
        extractor = AudioFeatureExtractor(
            sample_rate=config["sample_rate"],
            n_mfcc=config["n_mfcc"],
            config_filename=config["config_filename"] if config["config_filename"].strip() else None,
            max_spectrum_freq=config["max_spectrum_freq"]
        )
        
        # Determine overwrite policy
        skip_existing, overwrite_existing = self._parse_file_policy(file_policy)
        
        # Progress tracking
        total_scenarios = len(scenarios_df)
        progress_bar = st.progress(0)
        status_text = st.empty()
        results_container = st.empty()
        
        # Processing results
        results = {
            "processed": 0,
            "skipped": 0,
            "failed": 0,
            "details": []
        }
        
        try:
            for idx, (_, row) in enumerate(scenarios_df.iterrows()):
                scenario_path = row["path"]
                scenario_name = row["scenario"]
                
                # Update progress
                progress = (idx + 1) / total_scenarios
                progress_bar.progress(progress)
                status_text.text(f"Processing {idx + 1}/{total_scenarios}: {scenario_name}")
                
                # Check if we should skip this scenario
                if skip_existing:
                    mfcc_file = os.path.join(scenario_path, config["mfcc_filename"])
                    spectrum_file = os.path.join(scenario_path, config["spectrum_filename"])
                    if os.path.exists(mfcc_file) and os.path.exists(spectrum_file):
                        results["skipped"] += 1
                        results["details"].append(f"⏭️ Skipped {scenario_name} (files exist)")
                        continue
                
                # Process the scenario
                try:
                    success = extractor.process_scenario_folder(
                        scenario_folder=scenario_path,
                        wav_subfolder=config["wav_subfolder"],
                        recording_type=config["recording_type"],
                        mfcc_filename=config["mfcc_filename"],
                        spectrum_filename=config["spectrum_filename"],
                        dataset_path_for_config=dataset_root,
                        overwrite_existing_files=overwrite_existing
                    )
                    
                    if success:
                        results["processed"] += 1
                        results["details"].append(f"✅ Processed {scenario_name}")
                    else:
                        results["failed"] += 1
                        results["details"].append(f"❌ Failed {scenario_name} (no files or features)")
                        
                except Exception as e:
                    results["failed"] += 1
                    results["details"].append(f"❌ Error {scenario_name}: {str(e)}")
                
                # Update results display every few iterations
                if (idx + 1) % 3 == 0 or idx == total_scenarios - 1:
                    self._display_results(results, results_container)
        
        except Exception as e:
            st.error(f"Processing interrupted: {e}")
        
        # Final results
        progress_bar.progress(1.0)
        status_text.text("Processing complete!")
        self._display_results(results, results_container, final=True)
        
        # Offer to refresh scenarios
        if results["processed"] > 0:
            st.success(f"🎉 Successfully processed {results['processed']} scenarios!")
            if st.button("🔄 Refresh Scenarios List", help="Update scenarios list to reflect new features"):
                self.scenario_manager.clear_cache()
                st.rerun()
    
    def _parse_file_policy(self, file_policy: str) -> tuple[bool, bool]:
        """Parse file policy into skip_existing and overwrite_existing flags."""
        if file_policy == "Skip scenarios (both files present)":
            return True, False
        elif file_policy == "Keep existing (write missing only)":
            return False, False
        else:  # Overwrite
            return False, True
    
    def _display_results(self, results: dict, container, final: bool = False):
        """Display processing results in the given container."""
        with container.container():
            if final:
                st.markdown("### 📊 Processing Results")
            
            # Summary metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total", results["processed"] + results["skipped"] + results["failed"])
            with col2:
                st.metric("Processed", results["processed"])
            with col3:
                st.metric("Skipped", results["skipped"])
            with col4:
                st.metric("Failed", results["failed"])
            
            # Details
            if results["details"]:
                if final:
                    st.markdown("### 📝 Details")
                    # Show all details for final results
                    for detail in results["details"][-20:]:  # Last 20 entries
                        st.text(detail)
                    if len(results["details"]) > 20:
                        st.caption(f"... and {len(results['details']) - 20} more entries")
                else:
                    # Show just the latest few during processing
                    st.markdown("**Recent activity:**")
                    for detail in results["details"][-3:]:
                        st.text(detail)