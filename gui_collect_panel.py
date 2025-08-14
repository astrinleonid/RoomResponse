#!/usr/bin/env python3
"""
GUI Collection Panel

Handles data collection interface for both single scenarios and series collection.
Uses existing collection logic from collect_dataset.py and DatasetCollector.py
"""

import os
import json
from typing import List
import streamlit as st

# Collection modules
try:
    from DatasetCollector import SingleScenarioCollector
except ImportError:
    SingleScenarioCollector = None

try:
    from collect_dataset import collect_scenarios_series, _parse_series_expr, _load_defaults_from_config
except ImportError:
    collect_scenarios_series = None
    _parse_series_expr = None
    _load_defaults_from_config = None


class CollectionPanel:
    """Collection panel implementation with single and series collection modes."""
    
    def __init__(self, scenario_manager):
        """Initialize with scenario manager for cache management."""
        self.scenario_manager = scenario_manager
    
    def render(self):
        """Render the collection panel UI."""
        st.header("Collect - Data Collection")
        
        # Check dependencies
        if not self._check_dependencies():
            return
        
        # Get dataset root
        root = st.session_state.get("dataset_root", os.getcwd())
        if not os.path.isdir(root):
            st.error("❌ Please provide a valid dataset root directory to continue.", icon="📁")
            return
        
        # Load configuration
        config_data = self._load_configuration(root)
        
        # Collection mode selection
        collection_mode = self._render_mode_selection()
        
        # Common configuration
        common_config = self._render_common_configuration(config_data)
        
        # Mode-specific configuration and execution
        if collection_mode == "Single Scenario":
            self._render_single_scenario_mode(common_config)
        else:
            self._render_series_mode(common_config)
        
        # Post-collection actions
        self._render_post_collection_actions(config_data["config_file"])
    
    def _check_dependencies(self) -> bool:
        """Check if required collection modules are available."""
        if SingleScenarioCollector is None:
            st.error("❌ DatasetCollector module not found. Please ensure DatasetCollector.py and collect_dataset.py are available.", icon="🔧")
            st.info("The DatasetCollector.py and collect_dataset.py files should be in the same directory as this GUI application.")
            return False
        return True
    
    def _load_configuration(self, root: str) -> dict:
        """Load configuration defaults from recorderConfig.json."""
        config_file_path = os.path.join(root, "recorderConfig.json")
        if not os.path.exists(config_file_path):
            config_file_path = "recorderConfig.json"  # fallback to current directory
        
        defaults = {"computer": "Unknown_Computer", "room": "Unknown_Room"}
        if _load_defaults_from_config is not None:
            try:
                defaults = _load_defaults_from_config(config_file_path)
            except Exception:
                pass
        
        return {
            "defaults": defaults,
            "config_file": config_file_path
        }
    
    def _render_mode_selection(self) -> str:
        """Render collection mode selection."""
        st.markdown("### Collection Mode")
        return st.radio(
            "Choose collection mode:",
            options=["Single Scenario", "Series"],
            index=0,
            help="Single: collect one scenario. Series: collect multiple scenarios with delays and beeps."
        )
    
    def _render_common_configuration(self, config_data: dict) -> dict:
        """Render common configuration options."""
        st.markdown("### Common Configuration")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            computer_name = st.text_input(
                "Computer name",
                value=config_data["defaults"].get("computer", "Unknown_Computer"),
                help="Computer/device identifier for scenario naming"
            )
            
            room_name = st.text_input(
                "Room name", 
                value=config_data["defaults"].get("room", "Unknown_Room"),
                help="Room identifier for scenario naming"
            )
            
            num_measurements = st.number_input(
                "Number of measurements",
                min_value=1,
                max_value=1000,
                value=30,
                step=1,
                help="Number of audio measurements per scenario"
            )
        
        with col2:
            measurement_interval = st.number_input(
                "Measurement interval (seconds)",
                min_value=0.1,
                max_value=60.0,
                value=2.0,
                step=0.1,
                help="Time between individual measurements"
            )
            
            interactive_devices = st.checkbox(
                "Interactive device selection",
                value=False,
                help="Enable interactive audio device selection during collection"
            )
            
            config_file = st.text_input(
                "Config file path",
                value=config_data["config_file"],
                help="Path to recorderConfig.json"
            )
        
        # Output directory selection
        st.markdown("### Output Directory")
        
        output_col1, output_col2 = st.columns([3, 1])
        
        with output_col1:
            # Get default output directory (current dataset root)
            default_output = st.session_state.get("dataset_root", os.getcwd())
            
            output_dir = st.text_input(
                "Collection output directory",
                value=default_output,
                help="Directory where collected scenarios will be saved. Can be different from the main dataset root."
            )
            
            # Validate output directory
            output_dir = output_dir.strip() if output_dir else default_output
            if output_dir:
                output_dir = os.path.abspath(output_dir)
                if os.path.exists(output_dir) and os.path.isdir(output_dir):
                    st.success(f"✓ Output directory: {output_dir}", icon="📁")
                elif os.path.exists(output_dir):
                    st.error("⚠️ Path exists but is not a directory", icon="📁")
                else:
                    st.warning(f"⚠️ Directory will be created: {output_dir}", icon="📁")
        
        with output_col2:
            st.markdown("<br>", unsafe_allow_html=True)  # Spacing
            if st.button("📁 Use Dataset Root", help="Set output directory to current dataset root"):
                st.session_state["collection_output_override"] = st.session_state.get("dataset_root", os.getcwd())
                st.rerun()
            
            if st.button("📂 Browse...", help="Browse for output directory"):
                st.info("💡 Tip: Type the full path in the text field above, or use the file system to navigate.")
        
        # Remember user's choice
        if output_dir != default_output:
            st.session_state["collection_output_override"] = output_dir
        
        return {
            "computer_name": computer_name,
            "room_name": room_name,
            "num_measurements": num_measurements,
            "measurement_interval": measurement_interval,
            "interactive_devices": interactive_devices,
            "config_file": config_file,
            "output_dir": output_dir
        }
    
    def _render_single_scenario_mode(self, common_config: dict):
        """Render single scenario collection interface."""
        st.markdown("### Single Scenario Configuration")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            scenario_number = st.text_input(
                "Scenario number",
                value="1",
                help="Scenario identifier (can include letters and decimals, e.g., '1', '2.5', '3a')"
            )
        
        with col2:
            description = st.text_input(
                "Description",
                value=f"Room response measurement scenario {scenario_number}",
                help="Description of the measurement scenario"
            )
        
        # Preview scenario name and path
        if common_config["computer_name"] and common_config["room_name"] and scenario_number:
            scenario_name = f"{common_config['computer_name']}-Scenario{scenario_number}-{common_config['room_name']}"
            scenario_path = os.path.join(common_config["output_dir"], scenario_name)
            st.info(f"📁 Scenario will be saved as: `{scenario_name}`")
            st.caption(f"📍 Full path: `{scenario_path}`")
        
        # Collection controls
        st.markdown("### Execute Collection")
        
        if st.button("🎤 Start Single Scenario Collection", type="primary", use_container_width=True):
            SingleScenarioExecutor(self.scenario_manager).execute(
                common_config=common_config,
                scenario_number=scenario_number,
                description=description
            )
    
    def _render_series_mode(self, common_config: dict):
        """Render series collection interface."""
        st.markdown("### Series Configuration")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            series_scenarios = st.text_input(
                "Scenario numbers",
                value="1,2,3",
                help="Comma-separated list. Supports ranges: '1-3,5,7' = ['1','2','3','5','7']"
            )
            
            pre_delay = st.number_input(
                "Pre-delay (seconds)",
                min_value=0.0,
                max_value=600.0,
                value=60.0,
                step=10.0,
                help="Delay before the first scenario starts"
            )
            
            inter_delay = st.number_input(
                "Inter-scenario delay (seconds)",
                min_value=0.0,
                max_value=600.0,
                value=60.0,
                step=10.0,
                help="Delay between scenarios"
            )
        
        with col2:
            description_template = st.text_input(
                "Description template",
                value="Room response measurement scenario {n}",
                help="Template for scenario descriptions. {n} will be replaced with scenario number"
            )
            
            enable_beeps = st.checkbox(
                "Enable beep notifications",
                value=True,
                help="Play beeps: 1 beep after each scenario, 2 beeps at the end"
            )
            
            beep_config = {}
            if enable_beeps:
                beep_col1, beep_col2 = st.columns([1, 1])
                with beep_col1:
                    beep_config["volume"] = st.slider("Beep volume", 0.0, 1.0, 0.2, 0.05)
                    beep_config["freq"] = st.number_input("Beep frequency (Hz)", 200, 2000, 880, 50)
                with beep_col2:
                    beep_config["duration"] = st.number_input("Beep duration (ms)", 50, 1000, 200, 50)
        
        # Parse and preview series
        if series_scenarios and _parse_series_expr is not None:
            parsed_scenarios = self._preview_series(series_scenarios, common_config)
            
            # Show where scenarios will be saved
            if parsed_scenarios and common_config["output_dir"]:
                st.caption(f"📍 All scenarios will be saved to: `{common_config['output_dir']}`")
            
            # Collection controls
            st.markdown("### Execute Series Collection")
            
            if st.button("🎤 Start Series Collection", type="primary", use_container_width=True):
                if parsed_scenarios:
                    SeriesExecutor(self.scenario_manager).execute(
                        scenario_numbers=parsed_scenarios,
                        common_config=common_config,
                        description_template=description_template,
                        pre_delay=pre_delay,
                        inter_delay=inter_delay,
                        enable_beeps=enable_beeps,
                        beep_config=beep_config
                    )
                else:
                    st.error("❌ No valid scenarios to collect.")
    
    def _preview_series(self, series_scenarios: str, common_config: dict) -> List[str]:
        """Preview and validate series scenarios."""
        try:
            parsed_scenarios = _parse_series_expr(series_scenarios)
            if parsed_scenarios:
                st.info(f"📋 Series will collect {len(parsed_scenarios)} scenarios: {', '.join(parsed_scenarios)}")
                
                # Estimate total time
                total_measurements = len(parsed_scenarios) * common_config["num_measurements"]
                measurement_time = total_measurements * common_config["measurement_interval"]
                delay_time = 60.0 + (len(parsed_scenarios) - 1) * 60.0  # Default delays
                estimated_total = (measurement_time + delay_time) / 60.0  # minutes
                
                st.caption(f"⏱️ Estimated total time: {estimated_total:.1f} minutes ({total_measurements} total measurements)")
                return parsed_scenarios
            else:
                st.warning("⚠️ No valid scenarios parsed from the input.")
                return []
        except Exception as e:
            st.error(f"❌ Error parsing series: {e}")
            return []
    
    def _render_post_collection_actions(self, config_file: str):
        """Render post-collection action buttons."""
        st.markdown("### Post-Collection")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            if st.button("🔄 Refresh Scenarios List", help="Update scenarios list after collection"):
                self.scenario_manager.clear_cache()
                st.rerun()
        
        with col2:
            if st.button("💾 Save Settings as Defaults", help="Save computer and room names to config file"):
                # This would need access to the current form values
                pass


class SingleScenarioExecutor:
    """Handles single scenario collection execution."""
    
    def __init__(self, scenario_manager):
        self.scenario_manager = scenario_manager
    
    def execute(self, common_config: dict, scenario_number: str, description: str):
        """Execute single scenario collection."""
        # Validate inputs
        if not self._validate_inputs(common_config, scenario_number):
            return
        
        scenario_name = f"{common_config['computer_name']}-Scenario{scenario_number}-{common_config['room_name']}"
        
        # Show collection info
        self._show_collection_info(scenario_name, common_config)
        
        try:
            # Setup scenario parameters
            scenario_params = {
                "scenario_number": scenario_number.strip(),
                "description": description.strip(),
                "computer_name": common_config["computer_name"].strip(),
                "room_name": common_config["room_name"].strip(),
                "num_measurements": common_config["num_measurements"],
                "measurement_interval": common_config["measurement_interval"]
            }
            
            # Initialize and run collector
            collector = SingleScenarioCollector(
                base_output_dir=common_config["output_dir"],
                recorder_config=common_config["config_file"],
                scenario_config=scenario_params,
                merge_mode="append",
                allow_config_mismatch=False,
                resume=True
            )
            
            st.info("🎵 Collection started! Monitor console/terminal for progress and controls.")
            st.warning("⚠️ This will block the GUI during collection. Check your terminal for progress updates.")
            
            # Run the collection
            collector.collect_scenario(
                interactive_devices=common_config["interactive_devices"], 
                confirm_start=False
            )
            
            # Success
            st.success(f"🎉 Successfully collected scenario: {scenario_name}")
            st.info(f"📁 Data saved to: {collector.scenario_dir}")
            
            # Offer next steps
            if st.button("🔄 Refresh Scenarios & Go to Process", help="Update scenarios list and go to processing"):
                self.scenario_manager.clear_cache()
                st.info("Scenarios refreshed! Switch to Process panel to extract features.")
            
        except Exception as e:
            st.error(f"❌ Collection failed: {str(e)}")
            st.info("Check the terminal/console for detailed error information.")
    
    def _validate_inputs(self, common_config: dict, scenario_number: str) -> bool:
        """Validate input parameters."""
        if not common_config["computer_name"].strip() or not common_config["room_name"].strip() or not scenario_number.strip():
            st.error("❌ Computer name, room name, and scenario number are required.")
            return False
        return True
    
    def _show_collection_info(self, scenario_name: str, common_config: dict):
        """Show collection information."""
        with st.container():
            st.markdown("### 🎤 Starting Single Scenario Collection")
            st.text(f"Scenario: {scenario_name}")
            st.text(f"Measurements: {common_config['num_measurements']}")
            st.text(f"Interval: {common_config['measurement_interval']}s")
            st.text(f"Interactive devices: {common_config['interactive_devices']}")
            st.text("Initializing...")


class SeriesExecutor:
    """Handles series collection execution."""
    
    def __init__(self, scenario_manager):
        self.scenario_manager = scenario_manager
    
    def execute(self, scenario_numbers: List[str], common_config: dict, description_template: str,
               pre_delay: float, inter_delay: float, enable_beeps: bool, beep_config: dict):
        """Execute series collection."""
        if collect_scenarios_series is None:
            st.error("❌ Series collection function not available.")
            return
        
        # Show collection info
        self._show_collection_info(scenario_numbers, common_config, pre_delay, inter_delay, enable_beeps)
        
        try:
            st.info("🎵 Series collection started! Monitor console/terminal for progress and controls.")
            st.warning("⚠️ This will block the GUI during collection. Check your terminal for progress updates and beep notifications.")
            
            # Use the existing series collection function
            collect_scenarios_series(
                scenario_numbers=scenario_numbers,
                base_output_dir=common_config["output_dir"],
                config_file=common_config["config_file"],
                base_computer=common_config["computer_name"].strip(),
                base_room=common_config["room_name"].strip(),
                description_template=description_template,
                num_measurements=common_config["num_measurements"],
                measurement_interval=common_config["measurement_interval"],
                interactive_devices=common_config["interactive_devices"],
                pre_delay=pre_delay,
                inter_delay=inter_delay,
                enable_beeps=enable_beeps,
                beep_volume=beep_config.get("volume", 0.2) if enable_beeps else 0.2,
                beep_freq=beep_config.get("freq", 880) if enable_beeps else 880,
                beep_dur_ms=beep_config.get("duration", 200) if enable_beeps else 200
            )
            
            st.success(f"🎉 Successfully collected {len(scenario_numbers)} scenarios!")
            
            # Offer next steps
            if st.button("🔄 Refresh Scenarios & Go to Process", help="Update scenarios list and go to processing"):
                self.scenario_manager.clear_cache()
                st.info("Scenarios refreshed! Switch to Process panel to extract features.")
            
        except Exception as e:
            st.error(f"❌ Series collection failed: {str(e)}")
            st.info("Check the terminal/console for detailed error information.")
    
    def _show_collection_info(self, scenario_numbers: List[str], common_config: dict, 
                             pre_delay: float, inter_delay: float, enable_beeps: bool):
        """Show series collection information."""
        with st.container():
            st.markdown("### 🎤 Starting Series Collection")
            total_scenarios = len(scenario_numbers)
            st.text(f"Scenarios: {', '.join(scenario_numbers)} ({total_scenarios} total)")
            st.text(f"Pre-delay: {pre_delay}s | Inter-delay: {inter_delay}s")
            st.text(f"Measurements per scenario: {common_config['num_measurements']}")
            st.text(f"Beeps: {'enabled' if enable_beeps else 'disabled'}")
            st.text("Starting series collection...")


def save_collection_defaults(config_file: str, computer_name: str, room_name: str):
    """Save computer and room names as defaults in the config file."""
    try:
        # Read existing config
        config_data = {}
        if os.path.exists(config_file):
            with open(config_file, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
        
        # Update defaults
        config_data['computer'] = computer_name.strip()
        config_data['room'] = room_name.strip()
        
        # Write back
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config_data, f, indent=2)
        
        st.success(f"✅ Defaults saved to {config_file}")
        
    except Exception as e:
        st.error(f"❌ Failed to save defaults: {e}")