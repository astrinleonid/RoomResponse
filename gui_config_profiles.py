#!/usr/bin/env python3
"""
Configuration Profile Manager

Manages save/load of named configuration profiles for the Room Response system.
Profiles include recorder settings, multichannel configuration, and calibration thresholds.

Features:
- Save current configuration as named profile
- Load saved profiles
- Delete profiles
- List available profiles
- Auto-save last used profile
"""

import json
import streamlit as st
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime


class ConfigProfileManager:
    """Manager for configuration profiles"""

    def __init__(self, config_dir: str = "configs", default_config_file: str = "recorderConfig.json", recorder=None):
        """
        Initialize the profile manager.

        Args:
            config_dir: Directory to store profile files
            default_config_file: Path to the default/active configuration file
            recorder: Optional RoomResponseRecorder instance to update after loading
        """
        self.config_dir = Path(config_dir)
        self.default_config_file = Path(default_config_file)
        self.config_dir.mkdir(exist_ok=True)
        self.recorder = recorder

        # Session state keys
        self.SK_CURRENT_PROFILE = "config_profile_current"
        self.SK_PROFILE_LIST = "config_profile_list"

    def get_profile_path(self, profile_name: str) -> Path:
        """Get the file path for a profile"""
        # Sanitize profile name
        safe_name = "".join(c for c in profile_name if c.isalnum() or c in (' ', '_', '-')).strip()
        return self.config_dir / f"{safe_name}.json"

    def list_profiles(self) -> List[str]:
        """List all available profile names"""
        profiles = []
        for file_path in self.config_dir.glob("*.json"):
            profiles.append(file_path.stem)
        return sorted(profiles)

    def load_config(self, file_path: Optional[Path] = None) -> Dict[str, Any]:
        """
        Load configuration from file.

        Args:
            file_path: Path to config file (defaults to active config)

        Returns:
            Configuration dictionary
        """
        if file_path is None:
            file_path = self.default_config_file

        try:
            with open(file_path, 'r') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            st.error(f"Failed to load config from {file_path}: {e}")
            return {}

    def save_config(self, config: Dict[str, Any], file_path: Optional[Path] = None) -> bool:
        """
        Save configuration to file.

        Args:
            config: Configuration dictionary
            file_path: Path to save to (defaults to active config)

        Returns:
            True if successful, False otherwise
        """
        if file_path is None:
            file_path = self.default_config_file

        try:
            with open(file_path, 'w') as f:
                json.dump(config, f, indent=2)
            return True
        except Exception as e:
            st.error(f"Failed to save config to {file_path}: {e}")
            return False

    def save_profile(self, profile_name: str, config: Optional[Dict[str, Any]] = None) -> bool:
        """
        Save current configuration as a named profile.

        Args:
            profile_name: Name for the profile
            config: Configuration to save (if None, loads from default config file)

        Returns:
            True if successful, False otherwise
        """
        if not profile_name:
            st.error("Profile name cannot be empty")
            return False

        if config is None:
            config = self.load_config()

        if not config:
            st.error("No configuration to save")
            return False

        # Add metadata
        config['_profile_metadata'] = {
            'name': profile_name,
            'created': datetime.now().isoformat(),
            'description': f"Configuration profile: {profile_name}"
        }

        profile_path = self.get_profile_path(profile_name)
        success = self.save_config(config, profile_path)

        if success:
            st.success(f"✓ Profile '{profile_name}' saved to {profile_path.name}")
            # Refresh profile list in session state
            st.session_state[self.SK_PROFILE_LIST] = self.list_profiles()

        return success

    def _reload_recorder_config(self, config: Dict[str, Any]) -> bool:
        """
        Reload recorder configuration from config dictionary.

        Args:
            config: Configuration dictionary to apply

        Returns:
            True if successful, False otherwise
        """
        if self.recorder is None:
            return False

        try:
            # Update basic recording parameters
            if 'sample_rate' in config:
                self.recorder.sample_rate = config['sample_rate']
            if 'pulse_duration' in config:
                self.recorder.pulse_duration = config['pulse_duration']
            if 'pulse_fade' in config:
                self.recorder.pulse_fade = config['pulse_fade']
            if 'cycle_duration' in config:
                self.recorder.cycle_duration = config['cycle_duration']
            if 'num_pulses' in config:
                self.recorder.num_pulses = config['num_pulses']
            if 'volume' in config:
                self.recorder.volume = config['volume']
            if 'pulse_frequency' in config:
                self.recorder.pulse_frequency = config['pulse_frequency']
            if 'impulse_form' in config:
                self.recorder.impulse_form = config['impulse_form']
            if 'input_device' in config:
                self.recorder.input_device = config['input_device']
            if 'output_device' in config:
                self.recorder.output_device = config['output_device']

            # Update multi-channel configuration
            if 'multichannel_config' in config:
                mc_config = config['multichannel_config']
                self.recorder.multichannel_config['enabled'] = mc_config.get('enabled', False)
                self.recorder.multichannel_config['num_channels'] = mc_config.get('num_channels', 1)
                self.recorder.multichannel_config['channel_names'] = mc_config.get('channel_names', ['Channel 0'])
                self.recorder.multichannel_config['calibration_channel'] = mc_config.get('calibration_channel')
                self.recorder.multichannel_config['reference_channel'] = mc_config.get('reference_channel', 0)
                self.recorder.multichannel_config['response_channels'] = mc_config.get('response_channels', [0])

            # Update calibration quality configuration
            if 'calibration_quality_config' in config:
                self.recorder.calibration_quality_config = config['calibration_quality_config']

            # Recalculate derived values
            self.recorder._update_derived_values()

            return True
        except Exception as e:
            st.error(f"Failed to reload recorder configuration: {e}")
            return False

    def load_profile(self, profile_name: str, apply_to_active: bool = True) -> Optional[Dict[str, Any]]:
        """
        Load a named profile.

        Args:
            profile_name: Name of profile to load
            apply_to_active: If True, also save to the active config file

        Returns:
            Configuration dictionary if successful, None otherwise
        """
        profile_path = self.get_profile_path(profile_name)

        if not profile_path.exists():
            st.error(f"Profile '{profile_name}' not found")
            return None

        config = self.load_config(profile_path)

        if not config:
            return None

        if apply_to_active:
            # Remove metadata before saving to active config
            config_to_save = {k: v for k, v in config.items() if k != '_profile_metadata'}
            if self.save_config(config_to_save):
                # Reload recorder configuration if recorder instance is available
                if self._reload_recorder_config(config_to_save):
                    st.success(f"✓ Profile '{profile_name}' loaded and applied to recorder")
                else:
                    st.success(f"✓ Profile '{profile_name}' loaded to config file")
                    st.warning("⚠️ Recorder configuration not updated in memory. Restart the GUI to apply changes.")
                st.session_state[self.SK_CURRENT_PROFILE] = profile_name

        return config

    def delete_profile(self, profile_name: str) -> bool:
        """
        Delete a named profile.

        Args:
            profile_name: Name of profile to delete

        Returns:
            True if successful, False otherwise
        """
        profile_path = self.get_profile_path(profile_name)

        if not profile_path.exists():
            st.error(f"Profile '{profile_name}' not found")
            return False

        try:
            profile_path.unlink()
            st.success(f"✓ Profile '{profile_name}' deleted")

            # Clear from session state if it was active
            if st.session_state.get(self.SK_CURRENT_PROFILE) == profile_name:
                st.session_state[self.SK_CURRENT_PROFILE] = None

            # Refresh profile list
            st.session_state[self.SK_PROFILE_LIST] = self.list_profiles()
            return True

        except Exception as e:
            st.error(f"Failed to delete profile: {e}")
            return False

    def render_sidebar_ui(self):
        """
        Render the configuration profile manager UI in the sidebar.
        Should be called from the main GUI launcher.
        """
        st.sidebar.markdown("---")
        st.sidebar.markdown("### Configuration Profiles")

        # Initialize session state
        if self.SK_PROFILE_LIST not in st.session_state:
            st.session_state[self.SK_PROFILE_LIST] = self.list_profiles()
        if self.SK_CURRENT_PROFILE not in st.session_state:
            st.session_state[self.SK_CURRENT_PROFILE] = None

        profiles = st.session_state[self.SK_PROFILE_LIST]
        current = st.session_state[self.SK_CURRENT_PROFILE]

        # Current profile display
        if current:
            st.sidebar.info(f"📋 Active: **{current}**")
        else:
            st.sidebar.caption("📋 Using default configuration")

        # Profile selector
        if profiles:
            st.sidebar.markdown("#### Load Profile")

            # Add "None" option to allow deselecting
            profile_options = ["<select profile>"] + profiles
            default_idx = 0
            if current and current in profiles:
                default_idx = profiles.index(current) + 1

            selected = st.sidebar.selectbox(
                "Available Profiles",
                options=profile_options,
                index=default_idx,
                key="profile_selector"
            )

            col1, col2 = st.sidebar.columns(2)

            with col1:
                if st.button("📂 Load", key="load_profile_btn", use_container_width=True):
                    if selected != "<select profile>":
                        self.load_profile(selected, apply_to_active=True)
                        st.rerun()

            with col2:
                if st.button("🗑️ Delete", key="delete_profile_btn", use_container_width=True):
                    if selected != "<select profile>":
                        if self.delete_profile(selected):
                            st.rerun()
        else:
            st.sidebar.caption("No saved profiles yet")

        # Save new profile section
        st.sidebar.markdown("#### Save Current Config")

        new_profile_name = st.sidebar.text_input(
            "Profile Name",
            placeholder="e.g., 8ch_piano_hammer",
            key="new_profile_name",
            help="Enter a name for the new profile"
        )

        if st.sidebar.button("💾 Save Profile", key="save_profile_btn", use_container_width=True):
            if new_profile_name:
                if self.save_profile(new_profile_name):
                    st.session_state[self.SK_CURRENT_PROFILE] = new_profile_name
                    st.rerun()
            else:
                st.sidebar.error("Please enter a profile name")

        # Show profile count
        st.sidebar.caption(f"📁 {len(profiles)} profile(s) saved in {self.config_dir}/")

    def render_profile_info(self, profile_name: str):
        """
        Render detailed information about a profile.

        Args:
            profile_name: Name of profile to display
        """
        config = self.load_config(self.get_profile_path(profile_name))

        if not config:
            st.error(f"Failed to load profile '{profile_name}'")
            return

        metadata = config.get('_profile_metadata', {})

        st.markdown(f"## Profile: {profile_name}")

        if metadata:
            st.markdown("### Metadata")
            st.json(metadata)

        # Show key settings
        st.markdown("### Configuration Summary")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Recording Settings")
            st.write(f"- Sample Rate: {config.get('sample_rate', 'N/A')} Hz")
            st.write(f"- Pulse Duration: {config.get('pulse_duration', 'N/A')} s")
            st.write(f"- Cycle Duration: {config.get('cycle_duration', 'N/A')} s")
            st.write(f"- Num Pulses: {config.get('num_pulses', 'N/A')}")
            st.write(f"- Volume: {config.get('volume', 'N/A')}")

        with col2:
            st.markdown("#### Multi-Channel Settings")
            mc_config = config.get('multichannel_config', {})
            enabled = mc_config.get('enabled', False)
            st.write(f"- Enabled: {'✓ Yes' if enabled else '✗ No'}")
            if enabled:
                st.write(f"- Channels: {mc_config.get('num_channels', 'N/A')}")
                st.write(f"- Calibration Ch: {mc_config.get('calibration_channel', 'N/A')}")
                st.write(f"- Reference Ch: {mc_config.get('reference_channel', 'N/A')}")


if __name__ == "__main__":
    # Test/demo mode
    st.set_page_config(page_title="Config Profile Manager", layout="wide")

    manager = ConfigProfileManager()

    st.title("Configuration Profile Manager")
    st.markdown("Manage configuration profiles for the Room Response system")

    manager.render_sidebar_ui()

    st.markdown("---")
    st.markdown("### All Profiles")

    profiles = manager.list_profiles()

    if profiles:
        for profile in profiles:
            with st.expander(f"📋 {profile}"):
                manager.render_profile_info(profile)
    else:
        st.info("No profiles saved yet. Create one using the sidebar.")
