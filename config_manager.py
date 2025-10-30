#!/usr/bin/env python3
"""
Centralized Configuration Manager for RoomResponse
Ensures all components use the same configuration file
"""

import json
import time
from pathlib import Path
from typing import Dict, Any, Optional


class ConfigManager:
    """Singleton configuration manager for the entire application."""

    # Single source of truth for config file location
    CONFIG_FILE = "recorderConfig.json"

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ConfigManager, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        self.config_path = Path(self.CONFIG_FILE)

    def get_config_path(self) -> Path:
        """Get the absolute path to the configuration file."""
        return self.config_path.absolute()

    def load_config(self) -> Dict[str, Any]:
        """Load configuration from file.

        Returns:
            Dictionary with configuration values, or empty dict if file doesn't exist
        """
        try:
            if self.config_path.exists():
                with open(self.config_path, 'r') as f:
                    return json.load(f)
        except Exception as e:
            print(f"Warning: Failed to load config from {self.config_path}: {e}")
        return {}

    def save_config(self, config: Dict[str, Any], updated_by: str = "ConfigManager") -> bool:
        """Save configuration to file.

        Args:
            config: Configuration dictionary to save
            updated_by: Name of the component saving the config

        Returns:
            True if successful, False otherwise
        """
        try:
            # Add global timestamp
            config['last_updated'] = time.time()
            config['last_updated_by'] = updated_by

            # Write to file with pretty formatting
            with open(self.config_path, 'w') as f:
                json.dump(config, f, indent=2)

            return True
        except Exception as e:
            print(f"Error: Failed to save config to {self.config_path}: {e}")
            return False

    def update_section(self, section_name: str, section_data: Dict[str, Any], updated_by: str = "ConfigManager") -> bool:
        """Update a specific section of the configuration.

        Args:
            section_name: Name of the section to update (e.g., 'multichannel_config', 'series_config')
            section_data: Data to store in that section
            updated_by: Name of the component updating the config

        Returns:
            True if successful, False otherwise
        """
        try:
            # Load existing config
            config = self.load_config()

            # Update the section
            config[section_name] = section_data

            # Add section-specific timestamp
            if isinstance(section_data, dict):
                config[section_name]['updated_timestamp'] = time.time()
                config[section_name]['updated_by'] = updated_by

            # Save back
            return self.save_config(config, updated_by)
        except Exception as e:
            print(f"Error: Failed to update section {section_name}: {e}")
            return False

    def get_section(self, section_name: str, default: Any = None) -> Any:
        """Get a specific section from the configuration.

        Args:
            section_name: Name of the section to retrieve
            default: Default value if section doesn't exist

        Returns:
            Section data or default value
        """
        config = self.load_config()
        return config.get(section_name, default)

    def update_value(self, key: str, value: Any, updated_by: str = "ConfigManager") -> bool:
        """Update a single top-level value in the configuration.

        Args:
            key: Configuration key to update
            value: New value
            updated_by: Name of the component updating the config

        Returns:
            True if successful, False otherwise
        """
        try:
            config = self.load_config()
            config[key] = value
            return self.save_config(config, updated_by)
        except Exception as e:
            print(f"Error: Failed to update value {key}: {e}")
            return False

    def get_value(self, key: str, default: Any = None) -> Any:
        """Get a single top-level value from the configuration.

        Args:
            key: Configuration key to retrieve
            default: Default value if key doesn't exist

        Returns:
            Value or default
        """
        config = self.load_config()
        return config.get(key, default)


# Global instance
config_manager = ConfigManager()
