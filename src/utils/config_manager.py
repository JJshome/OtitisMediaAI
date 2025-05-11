"""
Configuration management utilities for OtitisMediaAI.
"""

import os
import json
import yaml
import logging
from typing import Dict, Any, Optional


class ConfigManager:
    """
    Manages configuration loading and access for the OtitisMediaAI system.
    Supports JSON and YAML configuration files.
    """
    
    def __init__(self, config_path: str):
        """
        Initialize the ConfigManager with a configuration file path.
        
        Args:
            config_path: Path to the configuration file (JSON or YAML)
        """
        self.logger = logging.getLogger('otitismedia_ai.config_manager')
        self.config_path = config_path
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """
        Load configuration from file.
        
        Returns:
            Configuration dictionary
        
        Raises:
            FileNotFoundError: If the configuration file doesn't exist
            ValueError: If the configuration file has an unsupported format
        """
        if not os.path.exists(self.config_path):
            self.logger.error(f"Configuration file not found: {self.config_path}")
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        
        self.logger.info(f"Loading configuration from {self.config_path}")
        
        file_ext = os.path.splitext(self.config_path)[1].lower()
        
        try:
            if file_ext == '.json':
                with open(self.config_path, 'r') as f:
                    config = json.load(f)
            elif file_ext in ['.yaml', '.yml']:
                with open(self.config_path, 'r') as f:
                    config = yaml.safe_load(f)
            else:
                self.logger.error(f"Unsupported configuration file format: {file_ext}")
                raise ValueError(f"Unsupported configuration file format: {file_ext}")
        except Exception as e:
            self.logger.error(f"Error loading configuration: {str(e)}")
            raise
        
        self.logger.info("Configuration loaded successfully")
        return config
    
    def get_config(self) -> Dict[str, Any]:
        """
        Get the complete configuration.
        
        Returns:
            Complete configuration dictionary
        """
        return self.config
    
    def get_section(self, section: str) -> Dict[str, Any]:
        """
        Get a specific section of the configuration.
        
        Args:
            section: Name of the configuration section
            
        Returns:
            Configuration section as a dictionary
            
        Raises:
            KeyError: If the requested section doesn't exist
        """
        if section not in self.config:
            self.logger.warning(f"Configuration section not found: {section}")
            raise KeyError(f"Configuration section not found: {section}")
        
        return self.config[section]
    
    def get_value(self, key: str, default: Optional[Any] = None) -> Any:
        """
        Get a specific configuration value.
        
        Args:
            key: Configuration key, can be in dot notation for nested keys
            default: Default value to return if the key doesn't exist
            
        Returns:
            Configuration value or default if not found
        """
        keys = key.split('.')
        value = self.config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            self.logger.debug(f"Configuration key not found: {key}, using default: {default}")
            return default
    
    def save_config(self, config_path: Optional[str] = None) -> None:
        """
        Save the current configuration to a file.
        
        Args:
            config_path: Path to save the configuration to, uses the original path if None
            
        Raises:
            ValueError: If the file format is unsupported
        """
        save_path = config_path or self.config_path
        self.logger.info(f"Saving configuration to {save_path}")
        
        file_ext = os.path.splitext(save_path)[1].lower()
        
        try:
            if file_ext == '.json':
                with open(save_path, 'w') as f:
                    json.dump(self.config, f, indent=2)
            elif file_ext in ['.yaml', '.yml']:
                with open(save_path, 'w') as f:
                    yaml.dump(self.config, f, default_flow_style=False)
            else:
                self.logger.error(f"Unsupported configuration file format: {file_ext}")
                raise ValueError(f"Unsupported configuration file format: {file_ext}")
        except Exception as e:
            self.logger.error(f"Error saving configuration: {str(e)}")
            raise
        
        self.logger.info("Configuration saved successfully")
    
    def update_config(self, updates: Dict[str, Any], save: bool = True) -> None:
        """
        Update the configuration with new values.
        
        Args:
            updates: Dictionary of configuration updates
            save: Whether to save the updated configuration to file
        """
        self.logger.info("Updating configuration")
        
        def update_recursive(target, source):
            for key, value in source.items():
                if key in target and isinstance(target[key], dict) and isinstance(value, dict):
                    update_recursive(target[key], value)
                else:
                    target[key] = value
        
        update_recursive(self.config, updates)
        
        if save:
            self.save_config()
