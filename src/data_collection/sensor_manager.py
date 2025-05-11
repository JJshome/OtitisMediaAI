"""
SensorManager: Unified interface for multi-sensor probe control and data acquisition.
"""

import os
import time
import logging
import threading
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any

from .sensors.image_sensor import ImageSensor
from .sensors.vibration_sensor import VibrationSensor
from .sensors.temperature_sensor import TemperatureSensor
from .sensors.pressure_sensor import PressureSensor
from .sensors.acoustic_sensor import AcousticSensor
from .sensors.eeg_sensor import EEGSensor
from .probe_controller import ProbeController
from ..utils.data_structures import SensorData, SensorType


class SensorManager:
    """
    Manages the collection and synchronization of data from multiple sensors.
    Provides a unified interface for probe control and data acquisition.
    """
    
    def __init__(self, config: Dict[str, Any], mode: str = 'clinical'):
        """
        Initialize the SensorManager with the provided configuration.
        
        Args:
            config: Configuration dictionary with sensor settings
            mode: Operation mode ('clinical', 'research', 'simulation', 'training')
        """
        self.logger = logging.getLogger('otitismedia_ai.sensor_manager')
        self.config = config
        self.mode = mode
        self.is_simulation = (mode == 'simulation')
        self.is_running = False
        self.data_lock = threading.Lock()
        self.latest_data = {}
        
        # Store enabled sensors based on configuration
        self.enabled_sensors = {
            SensorType.IMAGE: config.get('enable_image_sensor', True),
            SensorType.VIBRATION: config.get('enable_vibration_sensor', True),
            SensorType.TEMPERATURE: config.get('enable_temperature_sensor', True),
            SensorType.PRESSURE: config.get('enable_pressure_sensor', True),
            SensorType.ACOUSTIC: config.get('enable_acoustic_sensor', True),
            SensorType.EEG: config.get('enable_eeg_sensor', True)
        }
        
        self.logger.info(f"Initializing SensorManager in {mode} mode")
        self.logger.info(f"Enabled sensors: {[k.name for k, v in self.enabled_sensors.items() if v]}")
        
        # Initialize probe controller
        self.probe_controller = ProbeController(
            config.get('probe_config', {}),
            simulation_mode=self.is_simulation
        )
        
        # Initialize sensors
        self._initialize_sensors()
        
        # Create data collection thread
        self.collection_thread = None
        self.stop_event = threading.Event()
        
        self.logger.info("SensorManager initialization complete")
    
    def _initialize_sensors(self):
        """Initialize individual sensor objects based on configuration."""
        self.sensors = {}
        
        # Image sensor
        if self.enabled_sensors[SensorType.IMAGE]:
            self.sensors[SensorType.IMAGE] = ImageSensor(
                self.config.get('image_sensor_config', {}),
                simulation_mode=self.is_simulation
            )
        
        # Vibration sensor
        if self.enabled_sensors[SensorType.VIBRATION]:
            self.sensors[SensorType.VIBRATION] = VibrationSensor(
                self.config.get('vibration_sensor_config', {}),
                simulation_mode=self.is_simulation
            )
        
        # Temperature sensor
        if self.enabled_sensors[SensorType.TEMPERATURE]:
            self.sensors[SensorType.TEMPERATURE] = TemperatureSensor(
                self.config.get('temperature_sensor_config', {}),
                simulation_mode=self.is_simulation
            )
        
        # Pressure sensor
        if self.enabled_sensors[SensorType.PRESSURE]:
            self.sensors[SensorType.PRESSURE] = PressureSensor(
                self.config.get('pressure_sensor_config', {}),
                simulation_mode=self.is_simulation
            )
        
        # Acoustic sensor
        if self.enabled_sensors[SensorType.ACOUSTIC]:
            self.sensors[SensorType.ACOUSTIC] = AcousticSensor(
                self.config.get('acoustic_sensor_config', {}),
                simulation_mode=self.is_simulation
            )
        
        # EEG sensor
        if self.enabled_sensors[SensorType.EEG]:
            self.sensors[SensorType.EEG] = EEGSensor(
                self.config.get('eeg_sensor_config', {}),
                simulation_mode=self.is_simulation
            )
    
    def start_data_collection(self):
        """Start the data collection process in a separate thread."""
        if self.is_running:
            self.logger.warning("Data collection is already running")
            return
        
        self.logger.info("Starting data collection")
        
        # Connect to all sensors
        for sensor_type, sensor in self.sensors.items():
            try:
                sensor.connect()
                self.logger.info(f"Connected to {sensor_type.name} sensor")
            except Exception as e:
                self.logger.error(f"Failed to connect to {sensor_type.name} sensor: {str(e)}")
                # Disable sensor if connection fails
                self.enabled_sensors[sensor_type] = False
        
        # Connect to probe controller
        try:
            self.probe_controller.connect()
            self.logger.info("Connected to probe controller")
        except Exception as e:
            self.logger.error(f"Failed to connect to probe controller: {str(e)}")
            if not self.is_simulation:
                raise
        
        # Reset stop event
        self.stop_event.clear()
        
        # Start collection thread
        self.collection_thread = threading.Thread(target=self._collection_loop)
        self.collection_thread.daemon = True
        self.collection_thread.start()
        
        self.is_running = True
        self.logger.info("Data collection started")
    
    def stop_data_collection(self):
        """Stop the data collection process."""
        if not self.is_running:
            self.logger.warning("Data collection is not running")
            return
        
        self.logger.info("Stopping data collection")
        
        # Signal thread to stop
        self.stop_event.set()
        
        # Wait for thread to terminate
        if self.collection_thread:
            self.collection_thread.join(timeout=5.0)
            if self.collection_thread.is_alive():
                self.logger.warning("Collection thread did not terminate cleanly")
        
        # Disconnect from all sensors
        for sensor_type, sensor in self.sensors.items():
            try:
                sensor.disconnect()
                self.logger.info(f"Disconnected from {sensor_type.name} sensor")
            except Exception as e:
                self.logger.error(f"Error disconnecting from {sensor_type.name} sensor: {str(e)}")
        
        # Disconnect from probe controller
        try:
            self.probe_controller.disconnect()
            self.logger.info("Disconnected from probe controller")
        except Exception as e:
            self.logger.error(f"Error disconnecting from probe controller: {str(e)}")
        
        self.is_running = False
        self.logger.info("Data collection stopped")
    
    def _collection_loop(self):
        """Main data collection loop that runs in a separate thread."""
        sample_interval = 1.0 / self.config.get('sample_rate', 30)  # Default to 30 Hz
        
        while not self.stop_event.is_set():
            loop_start = time.time()
            
            try:
                # Collect data from each sensor
                current_data = {}
                
                for sensor_type, sensor in self.sensors.items():
                    if self.enabled_sensors[sensor_type]:
                        try:
                            sensor_data = sensor.read()
                            current_data[sensor_type] = sensor_data
                        except Exception as e:
                            self.logger.error(f"Error reading from {sensor_type.name} sensor: {str(e)}")
                
                # Get probe status
                try:
                    probe_status = self.probe_controller.get_status()
                    current_data['probe_status'] = probe_status
                except Exception as e:
                    self.logger.error(f"Error getting probe status: {str(e)}")
                
                # Synchronize and store the latest data
                with self.data_lock:
                    timestamp = time.time()
                    self.latest_data = {
                        'timestamp': timestamp,
                        'data': current_data
                    }
                
                # Notify listeners that new data is available
                self._notify_data_available()
                
            except Exception as e:
                self.logger.error(f"Error in data collection loop: {str(e)}")
            
            # Calculate sleep time to maintain target sample rate
            elapsed = time.time() - loop_start
            sleep_time = max(0, sample_interval - elapsed)
            
            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                self.logger.warning(f"Collection loop took {elapsed:.3f}s, exceeding sample interval {sample_interval:.3f}s")
    
    def _notify_data_available(self):
        """Notify listeners that new data is available."""
        # This would typically trigger callbacks to registered listeners
        pass
    
    def get_latest_data(self) -> Dict[str, Any]:
        """
        Get the latest synchronized data from all sensors.
        
        Returns:
            Dictionary containing the latest data and timestamp
        """
        with self.data_lock:
            return self.latest_data.copy()
    
    def collect_diagnostic_data(self, duration_seconds: float = 5.0) -> Dict[str, Any]:
        """
        Collect a complete set of diagnostic data over the specified duration.
        
        Args:
            duration_seconds: Duration in seconds to collect data
            
        Returns:
            Dictionary containing all collected sensor data for the diagnostic session
        """
        self.logger.info(f"Collecting diagnostic data for {duration_seconds} seconds")
        
        if not self.is_running:
            self.logger.info("Starting data collection for diagnostic session")
            self.start_data_collection()
            # Give sensors time to stabilize
            time.sleep(0.5)
        
        # Initialize diagnostic data container
        diagnostic_data = {
            'start_time': time.time(),
            'sensor_data': {},
            'metadata': {
                'duration': duration_seconds,
                'mode': self.mode,
                'enabled_sensors': {k.name: v for k, v in self.enabled_sensors.items()}
            }
        }
        
        # Store samples for each sensor type
        for sensor_type in self.sensors.keys():
            if self.enabled_sensors[sensor_type]:
                diagnostic_data['sensor_data'][sensor_type.name] = []
        
        # Collect data for the specified duration
        end_time = time.time() + duration_seconds
        while time.time() < end_time:
            latest = self.get_latest_data()
            
            if 'data' in latest:
                for sensor_type, sensor_data in latest['data'].items():
                    if isinstance(sensor_type, SensorType) and sensor_type.name in diagnostic_data['sensor_data']:
                        diagnostic_data['sensor_data'][sensor_type.name].append({
                            'timestamp': latest.get('timestamp', time.time()),
                            'data': sensor_data
                        })
            
            time.sleep(0.01)
        
        diagnostic_data['end_time'] = time.time()
        
        self.logger.info(f"Diagnostic data collection completed. Duration: {diagnostic_data['end_time'] - diagnostic_data['start_time']:.2f}s")
        
        return diagnostic_data
    
    def load_simulation_data(self, file_path: str) -> None:
        """
        Load simulation data from a file for use in simulation mode.
        
        Args:
            file_path: Path to the simulation data file
        """
        if not self.is_simulation:
            self.logger.warning("Cannot load simulation data outside of simulation mode")
            return
        
        self.logger.info(f"Loading simulation data from {file_path}")
        
        # Implementation would load data from file and configure sensors to use it
        for sensor_type, sensor in self.sensors.items():
            if self.enabled_sensors[sensor_type]:
                try:
                    sensor.load_simulation_data(file_path)
                except Exception as e:
                    self.logger.error(f"Error loading simulation data for {sensor_type.name} sensor: {str(e)}")
    
    def calibrate_sensors(self) -> bool:
        """
        Run calibration procedure for all enabled sensors.
        
        Returns:
            True if calibration was successful for all sensors, False otherwise
        """
        self.logger.info("Starting sensor calibration")
        
        all_success = True
        
        for sensor_type, sensor in self.sensors.items():
            if self.enabled_sensors[sensor_type]:
                try:
                    success = sensor.calibrate()
                    self.logger.info(f"Calibration for {sensor_type.name} sensor: {'Success' if success else 'Failed'}")
                    all_success = all_success and success
                except Exception as e:
                    self.logger.error(f"Error during calibration of {sensor_type.name} sensor: {str(e)}")
                    all_success = False
        
        return all_success
