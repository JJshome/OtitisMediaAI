#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
OtitisMediaAI Simulation Environment

This module provides a simulation environment for testing the OtitisMediaAI system
without requiring actual hardware sensors. It generates synthetic sensor data
based on predefined patient scenarios.
"""

import os
import sys
import argparse
import numpy as np
import json
import logging
import time
import cv2
from typing import Dict, List, Optional, Tuple, Union, Any
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import threading

# Add src to path for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(parent_dir)

from src.utils.logger import setup_logger
from src.utils.config_manager import ConfigManager
from src.utils.data_structures import SensorType, DiagnosticResult, PrognosisResult


class SimulationCase:
    """
    Represents a simulated patient case with predefined sensor data and expected results.
    """
    
    def __init__(self, 
                case_id: str, 
                name: str, 
                description: str,
                diagnosis: str,
                severity: float,
                has_effusion: bool,
                effusion_type: Optional[str] = None,
                data_dir: Optional[str] = None):
        """
        Initialize a simulation case.
        
        Args:
            case_id: Unique identifier for the case
            name: Case name
            description: Case description
            diagnosis: Expected diagnosis (e.g., 'Normal', 'OME', 'AOM', 'CSOM')
            severity: Expected severity (0.0-1.0)
            has_effusion: Whether the case has middle ear effusion
            effusion_type: Type of effusion if present
            data_dir: Directory containing case-specific sensor data files
        """
        self.case_id = case_id
        self.name = name
        self.description = description
        self.diagnosis = diagnosis
        self.severity = severity
        self.has_effusion = has_effusion
        self.effusion_type = effusion_type
        self.data_dir = data_dir or os.path.join("data", "simulation", case_id)
        
        # Validate data directory
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir, exist_ok=True)
    
    def get_sensor_data(self, sensor_type: SensorType) -> np.ndarray:
        """
        Get simulated sensor data for this case.
        
        Args:
            sensor_type: Type of sensor data to retrieve
            
        Returns:
            Numpy array containing sensor data
        """
        # Get file path for the sensor data
        file_path = self._get_sensor_file_path(sensor_type)
        
        # If file exists, load data from it
        if os.path.exists(file_path):
            return self._load_sensor_data(file_path, sensor_type)
        else:
            # Generate synthetic data if no file exists
            return self._generate_synthetic_data(sensor_type)
    
    def _get_sensor_file_path(self, sensor_type: SensorType) -> str:
        """Get the file path for a specific sensor data file."""
        file_names = {
            SensorType.IMAGE: "tympanic_image.jpg",
            SensorType.VIBRATION: "vibration_data.npy",
            SensorType.TEMPERATURE: "temperature_data.npy",
            SensorType.PRESSURE: "pressure_data.npy",
            SensorType.ACOUSTIC: "acoustic_data.npy",
            SensorType.EEG: "eeg_data.npy"
        }
        
        return os.path.join(self.data_dir, file_names.get(sensor_type, f"{sensor_type.name.lower()}_data.npy"))
    
    def _load_sensor_data(self, file_path: str, sensor_type: SensorType) -> np.ndarray:
        """Load sensor data from file."""
        if sensor_type == SensorType.IMAGE:
            # Load image file
            return cv2.imread(file_path)
        else:
            # Load numpy array
            return np.load(file_path)
    
    def _generate_synthetic_data(self, sensor_type: SensorType) -> np.ndarray:
        """Generate synthetic sensor data based on case parameters."""
        if sensor_type == SensorType.IMAGE:
            # Generate synthetic tympanic membrane image
            # Base color depends on diagnosis and severity
            base_color = np.zeros((512, 512, 3), dtype=np.uint8)
            
            if self.diagnosis == "Normal":
                # Normal - light pink/yellow
                base_color[:, :, 0] = 235  # B
                base_color[:, :, 1] = 215  # G
                base_color[:, :, 2] = 190  # R
            elif self.diagnosis == "OME":
                # OME - amber/yellow with blue tint for fluid
                base_color[:, :, 0] = 180 + int(40 * self.severity)  # B
                base_color[:, :, 1] = 200 - int(30 * self.severity)  # G
                base_color[:, :, 2] = 180 - int(20 * self.severity)  # R
            elif self.diagnosis == "AOM":
                # AOM - red/inflamed
                base_color[:, :, 0] = 90 - int(40 * self.severity)  # B
                base_color[:, :, 1] = 90 - int(50 * self.severity)  # G
                base_color[:, :, 2] = 180 + int(70 * self.severity)  # R
            elif self.diagnosis == "CSOM":
                # CSOM - darker red with white/yellow areas
                base_color[:, :, 0] = 60  # B
                base_color[:, :, 1] = 70  # G
                base_color[:, :, 2] = 150 + int(50 * self.severity)  # R
            
            # Create circular tympanic membrane
            center = (256, 256)
            radius = 200
            mask = np.zeros((512, 512), dtype=np.uint8)
            cv2.circle(mask, center, radius, 255, -1)
            
            # Add noise and texture
            noise = np.random.normal(0, 10, (512, 512, 3)).astype(np.int32)
            image = np.clip(base_color + noise, 0, 255).astype(np.uint8)
            
            # Apply mask
            for c in range(3):
                image[:, :, c] = cv2.bitwise_and(image[:, :, c], mask)
            
            # Add features based on diagnosis
            if self.diagnosis == "OME" or self.diagnosis == "AOM":
                # Add reflection of light (bright spot)
                reflection_center = (center[0] + 50, center[1] - 40)
                reflection_radius = 20
                cv2.circle(image, reflection_center, reflection_radius, (255, 255, 255), -1)
                cv2.GaussianBlur(image, (15, 15), 5, image)
                
                # Add malleus handle
                malleus_start = (center[0], center[1] - 100)
                malleus_end = (center[0], center[1] + 20)
                cv2.line(image, malleus_start, malleus_end, (200, 200, 200), 5)
                
                # Add fluid level if has effusion
                if self.has_effusion:
                    fluid_level = center[1] + int(100 * (self.severity - 0.5))
                    fluid_color = (180, 180, 110) if self.effusion_type == "Serous" else (160, 140, 120)
                    fluid_region = np.zeros_like(image)
                    cv2.circle(fluid_region, center, radius, (255, 255, 255), -1)
                    fluid_region[0:fluid_level, :, :] = 0
                    fluid_mask = fluid_region[:, :, 0] > 0
                    
                    # Apply fluid color
                    for c in range(3):
                        channel = image[:, :, c]
                        channel[fluid_mask] = (channel[fluid_mask] * 0.7 + fluid_color[c] * 0.3).astype(np.uint8)
                        image[:, :, c] = channel
            
            # Save the generated image
            os.makedirs(os.path.dirname(self._get_sensor_file_path(sensor_type)), exist_ok=True)
            cv2.imwrite(self._get_sensor_file_path(sensor_type), image)
            
            return image
            
        elif sensor_type == SensorType.VIBRATION:
            # Generate synthetic vibration data
            # Lower mobility for OME/effusion
            sample_rate = 1000  # Hz
            duration = 0.5  # seconds
            n_samples = int(sample_rate * duration)
            
            # Base signal (sine wave at 226Hz - standard tympanometry probe tone)
            t = np.linspace(0, duration, n_samples)
            freq = 226.0  # Hz
            
            # Amplitude depends on diagnosis and severity
            if self.diagnosis == "Normal":
                amplitude = 1.0
            elif self.has_effusion:
                # Reduced mobility with effusion
                amplitude = 0.4 - 0.3 * self.severity
            else:
                amplitude = 0.8 - 0.4 * self.severity
            
            # Add phase based on pressure
            if self.diagnosis == "OME":
                phase_shift = np.pi * 0.3
            else:
                phase_shift = 0
            
            # Generate signal
            signal = amplitude * np.sin(2 * np.pi * freq * t + phase_shift)
            
            # Add noise
            noise_level = 0.05 + 0.1 * self.severity
            noise = np.random.normal(0, noise_level, n_samples)
            signal = signal + noise
            
            # Reshape to match expected format (time, channels)
            vibration_data = np.column_stack((signal, np.gradient(signal), np.gradient(np.gradient(signal))))
            
            # Save the generated data
            os.makedirs(os.path.dirname(self._get_sensor_file_path(sensor_type)), exist_ok=True)
            np.save(self._get_sensor_file_path(sensor_type), vibration_data)
            
            return vibration_data
            
        elif sensor_type == SensorType.TEMPERATURE:
            # Generate synthetic temperature data
            # Higher temperature for inflammation (AOM)
            if self.diagnosis == "AOM":
                temp = 37.8 + 0.7 * self.severity
            elif self.diagnosis == "CSOM":
                temp = 37.5 + 0.5 * self.severity
            elif self.diagnosis == "OME":
                temp = 37.0 + 0.3 * self.severity
            else:
                temp = 36.9 + 0.1 * np.random.random()
            
            # Ambient temperature
            ambient_temp = 24.0 + 2.0 * np.random.random()
            
            # Temperature data as [temp, ambient_temp]
            temp_data = np.array([temp, ambient_temp])
            
            # Save the generated data
            os.makedirs(os.path.dirname(self._get_sensor_file_path(sensor_type)), exist_ok=True)
            np.save(self._get_sensor_file_path(sensor_type), temp_data)
            
            return temp_data
            
        elif sensor_type == SensorType.PRESSURE:
            # Generate synthetic pressure data (tympanometry curve)
            pressure_range = np.linspace(-400, 400, 200)  # daPa
            
            # Compliance curve depends on diagnosis
            if self.diagnosis == "Normal":
                # Normal curve - peak near 0 daPa
                peak_pressure = np.random.normal(-10, 20)
                width = 80
                height = 1.0
            elif self.diagnosis == "OME":
                # Flat curve for effusion
                peak_pressure = np.random.normal(-150, 50)
                width = 100 + self.severity * 50
                height = 0.3 - 0.2 * self.severity
            elif self.diagnosis == "AOM":
                # Negative pressure for AOM
                peak_pressure = np.random.normal(-200, 40)
                width = 120
                height = 0.5 - 0.3 * self.severity
            else:
                # CSOM - very flat
                peak_pressure = np.random.normal(-100, 80)
                width = 150
                height = 0.2
            
            # Generate bell curve
            compliance = height * np.exp(-((pressure_range - peak_pressure) / width) ** 2)
            
            # Add noise
            noise = np.random.normal(0, 0.02, len(pressure_range))
            compliance = compliance + noise
            compliance = np.clip(compliance, 0, 1)
            
            # Combine pressure and compliance
            pressure_data = np.column_stack((pressure_range, compliance))
            
            # Save the generated data
            os.makedirs(os.path.dirname(self._get_sensor_file_path(sensor_type)), exist_ok=True)
            np.save(self._get_sensor_file_path(sensor_type), pressure_data)
            
            return pressure_data
            
        elif sensor_type == SensorType.ACOUSTIC:
            # Generate synthetic acoustic data (DPOAE)
            sample_rate = 44100  # Hz
            duration = 0.2  # seconds
            n_samples = int(sample_rate * duration)
            
            # Generate time array
            t = np.linspace(0, duration, n_samples)
            
            # Frequencies for distortion product (f1, f2)
            f1, f2 = 1000, 1200  # Hz
            
            # DPOAE amplitude depends on diagnosis
            if self.diagnosis == "Normal":
                amplitude = 0.8
            elif self.has_effusion:
                # Reduced response with effusion
                amplitude = 0.2 - 0.15 * self.severity
            else:
                amplitude = 0.5 - 0.3 * self.severity
            
            # Generate primary tones
            tone1 = 0.7 * np.sin(2 * np.pi * f1 * t)
            tone2 = 0.7 * np.sin(2 * np.pi * f2 * t)
            
            # Generate distortion product at 2*f1-f2
            dp_freq = 2 * f1 - f2
            distortion = amplitude * np.sin(2 * np.pi * dp_freq * t)
            
            # Add noise
            noise_level = 0.1 + 0.2 * self.severity
            noise = np.random.normal(0, noise_level, n_samples)
            
            # Combine signals
            signal = tone1 + tone2 + distortion + noise
            signal = np.clip(signal, -1, 1)
            
            # Save the generated data
            os.makedirs(os.path.dirname(self._get_sensor_file_path(sensor_type)), exist_ok=True)
            np.save(self._get_sensor_file_path(sensor_type), signal)
            
            return signal
            
        elif sensor_type == SensorType.EEG:
            # Generate synthetic EEG data for pain detection
            sample_rate = 256  # Hz
            duration = 2.0  # seconds
            n_samples = int(sample_rate * duration)
            
            # Generate time array
            t = np.linspace(0, duration, n_samples)
            
            # Base signal (alpha, beta, theta waves)
            alpha_amp = 0.5
            beta_amp = 0.3
            theta_amp = 0.2
            
            # Adjust amplitudes based on pain (primarily in AOM)
            if self.diagnosis == "AOM":
                # Pain increases beta, decreases alpha
                pain_factor = self.severity * 0.8
                alpha_amp = alpha_amp * (1 - pain_factor)
                beta_amp = beta_amp * (1 + pain_factor)
                theta_amp = theta_amp * (1 + pain_factor * 0.5)
            
            # Generate waves
            alpha = alpha_amp * np.sin(2 * np.pi * 10 * t)  # Alpha: 8-13 Hz
            beta = beta_amp * np.sin(2 * np.pi * 20 * t)    # Beta: 14-30 Hz
            theta = theta_amp * np.sin(2 * np.pi * 5 * t)   # Theta: 4-7 Hz
            
            # Add noise
            noise = np.random.normal(0, 0.1, n_samples)
            
            # Combine signals
            signal = alpha + beta + theta + noise
            
            # Save the generated data
            os.makedirs(os.path.dirname(self._get_sensor_file_path(sensor_type)), exist_ok=True)
            np.save(self._get_sensor_file_path(sensor_type), signal)
            
            return signal
        
        # Default case - return empty array
        return np.array([])


class SimulationManager:
    """
    Manages simulation cases and provides an interface to the OtitisMediaAI system.
    """
    
    def __init__(self, config_path: str = None):
        """
        Initialize the simulation manager.
        
        Args:
            config_path: Path to configuration file
        """
        # Set up logging
        self.logger = setup_logger('simulation_manager', 'logs/simulation.log')
        self.logger.info("Initializing Simulation Manager")
        
        # Load configuration
        if config_path and os.path.exists(config_path):
            self.config_manager = ConfigManager(config_path)
            self.config = self.config_manager.get_config()
        else:
            # Default configuration
            self.config = {
                'simulation': {
                    'cases_dir': 'data/simulation',
                    'default_cases': [
                        {
                            'id': 'normal_01',
                            'name': 'Normal Ear',
                            'description': 'Healthy tympanic membrane with good mobility',
                            'diagnosis': 'Normal',
                            'severity': 0.0,
                            'has_effusion': False
                        },
                        {
                            'id': 'ome_mild_01',
                            'name': 'Mild OME',
                            'description': 'Mild otitis media with effusion, serous fluid',
                            'diagnosis': 'OME',
                            'severity': 0.3,
                            'has_effusion': True,
                            'effusion_type': 'Serous'
                        },
                        {
                            'id': 'ome_severe_01',
                            'name': 'Severe OME',
                            'description': 'Severe otitis media with effusion, mucoid fluid',
                            'diagnosis': 'OME',
                            'severity': 0.8,
                            'has_effusion': True,
                            'effusion_type': 'Mucoid'
                        },
                        {
                            'id': 'aom_01',
                            'name': 'Acute Otitis Media',
                            'description': 'Acute otitis media with significant inflammation',
                            'diagnosis': 'AOM',
                            'severity': 0.7,
                            'has_effusion': True,
                            'effusion_type': 'Purulent'
                        },
                        {
                            'id': 'csom_01',
                            'name': 'Chronic Suppurative Otitis Media',
                            'description': 'Chronic infection with tympanic membrane perforation',
                            'diagnosis': 'CSOM',
                            'severity': 0.6,
                            'has_effusion': True,
                            'effusion_type': 'Purulent'
                        }
                    ]
                }
            }
        
        # Load simulation cases
        self.cases = {}
        self._load_default_cases()
        self._discover_additional_cases()
        
        self.logger.info(f"Loaded {len(self.cases)} simulation cases")
    
    def _load_default_cases(self):
        """Load default simulation cases from configuration."""
        for case_info in self.config['simulation']['default_cases']:
            case = SimulationCase(
                case_id=case_info['id'],
                name=case_info['name'],
                description=case_info['description'],
                diagnosis=case_info['diagnosis'],
                severity=case_info['severity'],
                has_effusion=case_info['has_effusion'],
                effusion_type=case_info.get('effusion_type')
            )
            self.cases[case.case_id] = case
    
    def _discover_additional_cases(self):
        """Discover additional simulation cases from the cases directory."""
        cases_dir = self.config['simulation']['cases_dir']
        if not os.path.exists(cases_dir):
            os.makedirs(cases_dir, exist_ok=True)
            return
        
        # Look for case definition files
        for item in os.listdir(cases_dir):
            case_dir = os.path.join(cases_dir, item)
            if os.path.isdir(case_dir):
                case_def_file = os.path.join(case_dir, 'case_definition.json')
                if os.path.exists(case_def_file):
                    try:
                        with open(case_def_file, 'r') as f:
                            case_info = json.load(f)
                        
                        # Create case if it's not already loaded
                        if case_info['id'] not in self.cases:
                            case = SimulationCase(
                                case_id=case_info['id'],
                                name=case_info['name'],
                                description=case_info['description'],
                                diagnosis=case_info['diagnosis'],
                                severity=case_info['severity'],
                                has_effusion=case_info['has_effusion'],
                                effusion_type=case_info.get('effusion_type'),
                                data_dir=case_dir
                            )
                            self.cases[case.case_id] = case
                    except Exception as e:
                        self.logger.error(f"Error loading case from {case_def_file}: {str(e)}")
    
    def get_case(self, case_id: str) -> Optional[SimulationCase]:
        """
        Get a simulation case by ID.
        
        Args:
            case_id: Case identifier
            
        Returns:
            SimulationCase object or None if not found
        """
        return self.cases.get(case_id)
    
    def get_all_cases(self) -> List[SimulationCase]:
        """
        Get all available simulation cases.
        
        Returns:
            List of SimulationCase objects
        """
        return list(self.cases.values())
    
    def create_custom_case(self, 
                         name: str,
                         description: str,
                         diagnosis: str,
                         severity: float,
                         has_effusion: bool,
                         effusion_type: Optional[str] = None) -> SimulationCase:
        """
        Create a custom simulation case.
        
        Args:
            name: Case name
            description: Case description
            diagnosis: Expected diagnosis
            severity: Severity score (0.0-1.0)
            has_effusion: Whether the case has middle ear effusion
            effusion_type: Type of effusion if present
            
        Returns:
            The created SimulationCase object
        """
        # Generate a unique case ID
        timestamp = int(time.time())
        case_id = f"custom_{diagnosis.lower()}_{timestamp}"
        
        # Create the case
        case = SimulationCase(
            case_id=case_id,
            name=name,
            description=description,
            diagnosis=diagnosis,
            severity=severity,
            has_effusion=has_effusion,
            effusion_type=effusion_type
        )
        
        # Add to cases dictionary
        self.cases[case_id] = case
        
        # Save case definition
        case_dir = os.path.join(self.config['simulation']['cases_dir'], case_id)
        os.makedirs(case_dir, exist_ok=True)
        
        case_def = {
            'id': case_id,
            'name': name,
            'description': description,
            'diagnosis': diagnosis,
            'severity': severity,
            'has_effusion': has_effusion,
            'effusion_type': effusion_type
        }
        
        with open(os.path.join(case_dir, 'case_definition.json'), 'w') as f:
            json.dump(case_def, f, indent=2)
        
        self.logger.info(f"Created custom case: {case_id}")
        return case
    
    def run_simulation(self, case_id: str) -> Dict[str, Any]:
        """
        Run a simulation with the specified case and return results.
        
        Args:
            case_id: Case identifier
            
        Returns:
            Dictionary containing simulation results
        """
        case = self.get_case(case_id)
        if not case:
            self.logger.error(f"Case not found: {case_id}")
            return {'error': f"Case not found: {case_id}"}
        
        self.logger.info(f"Running simulation for case: {case.name} ({case_id})")
        
        # Collect sensor data
        sensor_data = {}
        for sensor_type in SensorType:
            try:
                data = case.get_sensor_data(sensor_type)
                sensor_data[sensor_type] = data
            except Exception as e:
                self.logger.error(f"Error generating {sensor_type.name} data: {str(e)}")
        
        # Create expected diagnostic result
        diagnostic_result = DiagnosticResult(
            timestamp=time.time(),
            diagnosis=case.diagnosis,
            confidence=0.95,
            severity=case.severity,
            effusion_type=case.effusion_type if case.has_effusion else None,
            effusion_volume=0.5 * case.severity if case.has_effusion else None,
            inflammation_level=0.8 * case.severity if case.diagnosis == "AOM" else 0.3 * case.severity,
            membrane_mobility=0.9 if case.diagnosis == "Normal" else 0.6 - 0.5 * case.severity,
            pain_level=7.0 * case.severity if case.diagnosis == "AOM" else 2.0 * case.severity,
            additional_findings={}
        )
        
        # Create expected prognosis result
        recovery_probability = 0.9 if case.diagnosis == "Normal" else 0.7 - 0.6 * case.severity
        surgery_recommendation = 0.1 if case.diagnosis == "Normal" else 0.3 + 0.6 * case.severity
        
        prognosis_result = PrognosisResult(
            natural_recovery_probability=recovery_probability,
            surgery_recommendation=surgery_recommendation,
            estimated_recovery_time=0 if case.diagnosis == "Normal" else int(10 + 80 * case.severity),
            treatment_recommendations=["No treatment needed"] if case.diagnosis == "Normal" else ["Antibiotics", "Follow-up in 2 weeks"],
            follow_up_recommendation="Routine check-up in 1 year" if case.diagnosis == "Normal" else "Follow-up in 2 weeks",
            risk_factors=[]
        )
        
        # Return simulation results
        result = {
            'case': {
                'id': case.case_id,
                'name': case.name,
                'description': case.description,
                'diagnosis': case.diagnosis,
                'severity': case.severity,
                'has_effusion': case.has_effusion,
                'effusion_type': case.effusion_type
            },
            'sensor_data': {
                sensor_type.name: data.shape if isinstance(data, np.ndarray) else None
                for sensor_type, data in sensor_data.items()
            },
            'expected_diagnosis': {
                'diagnosis': diagnostic_result.diagnosis,
                'confidence': diagnostic_result.confidence,
                'severity': diagnostic_result.severity,
                'effusion_type': diagnostic_result.effusion_type,
                'membrane_mobility': diagnostic_result.membrane_mobility
            },
            'expected_prognosis': {
                'recovery_probability': prognosis_result.natural_recovery_probability,
                'surgery_recommendation': prognosis_result.surgery_recommendation,
                'recovery_time': prognosis_result.estimated_recovery_time
            },
            'timestamp': time.time()
        }
        
        self.logger.info(f"Simulation completed for case: {case.name}")
        return result


class SimulationUI:
    """
    User interface for the OtitisMediaAI simulation environment.
    """
    
    def __init__(self, simulation_manager: SimulationManager):
        """
        Initialize the simulation UI.
        
        Args:
            simulation_manager: Simulation manager instance
        """
        self.simulation_manager = simulation_manager
        self.logger = logging.getLogger('simulation_ui')
        
        # Create main window
        self.root = tk.Tk()
        self.root.title("OtitisMediaAI Simulation Environment")
        self.root.geometry("1200x800")
        
        # Set up the UI components
        self._setup_ui()
    
    def _setup_ui(self):
        """Set up the user interface components."""
        # Main frame
        self.main_frame = ttk.Frame(self.root, padding="10")
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Title
        ttk.Label(self.main_frame, text="OtitisMediaAI Simulation Environment", 
                 font=("Arial", 16, "bold")).pack(pady=10)
        
        # Split into left and right panels
        self.panel_frame = ttk.Frame(self.main_frame)
        self.panel_frame.pack(fill=tk.BOTH, expand=True)
        
        # Left panel - Case selection
        self.left_panel = ttk.LabelFrame(self.panel_frame, text="Simulation Cases", padding="10")
        self.left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        # Cases listbox
        self.cases_var = tk.StringVar()
        self.cases_listbox = tk.Listbox(self.left_panel, listvariable=self.cases_var, height=10)
        self.cases_listbox.pack(fill=tk.BOTH, expand=True, pady=5)
        self.cases_listbox.bind('<<ListboxSelect>>', self._on_case_selected)
        
        # Populate cases
        self._refresh_cases_list()
        
        # Case details
        self.case_details_frame = ttk.LabelFrame(self.left_panel, text="Case Details", padding="10")
        self.case_details_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Case details labels
        self.case_name_var = tk.StringVar()
        self.case_desc_var = tk.StringVar()
        self.case_diagnosis_var = tk.StringVar()
        self.case_severity_var = tk.StringVar()
        self.case_effusion_var = tk.StringVar()
        
        ttk.Label(self.case_details_frame, text="Name:").grid(row=0, column=0, sticky=tk.W, pady=2)
        ttk.Label(self.case_details_frame, textvariable=self.case_name_var).grid(row=0, column=1, sticky=tk.W, pady=2)
        
        ttk.Label(self.case_details_frame, text="Description:").grid(row=1, column=0, sticky=tk.W, pady=2)
        ttk.Label(self.case_details_frame, textvariable=self.case_desc_var).grid(row=1, column=1, sticky=tk.W, pady=2)
        
        ttk.Label(self.case_details_frame, text="Diagnosis:").grid(row=2, column=0, sticky=tk.W, pady=2)
        ttk.Label(self.case_details_frame, textvariable=self.case_diagnosis_var).grid(row=2, column=1, sticky=tk.W, pady=2)
        
        ttk.Label(self.case_details_frame, text="Severity:").grid(row=3, column=0, sticky=tk.W, pady=2)
        ttk.Label(self.case_details_frame, textvariable=self.case_severity_var).grid(row=3, column=1, sticky=tk.W, pady=2)
        
        ttk.Label(self.case_details_frame, text="Effusion:").grid(row=4, column=0, sticky=tk.W, pady=2)
        ttk.Label(self.case_details_frame, textvariable=self.case_effusion_var).grid(row=4, column=1, sticky=tk.W, pady=2)
        
        # Run simulation button
        self.run_button = ttk.Button(self.left_panel, text="Run Simulation", command=self._run_simulation)
        self.run_button.pack(fill=tk.X, pady=10)
        
        # Create custom case button
        self.create_case_button = ttk.Button(self.left_panel, text="Create Custom Case", command=self._create_custom_case)
        self.create_case_button.pack(fill=tk.X, pady=5)
        
        # Right panel - Simulation results
        self.right_panel = ttk.LabelFrame(self.panel_frame, text="Simulation Results", padding="10")
        self.right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))
        
        # Notebook for different result views
        self.result_notebook = ttk.Notebook(self.right_panel)
        self.result_notebook.pack(fill=tk.BOTH, expand=True)
        
        # Image tab
        self.image_tab = ttk.Frame(self.result_notebook)
        self.result_notebook.add(self.image_tab, text="Tympanic Image")
        
        # Image display
        self.image_label = ttk.Label(self.image_tab)
        self.image_label.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Sensor Data tab
        self.sensor_tab = ttk.Frame(self.result_notebook)
        self.result_notebook.add(self.sensor_tab, text="Sensor Data")
        
        # Sensor data plots
        self.fig, self.axes = plt.subplots(2, 2, figsize=(8, 6))
        self.fig.tight_layout(pad=3.0)
        
        # Canvas for matplotlib figure
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.sensor_tab)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Results tab
        self.results_tab = ttk.Frame(self.result_notebook)
        self.result_notebook.add(self.results_tab, text="Diagnosis Results")
        
        # Results display
        self.results_frame = ttk.Frame(self.results_tab, padding="10")
        self.results_frame.pack(fill=tk.BOTH, expand=True)
        
        # Results text
        self.results_text = tk.Text(self.results_frame, height=20, width=50)
        self.results_text.pack(fill=tk.BOTH, expand=True)
        
        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        self.status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
    
    def _refresh_cases_list(self):
        """Refresh the list of available cases."""
        cases = self.simulation_manager.get_all_cases()
        self.cases_var.set([case.name for case in cases])
        self.cases = cases  # Store for reference
    
    def _on_case_selected(self, event):
        """Handle case selection event."""
        selection = self.cases_listbox.curselection()
        if not selection:
            return
        
        index = selection[0]
        if index < len(self.cases):
            case = self.cases[index]
            self._display_case_details(case)
    
    def _display_case_details(self, case: SimulationCase):
        """Display details of the selected case."""
        self.case_name_var.set(case.name)
        self.case_desc_var.set(case.description)
        self.case_diagnosis_var.set(case.diagnosis)
        self.case_severity_var.set(f"{case.severity:.2f}")
        
        if case.has_effusion:
            effusion_text = f"Yes ({case.effusion_type})" if case.effusion_type else "Yes"
        else:
            effusion_text = "No"
        
        self.case_effusion_var.set(effusion_text)
        
        # Also try to display the image
        try:
            image_data = case.get_sensor_data(SensorType.IMAGE)
            self._display_image(image_data)
        except Exception as e:
            self.logger.error(f"Error displaying image: {str(e)}")
    
    def _display_image(self, image_data: np.ndarray):
        """Display an image in the UI."""
        if image_data is None:
            return
        
        # Convert from BGR to RGB (if needed)
        if len(image_data.shape) == 3 and image_data.shape[2] == 3:
            image_data = cv2.cvtColor(image_data, cv2.COLOR_BGR2RGB)
        
        # Resize if needed
        max_size = 400
        h, w = image_data.shape[:2]
        if h > max_size or w > max_size:
            scale = max_size / max(h, w)
            new_size = (int(w * scale), int(h * scale))
            image_data = cv2.resize(image_data, new_size)
        
        # Convert to PIL image and display
        pil_image = Image.fromarray(image_data)
        tk_image = ImageTk.PhotoImage(pil_image)
        
        # Keep a reference to prevent garbage collection
        self.current_image = tk_image
        
        # Display in label
        self.image_label.configure(image=tk_image)
    
    def _run_simulation(self):
        """Run the simulation for the selected case."""
        selection = self.cases_listbox.curselection()
        if not selection:
            self.status_var.set("Error: No case selected")
            return
        
        index = selection[0]
        if index < len(self.cases):
            case = self.cases[index]
            self.status_var.set(f"Running simulation for {case.name}...")
            
            # Run in a separate thread to avoid freezing the UI
            threading.Thread(target=self._run_simulation_thread, args=(case,)).start()
    
    def _run_simulation_thread(self, case: SimulationCase):
        """Run simulation in a separate thread."""
        try:
            # Run the simulation
            result = self.simulation_manager.run_simulation(case.case_id)
            
            # Update UI with results
            self.root.after(0, lambda: self._update_results(case, result))
        except Exception as e:
            self.logger.error(f"Error in simulation: {str(e)}")
            self.root.after(0, lambda: self.status_var.set(f"Error: {str(e)}"))
    
    def _update_results(self, case: SimulationCase, result: Dict[str, Any]):
        """Update UI with simulation results."""
        # Display the tympanic image
        try:
            image_data = case.get_sensor_data(SensorType.IMAGE)
            self._display_image(image_data)
        except Exception as e:
            self.logger.error(f"Error displaying image: {str(e)}")
        
        # Plot sensor data
        self._plot_sensor_data(case)
        
        # Display diagnostic results
        self._display_results(result)
        
        # Update status
        self.status_var.set(f"Simulation completed for {case.name}")
    
    def _plot_sensor_data(self, case: SimulationCase):
        """Plot sensor data in the UI."""
        # Clear previous plots
        for ax in self.axes.flatten():
            ax.clear()
        
        # Plot vibration data
        try:
            vibration_data = case.get_sensor_data(SensorType.VIBRATION)
            if vibration_data is not None and len(vibration_data) > 0:
                self.axes[0, 0].plot(vibration_data[:, 0])
                self.axes[0, 0].set_title("Vibration Data")
                self.axes[0, 0].set_xlabel("Time (ms)")
                self.axes[0, 0].set_ylabel("Amplitude")
        except Exception as e:
            self.logger.error(f"Error plotting vibration data: {str(e)}")
        
        # Plot pressure data
        try:
            pressure_data = case.get_sensor_data(SensorType.PRESSURE)
            if pressure_data is not None and len(pressure_data) > 0:
                self.axes[0, 1].plot(pressure_data[:, 0], pressure_data[:, 1])
                self.axes[0, 1].set_title("Tympanometry")
                self.axes[0, 1].set_xlabel("Pressure (daPa)")
                self.axes[0, 1].set_ylabel("Compliance")
        except Exception as e:
            self.logger.error(f"Error plotting pressure data: {str(e)}")
        
        # Plot acoustic data
        try:
            acoustic_data = case.get_sensor_data(SensorType.ACOUSTIC)
            if acoustic_data is not None and len(acoustic_data) > 0:
                if len(acoustic_data.shape) == 1:
                    # Time domain
                    self.axes[1, 0].plot(acoustic_data)
                    self.axes[1, 0].set_title("Acoustic Data")
                    self.axes[1, 0].set_xlabel("Time (samples)")
                    self.axes[1, 0].set_ylabel("Amplitude")
                else:
                    # Multiple channels
                    for i in range(min(acoustic_data.shape[1], 3)):
                        self.axes[1, 0].plot(acoustic_data[:, i])
                    self.axes[1, 0].set_title("Acoustic Data")
                    self.axes[1, 0].set_xlabel("Time (samples)")
                    self.axes[1, 0].set_ylabel("Amplitude")
        except Exception as e:
            self.logger.error(f"Error plotting acoustic data: {str(e)}")
        
        # Plot EEG data
        try:
            eeg_data = case.get_sensor_data(SensorType.EEG)
            if eeg_data is not None and len(eeg_data) > 0:
                if len(eeg_data.shape) == 1:
                    # Single channel
                    self.axes[1, 1].plot(eeg_data)
                    self.axes[1, 1].set_title("EEG Data")
                    self.axes[1, 1].set_xlabel("Time (samples)")
                    self.axes[1, 1].set_ylabel("Amplitude")
                else:
                    # Multiple channels
                    for i in range(min(eeg_data.shape[1], 3)):
                        self.axes[1, 1].plot(eeg_data[:, i])
                    self.axes[1, 1].set_title("EEG Data")
                    self.axes[1, 1].set_xlabel("Time (samples)")
                    self.axes[1, 1].set_ylabel("Amplitude")
        except Exception as e:
            self.logger.error(f"Error plotting EEG data: {str(e)}")
        
        self.fig.tight_layout()
        self.canvas.draw()
    
    def _display_results(self, result: Dict[str, Any]):
        """Display diagnostic results in the UI."""
        # Clear previous results
        self.results_text.delete(1.0, tk.END)
        
        # Add result information
        self.results_text.insert(tk.END, "SIMULATION RESULTS\n")
        self.results_text.insert(tk.END, "=================\n\n")
        
        # Case info
        self.results_text.insert(tk.END, f"Case: {result['case']['name']}\n")
        self.results_text.insert(tk.END, f"Description: {result['case']['description']}\n\n")
        
        # Expected diagnosis
        self.results_text.insert(tk.END, "Expected Diagnosis:\n")
        self.results_text.insert(tk.END, f"- Diagnosis: {result['expected_diagnosis']['diagnosis']}\n")
        self.results_text.insert(tk.END, f"- Confidence: {result['expected_diagnosis']['confidence']:.2f}\n")
        self.results_text.insert(tk.END, f"- Severity: {result['expected_diagnosis']['severity']:.2f}\n")
        
        if result['expected_diagnosis']['effusion_type']:
            self.results_text.insert(tk.END, f"- Effusion Type: {result['expected_diagnosis']['effusion_type']}\n")
        
        self.results_text.insert(tk.END, f"- Membrane Mobility: {result['expected_diagnosis']['membrane_mobility']:.2f}\n\n")
        
        # Expected prognosis
        self.results_text.insert(tk.END, "Expected Prognosis:\n")
        self.results_text.insert(tk.END, f"- Natural Recovery Probability: {result['expected_prognosis']['recovery_probability']:.2f}\n")
        self.results_text.insert(tk.END, f"- Surgery Recommendation: {result['expected_prognosis']['surgery_recommendation']:.2f}\n")
        
        if result['expected_prognosis']['recovery_time'] > 0:
            self.results_text.insert(tk.END, f"- Estimated Recovery Time: {result['expected_prognosis']['recovery_time']} days\n\n")
        else:
            self.results_text.insert(tk.END, "- Estimated Recovery Time: Not applicable (healthy)\n\n")
        
        # Sensor data summary
        self.results_text.insert(tk.END, "Sensor Data Summary:\n")
        for sensor_name, shape in result['sensor_data'].items():
            if shape:
                self.results_text.insert(tk.END, f"- {sensor_name}: Data shape {shape}\n")
            else:
                self.results_text.insert(tk.END, f"- {sensor_name}: No data\n")
    
    def _create_custom_case(self):
        """Open dialog to create a custom simulation case."""
        # Create a new window
        custom_window = tk.Toplevel(self.root)
        custom_window.title("Create Custom Case")
        custom_window.geometry("400x500")
        custom_window.grab_set()  # Make modal
        
        # Form
        form_frame = ttk.Frame(custom_window, padding="20")
        form_frame.pack(fill=tk.BOTH, expand=True)
        
        # Fields
        ttk.Label(form_frame, text="Name:").grid(row=0, column=0, sticky=tk.W, pady=5)
        name_entry = ttk.Entry(form_frame, width=30)
        name_entry.grid(row=0, column=1, pady=5)
        
        ttk.Label(form_frame, text="Description:").grid(row=1, column=0, sticky=tk.W, pady=5)
        desc_entry = ttk.Entry(form_frame, width=30)
        desc_entry.grid(row=1, column=1, pady=5)
        
        ttk.Label(form_frame, text="Diagnosis:").grid(row=2, column=0, sticky=tk.W, pady=5)
        diagnosis_var = tk.StringVar()
        diagnosis_combo = ttk.Combobox(form_frame, textvariable=diagnosis_var, 
                                     values=["Normal", "OME", "AOM", "CSOM"])
        diagnosis_combo.grid(row=2, column=1, pady=5)
        diagnosis_combo.current(0)
        
        ttk.Label(form_frame, text="Severity (0.0-1.0):").grid(row=3, column=0, sticky=tk.W, pady=5)
        severity_var = tk.DoubleVar(value=0.5)
        severity_scale = ttk.Scale(form_frame, from_=0.0, to=1.0, variable=severity_var, 
                                 orient=tk.HORIZONTAL, length=200)
        severity_scale.grid(row=3, column=1, pady=5)
        
        # Add a label to show the severity value
        severity_label = ttk.Label(form_frame, text="0.50")
        severity_label.grid(row=3, column=2, pady=5)
        
        # Update severity label when scale changes
        def update_severity_label(*args):
            severity_label.config(text=f"{severity_var.get():.2f}")
        
        severity_var.trace_add("write", update_severity_label)
        
        ttk.Label(form_frame, text="Has Effusion:").grid(row=4, column=0, sticky=tk.W, pady=5)
        has_effusion_var = tk.BooleanVar(value=False)
        has_effusion_check = ttk.Checkbutton(form_frame, variable=has_effusion_var)
        has_effusion_check.grid(row=4, column=1, sticky=tk.W, pady=5)
        
        ttk.Label(form_frame, text="Effusion Type:").grid(row=5, column=0, sticky=tk.W, pady=5)
        effusion_type_var = tk.StringVar()
        effusion_type_combo = ttk.Combobox(form_frame, textvariable=effusion_type_var,
                                         values=["", "Serous", "Mucoid", "Purulent"])
        effusion_type_combo.grid(row=5, column=1, pady=5)
        effusion_type_combo.current(0)
        
        # Image preview
        ttk.Label(form_frame, text="Image Preview:").grid(row=6, column=0, sticky=tk.W, pady=5)
        preview_frame = ttk.Frame(form_frame, borderwidth=1, relief=tk.SUNKEN, width=200, height=200)
        preview_frame.grid(row=6, column=1, columnspan=2, pady=10)
        
        preview_label = ttk.Label(preview_frame)
        preview_label.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Function to update the preview image
        def update_preview():
            # Create a temporary case to generate the image
            temp_case = SimulationCase(
                case_id="temp",
                name=name_entry.get() or "Custom Case",
                description=desc_entry.get() or "Custom case description",
                diagnosis=diagnosis_var.get(),
                severity=severity_var.get(),
                has_effusion=has_effusion_var.get(),
                effusion_type=effusion_type_var.get() if has_effusion_var.get() else None
            )
            
            # Generate and display the image
            try:
                image_data = temp_case._generate_synthetic_data(SensorType.IMAGE)
                
                # Convert from BGR to RGB
                image_data = cv2.cvtColor(image_data, cv2.COLOR_BGR2RGB)
                
                # Resize for preview
                h, w = image_data.shape[:2]
                scale = 180 / max(h, w)
                new_size = (int(w * scale), int(h * scale))
                resized = cv2.resize(image_data, new_size)
                
                # Convert to PIL image
                pil_image = Image.fromarray(resized)
                tk_image = ImageTk.PhotoImage(pil_image)
                
                # Keep reference and display
                preview_label.image = tk_image
                preview_label.configure(image=tk_image)
            except Exception as e:
                self.logger.error(f"Error generating preview: {str(e)}")
        
        # Update preview when parameters change
        update_button = ttk.Button(form_frame, text="Update Preview", command=update_preview)
        update_button.grid(row=7, column=1, pady=10)
        
        # Create case button
        def on_create():
            try:
                case = self.simulation_manager.create_custom_case(
                    name=name_entry.get() or "Custom Case",
                    description=desc_entry.get() or "Custom case description",
                    diagnosis=diagnosis_var.get(),
                    severity=severity_var.get(),
                    has_effusion=has_effusion_var.get(),
                    effusion_type=effusion_type_var.get() if has_effusion_var.get() else None
                )
                
                # Refresh case list
                self._refresh_cases_list()
                
                # Select the new case
                case_index = self.cases.index(case)
                self.cases_listbox.selection_clear(0, tk.END)
                self.cases_listbox.selection_set(case_index)
                self.cases_listbox.see(case_index)
                self._display_case_details(case)
                
                # Close the dialog
                custom_window.destroy()
                
                # Update status
                self.status_var.set(f"Created custom case: {case.name}")
            except Exception as e:
                self.logger.error(f"Error creating custom case: {str(e)}")
                tk.messagebox.showerror("Error", f"Failed to create case: {str(e)}")
        
        create_button = ttk.Button(form_frame, text="Create Case", command=on_create)
        create_button.grid(row=8, column=0, columnspan=3, pady=10)
        
        # Initial preview
        update_preview()
    
    def run(self):
        """Run the UI main loop."""
        self.root.mainloop()


def main():
    """Main entry point for the simulation environment."""
    parser = argparse.ArgumentParser(description='OtitisMediaAI Simulation Environment')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to configuration file')
    parser.add_argument('--case', type=str, default=None,
                        help='Case ID to run automatically')
    parser.add_argument('--headless', action='store_true',
                        help='Run in headless mode without UI')
    parser.add_argument('--output', type=str, default=None,
                        help='Output JSON file for results (headless mode only)')
    
    args = parser.parse_args()
    
    # Initialize simulation manager
    sm = SimulationManager(args.config)
    
    # Run specific case if requested
    if args.case:
        result = sm.run_simulation(args.case)
        
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(result, f, indent=2)
        else:
            print(json.dumps(result, indent=2))
        
        return
    
    # Run UI if not headless
    if not args.headless:
        ui = SimulationUI(sm)
        ui.run()
    else:
        print("Headless mode activated. Use --case to specify a case ID to run.")


if __name__ == "__main__":
    main()
