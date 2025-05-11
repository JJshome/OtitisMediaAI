"""
Data structure definitions for the OtitisMediaAI system.
"""

import enum
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np


class SensorType(enum.Enum):
    """Enumeration of sensor types in the multi-sensor probe."""
    IMAGE = 1       # High-resolution camera for tympanic membrane imaging
    VIBRATION = 2   # MEMS accelerometer for membrane mobility assessment
    TEMPERATURE = 3 # Infrared temperature sensor for inflammation detection
    PRESSURE = 4    # MEMS pressure sensor for middle ear pressure
    ACOUSTIC = 5    # Microphone for acoustic emission measurements
    EEG = 6         # EEG sensor for pain pattern detection


@dataclass
class SensorData:
    """Base class for all sensor data."""
    timestamp: float  # Timestamp in seconds since epoch
    sensor_type: SensorType
    raw_data: Any
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ImageData(SensorData):
    """Data from the high-resolution imaging sensor."""
    image: np.ndarray  # Image data as numpy array (height, width, channels)
    resolution: Tuple[int, int]  # Resolution (width, height)
    format: str = "RGB"  # Image format (RGB, BGR, etc.)
    focus_quality: float = 1.0  # Focus quality metric (0.0-1.0)


@dataclass
class VibrationData(SensorData):
    """Data from the vibration sensor."""
    displacement: np.ndarray  # Membrane displacement measurements
    velocity: np.ndarray  # Velocity measurements
    frequency_response: Optional[np.ndarray] = None  # Frequency domain response
    sampling_rate: int = 1000  # Sampling rate in Hz


@dataclass
class TemperatureData(SensorData):
    """Data from the temperature sensor."""
    temperature: float  # Temperature in degrees Celsius
    ambient_temperature: float  # Ambient temperature in degrees Celsius
    temperature_map: Optional[np.ndarray] = None  # Spatial temperature distribution if available


@dataclass
class PressureData(SensorData):
    """Data from the pressure sensor."""
    pressure: float  # Pressure in daPa
    compliance: Optional[float] = None  # Compliance measurement if available
    pressure_changes: Optional[np.ndarray] = None  # Time-series of pressure changes


@dataclass
class AcousticData(SensorData):
    """Data from the acoustic emission sensor."""
    waveform: np.ndarray  # Time-domain acoustic waveform
    spectrum: Optional[np.ndarray] = None  # Frequency-domain spectrum
    sampling_rate: int = 44100  # Sampling rate in Hz
    dpoae_amplitude: Optional[float] = None  # DPOAE amplitude in dB SPL if measured


@dataclass
class EEGData(SensorData):
    """Data from the EEG sensor."""
    signals: np.ndarray  # EEG signals as numpy array (channels, samples)
    channels: List[str]  # Channel names
    sampling_rate: int = 256  # Sampling rate in Hz
    pain_score: Optional[float] = None  # Pain score (0-10) if calculated


@dataclass
class ProbeStatus:
    """Status information about the multi-sensor probe."""
    position: Tuple[float, float, float]  # 3D position (x, y, z)
    orientation: Tuple[float, float, float]  # Orientation (pitch, yaw, roll)
    insertion_depth: float  # Insertion depth in mm
    insertion_angle: float  # Insertion angle in degrees
    battery_level: float  # Battery level (0.0-1.0)
    temperature: float  # Internal temperature of the probe
    errors: List[str] = field(default_factory=list)  # List of error messages if any


@dataclass
class DiagnosticResult:
    """Diagnostic result from the AI analysis."""
    timestamp: float  # Timestamp of the diagnosis
    diagnosis: str  # Primary diagnosis (e.g., "OME", "Normal", "AOM")
    confidence: float  # Confidence level (0.0-1.0)
    severity: float  # Severity score (0.0-1.0)
    effusion_type: Optional[str] = None  # Type of effusion if present
    effusion_volume: Optional[float] = None  # Estimated volume in ml if present
    inflammation_level: Optional[float] = None  # Inflammation level (0.0-1.0)
    membrane_mobility: Optional[float] = None  # Membrane mobility (0.0-1.0)
    pain_level: Optional[float] = None  # Estimated pain level (0-10)
    additional_findings: Dict[str, Any] = field(default_factory=dict)  # Additional diagnostic findings


@dataclass
class PrognosisResult:
    """Prognosis prediction for the diagnosed condition."""
    natural_recovery_probability: float  # Probability of natural recovery (0.0-1.0)
    surgery_recommendation: float  # Recommendation score for surgery (0.0-1.0)
    estimated_recovery_time: Optional[int] = None  # Estimated recovery time in days
    treatment_recommendations: List[str] = field(default_factory=list)  # Recommended treatments
    follow_up_recommendation: Optional[str] = None  # Recommended follow-up schedule
    risk_factors: List[str] = field(default_factory=list)  # Identified risk factors
