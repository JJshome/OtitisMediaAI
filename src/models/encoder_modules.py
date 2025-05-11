"""
Sensor-specific encoder modules for the multi-sensor transformer model.
"""

import logging
import numpy as np
import tensorflow as tf
from typing import Dict, List, Optional, Tuple, Union, Any, Type
import cv2

from ..utils.data_structures import SensorType


class BaseEncoder(tf.keras.layers.Layer):
    """Base class for all sensor-specific encoders."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the encoder with the provided configuration.
        
        Args:
            config: Encoder configuration parameters
        """
        super(BaseEncoder, self).__init__()
        self.logger = logging.getLogger(f'otitismedia_ai.{self.__class__.__name__}')
        self.config = config
        self.input_shape = self._get_input_shape()
        self.output_dim = config.get('output_dim', 256)
        
        self._build_layers()
        self.logger.info(f"Initialized {self.__class__.__name__} with output_dim={self.output_dim}")
    
    def _get_input_shape(self) -> Tuple[int, ...]:
        """Get the expected input shape for this encoder."""
        raise NotImplementedError("Subclasses must implement _get_input_shape")
    
    def _build_layers(self):
        """Build the encoder layers."""
        raise NotImplementedError("Subclasses must implement _build_layers")
    
    def preprocess(self, data: Any) -> np.ndarray:
        """
        Preprocess the sensor data before encoding.
        
        Args:
            data: Raw sensor data
            
        Returns:
            Preprocessed data as numpy array
        """
        raise NotImplementedError("Subclasses must implement preprocess")
    
    def call(self, inputs):
        """
        Forward pass for the encoder.
        
        Args:
            inputs: Input tensor
            
        Returns:
            Encoded output tensor
        """
        raise NotImplementedError("Subclasses must implement call")


class ImageEncoder(BaseEncoder):
    """Encoder for tympanic membrane images."""
    
    def _get_input_shape(self) -> Tuple[int, ...]:
        """Get the expected input shape for image data."""
        return (
            self.config.get('image_height', 224),
            self.config.get('image_width', 224),
            self.config.get('image_channels', 3)
        )
    
    def _build_layers(self):
        """Build CNN-based encoder layers for image data."""
        base_filters = self.config.get('base_filters', 64)
        dropout_rate = self.config.get('dropout_rate', 0.2)
        
        # Use a pre-trained model as the base if specified
        pretrained_base = self.config.get('pretrained_base', 'resnet50')
        if pretrained_base:
            if pretrained_base == 'resnet50':
                self.base_model = tf.keras.applications.ResNet50(
                    include_top=False,
                    weights='imagenet',
                    input_shape=self.input_shape
                )
            elif pretrained_base == 'efficientnet':
                self.base_model = tf.keras.applications.EfficientNetB0(
                    include_top=False,
                    weights='imagenet',
                    input_shape=self.input_shape
                )
            elif pretrained_base == 'mobilenet':
                self.base_model = tf.keras.applications.MobileNetV2(
                    include_top=False,
                    weights='imagenet',
                    input_shape=self.input_shape
                )
            else:
                self.logger.warning(f"Unknown pretrained base {pretrained_base}, using ResNet50")
                self.base_model = tf.keras.applications.ResNet50(
                    include_top=False,
                    weights='imagenet',
                    input_shape=self.input_shape
                )
            
            # Freeze base model if specified
            if self.config.get('freeze_base', True):
                self.base_model.trainable = False
        else:
            # Build a custom CNN if no pretrained model is specified
            self.conv1 = tf.keras.layers.Conv2D(base_filters, 7, strides=2, padding='same', activation='relu')
            self.bn1 = tf.keras.layers.BatchNormalization()
            self.pool1 = tf.keras.layers.MaxPooling2D(3, strides=2, padding='same')
            
            self.conv2 = tf.keras.layers.Conv2D(base_filters*2, 3, padding='same', activation='relu')
            self.bn2 = tf.keras.layers.BatchNormalization()
            self.pool2 = tf.keras.layers.MaxPooling2D(2, strides=2, padding='same')
            
            self.conv3 = tf.keras.layers.Conv2D(base_filters*4, 3, padding='same', activation='relu')
            self.bn3 = tf.keras.layers.BatchNormalization()
            self.pool3 = tf.keras.layers.MaxPooling2D(2, strides=2, padding='same')
            
            self.conv4 = tf.keras.layers.Conv2D(base_filters*8, 3, padding='same', activation='relu')
            self.bn4 = tf.keras.layers.BatchNormalization()
            self.pool4 = tf.keras.layers.MaxPooling2D(2, strides=2, padding='same')
        
        # Common layers for all variants
        self.global_pool = tf.keras.layers.GlobalAveragePooling2D()
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        self.dense1 = tf.keras.layers.Dense(self.output_dim, activation='relu')
        self.reshape = tf.keras.layers.Reshape((1, self.output_dim))
    
    def preprocess(self, data: np.ndarray) -> np.ndarray:
        """
        Preprocess the image data before encoding.
        
        Args:
            data: Raw image data (height, width, channels)
            
        Returns:
            Preprocessed image data
        """
        # Ensure data is in the correct format
        if len(data.shape) == 3:  # Single image
            data = np.expand_dims(data, axis=0)
        
        # Resize to the expected input shape
        resized_images = []
        for img in data:
            target_h, target_w, _ = self.input_shape
            if img.shape[0] != target_h or img.shape[1] != target_w:
                resized = cv2.resize(img, (target_w, target_h))
                resized_images.append(resized)
            else:
                resized_images.append(img)
        
        processed = np.array(resized_images)
        
        # Apply normalization based on the pretrained model if used
        pretrained_base = self.config.get('pretrained_base', 'resnet50')
        if pretrained_base == 'resnet50':
            processed = tf.keras.applications.resnet50.preprocess_input(processed)
        elif pretrained_base == 'efficientnet':
            processed = tf.keras.applications.efficientnet.preprocess_input(processed)
        elif pretrained_base == 'mobilenet':
            processed = tf.keras.applications.mobilenet_v2.preprocess_input(processed)
        else:
            # Simple normalization if no specific preprocessing
            processed = processed / 255.0
        
        return processed
    
    def call(self, inputs):
        """
        Forward pass for the image encoder.
        
        Args:
            inputs: Input image tensor
            
        Returns:
            Encoded output tensor
        """
        pretrained_base = self.config.get('pretrained_base', 'resnet50')
        
        if pretrained_base:
            x = self.base_model(inputs)
        else:
            x = self.conv1(inputs)
            x = self.bn1(x)
            x = self.pool1(x)
            
            x = self.conv2(x)
            x = self.bn2(x)
            x = self.pool2(x)
            
            x = self.conv3(x)
            x = self.bn3(x)
            x = self.pool3(x)
            
            x = self.conv4(x)
            x = self.bn4(x)
            x = self.pool4(x)
        
        x = self.global_pool(x)
        x = self.dropout(x)
        x = self.dense1(x)
        x = self.reshape(x)
        
        return x


class VibrationEncoder(BaseEncoder):
    """Encoder for vibration sensor data."""
    
    def _get_input_shape(self) -> Tuple[int, ...]:
        """Get the expected input shape for vibration data."""
        return (
            self.config.get('sequence_length', 512),
            self.config.get('feature_dims', 3)
        )
    
    def _build_layers(self):
        """Build encoder layers for vibration time series data."""
        filters = self.config.get('filters', [64, 128, 256])
        kernel_sizes = self.config.get('kernel_sizes', [9, 5, 3])
        dropout_rate = self.config.get('dropout_rate', 0.2)
        
        # 1D CNN for temporal features
        self.conv_layers = []
        self.bn_layers = []
        self.pool_layers = []
        
        for i, (f, k) in enumerate(zip(filters, kernel_sizes)):
            self.conv_layers.append(
                tf.keras.layers.Conv1D(f, k, padding='same', activation='relu')
            )
            self.bn_layers.append(
                tf.keras.layers.BatchNormalization()
            )
            self.pool_layers.append(
                tf.keras.layers.MaxPooling1D(2)
            )
        
        # Additional processing layers
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        self.bidirectional = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(128, return_sequences=True)
        )
        self.global_pool = tf.keras.layers.GlobalAveragePooling1D()
        self.dense = tf.keras.layers.Dense(self.output_dim, activation='relu')
        self.reshape = tf.keras.layers.Reshape((1, self.output_dim))
    
    def preprocess(self, data: np.ndarray) -> np.ndarray:
        """
        Preprocess the vibration data before encoding.
        
        Args:
            data: Raw vibration data (time_steps, features)
            
        Returns:
            Preprocessed vibration data
        """
        # Ensure data is in the correct format
        if len(data.shape) == 2:  # Single sample
            data = np.expand_dims(data, axis=0)
        
        # Ensure consistent length
        target_length = self.input_shape[0]
        processed = []
        
        for sample in data:
            if sample.shape[0] > target_length:
                # Downsample to target length
                indices = np.linspace(0, sample.shape[0]-1, target_length, dtype=int)
                resampled = sample[indices]
                processed.append(resampled)
            elif sample.shape[0] < target_length:
                # Pad to target length
                padding = np.zeros((target_length - sample.shape[0], sample.shape[1]))
                padded = np.vstack([sample, padding])
                processed.append(padded)
            else:
                processed.append(sample)
        
        # Convert to numpy array and normalize
        processed = np.array(processed)
        
        # Z-score normalization (per channel)
        for i in range(processed.shape[0]):
            for j in range(processed.shape[2]):
                channel_data = processed[i, :, j]
                mean = np.mean(channel_data)
                std = np.std(channel_data)
                if std > 0:
                    processed[i, :, j] = (channel_data - mean) / std
        
        return processed
    
    def call(self, inputs):
        """
        Forward pass for the vibration encoder.
        
        Args:
            inputs: Input vibration tensor
            
        Returns:
            Encoded output tensor
        """
        x = inputs
        
        # Apply CNN layers
        for conv, bn, pool in zip(self.conv_layers, self.bn_layers, self.pool_layers):
            x = conv(x)
            x = bn(x)
            x = pool(x)
        
        # Apply RNN layer
        x = self.bidirectional(x)
        
        # Final processing
        x = self.global_pool(x)
        x = self.dropout(x)
        x = self.dense(x)
        x = self.reshape(x)
        
        return x


class TemperatureEncoder(BaseEncoder):
    """Encoder for temperature sensor data."""
    
    def _get_input_shape(self) -> Tuple[int, ...]:
        """Get the expected input shape for temperature data."""
        # For temperature maps
        if self.config.get('use_temperature_map', False):
            return (
                self.config.get('map_height', 64),
                self.config.get('map_width', 64),
                1
            )
        else:
            # For scalar temperature and ambient values
            return (2,)
    
    def _build_layers(self):
        """Build encoder layers for temperature data."""
        # Check if using temperature maps
        if self.config.get('use_temperature_map', False):
            # CNN for temperature maps
            self.conv1 = tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu')
            self.pool1 = tf.keras.layers.MaxPooling2D(2)
            self.conv2 = tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu')
            self.pool2 = tf.keras.layers.MaxPooling2D(2)
            self.flatten = tf.keras.layers.Flatten()
        
        # Common layers for all input types
        self.dense1 = tf.keras.layers.Dense(128, activation='relu')
        self.dropout = tf.keras.layers.Dropout(self.config.get('dropout_rate', 0.1))
        self.dense2 = tf.keras.layers.Dense(self.output_dim, activation='relu')
        self.reshape = tf.keras.layers.Reshape((1, self.output_dim))
    
    def preprocess(self, data: Union[float, Tuple[float, float], np.ndarray]) -> np.ndarray:
        """
        Preprocess the temperature data before encoding.
        
        Args:
            data: Raw temperature data (can be scalar, tuple of [temp, ambient], or temperature map)
            
        Returns:
            Preprocessed temperature data
        """
        # Check if using temperature maps
        if self.config.get('use_temperature_map', False):
            if isinstance(data, np.ndarray) and len(data.shape) >= 2:
                # Temperature map data
                if len(data.shape) == 2:
                    # Add channel dimension
                    data = np.expand_dims(data, axis=-1)
                
                if len(data.shape) == 3 and data.shape[0] != 1:
                    # Add batch dimension
                    data = np.expand_dims(data, axis=0)
                
                # Resize if needed
                target_h, target_w = self.input_shape[0], self.input_shape[1]
                if data.shape[1] != target_h or data.shape[2] != target_w:
                    resized = np.zeros((data.shape[0], target_h, target_w, data.shape[3]))
                    for i in range(data.shape[0]):
                        for j in range(data.shape[3]):
                            resized[i, :, :, j] = cv2.resize(data[i, :, :, j], (target_w, target_h))
                    data = resized
                
                # Normalize to 0-1 range
                min_temp = self.config.get('min_temp', 25.0)
                max_temp = self.config.get('max_temp', 45.0)
                data = (data - min_temp) / (max_temp - min_temp)
                data = np.clip(data, 0, 1)
                
                return data
            else:
                self.logger.warning("Expected temperature map but received scalar values")
                # Create a dummy map with uniform values
                dummy_map = np.zeros((1, self.input_shape[0], self.input_shape[1], 1))
                
                # Set values based on scalar temperature
                if isinstance(data, tuple):
                    temp_value = data[0]
                else:
                    temp_value = data
                
                # Normalize to 0-1 range
                min_temp = self.config.get('min_temp', 25.0)
                max_temp = self.config.get('max_temp', 45.0)
                normalized_temp = (temp_value - min_temp) / (max_temp - min_temp)
                normalized_temp = max(0, min(1, normalized_temp))
                
                dummy_map.fill(normalized_temp)
                return dummy_map
        else:
            # Scalar temperature processing
            if isinstance(data, np.ndarray) and len(data.shape) >= 2:
                self.logger.warning("Received temperature map but scalar values were expected")
                # Extract average temperature from map
                temp_value = np.mean(data)
                ambient_temp = self.config.get('default_ambient_temp', 25.0)
                data = np.array([[temp_value, ambient_temp]])
            elif isinstance(data, tuple):
                # Temperature and ambient temperature
                data = np.array([[data[0], data[1]]])
            else:
                # Only temperature, use default for ambient
                ambient_temp = self.config.get('default_ambient_temp', 25.0)
                data = np.array([[data, ambient_temp]])
            
            # Normalize to physiological ranges
            min_temp = self.config.get('min_temp', 25.0)
            max_temp = self.config.get('max_temp', 45.0)
            min_ambient = self.config.get('min_ambient', 15.0)
            max_ambient = self.config.get('max_ambient', 40.0)
            
            normalized = np.zeros_like(data)
            normalized[:, 0] = (data[:, 0] - min_temp) / (max_temp - min_temp)
            normalized[:, 1] = (data[:, 1] - min_ambient) / (max_ambient - min_ambient)
            normalized = np.clip(normalized, 0, 1)
            
            return normalized
    
    def call(self, inputs):
        """
        Forward pass for the temperature encoder.
        
        Args:
            inputs: Input temperature tensor
            
        Returns:
            Encoded output tensor
        """
        # Check if using temperature maps
        if self.config.get('use_temperature_map', False):
            x = self.conv1(inputs)
            x = self.pool1(x)
            x = self.conv2(x)
            x = self.pool2(x)
            x = self.flatten(x)
        else:
            x = inputs
        
        # Common processing
        x = self.dense1(x)
        x = self.dropout(x)
        x = self.dense2(x)
        x = self.reshape(x)
        
        return x


class PressureEncoder(BaseEncoder):
    """Encoder for pressure sensor data."""
    
    def _get_input_shape(self) -> Tuple[int, ...]:
        """Get the expected input shape for pressure data."""
        return (self.config.get('sequence_length', 256), 1)
    
    def _build_layers(self):
        """Build encoder layers for pressure time series data."""
        lstm_units = self.config.get('lstm_units', 128)
        dropout_rate = self.config.get('dropout_rate', 0.2)
        
        # 1D CNN for feature extraction
        self.conv1 = tf.keras.layers.Conv1D(32, 5, activation='relu', padding='same')
        self.pool1 = tf.keras.layers.MaxPooling1D(2)
        self.conv2 = tf.keras.layers.Conv1D(64, 3, activation='relu', padding='same')
        self.pool2 = tf.keras.layers.MaxPooling1D(2)
        
        # RNN for temporal patterns
        self.lstm = tf.keras.layers.LSTM(lstm_units)
        
        # Output layers
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        self.dense1 = tf.keras.layers.Dense(self.output_dim, activation='relu')
        self.reshape = tf.keras.layers.Reshape((1, self.output_dim))
    
    def preprocess(self, data: np.ndarray) -> np.ndarray:
        """
        Preprocess the pressure data before encoding.
        
        Args:
            data: Raw pressure data (time_steps, 1)
            
        Returns:
            Preprocessed pressure data
        """
        # Ensure data is in the correct format
        if len(data.shape) == 1:  # Single channel time series
            data = np.expand_dims(data, axis=-1)
        
        if len(data.shape) == 2:  # Add batch dimension
            data = np.expand_dims(data, axis=0)
        
        # Ensure consistent length
        target_length = self.input_shape[0]
        processed = []
        
        for sample in data:
            if sample.shape[0] > target_length:
                # Downsample to target length
                indices = np.linspace(0, sample.shape[0]-1, target_length, dtype=int)
                resampled = sample[indices]
                processed.append(resampled)
            elif sample.shape[0] < target_length:
                # Pad to target length
                padding = np.zeros((target_length - sample.shape[0], sample.shape[1]))
                padded = np.vstack([sample, padding])
                processed.append(padded)
            else:
                processed.append(sample)
        
        # Convert to numpy array
        processed = np.array(processed)
        
        # Normalize pressure values to standard range
        min_pressure = self.config.get('min_pressure', -400)
        max_pressure = self.config.get('max_pressure', 400)
        
        normalized = (processed - min_pressure) / (max_pressure - min_pressure)
        normalized = np.clip(normalized, 0, 1)
        
        return normalized
    
    def call(self, inputs):
        """
        Forward pass for the pressure encoder.
        
        Args:
            inputs: Input pressure tensor
            
        Returns:
            Encoded output tensor
        """
        x = self.conv1(inputs)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.lstm(x)
        x = self.dropout(x)
        x = self.dense1(x)
        x = self.reshape(x)
        
        return x


class AcousticEncoder(BaseEncoder):
    """Encoder for acoustic emission sensor data."""
    
    def _get_input_shape(self) -> Tuple[int, ...]:
        """Get the expected input shape for acoustic data."""
        return (
            self.config.get('sequence_length', 1024),
            self.config.get('feature_dims', 1)
        )
    
    def _build_layers(self):
        """Build encoder layers for acoustic time series data."""
        filters = self.config.get('filters', [64, 128, 256])
        kernel_sizes = self.config.get('kernel_sizes', [9, 5, 3])
        dropout_rate = self.config.get('dropout_rate', 0.2)
        
        # 1D CNN for acoustic features
        self.conv_layers = []
        self.bn_layers = []
        self.pool_layers = []
        
        for i, (f, k) in enumerate(zip(filters, kernel_sizes)):
            self.conv_layers.append(
                tf.keras.layers.Conv1D(f, k, padding='same', activation='relu')
            )
            self.bn_layers.append(
                tf.keras.layers.BatchNormalization()
            )
            self.pool_layers.append(
                tf.keras.layers.MaxPooling1D(2)
            )
        
        # Frequency domain features extraction
        self.stft_layer = tf.keras.layers.Lambda(lambda x: tf.signal.stft(
            x[:, :, 0], 
            frame_length=self.config.get('frame_length', 256),
            frame_step=self.config.get('frame_step', 128)
        ))
        
        self.mag_layer = tf.keras.layers.Lambda(lambda x: tf.abs(x))
        self.log_layer = tf.keras.layers.Lambda(lambda x: tf.math.log(x + 1e-6))
        
        # Final processing
        self.global_avg_pool = tf.keras.layers.GlobalAveragePooling1D()
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        self.dense1 = tf.keras.layers.Dense(self.output_dim, activation='relu')
        self.reshape = tf.keras.layers.Reshape((1, self.output_dim))
    
    def preprocess(self, data: np.ndarray) -> np.ndarray:
        """
        Preprocess the acoustic data before encoding.
        
        Args:
            data: Raw acoustic data (time_steps, channels)
            
        Returns:
            Preprocessed acoustic data
        """
        # Ensure data is in the correct format
        if len(data.shape) == 1:  # Single channel time series
            data = np.expand_dims(data, axis=-1)
        
        if len(data.shape) == 2:  # Add batch dimension
            data = np.expand_dims(data, axis=0)
        
        # Ensure consistent length
        target_length = self.input_shape[0]
        processed = []
        
        for sample in data:
            if sample.shape[0] > target_length:
                # Downsample to target length
                indices = np.linspace(0, sample.shape[0]-1, target_length, dtype=int)
                resampled = sample[indices]
                processed.append(resampled)
            elif sample.shape[0] < target_length:
                # Pad to target length
                padding = np.zeros((target_length - sample.shape[0], sample.shape[1]))
                padded = np.vstack([sample, padding])
                processed.append(padded)
            else:
                processed.append(sample)
        
        # Convert to numpy array
        processed = np.array(processed)
        
        # Normalize to [-1, 1] range for audio
        for i in range(processed.shape[0]):
            for j in range(processed.shape[2]):
                channel_data = processed[i, :, j]
                max_val = np.max(np.abs(channel_data))
                if max_val > 0:
                    processed[i, :, j] = channel_data / max_val
        
        return processed
    
    def call(self, inputs):
        """
        Forward pass for the acoustic encoder.
        
        Args:
            inputs: Input acoustic tensor
            
        Returns:
            Encoded output tensor
        """
        # Time domain processing
        x_time = inputs
        for conv, bn, pool in zip(self.conv_layers, self.bn_layers, self.pool_layers):
            x_time = conv(x_time)
            x_time = bn(x_time)
            x_time = pool(x_time)
        
        x_time = self.global_avg_pool(x_time)
        
        # Frequency domain processing
        x_freq = self.stft_layer(inputs)
        x_freq = self.mag_layer(x_freq)
        x_freq = self.log_layer(x_freq)
        x_freq = self.global_avg_pool(x_freq)
        
        # Combine time and frequency domain features
        x = tf.concat([x_time, x_freq], axis=-1)
        x = self.dropout(x)
        x = self.dense1(x)
        x = self.reshape(x)
        
        return x


class EEGEncoder(BaseEncoder):
    """Encoder for EEG sensor data."""
    
    def _get_input_shape(self) -> Tuple[int, ...]:
        """Get the expected input shape for EEG data."""
        return (
            self.config.get('sequence_length', 512),
            self.config.get('num_channels', 1)
        )
    
    def _build_layers(self):
        """Build encoder layers for EEG time series data."""
        filters = self.config.get('filters', [64, 128, 256])
        kernel_sizes = self.config.get('kernel_sizes', [11, 7, 5])
        dropout_rate = self.config.get('dropout_rate', 0.3)
        
        # 1D CNN for EEG signal processing
        self.conv_layers = []
        self.bn_layers = []
        self.pool_layers = []
        
        for i, (f, k) in enumerate(zip(filters, kernel_sizes)):
            self.conv_layers.append(
                tf.keras.layers.Conv1D(f, k, padding='same', activation='elu')
            )
            self.bn_layers.append(
                tf.keras.layers.BatchNormalization()
            )
            self.pool_layers.append(
                tf.keras.layers.MaxPooling1D(2)
            )
        
        # Bandpower extraction
        self.bandpower_layer = tf.keras.layers.Lambda(self._extract_bandpowers)
        
        # Final processing
        self.global_avg_pool = tf.keras.layers.GlobalAveragePooling1D()
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        self.dense1 = tf.keras.layers.Dense(128, activation='relu')
        self.dense2 = tf.keras.layers.Dense(self.output_dim, activation='relu')
        self.reshape = tf.keras.layers.Reshape((1, self.output_dim))
    
    def _extract_bandpowers(self, x):
        """Extract standard EEG bandpowers."""
        # This would typically use FFT to extract power in different frequency bands
        # Simplified implementation for the model architecture
        return x
    
    def preprocess(self, data: np.ndarray) -> np.ndarray:
        """
        Preprocess the EEG data before encoding.
        
        Args:
            data: Raw EEG data (time_steps, channels)
            
        Returns:
            Preprocessed EEG data
        """
        # Ensure data is in the correct format
        if len(data.shape) == 1:  # Single channel time series
            data = np.expand_dims(data, axis=-1)
        
        if len(data.shape) == 2:  # Add batch dimension
            data = np.expand_dims(data, axis=0)
        
        # Ensure consistent length
        target_length = self.input_shape[0]
        processed = []
        
        for sample in data:
            if sample.shape[0] > target_length:
                # Downsample to target length
                indices = np.linspace(0, sample.shape[0]-1, target_length, dtype=int)
                resampled = sample[indices]
                processed.append(resampled)
            elif sample.shape[0] < target_length:
                # Pad to target length
                padding = np.zeros((target_length - sample.shape[0], sample.shape[1]))
                padded = np.vstack([sample, padding])
                processed.append(padded)
            else:
                processed.append(sample)
        
        # Convert to numpy array
        processed = np.array(processed)
        
        # Apply bandpass filtering (simulate with normalization here)
        for i in range(processed.shape[0]):
            for j in range(processed.shape[2]):
                channel_data = processed[i, :, j]
                # Z-score normalization
                mean = np.mean(channel_data)
                std = np.std(channel_data)
                if std > 0:
                    processed[i, :, j] = (channel_data - mean) / std
        
        return processed
    
    def call(self, inputs):
        """
        Forward pass for the EEG encoder.
        
        Args:
            inputs: Input EEG tensor
            
        Returns:
            Encoded output tensor
        """
        x = inputs
        
        # Apply CNN layers
        for conv, bn, pool in zip(self.conv_layers, self.bn_layers, self.pool_layers):
            x = conv(x)
            x = bn(x)
            x = pool(x)
        
        # Extract bandpowers
        bandpowers = self.bandpower_layer(inputs)
        
        # Process CNN output
        x = self.global_avg_pool(x)
        
        # Combine CNN features with bandpowers
        x = tf.concat([x, self.global_avg_pool(bandpowers)], axis=-1)
        
        # Final processing
        x = self.dropout(x)
        x = self.dense1(x)
        x = self.dropout(x)
        x = self.dense2(x)
        x = self.reshape(x)
        
        return x
