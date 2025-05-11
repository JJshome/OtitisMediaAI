"""
Transformer-based model for multi-sensor data fusion and otitis media diagnosis.
"""

import os
import logging
import numpy as np
import tensorflow as tf
from typing import Dict, List, Optional, Tuple, Union, Any
import json
import time

from .encoder_modules import ImageEncoder, VibrationEncoder, TemperatureEncoder
from .encoder_modules import PressureEncoder, AcousticEncoder, EEGEncoder
from .attention_modules import MultiHeadAttention, FeedForward
from ..utils.data_structures import SensorType, DiagnosticResult, PrognosisResult


class TransformerModel:
    """
    Transformer-based model for multi-sensor data fusion and analysis.
    Processes and integrates data from different sensor types using
    attention mechanisms to make accurate diagnostic predictions.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the transformer model with the provided configuration.
        
        Args:
            config: Model configuration parameters
        """
        self.logger = logging.getLogger('otitismedia_ai.transformer_model')
        self.config = config
        self.model_path = config.get('model_path', 'models/transformer')
        self.weights_path = config.get('weights_path', None)
        self.input_dim = config.get('input_dim', 256)
        self.num_heads = config.get('num_heads', 8)
        self.num_encoder_layers = config.get('num_encoder_layers', 6)
        self.dropout_rate = config.get('dropout_rate', 0.1)
        self.use_gpu = config.get('use_gpu', True)
        
        # Set up device configuration
        if self.use_gpu and tf.config.list_physical_devices('GPU'):
            self.device = '/GPU:0'
            self.logger.info("Using GPU for model inference")
        else:
            self.device = '/CPU:0'
            self.logger.info("Using CPU for model inference (no GPU available or disabled)")
        
        # Initialize sensor-specific encoders
        self._initialize_encoders()
        
        # Build the transformer model
        self._build_model()
        
        # Load pre-trained weights if provided
        if self.weights_path:
            self._load_weights()
        
        self.logger.info("Transformer model initialized")
    
    def _initialize_encoders(self):
        """Initialize sensor-specific encoder modules."""
        self.encoders = {
            SensorType.IMAGE: ImageEncoder(self.config.get('image_encoder_config', {})),
            SensorType.VIBRATION: VibrationEncoder(self.config.get('vibration_encoder_config', {})),
            SensorType.TEMPERATURE: TemperatureEncoder(self.config.get('temperature_encoder_config', {})),
            SensorType.PRESSURE: PressureEncoder(self.config.get('pressure_encoder_config', {})),
            SensorType.ACOUSTIC: AcousticEncoder(self.config.get('acoustic_encoder_config', {})),
            SensorType.EEG: EEGEncoder(self.config.get('eeg_encoder_config', {}))
        }
    
    def _build_model(self):
        """Build the transformer model architecture."""
        with tf.device(self.device):
            # Define input placeholders for each sensor type
            self.inputs = {
                sensor_type: tf.keras.layers.Input(shape=encoder.input_shape, name=f"{sensor_type.name.lower()}_input")
                for sensor_type, encoder in self.encoders.items()
            }
            
            # Sensor-specific encoding
            encoded_features = {
                sensor_type: encoder(self.inputs[sensor_type])
                for sensor_type, encoder in self.encoders.items()
            }
            
            # Concatenate all encoded features
            concat_features = tf.keras.layers.Concatenate(axis=1)(list(encoded_features.values()))
            
            # Position encoding
            pos_encoding = self._get_positional_encoding(tf.shape(concat_features)[1], self.input_dim)
            concat_features = concat_features + pos_encoding
            
            # Transformer encoder layers
            encoder_output = concat_features
            for i in range(self.num_encoder_layers):
                encoder_output = self._transformer_encoder_layer(
                    encoder_output,
                    units=self.input_dim,
                    d_model=self.input_dim,
                    num_heads=self.num_heads,
                    dropout=self.dropout_rate,
                    name=f"encoder_layer_{i}"
                )
            
            # Global average pooling
            pooled = tf.keras.layers.GlobalAveragePooling1D()(encoder_output)
            
            # Diagnostic head
            diagnostic_head = tf.keras.layers.Dense(512, activation='relu')(pooled)
            diagnostic_head = tf.keras.layers.Dropout(self.dropout_rate)(diagnostic_head)
            diagnostic_head = tf.keras.layers.Dense(256, activation='relu')(diagnostic_head)
            
            # Multiple output heads for different diagnostic aspects
            diagnosis_output = tf.keras.layers.Dense(4, activation='softmax', name='diagnosis')(diagnostic_head)  # Normal, OME, AOM, CSOM
            severity_output = tf.keras.layers.Dense(1, activation='sigmoid', name='severity')(diagnostic_head)
            effusion_type_output = tf.keras.layers.Dense(3, activation='softmax', name='effusion_type')(diagnostic_head)  # None, Serous, Mucoid
            
            # Prognosis head
            prognosis_head = tf.keras.layers.Dense(256, activation='relu')(pooled)
            prognosis_head = tf.keras.layers.Dropout(self.dropout_rate)(prognosis_head)
            prognosis_head = tf.keras.layers.Dense(128, activation='relu')(prognosis_head)
            
            # Multiple output heads for different prognosis aspects
            recovery_output = tf.keras.layers.Dense(1, activation='sigmoid', name='recovery_probability')(prognosis_head)
            surgery_output = tf.keras.layers.Dense(1, activation='sigmoid', name='surgery_probability')(prognosis_head)
            
            # Define the model with multiple inputs and outputs
            self.model = tf.keras.Model(
                inputs=list(self.inputs.values()),
                outputs=[
                    diagnosis_output,
                    severity_output,
                    effusion_type_output,
                    recovery_output,
                    surgery_output
                ]
            )
            
            # Compile the model
            self.model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=self.config.get('learning_rate', 0.0001)),
                loss={
                    'diagnosis': 'categorical_crossentropy',
                    'severity': 'mse',
                    'effusion_type': 'categorical_crossentropy',
                    'recovery_probability': 'mse',
                    'surgery_probability': 'mse'
                },
                metrics={
                    'diagnosis': 'accuracy',
                    'severity': 'mae',
                    'effusion_type': 'accuracy',
                    'recovery_probability': 'mae',
                    'surgery_probability': 'mae'
                }
            )
            
            self.logger.info(f"Model built with {self.model.count_params()} parameters")
    
    def _transformer_encoder_layer(self, inputs, units, d_model, num_heads, dropout, name):
        """Single transformer encoder layer with multi-head attention and feed-forward network."""
        # Multi-head attention
        attention_output = MultiHeadAttention(d_model, num_heads)(inputs, inputs, inputs)
        attention_output = tf.keras.layers.Dropout(dropout)(attention_output)
        attention_output = tf.keras.layers.LayerNormalization(epsilon=1e-6)(inputs + attention_output)
        
        # Feed-forward network
        ffn_output = FeedForward(units, d_model)(attention_output)
        ffn_output = tf.keras.layers.Dropout(dropout)(ffn_output)
        encoder_output = tf.keras.layers.LayerNormalization(epsilon=1e-6)(attention_output + ffn_output)
        
        return encoder_output
    
    def _get_positional_encoding(self, seq_len, d_model):
        """Generate positional encoding for transformer inputs."""
        positions = np.arange(seq_len)[:, np.newaxis]
        depths = np.arange(d_model)[np.newaxis, :]
        angle_rates = 1 / np.power(10000, (2 * (depths // 2)) / np.float32(d_model))
        angle_rads = positions * angle_rates
        
        # Apply sin to even indices, cos to odd indices
        pos_encoding = np.zeros(angle_rads.shape)
        pos_encoding[:, 0::2] = np.sin(angle_rads[:, 0::2])
        pos_encoding[:, 1::2] = np.cos(angle_rads[:, 1::2])
        
        return tf.cast(pos_encoding[np.newaxis, ...], dtype=tf.float32)
    
    def _load_weights(self):
        """Load pre-trained weights if available."""
        if not os.path.exists(self.weights_path):
            self.logger.warning(f"Weights file not found at {self.weights_path}")
            return False
        
        try:
            self.model.load_weights(self.weights_path)
            self.logger.info(f"Model weights loaded from {self.weights_path}")
            return True
        except Exception as e:
            self.logger.error(f"Error loading model weights: {str(e)}")
            return False
    
    def save_weights(self, path: Optional[str] = None):
        """
        Save model weights to file.
        
        Args:
            path: Path to save weights, uses default path if None
        """
        save_path = path or self.weights_path
        if save_path is None:
            save_path = os.path.join(self.model_path, 'weights', f'transformer_weights_{int(time.time())}.h5')
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        try:
            self.model.save_weights(save_path)
            self.logger.info(f"Model weights saved to {save_path}")
            return True
        except Exception as e:
            self.logger.error(f"Error saving model weights: {str(e)}")
            return False
    
    def train(self, train_data, validation_data=None, epochs=10, batch_size=32, callbacks=None):
        """
        Train the transformer model on the provided data.
        
        Args:
            train_data: Training data as (inputs, targets) tuple
            validation_data: Validation data as (inputs, targets) tuple
            epochs: Number of training epochs
            batch_size: Batch size for training
            callbacks: List of Keras callbacks
            
        Returns:
            Training history
        """
        self.logger.info(f"Starting model training for {epochs} epochs with batch size {batch_size}")
        
        # Default callbacks if none provided
        if callbacks is None:
            callbacks = [
                tf.keras.callbacks.ModelCheckpoint(
                    filepath=os.path.join(self.model_path, 'checkpoints', 'ckpt_{epoch:02d}'),
                    save_weights_only=True,
                    save_best_only=True,
                    monitor='val_loss' if validation_data else 'loss'
                ),
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss' if validation_data else 'loss',
                    patience=5,
                    restore_best_weights=True
                ),
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss' if validation_data else 'loss',
                    factor=0.5,
                    patience=2
                ),
                tf.keras.callbacks.TensorBoard(
                    log_dir=os.path.join(self.model_path, 'logs'),
                    histogram_freq=1
                )
            ]
        
        # Train the model
        history = self.model.fit(
            train_data[0],
            train_data[1],
            epochs=epochs,
            batch_size=batch_size,
            validation_data=validation_data,
            callbacks=callbacks
        )
        
        self.logger.info("Model training completed")
        return history
    
    def predict(self, sensor_data: Dict[SensorType, Any]) -> Tuple[DiagnosticResult, PrognosisResult]:
        """
        Generate diagnostic and prognosis predictions from sensor data.
        
        Args:
            sensor_data: Dictionary mapping sensor types to corresponding data
            
        Returns:
            Tuple of (DiagnosticResult, PrognosisResult)
        """
        self.logger.info("Generating predictions from sensor data")
        
        # Encode sensor data
        model_inputs = {}
        for sensor_type, data in sensor_data.items():
            if sensor_type in self.encoders:
                try:
                    processed_data = self.encoders[sensor_type].preprocess(data)
                    model_inputs[sensor_type.name.lower() + '_input'] = processed_data
                except Exception as e:
                    self.logger.error(f"Error preprocessing {sensor_type.name} data: {str(e)}")
        
        # Ensure all required inputs are present
        missing_inputs = set(self.inputs.keys()) - set(sensor_data.keys())
        if missing_inputs:
            self.logger.warning(f"Missing inputs for sensor types: {[t.name for t in missing_inputs]}")
            # Generate dummy inputs for missing sensors
            for sensor_type in missing_inputs:
                if sensor_type in self.encoders:
                    dummy_shape = self.encoders[sensor_type].input_shape
                    model_inputs[sensor_type.name.lower() + '_input'] = np.zeros((1,) + dummy_shape)
        
        # Run inference
        with tf.device(self.device):
            try:
                predictions = self.model.predict(model_inputs)
                
                # Extract predictions
                diagnosis_pred, severity_pred, effusion_type_pred, recovery_pred, surgery_pred = predictions
                
                # Map diagnosis index to name
                diagnosis_map = ['Normal', 'OME', 'AOM', 'CSOM']
                diagnosis_idx = np.argmax(diagnosis_pred[0])
                diagnosis = diagnosis_map[diagnosis_idx]
                diagnosis_confidence = float(diagnosis_pred[0][diagnosis_idx])
                
                # Map effusion type index to name
                effusion_map = ['None', 'Serous', 'Mucoid']
                effusion_idx = np.argmax(effusion_type_pred[0])
                effusion_type = effusion_map[effusion_idx]
                
                # Create diagnostic result
                diagnostic_result = DiagnosticResult(
                    timestamp=time.time(),
                    diagnosis=diagnosis,
                    confidence=diagnosis_confidence,
                    severity=float(severity_pred[0][0]),
                    effusion_type=effusion_type if effusion_idx > 0 else None,
                    effusion_volume=self._estimate_effusion_volume(sensor_data) if effusion_idx > 0 else None,
                    inflammation_level=self._estimate_inflammation(sensor_data),
                    membrane_mobility=self._estimate_mobility(sensor_data),
                    pain_level=self._estimate_pain_level(sensor_data),
                    additional_findings={}
                )
                
                # Create prognosis result
                prognosis_result = PrognosisResult(
                    natural_recovery_probability=float(recovery_pred[0][0]),
                    surgery_recommendation=float(surgery_pred[0][0]),
                    estimated_recovery_time=self._estimate_recovery_time(diagnostic_result),
                    treatment_recommendations=self._generate_treatment_recommendations(diagnostic_result),
                    follow_up_recommendation=self._generate_followup_recommendation(diagnostic_result)
                )
                
                self.logger.info(
                    f"Prediction generated: {diagnosis} (confidence: {diagnosis_confidence:.2f}), "
                    f"severity: {diagnostic_result.severity:.2f}"
                )
                
                return diagnostic_result, prognosis_result
                
            except Exception as e:
                self.logger.error(f"Error during prediction: {str(e)}")
                raise
    
    def _estimate_effusion_volume(self, sensor_data: Dict[SensorType, Any]) -> float:
        """Estimate effusion volume based on sensor data."""
        # Placeholder implementation - would integrate relevant sensor data
        # such as tympanometry, vibration response, etc.
        return 0.5  # Default placeholder value
    
    def _estimate_inflammation(self, sensor_data: Dict[SensorType, Any]) -> float:
        """Estimate inflammation level based on sensor data."""
        # Placeholder implementation - would combine temperature and image data
        return 0.3  # Default placeholder value
    
    def _estimate_mobility(self, sensor_data: Dict[SensorType, Any]) -> float:
        """Estimate tympanic membrane mobility based on sensor data."""
        # Placeholder implementation - would use vibration sensor data
        return 0.7  # Default placeholder value
    
    def _estimate_pain_level(self, sensor_data: Dict[SensorType, Any]) -> Optional[float]:
        """Estimate pain level based on sensor data, primarily EEG."""
        # Placeholder implementation
        return 3.0  # Default placeholder value (0-10 scale)
    
    def _estimate_recovery_time(self, diagnostic_result: DiagnosticResult) -> Optional[int]:
        """Estimate recovery time in days based on diagnostic findings."""
        # Placeholder implementation - would be based on diagnosis, severity, etc.
        if diagnostic_result.diagnosis == "Normal":
            return 0
        elif diagnostic_result.diagnosis == "OME":
            return int(30 + 30 * diagnostic_result.severity)
        elif diagnostic_result.diagnosis == "AOM":
            return int(7 + 7 * diagnostic_result.severity)
        else:  # CSOM
            return 90
    
    def _generate_treatment_recommendations(self, diagnostic_result: DiagnosticResult) -> List[str]:
        """Generate treatment recommendations based on diagnostic findings."""
        # Placeholder implementation
        recommendations = []
        
        if diagnostic_result.diagnosis == "Normal":
            recommendations.append("No treatment required")
        elif diagnostic_result.diagnosis == "OME":
            if diagnostic_result.severity < 0.3:
                recommendations.append("Watchful waiting")
                recommendations.append("Follow-up in 3 months")
            elif diagnostic_result.severity < 0.7:
                recommendations.append("Nasal steroids")
                recommendations.append("Follow-up in 1 month")
            else:
                recommendations.append("Consider ventilation tube insertion")
        elif diagnostic_result.diagnosis == "AOM":
            recommendations.append("Antibiotics")
            recommendations.append("Pain management")
            recommendations.append("Follow-up in 2 weeks")
        else:  # CSOM
            recommendations.append("Topical antibiotics")
            recommendations.append("ENT referral")
        
        return recommendations
    
    def _generate_followup_recommendation(self, diagnostic_result: DiagnosticResult) -> Optional[str]:
        """Generate follow-up recommendation based on diagnostic findings."""
        # Placeholder implementation
        if diagnostic_result.diagnosis == "Normal":
            return "Routine check-up in 1 year"
        elif diagnostic_result.diagnosis == "OME":
            if diagnostic_result.severity < 0.5:
                return "Follow-up in 3 months"
            else:
                return "Follow-up in 1 month"
        elif diagnostic_result.diagnosis == "AOM":
            return "Follow-up in 2 weeks"
        else:  # CSOM
            return "ENT referral within 1 week"
