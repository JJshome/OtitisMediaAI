"""
Visualization Manager for 3D rendering and AR visualization of ear anatomy and diagnostic data.
"""

import os
import logging
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
import json
import time

from .ear_model_generator import EarModelGenerator
from .ar_renderer import ARRenderer
from ..utils.data_structures import SensorType, DiagnosticResult


class VisualizationManager:
    """
    Manages 3D modeling and AR visualization of ear anatomy and diagnostic results.
    Provides a unified interface for generating and manipulating visualizations.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the visualization manager with the provided configuration.
        
        Args:
            config: Visualization configuration parameters
        """
        self.logger = logging.getLogger('otitismedia_ai.visualization_manager')
        self.config = config
        self.enabled = config.get('enabled', True)
        self.output_dir = config.get('output_dir', 'output/visualizations')
        self.ar_mode = config.get('ar_mode', False)
        self.model_quality = config.get('model_quality', 'medium')  # Options: low, medium, high
        
        # Create output directory if it doesn't exist
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize model generator and AR renderer if enabled
        if self.enabled:
            self.model_generator = EarModelGenerator(
                self.config.get('model_generator_config', {})
            )
            
            if self.ar_mode:
                self.ar_renderer = ARRenderer(
                    self.config.get('ar_renderer_config', {})
                )
                self.logger.info("AR visualization mode enabled")
            else:
                self.ar_renderer = None
                self.logger.info("Standard 3D visualization mode enabled")
        else:
            self.model_generator = None
            self.ar_renderer = None
            self.logger.info("Visualization is disabled")
    
    def generate_ear_model(self, 
                          image_data: np.ndarray,
                          diagnostic_result: Optional[DiagnosticResult] = None) -> Dict[str, Any]:
        """
        Generate a 3D model of the ear based on image data and diagnostic results.
        
        Args:
            image_data: Tympanic membrane image data
            diagnostic_result: Diagnostic results to incorporate into the model
            
        Returns:
            Dictionary with model information and file paths
        """
        if not self.enabled or self.model_generator is None:
            self.logger.warning("Visualization is disabled, cannot generate ear model")
            return {"error": "Visualization is disabled"}
        
        self.logger.info("Generating 3D ear model")
        
        try:
            # Generate the base ear model
            model_info = self.model_generator.generate_model(
                image_data=image_data,
                quality=self.model_quality
            )
            
            # If diagnostic results are provided, enhance the model with diagnostic data
            if diagnostic_result:
                model_info = self.model_generator.enhance_model_with_diagnosis(
                    model_info=model_info,
                    diagnostic_result=diagnostic_result
                )
            
            # Save the model
            timestamp = int(time.time())
            model_file = os.path.join(self.output_dir, f"ear_model_{timestamp}")
            
            model_paths = self.model_generator.save_model(
                model_info=model_info,
                base_path=model_file
            )
            
            result = {
                "model_info": model_info,
                "model_paths": model_paths,
                "timestamp": timestamp
            }
            
            self.logger.info(f"3D ear model generated successfully: {model_paths['obj']}")
            return result
            
        except Exception as e:
            self.logger.error(f"Error generating ear model: {str(e)}")
            return {"error": str(e)}
    
    def render_ar_visualization(self, 
                              model_info: Dict[str, Any],
                              diagnostic_result: DiagnosticResult) -> Dict[str, Any]:
        """
        Render an AR visualization based on the 3D model and diagnostic results.
        
        Args:
            model_info: The 3D model information from generate_ear_model
            diagnostic_result: Diagnostic results to visualize
            
        Returns:
            Dictionary with AR visualization information and file paths
        """
        if not self.enabled or not self.ar_mode or self.ar_renderer is None:
            self.logger.warning("AR visualization is disabled")
            return {"error": "AR visualization is disabled"}
        
        self.logger.info("Rendering AR visualization")
        
        try:
            # Set up the AR scene
            ar_scene = self.ar_renderer.create_scene()
            
            # Add the ear model to the scene
            self.ar_renderer.add_model_to_scene(
                scene=ar_scene,
                model_info=model_info
            )
            
            # Add diagnostic annotations
            ar_scene = self.ar_renderer.add_diagnostic_annotations(
                scene=ar_scene,
                diagnostic_result=diagnostic_result
            )
            
            # Render the scene
            timestamp = int(time.time())
            output_file = os.path.join(self.output_dir, f"ar_visualization_{timestamp}")
            
            render_result = self.ar_renderer.render_scene(
                scene=ar_scene,
                output_path=output_file
            )
            
            result = {
                "visualization_info": render_result,
                "timestamp": timestamp
            }
            
            self.logger.info(f"AR visualization rendered successfully: {render_result['file_path']}")
            return result
            
        except Exception as e:
            self.logger.error(f"Error rendering AR visualization: {str(e)}")
            return {"error": str(e)}
    
    def generate_complete_visualization(self,
                                     image_data: np.ndarray,
                                     diagnostic_result: DiagnosticResult) -> Dict[str, Any]:
        """
        Generate a complete visualization including 3D model and AR rendering.
        
        Args:
            image_data: Tympanic membrane image data
            diagnostic_result: Diagnostic results to visualize
            
        Returns:
            Dictionary with complete visualization information
        """
        # Generate the 3D model
        model_result = self.generate_ear_model(
            image_data=image_data,
            diagnostic_result=diagnostic_result
        )
        
        if "error" in model_result:
            return model_result
        
        # If AR mode is enabled, also generate the AR visualization
        ar_result = {}
        if self.ar_mode and self.ar_renderer is not None:
            ar_result = self.render_ar_visualization(
                model_info=model_result["model_info"],
                diagnostic_result=diagnostic_result
            )
        
        # Combine the results
        result = {
            "model": model_result,
            "ar_visualization": ar_result if ar_result else None,
            "timestamp": int(time.time())
        }
        
        return result
    
    def save_visualization_metadata(self, visualization_data: Dict[str, Any], id: str) -> str:
        """
        Save visualization metadata for later reference.
        
        Args:
            visualization_data: Visualization data to save
            id: Unique identifier for the visualization
            
        Returns:
            Path to the saved metadata file
        """
        metadata_dir = os.path.join(self.output_dir, "metadata")
        os.makedirs(metadata_dir, exist_ok=True)
        
        file_path = os.path.join(metadata_dir, f"visualization_{id}.json")
        
        # Remove large binary data before saving
        metadata = {}
        for key, value in visualization_data.items():
            if key != "model_info" and key != "raw_data":
                metadata[key] = value
        
        with open(file_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        self.logger.info(f"Visualization metadata saved to {file_path}")
        return file_path
    
    def load_visualization(self, id: str) -> Dict[str, Any]:
        """
        Load a previously saved visualization by ID.
        
        Args:
            id: Unique identifier for the visualization
            
        Returns:
            Visualization data
        """
        file_path = os.path.join(self.output_dir, "metadata", f"visualization_{id}.json")
        
        if not os.path.exists(file_path):
            self.logger.error(f"Visualization metadata not found: {file_path}")
            return {"error": "Visualization not found"}
        
        try:
            with open(file_path, 'r') as f:
                metadata = json.load(f)
            
            self.logger.info(f"Visualization metadata loaded from {file_path}")
            return metadata
        except Exception as e:
            self.logger.error(f"Error loading visualization metadata: {str(e)}")
            return {"error": str(e)}
    
    def get_available_visualizations(self) -> List[Dict[str, Any]]:
        """
        Get a list of all available visualizations.
        
        Returns:
            List of visualization metadata summaries
        """
        metadata_dir = os.path.join(self.output_dir, "metadata")
        if not os.path.exists(metadata_dir):
            return []
        
        visualizations = []
        for filename in os.listdir(metadata_dir):
            if filename.endswith(".json"):
                try:
                    with open(os.path.join(metadata_dir, filename), 'r') as f:
                        metadata = json.load(f)
                    
                    summary = {
                        "id": filename.replace("visualization_", "").replace(".json", ""),
                        "timestamp": metadata.get("timestamp", 0),
                        "type": "AR" if metadata.get("ar_visualization") else "3D"
                    }
                    
                    visualizations.append(summary)
                except Exception as e:
                    self.logger.error(f"Error loading metadata from {filename}: {str(e)}")
        
        # Sort by timestamp (newest first)
        visualizations.sort(key=lambda x: x["timestamp"], reverse=True)
        
        return visualizations
    
    def update_ar_settings(self, new_settings: Dict[str, Any]) -> bool:
        """
        Update AR visualization settings.
        
        Args:
            new_settings: New AR settings to apply
            
        Returns:
            True if settings were updated successfully, False otherwise
        """
        if not self.enabled or not self.ar_mode or self.ar_renderer is None:
            self.logger.warning("AR visualization is disabled, cannot update settings")
            return False
        
        try:
            self.ar_renderer.update_settings(new_settings)
            self.logger.info("AR settings updated successfully")
            return True
        except Exception as e:
            self.logger.error(f"Error updating AR settings: {str(e)}")
            return False
    
    def generate_animation(self, 
                         model_info: Dict[str, Any], 
                         animation_type: str,
                         duration_seconds: float = 5.0) -> Dict[str, Any]:
        """
        Generate an animation of the 3D model.
        
        Args:
            model_info: The 3D model information
            animation_type: Type of animation (e.g., 'rotation', 'section_cut', 'fluid_flow')
            duration_seconds: Duration of the animation in seconds
            
        Returns:
            Dictionary with animation information and file path
        """
        if not self.enabled or self.model_generator is None:
            self.logger.warning("Visualization is disabled, cannot generate animation")
            return {"error": "Visualization is disabled"}
        
        self.logger.info(f"Generating {animation_type} animation")
        
        try:
            timestamp = int(time.time())
            output_file = os.path.join(self.output_dir, f"animation_{animation_type}_{timestamp}")
            
            animation_result = self.model_generator.generate_animation(
                model_info=model_info,
                animation_type=animation_type,
                output_path=output_file,
                duration_seconds=duration_seconds
            )
            
            result = {
                "animation_info": animation_result,
                "animation_type": animation_type,
                "duration_seconds": duration_seconds,
                "timestamp": timestamp
            }
            
            self.logger.info(f"Animation generated successfully: {animation_result['file_path']}")
            return result
            
        except Exception as e:
            self.logger.error(f"Error generating animation: {str(e)}")
            return {"error": str(e)}
