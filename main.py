#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
OtitisMediaAI: Multi-Sensor Fusion and AI-Based Otitis Media Diagnosis System
Main application entry point
"""

import os
import sys
import argparse
import logging
from datetime import datetime
import json
import traceback

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data_collection.sensor_manager import SensorManager
from preprocessing.data_preprocessor import DataPreprocessor
from models.transformer_model import TransformerModel
from visualization.visualization_manager import VisualizationManager
from prognosis.prognosis_predictor import PrognosisPredictor
from ui.application_interface import ApplicationInterface
from utils.config_manager import ConfigManager
from utils.logger import setup_logger


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='OtitisMediaAI Diagnostic System')
    
    parser.add_argument('--config', type=str, default='config/default.json',
                        help='Path to configuration file')
    parser.add_argument('--mode', type=str, choices=['clinical', 'research', 'simulation', 'training'],
                        default='clinical', help='Operation mode')
    parser.add_argument('--log-level', type=str, choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                        default='INFO', help='Logging level')
    parser.add_argument('--device', type=str, default=None,
                        help='Sensor device path (if applicable)')
    parser.add_argument('--output-dir', type=str, default='output',
                        help='Directory for output files')
    parser.add_argument('--disable-cloud', action='store_true',
                        help='Disable cloud connectivity')
    parser.add_argument('--disable-visualization', action='store_true',
                        help='Disable 3D/AR visualization')
    parser.add_argument('--load-model', type=str, default=None,
                        help='Path to pre-trained model weights')
    
    return parser.parse_args()


def main():
    """Main application entry point."""
    # Parse arguments
    args = parse_arguments()
    
    # Setup logging
    log_dir = os.path.join(args.output_dir, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f'otitismedia_ai_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    logger = setup_logger('otitismedia_ai', log_file, level=getattr(logging, args.log_level))
    
    logger.info(f"Starting OtitisMediaAI in {args.mode} mode")
    
    try:
        # Load configuration
        config_manager = ConfigManager(args.config)
        config = config_manager.get_config()
        
        # Override configuration with command line arguments
        if args.device:
            config['sensor_config']['device_path'] = args.device
        if args.disable_cloud:
            config['cloud_config']['enabled'] = False
        if args.disable_visualization:
            config['visualization_config']['enabled'] = False
        if args.load_model:
            config['model_config']['weights_path'] = args.load_model
        
        # Initialize components
        logger.info("Initializing system components")
        
        # Sensor manager for data collection
        sensor_manager = SensorManager(
            config['sensor_config'],
            mode=args.mode
        )
        
        # Data preprocessor
        preprocessor = DataPreprocessor(
            config['preprocessing_config']
        )
        
        # AI model
        model = TransformerModel(
            config['model_config']
        )
        
        # Visualization manager
        visualization_manager = VisualizationManager(
            config['visualization_config']
        )
        
        # Prognosis predictor
        prognosis_predictor = PrognosisPredictor(
            config['prognosis_config']
        )
        
        # User interface
        app_interface = ApplicationInterface(
            sensor_manager=sensor_manager,
            preprocessor=preprocessor,
            model=model,
            visualization_manager=visualization_manager,
            prognosis_predictor=prognosis_predictor,
            config=config,
            mode=args.mode
        )
        
        # Start the application
        logger.info("Starting application interface")
        app_interface.run()
        
    except Exception as e:
        logger.error(f"Error during initialization: {str(e)}")
        logger.error(traceback.format_exc())
        print(f"Error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
