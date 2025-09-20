#!/usr/bin/env python3
"""
CartPole DQN Model Serving API
"""

import os
import torch
import numpy as np
import mlflow
import mlflow.pytorch
from flask import Flask, request, jsonify
import gymnasium as gym
from dotenv import load_dotenv
import json
import logging

from dqn_agent import DQNAgent, DQN
from mlflow_utils import get_best_model_uri

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)

# Global model variable
model = None
agent = None


class CartPolePredictor:
    """CartPole model predictor wrapper."""
    
    def __init__(self, model_path=None, model_uri=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.agent = None
        
        if model_path and os.path.exists(model_path):
            self.load_from_path(model_path)
        elif model_uri:
            self.load_from_mlflow(model_uri)
        else:
            raise ValueError("Either model_path or model_uri must be provided")
    
    def load_from_path(self, model_path):
        """Load model from local path."""
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # Create agent and load model
            self.agent = DQNAgent(
                state_size=checkpoint.get('state_size', 4),
                action_size=checkpoint.get('action_size', 2)
            )
            self.agent.load_model(model_path)
            self.model = self.agent.q_network
            
            logger.info(f"Model loaded successfully from {model_path}")
            
        except Exception as e:
            logger.error(f"Error loading model from path: {e}")
            raise
    
    def load_from_mlflow(self, model_uri):
        """Load model from MLflow."""
        try:
            # Load model from MLflow
            self.model = mlflow.pytorch.load_model(model_uri)
            
            # Create agent wrapper
            self.agent = DQNAgent(state_size=4, action_size=2)
            self.agent.q_network = self.model
            
            logger.info(f"Model loaded successfully from MLflow: {model_uri}")
            
        except Exception as e:
            logger.error(f"Error loading model from MLflow: {e}")
            raise
    
    def predict(self, state):
        """Predict action for given state."""
        if self.agent is None:
            raise ValueError("Model not loaded")
        
        try:
            # Ensure state is numpy array
            if isinstance(state, list):
                state = np.array(state)
            
            # Predict action (no exploration)
            action = self.agent.act(state, training=False)
            
            # Get Q-values for additional info
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_values = self.model(state_tensor).cpu().numpy()[0]
            
            return {
                'action': int(action),
                'q_values': q_values.tolist(),
                'confidence': float(np.max(q_values))
            }
            
        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            raise
    
    def evaluate_episode(self, max_steps=500, render=False):
        """Evaluate model on a single episode."""
        try:
            env = gym.make('CartPole-v1', render_mode="human" if render else None)
            state, _ = env.reset()
            total_reward = 0
            steps = 0
            
            while steps < max_steps:
                action = self.agent.act(state, training=False)
                state, reward, terminated, truncated, _ = env.step(action)
                total_reward += reward
                steps += 1
                
                if terminated or truncated:
                    break
            
            env.close()
            
            return {
                'total_reward': total_reward,
                'steps': steps,
                'success': total_reward >= 195
            }
            
        except Exception as e:
            logger.error(f"Error during evaluation: {e}")
            raise


def load_model():
    """Load the trained model."""
    global model, agent
    
    try:
        # Try to load from environment variable path first
        model_path = os.getenv('MODEL_PATH', 'models/best_model.pth')
        
        if os.path.exists(model_path):
            predictor = CartPolePredictor(model_path=model_path)
            logger.info(f"Model loaded from local path: {model_path}")
        else:
            # Try to load best model from MLflow
            tracking_uri = os.getenv('MLFLOW_TRACKING_URI')
            if tracking_uri:
                mlflow.set_tracking_uri(tracking_uri)
                
            model_uri = get_best_model_uri()
            if model_uri:
                predictor = CartPolePredictor(model_uri=model_uri)
                logger.info(f"Model loaded from MLflow: {model_uri}")
            else:
                raise ValueError("No model found in MLflow")
        
        return predictor
        
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return None


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None
    })


@app.route('/predict', methods=['POST'])
def predict():
    """Prediction endpoint."""
    global model
    
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    try:
        data = request.get_json()
        
        if 'state' not in data:
            return jsonify({'error': 'Missing state in request'}), 400
        
        state = data['state']
        
        # Validate state format
        if not isinstance(state, (list, np.ndarray)) or len(state) != 4:
            return jsonify({'error': 'State must be a list/array of 4 numbers'}), 400
        
        # Make prediction
        result = model.predict(state)
        
        return jsonify({
            'success': True,
            'prediction': result,
            'state': state
        })
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/evaluate', methods=['POST'])
def evaluate():
    """Evaluation endpoint."""
    global model
    
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    try:
        data = request.get_json() or {}
        max_steps = data.get('max_steps', 500)
        render = data.get('render', False)
        
        # Run evaluation
        result = model.evaluate_episode(max_steps=max_steps, render=render)
        
        return jsonify({
            'success': True,
            'evaluation': result
        })
        
    except Exception as e:
        logger.error(f"Evaluation error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/model_info', methods=['GET'])
def model_info():
    """Get model information."""
    global model
    
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    try:
        return jsonify({
            'success': True,
            'model_info': {
                'state_size': 4,
                'action_size': 2,
                'device': str(model.device),
                'model_type': 'DQN',
                'environment': 'CartPole-v1'
            }
        })
        
    except Exception as e:
        logger.error(f"Model info error: {e}")
        return jsonify({'error': str(e)}), 500


@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404


@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500


def main():
    """Main serving function."""
    global model
    
    # Load model
    logger.info("Loading model...")
    model = load_model()
    
    if model is None:
        logger.error("Failed to load model. Exiting.")
        return
    
    logger.info("Model loaded successfully!")
    
    # Start Flask app
    host = os.getenv('SERVE_HOST', '0.0.0.0')
    port = int(os.getenv('SERVE_PORT', 8080))
    debug = os.getenv('DEBUG', 'false').lower() == 'true'
    
    logger.info(f"Starting model server on {host}:{port}")
    app.run(host=host, port=port, debug=debug)


if __name__ == "__main__":
    main()