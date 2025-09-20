#!/usr/bin/env python3
"""
Simple example script showing how to use the trained CartPole model
"""

import requests
import json
import numpy as np
import time


def test_model_api(base_url="http://localhost:8080"):
    """Test the CartPole model API."""
    
    print("Testing CartPole DQN Model API")
    print("=" * 50)
    
    # Test health check
    print("\n1. Health Check:")
    try:
        response = requests.get(f"{base_url}/health")
        print(f"Status: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
    except Exception as e:
        print(f"Health check failed: {e}")
        return
    
    # Test model info
    print("\n2. Model Information:")
    try:
        response = requests.get(f"{base_url}/model_info")
        print(f"Status: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
    except Exception as e:
        print(f"Model info failed: {e}")
    
    # Test predictions with different states
    print("\n3. Prediction Tests:")
    
    test_states = [
        [0.0, 0.0, 0.0, 0.0],  # Balanced state
        [0.1, 0.5, 0.1, 0.2],  # Slightly off balance
        [0.3, -0.2, 0.2, -0.1], # Different configuration
        [-0.1, 0.1, -0.05, 0.3], # Another state
    ]
    
    for i, state in enumerate(test_states, 1):
        print(f"\n  Test {i} - State: {state}")
        try:
            response = requests.post(
                f"{base_url}/predict",
                json={"state": state},
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code == 200:
                result = response.json()
                prediction = result['prediction']
                print(f"    Action: {prediction['action']} ({'Left' if prediction['action'] == 0 else 'Right'})")
                print(f"    Q-values: {[f'{q:.3f}' for q in prediction['q_values']]}")
                print(f"    Confidence: {prediction['confidence']:.3f}")
            else:
                print(f"    Prediction failed: {response.status_code}")
                print(f"    Response: {response.text}")
                
        except Exception as e:
            print(f"    Error: {e}")
    
    # Test model evaluation
    print("\n4. Model Evaluation:")
    try:
        response = requests.post(
            f"{base_url}/evaluate",
            json={"max_steps": 200, "render": False},
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            result = response.json()
            evaluation = result['evaluation']
            print(f"Total Reward: {evaluation['total_reward']}")
            print(f"Steps: {evaluation['steps']}")
            print(f"Success: {'Yes' if evaluation['success'] else 'No'}")
        else:
            print(f"Evaluation failed: {response.status_code}")
            
    except Exception as e:
        print(f"Evaluation error: {e}")
    
    print("\n" + "=" * 50)
    print("API testing completed!")


def simulate_cartpole_episode(base_url="http://localhost:8080", max_steps=200):
    """Simulate a CartPole episode using the API."""
    
    print("\nSimulating CartPole Episode")
    print("-" * 30)
    
    # Simple CartPole simulation (approximation)
    state = [0.0, 0.0, 0.1, 0.0]  # Start with slight pole angle
    total_reward = 0
    
    for step in range(max_steps):
        # Get action from model
        try:
            response = requests.post(
                f"{base_url}/predict",
                json={"state": state},
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code != 200:
                print(f"Prediction failed at step {step}")
                break
                
            action = response.json()['prediction']['action']
            
            # Simple state update (this is a rough approximation)
            # In reality, this would be handled by the Gym environment
            cart_pos, cart_vel, pole_angle, pole_vel = state
            
            # Apply action (simplified physics)
            force = 1.0 if action == 1 else -1.0
            cart_vel += force * 0.1
            cart_pos += cart_vel * 0.02
            pole_vel += np.sin(pole_angle) * 0.1
            pole_angle += pole_vel * 0.02
            
            # Update state
            state = [cart_pos, cart_vel, pole_angle, pole_vel]
            
            # Check if episode should end (simplified)
            if abs(cart_pos) > 2.4 or abs(pole_angle) > 0.2:
                print(f"Episode ended at step {step} - pole fell or cart out of bounds")
                break
            
            total_reward += 1
            
            if step % 50 == 0:
                print(f"Step {step}: Action={action}, Angle={pole_angle:.3f}, Reward={total_reward}")
                
        except Exception as e:
            print(f"Error at step {step}: {e}")
            break
    
    print(f"\nEpisode Summary:")
    print(f"Total Steps: {step + 1}")
    print(f"Total Reward: {total_reward}")
    print(f"Success: {'Yes' if total_reward >= 195 else 'No'}")


def main():
    """Main function."""
    print("CartPole DQN Model Testing")
    print("Make sure the model server is running: docker-compose --profile serving up model-server")
    print("Or locally: python src/serve.py")
    
    # Wait a moment for user to read
    time.sleep(2)
    
    # Test the API
    test_model_api()
    
    # Simulate an episode
    simulate_cartpole_episode()


if __name__ == "__main__":
    main()