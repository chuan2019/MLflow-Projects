#!/usr/bin/env python3
"""
CartPole DQN Training Script with MLflow Integration
"""

import os
import gymnasium as gym
import numpy as np
import torch
from dotenv import load_dotenv
import argparse
import json
from datetime import datetime

from dqn_agent import DQNAgent
from mlflow_utils import MLflowTracker

# Load environment variables
load_dotenv()


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train DQN agent on CartPole')
    parser.add_argument('--episodes', type=int, default=int(os.getenv('EPISODES', 1000)),
                        help='Number of training episodes')
    parser.add_argument('--lr', type=float, default=float(os.getenv('LEARNING_RATE', 0.001)),
                        help='Learning rate')
    parser.add_argument('--batch-size', type=int, default=int(os.getenv('BATCH_SIZE', 32)),
                        help='Batch size for training')
    parser.add_argument('--memory-size', type=int, default=int(os.getenv('MEMORY_SIZE', 10000)),
                        help='Replay buffer size')
    parser.add_argument('--target-update', type=int, default=int(os.getenv('TARGET_UPDATE', 10)),
                        help='Target network update frequency')
    parser.add_argument('--epsilon-start', type=float, default=float(os.getenv('EPSILON_START', 1.0)),
                        help='Starting epsilon value')
    parser.add_argument('--epsilon-end', type=float, default=float(os.getenv('EPSILON_END', 0.01)),
                        help='Final epsilon value')
    parser.add_argument('--epsilon-decay', type=float, default=float(os.getenv('EPSILON_DECAY', 0.995)),
                        help='Epsilon decay rate')
    parser.add_argument('--no-mlflow', action='store_true',
                        help='Disable MLflow tracking')
    parser.add_argument('--render', action='store_true',
                        help='Render environment during training')
    parser.add_argument('--save-model', type=str, default='models/best_model.pth',
                        help='Path to save the best model')
    
    return parser.parse_args()


def train_dqn_agent(args, tracker=None):
    """Train the DQN agent on CartPole environment."""
    
    # Create environment
    render_mode = "human" if args.render else None
    env = gym.make('CartPole-v1', render_mode=render_mode)
    
    # Get environment dimensions
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    
    print(f"Environment: CartPole-v1")
    print(f"State size: {state_size}")
    print(f"Action size: {action_size}")
    print(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    
    # Create DQN agent
    agent = DQNAgent(
        state_size=state_size,
        action_size=action_size,
        lr=args.lr,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        epsilon_decay=args.epsilon_decay,
        memory_size=args.memory_size,
        batch_size=args.batch_size,
        target_update=args.target_update
    )
    
    # Log hyperparameters to MLflow
    if tracker:
        hyperparams = {
            'learning_rate': args.lr,
            'batch_size': args.batch_size,
            'memory_size': args.memory_size,
            'target_update': args.target_update,
            'epsilon_start': args.epsilon_start,
            'epsilon_end': args.epsilon_end,
            'epsilon_decay': args.epsilon_decay,
            'episodes': args.episodes,
            'gamma': agent.gamma,
            'state_size': state_size,
            'action_size': action_size
        }
        tracker.log_hyperparameters(hyperparams)
        tracker.log_environment_info('CartPole-v1', env.observation_space, env.action_space)
    
    # Training variables
    rewards_per_episode = []
    losses = []
    best_avg_reward = -float('inf')
    solved = False
    solved_episode = None
    
    print(f"\nStarting training for {args.episodes} episodes...")
    print("-" * 50)
    
    for episode in range(args.episodes):
        state, _ = env.reset()
        total_reward = 0
        episode_losses = []
        
        while True:
            # Choose action
            action = agent.act(state)
            
            # Take action
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # Store experience
            agent.remember(state, action, reward, next_state, done)
            
            # Train agent
            loss = agent.replay()
            if loss is not None:
                episode_losses.append(loss)
            
            state = next_state
            total_reward += reward
            
            if done:
                break
        
        # Update target network
        if episode % args.target_update == 0:
            agent.update_target_network()
        
        # Store metrics
        rewards_per_episode.append(total_reward)
        if episode_losses:
            avg_loss = np.mean(episode_losses)
            losses.append(avg_loss)
        else:
            avg_loss = None
        
        # Check if solved (average reward over last 100 episodes >= 195)
        if len(rewards_per_episode) >= 100:
            avg_reward = np.mean(rewards_per_episode[-100:])
            if avg_reward >= 195.0 and not solved:
                solved = True
                solved_episode = episode
                print(f"\nEnvironment solved in {episode} episodes!")
                print(f"Average reward over last 100 episodes: {avg_reward:.2f}")
        else:
            avg_reward = np.mean(rewards_per_episode)
        
        # Save best model
        if avg_reward > best_avg_reward:
            best_avg_reward = avg_reward
            os.makedirs(os.path.dirname(args.save_model), exist_ok=True)
            agent.save_model(args.save_model)
        
        # Log metrics to MLflow
        if tracker:
            tracker.log_episode_metrics(
                episode=episode,
                reward=total_reward,
                epsilon=agent.epsilon,
                loss=avg_loss,
                success=1 if total_reward >= 195 else 0
            )
        
        # Print progress
        if episode % 100 == 0 or episode == args.episodes - 1:
            print(f"Episode {episode:4d} | "
                  f"Reward: {total_reward:6.1f} | "
                  f"Avg Reward: {avg_reward:6.1f} | "
                  f"Epsilon: {agent.epsilon:.3f} | "
                  f"Loss: {avg_loss:.4f if avg_loss else 'N/A':>6}")
    
    env.close()
    
    # Log final results
    print(f"\nTraining completed!")
    print(f"Best average reward: {best_avg_reward:.2f}")
    if solved:
        print(f"Environment solved in episode {solved_episode}")
    else:
        print("Environment not solved during training")
    
    return agent, rewards_per_episode, losses, solved


def main():
    """Main training function."""
    args = parse_args()
    
    # Initialize MLflow tracker
    tracker = None
    if not args.no_mlflow:
        try:
            tracking_uri = os.getenv('MLFLOW_TRACKING_URI')
            experiment_name = os.getenv('MLFLOW_EXPERIMENT_NAME', 'cartpole-dqn')
            tracker = MLflowTracker(experiment_name, tracking_uri)
            
            # Start MLflow run
            run_name = f"dqn_cartpole_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            tracker.start_run(run_name)
            print(f"MLflow tracking enabled - Experiment: {experiment_name}")
            
        except Exception as e:
            print(f"Warning: MLflow tracking disabled due to error: {e}")
            tracker = None
    
    try:
        # Train the agent
        agent, rewards, losses, solved = train_dqn_agent(args, tracker)
        
        # Log final artifacts to MLflow
        if tracker:
            # Log performance plots
            tracker.log_reward_plot(rewards)
            if losses:
                tracker.log_loss_plot(losses)
            
            # Log performance summary
            tracker.log_performance_summary(rewards)
            
            # Log the trained model
            tracker.log_model(agent, "cartpole_dqn_model")
            
            print(f"MLflow artifacts logged successfully")
    
    except Exception as e:
        print(f"Training failed: {e}")
        raise
    
    finally:
        # End MLflow run
        if tracker:
            tracker.end_run()


if __name__ == "__main__":
    main()