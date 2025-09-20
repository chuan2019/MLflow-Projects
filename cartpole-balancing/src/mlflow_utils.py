import mlflow
import mlflow.pytorch
import mlflow.sklearn
import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime
import torch


class MLflowTracker:
    """MLflow integration for DQN training tracking."""
    
    def __init__(self, experiment_name="cartpole-dqn", tracking_uri=None):
        self.experiment_name = experiment_name
        
        # Set tracking URI
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        
        # Set or create experiment
        try:
            experiment = mlflow.get_experiment_by_name(experiment_name)
            if experiment is None:
                experiment_id = mlflow.create_experiment(experiment_name)
            else:
                experiment_id = experiment.experiment_id
            mlflow.set_experiment(experiment_name)
        except Exception as e:
            print(f"Warning: Could not set experiment: {e}")
    
    def start_run(self, run_name=None):
        """Start a new MLflow run."""
        if run_name is None:
            run_name = f"dqn_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        mlflow.start_run(run_name=run_name)
        return mlflow.active_run()
    
    def log_hyperparameters(self, params):
        """Log hyperparameters to MLflow."""
        for key, value in params.items():
            mlflow.log_param(key, value)
    
    def log_metrics(self, metrics, step=None):
        """Log metrics to MLflow."""
        for key, value in metrics.items():
            mlflow.log_metric(key, value, step=step)
    
    def log_episode_metrics(self, episode, reward, epsilon, loss=None, success=None):
        """Log episode-specific metrics."""
        metrics = {
            "episode_reward": reward,
            "epsilon": epsilon
        }
        
        if loss is not None:
            metrics["loss"] = loss
        
        if success is not None:
            metrics["success"] = success
        
        self.log_metrics(metrics, step=episode)
    
    def log_model(self, agent, model_name="dqn_model"):
        """Log the trained model to MLflow."""
        # Save model locally first
        model_path = f"models/{model_name}.pth"
        os.makedirs("models", exist_ok=True)
        agent.save_model(model_path)
        
        # Log model to MLflow
        mlflow.pytorch.log_model(
            agent.q_network,
            "model",
            registered_model_name=model_name
        )
        
        # Also log the model file as artifact
        mlflow.log_artifact(model_path, "model_files")
    
    def log_reward_plot(self, rewards, window_size=100):
        """Create and log reward plot."""
        plt.figure(figsize=(12, 6))
        
        # Plot raw rewards
        plt.subplot(1, 2, 1)
        plt.plot(rewards)
        plt.title('Episode Rewards')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.grid(True)
        
        # Plot moving average
        plt.subplot(1, 2, 2)
        if len(rewards) >= window_size:
            moving_avg = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
            plt.plot(range(window_size-1, len(rewards)), moving_avg)
        plt.title(f'Moving Average Rewards (window={window_size})')
        plt.xlabel('Episode')
        plt.ylabel('Average Reward')
        plt.grid(True)
        
        plt.tight_layout()
        
        # Save and log plot
        plot_path = "plots/reward_plot.png"
        os.makedirs("plots", exist_ok=True)
        plt.savefig(plot_path)
        mlflow.log_artifact(plot_path, "plots")
        plt.close()
    
    def log_loss_plot(self, losses):
        """Create and log loss plot."""
        if not losses:
            return
            
        plt.figure(figsize=(10, 6))
        plt.plot(losses)
        plt.title('Training Loss')
        plt.xlabel('Training Step')
        plt.ylabel('Loss')
        plt.grid(True)
        
        # Save and log plot
        plot_path = "plots/loss_plot.png"
        os.makedirs("plots", exist_ok=True)
        plt.savefig(plot_path)
        mlflow.log_artifact(plot_path, "plots")
        plt.close()
    
    def log_performance_summary(self, rewards, solved_threshold=195, window_size=100):
        """Log performance summary metrics."""
        if len(rewards) == 0:
            return
        
        # Calculate summary statistics
        total_episodes = len(rewards)
        max_reward = max(rewards)
        min_reward = min(rewards)
        avg_reward = np.mean(rewards)
        std_reward = np.std(rewards)
        
        # Calculate success rate (if solved threshold is met)
        success_episodes = sum(1 for r in rewards if r >= solved_threshold)
        success_rate = success_episodes / total_episodes if total_episodes > 0 else 0
        
        # Calculate episodes to solve (first time hitting solved threshold consistently)
        episodes_to_solve = None
        if len(rewards) >= window_size:
            for i in range(window_size, len(rewards) + 1):
                window_avg = np.mean(rewards[i-window_size:i])
                if window_avg >= solved_threshold:
                    episodes_to_solve = i
                    break
        
        # Log summary metrics
        summary_metrics = {
            "total_episodes": total_episodes,
            "max_reward": max_reward,
            "min_reward": min_reward,
            "avg_reward": avg_reward,
            "std_reward": std_reward,
            "success_rate": success_rate,
            "final_100_avg": np.mean(rewards[-100:]) if len(rewards) >= 100 else avg_reward
        }
        
        if episodes_to_solve:
            summary_metrics["episodes_to_solve"] = episodes_to_solve
        
        for key, value in summary_metrics.items():
            mlflow.log_metric(f"summary_{key}", value)
    
    def end_run(self):
        """End the current MLflow run."""
        mlflow.end_run()
    
    def log_environment_info(self, env_name, observation_space, action_space):
        """Log environment information."""
        mlflow.log_param("environment", env_name)
        mlflow.log_param("observation_space", str(observation_space))
        mlflow.log_param("action_space", str(action_space))


def get_best_model_uri(experiment_name="cartpole-dqn", metric_name="summary_avg_reward"):
    """Get the URI of the best model based on a metric."""
    try:
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if not experiment:
            return None
        
        # Search for runs in the experiment
        runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=[f"metrics.{metric_name} DESC"],
            max_results=1
        )
        
        if len(runs) > 0:
            best_run_id = runs.iloc[0]['run_id']
            return f"runs:/{best_run_id}/model"
        
    except Exception as e:
        print(f"Error finding best model: {e}")
    
    return None