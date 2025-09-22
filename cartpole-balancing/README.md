# CartPole DQN with MLflow Integration

A complete reinforcement learning project implementing Deep Q-Network (DQN) for the CartPole environment with MLflow experiment tracking and Docker deployment.

## Project Overview

This project demonstrates how to:
- Implement a DQN algorithm for the CartPole environment
- Track experiments using MLflow
- Deploy models using Docker Compose
- Serve trained models via REST API

## Project Structure

```
mlflow-project/
├── src/
│   ├── dqn_agent.py      # DQN implementation and agent
│   ├── mlflow_utils.py   # MLflow tracking utilities
│   ├── train.py          # Training script
│   └── serve.py          # Model serving API
├── models/               # Saved models directory
├── notebooks/            # Jupyter notebooks for exploration
├── docker-compose.yml    # Docker services configuration
├── Dockerfile.trainer    # Training container
├── Dockerfile.server     # Serving container
├── requirements.txt      # Python dependencies
├── .env                  # Environment variables
└── README.md            # This file
```

## Quick Start

### Prerequisites

- Docker and Docker Compose
- GNU Make
- Python 3.9+ (optional, for local development only)

### Simple Workflow

```bash
# 1. Setup workspace
make workspace-setup

# 2. Train the model  
make train

# 3. Serve the model
make serve

# 4. Test the API
make test-api

# Stop everything when done
make stop
```

### See All Available Commands

```bash
make help
```

### 1. Setup Workspace

```bash
# Complete workspace setup (build images + start MLflow)
make workspace-setup

# Or just start MLflow tracking server
make setup

# MLflow UI will be available at http://localhost:5000
```

### 2. Development & Training

This project supports two development approaches:

#### Option A: Docker-Only (Recommended for Testing/Production)

**When to use**: Testing, CI/CD, production deployment, or when you want guaranteed environment consistency.

```bash
# Train using Docker (recommended)
make train

# Quick training with fewer episodes for testing
make quick-train

# No local Python installation required!
```

**Pros**: 
- Identical environment across all machines
- No local dependency management
- Perfect for testing and production

#### Option B: Local Development with Virtual Environment

**When to use**: Active development, debugging, IDE integration, quick iteration.

```bash
# Optional: Install dependencies locally for IDE support
make install

# Train locally (for development)
make train-local
```

**Pros**: 
- Full IDE support (code completion, debugging)
- Faster iteration for small changes
- Easy debugging and profiling

**Note**: Local virtual environment is **optional** and primarily for IDE support. All testing and deployment should use Docker.

### 3. Serve the Model

```bash
# Start model serving using Docker
make serve

# Or serve locally (for development)
make serve-local

# API will be available at http://localhost:8080
```

### 4. Test the Complete Pipeline

```bash
# Test API endpoints
make test-api

# Run comprehensive tests (setup + train + serve + test)
make test-all

# Quick demo of everything
make demo
```

## Available Make Commands

Run `make help` to see all available commands:

```bash
make help
```

**Key Commands:**
- `make workspace-setup` - Complete workspace setup
- `make train` - Train model using Docker
- `make serve` - Serve model using Docker  
- `make test-api` - Test API endpoints
- `make test-all` - Full integration test
- `make status` - Show service status
- `make logs` - Show service logs
- `make stop` - Stop all services
- `make clean` - Clean up containers and volumes, prune Docker system resources, and remove the Python virtual environment

## Common Workflows

### First Time Setup
```bash
make workspace-setup    # Build images and start MLflow
make train              # Train the model
make serve              # Start serving
make test-api           # Verify everything works
```

### Development Cycle
```bash
make train              # Train with new changes
make serve              # Update serving
make test-api           # Test the changes
```

### Quick Testing
```bash
make demo               # Complete automated demo
```

### Troubleshooting
```bash
make status             # Check service status
make logs               # View logs
make restart            # Restart services
make clean              # Clean up and restart fresh
```

## Environment Details

- **Environment**: CartPole-v1 from OpenAI Gymnasium
- **State Space**: 4 continuous variables (cart position, cart velocity, pole angle, pole angular velocity)
- **Action Space**: 2 discrete actions (move left or right)
- **Goal**: Balance the pole for 200+ steps
- **Solved Threshold**: Average reward ≥ 195 over 100 consecutive episodes

## Algorithm Details

### Deep Q-Network (DQN)

- **Network Architecture**: 
  - Input: 4 state variables
  - Hidden layers: 2 x 128 neurons with ReLU activation
  - Output: 2 Q-values (one per action)

- **Key Features**:
  - Experience replay buffer
  - Target network for stable training
  - Epsilon-greedy exploration
  - Gradient clipping for stability

## MLflow Integration

The project tracks the following metrics and artifacts:

### Hyperparameters
- Learning rate, batch size, memory size
- Epsilon decay parameters
- Network architecture details

### Metrics
- Episode rewards
- Training loss
- Success rate
- Epsilon values

### Artifacts
- Trained model weights
- Reward and loss plots
- Performance summaries

### Model Registry
- Best models are automatically registered
- Model versioning and comparison

## Docker Services

### MLflow Tracking Server
- **Port**: 5000
- **Purpose**: Experiment tracking and model registry
- **UI**: http://localhost:5000

### Training Container
- **Profile**: `training`
- **Purpose**: Train DQN models
- **Volumes**: Shared models and MLflow artifacts

### Model Server
- **Port**: 8080
- **Purpose**: Serve trained models via REST API
- **Profile**: `serving`

## API Endpoints

### Health Check
```bash
GET /health
```

### Predict Action
```bash
POST /predict
Content-Type: application/json

{
  "state": [0.1, 0.2, 0.3, 0.4]
}
```

### Evaluate Model
```bash
POST /evaluate
Content-Type: application/json

{
  "max_steps": 500,
  "render": false
}
```

### Model Information
```bash
GET /model_info
```

## Configuration

### Environment Variables (.env)

```bash
# MLflow Configuration
MLFLOW_TRACKING_URI=http://localhost:5000
MLFLOW_EXPERIMENT_NAME=cartpole-dqn

# Training Parameters
EPISODES=1000
LEARNING_RATE=0.001
BATCH_SIZE=32
MEMORY_SIZE=10000
TARGET_UPDATE=10
EPSILON_START=1.0
EPSILON_END=0.01
EPSILON_DECAY=0.995
```

### Training Arguments

```bash
python src/train.py --help

Options:
  --episodes INT       Number of training episodes [default: 1000]
  --lr FLOAT          Learning rate [default: 0.001]
  --batch-size INT    Batch size [default: 32]
  --memory-size INT   Replay buffer size [default: 10000]
  --target-update INT Target network update frequency [default: 10]
  --epsilon-start FLOAT Starting epsilon [default: 1.0]
  --epsilon-end FLOAT   Final epsilon [default: 0.01]
  --epsilon-decay FLOAT Epsilon decay rate [default: 0.995]
  --no-mlflow         Disable MLflow tracking
  --render            Render environment during training
  --save-model PATH   Model save path [default: models/best_model.pth]
```

## Usage Examples

### 1. Basic Training

```bash
# Train with default parameters
python src/train.py

# Train with custom parameters
python src/train.py --episodes 500 --lr 0.0005 --batch-size 64
```

### 2. Hyperparameter Tuning

```bash
# Try different learning rates
python src/train.py --lr 0.001 --episodes 500
python src/train.py --lr 0.0005 --episodes 500
python src/train.py --lr 0.002 --episodes 500
```

### 3. Model Serving

```bash
# Start serving
python src/serve.py

# Test prediction
curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d '{"state": [0.1, 0.2, 0.3, 0.4]}'

# Evaluate model
curl -X POST http://localhost:8080/evaluate \
  -H "Content-Type: application/json" \
  -d '{"max_steps": 500}'
```

## Development

### Local Setup

```bash
# Clone and setup
git clone <repository>
cd mlflow-project

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Start MLflow locally
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns --host 0.0.0.0 --port 5000
```

### Jupyter Notebooks

```bash
# Start Jupyter for experimentation
jupyter notebook notebooks/
```

## Performance Expectations

- **Training Time**: 5-15 minutes on CPU, 2-5 minutes on GPU
- **Convergence**: Usually solves in 100-500 episodes
- **Success Criteria**: Average reward ≥ 195 over 100 episodes
- **Best Performance**: Can achieve 500+ reward consistently

## Troubleshooting

### Common Issues

1. **MLflow Connection Error**
   ```bash
   # Check if MLflow server is running
   curl http://localhost:5000/health
   
   # Restart MLflow server
   docker-compose restart mlflow-server
   ```

2. **GPU Not Detected**
   ```bash
   # Check PyTorch CUDA installation
   python -c "import torch; print(torch.cuda.is_available())"
   ```

3. **Model Not Loading**
   ```bash
   # Check model file exists
   ls -la models/
   
   # Check MLflow model registry
   # Visit http://localhost:5000
   ```

4. **Training Slow/Not Converging**
   - Try different learning rates (0.0001 - 0.01)
   - Increase memory buffer size
   - Adjust target network update frequency

### Logs and Debugging

```bash
# View container logs
docker-compose logs mlflow-server
docker-compose logs cartpole-trainer
docker-compose logs model-server

# Debug training locally
python src/train.py --render --episodes 100
```

## Next Steps

1. **Experiment with hyperparameters** using MLflow
2. **Try different RL algorithms** (A2C, PPO, SAC)
3. **Deploy to cloud** (AWS, GCP, Azure)
4. **Add more complex environments** (Atari, MuJoCo)
5. **Implement model monitoring** and retraining

## Learning Resources

- [OpenAI Gymnasium Documentation](https://gymnasium.farama.org/)
- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [Deep Reinforcement Learning](https://spinningup.openai.com/en/latest/)
- [PyTorch RL Tutorial](https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html)

## References

1. Swagat K. (2020) _Balancing a CartPole System with Reinforcement Learning - A Tutorial_ [arXiv:2006.04938](https://arxiv.org/pdf/2006.04938)
2. Melrose R., James M.G., and Stefanie T. (2017) _Implementing the Deep Q-Network_ [arXiv:1711.07478](https://arxiv.org/pdf/1711.07478)
3. Mark T. (2017) _Reinforcement Learning (DQN) Tutorial_ [PyTorch Tutorials](https://docs.pytorch.org/tutorials/intermediate/reinforcement_q_learning.html)
4. _DQN debugging using Open AI gym Cartpole_ [ADG Efficiency Blog](https://adgefficiency.com/dqn-debugging/); _DDQN hyperparameter tuning using Open AI gym Cartpole_ [ADG Efficiency Blog](https://adgefficiency.com/dqn-tuning/); _Solving Open AI gym Cartpole using DDQN_ [ADG Efficiency Blog](https://adgefficiency.com/dqn-solving/)
5. Erfan A. (2023) _Reinforcement Learning with PyTorch: Mastering CartPole-v0!_ [LinkedIn](https://www.linkedin.com/pulse/reinforcement-learning-pytorch-mastering-cartpole-v0-akbarnezhad/)
