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
- Python 3.9+ (for local development)

### 1. Start MLflow Tracking Server

```bash
# Start only the MLflow tracking server
docker-compose up mlflow-server -d

# MLflow UI will be available at http://localhost:5000
```

### 2. Train the Model

#### Option A: Using Docker

```bash
# Train using Docker
docker-compose --profile training up cartpole-trainer
```

#### Option B: Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Train locally
python src/train.py
```

### 3. Serve the Model

```bash
# Start model serving
docker-compose --profile serving up model-server

# API will be available at http://localhost:8080
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

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.