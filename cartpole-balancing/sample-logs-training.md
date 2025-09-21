# Sample Logs - Training

## Training Model in Docker Container

```
chuan@chuan2025:~/mlflow-projects/cartpole-balancing$
chuan@chuan2025:~/mlflow-projects/cartpole-balancing$
chuan@chuan2025:~/mlflow-projects/cartpole-balancing$ make train
Starting DQN training...
docker compose --profile training up cartpole-trainer
[+] Running 2/2
 ✔ Container mlflow-tracking-server  Running                                                                                                                                       0.0s
 ✔ Container cartpole-trainer        Created                                                                                                                                       0.1s
Attaching to cartpole-trainer
cartpole-trainer  | 2025/09/21 04:35:23 WARNING mlflow.utils.git_utils: Failed to import Git (the Git executable is probably not on your PATH), so Git SHA is not available. Error: Failed to initialize: Bad git executable.
cartpole-trainer  | The git executable must be specified in one of the following ways:
cartpole-trainer  |     - be included in your $PATH
cartpole-trainer  |     - be set via $GIT_PYTHON_GIT_EXECUTABLE
cartpole-trainer  |     - explicitly set via git.refresh(<full-path-to-git-executable>)
cartpole-trainer  |
cartpole-trainer  | All git commands will error until this is rectified.
cartpole-trainer  |
cartpole-trainer  | This initial message can be silenced or aggravated in the future by setting the
cartpole-trainer  | $GIT_PYTHON_REFRESH environment variable. Use one of the following values:
cartpole-trainer  |     - quiet|q|silence|s|silent|none|n|0: for no message or exception
cartpole-trainer  |     - warn|w|warning|log|l|1: for a warning message (logging level CRITICAL, displayed by default)
cartpole-trainer  |     - error|e|exception|raise|r|2: for a raised exception
cartpole-trainer  |
cartpole-trainer  | Example:
cartpole-trainer  |     export GIT_PYTHON_REFRESH=quiet
cartpole-trainer  |
cartpole-trainer  | Successfully registered model 'cartpole_dqn_model'.
cartpole-trainer  | 2025/09/21 04:41:56 INFO mlflow.tracking._model_registry.client: Waiting up to 300 seconds for model version to finish creation. Model name: cartpole_dqn_model, version 1
cartpole-trainer  | Created version '1' of model 'cartpole_dqn_model'.
cartpole-trainer  | MLflow tracking enabled - Experiment: cartpole-dqn
cartpole-trainer  | Environment: CartPole-v1
cartpole-trainer  | State size: 4
cartpole-trainer  | Action size: 2
cartpole-trainer  | Device: CPU
cartpole-trainer  |
cartpole-trainer  | Starting training for 1000 episodes...
cartpole-trainer  | --------------------------------------------------
cartpole-trainer  | Episode    0 | Reward:   12.0 | Avg Reward:   12.0 | Epsilon: 1.000 | Loss:    N/A
cartpole-trainer  | Episode  100 | Reward:  177.0 | Avg Reward:  109.9 | Epsilon: 0.010 | Loss: 0.1565
cartpole-trainer  | Episode  200 | Reward:  128.0 | Avg Reward:  121.5 | Epsilon: 0.010 | Loss: 0.3793
cartpole-trainer  | Episode  300 | Reward:   37.0 | Avg Reward:  127.6 | Epsilon: 0.010 | Loss: 2.9504
cartpole-trainer  |
cartpole-trainer  | Environment solved in 363 episodes!
cartpole-trainer  | Average reward over last 100 episodes: 196.58
cartpole-trainer  | Episode  400 | Reward:   19.0 | Avg Reward:  181.3 | Epsilon: 0.010 | Loss: 0.6830
cartpole-trainer  | Episode  500 | Reward:  144.0 | Avg Reward:  132.3 | Epsilon: 0.010 | Loss: 0.4029
cartpole-trainer  | Episode  600 | Reward:  500.0 | Avg Reward:  255.3 | Epsilon: 0.010 | Loss: 1.0489
cartpole-trainer  | Episode  700 | Reward:  102.0 | Avg Reward:  119.1 | Epsilon: 0.010 | Loss: 3.0253
cartpole-trainer  | Episode  800 | Reward:  235.0 | Avg Reward:  223.4 | Epsilon: 0.010 | Loss: 3.9000
cartpole-trainer  | Episode  900 | Reward:   60.0 | Avg Reward:  395.4 | Epsilon: 0.010 | Loss: 9.7144
cartpole-trainer  | Episode  999 | Reward:  349.0 | Avg Reward:  376.5 | Epsilon: 0.010 | Loss: 12.8060
cartpole-trainer  |
cartpole-trainer  | Training completed!
cartpole-trainer  | Best average reward: 440.44
cartpole-trainer  | Environment solved in episode 363
cartpole-trainer  | MLflow artifacts logged successfully
cartpole-trainer exited with code 0
chuan@chuan2025:~/mlflow-projects/cartpole-balancing$
chuan@chuan2025:~/mlflow-projects/cartpole-balancing$ docker ps -a
CONTAINER ID   IMAGE                                 COMMAND                   CREATED         STATUS                      PORTS                    NAMES
0f1ec95af5f9   cartpole-balancing-cartpole-trainer   "python src/train.py"     7 minutes ago   Exited (0) 27 seconds ago                            cartpole-trainer
ea8d28e17020   python:3.9-slim                       "bash -c '\n  pip ins…"   8 minutes ago   Up 8 minutes                0.0.0.0:5000->5000/tcp   mlflow-tracking-server
chuan@chuan2025:~/mlflow-projects/cartpole-balancing$
chuan@chuan2025:~/mlflow-projects/cartpole-balancing$
chuan@chuan2025:~/mlflow-projects/cartpole-balancing$
```


## Training Model in Locally in Virtual Environment

```
chuan@chuan2025:~/mlflow-projects/cartpole-balancing$
chuan@chuan2025:~/mlflow-projects/cartpole-balancing$
chuan@chuan2025:~/mlflow-projects/cartpole-balancing$ make train-local
Starting local DQN training...
./venv.sh run python src/train.py
/home/chuan/Documents/My_Study/AI/mlflow-courses/mlflow-projects/cartpole-balancing/.venv/lib/python3.11/site-packages/mlflow/utils/requirements_utils.py:20: UserWarning: pkg_resources is deprecated as an API. See https://setuptools.pypa.io/en/latest/pkg_resources.html. The pkg_resources package is slated for removal as early as 2025-11-30. Refrain from using this package or pin to Setuptools<81.
  import pkg_resources  # noqa: TID251
MLflow tracking enabled - Experiment: cartpole-dqn
Environment: CartPole-v1
State size: 4
Action size: 2
Device: CUDA

Starting training for 1000 episodes...
--------------------------------------------------
Episode    0 | Reward:   21.0 | Avg Reward:   21.0 | Epsilon: 1.000 | Loss:    N/A
Episode  100 | Reward:  157.0 | Avg Reward:  121.9 | Epsilon: 0.010 | Loss: 0.3067
Episode  200 | Reward:   14.0 | Avg Reward:  125.0 | Epsilon: 0.010 | Loss: 0.4109
Episode  300 | Reward:  188.0 | Avg Reward:  149.3 | Epsilon: 0.010 | Loss: 2.9248
Episode  400 | Reward:  123.0 | Avg Reward:  149.6 | Epsilon: 0.010 | Loss: 1.7138
Episode  500 | Reward:  246.0 | Avg Reward:  170.4 | Epsilon: 0.010 | Loss: 0.2656
Episode  600 | Reward:   34.0 | Avg Reward:  186.7 | Epsilon: 0.010 | Loss: 3.0145

Environment solved in 613 episodes!
Average reward over last 100 episodes: 196.34
Episode  700 | Reward:  121.0 | Avg Reward:  330.1 | Epsilon: 0.010 | Loss: 3.3306
Episode  800 | Reward:  133.0 | Avg Reward:  243.9 | Epsilon: 0.010 | Loss: 10.3967
Episode  900 | Reward:  500.0 | Avg Reward:  395.3 | Epsilon: 0.010 | Loss: 6.9796
Episode  999 | Reward:  500.0 | Avg Reward:  489.4 | Epsilon: 0.010 | Loss: 6.3149

Training completed!
Best average reward: 489.45
Environment solved in episode 613
Training failed: [Errno 13] Permission denied: '/app'
Traceback (most recent call last):
  File "/home/chuan/Documents/My_Study/AI/mlflow-courses/mlflow-projects/cartpole-balancing/src/train.py", line 247, in <module>
    main()
  File "/home/chuan/Documents/My_Study/AI/mlflow-courses/mlflow-projects/cartpole-balancing/src/train.py", line 224, in main
    tracker.log_reward_plot(rewards)
  File "/home/chuan/Documents/My_Study/AI/mlflow-courses/mlflow-projects/cartpole-balancing/src/mlflow_utils.py", line 111, in log_reward_plot
    mlflow.log_artifact(plot_path, "plots")
  File "/home/chuan/Documents/My_Study/AI/mlflow-courses/mlflow-projects/cartpole-balancing/.venv/lib/python3.11/site-packages/mlflow/tracking/fluent.py", line 874, in log_artifact
    MlflowClient().log_artifact(run_id, local_path, artifact_path)
  File "/home/chuan/Documents/My_Study/AI/mlflow-courses/mlflow-projects/cartpole-balancing/.venv/lib/python3.11/site-packages/mlflow/tracking/client.py", line 1092, in log_artifact
    self._tracking_client.log_artifact(run_id, local_path, artifact_path)
  File "/home/chuan/Documents/My_Study/AI/mlflow-courses/mlflow-projects/cartpole-balancing/.venv/lib/python3.11/site-packages/mlflow/tracking/_tracking_service/client.py", line 454, in log_artifact
    artifact_repo.log_artifact(local_path, artifact_path)
  File "/home/chuan/Documents/My_Study/AI/mlflow-courses/mlflow-projects/cartpole-balancing/.venv/lib/python3.11/site-packages/mlflow/store/artifact/local_artifact_repo.py", line 36, in log_artifact
    mkdir(artifact_dir)
  File "/home/chuan/Documents/My_Study/AI/mlflow-courses/mlflow-projects/cartpole-balancing/.venv/lib/python3.11/site-packages/mlflow/utils/file_utils.py", line 192, in mkdir
    raise e
  File "/home/chuan/Documents/My_Study/AI/mlflow-courses/mlflow-projects/cartpole-balancing/.venv/lib/python3.11/site-packages/mlflow/utils/file_utils.py", line 189, in mkdir
    os.makedirs(target)
  File "<frozen os>", line 215, in makedirs
  File "<frozen os>", line 215, in makedirs
  File "<frozen os>", line 215, in makedirs
  [Previous line repeated 2 more times]
  File "<frozen os>", line 225, in makedirs
PermissionError: [Errno 13] Permission denied: '/app'
make: *** [Makefile:46: train-local] Error 1
chuan@chuan2025:~/mlflow-projects/cartpole-balancing$
chuan@chuan2025:~/mlflow-projects/cartpole-balancing$
chuan@chuan2025:~/mlflow-projects/cartpole-balancing$
```

**TODO**: fix the training code to make training model locally working 