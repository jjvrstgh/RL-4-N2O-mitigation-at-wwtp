from stable_baselines3 import DQN
from sumo_env9 import sumo_env
import os
from datetime import datetime


# Define directories and settings
current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
model_name = f"model_{current_time}"
models_dir = f"models/{model_name}/"
logdir = f"logs/{model_name}/"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if not os.path.exists(logdir):
    os.makedirs(logdir)

# Create the environment
env = sumo_env("train2.csv", 1)

# Create the DQN model
model = DQN('MlpPolicy', 
            env,
            buffer_size=1200,
            learning_starts=2400,
            batch_size=24,
            gamma=0.99,
            exploration_fraction=0.60,
            exploration_initial_eps=0.9,
            exploration_final_eps=0,
            verbose=1, 
            tensorboard_log=logdir)

model.learn(total_timesteps=5240, reset_num_timesteps=False, log_interval=1)
model.save(f"{models_dir}/final_model")