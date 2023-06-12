from stable_baselines3 import DQN
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor

from sumo_env7 import sumo_env
import os
from datetime import datetime
import subprocess

from wandb.integration.sb3 import WandbCallback
import wandb

# Define directories and settings
current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
model_name = f"model_{current_time}"
models_dir = f"models/{model_name}/"
logdir = f"logs/{model_name}/"

conf_dict = {"Model": "v2",
             "policy": "MlpPolicy",
             "model_save_name": model_name}

run = wandb.init(
    project="GHG_mitigation",
    entity="jerms",
    config=conf_dict,
    sync_tensorboard=True,
    save_code=True,
)

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if not os.path.exists(logdir):
    os.makedirs(logdir)

# Create the environment
env = sumo_env("merged_data.csv", 1)

# Create the DQN model
model = DQN('MlpPolicy', env, verbose=1, tensorboard_log=logdir)

# Training parameters
EPISODES = 500  # Total number of episodes
TIMESTEPS_PER_EPISODE = 10  # Timesteps per episode

for episode in range(EPISODES):
    obs = env.reset()
    n_episode = env.episode

    # Perform a learning update
    model.learn(total_timesteps=TIMESTEPS_PER_EPISODE, reset_num_timesteps=False, log_interval=1, callback=WandbCallback())
    model.save(f"{models_dir}/episode_{n_episode}")


# Save the final trained model
model.save(f"{models_dir}/final_model")