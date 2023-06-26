from stable_baselines3 import DQN
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor

from sumo_env9 import sumo_env
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

'''
conf_dict = {"Model": "v5",
             "policy": "MlpPolicy",
             "model_save_name": model_name}

run = wandb.init(
    project="GHG_mitigation",
    entity="jerms",
    config=conf_dict,
    sync_tensorboard=True,
    save_code=True,
)
'''

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
model.save(f"{models_dir}/train")

'''
# Training parameters
EPISODES = 5  # Total number of episodes
TIMESTEPS_PER_EPISODE = 96  # Timesteps per episode

# load pre-trained model
saved_model = f"{models_dir}/train.zip"
model2 = DQN.load(saved_model)

for episode in range(EPISODES):
    obs = env.reset()
    n_episode = env.episode

    # Perform a learning update
    model2.learn(total_timesteps=TIMESTEPS_PER_EPISODE, reset_num_timesteps=False, log_interval=1)#, callback=WandbCallback())
    model2.save(f"{models_dir}/episode_{n_episode}")
'''

# Save the final trained model
model.save(f"{models_dir}/final_model")