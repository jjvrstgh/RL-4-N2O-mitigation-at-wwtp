from stable_baselines3.common.env_checker import check_env
from sumo_env9 import sumo_env


env = sumo_env("train2.csv", 1)
# It will check your custom environment and output additional warnings if needed
check_env(env)