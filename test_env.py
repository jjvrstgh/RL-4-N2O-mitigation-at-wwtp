import gym
from sumo_env8 import sumo_env


env = sumo_env("train.csv", 1)
observation = env.reset()
print(f"Initial observation: {observation.shape}")

for step in range(5):
    action = env.action_space.sample()
    print(f"Step {step + 1}:")

    next_observation, reward, done, _ = env.step(action)
    print(f"Next observation: {next_observation.shape}")
    print(f"Reward: {reward}")
    print(f"Done: {done}")

    if done:
        print("Episode finished.")
        break  