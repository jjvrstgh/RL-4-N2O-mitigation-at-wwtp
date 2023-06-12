import time
import gym
from gym import spaces
import numpy as np
import pandas as pd

import os
import pickle

import dynamita.scheduler as ds
import dynamita.tool as dtool

# Import the reward function from the reward file
from reward import calculate_reward
from sumo_runner2 import SumoSimulation

n_variables = 10  # Determine observation space
n_actions = 9  # Determine the number of actions
do_step = 0.5 # Determine the step size of DO-sp

#training parameters
EPISODES = 500 # Total number of episodes
TIMESTEPS_PER_EPISODE = 10  # Timesteps per episode
total_timesteps = EPISODES * TIMESTEPS_PER_EPISODE

# Create the output dictionary
output_dict = {i: 0.5 + (i * do_step) for i in range(n_actions)}


import os
import time
import pickle
import csv

def wait_for_file(file_path, max_retries):
    retries = 0
    while retries < max_retries:
        for _ in range(10):
            if os.path.exists(file_path):
                return os.path.abspath(file_path)
            time.sleep(1)
        retries += 1
        if retries < max_retries:
            time.sleep(5)

        emergency_data = "backup_output.csv"
        print("emergency file was used")

        return os.path.abspath(emergency_data)


class sumo_env(gym.Env):
    """Gym-like environment that can work with Dynamita's Sumo to optimize WWTPs"""

    def __init__(self, csv_file, n_run):
        super(sumo_env, self).__init__()

        # Define the action space
        self.action_space = spaces.Discrete(n_actions)

        # Define the observation space
        self.observation_space = spaces.Box(low=0, high=40000, shape=(n_variables,), dtype=np.float32)

        # Calculate and store the length of the CSV file
        self.len_csv = self._get_csv_length(csv_file)

        # Initialize influent getter
        self.current_row_index = 0
        self.csv_data = self._load_csv_data(csv_file)

        # indicates which run this is [1:6]
        self.n_run = n_run

        # indicates episode
        self.episode = 0
        self.internal_counter = 0
        self.step_count = 0

        # Check if the file exists
        if os.path.isfile("state_rwd_action.pkl"):
            os.remove("state_rwd_action.pkl")
            print("Existing pickle file deleted.")

        # Check if the file exists
        if os.path.isfile("perma_data.csv"):
            os.remove("perma_data.csv")
            print("Existing pickle file deleted.")


    def _get_csv_length(self, csv_file):
        df = pd.read_csv(csv_file)
        length = len(df) - 1  # Subtract 1 to exclude the header row
        return length


    def _load_csv_data(self, csv_file):
        # Read the CSV file and return the data as a list of rows
        df = pd.read_csv(csv_file)
        data = df.values[1:]  # Skip the header row

        return data


    def _load_next_row(self):
        self.internal_counter += 1

        if self.episode == 0:
            # Get the first 250 rows from the file
            rows = self.csv_data[:TIMESTEPS_PER_EPISODE]
            current_row = rows[self.internal_counter - 1]
        else:
            # Get the next 250 rows from the file
            start_index = self.episode * TIMESTEPS_PER_EPISODE
            end_index = (self.episode + 1) * TIMESTEPS_PER_EPISODE
            rows = self.csv_data[start_index:end_index]
            current_row = rows[self.internal_counter - 1]

        # Extract the timestep and variables from the row
        timestep = current_row[0]
        variables = current_row[1:]

        # Check if all rows have been read
        if self.internal_counter == TIMESTEPS_PER_EPISODE:
            print('episode done')
            self.episode += 1
            self.done = True  # Mark episode as done
            self.internal_counter = 0  # Reset the internal counter for the next episode
            if self.episode * TIMESTEPS_PER_EPISODE >= len(self.csv_data):
                self.episode = 0  # Reset to episode 0 if all rows have been read
        else:
            self.done = False

        return timestep, variables


    def step(self, action):
        temp_output = f"model_output_{self.n_run}.csv"
        if os.path.isfile(temp_output):
            os.remove(temp_output)

        # Use the action as a key in the output_dict
        do_setpoint = output_dict[int(action)]

        # Load the next row as the new observation
        _, influent = self._load_next_row()
        
        simulation = SumoSimulation(influent, do_setpoint, self.n_run)
        simulation.run_simulation()

        # check for new temp output file
        get_file = wait_for_file(temp_output, 3)
        model_output_from_file = pd.read_csv(get_file)
        model_output = model_output_from_file.values

        # Combine the contents of "temp_state.csv" and the new row as floats
        observation_current_row = np.concatenate((influent, model_output.flatten()))
        observation = observation_current_row.astype(np.float32)  # Convert to float32

        # Generate reward using the imported reward function
        reward = calculate_reward(observation)

        # Load the existing data from the pkl file
        try:
            with open("state_rwd_action.pkl", "rb") as f:
                existing_data = pickle.load(f)
        except FileNotFoundError:
            existing_data = {}

        '''
        # Normalize each variable in the observation
        state_data = existing_data['state']
        print(state_data)
        normalized_variables = []
        for i in range(n_variables):
            var = observation[i]
            variable_data = state_data[i]
            mean_value = np.mean(variable_data)
            min_value = np.min(variable_data)
            max_value = np.max(variable_data)

            if max_value == min_value:
                normalized_var = 0  # Assign a default value when max_value and min_value are equal
            else:
                normalized_var = (var - mean_value) / (max_value - min_value)

            normalized_variables.append(normalized_var)

        print(normalized_variables)
        '''

        # Update the data with the new observation, reward, action, and done flag
        if 'state' in existing_data:
            existing_data['state'] = np.append(existing_data['state'], observation)
            existing_data['reward'] = np.append(existing_data['reward'], reward)
            existing_data['action'] = np.append(existing_data['action'], action)
            existing_data['done'] = np.append(existing_data['done'], False)
        else:
            existing_data['state'] = np.array([observation])
            existing_data['reward'] = np.array([reward])
            existing_data['action'] = np.array([action])
            existing_data['done'] = np.array([False])

        # Write the updated data back to the pkl file
        with open("state_rwd_action.pkl", "wb") as f:
            pickle.dump(existing_data, f)

        
        # Increment step count
        self.step_count += 1

        # Append the values to a CSV file
        with open('perma_data.csv', 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([self.step_count, observation, reward, action])



        return observation, reward, self.done, {}


    def reset(self):
        # stop all sumo activities
        ds.sumo.cleanup()

        # Reset the current row index
        self.current_row_index = 0
        self.done = False

        #Return the initial observation
        observation = np.ones(n_variables)

        data = {"state": observation, "reward": 0, "action": None, "done": False}
        with open("state_rwd_action.pkl", "wb") as f:
            pickle.dump(data, f)
        print("New file created with initial data.")
        

        return observation
