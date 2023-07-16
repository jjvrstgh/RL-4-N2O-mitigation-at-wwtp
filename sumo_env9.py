import time
import gym
from gym import spaces
import numpy as np
import pandas as pd

import csv
import os
import pickle

import dynamita.scheduler as ds
import dynamita.tool as dtool

# Import the reward function from the reward file
from reward import calculate_reward
from sumo_runner4 import run_simulation

n_variables = 14  # Determine observation space
t_state = 3 # n timesteps in statespace
s_space = n_variables * t_state
n_actions = 24  # Determine the number of actions

#training parameters
EPISODES = 50 # Total number of episodes
TIMESTEPS_PER_EPISODE = 96  # Timesteps per episode

# Create the output dictionary
output_dict = {
  0: (0.10, 0.10),
  1: (0.10, 0.25),
  2: (0.10, 0.5),
  3: (0.10, 1.0),
  4: (0.25, 0.10),
  5: (0.25, 0.25),
  6: (0.25, 0.5),
  7: (0.25, 1.0),
  8: (0.5, 0.10),
  9: (0.5, 0.25),
  10: (0.5, 0.5),
  11: (0.5, 1.0),
  12: (0.5, 1.5),
  13: (1.0, 0.10),
  14: (1.0, 0.25),
  15: (1.0, 0.5),
  16: (1.0, 1.0),
  17: (1.0, 1.5),
  18: (1.5, 0.5),
  19: (1.5, 1.0),
  20: (1.5, 1.5),
  21: (2.0, 2.0),
  22: (2.5, 2.5),
  23: (3.0, 3.0)
}


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
        self.observation_space = spaces.Box(low=-1, high=1, shape=(s_space,), dtype=np.float32)

        # Calculate and store the length of the CSV file
        self.len_csv = self._get_csv_length(csv_file)

        # Initialize influent getter
        self.current_row_index = 0
        self.csv_data = self._load_csv_data(csv_file)

        # indicates which run this is [1:6]
        self.n_run = n_run

        # sumo init steady state
        self.reset_state = True

        # indicates episode
        self.episode = 0
        self.internal_counter = 0
        self.step_count = 0
        
        # List of files to check and create
        files = [
            "state_rwd_action.pkl",
            "normalized_state_rwd_action.pkl"
        ]

        # Delete existing pickle files if they exist
        for file in files:
            if os.path.isfile(file):
                os.remove(file)
                print(f"Existing {file} deleted.")

        # Create an initial data dictionary
        observation = np.zeros(n_variables)
        data = {"state": observation, "reward": 30, "action": 0, "done": False}

        # Save the initial data to the pickle files
        for file in files:
            with open(file, "wb") as f:
                pickle.dump(data, f)
            print(f"New {file} created with initial data.")


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

        if self.internal_counter > len(self.csv_data):
            self.internal_counter = 0
        
        current_row = self.csv_data[self.internal_counter - 1]

        # Extract the timestep and variables from the row
        timestep = current_row[0]
        variables = current_row[1:]

        # Check if the row number is a multiple of 96
        if self.internal_counter % 240 == 0 or self.internal_counter == 1:
            self.reset_state = True
        else:
            self.reset_state = False

        # Check if all rows have been read
        if self.internal_counter == len(self.csv_data):
            print('Episode done')
            self.internal_counter = 0  # Reset the internal counter for the next episode
            self.done = True  # Mark episode as done
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
        
        run_simulation(influent, do_setpoint, self.reset_state)

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
        Normalization procedure happens before new state is updated
        '''

        normalization_range = n_variables * 8 # 96 == 1 day

        # Load the existing data from the pkl file
        try:
            with open("normalized_state_rwd_action.pkl", "rb") as f:
                normalized_data = pickle.load(f)
        except FileNotFoundError:
            normalized_data = {}
        
        state_data = existing_data['state']
        num_rows = len(state_data)  # Get the number of available rows

        if num_rows >= normalization_range:
            state_data = state_data[-normalization_range:]  # Consider the last 96 rows
        else:
            state_data = state_data[-num_rows:]  # Consider all available rows

        state_data = state_data.reshape(-1, n_variables)
        df_state = pd.DataFrame(state_data, columns=range(n_variables))  # Create a DataFrame with n_variables columns

        # Normalize each variable in the observation
        normalized_state = []
        for i in range(n_variables):
            var = observation[i]
            variable_data = df_state[i]  # Consider the selected rows for normalization
            mean_value = np.mean(variable_data)
            min_value = np.min(variable_data)
            max_value = np.max(variable_data)

            if max_value == min_value:
                normalized_var = 0  # Assign a default value when max_value and min_value are equal
            else:
                normalized_var = (var - mean_value) / (max_value - min_value)

            normalized_state.append(normalized_var)

        normalized_state = np.array(normalized_state, dtype=np.float32)


        # Normalize reward

        reward_data = pd.Series(existing_data['reward'])
        num_rows_re = len(reward_data)  # Get the number of available rows

        if num_rows_re >= normalization_range:
            reward_data = reward_data[-normalization_range:]  # Consider the last n * t rows
        else:
            reward_data = reward_data[-num_rows_re:]  # Consider all available rows

        df_reward = pd.Series(reward_data)  # Create a Series with a single column

        reward_mean = df_reward.mean()
        reward_min = df_reward.min()
        reward_max = df_reward.max()

        if reward_max == reward_min:
            normalized_reward = 0  # Assign a default value when max_value and min_value are equal
        else:
            normalized_reward = (reward - reward_mean) / (reward_max - reward_min)

        normalized_reward = float(normalized_reward)


        # Update the data with the new normalized observation, reward, action, and done flag
        if 'state' in normalized_data:
            normalized_data['state'] = np.append(normalized_data['state'], normalized_state)
            normalized_data['reward'] = np.append(normalized_data['reward'], normalized_reward)
            normalized_data['action'] = np.append(normalized_data['action'], action)
            normalized_data['done'] = np.append(normalized_data['done'], False)
        else:
            normalized_data['state'] = np.array([normalized_state])
            normalized_data['reward'] = np.array([reward])
            normalized_data['action'] = np.array([action])
            normalized_data['done'] = np.array([False])       
        

        # Select the last n row data entries from normalized_data['state']
        normalized_last_n_states = normalized_data['state'][-n_variables * t_state:]

        if normalized_last_n_states.shape[0] != n_variables * t_state:
            normalized_last_n_states =  np.zeros(n_variables * t_state)
            print('dummy observation used')

        # Write the updated data back to the pkl file
        with open("normalized_state_rwd_action.pkl", "wb") as f:
            pickle.dump(normalized_data, f)

        '''
        Normalization finished
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

        # keep reset on false
        self.reset_state = False

        return normalized_last_n_states, reward, self.done, {}


    def reset(self):
        # stop all sumo activities
        ds.sumo.cleanup()

        # sumo init steady state
        self.reset_state = True

        init_observation = np.zeros(n_variables * t_state)

        return init_observation
