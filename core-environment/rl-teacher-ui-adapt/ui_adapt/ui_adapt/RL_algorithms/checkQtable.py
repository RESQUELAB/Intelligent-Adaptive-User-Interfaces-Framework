import numpy as np
import gym
import pickle
import matplotlib.pyplot as plt

import pandas as pd

SIGMA = 1

env_name = "UIAdaptation-v0"
models_dir = 'RL_algorithms/models'
filename = "QLearning_Sigma_" + str(SIGMA) + "_" + env_name + ".pickle"
# Load the saved Q-table and metrics
file_path = models_dir+"/"+filename

with open(file_path, 'rb') as file:
    saved_data = pickle.load(file)


q_table = saved_data["q_table"]


import numpy as np

# Assuming your QTable is stored in a variable named q_table
dimensions = q_table.shape
print("QTable Dimensions:", dimensions)

unvisited_states_mask = np.all(q_table == 0, axis=1)
unvisited_states_indices = np.where(unvisited_states_mask)[0]

if len(unvisited_states_indices) > 0:
    print("Unvisited States Found at Indices:", unvisited_states_indices)
else:
    print("All states have been visited.")

