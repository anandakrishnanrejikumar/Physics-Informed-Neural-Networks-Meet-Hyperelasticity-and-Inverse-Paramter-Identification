import numpy as np
import os
import yaml

script_dir = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(script_dir, "config.yaml")

with open(config_path, "r") as file:
    config = yaml.safe_load(file)

base_dir = config["Directory"]["File_directory"]

# Define the input folder path
input_folder = os.path.join(base_dir, "Input/Element512")

# Load data
connectivity = np.loadtxt(os.path.join(input_folder, "connectivity.txt"), dtype=int, delimiter=',')
constraintInfo = np.loadtxt(os.path.join(input_folder, "displacement_0.500000.txt"), dtype=float, delimiter=',')
constraintInfo = constraintInfo[:, 6:10] # Extract constraint columns from file

num_nodes_per_element = 8
dof_per_node = 3
total_nodes = np.max(connectivity) + 1

num_elements = connectivity.shape[0]
dof_matrix = np.zeros((num_elements, num_nodes_per_element * dof_per_node), dtype=int)

is_dof_constrained = np.zeros(total_nodes * dof_per_node, dtype = bool)

node_to_dof = {}

dof_counter = 0

for element_idx in range(num_elements):
    for node_idx in range(num_nodes_per_element):
        node = connectivity[element_idx, node_idx]

        if node not in node_to_dof:
            node_to_dof[node] = [dof_counter, dof_counter + 1, dof_counter + 2]
            for i in range(dof_per_node):
                if constraintInfo[node, i] == 1:
                    is_dof_constrained[dof_counter + i] = True
            dof_counter += 3

        dof_matrix[element_idx, node_idx * dof_per_node:(node_idx + 1) * dof_per_node] = node_to_dof[node]

