import os
from enumerate_dof import dof_matrix
import yaml
import pandas as pd
import pyvista as pv
import numpy as np
import enumerate_dof as ed

script_dir = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(script_dir, "config.yaml")

with open(config_path, "r") as file:
    config = yaml.safe_load(file)

base_dir = config["Directory"]["File_directory"]

input_folder = os.path.join(base_dir, "Input\Element512")
output_folder = os.path.join(base_dir, "Output\Element512")
VtkData_folder = os.path.join(output_folder)

nodes = np.loadtxt(os.path.join(input_folder, "nodes.txt"), delimiter=',')  # Shape: (n_nodes, 3)
connectivity = np.loadtxt(os.path.join(input_folder, "connectivity.txt"), dtype=int, delimiter=',')
# force_calc_pinn_laststep = np.loadtxt(os.path.join(output_folder,"Force_Calc_500000_PINN.txt"),dtype=float,delimiter = ',')
# force_calc_pinn_laststep = force_calc_pinn_laststep.reshape(ed.total_nodes,3)
dof_per_node = 3
num_elements =connectivity.shape[0]
total_nodes = np.max(connectivity) + 1
force_rearrangend = np.zeros((total_nodes,dof_per_node))
num_nodes_per_element = 8
node_to_dof = {}
dof_counter = 0
i =0

# for element_idx in range(num_elements):
#     for node_idx in range(num_nodes_per_element):
#         node = connectivity[element_idx, node_idx]
#
#         if node not in node_to_dof:
#             node_to_dof[node] = [dof_counter, dof_counter + 1, dof_counter + 2]
#             force_rearrangend[node] = force_calc_pinn_laststep[i,:]
#             dof_counter += 3
#             i = i+1
#
#         dof_matrix[element_idx, node_idx * dof_per_node:(node_idx + 1) * dof_per_node] = node_to_dof[node]

file_path = os.path.join(VtkData_folder,"Cumulative_Data_Paraview.xlsx")
df = pd.read_excel(file_path)
df_connectivity = pd.read_excel(file_path, sheet_name="Connectivity")
connectivity = df_connectivity.values.astype(int)
# Extract columns
points = df[["X", "Y", "Z"]].values  # Node positions
displacements = df[["Ux", "Uy", "Uz"]].values  # Displacement vectors
forces = df[["Fx", "Fy", "Fz"]].values  # Force vectors


# Prepare connectivity for PyVista 8 is added here below just for paraview syntax to be sure of the element consists of 8 nodes
cells = np.insert(connectivity, 0, num_nodes_per_element, axis=1)
celltypes = np.full(num_elements, pv.CellType.HEXAHEDRON, dtype=np.int8)

# Create PyVista UnstructuredGrid
grid = pv.UnstructuredGrid(cells, celltypes, points)
grid.point_data["Displacement"] = displacements
grid.point_data["Forces"] = forces

# Save to VTU file
grid.save("Inverse_Model_Error.vtu")

print("FEM model successfully saved as FEM_Model.vtu")





