import os
from collections import namedtuple
import numpy as np
import torch
import torch.nn as nn
import yaml
import finite_element_model_fully_vectorized as tf
from enumerate_dof import dof_matrix

script_dir = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(script_dir, "config.yaml")

with open(config_path, "r") as file:
    config = yaml.safe_load(file)

base_dir = config["Directory"]["File_directory"]

# Define the input folder path
input_folder = os.path.join(base_dir, "Input","Element512")
output_folder = os.path.join(base_dir, "Output","Element512")
source_folder = os.path.join(base_dir, "Source")
file_path = os.path.join(output_folder, f"{config['Output File Names']['LossvsEpoch']}")
file_residual_force = os.path.join(output_folder, f"{config['Output File Names']['ResidualvsEpoch']}")
file_pinn  = os.path.join(output_folder, f"{config['Output File Names']['PinnModel']}")
nEl = config["FEM Parameters"]["Total Elements"]

# Load data
nodes = np.loadtxt(os.path.join(input_folder, "nodes.txt"), delimiter=',',dtype=np.float64)  # Shape: (n_nodes, 3)
connectivity = np.loadtxt(os.path.join(input_folder, "connectivity.txt"), dtype=int, delimiter=',')

# Convert to PyTorch tensors
X = torch.tensor(nodes, dtype=torch.float64)  # Nodal coordinates
nLoadSteps = config["General"]["Number Load Steps"]
t = torch.linspace(0.1, 1, nLoadSteps)  # Time values (here: <nLoadSteps> time steps)
tmp = t.repeat(nodes.shape[0], 1).T.flatten() # 1350 x 1
X_with_time = torch.cat((X.repeat(nLoadSteps, 1), tmp.view(-1, 1)), dim=1)
X_with_time.to(X.dtype)

# Precalculate shape function data
# Assuming shape_grad and detJ are tensors or arrays computed earlier
shape_grad, detJ = tf.Precalculate_shapeFunctionDetails(connectivity, nodes)
PreCalc = namedtuple('PreCalc', ['shape_grad', 'detJ'])
preCalc = PreCalc(shape_grad=shape_grad, detJ=detJ)
required_dof_for_loss_calculation = tf.Extract_nodes_force_for_loss(connectivity, nodes, dof_matrix)

# PINN
class PINN(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=50, output_dim=3, num_hidden_layers=5):
        super(PINN, self).__init__()

        # Build layers
        layers = [nn.Linear(input_dim, hidden_dim),nn.ReLU()]  # Input layer with activation
        for _ in range(num_hidden_layers):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())  # Add ReLU after each hidden layer
        layers.append(nn.Linear(hidden_dim, output_dim))  # Output layer

        self.network = nn.Sequential(*layers)  # Combine all layers into a single Sequential module

    def forward(self, X_with_time_data):
        u_nn = self.network(X_with_time_data)  # Output from the network

        # Extract coordinates and time
        X1 = X_with_time_data[:, 0:1]  # First coordinate (x)
        X2 = X_with_time_data[:, 1:2]  # Second coordinate (y)
        X3 = X_with_time_data[:, 2:3]  # Third coordinate (z)
        time = X_with_time_data[:, 3:4]   # Time
        u_bar = 0.5 * time # Applied displacement

        u_x = X1 * (X2 - 0.5) * u_nn[:,0:1]
        u_y = (0.5 + X2) * u_bar + 4 * X2 * (0.5 - X2) * u_nn[:, 1:2]
        u_z = (X3 - 0.1) * u_nn[:, 2:3]

        # Horizontally concatenate displacement vectors
        u_corrected = torch.cat([u_x, u_y, u_z], dim=1)

        return u_corrected


# Model, optimizer, and training setup
if __name__ == "__main__":
    model = PINN(input_dim=4, hidden_dim=50, output_dim=3, num_hidden_layers=5)
    model = model.double()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)



    # Training loop
    epochs = 100000
    for epoch in range(epochs):
        total_loss = 0.0

        # for i, t in enumerate(t_values):
        optimizer.zero_grad()

        # Forward pass
        displacement_nn = model(X_with_time)

        # Compute energy and forces and loss
        loss,Residual_Norm = tf.main(nEl,connectivity, displacement_nn, preCalc)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss:.6f}, Residual_norm: {Residual_Norm:.12f}")

        with open(file_path, "a") as file:
            file.write(f"Epoch {epoch + 1}, Loss: {total_loss:.12f}\n")

        with open(file_residual_force,"a") as file:
            file.write(f"Epoch {epoch + 1}, Residual_norm: {Residual_Norm:.12f}\n")

        if epoch == 100000:
            os.system(f'python "{os.path.join(source_folder, "Plot_EpochvsLoss.py")}"')
        # Save the model
    torch.save(model.state_dict(), file_pinn)




