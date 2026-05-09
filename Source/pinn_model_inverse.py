import torch
import torch.nn as nn
import numpy as np
from collections import namedtuple
import finite_element_model_fully_vectorised_inverse as tfinv
from enumerate_dof import dof_matrix
import os
import yaml
import enumerate_dof as ed

script_dir = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(script_dir, "config.yaml")

with open(config_path, "r") as file:
    config = yaml.safe_load(file)

base_dir = config["Directory"]["File_directory"]
nLoadSteps = config["General"]["Number Load Steps"]

# Define the input folder path
input_folder = os.path.join(base_dir, "Input", "Element512")
output_folder = os.path.join(base_dir, "Output", "Element512")
source_folder = os.path.join(base_dir, "Source")
file_path = os.path.join(output_folder, f"{config['Output File Names']['LossvsEpoch']}")
file_residual_force = os.path.join(output_folder, f"{config['Output File Names']['ResidualvsEpoch']}")
file_pinn = os.path.join(output_folder, f"{config['Output File Names']['PinnModel']}")
nEl = config["FEM Parameters"]["Total Elements"]

# Load data
nodes = np.loadtxt(os.path.join(input_folder, "nodes.txt"), delimiter=',', dtype=np.float64)  # Shape: (n_nodes, 3)
connectivity = np.loadtxt(os.path.join(input_folder, "connectivity.txt"), dtype=int, delimiter=',')
force_all_load_steps = np.loadtxt(os.path.join(input_folder, "force_global_10_load_steps.txt"), delimiter=',',
                                  dtype=np.float64)
# Delete
time_values = [f"{i * 0.05:.6f}" for i in range(1, nLoadSteps + 1)]
displacement_prescribed = torch.zeros((10, ed.total_nodes, 3)).to(torch.float64)
for i in range(nLoadSteps):
    input_folder = os.path.join(base_dir, "Input", "Element512")
    file_name = os.path.join(input_folder, f"displacement_{time_values[i]}.txt")  # Construct the file name
    tmp = np.loadtxt(file_name, delimiter=',', dtype=np.float64)
    tmp = torch.from_numpy(tmp)
    displacement_prescribed[i] = tmp[:, 3:6]
Loss_displacement = displacement_prescribed.view(nLoadSteps, -1, ed.dof_per_node)

# Convert to PyTorch tensors
X = torch.tensor(nodes, dtype=torch.float64)  # Nodal coordinates
t = torch.linspace(0.1, 1, nLoadSteps)  # Time values (here: <nLoadSteps> time steps)
tmp = t.repeat(nodes.shape[0], 1).T.flatten()  # 1350 x 1
X_with_time = torch.cat((X.repeat(nLoadSteps, 1), tmp.view(-1, 1)), dim=1)
X_with_time = X_with_time.to(dtype=torch.float64)
force_all_load_steps = torch.tensor(force_all_load_steps, dtype=torch.float64)

# Precalculate shape function data
# Assuming shape_grad and detJ are tensors or arrays computed earlier
shape_grad, detJ = tfinv.Precalculate_shapeFunctionDetails(connectivity, nodes)
PreCalc = namedtuple('PreCalc', ['shape_grad', 'detJ'])
preCalc = PreCalc(shape_grad=shape_grad, detJ=detJ)
required_dof_for_loss_calculation = tfinv.Extract_nodes_force_for_loss(connectivity, nodes)


# PINN
class PINN(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=50, output_dim=3, num_hidden_layers=5):
        super(PINN, self).__init__()
        # Apply Softplus to ensure Young's Modulus is always positive
        self.raw_bulkmod = nn.Parameter(
            torch.tensor(10, dtype=torch.float64, requires_grad=True))  # Example initial value
        self.softplus = nn.Softplus()
        self.raw_shearmod = nn.Parameter(torch.tensor(15, dtype=torch.float64, requires_grad=True))

        # Build layers
        layers = [nn.Linear(input_dim, hidden_dim), nn.Tanh()]  # Input layer with activation
        for _ in range(num_hidden_layers):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.Tanh())  # Add ReLU after each hidden layer
        layers.append(nn.Linear(hidden_dim, output_dim))  # Output layer

        self.network = nn.Sequential(*layers)  # Combine all layers into a single Sequential module

    def forward(self, X_with_timedata):
        Bulkmod_nn = self.softplus(self.raw_bulkmod)
        shearmod_nn = self.softplus(self.raw_shearmod)

        # Apply Sigmoid and scale to ensure Poisson ratio (nu) is between 0 and 0.5
        u_nn = self.network(X_with_timedata)  # Output from the network

        # Extract coordinates and time
        X1 = X_with_timedata[:, 0:1]  # First coordinate (x)
        X2 = X_with_timedata[:, 1:2]  # Second coordinate (y)
        X3 = X_with_timedata[:, 2:3]  # Third coordinate (z)
        time = X_with_timedata[:, 3:4]  # Time
        u_bar = 0.5 * time  # Applied displacement

        u_x = X1 * (X2 - 0.5) * u_nn[:, 0:1]
        u_y = (0.5 + X2) * u_bar + 4 * X2 * (0.5 - X2) * u_nn[:, 1:2]
        u_z = (X3 - 0.1) * u_nn[:, 2:3]

        # Horizontally concatenate displacement vectors
        u_corrected = torch.cat([u_x, u_y, u_z], dim=1)

        return u_corrected, Bulkmod_nn, shearmod_nn


# Model, optimizer, and training setup
if __name__ == "__main__":

    model = PINN(input_dim=4, hidden_dim=50, output_dim=3, num_hidden_layers=5)
    model = model.double()
    # optimizer_nn = torch.optim.Adam([param for name, param in model.named_parameters() if
    #                                  "raw_shearmod" not in name and "raw_bulkmod" not in name], lr=1e-3)
    # optimizer_material = torch.optim.Adam([model.raw_shearmod,model.raw_bulkmod],
    #                                       lr=0.001)  # Learning rate for material params

    # scheduler_nn = torch.optim.lr_scheduler.StepLR(optimizer_nn, step_size=5000, gamma=0.95)
    # scheduler_material = torch.optim.lr_scheduler.StepLR(optimizer_material, step_size=5000, gamma=0.95)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    # lbfgs_optimizer = torch.optim.LBFGS(model.parameters(), lr=1.0, max_iter=20, history_size=10)
    # Training loop
    epochs = 500000
    for epoch in range(epochs):
        total_loss = 0
        optimizer.zero_grad()
        displacement_nn, Bulk_mod, shear_mod = model(X_with_time)

        # Compute energy and forces and loss
        loss, Bulkmodulus, shearmodulus, L_U, L_F, L_R = tfinv.main(nEl, connectivity, displacement_nn,
                                                                    preCalc,
                                                                    required_dof_for_loss_calculation,
                                                                    Bulk_mod, shear_mod, force_all_load_steps,
                                                                    Loss_displacement)  #add force_all_load_steps,removed for extraction currently

        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        print(
            f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss:.12f},Bulk Modulus: {Bulkmodulus:.12f},Shear Modulus: {shearmodulus:.12f}")

        with open(file_path, "a") as file:
            file.write(
                f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss:.12f},Bulk Modulus: {Bulkmodulus:.12f},Shear Mod:{shearmodulus:.12f},L_R:{L_R:.12f},L_F:{L_F:.12f},L_U:{L_U:.12f},Shear_Mod_Gradient;{model.raw_shearmod.grad},Bulk_Mod_Gradient:{model.raw_bulkmod.grad}\n")

    # Save the model
    torch.save(model.state_dict(), file_pinn)
