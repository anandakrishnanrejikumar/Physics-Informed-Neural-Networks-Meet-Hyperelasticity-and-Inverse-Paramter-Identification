import os
from collections import namedtuple
import numpy as np
import torch
import torch.nn as nn
import yaml
import finite_element_model_fully_vectorized as tf
import finite_element_model_fully_vectorised_inverse as tfinv
import enumerate_dof as ed

script_dir = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(script_dir, "config.yaml")

with open(config_path, "r") as file:
    config = yaml.safe_load(file)

base_dir = config["Directory"]["File_directory"]

# Define the input folder path
input_folder = os.path.join(base_dir, "Input", "Element512")
output_folder = os.path.join(base_dir, "Output", "Element512")
file_path = os.path.join(output_folder, f"{config['Output File Names']['LossvsEpoch']}")
file_residual_force = os.path.join(output_folder, f"{config['Output File Names']['ResidualvsEpoch']}")
file_pinn = os.path.join(output_folder, f"{config['Output File Names']['PinnModel']}")
nEl = config["FEM Parameters"]["Total Elements"]
nLoadSteps = config["General"]["Number Load Steps"]

# Load data
nodes = np.loadtxt(os.path.join(input_folder, "nodes.txt"), delimiter=',', dtype=np.float64)  # Shape: (n_nodes, 3)
connectivity = np.loadtxt(os.path.join(input_folder, "connectivity.txt"), dtype=int, delimiter=',')

# Convert to PyTorch tensors
X = torch.tensor(nodes, dtype=torch.float64)  # Nodal coordinates
t = torch.linspace(0.1, 1, nLoadSteps)
tmp = t.repeat(nodes.shape[0], 1).T.flatten()
X_with_time = torch.cat((X.repeat(nLoadSteps, 1), tmp.view(-1, 1)), dim=1)
X_with_time.to(X.dtype)

# Precalculate shape function data
shape_grad, detJ = tf.Precalculate_shapeFunctionDetails(connectivity, nodes)
PreCalc = namedtuple('PreCalc', ['shape_grad', 'detJ'])
preCalc = PreCalc(shape_grad=shape_grad, detJ=detJ)

time_values = [f"{i * 0.05:.6f}" for i in range(1, nLoadSteps + 1)]
displacement_prescribed = torch.zeros((10, ed.total_nodes, 3)).to(torch.float64)
for i in range(nLoadSteps):
    input_folder = os.path.join(base_dir, "Input", "Element512")
    file_name = os.path.join(input_folder, f"displacement_{time_values[i]}.txt")  # Construct the file name
    tmp = np.loadtxt(file_name, delimiter=',', dtype=np.float64)
    tmp = torch.from_numpy(tmp)
    displacement_prescribed[i] = tmp[:, 3:6]
Loss_displacement = displacement_prescribed.view(nLoadSteps, -1, ed.dof_per_node)


class PINN(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=50, output_dim=3, num_hidden_layers=5):
        super(PINN, self).__init__()
        layers = [nn.Linear(input_dim, hidden_dim), nn.Tanh()]
        for _ in range(num_hidden_layers):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.Tanh())
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, X_with_time_data):
        u_nn = self.network(X_with_time_data)
        X1, X2, X3, time = X_with_time_data[:, 0:1], X_with_time_data[:, 1:2], X_with_time_data[:,
                                                                               2:3], X_with_time_data[:, 3:4]
        u_bar = 0.5 * time
        u_x = X1 * (X2 - 0.5) * u_nn[:, 0:1]
        u_y = (0.5 + X2) * u_bar + 4 * X2 * (0.5 - X2) * u_nn[:, 1:2]
        u_z = (X3 - 0.1) * u_nn[:, 2:3]
        return torch.cat([u_x, u_y, u_z], dim=1)


class InversePINN(PINN):
    def __init__(self):
        super(InversePINN, self).__init__()
        self.raw_bulkmod = nn.Parameter(
            torch.tensor(10, dtype=torch.float64, requires_grad=True))  # Example initial value
        self.softplus = nn.Softplus()
        self.raw_shearmod = nn.Parameter(torch.tensor(15, dtype=torch.float64, requires_grad=True))

    def forward(self, X_with_time_data):
        Bulkmod_nn = self.softplus(self.raw_bulkmod)
        shearmod_nn = self.softplus(self.raw_shearmod)
        u_corrected = super().forward(X_with_time_data)
        return u_corrected, Bulkmod_nn, shearmod_nn


def get_model(Model_Name):
    models = {"Forward": PINN, "Inverse": InversePINN}
    return models.get(Model_Name, None)


if __name__ == "__main__":
    model_name = config["PINN"]["Type"]
    if model_name:
        selected_model = get_model(model_name)
        model_instance = selected_model()
        model_instance = model_instance.double()

        optimizer = torch.optim.Adam(model_instance.parameters(), lr=1e-3)
        epochs = 100000 if model_name == "pinn" else 500000

        if model_name == "Forward":
            for epoch in range(epochs):
                total_loss = 0
                optimizer.zero_grad()
                displacement_nn = model_instance(X_with_time)
                loss, Residual_Norm = tf.main(nEl, connectivity, displacement_nn, preCalc)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss:.6f}, Residual_norm: {Residual_Norm:.12f}")
                with open(file_path, "a") as file:
                    file.write(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.6f}\n")

        else:
            force_all_load_steps = torch.tensor(
                np.loadtxt(os.path.join(input_folder, "force_global_10_load_steps.txt"), delimiter=','),
                dtype=torch.float64)
            required_dof_for_loss_calculation = tfinv.Extract_nodes_force_for_loss(connectivity, nodes)
            for epoch in range(epochs):
                total_loss = 0
                optimizer.zero_grad()
                displacement_nn, Bulk_mod, shear_mod = model_instance(X_with_time)
                loss, Bulkmodulus, shearmodulus, L_U, L_F, L_R = tfinv.main(nEl, connectivity, displacement_nn,
                                                                            preCalc,
                                                                            required_dof_for_loss_calculation,
                                                                            Bulk_mod, shear_mod, force_all_load_steps,
                                                                            Loss_displacement)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                print(
                    f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss:.12f},Bulk Modulus: {Bulkmodulus:.12f},Shear "
                    f"Modulus: {shearmodulus:.12f}")
                with open(file_path, "a") as file:
                    file.write(
                        f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss:.12f},Bulk Modulus: {Bulkmodulus:.12f},Shear "
                        f"Mod:{shearmodulus:.12f},L_R:{L_R:.12f},L_F:{L_F:.12f},L_U:{L_U:.12f},Shear_Mod_Gradient;"
                        f"{model_instance.raw_shearmod.grad},Bulk_Mod_Gradient:{model_instance.raw_bulkmod.grad}\n")

        torch.save(model_instance.state_dict(), file_pinn)
