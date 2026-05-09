import torch
import os
from pinn_model_inverse import PINN  # Import the PINN class
import yaml

script_dir = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(script_dir, "config.yaml")

with open(config_path, "r") as file:
    config = yaml.safe_load(file)
# Define the output folder path and model file
output_folder = config["Directory"]["File_directory"] # Change this to your actual output folder path
file_pinn = os.path.join(output_folder, "Report Writing","Inverse","Inverse08.pth")

# Recreate the model architecture (same structure as in training)
model = PINN(input_dim=4, hidden_dim=50, output_dim=3, num_hidden_layers=5)

# Check if trained model exists
if os.path.exists(file_pinn):
    print("Loading pre-trained model...")
    model.load_state_dict(torch.load(file_pinn))
    model.eval()  # Set the model to evaluation mode
else:
    print("No pre-trained model found. Exiting...")
    exit()

# Extracting Weights and Biases
print("\nExtracting Weights and Biases:")
for name, param in model.named_parameters():
    if "weight" in name:
        print(f"Layer: {name} - Weights:")
        print(param.data.numpy())  # Convert to numpy for better readability
        print("-" * 50)
    elif "bias" in name:
        print(f"Layer: {name} - Biases:")
        print(param.data.numpy())  # Convert to numpy for better readability
        print("-" * 50)

# Extract Young's Modulus and Poisson's Ratio
# Bulk_Modulus_NN = model.softplus(model.raw_bulkmod)
# Shear_Modulus_NN = model.softplus(model.raw_shearmod)  # Ensure Poisson ratio is in the range (0, 0.5)

# print(f"\nExtracted Trainable Parameters:")
# print(f"Bulk Modulus: {Bulk_Modulus_NN.item():.12f}")
# print(f"Shear Modulus: {Shear_Modulus_NN.item():.12f}")
