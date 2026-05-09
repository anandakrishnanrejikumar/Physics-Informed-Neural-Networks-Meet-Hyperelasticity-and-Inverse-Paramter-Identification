import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import yaml

# Load configuration file
script_dir = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(script_dir, "config.yaml")

with open(config_path, "r") as file:
    config = yaml.safe_load(file)

base_dir = config["Directory"]["File_directory"]

if __name__ == "__main__":
    plt.figure(figsize=(10, 6))

    # File path
    filename = "Force_Summed.xlsx"
    file_path = os.path.join(base_dir, "Report Writing", "Forward", filename)

    # Read Excel file
    df = pd.read_excel(file_path)

    # Extract columns
    x = df.iloc[:, 0]           # Load steps
    fem_y = df.iloc[:, 1]       # FEM force (red points)
    predicted_y = df.iloc[:, 2] # Predicted force (line)

    # Plot FEM data as red points
    plt.plot(x, fem_y, 'ro', label='FEM')

    # Plot predicted data as a line
    plt.plot(x, predicted_y, 'b-', label='PINN')

    # Labels and formatting
    plt.xlabel("Applied Displacement",fontsize=20)
    plt.ylabel("Force",fontsize=20)
    plt.title("Force vs Applied Displacement",fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.legend(fontsize=18)

    plt.grid(True, which="both", linestyle="--", linewidth=0.5)

    # Save final plot
    save_path = os.path.join(base_dir, "Output", "Element512", "Forward_ForceVSLoadStep.pdf")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)  # Ensure directory exists
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
