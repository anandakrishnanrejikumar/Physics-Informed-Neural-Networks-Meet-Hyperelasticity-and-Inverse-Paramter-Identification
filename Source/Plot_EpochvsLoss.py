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


    for i in range(1, 11):  # From 1 to 10
        filename = f"Forward0{i}Loss_ResidualVsEpoch.txt"
        file_path = os.path.join(base_dir, "Report Writing", "Forward", filename)

        try:
            df = pd.read_csv(file_path, sep='[;:, /]', engine='python', header=None)
            x = df.iloc[:, 1]  # Epochs
            y = df.iloc[:, 6]  # Loss
            Bulk_mod = 2.166666666666666666666665
            shear_mod = 1.0
            # y_rel = np.abs((y-Bulk_mod)/Bulk_mod) * 100
            plt.plot(x, y, label=f'Forward0{i}')  # Label for legend
        except FileNotFoundError:
            print(f"File not found: {filename}")
        except Exception as e:
            print(f"Error processing {filename}: {e}")

    # filename = f"Forward08Loss_ResidualVsEpoch.txt"
    # file_path = os.path.join(base_dir, "Report Writing", "Forward", filename)
    # df = pd.read_csv(file_path, sep='[;:, /]', engine='python', header=None)
    # x = df.iloc[:, 1]  # Epochs
    # y = df.iloc[:, -1]  # Loss
    # plt.plot(x, y, label=f'Forward08')

    # Plot settings
    plt.xscale('log')
    # plt.yscale('log')
    plt.xlim(1, 100000)
    plt.xlabel("Epochs (log scale)", fontsize=20)
    plt.ylabel("Loss", fontsize=20)
    plt.title("Loss vs Epoch", fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.legend(fontsize=18)
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)

    # Save final plot
    save_path = os.path.join(base_dir, "Output", "Element512", "Forward_LossVSEpoch_All.pdf")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
