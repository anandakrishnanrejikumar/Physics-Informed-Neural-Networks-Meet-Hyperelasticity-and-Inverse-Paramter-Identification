import os
import matplotlib.pyplot as plt
import pandas as pd
import yaml

script_dir = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(script_dir, "config.yaml")

with open(config_path, "r") as file:
    config = yaml.safe_load(file)
#512
base_dir = config["Directory"]["File_directory"]
if __name__ == "__main__":
    # Load the data
    file_path1 = os.path.join(base_dir,"Report Writing","HyperParameter Study","EpochVsLosstestTanhBulkandShear500000_100000_2000_1Inverse.txt")
    # save_path1 = os.path.join(base_dir,"Output","Element512","name.png")  # Output file path
    # file_path2 = os.path.join(base_dir,"Output","Element512","Residual.txt")
    # save_path2 = os.path.join(base_dir,"Output","Element512","Residual.png")

    df1 = pd.read_csv(file_path1, sep='[;:, /]', engine='python', header=None) # Adjust separator if needed
    # df2 = pd.read_csv(file_path2, sep='\s+', header=None)
    # Define x and y values
    a = 0
    x = df1.iloc[a:,1]  # First column as x-axis (Epochs)
    # i = [6,10,13,15,17,19]
    i = [10,13]
    # Title_names = ["Loss","Bulk Modulus","Shear Modulus","L_R","L_F","L_U"]
    Title_names = ["Bulk Modulus", "Shear Modulus"]
    for index,i in enumerate(i,start=0):
        save_path1 = os.path.join(base_dir, "Report Writing", "HyperParameter Study", f"Epochvs500000_100000_2000_1{Title_names[index]}.pdf")
        y_columns = df1.iloc[a:,i]  # Remaining columns as y-axis (Loss values)

        # Create plot
        plt.figure(figsize=(10, 6))
        plt.plot(x, y_columns, label=Title_names[index])

        # Mark the last y value
        last_y = y_columns.iloc[100000]
        plt.axhline(y=last_y, color='red', linestyle=':', linewidth=1.5,
                    label=f'Final {Title_names[index]} = {last_y:.6f}')
        # Optional: add annotation
        # plt.annotate(f'{last_y:.4f}', xy=(x.iloc[-1], last_y), xytext=(-60, 10),
        #              textcoords='offset points', arrowprops=dict(arrowstyle='->'), fontsize=9, color='red')

        # Set log scale for x-axis and define limits
        plt.xscale('log')
        plt.xlim(a + 1, 100000)

        plt.xlabel("Epochs (log scale)", fontsize=20)
        plt.ylabel(f"{Title_names[index]}",fontsize=20)
        plt.title(f"Epoch vs {Title_names[index]}",fontsize=20)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.legend(fontsize=15)
        plt.grid(True, which="both", linestyle="--", linewidth=0.5)

        # Save the plot
        plt.savefig(save_path1, dpi=300, bbox_inches='tight')
    # plt.show()

    # y_columns = df2.iloc[:,3]  # Remaining columns as y-axis (Loss values)
    #
    # # Create plot
    # plt.figure(figsize=(10, 6))
    # # for i, col in enumerate(y_columns, start=1):
    # #     plt.plot(x, df[col], label=f'Series {i}')
    #
    # plt.plot(x,df2[3])
    #
    # # Set log scale for x-axis and define limits
    # plt.xscale('log')
    # plt.xlim(1, 100000)  # Set x-axis limits from 1 to 100000
    #
    # # Get min and max y values for setting refined ticks
    # y_min, y_max = df2.iloc[:, 3].min().min(), df2.iloc[:, 3].max().max()
    #
    # # Refine y-ticks with more granularity
    # num_ticks = 10  # Increase this number for finer tick marks
    # y_ticks = np.linspace(y_min, y_max, num_ticks)  # Generate evenly spaced tick marks
    # plt.yticks(y_ticks)  # Apply refined ticks
    #
    # plt.xlabel("Epochs (log scale)")
    # plt.ylabel("Residual")
    # plt.title("Epoch vs Residual")
    # plt.legend()
    # plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    #
    # # Save the plot
    # plt.savefig(save_path2, dpi=300, bbox_inches='tight')  # High-resolution save
    #
    # # Show the plot (optional)
    # # plt.show()