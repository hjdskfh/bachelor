
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os


style_file = "Presentation_style_1_adjusted_no_grid.mplstyle"

base_path = os.path.dirname(os.path.abspath(__file__))
mlp=os.path.join(base_path, style_file)

plt.style.use(r"C:\\Users\\leavi\\bachelor\\code\\Presentation_style_1_adjusted_no_grid.mplstyle")


# Load the CSV file
file_path = r"c:\Users\leavi\bachelor\data\eam_static_results_renormalized.csv"
data = pd.read_csv(file_path)

# Convert Transmission to dB
data['Transmission (dB)'] = 10 * np.log10(data['Transmission'])

# Plot the data
plt.figure(figsize=(10, 6))
plt.plot(data['Voltage (V)'], data['Transmission (dB)'], marker='o', label='Transmission (dB)')
plt.xlabel('Voltage (V)')
plt.ylabel('Transmission (dB)')
plt.title('Transmission vs Voltage (in dB)')
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.legend()
plt.tight_layout()
plt.savefig(r"c:\Users\leavi\bachelor\code\eam_transmission_dB.png", dpi=300)  # Save the figure
# Show the plot


# import os
# import pandas as pd
# import matplotlib.pyplot as plt

# # Path to the data folder
# data_folder = r"c:\Users\leavi\bachelor\data"

# # Iterate through all files in the folder
# for file_name in os.listdir(data_folder):
#     file_path = os.path.join(data_folder, file_name)
    
#     # Check if the file is a CSV
#     if file_name.endswith(".csv"):
#         print(f"Processing file: {file_name}")
        
#         # Load the CSV file
#         data = pd.read_csv(file_path)
        
#         # Ensure the file has at least two columns (x and y axes)
#         if data.shape[1] < 2:
#             print(f"Skipping {file_name}: Not enough columns for plotting.")
#             continue
        
#         # Extract column names for x and y axes
#         x_axis = data.columns[0]  # First column as x-axis
#         y_axes = data.columns[1:]  # Remaining columns as y-axes
        
#         # Plot the data
#         plt.figure(figsize=(10, 6))
#         for y_axis in y_axes:
#             plt.plot(data[x_axis], data[y_axis], label=y_axis)
        
#         # Add labels and title
#         plt.xlabel(x_axis)
#         plt.ylabel("Values")
#         plt.title(f"Plot for {file_name}")
#         plt.legend()
#         plt.grid(True, which='both', linestyle='--', linewidth=0.5)
#         plt.tight_layout()
        
#         # Show the plot
#         plt.show()