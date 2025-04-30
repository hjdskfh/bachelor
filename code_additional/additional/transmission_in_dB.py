# import pandas as pd

# # Load the CSV file
# file_path = r"c:\Users\leavi\Downloads\eam_static_results [MConverter.eu].csv"
# data = pd.read_csv(file_path)
# data.columns = data.columns.str.strip()
# # Find the transmission value at 0.5 V
# transmission_at_0_5V = data.loc[data['Voltage (V)'] == 0.5, 'Transmission'].values[0]

# # Renormalize the transmission values
# data['Transmission'] = data['Transmission'] / transmission_at_0_5V

# # Save the renormalized data to a new CSV file
# output_file_path = r"c:\Users\leavi\Downloads\eam_static_results_renormalized.csv"
# data.to_csv(output_file_path, index=False)

# print(f"Renormalized data saved to {output_file_path}")


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

plt.style.use("C:\\Users\\leavi\\bachelor\\code\\Presentation_style_1_adjusted_no_grid.mplstyle")


# Load the CSV file
file_path = r"c:\Users\leavi\bachelor\data\eam_static_results_renormalized.csv"
data = pd.read_csv(file_path)

# Convert Transmission to dB
data['Transmission (dB)'] = 10 * np.log10(data['Transmission'])

# Plot the data
# Plot the data
plt.figure(figsize=(10, 6))
plt.plot(data['Voltage (V)'], data['Transmission (dB)'], color = 'green', label='Transmission (dB)')

# Add vertical lines
plt.axvline(x=-1.3, color='red',  label='Original EAM Voltage Setting')
plt.axvline(x=0.2, color='red')
plt.axvline(x=-2.1, color='blue', label='Modified EAM Voltage Settings')
plt.axvline(x=0.4, color='blue')

# Add labels, title, and grid
plt.xlabel('Voltage (V)')
plt.ylabel('Transmission (dB)')
# plt.title('Transmission vs Voltage (in dB)')
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.legend()
plt.tight_layout()

# Save the plot to a file
plt.savefig(r"c:\Users\leavi\Downloads\transmission_vs_voltage.png")

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