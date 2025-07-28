"""
main_readin_hist.py

Reads and processes histogram data from QKD simulations for fixed and random symbol pairs from PC or from the Cluster, including plotting and style configuration.
"""

import time
from tracemalloc import start
from datamanager import DataManager
from config import SimulationConfig
from simulationmanager import SimulationManager
from saver import Saver
from dataprocessor import DataProcessor
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import math
import json

plt.style.use("C:\\Users\\leavi\\bachelor\\code\\Presentation_style_1_adjusted_no_grid.mplstyle")


# # 16. hist fixed
file_name = r'C:\Users\leavi\bachelor\stuff_from_cluster\2025_04_16\20250415_210351_histograms_fixed.npz'

# check if all keys that are extracted are present in the file
if os.path.exists(file_name):
    print("File exists!")
else:
    print("File does not exist!")
data = np.load(file_name, allow_pickle=True)
for key in data.keys():
    print(f"{key}")

# Extract raw data from the simulation file
bins_per_symbol_hist = data["bins_per_symbol_hist"]
final_time_one_symbol = data["final_time_one_symbol"]
global_histogram_counts_x = data["global_histogram_counts_x"]
global_histogram_counts_z = data["global_histogram_counts_z"]
final_lookup_array = data["final_lookup_array"]
total_symbols = data["total_symbols"]
print(f"total_symbols: {total_symbols}")

# Plot the histogram data for fixed symbols
DataProcessor.plot_histogram_batch(bins_per_symbol_hist, final_time_one_symbol,
                                global_histogram_counts_x, global_histogram_counts_z,
                                final_lookup_array, total_symbols, start_symbol=0, end_symbol=5, name="4")

# Possibility to combine bins in the plotting of the histogram                        
# global_histogram_counts_x, bins_per_symbol_hist = DataProcessor.combine_bins(global_histogram_counts_x, bins_per_symbol_hist)
# global_histogram_counts_z, bins_per_symbol_hist = DataProcessor.combine_bins(global_histogram_counts_z, bins_per_symbol_hist)

# Plot the historgram for random symbols
# start_pair_arr = np.array([0,6,12,18,24,30])
# for i in start_pair_arr:
#     DataProcessor.plot_histogram_batch_random(bins_per_symbol_hist, final_time_one_symbol,
#                                 global_histogram_counts_x, global_histogram_counts_z,
#                                 final_combined_list_array, total_symbols, start_pair=i, end_pair=i+2, name="random")

