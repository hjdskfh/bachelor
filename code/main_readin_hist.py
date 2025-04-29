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

# file_name = '../stuff_from_cluster/2025_04_13/20250410_190215_histograms.npz'
# file_name = r'C:\Users\leavi\bachelor\stuff_from_cluster\2025_04_13\20250413_150050_histograms_fixed.npz'
# file_name = r'C:\Users\leavi\OneDrive\Dokumente\Uni\Semester 7\NeuMoQP\Programm\stuff_from_cluster\2025_04_13\random\20250413_172141_histograms_random.npz'
# file_name = r'C:\Users\leavi\OneDrive\Dokumente\Uni\Semester 7\NeuMoQP\Programm\stuff_from_cluster\2025_04_13\fixed\20250413_173800_histograms_fixed.npz'
# file_name = r'C:\Users\leavi\OneDrive\Dokumente\Uni\Semester 7\NeuMoQP\Programm\stuff_from_cluster\2025_04_21\histograms_random_results_later.npz'
# file_name = r'C:\Users\leavi\bachelor\results_data\20250414_154555_histograms_random.npz'
# file_name = r'C:\Users\leavi\bachelor\results_data\20250415_112451_histograms_random.npz'
# file_name = r'C:\Users\leavi\bachelor\stuff_from_cluster\2025_04_23\hist_rand\20250423_041529_histograms_random.npz'
# file_name = r'C:\Users\leavi\bachelor\stuff_from_cluster\2025_04_18\20250417_160300_histograms_fixed.npz'
# 27.
# file_name = r'C:\Users\leavi\bachelor\stuff_from_cluster\2025_04_28_neu\20250428_093031_histograms_random_con_12_12_total_batches_300_600_bins_60_80.npz'
# 28.
# file_name = r'C:\Users\leavi\bachelor\rest_cluster\results_data\20250428_093031_histograms_random_con_12_12_total_batches_300_600_bins_60_80.npz'
# # 18. hist fixed
# file_name = r'C:\Users\leavi\bachelor\stuff_from_cluster\2025_04_18\20250417_160300_histograms_fixed.npz'
# # 16. hist fixed
file_name = r'C:\Users\leavi\bachelor\stuff_from_cluster\2025_04_16\20250415_210351_histograms_fixed.npz'

if os.path.exists(file_name):
    print("File exists!")
else:
    print("File does not exist!")
data = np.load(file_name, allow_pickle=True)
for key in data.keys():
    print(f"{key}")

# print(data)
# np.set_printoptions(threshold=np.inf)

# hist_x = data["hist_x"]
# hist_z = data["hist_z"]
# t_sym = data["t_sym"]
# lookup_array = data["lookup_array"]
# print(f"hist_x: {hist_x}")
# print(f"hist_z: {hist_z}")
# print(f"t_sym: {t_sym}")
# print(f"lookup_array: {lookup_array}")

bins_per_symbol_hist = data["bins_per_symbol_hist"]
final_time_one_symbol = data["final_time_one_symbol"]
global_histogram_counts_x = data["global_histogram_counts_x"]
global_histogram_counts_z = data["global_histogram_counts_z"]
final_lookup_array = data["final_lookup_array"]
total_symbols = data["total_symbols"]
# # total_symbols = data["total_samples"]
# final_combined_list_array = data["final_combined_list_array"]
print(f"total_symbols: {total_symbols}")

# start_pair_arr = np.array([0,6,12,18,24,30])
# for i in start_pair_arr:
DataProcessor.plot_histogram_batch(bins_per_symbol_hist, final_time_one_symbol,
                                global_histogram_counts_x, global_histogram_counts_z,
                                final_lookup_array, total_symbols, start_symbol=0, end_symbol=15, name="random")
# global_histogram_counts_x, bins_per_symbol_hist = DataProcessor.combine_bins(global_histogram_counts_x, bins_per_symbol_hist)
# global_histogram_counts_z, bins_per_symbol_hist = DataProcessor.combine_bins(global_histogram_counts_z, bins_per_symbol_hist)

# start_pair_arr = np.array([0,6,12,18,24,30])
# for i in start_pair_arr:
#     DataProcessor.plot_histogram_batch_random(bins_per_symbol_hist, final_time_one_symbol,
#                                 global_histogram_counts_x, global_histogram_counts_z,
#                                 final_combined_list_array, total_symbols, start_pair=i, end_pair=i+2, name="random")
#0to2
#6to8
