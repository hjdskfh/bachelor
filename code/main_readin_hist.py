import time
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

# file_name = '../stuff_from_cluster/2025_04_13/20250410_190215_histograms.npz'
# file_name = r'C:\Users\leavi\bachelor\stuff_from_cluster\2025_04_13\20250413_150050_histograms_fixed.npz'
# file_name = r'C:\Users\leavi\OneDrive\Dokumente\Uni\Semester 7\NeuMoQP\Programm\stuff_from_cluster\2025_04_13\random\20250413_172141_histograms_random.npz'
# file_name = r'C:\Users\leavi\OneDrive\Dokumente\Uni\Semester 7\NeuMoQP\Programm\stuff_from_cluster\2025_04_13\fixed\20250413_173800_histograms_fixed.npz'
# file_name = r'C:\Users\leavi\OneDrive\Dokumente\Uni\Semester 7\NeuMoQP\Programm\stuff_from_cluster\2025_04_21\histograms_random_results_later.npz'
# file_name = r'C:\Users\leavi\bachelor\results_data\20250414_154555_histograms_random.npz'
# file_name = r'C:\Users\leavi\bachelor\results_data\20250415_112451_histograms_random.npz'
# file_name = r'C:\Users\leavi\bachelor\stuff_from_cluster\2025_04_23\hist_rand\20250423_041529_histograms_random.npz'
# file_name = r'C:\Users\leavi\bachelor\stuff_from_cluster\2025_04_18\20250417_160300_histograms_fixed.npz'
# file_name = r'C:\Users\leavi\bachelor\stuff_from_cluster\2025_04_23\20250423_110610_histograms_random.npz'
# file_name = r'C:\Users\leavi\bachelor\results\20250424_134045_t_and_input_power_DLI_mixed_pulses.npz'
# file_name = r'C:\Users\leavi\bachelor\stuff_from_cluster\2025_04_24\20250424_125225_histograms_random_end.npz'

# if os.path.exists(file_name):
#     print("File exists!")
# else:
#     print("File does not exist!")
# data = np.load(file_name, allow_pickle=True)
# for key in data.keys():
#     print(f"{key}")

# t_original = data["t_original"]
# t_upsampled = data["t_upsampled"]
# input_power_original = data["input_power_original"]
# input_power_upsampled = data["input_power_upsampled"]

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


# bins_per_symbol_hist = data["bins_per_symbol_hist"]
# final_time_one_symbol = data["final_time_one_symbol"]
# global_histogram_counts_x = data["global_histogram_counts_x"]
# global_histogram_counts_z = data["global_histogram_counts_z"]
# # final_lookup_array = data["final_lookup_array"]
# total_symbols = data["total_symbols"]
# # total_symbols = data["total_samples"]
# final_combined_list_array = data["final_combined_list_array"]
# p_decoy = 0.1

# # DataProcessor.plot_histogram_batch(bins_per_symbol_hist, final_time_one_symbol,
# #                             global_histogram_counts_x, global_histogram_counts_z,
# #                             final_lookup_array, total_symbols, start_symbol=0, end_symbol=4, name="random")

# # start ymbol has to be the non decoy symbol
# DataProcessor.plot_histogram_batch_random(bins_per_symbol_hist, final_time_one_symbol,
#                             global_histogram_counts_x, global_histogram_counts_z,
#                             final_combined_list_array, total_symbols, start_pair=3, end_pair=5, name="random", p_decoy = None)

# len_wrong_x_dec = data["global_len_wrong_x_dec"]
# len_wrong_x_non_dec = data["global_len_wrong_x_non_dec"]
# len_wrong_z_dec = data["global_len_wrong_z_dec"]
# len_wrong_z_non_dec = data["global_len_wrong_z_non_dec"]
# len_Z_checked_dec = data["global_len_Z_checked_dec"]
# len_Z_checked_non_dec = data["global_len_Z_checked_non_dec"]
# X_P_calc_dec = data["global_X_P_calc_dec"]
# X_P_calc_non_dec = data["global_X_P_calc_non_dec"]
# total_symbols = data["total_symbols"]

config = SimulationConfig(None)
data_processor = DataProcessor(config)
json_filepath = r'C:\Users\leavi\bachelor\stuff_from_cluster\2025_04_25\simfor_200000_batch_sims_simulation_config_20250425_120954.json.json'

# Load the JSON file
with open(json_filepath, 'r') as file:
    config_loaded = json.load(file)

config.mean_photon_nr = config_loaded["mean_photon_nr"]
config.mean_photon_decoy = config_loaded["mean_photon_decoy"]
config.p_decoy = config_loaded["p_decoy"]
config.p_z_alice = config_loaded["p_z_alice"]

# factor_arr = np.array([1, 10, 100, 1000, 10000, 100000, 1000000, 10000000, 100000000, 1000000000, 1e10, 1e11, 1e12, 1e13, 1e14, 1e15])
# factor_arr_total = np.array([1, 10, 100, 1000, 10000, 100000, 1000000, 10000000, 100000000, 1000000000, 1e10, 1e11, 1e12, 1e13, 1e14, 1e15])
# for factor in factor_arr:
#     for factor_total in factor_arr_total:
        # len_wrong_x_dec = 1 * factor
        # len_wrong_x_non_dec = 4 * factor
        # len_wrong_z_dec = 135 * factor
        # len_wrong_z_non_dec = 141 * factor
        # len_Z_checked_dec = 2539 * factor
        # len_Z_checked_non_dec = 4920 * factor
        # X_P_calc_dec = 32 * factor
        # X_P_calc_non_dec = 96 * factor
        # total_symbols = 1200000 * factor
config.p_decoy = 0.19
config.p_z_alice = 0.8

len_wrong_x_dec = 0.152 
len_wrong_x_non_dec = 2.592 
len_wrong_z_dec = 82.08
len_wrong_z_non_dec = 365.472 
len_Z_checked_dec = 1543.7
len_Z_checked_non_dec = 12726.7 
X_P_calc_dec = 62.2 
X_P_calc_non_dec = 4.864 
total_symbols = 1200000

print(f"len_wrong_x_dec: {len_wrong_x_dec}, len_wrong_x_non_dec: {len_wrong_x_non_dec}, len_wrong_z_dec: {len_wrong_z_dec}, len_wrong_z_non_dec: {len_wrong_z_non_dec}")
print(f"len_Z_checked_dec: {len_Z_checked_dec}, len_Z_checked_non_dec: {len_Z_checked_non_dec}")
print(f"X_P_calc_dec: {X_P_calc_dec}, X_P_calc_non_dec: {X_P_calc_non_dec}")
print(f"total_symbols: {total_symbols}")


skr = data_processor.calc_SKR(len_wrong_x_dec, len_wrong_x_non_dec, len_wrong_z_dec, len_wrong_z_non_dec, 
                len_Z_checked_dec, len_Z_checked_non_dec, X_P_calc_dec, X_P_calc_non_dec, total_symbols)

print(f"SKR: {skr} ")#for factor {factor} and ")#total factor {factor_total}")

#  len_wrong_x_dec = 20 * factor
# len_wrong_x_non_dec = 212 * factor
# len_wrong_z_dec = 224 * factor
# len_wrong_z_non_dec = 1951 * factor
# len_Z_checked_dec = 1619 * factor
# len_Z_checked_non_dec = 95060 * factor
# X_P_calc_dec = 24 * factor
# X_P_calc_non_dec = 5232 * factor
# total_symbols = 4000000 * factor