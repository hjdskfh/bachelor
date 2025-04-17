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
# file_name = r'C:\Users\leavi\OneDrive\Dokumente\Uni\Semester 7\NeuMoQP\Programm\stuff_from_cluster\2025_04_13\other\20250413_173800_histograms_fixed.npz'
# file_name = r'C:\Users\leavi\bachelor\results_data\20250414_154555_histograms_random.npz'
# file_name = r'C:\Users\leavi\bachelor\results_data\20250415_112451_histograms_random.npz'

# if os.path.exists(file_name):
#     print("File exists!")
# else:
#     print("File does not exist!")
# data = np.load(file_name, allow_pickle=True)
# for key in data.keys():
#     print(f"{key}")

'''bins_per_symbol_hist = data["bins_per_symbol_hist"]
final_time_one_symbol = data["final_time_one_symbol"]
global_histogram_counts_x = data["global_histogram_counts_x"]
global_histogram_counts_z = data["global_histogram_counts_z"]
final_lookup_array = data["final_lookup_array"]
total_symbols = data["total_symbols"]
# total_symbols = data["total_samples"]

# global_histogram_counts_z = np.empty_like(global_histogram_counts_z)

DataProcessor.plot_histogram_batch(bins_per_symbol_hist, final_time_one_symbol,
                            global_histogram_counts_x, global_histogram_counts_z,
                            final_lookup_array, total_symbols, start_symbol=0, end_symbol=10, name="random", leave_z = True)'''


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
json_filepath = r"c:\Users\leavi\bachelor\stuff_from_cluster\2025_04_17\repeat\simulation_config_20250416_104314.json"

# Load the JSON file
with open(json_filepath, 'r') as file:
    config_loaded = json.load(file)

config.mean_photon_nr = config_loaded["mean_photon_nr"]
config.mean_photon_decoy = config_loaded["mean_photon_decoy"]
config.p_decoy = config_loaded["p_decoy"]
config.p_z_alice = config_loaded["p_z_alice"]

factor_arr = np.array([1, 10, 100, 1000, 10000, 100000, 1000000, 10000000, 100000000, 1000000000])
for factor in factor_arr:
        
    len_wrong_x_dec = 0 * factor
    len_wrong_x_non_dec = 0 * factor
    len_wrong_z_dec = 5430 * factor
    len_wrong_z_non_dec = 7906 * factor
    len_Z_checked_dec = 52238 * factor
    len_Z_checked_non_dec = 310049 * factor
    X_P_calc_dec = 2152 * factor
    X_P_calc_non_dec = 35200 * factor
    total_symbols = 24000000 * factor


    
    skr = data_processor.calc_SKR(len_wrong_x_dec, len_wrong_x_non_dec, len_wrong_z_dec, len_wrong_z_non_dec, 
                    len_Z_checked_dec, len_Z_checked_non_dec, X_P_calc_dec, X_P_calc_non_dec, total_symbols)

    print(f"SKR: {skr} for factor {factor}")