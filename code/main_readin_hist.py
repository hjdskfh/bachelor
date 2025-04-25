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