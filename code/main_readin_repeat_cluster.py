from operator import length_hint
import time
from tkinter import N
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
import re
from collections import defaultdict
import datetime
import csv

# 909
# file_name = r"C:\Users\leavi\bachelor\stuff_from_cluster\2025_04_27\repeat_14_0_mpn_0_7_20db_att\20250427_133436_counts_repeat_max_12_2_50_20db.npz"
#900
file_name = r"C:\Users\leavi\bachelor\stuff_from_cluster\2025_04_27\repeat_900_deadtime_25ns\20250427_144011_counts_repeat.npz"

if os.path.exists(file_name):
    print("File exists!")
else:
    print("File does not exist!")
data = np.load(file_name, allow_pickle=True)
# for key in data.keys():
#     print(f"{key}")

print(data)

m_X_mud_in = data["global_len_wrong_x_dec"]
m_X_mus_in = data["global_len_wrong_x_non_dec"]
m_Z_mud_in = data["global_len_wrong_z_dec"]
m_Z_mus_in = data["global_len_wrong_z_non_dec"]
n_Z_mud_in = data["global_len_Z_checked_dec"]
n_Z_mus_in = data["global_len_Z_checked_non_dec"]
n_X_mud_in = data["global_X_P_calc_dec"]
n_X_mus_in = data["global_X_P_calc_non_dec"]
total_symbols = data["total_symbols"]
print(f"m_X_mud_in: {m_X_mud_in}")
print(f"m_X_mus_in: {m_X_mus_in}")
print(f"m_Z_mud_in: {m_Z_mud_in}")
print(f"m_Z_mus_in: {m_Z_mus_in}")
print(f"n_Z_mud_in: {n_Z_mud_in}")
print(f"n_Z_mus_in: {n_Z_mus_in}")
print(f"n_X_mud_in: {n_X_mud_in}")
print(f"n_X_mus_in: {n_X_mus_in}")
print(f"total_symbols: {total_symbols}")


config = SimulationConfig(None)
data_processor = DataProcessor(config)
# 909
# json_filepath = r"C:\Users\leavi\bachelor\stuff_from_cluster\2025_04_27\repeat_14_0_mpn_0_7_20db_att\simulation_config_20250427_115026.json"
#900
json_filepath = r"C:\Users\leavi\bachelor\stuff_from_cluster\2025_04_27\repeat_900_deadtime_25ns\simulation_config_20250427_113247.json"

# Load the JSON file
with open(json_filepath, 'r') as file:
    config_loaded = json.load(file)

config.mean_photon_nr = config_loaded["mean_photon_nr"]
config.mean_photon_decoy = config_loaded["mean_photon_decoy"]
config.p_decoy = config_loaded["p_decoy"]
config.p_z_alice = config_loaded["p_z_alice"]

desired_p_decoy_arr =  np.arange(0.02, 1, 0.02)
desired_p_z_alice_arr = np.arange(0.02, 1, 0.02)
# factor_x_mud_arr = np.arange(0, 100, 1)
factor_x_mud_arr = np.array([1])
length_multiply_arr = np.array([1e7, 1e8, 1e9])

for length_multiply in length_multiply_arr:
    for factor_x_mud_value in factor_x_mud_arr:
        for desired_p_decoy in desired_p_decoy_arr:
            for desired_p_z_alice in desired_p_z_alice_arr:
                # print(f"desired_p_decoy: {desired_p_decoy}, desired_p_z_alice: {desired_p_z_alice}")
                weight_Z = desired_p_z_alice
                weight_X = 1 - desired_p_z_alice
                weight_d = desired_p_decoy
                weight_s = 1 - desired_p_decoy

                # Apply the weights accordingly
                weighted_n_Z_mus = 4 * n_Z_mus_in * weight_Z * weight_s  # Z basis, signal
                weighted_n_Z_mud = 4 * n_Z_mud_in * weight_Z * weight_d  # Z basis, decoy
                weighted_n_X_mus = 4 * n_X_mus_in * weight_X * weight_s  # X basis, signal
                weighted_n_X_mud = 4 * n_X_mud_in * weight_X * weight_d  * factor_x_mud_value # X basis, decoy
                weighted_m_Z_mus = 4 * m_Z_mus_in * weight_Z * weight_s  # Z basis, signal
                weighted_m_Z_mud = 4 * m_Z_mud_in * weight_Z * weight_d  # Z basis, decoy
                weighted_m_X_mus = 4 * m_X_mus_in * weight_X * weight_s  # X basis, signal
                weighted_m_X_mud = 4 * m_X_mud_in * weight_X * weight_d  # X basis, decoy

                # print(f"weighted_n_Z_mus: {weighted_n_Z_mus}, weighted_n_Z_mud: {weighted_n_Z_mud}, weighted_n_X_mus: {weighted_n_X_mus}, weighted_n_X_mud: {weighted_n_X_mud}")
                # print(f"weighted_m_Z_mus: {weighted_m_Z_mus}, weighted_m_Z_mud: {weighted_m_Z_mud}, weighted_m_X_mus: {weighted_m_X_mus}, weighted_m_X_mud: {weighted_m_X_mud}")

                # put desired p_decoy and p_z_alice in the config
                config.p_decoy = desired_p_decoy
                config.p_z_alice = desired_p_z_alice

                # factor to get up to a billion symbols
                if weighted_n_Z_mus != 0:
                    factor = length_multiply / weighted_n_Z_mus
                else:
                    factor = 1
                # factor = 1

                # print(f"factor: {factor}")
                # funktionierte auch mit 10^6

                skr = data_processor.calc_SKR(  weighted_n_Z_mus,
                                                weighted_n_Z_mud,
                                                weighted_n_X_mus,
                                                weighted_n_X_mud,
                                                weighted_m_Z_mus,
                                                weighted_m_Z_mud,
                                                weighted_m_X_mus,
                                                weighted_m_X_mud,
                                                total_symbols,
                                                factor
                                            )

                # print(f"SKR: {skr} ")#for factor {factor} and ")#total factor {factor_total}
                # print(f"total symbols: {total_symbols}")
#                 if not math.isnan(skr) and skr > 0:
#                     timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
#                     with open(f"deadtime_SKR_results_{timestamp}.txt", "a") as f:
#                         f.write(f"length_multiply: {length_multiply:.2e}, mpn_s: {config.mean_photon_nr:.2f}, mpn_d: {config.mean_photon_decoy:.2f}, desired_p_decoy: {desired_p_decoy:.2f}, desired_p_z_alice: {desired_p_z_alice:.2f}, factor_x_mud: {factor_x_mud_value}, SKR: {skr}\n")
#                     # raise ValueError(f"SKR: {skr} is not NaN and > 0 for p_decoy: {desired_p_decoy}, p_z_alice: {desired_p_z_alice}"
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
                # csv_filename = f"deadtime_SKR_results_{timestamp}_tot_sym_{total_symbols}.csv"
                # # Check if the file already exists to write the header only once
                # file_exists = os.path.isfile(csv_filename)
                # # Open the CSV file in append mode
                # with open(csv_filename, "a", newline="") as csvfile:
                #     csv_writer = csv.writer(csvfile)

                #     # Write the header if the file is being created for the first time
                #     if not file_exists:
                #         csv_writer.writerow(["length_multiply", "mpn_s", "mpn_d", "desired_p_decoy", "desired_p_z_alice", "factor_x_mud", "SKR"])

                #     # Write the data row
                #     if not math.isnan(skr) and skr > 0:
                #         csv_writer.writerow([
                #             length_multiply,
                #             config.mean_photon_nr,
                #             config.mean_photon_decoy,
                #             desired_p_decoy,
                #             desired_p_z_alice,
                #               factor_x_mud_value,
                #               skr
                #         ])
# Calculate the ratios
if n_Z_mus_in != 0:
    QBER_signal = m_Z_mus_in / n_Z_mus_in
else:
    QBER_signal = None  # Avoid division by zero

# Calculate QBER_decoy (ratio_m_Z_mud_in)
if n_Z_mud_in != 0:
    QBER_decoy = m_Z_mud_in / n_Z_mud_in
else:
    QBER_decoy = None

# Calculate Pherr_signal (ratio_m_X_mus_in)
if n_X_mus_in != 0:
    Pherr_signal = m_X_mus_in / n_X_mus_in
else:
    Pherr_signal = None

# Calculate Pherr_decoy (ratio_m_X_mud_in)
if n_X_mud_in != 0:
    Pherr_decoy = m_X_mud_in / n_X_mud_in
else:
    Pherr_decoy = None

with open(f"deadtime_SKR_results_{timestamp}.txt", "a") as f:
    f.write(f"n_Z_mus_in: {n_Z_mus_in}, n_Z_mud_in: {n_Z_mud_in}, n_X_mus_in: {n_X_mus_in}, n_X_mud_in: {n_X_mud_in}\n")
    f.write(f"m_Z_mus_in: {m_Z_mus_in}, m_Z_mud_in: {m_Z_mud_in}, m_X_mus_in: {m_X_mus_in}, m_X_mud_in: {m_X_mud_in}\n")
    f.write(f"QBER_signal: {QBER_signal}, QBER_decoy: {QBER_decoy}, Pherr_signal: {Pherr_signal}, Pherr_decoy: {Pherr_decoy}\n")
    f.write(f"total symbols: {total_symbols}\n")