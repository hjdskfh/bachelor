from operator import length_hint
import time
from tkinter import N

from matplotlib import scale
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
# json_filepath = r"C:\Users\leavi\bachelor\stuff_from_cluster\2025_04_27\repeat_14_0_mpn_0_7_20db_att\simulation_config_20250427_115026.json"
#900
# json_filepath = r"C:\Users\leavi\bachelor\stuff_from_cluster\2025_04_27\repeat_900_deadtime_25ns\simulation_config_20250427_113247.json"
# 14
# json_filepath = r'C:\Users\leavi\bachelor\stuff_from_cluster\2025_04_28\repeat\14_simulation_config_20250428_075954.json'
# 1
# json_filepath = r'C:\Users\leavi\bachelor\stuff_from_cluster\2025_04_28\repeat\1_simulation_config_20250428_041909.json'
# 10
# json_filepath = r'C:\Users\leavi\bachelor\stuff_from_cluster\2025_04_28\repeat\10_simulation_config_20250428_104035.json'
# Mo nach 1 att
# json_filepath = r'C:\Users\leavi\bachelor\stuff_from_cluster\2025_04_28_neu\repeat_diff_att\simulation_config_20250428_144433.json'
# Mo nach 2 att
# json_filepath = r'C:\Users\leavi\bachelor\stuff_from_cluster\2025_04_28_neu\repeat_diff_att\simulation_config_20250428_145342.json'
# Mo nach 3 att
# json_filepath = r'C:\Users\leavi\bachelor\stuff_from_cluster\2025_04_28_neu\repeat_diff_att\simulation_config_20250428_145417.json'
# Mo nach 4 att
# json_filepath = r'C:\Users\leavi\bachelor\stuff_from_cluster\2025_04_28_neu\repeat_diff_att\simulation_config_20250428_145457.json'
# Mo Abend 1 att
# json_filepath = r'C:\Users\leavi\bachelor\stuff_from_cluster\2025_04_28_neu\repeat_diff_att_neu\simulation_config_20250428_180803.json'
# Mo abend 2 att
# json_filepath = r'C:\Users\leavi\bachelor\stuff_from_cluster\2025_04_28_neu\repeat_diff_att_neu\simulation_config_20250428_180905.json'
# Mo abend 3 att
# json_filepath = r'C:\Users\leavi\bachelor\stuff_from_cluster\2025_04_28_neu\repeat_diff_att_neu\simulation_config_20250428_180945.json'
# Mo abend 4 att
json_filepath = r'C:\Users\leavi\bachelor\stuff_from_cluster\2025_04_28_neu\repeat_diff_att_neu\simulation_config_20250428_181019.json'

# 909
# file_name = r"C:\Users\leavi\bachelor\stuff_from_cluster\2025_04_27\repeat_14_0_mpn_0_7_20db_att\20250427_133436_counts_repeat_max_12_2_50_20db.npz"
#900
# file_name = r"C:\Users\leavi\bachelor\stuff_from_cluster\2025_04_27\repeat_900_deadtime_25ns\20250427_144011_counts_repeat.npz"
# 14
# file_name = r'C:\Users\leavi\bachelor\stuff_from_cluster\2025_04_28\repeat\14_20250428_094849_counts_repeat_mpn_0.07_decoy_0.035_att_-3_total_2000000_batch_50_max_12_volt_0.0011_current_0.00041.npz'
# 1 
# file_name = r'C:\Users\leavi\bachelor\stuff_from_cluster\2025_04_28\repeat\1_20250428_060659_counts_repeat_mpn_0.15_decoy_0.075_att_-3_total_2000000_batch_50_max_12_volt_0.011_current_0.00041.npz'
# 10
# file_name = r'C:\Users\leavi\bachelor\stuff_from_cluster\2025_04_28\repeat\10_20250428_122458_counts_repeat_mpn_0.3_decoy_0.15_att_-6_total_2000000_batch_50_max_12_volt_0.0011_current_0.00041.npz'
# Mo nach 1 att -> nur mit faktor 2 x_mud
# file_name = r'C:\Users\leavi\bachelor\stuff_from_cluster\2025_04_28_neu\repeat_diff_att\20250428_162803_counts_repeat_0_15_0_075_20000_-3_0_0011_0_00041.npz'
# Mo nach 2 att -> gar nicht
# file_name = r'C:\Users\leavi\bachelor\stuff_from_cluster\2025_04_28_neu\repeat_diff_att\20250428_164006_counts_repeat_job_id_6760986_fiber_attenuation_-6_decoy_0.125_non_decoy_0.25_samples_100000_simulations_in_batch_2_total_batches_10.npz'
# Mo nach 3 att -> gar nicht
# file_name = r'C:\Users\leavi\bachelor\stuff_from_cluster\2025_04_28_neu\repeat_diff_att\20250428_164252_counts_repeat_job_id_6760987_fiber_attenuation_-6_decoy_0.15_non_decoy_0.3_samples_20000_simulations_in_batch_2_total_batches_50.npz'
# Mo nach 4 att
# file_name = r'C:\Users\leavi\bachelor\stuff_from_cluster\2025_04_28_neu\repeat_diff_att\20250428_164512_counts_repeat_job_id_6760988_fiber_attenuation_-6_decoy_0.3_non_decoy_0.6_samples_20000_simulations_in_batch_2_total_batches_50.npz'

# if os.path.exists(file_name):
#     print("File exists!")
# else:
#     print("File does not exist!")
# data = np.load(file_name, allow_pickle=True)
# # for key in data.keys():
# #     print(f"{key}")

# print(data)


# ---------- TXT --------------
# Define the path to the text file
# Mo abend 1 att
# txt_file_path = r'C:\Users\leavi\bachelor\stuff_from_cluster\2025_04_28_neu\repeat_diff_att_neu\output_20250428_195044_n_samples_None_function_max_4_batch_100.txt'
# Mo abend 2 att
# txt_file_path = r'C:\Users\leavi\bachelor\stuff_from_cluster\2025_04_28_neu\repeat_diff_att_neu\output_20250428_195241_n_samples_None_function_max_4_batch_100.txt'
# Mo abend 3 att
# txt_file_path = r'C:\Users\leavi\bachelor\stuff_from_cluster\2025_04_28_neu\repeat_diff_att_neu\output_20250428_195343_n_samples_None_function_max_4_batch_100.txt'
# Mo abend 4 att
txt_file_path = r'C:\Users\leavi\bachelor\stuff_from_cluster\2025_04_28_neu\repeat_diff_att_neu\output_20250428_195900_n_samples_None_function_max_4_batch_100.txt'


# Initialize a dictionary to store the parsed data
data = {}


# Read the file line by line
with open(txt_file_path, 'r') as file:
    for line in file:
        # Split the line into key and value based on the colon
        if ':' in line:
            key, value = line.strip().split(':', 1)
            key = key.strip()  # Remove extra spaces around the key
            value = value.strip()  # Remove extra spaces around the value

            # Parse the value
            if value.startswith('[') and value.endswith(']'):  # Handle lists
                value = eval(value)  # Convert string representation of list to actual list
                if len(value) == 1:  # Convert single-element lists to scalars
                    value = value[0]
            elif value == 'None':  # Handle None values
                value = None
            elif value.replace('.', '', 1).isdigit():  # Handle numeric values
                value = float(value) if '.' in value else int(value)
            # Store the key-value pair in the dictionary
            data[key] = value

# Print the parsed data
print(data)

# Access specific variables like in the marked code
m_X_mud_in = data.get("global_len_wrong_x_dec", None)
m_X_mus_in = data.get("global_len_wrong_x_non_dec", None)
m_Z_mud_in = data.get("global_len_wrong_z_dec", None)
m_Z_mus_in = data.get("global_len_wrong_z_non_dec", None)
n_Z_mud_in = data.get("global_len_Z_checked_dec", None)
n_Z_mus_in = data.get("global_len_Z_checked_non_dec", None)
n_X_mud_in = data.get("global_X_P_calc_dec", None)
n_X_mus_in = data.get("global_X_P_calc_non_dec", None)
total_symbols = data.get("total_symbols", None)
print(f"m_X_mud_in: {m_X_mud_in}")
print(f"m_X_mus_in: {m_X_mus_in}")
print(f"m_Z_mud_in: {m_Z_mud_in}")
print(f"m_Z_mus_in: {m_Z_mus_in}")
print(f"n_Z_mud_in: {n_Z_mud_in}")
print(f"n_Z_mus_in: {n_Z_mus_in}")
print(f"n_X_mud_in: {n_X_mud_in}")
print(f"n_X_mus_in: {n_X_mus_in}")
print(f"total_symbols: {total_symbols}")

# ------------- ende TXT --------------


# m_X_mud_in = data["global_len_wrong_x_dec"]
# m_X_mus_in = data["global_len_wrong_x_non_dec"]
# m_Z_mud_in = data["global_len_wrong_z_dec"]
# m_Z_mus_in = data["global_len_wrong_z_non_dec"]
# n_Z_mud_in = data["global_len_Z_checked_dec"]
# n_Z_mus_in = data["global_len_Z_checked_non_dec"]
# n_X_mud_in = data["global_X_P_calc_dec"]
# n_X_mus_in = data["global_X_P_calc_non_dec"]
# total_symbols = data["total_symbols"]
# print(f"m_X_mud_in: {m_X_mud_in}")
# print(f"m_X_mus_in: {m_X_mus_in}")
# print(f"m_Z_mud_in: {m_Z_mud_in}")
# print(f"m_Z_mus_in: {m_Z_mus_in}")
# print(f"n_Z_mud_in: {n_Z_mud_in}")
# print(f"n_Z_mus_in: {n_Z_mus_in}")
# print(f"n_X_mud_in: {n_X_mud_in}")
# print(f"n_X_mus_in: {n_X_mus_in}")
# print(f"total_symbols: {total_symbols}")


config = SimulationConfig(None)
data_processor = DataProcessor(config)


# Load the JSON file
with open(json_filepath, 'r') as file:
    config_loaded = json.load(file)

config.mean_photon_nr = config_loaded["mean_photon_nr"]
config.mean_photon_decoy = config_loaded["mean_photon_decoy"]
config.p_decoy = config_loaded["p_decoy"]
config.p_z_alice = config_loaded["p_z_alice"]
print(f"mean_photon_nr: {config.mean_photon_nr}")
print(f"mean_photon_decoy: {config.mean_photon_decoy}")

desired_p_decoy_arr =  np.arange(0.02, 1, 0.02)
desired_p_z_alice_arr = np.arange(0.02, 1, 0.02)
# factor_x_mud_arr = np.arange(0, 100, 1)
factor_x_mud_arr = np.array([1])
length_multiply_arr = np.array([1e7, 1e8, 1e9])
scale_factor_symbol_arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 50, 100, 1000]) #np.arange(1, 1000, 1)

for scale_factor_symbol in scale_factor_symbol_arr:
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

                    total_symbols = scale_factor_symbol * total_symbols
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
                    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H")
                    csv_filename = f"4_att_SKR_results_{timestamp}.csv"
                    # Check if the file already exists to write the header only once
                    file_exists = os.path.isfile(csv_filename)
                    # Open the CSV file in append mode
                    with open(csv_filename, "a", newline="") as csvfile:
                        csv_writer = csv.writer(csvfile)

                        # Write the header if the file is being created for the first time
                        if not file_exists:
                            csv_writer.writerow(["length_multiply", "mpn_s", "mpn_d", "desired_p_decoy", "desired_p_z_alice", "factor_x_mud","scale_factor_symbol", "SKR"])

                        # Write the data row
                        if not math.isnan(skr) and skr > 0:
                            csv_writer.writerow([
                                length_multiply,
                                config.mean_photon_nr,
                                config.mean_photon_decoy,
                                desired_p_decoy,
                                desired_p_z_alice,
                                factor_x_mud_value,
                                scale_factor_symbol,
                                skr
                            ])
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

with open(f"4_att_SKR_results_{timestamp}.txt", "a") as f:
    f.write(f"n_Z_mus_in: {n_Z_mus_in}, n_Z_mud_in: {n_Z_mud_in}, n_X_mus_in: {n_X_mus_in}, n_X_mud_in: {n_X_mud_in}\n")
    f.write(f"m_Z_mus_in: {m_Z_mus_in}, m_Z_mud_in: {m_Z_mud_in}, m_X_mus_in: {m_X_mus_in}, m_X_mud_in: {m_X_mud_in}\n")
    f.write(f"QBER_signal: {QBER_signal}, QBER_decoy: {QBER_decoy}, Pherr_signal: {Pherr_signal}, Pherr_decoy: {Pherr_decoy}\n")
    f.write(f"total symbols: {total_symbols}\n")