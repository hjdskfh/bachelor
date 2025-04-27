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

def extract_last_cumulative_totals(log_file_path):
    cumulative_totals = defaultdict(float)
    constant_fields = {"p_z_alice", "p_decoy", "p_z_bob", "mu_signal", "mu_decoy"}
    first_non_none_values = {}

    with open(log_file_path, 'r') as f:
        lines = f.readlines()

    in_cumulative = False
    current_cumulative_data = {}  # Ensure this is initialized outside of the loop
    last_cumulative_data = {}

    for line in lines:
        if "Cumulative Totals:" in line:
            in_cumulative = True
            current_cumulative_data = {}  # Reset current data at the start of a new section

        if in_cumulative:
            if not line.strip() or re.match(r"\d{4}-\d{2}-\d{2}", line):  # End of the current section
                if current_cumulative_data:
                    last_cumulative_data = current_cumulative_data  # Save the last occurrence
                in_cumulative = False

            match = re.match(r"\s*(\w+):\s+([-+]?[0-9]*\.?[0-9]+)", line)
            if match:
                key, val = match.groups()
                current_cumulative_data[key] = float(val)

        # For constant fields, only store the first non-None value
        for key in constant_fields:
            if key in current_cumulative_data and current_cumulative_data[key] is not None:
                if key not in first_non_none_values:
                    first_non_none_values[key] = current_cumulative_data[key]

    # Now, populate the cumulative_totals dictionary
    for key, value in last_cumulative_data.items():
        if key in constant_fields:
            # Use the first non-None value for constant fields
            if key in first_non_none_values:
                cumulative_totals[key] = first_non_none_values[key]
        else:
            cumulative_totals[key] += value  # Sum non-constant fields

    return dict(cumulative_totals)

def combine_cumulative_totals(log_files):
    additive_keys = [
        "amount_run",
        "len_wrong_x_dec", "len_wrong_x_non_dec",
        "len_wrong_z_dec", "len_wrong_z_non_dec",
        "len_Z_checked_dec", "len_Z_checked_non_dec",
        "X_P_calc_non_dec", "X_P_calc_dec",
        "total_symbols",
        "gain_Z_non_dec", "gain_Z_dec",
        "gain_X_non_dec", "gain_X_dec",
        "qber_z_dec", "qber_z_non_dec",
        "qber_x_dec", "qber_x_non_dec",
        "total_amount_detections"
    ]

    shared_keys = [
        "p_z_alice", "p_decoy", "p_z_bob",
        "mu_signal", "mu_decoy"
    ]

    combined = {key: 0 for key in additive_keys}
    shared_values_set = False

    for path in log_files:
        data = extract_last_cumulative_totals(path)

        for key in additive_keys:
            combined[key] += data.get(key, 0)

        if not shared_values_set:
            for key in shared_keys:
                combined[key] = data.get(key, None)
            shared_values_set = True

    return combined


config = SimulationConfig(None)
data_processor = DataProcessor(config)
# json_filepath = r'C:\Users\leavi\OneDrive\Dokumente\Uni\Semester 7\NeuMoQP\Programm\stuff_from_cluster\2025_04_26\simfor_200000_batch_sims_simulation_config_20250425_120954.json'
# json_filepath = r'C:\Users\leavi\bachelor\stuff_from_cluster\2025_04_27\repeat_uhr_10_15\simulation_config_20250426_215842.json'
# json_filepath = r'C:\Users\leavi\bachelor\stuff_from_cluster\2025_04_27\repeat_uhr_10_15\simulation_config_20250426_215909.json'
json_filepath = r'C:\Users\leavi\bachelor\stuff_from_cluster\2025_04_27\repeat_uhr_10_15\simulation_config_20250426_215945.json'
# Load the JSON file
with open(json_filepath, 'r') as file:
    config_loaded = json.load(file)

# file1 = r'C:\Users\leavi\OneDrive\Dokumente\Uni\Semester 7\NeuMoQP\Programm\stuff_from_cluster\2025_04_26\simulation_tracking_20250425_120949.log'
# file2 = r'C:\Users\leavi\OneDrive\Dokumente\Uni\Semester 7\NeuMoQP\Programm\stuff_from_cluster\2025_04_26\simulation_tracking_20250425_120954.log'
# file3 = r'C:\Users\leavi\OneDrive\Dokumente\Uni\Semester 7\NeuMoQP\Programm\stuff_from_cluster\2025_04_26\simulation_tracking_20250425_120957.log'
#alle nicht funktioniert
# file1 = r'C:\Users\leavi\bachelor\stuff_from_cluster\2025_04_27\repeat_uhr_10_15\simulation_tracking_mpn_0_7_mpn_d_0_1_20250426_215842.log'
# file1 = r'C:\Users\leavi\bachelor\stuff_from_cluster\2025_04_27\repeat_uhr_10_15\simulation_tracking_mpn_0_35_mpn_d_0_175_20250426_215909.log'
file1 = r'C:\Users\leavi\bachelor\stuff_from_cluster\2025_04_27\repeat_uhr_10_15\simulation_tracking_mpn_0_15_mpn_d_0_075_20250426_215944.log'

# log_files = [file1, file2, file3]
log_files = [file1]
cumulative_data = combine_cumulative_totals(log_files)
# cumulative_data = extract_last_cumulative_totals("simulation_tracking_20250425_120949.log")

n_Z_mus_in = cumulative_data["len_Z_checked_non_dec"]
n_Z_mud_in = cumulative_data["len_Z_checked_dec"]
n_X_mus_in = cumulative_data["X_P_calc_non_dec"]
n_X_mud_in = cumulative_data["X_P_calc_dec"]
m_Z_mus_in = cumulative_data["len_wrong_z_non_dec"]
m_Z_mud_in = cumulative_data["len_wrong_z_dec"]
m_X_mus_in = cumulative_data["len_wrong_x_non_dec"]
m_X_mud_in = cumulative_data["len_wrong_x_dec"]
total_symbols = cumulative_data["total_symbols"]

print(f"n_Z_mus_in: {n_Z_mus_in}, n_Z_mud_in: {n_Z_mud_in}, n_X_mus_in: {n_X_mus_in}, n_X_mud_in: {n_X_mud_in}")
print(f"m_Z_mus_in: {m_Z_mus_in}, m_Z_mud_in: {m_Z_mud_in}, m_X_mus_in: {m_X_mus_in}, m_X_mud_in: {m_X_mud_in}")
print(f"total_symbols: {total_symbols}")

# m_X_mud_in = 1           # len_wrong_x_dec
# m_X_mus_in = 4            # len_wrong_x_non_dec

# m_Z_mud_in = 135          # len_wrong_z_dec
# m_Z_mus_in = 141       # len_wrong_z_non_dec

# n_Z_mud_in = 2539         # len_Z_checked_dec
# n_Z_mus_in = 4920        # len_Z_checked_non_dec

# n_X_mus_in = 96           # X_P_calc_dec
# n_X_mud_in = 32           # X_P_calc_non_dec

# total_symbols = 1200000

manual_mpn_s = 0.15
manual_mpn_d = 0.075
config.mean_photon_nr = config_loaded.get("mean_photon_nr", manual_mpn_s )
config.mean_photon_decoy = config_loaded.get("mean_photon_decoy", manual_mpn_d)
print(f"config.mean_photon_nr: {config.mean_photon_nr}, config.mean_photon_decoy: {config.mean_photon_decoy}")


# Weighting factors
desired_p_decoy = 0.19
desired_p_z_alice = 0.8
weight_Z = desired_p_z_alice
weight_X = 1 - desired_p_z_alice
weight_d = desired_p_decoy
weight_s = 1 - desired_p_decoy

# Apply the weights accordingly
weighted_n_Z_mus = 4 * n_Z_mus_in * weight_Z * weight_s  # Z basis, signal
weighted_n_Z_mud = 4 * n_Z_mud_in * weight_Z * weight_d  # Z basis, decoy
weighted_n_X_mus = 4 * n_X_mus_in * weight_X * weight_s  # X basis, signal
weighted_n_X_mud = 4 * n_X_mud_in * weight_X * weight_d  # X basis, decoy
weighted_m_Z_mus = 4 * m_Z_mus_in * weight_Z * weight_s  # Z basis, signal
weighted_m_Z_mud = 4 * m_Z_mud_in * weight_Z * weight_d  # Z basis, decoy
weighted_m_X_mus = 4 * m_X_mus_in * weight_X * weight_s  # X basis, signal
weighted_m_X_mud = 4 * m_X_mud_in * weight_X * weight_d  # X basis, decoy

# store_n_X_mus = weighted_n_X_mus
# weighted_n_X_mus = weighted_n_X_mud
# weighted_n_X_mud = store_n_X_mus
# weighted_m_X_mus = 2.592            # weighted len_wrong_x_non_dec
# weighted_m_X_mud = 0.152            # weighted len_wrong_x_dec

# weighted_m_Z_mus = 365.472          # weighted len_wrong_z_non_dec
# weighted_m_Z_mud = 82.08            # weighted len_wrong_z_dec

# weighted_n_Z_mus = 12726.7          # weighted len_Z_checked_non_dec
# weighted_n_Z_mud = 1543.7           # weighted len_Z_checked_dec

# weighted_n_X_mus = 62.2             # weighted X_P_calc_non_dec
# weighted_n_X_mud = 4.864            # weighted X_P_calc_dec

# total_symbols = 1200000    # weighted total_symbols


print(f"weighted_n_Z_mus: {weighted_n_Z_mus}, weighted_n_Z_mud: {weighted_n_Z_mud}, weighted_n_X_mus: {weighted_n_X_mus}, weighted_n_X_mud: {weighted_n_X_mud}")
print(f"weighted_m_Z_mus: {weighted_m_Z_mus}, weighted_m_Z_mud: {weighted_m_Z_mud}, weighted_m_X_mus: {weighted_m_X_mus}, weighted_m_X_mud: {weighted_m_X_mud}")

# put desired p_decoy and p_z_alice in the config
config.p_decoy = desired_p_decoy
config.p_z_alice = desired_p_z_alice

# factor to get up to a billion symbols
if weighted_n_Z_mus != 0:
    factor = 1e9 / weighted_n_Z_mus
else:
    factor = 1
# factor = 1

print(f"factor: {factor}")
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

print(f"SKR: {skr} ")#for factor {factor} and ")#total factor {factor_total}

#  len_wrong_x_dec = 20 * factor
# len_wrong_x_non_dec = 212 * factor
# len_wrong_z_dec = 224 * factor
# len_wrong_z_non_dec = 1951 * factor
# len_Z_checked_dec = 1619 * factor
# len_Z_checked_non_dec = 95060 * factor
# X_P_calc_dec = 24 * factor
# X_P_calc_non_dec = 5232 * factor
# total_symbols = 4000000 * factor

# len_wrong_x_dec = 0.152 
# len_wrong_x_non_dec = 2.592 
# len_wrong_z_dec = 82.08
# len_wrong_z_non_dec = 365.472 
# len_Z_checked_dec = 1543.7
# len_Z_checked_non_dec = 12726.7 
# X_P_calc_dec = 62.2 
# X_P_calc_non_dec = 4.864 
# total_symbols = 1200000


# (venv) C:\Users\leavi\bachelor>python code/main_readin_data_hist.py
# len_wrong_x_dec: 0.152, len_wrong_x_non_dec: 2.592, len_wrong_z_dec: 82.08, len_wrong_z_non_dec: 365.472
# len_Z_checked_dec: 1543.7, len_Z_checked_non_dec: 12726.7
# X_P_calc_dec: 62.2, X_P_calc_non_dec: 4.864
# total_symbols: 1200000
# initial_params: [0.182, 0.1, 0.81, 0.8]
# factor: 100000
# skl: 39458439.52749339, secret_key_length: 182115874.74227718, total_bit_sequence_length: 120000000000
# SKR: 39458439.52749339
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
                # if not math.isnan(skr) and skr > 0:
                #     with open(f"OG_SKR_results_{timestamp}.txt", "a") as f:
                #         f.write(f"length_multiply: {length_multiply:.2e}, mpn_s: {config.mean_photon_nr:.2f}, mpn_d: {config.mean_photon_decoy:.2f}, desired_p_decoy: {desired_p_decoy:.2f}, desired_p_z_alice: {desired_p_z_alice:.2f}, factor_x_mud: {factor_x_mud_value}, SKR: {skr}\n")
                #     # raise ValueError(f"SKR: {skr} is not NaN and > 0 for p_decoy: {desired_p_decoy}, p_z_alice: {desired_p_z_alice}")
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
                # csv_filename = f"OG_SKR_results_{timestamp}_tot_sym_{total_symbols}.csv"
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
                #             factor_x_mud_value,
                #             skr
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

with open(f"OG_SKR_results_{timestamp}.txt", "a") as f:
    f.write(f"n_Z_mus_in: {n_Z_mus_in}, n_Z_mud_in: {n_Z_mud_in}, n_X_mus_in: {n_X_mus_in}, n_X_mud_in: {n_X_mud_in}\n")
    f.write(f"m_Z_mus_in: {m_Z_mus_in}, m_Z_mud_in: {m_Z_mud_in}, m_X_mus_in: {m_X_mus_in}, m_X_mud_in: {m_X_mud_in}\n")
    f.write(f"QBER_signal: {QBER_signal}, QBER_decoy: {QBER_decoy}, Pherr_signal: {Pherr_signal}, Pherr_decoy: {Pherr_decoy}\n")
    f.write(f"total symbols: {total_symbols}\n") 
    