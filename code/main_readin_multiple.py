import os
import numpy as np
import json
import datetime
import csv
from operator import length_hint
from config import SimulationConfig
from saver import Saver
from dataprocessor import DataProcessor
import numpy as np
import os
import json
import datetime
import csv
import math

print("Starting to read in the data")
config = SimulationConfig(None)
data_processor = DataProcessor(config)


# Define the directory containing the input files
# input_dir = r"C:\Users\leavi\bachelor\stuff_from_cluster\2025_04_29\morning_files"
input_dir = r'C:\Users\leavi\bachelor\stuff_from_cluster\2025_04_29\abend_files'
# input_dir = r'C:\Users\leavi\bachelor\stuff_from_cluster\2025_04_29\10_abends_files'
# input_dir = r'C:\Users\leavi\bachelor\stuff_from_cluster\2025_04_30\nachtmessung'

# Get all files in the directory
all_files = os.listdir(input_dir)

# Group files by prefix (e.g., 1_, 2_, 3_)
prefixes = set(f.split('_')[0] for f in all_files if f.endswith('.npz') or f.endswith('.json'))
print(f"Prefixes found: {prefixes}")

# Process each prefix group
for prefix in prefixes:
    print(f"Processing files for prefix: {prefix}")
    # Get all .npz and .json files for the current prefix
    npz_files = [f for f in all_files if f.startswith(f"{prefix}_") and f.endswith('.npz')]
    json_files = [f for f in all_files if f.startswith(f"{prefix}_") and f.endswith('.json')]
    
    print(f"Prefix: {prefix}, NPZ files: {npz_files}, JSON files: {json_files}")

    # Match .npz files with corresponding .json files
    file_pairs = []
    for npz_file in npz_files:
        # Extract the prefix from the .npz file
        npz_prefix = npz_file.split('_')[0]
        # Find the corresponding .json file with the same prefix
        json_file = next((f for f in json_files if f.startswith(f"{npz_prefix}_")), None)
        if json_file:
            file_pairs.append((os.path.join(input_dir, npz_file), os.path.join(input_dir, json_file)))

    print(f"File pairs for prefix {prefix}: {file_pairs}")

    # Process each valid file pair
    for npz_path, json_path in file_pairs:
        print(f"Processing: {npz_path} with {json_path}")

        # Load the .npz file
        if os.path.exists(npz_path):
            data = np.load(npz_path, allow_pickle=True)
        else:
            print(f"File does not exist: {npz_path}")
            continue

        # Extract data from the .npz file
        m_X_mud_in = data["global_len_wrong_x_dec"]
        m_X_mus_in = data["global_len_wrong_x_non_dec"]
        m_Z_mud_in = data["global_len_wrong_z_dec"]
        m_Z_mus_in = data["global_len_wrong_z_non_dec"]
        n_Z_mud_in = data["global_len_Z_checked_dec"]
        n_Z_mus_in = data["global_len_Z_checked_non_dec"]
        n_X_mud_in = data["global_X_P_calc_dec"]
        n_X_mus_in = data["global_X_P_calc_non_dec"]
        total_symbols = data["total_symbols"]
        print(f"m_X_mud_in: {m_X_mud_in}, m_X_mus_in: {m_X_mus_in}")
        print(f"m_Z_mud_in: {m_Z_mud_in}, m_Z_mus_in: {m_Z_mus_in}")
        print(f"n_Z_mud_in: {n_Z_mud_in}, n_Z_mus_in: {n_Z_mus_in}")
        print(f"n_X_mud_in: {n_X_mud_in}, n_X_mus_in: {n_X_mus_in}")
        print(f"total_symbols: {total_symbols}")

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
        
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H")
        txt_filename = f"new_results_{prefix}_{timestamp}.txt"
        with open(txt_filename, "a") as f:
            f.write(f"n_Z_mus_in: {n_Z_mus_in}, n_Z_mud_in: {n_Z_mud_in}, n_X_mus_in: {n_X_mus_in}, n_X_mud_in: {n_X_mud_in}\n")
            f.write(f"m_Z_mus_in: {m_Z_mus_in}, m_Z_mud_in: {m_Z_mud_in}, m_X_mus_in: {m_X_mus_in}, m_X_mud_in: {m_X_mud_in}\n")
            f.write(f"QBER_signal: {QBER_signal}, QBER_decoy: {QBER_decoy}, Pherr_signal: {Pherr_signal}, Pherr_decoy: {Pherr_decoy}\n")
            f.write(f"total symbols: {total_symbols}\n")

        # Load the JSON file
        with open(json_path, 'r') as file:
            config_loaded = json.load(file)

        # Extract configuration parameters
        mean_photon_nr = config_loaded["mean_photon_nr"]
        mean_photon_decoy = config_loaded["mean_photon_decoy"]
        print(f"initial mean_photon_nr: {mean_photon_nr}, mean_photon_decoy: {mean_photon_decoy}")
        p_decoy = config_loaded["p_decoy"]
        p_z_alice = config_loaded["p_z_alice"]

        # Perform calculations (example)
        desired_p_decoy_arr = np.arange(0.02, 1, 0.02)
        desired_p_z_alice_arr = np.arange(0.02, 1, 0.02)
        factor_x_mud_arr = np.array([1])
        length_multiply_arr = np.array([1e7, 1e8, 1e9])
        factor_total_arr = np.array([1])

        for factor_total in factor_total_arr:
            for length_multiply in length_multiply_arr:
                for factor_x_mud_value in factor_x_mud_arr:
                    for desired_p_decoy in desired_p_decoy_arr:
                        for desired_p_z_alice in desired_p_z_alice_arr:
                            weight_Z = desired_p_z_alice
                            weight_X = 1 - desired_p_z_alice
                            weight_d = desired_p_decoy
                            weight_s = 1 - desired_p_decoy

                            # Apply weights
                            weighted_n_Z_mus = 4 * n_Z_mus_in * weight_Z * weight_s  # Z basis, signal
                            weighted_n_Z_mud = 4 * n_Z_mud_in * weight_Z * weight_d  # Z basis, decoy
                            weighted_n_X_mus = 4 * n_X_mus_in * weight_X * weight_s  # X basis, signal
                            weighted_n_X_mud = 4 * n_X_mud_in * weight_X * weight_d  * factor_x_mud_value # X basis, decoy
                            weighted_m_Z_mus = 4 * m_Z_mus_in * weight_Z * weight_s  # Z basis, signal
                            weighted_m_Z_mud = 4 * m_Z_mud_in * weight_Z * weight_d  # Z basis, decoy
                            weighted_m_X_mus = 4 * m_X_mus_in * weight_X * weight_s  # X basis, signal
                            weighted_m_X_mud = 4 * m_X_mud_in * weight_X * weight_d  # X basis, decoy
                            total_symbols = total_symbols * factor_total

                            # put desired p_decoy and p_z_alice in the config
                            config.p_decoy = desired_p_decoy
                            config.p_z_alice = desired_p_z_alice
                            # print(f"p_decoy: {config.p_decoy}, p_z_alice: {config.p_z_alice}")

                            # Calculate SKR (example)
                            if weighted_n_Z_mus != 0:
                                factor = length_multiply / weighted_n_Z_mus
                            else:
                                factor = 1

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

                            # Save results to a CSV file
                            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H")
                            csv_filename = f"new_results_{prefix}_{timestamp}.csv"
                            file_exists = os.path.isfile(csv_filename)
                            with open(csv_filename, "a", newline="") as csvfile:
                                csv_writer = csv.writer(csvfile)

                                # Write the header if the file is being created for the first time
                                if not file_exists:
                                    csv_writer.writerow(["length_multiply", "mpn_s", "mpn_d", "desired_p_decoy", "desired_p_z_alice", "factor_x_mud", "factor_total", "SKR"])

                                # Write the data row
                                if not math.isnan(skr) and skr > 0:
                                    csv_writer.writerow([
                                        length_multiply,
                                        mean_photon_nr,
                                        mean_photon_decoy,
                                        desired_p_decoy,
                                        desired_p_z_alice,
                                        factor_x_mud_value,
                                        factor_total,
                                        skr
                                    ])

