import matplotlib.pyplot as plt
from pathlib import Path
import datetime
import json
import psutil
import os
import time
import threading
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd
import inspect
import math


class Saver:
    def __init__(self, config):
        self.config = config

    @staticmethod
    def save_plot(filename, dpi=600):
        """Saves the current Matplotlib plot to a file in a folder next to 'code'."""
        
        # Get the script's parent directory (the directory where the script is located)
        script_dir = Path(__file__).parent
        
        # Navigate to the parent folder (next to 'code') and then to the 'data' folder
        target_dir = script_dir.parent / 'images'
        
        # Create the directory if it doesn't exist
        target_dir.mkdir(exist_ok=True)
        
        # Generate a timestamp (e.g., '20231211_153012' for 11th December 2023 at 15:30:12)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Append the timestamp to the filename
        filename_with_timestamp = f"{timestamp}_{filename}"
        
        # Define the file path
        filepath = target_dir / filename_with_timestamp
        
        # Save the plot
        plt.savefig(filepath, dpi=dpi)
        
        # Close the plot to free up memory
        plt.close()

    @staticmethod
    def save_array_as_npz(filename, **kwargs):
        # Get the script's parent directory (the directory where the script is located)
        script_dir = Path(__file__).parent

        # Navigate to the parent folder (next to 'code') and then to the 'images' folder
        target_dir = script_dir.parent / 'results'

        # Create the directory if it doesn't exist
        target_dir.mkdir(exist_ok=True)

        # Generate a timestamp (e.g., '20231211_153012' for 11th December 2023 at 15:30:12)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        # Append the timestamp to the filename
        filename_with_timestamp = f"{timestamp}_{filename}.npz"

        # Define the file path
        filepath = target_dir / filename_with_timestamp

        # Save all provided arrays into the file
        np.savez(filepath, **kwargs)

    @staticmethod
    def save_array_as_npz_data(filename, **kwargs):
        # Get the script's parent directory (the directory where the script is located)
        script_dir = Path(__file__).parent

        # Navigate to the parent folder (next to 'code') and then to the 'images' folder
        target_dir = script_dir.parent / 'results_data'

        # Create the directory if it doesn't exist
        target_dir.mkdir(exist_ok=True)

        # Generate a timestamp (e.g., '20231211_153012' for 11th December 2023 at 15:30:12)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        # Append the timestamp to the filename
        filename_with_timestamp = f"{timestamp}_{filename}.npz"

        # Define the file path
        filepath = target_dir / filename_with_timestamp

        # Save all provided arrays into the file
        np.savez(filepath, **kwargs)


    @staticmethod
    def save_to_json(config_object):
        """Save data to a JSON file with timestamp in the 'logs' folder next to the code."""
        
        # Get the directory of the current script (code folder)
        script_dir = Path(__file__).parent  # The folder where the script is located
        
        # Navigate to the parent directory (next to the code) and then to the 'logs' folder
        logs_dir = script_dir.parent / 'logs'  # Go one level up and look for 'logs'
        
        # Create the 'logs' directory if it doesn't exist
        logs_dir.mkdir(exist_ok=True)

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        # Construct the filename with the timestamp
        filename_with_timestamp = f"simulation_config_{timestamp}.json"

        try:                   
            
            # If 'rng' is present in the config, remove it and store only the seed
            if 'rng' in config_object:
                del config_object['rng']  # Remove the rng object from the dictionary


            # Define the full file path where the JSON file will be saved
            file_path = logs_dir / filename_with_timestamp  # Combine 'logs' folder and the filename
            
            # Write the dictionary to a JSON file
            with open(file_path, 'w') as f:
                json.dump(config_object, f, indent=4)
            
            print(f"Configuration saved to {file_path}")
        
        except Exception as e:
            print(f"Error saving to JSON: {e}")
    
    @staticmethod
    def save_results_to_txt(function_used = None, n_samples = None, seed = None, non_signal_voltage=None, voltage_decoy=None, voltage=None, voltage_decoy_sup=None, voltage_sup=None, p_indep_x_states_non_dec=None, p_indep_x_states_dec=None, **kwargs):
        """
        Saves key-value pairs to a timestamped text file.

        Parameters:
        - output_dir (str): Directory where the output file should be saved.
        - kwargs: Any number of key-value pairs to be written to the file.

        Returns:
        - filepath (str): Path of the saved file.
        """

        # Generate timestamp for filename
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        script_dir = Path(__file__).parent  # The folder where the script is located
        
        # Navigate to the parent directory (next to the code) and then to the 'results' folder
        logs_dir = script_dir.parent / 'results'  # Go one level up and look for 'results'
        
        # Create the 'logs' directory if it doesn't exist
        logs_dir.mkdir(exist_ok=True)

        # Define the file path
        filepath = os.path.join(logs_dir, f"output_{timestamp}_n_samples_{n_samples}.txt")

        # Write the key-value pairs to the file
        with open(filepath, "w") as f:
            f.write(f"function_used: {function_used}\n")
            f.write(f"n_samples: {n_samples}\n")
            f.write(f"seed: {seed}\n")
            f.write(f"non_signal_voltage: {non_signal_voltage}\n")
            f.write(f"voltage_decoy: {voltage_decoy}\n")
            f.write(f"voltage: {voltage}\n")
            f.write(f"voltage_decoy_sup: {voltage_decoy_sup}\n")
            f.write(f"voltage_sup: {voltage_sup}\n")
            f.write(f"p_indep_x_states_non_dec: {p_indep_x_states_non_dec}\n")
            f.write(f"p_indep_x_states_dec: {p_indep_x_states_dec}\n")
        
            for key, value in kwargs.items():
                f.write(f"{key}: {value}\n")

    # ========== Memory Helper ==========

    @staticmethod
    def memory_usage(description):
        process = psutil.Process(os.getpid())  # Get the current process
        memory_info = process.memory_info()
        memory_used = memory_info.rss  # Memory used by the process in bytes
        memory_used_mb = memory_used / (1024 ** 2)  # Convert to MB
        print(f"[{description}] Memory used: {memory_used_mb:.2f} MB")
        

    # monitor memory usage and terminate if it exceeds the limit
    @staticmethod
    def monitor_memory():
        MEMORY_LIMIT_MB = 6000
        """Check memory usage and terminate if it exceeds the limit."""
        process = psutil.Process(os.getpid())
        while True:
            memory_usage = process.memory_info().rss / (1024 * 1024)  # Convert to MB
            if memory_usage > MEMORY_LIMIT_MB:
                print(f"⚠️ Memory limit exceeded! ({memory_usage:.2f} MB). Terminating process.")
                os._exit(1)  # Forcefully kill the program
            time.sleep(1)     # Check every second   

    @staticmethod
    def save_arrays_to_csv(filename, **arrays):
        """
        Save multiple arrays to a CSV file with column names based on variable names.
        Handles cases where single numbers (int, float) are passed instead of arrays.

        Args:
            filename (str): Name of the CSV file (e.g., "output.csv").
            **arrays: Any number of named arrays to save.
        """

        # Generate timestamp for filename
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        output_dir = r"C:\Users\leavi\OneDrive\Dokumente\Uni\Semester 7\NeuMoQP\Programm\results"

        # Ensure the output directory exists
        os.makedirs(output_dir, exist_ok=True)

        # Define the file path
        filepath = os.path.join(output_dir, f"output_{timestamp}_{filename}.csv")

        # ✅ Convert all values to NumPy arrays safely
        data_dict = {name: np.atleast_1d(values) for name, values in arrays.items()}  # Ensures all values are at least 1D

        # ✅ Convert integer arrays to float before padding
        for key in data_dict:
            if data_dict[key].dtype == np.int64:  # If it's an integer array
                data_dict[key] = data_dict[key].astype(float)  # Convert to float

        # ✅ Ensure all arrays have the same length
        max_length = max(len(arr) for arr in data_dict.values())

        # ✅ Pad arrays so they all have the same length
        for key in data_dict:
            if len(data_dict[key]) < max_length:
                data_dict[key] = np.pad(data_dict[key], (0, max_length - len(data_dict[key])), constant_values=np.nan)

        # ✅ Convert to DataFrame and save
        df = pd.DataFrame(data_dict)
        df.to_csv(filepath, index=False)

        print(f"✅ Saved to {filename} with columns: {list(data_dict.keys())}")

    # ========== Batchsize Calculator Helper for Histogramm ==========

    def find_best_batchsize(length_of_chain, n_rep, target=1000):
        n_samples = length_of_chain * n_rep  # Compute total number of samples

        # Find divisors of n_samples close to the target batchsize
        best_batchsize = min(
            (b for b in range(1, n_samples + 1) if n_samples % b == 0), 
            key=lambda x: abs(x - target)
        )
        
        return best_batchsize

    # ========== Prepare Data for Histogram ==========
    def update_histogram_batches(
        length_of_chain,
        time_photons_det_x, 
        time_photons_det_z,
        time_one_symbol,
        total_symbols,
        index_where_photons_det_x, index_where_photons_det_z,
        amount_bins_hist,
        bins_per_symbol = 30 
    ):
        
        print(f"total_symbols: {total_symbols}")
        n_rep = total_symbols // length_of_chain
        print(f"n_rep: {n_rep}")
        print(f"lengthofchain:{length_of_chain}")
        print(f"first blub")

        local_histogram_counts_x = np.zeros(amount_bins_hist, dtype=int)
        local_histogram_counts_z = np.zeros(amount_bins_hist, dtype=int)
        # Define bins spanning the time interval for this batch.
        bins_arr_per_symbol = np.linspace(0, time_one_symbol, bins_per_symbol + 1)        
        # Loop over each cycle (repetition)
        for rep in range(n_rep):
            for s in range(length_of_chain):  # which symbol
                row_idx = rep * length_of_chain + s
                if np.isin(row_idx, index_where_photons_det_x):
                    ind_short = np.where(index_where_photons_det_x == row_idx)[0]
                    valid_x = time_photons_det_x[ind_short][~np.isnan(time_photons_det_x[ind_short])]
                    bin_index = np.digitize(valid_x, bins_arr_per_symbol) - 1
                    # insert into histogram_counts_z with 30*symbol + bin_index 
                    print(f"bins_per_symbol:{bins_per_symbol}, s:{s}, bin_index:{bin_index}")
                    local_histogram_counts_x[bins_per_symbol * s + bin_index] += 1
                if np.isin(row_idx, index_where_photons_det_z):
                    ind_short = np.where(index_where_photons_det_z == row_idx)[0]
                    valid_z = time_photons_det_z[ind_short][~np.isnan(time_photons_det_z[ind_short])]
                    bin_index = np.digitize(valid_z, bins_arr_per_symbol) - 1
                    # insert into histogram_counts_z with 30*symbol + bin_index 
                    print(f"bins_per_symbol:{bins_per_symbol}, s:{s}, bin_index:{bin_index}")
                    local_histogram_counts_z[bins_per_symbol * s + bin_index] += 1
        return local_histogram_counts_x, local_histogram_counts_z

    def plot_histogram_batch(length_of_chain, bins_per_symbol, time_one_symbol, histogram_counts_x, histogram_counts_z, lookup_arr, total_symbols, start_symbol=3, end_symbol=10):
        assert 0 <= start_symbol <= end_symbol <= 64
        amount_of_symbols_incl_start_and_end = end_symbol - start_symbol + 1
        bins = np.linspace(0, amount_of_symbols_incl_start_and_end * time_one_symbol, bins_per_symbol * amount_of_symbols_incl_start_and_end + 1)    

        plt.figure(figsize=(10, 6))
        # Plot as bar chart; you can also use plt.hist with precomputed counts.
        width = (bins[1] - bins[0])
        plt.bar(bins[:-1], histogram_counts_x[start_symbol * bins_per_symbol :(end_symbol + 1) * bins_per_symbol], width=width, alpha=0.6, label='X basis', color='blue')
        plt.bar(bins[:-1], histogram_counts_z[start_symbol * bins_per_symbol :(end_symbol + 1) * bins_per_symbol], width=width, alpha=0.6, label='Z basis', color='red')
        for i in range(amount_of_symbols_incl_start_and_end):
            plt.axvline(x=i * time_one_symbol, color='grey', linestyle='--', linewidth=1)

            # Place the symbol halfway between this line and the next
            if i < amount_of_symbols_incl_start_and_end - 1:
                x_mid = i * time_one_symbol + time_one_symbol / 2
                symbol = lookup_arr[start_symbol + i]
                y_max = max(max(histogram_counts_x), max(histogram_counts_z))
                x_mid = i * time_one_symbol + time_one_symbol / 2
                basis = symbol[0]  # assuming symbol is like 'X0' or 'Z1'
                color = 'green' if basis == 'X' else 'purple'

                plt.text(x_mid, y_max * 0.9, symbol, ha='center', va='bottom', fontsize=14, color=color, fontweight='bold')

        plt.xlabel("Time ")
        plt.ylabel("Cumulative Counts")
        plt.title(f"Cumulative Histogram for symbols {lookup_arr[start_symbol:end_symbol + 1]} with {total_symbols} symbols sent")
        plt.legend()
        plt.tight_layout()
        Saver.save_plot(f"hist_symbols_{start_symbol}_to_{end_symbol}")

   
    def get_all_pair_indices(lookup_arr):
        """
        Given a 1D array (or list) of symbol identifiers (for one chain),
        return a dictionary mapping each adjacent pair (as a tuple)
        to a numpy array of indices where that pair occurs.
        
        The returned index i indicates that lookup_arr[i] and lookup_arr[i+1] form that pair.
        """
        lookup_arr = np.array(lookup_arr)
        # Create an array of shape (N-1, 2) with each row as a pair (lookup_arr[i], lookup_arr[i+1])
        pairs = np.column_stack((lookup_arr[:-1], lookup_arr[1:]))
        # Get the unique pairs (each row is a unique pair)
        unique_pairs = np.unique(pairs, axis=0)
        
        pair_indices_dict = {}
        for pair in unique_pairs:
            pair_tuple = tuple(pair)
            # Find indices where this exact pair occurs (vectorized)
            indices = np.nonzero((pairs[:, 0] == pair_tuple[0]) & (pairs[:, 1] == pair_tuple[1]))[0]
            pair_indices_dict[pair_tuple] = indices
        return pair_indices_dict


    def update_histogram_batches_all_pairs(length_of_chain, time_one_symbol, time_photons_det_z, time_photons_det_x,
                                        index_where_photons_det_z, index_where_photons_det_x, amount_bins_hist,
                                        bins_per_symbol, lookup_arr, basis, value, decoy):
        """
        Update the histogram counts for all adjacent pairs in the given sequence.
        
        The simulation data (time_photons_det and index_where_photons_det) is assumed to be stored
        in an NPZ file, and total_symbols is the total number of symbols over all chains.
        
        Parameters:
        length_of_chain: int
            Number of symbols in one chain (e.g. 65).
        time_one_symbol: float
            Duration (time window) for one symbol.
        time_photons_det: 1D numpy array
            The arrival times (relative to each symbol start) of detected photons.
        index_where_photons_det: numpy array
            Global symbol indices (across the batch) where detections occurred.
        amount_bins_hist: int
            Total number of histogram bins (typically bins_per_symbol * length_of_chain).
        bins_per_symbol: int
            Number of bins per symbol.
        lookup_arr: list of str
            The lookup array for one chain, e.g.:
            ['Z0', 'Z0', 'Z1', 'Z0', 'X0', 'Z0', 'X1', ...]
            (Here, strings are not normalized, so "Z0" and "Z0*" are distinct.)
        
        Returns:
        local_histogram_counts: 1D numpy array of length amount_bins_hist.
        """
        raw_symbol_lookup = {
            (1, 1, 0): "Z0",
            (1, 0, 0): "Z1",
            (0, -1, 0): "X+",
            (1, 1, 1): "Z0*",  # or "Z0_decoy" if you prefer
            (1, 0, 1): "Z1*",  # or "Z1_decoy"
            (0, -1, 1): "X+*",  # or "X+_decoy"
        }

        local_histogram_counts_x = np.zeros(amount_bins_hist, dtype=int)
        local_histogram_counts_z = np.zeros(amount_bins_hist, dtype=int)
        
        # Define bin edges for one symbol's time window.
        bins_arr = np.linspace(0, time_one_symbol, bins_per_symbol + 1)
        
        # Get dictionary mapping each adjacent pair (for one chain) to the positions where they occur.
        pair_indices_dict = Saver.get_all_pair_indices(lookup_arr)
        
        def process(idx_left, idx_right, index_where_photons_det, time_photons_det, local_histogram_counts):
            if idx_left > 0 and idx_right < length_of_chain - 1:
                # looked at index is second symbol:
                pair_key = (raw_symbol_lookup[(basis[idx_left], value[idx_left], decoy[idx_left])], 
                            raw_symbol_lookup[(basis[idx_right], value[idx_right], decoy[idx_right])])
                print(f"pair_key:{pair_key}")
                print(f"pair_indices_dict:{pair_indices_dict}")
                #if pair_key in pair_indices_dict:  
                position_brujin_left = pair_indices_dict[pair_key].item()
                
                # Process the first symbol of the pair:
                if idx_left in index_where_photons_det:
                    inds_first = np.where(index_where_photons_det == idx_left)[0]
                    valid_times_first = time_photons_det[inds_first]
                    valid_times_first = valid_times_first[~np.isnan(valid_times_first)]
                    bin_indices_first = np.digitize(valid_times_first, bins_arr) - 1
                    # Update histogram: position = pos (for first symbol)
                    for b in bin_indices_first:
                        if 0 <= b < bins_per_symbol:
                            overall_bin = position_brujin_left * bins_per_symbol + b
                            local_histogram_counts[overall_bin] += 1

                # Process the second symbol of the pair:
                if idx_right in index_where_photons_det:
                    inds_first = np.where(index_where_photons_det == idx_right)[0]
                    valid_times_first = time_photons_det[inds_first]
                    valid_times_first = valid_times_first[~np.isnan(valid_times_first)]
                    bin_indices_first = np.digitize(valid_times_first, bins_arr) - 1
                    # Update histogram: position = pos (for first symbol)
                    for b in bin_indices_first:
                        if 0 <= b < bins_per_symbol:
                            overall_bin = (position_brujin_left + 1) * bins_per_symbol + b
                            local_histogram_counts[overall_bin] += 1

        # Process each chain (repetition)
        for idx_where_z in index_where_photons_det_z:
            idx_before_z = idx_where_z - 1
            idx_after_z = idx_where_z + 1
        
            # looked at photon is right part of symbol
            process(idx_before_z, idx_where_z, index_where_photons_det_z, time_photons_det_z, local_histogram_counts_z)
            # looked at photon is left part of symbol
            process(idx_where_z, idx_after_z, index_where_photons_det_z, time_photons_det_z, local_histogram_counts_z)

        for idx_where_x in index_where_photons_det_x:
            idx_before_x = idx_where_x - 1
            idx_after_x = idx_where_x + 1
        
            # looked at photon is right part of symbol
            process(idx_before_x, idx_where_x, index_where_photons_det_x, time_photons_det_x, local_histogram_counts_x)
            # looked at photon is left part of symbol
            process(idx_where_x, idx_after_x, index_where_photons_det_x, time_photons_det_x, local_histogram_counts_x)

        
        return local_histogram_counts_z, local_histogram_counts_x

