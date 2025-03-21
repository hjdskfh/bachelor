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

        output_dir = r"C:\Users\leavi\OneDrive\Dokumente\Uni\Semester 7\NeuMoQP\Programm\results"

        # Ensure the output directory exists
        os.makedirs(output_dir, exist_ok=True)

        # Define the file path
        filepath = os.path.join(output_dir, f"output_{timestamp}_n_samples_{n_samples}.txt")

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

    def prepare_data_for_histogram(time_photons_det_x, time_photons_det_z, bins_per_symbol, histogram_matrix_bins_x, histogram_matrix_bins_z):
        """Prepare the data for the histogram by binning the photon arrival times."""
        # Example parameters
        num_symbols = length_of_chain
        bins_per_symbol = 30
        symbol_window = 100  # Example time window for one symbol (adjust as needed)

        # Create bins
        bins = np.linspace(0, symbol_window, bins_per_symbol + 1)  # 30 bins

        # Loop over symbols
        for i in range(min(num_symbols, time_photons_det_x.shape[0])):
            # Extract non-NaN times for this symbol
            times_x = time_photons_det_x[i, ~np.isnan(time_photons_det_x[i])]
            times_z = time_photons_det_z[i, ~np.isnan(time_photons_det_z[i])]

            # Histogram for X basis
            counts_x, _ = np.histogram(times_x, bins=bins)
            histogram_matrix_bins_x[i, :] = counts_x
            
            # Histogram for Z basis
            counts_z, _ = np.histogram(times_z, bins=bins)
            histogram_matrix_bins_z[i, :] = counts_z

        return histogram_matrix_bins_x, histogram_matrix_bins_z
    
    def prepare_data_for_histogram(
        time_photons_det_x,
        time_photons_det_z,
        bins_per_symbol,
        histogram_matrix_bins_x,
        histogram_matrix_bins_z,
        symbol_window=100  # Time window per symbol in ns or ps
    ):
        """
        Bins photon arrival times into a fixed number of bins per symbol and accumulates counts.

        Parameters:
        - time_photons_det_x (np.ndarray): 2D array of photon times for X basis, shape (symbols, detections)
        - time_photons_det_z (np.ndarray): 2D array of photon times for Z basis, shape (symbols, detections)
        - bins_per_symbol (int): number of bins per symbol
        - histogram_matrix_bins_x (np.ndarray): running bin count matrix for X, shape (symbols, bins)
        - histogram_matrix_bins_z (np.ndarray): running bin count matrix for Z, shape (symbols, bins)
        - symbol_window (float): max time window per symbol (e.g., 100 ns)

        Returns:
        - histogram_matrix_bins_x, histogram_matrix_bins_z: updated bin matrices
        """

        # Define time bin edges
        bins = np.linspace(0, symbol_window, bins_per_symbol + 1)

        # Ensure we don't go out of bounds
        num_symbols = min(time_photons_det_x.shape[0], histogram_matrix_bins_x.shape[0])

        for i in range(num_symbols):
            # Clean and bin X basis times
            valid_times_x = time_photons_det_x[i][~np.isnan(time_photons_det_x[i])]
            counts_x, _ = np.histogram(valid_times_x, bins=bins)
            histogram_matrix_bins_x[i] += counts_x

            # Clean and bin Z basis times
            valid_times_z = time_photons_det_z[i][~np.isnan(time_photons_det_z[i])]
            counts_z, _ = np.histogram(valid_times_z, bins=bins)
            histogram_matrix_bins_z[i] += counts_z

        return histogram_matrix_bins_x, histogram_matrix_bins_z

    def update_histogram_batches(
        length_of_chain,
        time_photons_det_x, 
        time_photons_det_z,
        histogram_batches_x, histogram_batches_z,
        batch_size = 10, bins_per_symbol = 30, symbol_window = 100    
    ):
        """
        Update cumulative histograms for batches of in-cycle symbols.
        
        Parameters:
        - time_photons_det_x, time_photons_det_z: 2D arrays of shape (length_of_chain*n_rep, detections_per_symbol).
        - batch_size: number of in-cycle symbols to group in one batch (e.g., 10).
        - bins_per_symbol: number of histogram bins per symbol.
        - symbol_window: the time interval allocated for each symbol.
        - histogram_batches_x, histogram_batches_z: cumulative histogram arrays to update.
        
        Returns:
        Updated histogram_batches_x and histogram_batches_z.
        """
        total_symbols = time_photons_det_x.shape[0]  # Should equal length_of_chain * n_rep
        n_rep = total_symbols // length_of_chain
        
        num_batches = math.ceil(length_of_chain / batch_size)
        
        # Process each batch
        for batch in range(num_batches):
            # Determine the in-cycle symbol indices for this batch.
            in_cycle_start = batch * batch_size
            in_cycle_end = min(length_of_chain, in_cycle_start + batch_size)
            num_symbols_in_batch = in_cycle_end - in_cycle_start
            # Total bins for this batch
            batch_bins = bins_per_symbol * num_symbols_in_batch
            print(f"batchbins: {batch_bins} in batch {batch}")
            # Define bins spanning the time interval for this batch.
            bins = np.linspace(0, symbol_window, bins_per_symbol + 1)    
            print(f"Bins range: {bins.min()} to {bins.max()}")
            print(f"First 10 X photon times: {time_photons_det_x[:10]}")
            print(f"First 10 Z photon times: {time_photons_det_z[:10]}")
            print(f"Histogram bins: {bins}")
            # Temporary count arrays for this batch.
            counts_x_batch = np.zeros(batch_bins, dtype=int)
            counts_z_batch = np.zeros(batch_bins, dtype=int)
            
            # Loop over each cycle (repetition)
            for rep in range(n_rep):
                # For each symbol in the in-cycle indices of this batch:
                for s in range(in_cycle_start, in_cycle_end):
                    row_idx = rep * length_of_chain + s
                    # Get valid times (remove NaN)
                    valid_x = time_photons_det_x[row_idx][~np.isnan(time_photons_det_x[row_idx])]
                    valid_z = time_photons_det_z[row_idx][~np.isnan(time_photons_det_z[row_idx])]
                    
                    # Compute histogram for this symbol
                    counts_x, _ = np.histogram(valid_x, bins=bins)
                    counts_z, _ = np.histogram(valid_z, bins=bins)
                    
                     # Store histogram in the correct batch index range
                    start_idx = (s - in_cycle_start) * bins_per_symbol
                    end_idx = start_idx + bins_per_symbol
                    counts_x_batch[start_idx:end_idx] += counts_x
                    counts_z_batch[start_idx:end_idx] += counts_z

                    print(f"counts_x_batch after symbol {s}: {counts_x_batch}")
            
            # Update the corresponding batch in the cumulative histogram arrays.
            # For a full batch, the number of bins is bins_per_symbol * batch_size,
            # but if it's the last batch, we only update the used portion.
            histogram_batches_x[batch, :batch_bins] += counts_x_batch
            histogram_batches_z[batch, :batch_bins] += counts_z_batch
        
        return histogram_batches_x, histogram_batches_z

    def plot_histogram_batch(length_of_chain, batch_index, batch_size, bins_per_symbol, symbol_window, hist_counts_x, hist_counts_z):
        """
        Plot the cumulative histogram for a given batch.
        
        Parameters:
        - batch_index: which batch to plot (0-indexed).
        - batch_size: number of in-cycle symbols per batch (except possibly the last one).
        - bins_per_symbol: number of histogram bins per symbol.
        - symbol_window: time interval allocated per symbol.
        - hist_counts_x, hist_counts_z: cumulative histogram arrays for the batch.
        """
        # Determine the number of symbols in this batch.
        # For the last batch, it might be less than batch_size.
        # We assume length_of_chain total symbols; hence:
        if batch_index == (math.ceil(length_of_chain / batch_size) - 1):
            # Last batch: in-cycle indices from batch_index*batch_size to length_of_chain
            num_symbols_in_batch = length_of_chain - batch_index * batch_size
        else:
            num_symbols_in_batch = batch_size
        
        batch_bins = bins_per_symbol * num_symbols_in_batch
        bins = np.linspace(0, num_symbols_in_batch * symbol_window, batch_bins + 1)    
        print(f"bins: {bins}")

        plt.figure(figsize=(10, 6))
        # Plot as bar chart; you can also use plt.hist with precomputed counts.
        width = (bins[1] - bins[0])
        plt.bar(bins[:-1], hist_counts_x[:batch_bins], width=width, alpha=0.6, label='X basis', color='blue')
        plt.bar(bins[:-1], hist_counts_z[:batch_bins], width=width, alpha=0.6, label='Z basis', color='red')
        plt.xlabel("Time (offset for each symbol in batch)")
        plt.ylabel("Cumulative Counts")
        plt.title(f"Cumulative Histogram for Batch {batch_index+1} (In-cycle symbols {batch_index*batch_size} to {batch_index*batch_size + num_symbols_in_batch - 1})")
        plt.legend()
        plt.tight_layout()
        plt.show()

            
