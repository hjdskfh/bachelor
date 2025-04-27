import matplotlib.pyplot as plt
from pathlib import Path
import datetime
import json
import psutil
import os
import time
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd


class Saver:
    def __init__(self, config):
        self.config = config

    @staticmethod
    def save_plot(filename, dpi=600, no_date=False):
        """Saves the current Matplotlib plot to a file in a folder next to 'code'."""
        
        # Get the script's parent directory (the directory where the script is located)
        script_dir = Path(__file__).parent
        
        # Navigate to the parent folder (next to 'code') and then to the 'data' folder
        target_dir = script_dir.parent / 'images'
        
        # Create the directory if it doesn't exist
        target_dir.mkdir(exist_ok=True)
        print(f"no_date: {no_date}")
        # Generate a timestamp (e.g., '20231211_153012' for 11th December 2023 at 15:30:12)
        if not no_date:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            # Append the timestamp to the filename
            filename_with_timestamp = f"{timestamp}_{filename}"
        else:
            filename_with_timestamp = filename
        
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
        filepath = os.path.join(logs_dir, f"output_{timestamp}_n_samples_{n_samples}_function_{function_used}.txt")

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
    @staticmethod
    def find_best_batchsize(length_of_chain, n_rep, target=1000):
        n_samples = length_of_chain * n_rep  # Compute total number of samples

        # Find divisors of n_samples close to the target batchsize
        best_batchsize = min(
            (b for b in range(1, n_samples + 1) if n_samples % b == 0), 
            key=lambda x: abs(x - target)
        )
        
        return best_batchsize
