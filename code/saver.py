import matplotlib.pyplot as plt
from pathlib import Path
import datetime
import json
import psutil
import os
import time
import threading
from matplotlib.ticker import MaxNLocator


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
    def memory_usage(description):
        process = psutil.Process(os.getpid())  # Get the current process
        memory_info = process.memory_info()
        memory_used = memory_info.rss  # Memory used by the process in bytes
        memory_used_mb = memory_used / (1024 ** 2)  # Convert to MB
        print(f"[{description}] Memory used: {memory_used_mb:.2f} MB")
        
    # Function to track peak memory usage during the simulation
    @staticmethod
    def track_peak_memory_usage(check_interval=0.1):
        """
        Monitors and returns the peak memory usage of the current process.
        
        Args:
            check_interval (int): The interval (in seconds) at which to check memory usage.
            
        Returns:
            peak_memory (int): The peak memory usage in bytes.
        """
        # Get the current process
        process = psutil.Process(os.getpid())
        
        # Variable to store the peak memory usage
        peak_memory = 0
        
        # Function to monitor memory usage in a separate thread
        def monitor_memory():
            nonlocal peak_memory
            while True:
                current_memory = process.memory_info().rss  # in bytes
                peak_memory = max(peak_memory, current_memory)
                time.sleep(check_interval)  # Check memory every 'check_interval' seconds
        
        # Start memory monitoring in a separate daemon thread
        memory_thread = threading.Thread(target=monitor_memory, daemon=True)
        memory_thread.start()
        
        return peak_memory, memory_thread
    
    @staticmethod
    def save_results_to_txt(n_samples = None, seed = None, **kwargs):
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
            f.write(f"n_samples: {n_samples}\n")
            f.write(f"seed: {seed}\n")
            for key, value in kwargs.items():
                f.write(f"{key}: {value}\n")

    
                

    
