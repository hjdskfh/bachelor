import matplotlib.pyplot as plt
from pathlib import Path
import datetime
import json


class Saver:
    def __init__(self):
        None

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
