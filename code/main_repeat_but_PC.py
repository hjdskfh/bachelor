import cProfile
import pstats
import time
from datamanager import DataManager
from config import SimulationConfig
from simulationmanager import SimulationManager
from simulationhelper import SimulationHelper
from saver import Saver
from dataprocessor import DataProcessor
import numpy as np
import psutil
import threading
import os
import sys
import datetime

# Define the log file name with a timestamp
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = f"simulation_tracking_{timestamp}.log"

# Initialize cumulative totals
cumulative_totals = {
    "len_wrong_x_dec": 0,
    "len_wrong_x_non_dec": 0,
    "len_wrong_z_dec": 0,
    "len_wrong_z_non_dec": 0,
    "len_Z_checked_dec": 0,
    "len_Z_checked_non_dec": 0,
    "X_P_calc_non_dec": 0.0,
    "X_P_calc_dec": 0.0,
}

def log_simulation_results(results):
    """Log simulation results and update cumulative totals."""
    global cumulative_totals

    # Update cumulative totals
    cumulative_totals["len_wrong_x_dec"] += results["len_wrong_x_dec"]
    cumulative_totals["len_wrong_x_non_dec"] += results["len_wrong_x_non_dec"]
    cumulative_totals["len_wrong_z_dec"] += results["len_wrong_z_dec"]
    cumulative_totals["len_wrong_z_non_dec"] += results["len_wrong_z_non_dec"]
    cumulative_totals["len_Z_checked_dec"] += results["len_Z_checked_dec"]
    cumulative_totals["len_Z_checked_non_dec"] += results["len_Z_checked_non_dec"]
    cumulative_totals["X_P_calc_non_dec"] += results["X_P_calc_non_dec"]
    cumulative_totals["X_P_calc_dec"] += results["X_P_calc_dec"]

    # Write results to the log file
    with open(log_file, 'a') as f:
        f.write(f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - Simulation Results:\n")
        for key, value in results.items():
            f.write(f"  {key}: {value}\n")
        f.write("Cumulative Totals:\n")
        for key, value in cumulative_totals.items():
            f.write(f"  {key}: {value}\n")
        f.write("\n")

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


for simulation_run in range(100):  # Replace with your actual simulation loop

    Saver.memory_usage("Before everything")

    #measure execution time
    start_time = time.time()  # Record start time

    #database
    database = DataManager()

    #readin
    database.add_data('data/current_power_data.csv', 'Current (mA)', 'Optical Power (mW)', 9, 'current_power') 
    database.add_data('data/voltage_shift_data.csv', 'Voltage (V)', 'Wavelength Shift (nm)', 20, 'voltage_shift', do_inverse=True, parabola=True)
    # database.add_data('data/eam_transmission_data.csv', 'Voltage (V)', 'Transmission', 11, 'eam_transmission') 
    database.add_data('data/eam_static_results_renormalized.csv', 'Voltage (V)', 'Transmission', 16, 'eam_transmission')
    database.add_data('data/wavelength_neff.csv', 'Wavelength (nm)', 'neff', 20, 'wavelength_neff')

    detector_jitter = 1e-11
    database.add_jitter(detector_jitter, 'detector')

    # Define file name
    style_file = "Presentation_style_1_adjusted_no_grid.mplstyle"

    base_path = os.path.dirname(os.path.abspath(__file__))

    #create simulation mean current 0.08 , mena_voltage = 0.98 weil aus voltage_sweep, 0.9835 # mean voltage mit skript 1.6094623981710416 rausbekommen
    config = SimulationConfig(database, n_samples=20, batchsize=10, non_signal_voltage = -1.3, voltage_decoy=0.2,
                    voltage=0.2, voltage_decoy_sup=0.2, voltage_sup=0.2, 
                            detector_jitter=detector_jitter, mlp=os.path.join(base_path, style_file), script_name = os.path.basename(__file__))
    simulation = SimulationManager(config)

    # Convert the config object to a dictionary
    config_params = config.to_dict()

    # Save the config parameters to a JSON file
    Saver.save_to_json(config_params)

    # Read in time
    end_time_read = time.time()  # Record end time  
    execution_time_read = end_time_read - start_time  # Calculate execution time for reading
    print(f"Execution time for reading: {execution_time_read:.9f} seconds for {config.n_samples} samples")

    len_wrong_x_dec, len_wrong_x_non_dec, len_wrong_z_dec, len_wrong_z_non_dec, len_Z_checked_dec, len_Z_checked_non_dec, X_P_calc_non_dec, X_P_calc_dec = simulation.run_simulation_repeat(save_output = False)
    
    # Prepare results dictionary
    results = {
        "len_wrong_x_dec": len_wrong_x_dec,
        "len_wrong_x_non_dec": len_wrong_x_non_dec,
        "len_wrong_z_dec": len_wrong_z_dec,
        "len_wrong_z_non_dec": len_wrong_z_non_dec,
        "len_Z_checked_dec": len_Z_checked_dec,
        "len_Z_checked_non_dec": len_Z_checked_non_dec,
        "X_P_calc_non_dec": X_P_calc_non_dec,
        "X_P_calc_dec": X_P_calc_dec,
    }
    
    # Log the results
    log_simulation_results(results)

    end_time_simulation = time.time()  # Record end time for simulation
    execution_time_simulation = end_time_simulation - end_time_read  # Calculate execution time for simulation
    print(f"Execution time for simulation: {execution_time_simulation:.9f} seconds for {config.n_samples} samples")

