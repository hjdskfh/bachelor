#blub
import time
from datamanager import DataManager
from config import SimulationConfig
from simulationmanager import SimulationManager
from saver import Saver
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import random
import math
from joblib import Parallel, delayed

Saver.memory_usage("START of Simulation: Before everything")
start_time = time.time()  # Record start time

#database
database = DataManager()
database.add_data('data/current_power_data.csv', 'Current (mA)', 'Optical Power (mW)', 9, 'current_power') 
database.add_data('data/voltage_shift_data.csv', 'Voltage (V)', 'Wavelength Shift (nm)', 20, 'voltage_shift')
database.add_data('data/eam_transmission_data.csv', 'Voltage (V)', 'Transmission', 11, 'eam_transmission') 
database.add_data('data/wavelength_neff.csv', 'Wavelength (nm)', 'neff', 20, 'wavelength_neff')


jitter = 1e-11
database.add_jitter(jitter, 'laser')
detector_jitter = 100e-12
database.add_jitter(detector_jitter, 'detector')
n_samples_set = 20000


# Memory considerations:
# 20,000 symbols ~ 6 GB per simulation.
# To be safe, use up to ~75% of 256 GB → ~192 GB usable.
# Maximum concurrent simulations ≈ 192 / 6 ≈ 32.
# Here we choose a conservative maximum number of parallel tasks.
max_concurrent_tasks = 16

# How many simulations per batch (each batch runs sequentially inside one task)
simulations_in_batch = 2  # adjust this to increase per-task workload

# Total number of batches to run (total simulations = simulations_in_batch * total_batches)
total_batches = 100  # e.g., total simulations = 2 * 50 = 100  # 340 circa 4,5 stunden mit 2 sim per batch

# Define file name
style_file = "Presentation_style_1_adjusted_no_grid.mplstyle"

base_path = os.path.dirname(os.path.abspath(__file__))

config = SimulationConfig(database, seed=None, n_samples=n_samples_set, n_pulses=4, batchsize=1000, mean_voltage=-1.708, mean_current=0.080, voltage_amplitude=0.02, current_amplitude=0.0005,
                p_z_alice=0.5, p_decoy=0.1, p_z_bob=0.85, sampling_rate_FPGA=6.5e9, bandwidth=4e9, jitter=jitter, 
                non_signal_voltage=-1.1, voltage_decoy=-0.1, voltage=-0.1, voltage_decoy_sup=-0.1, voltage_sup=-0.1,
                mean_photon_nr=0.7, mean_photon_decoy=0.1, 
                fiber_attenuation=-3, detector_efficiency=0.3, dark_count_frequency=10, detection_time=1e-10, detector_jitter=detector_jitter,
                mlp=os.path.join(base_path, style_file), script_name = os.path.basename(__file__)
                )
simulation = SimulationManager(config)

# Convert the config object to a dictionary
config_params = config.to_dict()
Saver.save_to_json(config_params)

# Read in time
end_time_read = time.time()  # Record end time
execution_time_read = end_time_read - start_time  # Calculate execution time for reading
print(f"Execution time for reading: {execution_time_read:.9f} seconds for {config.n_samples} samples")

def run_simulation_and_update_hist(i, base_path, style_file, database, jitter,
                                   detector_jitter, n_samples_set):
    # Create the simulation config locally
    config = SimulationConfig(database, seed=None, n_samples=n_samples_set, n_pulses=4, batchsize=1000, mean_voltage=-1.708, mean_current=0.080, voltage_amplitude=0.02, current_amplitude=0.0005,
                p_z_alice=0.5, p_decoy=0.1, p_z_bob=0.85, sampling_rate_FPGA=6.5e9, bandwidth=4e9, jitter=jitter, 
                non_signal_voltage=-1.1, voltage_decoy=-0.1, voltage=-0.1, voltage_decoy_sup=-0.1, voltage_sup=-0.1,
                mean_photon_nr=0.7, mean_photon_decoy=0.1, 
                fiber_attenuation=-3, detector_efficiency=0.3, dark_count_frequency=10, detection_time=1e-10, detector_jitter=detector_jitter,
                mlp=os.path.join(base_path, style_file), script_name = os.path.basename(__file__)
                )
    simulation = SimulationManager(config)

    if random.random() < 0.01:
        save_output_var = True
    else:
        save_output_var = False
    len_wrong_x_dec, len_wrong_x_non_dec, len_wrong_z_dec, len_wrong_z_non_dec, len_Z_checked_dec, len_Z_checked_non_dec, X_P_calc_non_dec, X_P_calc_dec = simulation.run_simulation_classificator(save_output = save_output_var)

    return len_wrong_x_dec, len_wrong_x_non_dec, len_wrong_z_dec, len_wrong_z_non_dec, len_Z_checked_dec, len_Z_checked_non_dec, X_P_calc_non_dec, X_P_calc_dec

def run_simulation_batch(batch_id, base_path, style_file,
                         database, jitter, detector_jitter, n_samples_set):
    """
    Run a batch of simulations sequentially and aggregate the local histograms.
    """
    # Initialize local histograms for the batch
    total_len_wrong_x_dec = np.zeros(1, dtype = int)
    total_len_wrong_x_non_dec = np.zeros(1, dtype = int)
    total_len_wrong_z_dec = np.zeros(1, dtype = int)
    total_len_wrong_z_non_dec = np.zeros(1, dtype = int)
    total_len_Z_checked_dec = np.zeros(1, dtype = int)
    total_len_Z_checked_non_dec = np.zeros(1, dtype = int)
    total_X_P_calc_non_dec = np.zeros(1, dtype = float)
    total_X_P_calc_dec = np.zeros(1, dtype = float)

    for j in range(simulations_in_batch):
        # We pass a unique identifier if needed (here simply j)
        len_wrong_x_dec, len_wrong_x_non_dec, len_wrong_z_dec, len_wrong_z_non_dec, len_Z_checked_dec, len_Z_checked_non_dec, X_P_calc_non_dec, X_P_calc_dec = run_simulation_and_update_hist(
            j, base_path, style_file, database, jitter, detector_jitter, n_samples_set)
        
        total_len_wrong_x_dec += len_wrong_x_dec
        total_len_wrong_x_non_dec += len_wrong_x_non_dec
        total_len_wrong_z_dec += len_wrong_z_dec
        total_len_wrong_z_non_dec += len_wrong_z_non_dec
        total_len_Z_checked_dec += len_Z_checked_dec
        total_len_Z_checked_non_dec += len_Z_checked_non_dec
        total_X_P_calc_non_dec += X_P_calc_non_dec
        total_X_P_calc_dec += X_P_calc_dec

    return total_len_wrong_x_dec, total_len_wrong_x_non_dec, total_len_wrong_z_dec, total_len_wrong_z_non_dec, total_len_Z_checked_dec, total_len_Z_checked_non_dec, total_X_P_calc_non_dec, total_X_P_calc_dec

# --- Run Batches in Parallel ---

results = Parallel(n_jobs=max_concurrent_tasks)(
    delayed(run_simulation_batch)(
         batch_id, base_path, style_file, database, jitter, detector_jitter, n_samples_set
    )
    for batch_id in range(total_batches)
)

# --- Aggregate Global Histograms ---
global_len_wrong_x_dec = 0
global_len_wrong_x_non_dec = 0
global_len_wrong_z_dec = 0
global_len_wrong_z_non_dec = 0
global_len_Z_checked_dec = 0
global_len_Z_checked_non_dec = 0
global_X_P_calc_non_dec = 0
global_X_P_calc_dec = 0

for len_wrong_x_dec, len_wrong_x_non_dec, len_wrong_z_dec, len_wrong_z_non_dec, len_Z_checked_dec, len_Z_checked_non_dec, X_P_calc_non_dec, X_P_calc_dec in results:
    global_len_wrong_x_dec += len_wrong_x_dec
    global_len_wrong_x_non_dec += len_wrong_x_non_dec
    global_len_wrong_z_dec += len_wrong_z_dec
    global_len_wrong_z_non_dec += len_wrong_z_non_dec
    global_len_Z_checked_dec += len_Z_checked_dec
    global_len_Z_checked_non_dec += len_Z_checked_non_dec
    global_X_P_calc_non_dec += X_P_calc_non_dec
    global_X_P_calc_dec += X_P_calc_dec

print(f"global_len_wrong_x_dec: {global_len_wrong_x_dec}")
print(f"global_len_wrong_x_non_dec: {global_len_wrong_x_non_dec}")
print(f"global_len_wrong_z_dec: {global_len_wrong_z_dec}")
print(f"global_len_wrong_z_non_dec: {global_len_wrong_z_non_dec}")
print(f"global_len_Z_checked_dec: {global_len_Z_checked_dec}")
print(f"global_len_Z_checked_non_dec: {global_len_Z_checked_non_dec}")
print(f"global_X_P_calc_non_dec: {global_X_P_calc_non_dec}")
print(f"global_X_P_calc_dec: {global_X_P_calc_dec}")

total_symbols = n_samples_set * simulations_in_batch * total_batches

Saver.save_results_to_txt(global_len_wrong_x_dec=global_len_wrong_x_dec, global_len_wrong_x_non_dec=global_len_wrong_x_non_dec, global_len_wrong_z_dec=global_len_wrong_z_dec,
                        global_len_wrong_z_non_dec=global_len_wrong_z_non_dec, global_len_Z_checked_dec=global_len_Z_checked_dec, global_len_Z_checked_non_dec=global_len_Z_checked_non_dec,
                        global_X_P_calc_dec=global_X_P_calc_dec, global_X_P_calc_non_dec=global_X_P_calc_non_dec, total_symbols=total_symbols)

# --- Timing ---
end_time_simulation = time.time()
execution_time_simulation = end_time_simulation - start_time
print(f"Execution time for simulation: {execution_time_simulation:.9f} seconds")