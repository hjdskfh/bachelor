#blub
import time
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
from joblib import Parallel, delayed
import sys

sys.stdout.flush()

job_id = os.getenv("SLURM_JOB_ID")
print(f"Running SLURM Job ID: {job_id}")


os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

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


# Memory considerations:
# 20,000 symbols ~ 6 GB per simulation.
# To be safe, use up to ~75% of 256 GB → ~192 GB usable.
# Maximum concurrent simulations ≈ 192 / 6 ≈ 32.
# Here we choose a conservative maximum number of parallel tasks.
max_concurrent_tasks = 16

# How many simulations per batch (each batch runs sequentially inside one task)
simulations_in_batch = 2  # adjust this to increase per-task workload

# Total number of batches to run (total simulations = simulations_in_batch * total_batches)
total_batches = 100  # e.g., total simulations = 2 * 50 = 100

length_of_chain = 6*6 +1
n_rep = 500
bins_per_symbol_hist = 30
amount_bins_hist = bins_per_symbol_hist * length_of_chain

# Define file name
style_file = "Presentation_style_1_adjusted_no_grid.mplstyle"

base_path = os.path.dirname(os.path.abspath(__file__))


best_batchsize = Saver.find_best_batchsize(length_of_chain, n_rep)

config = SimulationConfig(database, n_samples=int(length_of_chain*n_rep), batchsize=best_batchsize, 
                 jitter=jitter, detector_jitter=detector_jitter,
                mlp=os.path.join(base_path, style_file), script_name=os.path.basename(__file__), job_id=job_id
                )
simulation = SimulationManager(config)

# Convert the config object to a dictionary
config_params = config.to_dict()
Saver.save_to_json(config_params)

# Read in time
end_time_read = time.time()  # Record end time
execution_time_read = end_time_read - start_time  # Calculate execution time for reading
print(f"Execution time for reading: {execution_time_read:.9f} seconds for {config.n_samples} samples")

def run_simulation_and_update_hist(i, length_of_chain, n_rep, base_path, style_file, database, jitter,
                                   detector_jitter, best_batchsize, bins_per_symbol, amount_bins):
    # Create the simulation config locally
    config = SimulationConfig(database, n_samples=int(length_of_chain*n_rep), batchsize=best_batchsize, 
                 jitter=jitter, detector_jitter=detector_jitter,
                mlp=os.path.join(base_path, style_file), script_name=os.path.basename(__file__), job_id=job_id
                )
    simulation = SimulationManager(config)

    # Run one simulation
    time_photons_det_x, time_photons_det_z, index_where_photons_det_x, index_where_photons_det_z, \
        time_one_symbol, lookup_arr = simulation.run_simulation_hist_final()

    # Compute local histograms
    local_hist_x, local_hist_z = DataProcessor.update_histogram_batches(length_of_chain,
        time_photons_det_x, time_photons_det_z, time_one_symbol, int(length_of_chain * n_rep),
        index_where_photons_det_x, index_where_photons_det_z, amount_bins_hist, bins_per_symbol=bins_per_symbol)

    return local_hist_x, local_hist_z, time_one_symbol, lookup_arr

def run_simulation_batch(batch_id, simulations_in_batch, length_of_chain, n_rep, base_path, style_file,
                         database, jitter, detector_jitter, best_batchsize, bins_per_symbol, amount_bins):
    """
    Run a batch of simulations sequentially and aggregate the local histograms.
    """
    # Initialize local histograms for the batch
    local_hist_total_x = np.zeros(amount_bins, dtype=int)
    local_hist_total_z = np.zeros(amount_bins, dtype=int)
    time_one_symbol_final = None
    lookup_arr_final = None

    for j in range(simulations_in_batch):
        # We pass a unique identifier if needed (here simply j)
        local_hist_x, local_hist_z, time_one_symbol, lookup_arr = run_simulation_and_update_hist(
            j, length_of_chain, n_rep, base_path, style_file, database, jitter,
            detector_jitter, best_batchsize, bins_per_symbol, amount_bins
        )
        local_hist_total_x += local_hist_x
        local_hist_total_z += local_hist_z
        time_one_symbol_final = time_one_symbol  # assume it's the same for each simulation in the batch
        lookup_arr_final = lookup_arr

    return local_hist_total_x, local_hist_total_z, time_one_symbol_final, lookup_arr_final

# --- Run Batches in Parallel ---

results = Parallel(n_jobs=max_concurrent_tasks)(
    delayed(run_simulation_batch)(
         batch_id,
         simulations_in_batch,
         length_of_chain,
         n_rep,
         base_path,
         style_file,
         database,
         jitter,
         detector_jitter,
         best_batchsize,
         bins_per_symbol_hist,
         amount_bins_hist
    )
    for batch_id in range(total_batches)
)

# --- Aggregate Global Histograms ---
global_histogram_counts_x = np.zeros(amount_bins_hist, dtype=int)
global_histogram_counts_z = np.zeros(amount_bins_hist, dtype=int)
final_time_one_symbol = None
final_lookup_arr = None

for local_hist_x, local_hist_z, time_one_symbol, lookup_arr in results:
    global_histogram_counts_x += local_hist_x
    global_histogram_counts_z += local_hist_z
    if final_time_one_symbol is None:
        final_time_one_symbol = time_one_symbol  # Set the time_one_symbol from the first result
    if final_lookup_arr is None:
        final_lookup_arr = lookup_arr
    
# --- Plot and Save Results ---
total_symbols = int(length_of_chain*n_rep) * simulations_in_batch * total_batches

DataProcessor.plot_histogram_batch(bins_per_symbol_hist, final_time_one_symbol,
                           global_histogram_counts_x, global_histogram_counts_z,
                           final_lookup_arr, total_symbols, start_symbol=3, end_symbol=10, name="fixed")

Saver.save_array_as_npz_data("histograms_fixed",
                        bins_per_symbol_hist=bins_per_symbol_hist,
                        final_time_one_symbol=final_time_one_symbol,
                        global_histogram_counts_x=global_histogram_counts_x,
                        global_histogram_counts_z=global_histogram_counts_z,
                        final_lookup_array=final_lookup_arr,
                        total_symbols=total_symbols,
                        )

# --- Timing ---
end_time_simulation = time.time()
execution_time_simulation = end_time_simulation - start_time
print(f"Execution time for simulation: {execution_time_simulation:.9f} seconds")