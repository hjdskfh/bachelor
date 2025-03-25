
import time
from datamanager import DataManager
from config import SimulationConfig
from simulationmanager import SimulationManager
from saver import Saver
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import math
from joblib import Parallel, delayed

# savely 192 GB -> 640000 samples / 65 = 10000 max reps
Saver.memory_usage("START of Simulation: Before everything")
start_time = time.time()  # Record start time

#database
database = DataManager()
database.add_data('data/current_power_data.csv', 'Current (mA)', 'Optical Power (mW)', 9, 'current_power') 
database.add_data('data/voltage_shift_data.csv', 'Voltage (V)', 'Wavelength Shift (nm)', 20, 'voltage_shift')
database.add_data('data/eam_transmission_data.csv', 'Voltage (V)', 'Transmission', 11, 'eam_transmission') 

jitter = 1e-11
database.add_jitter(jitter, 'laser')
detector_jitter = 100e-12
database.add_jitter(detector_jitter, 'detector')

times_per_n = 100
length_of_chain = 8*8 +1
n_rep = 50
round_counter = 0
bins_per_symbol_hist = 30
amount_bins_hist = bins_per_symbol_hist * length_of_chain

# Define file name
style_file = "Presentation_style_1_adjusted_no_grid.mplstyle"

# Check if running on Windows or Linux (Cluster)
if os.name == "nt":  # Windows (Your PC)
    base_path = "C:/Users/leavi/OneDrive/Dokumente/Uni/Semester 7/NeuMoQP/Programm/code/"
else:  # Linux (Cluster)
    base_path = "/wang/users/leavic98/cluster_home/NeuMoQP/Programm/code/"

best_batchsize = Saver.find_best_batchsize(length_of_chain, n_rep)

config = SimulationConfig(database, seed=None, n_samples=int(length_of_chain*n_rep), n_pulses=4, batchsize=best_batchsize, mean_voltage=1.0, mean_current=0.080, voltage_amplitude=0.050, current_amplitude=0.0005,
                p_z_alice=0.5, p_decoy=0.1, p_z_bob=0.85, sampling_rate_FPGA=6.5e9, bandwidth=4e9, jitter=jitter, 
                non_signal_voltage=-1.1, voltage_decoy=-0.1, voltage=-0.1, voltage_decoy_sup=-0.1, voltage_sup=-0.1,
                mean_photon_nr=0.7, mean_photon_decoy=0.1, 
                fiber_attenuation=-3, insertion_loss_dli=-1, n_eff_in_fiber=1.558, detector_efficiency=0.3, dark_count_frequency=10, detection_time=1e-10, detector_jitter=detector_jitter,
                p_indep_x_states_non_dec=None, p_indep_x_states_dec=None,
                mlp=os.path.join(base_path, style_file), script_name = os.path.basename(__file__)
                )
simulation = SimulationManager(config)

# Convert the config object to a dictionary
config_params = config.to_dict()

# Save the config parameters to a JSON file
Saver.save_to_json(config_params)

# Read in time
end_time_read = time.time()  # Record end time
execution_time_read = end_time_read - start_time  # Calculate execution time for reading
print(f"Execution time for reading: {execution_time_read:.9f} seconds for {config.n_samples} samples")

def run_simulation_and_update_hist(i, length_of_chain, n_rep, base_path, style_file, database, jitter,
                                   detector_jitter, best_batchsize, bins_per_symbol):
    # Create the simulation config locally
    config = SimulationConfig(
        database, seed=None,
        n_samples=int(length_of_chain * n_rep),
        n_pulses=4,
        batchsize=best_batchsize,
        mean_voltage=1.0,
        mean_current=0.080,
        voltage_amplitude=0.050,
        current_amplitude=0.0005,
        p_z_alice=0.5,
        p_decoy=0.1,
        p_z_bob=0.85,
        sampling_rate_FPGA=6.5e9,
        bandwidth=4e9,
        jitter=jitter,
        non_signal_voltage=-1.1,
        voltage_decoy=-0.1,
        voltage=-0.1,
        voltage_decoy_sup=-0.1,
        voltage_sup=-0.1,
        mean_photon_nr=0.7,
        mean_photon_decoy=0.1,
        fiber_attenuation=-3,
        insertion_loss_dli=-1,
        n_eff_in_fiber=1.558,
        detector_efficiency=0.3,
        dark_count_frequency=10,
        detection_time=1e-10,
        detector_jitter=detector_jitter,
        p_indep_x_states_non_dec=None,
        p_indep_x_states_dec=None,
        mlp=os.path.join(base_path, style_file),
        script_name=os.path.basename(__file__)
    )
    simulation = SimulationManager(config)

    # Run one simulation
    time_photons_det_x, time_photons_det_z, index_where_photons_det_x, index_where_photons_det_z, \
        time_one_symbol, lookup_arr = simulation.run_simulation_hist_final()

    # Compute local histograms
    local_hist_x, local_hist_z = Saver.update_histogram_batches(
        length_of_chain,
        time_photons_det_x, 
        time_photons_det_z,
        time_one_symbol,
        int(length_of_chain * n_rep),
        index_where_photons_det_x,
        index_where_photons_det_z,
        amount_bins_hist,
        bins_per_symbol=bins_per_symbol
    )

    return local_hist_x, local_hist_z, time_one_symbol, lookup_arr

# Initialize global histograms to zero
global_histogram_counts_x = np.zeros(amount_bins_hist, dtype=int)
global_histogram_counts_z = np.zeros(amount_bins_hist, dtype=int)

# Optionally, process in batches if times_per_n is large
results = Parallel(n_jobs=64)(
    delayed(run_simulation_and_update_hist)(
        i, length_of_chain, n_rep, base_path, style_file, database, jitter,
        detector_jitter, best_batchsize, bins_per_symbol_hist
    )
    for i in range(times_per_n)
)

# Merge (sum) the local histograms into the global histogram arrays
for local_hist_x, local_hist_z, time_one_symbol, lookup_arr in results:
    global_histogram_counts_x += local_hist_x
    global_histogram_counts_z += local_hist_z

# Now you can plot using your existing plot_histogram_batch function.
Saver.plot_histogram_batch(length_of_chain, bins_per_symbol_hist, time_one_symbol,
                     global_histogram_counts_x, global_histogram_counts_z,
                     lookup_arr, start_symbol=3, end_symbol=10)

np.set_printoptions(threshold=2000)
Saver.save_array_as_npz("histograms", global_histogram_counts_x=global_histogram_counts_x, global_histogram_counts_z=global_histogram_counts_z)

end_time_simulation = time.time()  # Record end time for simulation
execution_time_simulation = end_time_simulation - end_time_read  # Calculate execution time for simulation
print(f"Execution time for simulation: {execution_time_simulation:.9f} seconds for {config.n_samples} samples")