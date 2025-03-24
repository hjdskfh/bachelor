
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

Saver.memory_usage("START of Simulation: Before everything")

#measure execution time
start_time = time.time()  # Record start time

#database
database = DataManager()

#readin
database.add_data('data/current_power_data.csv', 'Current (mA)', 'Optical Power (mW)', 9, 'current_power') 
database.add_data('data/voltage_shift_data.csv', 'Voltage (V)', 'Wavelength Shift (nm)', 20, 'voltage_shift')
database.add_data('data/eam_transmission_data.csv', 'Voltage (V)', 'Transmission', 11, 'eam_transmission') 

jitter = 1e-11
database.add_jitter(jitter, 'laser')
detector_jitter = 100e-12
database.add_jitter(detector_jitter, 'detector')

times_per_n = 1
length_of_chain = 8*8 +1
n_rep = 50
round_counter = 0
bins_per_symbol_hist = 30
amount_bins = bins_per_symbol_hist * length_of_chain
histogram_counts_x = np.zeros(amount_bins, dtype=int)
histogram_counts_z = np.zeros(amount_bins, dtype=int)

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

for i in range(times_per_n):#create simulation mean current 0.08, int(length_of_chain*n_rep)
    # Run the simulation
    time_photons_det_x, time_photons_det_z, index_where_photons_det_x, index_where_photons_det_z, time_one_symbol, lookup_arr = simulation.run_simulation_hist_final()
    print(f"time_photons_det_x[:10]: {time_photons_det_x[:10]}")
    print(f"time_photons_det_z[:10]: {time_photons_det_z[:10]}")
    print(f"time_one_symbol: {time_one_symbol}")

    Saver.update_histogram_batches(length_of_chain, time_photons_det_x, time_photons_det_z, time_one_symbol, int(length_of_chain * n_rep),
                                   index_where_photons_det_x, index_where_photons_det_z,
                                   histogram_counts_x, histogram_counts_z,
                                   bins_per_symbol = 30)

Saver.save_array_as_npz("histograms", histogram_counts_x=histogram_counts_x, histogram_counts_z=histogram_counts_z)
np.set_printoptions(threshold=2000)
print(f"hist z: {histogram_counts_z}")
print(f"hist x: {histogram_counts_x}")
print(f"histz.shape:{histogram_counts_z.shape}")
print(f"histx.shape:{histogram_counts_x.shape}")

Saver.plot_histogram_batch(length_of_chain, bins_per_symbol_hist, time_one_symbol, histogram_counts_x, histogram_counts_z, lookup_arr, start_symbol=3, end_symbol=10)



'''pd.DataFrame(histogram_matrix_bins_x).to_csv("hist_x.csv", index=False, header=False)
pd.DataFrame(histogram_matrix_bins_z).to_csv("hist_z.csv", index=False, header=False)'''

'''plt.hist(histogram_matrix_bins_z.flatten(), bins=10)
plt.xlabel('Histogram Bins Z')
plt.ylabel('Frequency')
plt.title('Histogram of Histogram Bins Z')
Saver.save_plot(f"hist_bins_z")

plt.hist(histogram_matrix_bins_x.flatten(), bins=10)
plt.xlabel('Histogram Bins X')
plt.ylabel('Frequency')
plt.title('Histogram of Histogram Bins X')
Saver.save_plot(f"hist_bins_x")'''

end_time_simulation = time.time()  # Record end time for simulation
execution_time_simulation = end_time_simulation - end_time_read  # Calculate execution time for simulation
print(f"Execution time for simulation: {execution_time_simulation:.9f} seconds for {config.n_samples} samples")