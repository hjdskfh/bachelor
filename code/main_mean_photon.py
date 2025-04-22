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
import functools

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
print = functools.partial(print, flush=True)

Saver.memory_usage("Before everything")

#measure execution time
start_time = time.time()  # Record start time

#database
database = DataManager()

#readin
database.add_data('data/current_power_data.csv', 'Current (mA)', 'Optical Power (mW)', 9, 'current_power') 
database.add_data('data/voltage_shift_data.csv', 'Voltage (V)', 'Wavelength Shift (nm)', 20, 'voltage_shift', do_inverse=True, parabola=True)
database.add_data('data/eam_transmission_data.csv', 'Voltage (V)', 'Transmission', 11, 'eam_transmission') 
database.add_data('data/wavelength_neff.csv', 'Wavelength (nm)', 'neff', 20, 'wavelength_neff')

detector_jitter = 5e-12
database.add_jitter(detector_jitter, 'detector')

# Define file name
style_file = "Presentation_style_1_adjusted_no_grid.mplstyle"

base_path = os.path.dirname(os.path.abspath(__file__))

#create simulation mean current 0.08 , mena_voltage = 0.98 weil aus voltage_sweep, 0.9835 # mean voltage mit skript 1.6094623981710416 rausbekommen
config = SimulationConfig(database, n_samples=630000, batchsize=1000, 
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

# Run the simulation
# lookup_results = simulation.lookup()
# print(lookup_results)
# len_wrong_x_dec, len_wrong_x_non_dec, len_wrong_z_dec, len_wrong_z_non_dec, \
# len_Z_checked_dec, len_Z_checked_non_dec, X_P_calc_non_dec, X_P_calc_dec = simulation.run_simulation_classificator()
# print(f"len_wrong_x_dec: {len_wrong_x_dec}, len_wrong_x_non_dec: {len_wrong_x_non_dec}, len_wrong_z_dec: {len_wrong_z_dec}, len_wrong_z_non_dec: {len_wrong_z_non_dec}")
# print(f"len_Z_checked_dec: {len_Z_checked_dec}, len_Z_checked_non_dec: {len_Z_checked_non_dec}")
# print(f"X_P_calc_non_dec: {X_P_calc_non_dec}, X_P_calc_dec: {X_P_calc_dec}")
# simulation.run_DLI()
# simulation.run_simulation_till_DLI()
len_wrong_x_dec, len_wrong_x_non_dec, len_wrong_z_dec, len_wrong_z_non_dec, len_Z_checked_dec, len_Z_checked_non_dec, X_P_calc_non_dec, X_P_calc_dec, time_photons_det_x, time_photons_det_z, time_one_symbol, index_where_photons_det_z, index_where_photons_det_x, \
                basis, value, decoy, lookup_array = simulation.run_simulation_detection_tester()
# simulation.run_simulation_states()
# time_photons_det_x, time_photons_det_z, index_where_photons_det_x, index_where_photons_det_z, time_one_symbol, lookup_array, basis, value, decoy = simulation.run_simulation_hist_final()
# time_one_symbol, time_photons_det_z, time_photons_det_x, index_where_photons_det_z, index_where_photons_det_x, lookup_array, basis, value, decoy = simulation.run_simulation_hist_pick_symbols()



end_time_simulation = time.time()  # Record end time for simulation
execution_time_simulation = end_time_simulation - end_time_read  # Calculate execution time for simulation
print(f"Execution time for simulation: {execution_time_simulation:.9f} seconds for {config.n_samples} samples")
