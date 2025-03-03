import cProfile
import pstats
import time
from datamanager import DataManager
from config import SimulationConfig
from simulationmanager import SimulationManager
from simulationhelper import SimulationHelper
from saver import Saver
import numpy as np
import psutil
import threading
import os

threading.Thread(target=Saver.monitor_memory, daemon=True).start()

Saver.memory_usage("Before everything")

#measure execution time
start_time = time.time()  # Record start time

#database
database = DataManager()

#readin
database.add_data('data/current_power_data.csv', 'Current (mA)', 'Optical Power (mW)', 9, 'current_power') 
database.add_data('data/voltage_shift_data.csv', 'Voltage (V)', 'Wavelength Shift (nm)', 20, 'voltage_shift')
database.add_data('data/current_wavelength_modified.csv', 'Current (mA)', 'Wavelength (nm)', 9, 'current_wavelength')#modified sodass mA Werte stimmen (/1000)
database.add_data('data/eam_transmission_data.csv', 'Voltage (V)', 'Transmission', 11, 'eam_transmission') 

jitter = 1e-11
database.add_jitter(jitter, 'laser')
detector_jitter = 100e-12
database.add_jitter(detector_jitter, 'detector')

seed = 456725

#create simulation
config = SimulationConfig(database, seed=seed, n_samples=20000, n_pulses=4, batchsize=1000, mean_voltage=1.0, mean_current=0.08, current_amplitude=0.02,
                p_z_alice=0.5, p_decoy=0.1, p_z_bob=0.85, sampling_rate_FPGA=6.5e9, bandwidth=4e9, jitter=jitter, 
                non_signal_voltage=-1.2, voltage_decoy=-0.2, voltage=-0.2, voltage_decoy_sup=-0.2, voltage_sup=-0.2,
                mean_photon_nr=0.7, mean_photon_decoy=0.1, 
                fiber_attenuation=-3, insertion_loss_dli=1, n_eff_in_fiber=1.558, detector_efficiency=0.3, dark_count_frequency=10, detection_time=1e-10, detector_jitter=detector_jitter,
                mlp='C:/Users/leavi/OneDrive/Dokumente/Uni/Semester 7/NeuMoQP/Programm/code/Presentation_style_1_adjusted_no_grid.mplstyle'
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

# Plot results
simulation.run_simulation_classificator()

end_time_simulation = time.time()  # Record end time for simulation
execution_time_simulation = end_time_simulation - end_time_read  # Calculate execution time for simulation
print(f"Execution time for simulation: {execution_time_simulation:.9f} seconds for {config.n_samples} samples")