import cProfile
import pstats
import time

from datamanager import DataManager
from config import SimulationConfig
from simulationmanager import SimulationManager
from saver import Saver

#measure execution time
start_time = time.time()  # Record start time

# Enable profiler
profiler = cProfile.Profile()
profiler.enable()

#database
database = DataManager()

#readin
database.add_data('data/current_power_data.csv', 'Current (mA)', 'Optical Power (mW)', 9, 'current_power') 
database.add_data('data/voltage_shift_data.csv', 'Voltage (V)', 'Wavelength Shift (nm)', 20, 'voltage_shift')
database.add_data('data/current_wavelength_modified.csv', 'Current (mA)', 'Wavelength (nm)', 9, 'current_wavelength')#modified sodass mA Werte stimmen (/1000)
database.add_data('data/eam_transmission_data.csv', 'Voltage (V)', 'Transmission', 11, 'eam_transmission') #modified,VZ geflippt von Spannungswerten

jitter = 1e-11
database.add_jitter(jitter, 'laser')
detector_jitter = 100e-12
database.add_jitter(detector_jitter, 'detector')

#create simulation
config = SimulationConfig(database, seed = None, n_samples=50, n_pulses=4, mean_voltage=1.0, mean_current=0.08, current_amplitude=0.02,
                 p_z_alice=0.5, p_decoy=0.1, p_z_bob = 0.5, sampling_rate_FPGA=6.5e9, bandwidth = 4e9, jitter=jitter, 
                 voltage_decoy=0, voltage=0, voltage_decoy_sup=0, voltage_sup=0,
                 mean_photon_nr=0.7, mean_photon_decoy=0.1, 
                 fiber_attenuation=-3, detector_efficiency = 0.3, dark_count_frequency = 1000, detection_time = 5e-10, detector_jitter = detector_jitter,
                 mlp = 'C:/Users/leavi/OneDrive/Dokumente/Uni/Semester 7/NeuMoQP/Programm/code/Presentation_style_1_adjusted_no_grid.mplstyle'
                 )

# Extract the parameters as a dictionary
simulation = SimulationManager(config)

#plot results
simulation.run_simulation_states_old()
#simulation.run_simulation_histograms()

# Disable profiler
profiler.disable()
profiler.dump_stats('profile_output.prof')

# Print profiling results
with open('profile_output.txt', 'w') as f:
    stats = pstats.Stats('profile_output.prof', stream=f)
    stats.sort_stats('cumulative')
    stats.print_stats()

end_time = time.time()  # Record end time
execution_time = end_time - start_time  # Calculate execution time
print(f"Execution time: {execution_time:.9f} seconds for {config.n_samples} samples")

# Convert the config object to a dictionary
config_params = config.to_dict()    

# Save the config parameters to a JSON file
Saver.save_to_json(config_params)
