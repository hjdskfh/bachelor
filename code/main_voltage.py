import cProfile
import pstats
import time
from datamanager import DataManager
from config import SimulationConfig
from simulationmanager import SimulationManager
from saver import Saver
import numpy as np
import matplotlib.pyplot as plt
import os 


Saver.memory_usage("Before everything")

#database
database = DataManager()

#readin
database.add_data('data/current_power_data.csv', 'Current (mA)', 'Optical Power (mW)', 9, 'current_power') 
database.add_data('data/voltage_shift_data.csv', 'Voltage (V)', 'Wavelength Shift (nm)', 20, 'voltage_shift')
# database.add_data('data/eam_transmission_data.csv', 'Voltage (V)', 'Transmission', 11, 'eam_transmission') #modified,VZ geflippt von Spannungswerten
database.add_data('data/wavelength_neff.csv', 'Wavelength (nm)', 'neff', 20, 'wavelength_neff')
database.add_data('data/eam_static_results_renormalized.csv', 'Voltage (V)', 'Transmission', 16, 'eam_transmission')


detector_jitter = 5e-12
database.add_jitter(detector_jitter, 'detector')

#seed = 45

#n_samples = np.arange(22000, 27000, 2000, dtype=int)
times_per_n = 1
#seed_arr = np.arange(1, times_per_n + 1, 1)
arr_voltage = np.arange(0.975, 1.015, 0.005)
peak_wavelength = np.empty(len(arr_voltage) * times_per_n)
amount_detection_x_late_bin = np.empty(len(arr_voltage) * times_per_n)
round_counter = 0

# Define file name
style_file = "Presentation_style_1_adjusted_no_grid.mplstyle"

base_path = os.path.dirname(os.path.abspath(__file__))

#for n in n_samples:
for idx, var_voltage in enumerate(arr_voltage):
    p_indep_x_states_dec_var = None
    p_indep_x_states_non_dec_var = None

    for i in range(times_per_n):
        #measure execution time
        start_time = time.time()  # Record start time

        round_counter += 1

        print(f"VARVOLTAGE: {var_voltage}")

        #create simulation
        config = SimulationConfig(database, round=round_counter, mean_voltage=var_voltage, 
                                   detector_jitter=detector_jitter,
                                  mlp=os.path.join(base_path, style_file), script_name=os.path.basename(__file__))
        simulation = SimulationManager(config)

        # Convert the config object to a dictionary
        config_params = config.to_dict()

        # Save the config parameters to a JSON file
        Saver.save_to_json(config_params)

        #readin time
        end_time_read = time.time()  # Record end time  
        execution_time = end_time_read - start_time  # Calculate execution time
        print(f"Execution time for readin: {execution_time:.9f} seconds for {config.n_samples} samples")

        #plot results
        # simulation.run_simulation_classificator()
        # peak_wavelength[idx * times_per_n + i], amount_detection_x_late_bin[idx * times_per_n + i] = simulation.run_simulation_det_peak_wave()
        simulation.run_simulation_till_DLI()


        end_time = time.time()  # Record end time  
        execution_time = end_time - start_time  # Calculate execution time
        print(f"Execution time: {execution_time:.9f} seconds for {config.n_samples} samples")

'''plt.bar(peak_wavelength, amount_detection_x_late_bin, width=0.8)  # adjust width for nicer display if needed
plt.xlabel('Peak Wavelength')
plt.ylabel('Number of Detections (Late Bin, X-basis)')
plt.title('Number of Detections per Peak Wavelength')
Saver.save_plot('nr_det_late_x_over_peak_wavelength')'''
