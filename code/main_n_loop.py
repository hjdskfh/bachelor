import cProfile
import pstats
import time
from datamanager import DataManager
from config import SimulationConfig
from simulationmanager import SimulationManager
from saver import Saver
import numpy as np
import matplotlib.pyplot as plt


Saver.memory_usage("Before everything")



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

#seed = 45

#n_samples = np.arange(22000, 27000, 2000, dtype=int)
times_per_n = 1
#seed_arr = np.arange(1, times_per_n + 1, 1)
#for n in n_samples:
arr_current = np.arange(0.08205, 0.08215, 0.00001)
peak_wavelength = np.empty(len(arr_current) * times_per_n)
amount_detection_x_late_bin = np.empty(len(arr_current) * times_per_n)

for idx, var_current in enumerate(arr_current):
    for i in range(times_per_n):
        #measure execution time
        start_time = time.time()  # Record start time

        #create simulation
        config = SimulationConfig(database, seed=624537, n_samples=20000, n_pulses=4, batchsize=1000, mean_voltage=1.0, mean_current=var_current, current_amplitude=0.02,
                        p_z_alice=0.5, p_decoy=0.1, p_z_bob=0.15, sampling_rate_FPGA=6.5e9, bandwidth=4e9, jitter=jitter, 
                        non_signal_voltage=-1.2, voltage_decoy=-0.2, voltage=-0.2, voltage_decoy_sup=-0.2, voltage_sup=-0.2,
                        mean_photon_nr=0.7, mean_photon_decoy=0.1, 
                        fiber_attenuation=-3, insertion_loss_dli=-1, n_eff_in_fiber=1.558, detector_efficiency=0.3, dark_count_frequency=10, detection_time=1e-10, detector_jitter=detector_jitter,
                        mlp='C:/Users/leavi/OneDrive/Dokumente/Uni/Semester 7/NeuMoQP/Programm/code/Presentation_style_1_adjusted_no_grid.mplstyle'
                        )
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
        peak_wavelength[idx * times_per_n + i], amount_detection_x_late_bin[idx * times_per_n + i] = simulation.run_simulation_classificator()


        end_time = time.time()  # Record end time  
        execution_time = end_time - start_time  # Calculate execution time
        print(f"Execution time: {execution_time:.9f} seconds for {config.n_samples} samples")

plt.bar(peak_wavelength, amount_detection_x_late_bin, width=0.8)  # adjust width for nicer display if needed
plt.xlabel('Peak Wavelength')
plt.ylabel('Number of Detections (Late Bin, X-basis)')
plt.title('Number of Detections per Peak Wavelength')
plt.show()






