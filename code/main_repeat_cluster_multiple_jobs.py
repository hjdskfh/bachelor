import time
import os
import functools
import random
import numpy as np
from joblib import Parallel, delayed
from datamanager import DataManager
from config import SimulationConfig
from simulationmanager import SimulationManager
from saver import Saver

# Ensure print is always flushed
print = functools.partial(print, flush=True)

def start_simulation_batches(total_batches=300, simulations_in_batch=2, max_concurrent_tasks=3, mean_photon_nr=0.182, mean_photon_decoy=0.1, fiber_attenuation=-3, voltage_amplitude=0.0011, current_amplitude = 0.00041):
    print("Starting simulation batches...")

    Saver.memory_usage("START of Simulation: Before everything")
    start_time = time.time()

    job_id = os.getenv("SLURM_JOB_ID")
    print(f"Running SLURM Job ID: {job_id}")

    # Setup database
    database = DataManager()
    database.add_data('data/current_power_data.csv', 'Current (mA)', 'Optical Power (mW)', 9, 'current_power') 
    database.add_data('data/voltage_shift_data.csv', 'Voltage (V)', 'Wavelength Shift (nm)', 20, 'voltage_shift')
    database.add_data('data/wavelength_neff.csv', 'Wavelength (nm)', 'neff', 20, 'wavelength_neff')
    database.add_data('data/eam_static_results_renormalized.csv', 'Voltage (V)', 'Transmission', 16, 'eam_transmission')

    detector_jitter = 5e-12
    database.add_jitter(detector_jitter, 'detector')

    base_path = os.path.dirname(os.path.abspath(__file__))
    style_file = "Presentation_style_1_adjusted_no_grid.mplstyle"
    n_samples_set = 20000

    config = SimulationConfig(database, n_samples=n_samples_set, mean_photon_nr=mean_photon_nr, mean_photon_decoy=mean_photon_decoy, fiber_attenuation=fiber_attenuation,
                                voltage_amplitude=voltage_amplitude, current_amplitude=current_amplitude,
                               detector_jitter=detector_jitter,
                               mlp=os.path.join(base_path, style_file), 
                               script_name=os.path.basename(__file__), 
                               job_id=job_id)
    
    simulation = SimulationManager(config)
    config_params = config.to_dict()
    Saver.save_to_json(config_params)

    end_time_read = time.time()
    execution_time_read = end_time_read - start_time
    print(f"Execution time for reading: {execution_time_read:.9f} seconds for {config.n_samples} samples")

    def run_simulation_and_update_hist(i):
        config = SimulationConfig(database, n_samples=n_samples_set, mean_photon_nr=mean_photon_nr, mean_photon_decoy=mean_photon_decoy, fiber_attenuation=fiber_attenuation,
                                  voltage_amplitude=voltage_amplitude, current_amplitude=current_amplitude,
                                  detector_jitter=detector_jitter,
                                  mlp=os.path.join(base_path, style_file), 
                                  script_name=os.path.basename(__file__), 
                                  job_id=job_id)
        simulation = SimulationManager(config)

        save_output_var = random.random() < 0.01
        return simulation.run_simulation_repeat(save_output=save_output_var)

    def run_simulation_batch(batch_id):
        total_len_wrong_x_dec = np.zeros(1, dtype=int)
        total_len_wrong_x_non_dec = np.zeros(1, dtype=int)
        total_len_wrong_z_dec = np.zeros(1, dtype=int)
        total_len_wrong_z_non_dec = np.zeros(1, dtype=int)
        total_len_Z_checked_dec = np.zeros(1, dtype=int)
        total_len_Z_checked_non_dec = np.zeros(1, dtype=int)
        total_X_P_calc_non_dec = np.zeros(1, dtype=float)
        total_X_P_calc_dec = np.zeros(1, dtype=float)

        for j in range(simulations_in_batch):
            results = run_simulation_and_update_hist(j)
            (len_wrong_x_dec, len_wrong_x_non_dec, len_wrong_z_dec, 
             len_wrong_z_non_dec, len_Z_checked_dec, len_Z_checked_non_dec, 
             X_P_calc_non_dec, X_P_calc_dec) = results

            total_len_wrong_x_dec += len_wrong_x_dec
            total_len_wrong_x_non_dec += len_wrong_x_non_dec
            total_len_wrong_z_dec += len_wrong_z_dec
            total_len_wrong_z_non_dec += len_wrong_z_non_dec
            total_len_Z_checked_dec += len_Z_checked_dec
            total_len_Z_checked_non_dec += len_Z_checked_non_dec
            total_X_P_calc_non_dec += X_P_calc_non_dec
            total_X_P_calc_dec += X_P_calc_dec

        return (total_len_wrong_x_dec, total_len_wrong_x_non_dec, 
                total_len_wrong_z_dec, total_len_wrong_z_non_dec, 
                total_len_Z_checked_dec, total_len_Z_checked_non_dec, 
                total_X_P_calc_non_dec, total_X_P_calc_dec)

    # Run batches in parallel
    results = Parallel(n_jobs=max_concurrent_tasks)(
        delayed(run_simulation_batch)(batch_id) for batch_id in range(total_batches)
    )

    # Aggregate results
    global_len_wrong_x_dec = 0
    global_len_wrong_x_non_dec = 0
    global_len_wrong_z_dec = 0
    global_len_wrong_z_non_dec = 0
    global_len_Z_checked_dec = 0
    global_len_Z_checked_non_dec = 0
    global_X_P_calc_non_dec = 0
    global_X_P_calc_dec = 0

    for result in results:
        (len_wrong_x_dec, len_wrong_x_non_dec, len_wrong_z_dec, len_wrong_z_non_dec,
         len_Z_checked_dec, len_Z_checked_non_dec, 
         X_P_calc_non_dec, X_P_calc_dec) = result

        global_len_wrong_x_dec += len_wrong_x_dec
        global_len_wrong_x_non_dec += len_wrong_x_non_dec
        global_len_wrong_z_dec += len_wrong_z_dec
        global_len_wrong_z_non_dec += len_wrong_z_non_dec
        global_len_Z_checked_dec += len_Z_checked_dec
        global_len_Z_checked_non_dec += len_Z_checked_non_dec
        global_X_P_calc_non_dec += X_P_calc_non_dec
        global_X_P_calc_dec += X_P_calc_dec

    total_symbols = n_samples_set * simulations_in_batch * total_batches

    Saver.save_array_as_npz_data(f"counts_repeat_mpn_{mean_photon_nr}_decoy_{mean_photon_decoy}_att_{fiber_attenuation}_total_{total_symbols}_batch_{total_batches}_max_{max_concurrent_tasks}_volt_{voltage_amplitude}_current_{current_amplitude}",
                                 global_len_wrong_x_dec=global_len_wrong_x_dec,
                                 global_len_wrong_x_non_dec=global_len_wrong_x_non_dec,
                                 global_len_wrong_z_dec=global_len_wrong_z_dec,
                                 global_len_wrong_z_non_dec=global_len_wrong_z_non_dec,
                                 global_len_Z_checked_dec=global_len_Z_checked_dec,
                                 global_len_Z_checked_non_dec=global_len_Z_checked_non_dec,
                                 global_X_P_calc_dec=global_X_P_calc_dec,
                                 global_X_P_calc_non_dec=global_X_P_calc_non_dec,
                                 total_symbols=total_symbols)

    end_time_simulation = time.time()
    execution_time_simulation = end_time_simulation - start_time
    print(f"Execution time for simulation: {execution_time_simulation:.9f} seconds")

# eahc 1h44min hours for 50 batches of 2 simulations with 12 concurrent tasks
# batch 1
start_simulation_batches(total_batches=50, simulations_in_batch=2, max_concurrent_tasks=12, mean_photon_nr=0.15, mean_photon_decoy=0.075, fiber_attenuation=-3, voltage_amplitude=0.0011, current_amplitude = 0.00041)
start_simulation_batches(total_batches=50, simulations_in_batch=2, max_concurrent_tasks=12, mean_photon_nr=0.2, mean_photon_decoy=0.1, fiber_attenuation=-3, voltage_amplitude=0.0011, current_amplitude = 0.00041)
start_simulation_batches(total_batches=50, simulations_in_batch=2, max_concurrent_tasks=12, mean_photon_nr=0.25, mean_photon_decoy=0.175, fiber_attenuation=-6, voltage_amplitude=0.0011, current_amplitude = 0.00041)
start_simulation_batches(total_batches=50, simulations_in_batch=2, max_concurrent_tasks=12, mean_photon_nr=0.3, mean_photon_decoy=0.15, fiber_attenuation=-6, voltage_amplitude=0.0011, current_amplitude = 0.00041)
start_simulation_batches(total_batches=50, simulations_in_batch=2, max_concurrent_tasks=12, mean_photon_nr=0.5, mean_photon_decoy=0.25, fiber_attenuation=-9, voltage_amplitude=0.0011, current_amplitude = 0.00041)
#batch 2
# start_simulation_batches(total_batches=50, simulations_in_batch=2, max_concurrent_tasks=12, mean_photon_nr=0.6, mean_photon_decoy=0.3, fiber_attenuation=-9, voltage_amplitude=0.0011, current_amplitude = 0.00041)
# start_simulation_batches(total_batches=50, simulations_in_batch=2, max_concurrent_tasks=12, mean_photon_nr=0.15, mean_photon_decoy=0.075, fiber_attenuation=-3, voltage_amplitude=0.0055, current_amplitude = 0.00041)
# start_simulation_batches(total_batches=50, simulations_in_batch=2, max_concurrent_tasks=12, mean_photon_nr=0.15, mean_photon_decoy=0.075, fiber_attenuation=-3, voltage_amplitude=0.011, current_amplitude = 0.00041)
# start_simulation_batches(total_batches=50, simulations_in_batch=2, max_concurrent_tasks=12, mean_photon_nr=0.10, mean_photon_decoy=0.05, fiber_attenuation=-3, voltage_amplitude=0.0011, current_amplitude = 0.00041)
# start_simulation_batches(total_batches=50, simulations_in_batch=2, max_concurrent_tasks=12, mean_photon_nr=0.07, mean_photon_decoy=0.035, fiber_attenuation=-3, voltage_amplitude=0.0011, current_amplitude = 0.00041)
# batch 3
# start_simulation_batches(total_batches=50, simulations_in_batch=2, max_concurrent_tasks=12, mean_photon_nr=0.6, mean_photon_decoy=0.3, fiber_attenuation=-10, voltage_amplitude=0.0011, current_amplitude = 0.00041)
# start_simulation_batches(total_batches=50, simulations_in_batch=2, max_concurrent_tasks=12, mean_photon_nr=0.7, mean_photon_decoy=0.35, fiber_attenuation=-13, voltage_amplitude=0.0011, current_amplitude = 0.00041)
# start_simulation_batches(total_batches=50, simulations_in_batch=2, max_concurrent_tasks=12, mean_photon_nr=0.7, mean_photon_decoy=0.35, fiber_attenuation=-15, voltage_amplitude=0.0011, current_amplitude = 0.00041)
# start_simulation_batches(total_batches=50, simulations_in_batch=2, max_concurrent_tasks=12, mean_photon_nr=0.15, mean_photon_decoy=0.075, fiber_attenuation=-3, voltage_amplitude=0.007, current_amplitude = 0.00041)
# start_simulation_batches(total_batches=50, simulations_in_batch=2, max_concurrent_tasks=12, mean_photon_nr=0.15, mean_photon_decoy=0.075, fiber_attenuation=-3, voltage_amplitude=0.004, current_amplitude = 0.00041)
