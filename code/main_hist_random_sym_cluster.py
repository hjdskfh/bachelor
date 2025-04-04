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


# Assuming Saver, DataManager, SimulationConfig, SimulationManager are already defined and imported

def run_simulation_and_update_hist_all_pairs(i, n_samples_set, length_of_chain, base_path, style_file, database, jitter,
                                             detector_jitter, bins_per_symbol):
    config = SimulationConfig(
        database, seed=None, n_samples=n_samples_set, n_pulses=4, batchsize=1000,
        mean_voltage=0.9825, mean_current=0.080, voltage_amplitude=0.050, current_amplitude=0.0005,
        p_z_alice=0.5, p_decoy=0.1, p_z_bob=0.85,
        sampling_rate_FPGA=6.5e9, bandwidth=4e9, jitter=jitter,
        non_signal_voltage=-1.1, voltage_decoy=-0.1, voltage=-0.1,
        voltage_decoy_sup=-0.1, voltage_sup=-0.1,
        mean_photon_nr=0.7, mean_photon_decoy=0.1, fiber_attenuation=-3,
        detector_efficiency=0.3, dark_count_frequency=10, detection_time=1e-10,
        detector_jitter=detector_jitter, p_indep_x_states_non_dec=None, p_indep_x_states_dec=None,
        mlp=os.path.join(base_path, style_file), script_name=os.path.basename(__file__)
    )
    simulation = SimulationManager(config)

    simulation.run_simulation_hist_pick_symbols()
    
    # load data either from just simulated or from previous simulation
    data = np.load("simulation_data.npz")
    time_photons_det_z = data["time_photons_det_z"]
    time_photons_det_x = data["time_photons_det_x"]
    index_where_photons_det_z = data["index_where_photons_det_z"]
    index_where_photons_det_x = data["index_where_photons_det_x"]
    time_one_symbol = data["time_one_symbol"]
    basis = data["basis"]
    value = data["value"]
    decoy = data["decoy"]
    lookup_array = data["lookup_array"]
    
    hist_z, hist_x = DataProcessor.update_histogram_batches_all_pairs(length_of_chain, time_one_symbol, time_photons_det_z, time_photons_det_x,
                                                            index_where_photons_det_z, index_where_photons_det_x, amount_bins_hist,
                                                            bins_per_symbol, lookup_array, basis, value, decoy)

    return hist_x, hist_z, time_one_symbol, lookup_array

def run_simulation_batch_all_pairs(batch_id, n_samples_set, length_of_chain, base_path, style_file, database, jitter,
                                   detector_jitter, bins_per_symbol, amount_bins):
    hist_total_x = np.zeros(amount_bins, dtype=int)
    hist_total_z = np.zeros(amount_bins, dtype=int)
    t_sym_final = None
    lookup_array_final = None

    for j in range(simulations_in_batch):
        hist_x, hist_z, t_sym, lookup_array = run_simulation_and_update_hist_all_pairs(
            j, n_samples_set, length_of_chain, base_path, style_file, database, jitter,
            detector_jitter, bins_per_symbol
        )
        hist_total_x += hist_x
        hist_total_z += hist_z
        t_sym_final = t_sym
        lookup_array_final = lookup_array

    return hist_total_x, hist_total_z, t_sym_final, lookup_array_final

if __name__ == '__main__':
    Saver.memory_usage("START of Simulation: Before everything")
    start_time = time.time()

    database = DataManager()
    database.add_data('data/current_power_data.csv', 'Current (mA)', 'Optical Power (mW)', 9, 'current_power')
    database.add_data('data/voltage_shift_data.csv', 'Voltage (V)', 'Wavelength Shift (nm)', 20, 'voltage_shift')
    database.add_data('data/eam_transmission_data.csv', 'Voltage (V)', 'Transmission', 11, 'eam_transmission')
    database.add_data('data/wavelength_neff.csv', 'Wavelength (nm)', 'neff', 20, 'wavelength_neff')

    jitter = 1e-11
    detector_jitter =  100e-12
    n_samples_set = 20000
    database.add_jitter(jitter, 'laser')
    database.add_jitter(detector_jitter, 'detector')

    max_concurrent_tasks = 32
    # How many simulations per batch (each batch runs sequentially inside one task)
    simulations_in_batch = 2  # adjust this to increase per-task workload
    # Total number of batches to run (total simulations = simulations_in_batch * total_batches)
    total_batches = 50  # e.g., total simulations = 2 * 50 = 100  # 340 circa 4,5 stunden mit 2 sim per batch

    length_of_chain = 6*6 + 1
    bins_per_symbol_hist = 30
    amount_bins_hist = bins_per_symbol_hist * length_of_chain

    style_file = "Presentation_style_1_adjusted_no_grid.mplstyle"
    base_path = os.path.dirname(os.path.abspath(__file__))

    results = Parallel(n_jobs=max_concurrent_tasks)(
        delayed(run_simulation_batch_all_pairs)( 
            batch_id, n_samples_set, length_of_chain, base_path, style_file, database, jitter,
            detector_jitter, bins_per_symbol_hist, amount_bins_hist
        ) for batch_id in range(total_batches)
    )

    global_histogram_counts_x = np.zeros(amount_bins_hist, dtype=int)
    global_histogram_counts_z = np.zeros(amount_bins_hist, dtype=int)
    final_time_one_symbol, final_lookup_array = None, None

    for hist_x, hist_z, t_sym, lookup_array in results:
        global_histogram_counts_x += hist_x
        global_histogram_counts_z += hist_z
        final_time_one_symbol = t_sym
        final_lookup_array = lookup_array

    total_symbols = n_samples_set * simulations_in_batch * total_batches

    DataProcessor.plot_histogram_batch(bins_per_symbol_hist, final_time_one_symbol,
                               global_histogram_counts_x, global_histogram_counts_z,
                               final_lookup_array, total_symbols, start_symbol=3, end_symbol=10, name="random")

    Saver.save_array_as_npz("histograms_all_pairs",
                            histogram_counts_x=global_histogram_counts_x,
                            histogram_counts_z=global_histogram_counts_z)

    print(f"Execution time for simulation: {time.time() - start_time:.9f} seconds")
