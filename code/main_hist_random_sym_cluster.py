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
import functools

print = functools.partial(print, flush=True)
job_id = os.getenv("SLURM_JOB_ID")
print(f"Running SLURM Job ID: {job_id}")


os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

def run_simulation_and_update_hist_all_pairs(i, n_samples_set, length_of_chain, base_path, style_file, database,
                                             detector_jitter, bins_per_symbol):
    config = SimulationConfig(
        database, n_samples=n_samples_set,
        detector_jitter=detector_jitter,
        mlp=os.path.join(base_path, style_file), script_name=os.path.basename(__file__), job_id=job_id)
    
    simulation = SimulationManager(config)

    time_one_symbol, time_photons_det_z, time_photons_det_x, index_where_photons_det_z, index_where_photons_det_x, lookup_array, basis, value, decoy = simulation.run_simulation_hist_pick_symbols()
    
    '''# load data either from just simulated or from previous simulation
    data = np.load("simulation_data.npz", allow_pickle=True)
    time_photons_det_z = data["time_photons_det_z"]
    time_photons_det_x = data["time_photons_det_x"]
    index_where_photons_det_z = data["index_where_photons_det_z"]
    index_where_photons_det_x = data["index_where_photons_det_x"]
    time_one_symbol = data["time_one_symbol"]
    basis = data["basis"]
    value = data["value"]
    decoy = data["decoy"]
    lookup_array = data["lookup_array"]'''
    
    hist_z, hist_x, combined_list = DataProcessor.update_histogram_batches_all_pairs(length_of_chain, time_one_symbol, time_photons_det_z, time_photons_det_x,
                                                            index_where_photons_det_z, index_where_photons_det_x, amount_bins_hist,
                                                            bins_per_symbol, lookup_array, basis, value, decoy)

    with np.printoptions(threshold=np.inf):
        print("hist_x:", hist_x)
        print("hist_z:", hist_z)
        print("time_one_symbol:", time_one_symbol)
        print("combined_list:", combined_list)

    return hist_x, hist_z, time_one_symbol, combined_list

def run_simulation_batch_all_pairs(batch_id, n_samples_set, length_of_chain, base_path, style_file, database,
                                   detector_jitter, bins_per_symbol, amount_bins):
    hist_total_x = np.zeros(amount_bins, dtype=int)
    hist_total_z = np.zeros(amount_bins, dtype=int)
    t_sym_final = None
    combined_list_final = None

    for j in range(simulations_in_batch):
        hist_x, hist_z, t_sym, combined_list = run_simulation_and_update_hist_all_pairs(
            j, n_samples_set, length_of_chain, base_path, style_file, database,
            detector_jitter, bins_per_symbol
        )
        hist_total_x += hist_x
        hist_total_z += hist_z
        t_sym_final = t_sym
        combined_list_final = combined_list

    return hist_total_x, hist_total_z, t_sym_final, combined_list_final

if __name__ == '__main__':
    Saver.memory_usage("START of Simulation: Before everything")
    start_time = time.time()

    database = DataManager()
    database.add_data('data/current_power_data.csv', 'Current (mA)', 'Optical Power (mW)', 9, 'current_power')
    database.add_data('data/voltage_shift_data.csv', 'Voltage (V)', 'Wavelength Shift (nm)', 20, 'voltage_shift')
    # database.add_data('data/eam_transmission_data.csv', 'Voltage (V)', 'Transmission', 11, 'eam_transmission')
    database.add_data('data/wavelength_neff.csv', 'Wavelength (nm)', 'neff', 20, 'wavelength_neff')
    database.add_data('data/eam_static_results_renormalized.csv', 'Voltage (V)', 'Transmission', 16, 'eam_transmission')

    detector_jitter =  5e-12
    n_samples_set = 20000
    database.add_jitter(detector_jitter, 'detector')

    style_file = "Presentation_style_1_adjusted_no_grid.mplstyle"

    base_path = os.path.dirname(os.path.abspath(__file__))

    config = SimulationConfig(
        database, n_samples=n_samples_set,
        detector_jitter=detector_jitter,
        mlp=os.path.join(base_path, style_file), script_name=os.path.basename(__file__), job_id=job_id
    )
    simulation = SimulationManager(config)

    # Convert the config object to a dictionary
    config_params = config.to_dict()
    Saver.save_to_json(config_params)


    max_concurrent_tasks = 2
    # How many simulations per batch (each batch runs sequentially inside one task)
    simulations_in_batch = 2  # adjust this to increase per-task workload
    # Total number of batches to run (total simulations = simulations_in_batch * total_batches)
    total_batches = 100 # e.g., total simulations = 2 * 50 = 100  # 340 circa 4,5 stunden mit 2 sim per batch, 800 10 stunden

    length_of_chain = 6*6*2
    bins_per_symbol_hist = 100
    amount_bins_hist = bins_per_symbol_hist * length_of_chain
    

    style_file = "Presentation_style_1_adjusted_no_grid.mplstyle"
    base_path = os.path.dirname(os.path.abspath(__file__))

    results = Parallel(n_jobs=max_concurrent_tasks)(
        delayed(run_simulation_batch_all_pairs)( 
            batch_id, n_samples_set, length_of_chain, base_path, style_file, database,
            detector_jitter, bins_per_symbol_hist, amount_bins_hist
        ) for batch_id in range(total_batches)
    )

    # PRINT RESULTS
    # Unpack all components into separate lists
    hist_x_list = []
    hist_z_list = []
    t_sym_list = []
    combined_list_list = []

    for hist_x, hist_z, t_sym, lookup_array in results:
        hist_x_list.append(hist_x)
        hist_z_list.append(hist_z)
        t_sym_list.append(t_sym)
        combined_list_list.append(lookup_array)

    # Now save them — use object dtype if arrays are different shapes
    np.savez("histograms_random_results_middle.npz",
            hist_x=np.array(hist_x_list, dtype=object),
            hist_z=np.array(hist_z_list, dtype=object),
            t_sym=np.array(t_sym_list, dtype=object),
            combined_list=np.array(combined_list_list, dtype=object))

    global_histogram_counts_x = np.zeros(amount_bins_hist, dtype=int)
    global_histogram_counts_z = np.zeros(amount_bins_hist, dtype=int)
    final_time_one_symbol, final_combined_list_array = None, None

    for hist_x, hist_z, t_sym, combined_list in results:
        global_histogram_counts_x += hist_x
        global_histogram_counts_z += hist_z
        final_time_one_symbol = t_sym
        final_combined_list_array = combined_list

    total_symbols = n_samples_set * simulations_in_batch * total_batches
    
    DataProcessor.plot_histogram_batch_random(bins_per_symbol_hist, final_time_one_symbol,
                            global_histogram_counts_x, global_histogram_counts_z,
                            final_combined_list_array, total_symbols, start_pair=0, end_pair=4, name="random")

    Saver.save_array_as_npz_data(f"histograms_random_con_{max_concurrent_tasks}_12_total_batches_{total_batches}_600_bins_{bins_per_symbol_hist}_80",
                            bins_per_symbol_hist=bins_per_symbol_hist,
                            final_time_one_symbol=final_time_one_symbol,
                            global_histogram_counts_x=global_histogram_counts_x,
                            global_histogram_counts_z=global_histogram_counts_z,
                            final_combined_list_array=final_combined_list_array,
                            total_symbols=total_symbols)

    print(f"Execution time for simulation: {time.time() - start_time:.9f} seconds")
