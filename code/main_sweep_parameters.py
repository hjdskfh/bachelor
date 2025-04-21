import time
from datamanager import DataManager
from config import SimulationConfig
from simulationmanager import SimulationManager
from simulationhelper import SimulationHelper
from saver import Saver
from datamanager import DataManager
from dataprocessor import DataProcessor
import numpy as np
import os
import itertools
import json


def run_parameter_sweep():
    # Initialize DataManager and add data.
    database = DataManager()
    database.add_data('data/current_power_data.csv', 'Current (mA)', 'Optical Power (mW)', 9, 'current_power')
    database.add_data('data/voltage_shift_data.csv', 'Voltage (V)', 'Wavelength Shift (nm)', 20, 'voltage_shift')
    database.add_data('data/eam_transmission_data.csv', 'Voltage (V)', 'Transmission', 11, 'eam_transmission')
    database.add_data('data/wavelength_neff.csv', 'Wavelength (nm)', 'neff', 20, 'wavelength_neff')

    # Add jitter values.
   
    detector_jitter = 1e-11
    database.add_jitter(detector_jitter, 'detector')
    total_symbols = 2000    # Total number of symbols to be processed.

    # Fixed parameters (hard-coded) as one-liners.
    style_file = "Presentation_style_1_adjusted_no_grid.mplstyle"
    base_path = os.path.dirname(os.path.abspath(__file__))

    # Parameter sweep ranges.
    mean_photon_nr_values = [0.5] #[0.65, 0.7, 0.75]
    mean_photon_decoy_values = [0.25] #[0.05, 0.1, 0.15]
    p_decoy_values = [0.3] #[0.05, 0.1, 0.15]
    p_z_alice_values = [0.9] #[0.4, 0.5, 0.6]

    results = []
    overall_start = time.time()
    first = True
    # Iterate over every combination of the four parameters.
    for mean_photon_nr, mean_photon_decoy, p_z_alice, p_decoy in itertools.product(
            mean_photon_nr_values, mean_photon_decoy_values, p_z_alice_values, p_decoy_values):

        # Create a SimulationConfig with fixed parameters in one long call.
        config = SimulationConfig(database, n_samples=total_symbols, 
                mean_photon_nr=mean_photon_nr, mean_photon_decoy=mean_photon_decoy, 
                detector_jitter=detector_jitter,
                mlp=os.path.join(base_path, style_file), script_name = os.path.basename(__file__)
                )

        # Create the simulation manager and run the simulation.
        simulation = SimulationManager(config)
        data_processor = DataProcessor(config)
        if first:
            config_params = config.to_dict()
            Saver.save_to_json(config_params)
            first = False

        len_wrong_x_dec, len_wrong_x_non_dec, len_wrong_z_dec, len_wrong_z_non_dec, len_Z_checked_dec, len_Z_checked_non_dec, X_P_calc_non_dec, X_P_calc_dec = simulation.run_simulation_classificator()
        skr = data_processor.calc_SKR(len_wrong_x_dec, len_wrong_x_non_dec, len_wrong_z_dec, len_wrong_z_non_dec, 
                 len_Z_checked_dec, len_Z_checked_non_dec, X_P_calc_non_dec, X_P_calc_dec, total_symbols)
        
        # Retrieve a metric from the simulation (replace with your actual metric function).
        results.append({
            "mean_photon_nr": mean_photon_nr,
            "mean_photon_decoy": mean_photon_decoy,
            "p_z_alice": p_z_alice,
            "p_decoy": p_decoy,
            "len_wrong_x_dec": len_wrong_x_dec,
            "len_wrong_x_non_dec": len_wrong_x_non_dec,
            "len_wrong_z_dec": len_wrong_z_dec,
            "len_wrong_z_non_dec": len_wrong_z_non_dec,
            "len_Z_checked_dec": len_Z_checked_dec,
            "len_Z_checked_non_dec": len_Z_checked_non_dec,
            "X_P_calc_non_dec": X_P_calc_non_dec,
            "X_P_calc_dec": X_P_calc_dec,
            "SKR": skr
        })

        print(f"Swept: mean_photon_nr={mean_photon_nr}, mean_photon_decoy={mean_photon_decoy}, "
              f"p_z_alice={p_z_alice}, p_decoy={p_decoy} => skr: {skr}")
        # Save sweep results to a JSON file.
        with open("sweep_results.json", "w") as f:
            json.dump(results, f, indent=4)


    overall_end = time.time()
    print(f"Total sweep execution time: {overall_end - overall_start:.3f} seconds")

    # Save sweep results to a JSON file.
    with open("sweep_results.json", "w") as f:
        json.dump(results, f, indent=4)

    print("Sweep results saved to sweep_results.json")

if __name__ == '__main__':
    run_parameter_sweep()