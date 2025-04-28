from operator import length_hint
import time
from tkinter import N
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
import json
import re
from collections import defaultdict

config = SimulationConfig(None)
data_processor = DataProcessor(config)
json_filepath = r"C:\Users\leavi\bachelor\stuff_from_cluster\2025_04_23\repeat\simulation_config_20250423_133838.json.json"

# Load the JSON file
with open(json_filepath, 'r') as file:
    config_loaded = json.load(file)

config.mean_photon_nr = 0.182 #config_loaded["mean_photon_nr"]
config.mean_photon_decoy = 0.1 #config_loaded["mean_photon_decoy"]
config.p_decoy = 0.19 #config_loaded["p_decoy"]
config.p_z_alice = 0.8 #config_loaded["p_z_alice"]


factor_arr = np.array([1, 10, 100, 1000, 10000, 100000, 1000000, 10000000, 100000000, 1000000000])
for factor in factor_arr:
        
    len_wrong_x_dec = 0.152 
    len_wrong_x_non_dec = 2.592 
    len_wrong_z_dec = 82.08
    len_wrong_z_non_dec = 365.472 
    len_Z_checked_dec = 1543.7
    len_Z_checked_non_dec = 12726.7 
    X_P_calc_dec = 4.864 *1.189
    X_P_calc_non_dec = 62.2 
    total_symbols = 1200000
    
    skr = data_processor.calc_SKR_old(len_wrong_x_dec, len_wrong_x_non_dec, len_wrong_z_dec, len_wrong_z_non_dec, 
                    len_Z_checked_dec, len_Z_checked_non_dec, X_P_calc_dec, X_P_calc_non_dec, total_symbols)

    print(f"SKR: {skr} for factor {factor}")