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

# file_name = '../stuff_from_cluster/2025_04_11/20250410_190215_histograms.npz'
file_name = r'C:\Users\leavi\bachelor\stuff_from_cluster\2025_04_11\20250410_190215_histograms.npz'
data = np.load(file_name, allow_pickle=True)
print(data)
bins_per_symbol_hist = data["bins_per_symbol_hist"]
final_time_one_symbol = data["final_time_one_symbol"]
global_histogram_counts_x = data["global_histogram_counts_x"]
global_histogram_counts_z = data["global_histogram_counts_z"]
final_lookup_array = data["final_lookup_array"]
total_symbols = data["total_symbols"]

DataProcessor.plot_histogram_batch(bins_per_symbol_hist, final_time_one_symbol,
                            global_histogram_counts_x, global_histogram_counts_z,
                            final_lookup_array, total_symbols, start_symbol=3, end_symbol=10, name="random")

