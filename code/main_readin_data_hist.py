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

# file_name = '../stuff_from_cluster/2025_04_13/20250410_190215_histograms.npz'
# file_name = r'C:\Users\leavi\bachelor\stuff_from_cluster\2025_04_13\20250413_150050_histograms_fixed.npz'
# file_name = r'C:\Users\leavi\OneDrive\Dokumente\Uni\Semester 7\NeuMoQP\Programm\stuff_from_cluster\2025_04_13\random\20250413_172141_histograms_random.npz'
# file_name = r'C:\Users\leavi\OneDrive\Dokumente\Uni\Semester 7\NeuMoQP\Programm\stuff_from_cluster\2025_04_13\fixed\20250413_173800_histograms_fixed.npz'
file_name = r'C:\Users\leavi\OneDrive\Dokumente\Uni\Semester 7\NeuMoQP\Programm\stuff_from_cluster\2025_04_13\other\20250413_173800_histograms_fixed.npz'

if os.path.exists(file_name):
    print("File exists!")
else:
    print("File does not exist!")
data = np.load(file_name, allow_pickle=True)
print(data)
bins_per_symbol_hist = data["bins_per_symbol_hist"]
final_time_one_symbol = data["final_time_one_symbol"]
global_histogram_counts_x = data["global_histogram_counts_x"]
global_histogram_counts_z = data["global_histogram_counts_z"]
final_lookup_array = data["final_lookup_array"]
total_symbols = data["total_symbols"]
np.set_printoptions(threshold=np.inf)
print(f"bins_per_symbol_hist: {bins_per_symbol_hist}")
print(f"final_time_one_symbol: {final_time_one_symbol}")
print(f"global_histogram_counts_x: {global_histogram_counts_x}")
print(f"global_histogram_counts_z: {global_histogram_counts_z}")
print(f"final_lookup_array: {final_lookup_array}")
print(f"total_symbols: {total_symbols}")


DataProcessor.plot_histogram_batch(bins_per_symbol_hist, final_time_one_symbol,
                            global_histogram_counts_x, global_histogram_counts_z,
                            final_lookup_array, total_symbols, start_symbol=0, end_symbol=20, name="random")

