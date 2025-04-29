from math import comb
from re import M
from tracemalloc import start
from matplotlib.pylab import f
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd
from saver import Saver


class DataProcessor:
    def __init__(self, config):
        self.config = config

    # ========== Prepare Data for Histogram ==========
    @staticmethod
    def update_histogram_batches(length_of_chain, time_photons_det_x, time_photons_det_z, time_one_symbol, total_symbols,
                                index_where_photons_det_x, index_where_photons_det_z, amount_bins_hist, bins_per_symbol = 30):
        # print(f"total_symbols: {total_symbols}")
        n_rep = total_symbols // length_of_chain
    
        assert len(time_photons_det_x) == len(index_where_photons_det_x), f"Length mismatch: {len(time_photons_det_x)} vs {len(index_where_photons_det_x)}"
        assert len(time_photons_det_z) == len(index_where_photons_det_z), f"Length mismatch: {len(time_photons_det_z)} vs {len(index_where_photons_det_z)}"

        local_histogram_counts_x = np.zeros(amount_bins_hist, dtype=int)
        local_histogram_counts_z = np.zeros(amount_bins_hist, dtype=int)
        # Define bins spanning the time interval for this batch.
        bins_arr_per_symbol = np.linspace(0, time_one_symbol, bins_per_symbol + 1)        
        # Loop over each cycle (repetition)
        for rep in range(n_rep):
            for s in range(length_of_chain):  # which symbol
                row_idx = rep * length_of_chain + s 
                print(f"row_idx: {row_idx}, index_where_photons_det_x: {index_where_photons_det_x}")
                if np.isin(row_idx, index_where_photons_det_x):
                    ind_short = np.where(index_where_photons_det_x == row_idx)[0]
                    valid_x = time_photons_det_x[ind_short][~np.isnan(time_photons_det_x[ind_short])]
                    bin_index = np.digitize(valid_x, bins_arr_per_symbol) - 1
                    bin_index = np.minimum(bin_index, bins_per_symbol - 1)  # Ensure it counts in the last bin if exactly time_one_symbol
                    # insert into histogram_counts_z with 30*symbol + bin_index 
                    idx_hist = bins_per_symbol * s + bin_index
                    if np.size(idx_hist) > 1:
                        print("idx has multiple values!")
                        print(f"bins_per_symbol:{bins_per_symbol}, s:{s}, bin_index:{bin_index}")
                    else:
                        if idx_hist >= len(local_histogram_counts_x):
                            print(f"Out of bounds: idx_hist={idx_hist}, len={len(local_histogram_counts_x)}, s={s}, bin_index={bin_index}")
                    assert np.all(0 <= bin_index) and np.all(bin_index < bins_per_symbol), f"bin_index {bin_index} out of bounds for symbol {s}"
                    np.add.at(local_histogram_counts_x, idx_hist, 1)
                if np.isin(row_idx, index_where_photons_det_z):
                    ind_short = np.where(index_where_photons_det_z == row_idx)[0]
                    valid_z = time_photons_det_z[ind_short][~np.isnan(time_photons_det_z[ind_short])]
                    bin_index = np.digitize(valid_z, bins_arr_per_symbol) - 1
                    bin_index = np.minimum(bin_index, bins_per_symbol - 1)  # Ensure it counts in the last bin if exactly time_one_symbol
                    # insert into histogram_counts_z with 30*symbol + bin_index 
                    idx_hist = bins_per_symbol * s + bin_index
                    if np.size(idx_hist) > 1:
                        print("idx has multiple values!")
                        print(f"bins_per_symbol:{bins_per_symbol}, s:{s}, bin_index:{bin_index}")
                    else:
                        if idx_hist >= len(local_histogram_counts_z):
                            print(f"Out of bounds: idx_hist={idx_hist}, len={len(local_histogram_counts_z)}, s={s}, bin_index={bin_index}")
                    assert np.all(0 <= bin_index) and np.all(bin_index < bins_per_symbol), f"bin_index {bin_index} out of bounds for symbol {s}"
                    np.add.at(local_histogram_counts_z, idx_hist, 1)
        if np.any(local_histogram_counts_x > 0) or np.any(local_histogram_counts_z > 0):
            print(f"local_histogram_counts_x: {local_histogram_counts_x}")
            print(f"local_histogram_counts_z: {local_histogram_counts_z}")

        return local_histogram_counts_x, local_histogram_counts_z

    @staticmethod
    def plot_histogram_batch(bins_per_symbol, time_one_symbol, histogram_counts_x, histogram_counts_z, lookup_arr, total_symbols, start_symbol=3, end_symbol=10, name=' ', leave_x = False, leave_z = False):
        assert 0 <= start_symbol <= end_symbol <= 36
        def normalize_histogram(histogram_counts, slice_start, slice_end, bins_per_symbol):
            # Get the maximum value in the specified slice
            max_in_slice = np.max(histogram_counts) if np.max(histogram_counts) > 0 else 1
            # Normalize the entire histogram using the maximum value in the slice
            normalized_histogram = histogram_counts / max_in_slice
            return normalized_histogram
        
        amount_of_symbols_incl_start_and_end = end_symbol - start_symbol + 1
        bins = np.linspace(0, amount_of_symbols_incl_start_and_end * time_one_symbol, bins_per_symbol * amount_of_symbols_incl_start_and_end + 1)    
        
        # Normalize histogram counts so the highest peak is at 1
        histogram_counts_x = normalize_histogram(histogram_counts_x, start_symbol, end_symbol +1, bins_per_symbol)
        histogram_counts_z = normalize_histogram(histogram_counts_z, start_symbol, end_symbol +1, bins_per_symbol)

        plt.figure(figsize=(10, 6))
        # Plot as bar chart; you can also use plt.hist with precomputed counts.
        width = (bins[1] - bins[0])
        if leave_x == False:
            plt.bar(bins[:-1], histogram_counts_x[start_symbol * bins_per_symbol :(end_symbol + 1) * bins_per_symbol], width=width, alpha=0.6, label='X basis', color='blue')
        if leave_z == False:
            plt.bar(bins[:-1], histogram_counts_z[start_symbol * bins_per_symbol :(end_symbol + 1) * bins_per_symbol], width=width, alpha=0.6, label='Z basis', color='red')

        for i in range(amount_of_symbols_incl_start_and_end):
            plt.axvline(x=i * time_one_symbol, color='grey', linestyle='--', linewidth=1)

            # Place the symbol halfway between this line and the next
            if i < amount_of_symbols_incl_start_and_end:
                x_mid = i * time_one_symbol + time_one_symbol / 2
                symbol = lookup_arr[start_symbol + i]
                if leave_x == False and leave_z == False:
                    y_max = max(max(histogram_counts_x), max(histogram_counts_z))
                elif leave_x == False:
                    y_max = max(histogram_counts_x)
                elif leave_z == False:
                    y_max = max(histogram_counts_z)
                else:
                    y_max = 1  # Default value if neither histogram is plotted
                basis = symbol[0]  # assuming symbol is like 'X0' or 'Z1'
                color = 'green' if basis == 'X' else 'purple'
                
                plt.text(x_mid, y_max * 0.915, symbol, ha='center', va='bottom', fontsize=18, color=color, fontweight='bold')

        plt.xlabel("Time (s)", fontsize = 18)
        plt.ylabel("Cumulative Photon Counts", fontsize = 18)
        # plt.title(f"Cumulative Histogram for {start_symbol} to {end_symbol} for {total_symbols} {name} symbols")
        plt.legend(fontsize = 16)
        plt.ylim(0, 1.1)
        plt.tick_params(axis='both', which='major', labelsize=18)  # Increase tick size for major ticks
        plt.tight_layout()
        Saver.save_plot(f"hist_fixed_symbols_{start_symbol}_to_{end_symbol}", no_time = True)

    @staticmethod
    def get_all_pair_indices(lookup_arr):
        """
        Given a 1D array (or list) of symbol identifiers (for one chain),
        return a dictionary mapping each adjacent pair (as a tuple)
        to a numpy array of indices where that pair occurs.
        
        The returned index i indicates that lookup_arr[i] and lookup_arr[i+1] form that pair.
        """
        lookup_arr = np.array(lookup_arr)
        # Create an array of shape (N-1, 2) with each row as a pair (lookup_arr[i], lookup_arr[i+1])
        pairs = np.column_stack((lookup_arr[:-1], lookup_arr[1:]))
        # Get the unique pairs (each row is a unique pair)
        unique_pairs = np.unique(pairs, axis=0)
        print(f"unique_pairs: {unique_pairs}")
        
        pair_indices_dict = {}
        for pair in unique_pairs:
            pair_tuple = tuple(pair)
            # Find indices where this exact pair occurs (vectorized)
            indices = np.nonzero((pairs[:, 0] == pair_tuple[0]) & (pairs[:, 1] == pair_tuple[1]))[0]
            pair_indices_dict[pair_tuple] = indices
        return pair_indices_dict


    def create_pair_mapping(raw_symbol_lookup):
        """
        Create all possible pairs from the raw_symbol_lookup dictionary and store them sequentially.
        Each pair is assigned a unique position in a combined list.

        Parameters:
        - raw_symbol_lookup: dict
            A dictionary mapping tuples to symbol strings (e.g., {(1, 1, 0): "Z0", ...}).

        Returns:
        - pair_mapping: dict
            A dictionary where keys are pairs (e.g., ('Z0', 'Z1')) and values are their positions in the combined list.
        - combined_list: list
            A list where all pairs are stored sequentially.
        """
        # Extract all symbols from the raw_symbol_lookup
        symbols = list(raw_symbol_lookup.values())

        # Generate all possible pairs
        all_pairs = [(a, b) for a in symbols for b in symbols]

        # Create a mapping of pairs to their positions
        pair_mapping = {pair: idx for idx, pair in enumerate(all_pairs)}

        # Create a combined list where all pairs are stored sequentially
        combined_list = [pair for pair in all_pairs]
        

        return pair_mapping, combined_list


    def update_histogram_batches_all_pairs(length_of_chain, time_one_symbol, time_photons_det_z, time_photons_det_x,
                                        index_where_photons_det_z, index_where_photons_det_x, amount_bins_hist,
                                        bins_per_symbol, lookup_arr, basis, value, decoy):
        """
        Update the histogram counts for all adjacent pairs in the given sequence.
        
        The simulation data (time_photons_det and index_where_photons_det) is assumed to be stored
        in an NPZ file, and total_symbols is the total number of symbols over all chains.
        
        Parameters:
        length_of_chain: int
            Number of symbols in one chain (e.g. 65).
        time_one_symbol: float
            Duration (time window) for one symbol.
        time_photons_det: 1D numpy array
            The arrival times (relative to each symbol start) of detected photons.
        index_where_photons_det: numpy array
            Global symbol indices (across the batch) where detections occurred.
        amount_bins_hist: int
            Total number of histogram bins (typically bins_per_symbol * length_of_chain).
        bins_per_symbol: int
            Number of bins per symbol.
        lookup_arr: list of str
            The lookup array for one chain, e.g.:
            ['Z0', 'Z0', 'Z1', 'Z0', 'X0', 'Z0', 'X1', ...]
            (Here, strings are not normalized, so "Z0" and "Z0*" are distinct.)
        
        Returns:
        local_histogram_counts: 1D numpy array of length amount_bins_hist.
        """
        raw_symbol_lookup = {
            (1, 1, 0): "Z0",
            (1, 0, 0): "Z1",
            (0, -1, 0): "X+",
            (1, 1, 1): "Z0*",  # or "Z0_decoy" if you prefer
            (1, 0, 1): "Z1*",  # or "Z1_decoy"
            (0, -1, 1): "X+*",  # or "X+_decoy"
        }
        assert len(time_photons_det_x) == len(index_where_photons_det_x), f"Length mismatch: {len(time_photons_det_x)} vs {len(index_where_photons_det_x)}"
        assert len(time_photons_det_z) == len(index_where_photons_det_z), f"Length mismatch: {len(time_photons_det_z)} vs {len(index_where_photons_det_z)}"


        local_histogram_counts_x = np.zeros(amount_bins_hist, dtype=int)
        local_histogram_counts_z = np.zeros(amount_bins_hist, dtype=int)
        
        # Define bin edges for one symbol's time window.
        bins_arr = np.linspace(0, time_one_symbol, bins_per_symbol + 1)
        
        # Get dictionary mapping each adjacent pair (for one chain) to the positions where they occur.
        # pair_indices_dict = DataProcessor.get_all_pair_indices(lookup_arr)
        pair_mapping, combined_list = DataProcessor.create_pair_mapping(raw_symbol_lookup)


        def process(idx_left, idx_right, index_where_photons_det, time_photons_det, local_histogram_counts):
            
            if idx_left >= 0 and idx_right < len(basis):
                # looked at index is second symbol:
                pair_key = (raw_symbol_lookup[(basis[idx_left], value[idx_left], decoy[idx_left])], 
                            raw_symbol_lookup[(basis[idx_right], value[idx_right], decoy[idx_right])])
                # print(f"pair_key:   {pair_key}")
                position_pair = pair_mapping.get(pair_key, -1)
                # print(f"position_brujin_left: {position_brujin_left}")
            
                # Process the first symbol of the pair:
                if idx_left in index_where_photons_det:
                    # print(f"idx_left in index_where_photons_det: {idx_left}")
                    inds_first = np.where(index_where_photons_det == idx_left)[0]
                    valid_times_first = time_photons_det[inds_first]
                    # print(f"valid_times_first: {valid_times_first}")
                    valid_times_first = valid_times_first[~np.isnan(valid_times_first)]
                    # print(f"valid_times_first without nan: {valid_times_first}")
                    bin_indices_first = np.digitize(valid_times_first, bins_arr) - 1
                    # Update histogram: position = pos (for first symbol)
                    for b in bin_indices_first:
                        if 0 <= b < bins_per_symbol:
                            overall_bin = (2 * position_pair) * bins_per_symbol + b
                            local_histogram_counts[overall_bin] += 1

                # Process the second symbol of the pair:
                if idx_right in index_where_photons_det:
                    # print(f"idx_right in index_where_photons_det: {idx_right}")
                    inds_first = np.where(index_where_photons_det == idx_right)[0]
                    valid_times_first = time_photons_det[inds_first]
                    # print(f"valid_times_first: {valid_times_first}")
                    valid_times_first = valid_times_first[~np.isnan(valid_times_first)]
                    # print(f"valid_times_first without nan: {valid_times_first}")
                    bin_indices_first = np.digitize(valid_times_first, bins_arr) - 1
                    # Update histogram: position = pos (for first symbol)
                    for b in bin_indices_first:
                        if 0 <= b < bins_per_symbol:
                            overall_bin = (2 * position_pair + 1) * bins_per_symbol + b
                            local_histogram_counts[overall_bin] += 1

        # Process each chain (repetition)
        print(f"index_where_photons_det_z max: {index_where_photons_det_z.max()}, index_where_photons_det_x max: {index_where_photons_det_x.max()}")
        for idx_where_z in index_where_photons_det_z:
            idx_before_z = idx_where_z - 1
            idx_after_z = idx_where_z + 1

            # print(f"idx_before_z:{idx_before_z}, idx_after_z:{idx_after_z}, idx_where_z:{idx_where_z}")
            # looked at photon is right part of symbol
            process(idx_before_z, idx_where_z, index_where_photons_det_z, time_photons_det_z, local_histogram_counts_z)
            # looked at photon is left part of symbol
            process(idx_where_z, idx_after_z, index_where_photons_det_z, time_photons_det_z, local_histogram_counts_z)

        for idx_where_x in index_where_photons_det_x:
            idx_before_x = idx_where_x - 1
            idx_after_x = idx_where_x + 1

            # print(f"idx_before_x:{idx_before_x}, idx_after_x:{idx_after_x}, idx_where_x:{idx_where_x}")
            # looked at photon is right part of symbol
            process(idx_before_x, idx_where_x, index_where_photons_det_x, time_photons_det_x, local_histogram_counts_x)
            # looked at photon is left part of symbol
            process(idx_where_x, idx_after_x, index_where_photons_det_x, time_photons_det_x, local_histogram_counts_x)

        
        return local_histogram_counts_z, local_histogram_counts_x, combined_list
    
    def plot_histogram_batch_random(bins_per_symbol, time_one_symbol, histogram_counts_x, histogram_counts_z, combined_list, total_symbols, start_pair=3, end_pair=10, name=' ', leave_x = False, leave_z = False, p_decoy = None):
        assert 0 <= start_pair <= end_pair <= 35
        def normalize_histogram(histogram_counts, slice_start, slice_end, bins_per_symbol):
            # Get the maximum value in the specified slice
            max_in_slice = np.max(histogram_counts[slice_start * bins_per_symbol:(slice_end + 1) * bins_per_symbol]) if np.max(histogram_counts[slice_start * bins_per_symbol:(slice_end + 1) * bins_per_symbol]) > 0 else 1
            # Normalize the entire histogram using the maximum value in the slice
            normalized_histogram = histogram_counts / max_in_slice
            return normalized_histogram
        
        print(f"combined_list: {combined_list}")
       
        amount_of_pairs_incl_start_and_end = end_pair - start_pair + 1
        amount_symbols = 2*amount_of_pairs_incl_start_and_end

        start_symbol = start_pair * 2
        end_symbol = end_pair * 2 + 1
        
        
        bins = np.linspace(0, amount_symbols * time_one_symbol, bins_per_symbol * amount_symbols + 1)    
        
        # Normalize histogram counts so the highest peak is at 1
        # histogram_counts_x = normalize_histogram(histogram_counts_x, start_symbol, end_symbol + 1, bins_per_symbol)
        # histogram_counts_z = normalize_histogram(histogram_counts_z, start_symbol, end_symbol + 1, bins_per_symbol)

        plt.figure(figsize=(10, 6))
        # Plot as bar chart; you can also use plt.hist with precomputed counts.
        width = (bins[1] - bins[0])
        if leave_x == False:
            plt.bar(bins[:-1], histogram_counts_x[(start_symbol) * bins_per_symbol :((end_symbol) + 1) * bins_per_symbol], width=width, alpha=0.6, label='X basis', color='blue')
        if leave_z == False:
            plt.bar(bins[:-1], histogram_counts_z[(start_symbol) * bins_per_symbol :((end_symbol) + 1) * bins_per_symbol], width=width, alpha=0.6, label='Z basis', color='red')

        for i in range(amount_symbols):
            if (start_symbol + i) % 2 == 0:
                plt.axvline(x=i * time_one_symbol, color='grey', linestyle='--', linewidth=3)
            else:
                plt.axvline(x=i * time_one_symbol, color='grey', linestyle='--', linewidth=1)

            # Place the symbol halfway between this line and the next
            if i < amount_symbols:
                print(f"blub")
                x_mid = i * time_one_symbol + time_one_symbol / 2
                symbol_index = (start_symbol + i) // 2
                if (start_symbol + i)  % 2 == 0:
                    symbol = combined_list[symbol_index][0]
                else:
                    symbol = combined_list[symbol_index][1]
                print(f"symbol: {symbol}")

                if leave_x == False and leave_z == False:
                    y_max = max(max(histogram_counts_x), max(histogram_counts_z))
                elif leave_x == False:
                    y_max = max(histogram_counts_x)
                elif leave_z == False:
                    y_max = max(histogram_counts_z)
                else:
                    y_max = 1  # Default value if neither histogram is plotted
                basis = symbol[0]  # assuming symbol is like 'X0' or 'Z1'
                color = 'green' if basis == 'X' else 'purple'
                print(f"blub2")
                plt.text(x_mid, y_max * 0.5, symbol, ha='center', va='bottom', fontsize=14, color=color, fontweight='bold')
        
        plt.axvline(x=amount_symbols * time_one_symbol, color='grey', linestyle='--', linewidth=3)

        plt.xlabel("Time (s)")
        plt.ylabel("Cumulative Counts")
        # plt.title(f"Cumulative Histogram for {start_pair} to {end_pair} for {total_symbols} {name} symbols")
        plt.legend()
        plt.tight_layout()
        Saver.save_plot(f"hist_symbols_{start_pair}_to_{end_pair}", no_time = True)

    #  -------SKR ------
    # Entropy function
    @staticmethod
    def entropy(p):
        return -p * np.log2(p) - (1 - p) * np.log2(1 - p)


    # Finite key correction
    @staticmethod
    def n_finite_key_corrected(sign, k, p_k, n_k, n, epsilon):
        if sign == "+":
            return np.exp(k) / p_k * (n_k + np.sqrt(n / 2 * np.log(1 / epsilon)))
        elif sign == "-":
            return np.exp(k) / p_k * (n_k - np.sqrt(n / 2 * np.log(1 / epsilon)))
        else:
            raise ValueError("Invalid sign value. Must be '+' or '-'.")


    # Gamma function
    @staticmethod
    def gamma(a, b, c, d):
        return np.sqrt(
            (c + d)
            * (1 - b)
            * b
            / (c * d * np.log(2))
            * np.log((c + d) * 21**2 / (c * d * (1 - b) * a**2))
        )
    
    def calc_SKR(self, n_Z_mus_in, n_Z_mud_in, n_X_mus_in, n_X_mud_in, m_Z_mus_in, m_Z_mud_in, m_X_mus_in, m_X_mud_in, total_symbols, factor):
        block_size = None  
        
        total_bit_sequence_length = total_symbols * factor  # Number of detections in key generating basis
        q_Z = self.config.p_z_bob  # Bob chooses a basis Z and X with probabilities qz
       
        epsilon_sec = 1e-9  # 
        epsilon_cor = 1e-15  # Secret key are identical except of probability epsilon_cor
        repetition_rate = self.config.sampling_rate_FPGA / self.config.n_pulses  # Pulse (symbol) repetition rate
        # print(f"self.config.sampling_rate_FPGA: {self.config.sampling_rate_FPGA}, self.config.n_pulses: {self.config.n_pulses}, repetition_rate: {repetition_rate}")	
        fEC = 1.19  # Error correction effciency
        epsilon_1 = epsilon_sec / 19

        def calculate_skr(params, total_bit_sequence_length):
            mus, mud, p_mus, p_Z = params
            p_mud = 1 - p_mus
            p_X = 1 - p_Z
            q_X = 1 - q_Z

            # Compute gain
            # eta_ch_Z = np.power(10, -channel_attenuation_Z / 10)
            # eta_ch_X = np.power(10, -channel_attenuation_X / 10)
            # gain_Z_mus = 1 - (1 - y_0) * np.exp(-1 * mus * eta_bob * eta_ch_Z)
            # gain_Z_mud = 1 - (1 - y_0) * np.exp(-1 * mud * eta_bob * eta_ch_Z)
            # gain_X_mus = 1 - (1 - y_0) * np.exp(-1 * mus * eta_bob * eta_ch_X)
            # gain_X_mud = 1 - (1 - y_0) * np.exp(-1 * mud * eta_bob * eta_ch_X)

            # Recalucalte total_bit_sequence_length to match desired nZ
            # if block_size is not None:
            #     total_bit_sequence_length = block_size / (
            #         p_Z * (p_mus * gain_Z_mus + p_mud * gain_Z_mud)
            #     )

            # Compute total detection events
            # n_Z_mus = p_Z * q_Z * p_mus * gain_Z_mus * total_bit_sequence_length
            # n_Z_mud = p_Z * q_Z * p_mud * gain_Z_mud * total_bit_sequence_length
            # n_X_mus = p_X * q_X * p_mus * gain_X_mus * total_bit_sequence_length
            # n_X_mud = p_X * q_X * p_mud * gain_X_mud * total_bit_sequence_length
            # n_Z = n_Z_mus + n_Z_mud
            # n_X = n_X_mus + n_X_mud
            # print(f"factor: {factor}")
            n_Z_mus = n_Z_mus_in * factor
            n_Z_mud = n_Z_mud_in * factor
            n_X_mus = n_X_mus_in * factor
            n_X_mud = n_X_mud_in * factor
            n_Z = n_Z_mus + n_Z_mud
            n_X = n_X_mus + n_X_mud

            # Compute error
            # error_Z_mus = y_0 * (e_0 - e_detector_Z) + e_detector_Z * gain_Z_mus
            # error_Z_mud = y_0 * (e_0 - e_detector_Z) + e_detector_Z * gain_Z_mud
            # error_X_mus = y_0 * (e_0 - e_detector_X) + e_detector_X * gain_X_mus
            # error_X_mud = y_0 * (e_0 - e_detector_X) + e_detector_X * gain_X_mud

            # # Compute total error events
            # m_Z_mus = p_Z * q_Z * p_mus * error_Z_mus * total_bit_sequence_length
            # m_Z_mud = p_Z * q_Z * p_mud * error_Z_mud * total_bit_sequence_length
            # m_X_mus = p_X * p_X * p_mus * error_X_mus * total_bit_sequence_length
            # m_X_mud = p_X * p_X * p_mud * error_X_mud * total_bit_sequence_length

            m_Z_mus = m_Z_mus_in * factor
            m_Z_mud = m_Z_mud_in * factor
            m_X_mus = m_X_mus_in * factor
            m_X_mud = m_X_mud_in * factor
            m_Z = m_Z_mus + m_Z_mud
            m_X = m_X_mus + m_X_mud

            # Probabilites sending vaccum and single photon states
            tau_0 = p_mus * np.exp(-mus) + p_mud * np.exp(-mud)
            tau_1 = p_mus * mus * np.exp(-mus) + p_mud * mud * np.exp(-mud)

            # Compute finite-key security bounds
            s_l_Z0 = (
                tau_0
                / (mus - mud)
                * (
                    mus * DataProcessor.n_finite_key_corrected("-", mud, p_mud, n_Z_mud, n_Z, epsilon_1)
                    - mud * DataProcessor.n_finite_key_corrected("+", mus, p_mus, n_Z_mus, n_Z, epsilon_1)
                )
            )
            s_u_Z0 = 2 * (
                tau_0 * DataProcessor.n_finite_key_corrected("+", mus, p_mus, m_Z_mus, m_Z, epsilon_1) #warum signal?? in supplemetary A16 steht einfach nur k???
                + np.sqrt(n_Z / 2 * np.log(1 / epsilon_1))  # und es sind verschiedene eplisons??
            )
            s_l_Z1 = (
                tau_1
                * mus
                / (mud * (mus - mud))
                * (
                    DataProcessor.n_finite_key_corrected("-", mud, p_mud, n_Z_mud, n_Z, epsilon_1)
                    - mud**2
                    / mus**2
                    * DataProcessor.n_finite_key_corrected("+", mus, p_mus, n_Z_mus, n_Z, epsilon_1)
                    - (mus**2 - mud**2) / (mus**2 * tau_0) * s_u_Z0
                )
            )
            s_u_X0 = 2 * (
                tau_0 * DataProcessor.n_finite_key_corrected("+", mud, p_mud, m_X_mud, m_X, epsilon_1)
                + np.sqrt(n_X / 2 * np.log(1 / epsilon_1))
            )
            s_l_X1 = (
                tau_1
                * mus
                / (mud * (mus - mud))
                * (
                    DataProcessor.n_finite_key_corrected("-", mud, p_mud, n_X_mud, n_X, epsilon_1)
                    - mud**2
                    / mus**2
                    * DataProcessor.n_finite_key_corrected("+", mus, p_mus, n_X_mus, n_X, epsilon_1)
                    - (mus**2 - mud**2) / (mus**2 * tau_0) * s_u_X0
                )
            )
            # print(f"s_l_Z0: {s_l_Z0}, s_u_Z0: {s_u_Z0}, s_l_Z1: {s_l_Z1}, s_u_X0: {s_u_X0}, s_l_X1: {s_l_X1}")
            v_u_X1 = (
                tau_1
                / (mus - mud)
                * (
                    DataProcessor.n_finite_key_corrected("+", mus, p_mus, m_X_mus, m_X, epsilon_1)
                    - DataProcessor.n_finite_key_corrected("-", mud, p_mud, m_X_mud, m_X, epsilon_1)
                )
            )
            # print(f"v_u_X1: {v_u_X1}")
            phi_u_Z1 = v_u_X1 / s_l_X1 * DataProcessor.gamma(epsilon_sec, v_u_X1 / s_l_X1, s_l_Z1, s_l_X1)
            # print(f"phi_u_Z1: {phi_u_Z1}")

            # Error correction term
            lambda_EC = n_Z * fEC * DataProcessor.entropy(m_Z / n_Z)

            # Compute secret key length
            secret_key_length = (
                s_l_Z0
                + s_l_Z1 * (1 - DataProcessor.entropy(phi_u_Z1))
                - lambda_EC
                - 6 * np.log2(19 / epsilon_sec)
                - np.log2(2 / epsilon_cor)
            )
            skr = repetition_rate * secret_key_length / total_bit_sequence_length
            # print(f"skl: {skr}, secret_key_length: {secret_key_length}, total_bit_sequence_length: {total_bit_sequence_length}")
            # print(f"repetition_rate: {repetition_rate}")
            return skr
        
        initial_params = [self.config.mean_photon_nr, self.config.mean_photon_decoy, 1-self.config.p_decoy, self.config.p_z_alice]  # mus, mud, p_mus, p_Z
        # print(f"initial_params: {initial_params}")
        skr = calculate_skr(initial_params, total_bit_sequence_length)
        
        return skr
    
    def calc_SKR_Simon(self, n_Z_mus_in, n_Z_mud_in, n_X_mus_in, n_X_mud_in, m_Z_mus_in, m_Z_mud_in, m_X_mus_in, m_X_mud_in, total_symbols, factor):
        block_size = None  
        
        total_bit_sequence_length = total_symbols * factor  # Number of detections in key generating basis
        eta_bob = 4.5 / 100  # Transmittance in Bob’s side, including internal transmittance of optical components and detector efficiency
        y_0 = 1.7e-6  # Background rate, which includes the detector dark count and other background contributions such as the stray light from timing pulses
        channel_attenuation_Z = 26  # Channel transmittance [dB], can be derived from the loss coefficient alpha measured in dB/km and the length of the fiber l in km, alpha = 0.21
        channel_attenuation_X = 26
        q_Z = self.config.p_z_bob  # Bob chooses a basis Z and X with probabilities qz
       
        epsilon_sec = 1e-9  # 
        epsilon_cor = 1e-15  # Secret key are identical except of probability epsilon_cor
        repetition_rate = self.config.sampling_rate_FPGA / self.config.n_pulses  # Pulse (symbol) repetition rate
        print(f"self.config.sampling_rate_FPGA: {self.config.sampling_rate_FPGA}, self.config.n_pulses: {self.config.n_pulses}, repetition_rate: {repetition_rate}")	
        fEC = 1.19  # Error correction effciency
        epsilon_1 = epsilon_sec / 19

        def calculate_skr(params, total_bit_sequence_length):
            mus, mud, p_mus, p_Z = params
            p_mud = 1 - p_mus
            p_X = 1 - p_Z
            q_X = 1 - q_Z

            # Compute gain
            # eta_ch_Z = np.power(10, -channel_attenuation_Z / 10)
            # eta_ch_X = np.power(10, -channel_attenuation_X / 10)
            # gain_Z_mus = 1 - (1 - y_0) * np.exp(-1 * mus * eta_bob * eta_ch_Z)
            # gain_Z_mud = 1 - (1 - y_0) * np.exp(-1 * mud * eta_bob * eta_ch_Z)
            # gain_X_mus = 1 - (1 - y_0) * np.exp(-1 * mus * eta_bob * eta_ch_X)
            # gain_X_mud = 1 - (1 - y_0) * np.exp(-1 * mud * eta_bob * eta_ch_X)

            # Recalucalte total_bit_sequence_length to match desired nZ
            # if block_size is not None:
            #     total_bit_sequence_length = block_size / (
            #         p_Z * (p_mus * gain_Z_mus + p_mud * gain_Z_mud)
            #     )

            # Compute total detection events
            # n_Z_mus = p_Z * q_Z * p_mus * gain_Z_mus * total_bit_sequence_length
            # n_Z_mud = p_Z * q_Z * p_mud * gain_Z_mud * total_bit_sequence_length
            # n_X_mus = p_X * q_X * p_mus * gain_X_mus * total_bit_sequence_length
            # n_X_mud = p_X * q_X * p_mud * gain_X_mud * total_bit_sequence_length
            # n_Z = n_Z_mus + n_Z_mud
            # n_X = n_X_mus + n_X_mud
            print(f"factor: {factor}")
            n_Z_mus = n_Z_mus_in * factor
            n_Z_mud = n_Z_mud_in * factor
            n_X_mus = n_X_mus_in * factor
            n_X_mud = n_X_mud_in * factor
            n_Z = n_Z_mus + n_Z_mud
            n_X = n_X_mus + n_X_mud

            # Compute error
            # error_Z_mus = y_0 * (e_0 - e_detector_Z) + e_detector_Z * gain_Z_mus
            # error_Z_mud = y_0 * (e_0 - e_detector_Z) + e_detector_Z * gain_Z_mud
            # error_X_mus = y_0 * (e_0 - e_detector_X) + e_detector_X * gain_X_mus
            # error_X_mud = y_0 * (e_0 - e_detector_X) + e_detector_X * gain_X_mud

            # # Compute total error events
            # m_Z_mus = p_Z * q_Z * p_mus * error_Z_mus * total_bit_sequence_length
            # m_Z_mud = p_Z * q_Z * p_mud * error_Z_mud * total_bit_sequence_length
            # m_X_mus = p_X * p_X * p_mus * error_X_mus * total_bit_sequence_length
            # m_X_mud = p_X * p_X * p_mud * error_X_mud * total_bit_sequence_length

            m_Z_mus = m_Z_mus_in * factor
            m_Z_mud = m_Z_mud_in * factor
            m_X_mus = m_X_mus_in * factor
            m_X_mud = m_X_mud_in * factor
            m_Z = m_Z_mus + m_Z_mud
            m_X = m_X_mus + m_X_mud

            # Probabilites sending vaccum and single photon states
            tau_0 = p_mus * np.exp(-mus) + p_mud * np.exp(-mud)
            tau_1 = p_mus * mus * np.exp(-mus) + p_mud * mud * np.exp(-mud)

            # Compute finite-key security bounds
            s_l_Z0 = (
                tau_0
                / (mus - mud)
                * (
                    mus * DataProcessor.n_finite_key_corrected("-", mud, p_mud, n_Z_mud, n_Z, epsilon_1)
                    - mud * DataProcessor.n_finite_key_corrected("+", mus, p_mus, n_Z_mus, n_Z, epsilon_1)
                )
            )
            s_u_Z0 = 2 * (
                tau_0 * DataProcessor.n_finite_key_corrected("+", mus, p_mus, m_Z_mus, m_Z, epsilon_1)
                + np.sqrt(n_Z / 2 * np.log(1 / epsilon_1))
            )
            s_l_Z1 = (
                tau_1
                * mus
                / (mud * (mus - mud))
                * (
                    DataProcessor.n_finite_key_corrected("-", mud, p_mud, n_Z_mud, n_Z, epsilon_1)
                    - mud**2
                    / mus**2
                    * DataProcessor.n_finite_key_corrected("+", mus, p_mus, n_Z_mus, n_Z, epsilon_1)
                    - (mus**2 - mud**2) / (mus**2 * tau_0) * s_u_Z0
                )
            )
            s_u_X0 = 2 * (
                tau_0 * DataProcessor.n_finite_key_corrected("+", mud, p_mud, m_X_mud, m_X, epsilon_1)
                + np.sqrt(n_X / 2 * np.log(1 / epsilon_1))
            )
            s_l_X1 = (
                tau_1
                * mus
                / (mud * (mus - mud))
                * (
                    DataProcessor.n_finite_key_corrected("-", mud, p_mud, n_X_mud, n_X, epsilon_1)
                    - mud**2
                    / mus**2
                    * DataProcessor.n_finite_key_corrected("+", mus, p_mus, n_X_mus, n_X, epsilon_1)
                    - (mus**2 - mud**2) / (mus**2 * tau_0) * s_u_X0
                )
            )
            # print(f"s_l_Z0: {s_l_Z0}, s_u_Z0: {s_u_Z0}, s_l_Z1: {s_l_Z1}, s_u_X0: {s_u_X0}, s_l_X1: {s_l_X1}")
            v_u_X1 = (
                tau_1
                / (mus - mud)
                * (
                    DataProcessor.n_finite_key_corrected("+", mus, p_mus, m_X_mus, m_X, epsilon_1)
                    - DataProcessor.n_finite_key_corrected("-", mud, p_mud, m_X_mud, m_X, epsilon_1)
                )
            )
            # print(f"v_u_X1: {v_u_X1}")
            phi_u_Z1 = v_u_X1 / s_l_X1 * DataProcessor.gamma(epsilon_sec, v_u_X1 / s_l_X1, s_l_Z1, s_l_X1)
            # print(f"phi_u_Z1: {phi_u_Z1}")

            # Error correction term
            lambda_EC = n_Z * fEC * DataProcessor.entropy(m_Z / n_Z)

            # Compute secret key length
            secret_key_length = (
                s_l_Z0
                + s_l_Z1 * (1 - DataProcessor.entropy(phi_u_Z1))
                - lambda_EC
                - 6 * np.log2(19 / epsilon_sec)
                - np.log2(2 / epsilon_cor)
            )
            skr = repetition_rate * secret_key_length / total_bit_sequence_length
            print(f"skl: {skr}, secret_key_length: {secret_key_length}, total_bit_sequence_length: {total_bit_sequence_length}")
            print(f"repetition_rate: {repetition_rate}")
            return skr
        
        initial_params = [self.config.mean_photon_nr, self.config.mean_photon_decoy, 1-self.config.p_decoy, self.config.p_z_alice]  # mus, mud, p_mus, p_Z
        print(f"initial_params: {initial_params}")
        skr = calculate_skr(initial_params, total_bit_sequence_length)
        
        return skr
    
    def calc_SKR_old(self, len_wrong_x_dec, len_wrong_x_non_dec, len_wrong_z_dec, len_wrong_z_non_dec, 
                 len_Z_checked_dec, len_Z_checked_non_dec, X_P_calc_dec, X_P_calc_non_dec, total_symbols):
        block_size = None  
        #factor to get up to a billion symbols
        # if len_Z_checked_non_dec != 0:
        #     factor = 1e9 / len_Z_checked_non_dec
        # else:
        factor = 1e4
        print(f"factor: {factor}")
        total_bit_sequence_length = total_symbols * factor  # Number of detections in key generating basis
        eta_bob = 4.5 / 100  # Transmittance in Bob’s side, including internal transmittance of optical components and detector efficiency
        y_0 = 1.7e-6  # Background rate, which includes the detector dark count and other background contributions such as the stray light from timing pulses
        channel_attenuation_Z = 26  # Channel transmittance [dB], can be derived from the loss coefficient alpha measured in dB/km and the length of the fiber l in km, alpha = 0.21
        channel_attenuation_X = 26
        q_Z = self.config.p_z_bob  # Bob chooses a basis Z and X with probabilities qz
       
        epsilon_sec = 1e-9  # 
        epsilon_cor = 1e-15  # Secret key are identical except of probability epsilon_cor
        repetition_rate = self.config.sampling_rate_FPGA / self.config.n_pulses  # Pulse (symbol) repetition rate
        fEC = 1.19  # Error correction effciency
        epsilon_1 = epsilon_sec / 19

        def calculate_skr(params, total_bit_sequence_length):
            mus, mud, p_mus, p_Z = params
            p_mud = 1 - p_mus
            p_X = 1 - p_Z
            q_X = 1 - q_Z

            # Compute gain
            # eta_ch_Z = np.power(10, -channel_attenuation_Z / 10)
            # eta_ch_X = np.power(10, -channel_attenuation_X / 10)
            # gain_Z_mus = 1 - (1 - y_0) * np.exp(-1 * mus * eta_bob * eta_ch_Z)
            # gain_Z_mud = 1 - (1 - y_0) * np.exp(-1 * mud * eta_bob * eta_ch_Z)
            # gain_X_mus = 1 - (1 - y_0) * np.exp(-1 * mus * eta_bob * eta_ch_X)
            # gain_X_mud = 1 - (1 - y_0) * np.exp(-1 * mud * eta_bob * eta_ch_X)

            # Recalucalte total_bit_sequence_length to match desired nZ
            # if block_size is not None:
            #     total_bit_sequence_length = block_size / (
            #         p_Z * (p_mus * gain_Z_mus + p_mud * gain_Z_mud)
            #     )

            # Compute total detection events
            # n_Z_mus = p_Z * q_Z * p_mus * gain_Z_mus * total_bit_sequence_length
            # n_Z_mud = p_Z * q_Z * p_mud * gain_Z_mud * total_bit_sequence_length
            # n_X_mus = p_X * q_X * p_mus * gain_X_mus * total_bit_sequence_length
            # n_X_mud = p_X * q_X * p_mud * gain_X_mud * total_bit_sequence_length
            # n_Z = n_Z_mus + n_Z_mud
            # n_X = n_X_mus + n_X_mud

            n_Z_mus = len_Z_checked_non_dec * factor
            n_Z_mud = len_Z_checked_dec * factor
            n_X_mus = X_P_calc_non_dec * factor
            n_X_mud = X_P_calc_dec * factor
            n_Z = n_Z_mus + n_Z_mud
            n_X = n_X_mus + n_X_mud

            # Compute error
            # error_Z_mus = y_0 * (e_0 - e_detector_Z) + e_detector_Z * gain_Z_mus
            # error_Z_mud = y_0 * (e_0 - e_detector_Z) + e_detector_Z * gain_Z_mud
            # error_X_mus = y_0 * (e_0 - e_detector_X) + e_detector_X * gain_X_mus
            # error_X_mud = y_0 * (e_0 - e_detector_X) + e_detector_X * gain_X_mud

            # # Compute total error events
            # m_Z_mus = p_Z * q_Z * p_mus * error_Z_mus * total_bit_sequence_length
            # m_Z_mud = p_Z * q_Z * p_mud * error_Z_mud * total_bit_sequence_length
            # m_X_mus = p_X * p_X * p_mus * error_X_mus * total_bit_sequence_length
            # m_X_mud = p_X * p_X * p_mud * error_X_mud * total_bit_sequence_length

            m_Z_mus = len_wrong_z_non_dec * factor
            m_Z_mud = len_wrong_z_dec * factor
            m_X_mus = len_wrong_x_non_dec * factor
            m_X_mud = len_wrong_x_dec * factor
            m_Z = m_Z_mus + m_Z_mud
            m_X = m_X_mus + m_X_mud

            # Probabilites sending vaccum and single photon states
            tau_0 = p_mus * np.exp(-mus) + p_mud * np.exp(-mud)
            tau_1 = p_mus * mus * np.exp(-mus) + p_mud * mud * np.exp(-mud)

            # Compute finite-key security bounds
            s_l_Z0 = (
                tau_0
                / (mus - mud)
                * (
                    mus * DataProcessor.n_finite_key_corrected("-", mud, p_mud, n_Z_mud, n_Z, epsilon_1)
                    - mud * DataProcessor.n_finite_key_corrected("+", mus, p_mus, n_Z_mus, n_Z, epsilon_1)
                )
            )
            s_u_Z0 = 2 * (
                tau_0 * DataProcessor.n_finite_key_corrected("+", mus, p_mus, m_Z_mus, m_Z, epsilon_1)
                + np.sqrt(n_Z / 2 * np.log(1 / epsilon_1))
            )
            s_l_Z1 = (
                tau_1
                * mus
                / (mud * (mus - mud))
                * (
                    DataProcessor.n_finite_key_corrected("-", mud, p_mud, n_Z_mud, n_Z, epsilon_1)
                    - mud**2
                    / mus**2
                    * DataProcessor.n_finite_key_corrected("+", mus, p_mus, n_Z_mus, n_Z, epsilon_1)
                    - (mus**2 - mud**2) / (mus**2 * tau_0) * s_u_Z0
                )
            )
            s_u_X0 = 2 * (
                tau_0 * DataProcessor.n_finite_key_corrected("+", mud, p_mud, m_X_mud, m_X, epsilon_1)
                + np.sqrt(n_X / 2 * np.log(1 / epsilon_1))
            )
            s_l_X1 = (
                tau_1
                * mus
                / (mud * (mus - mud))
                * (
                    DataProcessor.n_finite_key_corrected("-", mud, p_mud, n_X_mud, n_X, epsilon_1)
                    - mud**2
                    / mus**2
                    * DataProcessor.n_finite_key_corrected("+", mus, p_mus, n_X_mus, n_X, epsilon_1)
                    - (mus**2 - mud**2) / (mus**2 * tau_0) * s_u_X0
                )
            )
            v_u_X1 = (
                tau_1
                / (mus - mud)
                * (
                    DataProcessor.n_finite_key_corrected("+", mus, p_mus, m_X_mus, m_X, epsilon_1)
                    - DataProcessor.n_finite_key_corrected("-", mud, p_mud, m_X_mud, m_X, epsilon_1)
                )
            )
            phi_u_Z1 = v_u_X1 / s_l_X1 * DataProcessor.gamma(epsilon_sec, v_u_X1 / s_l_X1, s_l_Z1, s_l_X1)

            # Error correction term
            lambda_EC = n_Z * fEC * DataProcessor.entropy(m_Z / n_Z)

            # Compute secret key length
            secret_key_length = (
                s_l_Z0
                + s_l_Z1 * (1 - DataProcessor.entropy(phi_u_Z1))
                - lambda_EC
                - 6 * np.log2(19 / epsilon_sec)
                - np.log2(2 / epsilon_cor)
            )
            skr = repetition_rate * secret_key_length / total_bit_sequence_length
            return skr
        
        initial_params = [self.config.mean_photon_nr, self.config.mean_photon_decoy, 1-self.config.p_decoy, self.config.p_z_alice]  # mus, mud, p_mus, p_Z
        print(f"initial_params: {initial_params}")
        skr = calculate_skr(initial_params, total_bit_sequence_length)
        
        return skr
        