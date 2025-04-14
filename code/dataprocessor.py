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
        return local_histogram_counts_x, local_histogram_counts_z

    @staticmethod
    def plot_histogram_batch(bins_per_symbol, time_one_symbol, histogram_counts_x, histogram_counts_z, lookup_arr, total_symbols, start_symbol=3, end_symbol=10, name=' '):
        assert 0 <= start_symbol <= end_symbol <= 64
        amount_of_symbols_incl_start_and_end = end_symbol - start_symbol + 1
        bins = np.linspace(0, amount_of_symbols_incl_start_and_end * time_one_symbol, bins_per_symbol * amount_of_symbols_incl_start_and_end + 1)    

        plt.figure(figsize=(10, 6))
        # Plot as bar chart; you can also use plt.hist with precomputed counts.
        width = (bins[1] - bins[0])
        plt.bar(bins[:-1], histogram_counts_x[start_symbol * bins_per_symbol :(end_symbol + 1) * bins_per_symbol], width=width, alpha=0.6, label='X basis', color='blue')
        plt.bar(bins[:-1], histogram_counts_z[start_symbol * bins_per_symbol :(end_symbol + 1) * bins_per_symbol], width=width, alpha=0.6, label='Z basis', color='red')
        
        for i in range(amount_of_symbols_incl_start_and_end):
            plt.axvline(x=i * time_one_symbol, color='grey', linestyle='--', linewidth=1)

            # Place the symbol halfway between this line and the next
            if i < amount_of_symbols_incl_start_and_end:
                x_mid = i * time_one_symbol + time_one_symbol / 2
                symbol = lookup_arr[start_symbol + i]
                y_max = max(max(histogram_counts_x), max(histogram_counts_z))
                basis = symbol[0]  # assuming symbol is like 'X0' or 'Z1'
                color = 'green' if basis == 'X' else 'purple'

                plt.text(x_mid, y_max * 0.9, symbol, ha='center', va='bottom', fontsize=14, color=color, fontweight='bold')

        plt.xlabel("Time ")
        plt.ylabel("Cumulative Counts")
        plt.title(f"Cumulative Histogram for {lookup_arr[start_symbol:end_symbol + 1]} for {total_symbols} {name} symbols")
        plt.legend()
        plt.tight_layout()
        Saver.save_plot(f"hist_symbols_{start_symbol}_to_{end_symbol}")

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
        
        pair_indices_dict = {}
        for pair in unique_pairs:
            pair_tuple = tuple(pair)
            # Find indices where this exact pair occurs (vectorized)
            indices = np.nonzero((pairs[:, 0] == pair_tuple[0]) & (pairs[:, 1] == pair_tuple[1]))[0]
            pair_indices_dict[pair_tuple] = indices
        return pair_indices_dict


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

        local_histogram_counts_x = np.zeros(amount_bins_hist, dtype=int)
        local_histogram_counts_z = np.zeros(amount_bins_hist, dtype=int)
        
        # Define bin edges for one symbol's time window.
        bins_arr = np.linspace(0, time_one_symbol, bins_per_symbol + 1)
        
        # Get dictionary mapping each adjacent pair (for one chain) to the positions where they occur.
        pair_indices_dict = DataProcessor.get_all_pair_indices(lookup_arr)
        
        def process(idx_left, idx_right, index_where_photons_det, time_photons_det, local_histogram_counts):
            if idx_left > 0 and idx_right < length_of_chain - 1:
                # looked at index is second symbol:
                pair_key = (raw_symbol_lookup[(basis[idx_left], value[idx_left], decoy[idx_left])], 
                            raw_symbol_lookup[(basis[idx_right], value[idx_right], decoy[idx_right])])
                # print(f"pair_key:{pair_key}")
                # print(f"pair_indices_dict:{pair_indices_dict}")
                #if pair_key in pair_indices_dict:  
                position_brujin_left = pair_indices_dict[pair_key].item()
                
                # Process the first symbol of the pair:
                if idx_left in index_where_photons_det:
                    inds_first = np.where(index_where_photons_det == idx_left)[0]
                    valid_times_first = time_photons_det[inds_first]
                    valid_times_first = valid_times_first[~np.isnan(valid_times_first)]
                    bin_indices_first = np.digitize(valid_times_first, bins_arr) - 1
                    # Update histogram: position = pos (for first symbol)
                    for b in bin_indices_first:
                        if 0 <= b < bins_per_symbol:
                            overall_bin = position_brujin_left * bins_per_symbol + b
                            local_histogram_counts[overall_bin] += 1

                # Process the second symbol of the pair:
                if idx_right in index_where_photons_det:
                    inds_first = np.where(index_where_photons_det == idx_right)[0]
                    valid_times_first = time_photons_det[inds_first]
                    valid_times_first = valid_times_first[~np.isnan(valid_times_first)]
                    bin_indices_first = np.digitize(valid_times_first, bins_arr) - 1
                    # Update histogram: position = pos (for first symbol)
                    for b in bin_indices_first:
                        if 0 <= b < bins_per_symbol:
                            overall_bin = (position_brujin_left + 1) * bins_per_symbol + b
                            local_histogram_counts[overall_bin] += 1

        # Process each chain (repetition)
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

        
        return local_histogram_counts_z, local_histogram_counts_x
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
    
    def calc_SKR(self, len_wrong_x_dec, len_wrong_x_non_dec, len_wrong_z_dec, len_wrong_z_non_dec, 
                 len_Z_checked_dec, len_Z_checked_non_dec, X_P_calc_non_dec, X_P_calc_dec, total_symbols):
        block_size = None  
        #factor to get up to a billion symbols
        if len_Z_checked_non_dec != 0:
            factor = 1e9 / len_Z_checked_non_dec
        else:
            factor = 1
        total_bit_sequence_length = total_symbols  # Number of detections in key generating basis
        eta_bob = 4.5 / 100  # Transmittance in Bob’s side, including internal transmittance of optical components and detector efficiency
        y_0 = 1.7e-6  # Background rate, which includes the detector dark count and other background contributions such as the stray light from timing pulses
        channel_attenuation_Z = 26  # Channel transmittance [dB], can be derived from the loss coefficient alpha measured in dB/km and the length of the fiber l in km, alpha = 0.21
        channel_attenuation_X = 26
        q_Z = self.config.p_z_bob  # Bob chooses a basis Z and X with probabilities qz
        e_detector_Z = (
            3.3 / 100
        )  # e_detector, characterizes the alignment and stability, characterizes the alignment and stability, assume constant
        e_detector_X = 3.3 / 100
        e_0 = 1 / 2  # error rate of the background, will assume that the background is random
        epsilon_sec = 1e-10  # It's called a "epsilon_sec-secret"
        epsilon_cor = 1e-15  # Secret key are identical except of probability epsilon_cor
        repetition_rate = 1e9  # Pulse (symbol) repetition rate
        fEC = 1.22  # Error correction effciency
        epsilon_1 = epsilon_sec / 19

        def calculate_skr(params, total_bit_sequence_length):
            mus, mud, p_mus, p_Z = params
            p_mud = 1 - p_mus
            p_X = 1 - p_Z
            q_X = 1 - q_Z

            # Compute gain
            eta_ch_Z = np.power(10, -channel_attenuation_Z / 10)
            eta_ch_X = np.power(10, -channel_attenuation_X / 10)
            gain_Z_mus = 1 - (1 - y_0) * np.exp(-1 * mus * eta_bob * eta_ch_Z)
            gain_Z_mud = 1 - (1 - y_0) * np.exp(-1 * mud * eta_bob * eta_ch_Z)
            gain_X_mus = 1 - (1 - y_0) * np.exp(-1 * mus * eta_bob * eta_ch_X)
            gain_X_mud = 1 - (1 - y_0) * np.exp(-1 * mud * eta_bob * eta_ch_X)

            # Recalucalte total_bit_sequence_length to match desired nZ
            if block_size is not None:
                total_bit_sequence_length = block_size / (
                    p_Z * (p_mus * gain_Z_mus + p_mud * gain_Z_mud)
                )

            # Compute total detection events
            # n_Z_mus = p_Z * q_Z * p_mus * gain_Z_mus * total_bit_sequence_length
            # n_Z_mud = p_Z * q_Z * p_mud * gain_Z_mud * total_bit_sequence_length
            # n_X_mus = p_X * q_X * p_mus * gain_X_mus * total_bit_sequence_length
            # n_X_mud = p_X * q_X * p_mud * gain_X_mud * total_bit_sequence_length
            # n_Z = n_Z_mus + n_Z_mud
            # n_X = n_X_mus + n_X_mud

            n_Z_mus = len_Z_checked_non_dec * factor
            print(f"n_Z_mus: {n_Z_mus}")
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

            error_Z_mus = len_wrong_z_non_dec * factor
            error_Z_mud = len_wrong_z_dec * factor
            error_X_mus = len_wrong_x_non_dec * factor
            error_X_mud = len_wrong_x_dec * factor

            # # Compute total error events
            m_Z_mus = p_Z * q_Z * p_mus * error_Z_mus * total_bit_sequence_length
            m_Z_mud = p_Z * q_Z * p_mud * error_Z_mud * total_bit_sequence_length
            m_X_mus = p_X * p_X * p_mus * error_X_mus * total_bit_sequence_length
            m_X_mud = p_X * p_X * p_mud * error_X_mud * total_bit_sequence_length
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
        skr = calculate_skr(initial_params, total_bit_sequence_length)
        
        return skr
        