from matplotlib.pylab import f, norm
from matplotlib import pyplot as plt
import numpy as np
from scipy.fftpack import fft, ifft, fftfreq
from scipy.interpolate import interp1d
from scipy.special import factorial
from scipy import constants
import time
import gc
import psutil
import os

from saver import Saver



class SimulationHelper:
    def __init__(self, config):
        self.config = config 
        self.config

    # ========== Main Helper Functions for generate alice choice ==========
    
    def count_alice_choices(self, basis, value, decoy):
        """Count Alice's choices and return them as separate values."""
        Z1_sent_norm = np.sum((basis == 1) & (value == 1) & (decoy == 0))  # Z1_sent_norm
        Z1_sent_dec = np.sum((basis == 1) & (value == 1) & (decoy == 1))  # Z1_sent_dec
        Z0_sent_norm = np.sum((basis == 1) & (value == 0) & (decoy == 0))  # Z0_sent_norm
        Z0_sent_dec = np.sum((basis == 1) & (value == 0) & (decoy == 1))  # Z0_sent_dec
        XP_sent_norm = np.sum((basis == 0) & (decoy == 0))                # XP_sent_norm
        XP_sent_dec = np.sum((basis == 0) & (decoy == 1))                # XP_sent_dec

        return Z1_sent_norm, Z1_sent_dec, Z0_sent_norm, Z0_sent_dec, XP_sent_norm, XP_sent_dec
    
    def create_all_symbol_combinations_for_hist(self):
        def de_bruijn(k, n):
            """
            Generate a de Bruijn sequence for alphabet size k and subsequences of length n.
            """
            alphabet = list(range(k))
            a = [0] * k * n
            sequence = []

            def db(t, p):
                if t > n:
                    if n % p == 0:
                        sequence.extend(a[1:p + 1])
                else:
                    a[t] = a[t - p]
                    db(t + 1, p)
                    for j in range(a[t - p] + 1, k):
                        a[t] = j
                        db(t + 1, t)

            db(1, 1)
            return sequence

        # Generate de Bruijn sequence for 8 symbols (order 2)

        # Map numbers to your symbols
        symbols = ['Z0', 'Z1', 'X+', 'Z0*', 'Z1*', 'X+*']
        seq = de_bruijn(len(symbols), 2)
        symbol_sequence = [symbols[i] for i in seq]

        # Assuming symbols dictionary as before
        symbols_dict = {
            'Z0':  (1, 1, 0),
            'Z1':  (1, 0, 0),
            'X+':  (0, -1, 0),
            'Z0*': (1, 1, 1),
            'Z1*': (1, 0, 1),
            'X+*': (0, -1, 1),
        }

        basis_array = np.empty(len(symbols_dict)**2 + 1, dtype=int)
        value_array = np.empty(len(symbols_dict)**2 + 1, dtype=int)
        decoy_array = np.empty(len(symbols_dict)**2 + 1, dtype=int)
        lookup_array = []  # Here is your lookup array!

        # Flatten basis, value, decoy
        for idx, sym in enumerate(symbol_sequence):
            b, v, d = symbols_dict[sym]
            basis_array[idx] = b
            value_array[idx] = v
            decoy_array[idx] = d
            lookup_array.append(sym)

        basis_array[-1] = basis_array[0]
        value_array[-1] = value_array[0]
        decoy_array[-1] = decoy_array[0]
        lookup_array.append(lookup_array[0])

        print("Basis Array (sample):", basis_array[:10])
        print("Value Array (sample):", value_array[:10])
        print("Decoy Array (sample):", decoy_array[:10])

        return basis_array, value_array, decoy_array, lookup_array
        
    # ========== Main Helper Functions for signal generation ==========

    def get_pulse_height(self, basis, decoy):
        """
        Determine the pulse height based on the basis and decoy state.
        Args:
            basis (int): 0 for X-basis (superposition), 1 for Z-basis (computational).
            decoy (int): 0 for standard pulse, 1 for decoy pulse.
        """
        return np.where(decoy == 0,
                        np.where(basis == 0, self.config.voltage_sup, self.config.voltage),
                        np.where(basis == 0, self.config.voltage_decoy_sup, self.config.voltage_decoy))

    def get_jitter(self, name_jitter, size_jitter=1):
        probabilities, jitter_values = self.config.data.get_probabilities(x_data=None, name='probabilities' + name_jitter)
        jitter_shifts = self.config.rng.choice(jitter_values, size = size_jitter, p=probabilities) # für jeden Querstrich kann man jitter haben
        return jitter_shifts

    def encode_pulse(self, value):
        """Return a binary pattern for a square pulse based on the given value"""
        pattern = np.zeros((self.config.n_samples, self.config.n_pulses), dtype=int) #hatte eig len(value)
        pattern[value == 1, 0] = 1
        pattern[value == 0, self.config.n_pulses // 2] = 1
        pattern[value == -1, 0] = 1
        pattern[value == -1, self.config.n_pulses // 2] = 1
        return pattern

    def generate_square_pulse(self, pulse_height, pulse_duration, sampling_rate_fft, pattern):
        """Generate a square pulse signal for a given height and pattern."""
        # make len(t) divisible by n_pulses
        samples_per_pulse = int(pulse_duration * sampling_rate_fft)
        total_samples = self.config.n_pulses * samples_per_pulse
        # t = np.linspace(0, self.config.n_pulses * pulse_duration, total_samples, endpoint=False)
        # t = np.arange(0, total_samples, 1 / sampling_rate_fft)
        dt = 1 / sampling_rate_fft
        t = np.linspace(0, total_samples * dt, total_samples, endpoint=False)
        repeating_square_pulses = np.full((len(pulse_height), len(t)), self.config.non_signal_voltage, dtype=np.float64)
        one_pulse = len(t) // self.config.n_pulses
        indices = np.arange(len(t))
        for i, p in enumerate(pattern):
            for j, bit in enumerate(p):
                pulse_indices = (indices // one_pulse) == j
                if bit == 1:
                    repeating_square_pulses[i, pulse_indices] = pulse_height[i] 
        return t, repeating_square_pulses
    
    def generate_encoded_pulse(self, pulse_height, pulse_duration, value, sampling_rate_fft):
        return self.generate_square_pulse(pulse_height, pulse_duration, sampling_rate_fft, pattern = self.encode_pulse(value))

    def apply_bandwidth_filter(self, signal, sampling_rate_fft):
        """Apply a frequency-domain filter to a signal."""
        S_fourier = fft(signal)
        frequencies = fftfreq(len(signal), d=1 / sampling_rate_fft)

        freq_x = [0, self.config.bandwidth * 0.8, self.config.bandwidth, self.config.bandwidth * 1.2, sampling_rate_fft / 2]
        freq_y = [1, 1, 0.7, 0.01, 0.001]  # Smooth drop-off

        np.multiply(S_fourier, np.interp(np.abs(frequencies), freq_x, freq_y), out=S_fourier)

        return np.real(ifft(S_fourier))

    def apply_jitter_to_pulse(self, t, signals, jitter_shifts):
        dt = t[1] - t[0]  # sampling interval in seconds
        # Convert jitter from seconds to integer index shift
        index_shift_per_symbol = (jitter_shifts / dt).astype(int)
        # One pulse = how many samples?
        index_one_signal = len(t) // self.config.n_pulses
        # Where transitions would normally occur (e.g., 128, 256, 384...)
        transition_indices = np.arange(index_one_signal, len(signals), index_one_signal)
        # Sanity check: align lengths
        min_len = min(len(transition_indices), len(index_shift_per_symbol))
        transition_indices = transition_indices[:min_len]
        index_shift_per_symbol = index_shift_per_symbol[:min_len]
        # Apply jitter to transitions
        new_transition_indices = transition_indices + index_shift_per_symbol
        for i, old_idx in enumerate(transition_indices):
            new_idx = new_transition_indices[i]
            # Make sure new_idx stays within bounds
            new_idx = np.clip(new_idx, 0, len(signals) - 1)
            if old_idx < new_idx:
                signals[old_idx:new_idx] = signals[old_idx - 1]
            elif old_idx > new_idx:
                signals[new_idx:old_idx] = signals[old_idx + 1]
        return signals
    
    # ========= Delay Line Interferometer Helpers ==========

    def DLI(self, P_in, dt, tau, delta_L, f_0,  n_eff, splitting_ratio = 0.5, plot = False):
        """
        Simulates the behavior of a delay line interferometer (DLI), which is used to 
        measure phase differences between two optical signals.#
            P_in (array-like): Input optical power as a function of time.
            dt (float): Sampling time step for calculation, not (!) signal sampling rate (seconds).
            tau (float): Time delay introduced by the interferometer (seconds).
            delta_L (float): Path length difference between the two arms of the interferometer (meters).
            f0 (float): Optical carrier frequency of the input signal (Hz).
            n_eff (float): Effective refractive index of the waveguide.
            splitting_ratio (float, optional): Splitting ratio of the coupler. Defaults to 0.5 
                    (ideal 50/50 coupler).
            tuple: A tuple containing:
                - np.ndarray: Output power at the first port of the interferometer.
                - np.ndarray: Output power at the second port of the interferometer.
                - np.ndarray: Time array corresponding to the input signal.
        """
        # Time array
        t = np.arange(len(P_in)) * dt
        # print(f"shape t: {t.shape}")
        
        # Input optical field (assuming carrier frequency)
        E0 = np.sqrt(P_in/2)
        E_in = E0 * np.exp(1j * 2 * np.pi * f_0 * t)
        # print(f"shape E_in: {E_in.shape}")
        # Interpolate for delayed version
        interp_real = interp1d(t, np.real(E_in),kind='cubic', fill_value="extrapolate")
        interp_imag = interp1d(t, np.imag(E_in),kind='cubic', fill_value="extrapolate")

        E_in_delayed = interp_real(t - tau) + 1j * interp_imag(t - tau)
        
        # #remove 
        remove_samples = int(0.6e-9 / dt)
        # E_in = E_in[remove_samples:]
        # E_in_delayed = E_in_delayed[remove_samples:]
        # t = t[remove_samples:]

        #Plot the real and imaginary parts of the interpolated signal
        if plot == True:
            plt.figure(figsize=(10, 4))
            plt.plot(t * 1e9, np.real(E_in), label="Real Part")
            plt.plot(t * 1e9, np.imag(E_in), label="Imaginary Part")
            plt.plot(t * 1e9, np.real(E_in_delayed), label="Real Part (Delayed)", linestyle='--')
            plt.plot(t * 1e9, np.imag(E_in_delayed), label="Imaginary Part (Delayed)", linestyle='--')
            plt.xlim(2,6)
            plt.ylim(-0.0001,0.0001)
            plt.legend()
            plt.show()
            
        # Phase shift from path length difference
        phi = 2 * np.pi * f_0 * n_eff * delta_L / constants.c
        E_in_delayed *= np.exp(1j * phi)

        # Interpolate for E_in and E_in_delayed to ensure alignment
        # interp_E_in_real = interp1d(t, np.real(E_in), kind='cubic', fill_value="extrapolate")
        # interp_E_in_imag = interp1d(t, np.imag(E_in), kind='cubic', fill_value="extrapolate")
        # interp_E_in_delayed_real = interp1d(t, np.real(E_in_delayed), kind='cubic', fill_value="extrapolate")
        # interp_E_in_delayed_imag = interp1d(t, np.imag(E_in_delayed), kind='cubic', fill_value="extrapolate")

        # Recalculate aligned signals
        # E_in_aligned = interp_E_in_real(t) + 1j * interp_E_in_imag(t)
        # E_in_delayed_aligned = interp_E_in_delayed_real(t) + 1j * interp_E_in_delayed_imag(t)
      

        # Calculate outputs
        E_out1 = np.sqrt(splitting_ratio) * (E_in - E_in_delayed)
        E_out2 = np.sqrt(1 - splitting_ratio) * 1j * (E_in + E_in_delayed)

        # remove samples 
        E_out1 = E_out1[remove_samples:]
        E_out2 = E_out2[remove_samples:]
        t = t[remove_samples:]

        return np.abs(E_out1)**2, np.abs(E_out2)**2, t

    # ========== Detector Helper ==========

    def choose_photons(self, t, power_dampened, peak_wavelength, start_time, fixed_nr_photons=None):
        """Calculate and choose photons based on the transmission and jitter."""
        # Calculate the mean photon number
        energy_per_pulse = np.trapz(power_dampened, t, axis=1)
        # delete power_dampened
        
        energy_one_photon = constants.h * constants.c / peak_wavelength
        calc_mean_photon_nr_detector = energy_per_pulse / energy_one_photon
        # Use Poisson distribution to get the number of photons
        nr_photons = fixed_nr_photons if fixed_nr_photons is not None else self.poisson_distr(calc_mean_photon_nr_detector)
        all_time_max_nr_photons = max(nr_photons)
            
        # Find the indices where the number of photons is greater than 0
        non_zero_photons = nr_photons > 0
        index_where_photons = np.where(non_zero_photons)[0]
        nr_iterations_where_photons = len(index_where_photons)

        # Pre-allocate arrays for the photon properties        
        energy_per_photon = np.full((nr_iterations_where_photons, all_time_max_nr_photons), np.nan)
        wavelength_photons = np.full_like(energy_per_photon, np.nan)
        time_photons = np.full_like(energy_per_photon, np.nan)
        nr_photons = nr_photons[non_zero_photons] #delete rows where number of photons is 0

        Saver.memory_usage("in choose photon after pre-allocate arrays")
        # Generate the photon properties for each sample with non-zero photons
        for i, idx in enumerate(index_where_photons):
            photon_count = nr_photons[i]
            energy_per_photon[i, :photon_count] = energy_one_photon[idx]
            wavelength_photons[i, :photon_count] = peak_wavelength[idx]

            norm_prob_time_symbol = np.divide(power_dampened[idx], power_dampened[idx].sum())
            # print(f"norm_prob_time_symbol: {norm_prob_time_symbol.sum()}")
            # plt.plot(t, norm_prob_time_symbol, label=f"Symbol {idx}")
            # plt.show()

            time_photons[i, :photon_count] = self.config.rng.choice(t, size=photon_count, p=norm_prob_time_symbol) #t ist konstant
        
        del power_dampened
        gc.collect()

        return wavelength_photons, time_photons, nr_photons, index_where_photons, all_time_max_nr_photons, calc_mean_photon_nr_detector

    def filter_photons_detection_time(self, time_photons_det, wavelength_photons_det):
        """
            Filter photon detections that are too close in time across symbols by first
            "unfolding" the 2D array. For each row (symbol), add an offset equal to
            row_index * (n_pulses * pulse_length)
            to each non-NaN detection time. This creates a continuous time axis across rows.
            
            Then, sort the valid (non-NaN) adjusted times and reject detections that are
            closer than detection_time to the previous accepted detection.
            
            The detections that fail the test are marked as NaN in both time_photons_det
            and wavelength_photons_det.
            
            Parameters:
            time_photons_det       : 2D numpy array of detection times (rows = symbols)
            wavelength_photons_det : 2D numpy array of corresponding wavelengths
            pulse_length           : Duration of one pulse
            
            Returns:
            Updated (filtered) time_photons_det and wavelength_photons_det arrays.
            """
        pulse_length = 1 / self.config.sampling_rate_FPGA

        num_rows, num_cols = time_photons_det.shape

        # Create an offset array for each row: shape (num_rows, 1)
        row_offsets = np.arange(num_rows).reshape(num_rows, 1) * (self.config.n_pulses * pulse_length)
        
        # Create an adjusted time array: add the offset to each valid (non-NaN) time
        adjusted_time = np.where(~np.isnan(time_photons_det), time_photons_det + row_offsets, np.nan)
        
        # Get indices of valid (non-NaN) detections
        valid_mask = ~np.isnan(adjusted_time)
        valid_times = adjusted_time[valid_mask]           # 1D array of valid, adjusted times
        valid_indices = np.argwhere(valid_mask)             # Each row is [row, col] of a valid detection

        # Sort the valid detections by adjusted time
        sort_order = np.argsort(valid_times)
        valid_times_sorted = valid_times[sort_order]
        valid_indices_sorted = valid_indices[sort_order]    # Sorted list of [row, col] indices

        # Mark which detections are accepted (initialize all as accepted)
        accepted = np.ones(valid_times_sorted.shape, dtype=bool)
        
        # Process the sorted valid times: if the time difference from the last accepted
        # detection is less than detection_time, mark this detection as rejected.
        last_accepted_time = -np.inf
        for idx, times in enumerate(valid_times_sorted):
            if times - last_accepted_time < self.config.detection_time:
                accepted[idx] = False
            else:
                last_accepted_time = times

        # For detections that were not accepted, set the original arrays to NaN.
        # valid_indices_sorted[~accepted] is a 2D array of [row, col] indices.
        for row, col in valid_indices_sorted[~accepted]:
            time_photons_det[row, col] = np.nan
            wavelength_photons_det[row, col] = np.nan

        return time_photons_det, wavelength_photons_det
    
    def add_detection_jitter(self, t, time_photons_det):
        jitter_shifts = self.get_jitter('detector', size_jitter = time_photons_det.shape)

        #Apply jitter to non-NaN values, keeping NaNs unchanged
        time_photons_det[~np.isnan(time_photons_det)] += jitter_shifts[~np.isnan(time_photons_det)]
        #Clip the values to the time range so jitter doesn't put values out of bounds
        time_photons_det = np.clip(time_photons_det, 0, t[-1])

        return time_photons_det
    
    def darkcount(self, index_where_photons_det, time_photons_det):
        """Calculate the number of dark count photons detected."""
        pulse_duration = 1 / self.config.sampling_rate_FPGA
        symbol_duration = pulse_duration * self.config.n_pulses
        num_dark_counts = self.config.rng.poisson(self.config.dark_count_frequency / 2 * symbol_duration, size=self.config.n_samples)  # / 2 weil wird ja bei 2 detektoren
        dark_count_times = [self.config.rng.uniform(0, symbol_duration, count) if count > 0 else np.empty(0) for count in num_dark_counts]
        
        num_columns = time_photons_det.shape[1]
        new_rows = []
        new_indices = []

        for i, count in enumerate(num_dark_counts):
            if count > 0:
                if count > num_columns:
                    count = num_columns
                    dark_count_times[i] = dark_count_times[i][:count]
                if i in index_where_photons_det:
                    idx_short_i = np.where(index_where_photons_det == i)[0][0]
                    time_photons_det[idx_short_i, -count:] = dark_count_times[i]
                else:
                    new_row = np.full((1, num_columns), np.nan)
                    new_row[0, -count:] = dark_count_times[i]
                    new_rows.append(new_row)
                    new_indices.append(i)
    
        # Add all new rows to time_photons_det at once
        if new_rows:
            new_rows_array = np.vstack(new_rows)  # Stack all new rows into a single array
            time_photons_det = np.vstack((time_photons_det, new_rows_array))

        # Add all new indices to index_where_photons_det at once
        if new_indices:
            index_where_photons_det = np.concatenate((index_where_photons_det, new_indices))

        # Sort index_where_photons_det and rearrange time_photons_det accordingly
        sort_order = np.argsort(index_where_photons_det)
        index_where_photons_det = index_where_photons_det[sort_order]
        time_photons_det = time_photons_det[sort_order]

        print(f"total darkcounts: {np.sum(num_dark_counts)}")
        return dark_count_times, num_dark_counts

    # ========== Classificator Helper ========== 

    def classificator_sift_z_vacuum(self, basis, detected_indices_z, index_where_photons_det_z):
        # nur z basis sendung
        mask_z_short = basis[index_where_photons_det_z] == 1 # alle indices bei denen was detected und das gesendete in X basis
        detected_indices_z_det_z_basis = detected_indices_z[mask_z_short]
        early_indices_short = np.where(np.sum(detected_indices_z == 0, axis=1) >= 1)[0]
        late_indices_short = np.where(np.sum(detected_indices_z == 1, axis=1) >= 1)[0]
        # assign random value to double counts
        non_negative_counts = np.sum(detected_indices_z_det_z_basis != -1, axis=1)
        # Identify rows with more than one non-`-1` value
        rows_with_multiple_detections = np.where(non_negative_counts > 1)[0]

        # change detected_indices_z_det_z_basis to -1 except one for rows with multiple detections
        for row in rows_with_multiple_detections:
            non_negative_indices = np.where(detected_indices_z_det_z_basis[row] != -1)[0]
            
            # Randomly select one index to keep
            selected_index = np.random.choice(non_negative_indices)
            
            # Set all other indices to -1
            detected_indices_z_det_z_basis[row, :] = -1
            detected_indices_z_det_z_basis[row, selected_index] = detected_indices_z_det_z_basis[row, selected_index]

        total_sift_z_basis_short = np.union1d(np.union1d(early_indices_short, late_indices_short), rows_with_multiple_detections)                  
        get_original_indexing_z = index_where_photons_det_z[basis[index_where_photons_det_z] == 1]

        # get vacuums
        indices_z_long = np.where(basis == 1)[0]
        amount_Z_sent = len(indices_z_long)
        amount_vacuum = amount_Z_sent - len(total_sift_z_basis_short)  # updated to use total_sift_z_basis_short
        if amount_Z_sent != 0:
            p_vacuum_z = amount_vacuum / amount_Z_sent
        else:
            p_vacuum_z = 0
        
        return detected_indices_z_det_z_basis, p_vacuum_z, total_sift_z_basis_short, indices_z_long, mask_z_short, get_original_indexing_z

    def classificator_sift_x_vacuum(self, basis, detected_indices_x, index_where_photons_det_x):
        # nur x-basis sendund
        mask_x_short = basis[index_where_photons_det_x] == 0
        detected_indices_x_det_x_basis = detected_indices_x[mask_x_short]
        # no measurement indices
        indices_x_long = np.where(basis == 0)[0]
        nothing_in_det_indices_short = np.where(np.all(detected_indices_x == -1, axis=1))[0]
        nothing_in_det_indices_long = index_where_photons_det_x[nothing_in_det_indices_short]
        full_range_array = np.arange(0, self.config.n_samples) 
        indices_x_no_photons_long = np.setdiff1d(full_range_array, index_where_photons_det_x)

        # assign random value to double counts
        non_negative_counts = np.sum(detected_indices_x_det_x_basis != -1, axis=1)
        rows_with_multiple_detections = np.where(non_negative_counts > 1)[0]

        # change detected_indices_z_det_z_basis to -1 except one for rows with multiple detections
        for row in rows_with_multiple_detections:
            non_negative_indices = np.where(detected_indices_x_det_x_basis[row] != -1)[0]
            
            # Randomly select one index to keep
            selected_index = np.random.choice(non_negative_indices)
            
            # Set all other indices to -1
            detected_indices_x_det_x_basis[row, :] = -1
            detected_indices_x_det_x_basis[row, selected_index] = detected_indices_x_det_x_basis[row, selected_index]

        vacuum_indices_x_long = np.union1d(indices_x_no_photons_long, nothing_in_det_indices_long)
        get_original_indexing_x = index_where_photons_det_x[basis[index_where_photons_det_x] == 1]

        '''with np.printoptions(threshold=100):
            Saver.save_results_to_txt(  # Save the results to a text file
                function_used = "sift_vac_x",
                n_samples=self.config.n_samples,
                seed=self.config.seed,
                non_signal_voltage=self.config.non_signal_voltage,
                voltage_decoy=self.config.voltage_decoy, 
                voltage=self.config.voltage, 
                voltage_decoy_sup=self.config.voltage_decoy_sup, 
                voltage_sup=self.config.voltage_sup,
                p_indep_x_states_non_dec=self.config.p_indep_x_states_non_dec,
                p_indep_x_states_dec=self.config.p_indep_x_states_dec,
                get_original_indexing_x = get_original_indexing_x,
                detected_indices_x_det_x_basis = detected_indices_x_det_x_basis,
                index_where_photons_det_x = index_where_photons_det_x,
                detected_indices_x = detected_indices_x.shape
                )'''

        # 1 or 2 signals in X basis
        sum_det_ind = np.sum(detected_indices_x >= 0, axis=1)
        one_or_two_in_x_short = np.where((sum_det_ind == 1) | (sum_det_ind == 2))[0]
        one_or_two_in_x_long = index_where_photons_det_x[one_or_two_in_x_short]

        total_sift_x_basis_long = np.union1d(vacuum_indices_x_long, one_or_two_in_x_long)

        return detected_indices_x_det_x_basis, total_sift_x_basis_long, vacuum_indices_x_long, indices_x_long, mask_x_short, get_original_indexing_x

    def classificator_identify_z(self, mask_x_short, value, total_sift_z_basis_short, detected_indices_x_det_x_basis, index_where_photons_det_z, decoy, indices_z_long, get_original_indexing_z, get_original_indexing_x):
        if index_where_photons_det_z.size == 0:
            return 0, 0, 0, 0
        # Z basis
        # all indices
        Z_indices_measured_long = index_where_photons_det_z[total_sift_z_basis_short]
        mask_Z0_long = value[Z_indices_measured_long] == 1
        mask_Z1_long = value[Z_indices_measured_long] == 0
        # only overlaps between Z0 vs Z1 sent and measured
        ind_Z0_verified_long = Z_indices_measured_long[mask_Z0_long]
        ind_Z1_verified_long = Z_indices_measured_long[mask_Z1_long]
        if get_original_indexing_x.size != 0:
            # check if no detection in late_bin X basis
            one_in_x_short = np.where(np.any(detected_indices_x_det_x_basis == 1, axis=1))[0]
            one_in_x_long = get_original_indexing_x[one_in_x_short]
            all_ind = np.arange(self.config.n_samples)
            no_one_in_x_long = np.setdiff1d(all_ind, one_in_x_long)
            ind_Z0_checked = np.intersect1d(ind_Z0_verified_long, no_one_in_x_long)
            ind_Z1_checked = np.intersect1d(ind_Z1_verified_long, no_one_in_x_long)
        else:
            ind_Z0_checked = ind_Z0_verified_long
            ind_Z1_checked = ind_Z1_verified_long

        # decoy and non-decoy
        ind_sent_non_dec = np.where((decoy == 0))[0]
        ind_Z0_checked_non_dec = np.intersect1d(ind_Z0_checked, ind_sent_non_dec)
        ind_Z1_checked_non_dec = np.intersect1d(ind_Z1_checked, ind_sent_non_dec)
        ind_sent_dec = np.where((decoy == 1))[0]
        ind_Z0_checked_dec = np.intersect1d(ind_Z0_checked, ind_sent_dec)
        ind_Z1_checked_dec = np.intersect1d(ind_Z1_checked, ind_sent_dec)

        # gain  
        ind_Z_sent_non_dec = np.intersect1d(indices_z_long, ind_sent_non_dec)
        '''print(f"indices_z_long: {indices_z_long}")
        print(f"ind_sent_non_dec: {ind_sent_non_dec}")
        print(f"ind_Z_sent_non_dec: {ind_Z_sent_non_dec}")'''
        len_Z_checked_non_dec = len(ind_Z0_checked_non_dec) + len(ind_Z1_checked_non_dec)

        if len(ind_Z_sent_non_dec) != 0:
            gain_Z_non_dec = len_Z_checked_non_dec / len(ind_Z_sent_non_dec)	
        else:
            gain_Z_non_dec = 0 #raise ValueError("No Z sent detected")
        
        # gain Z dec
        ind_Z_sent_dec = np.intersect1d(indices_z_long, ind_sent_dec)
        len_Z_checked_dec = len(ind_Z0_checked_dec) + len(ind_Z1_checked_dec)

        if len(ind_Z_sent_dec) != 0:
            gain_Z_dec = len_Z_checked_dec / len(ind_Z_sent_dec)	
        else:
            gain_Z_dec = 0 #raise ValueError("No Z decoy sent detected")
        
        return gain_Z_non_dec, gain_Z_dec, len_Z_checked_dec, len_Z_checked_non_dec

    def classificator_identify_x(self, mask_x_short, mask_z_short, detected_indices_x_det_x_basis, detected_indices_z_det_z_basis, basis, value, decoy, indices_x_long, get_original_indexing_x, get_original_indexing_z):
        # X basis
        # empty late in x basis
        # print(f"detected_indices_x_det_x_basis: {detected_indices_x_det_x_basis}")
        if get_original_indexing_x.size != 0:
            one_in_x_short = np.where(np.any(detected_indices_x_det_x_basis == 1, axis=1))[0]
            one_in_x_long = get_original_indexing_x[one_in_x_short]
            all_ind = np.arange(self.config.n_samples)
            no_one_in_x_long = np.setdiff1d(all_ind, one_in_x_long)    
            if get_original_indexing_z.size != 0:
                # no detection in Z basis
                zero_or_one_in_z_short = np.where(np.any(detected_indices_z_det_z_basis == 1, axis=1) | np.any(detected_indices_z_det_z_basis == 0))[0]
                zero_or_one_in_z_long = get_original_indexing_z[zero_or_one_in_z_short]
                all_ind = np.arange(self.config.n_samples)
                no_zero_or_one_in_z_long = np.setdiff1d(all_ind, zero_or_one_in_z_long)
                X_P_prime_checked_long = np.intersect1d(no_one_in_x_long, no_zero_or_one_in_z_long)
            else: 
                X_P_prime_checked_long = no_one_in_x_long
        else:
            no_one_in_x_long = np.arange(self.config.n_samples)
            X_P_prime_checked_long = no_one_in_x_long
        # print(f"X_P_prime_checked_long part: {X_P_prime_checked_long[:10]}")

        # decoy or not
        ind_sent_non_dec_long = np.where((decoy == 0))[0]
        ind_XP_prime_checked_non_dec = np.intersect1d(X_P_prime_checked_long, ind_sent_non_dec_long)
        ind_sent_dec_long = np.where((decoy == 1))[0]
        ind_XP_prime_checked_dec = np.intersect1d(X_P_prime_checked_long, ind_sent_dec_long)


        # sort out symbols for p_indep_x_states
        # create signal Z0X+ and then X+Z1
        if get_original_indexing_x.size != 0:
            Z0_alice_s = np.where((basis == 1) & (value == 1) & (decoy == 0))[0]  # Indices where Z0 was sent
            XP_alice_s = np.where((basis == 0) & (decoy == 0))[0]  # Indices where XP was sent
            Z0_XP_alice_s = XP_alice_s[np.isin(XP_alice_s - 1, Z0_alice_s)]  # Indices where Z1Z0 was sent (index of Z0 used aka the higher index at which time we measure the X+ state)
            has_0_short = np.where(np.any(detected_indices_x_det_x_basis == 0, axis=1))[0]
            # print(f"has_0_short: {has_0_short}, shape: {has_0_short.shape}")
            has_0_long = get_original_indexing_x[has_0_short]
            # print(f"has_0_long: {has_0_long}")
            has_0_z0xp_s = np.intersect1d(has_0_long, Z0_XP_alice_s)
            ind_has_0_z0xp_s = len(np.where(has_0_z0xp_s)[0])
            
            Z1_alice_s = np.where((basis == 1) & (value == 0) & (decoy == 0))[0]  # Indices where Z0 was sent
            XP_Z1_alice_s = Z1_alice_s[np.isin(Z1_alice_s - 1, XP_alice_s)]  # Indices where Z1Z0 was sent (index of Z0 used aka the higher index at which time we measure the X+ state)
            has_0_xpz1_s = np.intersect1d(has_0_long, XP_Z1_alice_s)
            ind_has_0_xpz1_s = len(np.where(has_0_xpz1_s)[0])
            print(f"ind_has_0_xpz1_s: {ind_has_0_xpz1_s}")
            print(f"XP_Z1_alice_s: {XP_Z1_alice_s}, shape: {XP_Z1_alice_s.shape}")

            print(f"ind_has_0_z0xp_s: {ind_has_0_z0xp_s}")
            print(f"Z0_XP_alice_s: {Z0_XP_alice_s}, shape: {Z0_XP_alice_s.shape}")

            X_P_calc_non_dec = (ind_has_0_xpz1_s + ind_has_0_z0xp_s) / ( (1 / 4) * self.config.p_z_alice)
        else:
            X_P_calc_non_dec = 0
       
        # create signal Z0X+ and then X+Z1
        if get_original_indexing_x.size != 0:
            Z0_alice_d = np.where((basis == 1) & (value == 1) & (decoy == 1))[0]  # Indices where Z0 was sent
            XP_alice_d = np.where((basis == 0) & (decoy == 1))[0]  # Indices where XP was sent
            Z0_XP_alice_d = XP_alice_d[np.isin(XP_alice_d - 1, Z0_alice_d)]  # Indices where Z1Z0 was sent (index of Z0 used aka the higher index at which time we measure the X+ state)
            has_0_short = np.where(np.any(detected_indices_x_det_x_basis == 0, axis=1))[0]
            has_0_long = get_original_indexing_x[has_0_short]
            has_0_z0xp_d = np.intersect1d(has_0_long, Z0_XP_alice_d)
            ind_has_0_z0xp_d = len(np.where(has_0_z0xp_d)[0])
            
            Z1_alice_d = np.where((basis == 1) & (value == 0) & (decoy == 1))[0]  # Indices where Z1 was sent
            XP_Z1_alice_d = Z1_alice_d[np.isin(Z1_alice_d - 1, XP_alice_d)]  # Indices where Z1Z0 was sent (index of Z0 used aka the higher index at which time we measure the X+ state)
            has_0_xpz1_d = np.intersect1d(has_0_long, XP_Z1_alice_d)
            ind_has_0_xpz1_d = len(np.where(has_0_xpz1_d)[0])
            print(f"ind_has_0_xpz1_d: {ind_has_0_xpz1_d}")
            print(f"XP_Z1_alice_d: {XP_Z1_alice_d}, shape: {XP_Z1_alice_d.shape}")

            print(f"ind_has_0_z0xp_d: {ind_has_0_z0xp_d}")
            print(f"Z0_XP_alice_d: {Z0_XP_alice_d}, shape: {Z0_XP_alice_d.shape}")

            X_P_calc_dec = (ind_has_0_xpz1_d + ind_has_0_z0xp_d) / ( (1 / 4) * self.config.p_z_alice)
            print(f"X_P_calc_dec:{X_P_calc_dec}")
            print(f"X_P_calc_non_dec:{X_P_calc_non_dec}")
        else:
            X_P_calc_dec = 0

        # gain non dec 
        ind_sent_non_dec_long = np.where((decoy == 0))[0]
        ind_x_sent_non_dec_long = np.intersect1d(indices_x_long, ind_sent_non_dec_long)

        if len(ind_x_sent_non_dec_long) != 0:
            gain_X_non_dec = X_P_calc_non_dec / len(ind_x_sent_non_dec_long)
        else:
            gain_X_non_dec = 0 #raise ValueError("No Z sent detected")
        
        # gain X dec
        ind_sent_dec_long = np.where((decoy == 1))[0]
        ind_x_sent_dec_long = np.intersect1d(indices_x_long, ind_sent_dec_long)

        if len(ind_x_sent_dec_long) != 0:
            gain_X_dec = X_P_calc_dec / len(ind_x_sent_dec_long)
        else:
            gain_X_dec = 0 #raise ValueError("No Z decoy sent detected")
        # print(f"Returning: {X_P_calc_non_dec}, {X_P_calc_dec}, {gain_X_non_dec}, {gain_X_dec}")

            
        return X_P_calc_non_dec, X_P_calc_dec, gain_X_non_dec, gain_X_dec
    
    
    def classificator_errors(self, mask_x_short, mask_z_short, indices_z_long, indices_x_long, value, detected_indices_z_det_z_basis, detected_indices_x_det_x_basis, basis, decoy, get_original_indexing_x, get_original_indexing_z):
        wrong_detection_mask_z = np.zeros(len(basis), dtype=bool)
        wrong_detection_mask_x = np.zeros(len(basis), dtype=bool)
        ind_sent_non_dec_long = np.where((decoy == 0))[0]
        ind_sent_dec_long = np.where((decoy == 1))[0]
        #Step 1: Check for wrong detections in the Z basis (Z0 and Z1)
        #check if wrong_detection_mask is empty
        if get_original_indexing_z.size != 0:
            # measure in late for Z0 (wrong detection): value 1 für Z0
            has_1_short = np.where(np.any(detected_indices_z_det_z_basis == 1, axis=1))[0]
            has_1_long = get_original_indexing_z[has_1_short]       # detected indices has shape of time_photons_det
            has_sent_Z0_long = np.intersect1d(indices_z_long, np.where(value == 1)[0])
            has_1_and_z0_long = np.intersect1d(has_1_long, has_sent_Z0_long)
            wrong_detection_mask_z[np.where(has_1_and_z0_long)[0]] = True
            #Condition 3: Measure in early for Z1 (wrong detection), value 0 for Z1
            has_0_short = np.where(np.any(detected_indices_z_det_z_basis == 0, axis=1))[0]
            has_0_long = get_original_indexing_z[has_0_short]
            has_sent_Z1_long = np.intersect1d(indices_z_long, np.where(value == 0)[0])
            has_0_and_z1_long = np.intersect1d(has_0_long, has_sent_Z1_long)
            wrong_detection_mask_z[np.where(has_0_and_z1_long)[0]] = True
            # filter out decoy cases
            wrong_detections_z = np.where(wrong_detection_mask_z)[0]
            wrong_detections_z_dec = np.intersect1d(wrong_detections_z, ind_sent_dec_long)
            wrong_detections_z_non_dec = np.intersect1d(wrong_detections_z, ind_sent_non_dec_long)
        else:
            wrong_detections_z_dec = np.array([])
            wrong_detections_z_non_dec = np.array([])

        if get_original_indexing_x.size != 0:
            #Condition 6: Late detection in X+ after X+ sent
            has_1_short = np.where(np.any(detected_indices_x_det_x_basis == 1, axis=1))[0]
            has_1_long = get_original_indexing_x[has_1_short]
            has_sent_xp_long = indices_x_long
            has_1_and_xp_long = np.intersect1d(has_1_long, has_sent_xp_long)
            wrong_detection_mask_x[np.where(has_1_and_xp_long)[0]] = True
            wrong_detections_x = np.where(wrong_detection_mask_x)[0]

            #Condition 7: Early detection in Z0 after Z1 sent (wrong detection), value 0 for Z0
            Z1_alice_s = np.where((basis == 1) & (value == 0) & (decoy == 0))[0]  # Indices where Z1 was sent
            Z0_alice_s = np.where((basis == 1) & (value == 1) & (decoy == 0))[0]  # Indices where Z0 was sent
            Z1_Z0_alice_s = Z0_alice_s[np.isin(Z0_alice_s - 1, Z1_alice_s)]  # Indices where Z1Z0 was sent (index of Z0 used aka the higher index at which time we measure the X+ state)
            Z1_alice_d = np.where((basis == 1) & (value == 0) & (decoy == 1))[0]  # Indices where Z1 was sent
            Z0_alice_d = np.where((basis == 1) & (value == 1) & (decoy == 1))[0]  # Indices where Z0 was sent
            Z1_Z0_alice_d = Z0_alice_d[np.isin(Z0_alice_d - 1, Z1_alice_d)]  # Indices where Z1Z0 was sent (index of Z0 used aka the higher index at which time we measure the X+ state)
            
            has_0_short = np.where(np.any(detected_indices_x_det_x_basis == 0, axis=1))[0]
            has_0_long = get_original_indexing_x[has_0_short]
            has_0_z1z0_s = np.intersect1d(has_0_long, Z1_Z0_alice_s)
            wrong_detection_mask_x[np.where(has_0_z1z0_s)[0]] = True
            wrong_detections_x = np.where(wrong_detection_mask_x)[0]

            has_0_z1z0_d = np.intersect1d(has_0_long, Z1_Z0_alice_d)
            wrong_detection_mask_x[np.where(has_0_z1z0_d)[0]] = True
            wrong_detections_x = np.where(wrong_detection_mask_x)[0]
                     
            # decoy and non-decoy
            wrong_detections_x_dec = np.intersect1d(wrong_detections_x, ind_sent_dec_long)
            wrong_detections_x_non_dec = np.intersect1d(wrong_detections_x, ind_sent_non_dec_long)
                        
        else:
            wrong_detections_x_dec = np.array([])
            wrong_detections_x_non_dec = np.array([])
            
        wrong_detections_z_dec = np.sort(wrong_detections_z_dec)
        wrong_detections_z_non_dec = np.sort(wrong_detections_z_non_dec)
        wrong_detections_x_dec = np.sort(wrong_detections_x_dec)
        wrong_detections_x_non_dec = np.sort(wrong_detections_x_non_dec)

        return wrong_detections_z_dec, wrong_detections_z_non_dec, wrong_detections_x_dec, wrong_detections_x_non_dec
    
    def classificator_qber_rkr(self, t, wrong_detections_z_dec, wrong_detections_z_non_dec, wrong_detections_x_dec, wrong_detections_x_non_dec, len_Z_checked_dec, len_Z_checked_non_dec, X_P_calc_non_dec, X_P_calc_dec):
        if len_Z_checked_dec != 0:
            qber_z_dec = len(wrong_detections_z_dec) / len_Z_checked_dec
        else:
            qber_z_dec = 0

        if len_Z_checked_non_dec != 0:
            qber_z_non_dec = len(wrong_detections_z_non_dec) / len_Z_checked_non_dec
        else:
            qber_z_non_dec = 0

        if X_P_calc_dec != 0:
            qber_x_dec = len(wrong_detections_x_dec) / X_P_calc_dec
        else:
            qber_x_dec = 0

        if X_P_calc_non_dec != 0:
            qber_x_non_dec = len(wrong_detections_x_non_dec) / X_P_calc_non_dec
        else:
            qber_x_non_dec = 0

        total_amount_detections = len_Z_checked_dec + len_Z_checked_non_dec + X_P_calc_dec + X_P_calc_non_dec   

        raw_key_rate = total_amount_detections / (t[-1] * self.config.n_samples)

        return qber_z_dec, qber_z_non_dec, qber_x_dec, qber_x_non_dec, raw_key_rate, total_amount_detections
    
    # ========== Data Processing Helper ==========

    def poisson_distr(self, calc_value):
        """Calculate the number of photons based on a Poisson distribution."""
        assert np.all(calc_value >= 0), "calc_value must be non-negative."
    
        # Use NumPy's built-in Poisson sampler
        nr_photons = self.config.rng.poisson(calc_value)

        return nr_photons
    
# --------- check mean photon number at different points -----------

    def calculate_mean_photon_number(self, power_dampened, peak_wavelength, t, one_peak= False):
        if one_peak == 'first':
            # Select the first peak (e.g., first half of the pulse)
            midpoint = len(t) // 2
            power_dampened = power_dampened[:, :midpoint]
            t = t[:midpoint]
        elif one_peak == 'last':
            # Select the last peak (e.g., second half of the pulse)
            midpoint = len(t) // 2
            power_dampened = power_dampened[:, midpoint:]
            t = t[midpoint:]

        energy_per_pulse = np.trapz(power_dampened, t, axis=1)
        energy_one_photon = constants.h * constants.c / peak_wavelength
        calc_mean_photon_nr = energy_per_pulse / energy_one_photon

        return calc_mean_photon_nr

