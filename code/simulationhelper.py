import numpy as np
from scipy.fftpack import fft, ifft, fftfreq
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
        pattern = np.zeros((len(value), self.config.n_pulses), dtype=int)
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
        t = np.linspace(0, self.config.n_pulses * pulse_duration, total_samples, endpoint=False)
        #t = np.arange(0, self.config.n_pulses * pulse_duration, inv_sampling, dtype=np.float64)
        repeating_square_pulses = np.full((len(pulse_height), len(t)), self.config.non_signal_voltage, dtype=np.float64)
        one_pulse = len(t) // self.config.n_pulses
        indices = np.arange(len(t))
        for i, pattern in enumerate(pattern):
            for j, bit in enumerate(pattern):
                if bit == 1:
                    repeating_square_pulses[i, (indices // one_pulse) == j] = pulse_height[i]
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
        index_shift_per_symbol = ((len(t) // t[-1]) * jitter_shifts).astype(int)
        index_shift_per_symbol = index_shift_per_symbol[: self.config.n_pulses * self.config.batchsize - 1]  #size = n_pulses * batchsize - 1 (minus Anfang und Ende)
        index_one_signal = len(t) // self.config.n_pulses
        transition_indices = np.arange(index_one_signal, len(signals), index_one_signal)

        new_transition_indices = transition_indices + index_shift_per_symbol
        # Step through and shift transitions **in-place**
        for i, old_idx in enumerate(transition_indices):
            new_idx = new_transition_indices[i]  # Get the new shifted index
            
            if old_idx < new_idx:
                # Shift right: Fill original transition spot with previous value
                signals[old_idx:new_idx] = signals[old_idx - 1]
            elif old_idx > new_idx:
                # Shift left: Fill new transition spot with flipped value
                signals[new_idx:old_idx] = signals[old_idx + 1]

        return signals

    # ========== Detector Helper ==========
    def choose_photons(self, norm_transmission, t, power_dampened, peak_wavelength, start_time, fixed_nr_photons=None):
        """Calculate and choose photons based on the transmission and jitter."""
        # Calculate the mean photon number
        Saver.memory_usage("in choose photon before anything")
        energy_per_pulse = np.trapezoid(power_dampened, t, axis=1)
        Saver.memory_usage("in choose photon after energy_per_pulse")
        # delete power_dampened
        del power_dampened
        gc.collect()
        Saver.memory_usage("in choose photon after gc.collect")
        energy_one_photon = constants.h * constants.c / peak_wavelength
        calc_mean_photon_nr_detector = energy_per_pulse / energy_one_photon
        # Use Poisson distribution to get the number of photons
        Saver.memory_usage("in choose photon after calc_mean_photon_nr_detector")
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
            time_photons[i, :photon_count] = self.config.rng.choice(t, size=photon_count, p=norm_transmission[idx]) #t ist konstant
            '''if np.any(wavelength_photons[i] > 1.6e-6):
                print(f"wavelength_photons: {wavelength_photons[i]}")
                print(f"energy_per_photon: {energy_per_photon[i]}")
                print(f"energy_per_pulse: {energy_per_pulse[idx]}")'''
                

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
    
    def darkcount(self):
        """Calculate the number of dark count photons detected."""
        pulse_duration = 1 / self.config.sampling_rate_FPGA
        symbol_duration = pulse_duration * self.config.n_pulses
        num_dark_counts = self.config.rng.poisson(self.config.dark_count_frequency * symbol_duration, size=self.config.n_samples)
        dark_count_times = [np.sort(self.config.rng.uniform(0, symbol_duration, count)) if count > 0 else np.empty(0) for count in num_dark_counts]
        return dark_count_times, num_dark_counts

    # ========== Classificator Helper ========== 

    def classificator_sift_z_vacuum(self, basis, detected_indices_z, index_where_photons_det_z):
        # nur z basis sendung
        detected_indices_z = detected_indices_z[basis[index_where_photons_det_z] == 1]
        early_indices = np.where(np.sum(detected_indices_z == 0, axis=1) == 1)[0]
        late_indices = np.where(np.sum(detected_indices_z == 1, axis=1) == 1)[0]
        total_sift_z = np.union1d(early_indices, late_indices)                  

        # get vacuums
        indices_z = np.where(basis == 1)[0]
        amount_Z_sent = len(indices_z)
        amount_vacuum = amount_Z_sent - len(total_sift_z)  # corrected from combined_indices_z to total_sift_z
        p_vacuum_z = amount_vacuum / amount_Z_sent
    
        return detected_indices_z, p_vacuum_z, total_sift_z, indices_z

    def classificator_sift_x_vacuum(self, basis, detected_indices_x, index_where_photons_det_x):
        # nur x-basis sendung
        detected_indices_x = detected_indices_x[basis[index_where_photons_det_x] == 0]
        # no measurement indices
        indices_x = np.where(basis == 0)[0]
        nothing_in_det_indices = np.where(np.all(detected_indices_x == -1, axis=1))[0]
        full_range_array = np.arange(0, self.config.n_samples)  # From 0 to 20
        indices_x_no_photons = np.setdiff1d(full_range_array, index_where_photons_det_x)
        vacuum_indices_x = np.union1d(indices_x_no_photons, nothing_in_det_indices)

        # 1 or 2 signals in X basis
        sum_det_ind = np.sum(detected_indices_x >= 0, axis=1)
        one_or_two_in_x = np.where((sum_det_ind == 1) | (sum_det_ind == 2))[0]

        total_sift_x = np.union1d(vacuum_indices_x, one_or_two_in_x)

        return detected_indices_x, total_sift_x, vacuum_indices_x, indices_x

    def classificator_identify_z(self, value, total_sift_z, detected_indices_z, detected_indices_x, index_where_photons_det_z, decoy, indices_z):
        # Z basis
        # on whole array 
        mask_Z0_whole = value[Z0_indices_measured] == 1
        mask_Z1_whole = value[Z1_indices_measured] == 0
        ind_Z0_sent = index_where_photons_det_z[mask_Z0_whole]
        ind_Z1_sent = index_where_photons_det_z[mask_Z1_whole]
        # alle indices
        Z0_indices = np.where(np.sum(detected_indices_z[total_sift_z] == 0, axis=1) == 1)[0]
        Z1_indices = np.where(np.sum(detected_indices_z[total_sift_z] == 1, axis=1) == 1)[0]
        Z0_indices_measured = total_sift_z[Z0_indices]
        Z1_indices_measured = total_sift_z[Z1_indices]
        # only overlaps between sent and measured
        ind_Z0_verified = np.intersect1d(ind_Z0_sent, Z0_indices_measured)
        ind_Z1_verified = np.intersect1d(ind_Z1_sent, Z1_indices_measured)

        # check if no detection in late_bin X basis
        one_in_x = np.where(np.any(detected_indices_x == 1, axis=1))[0]
        all_ind = np.arange(self.config.n_samples)
        no_one_in_x = np.setdiff1d(all_ind, one_in_x)
        ind_Z0_checked = np.intersect1d(ind_Z0_verified, no_one_in_x)
        ind_Z1_checked = np.intersect1d(ind_Z1_verified, no_one_in_x)

        # gain  !! stimmt nich
        ind_sent_non_dec = np.where((decoy == 0))[0]
        ind_Z_sent_det_non_dec = np.intersect1d(indices_z, ind_sent_non_dec)
        ind_Z0_checked_non_dec = np.intersect1d(ind_Z0_checked, ind_Z_sent_det_non_dec)
        ind_Z1_checked_non_dec = np.intersect1d(ind_Z1_checked, ind_Z_sent_det_non_dec)
        len_Z_checked_non_dec = len(ind_Z0_checked_non_dec) + len(ind_Z1_checked_non_dec)

        if len(ind_Z_sent_det) != 0:
            gain_Z = len_Z_checked_non_dec / len(ind_Z_sent_det)	
        else:
            raise ValueError("No Z sent detected")
        
        # gain Z dec
        ind_sent_dec = np.where((decoy == 1))[0]
        ind_Z_sent_det = np.intersect1d(indices_z, ind_sent_dec)
        ind_Z0_checked_dec = np.intersect1d(ind_Z0_checked, ind_Z_sent_det)
        ind_Z1_checked_dec = np.intersect1d(ind_Z1_checked, ind_Z_sent_det)
        len_Z_checked_dec = len(ind_Z0_checked_dec) + len(ind_Z1_checked_dec)

        if len(ind_Z_sent_det) != 0:
            gain_Z = len_Z_checked_dec / len(ind_Z_sent_det)	
        else:
            raise ValueError("No Z decoy sent detected")

        return ind_Z0_checked, ind_Z1_checked

    def classificator_identify_x(self, total_sift_x, detected_indices_x, detected_indices_z, index_where_photons_det_x, basis, value, decoy, indices_x):
        # X basis
        # empty late in x basis
        no_one_in_x = np.where(np.sum(detected_indices_x[total_sift_x] != 1, axis=1) == 0)[0]
        X_P_prime = total_sift_x[no_one_in_x]

        # no detection in Z basis
        zero_or_one_in_z = np.where(np.any(detected_indices_z == 1, axis=1) | np.any(detected_indices_z == 0))[0]
        all_ind = np.arange(self.config.n_samples)
        no_zero_or_one_in_z = np.setdiff1d(all_ind, zero_or_one_in_z)
        X_P_prime_checked = np.intersect1d(X_P_prime, no_zero_or_one_in_z)

        # calculate p_indepedent_x_states
        # Z0X+
        Z0_alice = np.where((basis == 1) & (value == 0))[0]  # Indices where Z0 was sent
        XP_alice = np.where((basis == 0) & (value == 0))[0]  # Indices where XP was sent
        Z0_XP_alice = XP_alice[np.isin(XP_alice - 1, Z0_alice)]  # Indices where Z0XP was sent (index of Z0 used aka the higher index at which time we measure the X+ state)
        has_0_and_Z0XP = np.any(detected_indices_x == 0, axis=1) &  np.isin(index_where_photons_det_x, Z0_XP_alice)
        # X+Z1
        Z1_alice = np.where((basis ==1) & (value == 0))[0]  # Indices where Z0 was sent
        XP_Z1_alice = XP_alice[np.isin(Z1_alice - 1, XP_alice)]  # Indices where Z1XP was sent (index of X+ used aka the higher index at which time we measure the X+ state)
        has_0_and_XPZ1 = np.any(detected_indices_x == 0, axis=1) &  np.isin(index_where_photons_det_x, XP_Z1_alice)

        p_indep_x_states = np.sum(has_0_and_Z0XP) + np.sum(has_0_and_XPZ1) / (len(Z0_XP_alice) + len(XP_Z1_alice))

        X_P_calc = X_P_prime_checked * p_indep_x_states 
        return X_P_calc, p_indep_x_states
    
    def classificator_errors(self, index_where_photons_det_x, index_where_photons_det_z, detected_indices_z, detected_indices_x, basis):
        wrong_detection_mask_z = np.zeros(len(index_where_photons_det_z), dtype=bool)
        print(f"wrong_detection_mask_z.shape: {wrong_detection_mask_z.shape}")
        wrong_detection_mask_x = np.zeros(len(index_where_photons_det_x), dtype=bool)

        #Step 1: Check for wrong detections in the Z basis (Z0 and Z1)
        #check if wrong_detection_mask is empty
        if wrong_detection_mask_z.size != 0:
            #Condition 2: Measure in late for Z0 (wrong detection)
            has_one_and_z0 = np.any(detected_indices_z == 1, axis=1) & basis[index_where_photons_det_z] == 1        # detected indices has shape of time_photons_det
            wrong_detection_mask_z[np.where(has_one_and_z0)[0]] = True
            #Condition 3: Measure in early for Z1 (wrong detection)
            has_0_and_z1 = np.any(detected_indices_z == 0, axis=1) & basis[index_where_photons_det_z] == 1
            wrong_detection_mask_z[np.where(has_0_and_z1)[0]] = True
            #get wrong_detections thruough correct indexing
            wrong_detections_z = index_where_photons_det_z[wrong_detection_mask_z]
        else:
            wrong_detections_z = np.array([])
        


    # ab hier eig nich mehr brauch
    def classificator_det_ind(self, timebins, decoy, time_photons_det, index_where_photons_det, is_decoy):
        # Apply np.digitize and shift indices
        detected_indices = np.where(
            np.isnan(time_photons_det),                                         # Check for NaN values
            -1,                                                                 # Assign -1 for undetected photons
            np.digitize(time_photons_det, timebins) - 1                         # Early = 0, Late = 1
        )                                                                       # 0 wenn early timebin, 1 wenn late timebin, -1 wenn nicht detektiert
        reduced_decoy = decoy[index_where_photons_det]
        if is_decoy == False:
            detected_indices = detected_indices[reduced_decoy == 0]
        else:
            detected_indices = detected_indices[reduced_decoy == 1]
        return detected_indices

    def classificator_z(self, basis, value, decoy, index_where_photons_det_z, index_where_photons_det_x, detected_indices_z, detected_indices_x, is_decoy):
        # Z basis
        if index_where_photons_det_z.size == 0 or index_where_photons_det_x.size == 0:
            return 0, 0
        #Find indices where there is exactly one 0 (Z0 detection)
        Z0_indices_measured_reduced = np.where(np.sum(detected_indices_z == 0, axis=1) == 1)[0]
        #Find indices where there is exactly one 1 (Z1 detection)
        Z1_indices_measured_reduced = np.where(np.sum(detected_indices_z == 1, axis=1) == 1)[0]
        #get indices in original indexing
        Z0_indices_measured = index_where_photons_det_z[Z0_indices_measured_reduced]
        Z1_indices_measured = index_where_photons_det_z[Z1_indices_measured_reduced]
        #only keep those where alice sent Z:
        mask_Z0 = (basis[Z0_indices_measured] == 1) & (value[Z0_indices_measured] == 1)
        Z0_indices_checked_with_send = Z0_indices_measured[mask_Z0]
        mask_Z1 = (basis[Z1_indices_measured] == 1) & (value[Z1_indices_measured] == 0)
        Z1_indices_checked_with_send = Z1_indices_measured[mask_Z1]

        # find indices where there is no detection in late_bin X basis
        one_in_x = np.where(np.any(detected_indices_x == 1, axis=1))[0]
        all_ind = np.arange(self.config.n_samples)
        no_one_in_x = np.setdiff1d(all_ind, one_in_x)

        Z1_indices_checked_no_one_in_z = np.intersect1d(Z1_indices_checked_with_send, no_one_in_x)
        Z0_indices_checked_no_one_in_z = np.intersect1d(Z0_indices_checked_with_send, no_one_in_x)

        amount_Z_det = len(Z0_indices_checked_no_one_in_z) + len(Z1_indices_checked_no_one_in_z)
        if is_decoy == False:
            Z_sent = np.sum((basis == 1) & (decoy == 0))
        else:
            Z_sent = np.sum((basis == 1) & (decoy == 1))
        if Z_sent != 0:
            gain_Z = amount_Z_det / Z_sent
        else:
            gain_Z = np.nan
        return gain_Z, amount_Z_det
    
    def classificator_x(self, basis, value, decoy, index_where_photons_det_x, index_where_photons_det_z, detected_indices_x, detected_indices_z, gain_Z, is_decoy):
        # X detection
        # if no detect indices, return 0
        if index_where_photons_det_x.size == 0 or index_where_photons_det_z.size == 0:
            return 0, 0
        
        # Find indices where there are no detection in late timebins
        no_ones_rows_reduced = np.where(np.all((detected_indices_x != 1), axis=1))[0]
        no_ones_rows_full = index_where_photons_det_x[no_ones_rows_reduced]
        all_ind = np.arange(self.config.n_samples)
        remaining_indices = np.setdiff1d(all_ind, index_where_photons_det_x)
        XP_indices_measured = np.concatenate((no_ones_rows_full, remaining_indices))
        # only keep those where alice sent X+
        mask_x = basis[XP_indices_measured] == 0
        XP_indices_checked_with_send = XP_indices_measured[mask_x]

        # find indices where there is no detection in Z basis ! that has to be fulfilled aswell SIFTEN
        zero_or_one_in_z = np.where(np.any(detected_indices_z == 1, axis=1) | np.any(detected_indices_z == 0))[0]
        no_zero_or_one_in_z = np.setdiff1d(all_ind, zero_or_one_in_z)

        XP_indices_checked_no_z = np.intersect1d(XP_indices_checked_with_send, no_zero_or_one_in_z)

        amount_XP_det = round(gain_Z * len(XP_indices_checked_no_z))
        if is_decoy == False:
            XP_sent = np.sum((basis == 0) & (decoy == 0))
        else:
            XP_sent = np.sum((basis == 0) & (decoy == 1))
        if XP_sent != 0:
            gain_XP = amount_XP_det / XP_sent
        else:
            gain_XP = np.nan
        return gain_XP, amount_XP_det


    def classificator_error_cases(self, basis, value, index_where_photons_det_x, index_where_photons_det_z, total_detected_indices_x, total_detected_indices_z):
        # Error cases
        #Initialize a boolean array to track wrong detections (same length as number of detections)
        wrong_detection_mask_z = np.zeros(len(index_where_photons_det_z), dtype=bool)
        print(f"wrong_detection_mask_z.shape: {wrong_detection_mask_z.shape}")
        wrong_detection_mask_x = np.zeros(len(index_where_photons_det_x), dtype=bool)

        #Step 1: Check for wrong detections in the Z basis (Z0 and Z1)
        #check if wrong_detection_mask is empty
        if wrong_detection_mask_z.size != 0:
            #Condition 2: Measure in late for Z0 (wrong detection)
            has_one_and_z0 = np.any(total_detected_indices_z == 1, axis=1) & basis[index_where_photons_det_z] == 1        # detected indices has shape of time_photons_det
            wrong_detection_mask_z[np.where(has_one_and_z0)[0]] = True
            #Condition 3: Measure in early for Z1 (wrong detection)
            has_0_and_z1 = np.any(total_detected_indices_z == 0, axis=1) & basis[index_where_photons_det_z] == 1
            wrong_detection_mask_z[np.where(has_0_and_z1)[0]] = True
            #get wrong_detections thruough correct indexing
            wrong_detections_z = index_where_photons_det_z[wrong_detection_mask_z]
        else:
            wrong_detections_z = np.array([])
        
        #Step 2: Check for wrong detections in the X+ basis
        if wrong_detection_mask_x.size != 0:
            #Condition 4: Early detection in X+ state after Z1Z0 in when Z0 got sent
            Z1_alice = np.where((basis == 1) & (value == 1))[0]  # Indices where Z1 was sent
            Z0_alice = np.where((basis == 1) & (value == 0))[0]  # Indices where Z0 was sent
            Z1_Z0_alice = Z0_alice[np.isin(Z0_alice - 1, Z1_alice)]  # Indices where Z1Z0 was sent (index of Z0 used aka the higher index at which time we measure the X+ state)
            has_0_and_z0z1 = np.any(total_detected_indices_x == 0, axis=1) &  np.isin(index_where_photons_det_x, Z1_Z0_alice)
            wrong_detection_mask_x[np.where(has_0_and_z0z1)[0]] = True
            #Condition 5: Early detection in X+ after Z1X+
            XP_alice = np.where((basis == 0))[0] # Indices where X+ was sent
            Z1_XP_alice = XP_alice[np.isin(XP_alice - 1, Z1_alice)]  # Indices where Z1X+ was sent (index of X+ used aka the higher index at which time we measure the X+ state)
            has_0_and_z1xp = np.any(total_detected_indices_x == 0, axis=1) &  np.isin(index_where_photons_det_x, Z1_XP_alice)
            wrong_detection_mask_x[np.where(has_0_and_z1xp)[0]] = True
            #Condition 6: Late detection in X+ after X+ sent
            has_1_and_xp = np.any(total_detected_indices_x == 1, axis=1) & (basis[index_where_photons_det_x] == 0)
            wrong_detection_mask_x[np.where(has_1_and_xp)[0]] = True
            #`wrong_detection_mask` is a boolean array where True indicates a wrong detection
            wrong_detections_x = index_where_photons_det_x[wrong_detection_mask_x]
        else:
            wrong_detections_x = np.array([])
            
        #Combine the wrong detections from both bases
        wrong_detections_z = np.sort(wrong_detections_z)
        wrong_detections_x = np.sort(wrong_detections_x)                                        # now sorted
        return wrong_detections_z, wrong_detections_x         # all indices of wrong detections
    
    # ========== Data Processing Helper ==========

    def poisson_distr(self, calc_value):
        """Calculate the number of photons based on a Poisson distribution."""
        #print(f"np.any(array < 0): {np.isnan(calc_value < 0)}")
        #print(f"np.any(np.isnan(array)):{np.any(np.isnan(calc_value))}")

        upper_bounds = (calc_value + 5 * np.sqrt(calc_value)).astype(int) # shape: (n_samples,)
        max_upper_bound = upper_bounds.max()
        x = np.arange(0, max_upper_bound + 1)
        probabilities_array_poisson = np.exp(-calc_value[:, None]) * (calc_value[:, None] ** x) / factorial(x) #shape: (n_samples, max_upper_bound + 1)
        # Normalize the probabilities for every row
        probabilities_array_poisson /= probabilities_array_poisson.sum(axis=1, keepdims=True)
        # Step 1: Compute cumulative probabilities
        cumulative_probabilities = np.cumsum(probabilities_array_poisson, axis=1) #shape: (n_samples, max_upper_bound + 1)
        # Step 2: Generate random values
        random_values = self.config.rng.random(size=probabilities_array_poisson.shape[0]) # shape: (n_samples,)
        # Step 3: Find indices using searchsorted
        sampled_indices = (cumulative_probabilities.T < random_values).sum(axis=0) #cumulative_probabilities.T < random_values boolean  matrix False or True 
        # Step 4: Map indices to x values
        nr_photons = x[sampled_indices]
        return nr_photons
    

  
     

