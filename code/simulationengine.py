import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import splev
from scipy.fftpack import fft, ifft, fftfreq
from scipy import constants
from scipy.special import factorial
import time

from saver import Saver
from simulationsingle import SimulationSingle
from simulationhelper import SimulationHelper


class SimulationEngine:
    def __init__(self, config):
        self.config = config
        self.simulation_single = SimulationSingle(config)
        self.simulation_helper = SimulationHelper(config)

    def get_interpolated_value(self, x_data, name):
        #calculate tck for which curve
        tck = self.config.data.get_data(x_data, name)
        return splev(x_data, tck)

    def random_laser_output(self, current_power, voltage_shift, current_wavelength):
        'every batchsize values we get a new chosen value'
        # Generate a random time within the desired range
        times = self.config.rng.uniform(0, 10, self.config.n_samples // self.config.batchsize)
        # Use sinusoidal modulation for the entire array
        chosen_voltage = self.config.mean_voltage + 0.050 * np.sin(2 * np.pi * 1 * times)
        chosen_current = (self.config.mean_current + self.config.current_amplitude * np.sin(2 * np.pi * 1 * times)) * 1e3
        optical_power_short = self.get_interpolated_value(chosen_current, current_power)
        peak_wavelength_short = self.get_interpolated_value(chosen_current, current_wavelength) + self.get_interpolated_value(chosen_voltage, voltage_shift)
        optical_power = np.repeat(optical_power_short, self.config.batchsize)
        peak_wavelength = np.repeat(peak_wavelength_short, self.config.batchsize)
        return optical_power * 1e-3, peak_wavelength * 1e-9  # in W and m

    def generate_alice_choices(self, basis=None, value=None, decoy=None):
        """Generates Alice's choices for a quantum communication protocol."""
        # Generate arrays if parameters are not provided

        if basis is None:
            basis = self.config.rng.choice(
                [0, 1], size=self.config.n_samples, p=[1 - self.config.p_z_alice, self.config.p_z_alice]
            )
        if value is None:
            value = self.config.rng.choice(
                [0, 1], size=self.config.n_samples, p=[0.5, 0.5]
            )
        if decoy is None:
            decoy = self.config.rng.choice(
                [0, 1], size=self.config.n_samples, p=[1 - self.config.p_decoy, self.config.p_decoy]
            )

        # Ensure all inputs are NumPy arrays
        basis = np.array(basis, dtype=int)
        value = np.array(value, dtype=int)
        decoy = np.array(decoy, dtype=int)
        
        # Adjust value for array case if basis is 0
        if isinstance(basis, np.ndarray):
            value[basis == 0] = -1
        else:
            value = -1 if basis == 0 else value

        # Ensure outputs are arrays of the correct size
        if basis.size == 1:  # Scalar case
            basis = np.full(self.config.n_samples, basis, dtype=int)
        if value.size == 1:
            value = np.full(self.config.n_samples, value, dtype=int)
        if decoy.size == 1:
            decoy = np.full(self.config.n_samples, decoy, dtype=int)

        return basis, value, decoy
    
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

    def signal_bandwidth_jitter(self, basis, values, decoy):
        pulse_heights = self.get_pulse_height(basis, decoy)
        jitter_shifts = self.get_jitter('laser', size_jitter = self.config.n_samples * self.config.n_pulses)
        pulse_duration = 1 / self.config.sampling_rate_FPGA
        sampling_rate_fft = 100e11
        t, signals = self.generate_encoded_pulse(pulse_heights, pulse_duration, values, sampling_rate_fft)
        #plt.plot(signals[0], label = 'before everything')
        #plt.legend()
        indexshift = len(t) // self.config.n_pulses // 2
        old_save_shift = np.zeros(indexshift)
        new_save_shift = np.empty(indexshift)
        for i in range(0, len(values), self.config.batchsize):
            signals_batch = signals[i:i + self.config.batchsize, :]
            flattened_signals_batch = signals_batch.reshape(-1)
            #plt.plot(flattened_signals_batch[:6*len(t)], label = 'before jitter')

            flattened_signals_batch = self.apply_jitter_to_pulse(t, flattened_signals_batch, jitter_shifts[self.config.n_pulses * i:self.config.n_pulses *(i + self.config.batchsize)])
            #plt.plot(flattened_signals_batch[:6*len(t)], label = 'after jitter')

            flattened_signals_batch = self.apply_bandwidth_filter(flattened_signals_batch, sampling_rate_fft)
            #roll array for better readability later at the detector
            ''' filtered_signals = np.roll(filtered_signals, indexshift)
            new_save_shift = filtered_signals[:indexshift]
            filtered_signals[: indexshift] = old_save_shift
            old_save_shift = new_save_shift'''

            flattened_signals_batch = flattened_signals_batch.reshape(self.config.batchsize, len(t))
            signals[i:i + self.config.batchsize, :] = flattened_signals_batch
            #plt.plot(filtered_signals[:len(t)], label = 'BW')
            '''plt.legend()
            Saver.save_plot('without_bandwidth')
            plt.plot(filtered_signals[:6*len(t)], label = 'ende')
            Saver.save_plot('with_bandwidth')'''
        return signals, t, jitter_shifts

    def eam_transmission(self, voltage_signal, optical_power, T1_dampening, peak_wavelength, t):
        """fastest? prob not: Calculate the transmission and power for all elements in the arrays."""
        # Create a mask where voltage_signal is less than 7.023775e-05 = x_max
        _, x_max = self.config.data.get_data_x_min_x_max('eam_transmission')            
        mask = voltage_signal < x_max

        # Compute interpolated values only for the values that meet the condition (<x_max)
        interpolated_values = self.get_interpolated_value(voltage_signal[mask], 'eam_transmission')

        # rest if the values: should be value at x_max (here 1.0)
        signal_over_threshold = self.get_interpolated_value(x_max, 'eam_transmission')

        # fill transmission array with signal_over_threshold
        transmission = np.full_like(voltage_signal, signal_over_threshold)
        # fill transmission array with interpolated_values where mask is True
        transmission[mask] = interpolated_values
       
        power_dampened = transmission * optical_power[:, None] / T1_dampening

        # Calculate the mean photon number
        energy_per_pulse = np.trapezoid(power_dampened, t, axis=1)
        calc_mean_photon_nr = energy_per_pulse / (constants.h * constants.c / peak_wavelength)

        return power_dampened, transmission, calc_mean_photon_nr, energy_per_pulse
    
    def fiber_attenuation(self, power_dampened):
        """Apply fiber attenuation to the power."""
        attenuation_factor = 10 ** (self.config.fiber_attenuation / 10)
        power_dampened = power_dampened * attenuation_factor
        return power_dampened
    
    def basis_selection_bob(self, power_dampened):
        """passive basis selection for Bob: in Beutel paper 85% without DLI"""
        power_dampened_x = power_dampened * self.config.p_z_bob
        power_dampened_z = power_dampened * (1 - self.config.p_z_bob)
        return power_dampened_x, power_dampened_z

    def delay_line_interferometer(self, power_dampened_x, t, peak_wavelength):
        assert self.config.fraction_long_arm <= 1
        eta_long = self.config.fraction_long_arm
        eta_short = 1 - eta_long

        # phase difference between the two arms: w*delta T_bin, w = 2pi*f = 2pic/lambda
        delta_t_bin = t[-1] / 2                             # Time bin duration (float64)
        frequency_symbol = constants.c / peak_wavelength     # Frequency of the symbol (float64)
        delta_phi = (2* np.pi * frequency_symbol) * delta_t_bin # Phase difference (radians) ALT:* constants.c / peak_wavelength

        # Time bin split point ( for late time bin start)
        split_point = len(t) // 2

        late_bin_ex_last = power_dampened_x[:-1, split_point:]
        early_bin_ex_first = power_dampened_x[1:, :split_point]
        whole_early_bin = power_dampened_x[:, :split_point]
        whole_late_bin = power_dampened_x[:, split_point:]

        # Calculate the interference term nth symbol and n+1th symbol (early-time and late-time bins)
        interference_term1 = (
            2 * np.sqrt(eta_long * eta_short) *
            np.multiply(np.sqrt(np.multiply(late_bin_ex_last, early_bin_ex_first)),np.cos(delta_phi[:-1]).reshape(-1,1)) #letzter Wert von delta_phi wird nicht verwendet weil zwischen 0 und n-1
        )
        
        # Interference term within the (n+1)th symbol (early-time and late-time bins)
        interference_term2 = (
            2 * np.sqrt(eta_short * eta_long) *
            np.multiply(np.sqrt(np.multiply(whole_early_bin, whole_late_bin)), np.cos(delta_phi).reshape(-1,1)) # alle n Werte für phi
        )

        # Pre-allocate arrays for the total power
        power_dampened_total = np.zeros((self.config.n_samples, len(t)))

        # Sum the contributions for total power (including interference)
        # For the nth and (n+1)th symbol:
        power_dampened_total[1:, :split_point] = (
            late_bin_ex_last + early_bin_ex_first + interference_term1  # nth and (n+1)th interference
        )

        # For the (n+1)th symbol:
        power_dampened_total[:, split_point:] = (
            whole_early_bin + whole_late_bin + interference_term2  # (n+1)th symbol interference
        )

        # for first row early bin
        power_dampened_total[0, :split_point] = power_dampened_x[0, :split_point]
        
        return power_dampened_total

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

    def choose_photons(self, power_dampened, transmission, t, peak_wavelength, calc_mean_photon_nr, energy_per_pulse, fixed_nr_photons=None):
        """Calculate and choose photons based on the transmission and jitter."""
        
        # Use Poisson distribution to get the number of photons
        nr_photons = fixed_nr_photons if fixed_nr_photons is not None else self.poisson_distr(calc_mean_photon_nr)
        all_time_max_nr_photons = max(nr_photons)
        sum_nr_photons_at_chosen = nr_photons.sum()
            
        # Find the indices where the number of photons is greater than 0
        non_zero_photons = nr_photons > 0
        index_where_photons = np.where(non_zero_photons)[0]
        nr_iterations_where_photons = len(index_where_photons)

        # Pre-allocate arrays for the photon properties        
        energy_per_photon = np.full((nr_iterations_where_photons, all_time_max_nr_photons), np.nan)
        wavelength_photons = np.full_like(energy_per_photon, np.nan)
        time_photons = np.full_like(energy_per_photon, np.nan)
        nr_photons = nr_photons[non_zero_photons] #delete rows where number of photons is 0

        # Calculate the normalized transmission
        norm_transmission = transmission / transmission.sum(axis=1, keepdims=True)

        # Generate the photon properties for each sample with non-zero photons
        for i, idx in enumerate(index_where_photons):
            photon_count = nr_photons[i]
            energy_per_photon[i, :photon_count] = energy_per_pulse[idx] / photon_count
            wavelength_photons[i, :photon_count] = (constants.h * constants.c) / energy_per_photon[i, :photon_count]
            time_photons[i, :photon_count] = self.config.rng.choice(t, size=photon_count, p=norm_transmission[idx]) #t ist konstant

        return wavelength_photons, time_photons, nr_photons, index_where_photons, all_time_max_nr_photons, sum_nr_photons_at_chosen
    
    def detector(self, t, wavelength_photons, time_photons, nr_photons, index_where_photons, all_time_max_nr_photons):
        """Simulate the detector process."""
        # Will the photons pass the detection efficiency?
        pass_detection = self.config.rng.choice([False, True], size=(len(index_where_photons), all_time_max_nr_photons), p=[1 - self.config.detector_efficiency, self.config.detector_efficiency])

        # Apply the detection efficiency to the photon properties
        wavelength_photons = np.where(pass_detection, wavelength_photons, np.nan)
        time_photons = np.where(pass_detection, time_photons, np.nan)

        # delete all Nan values
        valid_rows = ~np.isnan(wavelength_photons).all(axis=1)
        nr_photons_det = nr_photons[valid_rows] #nr_photons only for the ones we carry, rest 0
        index_where_photons_det = index_where_photons[valid_rows]
        wavelength_photons_det = wavelength_photons[valid_rows]
        time_photons_det = time_photons[valid_rows]

        # Last photon detected --> can next photon be detected? --> sort stuff
        sorted_indices = np.array([np.argsort(time) for time in time_photons_det])
        wavelength_photons = np.array([wavelength_photon[indices] for wavelength_photon, indices in zip(wavelength_photons_det, sorted_indices)])
        time_photons = np.array([time_photon[indices] for time_photon, indices in zip(time_photons_det, sorted_indices)])

        pulse_length = 1 / self.config.sampling_rate_FPGA
        # Initialize the adjusted times array with NaN values
        for i, time_p in enumerate(time_photons_det):
            # Initialize a list to store valid photons for this row
            last_valid_time = 0  # Start with time 0 to allow the first photon

            for j, time in enumerate(time_p):
                if np.isnan(time):
                    # If it's NaN, skip it (no photon detected)
                    continue
                else:
                    # If it's a valid photon, check if it's after the dead time
                    if time - last_valid_time < self.config.detection_time:
                        # If it's to close to the last valid time, skip it
                        time_photons_det[i, j] = np.nan  # Set the invalid photon to NaN
                        wavelength_photons_det[i, j] = np.nan  # Set the invalid wavelength to NaN
                    else:
                        last_valid_time = time - pulse_length*self.config.n_pulses  # Update the last valid time

        valid_rows = ~np.isnan(wavelength_photons_det).all(axis=1)
        nr_photons_det = nr_photons_det[valid_rows] #nr_photons only for the ones we carry, rest 0
        index_where_photons_det = index_where_photons_det[valid_rows]
        wavelength_photons_det = wavelength_photons_det[valid_rows]
        time_photons_det = time_photons_det[valid_rows]
        
        
        # jitter detector: timing jitter
        jitter_shifts = self.get_jitter('detector', size_jitter = time_photons_det.shape)

        #Apply jitter to non-NaN values, keeping NaNs unchanged
        time_photons_det[~np.isnan(time_photons_det)] += jitter_shifts[~np.isnan(time_photons_det)]
        #Create a mask for valid (non-NaN) entries
        valid_mask = ~np.isnan(time_photons_det)

        #Apply jitter until all values are within bounds (0 <= time_photons_det <= t[-1])
        while True:
            #Create the mask for out-of-bounds values (non-NaN)
            out_of_bounds_mask = (time_photons_det[valid_mask] < 0) | (time_photons_det[valid_mask] > t[-1])
            #If no out-of-bounds values, exit the loop
            if not np.any(out_of_bounds_mask):
                break
            #Remove the old jitter (subtract the previous jitter) for the out-of-bounds values
            time_photons_det[valid_mask][out_of_bounds_mask] -= jitter_shifts[valid_mask][out_of_bounds_mask]
            #Generate new jitter for the out-of-bounds values only
            jitter_shifts[valid_mask][out_of_bounds_mask] = self.get_jitter('detector', size=np.sum(out_of_bounds_mask))
            #Apply the new jitter to the out-of-bounds values
            time_photons_det[valid_mask][out_of_bounds_mask] += jitter_shifts[valid_mask][out_of_bounds_mask]

        return time_photons_det, wavelength_photons_det, nr_photons_det, index_where_photons_det
    
    def darkcount(self):
        """Calculate the number of dark count photons detected."""
        pulse_duration = 1 / self.config.sampling_rate_FPGA
        symbol_duration = pulse_duration * self.config.n_pulses
        num_dark_counts = self.config.rng.poisson(self.config.dark_count_frequency * symbol_duration, size=self.config.n_samples)
        dark_count_times = [np.sort(self.config.rng.uniform(0, symbol_duration, count)) if count > 0 else np.empty(0) for count in num_dark_counts]
        return dark_count_times, num_dark_counts
    
    def classificator_det_ind(self, timebins, decoy, time_photons_det, index_where_photons_det, is_decoy):
        detected_indices = np.array([np.digitize(time_photon, timebins) - 1 for time_photon in time_photons_det])  # 0 wenn early timebin, 1 wenn late timebin, -1 wenn nicht detektiert
        reduced_decoy = decoy[index_where_photons_det]
        if is_decoy == False:
            detected_indices = detected_indices[reduced_decoy == 0]
        else:
            detected_indices = detected_indices[reduced_decoy == 1]
        return detected_indices

    def classificator_z(self, basis, value, decoy, index_where_photons_det_z, detected_indices_z, is_decoy):
        # Z basis
        #Find indices where there is exactly one 0 (Z0 detection)
        Z0_indices_measured_reduced = np.where(np.sum(detected_indices_z == 0, axis=1) == 1)[0]
        #Find indices where there is exactly one 1 (Z1 detection)
        Z1_indices_measured_reduced = np.where(np.sum(detected_indices_z == 1, axis=1) == 1)[0]
        #get indices in original indexing
        Z0_indices_measured = index_where_photons_det_z[Z0_indices_measured_reduced]
        Z1_indices_measured = index_where_photons_det_z[Z1_indices_measured_reduced]
        #only keep those where alice sent Z: CHECK
        mask_Z0 = (basis[Z0_indices_measured] == 1) & (value[Z0_indices_measured] == 1)
        Z0_indices_checked = Z0_indices_measured[mask_Z0]
        mask_Z1 = (basis[Z1_indices_measured] == 1) & (value[Z1_indices_measured] == 0)
        Z1_indices_checked = Z1_indices_measured[mask_Z1]
        amount_Z_det = len(Z0_indices_checked) + len(Z1_indices_checked)
        if is_decoy == False:
            Z_sent = np.sum((basis == 1) & (decoy == 0))
        else:
            Z_sent = np.sum((basis == 1) & (decoy == 1))
        gain_Z = amount_Z_det / Z_sent
        return gain_Z, amount_Z_det
    
    def classificator_x(self, basis, value, decoy, index_where_photons_det_x, detected_indices_x, gain_Z, is_decoy):
        # X detection
        no_ones_rows_reduced = np.where(np.all((detected_indices_x != 1), axis=1))[0]
        no_ones_rows_full = index_where_photons_det_x[no_ones_rows_reduced]
        all_ind = np.arange(self.config.n_samples)
        remaining_indices = np.setdiff1d(all_ind, index_where_photons_det_x)
        XP_indices_measured = np.concatenate((no_ones_rows_full, remaining_indices))
        #only keep those where alice sent X+
        mask_x = basis[XP_indices_measured] == 0
        XP_indices_checked = XP_indices_measured[mask_x]
        amount_XP_det = gain_Z * len(XP_indices_checked)
        if is_decoy == False:
            XP_sent = np.sum((basis == 0) & (decoy == 0))
        else:
            XP_sent = np.sum((basis == 0) & (decoy == 1))
        gain_XP = amount_XP_det / XP_sent
        return gain_XP, amount_XP_det


    def classificator_error_cases(self, basis, value, index_where_photons_det_x, index_where_photons_det_z, total_detected_indices_x, total_detected_indices_z):
        # Error cases
        #Initialize a boolean array to track wrong detections (same length as number of detections)
        wrong_detection_mask_z = np.zeros(len(index_where_photons_det_z), dtype=bool)
        wrong_detection_mask_x = np.zeros(len(index_where_photons_det_x), dtype=bool)
        #Step 1: Check for wrong detections in the Z basis (Z0 and Z1)
        #Condition 1: Measure both bins in Z (both early and late detection)
        wrong_detection_mask_z |= (total_detected_indices_z[:, 0] == 1) & (total_detected_indices_z[:, 1] == 1)
        #Condition 2: Measure in late for Z0 (wrong detection)
        wrong_detection_mask_z |= (total_detected_indices_z[:, 0] == 1) & (basis == 0)
        #Condition 3: Measure in early for Z1 (wrong detection)
        wrong_detection_mask_z |= (total_detected_indices_z[:, 1] == 0) & (basis == 1)
        #get wrong_detections thruough correct indexing
        wrong_detections_z = index_where_photons_det_z[wrong_detection_mask_z]
        
        #Step 2: Check for wrong detections in the X+ basis
        #Condition 4: Early detection in X+ state after Z1Z0 !!!!!!!check still
        Z1_alice = np.where((basis == 1) & (value == 1))[0]  # Indices where Z1 was sent
        Z0_alice = np.where((basis == 1) & (value == 0))[0]  # Indices where Z0 was sent
        Z1_Z0_alice = Z0_alice[np.isin(Z0_alice - 1, Z1_alice)]  # Indices where Z1Z0 was sent
        wrong_detection_mask_x = np.isin(index_where_photons_det_z, Z1_Z0_alice) & (total_detected_indices_x[:, 0] == 0)
        #Condition 5: Early detection in X+ after Z1X+
        XP_alice = np.where((basis == 0)) # Indices where X+ was sent
        Z1_XP_alice = XP_alice[np.isin(XP_alice - 1, Z1_alice)]  # Indices where Z1Z0 was sent
        wrong_detection_mask_x |= (total_detected_indices_x[:, 0] == 0) & np.isin(index_where_photons_det_x, Z1_XP_alice)
        #Condition 6: Late detection in X+ after X+ sent
        wrong_detection_mask_x |= (total_detected_indices_x[:, 1] == 1) & (basis == 0) 
        #`wrong_detection_mask` is a boolean array where True indicates a wrong detection
        wrong_detections_x = index_where_photons_det_x[wrong_detection_mask_x]
        #Combine the wrong detections from both bases
        wrong_detections = np.concatenate([wrong_detections_x, wrong_detections_z])         # not sorted!
        wrong_detections = np.sort(wrong_detections)                                        # now sorted
        return wrong_detections


    def classificator(self, t, time_photons_det_x, index_where_photons_det_x, time_photons_det_z, index_where_photons_det_z, basis, value, decoy):
        """Classify time bins."""
        timebins = np.linspace(0, t[-1], self.config.n_pulses // 2)
        detected_indices_z_norm = self.classificator_det_ind(timebins, decoy, time_photons_det_z, index_where_photons_det_z, is_decoy = False)
        gain_Z_norm, amount_Z_det_norm = self.classificator_z(basis, value, decoy, index_where_photons_det_z, detected_indices_z_norm, is_decoy = False)

        detected_indices_z_dec = self.classificator_det_ind(timebins, decoy, time_photons_det_z, index_where_photons_det_z, is_decoy = True)
        gain_Z_dec, amount_Z_det_dec = self.classificator_z(basis, value, decoy, index_where_photons_det_z, detected_indices_z_dec, is_decoy = True)

        detected_indices_x_norm = self.classificator_det_ind(timebins, decoy, time_photons_det_x, index_where_photons_det_x, is_decoy = False)
        gain_XP_norm, amount_XP_det_norm = self.classificator_x(basis, value, decoy, index_where_photons_det_x, detected_indices_x_norm, gain_Z_norm, is_decoy = False)

        detected_indices_x_dec = self.classificator_det_ind(timebins, decoy, time_photons_det_x, index_where_photons_det_x, is_decoy = True)
        gain_XP_dec, amount_XP_det_dec = self.classificator_x(basis, value, decoy, index_where_photons_det_x, detected_indices_x_dec, gain_Z_dec, is_decoy = True)

        total_detected_indices_x = np.unique(np.concatenate(detected_indices_x_dec, detected_indices_x_norm))
        total_detected_indices_z = np.unique(np.concatenate(detected_indices_z_dec, detected_indices_z_norm))

        wrong_detections = self.classificator_error_cases(basis, value, index_where_photons_det_x, index_where_photons_det_z, total_detected_indices_x, total_detected_indices_z)

        total_amount_detections = amount_Z_det_norm + amount_Z_det_dec + amount_XP_det_norm + amount_XP_det_dec
        qber = len(wrong_detections) / total_amount_detections
        raw_key_rate = total_amount_detections / (t[-1] * self.config.n_samples)

        return wrong_detections, total_amount_detections, qber, raw_key_rate
    
    def initialize(self):
        plt.style.use(self.config.mlp)
        self.config.validate_parameters() #some checks if parameters are in valid ranges
        #calculate T1 dampening 
        lower_limit_t1, upper_limit_t1, tol_t1 = 0, 100, 1e-3
        T1_dampening = self.simulation_single.find_T1(lower_limit_t1, upper_limit_t1, tol_t1)
        if T1_dampening > (upper_limit_t1 - 10*tol_t1) or T1_dampening < (lower_limit_t1 + 10*tol_t1):
            raise ValueError(f"T1 dampening is very close to limit [{lower_limit_t1}, {upper_limit_t1}] with tolerance {tol_t1}")
        #print('T1_dampening at initialize end: ' +str(T1_dampening))

        #with simulated decoy state: calculate decoy height
        self.simulation_single.find_voltage_decoy(T1_dampening, lower_limit=-1, upper_limit=1.5, tol=1e-7)
        if self.voltage_decoy > (upper_limit_t1 - 10*tol_t1) or self.voltage_decoy < (lower_limit_t1 + 10*tol_t1):
            raise ValueError(f"T1 dampening is very close to limit [{lower_limit_t1}, {upper_limit_t1}] with tolerance {tol_t1}")
        #print('voltage at initialize end: ' + str(self.config.voltage))
        #print('Voltage_decoy at initialize end: ' + str(self.config.voltage_decoy))
        #print('Voltage_decoy_sup at initialize end: ' + str(self.config.voltage_decoy_sup))
        #print('Voltage_sup at initialize end: ' + str(self.config.voltage_sup))
        return T1_dampening
    
   