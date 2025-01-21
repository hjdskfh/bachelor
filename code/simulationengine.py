import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import splev
from scipy.fftpack import fft, ifft, fftfreq
from scipy import constants
from scipy.special import factorial
import time

from saver import Saver
from simulationsingle import SimulationSingle

class SimulationEngine:
    def __init__(self, config):
        self.config = config
        self.simulation_single = SimulationSingle(config)

    def get_interpolated_value_array(self, x_data, name):
        #calculate tck for which curve
        tck = self.config.data.get_data_array(x_data, name)
        return splev(x_data, tck)

    def random_laser_output(self, current_power, voltage_shift, current_wavelength):
        # Generate a random time within the desired range
        times = self.config.rng.uniform(0, 10, self.config.n_samples)
            
        # Use sinusoidal modulation for the entire array
        chosen_voltage = self.config.mean_voltage + 0.050 * np.sin(2 * np.pi * 1 * times)
        chosen_current = (self.config.mean_current + self.config.current_amplitude * np.sin(2 * np.pi * 1 * times)) * 1e3

        optical_power = self.get_interpolated_value_array(chosen_current, current_power)
        peak_wavelength = self.get_interpolated_value_array(chosen_current, current_wavelength) + self.get_interpolated_value_array(chosen_voltage, voltage_shift)
        return optical_power * 1e-3, peak_wavelength * 1e-9  # in W and m

    def generate_alice_choices(self, basis=None, value=None, decoy=None):
        """Generates Alice's choices for a quantum communication protocol."""
        # Generate arrays if any parameter is missing
        basis = basis or self.config.rng.choice([0, 1], size=self.config.n_samples, p=[1 - self.config.p_z_alice, self.config.p_z_alice])
        value = value or self.config.rng.choice([0, 1], size=self.config.n_samples, p=[1 - 0.5, 0.5])
        decoy = decoy or self.config.rng.choice([0, 1], size=self.config.n_samples, p=[1 - self.config.p_decoy, self.config.p_decoy])
        
        # Adjust value for array case if basis is 0
        if isinstance(basis, np.ndarray):
            value[basis == 0] = -1
        else:
            value = -1 if basis == 0 else value

        # Ensure outputs are arrays for non-single-value case
        basis = np.full(self.config.n_samples, basis)
        value = np.full(self.config.n_samples, value)
        decoy = np.full(self.config.n_samples, decoy)

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

    def encode_pulse(self, value):
        """Return a binary pattern for a square pulse based on the given value."""
        pattern = np.zeros((len(value), self.config.n_pulses), dtype=int)
        pattern[value == 1, 0] = 1
        pattern[value == 0, self.config.n_pulses // 2] = 1
        pattern[value == -1, 0] = 1
        pattern[value == -1, self.config.n_pulses // 2] = 1
        return pattern

    def generate_square_pulse(self, pulse_height, pulse_duration, sampling_rate_fft, pattern):
        """Generate a square pulse signal for a given height and pattern."""
        t = np.arange(0, self.config.n_pulses * pulse_duration, 1 / sampling_rate_fft, dtype=np.float64)
        repeating_square_pulses = np.full((len(pulse_height), len(t)), self.config.non_signal_voltage, dtype=np.float64)
        one_signal = len(t) // self.config.n_pulses
        indices = np.arange(len(t))
        for i, pattern in enumerate(pattern):
            for j, bit in enumerate(pattern):
                if bit == 1:
                    repeating_square_pulses[i, (indices // one_signal) == j] = pulse_height[i]
        return t, repeating_square_pulses
    
    def generate_encoded_pulse(self, pulse_height, pulse_duration, value, sampling_rate_fft):
        return self.generate_square_pulse(pulse_height, pulse_duration, sampling_rate_fft, pattern = self.encode_pulse(value))

    def apply_bandwidth_filter(self, signal, sampling_rate_fft):
        filtered_signal = np.empty_like(signal)
        freq_x = [0, self.config.bandwidth * 0.8, self.config.bandwidth, self.config.bandwidth * 1.2, sampling_rate_fft / 2]
        freq_y = [1, 1, 0.7, 0.01, 0.001]
        for i, signal in enumerate(signal):
            S_fourier = fft(signal)
            frequencies = fftfreq(len(signal), d=1 / sampling_rate_fft)
            S_fourier *= np.interp(np.abs(frequencies), freq_x, freq_y)
            filtered_signal[i] = np.real(ifft(S_fourier))
        return filtered_signal

    def apply_jitter_to_t(self, t, name_jitter):
        probabilities, jitter_values = self.config.data.get_probabilities(x_data=None, name='probabilities' + name_jitter)
        jitter_shifts = self.config.rng.choice(jitter_values, size=self.config.n_samples, p=probabilities)
        return t + jitter_shifts[:, None]

    def signal_bandwidth_jitter(self, basis, values, decoy):
        pulse_heights = self.get_pulse_height(basis, decoy)
        pulse_duration = 1 / self.config.sampling_rate_FPGA
        sampling_rate_fft = 100e11
        t, signals = self.generate_encoded_pulse(pulse_heights, pulse_duration, values, sampling_rate_fft)
        filtered_signal = self.apply_bandwidth_filter(signals, sampling_rate_fft)
        t_jittered = self.apply_jitter_to_t(t, name_jitter='laser')
        return filtered_signal, t_jittered, signals
    
    def eam_transmission_1_mean_photon_number_new(self, voltage_signal, t_jitter, optical_power, peak_wavelength, T1_dampening, basis, value, decoy):
        #from where is transmission 0?
        #print(f"voltage_signal: {(voltage_signal < 7.023775e-05).sum()}")
        #print(f"difference: {len(voltage_signal) - (voltage_signal < 7.023775e-05).sum()}")
        signal_over_threshold = self.simulation_single.get_interpolated_value_single(7.023775e-05, 'eam_transmission')
        transmission = np.where(voltage_signal < 7.023775e-05, 
                                self.get_interpolated_value_array(voltage_signal, 'eam_transmission', array = True), 
                                signal_over_threshold)

        power = transmission * optical_power
        power_dampened = power / T1_dampening
        energy_pp = np.trapz(power_dampened, t_jitter)

        calc_mean_photon_nr = energy_pp / (constants.h * constants.c / peak_wavelength)
        return power_dampened, calc_mean_photon_nr, energy_pp, transmission

    def eam_transmission(self, voltage_signal, optical_power, T1_dampening):
        """Calculate the transmission and power for all elements in the arrays."""
        signal_over_threshold = np.full_like(voltage_signal, self.simulation_single.get_interpolated_value_single(7.023775e-05, 'eam_transmission'))
        transmission = np.where(voltage_signal[0] < 7.023775e-05, 
                                self.get_interpolated_value_array(voltage_signal[0], 'eam_transmission'), 
                                signal_over_threshold)
        power = transmission * optical_power[:, None]
        power_dampened = power / T1_dampening
        return power_dampened, transmission
    
    def fiber_attenuation(self, power_dampened):
        """Apply fiber attenuation to the power."""
        attenuation_factor = 10 ** (self.config.fiber_attenuation / 10)
        calc_power_fiber = power_dampened * attenuation_factor
        return calc_power_fiber
    
    def generate_bob_choices(self, basis_bob = None, value_bob = None, decoy_bob = None, fixed = None):
        """Generates Bob's choices for a quantum communication protocol."""
        # Bob's basis choice is random, muss nich 50:50
        if not fixed:
            basis_bob = self.config.rng.choice([0, 1], size=self.config.n_samples, p=[1 - self.config.p_z_bob, self.config.p_z_bob])
            value_bob = self.config.rng.choice([0, 1], size=self.config.n_samples, p=[1 - 0.5, 0.5])
            decoy_bob = self.config.rng.choice([0, 1], size=self.config.n_samples, p=[1 - self.config.p_decoy, self.config.p_decoy])
        value_bob[basis_bob == 0] = -1
        return basis_bob, value_bob, decoy_bob
    
    def poisson_distr(self, calc_value):
        """Calculate the number of photons based on a Poisson distribution."""
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

    def choose_photons(self, calc_power_fiber, transmission, t_jitter, peak_wavelength, fixed_nr_photons=None):
        """Calculate and choose photons based on the transmission and jitter."""
        # Calculate the mean photon number
        energy_per_pulse = np.trapz(calc_power_fiber, t_jitter, axis=1)
        calc_mean_photon_nr = energy_per_pulse / (constants.h * constants.c / peak_wavelength)
        
        # Use Poisson distribution to get the number of photons
        nr_photons = fixed_nr_photons if fixed_nr_photons is not None else self.poisson_distr(calc_mean_photon_nr)
        print(f"nr_photons: {nr_photons}")
        
        # Find the indices where the number of photons is greater than 0
        non_zero_photons = nr_photons > 0
        index_where_photons = np.where(non_zero_photons)[0]
        nr_iterations_where_photons = len(index_where_photons)

        # Pre-allocate arrays for the photon properties
        all_time_max_nr_photons = max(nr_photons)
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
            time_photons[i, :photon_count] = self.config.rng.choice(t_jitter[idx], size=photon_count, p=norm_transmission[idx])

        return wavelength_photons, time_photons, nr_photons, index_where_photons, all_time_max_nr_photons
    
    def detector(self, t_jitter, wavelength_photons, time_photons, nr_photons, index_where_photons, all_time_max_nr_photons):
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
        #print(f"wavelength_photons after del nan: {wavelength_photons.shape}")
        time_photons_det = time_photons[valid_rows]

        # Last photon detected --> can next photon be detected? --> sort stuff
        sorted_indices = [np.argsort(time) for time in time_photons_det]
        wavelength_photons = [wavelength_photon[indices] for wavelength_photon, indices in zip(wavelength_photons_det, sorted_indices)]
        time_photons = [time_photon[indices] for time_photon, indices in zip(time_photons_det, sorted_indices)]
        #print(f"in fct time_photons: {time_photons}")
        '''# Compute differences with vectorized operations
        last_photon_time_minus_end_time = np.empty(self.config.n_samples)
        last_photon_time_minus_end_time[0] = 0
        last_photon_time_minus_end_time = [time_p[-1] - t_jitter[i, -1] if len(time_p) > 0 
                                        else last_photon_time_minus_end_time[i] - t_jitter[i, -1] 
                                        for i, time_p in enumerate(time_photons)]
        time_diffs = [np.diff(time_photon, prepend=last_photon_time_minus_end_time[i]) for i, time_photon in enumerate(time_photons)]
        print(f"time_diffs: {time_diffs}")

        # Create valid indices
        valid_indices = [np.where(time_diff >= self.config.detection_time)[0] for time_diff in time_diffs]
        print(f"valid_indices: {valid_indices}")'''

        '''last_index = 0
        simulation_time = 0
        last_photon_time_minus_end_time = 0 
        time_diffs = np.empty_like(time_photons)
        for k, time_phot in enumerate(time_photons):
            index_w = index_where_photons[k]
            index_skip = index_w - last_index
            if index_skip > 1:
                for l in range(index_skip):
                    last_photon_time_minus_end_time -= t_jitter[last_index + l, -1]
            last_photon_time_minus_end_time = time_photons[k][-1] - t_jitter[index_w, -1]
            last_index = index_w
            time_diff = np.diff(time_phot, prepend=last_photon_time_minus_end_time)
        
        time_diffs = [np.diff(time_photon, prepend=last_photon_time_minus_end_time[i]) for i, time_photon in enumerate(time_photons)]
        valid_indices = time_diffs >= self.config.detection_time
            '''
        print(f"time_photons_det vor {np.count_nonzero(~np.isnan(time_photons_det))}")
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
                    #print(f"differenz{time - last_valid_time}")
                    if time - last_valid_time < self.config.detection_time:
                        # If it's to close to the last valid time, skip it
                        time_photons_det[i, j] = np.nan  # Set the invalid photon to NaN
                        #print(f"delete time{time}")
                        wavelength_photons_det[i, j] = np.nan  # Set the invalid wavelength to NaN
                    else:
                        last_valid_time = time - t_jitter[i, -1]

        print(f"time_photons_det nach {np.count_nonzero(~np.isnan(time_photons_det))}")


        '''
        # Initialize the array
        last_photon_time_minus_end_time = np.zeros(self.config.n_samples)
        last_photon_time_minus_end_time[0] = 0  # First value is zero or some other initial value

        # Calculate the last photon time minus the end time for each time_p
        for i, time_p in enumerate(time_photons):
            if len(time_p) > 0:
                # Ignore NaN values by filtering out NaNs and then using the last valid time
                time_p_no_nan = [t for t in time_p if not np.isnan(t)]
                if time_p_no_nan:  # Ensure there's at least one valid time
                    last_photon_time_minus_end_time[i] = time_p_no_nan[-1] - t_jitter[i, -1]
            else:
                # Handle empty time_p
                last_photon_time_minus_end_time[i] = last_photon_time_minus_end_time[i] - t_jitter[i, -1]
            

        # Calculate time differences with dead time (detection time) consideration
        time_diffs = [np.diff(time_photon, prepend=last_photon_time_minus_end_time[i]) for i, time_photon in enumerate(time_photons)]

        # Create valid indices where the time difference is larger than the dead time (detection_time)
        valid_indices = [
            np.where(time_diff >= self.config.detection_time)[0]  # Indices where the time difference meets the detection threshold
            for time_diff in time_diffs
        ]'''

        '''# Use np.nanmax to get the last valid time in each row (for the last photon), time_photons hat schon mind 1 nicht NaN value
        last_photon_times = np.nanmax(time_photons, axis=1)

        # Adjust last photon time by subtracting the jitter time (t_jitter) for each row
        last_photon_time_minus_end_time = last_photon_times - np.nanmax(t_jitter, axis=1)

        # Calculate time differences, using np.diff and prepending the adjusted last photon time
        time_diffs = np.diff(time_photons, axis=1, prepend=last_photon_time_minus_end_time[:, np.newaxis])

        # Vectorized operation to calculate valid indices where time difference is greater than detection_time
        valid_indices = time_diffs >= self.config.detection_time

        # Now, valid_indices will be a boolean array where True means the photon is allowed
        # You can use this mask to filter `time_photons` and any other related data (like wavelengths)

        # Example usage of valid indices to filter the timestamps (ignoring NaN values)
        valid_photons = time_photons[valid_indices]'''
        valid_rows = ~np.isnan(wavelength_photons_det).all(axis=1)
        nr_photons_det = nr_photons_det[valid_rows] #nr_photons only for the ones we carry, rest 0
        index_where_photons_det = index_where_photons_det[valid_rows]
        wavelength_photons_det = wavelength_photons_det[valid_rows]
        time_photons_det = time_photons_det[valid_rows]


        t_detector_jittered = self.apply_jitter_to_t(t_jitter, name_jitter='detector')

        return time_photons_det, wavelength_photons_det, nr_photons_det, index_where_photons_det, t_detector_jittered
    
    def darkcount(self):
        """Calculate the number of dark count photons detected."""
        pulse_duration = 1 / self.config.sampling_rate_FPGA
        symbol_duration = pulse_duration * self.config.n_pulses
        num_dark_counts = self.config.rng.poisson(self.config.dark_count_frequency * symbol_duration, size=self.config.n_samples)
        dark_count_times = [np.sort(self.config.rng.uniform(0, symbol_duration, count)) if count > 0 else np.empty(0) for count in num_dark_counts]
        return dark_count_times, num_dark_counts
    
    def classificator_old(self, t, valid_timestamps, valid_wavelengths, valid_nr_photons, value):
        if valid_nr_photons > 0:
            # classify timebins
            timebins = np.linspace(0, t[-1], self.config.n_pulses)
            detected_indices = np.digitize(valid_timestamps, timebins) - 1
            pattern = self.encode_pulse(value)
            # All photons are classified as correct (True)
            if np.all(pattern[detected_indices] == 1):
                return np.ones(valid_nr_photons, dtype=bool) 
            else:  # Any photon in a wrong bin gets classified as incorrect (False)
                return np.zeros(valid_nr_photons, dtype=bool)
        else: # If no valid photons, return an empty array
            return np.empty(0) 

    def classificator(self, t, valid_timestamps, valid_wavelengths, valid_nr_photons, values):
        """Classify time bins."""
        timebins = np.linspace(0, t[-1], self.config.n_pulses)
        detected_indices = [np.digitize(valid_timestamp, timebins) - 1 for valid_timestamp in valid_timestamps]
        patterns = self.encode_pulse(values)
        mask_classifications = np.zeros_like(patterns, dtype=bool)  # Initialize mask of the same shape as patterns
        for i, indices in enumerate(detected_indices):  # Fill mask based on detected_indices
            mask_classifications[i, indices] = True
        # Check where the mask is True and the values in patterns are 1
        classifications = np.all(np.where(mask_classifications, patterns == 1, True), axis=1)
        return classifications
    
    def initialize(self):
        plt.style.use(self.config.mlp)
        # check if voltage input makes sense
        if (self.config.non_signal_voltage - self.config.voltage) != -1:
            raise ValueError(f"The difference between non_signal_voltage and voltage is not 1")
        if self.config.voltage != self.config.voltage_decoy or self.config.voltage != self.config.voltage_sup or self.config.voltage_decoy_sup:
            raise ValueError(f"The starting voltage values are not the same") 
        #calculate T1 dampening 
        lower_limit_t1, upper_limit_t1, tol_t1 = 0, 100, 1e-3
        T1_dampening = self.simulation_single.find_T1(lower_limit_t1, upper_limit_t1, tol_t1)
        if T1_dampening > (upper_limit_t1 - 10*tol_t1) or T1_dampening < (lower_limit_t1 + 10*tol_t1):
            raise ValueError(f"T1 dampening is very close to limit [{lower_limit_t1}, {upper_limit_t1}] with tolerance {tol_t1}")
        #print('T1_dampening at initialize end: ' +str(T1_dampening))
        #T1_dampening_in_dB = 10* np.log(1/T1_dampening) 
        #print('T1_dampening at initialize end in dB: ' + str(T1_dampening_in_dB))

        #with simulated decoy state: calculate decoy height
        self.simulation_single.find_voltage_decoy(T1_dampening, lower_limit=-1, upper_limit=1.5, tol=1e-7)
        #print('voltage at initialize end: ' + str(self.config.voltage))
        #print('Voltage_decoy at initialize end: ' + str(self.config.voltage_decoy))
        #print('Voltage_decoy_sup at initialize end: ' + str(self.config.voltage_decoy_sup))
        #print('Voltage_sup at initialize end: ' + str(self.config.voltage_sup))
        return T1_dampening
    
   