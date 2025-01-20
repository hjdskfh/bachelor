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
        #calc mean photon number
        #energy_per_pulse = np.array([np.trapz(calc_power_fiber[i], t_jitter[i])
        #                      for i in range(self.config.n_samples)])
        energy_per_pulse = np.trapz(calc_power_fiber, t_jitter, axis=1)
        calc_mean_photon_nr = energy_per_pulse / (constants.h * constants.c / peak_wavelength)
        #Poisson distribution to get amount of photons
        nr_photons = fixed_nr_photons if fixed_nr_photons is not None else self.poisson_distr(calc_mean_photon_nr)
        print(f"nr_photons: {nr_photons}")
        non_zero_photons = nr_photons > 0
        index_where_photons = np.where(non_zero_photons)[0]

        nr_iterations_where_photons = non_zero_photons.sum()
        #energy_per_photon = np.where(non_zero_photons, energy_per_pulse / nr_photons, 0)
        #wavelength_photons = np.where(non_zero_photons[:, None], (constants.h * constants.c) / energy_per_photon[:, None], 0)   
        norm_transmission = transmission / transmission.sum(axis=1, keepdims=True)

        

        start_time_choose = time.time()  # Record start time
        energy_per_photon = np.ones((nr_iterations_where_photons, max(nr_photons)))*-1
        wavelength_photons = np.ones_like(energy_per_photon)*-1
        time_photons = np.ones_like(energy_per_photon)*-1
        nr_photons = nr_photons[non_zero_photons] #shape: (nr_iterations_where_photons,)
        
        for i, index_phot in enumerate(index_where_photons):
                energy_per_photon[i] = energy_per_pulse[index_phot] / nr_photons[i]
                wavelength_photons[i] = (constants.h * constants.c) / energy_per_photon[i]
                time_photons[i] = self.config.rng.choice(t_jitter[index_phot], size=nr_photons[i], p=norm_transmission[index_phot])
        end_time_choose = time.time()  # Record end time
        execution_time = end_time_choose - start_time_choose  # Calculate execution time
        print(f"Execution time for choose: {execution_time:.9f} seconds for {self.config.n_samples} samples")
        

        '''start_time_choose_2 = time.time()  # Record start time
        energy_per_photon = []
        wavelength_photons = []
        time_photons = []
        for idx, photons in enumerate(nr_photons):
            if photons > 0:
                energy_per_photon_current = energy_per_pulse[idx] / photons
                energy_per_photon.append(energy_per_photon_current)
                wavelength_photons.append((constants.h * constants.c) / energy_per_photon_current)
                time_photons.append(self.config.rng.choice(t_jitter[idx], size=nr_photons[idx], p=norm_transmission[idx]))
        end_time_choose_2 = time.time()  # Record end time
        execution_time_2 = end_time_choose_2 - start_time_choose_2  # Calculate execution time
        print(f"Execution time var 2 for choose: {execution_time_2:.9f} seconds for {self.config.n_samples} samples")'''
        return wavelength_photons, time_photons, nr_photons, index_where_photons, non_zero_photons
    
    def detector(self, t_jitter, wavelength_photons, time_photons, nr_photons):
        """Simulate the detector process."""
        #will the photons pass the detection efficiency?
        pass_detection = self.config.rng.choice([False, True], size=(len(nr_photons), max(nr_photons)), p=[1 - self.config.detector_efficiency, self.config.detector_efficiency])
        print(f"pass_detection: {pass_detection.shape}")
        wavelength_photons_det = [wavelength_photon[pass_detection[i]] for i, wavelength_photon in enumerate(wavelength_photons)]
        time_photons_det = [time_photon[pass_detection[i]] for i, time_photon in enumerate(time_photons)]
        nr_photons_det = np.sum(pass_detection, axis=1)

        #last photon detected --> can next photon be detected? --> sort stuff
        sorted_indices = [np.argsort(time_photon) for time_photon in time_photons_det]
        wavelength_photons_det = [wavelength_photon[indices] for wavelength_photon, indices in zip(wavelength_photons_det, sorted_indices)]
        time_photons_det = [time_photon[indices] for time_photon, indices in zip(time_photons_det, sorted_indices)]
        # Compute differences with vectorized operations
        last_photon_time_minus_end_time = np.empty(self.config.n_samples)
        last_photon_time_minus_end_time[0] = 0
        last_photon_time_minus_end_time = [time_photon[-1] - t_jitter[i, -1] if len(time_photon) > 0 
                                           else last_photon_time_minus_end_time[i] - t_jitter[i, -1] 
                                           for i, time_photon in enumerate(time_photons_det)]
        time_diffs = [np.diff(time_photon, prepend=last_photon_time_minus_end_time[i]) for i, time_photon in enumerate(time_photons_det)]
        # Create valid indices
        valid_indices = [np.where(time_diff >= self.config.detection_time)[0] for time_diff in time_diffs]
        # Apply the mask to both timestamps and wavelengths
        valid_timestamps = [time_photon[indices] for time_photon, indices in zip(time_photons_det, valid_indices)]
        valid_wavelengths = [wavelength_photon[indices] for wavelength_photon, indices in zip(wavelength_photons_det, valid_indices)]
        valid_nr_photons = [len(indices) for indices in valid_indices]            
        
        t_detector_jittered = self.apply_jitter_to_t(t_jitter, name_jitter='detector')

        return valid_timestamps, valid_wavelengths, valid_nr_photons, t_detector_jittered
    
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
    
   