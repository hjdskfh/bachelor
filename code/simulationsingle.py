import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import splev
from scipy.fftpack import fft, ifft, fftfreq
from scipy import constants
from scipy.special import factorial



class SimulationSingle:
    def __init__(self, config):
        self.config = config   
    
    def get_interpolated_value_single(self, x_data, name):
        #calculate tck for which curve
        tck = self.config.data.get_data(x_data, name)
        return splev(x_data, tck)
    
    def random_laser_output_single(self, current_power, voltage_shift, current_wavelength, fixed = None):
        # Use fixed values for a single instance
        time = self.config.rng.uniform(0, 10)
        chosen_voltage = self.config.mean_voltage if fixed else self.config.mean_voltage + 0.050 * np.sin(2 * np.pi * 1 * time)
        chosen_current = (self.config.mean_current if fixed else (self.config.mean_current + self.config.current_amplitude * np.sin(2 * np.pi * 1 * time))) * 1e3
        optical_power = self.get_interpolated_value_single(chosen_current, current_power)
        peak_wavelength = self.get_interpolated_value_single(chosen_current, current_wavelength) + self.get_interpolated_value_single(chosen_voltage, voltage_shift)
        return optical_power * 1e-3, peak_wavelength * 1e-9  # in W and m
        
    def generate_alice_choices_single(self, basis=None, value=None, decoy=None):
        # Generate single values if any parameter is missing
        basis = basis or self.config.rng.choice([0, 1], p=[1 - self.config.p_z_alice, self.config.p_z_alice])
        value = value or self.config.rng.choice([0, 1], p=[1 - 0.5, 0.5])
        decoy = decoy or self.config.rng.choice([0, 1], p=[1 - self.config.p_decoy, self.config.p_decoy])
        # Adjust value if basis is 0
        if basis == 0:
            value = -1
        return basis, value, decoy
    
    def get_pulse_height_single(self, basis, decoy):
        """
        Determine the pulse height based on the basis and decoy state.
        Args:
            basis (int): 0 for X-basis (superposition), 1 for Z-basis (computational).
            decoy (int): 0 for standard pulse, 1 for decoy pulse.
        Returns:
            float: The height of the square pulse in volts.
        """
        if decoy == 0:  # Non-decoy
            return self.config.voltage_sup if basis == 0 else self.config.voltage
        else:  # Decoy
            return self.config.voltage_decoy_sup if basis == 0 else self.config.voltage_decoy
        
    def encode_pulse_single(self, value):
        """Return a binary pattern for a square pulse based on the given value."""
        pattern = np.zeros(self.config.n_pulses, dtype=int)
        if value == 1:  # "1000"
            pattern[0] = 1
        elif value == 0:  # "0010"
            pattern[self.config.n_pulses // 2] = 1
        elif value == -1:  # "1010"
            pattern[0] = 1
            pattern[self.config.n_pulses // 2] = 1
        return pattern
    
    def generate_square_pulse_single(self, pulse_height, pulse_duration, pattern, sampling_rate_fft):
        """Generate a square pulse signal for a given height and pattern."""
        t = np.arange(0, self.config.n_pulses * pulse_duration, 1 / sampling_rate_fft, dtype = np.float64)
        repeating_square_pulse = np.full(len(t), self.config.non_signal_voltage, dtype = np.float64) #np.full(len(t), 0, dtype = np.float64)#np.zeros(len(t)) #np.full(len(t), 0) #np.full(len(t), self.config.non_signal_voltage) #np.zeros(len(t)) #
        one_signal = len(t) // self.config.n_pulses
        indices = np.arange(len(t))
        for i, bit in enumerate(pattern):
            if bit == 1:
                repeating_square_pulse[(indices // one_signal) == i] = pulse_height
        return t, repeating_square_pulse
    
    def generate_encoded_pulse_single(self, pulse_height, pulse_duration, value, sampling_rate_fft):
        pattern = self.encode_pulse_single(value)
        return self.generate_square_pulse_single(pulse_height, pulse_duration, pattern, sampling_rate_fft)
    
    def apply_bandwidth_filter_single(self, signal, sampling_rate_fft):
        """Apply a frequency-domain filter to a signal."""
        S_fourier = fft(signal)
        frequencies = fftfreq(len(signal), d=1 / sampling_rate_fft)

        freq_x = [0, self.config.bandwidth * 0.8, self.config.bandwidth, self.config.bandwidth * 1.2, sampling_rate_fft / 2]
        freq_y = [1, 1, 0.7, 0.01, 0.001]  # Smooth drop-off

        np.multiply(S_fourier, np.interp(np.abs(frequencies), freq_x, freq_y), out=S_fourier)
        return np.real(ifft(S_fourier))
    
    def apply_jitter_single(self, t, name_jitter):
        """Add jitter to the time array."""
        probabilities, jitter_values = self.config.data.get_probabilities(x_data=None, name='probabilities' + name_jitter)
        jitter_shift = self.config.rng.choice(jitter_values, p=probabilities)
        return t + jitter_shift

    def signal_bandwidth_jitter_single(self, basis, value, decoy):
        """Process signal with bandwidth limitation and apply jitter."""
        pulse_height = self.get_pulse_height_single(basis, decoy)
        pulse_duration = 1 / self.config.sampling_rate_FPGA
        sampling_rate_fft = 100e11
        t, signal = self.generate_encoded_pulse_single(pulse_height, pulse_duration, value, sampling_rate_fft)
        filtered_signal = self.apply_bandwidth_filter_single(signal, sampling_rate_fft)
        t_jittered = self.apply_jitter_single(t, name_jitter = 'laser')
        return filtered_signal, t_jittered, signal
    
    def eam_transmission_single(self, voltage_signal, optical_power, T1_dampening):
        #include the eam_voltage and multiply with calculated optical power from laser
        power = np.empty(len(voltage_signal))
        transmission = np.empty(len(voltage_signal))

        for i in range(len(voltage_signal)):
            if voltage_signal[i] < 0:
                transmission[i] = self.get_interpolated_value_single(voltage_signal[i], 'eam_transmission')
            else:
                transmission[i] = self.get_interpolated_value_single(0, 'eam_transmission')
            power[i] = transmission[i] * optical_power        

        power_dampened = power / T1_dampening
        return power_dampened, transmission
    
    def fiber_attenuation_single(self, power_dampened):
        """Apply fiber attenuation to the power."""
        attenuation_factor = 10 ** (self.config.fiber_attenuation / 10)
        power_dampened = power_dampened * attenuation_factor
        return power_dampened
    
    def find_T1(self, lower_limit, upper_limit, tol):
        # Generate a single instance for T1 dampening calculation
        optical_power, peak_wavelength = self.random_laser_output_single('current_power', 'voltage_shift', 'current_wavelength', fixed=True)
        basis, value, decoy = self.generate_alice_choices_single(basis = 1, value = 1, decoy = 0)

        while upper_limit - lower_limit > tol:
            T1_dampening = (lower_limit + upper_limit) / 2
            voltage_signal, t_jitter, _ = self.signal_bandwidth_jitter_single(basis, value, decoy)
            power_dampened, _ = self.eam_transmission_single(voltage_signal, optical_power, T1_dampening)
            energy_pp = np.trapezoid(power_dampened, t_jitter)
            calc_mean_photon_nr = energy_pp / (constants.h * constants.c / peak_wavelength)

            if calc_mean_photon_nr < self.config.mean_photon_nr:
                upper_limit = T1_dampening
            else:
                lower_limit = T1_dampening

        T1_dampening = (lower_limit + upper_limit) / 2
        return T1_dampening
    
    def _set_voltage(self, optical_power, peak_wavelength, lower_limit, upper_limit, tol, target_mean, voltage_type, T1_dampening, basis, value, decoy):
        """Helper method to perform binary search and set the voltage."""
        while upper_limit - lower_limit > tol:
            voltage = (lower_limit + upper_limit) / 2
            
            # Dynamically set the voltage attribute based on voltage_type
            setattr(self.config, voltage_type, voltage)

            voltage_signals, t_jitter, _ = self.signal_bandwidth_jitter_single(*self.generate_alice_choices_single(basis, value, decoy))
            power_dampened, _ = self.eam_transmission_single(voltage_signals, optical_power, T1_dampening)
            energy_pp = np.trapezoid(power_dampened, t_jitter)
            calc_mean_photon_nr = energy_pp / (constants.h * constants.c / peak_wavelength)

            if calc_mean_photon_nr > target_mean:
                upper_limit = voltage  # Reduce upper bound
            else:
                lower_limit = voltage  # Increase lower bound

        final_voltage = (lower_limit + upper_limit) / 2
        setattr(self.config, voltage_type, final_voltage)  # Set the final voltage dynamically
        return final_voltage
    
    def find_voltage_decoy(self, T1_dampening, lower_limit, upper_limit, tol):
        """Find the appropriate voltage values for decoy and non-decoy states using binary search."""
        # Store original limits to reset later
        store_lower_limit = lower_limit
        store_upper_limit = upper_limit

        # Set the optical power and peak wavelength for the simulation
        optical_power, peak_wavelength = self.random_laser_output_single('current_power', 'voltage_shift', 'current_wavelength', fixed=True)

        # Find voltage for 000 -> 1010 non-decoy state
        self.config.voltage_sup = self._set_voltage(optical_power, peak_wavelength, lower_limit, upper_limit, tol, 
                                             self.config.mean_photon_nr, "voltage_sup", T1_dampening, 
                                             basis = 0, value = 0, decoy = 0)
        # Check if voltage_sup is within the limits and tolerances
        if self.config.voltage_sup > (upper_limit - 10 * tol) or self.config.voltage_sup < (lower_limit + 10 * tol):
            raise ValueError(f"Voltage for non-decoy state (voltage_sup) is very close to limit [{store_lower_limit}, {store_upper_limit}] with tolerance {tol}")

        # Reset limits for next voltage calculation
        lower_limit, upper_limit = store_lower_limit, store_upper_limit

        # Find voltage for 111 -> 1000 decoy state
        self.config.voltage_decoy = self._set_voltage(optical_power, peak_wavelength, lower_limit, upper_limit, tol, 
                                               self.config.mean_photon_decoy, "voltage_decoy", T1_dampening, 
                                               basis = 1, value = 1, decoy = 1)
        # Check if voltage_sup is within the limits and tolerances
        if self.config.voltage_decoy > (upper_limit - 10 * tol) or self.config.voltage_sup < (lower_limit + 10 * tol):
            raise ValueError(f"Voltage for decoy state (voltage_decoy) is very close to limit [{store_lower_limit}, {store_upper_limit}] with tolerance {tol}")


        # Reset limits for next voltage calculation
        lower_limit, upper_limit = store_lower_limit, store_upper_limit

        # Find voltage for 001 -> 1010 decoy state
        self.config.voltage_decoy_sup = self._set_voltage(optical_power, peak_wavelength, lower_limit, upper_limit, tol, 
                                                   self.config.mean_photon_decoy, "voltage_decoy_sup", T1_dampening, 
                                                   basis = 0, value = 0, decoy = 1)
        # Check if voltage_sup is within the limits and tolerances
        if self.config.voltage_decoy > (upper_limit - 10 * tol) or self.config.voltage_sup < (lower_limit + 10 * tol):
            raise ValueError(f"Voltage for decoy sup state (voltage_decoy_sup) is very close to limit [{store_lower_limit}, {store_upper_limit}] with tolerance {tol}")

        return None