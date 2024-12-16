import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import splev
from scipy.fftpack import fft, ifft, fftfreq
from scipy import constants

class SimulationEngine:
    def __init__(self, config):
        self.config = config       

    def get_interpolated_value(self, x_data, name):
        #calculate tck for which curve
        tck = self.config.data.get_data(x_data, name)
        return splev(x_data, tck)
    
    def random_laser_output_fixed(self, current_power, voltage_shift, current_wavelength):
        # Generate a random time within the desired range
        time = np.random.uniform(0, 10) 
        
        # Calculate voltage and current based on this time of laserdiode and heater
            #voltage_heater = 1 in V, voltage_amplitude = 0.050 in V, voltage_frequency = 1
        chosen_voltage = self.config.mean_voltage
    
            #current_laserdiode = 0.08 in A, current_amplitude = 0.020 in A, current_frequency = 1
        chosen_current = (self.config.mean_current)* 1e3 #damit in mA

        optical_power = self.get_interpolated_value(chosen_current, current_power)
        peak_wavelength = self.get_interpolated_value(chosen_current, current_wavelength) + self.get_interpolated_value(chosen_voltage, voltage_shift)
        return optical_power * 1e-3, peak_wavelength * 1e-9  #in W and m

    def random_laser_output(self, current_power, voltage_shift, current_wavelength):
        # Generate a random time within the desired range
        time = np.random.uniform(0, 10) 
        
        # Calculate voltage and current based on this time of laserdiode and heater
            #voltage_heater = 1 in V, voltage_amplitude = 0.050 in V, voltage_frequency = 1
        chosen_voltage = self.config.mean_voltage + 0.050 * np.sin(2 * np.pi * 1 * time)  
    
            #current_laserdiode = 0.08 in A, current_amplitude = 0.020 in A, current_frequency = 1
        chosen_current = (self.config.mean_current + self.config.current_amplitude * np.sin(2 * np.pi * 1 * time) )* 1e3 #damit in mA

        optical_power = self.get_interpolated_value(chosen_current, current_power)
        peak_wavelength = self.get_interpolated_value(chosen_current, current_wavelength) + self.get_interpolated_value(chosen_voltage, voltage_shift)
        return optical_power * 1e-3, peak_wavelength * 1e-9  #in W and m
    
    def generate_alice_choices_fixed(self, basis, value, decoy):

        # Basis and value choices
        basis = np.array([basis])
        value = np.array([value])
        value[basis == 0] = -1  # Mark X basis values

        decoy = np.array([decoy])
        return (basis, value, decoy)

    def generate_alice_choices(self):

        """Generates Alice's choices for a quantum communication protocol, including 
        basis selection, value encoding, decoy states, and does the fft.

        Args:
        n_pulses: Elektronik kann so viele channels
        symbol_length: iterations per symbol
        p_z_alice: Probability of Alice choosing the Z basis.
        p_z_1: Probability of encoding a '1' in the Z basis.
        p_decoy: Probability of sending a decoy state.
        """

        # Basis and value choices
        basis = np.random.choice([0, 1], size = 1, p=[1-self.config.p_z_alice, self.config.p_z_alice]) # Randomly selects whether each pulse block is prepared in the Z-basis (0) or the X-basis (1) with a bias controlled by p_z_alice
        value = np.random.choice([0, 1], size = 1, p=[1-self.config.p_z_1, self.config.p_z_1]) #Assigns logical values (0 or 1) to the pulses with probabilities defined by p_z_1. If the basis is 0 (X-basis), the values are set to -1 to differentiate them.   
        value[basis == 0] = -1  # Mark X basis values

        # Decoy state selection
        decoy = np.random.choice([0, 1], size = 1, p=[1-self.config.p_decoy, self.config.p_decoy])

        return (basis, value, decoy)
    
    def get_pulse_height(self, basis, decoy):
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

    def encode_pulse(self, value):
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

    def generate_square_pulse(self, pulse_height, pulse_duration, pattern, sampling_rate_fft):
        """Generate a square pulse signal for a given height and pattern."""
        t = np.arange(0, self.config.n_pulses * pulse_duration, 1 / sampling_rate_fft)
        repeating_square_pulse = np.full(len(t), self.config.non_signal_voltage)
        one_signal = len(t) // self.config.n_pulses

        for i, bit in enumerate(pattern):
            if bit == 1:
                repeating_square_pulse[i * one_signal:(i + 1) * one_signal] = pulse_height

        return t, repeating_square_pulse
    
    def generate_encoded_pulse(self, pulse_height, pulse_duration, value, sampling_rate_fft):
        pattern = self.encode_pulse(value)
        return self.generate_square_pulse(pulse_height, pulse_duration, pattern, sampling_rate_fft)

    def apply_bandwidth_filter(self, signal, sampling_rate_fft):
        """Apply a frequency-domain filter to a signal."""
        S_f = fft(signal)
        frequencies = fftfreq(len(signal), d=1 / sampling_rate_fft)

        freq_x = [0, self.config.bandwidth * 0.8, self.config.bandwidth, self.config.bandwidth * 1.2, sampling_rate_fft / 2]
        freq_y = [1, 1, 0.7, 0.01, 0.001]  # Smooth drop-off
        S_filtered = S_f * np.interp(np.abs(frequencies), freq_x, freq_y)
        return np.real(ifft(S_filtered))

    def apply_jitter(self, t):
        """Add jitter to the time array."""
        probabilities, jitter_values = self.config.data.get_data(x_data=None, name='probabilities')
        jitter_shift = np.random.choice(jitter_values, p=probabilities)
        return t + jitter_shift

    def signal_bandwidth_jitter(self, basis, value, decoy):
        """Process signal with bandwidth limitation and apply jitter."""
        pulse_height = self.get_pulse_height(basis, decoy)
        pulse_duration = 1 / self.config.freq
        sampling_rate_fft = 100e11
        t, signal = self.generate_encoded_pulse(pulse_height, pulse_duration, value, sampling_rate_fft)
        filtered_signal = self.apply_bandwidth_filter(signal, sampling_rate_fft)
        t_jittered = self.apply_jitter(t)
        return filtered_signal, t_jittered, signal

    def eam_transmission_1_mean_photon_number(self, voltage_signal, t_jitter, optical_power, peak_wavelength, T1_dampening, basis, value, decoy):
        #include the eam_voltage and multiply with calculated optical power from laser
        power = np.empty(len(voltage_signal))
        transmission = np.empty(len(voltage_signal))

        for i in range(len(voltage_signal)):
            if voltage_signal[i] < 0:
                transmission[i] = self.get_interpolated_value(voltage_signal[i], 'eam_transmission')
            else:
                transmission[i] = self.get_interpolated_value(0, 'eam_transmission')
            power[i] = transmission[i] * optical_power        

        power_dampened = power / T1_dampening
        energy_pp = np.trapz(power_dampened, t_jitter)

        '''plt.plot(t_jitter * 1e9, power, label = 'after eam') #fehlt optical power
        plt.title("Power of Square Signal with Bandwidth Limitation with 1e-11 jitter")
        plt.xlabel("Time in ns")
        plt.ylabel("Power in W")
        plt.legend()
        plt.grid(True)
        #save_plot('power_after_transmission_with_4_GHz_bandwidth_and_1e-11s_jitter')
        plt.show()'''

        calc_mean_photon_nr = energy_pp / (constants.h*constants.c/peak_wavelength)
        return power_dampened, calc_mean_photon_nr, energy_pp, transmission
        
    def eam_transmission_2_choose_photons(self, calc_mean_photon_nr, energy_pp, transmission, t_jitter):
        #Poisson distribution to get amount of photons

        # Define a range of values (e.g., from 0 to an upper bound like mean + 5 standard deviations)
        upper_bound = int(calc_mean_photon_nr + 5 * np.sqrt(calc_mean_photon_nr))
        x = np.arange(0, upper_bound + 1)

        # Compute Poisson probabilities
        probabilities_array_poisson = np.exp(-calc_mean_photon_nr) * (calc_mean_photon_nr ** x) / factorial(x)

        # Normalize probabilities (optional, as Poisson probabilities already sum to ~1)
        probabilities_array_poisson = probabilities_array_poisson / probabilities_array_poisson.sum()
                
        #choose amount of photons and calculate energy per Photons and initialize Wavelength Arrays
        nr_photons = np.random.choice(x, p = probabilities_array_poisson)
        if nr_photons != 0:
            energy_per_photon = energy_pp / nr_photons
            wavelength_photons = np.zeros(nr_photons)

            #choose time for photons
            norm_transmission = transmission / transmission.sum()
            time_photons = np.zeros(nr_photons)
        
            for i in range(nr_photons):
                wavelength_photons[i] = (constants.h * constants.c) / energy_per_photon
                time_photons[i] = np.random.choice(t_jitter, p = norm_transmission)
        else:
            wavelength_photons = np.empty(0)
            time_photons = np.empty(0)

        return wavelength_photons, time_photons, nr_photons
    
    def find_T1(self, lower_limit, upper_limit, tol):
        # für X-Basis -> 110 ist 1000 non-decoy
        optical_power, peak_wavelength = self.random_laser_output_fixed('current_power','voltage_shift', 'current_wavelength')
        basis, value, decoy = self.generate_alice_choices_fixed(basis = 1, value = 1, decoy = 0)

        while upper_limit- lower_limit > tol:
            T1_dampening = (lower_limit + upper_limit) / 2
            voltage_signal, t_jitter, _ = self.signal_bandwidth_jitter(basis, value, decoy)
            _, calc_mean_photon_nr, _, _ = self.eam_transmission_1_mean_photon_number(voltage_signal, t_jitter, optical_power, peak_wavelength, T1_dampening, basis, value, decoy)

            #compare calculated mean with target mean
            if calc_mean_photon_nr < self.config.mean_photon_nr:  #!hier kleiner weil durch T1 geteilt wird ! anders als in find_voltage
                upper_limit = T1_dampening #reduce upper bound
            else:
                lower_limit = T1_dampening #increase lower bound

        #final voltage decoy
        T1_dampening = (lower_limit + upper_limit) / 2
        return T1_dampening

    def _set_voltage(self, optical_power, peak_wavelength, lower_limit, upper_limit, tol, target_mean, voltage_type, T1_dampening, basis, value, decoy):
        """Helper method to perform binary search and set the voltage."""
        while upper_limit - lower_limit > tol:
            voltage = (lower_limit + upper_limit) / 2
            
            # Dynamically set the voltage attribute based on voltage_type
            setattr(self.config, voltage_type, voltage)

            voltage_signal, t_jitter, _ = self.signal_bandwidth_jitter(*self.generate_alice_choices_fixed(basis, value, decoy))
            '''plt.plot(t_jitter, s_filtered_repeating, label = 'first run with voltage = '+ str(voltage))
            plt.show()'''
            _, calc_mean_photon_nr, _, _ = self.eam_transmission_1_mean_photon_number(
                voltage_signal, t_jitter, optical_power, peak_wavelength, T1_dampening, 
                *self.generate_alice_choices_fixed(basis, value, decoy))

            # Compare the calculated mean with the target mean
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
        optical_power, peak_wavelength = self.random_laser_output_fixed('current_power', 'voltage_shift', 'current_wavelength')

        # Find voltage for 000 -> 1010 non-decoy state
        self.config.voltage_sup = self._set_voltage(optical_power, peak_wavelength, lower_limit, upper_limit, tol, 
                                             self.config.mean_photon_nr, "voltage_sup", T1_dampening, 
                                             basis = 0, value = 0, decoy = 0)
        #print('voltage_sup', self.config.voltage_sup)

        # Reset limits for next voltage calculation
        lower_limit, upper_limit = store_lower_limit, store_upper_limit

        # Find voltage for 111 -> 1000 decoy state
        self.config.voltage_decoy = self._set_voltage(optical_power, peak_wavelength, lower_limit, upper_limit, tol, 
                                               self.config.mean_photon_decoy, "voltage_decoy", T1_dampening, 
                                               basis = 1, value = 1, decoy = 1)
        #print('voltage_decoy', self.config.voltage_decoy)


        # Reset limits for next voltage calculation
        lower_limit, upper_limit = store_lower_limit, store_upper_limit

        # Find voltage for 001 -> 1010 decoy state
        self.config.voltage_decoy_sup = self._set_voltage(optical_power, peak_wavelength, lower_limit, upper_limit, tol, 
                                                   self.config.mean_photon_decoy, "voltage_decoy_sup", T1_dampening, 
                                                   basis = 0, value = 0, decoy = 1)
        #print('voltage_decoy_sup', self.config.voltage_decoy_sup)

        return None
    
    def initialize(self):
        plt.style.use(self.config.mlp)
        #calculate T1 dampening 
        T1_dampening = self.find_T1(lower_limit = 0, upper_limit = 100, tol = 1e-3)
        #print('T1_dampening at initialize end: ' +str(T1_dampening))
        #T1_dampening_in_dB = 10* np.log(1/T1_dampening) 
        #print('T1_dampening at initialize end in dB: ' + str(T1_dampening_in_dB))

        #with simulated decoy state: calculate decoy height
        self.find_voltage_decoy(T1_dampening, lower_limit=-2, upper_limit=2, tol=1e-7, )
        #print('Voltage_decoy at initialize end' + str(self.config.voltage_decoy))
        #print('Voltage_decoy_sup at initialize end' + str(self.config.voltage_decoy_sup))
        #print('Voltage_sup at initialize end' + str(self.config.voltage_sup))

        return T1_dampening
    
   