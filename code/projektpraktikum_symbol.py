import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import splrep, splev
import pandas as pd 
import time
from pathlib import Path
from scipy.fftpack import fft, ifft, fftfreq
from scipy import constants
from scipy.special import factorial

class dataManager:
    def __init__(self):
        self.curves = {}

    def add_data(self, csv_file, column1, column2, rows, name):
               
        df = pd.read_csv(csv_file, nrows = rows)            
        df.columns = df.columns.str.strip()

        #df.sort_values(by=column1, ascending=True)
        if not all(df[column1].diff().dropna() > 0):  # Check if the values are not in ascending order
            df[column1] = df[column1].iloc[::-1].reset_index(drop=True)  # Reverse the order of the column
            df[column2] = df[column2].iloc[::-1].reset_index(drop=True)  # Reverse the corresponding column

        # Ensure valid input
        df = df.sort_values(by=column1).drop_duplicates(subset=column1)
        df = df.dropna(subset=[column1, column2])
        df[column1] = pd.to_numeric(df[column1], errors='coerce')
        df[column2] = pd.to_numeric(df[column2], errors='coerce')
             
        # Access first and last elements directly from the DataFrame
        x_min = df[column1].iloc[0]  # First element
        x_max = df[column1].iloc[-1]  # Last element

        self.curves[name] = {
            'tck': splrep(df[column1], df[column2]),  # Store the tck
            'x_min': x_min,  # Store minimum x-value
            'x_max': x_max   # Store maximum x-value
            }
    
    def add_jitter(self, jitter): #Gaussian
        # Calculate standard deviation based on FWHM
        std_dev = jitter / (2 * np.sqrt(2 * np.log(2)))

        # Define a range of values (e.g., -3 to 3 standard deviations)
        x = np.linspace(-3*std_dev, 3*std_dev, 100)

        # Compute Gaussian weights
        weights = np.exp(-0.5 * (x / std_dev) ** 2)

        # Normalize weights to get probabilities (sum to 1)
        probabilities_array = weights / weights.sum()
        self.curves['probabilities'] = {
            'prob': probabilities_array,
            'x': x
            }
    


    def get_data(self, x_data, name):
        if name == 'probabilities':
            return self.curves[name]['prob'], self.curves[name]['x']
        x_min = self.curves[name]['x_min']
        x_max = self.curves[name]['x_max']
        if x_data < x_min or x_data > x_max:
            raise ValueError(x_data, " x data isn't in table for ", name)
        if name not in self.curves:
            raise ValueError(f"Spline '{name}' not found.")
        return self.curves[name]['tck'] # Return tck
    
    def show_data(self, csv_file, column1, column2, rows):
        table = pd.read_csv(csv_file, nrows = rows)
        table.columns = table.columns.str.strip()
        plt.plot(table[column1], table[column2])
        if csv_file == 'data/eam_transmission_data.csv':
            plt.ylabel('transmission')
            plt.xlabel('voltage in V')
        plt.show()

class SimulationConfig:
    def __init__(self, data, n_samples=10000, n_pulses=4, mean_voltage=1.0, mean_amplitude=0.08,
                 p_z_alice=0.5, p_z_1=0.5, p_decoy=0.1, freq=6.75e9, jitter=1e-11,voltage_decoy=1.0,
                 voltage=1.0, voltage_decoy_sup=1.0, voltage_sup=1.0, 
                 eam_transmission_TP=-1.0, eam_transmission_HP=0.0, eam_transmission_TP_decoy=-1.0,
                 eam_transmission_HP_decoy=-0.6,
                 mean_photon_nr=0.7, mean_photon_decoy=0.1,
                 ):
        # Input data
        self.data = data

        # General simulation parameters
        self.n_samples = n_samples
        self.n_pulses = n_pulses

        # Voltage and amplitude settings
        self.mean_voltage = mean_voltage  # Voltage (in V)
        self.mean_amplitude = mean_amplitude  # Amplitude (in A)

        # Probability parameters
        self.p_z_alice = p_z_alice
        self.p_z_1 = p_z_1
        self.p_decoy = p_decoy

        # Sampling and frequency
        self.freq = freq  # FPGA frequency (Hz)

        # Timing parameters
        self.jitter = jitter  # Timing jitter (s)

        # Voltage configurations
        self.voltage_decoy = voltage_decoy
        self.voltage = voltage
        self.voltage_decoy_sup = voltage_decoy_sup  # Superposition voltage with decoy state
        self.voltage_sup = voltage_sup  # Superposition voltage without decoy state

        # Transmission settings
        self.eam_transmission_TP = eam_transmission_TP  # Transmission Tiefpunkt (-1V)
        self.eam_transmission_HP = eam_transmission_HP  # Transmission Hochpunkt (0V)
        self.eam_transmission_TP_decoy = eam_transmission_TP_decoy
        self.eam_transmission_HP_decoy = eam_transmission_HP_decoy

        # Photon number settings
        self.mean_photon_nr = mean_photon_nr
        self.mean_photon_decoy = mean_photon_decoy
    
class Simulation:
    def __init__(self, config: SimulationConfig):
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
        chosen_current = (self.config.mean_amplitude)* 1e3 #damit in mA

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
        chosen_current = (self.config.mean_amplitude + 0.02 * np.sin(2 * np.pi * 1 * time) )* 1e3 #damit in mA

        optical_power = self.get_interpolated_value(chosen_current, current_power)
        peak_wavelength = self.get_interpolated_value(chosen_current, current_wavelength) + self.get_interpolated_value(chosen_voltage, voltage_shift)
        return optical_power * 1e-3, peak_wavelength * 1e-9  #in W and m
    

    def generate_alice_choices_fixed(self, basis, value, decoy):

        # Basis and value choices
        basis_arr = np.array([basis])
        value_arr = np.array([value])
        value_arr[basis == 0] = -1  # Mark X basis values

        decoy_arr = np.array([decoy])
        return (basis_arr, value_arr, decoy_arr)

    
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
        repeating_square_pulse = np.zeros(len(t))
        one_signal = len(t) // self.config.n_pulses

        for i, bit in enumerate(pattern):
            if bit == 1:
                repeating_square_pulse[i * one_signal:(i + 1) * one_signal] = pulse_height

        return t, repeating_square_pulse
    
    def generate_encoded_pulse(self, pulse_height, pulse_duration, value, sampling_rate_fft):
        pattern = self.encode_pulse(value)
        return self.generate_square_pulse(pulse_height, pulse_duration, pattern, sampling_rate_fft)

    def apply_bandwidth_filter(self, signal, sampling_rate_fft, cutoff):
        """Apply a frequency-domain filter to a signal."""
        S_f = fft(signal)
        frequencies = fftfreq(len(signal), d=1 / sampling_rate_fft)

        freq_x = [0, cutoff * 0.8, cutoff, cutoff * 1.2, sampling_rate_fft / 2]
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
        filtered_signal = self.apply_bandwidth_filter(signal, sampling_rate_fft, cutoff=4e9)
        t_jittered = self.apply_jitter(t)
        return filtered_signal, t_jittered

    def eam_transmission_1_mean_photon_number(self, s_filtered_repeating, t_jitter, optical_power, peak_wavelength, T1_dampening, basis, value, decoy):
        #include the eam_voltage and multiply with calculated optical power from laser
        power = np.empty(len(s_filtered_repeating))
        transmission = np.empty(len(s_filtered_repeating))
        if decoy == 1:
            s_filtered_repeating_non_decoy, _ = self.signal_bandwidth_jitter(basis, value, decoy = 0)
            decoy = np.array([1])
            voltage_min = np.min(s_filtered_repeating_non_decoy)
            if voltage_min < -0.5:
                print('voltage_min' +str(voltage_min))
            voltage_max = np.max(s_filtered_repeating_non_decoy)
            if voltage_max > 2:
                print('voltage_max' +str(voltage_max))
        else:
            voltage_min = np.min(s_filtered_repeating)
            voltage_max = np.max(s_filtered_repeating)
        
        for i in range(len(s_filtered_repeating)):
            voltage_for_eam_table = (s_filtered_repeating[i]-voltage_min) / (voltage_max - voltage_min) * (self.config.eam_transmission_HP-self.config.eam_transmission_TP) + self.config.eam_transmission_TP 
            transmission[i] = self.get_interpolated_value(voltage_for_eam_table, 'eam_transmission')
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
        print('calc_mean_photon_nr: ' ,calc_mean_photon_nr)
        print('target mean photon number: ', self.config.mean_photon_nr)
        x = np.arange(0, upper_bound + 1)

        # Compute Poisson probabilities
        probabilities_array_poisson = np.exp(-calc_mean_photon_nr) * (calc_mean_photon_nr ** x) / factorial(x)

        # Normalize probabilities (optional, as Poisson probabilities already sum to ~1)
        probabilities_array_poisson = probabilities_array_poisson / probabilities_array_poisson.sum()
                
        #choose amount of photons and calculate energy per Photons and initialize Wavelength Arrays
        nr_photons = np.random.choice(x, p = probabilities_array_poisson)
        print('nr photons per pulse: ', nr_photons)
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

        return wavelength_photons, time_photons
    
    def find_T1(self, lower_limit, upper_limit, tol):
        # für X-Basis -> 110 ist 1000 non-decoy
        optical_power, peak_wavelength = self.random_laser_output('current_power','voltage_shift', 'current_wavelength')
        basis, value, decoy = self.generate_alice_choices_fixed(basis = 1, value = 1, decoy = 0)

        while upper_limit- lower_limit > tol:
            T1_dampening = (lower_limit + upper_limit) / 2
            s_filtered_repeating, t_jitter = self.signal_bandwidth_jitter(basis, value, decoy)
            _, calc_mean_photon_nr, _, _ = self.eam_transmission_1_mean_photon_number(s_filtered_repeating, t_jitter, optical_power, peak_wavelength, T1_dampening, basis, value, decoy)

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
            setattr(self, voltage_type, voltage)
            s_filtered_repeating, t_jitter = self.signal_bandwidth_jitter(*self.generate_alice_choices_fixed(basis, value, decoy))
            _, calc_mean_photon_nr, _, _ = self.eam_transmission_1_mean_photon_number(
                s_filtered_repeating, t_jitter, optical_power, peak_wavelength, T1_dampening, *self.generate_alice_choices_fixed(basis, value, decoy))

            # Compare the calculated mean with the target mean
            if calc_mean_photon_nr > target_mean:
                upper_limit = voltage  # Reduce upper bound
            else:
                lower_limit = voltage  # Increase lower bound

        # Final voltage assignment
        return (lower_limit + upper_limit) / 2

    def find_voltage_decoy(self, T1_dampening, lower_limit, upper_limit, tol):
        """Find the appropriate voltage values for decoy and non-decoy states using binary search."""
        # Store original limits to reset later
        store_lower_limit = lower_limit
        store_upper_limit = upper_limit

        # Set the optical power and peak wavelength for the simulation
        optical_power, peak_wavelength = self.random_laser_output_fixed('current_power', 'voltage_shift', 'current_wavelength')

        # Find voltage for 111 -> 1000 decoy state
        self.config.voltage_decoy = self._set_voltage(optical_power, peak_wavelength, lower_limit, upper_limit, tol, 
                                               self.config.mean_photon_decoy, "voltage_decoy", T1_dampening, 
                                               basis = 1, value = 1, decoy = 1)

        print('voltage_decoy', self.config.voltage_decoy)
        # Reset limits for next voltage calculation
        lower_limit, upper_limit = store_lower_limit, store_upper_limit

        # Find voltage for 000 -> 1010 non-decoy state
        self.config.voltage_sup = self._set_voltage(optical_power, peak_wavelength, lower_limit, upper_limit, tol, 
                                             self.config.mean_photon_nr, "voltage_sup", T1_dampening, 
                                             basis = 0, value = 0, decoy = 0)
        print('voltage_sup', self.config.voltage_sup)


        # Reset limits for next voltage calculation
        lower_limit, upper_limit = store_lower_limit, store_upper_limit

        # Find voltage for 001 -> 1010 decoy state
        self.config.voltage_decoy_sup = self._set_voltage(optical_power, peak_wavelength, lower_limit, upper_limit, tol, 
                                                   self.config.mean_photon_decoy, "voltage_decoy_sup", T1_dampening, 
                                                   basis = 0, value = 0, decoy = 0)
        print('voltage_decoy_sup', self.config.voltage_decoy_sup)

        return None
    
    def find_voltage_decoy_old(self, lower_limit, upper_limit, tol):
        basis, value, decoy = self.generate_alice_choices_fixed(basis = 0, value = 0, decoy = 1)
        optical_power, peak_wavelength = self.random_laser_output('current_power','voltage_shift', 'current_wavelength')

        while upper_limit - lower_limit > tol:
            self.config.voltage_decoy = (lower_limit + upper_limit) / 2
            s_filtered_repeating, t_jitter = self.signal_bandwidth_jitter(basis, value, decoy)
            _, calc_mean_photon_nr, _, _ = self.eam_transmission_1_mean_photon_number(s_filtered_repeating, t_jitter, optical_power, peak_wavelength, basis, value, decoy)
            #print('calc_mean_photon_nr: ' + str(calc_mean_photon_nr) + 'with voltage_decoy: ' +str(self.voltage_decoy))

            #compare calculated mean with target mean
            if calc_mean_photon_nr > self.config.mean_photon_decoy:
                upper_limit = self.config.voltage_decoy #reduce upper bound
            else:
                lower_limit = self.config.voltage_decoy #increase lower bound

        #final voltage decoy
        self.config.voltage_decoy = (lower_limit + upper_limit) / 2
        return None
    
    def initialize(self):
        #calculate T1 dampening 
        T1_dampening = self.find_T1(lower_limit = 0, upper_limit = 100, tol = 1e-3)
        print('T1_dampening at initialize end: ' +str(T1_dampening))
        T1_dampening_in_dB = 10* np.log(1/T1_dampening) 
        print('T1_dampening at initialize end in dB: ' + str(T1_dampening_in_dB))

        #with simulated decoy state: calculate decoy height
        self.find_voltage_decoy(T1_dampening, lower_limit=0, upper_limit=100, tol=1e-7, )
        print('Voltage_decoy at initialize end' + str(self.config.voltage_decoy))
        print('Voltage_decoy_sup at initialize end' + str(self.config.voltage_decoy_sup))
        print('Voltage_sup at initialize end' + str(self.config.voltage_sup))



        return T1_dampening
    
    def run_simulation(self):
        
        T1_dampening = self.initialize()
        optical_power, peak_wavelength = self.random_laser_output('current_power','voltage_shift', 'current_wavelength')
        basis, value, decoy = self.generate_alice_choices_fixed(basis = 0, value = 0, decoy = 0)   #11: 1000, 10: 0010, 00: 1010
        s_filtered_repeating, t_jitter = self.signal_bandwidth_jitter(basis, value, decoy)
        power_dampened, calc_mean_photon_nr, energy_pp, transmission = self.eam_transmission_1_mean_photon_number(s_filtered_repeating, t_jitter, optical_power, peak_wavelength, T1_dampening, basis, value, decoy)    
        wavelength_photons, time_photons = self.eam_transmission_2_choose_photons(calc_mean_photon_nr, energy_pp, transmission, t_jitter)
        print(f"wavelength photons: {wavelength_photons}, time photons: {time_photons}")

        plt.plot(t_jitter * 1e9, power_dampened * 1e3, label = 'dampened_power') #fehlt optical power
        plt.title("Power of Square Signal with Bandwidth Limitation with 1e-11 jitter")
        plt.xlabel("Time in ns")
        plt.ylabel("Power in mW")
        plt.legend()
        plt.grid(True)
        save_plot('new_T1_dampening_power_after_transmission_with_4_GHz_bandwidth_and_1e-11s_jitter')
        #plt.show()'''

    

def save_plot(filename, dpi=600):
  """Saves the current Matplotlib plot to a file in the 'img' directory."""
  script_dir = Path(__file__).parent
  img_dir = script_dir / 'img'
  img_dir.mkdir(exist_ok=True)
  filepath = img_dir / filename
  plt.savefig(filepath, dpi=dpi)
  plt.close()


# ================================================
# EXECUTION 
# ================================================

#measure execution time
start_time = time.time()  # Record start time

#database
database = dataManager()

#readin
database.add_data('data/current_power_data.csv', 'Current (mA)', 'Optical Power (mW)', 9, 'current_power') 
database.add_data('data/voltage_shift_data.csv', 'Voltage (V)', 'Wavelength Shift (nm)', 20, 'voltage_shift')
database.add_data('data/current_wavelength_modified.csv', 'Current (mA)', 'Wavelength (nm)', 9, 'current_wavelength')#modified sodass mA Werte stimmen (/1000)
database.add_data('data/eam_transmission_data.csv', 'Voltage (V)', 'Transmission', 11, 'eam_transmission') #modified,VZ geflippt von Spannungswerten

database.add_jitter(jitter = 1e-11)

#create simulation
config = SimulationConfig(database, n_samples=10000, n_pulses=4, mean_voltage=1.0, mean_amplitude=0.08,
                 p_z_alice=0.5, p_z_1=0.5, p_decoy=0.1, freq=6.75e9, jitter=1e-11,voltage_decoy=1.0,
                 voltage=1.0, voltage_decoy_sup=1.0, voltage_sup=1.0, 
                 eam_transmission_TP=-1.0, eam_transmission_HP=0.0, eam_transmission_TP_decoy=-1.0,
                 eam_transmission_HP_decoy=-0.6,
                 mean_photon_nr=0.7, mean_photon_decoy=0.1)
simulation = Simulation(config)   

#plot results
simulation.run_simulation()

end_time = time.time()  # Record end time
execution_time = end_time - start_time  # Calculate execution time
print(f"Execution time: {execution_time:.9f} seconds for {config.n_samples} samples")

end_time_2 = time.time()  # Record end time
execution_time_2 = end_time_2 - start_time  # Calculate execution time
print(f"Execution time after writing in Array: {execution_time_2:.9f} seconds for {simulation.config.n_samples} samples")

'''x = np.linspace(0, simulation.n_pulses // simulation.symbol_length -1, simulation.n_pulses // simulation.symbol_length)
plt.hist(all_jitter * 1e12, bins=30, label='jitter', alpha=0.7)
plt.title('jittershift over ' + str(simulation.n_samples) + ' iterations',size = 14)
plt.ylabel('iterations')
plt.xlabel('jitter in ps')
save_plot('jitter_over_' + str(simulation.n_samples) + '_iterations_05_11.png')
#plt.show()
'''
'''x = np.linspace(0, simulation.n_pulses // simulation.symbol_length -1, simulation.n_pulses // simulation.symbol_length)
plt.plot(x, alice_symbols, label='alice_symbol')
plt.title('alice symbol',size = 14)
plt.ylabel('shape of symbol')
plt.xlabel('pulses per symbol length')
plt.show()
print(alice_symbols[10], alice_symbols[11])'''

'''plt.hist(optical_power, bins=30, label='Optical Power', alpha=0.7)
plt.title('optical power over ' + str(simulation.n_samples) + ' iterations',size = 14)
plt.ylabel('iterations')
plt.xlabel('optical power in mW')
save_plot('power_over_' + str(simulation.n_samples) + '_iterations_05_11.png')
plt.show()


plt.hist(peak_wavelength, bins=30, label='Peak Wavelength', alpha=0.7)
plt.title('peak wavelengths over ' + str(simulation.n_samples) + ' iterations',size = 14)
plt.ylabel('iterations')
plt.xlabel('peak wavelength in nm')
save_plot('peak_wavelength_over_' + str(simulation.n_samples) + '_iterations_05_11.png')
plt.show()
'''