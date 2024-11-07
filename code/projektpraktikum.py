import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import splrep, splev
import pandas as pd 
import time
from pathlib import Path


class dataManager:
    def __init__(self):
        self.curves = {}

    def add_data(self, csv_file, column1, column2, rows, name):
        df = pd.read_csv(csv_file, nrows = rows)            
        df.columns = df.columns.str.strip()
        
        # Access first and last elements directly from the DataFrame
        x_min = df[column1].iloc[0]  # First element
        x_max = df[column1].iloc[-1]  # Last element

        self.curves[name] = {
            'tck': splrep(df[column1], df[column2]),  # Store the tck
            'x_min': x_min,  # Store minimum x-value
            'x_max': x_max   # Store maximum x-value
            }
        
    def get_data(self, x_data, name):
        x_min = self.curves[name]['x_min']
        x_max = self.curves[name]['x_max']
        if x_data < x_min or x_data > x_max:
            raise ValueError(f"x data isn't in table")
        if name not in self.curves:
            raise ValueError(f"Spline '{name}' not found.")
        return self.curves[name]['tck'] # Return tck

class Simulation:
    def __init__(self, data, n_samples=10000):
        self.data = data
        self.n_samples = n_samples  # Number of samples to generate

    def get_interpolated_value(self, x_data, name):
        #calculate tck for which curve
        tck = self.data.get_data(x_data, name)
        return splev(x_data, tck)

    def random_laser_output(self, current_power, voltage_shift, current_wavelength):
        # Generate a random time within the desired range
        time = np.random.uniform(0, 10) 
        
        # Calculate voltage and current based on this time
            #voltage_heater = 1 in V, voltage_amplitude = 0.050 in V, voltage_frequency = 1
        chosen_voltage = 1 + 0.050 * np.sin(2 * np.pi * 1 * time)  
    
            #current_laserdiode = 0.08 in A, current_amplitude = 0.020 in A, current_frequency = 1
        chosen_current = (0.08 + 0.02 * np.sin(2 * np.pi * 1 * time) )* 1e3 #damit in mA

        optical_power = self.get_interpolated_value(chosen_current, current_power)
        peak_wavelength = self.get_interpolated_value(chosen_current, current_wavelength) + self.get_interpolated_value(chosen_voltage, voltage_shift)
        return optical_power, peak_wavelength

    def generate_alice_choices_fixed(self, n_pulses, symbol_length=2, p_z_alice=0.5, p_z_1=0.5, p_decoy=0.1, dB_on=20, dB_off=10, dB_decoy=15, dB_channel_attenuation=5):

          """
        Generates Alice's choices for a quantum communication protocol, including 
        basis selection, value encoding, decoy states, and attenuation patterns.

        Args:
        n_pulses: Number of pulses to generate.
        symbol_length: Length of each symbol in pulses.
        p_z_alice: Probability of Alice choosing the Z basis.
        p_z_1: Probability of encoding a '1' in the Z basis.
        p_decoy: Probability of sending a decoy state.
        dB_on: volle attenuation
        dB_off: weniger attenuation
        dB_decoy: Attenuation in dB for decoy states.
        dB_channel_attenuation: Channel attenuation in dB.

        Returns:
        tuple: A tuple containing the basis choices, encoded values, decoy flags, 
            attenuation pattern, modulation multiplier, and channel multiplier.
        """

        # SET Basis and value choices
        basis = np.zeros(n_pulses // symbol_length)
        basis[: int(3/4 * n_pulses // symbol_length)] = 1
        value = np.zeros(n_pulses // symbol_length)
        value[: int(1/4 * n_pulses // symbol_length)] = 1
        value[basis == 0] = -1

        # Decoy state selection
        decoy = np.zeros(n_pulses // symbol_length)
        decoy[int(2/4 * n_pulses // symbol_length):int(3/4 * n_pulses // symbol_length)] = 1 

        # Attenuation pattern
        pattern_01 = np.concatenate([np.zeros(symbol_length // 2), np.ones(symbol_length // 2)])
        pattern_10 = np.concatenate([np.ones(symbol_length // 2), np.zeros(symbol_length // 2)])
        pattern_attenuator = np.zeros(n_pulses, dtype=int)
        for i, v in enumerate(value):
            start = i * symbol_length
            end = start + symbol_length
            if v == 0:
                pattern_attenuator[start:end] = pattern_10
            elif v == 1:
                pattern_attenuator[start:end] = pattern_01

        # Attenuation pattern in dB
        attenuation = np.zeros(n_pulses)
        attenuation[pattern_attenuator == 0] = dB_off
        attenuation[pattern_attenuator == 1] = dB_on
        attenuation[(pattern_attenuator == 0) & (np.repeat(decoy, symbol_length) == 1)] = dB_decoy
        attenuation[(np.repeat(value, symbol_length) == -1) & (np.repeat(decoy, symbol_length) == 1)] = dB_decoy

        # Calculate multipliers
        multiplier_modulation = np.power(10, -1 * attenuation / 10)
        multiplier_channel = np.power(10, -1 * dB_channel_attenuation / 10)

        return (basis, value, decoy, pattern_attenuator, 
                multiplier_modulation, multiplier_channel)


    def get_output(self):
        #alternative: self.laser_outputs = [self.random_laser_output() for _ in range(self.n_samples)]
        all_optical_power = np.empty(self.n_samples)
        all_peak_wavelength = np.empty(self.n_samples)

        for i in range(self.n_samples):
            all_optical_power[i], all_peak_wavelength[i] = self.random_laser_output('current_power','voltage_shift', 'current_wavelength')
        return all_optical_power, all_peak_wavelength

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

# PARAMETERS
n_pulses = int(1e4)
symbol_length = 2
jitter_source = 100e-12
repetition_rate = 1e9
time_resolution_factor = 4
factor_pulse_ampltiude = 2
p_z_alice = 0.8
p_z_1 = 0.5
p_decoy = 0.1 * 8
dB_on = 25
dB_off = 10
dB_decoy = 15
dB_channel_attenuation = 2
p_z_bob = p_z_alice
p_stray = 1e-4
p_short_path_DLI = 0.5
epsilon = 1e-3 # 2e-50
lambdaEC = 1.16

#measure execution time
start_time = time.time()  # Record start time

#set probabilities
prob_decoy = 0.1

#database
database = dataManager()

#readin
database.add_data('data/current_power_data.csv', 'Current (mA)', 'Optical Power (mW)', 9, 'current_power') 
database.add_data('data/voltage_shift_data.csv', 'Voltage (V)', 'Wavelength Shift (nm)', 20, 'voltage_shift')
database.add_data('data/current_wavelength_modified.csv', 'Current (mA)', 'Wavelength (nm)', 9, 'current_wavelength')#modified sodass mA Werte stimmen (/1000)

#create simulation
simulation = Simulation(database)   

end_time = time.time()  # Record end time
execution_time = end_time - start_time  # Calculate execution time
print(f"Execution time: {execution_time:.9f} seconds for {simulation.n_samples} samples")

#plot results
optical_power, peak_wavelength = simulation.get_output()

end_time_2 = time.time()  # Record end time
execution_time_2 = end_time_2 - start_time  # Calculate execution time
print(f"Execution time after writing in Array: {execution_time_2:.9f} seconds for {simulation.n_samples} samples")

plt.hist(optical_power, bins=30, label='Optical Power', alpha=0.7)
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
