import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import splrep, splev
import pandas as pd 
import time
from pathlib import Path
from scipy.fftpack import fft, ifft, fftfreq

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
    def __init__(self, data, n_samples = 10000, n_pulses = 4, symbol_length=1000, p_z_alice=0.5, p_z_1=0.5, 
                 p_decoy=0.1, sampling_rate_fft = 100e11, S21_dB = - 2, freq = 6.75e9):
        self.data = data
        self.n_samples = n_samples
        self.n_pulses = n_pulses
        self.symbol_length = symbol_length
        self.p_z_alice = p_z_alice
        self.p_z_1 = p_z_1
        self.p_decoy = p_decoy
        self.sampling_rate_fft = sampling_rate_fft
        self.S21_dB = S21_dB
        self.freq = freq

    def get_interpolated_value(self, x_data, name):
        #calculate tck for which curve
        tck = self.data.get_data(x_data, name)
        return splev(x_data, tck)

    def random_laser_output(self, current_power, voltage_shift, current_wavelength):
        # Generate a random time within the desired range
        time = np.random.uniform(0, 10) 
        
        # Calculate voltage and current based on this time of laserdiode and heater
            #voltage_heater = 1 in V, voltage_amplitude = 0.050 in V, voltage_frequency = 1
        chosen_voltage = 1 + 0.050 * np.sin(2 * np.pi * 1 * time)  
    
            #current_laserdiode = 0.08 in A, current_amplitude = 0.020 in A, current_frequency = 1
        chosen_current = (0.08 + 0.02 * np.sin(2 * np.pi * 1 * time) )* 1e3 #damit in mA

        optical_power = self.get_interpolated_value(chosen_current, current_power)
        peak_wavelength = self.get_interpolated_value(chosen_current, current_wavelength) + self.get_interpolated_value(chosen_voltage, voltage_shift)
        return optical_power, peak_wavelength

    def generate_alice_choices(self):
        """Generates Alice's choices for a quantum communication protocol, including 
        basis selection, value encoding, decoy states, and attenuation patterns.

        Args:
        n_pulses: Elektronik kann so viele channels
        symbol_length: iterations per symbol
        p_z_alice: Probability of Alice choosing the Z basis.
        p_z_1: Probability of encoding a '1' in the Z basis.
        p_decoy: Probability of sending a decoy state.

        not in
        dB_on: volle attenuation
        dB_off: weniger attenuation
        dB_decoy: Attenuation in dB for decoy states.
        dB_channel_attenuation: Channel attenuation in dB.

        Returns:
        tuple: A tuple containing the basis choices, encoded values, decoy flags, 
            attenuation pattern, modulation multiplier, and channel multiplier.
        """

        # Basis and value choices
        basis = np.random.choice([0, 1], size = 1, p=[1-self.p_z_alice, self.p_z_alice])
        value = np.random.choice([0, 1], size = 1, p=[1-self.p_z_1, self.p_z_1])
        value[basis == 0] = -1  # Mark X basis values

        # Decoy state selection
        decoy = np.random.choice([0, 1], size = 1, p=[1-self.p_decoy, self.p_decoy]) #size=(self.n_pulses // self.symbol_length)

        pulse_duration = 1 / self.freq  # Pulse duration for a 6.75e9 GHz square wave
        t = np.arange(0, self.n_pulses * pulse_duration, 1 / self.sampling_rate_fft)  # Time vector

        # Create a repeating square wave signal in time domain
        one_signal = len(t) // self.n_pulses
        if value == 1:
            #1000 
            repeating_square_pulse = np.zeros(len(t))
            repeating_square_pulse[:one_signal] = 1
        elif value == 0:
            #0010 
            repeating_square_pulse = np.zeros(len(t))
            repeating_square_pulse[2 * one_signal:3 * one_signal] = 1
            '''            repeating_square_pulse = np.concatenate([np.zeros(2* one_signal, dtype=int), 
                                                     np.ones(one_signal, dtype=int), 
                                                     np.zeros(len(t) - 3 * one_signal, dtype=int)])'''
        elif value == -1:
            #1010
            repeating_square_pulse = np.zeros(len(t))
            repeating_square_pulse[:one_signal] = 1
            repeating_square_pulse[2 * one_signal : 3* one_signal] = 1
            '''repeating_square_pulse = np.concatenate([np.ones(one_signal, dtype=int)*1/np.sqrt(2), 
                                                        np.zeros(one_signal, dtype=int)*1/np.sqrt(2), 
                                                        np.ones(one_signal, dtype=int)*1/np.sqrt(2), 
                                                        np.zeros(len(t) - 3 * one_signal, dtype=int)*1/np.sqrt(2)])'''
        # Fourier transform to frequency domain for the repeating signal
        n_repeating = len(t)
        S_f_repeating = fft(repeating_square_pulse)
        frequencies_repeating = fftfreq(n_repeating, d=1 / self.sampling_rate_fft)  

        # Plot the original signal
        plt.figure(figsize=(12, 6))
        plt.plot(t * 1e9, repeating_square_pulse, label="Original Signal", alpha=1, marker="")

        #calculate cutoff for S21_dB = -3dB
        '''k = np.sqrt(16 / (10**0.15 - 1))
        cutoff = (k * np.sqrt(10 ** (self.S21_dB / -20) - 1)) * 1e9
        print(cutoff)

        freq_x = [0, cutoff * 0.8, cutoff, cutoff * 1.2, self.sampling_rate_fft / 2]
        freq_y = [1, 1, 0.7, 0.01, 0.001]  # Gradual drop-off for a smoother response
    
        # Apply the frequency filter
        S_filtered_repeating = S_f_repeating * np.interp(np.abs(frequencies_repeating), freq_x, freq_y)
        
        # Inverse Fourier transform back to the time domain
        s_filtered_repeating = np.real(ifft(S_filtered_repeating))
        
        # Plot the filtered signal
        plt.plot(t * 1e9, s_filtered_repeating, label=f"Cutoff: {cutoff/1e9} GHz", alpha=0.7, marker ="")

        # Final plot adjustments
        plt.title("Square Signal with Bandwidth Limitation")
        plt.xlabel("Time (ns)")
        plt.ylabel("Amplitude")
        plt.legend()
        plt.grid(True)
        plt.show()'''

        cutoffs = [1e9, 2e9, 3e9, 4e9, 5e9, 10e9] #20e9, 30e9, 50e9, 80e9, 100e9]
        for cutoff in cutoffs:
            # Use a smoother transition for the frequency response
            freq_x = [0, cutoff * 0.8, cutoff, cutoff * 1.2, self.sampling_rate_fft / 2]
            freq_y = [1, 1, 0.7, 0.01, 0.001]  # Gradual drop-off for a smoother response
            
            # Apply the frequency filter
            S_filtered_repeating = S_f_repeating * np.interp(np.abs(frequencies_repeating), freq_x, freq_y)
            
            # Inverse Fourier transform back to the time domain
            s_filtered_repeating = np.real(ifft(S_filtered_repeating))
            
            # Plot the filtered signal
            plt.plot(t * 1e9, s_filtered_repeating, label=f"Cutoff: {cutoff/1e9} GHz", alpha=0.7, marker ="")

        # Final plot adjustments
        plt.title("Square Signal with Bandwidth Limitation")
        plt.xlabel("Time (ns)")
        plt.ylabel("Amplitude")
        plt.legend()
        plt.grid(True)
        plt.show()

        return (basis, value, decoy, repeating_square_pulse)

    def get_output(self):
        #alternative: self.laser_outputs = [self.random_laser_output() for _ in range(self.n_samples)]
        all_optical_power = np.empty(self.n_samples)
        all_peak_wavelength = np.empty(self.n_samples)

        for i in range(self.n_samples):
            all_optical_power[i], all_peak_wavelength[i] = self.random_laser_output('current_power','voltage_shift', 'current_wavelength')
        return all_optical_power, all_peak_wavelength
    
    def get_output_symbols(self):
        basis_alice, value_alice, decoy_alice = self.generate_alice_choices()
        return basis_alice

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

#create simulation
simulation = Simulation(database)   

end_time = time.time()  # Record end time
execution_time = end_time - start_time  # Calculate execution time
print(f"Execution time: {execution_time:.9f} seconds for {simulation.n_samples} samples")

#plot results
#optical_power, peak_wavelength = simulation.get_output()
alice_symbols = simulation.generate_alice_choices()

end_time_2 = time.time()  # Record end time
execution_time_2 = end_time_2 - start_time  # Calculate execution time
print(f"Execution time after writing in Array: {execution_time_2:.9f} seconds for {simulation.n_samples} samples")

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