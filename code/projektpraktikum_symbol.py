import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import splrep, splev
import pandas as pd 
import time
from pathlib import Path
from scipy.fftpack import fft, ifft, fftfreq
from scipy import constants

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
    
    def add_jitter(self, jitter):
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
            raise ValueError(str(x_data)  + "x data isn't in table")
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

    
    
class Simulation:
    def __init__(self, data, n_samples = 10000, n_pulses = 4, p_z_alice=0.5, p_z_1=0.5, 
                 p_decoy=0.1, sampling_rate_fft = 100e11, freq = 6.75e9, jitter = 1e-11, voltage_decoy = 1, 
                 voltage = 1, T1_dampening_0 = 1, T1_dampening_1 = 1,
                 eam_transmission_TP = -1, eam_transmission_HP = 0, eam_transmission_TP_decoy = -1, 
                 eam_transmission_HP_decoy = -0.6, mean_photon_nr = 0.7, mean_photon_decoy = 0.1):
        self.data = data
        self.n_samples = n_samples
        self.n_pulses = n_pulses
        self.p_z_alice = p_z_alice
        self.p_z_1 = p_z_1
        self.p_decoy = p_decoy
        self.sampling_rate_fft = sampling_rate_fft
        self.freq = freq #FPGA
        self.jitter = jitter
        self.voltage_decoy = voltage_decoy
        self.voltage = voltage
        self.T1_dampening_0 = T1_dampening_0
        self.T1_dampening_1 = T1_dampening_1
        self.eam_transmission_TP = eam_transmission_TP                  #V we use transmission curve till -1V Tiefpunkt
        self.eam_transmission_HP = eam_transmission_HP                  #V we use eam_transmission curve till 0V Hochpunkt
        self.eam_transmission_TP_decoy = eam_transmission_TP_decoy      
        self.eam_transmission_HP_decoy = eam_transmission_HP_decoy
        self.mean_photon_nr = mean_photon_nr
        self.mean_photon_decoy = mean_photon_decoy

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
        basis = np.random.choice([0, 1], size = 1, p=[1-self.p_z_alice, self.p_z_alice]) # Randomly selects whether each pulse block is prepared in the Z-basis (0) or the X-basis (1) with a bias controlled by p_z_alice
        value = np.random.choice([0, 1], size = 1, p=[1-self.p_z_1, self.p_z_1]) #Assigns logical values (0 or 1) to the pulses with probabilities defined by p_z_1. If the basis is 0 (X-basis), the values are set to -1 to differentiate them.   
        value[basis == 0] = -1  # Mark X basis values

        # Decoy state selection
        decoy = np.random.choice([0, 1], size = 1, p=[1-self.p_decoy, self.p_decoy])

        return (basis, value, decoy)
    
    def signal_bandwidth_jitter(self, value, decoy):
        
        pulse_duration = 1 / self.freq  # Pulse duration for a 6.75e9 GHz square wave
        t = np.arange(0, self.n_pulses * pulse_duration, 1 / self.sampling_rate_fft)  # Time vector

        # Create a repeating square wave signal in time domain
        one_signal = len(t) // self.n_pulses

        pulse_height = self.voltage_decoy if decoy == 1 else self.voltage

        if value == 1:
            #1000 
            repeating_square_pulse = np.zeros(len(t))
            repeating_square_pulse[:one_signal] = pulse_height
        elif value == 0:
            #0010 
            repeating_square_pulse = np.zeros(len(t))
            repeating_square_pulse[2 * one_signal:3 * one_signal] = pulse_height
        elif value == -1:
            #1010
            repeating_square_pulse = np.zeros(len(t))
            repeating_square_pulse[:one_signal] = pulse_height
            repeating_square_pulse[2 * one_signal : 3* one_signal] = pulse_height

        # Fourier transform to frequency domain for the repeating signal
        n_repeating = len(t)
        S_f_repeating = fft(repeating_square_pulse)
        frequencies_repeating = fftfreq(n_repeating, d=1 / self.sampling_rate_fft)  

        #calculate cutoff for S21_dB = -3dB
        #k = np.sqrt(16 / (10**0.15 - 1))
        #cutoff = (k * np.sqrt(10 ** (self.S21_dB / -20) - 1)) * 1e9
        cutoff = 4e9

        freq_x = [0, cutoff * 0.8, cutoff, cutoff * 1.2, self.sampling_rate_fft / 2]
        freq_y = [1, 1, 0.7, 0.01, 0.001]  # Gradual drop-off for a smoother response
    
        # Apply the frequency filter
        
        S_filtered_repeating = S_f_repeating * np.interp(np.abs(frequencies_repeating), freq_x, freq_y)
        
        # Inverse Fourier transform back to the time domain
        s_filtered_repeating = np.real(ifft(S_filtered_repeating))

        # Plot the filtered signal and the original signal
        '''#plt.figure(figsize=(12, 6))
        plt.plot(t * 1e9, repeating_square_pulse, label="Original Signal", alpha=1, marker="")
        plt.plot(t * 1e9, s_filtered_repeating, label="Cutoff: 4 GHz", alpha=0.7, marker ="")
        plt.show()
        
        # Final plot adjustments
        plt.title("voltage signal with bandwidth limitation and jitter")
        plt.xlabel("Time (ns)")
        plt.ylabel("Voltage (V)")
        plt.legend()
        plt.grid(True)
        #save_plot('new_v_with_4_GHz_bandwidth_and_1e-11s_jitter.png')
        #plt.show()'''
        
        #random choice for jitter
        probabilities, x = self.data.get_data(x_data = None, name = 'probabilities')
        jittershift = np.random.choice(x, p = probabilities)
        t_jitter = t + jittershift 
        
        return s_filtered_repeating, t_jitter

    def eam_transmission(self, s_filtered_repeating, t_jitter, optical_power, peak_wavelength, basis, value, decoy):
        #include the eam_voltage and multiply with calculated optical power from laser
        power = np.empty(len(s_filtered_repeating))
        if decoy == 1:
            s_filtered_repeating_non_decoy, _ = self.signal_bandwidth_jitter(value, decoy = 0)
            decoy = 1
            voltage_min = np.min(s_filtered_repeating_non_decoy)
            #print('voltage_min' +str(voltage_min))
            voltage_max = np.max(s_filtered_repeating_non_decoy)
            #print('voltage_max' +str(voltage_max))
        else:
            voltage_min = np.min(s_filtered_repeating)
            voltage_max = np.max(s_filtered_repeating)
        
        print('optical power' +str(optical_power))
        for i in range(len(s_filtered_repeating)):
            voltage_for_eam_table = (s_filtered_repeating[i]-voltage_min) / (voltage_max - voltage_min) * (self.eam_transmission_HP-self.eam_transmission_TP) + self.eam_transmission_TP 
            transmission = self.get_interpolated_value(voltage_for_eam_table, 'eam_transmission')
            power[i] = transmission * optical_power
        print('optical_power_imp'+str(optical_power))
        

        if basis == 0:
            power_dampened = power / self.T1_dampening_0
        else:
            power_dampened = power / self.T1_dampening_1
        print('peakwavelength' + str(peak_wavelength))
        energy_pp = np.trapz(power_dampened, t_jitter)

        '''plt.plot(t_jitter * 1e9, power, label = 'after eam') #fehlt optical power
        plt.title("Power of Square Signal with Bandwidth Limitation with 1e-11 jitter")
        plt.xlabel("Time in ns")
        plt.ylabel("Power in W")
        plt.legend()
        plt.grid(True)
        #save_plot('power_after_transmission_with_4_GHz_bandwidth_and_1e-11s_jitter')
        plt.show()'''

        print('energy_pp' + str(energy_pp))
        calc_mean_photon_nr = energy_pp / (constants.h*constants.c/peak_wavelength)
        
        return power, power_dampened, calc_mean_photon_nr

    '''def find_T1(self):
        #for 0 basis
        optical_power, peak_wavelength = self.random_laser_output('current_power','voltage_shift', 'current_wavelength')
        basis, value, decoy = self.generate_alice_choices_fixed(basis = 0, value = 0, decoy = 0)
        s_filtered_repeating, t_jitter = self.signal_bandwidth_jitter(value, decoy)
        power, _, calc_mean_photon_nr = self.eam_transmission(s_filtered_repeating, t_jitter, optical_power, peak_wavelength, basis, value, decoy)

        #first round: calculate T1 dampening_0
        if basis == 0:
            power_dampened = power / self.T1_dampening_0
        else:
            power_dampened = power / self.T1_dampening_1
        test_energy_pp = np.trapz(power_dampened, t_jitter)
        test_nr_photons_pp = test_energy_pp / (constants.h*constants.c/peak_wavelength)
        self.T1_dampening_0 = test_nr_photons_pp / self.mean_photon_nr
        T1_dampening_dB_0 = 10* np.log(self.mean_photon_nr / test_nr_photons_pp)  #<0 ist Abschwächung

        #for basis 1
        basis, value, decoy = self.generate_alice_choices_fixed(basis = 1, value = 0, decoy = 0)
        s_filtered_repeating, t_jitter = self.signal_bandwidth_jitter(value, decoy)
        power, _, calc_mean_photon_nr = self.eam_transmission(s_filtered_repeating, t_jitter, optical_power, peak_wavelength, basis, value, decoy)

        #first round: calculate T1 dampening_1
        test_energy_pp = np.trapz(power, t_jitter)
        test_nr_photons_pp = test_energy_pp / (constants.h*constants.c/peak_wavelength)
        self.T1_dampening_1 = test_nr_photons_pp / self.mean_photon_nr
        T1_dampening_dB_1 = 10* np.log(self.mean_photon_nr / test_nr_photons_pp)  #<0 ist Abschwächung
        return T1_dampening_dB_0, T1_dampening_dB_1'''
    
    def find_T1_neu(self, lower_limit, upper_limit, tol):
        # für 0 Basis
        optical_power, peak_wavelength = self.random_laser_output('current_power','voltage_shift', 'current_wavelength')
        basis, value, decoy = self.generate_alice_choices_fixed(basis = 0, value = 0, decoy = 0)

        while upper_limit - lower_limit > tol:
            self.T1_dampening_0 = (lower_limit + upper_limit) / 2
            s_filtered_repeating, t_jitter = self.signal_bandwidth_jitter(value, decoy)
            _, _, calc_mean_photon_nr = self.eam_transmission(s_filtered_repeating, t_jitter, optical_power, peak_wavelength, basis, value, decoy)
            print('calc_mean_photon_nr: ' + str(calc_mean_photon_nr) + 'with dampening t1_0: ' + str(self.T1_dampening_0))

            #compare calculated mean with target mean
            if calc_mean_photon_nr < self.mean_photon_nr:  #!hier kleiner weil durch T1 geteilt wird ! anders als in find_voltage
                upper_limit = self.T1_dampening_0 #reduce upper bound
            else:
                lower_limit = self.T1_dampening_0 #increase lower bound

        #final voltage decoy
        self.T1_dampening_0 = (lower_limit + upper_limit) / 2

        # für Basis 1
        optical_power, peak_wavelength = self.random_laser_output('current_power','voltage_shift', 'current_wavelength')
        basis, value, decoy = self.generate_alice_choices_fixed(basis = 1, value = 0, decoy = 0)

        while upper_limit - lower_limit > tol:
            self.T1_dampening_1 = (lower_limit + upper_limit) / 2
            s_filtered_repeating, t_jitter = self.signal_bandwidth_jitter(value, decoy)
            _, _, calc_mean_photon_nr = self.eam_transmission(s_filtered_repeating, t_jitter, optical_power, peak_wavelength, basis, value, decoy)
            print('calc_mean_photon_nr: ' + str(calc_mean_photon_nr))


            #compare calculated mean with target mean
            if calc_mean_photon_nr < self.mean_photon_decoy:   #!hier kleiner weil durch T1 geteilt wird !anders als in find_voltage
                upper_limit = self.T1_dampening_1 #reduce upper bound
            else:
                lower_limit = self.T1_dampening_1 #increase lower bound

        #final voltage decoy
        self.T1_dampening_1 = (lower_limit + upper_limit) / 2
        return None
    
    def binary_search_for_voltage_decoy(self, lower_limit, upper_limit, tol):
        basis, value, decoy = self.generate_alice_choices_fixed(basis = 0, value = 0, decoy = 1)
        optical_power, peak_wavelength = self.random_laser_output('current_power','voltage_shift', 'current_wavelength')

        while upper_limit - lower_limit > tol:
            print('T1_dampening_0: ' + str(self.T1_dampening_0))
            self.voltage_decoy = (lower_limit + upper_limit) / 2
            print('self.voltage_decoy: ' +str(self.voltage_decoy))
            s_filtered_repeating, t_jitter = self.signal_bandwidth_jitter(value, decoy)
            _, _, calc_mean_photon_nr = self.eam_transmission(s_filtered_repeating, t_jitter, optical_power, peak_wavelength, basis, value, decoy)
            print('calc_mean_photon_nr: ' + str(calc_mean_photon_nr))

            #compare calculated mean with target mean
            if calc_mean_photon_nr > self.mean_photon_decoy:
                upper_limit = self.voltage_decoy #reduce upper bound
            else:
                lower_limit = self.voltage_decoy #increase lower bound

        #final voltage decoy
        self.voltage_decoy = (lower_limit + upper_limit) / 2
        return None

    def initialize(self):
        #first round: calculate T1 dampening 
        self.find_T1_neu(lower_limit = 0, upper_limit = 100, tol = 1e-3)
        print('T1_dampening_0 at initialize end' +str(self.T1_dampening_0))
        print('T1_dampening_1 at initialize end' + str(self.T1_dampening_1))

        #with first decoy state: calculate decoy height
        self.binary_search_for_voltage_decoy(lower_limit=0, upper_limit=1, tol=1e-7)
        print('Voltage_decoy' + str(self.voltage_decoy))

        return None
    
    def run_simultation(self):
        
        self.initialize()
        optical_power, peak_wavelength = self.random_laser_output('current_power','voltage_shift', 'current_wavelength')
        basis, value, decoy = self.generate_alice_choices_fixed(basis = 0, value = 0, decoy = 0)   #11: 1000, 10: 0010, 00: 1010
        s_filtered_repeating, t_jitter = self.signal_bandwidth_jitter(value, decoy)
        power, power_dampened, calc_mean_photon_nr = self.eam_transmission(s_filtered_repeating, t_jitter, optical_power, peak_wavelength, basis, value, decoy)    
        
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
simulation = Simulation(database)   

end_time = time.time()  # Record end time
execution_time = end_time - start_time  # Calculate execution time
print(f"Execution time: {execution_time:.9f} seconds for {simulation.n_samples} samples")

#plot results
simulation.run_simultation()

end_time_2 = time.time()  # Record end time
execution_time_2 = end_time_2 - start_time  # Calculate execution time
print(f"Execution time after writing in Array: {execution_time_2:.9f} seconds for {simulation.n_samples} samples")

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