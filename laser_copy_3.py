import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import splrep, splev
import pandas as pd 
import time


class dataManager:
    def __init__(self):
        self.curves = {}

    def add_data(self, csv_file, column1, column2, rows, name):
        df = pd.read_csv(csv_file, nrows = rows)              #erste 9 Datenpunkte
        df.columns = df.columns.str.strip()
        self.curves[name] = {
            'tck': splrep(df[column1], df[column2]),  # Store the tck
            }


    def get_data(self, name):
        if name not in self.curves:
            raise ValueError(f"Spline '{name}' not found.")
        return self.curves[name]['tck'] # Return tck

class Simulation:
    def __init__(self, data, n_samples=10000):
        self.data = data
        self.n_samples = n_samples  # Number of samples to generate
        #self.outputs = [] 

    def get_interpolated_value(self, x_data, which_data):
        #calculate tck for which curve
        tck = self.data.get_data(which_data)
        return splev(x_data, tck)

    def random_laser_output(self, current_power, voltage_shift, current_wavelength):
        # Generate a random time within the desired range
        time = np.random.uniform(0, 10)
        
        # Calculate voltage and current based on this time
            #voltage_heater = 1 in V, voltage_amplitude = 0.050 in V, voltage_frequency = 1
        chosen_voltage = 1 + 0.050 * np.sin(2 * np.pi * 1 * time)  
            #current_laserdiode = 0.08 in A, current_amplitude = 0.020 in A, current_frequency = 1
        chosen_current = 0.08 + 0.02 * np.sin(2 * np.pi * 1 * time) * 1e3 #damit in mA

        optical_power = self.get_interpolated_value(chosen_current, current_power)
        peak_wavelength = self.get_interpolated_value(chosen_current, current_wavelength) + self.get_interpolated_value(chosen_voltage, voltage_shift)
        return optical_power, peak_wavelength


    def get_output(self):
        #alternative: self.laser_outputs = [self.random_laser_output() for _ in range(self.n_samples)]
        all_optical_power = np.empty(self.n_samples)
        all_peak_wavelength = np.empty(self.n_samples)

        for i in range(self.n_samples):
            all_optical_power[i], all_peak_wavelength[i] = self.random_laser_output('current_power','voltage_shift', 'current_wavelength')
        return all_optical_power, all_peak_wavelength




#PROGRAMM

#measure execution time
start_time = time.time()  # Record start time

#database
database = dataManager()

#readin
database.add_data('current_power_data.csv', 'Current (mA)', 'Optical Power (mW)', 9, 'current_power') 
database.add_data('voltage_shift_data.csv', 'Voltage (V)', 'Wavelength Shift (nm)', 20, 'voltage_shift')
database.add_data('current_wavelength_modified.csv', 'Current (mA)', 'Wavelength (nm)', 9, 'current_wavelength')#modified sodass mA Werte stimmen (/1000)

#create simulation
simulation = Simulation(database)   
#wenn Zeit hier gestoppt wird: 0.0000s für 10000 Iterationen
end_time = time.time()  # Record end time
execution_time = end_time - start_time  # Calculate execution time
print(f"Execution time: {execution_time:.9f} seconds for {simulation.n_samples} samples")

#plot results
optical_power, peak_wavelength = simulation.get_output()

end_time_2 = time.time()  # Record end time
execution_time_2 = end_time_2 - start_time  # Calculate execution time
print(f"Execution time after writing in Array: {execution_time_2:.9f} seconds for {simulation.n_samples} samples")

plt.hist(optical_power, bins=30, label='Optical Power', alpha=0.7)
plt.title('integrated optical power over all iterations',size = 14)
plt.xlabel('iterations')
plt.ylabel('integrated optical power in mW')
plt.show()

plt.hist(peak_wavelength, bins=30, label='Peak Wavelength', alpha=0.7)
plt.title('peak wavelengths over all iterations',size = 14)
plt.xlabel('iterations')
plt.ylabel('peak wavelength in nm')
plt.show()