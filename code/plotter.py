import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from saver import Saver

class Plotter:
    def __init__(self, config):
        self.config = config 

    def make_data_plottable(self, data):
            data = data[~np.isnan(data)]
            if str(data).startswith("nr_photons"):
                # Step 1: Calculate the number of zeros (no photons detected)
                num_zeros = self.config.n_samples - len(data)

                # Step 2: Create the full dataset by combining the nr_photons and the zeros
                zeros = np.zeros(num_zeros)
                data = np.concatenate((data, zeros))

            # If data is empty after removing NaNs, return an empty array
            if data.size == 0:
                return np.array([])

            # If data is a scalar or zero-dimensional array, return it as a 1D array
            if np.isscalar(data) or data.ndim == 0:
                return np.array([data])
                
            # If data contains a single array, return it directly
            if len(data) == 1:
                return np.array(data)  # No need for np.concatenate

            # Flatten and concatenate all elements
            return np.concatenate([arr.flatten() for arr in data])
    
    def plot_mean_photon_histogram(self, calc_mean_photon_nr, target_mean_photon_nr, type_photon_nr):
        """
        Plots a histogram of the mean photon number and saves the plot.

        Parameters:
        - calc_mean_photon_nr: Array of calculated mean photon numbers.
        - target_mean_photon_nr: Target mean photon number (vertical reference line).
        - filename: Name for saving the plot.
        """
        plt.figure(figsize=(8, 6))
        plt.hist(calc_mean_photon_nr, bins=40, alpha=0.7, label="Mean Photon Number")
        if target_mean_photon_nr is not None:
            for i in range(len(target_mean_photon_nr)):
                plt.axvline(target_mean_photon_nr[i], label=f'Target Mean Photon Number {i+1}')

        # Formatting title and labels
        plt.title(f"{type_photon_nr} over {self.config.n_samples} Iterations")
        plt.ylabel('Counts')
        plt.xlabel(r'$\Delta \langle \mu \rangle$')

        # Ensure y-axis is integer-labeled
        plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))
        plt.legend()
        plt.tight_layout()

        # Save the plot
        filename = f"hist_mean_photon_nr_{self.config.n_samples}"
        Saver.save_plot(filename)

    def plot_photon_number_histogram(self, nr_photons_det_x, nr_photons_det_z, type_photon_number = None):
        """
        Plots a histogram of detected photon numbers for X and Z bases and saves the plot.

        Parameters:
        - nr_photons_det_x: Array of detected photon numbers in the X basis.
        - nr_photons_det_z: Array of detected photon numbers in the Z basis.
        - type_photon_number: A string describing the type of photon number being plotted.
        """
        plt.figure(figsize=(8, 6))
        
        # Plot histograms for X and Z bases with specific bin range and style
        plt.hist(nr_photons_det_x, label='X basis', bins=np.arange(0, 11) - 0.5)
        plt.hist(nr_photons_det_z, label='Z basis', bins=np.arange(0, 11) - 0.5)

        # Formatting title and labels
        plt.title(f"{type_photon_number} over {self.config.n_samples} Iterations")
        plt.ylabel('Counts')
        plt.xlabel('Photon Number')

        # Set x-ticks to be integers
        plt.xticks(np.arange(0, 11))

        # Ensure y-axis is integer-labeled
        plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))
        plt.legend()
        plt.tight_layout()

        # Save the plot
        filename = f"hist_photon_number_{self.config.n_samples}"
        Saver.save_plot(filename)

    def plot_photon_time_histogram(self, time_photons_det_x, time_photons_det_z):
        """
        Plots a histogram of detected photon times for X and Z bases and saves the plot.

        Parameters:
        - time_photons_det_x: Array of photon detection times in the X basis.
        - time_photons_det_z: Array of photon detection times in the Z basis.
        """
        plt.figure(figsize=(8, 6))
        
        # Convert times to ns and plot histograms
        plt.hist(time_photons_det_x * 1e9, label='X basis', bins=40, zorder=2, alpha=0.7)
        plt.hist(time_photons_det_z * 1e9, label='Z basis', bins=40, zorder=1, alpha=0.7)

        # Formatting title and labels
        plt.title(f"Photon Time Distribution over {self.config.n_samples} Iterations")
        plt.ylabel('Counts')
        plt.xlabel('Photon Time (ns)')

        # Ensure y-axis is integer-labeled
        plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))
        plt.legend()
        plt.tight_layout()

        # Save the plot
        filename = f"hist_photon_time_{self.config.n_samples}"
        Saver.save_plot(filename)

    def plot_photon_wavelength_histogram(self, wavelength_photons_det_x, wavelength_photons_det_z):
        """
        Plots a histogram of detected photon wavelengths for X and Z bases and saves the plot.

        Parameters:
        - wavelength_photons_det_x: Array of detected photon wavelengths in the X basis (in meters).
        - wavelength_photons_det_z: Array of detected photon wavelengths in the Z basis (in meters).
        """
        plt.figure(figsize=(8, 6))
        
        # Convert wavelengths to nanometers and plot histograms for X and Z bases
        plt.hist(wavelength_photons_det_x * 1e9, label='X basis', bins=40, alpha=0.7, zorder=2)
        plt.hist(wavelength_photons_det_z * 1e9, label='Z basis', bins=40, alpha=0.7, zorder=1)

        # Formatting title and labels
        plt.title(f"photon wavelength over {self.config.n_samples} Iterations")
        plt.ylabel('Counts')
        plt.xlabel('Photon Wavelength (nm)')

        # Ensure y-axis is integer-labeled
        plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))
        plt.legend()
        plt.tight_layout()

        # Save the plot
        filename = f"hist_wavelength_{self.config.n_samples}"
        Saver.save_plot(filename)



        