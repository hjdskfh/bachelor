import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import gc
from cycler import cycler
import re


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
    
    def plot_power(self, t, second_power, amount_symbols_in_plot=4, where_plot_1=None, shortened_first_power=None, where_plot_2=None, title_rest=None, shift=0, is_DLI=False):	
        step_size = t[1] - t[0]
        # Calculate the new length
        new_length = len(t) * amount_symbols_in_plot
        # Generate the new array with the same step size
        t_plot1 = np.arange(t[0], t[0] + step_size * new_length, step_size)
        second_part = second_power[shift:shift + amount_symbols_in_plot]
        second_part = second_part.reshape(-1)
        plt.figure(figsize=(8, 6))
       
        if shortened_first_power is None:
            plt.plot(t_plot1 * 1e9, second_part * 1e3, label = where_plot_1)
            plt.title(f"Power {where_plot_1} over {self.config.n_samples} Iterations")
        else:
            shortened_first_power = shortened_first_power.reshape(-1)
            plt.plot(t_plot1 * 1e9, shortened_first_power * 1e3, label=where_plot_1)
            plt.plot(t_plot1 * 1e9, second_part * 1e3, label=where_plot_2)
            plt.title(f"Power {title_rest} over {amount_symbols_in_plot} Symbols")
        
        plt.ylabel('Power (mW)')
        plt.xlabel('Time (ns)')
        plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))
        plt.legend()
        plt.tight_layout()
        if shortened_first_power is None:
            Saver.save_plot(f"power_{where_plot_1.replace(' ', '_').lower()}_for_{amount_symbols_in_plot}_symbols")
        else:
            Saver.save_plot(f"power_{amount_symbols_in_plot}_{where_plot_2.replace(' ', '_').lower()}_symbols")

        '''if is_DLI:
            plt.plot(t_plot1 * 1e9, second_part * 1e3, label = where_plot_1)
            plt.title(f"single_second_Power over {self.config.n_samples} Iterations") 
            plt.ylabel('Power (mW)')
            plt.xlabel('Time (ns)')
            plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))
            plt.legend()
            plt.tight_layout()       
            Saver.save_plot(f"single_second_power_{where_plot_1.replace(' ', '_').lower()}_for_{amount_symbols_in_plot}_symbols")

            if shortened_first_power is not None:
                shortened_first_power = shortened_first_power.reshape(-1)
                plt.plot(t_plot1 * 1e9, shortened_first_power * 1e3, label=where_plot_1)
                # plt.show()
                plt.plot(t_plot1 * 1e9, second_part * 1e3, label=where_plot_2)
                # plt.show()
                plt.ylabel('Power (mW)')
                plt.xlabel('Time (ns)')
                plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))
                plt.legend()
                plt.tight_layout()
                plt.title(f"single_first_Power over {amount_symbols_in_plot} Symbols")
                Saver.save_plot(f"single_first_power_{amount_symbols_in_plot}_{where_plot_2.replace(' ', '_').lower()}_symbols")'''

        del second_power
        gc.collect()
    
    def plot_and_delete_mean_photon_histogram(self, calc_mean_photon_nr, target_mean_photon_nr, type_photon_nr):
        """
        Plots a histogram of the mean photon number and saves the plot.

        Parameters:
        - calc_mean_photon_nr: Array of calculated mean photon numbers.
        - target_mean_photon_nr: Target mean photon number (vertical reference line).
        - filename: Name for saving the plot.
        """
        calc_mean_photon_nr = self.make_data_plottable(calc_mean_photon_nr)

        plt.figure(figsize=(8, 6))

        if type_photon_nr == "Mean Photon Number at Detector X":
            # Save the original color cycle
            original_cycle = plt.rcParams['axes.prop_cycle']

            # Create a modified cycle that skips the first color
            default_colors = original_cycle.by_key()['color']
            new_cycle = cycler('color', default_colors[1:])

            # Use the modified cycle only for this histogram plot
            with plt.rc_context({'axes.prop_cycle': new_cycle}):
                plt.hist(calc_mean_photon_nr, bins=40, alpha=0.7, label="Mean Photon Number")
        else:
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
        del calc_mean_photon_nr
        gc.collect()


    def plot_and_delete_photon_nr_histogram(self, nr_photons_det_x, nr_photons_det_z):
        """
        Plots a histogram of detected photon numbers for X and Z bases and saves the plot.

        Parameters:
        - nr_photons_det_x: Array of detected photon numbers in the X basis.
        - nr_photons_det_z: Array of detected photon numbers in the Z basis.
        """
        nr_photons_det_x = self.make_data_plottable(nr_photons_det_x)
        nr_photons_det_z = self.make_data_plottable(nr_photons_det_z)

        plt.figure(figsize=(8, 6))
        
        # Plot histograms for X and Z bases with specific bin range and style
        plt.hist(nr_photons_det_z, label='Z basis', bins=np.arange(0, 11) - 0.5, alpha = 0.7)
        plt.hist(nr_photons_det_x, label='X basis', bins=np.arange(0, 11) - 0.5, alpha = 0.7)

        # Formatting title and labels
        plt.title(f"Photon Number over {self.config.n_samples} Iterations")
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
        del nr_photons_det_x
        del nr_photons_det_z
        gc.collect()

    def plot_and_delete_photon_time_histogram(self, time_photons_det_x, time_photons_det_z):
        """
        Plots a histogram of detected photon times for X and Z bases and saves the plot.

        Parameters:
        - time_photons_det_x: Array of photon detection times in the X basis.
        - time_photons_det_z: Array of photon detection times in the Z basis.
        """
        time_photons_det_x = self.make_data_plottable(time_photons_det_x)
        time_photons_det_z = self.make_data_plottable(time_photons_det_z)

        plt.figure(figsize=(8, 6))
        
        # Convert times to ns and plot histograms
        plt.hist(time_photons_det_z * 1e9, label='Z basis', bins=40, zorder=1, alpha=0.7)
        plt.hist(time_photons_det_x * 1e9, label='X basis', bins=40, zorder=2, alpha=0.7)

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
        del time_photons_det_x
        del time_photons_det_z
        gc.collect()

    def plot_and_delete_photon_wavelength_histogram(self, wavelength_photons_det_x, wavelength_photons_det_z):
        """
        Plots a histogram of detected photon wavelengths for X and Z bases and saves the plot.

        Parameters:
        - wavelength_photons_det_x: Array of detected photon wavelengths in the X basis (in meters).
        - wavelength_photons_det_z: Array of detected photon wavelengths in the Z basis (in meters).
        """
        wavelength_photons_det_x = self.make_data_plottable(wavelength_photons_det_x)
        wavelength_photons_det_z = self.make_data_plottable(wavelength_photons_det_z)

        plt.figure(figsize=(8, 6))
        # print("max wavelength x", max(wavelength_photons_det_x))
        # print("max wavelength z", max(wavelength_photons_det_z))
        # Convert wavelengths to nanometers and plot histograms for X and Z bases
        plt.hist(wavelength_photons_det_z * 1e9, label='Z basis', bins=40, alpha=0.7, zorder=1)
        plt.hist(wavelength_photons_det_x * 1e9, label='X basis', bins=40, alpha=0.7, zorder=2)

        # Formatting title and labels
        plt.title(f"Photon Wavelength over {self.config.n_samples} Iterations")
        plt.ylabel('Counts')
        plt.xlabel('Photon Wavelength (nm)')

        # Ensure y-axis is integer-labeled
        plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))
        plt.legend()
        plt.tight_layout()

        # Save the plot
        filename = f"hist_wavelength_{self.config.n_samples}"
        Saver.save_plot(filename)
        del wavelength_photons_det_x
        del wavelength_photons_det_z
        gc.collect()

    def plot_and_delete_photon_wavelength_histogram_two_diagrams(self, wavelength_photons_det_x, wavelength_photons_det_z):
        """
        Plots a histogram of detected photon wavelengths for X and Z bases and saves the plot.

        Parameters:
        - wavelength_photons_det_x: Array of detected photon wavelengths in the X basis (in meters).
        - wavelength_photons_det_z: Array of detected photon wavelengths in the Z basis (in meters).
        """
        wavelength_photons_det_x = self.make_data_plottable(wavelength_photons_det_x)
        wavelength_photons_det_z = self.make_data_plottable(wavelength_photons_det_z)

        plt.figure(figsize=(8, 6))
        
        # print("max wavelength x", max(wavelength_photons_det_x))
        # print("max wavelength z", max(wavelength_photons_det_z))
        # Convert wavelengths to nanometers and plot histograms for X and Z bases
        plt.hist(wavelength_photons_det_z * 1e9, label='Z basis', bins=40, alpha=0.7, zorder=1)

        # Formatting title and labels
        plt.title(f"Photon Wavelength over {self.config.n_samples} Iterations for Z basis")
        plt.ylabel('Counts')
        plt.xlabel('Photon Wavelength (nm)')

        # Ensure y-axis is integer-labeled
        plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))
        plt.legend()
        plt.tight_layout()

        # Save the plot
        filename = f"hist_wavelength_{self.config.n_samples}"
        Saver.save_plot(filename)

        plt.hist(wavelength_photons_det_x * 1e9, label='X basis', bins=40, alpha=0.7, zorder=2)

        # Formatting title and labels
        plt.title(f"Photon Wavelength over {self.config.n_samples} Iterations for X basis")
        plt.ylabel('Counts')
        plt.xlabel('Photon Wavelength (nm)')

        # Ensure y-axis is integer-labeled
        plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))
        plt.legend()
        plt.tight_layout()

        # Save the plot
        filename = f"hist_wavelength_{self.config.n_samples}"
        Saver.save_plot(filename)
        del wavelength_photons_det_x
        del wavelength_photons_det_z
        gc.collect()

    def sanitize_filename(self, s):
        # Remove anything not alphanumeric, dot, underscore, or dash
        return re.sub(r'[^\w\-_\. ]', '_', s).replace(' ', '_')


    def prepare_data_for_histogram(self, time_photons_det_x, time_photons_det_z, last_histogram_matrix=None):
        # Example parameters
        num_symbols = 65
        bins_per_symbol = 30
        symbol_window = 100  # Example time window for one symbol (adjust as needed)

        # Placeholder example time matrix (replace this with real data!)
        # Let's say we expect up to 10 detections per symbol repetition
        # NaN = no detection

        # Prepare histogram matrix
        if last_histogram_matrix is None:
            histogram_matrix = np.zeros((num_symbols, bins_per_symbol), dtype=int)
        else:
            histogram_matrix = last_histogram_matrix

        # Create bins
        bins = np.linspace(0, symbol_window, bins_per_symbol + 1)  # 30 bins

        # Loop over symbols
        for i in range(num_symbols):
            # Extract non-NaN times for this symbol
            times = time_photons_det_x[i, ~np.isnan(time_photons_det_x[i])]
            
            # Histogram
            counts, _ = np.histogram(times, bins=bins)
            
            # Store counts
            histogram_matrix[i, :] = counts

        print("Histogram matrix shape:", histogram_matrix.shape)  # (65, 30)
