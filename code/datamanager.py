import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import splrep, InterpolatedUnivariateSpline
import pandas as pd 
from saver import Saver


class DataManager:
    def __init__(self):
        self.curves = {}

    def add_data(self, csv_file, column1, column2, rows, name, do_inverse=False, parabola=False):
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

        if do_inverse and parabola:
            # Split the data into two parts: one for positive x, one for negative x
            positive_mask = df[column1] >= 0
            negative_mask = df[column1] < 0

            # For positive x-values, sort the data while preserving the index
            df_pos_sorted = df[positive_mask].sort_values(by=column1)
            x_vals_pos_sorted = df_pos_sorted[column1].values
            y_vals_pos_sorted = df_pos_sorted[column2].values

            # For negative x-values, sort the data while preserving the index
            df_neg_sorted = df[negative_mask].sort_values(by=column1)
            x_vals_neg_sorted = df_neg_sorted[column1].values
            y_vals_neg_sorted = df_neg_sorted[column2].values

            x_vals_neg_sorted = x_vals_neg_sorted[::-1]
            y_vals_neg_sorted = y_vals_neg_sorted[::-1]
            # Reverse the negative values to make sure they are strictly increasing for the spline
            x_vals_neg_sorted = -x_vals_neg_sorted  # Multiply by -1 to reverse the direction
            
            if not np.all(np.diff(x_vals_pos_sorted) > 0) or not np.all(np.diff(x_vals_neg_sorted) > 0):
                raise ValueError("After sorting and reversing, x values should be strictly increasing.")
        
            # Create the regular spline for the full curve
            tck = splrep(df[column1], df[column2])
            # Create inverse splines for both positive and negative regions
            inverse_spline = None            
            inverse_spline_positive = InterpolatedUnivariateSpline(y_vals_pos_sorted, x_vals_pos_sorted)
            inverse_spline_negative = InterpolatedUnivariateSpline(y_vals_neg_sorted, x_vals_neg_sorted)
            
        elif do_inverse:
            tck = splrep(df[column1], df[column2])  # Regular spline only
            inverse_spline = InterpolatedUnivariateSpline(df[column2], df[column1])
            inverse_spline_positive = None
            inverse_spline_negative = None   
        else:
            tck = splrep(df[column1], df[column2])
            inverse_spline = None
            inverse_spline_positive = None
            inverse_spline_negative = None         

        self.curves[name] = {
            'tck': tck,  # Store the tck
            'inverse_spline_positive': inverse_spline_positive,  # Positive inverse spline
            'inverse_spline_negative': inverse_spline_negative,  # Negative inverse spline
            'inverse_spline': inverse_spline,  # General inverse spline
            'x_min': x_min,  # Store minimum x-value
            'x_max': x_max   # Store maximum x-value
            }
        # plt.style.use("C:\\Users\\leavi\\bachelor\\code\\Presentation_style_1_adjusted_no_grid.mplstyle")
        # # Plot the data
        # title = ''
        # if name == 'voltage_shift':
        #     title = 'Wavelength Shift depending on Voltage of Heater'
        # if name == 'current_power':
        #     title = 'Optical Power depending on Current of Laser Diode'
        # if name == 'eam_transmission':
        #     title = 'Optical Transmission depending on Voltage given to the EAM'
        # if name == 'wavelength_neff':
        #     title = 'Effective Refractive Index in Fiber depending on Wavelength'
        # plt.plot(df[column1], df[column2])
        # plt.xlabel(column1)
        # plt.ylabel(column2)
        # plt.grid(True)
        # Saver.save_plot(f'{name}_data_plot.png', no_time = True)
    
    def add_jitter(self, jitter, name): #Gaussian
        # Calculate standard deviation based on FWHM
        std_dev = jitter / (2 * np.sqrt(2 * np.log(2)))

        # Define a range of values (e.g., -3 to 3 standard deviations)
        x = np.linspace(-3*std_dev, 3*std_dev, 100)

        # Compute Gaussian weights
        weights = np.exp(-0.5 * (x / std_dev) ** 2)

        # Normalize weights to get probabilities (sum to 1)
        probabilities_array = weights / weights.sum()
        self.curves['probabilities' + name] = {
            'prob': probabilities_array,
            'x': x
            }

    def get_probabilities(self, x_data, name):
        if not name.startswith("probabilities"):
            raise ValueError(f"Invalid name for probabilities: {name}")
        return self.curves[name]['prob'], self.curves[name]['x']

    def get_data(self, x_data, name, inverse=False):
        x_min = self.curves[name]['x_min']
        x_max = self.curves[name]['x_max']
        if isinstance(x_data, np.ndarray) and x_data.ndim == 2:
            print(f"First few elements of x_data: {x_data[:5, :5]}")
            print(f"x_min: {x_min}, x_max: {x_max}")
            print(f"x_data shape: {x_data.shape}")
        if np.isscalar(x_data):
            if x_data < x_min or x_data > x_max:
                raise ValueError(x_data, " x data isn't in table for ", name, " and x_min: ", x_min, " and x_max: ", x_max)
        elif isinstance(x_data, np.ndarray):
            if x_data.ndim == 1:
                if (x_data < x_min).any() or (x_data > x_max).any():
                    out_of_bounds = np.where((x_data < x_min) | (x_data > x_max))
                    raise ValueError(x_data, " x data array isn't in table for ", name, " x_data: ", x_data, 
                                    " x_min ", x_min, " and x_max: ", x_max, " out of bounds index: ", out_of_bounds, 
                                    " out of bounds: ",x_data[out_of_bounds])
            else:
                out_of_bounds = np.where((x_data < x_min) | (x_data > x_max))
                if out_of_bounds[0].size > 0:
                    out_of_bounds_values = x_data[out_of_bounds]
                    out_of_bounds_indices = list(zip(out_of_bounds[0], out_of_bounds[1]))  # Pair indices
                    raise ValueError(x_data, " x data array isn't in table for ", name, " x_data: ", x_data, 
                                    " x_min ", x_min, " and x_max: ", x_max, " out of bounds index: ", out_of_bounds, 
                                    " out of bounds: ",x_data[out_of_bounds])
        else:
            raise ValueError("Invalid x_data type")
        if name not in self.curves:
            raise ValueError(f"Spline '{name}' not found.")
        
        if inverse:
            if self.curves[name]['inverse_spline'] is not None:
                return self.curves[name]['inverse_spline']
            elif x_data >= 0:
                return self.curves[name]['inverse_spline_positive']
            elif x_data < 0:
                return self.curves[name]['inverse_spline_negative']
            else:
                raise ValueError("Invalid x_data value for inverse spline.")

        return self.curves[name]['tck'] # Return tck

    def get_data_x_min_x_max(self, name):
        x_min = self.curves[name]['x_min']
        x_max = self.curves[name]['x_max']
        return x_min, x_max
    
    def show_data(self, csv_file, column1, column2, rows):
        table = pd.read_csv(csv_file, nrows = rows)
        table.columns = table.columns.str.strip()
        plt.plot(table[column1], table[column2])
        if csv_file == 'data/eam_transmission_data.csv':
            plt.ylabel('transmission')
            plt.xlabel('voltage in V')
        plt.show()
