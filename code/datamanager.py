import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import splrep
import pandas as pd 
from saver import Saver


class DataManager:
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
        
        '''# Plot the data
        plt.figure(figsize=(10, 6))
        plt.plot(df[column1], df[column2], label=name, marker='o', linestyle='-', color='b')
        plt.xlabel(column1)
        plt.ylabel(column2)
        plt.title(f'{name} Data Plot')
        plt.grid(True)
        plt.legend()
        Saver.save_plot(f'{name}_data_plot.png')'''
    
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

    def get_data(self, x_data, name):
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
                print(f"out_of_bounds: {out_of_bounds}")
                if out_of_bounds[0].size > 0:
                    out_of_bounds_values = x_data[out_of_bounds]
                    out_of_bounds_indices = list(zip(out_of_bounds[0], out_of_bounds[1]))  # Pair indices
                    print("Out of bounds values:", out_of_bounds_values[:20])
                    print("Out of bounds indices:", out_of_bounds_indices)
                    raise ValueError(x_data, " x data array isn't in table for ", name, " x_data: ", x_data, 
                                    " x_min ", x_min, " and x_max: ", x_max, " out of bounds index: ", out_of_bounds, 
                                    " out of bounds: ",x_data[out_of_bounds])
        else:
            raise ValueError("Invalid x_data type")
        if name not in self.curves:
            raise ValueError(f"Spline '{name}' not found.")
        
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
