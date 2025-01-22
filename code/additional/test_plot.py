import numpy as np
import time

# Create random data of shape (1000000, 6153)
x_data = np.random.randn(100000, 6153)

# Define some min and max values for the bounds check
x_min = -2
x_max = 2

# Start the timer
start_time = time.time()

# Perform np.where on the large array
out_of_bounds = np.where((x_data < x_min) | (x_data > x_max))

# Measure the time elapsed
elapsed_time = time.time() - start_time
print(f"Execution time: {elapsed_time:.5f} seconds")
