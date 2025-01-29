import numpy as np
import time

# Test with smaller data size
rows = 50000  # Reducing rows from 100,000 to 50,000
cols = 6150   # Keep the same number of columns

# Create a random matrix with the shape (rows, cols)
matrix = np.random.random((rows, cols))

# Measure time to perform np.roll for each row
start_time = time.time()

rolled_matrix = np.copy(matrix)  # Copy to avoid in-place modification

for i in range(rolled_matrix.shape[0]):
    rolled_matrix[i] = np.roll(rolled_matrix[i], shift=30)

end_time = time.time()

# Output the time taken for the operation
elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time:.6f} seconds")
