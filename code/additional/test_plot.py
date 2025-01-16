import random
import numpy as np
import time

# Timing Python's random module for 100,000 * 3 iterations
start = time.time()
random.seed(42)
for _ in range(100000 * 3):
    random_number = random.randint(0, 10)  # generating one random number
random_time = time.time() - start
print("random module time:", random_time)

# Timing NumPy's default_rng for 100,000 * 3 iterations
start = time.time()
rng = np.random.default_rng(42)
for _ in range(100000 * 3):
    numpy_number = rng.integers(0, 10)  # generating one random number
numpy_time = time.time() - start
print("NumPy RNG time:", numpy_time)