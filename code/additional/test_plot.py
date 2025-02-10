import numpy as np
import time

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import entropy

# Simulated detection data
np.random.seed(42)
N = 100000  # Number of detected events

# Generate random bits and bases
alice_bits = np.random.randint(0, 2, N)
alice_bases = np.random.choice(['X', 'Z'], N, p=[0.5, 0.5])
bob_bases = np.random.choice(['X', 'Z'], N, p=[0.5, 0.5])

# Simulate Bob's measurements with a small error rate
error_rate = 0.02
bob_bits = np.where(np.random.rand(N) < error_rate, 1 - alice_bits, alice_bits)

# Sifting: Keep only events where Alice and Bob chose the same basis
sift_mask = alice_bases == bob_bases
alice_sifted = alice_bits[sift_mask]
bob_sifted = bob_bits[sift_mask]
num_sifted = len(alice_sifted)

# QBER Calculation
qber = np.mean(alice_sifted != bob_sifted)
print(f"Quantum Bit Error Rate (QBER): {qber:.4f}")

# Histogram of detected time bins
hist, bins = np.histogram(np.random.randn(N), bins=50)
plt.figure(figsize=(8, 4))
plt.bar(bins[:-1], hist, width=(bins[1] - bins[0]))
plt.xlabel("Time bin")
plt.ylabel("Counts")
plt.title("Detection Histogram")
plt.show()

# Raw Key Rate Calculation
raw_key_rate = num_sifted / N  # Proportion of key bits remaining after sifting
print(f"Raw Key Rate: {raw_key_rate:.4f}")

# Decoy State Analysis (Example with Three Intensities: signal, weak decoy, vacuum decoy)
mu_signal = 0.8
mu_weak = 0.1
mu_vacuum = 0.02
prob_signal = 0.7
prob_weak = 0.2
prob_vacuum = 0.1

num_signal = int(prob_signal * num_sifted)
num_weak = int(prob_weak * num_sifted)
num_vacuum = int(prob_vacuum * num_sifted)

error_signal = np.random.rand(num_signal) < error_rate
error_weak = np.random.rand(num_weak) < error_rate
error_vacuum = np.random.rand(num_vacuum) < error_rate

qber_signal = np.mean(error_signal)
qber_weak = np.mean(error_weak)
qber_vacuum = np.mean(error_vacuum)
print(f"QBER (Signal): {qber_signal:.4f}, QBER (Weak Decoy): {qber_weak:.4f}, QBER (Vacuum Decoy): {qber_vacuum:.4f}")

# Privacy Amplification (Using a simple hashing technique)
def hash_function(key, length):
    """Simple hashing function using XOR"""
    hashed_key = key[:length] ^ key[length:2*length]  # Example: XOR first half with second half
    return hashed_key

final_key_length = int(num_sifted * (1 - qber))  # Reduce key length based on errors
final_key = hash_function(alice_sifted, final_key_length)
print(f"Final Secret Key Length: {len(final_key)}")
