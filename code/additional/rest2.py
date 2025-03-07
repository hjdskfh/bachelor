import numpy as np
import matplotlib.pyplot as plt

# Define original parameters
num_samples = 6150
dt_original = 0.1e-9  # 0.1 ns
fs_original = 1 / dt_original
frequencies_original = np.fft.fftfreq(num_samples, d=dt_original)

# Define new dt (increased)
dt_new = 0.2e-9  # 0.2 ns
fs_new = 1 / dt_new
frequencies_new = np.fft.fftfreq(num_samples, d=dt_new)

# Plot frequency resolutions
plt.figure(figsize=(8, 4))
plt.plot(frequencies_original[:num_samples//2] / 1e12, label="Original dt=0.1 ns")
plt.plot(frequencies_new[:num_samples//2] / 1e12, label="Modified dt=0.2 ns", linestyle="dashed")
plt.xlabel("Frequency (THz)")
plt.ylabel("FFT Frequency Bins")
plt.title("Effect of Increasing dt on Frequency Resolution")
plt.legend()
plt.show()
