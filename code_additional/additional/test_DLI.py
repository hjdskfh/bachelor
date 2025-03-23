import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as constants

# Define wavelength and k
wavelength = 1550e-9  # 1550 nm
k = 2 * np.pi / wavelength  # Wave number

# Define optical path difference
delta_L = 10e-6  # 10 µm path difference
alpha = 0.9  # Transmission efficiency

# Create a time array (e.g., 1000 time samples over 1 ns)
time = np.linspace(0, 1e-9, 1000)  # 1 ns duration

# Define time-dependent input power P0(t) (e.g., random modulated signal)
P0_t = np.abs(np.sin(2 * np.pi * 5e9 * time))  # Example: 5 GHz sinusoidal power variation

# Apply interference formula dynamically over time
P_plus_t = (alpha * P0_t / 2) * (1 + np.cos(k * delta_L))
P_minus_t = (alpha * P0_t / 2) * (1 - np.cos(k * delta_L))

# Plot the results
plt.figure(figsize=(8, 5))
plt.plot(time * 1e9, P0_t, label="Input Power $P_0(t)$", linestyle='dotted', color="black")
plt.plot(time * 1e9, P_plus_t, label=r'$P_{out,+}(t)$')
plt.plot(time * 1e9, P_minus_t, label=r'$P_{out,-}(t)$')
plt.xlabel('Time (ns)')
plt.ylabel('Power')
plt.title('DLI Interference Over Time')
plt.legend()
plt.grid()
plt.show()
