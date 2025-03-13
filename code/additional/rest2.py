import numpy as np
import matplotlib.pyplot as plt

# Parameters
pulse_width = 0.3e-9  # 0.3 ns
time = np.linspace(-2e-9, 2e-9, 1000)  # time window

# Gaussian pulse
amplitude = np.exp(-time**2 / (2 * (pulse_width/2.355)**2))  # FWHM -> sigma

# Fourier Transform (spectrum)
spectrum = np.fft.fftshift(np.fft.fft(amplitude))
freq = np.fft.fftshift(np.fft.fftfreq(len(time), d=(time[1] - time[0])))

# Plot
plt.figure(figsize=(10, 5))
plt.plot(freq * 1e-9, np.abs(spectrum))  # GHz scale
plt.xlabel('Frequency (GHz)')
plt.ylabel('Amplitude Spectrum')
plt.title('Pulse Amplitude Spectrum (no linewidth)')
plt.grid(True)
plt.show()

# Define laser linewidth as Lorentzian (5 MHz)
linewidth = 5e6  # 5 MHz
lorentzian = 1 / (1 + ((freq) / (linewidth / 2))**2)
lorentzian /= np.sum(lorentzian)  # Normalize

# Convolve (approximate via multiplication and renormalization)
broadened_spectrum = np.abs(spectrum) * lorentzian

# Plot
plt.figure(figsize=(10, 5))
plt.plot(freq * 1e-9, broadened_spectrum)
plt.xlabel('Frequency (GHz)')
plt.ylabel('Amplitude Spectrum (with linewidth)')
plt.title('Pulse Spectrum with Laser Linewidth Broadening')
plt.grid(True)
plt.show()

