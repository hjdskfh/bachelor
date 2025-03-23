#import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft, ifft, fftfreq

# Set style for plot
#plt.style.use(os.path.abspath('C:/Users/juliu/Nextcloud/Documents/Python/Plotting/Presentation_style_1.mplstyle'))

# Constants
freq = 6.75e9 / 2  # Frequency of the square wave signal
pulse_duration = 1 / freq  # Pulse duration for a 3.25 GHz square wave
sampling_rate = 100e11  # High sampling rate for accurate FFT
t = np.arange(-5 * pulse_duration, 5 * pulse_duration, 1 / sampling_rate)  # Time vector

# Create a repeating square wave signal in time domain
num_cycles = 2  # Number of cycles of the square wave
t_repeating = np.arange(-num_cycles * pulse_duration, num_cycles * pulse_duration, 1 / sampling_rate)
repeating_square_pulse = np.sign(np.sin(2 * np.pi * freq * t_repeating))

# Fourier transform to frequency domain for the repeating signal
n_repeating = len(t_repeating)
S_f_repeating = fft(repeating_square_pulse)
frequencies_repeating = fftfreq(n_repeating, d=1 / sampling_rate)

# Plot the original signal
plt.figure(figsize=(12, 6))
plt.plot(t_repeating * 1e9, repeating_square_pulse, label="Original Signal", alpha=1, marker="")

# Define cutoffs and apply smoother filtering for frequency response
cutoffs = [1e9, 2e9, 3e9, 4e9, 5e9, 10e9, 20e9, 30e9]
for cutoff in cutoffs:
    # Use a smoother transition for the frequency response
    freq_x = [0, cutoff * 0.8, cutoff, cutoff * 1.2, sampling_rate / 2]
    freq_y = [1, 1, 0.7, 0.01, 0.001]  # Gradual drop-off for a smoother response
    
    # Apply the frequency filter
    S_filtered_repeating = S_f_repeating * np.interp(np.abs(frequencies_repeating), freq_x, freq_y)
    
    # Inverse Fourier transform back to the time domain
    s_filtered_repeating = np.real(ifft(S_filtered_repeating))
    
    # Plot the filtered signal
    plt.plot(t_repeating * 1e9, s_filtered_repeating, label=f"Cutoff: {cutoff/1e9} GHz", alpha=0.7, marker ="")

# Final plot adjustments
plt.title("Square Signal with Bandwidth Limitation")
plt.xlabel("Time (ns)")
plt.ylabel("Amplitude")
plt.legend()
plt.grid(True)
plt.show()
