import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import time

# Define DLI function
def delay_line_interferometer(P_in, dt, tau, delta_L, f0, splitting_ratio = 0.5):
    """
    Simulates the behavior of a delay line interferometer (DLI), which is used to 
    measure phase differences between two optical signals.
        P_in (array-like): Input optical power as a function of time.
        dt (float): Sampling time step for calculation, not (!) signal sampling rate (seconds).
        tau (float): Time delay introduced by the interferometer (seconds).
        delta_L (float): Path length difference between the two arms of the interferometer (meters).
        f0 (float): Optical carrier frequency of the input signal (Hz).
        splitting_ratio (float, optional): Splitting ratio of the coupler. Defaults to 0.5 
                (ideal 50/50 coupler).
        tuple: A tuple containing:
            - np.ndarray: Output power at the first port of the interferometer.
            - np.ndarray: Output power at the second port of the interferometer.
            - np.ndarray: Time array corresponding to the input signal.
    """
    c = 3e8  # speed of light in vacuum

    # Time array
    t = np.arange(len(P_in)) * dt
    
    # Input optical field (assuming carrier frequency)
    E0 = np.sqrt(P_in)
    E_in = E0 * np.exp(1j * 2 * np.pi * f0 * t)

    # Interpolate for delayed version
    interp_real = interp1d(t, np.real(E_in), bounds_error=False, fill_value=0.0)
    interp_imag = interp1d(t, np.imag(E_in), bounds_error=False, fill_value=0.0)
    E_in_delayed = interp_real(t - tau) + 1j * interp_imag(t - tau)

    # Phase shift from path length difference
    phi = 2 * np.pi * f0 * delta_L / c
    E_in_delayed *= np.exp(1j * phi)
    print(f"tau = {tau*1e12:.2f} ps, delta_L = {delta_L*1e6:.2f} µm, phase shift φ = {phi:.2f} rad")

    # Ideal 50/50 coupler outputs
    E_out1 = splitting_ratio * (E_in - E_in_delayed)
    E_out2 = (1-splitting_ratio)*1j* (E_in + E_in_delayed)

    return np.abs(E_out1)**2, np.abs(E_out2)**2, t

start_time = time.time()
# Signal and simulation setup
bit_sequence ="010100010101010000010100"
bit_rate = 3.25e9 
sample_rate = 1 / 2e-14  # too low relults in poor visibility? maybe only with square pulses
dt = 2e-14
samples_per_bit = int(sample_rate / bit_rate)

# Create pulse train with random amplitude variation
pulse_train = []
rng = np.random.default_rng(seed=42)
for bit in bit_sequence:
    if bit == "1":
        amplitude = 1.0 + rng.uniform(-0.01, 0.01) # to add imperfections
        t = np.linspace(-1, 1, samples_per_bit)
        gaussian_pulse = amplitude * np.exp(-0.5 * (t / 0.3)**2)  # Gaussian with std dev ~0.3
        pulse_train.extend(gaussian_pulse)
    else:
        pulse_train.extend([0.0] * samples_per_bit)

P_in = np.array(pulse_train)
t = np.arange(len(P_in)) * dt

# Delay and path length
tau = 2 / bit_rate  # Should be 1/bit_rate but that doesn make sense???
n_g = 2.05 # For calculatting path length difference
# Assuming the group refractive index of the waveguide
c = 3e8
delta_L = tau * c / n_g

print(f"Path length difference (delta_L): {delta_L:.6f} m")

# Wavelength sweep setup
lambda0 = 1550e-9  # 1550 nm
delta_lambda = 1e-10  # ±500 pm
wavelengths = np.linspace(lambda0 - delta_lambda / 2, lambda0 + delta_lambda / 2, 50)
frequencies = c / wavelengths  # convert to optical frequencies

# Run simulation over swept wavelengths
results = []
for f0 in frequencies:
    P1, P2, _ = delay_line_interferometer(P_in, dt, tau, delta_L, f0)
    results.append((f0, P1, P2))

'''# Plot result
plt.figure(figsize=(14, 6))
for f0, P1, P2 in results:
    wavelength = c / f0 * 1e9  # in nm
    plt.plot(t * 1e9, P1, label=f'Port 1 @ {wavelength:.2f} nm')
plt.xlabel("Time (ns)")
plt.ylabel("Power (a.u.)")
plt.title("DLI Output Port 1 Over Wavelength Sweep (~1550 nm ± 250 pm)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()'''

# Calculate visibility for each wavelength sweep
visibilities = []
wavelengths_nm = []

for f0, P1, _ in results:
    I_max = np.max(P1)
    I_min = np.min(P1)
    visibility = (I_max - I_min) / (I_max + I_min)
    visibilities.append(visibility)
    wavelengths_nm.append(c / f0 * 1e9)  # Convert frequency to wavelength in nm

# Find the index corresponding to 1 ns
target_time_ns = 1.0
target_index = np.argmin(np.abs(t * 1e9 - target_time_ns))

# Collect amplitudes at 1 ns for Port 1 and Port 2
amplitudes_port1 = []
amplitudes_port2 = []
wavelengths_nm = []

for f0, P1, P2 in results:
    amplitudes_port1.append(P1[target_index])
    amplitudes_port2.append(P2[target_index])
    wavelengths_nm.append(c / f0 * 1e9)  # Convert frequency to wavelength in nm

# Plot amplitudes at 1 ns vs wavelength
plt.figure(figsize=(8, 5))
plt.plot(wavelengths_nm, amplitudes_port1, marker='o', label='Port 1')
plt.plot(wavelengths_nm, amplitudes_port2, marker='s', label='Port 2')
plt.xlabel("Wavelength (nm)")
plt.ylabel("Power at 1 ns (a.u.)")
plt.title("Output Power at 1 ns vs Wavelength")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Step 1: Extract amplitudes at 1 ns
amplitudes_port1 = []
wavelengths_nm = []

for f0, P1, _ in results:
    amplitudes_port1.append(P1[target_index])
    wavelengths_nm.append(c / f0 * 1e9)

amplitudes_port1 = np.array(amplitudes_port1)
wavelengths_nm = np.array(wavelengths_nm)

# Step 2: Find max, min, and midpoint amplitude locations
idx_max = np.argmax(amplitudes_port1)
idx_min = np.argmin(amplitudes_port1)

# Get mid index between min and max (closest to halfway in wavelength)
lambda_mid = (wavelengths_nm[idx_max] + wavelengths_nm[idx_min]) / 2
idx_mid = np.argmin(np.abs(wavelengths_nm - lambda_mid))

# Gather selected indices and their corresponding data
selected_indices = [idx_max, idx_min, idx_mid]
selected_labels = ["Max", "Min", "Mid"]
selected_data = [results[i] for i in selected_indices]
selected_wavelengths = [wavelengths_nm[i] for i in selected_indices]

# Step 3: Plot input, port 1, and port 2 over time for each selected wavelength
plt.figure(figsize=(14, 10))
for i, (label, (f0, P1, P2), wavelength) in enumerate(zip(selected_labels, selected_data, selected_wavelengths)):
    plt.subplot(3, 1, i + 1)
    plt.plot(t * 1e9, P_in, label='Input Power', linestyle='--')
    plt.plot(t * 1e9, P1, label='Output Port 1')
    plt.plot(t * 1e9, P2, label='Output Port 2')
    plt.title(f"{label} Interference Case @ {wavelength:.2f} nm")
    plt.xlabel("Time (ns)")
    plt.ylabel("Power (a.u.)")
    plt.legend()
    plt.grid(True)

plt.tight_layout()
plt.show()
end_time = time.time()
print(f"Execution time: {end_time - start_time:.2f} seconds at dt:{dt:.3e} ps")
