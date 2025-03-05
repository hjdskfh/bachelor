import numpy as np
import matplotlib.pyplot as plt

# -- Simulation parameters --
num_samples = 2000     # total time samples for the array
dt = 1.0               # arbitrary time step (can be 1 "bin" in simple discrete sense)
time = np.arange(num_samples) * dt

# We'll place an "early" pulse at index ~200, and a "late" pulse at index ~200 + bin_delay
bin_delay = 50         # number of samples between "early" and "late"
pulse_center_early = 200
pulse_center_late  = pulse_center_early + bin_delay

# Make a simple Gaussian-like pulse shape
pulse_width = 5.0
def gaussian_pulse(t, t0, width=5.0):
    return np.exp(-0.5*((t - t0)/width)**2)

# -- 1) Create input pulses (time-bin states) --
E_in = np.zeros(num_samples, dtype=complex)

# For demonstration, let's place two pulses: an "early" and a "late"
E_in += gaussian_pulse(time, pulse_center_early, pulse_width)
E_in += gaussian_pulse(time, pulse_center_late,  pulse_width)

# Convert these to amplitude (already amplitude in this example).
# If you only had power, you'd do: E_in = np.sqrt(P_in).

# -- 2) First directional coupler (50:50) --
# We'll say E1 = (E_in)/sqrt(2), E2 = i*(E_in)/sqrt(2).
# This is just one valid convention for a 50:50 coupler.
E1 = E_in / np.sqrt(2)
E2 = 1j * E_in / np.sqrt(2)

# -- 3) Long vs. Short arm --
#   short arm: E1 is unchanged (no delay)
#   long arm : E2 is delayed by 'bin_delay' samples
# Optionally add loss factor alpha if needed (alpha < 1).
alpha = 1.0  # no extra loss for simplicity
E2_delayed = alpha * np.roll(E2, bin_delay)  # shift in time by bin_delay

# -- 4) Second directional coupler (50:50) to recombine --
# One typical formula for a 50:50 coupler is:
#   E_out1 = (E1 + i*E2_delayed)/sqrt(2)
#   E_out2 = (E1 - i*E2_delayed)/sqrt(2)
E_out1 = (E1 + 1j * E2_delayed) / np.sqrt(2)
E_out2 = (E1 - 1j * E2_delayed) / np.sqrt(2)

# -- 5) Compute final power in each output vs. time --
P_out1 = np.abs(E_out1)**2
P_out2 = np.abs(E_out2)**2

# Plot the final output powers
plt.figure()
plt.plot(time, P_out1, label="Output port 1")
plt.plot(time, P_out2, label="Output port 2")
plt.xlabel("Time (arb. units)")
plt.ylabel("Power (arb. units)")
plt.title("Delay-Line Interferometer with 1-Bin Offset (QKD Time-Bin)")
plt.legend()
plt.show()
