import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import datetime

# ------------------------------
# 1) Simulation Parameters
# ------------------------------
symbol_count = 6        # number of time-bin symbols to simulate
samples_per_symbol = 50 # how many time samples per symbol
dt = 1.0                # time step in "arbitrary" units
time_total = symbol_count * samples_per_symbol
time = np.arange(time_total) * dt

# The delay (in samples) between early and late pulses
# We'll say "early" is at time index 0 of each symbol block,
# and "late" is at time index late_offset within that block.
late_offset = samples_per_symbol // 2  # half the symbol period as an example

# A function to make a short Gaussian pulse
def gaussian_pulse(t_index, center, width=5.0):
    return np.exp(-0.5 * ((t_index - center) / width) ** 2)

# -----------------------------------
# 2) Define which pulses go in each symbol
# -----------------------------------
# We'll create an array of "states" for each symbol. For example:
#   'E' = early only
#   'L' = late only
#   'EL' = both early & late
#   '0'  = no pulse
#
# You could randomize these or specify them manually:
states = ['EL', '0', 'E', 'L', 'EL', '0']

# Ensure we have exactly symbol_count states:
assert len(states) == symbol_count, "states list must match symbol_count"

# -----------------------------------
# 3) Construct the Input Field E_in
# -----------------------------------
E_in = np.zeros(time_total, dtype=complex)

pulse_width = 5.0  # Gaussian width

for s_idx, state in enumerate(states):
    # each symbol spans [s_idx*samples_per_symbol ... (s_idx+1)*samples_per_symbol)
    symbol_start = s_idx * samples_per_symbol
    symbol_end   = symbol_start + samples_per_symbol
    
    # Place pulses according to state
    if 'E' in state:
        # Early pulse near the start of this symbol
        center_early = symbol_start + 5  # offset from block start by 5 samples
        for t_index in range(symbol_start, symbol_end):
            E_in[t_index] += gaussian_pulse(t_index, center_early, pulse_width)
            
    if 'L' in state:
        # Late pulse near 'late_offset'
        center_late = symbol_start + late_offset
        for t_index in range(symbol_start, symbol_end):
            E_in[t_index] += gaussian_pulse(t_index, center_late, pulse_width)
            
    # if '0', do nothing (no pulses)

# -----------------------------------
# 4) First 50:50 Coupler
# -----------------------------------
# We'll use one standard convention:
#   E1 =  (E_in) / sqrt(2)
#   E2 =  i*(E_in) / sqrt(2)
E1 = E_in / np.sqrt(2)
E2 = 1j * E_in / np.sqrt(2)

# -----------------------------------
# 5) Delay the Long Arm by 1 Full Bin
# -----------------------------------
# "One bin" can be interpreted as 'samples_per_symbol', or some fraction.
delay_samples = late_offset  # or samples_per_symbol, depending on your design
# e.g., let's do a "one-time-bin" offset = late_offset
E2_delayed = np.roll(E2, delay_samples)

# Optionally include extra loss in the long arm:
alpha = 1.0  # no additional loss for simplicity
E2_delayed *= alpha

# -----------------------------------
# 6) Second 50:50 Coupler
# -----------------------------------
# Another 50:50 coupler with the formula:
#   E_out1 = (E1 + i*E2_delayed)/sqrt(2)
#   E_out2 = (E1 - i*E2_delayed)/sqrt(2)
E_out1 = (E1 + 1j * E2_delayed) / np.sqrt(2)
E_out2 = (E1 - 1j * E2_delayed) / np.sqrt(2)

# -----------------------------------
# 7) Compute Power
# -----------------------------------
P_out1 = np.abs(E_out1)**2
P_out2 = np.abs(E_out2)**2

# -----------------------------------
# 8) Plot Results
# -----------------------------------
plt.figure(figsize=(10,5))
plt.plot(time, P_out1, label="Output port 1")
plt.plot(time, P_out2, label="Output port 2")
plt.plot(time, np.abs(E_in)**2, label="Input power", linestyle='--', color='gray')
plt.xlabel("Time (arb. units)")
plt.ylabel("Power (arb. units)")
plt.title("Multi-Symbol Delay-Line Interferometer (Early/Late pulses)")
plt.legend()

filename = "DLI"
# Get the script's parent directory (the directory where the script is located)
script_dir = Path(__file__).parent

# Navigate to the parent folder (next to 'code') and then to the 'data' folder
target_dir = script_dir.parent / 'images'

# Create the directory if it doesn't exist
target_dir.mkdir(exist_ok=True)

# Generate a timestamp (e.g., '20231211_153012' for 11th December 2023 at 15:30:12)
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

# Append the timestamp to the filename
filename_with_timestamp = f"{timestamp}_{filename}"

# Define the file path
filepath = target_dir / filename_with_timestamp

# Save the plot
plt.savefig(filepath)

# Close the plot to free up memory
plt.close()
