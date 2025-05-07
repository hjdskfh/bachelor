# import matplotlib.pyplot as plt

# plt.style.use(r'C:\Users\leavi\OneDrive\Dokumente\Uni\Semester 7\NeuMoQP\Programm\code\Presentation_style_1_adjusted_no_grid.mplstyle')

# # Original data (unsorted)
# vamp = [5.5, 4.4, 3.3, 2.2, 1.1]
# skr = [9.72, 9.89, 9.32, 9.93, 8.34]

# # Sort by Vamp (ascending)
# sorted_data = sorted(zip(vamp, skr))
# vamp_sorted, skr_sorted = zip(*sorted_data)

# # Plotting
# plt.figure(figsize=(8, 5))
# plt.plot(vamp_sorted, skr_sorted, marker='o', linestyle='-', color='blue', label='SKR (Mbit/s)')
# plt.xlabel('variance in heater voltage (mV)')
# plt.ylabel('SKR (Mbit/s)')
# plt.ylim(6,11)
# plt.title('Secure Key Rate vs. Vamp')
# plt.grid(True)
# plt.legend()
# plt.tight_layout()
# plt.savefig('SKR_vs_Vamp.png', dpi=300)

import matplotlib.pyplot as plt

plt.style.use(r'C:\Users\leavi\OneDrive\Dokumente\Uni\Semester 7\NeuMoQP\Programm\code\Presentation_style_1_adjusted_no_grid.mplstyle')

# Raw data
att = [1, 3, 6, 9]
skr_original = [12.71, 9.71, 7.84, 10.58]   # in Mbit/s
skr_modified = [13.62, 1.60, 9.46, 9.48]    # in Mbit/s

# Sort by attenuation
sorted_data = sorted(zip(att, skr_original, skr_modified))
att_sorted, skr_original_sorted, skr_modified_sorted = zip(*sorted_data)

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(att_sorted, skr_original_sorted, marker='o', label='Original (16.7 dB)', color='red')
plt.plot(att_sorted, skr_modified_sorted, marker='s', label='Modified (20.7 dB)', color='blue')
plt.xlabel('Attenuation (dB)', fontsize = 20)
plt.ylabel('SKR (Mbit/s)', fontsize = 20)
plt.ylim(0, 16)
plt.title('Secure Key Rate vs. Attenuation')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig('SKR_vs_Attenuation_ER.png', dpi=300)
