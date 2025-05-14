import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# # plt.style.use(r'C:\Users\leavi\OneDrive\Dokumente\Uni\Semester 7\NeuMoQP\Programm\code\Presentation_style_1_adjusted_no_grid.mplstyle')
# plt.style.use(r'C:\Users\leavi\bachelor\code\Presentation_style_1_adjusted_no_grid.mplstyle')

# # # Attenuation values (in dB), sorted descending
# # attenuation = [1, 3, 6, 9, 12, 15, 18, 21, 24]

# # # Corresponding SKR values (in MBit/s), sorted accordingly
# # skr_mbps = [85.97, 73.46, 69.73, 48.73, 21.22, 13.18, 7.09, 3.47, 1.70]

# # alte skr over att daten
# attenuation = [1, 3, 6, 9, 12, 17, 19, 25]
# skr_mbps = [12.71, 9.72, 7.84, 10.58, 4.99, 1.70, 0.60, np.nan]

# # Plotting
# plt.figure(figsize=(8, 5))
# plt.plot(attenuation, skr_mbps, marker='o', linestyle='-', color='blue')
# plt.xlabel('Attenuation (dB)')
# plt.ylabel('SKR (MBit/s)')
# plt.yscale('log')  # <- log scale on y-axis
# plt.grid(True)
# plt.tight_layout()
# # plt.savefig(r'C:\Users\leavi\OneDrive\Dokumente\Uni\Semester 7\NeuMoQP\Programm\code\attenuation_vs_skr_alt.png')
# plt.savefig(r'C:\Users\leavi\bachelor\code\attenuation_vs_skr_neu.png')

# import matplotlib.pyplot as plt
# import numpy as np
# import matplotlib.cm as cm

plt.style.use(r'C:\Users\leavi\bachelor\code\Presentation_style_1_adjusted_no_grid.mplstyle')


# # # Attenuation and SKR data
# attenuation = np.array([-1, -3, -6, -9, -12, -15, -18, -21, -24]) * -1 * (1/0.19)
# skr_mbps = np.array([85.97, 73.46, 69.73, 48.73, 21.22, 13.18, 7.09, 3.47, 1.70])
# mean_photon_number_signal = np.array([0.13, 0.165, 0.55, 0.7, 0.7, 0.7, 0.75, 0.75, 0.75])
# mean_photon_number_decoy = np.array([0.125, 0.16, 0.5, 0.65, 0.665, 0.665, 0.7125, 0.7125, 0.7125])

# fig, ax1 = plt.subplots(figsize=(8, 5))

# # First y-axis: SKR
# ax1.plot(attenuation, skr_mbps, marker='o', color='blue', label='SKR ($\mathrm{Mbit/s}$)', zorder = 1)
# ax1.set_xlabel('Fiber Length (km)')
# ax1.set_ylabel('SKR ($\mathrm{Mbit/s}$)', color='blue')
# ax1.set_yscale('log')
# ax1.set_ylim(1, 100)
# ax1.tick_params(axis='y', labelcolor='blue')
# ax1.grid(True)

# # Second y-axis: Mean photon number
# ax2 = ax1.twinx()
# line1, = ax2.plot(attenuation, mean_photon_number_signal, color = 'darkred', marker='s', label='$\mu_s$', zorder = 2)
# line2, = ax2.plot(attenuation, mean_photon_number_decoy, color = 'lightcoral', marker='d', label='$\mu_d$', zorder = 2)
# ax2.set_ylabel('Mean Photon Number $\mu$', color='red')
# ax2.tick_params(axis='y', labelcolor='red')
# ax2.set_ylim(0, 1)

# # Combine legends from both axes
# lines = [ax1.lines[0], line1, line2]
# labels = [l.get_label() for l in lines]
# ax1.legend(lines, labels, loc = 'lower center')

# fig.tight_layout()
# plt.savefig(r'C:\Users\leavi\bachelor\code\attenuation_vs_skr_and_photon.png')



# bachelor sKR
# Raw data
# # alte skr over att daten
attenuation = [1, 3, 6, 9, 12, 17, 19, 25]
skr_mbps = [12.71, 9.72, 7.84, 10.58, 4.99, 1.70, 0.60, np.nan]

#  Plotting
plt.figure(figsize=(8, 5))
plt.plot(attenuation, skr_mbps, marker='o', linestyle='-', color='blue')
plt.xlabel('Attenuation (dB)')
plt.ylabel('SKR (MBit/s)')
plt.yscale('log')  # <- log scale on y-axis
plt.grid(True)
plt.tight_layout()
# plt.savefig(r'C:\Users\leavi\OneDrive\Dokumente\Uni\Semester 7\NeuMoQP\Programm\code\attenuation_vs_skr_alt.png')
plt.savefig(r'C:\Users\leavi\bachelor\code\attenuation_vs_skr_log.png')
