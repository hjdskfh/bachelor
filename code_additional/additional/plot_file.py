import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

plt.style.use(r'C:\Users\leavi\OneDrive\Dokumente\Uni\Semester 7\NeuMoQP\Programm\code\Presentation_style_1_adjusted_no_grid.mplstyle')

# # Attenuation values (in dB), sorted descending
# attenuation = [1, 3, 6, 9, 12, 15, 18, 21, 24]

# # Corresponding SKR values (in MBit/s), sorted accordingly
# skr_mbps = [85.97, 73.46, 69.73, 48.73, 21.22, 13.18, 7.09, 3.47, 1.70]

# alte skr over att daten
attenuation = [1, 3, 6, 9, 12, 17, 19, 25]
skr_mbps = [12.71, 9.72, 7.84, 10.58, 4.99, 1.70, 0.60, np.nan]

# Plotting
plt.figure(figsize=(8, 5))
plt.plot(attenuation, skr_mbps, marker='o', linestyle='-', color='blue')
plt.xlabel('Attenuation (dB)')
plt.ylabel('SKR (MBit/s)')
plt.grid(True)
plt.tight_layout()
plt.savefig(r'C:\Users\leavi\OneDrive\Dokumente\Uni\Semester 7\NeuMoQP\Programm\code\attenuation_vs_skr_alt.png')

# import matplotlib.pyplot as plt
# import numpy as np
# import matplotlib.cm as cm

# # Attenuation and SKR data
# attenuation = [-1, -3, -6, -9, -12, -15, -18, -21, -24]
# skr_mbps = [85.97, 73.46, 69.73, 48.73, 21.22, 13.18, 7.09, 3.47, 1.70]

# # Normalize the index for color mapping
# colors = cm.rainbow(np.linspace(0, 1, len(attenuation)))

# # Plotting
# plt.figure(figsize=(8, 5))
# for i in range(len(attenuation)):
#     plt.plot(attenuation[i], skr_mbps[i], 'o', color=colors[i], markersize=10)

# # Connect the points with a line (optional)
# plt.plot(attenuation, skr_mbps, color='gray', linestyle='--', alpha=0.5)

# plt.xlabel('Attenuation (dB)')
# plt.ylabel('SKR (MBit/s)')
# plt.grid(True)
# plt.tight_layout()
# plt.show()