import numpy as np
import matplotlib.pyplot as plt

# Spiral coordinates
theta = np.linspace(0, 4 * np.pi, 500)
r = np.linspace(0.0, 1.0, 500)
x_spiral = r * np.cos(theta)
y_spiral = r * np.sin(theta)

# Embed spiral in a wave
x_wave = np.linspace(-2, 2, 1000)
y_wave = 0.2 * np.sin(2 * np.pi * x_wave)  # outer wave

# Combine parts
fig, ax = plt.subplots(figsize=(6, 2))
ax.plot(x_wave, y_wave, color='gray', linewidth=2)

# Shift and scale spiral to embed
x_spiral += 0  # position horizontally
y_spiral += 0  # position vertically
ax.plot(x_spiral, y_spiral, color='gray', linewidth=2)

# Style
ax.axis('off')
ax.set_aspect('equal')
plt.tight_layout()
plt.show()
