import matplotlib.pyplot as plt
import numpy as np
from cycler import cycler

# Define the color cycle using the viridis colormap
colors = plt.cm.viridis(np.linspace(0, 1, 7))  # Get 10 colors from viridis colormap

print(colors)

# Set the prop_cycle in the rcParams
plt.rcParams['axes.prop_cycle'] = cycler('color', colors)

# Create multiple plots and they will cycle through the colors automatically
for i in range(5):  # You can adjust this range to the number of plots you want
    x = np.linspace(0, 10, 100)
    y = np.sin(x + i)  # Just an example, modifying the sine curve
    plt.plot(x, y, label=f"Plot {i+1}")  # The color will automatically cycle

plt.legend()  # Show the legend
plt.title('Example with Viridis Color Cycle')
plt.show()
