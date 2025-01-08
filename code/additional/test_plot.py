import matplotlib.pyplot as plt
import numpy as np

# Example data (replace with your actual numbers)
iterations = np.arange(1, 11)  # Iteration numbers
photons_not_filtered = [50, 45, 40, 38, 35, 30, 28, 25, 22, 20]
photons_filtered = [10, 15, 20, 22, 25, 30, 32, 35, 38, 40]

# Bar width
bar_width = 0.4

# Plot the bar graph
plt.bar(iterations - bar_width/2, photons_not_filtered, width=bar_width, label='Not Filtered', color='blue')
plt.bar(iterations + bar_width/2, photons_filtered, width=bar_width, label='Filtered', color='orange')

# Add labels and title
plt.xlabel('Iteration')
plt.ylabel('Number of Photons')
plt.title('Photons Filtered vs Not Filtered per Iteration')
plt.xticks(iterations)  # Set x-axis ticks to iterations
plt.legend()

# Show the plot
plt.show()