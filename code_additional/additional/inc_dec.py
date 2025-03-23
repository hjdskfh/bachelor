import numpy as np
import matplotlib.pyplot as plt

# Define the range, avoiding division by zero
x = np.linspace(-0.1, 0.1, 1000)  # Small values around zero
x = x[x != 0]  # Remove zero to avoid division error

# Compute cos(1/x)
y = np.cos(1 / x)

# Plot the function
plt.figure(figsize=(8, 5))
plt.plot(x, y, label=r'$\cos(1/x)$', color='b')
plt.axhline(0, color='black', linewidth=0.5, linestyle='--')
plt.axvline(0, color='black', linewidth=0.5, linestyle='--')
plt.xlabel('x')
plt.ylabel('cos(1/x)')
plt.title('Plot of cos(1/x)')
plt.legend()
plt.grid(True)
plt.show()

