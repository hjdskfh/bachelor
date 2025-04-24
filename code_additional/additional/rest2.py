import numpy as np
import matplotlib.pyplot as plt

# Generate sample data: a Gaussian peak
x = np.linspace(-5, 5, 500)
y = np.exp(-x**2)

# Our FWHM function (copy-paste from before)
def calculate_fwhm(x, y):
    x = np.array(x)
    y = np.array(y)
    
    half_max = np.max(y) / 2.0
    peak_index = np.argmax(y)

    for i in range(peak_index - 1, -1, -1):
        if y[i] < half_max:
            x1, y1 = x[i], y[i]
            x2, y2 = x[i + 1], y[i + 1]
            x_left = x1 + (half_max - y1) * (x2 - x1) / (y2 - y1)
            left_indices = (i, i + 1)
            break
    else:
        x_left = x[0]
        left_indices = (0, 1)

    for i in range(peak_index + 1, len(y)):
        if y[i] < half_max:
            x1, y1 = x[i - 1], y[i - 1]
            x2, y2 = x[i], y[i]
            x_right = x1 + (half_max - y1) * (x2 - x1) / (y2 - y1)
            right_indices = (i - 1, i)
            break
    else:
        x_right = x[-1]
        right_indices = (len(x) - 2, len(x) - 1)

    fwhm = x_right - x_left
    return fwhm, x_left, x_right, left_indices, right_indices

# Call the function
fwhm, x_left, x_right, idx_left, idx_right = calculate_fwhm(x, y)

# Plotting
plt.plot(x, y, label='Function')
plt.axhline(np.max(y)/2, color='gray', linestyle='--', label='Half Maximum')
plt.axvline(x_left, color='red', linestyle='--', label='FWHM Points')
plt.axvline(x_right, color='red', linestyle='--')
plt.title(f'FWHM = {fwhm:.3f}')
plt.legend()
plt.grid(True)
plt.show()

