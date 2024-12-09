import matplotlib.pyplot as plt

# Example data (replace with actual data)
time = [0, 1, 2, 3, 4, 5]
voltages = {
    'Z0': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
    'Z1': [0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
    'X+': [0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
    'Z0_decoy': [0.15, 0.25, 0.35, 0.45, 0.55, 0.65],
    'Z1_decoy': [0.25, 0.35, 0.45, 0.55, 0.65, 0.75],
    'X+_decoy': [0.35, 0.45, 0.55, 0.65, 0.75, 0.85]
}
transmissions = {
    'Z0': [0.9, 0.85, 0.8, 0.75, 0.7, 0.65],
    'Z1': [0.88, 0.83, 0.78, 0.73, 0.68, 0.63],
    'X+': [0.86, 0.81, 0.76, 0.71, 0.66, 0.61],
    'Z0_decoy': [0.87, 0.82, 0.77, 0.72, 0.67, 0.62],
    'Z1_decoy': [0.85, 0.8, 0.75, 0.7, 0.65, 0.6],
    'X+_decoy': [0.84, 0.79, 0.74, 0.69, 0.64, 0.59]
}

# Create individual plots for each combination
for key in voltages.keys():
    fig, ax = plt.subplots(figsize=(8, 5))
    ax2 = ax.twinx()  # Create a second y-axis
    
    # Plot voltage (left y-axis)
    ax.plot(time, voltages[key], color='blue', label='Voltage', linestyle='-', marker='o')
    ax.set_ylabel('Voltage (V)', color='blue')
    ax.tick_params(axis='y', labelcolor='blue')

    # Plot transmission (right y-axis)
    ax2.plot(time, transmissions[key], color='red', label='Transmission', linestyle='--', marker='x')
    ax2.set_ylabel('Transmission', color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    
    # Titles and labels
    ax.set_title(f"Combination: {key}")
    ax.set_xlabel('Time')
    ax.grid(True)

    # Save or show the plot
    plt.tight_layout()
    plt.show()  # Use plt.savefig(f"{key}_plot.png") to save to file
