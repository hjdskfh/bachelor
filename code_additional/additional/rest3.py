'''

import itertools

# Define all 8 symbols and their (basis, value, decoy)
symbols = {
    'Z0':  (1, 1, 0),
    'Z1':  (1, 0, 0),
    'X0':  (0, 0, 0),
    'X1':  (0, 1, 0),
    'Z0*': (1, 1, 1),
    'Z1*': (1, 0, 1),
    'X0*': (0, 0, 1),
    'X1*': (0, 1, 1),
}

# List of symbol names
symbol_names = list(symbols.keys())

# Generate all possible ordered pairs (with self-pairs)
pairs = list(itertools.product(symbol_names, repeat=2))

# Prepare flattened arrays
basis_array = []
value_array = []
decoy_array = []
lookup_array = []

# Fill the arrays by "stringing" them together, and track symbol pairs
for sym1, sym2 in pairs:
    basis_1, value_1, decoy_1 = symbols[sym1]
    basis_2, value_2, decoy_2 = symbols[sym2]
    
    basis_array.extend([basis_1, basis_2])
    value_array.extend([value_1, value_2])
    decoy_array.extend([decoy_1, decoy_2])
    
    # Add to lookup
    lookup_array.append((sym1, sym2))

# OPTIONAL: Print samples
print("Basis Array (sample):", basis_array[:10])
print("Value Array (sample):", value_array[:10])
print("Decoy Array (sample):", decoy_array[:10])
print("Lookup Array in Pairs (sample):", lookup_array[50:60])'''

'''import numpy as np

def de_bruijn(k, n):
    """
    Generate a de Bruijn sequence for alphabet size k and subsequences of length n.
    """
    alphabet = list(range(k))
    a = [0] * k * n
    sequence = []

    def db(t, p):
        if t > n:
            if n % p == 0:
                sequence.extend(a[1:p + 1])
        else:
            a[t] = a[t - p]
            db(t + 1, p)
            for j in range(a[t - p] + 1, k):
                a[t] = j
                db(t + 1, t)

    db(1, 1)
    return sequence

# Generate de Bruijn sequence for 8 symbols (order 2)
seq = de_bruijn(8, 2)

# Map numbers to your symbols
symbols = ['Z0', 'Z1', 'X0', 'X1', 'Z0*', 'Z1*', 'X0*', 'X1*']
symbol_sequence = [symbols[i] for i in seq]

print("Symbol sequence:", symbol_sequence)
print("Length of sequence:", len(symbol_sequence))  # Should be 64

# Assuming symbols dictionary as before
symbols_dict = {
    'Z0':  (1, 1, 0),
    'Z1':  (1, 0, 0),
    'X0':  (0, 0, 0),
    'X1':  (0, 1, 0),
    'Z0*': (1, 1, 1),
    'Z1*': (1, 0, 1),
    'X0*': (0, 0, 1),
    'X1*': (0, 1, 1),
}

basis_array = np.empty(len(symbols_dict)**2 + 1, dtype=int)
value_array = np.empty(len(symbols_dict)**2 + 1, dtype=int)
decoy_array = np.empty(len(symbols_dict)**2 + 1, dtype=int)


# Flatten basis, value, decoy
for idx, sym in enumerate(symbol_sequence):
    b, v, d = symbols_dict[sym]
    basis_array[idx] = b
    value_array[idx] = v
    decoy_array[idx] = d

basis_array[-1] = basis_array[0]
value_array[-1] = value_array[0]
decoy_array[-1] = decoy_array[0]

print("Basis Array (sample):", basis_array[:10])
print("Value Array (sample):", value_array[:10])
print("Decoy Array (sample):", decoy_array[:10])
'''

import numpy as np

# Example of real photon times (replace this with actual data!)
photon_times = np.array([100, 105, 109, 135, 145, 155, 160, 300, 310, 320])

# Example parameters
num_symbols = 65
bins_per_symbol = 30
T_max = 6500  # Total experimental time in ns (adjust as needed)

symbol_window = T_max / num_symbols
bin_width = symbol_window / bins_per_symbol

# Create histogram matrix
histogram_matrix = np.zeros((num_symbols, bins_per_symbol), dtype=int)

# Build histogram
for symbol_idx in range(num_symbols):
    symbol_start = symbol_idx * symbol_window
    symbol_end = symbol_start + symbol_window

    # Select photon times within symbol window
    mask = (photon_times >= symbol_start) & (photon_times < symbol_end)
    photon_times_in_symbol = photon_times[mask]

    # Define bins inside symbol window
    bins = np.linspace(symbol_start, symbol_end, bins_per_symbol + 1)

    # Histogram
    counts, _ = np.histogram(photon_times_in_symbol, bins=bins)

    # Store
    histogram_matrix[symbol_idx, :] = counts

print("Histogram matrix shape:", histogram_matrix.shape)
