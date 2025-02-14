import numpy as np
from scipy.fftpack import fft, ifft, fftfreq

from simulationengine import SimulationEngine
from simulationsingle import SimulationSingle


class SimulationHelper:
    def __init__(self, config):
        self.config = config 
        self.simulation_engine = SimulationEngine(config)
        self.simulation_single = SimulationSingle(config)
    
    # ========== Main Helper Functions for signal generation ==========

    def get_pulse_height(self, basis, decoy):
        """
        Determine the pulse height based on the basis and decoy state.
        Args:
            basis (int): 0 for X-basis (superposition), 1 for Z-basis (computational).
            decoy (int): 0 for standard pulse, 1 for decoy pulse.
        """
        return np.where(decoy == 0,
                        np.where(basis == 0, self.config.voltage_sup, self.config.voltage),
                        np.where(basis == 0, self.config.voltage_decoy_sup, self.config.voltage_decoy))

    def get_jitter(self, name_jitter, size_jitter=1):
        probabilities, jitter_values = self.config.data.get_probabilities(x_data=None, name='probabilities' + name_jitter)
        jitter_shifts = self.config.rng.choice(jitter_values, size = size_jitter, p=probabilities) # für jeden Querstrich kann man jitter haben
        return jitter_shifts

    def encode_pulse(self, value):
        """Return a binary pattern for a square pulse based on the given value"""
        pattern = np.zeros((len(value), self.config.n_pulses), dtype=int)
        pattern[value == 1, 0] = 1
        pattern[value == 0, self.config.n_pulses // 2] = 1
        pattern[value == -1, 0] = 1
        pattern[value == -1, self.config.n_pulses // 2] = 1
        return pattern

    def generate_square_pulse(self, pulse_height, pulse_duration, sampling_rate_fft, pattern):
        """Generate a square pulse signal for a given height and pattern."""
        # make len(t) divisible by n_pulses
        samples_per_pulse = int(pulse_duration * sampling_rate_fft)
        total_samples = self.config.n_pulses * samples_per_pulse
        t = np.linspace(0, self.config.n_pulses * pulse_duration, total_samples, endpoint=False)
        #t = np.arange(0, self.config.n_pulses * pulse_duration, inv_sampling, dtype=np.float64)
        repeating_square_pulses = np.full((len(pulse_height), len(t)), self.config.non_signal_voltage, dtype=np.float64)
        one_pulse = len(t) // self.config.n_pulses
        indices = np.arange(len(t))
        for i, pattern in enumerate(pattern):
            for j, bit in enumerate(pattern):
                if bit == 1:
                    repeating_square_pulses[i, (indices // one_pulse) == j] = pulse_height[i]
        return t, repeating_square_pulses
    
    def generate_encoded_pulse(self, pulse_height, pulse_duration, value, sampling_rate_fft):
        return self.generate_square_pulse(pulse_height, pulse_duration, sampling_rate_fft, pattern = self.encode_pulse(value))

    def apply_bandwidth_filter(self, signal, sampling_rate_fft):
        """Apply a frequency-domain filter to a signal."""
        S_fourier = fft(signal)
        frequencies = fftfreq(len(signal), d=1 / sampling_rate_fft)

        freq_x = [0, self.config.bandwidth * 0.8, self.config.bandwidth, self.config.bandwidth * 1.2, sampling_rate_fft / 2]
        freq_y = [1, 1, 0.7, 0.01, 0.001]  # Smooth drop-off

        np.multiply(S_fourier, np.interp(np.abs(frequencies), freq_x, freq_y), out=S_fourier)

        return np.real(ifft(S_fourier))

    def apply_jitter_to_pulse(self, t, signals, jitter_shifts):
        index_shift_per_symbol = ((len(t) // t[-1]) * jitter_shifts).astype(int)
        index_shift_per_symbol = index_shift_per_symbol[: self.config.n_pulses * self.config.batchsize - 1]  #size = n_pulses * batchsize - 1 (minus Anfang und Ende)
        index_one_signal = len(t) // self.config.n_pulses
        transition_indices = np.arange(index_one_signal, len(signals), index_one_signal)

        new_transition_indices = transition_indices + index_shift_per_symbol
        # Step through and shift transitions **in-place**
        for i, old_idx in enumerate(transition_indices):
            new_idx = new_transition_indices[i]  # Get the new shifted index
            
            if old_idx < new_idx:
                # Shift right: Fill original transition spot with previous value
                signals[old_idx:new_idx] = signals[old_idx - 1]
            elif old_idx > new_idx:
                # Shift left: Fill new transition spot with flipped value
                signals[new_idx:old_idx] = signals[old_idx + 1]

        return signals
    
    # ========== Data Processing Helper ==========
    def poisson_distr(self, calc_value):
        """Calculate the number of photons based on a Poisson distribution."""
        #print(f"np.any(array < 0): {np.isnan(calc_value < 0)}")
        #print(f"np.any(np.isnan(array)):{np.any(np.isnan(calc_value))}")

        upper_bounds = (calc_value + 5 * np.sqrt(calc_value)).astype(int) # shape: (n_samples,)
        max_upper_bound = upper_bounds.max()
        x = np.arange(0, max_upper_bound + 1)
        probabilities_array_poisson = np.exp(-calc_value[:, None]) * (calc_value[:, None] ** x) / factorial(x) #shape: (n_samples, max_upper_bound + 1)
        # Normalize the probabilities for every row
        probabilities_array_poisson /= probabilities_array_poisson.sum(axis=1, keepdims=True)
        # Step 1: Compute cumulative probabilities
        cumulative_probabilities = np.cumsum(probabilities_array_poisson, axis=1) #shape: (n_samples, max_upper_bound + 1)
        # Step 2: Generate random values
        random_values = self.config.rng.random(size=probabilities_array_poisson.shape[0]) # shape: (n_samples,)
        # Step 3: Find indices using searchsorted
        sampled_indices = (cumulative_probabilities.T < random_values).sum(axis=0) #cumulative_probabilities.T < random_values boolean  matrix False or True 
        # Step 4: Map indices to x values
        nr_photons = x[sampled_indices]
        return nr_photons
