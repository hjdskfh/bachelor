import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from pathlib import Path
import math

# ================================================
# FUNCTIONS
# ================================================

def save_plot(filename, dpi=600):
  """Saves the current Matplotlib plot to a file in the 'img' directory."""
  script_dir = Path(__file__).parent
  img_dir = script_dir / 'img'
  img_dir.mkdir(exist_ok=True)
  filepath = img_dir / filename
  plt.savefig(filepath, dpi=dpi)
  plt.close()

def simulate_pulsed_source(n_pulses=6, jitter_source=100e-12, 
                          repetition_rate=1e9, time_resolution_factor=4, factor_pulse_ampltiude=1):
  """
  Simulates a pulsed source with Gaussian jitter and calculates the average 
  number of photons expected in each time bin.

  Args:
    n_pulses: Number of pulses.
    jitter_source: Standard deviation of the Gaussian jitter (in seconds).
    repetition_rate: Repetition rate of the source (in Hz).
    time_resolution_factor: Factor to increase time resolution.

  Returns:
    tuple: A tuple containing the time bin centers and the average number of 
           photons per time bin.
  """

  bin_width = 1 / (time_resolution_factor * repetition_rate)
  bin_centers = np.arange((n_pulses + 1) * time_resolution_factor) * bin_width
  gaussian_centers = np.arange(1, n_pulses + 1) / repetition_rate
  
  average_number_photons_per_time_bin = np.zeros_like(bin_centers)
  for center in gaussian_centers:
    average_number_photons_per_time_bin += (
        factor_pulse_ampltiude / repetition_rate * 
        stats.norm.pdf(bin_centers, loc=center, scale=jitter_source)
    )

  return bin_centers, average_number_photons_per_time_bin

def generate_alice_choices(n_pulses, symbol_length=2, p_z_alice=0.5, 
                           p_z_1=0.5, p_decoy=0.1, dB_on=20, dB_off=10, 
                           dB_decoy=15, dB_channel_attenuation=5):
    """
    Generates Alice's choices for a quantum communication protocol, including 
    basis selection, value encoding, decoy states, and attenuation patterns.

    Args:
        n_pulses: Number of pulses to generate.
        symbol_length: Length of each symbol in pulses.
        p_z_alice: Probability of Alice choosing the Z basis.
        p_z_1: Probability of encoding a '1' in the Z basis.
        p_decoy: Probability of sending a decoy state.
        dB_on: Attenuation in dB for '1' in the Z basis.
        dB_off: Attenuation in dB for '0' in the Z basis.
        dB_decoy: Attenuation in dB for decoy states.
        dB_channel_attenuation: Channel attenuation in dB.

    Returns:
        tuple: A tuple containing the basis choices, encoded values, decoy flags, 
            attenuation pattern, modulation multiplier, and channel multiplier.
    """

    # Basis and value choices
    basis = np.random.choice([0, 1], size=(n_pulses // symbol_length), 
                            p=[1-p_z_alice, p_z_alice])
    value = np.random.choice([0, 1], size=(n_pulses // symbol_length), 
                            p=[1-p_z_1, p_z_1])
    value[basis == 0] = -1  # Mark X basis values

    # Decoy state selection
    decoy = np.random.choice([0, 1], size=(n_pulses // symbol_length), 
                            p=[1-p_decoy, p_decoy])

    # Attenuation pattern
    pattern_01 = np.concatenate([np.zeros(symbol_length // 2), np.ones(symbol_length // 2)])
    pattern_10 = np.concatenate([np.ones(symbol_length // 2), np.zeros(symbol_length // 2)])
    pattern_attenuator = np.zeros(n_pulses, dtype=int)
    for i, v in enumerate(value):
        start = i * symbol_length
        end = start + symbol_length
        if v == 0:
            pattern_attenuator[start:end] = pattern_10
        elif v == 1:
            pattern_attenuator[start:end] = pattern_01

    # Attenuation pattern in dB
    attenuation = np.zeros(n_pulses)
    attenuation[pattern_attenuator == 0] = dB_off
    attenuation[pattern_attenuator == 1] = dB_on
    attenuation[(pattern_attenuator == 1) & (np.repeat(decoy, symbol_length) == 1)] = dB_decoy
    attenuation[(np.repeat(value, symbol_length) == -1) & (np.repeat(decoy, symbol_length) == 1)] = dB_decoy

    # Calculate multipliers
    multiplier_modulation = np.power(10, -1 * attenuation / 10)
    multiplier_channel = np.power(10, -1 * dB_channel_attenuation / 10)

    return (basis, value, decoy, pattern_attenuator, 
            multiplier_modulation, multiplier_channel)

def generate_alice_choices_fixed(n_pulses, symbol_length=2, p_z_alice=0.5, 
                           p_z_1=0.5, p_decoy=0.1, dB_on=20, dB_off=10, 
                           dB_decoy=15, dB_channel_attenuation=5):
    """
    Generates Alice's choices for a quantum communication protocol, including 
    basis selection, value encoding, decoy states, and attenuation patterns.

    Args:
        n_pulses: Number of pulses to generate.
        symbol_length: Length of each symbol in pulses.
        p_z_alice: Probability of Alice choosing the Z basis.
        p_z_1: Probability of encoding a '1' in the Z basis.
        p_decoy: Probability of sending a decoy state.
        dB_on: Attenuation in dB for '1' in the Z basis.
        dB_off: Attenuation in dB for '0' in the Z basis.
        dB_decoy: Attenuation in dB for decoy states.
        dB_channel_attenuation: Channel attenuation in dB.

    Returns:
        tuple: A tuple containing the basis choices, encoded values, decoy flags, 
            attenuation pattern, modulation multiplier, and channel multiplier.
    """

    # SET Basis and value choices
    basis = np.zeros(n_pulses // symbol_length)
    basis[: int(3/4 * n_pulses // symbol_length)] = 1
    value = np.zeros(n_pulses // symbol_length)
    value[: int(1/4 * n_pulses // symbol_length)] = 1
    value[basis == 0] = -1

    # Decoy state selection
    decoy = np.zeros(n_pulses // symbol_length)
    decoy[int(2/4 * n_pulses // symbol_length):int(3/4 * n_pulses // symbol_length)] = 1 

    # Attenuation pattern
    pattern_01 = np.concatenate([np.zeros(symbol_length // 2), np.ones(symbol_length // 2)])
    pattern_10 = np.concatenate([np.ones(symbol_length // 2), np.zeros(symbol_length // 2)])
    pattern_attenuator = np.zeros(n_pulses, dtype=int)
    for i, v in enumerate(value):
        start = i * symbol_length
        end = start + symbol_length
        if v == 0:
            pattern_attenuator[start:end] = pattern_10
        elif v == 1:
            pattern_attenuator[start:end] = pattern_01

    # Attenuation pattern in dB
    attenuation = np.zeros(n_pulses)
    attenuation[pattern_attenuator == 0] = dB_off
    attenuation[pattern_attenuator == 1] = dB_on
    attenuation[(pattern_attenuator == 0) & (np.repeat(decoy, symbol_length) == 1)] = dB_decoy
    attenuation[(np.repeat(value, symbol_length) == -1) & (np.repeat(decoy, symbol_length) == 1)] = dB_decoy

    # Calculate multipliers
    multiplier_modulation = np.power(10, -1 * attenuation / 10)
    multiplier_channel = np.power(10, -1 * dB_channel_attenuation / 10)

    return (basis, value, decoy, pattern_attenuator, 
            multiplier_modulation, multiplier_channel)

def simulate_bob_detection(average_number_photons_per_time_bin, n_pulses, 
                           time_resolution_factor, symbol_length=2, 
                           p_z_bob=0.5, p_stray=0, p_short_path_DLI=0.5):
  """
  Simulates Bob's detection events, including basis choice, stray light, and 
  potential delayed pulses from a DLI (Delayed Light Injection) attack.

  Args:
    average_number_photons_per_time_bin: Average number of photons in each time bin.
    n_pulses: Number of pulses.
    time_resolution_factor: Factor to increase time resolution.
    symbol_length: Length of each symbol in pulses.
    p_z_bob: Probability of Bob choosing the Z basis.
    p_stray: Average number of stray photons per time bin.
    p_short_path_DLI: Probability of a photon taking the short path in a DLI attack.

  Returns:
    tuple:  A tuple containing the photon count histograms for the Z and X bases.
  """

  # Calculate average photons in Z and X bases
  average_photons_z = (
      average_number_photons_per_time_bin * p_z_bob + p_stray
  )
  average_photons_x = (
      average_number_photons_per_time_bin * (1 - p_z_bob) + p_stray
  )

  # Simulate delayed pulses for DLI attack
  delayed_pulses_x = np.zeros_like(average_photons_x)
  delay_index = symbol_length * time_resolution_factor // 2
  delayed_pulses_x[delay_index:] = average_photons_x[:-delay_index].copy()

  # Combine direct and delayed pulses
  average_photons_x = (
      p_short_path_DLI * average_photons_x + 
      (1 - p_short_path_DLI) * delayed_pulses_x
  )

  # Generate photon count histograms
  histogram_z = np.random.poisson(lam=average_photons_z)
  histogram_x = np.random.poisson(lam=average_photons_x)

  return average_photons_z, average_photons_x, histogram_z, histogram_x

def reduce_histogram_bins(histogram, bin_reduction_factor):
  """
  Reduces the number of bins in a histogram by summing up groups of bins.

  Args:
    histogram: The original histogram as a NumPy array.
    bin_reduction_factor: The factor by which to reduce the number of bins. 
                           (e.g., 4 to sum groups of 4 bins)

  Returns:
    numpy.array: The reduced histogram.
  """

  reshaped_histogram = histogram.reshape(-1, bin_reduction_factor)
  reduced_histogram = np.sum(reshaped_histogram, axis=1)
  return reduced_histogram

def repeat_array(histogram, time_resolution_factor):
  """
  Reduces the number of bins in a histogram by summing up groups of bins.

  Args:
    histogram: The original histogram as a NumPy array.
    bin_reduction_factor: The factor by which to reduce the number of bins. 
                           (e.g., 4 to sum groups of 4 bins)

  Returns:
    numpy.array: The reduced histogram.
  """
  histogram = np.repeat(histogram, time_resolution_factor)
  front = np.repeat(histogram[0], time_resolution_factor // 2)
  tail = np.repeat(histogram[-1], time_resolution_factor // 2)
  histogram = np.concatenate((front, histogram, tail))
  return histogram

def decode_photon_counts(x_basis_sequence, z_basis_sequence): # Note: wrong and obsolete
    x_early_counts = x_basis_sequence[::2]  
    x_late_counts = x_basis_sequence[1::2]  
    
    z_early_counts = z_basis_sequence[::2] 
    z_late_counts = z_basis_sequence[1::2]

    basis_array = []
    value_array = []

    for i in range(len(x_early_counts)):
        basis = None
        value = None

        if z_early_counts[i] > 0 and z_late_counts[i] == 0:
            basis, value = 1, 1 
        elif z_late_counts[i] > 0 and z_early_counts[i] == 0:
            basis, value = 1, 0  
        elif x_late_counts[i] > 1:
            basis, value = 0, -1 
        else:
            basis, value = -3, -3
        
        basis_array.append(basis)
        value_array.append(value)
        
    return np.array(basis_array), np.array(value_array)

def calculate_gain_and_error(symbol_length, alice_basis, alice_values, alice_decoy, detection_hist_z, detection_hist_x): # Note: wrong and obsolete
    # intensity / decoy dependent

    # compute gain
    epsilon_temp = 1e-10
    detections_symbol_z = np.sum(detection_hist_z.reshape(-1, symbol_length), axis=1)
    detections_symbol_x = np.sum(detection_hist_x.reshape(-1, symbol_length), axis=1)

    alice_signal = np.array([1 if x == 0 else 0 for x in alice_decoy])
    gain_z_signal = detections_symbol_z[(alice_basis & alice_signal) == 1].sum()/(np.count_nonzero((alice_basis & alice_signal) == 1) + epsilon_temp)
    gain_z_decoy = detections_symbol_z[(alice_basis & alice_decoy) == 1].sum()/(np.count_nonzero((alice_basis & alice_decoy) == 1) + epsilon_temp)
    alice_x_basis = np.array([1 if x == 0 else 0 for x in alice_basis])
    gain_x_signal = detections_symbol_x[(alice_x_basis & alice_signal) == 1].sum()/(np.count_nonzero((alice_x_basis & alice_signal) == 1) + epsilon_temp)
    gain_z_decoy = detections_symbol_z[(alice_x_basis & alice_decoy) == 1].sum()/(np.count_nonzero((alice_x_basis & alice_decoy) == 1) + epsilon_temp)

    gain_Z = np.array([gain_z_signal, gain_z_decoy])
    gain_X = np.array([gain_x_signal, gain_z_decoy])

    # compute error
    bob_basis, bob_values = decode_photon_counts(detection_hist_z, detection_hist_x)

    matching_indices = np.where((alice_basis == 1) & (bob_basis == 1) & (alice_signal == 1))
    errors_z_signal = np.sum(alice_values[matching_indices] != bob_values[matching_indices])
    matching_indices = np.where((alice_basis == 1) & (bob_basis == 1) & (alice_signal == 0))
    errors_z_decoy = np.sum(alice_values[matching_indices] != bob_values[matching_indices])
    
    # !!!!!!!!!!!!!!!!!! errors dont make sense (will be zero in X Basis) !!!!!!!!!!!!!!!!!!
    matching_indices = np.where((alice_basis == 0) & (bob_basis == 0) & (alice_signal == 1))
    errors_x_signal = np.sum(alice_values[matching_indices] != bob_values[matching_indices])
    matching_indices = np.where((alice_basis == 0) & (bob_basis == 0) & (alice_signal == 0))
    errors_x_decoy = np.sum(alice_values[matching_indices] != bob_values[matching_indices])

    num_error_Z = np.array([errors_z_signal, errors_z_decoy])
    num_error_X = np.array([errors_x_signal, errors_x_decoy])

    return gain_Z, gain_X, num_error_Z, num_error_X

# -----------------------------------------------------------
# see: Security proof for a simplified BB84-like QKD protocol

def calculate_skr(pz, mu, p_mu, epsilon, alice_basis, alice_values, alice_decoy, detection_hist_z, detection_hist_x, repetition_rate, n_pulses, symbol_length, lambdaEC):
    total_number_of_symbols = n_pulses // symbol_length

    # Lower Bound Vacuum Events
    x_early_counts = detection_hist_x[::2]  
    n_early_01_signal = 0
    n_early_01_decoy = 0
    for i in range(len(alice_values) - 1):
        if alice_values[i] == -1 and alice_values[i + 1] == 1:
            if alice_decoy[i] == 0 and alice_decoy[i + 1] == 0:
                n_early_01_signal += x_early_counts[i + 1]
            if alice_decoy[i] == 1 and alice_decoy[i + 1] == 1:
                n_early_01_decoy += x_early_counts[i + 1]
    n_early_01 = np.array([n_early_01_signal, n_early_01_decoy])
    vacuum_events_lower_bound = d0_lower(n_early_01, mu, p_mu, epsilon)
    print("n_early_01: " + str(n_early_01))
    print("vacuum_events_lower_bound: " + str(vacuum_events_lower_bound))

    # Single Photon Events in Z-Basis
    single_photon_events_z_basis = d1_lower_z_basis(pz, mu, p_mu, epsilon, alice_basis, alice_values, alice_decoy, detection_hist_z, detection_hist_x)
    print("single_photon_events_z_basis: " + str(single_photon_events_z_basis))

    # Phase error rate in Z-Basis
    phase_error_rate_z_basis = d1_upper_phase_errors_z_basis(pz, mu, p_mu, epsilon, alice_basis, alice_values, alice_decoy, detection_hist_z, detection_hist_x)
    print("phase_error_rate_z_basis: " + str(phase_error_rate_z_basis))

    # Assume epsilon_sec = epsilon_cor
    secret_key_length = vacuum_events_lower_bound + single_photon_events_z_basis * (1 - h(phase_error_rate_z_basis)) - lambdaEC - 6 * np.log2(19 / epsilon) - np.log2(2 / epsilon)

    return secret_key_length / total_number_of_symbols * repetition_rate

def d0_lower(count, mu, p_mu, epsilon):
    tau_0 = tau_n(0, mu, p_mu)
    mu_difference = mu[0] - mu[1]
    n_mu2_m = n_pm_k(1, mu, p_mu, count, epsilon)[1]
    n_mu1_p = n_pm_k(0, mu, p_mu, count, epsilon)[0]
    print("tau_0: " + str(tau_0))
    print("tau_1: " + str(tau_n(1, mu, p_mu)))
    return tau_0 / mu_difference * (mu[0] * n_mu2_m - mu[1] * n_mu1_p)

def d1_upper_phase_errors_z_basis(pz, mu, p_mu, epsilon, alice_basis, alice_values, alice_decoy, detection_hist_z, detection_hist_x):
    phase_errors_x_basis =  d1_upper_error_x_basis(pz, mu, p_mu, epsilon, alice_basis, alice_values, alice_decoy, detection_hist_z, detection_hist_x)
    print("phase_errors_x_basis: " + str(phase_errors_x_basis))
    single_photon_events_Z_basis = d1_lower_z_basis(pz, mu, p_mu, epsilon, alice_basis, alice_values, alice_decoy, detection_hist_z, detection_hist_x)
    # measure: early, send: 01 - Note: Corresponds To Vaccum States
    x_early_counts = detection_hist_x[::2]  
    n_early_01 = 0
    for i in range(len(alice_values) - 1):
        if alice_values[i] == -1 and alice_values[i + 1] == 1:
            n_early_01 += x_early_counts[i + 1]
    detections_after_0_1_sequence = n_early_01
    # measure: early, send: 00 or 11
    n_early_00_or_11_signal = 0
    n_early_00_or_11_decoy = 0
    for i in range(len(alice_values) - 1):
        if (alice_values[i] == 1 and alice_values[i + 1] == 1) or (alice_values[i] == 0 and alice_values[i + 1] == 0):
            if alice_decoy[i] == 0 and alice_decoy[i + 1] == 0:
                #n_early_00_or_11_signal += x_early_counts[i]
                n_early_00_or_11_signal += x_early_counts[i + 1]
            if alice_decoy[i] == 1 and alice_decoy[i + 1] == 1:
                #n_early_00_or_11_decoy += x_early_counts[i]
                n_early_00_or_11_decoy += x_early_counts[i + 1]
    n_early_00_or_11 = np.array([n_early_00_or_11_signal, n_early_00_or_11_decoy])
    print("n_early_00_or_11: " + str(n_early_00_or_11))
    d1_lower_early_ZZ = d1_lower(n_early_00_or_11, mu, p_mu, epsilon, detections_after_0_1_sequence)
    print("d1_lower_early_ZZ: " + str(d1_lower_early_ZZ))
    gamma = gamma_function(epsilon, phase_errors_x_basis, single_photon_events_Z_basis, d1_lower_early_ZZ)
    return phase_errors_x_basis + gamma

def d1_lower_z_basis(pz, mu, p_mu, epsilon, alice_basis, alice_values, alice_decoy, detection_hist_z, detection_hist_x):
    # count detections in Z basis
    number_detection_per_symbol_z_basis = np.sum(detection_hist_z.reshape(-1, symbol_length), axis=1)
    n_Z_signal = np.sum(number_detection_per_symbol_z_basis[np.where(alice_decoy == 0)])
    n_Z_decoy = np.sum(number_detection_per_symbol_z_basis[np.where(alice_decoy == 1)])
    n_Z = np.array([n_Z_signal, n_Z_decoy])
    # measure: early, send: 01 - Note: Corresponds To Vaccum States
    x_early_counts = detection_hist_x[::2]  
    n_early_01 = 0
    for i in range(len(alice_values) - 1):
        if alice_values[i] == -1 and alice_values[i + 1] == 1:
            n_early_01 += x_early_counts[i + 1]
    detections_after_0_1_sequence = n_early_01
    single_photon_events_Z_basis_lower_bound = d1_lower(n_Z, mu, p_mu, epsilon, detections_after_0_1_sequence)
    return single_photon_events_Z_basis_lower_bound

def d1_upper_error_x_basis(pz, mu, p_mu, epsilon, alice_basis, alice_values, alice_decoy, detection_hist_z, detection_hist_x):
    # counts
    x_early_counts = detection_hist_x[::2]  
    x_late_counts = detection_hist_x[1::2]  
    # measure: late, send: +
    n_late_plus_signal = np.sum(x_late_counts[np.where((alice_values == -1) & (alice_decoy == 0))])
    n_late_plus_decoy = np.sum(x_late_counts[np.where((alice_values == -1) & (alice_decoy == 1))])
    n_late_plus = np.array([n_late_plus_signal, n_late_plus_decoy])
    print("n_late_plus: " + str(n_late_plus))
    # measure: early, send: 00 or 11
    n_early_00_or_11_signal = 0
    n_early_00_or_11_decoy = 0
    for i in range(len(alice_values) - 1):
        if (alice_values[i] == 1 and alice_values[i + 1] == 1) or (alice_values[i] == 0 and alice_values[i + 1] == 0):
            if alice_decoy[i] == 0 and alice_decoy[i + 1] == 0:
                #n_early_00_or_11_signal += x_early_counts[i]
                n_early_00_or_11_signal += x_early_counts[i + 1]
            if alice_decoy[i] == 1 and alice_decoy[i + 1] == 1:
                #n_early_00_or_11_decoy += x_early_counts[i]
                n_early_00_or_11_decoy += x_early_counts[i + 1]
    n_early_00_or_11 = np.array([n_early_00_or_11_signal, n_early_00_or_11_decoy])
    print("n_early_00_or_11: " + str(n_early_00_or_11))
    # measure: late, send: 0
    n_late_0_signal = np.sum(x_late_counts[np.where((alice_values == 0) & (alice_decoy == 0))])
    n_late_0_decoy = np.sum(x_late_counts[np.where((alice_values == 0) & (alice_decoy == 1))])
    n_late_0 = np.array([n_late_0_signal, n_late_0_decoy])
    print("n_late_0: " + str(n_late_0))
    # measure: late, send: 1
    n_late_1_signal = np.sum(x_late_counts[np.where((alice_values == 1) & (alice_decoy == 0))])
    n_late_1_decoy = np.sum(x_late_counts[np.where((alice_values == 1) & (alice_decoy == 1))])
    n_late_1 = np.array([n_late_1_signal, n_late_1_decoy])
    print("n_late_1: " + str(n_late_1))
    # measure: early, send: 0+
    n_early_0plus_signal = 0
    n_early_0plus_decoy = 0
    for i in range(len(alice_values) - 1):
        if alice_values[i] == 0 and alice_values[i + 1] == -1:
            if alice_decoy[i] == 0 and alice_decoy[i + 1] == 0:
                #n_early_0plus_signal += x_early_counts[i]
                n_early_0plus_signal += x_early_counts[i + 1]
            if alice_decoy[i] == 1 and alice_decoy[i + 1] == 1:
                #n_early_0plus_decoy += x_early_counts[i]
                n_early_0plus_decoy += x_early_counts[i + 1]
    n_early_0plus = np.array([n_early_0plus_signal, n_early_0plus_decoy])
    print("n_early_0plus: " + str(n_early_0plus))
    # measure: early, send: +1
    n_early_plus1_signal = 0
    n_early_plus1_decoy = 0
    for i in range(len(alice_values) - 1):
        if alice_values[i] == -1 and alice_values[i + 1] == 1:
            if alice_decoy[i] == 0 and alice_decoy[i + 1] == 0:
                #n_early_plus1_signal += x_early_counts[i]
                n_early_plus1_signal += x_early_counts[i + 1]
            if alice_decoy[i] == 1 and alice_decoy[i + 1] == 1:
                #n_early_plus1_decoy += x_early_counts[i]
                n_early_plus1_decoy += x_early_counts[i + 1]
    n_early_plus1 = np.array([n_early_plus1_signal, n_early_plus1_decoy])
    print("n_early_plus1: " + str(n_early_plus1))
    # measure: early, send: 01 - Note: Corresponds To Vaccum States
    n_early_01 = 0
    for i in range(len(alice_values) - 1):
        if alice_values[i] == -1 and alice_values[i + 1] == 1:
            n_early_01 += x_early_counts[i + 1]
    detections_after_0_1_sequence = n_early_01
    print("detections_after_0_1_sequence: " + str(detections_after_0_1_sequence))

    alpha, beta = calculate_alpha_and_beta(pz)

    # calculate d1(ex)
    term1 = alpha / 2 * d1_upper(n_late_plus, mu, p_mu, epsilon) / d1_lower(n_early_00_or_11, mu, p_mu, epsilon, detections_after_0_1_sequence)
    term2 = beta * d1_lower(n_late_0 + n_late_1, mu, p_mu, epsilon, detections_after_0_1_sequence) / d1_lower(n_early_00_or_11, mu, p_mu, epsilon, detections_after_0_1_sequence)
    term3 = alpha * d1_lower(n_early_0plus + n_early_plus1, mu, p_mu, epsilon, detections_after_0_1_sequence) / d1_lower(n_early_00_or_11, mu, p_mu, epsilon, detections_after_0_1_sequence)
    max = 0
    compare = 1 + term1 - term2 - term3
    if compare > max:
        max = compare
    print("d1_upper(n_late_plus, mu, p_mu, epsilon)", str(d1_upper(n_late_plus, mu, p_mu, epsilon)))
    print("d1_lower(n_early_00_or_11, mu, p_mu, epsilon, detections_after_0_1_sequence)", str(d1_lower(n_early_00_or_11, mu, p_mu, epsilon, detections_after_0_1_sequence)))
    print("term1: " + str(term1))
    print("term2: " + str(term2))
    print("term3: " + str(term3))
    print("max: " + str(max))
    return term1 + max

def calculate_alpha_and_beta(pz):
    return pz ** 2 / (4 * (1 - pz)), pz/4

def d1_upper(count, mu, p_mu, epsilon):
    tau_1 = tau_n(1, mu, p_mu)
    mu_difference = mu[0] - mu[1]
    n_mu1_p = n_pm_k(0, mu, p_mu, count, epsilon)[0]
    n_mu2_m = n_pm_k(1, mu, p_mu, count, epsilon)[1]
    return tau_1 / mu_difference * (n_mu1_p - n_mu2_m)

def d1_lower(count, mu, p_mu, epsilon, detections_after_0_1_sequence):
    tau_0 = tau_n(0, mu, p_mu)
    tau_1 = tau_n(1, mu, p_mu)
    n_mu2_m = n_pm_k(1, mu, p_mu, count, epsilon)[1]
    n_mu1_p = n_pm_k(0, mu, p_mu, count, epsilon)[0]

    a = tau_1 * mu[0] / (mu[1] * (mu[0] - mu[1]))
    b = n_mu2_m - np.power(mu[1], 2) / np.power(mu[0], 2) * n_mu1_p
    c = (np.power(mu[0], 2) - np.power(mu[1], 2)) / np.power(mu[0], 2) * d0_upper(detections_after_0_1_sequence, epsilon) / tau_0

    return a * (b + c)

def tau_n(n, mu, p_mu):
    tau = 0
    for i, k in enumerate(mu):
        tau += np.exp(-k) * k ** n * p_mu[i] / math.factorial(n)
    return tau

def n_pm_k(index, mu, p_mu, count, epsilon):
    n_p = np.exp(mu[index]) / p_mu[index] * (count[index] + np.sqrt(count.sum()/2 * np.log(1/epsilon)))
    n_m = np.exp(mu[index]) / p_mu[index] * (count[index] - np.sqrt(count.sum()/2 * np.log(1/epsilon)))
    return n_p, n_m 

def d0_upper(detections_after_0_1_sequence, epsilon):
    return detections_after_0_1_sequence + delta_vac(detections_after_0_1_sequence, epsilon)

def delta_vac(n, epsilon):
    return np.sqrt((n * np.log(1/epsilon)) / 2)

def gamma_function(a,b,c,d):
    a = (c + d) * (1 - b) * b / (c * d * np.log(2))
    b = np.log2((c + d) * 21 **2 / (c * d * (1 - b) * b * np.power(a, 2)))
    return np.sqrt(a * b)

def h(x: float) -> float:
    if x == 0 or x == 1:
        return 0
    return -x * np.log2(x) - (1 - x) * np.log2(1 - x)

# -----------------------------------------------------------


def plotting(bin_centers, average_number_photons_per_time_bin_source, basis_alice, value_alice, decoy_alice, mult_mod_repeated, average_number_photons_per_time_bin_after_eam, average_photons_z, average_photons_x, reduced_histogram_z_repeated, reduced_histogram_x_repeated):
    plt.plot(bin_centers, average_number_photons_per_time_bin_source) 
    plt.xlabel("Time (s)")
    plt.ylabel("Average number of photons expected in the time bin")
    plt.title("Pulsed Laser Source")
    save_plot("pulsed_laser_source.png")

    fig, ax = plt.subplots()
    state_labels = []
    for basis, value, decoy in zip(basis_alice, value_alice, decoy_alice):
        if basis == 1 and value == 0 and decoy == 1:
            state_labels.append('Z0_decoy')
        elif basis == 1 and value == 1 and decoy == 1:
            state_labels.append('Z1_decoy')
        elif basis == 0 and value == -1 and decoy == 1:
            state_labels.append('X1_decoy')
        elif basis == 1 and value == 0 and decoy == 0:
            state_labels.append('Z0')
        elif basis == 1 and value == 1 and decoy == 0:
            state_labels.append('Z1')
        elif basis == 0 and value == -1 and decoy == 0:
            state_labels.append('X1')
        else:
            state_labels.append('Unknown')

    color_map = {
        'Z0_decoy': 'darkblue', 
        'Z1_decoy': 'darkcyan', 
        'X1_decoy': 'darkred', 
        'Z0': 'blue', 
        'Z1': 'lightblue', 
        'X1': 'red', 
        'Unknown': 'gray'
    }
    colors = [color_map[label] for label in state_labels]

    ax.bar(range(len(basis_alice)), [1] * len(basis_alice), color=colors, tick_label=state_labels)
    ax.set_xticks(range(len(basis_alice)))
    ax.set_xticklabels(state_labels, rotation=45, ha='right')
    plt.title("Alice Value Choices")
    plt.tight_layout()
    save_plot("alice_values.png")

    plt.plot(bin_centers, mult_mod_repeated)
    plt.xlabel("Time (s)")
    plt.ylabel("Modulator Attenuation")
    plt.title("EAM")
    save_plot("eam.png")

    plt.plot(bin_centers, average_number_photons_per_time_bin_after_eam) 
    plt.xlabel("Time (s)")
    plt.ylabel("Average number of photons expected in the time bin")
    plt.title("After EAM")
    save_plot("after_eam.png")

    plt.plot(bin_centers, average_photons_z)
    plt.xlabel("Time (s)")
    plt.ylabel("Average photons Z")
    plt.title("Z-basis")
    save_plot("average_photons_z.png")

    plt.plot(bin_centers, average_photons_x)
    plt.xlabel("Time (s)")
    plt.ylabel("Average photons X")
    plt.title("X-basis")
    save_plot("average_photons_x.png")

    plt.plot(bin_centers, reduced_histogram_z_repeated)
    plt.xlabel("Time (s)")
    plt.ylabel("Histogram Z")
    plt.title("Z-basis")
    save_plot("histogram_z.png")

    plt.plot(bin_centers, reduced_histogram_x_repeated)
    plt.xlabel("Time (s)")
    plt.ylabel("Histogram X")
    plt.title("X-basis")
    save_plot("histogram_x.png")

# ================================================
# EXECUTION 
# ================================================

# PARAMETERS
n_pulses = int(1e4)
symbol_length = 2
jitter_source = 100e-12
repetition_rate = 1e9
time_resolution_factor = 4
factor_pulse_ampltiude = 2
p_z_alice = 0.8
p_z_1 = 0.5
p_decoy = 0.1 * 8
dB_on = 25
dB_off = 10
dB_decoy = 15
dB_channel_attenuation = 2
p_z_bob = p_z_alice
p_stray = 1e-4
p_short_path_DLI = 0.5
epsilon = 1e-3 # 2e-50
lambdaEC = 1.16

# simulate alice
bin_centers, average_number_photons_per_time_bin_source = simulate_pulsed_source(
    n_pulses, jitter_source, repetition_rate, time_resolution_factor, factor_pulse_ampltiude
)

basis_alice, value_alice, decoy_alice, pattern, mult_mod, mult_channel = generate_alice_choices(
    n_pulses, symbol_length, p_z_alice, p_z_1, p_decoy, 
    dB_on, dB_off, dB_decoy, dB_channel_attenuation
)

mult_mod_repeated = repeat_array(mult_mod, time_resolution_factor)
average_number_photons_per_time_bin_after_eam = average_number_photons_per_time_bin_source * mult_mod_repeated
average_number_photons_per_time_bin_after_channel = average_number_photons_per_time_bin_after_eam * mult_channel

# GET mu1, mu2 (and mu3)
mu_max_laser = np.max(average_number_photons_per_time_bin_source[:time_resolution_factor * 2])
mu1 = mu_max_laser * np.power(10, -1 * dB_off / 10)
mu2 = mu_max_laser * np.power(10, -1 * dB_decoy / 10)
mu3 = mu_max_laser * np.power(10, -1 * dB_on / 10) 
print("mu1: " + str(mu1) + ", mu2: " + str(mu2) + ", mu3: " + str(mu3))
mu = np.array([mu1, mu2])
p_mu = np.array([1-p_decoy, p_decoy])

# simulate detections
average_photons_z, average_photons_x, histogram_z, histogram_x = simulate_bob_detection(
    average_number_photons_per_time_bin_after_channel, n_pulses, time_resolution_factor, symbol_length, p_z_bob, p_stray, p_short_path_DLI
)

reduced_histogram_z = reduce_histogram_bins(histogram_z[time_resolution_factor // 2:-time_resolution_factor // 2], time_resolution_factor)
reduced_histogram_x = reduce_histogram_bins(histogram_x[time_resolution_factor // 2:-time_resolution_factor // 2], time_resolution_factor)

# CORRECT???? SIMULATE DESTRUCTIVE INTERFERENCE 
for i in range(1, len(reduced_histogram_x), 2):  # Iterate over every second element starting from the second element
    if reduced_histogram_x[i] >= 2:
      reduced_histogram_x[i] = 0

# plotting
# reduced_histogram_z_repeated = repeat_array(reduced_histogram_z, time_resolution_factor)
# reduced_histogram_x_repeated = repeat_array(reduced_histogram_x, time_resolution_factor)

#plotting(bin_centers, average_number_photons_per_time_bin_source, basis_alice, value_alice, mult_mod_repeated, average_number_photons_per_time_bin_after_eam, average_photons_z, average_photons_x, reduced_histogram_z_repeated, reduced_histogram_x_repeated)

# SKR
skr = calculate_skr(p_z_alice, mu, p_mu, epsilon, basis_alice, value_alice, decoy_alice, reduced_histogram_z, reduced_histogram_x, repetition_rate, n_pulses, symbol_length, lambdaEC)

print("SKR: " + str(skr))


# ================================================
# EXECUTION (FOR PLOTS)
# ================================================

"""
# PARAMETERS
n_pulses = int(32)
symbol_length = 2
jitter_source = 100e-12
repetition_rate = 1e9
time_resolution_factor = 8
factor_pulse_ampltiude = 2 * 100
p_z_alice = 0.5
p_z_1 = 0.5
p_decoy = 0.3
dB_on = 25
dB_off = 10
dB_decoy = 15
dB_channel_attenuation = 0
p_z_bob = p_z_alice
p_stray = 0
p_short_path_DLI = 0.5
epsilon = 1e-1 # 2e-50
lambdaEC = 1.16

# simulate alice
bin_centers, average_number_photons_per_time_bin_source = simulate_pulsed_source(
    n_pulses, jitter_source, repetition_rate, time_resolution_factor, factor_pulse_ampltiude
)

basis_alice, value_alice, decoy_alice, pattern, mult_mod, mult_channel = generate_alice_choices_fixed(
    n_pulses, symbol_length, p_z_alice, p_z_1, p_decoy, 
    dB_on, dB_off, dB_decoy, dB_channel_attenuation
)

mult_mod_repeated = repeat_array(mult_mod, time_resolution_factor)
average_number_photons_per_time_bin_after_eam = average_number_photons_per_time_bin_source * mult_mod_repeated
average_number_photons_per_time_bin_after_channel = average_number_photons_per_time_bin_after_eam * mult_channel

# GET mu1, mu2 (and mu3)
mu_max_laser = np.max(average_number_photons_per_time_bin_source[:time_resolution_factor * 2])
mu1 = mu_max_laser * np.power(10, -1 * dB_off / 10)
mu2 = mu_max_laser * np.power(10, -1 * dB_decoy / 10)
mu3 = mu_max_laser * np.power(10, -1 * dB_on / 10) 
print("mu1: " + str(mu1) + ", mu2: " + str(mu2) + ", mu3: " + str(mu3))
mu = np.array([mu1, mu2])
p_mu = np.array([1-p_decoy, p_decoy])

# simulate detections
average_photons_z, average_photons_x, histogram_z, histogram_x = simulate_bob_detection(
    average_number_photons_per_time_bin_after_channel, n_pulses, time_resolution_factor, symbol_length, p_z_bob, p_stray, p_short_path_DLI
)

reduced_histogram_z = reduce_histogram_bins(histogram_z[time_resolution_factor // 2:-time_resolution_factor // 2], time_resolution_factor)
reduced_histogram_x = reduce_histogram_bins(histogram_x[time_resolution_factor // 2:-time_resolution_factor // 2], time_resolution_factor)

# CORRECT???? SIMULATE DESTRUCTIVE INTERFERENCE 
#for i in range(1, len(reduced_histogram_x), 2):  # Iterate over every second element starting from the second element
#    if reduced_histogram_x[i] >= 2:
#        reduced_histogram_x[i] = 0

# plotting
reduced_histogram_z_repeated = repeat_array(reduced_histogram_z, time_resolution_factor)
reduced_histogram_x_repeated = repeat_array(reduced_histogram_x, time_resolution_factor)

plotting(bin_centers, average_number_photons_per_time_bin_source, basis_alice, value_alice, decoy_alice,mult_mod_repeated, average_number_photons_per_time_bin_after_eam, average_photons_z, average_photons_x, reduced_histogram_z_repeated, reduced_histogram_x_repeated)

# SKR
# skr = calculate_skr(p_z_alice, mu, p_mu, epsilon, basis_alice, value_alice, decoy_alice, reduced_histogram_z, reduced_histogram_x, repetition_rate, n_pulses, symbol_length, lambdaEC)
# print(skr)
"""