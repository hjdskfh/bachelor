import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import splev
from scipy.fftpack import fft, ifft, fftfreq
from scipy import constants
from scipy.special import factorial
import time
import gc

from saver import Saver
from simulationsingle import SimulationSingle
from simulationhelper import SimulationHelper
from plotter import Plotter


class SimulationEngine:
    def __init__(self, config):
        self.config = config
        self.simulation_single = SimulationSingle(config)
        self.simulation_helper = SimulationHelper(config)
        self.plotter = Plotter(config)


    def get_interpolated_value(self, x_data, name):
        #calculate tck for which curve
        tck = self.config.data.get_data(x_data, name)
        return splev(x_data, tck)

    def random_laser_output(self, current_power, voltage_shift, current_wavelength):
        'every batchsize values we get a new chosen value'
        # Generate a random time within the desired range
        times = self.config.rng.uniform(0, 10, self.config.n_samples // self.config.batchsize)
        # Use sinusoidal modulation for the entire array
        # chosen_voltage = self.config.mean_voltage + 0.050 * np.sin(2 * np.pi * 1 * times)
        # chosen_current = (self.config.mean_current) + self.config.current_amplitude * np.sin(2 * np.pi * 1 * times)) * 1e3
        chosen_voltage = np.ones(self.config.n_samples // self.config.batchsize) * self.config.mean_voltage
        chosen_current = np.ones(self.config.n_samples // self.config.batchsize) * (self.config.mean_current) * 1e3
        optical_power_short = self.get_interpolated_value(chosen_current, current_power)
        peak_wavelength_short = self.get_interpolated_value(chosen_current, current_wavelength) + self.get_interpolated_value(chosen_voltage, voltage_shift)
        print(f"peak_wavelength_short: {peak_wavelength_short}")
        optical_power = np.repeat(optical_power_short, self.config.batchsize)
        peak_wavelength = np.repeat(peak_wavelength_short, self.config.batchsize)
        return optical_power * 1e-3, peak_wavelength * 1e-9  # in W and m

    def generate_alice_choices(self, basis=None, value=None, decoy=None):
        """Generates Alice's choices for a quantum communication protocol."""
        # Generate arrays if parameters are not provided

        if basis is None:
            basis = self.config.rng.choice(
                [0, 1], size=self.config.n_samples, p=[1 - self.config.p_z_alice, self.config.p_z_alice]
            )
        if value is None:
            value = self.config.rng.choice(
                [0, 1], size=self.config.n_samples, p=[0.5, 0.5]
            )
        if decoy is None:
            decoy = self.config.rng.choice(
                [0, 1], size=self.config.n_samples, p=[1 - self.config.p_decoy, self.config.p_decoy]
            )

        # Ensure all inputs are NumPy arrays
        basis = np.array(basis, dtype=int)
        value = np.array(value, dtype=int)
        decoy = np.array(decoy, dtype=int)
        
        # Adjust value for array case if basis is 0
        if isinstance(basis, np.ndarray):
            value[basis == 0] = -1
        else:
            value = -1 if basis == 0 else value

        # Ensure outputs are arrays of the correct size
        if basis.size == 1:  # Scalar case
            basis = np.full(self.config.n_samples, basis, dtype=int)
        if value.size == 1:
            value = np.full(self.config.n_samples, value, dtype=int)
        if decoy.size == 1:
            decoy = np.full(self.config.n_samples, decoy, dtype=int)

        return basis, value, decoy
    
    def signal_bandwidth_jitter(self, basis, values, decoy):
        pulse_heights = self.simulation_helper.get_pulse_height(basis, decoy)
        jitter_shifts = self.simulation_helper.get_jitter('laser', size_jitter = self.config.n_samples * self.config.n_pulses)
        pulse_duration = 1 / self.config.sampling_rate_FPGA
        sampling_rate_fft = 100e11
        t, signals = self.simulation_helper.generate_encoded_pulse(pulse_heights, pulse_duration, values, sampling_rate_fft)
        #plt.plot(signals[0], label = 'before everything')
        #plt.legend()
        indexshift = len(t) // self.config.n_pulses // 2
        new_save_shift = np.empty(indexshift)
        old_save_shift = np.ones(indexshift) * self.config.non_signal_voltage

        for i in range(0, len(values), self.config.batchsize):
            signals_batch = signals[i:i + self.config.batchsize, :]
            flattened_signals_batch = signals_batch.reshape(-1)

            flattened_signals_batch = self.simulation_helper.apply_jitter_to_pulse(t, flattened_signals_batch, jitter_shifts[self.config.n_pulses * i:self.config.n_pulses *(i + self.config.batchsize)])

            if i == 0:
                amount_symbols_in_plot = 3
                pulse_duration = 1 / self.config.sampling_rate_FPGA
                sampling_rate_fft = 100e11
                samples_per_pulse = int(pulse_duration * sampling_rate_fft)
                total_samples = self.config.n_pulses * samples_per_pulse
                t_plot1 = np.linspace(0, amount_symbols_in_plot * self.config.n_pulses * pulse_duration, amount_symbols_in_plot * total_samples, endpoint=False)
                # print(f"shape flattened_signals_batch: {flattened_signals_batch.shape}")
                plt.plot(t_plot1 * 1e9, flattened_signals_batch[:len(t) * amount_symbols_in_plot])
                print(f"basis[:amount], value[:amount], decoy[:amount]: {basis[:amount_symbols_in_plot]}, {values[:amount_symbols_in_plot]}, {decoy[:amount_symbols_in_plot]}")
                for k in range(amount_symbols_in_plot + 1):
                    plt.axvline(t_plot1[-1] * 1e9 * (k) / (amount_symbols_in_plot), label=f'Target Mean Photon Number {i+1}', color = 'darkgreen')
                plt.title(f"Square Signal with jitter for the first 3 symbols")
                plt.xlabel('Time (ns)')
                plt.ylabel('Volt (V)')
                Saver.save_plot(f"square_signal")

            flattened_signals_batch = np.roll(flattened_signals_batch, indexshift)
            new_save_shift = flattened_signals_batch[:indexshift]
            flattened_signals_batch[: indexshift] = old_save_shift
            old_save_shift = new_save_shift
            
            
            
            flattened_signals_batch = self.simulation_helper.apply_bandwidth_filter(flattened_signals_batch, sampling_rate_fft)
            '''print(f"shape flattened_signals_batch: {flattened_signals_batch.shape}")
            plt.plot(flattened_signals_batch[:len(t)*3], label = 'BW')
            plt.show()'''

            flattened_signals_batch = flattened_signals_batch.reshape(self.config.batchsize, len(t))
            signals[i:i + self.config.batchsize, :] = flattened_signals_batch
          
            '''plt.legend()
            Saver.save_plot('without_bandwidth')
            plt.plot(filtered_signals[:6*len(t)], label = 'ende')
            Saver.save_plot('with_bandwidth')'''
            
        return signals, t, jitter_shifts

    def eam_transmission(self, voltage_signal, optical_power, T1_dampening, peak_wavelength, t):
        """fastest? prob not: Calculate the transmission and power for all elements in the arrays."""
        # Create a mask where voltage_signal is less than 7.023775e-05 = x_max
        _, x_max = self.config.data.get_data_x_min_x_max('eam_transmission')            
        mask = voltage_signal < x_max

        # Compute interpolated values only for the values that meet the condition (<x_max)
        interpolated_values = self.get_interpolated_value(voltage_signal[mask], 'eam_transmission')

        # rest if the values: should be value at x_max (here 1.0)
        signal_over_threshold = self.get_interpolated_value(x_max, 'eam_transmission')

        # fill transmission array with signal_over_threshold
        transmission = np.full_like(voltage_signal, signal_over_threshold)

        del voltage_signal
        gc.collect()

        # fill transmission array with interpolated_values where mask is True
        transmission[mask] = interpolated_values
       
        power_dampened = transmission * optical_power[:, None] / T1_dampening

        # Calculate the mean photon number
        energy_per_pulse = np.trapezoid(power_dampened, t, axis=1)
        calc_mean_photon_nr = energy_per_pulse / (constants.h * constants.c / peak_wavelength)

        # Normalize the transmission values
        norm_transmission = np.divide(transmission, transmission.sum(axis=1, keepdims=True), out=transmission)

        return power_dampened, norm_transmission, calc_mean_photon_nr, energy_per_pulse # leave these and recalculate them in detector
    
    def fiber_attenuation(self, power_dampened):    
        """Apply fiber attenuation to the power."""
        attenuation_factor = 10 ** (self.config.fiber_attenuation / 10)
        power_dampened = power_dampened * attenuation_factor
        return power_dampened

    def delay_line_interferometer(self, power_dampened, t, peak_wavelength):
    
        # phase difference between the two arms: w*delta T_bin, w = 2pi*f = 2pic/lambda
        delta_t_bin = t[-1] / 2                             # Time bin duration (float64)
        k_wave = 2 * constants.pi / peak_wavelength     # Frequency of the symbol (float64)
        print(f"peak_wavelength: {peak_wavelength[:10]}")
        effective_length_difference = constants.c * delta_t_bin / self.config.n_eff_in_fiber
        var_inside_cos = (effective_length_difference * (k_wave).reshape(-1, 1))
        var_inside_cos_mod = (var_inside_cos) % (2 * np.pi) / (2 * np.pi)
        print(f"delta_L * k mod 2pi: {var_inside_cos[:10], var_inside_cos[1000:1010], var_inside_cos[2000:2010]}")
        '''plt.plot(var_inside_cos)
        plt.show()
        plt.plot(var_inside_cos_mod)
        plt.show()'''
        attenuation_factor = 10 ** (self.config.insertion_loss_dli / 10)
        print(f"attenuation_factor in DLI: {attenuation_factor}")

        # power_dampened = power_dampened * attenuation_factor / 2 * (1 - np.cos((k_wave).reshape(-1, 1) * effective_length_difference))
        # self.plotter.plot_power(power_dampened, amount_symbols_in_plot=4, where_plot_1='+ after DLI')
        '''copy_power_dampened = copy_power_dampened * attenuation_factor / 2 * (1 + 2 * np.cos((k_wave).reshape(-1,1) * effective_length_difference))
        self.plotter.plot_power(copy_power_dampened, amount_symbols_in_plot=4, where_plot_1='- after DLI')'''

        sweep_wavelength = np.linspace(1552 * 1e-9, 1552.2 * 1e-9, 500)
        k_wave = 2 * constants.pi / sweep_wavelength     # Frequency of the symbol (float64)
        cos_term = np.cos((k_wave) * effective_length_difference)
        print(f"shape of cos_term in plot: {cos_term.shape}")
        
        plt.plot(sweep_wavelength *1e9, cos_term)
        plt.title(f"Cosine Term over wavelength")
        plt.ylabel('Amplitude')
        plt.xlabel('wavelength in nm')
        Saver.save_plot(f"cosine_term_plot")

        # phase difference between the two arms: w*delta T_bin, w = 2pi*f = 2pic/lambda
        delta_t_bin = t[-1] / 2                             # Time bin duration (float64)
        omega = 2 * constants.pi * constants.c / peak_wavelength     # Frequency of the symbol (float64)
        omega = (omega).reshape(-1, 1)
        omega_without_first = omega[1:]

        # get amplitude
        power_dampened = np.sqrt(power_dampened) # ignore phase bc is global phase
        amplitude = power_dampened

        # Time bin split point ( for late time bin start)
        split_point = len(t) // 2

        late_bin_ex_last = amplitude[:-1, split_point:]
        early_bin_ex_first = amplitude[1:, :split_point]
        whole_early_bin = amplitude[:, :split_point]
        whole_late_bin = amplitude[:, split_point:]
        '''plt.plot(early_bin_ex_first[2:6].flatten())
        plt.title("Early Bin Ex First (2-6th row)")
        plt.xlabel("Time Bins")
        plt.ylabel("Amplitude")
        plt.grid()
        plt.show()'''

        sampling_rate_fft = 100e11
        amplitude_fft = fft(amplitude)
        frequencies = fftfreq(len(amplitude), d=1 / sampling_rate_fft)
        omega = 2 * np.pi * frequencies  # Convert to angular frequency
        t_shift = t[-1] / 2
        phase_shift = np.exp(1j * omega * t_shift)
        amplitude_fft = amplitude_fft * phase_shift

        # Compute the inverse FFT to get the time-shifted signal, Take real part (since the IFFT might introduce small imaginary noise)
        amplitude = np.real(ifft(amplitude_fft))


        #???np.multiply(S_fourier, np.interp(np.abs(frequencies), freq_x, freq_y), out=S_fourier)

        #???return np.real(ifft(S_fourier))


        # Calculate the interference term nth symbol and n+1th symbol (early-time and late-time bins)

        # Pre-allocate arrays for the total power
        power_dampened_total = np.zeros((self.config.n_samples, len(t)))

        '''# Sum the contributions for total power (including interference)
        # For the nth and (n+1)th symbol:
        power_over_symbols_n_np1 = np.abs(1 / 2 * (early_bin_ex_first - late_bin_ex_last * np.exp(1j * omega_without_first * delta_t_bin)))**2
        # power_over_symbols_n_np1 = np.abs(1 / 2 * (1j * early_bin_ex_first + 1j * late_bin_ex_last * np.exp(1j * omega_without_first * delta_t_bin)))**2
        # power_over_symbols_n_np1 = np.abs(1 / 2 * (early_bin_ex_first + late_bin_ex_last * np.exp(1j * omega_without_first * delta_t_bin)))**2
        # power_over_symbols_n_np1 = np.abs(1 / 2 * (1j * early_bin_ex_first - 1j * late_bin_ex_last * np.exp(1j * omega_without_first * delta_t_bin)))**2


        plt.plot(power_over_symbols_n_np1[2:6].flatten())
        plt.title("Power Over Symbols N and N+1 (2-6th row)")
        plt.xlabel("Time Bins")
        plt.ylabel("Power Over Symbols")
        plt.grid()
        plt.show()

        power_dampened_total[1:, :split_point] = (
            power_over_symbols_n_np1  # nth and (n+1)th interference
        )

        # For the (n+1)th symbol:
        power_over_symbols_np1 = np.abs(1 / 2 * (whole_late_bin - whole_early_bin * np.exp(1j * omega * delta_t_bin)))**2
        # power_over_symbols_np1 = np.abs(1 / 2 * (1j * whole_late_bin + 1j * whole_early_bin * np.exp(1j * omega * delta_t_bin)))**2
        # power_over_symbols_np1 = np.abs(1 / 2 * (whole_late_bin + whole_early_bin * np.exp(1j * omega * delta_t_bin)))**2
        # power_over_symbols_np1 = np.abs(1 / 2 * (1j * whole_late_bin - 1j * whole_early_bin * np.exp(1j * omega * delta_t_bin)))**2

        power_dampened_total[:, split_point:] = (
            power_over_symbols_np1 # (n+1)th symbol interference
        )

        # for first row early bin
        #power_dampened_total[0, :split_point] = power_dampened[0, :split_point]

        part_power = power_dampened_total[2:6].flatten()
        plt.plot(part_power)
        plt.title("Power Dampened Total (3-6th row) - mit i")
        plt.xlabel("Time Bins")
        plt.ylabel("Power Dampened")
        plt.grid()
        Saver.save_plot("power_dampened_total")'''

        return power_dampened_total
    
    
    def detector(self, t, norm_transmission, peak_wavelength, power_dampened, start_time):
        """Simulate the detector process."""
        # choose photons
        wavelength_photons, time_photons, nr_photons, index_where_photons, all_time_max_nr_photons, calc_mean_photon_nr_detector = self.simulation_helper.choose_photons(norm_transmission, t, power_dampened, peak_wavelength, start_time, fixed_nr_photons=None)
        Saver.memory_usage("after choose: " + str(time.time() - start_time))

        # Will the photons pass the detection efficiency?
        pass_detection = self.config.rng.choice([False, True], size=(len(index_where_photons), all_time_max_nr_photons), p=[1 - self.config.detector_efficiency, self.config.detector_efficiency])
        
        # Apply the detection efficiency to the photon properties
        wavelength_photons = np.where(pass_detection, wavelength_photons, np.nan)
        time_photons = np.where(pass_detection, time_photons, np.nan)

        # delete all Nan values
        valid_rows = ~np.isnan(wavelength_photons).all(axis=1)
        nr_photons_det = nr_photons[valid_rows] #nr_photons only for the ones we carry, rest 0
        index_where_photons_det = index_where_photons[valid_rows]
        wavelength_photons_det = wavelength_photons[valid_rows]
        time_photons_det = time_photons[valid_rows]
        
        Saver.memory_usage("before detection time:" + str(time.time() - start_time))
        # Last photon detected --> can next photon be detected? --> sort stuff
        sorted_indices = np.array([np.argsort(time) for time in time_photons_det])
        wavelength_photons = np.array([wavelength_photon[indices] for wavelength_photon, indices in zip(wavelength_photons_det, sorted_indices)])
        time_photons = np.array([time_photon[indices] for time_photon, indices in zip(time_photons_det, sorted_indices)])

        # Apply the detection time to the photon properties
        time_photons_det, wavelength_photons_det = self.simulation_helper.filter_photons_detection_time(time_photons_det, wavelength_photons_det)

        # delete all Nan values
        valid_rows = ~np.isnan(wavelength_photons_det).all(axis=1)
        nr_photons_det = nr_photons_det[valid_rows] #nr_photons only for the ones we carry, rest 0
        index_where_photons_det = index_where_photons_det[valid_rows]
        wavelength_photons_det = wavelength_photons_det[valid_rows]
        time_photons_det = time_photons_det[valid_rows]
        
        Saver.memory_usage("before jitter detector: " + str(time.time() - start_time))

        # jitter detector: timing jitter
        time_photons_det = self.simulation_helper.add_detection_jitter(t, time_photons_det)

        Saver.memory_usage("before darkcount: " + str(time.time() - start_time))

        # calculate darkcount
        dark_count_times, num_dark_counts = self.simulation_helper.darkcount()

        return time_photons_det, wavelength_photons_det, nr_photons_det, index_where_photons_det, calc_mean_photon_nr_detector, dark_count_times, num_dark_counts
    
    def classificator(self, t, time_photons_det_x, index_where_photons_det_x, time_photons_det_z, index_where_photons_det_z, basis, value, decoy):
        """Classify time bins."""
        num_segments = self.config.n_pulses // 2
        timebins = np.linspace(t[-1] / num_segments, t[-1], num_segments)        
        detected_indices_z_norm = self.simulation_helper.classificator_det_ind(timebins, decoy, time_photons_det_z, index_where_photons_det_z, is_decoy = False)
        # print(f"shape detected_indices_z_norm: {detected_indices_z_norm}")

        detected_indices_z_dec = self.simulation_helper.classificator_det_ind(timebins, decoy, time_photons_det_z, index_where_photons_det_z, is_decoy = True)
        # print(f"shape detected_indices_z_dec: {detected_indices_z_dec}")

        detected_indices_x_norm = self.simulation_helper.classificator_det_ind(timebins, decoy, time_photons_det_x, index_where_photons_det_x, is_decoy = False)
        # print(f"shape detected_indices_x_norm: {detected_indices_x_norm}")

        detected_indices_x_dec = self.simulation_helper.classificator_det_ind(timebins, decoy, time_photons_det_x, index_where_photons_det_x, is_decoy=True)
        # print(f"detected_indices_x_dec shape: {detected_indices_x_dec.shape}")

        gain_Z_norm, amount_Z_det_norm = self.simulation_helper.classificator_z(basis, value, decoy, index_where_photons_det_z, detected_indices_z_norm, detected_indices_x_norm, is_decoy=False)
        gain_Z_dec, amount_Z_det_dec = self.simulation_helper.classificator_z(basis, value, decoy, index_where_photons_det_z, detected_indices_z_dec, detected_indices_x_dec, is_decoy=True)
        gain_XP_norm, amount_XP_det_norm = self.simulation_helper.classificator_x(basis, value, decoy, index_where_photons_det_x, detected_indices_x_norm, detected_indices_z_norm, gain_Z_norm, is_decoy=False)
        gain_XP_dec, amount_XP_det_dec = self.simulation_helper.classificator_x(basis, value, decoy, index_where_photons_det_x, detected_indices_x_dec, detected_indices_z_dec, gain_Z_dec, is_decoy=True)

        # Use np.hstack to concatenate, and it automatically handles empty arrays
        total_detected_indices_x = np.vstack((detected_indices_x_dec, detected_indices_x_norm)) if detected_indices_x_dec.size > 0 or detected_indices_x_norm.size > 0 else np.empty((0, 0))
        total_detected_indices_z = np.vstack((detected_indices_z_dec, detected_indices_z_norm)) if detected_indices_z_dec.size > 0 or detected_indices_z_norm.size > 0 else np.empty((0, 0))
        # print(f"total_detected_indices_x: {total_detected_indices_x.shape}")
        # print(f"total_detected_indices_z: {total_detected_indices_z.shape}")

        wrong_detections_z, wrong_detections_x = self.simulation_helper.classificator_error_cases(basis, value, index_where_photons_det_x, index_where_photons_det_z, total_detected_indices_x, total_detected_indices_z)
        len_wrong_detections_z = len(wrong_detections_z)
        len_wrong_detections_x = len(wrong_detections_x)
        amount_Z_detections = amount_Z_det_norm + amount_Z_det_dec
        amount_XP_detections = amount_XP_det_norm + amount_XP_det_dec
        total_amount_detections = amount_Z_det_norm + amount_Z_det_dec + amount_XP_det_norm + amount_XP_det_dec

        if amount_Z_det_norm != 0:
            qber = len(wrong_detections_z) / amount_Z_det_norm
        else:
            qber = 0

        if amount_XP_det_norm != 0:
            phase_error_rate = len(wrong_detections_x) / amount_XP_det_norm
        else:
            phase_error_rate = 0

        raw_key_rate = total_amount_detections / (t[-1] * self.config.n_samples)

        return len_wrong_detections_z, len_wrong_detections_x, total_amount_detections, amount_Z_detections, amount_XP_detections, qber, phase_error_rate, raw_key_rate, gain_XP_norm, gain_XP_dec, gain_Z_norm, gain_Z_dec, detected_indices_x_dec, detected_indices_x_norm, detected_indices_z_dec, detected_indices_z_norm
    
    def initialize(self):
        plt.style.use(self.config.mlp)
        self.config.validate_parameters() #some checks if parameters are in valid ranges
        print(f"seed: {self.config.seed}")

        # calculate T1 dampening 
        lower_limit_t1, upper_limit_t1, tol_t1 = 0, 100, 1e-3
        T1_dampening = self.simulation_single.find_T1(lower_limit_t1, upper_limit_t1, tol_t1)

        # test
        if T1_dampening > (upper_limit_t1 - 10*tol_t1) or T1_dampening < (lower_limit_t1 + 10*tol_t1):
            raise ValueError(f"T1 dampening is with {T1_dampening} very close to limit [{lower_limit_t1}, {upper_limit_t1}] with tolerance {tol_t1}")
        print('T1_dampening at initialize end: ' +str(T1_dampening))

        # with simulated decoy state: calculate decoy height
        lower_limit, upper_limit, tol = -1.2, 1, 1e-7
        var_voltage_decoy = self.config.voltage_decoy
        var_voltage_sup = self.config.voltage_sup
        var_voltage_decoy_sup = self.config.voltage_decoy_sup

        var_voltage_sup, var_voltage_decoy, var_voltage_decoy_sup = self.simulation_single.find_voltage_decoy(T1_dampening, var_voltage_decoy, var_voltage_sup, var_voltage_decoy_sup, lower_limit=lower_limit, upper_limit=upper_limit, tol=tol)
        
        self.config.voltage_sup = var_voltage_sup
        self.config.voltage_decoy = var_voltage_decoy
        self.config.voltage_decoy_sup = var_voltage_decoy_sup

        print('voltage at initialize end: ' + str(self.config.voltage))
        print('Voltage_decoy at initialize end: ' + str(self.config.voltage_decoy))
        print('Voltage_decoy_sup at initialize end: ' + str(self.config.voltage_decoy_sup))
        print('Voltage_sup at initialize end: ' + str(self.config.voltage_sup))

        # test if voltage values make sense
        if self.config.voltage_decoy > (upper_limit - 10*tol) or self.config.voltage_decoy < (lower_limit + 10*tol):
            raise ValueError(f"voltage decoy is with {self.config.voltage_decoy} very close to limit [{lower_limit}, {upper_limit}] with tolerance {tol}")
        self.config.voltage_decoy = var_voltage_decoy
        if self.config.voltage_decoy > self.config.voltage:
            raise ValueError(f"self.config.voltage_decoy > self.config.voltage is True")
        if self.config.voltage_decoy_sup > self.config.voltage_sup:
            raise ValueError(f"self.config.voltage_decoy_sup > self.config.voltage_decoy is True")
        
        '''# basis = 0, value = 0, decoy = 0:  1010 non-decoy state
        basis_fix, value_fix, decoy_fix = self.simulation_single.generate_alice_choices_single(basis = 0, value = 0, decoy = 0)
        signals, t, _ = self.simulation_single.signal_bandwidth_single(basis_fix, value_fix, decoy_fix)

        plt.plot(t * 1e9, signals, label = '1010 non-decoy')

        plt.title(f"Voltage Signal with Bandwidth and Jitter for one symbol")
        plt.ylabel('Volt (V)')
        plt.xlabel('Time (ns)')
        plt.legend()
        Saver.save_plot(f"signal_after_init_1010_voltage")
        
        optical_power, _ = self.simulation_single.random_laser_output_single('current_power', 'voltage_shift', 'current_wavelength')
        power_dampened, _ = self.simulation_single.eam_transmission_single(signals, optical_power, T1_dampening)
        plt.plot(t * 1e9, power_dampened * 1e3)
        plt.title(f"Power for one symbol")
        plt.ylabel('Power (mW)')
        plt.xlabel('Time (ns)')
        Saver.save_plot(f"signal_after_init_1010_power")'''

        return T1_dampening
    
   