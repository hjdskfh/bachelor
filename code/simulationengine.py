from re import A
import numpy as np
import matplotlib.pyplot as plt
from scipy import constants
import time
import gc
from scipy.interpolate import splrep, splev


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

    def get_interpolated_value(self, x_data, name, inverse_flag=False):
        # Otherwise, use the regular spline
        tck = self.config.data.get_data(x_data, name, inverse=inverse_flag)
        if inverse_flag:
            return tck(x_data)
        return splev(x_data, tck)  # Interpolate with the regular spline

    def random_laser_output(self, current_power, voltage_shift, fixed=None):
        'every batchsize values we get a new chosen value'
        # Generate a random time within the desired range
        times = self.config.rng.uniform(0, 10, self.config.n_samples // self.config.batchsize)
        # Use sinusoidal modulation for the entire array
        if fixed is None:
            chosen_voltage = self.config.mean_voltage + self.config.voltage_amplitude * np.sin(2 * np.pi * 1 * times)  # 50 mV passt
            chosen_current = ((self.config.mean_current) + self.config.current_amplitude * np.sin(2 * np.pi * 1 * times)) * 1e3
            '''# plot sinuses
            mean_voltage = 0.05  # 50 mV
            voltage_amplitude = 0.02  # 20 mV
            mean_current = 0.01  # 10 mA
            current_amplitude = 0.005  # 5 mA

            # Generate time values
            times = np.linspace(0, 1, 1000)  # 1 second with 1000 points

            # Generate sinusoidal signals
            chosen_voltage = mean_voltage + voltage_amplitude * np.sin(2 * np.pi * 1 * times)
            chosen_current = (mean_current + current_amplitude * np.sin(2 * np.pi * 1 * times)) * 1e3  # Convert to mA

            # Plot the signals
            plt.figure(figsize=(10, 6))
            plt.plot(times, chosen_voltage, label="Voltage (V)", color="blue")
            plt.plot(times, chosen_current, label="Current (mA)", color="red")
            plt.title("Sinusoidal Voltage and Current Signals")
            plt.xlabel("Time (s)")
            plt.ylabel("Amplitude")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.show()'''
        else:
            chosen_voltage = np.ones(self.config.n_samples // self.config.batchsize) * self.config.mean_voltage
            chosen_current = np.ones(self.config.n_samples // self.config.batchsize) * (self.config.mean_current) * 1e3
        optical_power_short = self.get_interpolated_value(chosen_current, current_power)
        peak_wavelength_short =  1550 + self.get_interpolated_value(chosen_voltage, voltage_shift)  # self.get_interpolated_value(chosen_current, current_wavelength) +
        optical_power = np.repeat(optical_power_short, self.config.batchsize)
        peak_wavelength = np.repeat(peak_wavelength_short, self.config.batchsize)
        return optical_power * 1e-3, peak_wavelength * 1e-9, chosen_voltage, chosen_current  # in W and m

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

        # Ensure outputs are arrays of the correct size
        if basis.size == 1:  # Scalar case
            basis = np.full(self.config.n_samples, basis, dtype=int)
        if value.size == 1:
            value = np.full(self.config.n_samples, value, dtype=int)
        if decoy.size == 1:
            decoy = np.full(self.config.n_samples, decoy, dtype=int)

        if basis.size > 1: # array-case
            basis = np.tile(basis, (self.config.n_samples // len(basis)) + 1)[:self.config.n_samples]
        if value.size > 1:
            value = np.tile(value, (self.config.n_samples // len(value)) + 1)[:self.config.n_samples]
        if decoy.size > 1:
            decoy = np.tile(decoy, (self.config.n_samples // len(decoy)) + 1)[:self.config.n_samples]

        # Adjust value for array case if basis is 0
        if isinstance(basis, np.ndarray):
            value[basis == 0] = -1
        else:
            value = -1 if basis == 0 else value
        return basis, value, decoy
    
    def signal_bandwidth_jitter(self, basis, values, decoy):
        pulse_heights = self.simulation_helper.get_pulse_height(basis, decoy)
        # jitter_shifts = self.simulation_helper.get_jitter('laser', size_jitter = self.config.n_samples * self.config.n_pulses)
        pulse_duration = 1 / self.config.sampling_rate_FPGA
        sampling_rate_fft = 100e11
        t, square_signals = self.simulation_helper.generate_encoded_pulse(pulse_heights, pulse_duration, values, sampling_rate_fft)
        signals = square_signals.copy()
        indexshift = len(t) // self.config.n_pulses // 2
        new_save_shift = np.empty(indexshift)
        old_save_shift = np.ones(indexshift) * self.config.non_signal_voltage
        new_save_shift_square = np.empty(indexshift)
        old_save_shift_square = np.ones(indexshift) * self.config.non_signal_voltage

        for i in range(0, len(values), self.config.batchsize):
            signals_batch = signals[i:i + self.config.batchsize, :]
            square_signals_batch = square_signals[i:i + self.config.batchsize, :]
            flattened_signals_batch = signals_batch.reshape(-1)
            square_signals_batch = square_signals_batch.reshape(-1)

            # flattened_signals_batch = self.simulation_helper.apply_jitter_to_pulse(t, flattened_signals_batch, jitter_shifts[self.config.n_pulses * i:self.config.n_pulses *(i + self.config.batchsize)])

            '''if i == 0:
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
                Saver.save_plot(f"square_signal")'''
            # roll for better detection
            flattened_signals_batch = np.roll(flattened_signals_batch, indexshift)
            new_save_shift = flattened_signals_batch[:indexshift]
            flattened_signals_batch[: indexshift] = old_save_shift
            old_save_shift = new_save_shift

            # also roll square_signals_batch for comparison
            square_signals_batch = np.roll(square_signals_batch, indexshift)
            new_save_shift_square = square_signals_batch[:indexshift]
            square_signals_batch[: indexshift] = old_save_shift_square
            old_save_shift_square = new_save_shift_square
            
            flattened_signals_batch = self.simulation_helper.apply_bandwidth_filter(flattened_signals_batch, sampling_rate_fft)
            '''print(f"shape flattened_signals_batch: {flattened_signals_batch.shape}")
            plt.plot(flattened_signals_batch[:len(t)*3], label = 'BW')
            plt.show()'''

            # reshape so one can write back into matrix
            flattened_signals_batch = flattened_signals_batch.reshape(self.config.batchsize, len(t))
            signals[i:i + self.config.batchsize, :] = flattened_signals_batch
            square_signals_batch = square_signals_batch.reshape(self.config.batchsize, len(t))
            square_signals[i:i + self.config.batchsize, :] = square_signals_batch

          
            '''plt.legend()
            Saver.save_plot('without_bandwidth')
            plt.plot(filtered_signals[:6*len(t)], label = 'ende')
            Saver.save_plot('with_bandwidth')'''
            
        return signals, t, square_signals

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
        energy_per_pulse = np.trapz(power_dampened, t, axis=1)
        calc_mean_photon_nr = energy_per_pulse / (constants.h * constants.c / peak_wavelength)

        # Normalize the transmission values
        norm_transmission = np.divide(transmission, transmission.sum(axis=1, keepdims=True), out=transmission)

        return power_dampened, calc_mean_photon_nr, energy_per_pulse # leave these and recalculate them in detector
    
    def fiber_attenuation(self, power_dampened):    
        """Apply fiber attenuation to the power."""
        attenuation_factor = 10 ** (self.config.fiber_attenuation / 10)
        power_dampened = power_dampened * attenuation_factor
        return power_dampened

    def delay_line_interferometer(self, power_dampened, t, peak_wavelength, value):
        dt_new = 1e-14

        tau = 2 / self.config.sampling_rate_FPGA  
        n_g = 2.05 # For calculatting path length difference
        # n_g = 1
        # Assuming the group refractive index of the waveguide
        n_eff = 1.56 # Effective refractive index
        
        delta_L = tau * constants.c / n_g

        for i in range(0, len(value), self.config.batchsize):

            power_dampened_batch = power_dampened[i:i + self.config.batchsize, :]
            flattened_power_batch = power_dampened_batch.reshape(-1)
            # print(f"flattened_power_batch: {flattened_power_batch.shape}, power_dampened_batch: {power_dampened_batch.shape}, t: {t.shape}")

            dt_original = t[1] - t[0]
            num_points = flattened_power_batch.size 
            # print(f"num_points:{num_points}")
            t_original_all_sym = np.arange(t[0], t[0] + num_points * dt_original, dt_original)     
            # print(f"t_original_all_sym: {t_original_all_sym.shape}, t: {t.shape}, step t_original: {t_original_all_sym[1]-t_original_all_sym[0]}, t step:{t[1]-t[0]}")       

            # calculate new time axis
            t_new_all_sym = np.arange(t_original_all_sym[0], t_original_all_sym[-1], dt_new)
            
            # Resample the data using np.interp (linear interpolation)
            flattened_power_batch_resampled = np.interp(t_new_all_sym, t_original_all_sym, flattened_power_batch)
            '''plt.plot(flattened_power_batch_resampled[:len(t)*100], color = 'green', label = 'after resample')
            plt.plot(flattened_power_batch[:len(t)*10], label = 'before resample')
            plt.show()'''
            
            flattened_power_batch_resampled_copy = flattened_power_batch_resampled.copy()

            # t_test ist gleich wie das t das in der Funktion gemacht wird
            t_test = np.arange(len(flattened_power_batch_resampled)) * dt_new
            # print(f"t_test:{t_test.shape}, t_test:{t_test[-1]}, t_new_all_sym:{t_new_all_sym.shape} t_new_all_sym: {t_new_all_sym[-1]}")
            
            f_0 = constants.c / peak_wavelength[i]
            
            n_eff_calculated = self.get_interpolated_value(peak_wavelength[i] * 1e9, 'wavelength_neff')
            print(f"n_eff_calculated: {n_eff_calculated} normal {n_eff}, peak_wavelength[i]: {peak_wavelength[i]}, f_0: {f_0}, i: {i}, batchsize: {self.config.batchsize}")

            '''print(f"f_0: {f_0}, i: {i}, batchsize: {self.config.batchsize}, peak_wavelength[i]: {peak_wavelength[i]}")'''

            power_1, power_2, _ = self.simulation_helper.DLI(flattened_power_batch_resampled, dt_new, tau, delta_L, f_0,  n_eff)
            '''plt.plot(power_1[:len(t)], label = 'after DLI 1')
            plt.plot(power_2[:len(t)], label = 'after DLI 2')
            plt.legend()
            Saver.save_plot(f"power_after_DLI_{i // self.config.batchsize}")'''
            
            '''# plot von power_1 und power_2
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
            # Plot the first graph in the first subplot
            ax1.plot(t_test *1e9, flattened_power_batch_resampled_copy, label='before DLI')
            ax1.plot(t_test *1e9, power_2, label='after DLI 2, ' + str(peak_wavelength[i // self.config.batchsize]))
            ax1.set_ylabel('Power (W)')
            ax1.set_xlabel('Time (ns)')
            ax1.set_title(f"Power after DLI port 2 wavelength {peak_wavelength[i]} for {self.config.batchsize} symbols")
            ax1.legend()
            # Plot the second graph in the second subplot
            ax2.plot(t_test *1e9, flattened_power_batch_resampled_copy, label='before DLI')
            ax2.plot(t_test *1e9, power_1, label='after DLI 1, ' + str(peak_wavelength[i // self.config.batchsize]))
            ax2.set_ylabel('Power (W)')
            ax2.set_xlabel('Time (ns)')
            ax2.set_title(f"Power after DLI port 1 wavelength {peak_wavelength[i]} for {self.config.batchsize} symbols")
            ax2.legend()
            # Save the plot with a custom filename
            Saver.save_plot(f"power_after_DLI_combined")'''

            # Use interpolation to compute signal values at new time points
            signal_downsampled = np.interp(t_original_all_sym, t_new_all_sym, power_1)
            '''plt.plot(signal_downsampled[:len(t)*100], label = 'after DLI 1')
            plt.plot(power_1[:len(t)*100], color = 'green', label = 'after DLI 1')
            plt.show()'''

            flattened_power_batch = signal_downsampled.reshape(self.config.batchsize, len(t))
            power_dampened[i:i + self.config.batchsize, :] = flattened_power_batch
        return power_dampened, f_0

    
    def detector(self, t, peak_wavelength, power_dampened, start_time=0):
        """Simulate the detector process."""
        # choose photons
        wavelength_photons, time_photons, nr_photons, index_where_photons, all_time_max_nr_photons, calc_mean_photon_nr_detector = self.simulation_helper.choose_photons(t, power_dampened, peak_wavelength, start_time, fixed_nr_photons=None)
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
        dark_count_times, num_dark_counts = self.simulation_helper.darkcount(index_where_photons_det, time_photons_det)

        return time_photons_det, wavelength_photons_det, nr_photons_det, index_where_photons_det, calc_mean_photon_nr_detector, dark_count_times, num_dark_counts
    
    def classificator_new(self, t, time_photons_det_x, index_where_photons_det_x, time_photons_det_z, index_where_photons_det_z, basis, value, decoy):
        """Classify time bins."""
        midpoint = t[-1] / 2  # Midpoint for classification
        detected_indices_z = np.where(
            np.isnan(time_photons_det_z),                                         # Check for NaN values
            -1,                                                                 # Assign -1 for undetected photons
            np.where(time_photons_det_z < midpoint, 0, 1)  # else, go deeper                         # Early = 0, Late = 1
            )
        
        detected_indices_x = np.where(
            np.isnan(time_photons_det_x),                                         # Check for NaN values
            -1,                                                                 # Assign -1 for undetected photons
            np.where(time_photons_det_x < midpoint, 0, 1)  # else, go deeper                        # Early = 0, Late = 1
            )
        
        detected_indices_z_det_z_basis, p_vacuum_z, total_sift_z_basis_short, \
        indices_z_long, mask_z_short, get_original_indexing_z = self.simulation_helper.classificator_sift_z_vacuum(basis, detected_indices_z, index_where_photons_det_z)
        detected_indices_x_det_x_basis, total_sift_x_basis_long, vacuum_indices_x_long, \
        indices_x_long, mask_x_short, get_original_indexing_x = self.simulation_helper.classificator_sift_x_vacuum(basis, detected_indices_x, index_where_photons_det_x)
        '''with np.printoptions(threshold=np.inf):
            print(f"time_photons_det_x part: {time_photons_det_x[:10]}")
            len(f"time_photons_det_x: {len(time_photons_det_x)}")
            print(f"detected_indices_x part: {detected_indices_x[:10]}")
            print(f"detected_indices_x shape: {detected_indices_x.shape}")
            print("index_where_photons_det_x part: ", index_where_photons_det_x[:10])
            print(f"detected_indices_x_det_x_basis: {detected_indices_x_det_x_basis}")
            print(f"get_original_indexing_x part: {get_original_indexing_x[:10]}")
            print(f"XP_alice_s part {np.where((basis == 0) & (decoy == 1))[0][:10]}")'''
        gain_Z_non_dec, gain_Z_dec, len_Z_checked_dec, len_Z_checked_non_dec = self.simulation_helper.classificator_identify_z(mask_x_short, value, total_sift_z_basis_short, 
                                                                                                                               detected_indices_x_det_x_basis, index_where_photons_det_z, decoy, indices_z_long, get_original_indexing_z, get_original_indexing_x)

        X_P_calc_non_dec, X_P_calc_dec, gain_X_non_dec, gain_X_dec = self.simulation_helper.classificator_identify_x(mask_x_short, mask_z_short, detected_indices_x_det_x_basis, 
                                                                                                                     detected_indices_z_det_z_basis, basis, value, decoy, indices_x_long, get_original_indexing_x, get_original_indexing_z)

        wrong_detections_z_dec, wrong_detections_z_non_dec, wrong_detections_x_dec, wrong_detections_x_non_dec = self.simulation_helper.classificator_errors(mask_x_short, mask_z_short, indices_z_long, indices_x_long, value, detected_indices_z_det_z_basis, detected_indices_x_det_x_basis, basis, decoy, get_original_indexing_x, get_original_indexing_z)


        qber_z_dec, qber_z_non_dec, qber_x_dec, qber_x_non_dec, raw_key_rate, total_amount_detections = self.simulation_helper.classificator_qber_rkr(t, wrong_detections_z_dec, wrong_detections_z_non_dec, wrong_detections_x_dec, wrong_detections_x_non_dec, len_Z_checked_dec, len_Z_checked_non_dec, X_P_calc_non_dec, X_P_calc_dec)

        return p_vacuum_z, vacuum_indices_x_long, len_Z_checked_dec, len_Z_checked_non_dec, gain_Z_non_dec, gain_Z_dec, gain_X_non_dec, gain_X_dec, X_P_calc_non_dec, X_P_calc_dec, wrong_detections_z_dec, wrong_detections_z_non_dec, wrong_detections_x_dec, wrong_detections_x_non_dec, qber_z_dec, qber_z_non_dec, qber_x_dec, qber_x_non_dec, raw_key_rate, total_amount_detections
    
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
    
    def find_p_indep_states_x_for_classifier(self, T1_dampening, simulation_length_factor=1000, is_decoy=False):
        # create signal Z0X+ and then X+Z1
        # Generate the repeating pattern for basis: [1, 0, 0, 1]
        basis_pattern = [1, 0, 0, 1]
        copy_old_n_samples = self.config.n_samples
        self.config.n_samples = len(basis_pattern) * simulation_length_factor

        # Generate the repeating pattern for value: [1, -1, -1, 0]
        value_pattern = [1, -1, -1, 0]

        # Set decoy array to all zeros
        if is_decoy:
            decoy_pattern = 1
        else:
            decoy_pattern = 0

        optical_power, peak_wavelength, chosen_voltage, chosen_current = self.random_laser_output('current_power', 'voltage_shift', 'current_wavelength')
        basis, value, decoy = self.generate_alice_choices(basis=basis_pattern, value=value_pattern, decoy=decoy_pattern)
        signals, t, _ = self.signal_bandwidth_jitter(basis, value, decoy)
        power_dampened, norm_transmission, _, _ = self.eam_transmission(signals, optical_power, T1_dampening, peak_wavelength, t)
        power_dampened = self.fiber_attenuation(power_dampened)

        # Z basis
        power_dampened = power_dampened * self.config.p_z_bob
        time_photons_det_z, _,_, index_where_photons_det_z,_,_, _ = self.detector(t, norm_transmission, peak_wavelength, power_dampened)
        power_dampened = power_dampened / self.config.p_z_bob

        # X basis
        power_dampened = power_dampened * (1 - self.config.p_z_bob)

        amount_symbols_in_first_part = 20
        first_power = power_dampened[:amount_symbols_in_first_part]

        power_dampened, _ = self.delay_line_interferometer(power_dampened, t, peak_wavelength)

        self.plotter.plot_power(power_dampened, amount_symbols_in_plot=amount_symbols_in_first_part, where_plot_1='before DLI',  shortened_first_power=first_power, where_plot_2='after DLI,', title_rest='- in fft, mean_volt: ' + str("{:.3f}".format(self.config.mean_voltage)) + ' voltage: ' + str("{:.3f}".format(chosen_voltage[0])) + ' V and ' + str("{:.3f}".format(peak_wavelength[0])))


        time_photons_det_x, _, _, index_where_photons_det_x, _, _, _ = self.detector(t, norm_transmission, peak_wavelength, power_dampened)        


        # classificator
        num_segments = self.config.n_pulses // 2
        timebins = np.linspace(t[-1] / num_segments, t[-1], num_segments)        
        
        detected_indices_z = np.where(
            np.isnan(time_photons_det_z),                                         # Check for NaN values
            -1,                                                                 # Assign -1 for undetected photons
            np.digitize(time_photons_det_z, timebins) - 1                         # Early = 0, Late = 1
            )
        
        detected_indices_x = np.where(
            np.isnan(time_photons_det_x),                                         # Check for NaN values
            -1,                                                                 # Assign -1 for undetected photons
            np.digitize(time_photons_det_x, timebins) - 1                         # Early = 0, Late = 1
            )
        
        detected_indices_x_det_x_basis, total_sift_x_basis_long, vacuum_indices_x_long, indices_x_long, mask_x_short = self.simulation_helper.classificator_sift_x_vacuum(basis, detected_indices_x, index_where_photons_det_x)
        p_indep_x_states, len_ind_has_one_0_and_every_second_symbol, len_ind_every_second_symbol = self.simulation_helper.classificator_identify_x_calc_p_indep_states_x(mask_x_short, detected_indices_x_det_x_basis)


        # set n_samples to normal
        self.config.n_samples = copy_old_n_samples
        print(f"type p_indep_x_states: {type(p_indep_x_states)}")
        print(f"len_ind_has_one_0_and_every_second_symbol: {len_ind_has_one_0_and_every_second_symbol}")
        print(f"len_ind_every_second_symbol: {len_ind_every_second_symbol}")
        print(f"Returning values: {p_indep_x_states}, {len_ind_has_one_0_and_every_second_symbol}, {len_ind_every_second_symbol}")
        
        return p_indep_x_states, len_ind_has_one_0_and_every_second_symbol, len_ind_every_second_symbol
   
 