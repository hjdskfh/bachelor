def delay_line_interferometer_old(self, power_dampened, t, peak_wavelength, value):
        dt_new = 1e-14

        tau = 2 / self.config.sampling_rate_FPGA  
        n_g = 2.05 # For calculatting path length difference
        # n_g = 1
        # Assuming the group refractive index of the waveguide
        n_eff = 1.54 # Effective refractive index
        # print(f"c:{constants.c}")
        delta_L = tau * constants.c / n_g
        # print(f"delta_L: {delta_L:.6f} m")
        # print(f"tau: {tau} s")
        

        for i in range(0, len(value), self.config.batchsize):

            power_dampened_batch = power_dampened[i:i + self.config.batchsize, :]
            # print(f"powerbatch size: {len(power_dampened_batch)}")
            flattened_power_batch = power_dampened_batch.reshape(-1)
            # print(f"flattened_power_batch: {flattened_power_batch.shape}, power_dampened_batch: {power_dampened_batch.shape}, t: {t.shape}")
            flattened_power_batch_copy = flattened_power_batch.copy()

            dt_original = t[1] - t[0] # t[-1]-t[0] / len(t) # t[-1] - t[0] # 1 
            num_points = flattened_power_batch.size 
            # print(f"num_points:{num_points}")
            np.set_printoptions(threshold=100)

            t_original_all_sym = np.arange(t[0], t[0] + (num_points) * dt_original, dt_original)     
            # print(f"t_original_all_sym: {t_original_all_sym.shape}, t: {t.shape}, step t_original: {t_original_all_sym[1]-t_original_all_sym[0]}, t step:{t[1]-t[0]}, t_original_all_sym[-1]: {t_original_all_sym[-1]}")       
            # print(f"t_original_all_sym[:-100]: {t_original_all_sym[-10:]}")

            # calculate new time axis
            t_new_all_sym = np.arange(t_original_all_sym[0], t_original_all_sym[-1] + dt_new, dt_new)
            # print(f"t_new_all_sym: {t_new_all_sym[-10:]}, t_new_all_sym.shape: {t_new_all_sym.shape}, t_new_all_sym[-1]: {t_new_all_sym[-1]}")
            # hinzugefügt + dt_new*self.config.batchsize

            # Resample the data using np.interp (linear interpolation)
            flattened_power_batch_resampled = np.interp(t_new_all_sym, t_original_all_sym, flattened_power_batch)
        
            flattened_power_batch_resampled_copy = flattened_power_batch_resampled.copy()
            # plt.plot(flattened_power_batch_resampled_copy[:len(t)*100], color = 'green', label = 'after resample')
            # plt.show()
            # t_test ist gleich wie das t das in der Funktion gemacht wird
            t_test = np.arange(len(flattened_power_batch_resampled)) * dt_new
            # print(f"t_test:{t_test.shape}, t_test:{t_test[-1]}, t_new_all_sym:{t_new_all_sym.shape} t_new_all_sym: {t_new_all_sym[-1]}")
            
            f_0 = constants.c / peak_wavelength[i]
            
            n_eff_calculated = self.get_interpolated_value(peak_wavelength[i] * 1e9, 'wavelength_neff')
            print(f"n_eff_calculated: {n_eff_calculated} normal {n_eff}, peak_wavelength[i]: {peak_wavelength[i]}, f_0: {f_0}, i: {i}, batchsize: {self.config.batchsize}")
            
            # upsampled input power!
            # timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            # script_dir = Path(__file__).parent  # The folder where the script is located
        
            # # Navigate to the parent directory (next to the code) and then to the 'results' folder
            # logs_dir = script_dir.parent / 'results'  # Go one level up and look for 'results'
            
            # # Create the 'logs' directory if it doesn't exist
            # logs_dir.mkdir(exist_ok=True)

            # # Define the file path
            # filepath = os.path.join(logs_dir, f"output_{timestamp}_n_samples_{self.config.n_samples}.txt")

            # with open(filepath, "w") as f:
            #     with np.printoptions(threshold=np.inf):
            #         f.write(f" {t}\n")
            #         f.write(f"{flattened_power_batch_resampled_copy}\n")
            
            Saver.save_array_as_npz("t_and_input_power_DLI_mixed_pulses",
                                    t_original = t_original_all_sym,
                                    t_upsampled = t_new_all_sym,
                                    input_power_original = flattened_power_batch_copy,
                                    input_power_upsampled = flattened_power_batch_resampled_copy)

            # print(f"len(flattened_power_batch): {len(flattened_power_batch_resampled)}")
            power_1, power_2, _ = self.simulation_helper.DLI(flattened_power_batch_resampled, dt_new, tau, delta_L, f_0,  n_eff)
            # plt.plot(power_1[:len(t)*100], label = 'after DLI 1')
            # plt.plot(power_2[:len(t)*100], label = 'after DLI 2')
            # plt.legend()
            # plt.show()
            # Use interpolation to compute signal values at new time points
            signal_downsampled = np.interp(t_original_all_sym, t_new_all_sym, power_1)
            # plt.plot(signal_downsampled)
            # plt.show()
            flattened_power_batch = signal_downsampled.reshape(self.config.batchsize, len(t))
            power_dampened[i:i + self.config.batchsize, :] = flattened_power_batch
        return power_dampened, f_0
