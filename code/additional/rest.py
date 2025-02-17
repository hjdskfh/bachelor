'''num_rows, num_cols = time_photons_det.shape

        # Get indices of non-NaN values for each row
        valid_indices = ~np.isnan(time_photons_det)
        
        # Create an array to store adjusted last valid times
        last_valid_time = np.full((num_rows, num_cols), np.nan)  # Start with NaNs

        for row in range(num_rows):
            # Get valid photon times and their indices
            valid_times = time_photons_det[row, valid_indices[row]]
            valid_idx = np.where(valid_indices[row])[0]  # Get original column indices
            
            if len(valid_times) == 0: # Should not be empty
                continue  # Skip empty rows
            
            # Compute adjusted last valid time for each valid detection
            last_valid_time_row = np.full_like(valid_times, np.nan)
            last_valid_time_row[0] = 0  # The first photon always passes

            for i in range(1, len(valid_times)):
                symbol_diff = (valid_idx[i] - valid_idx[i-1]) * pulse_length * self.config.n_pulses
                last_valid_time_row[i] = symbol_diff + valid_times[i-1]

            # Apply the filtering condition
            invalid_mask = (valid_times - last_valid_time_row) < self.config.detection_time
            
            # Assign NaN to invalid detections
            time_photons_det[row, valid_idx[invalid_mask]] = np.nan
            wavelength_photons_det[row, valid_idx[invalid_mask]] = np.nan'''
        '''# Initialize the adjusted times array with NaN values
        for i, time_p in enumerate(time_photons_det):
            # Initialize a list to store valid photons for this row
            last_valid_time = 0  # Start with time 0 to allow the first photon

            for j, time in enumerate(time_p):
                if np.isnan(time):
                    # If it's NaN, skip it (no photon detected)
                    continue
                else:
                    # If it's a valid photon, check if it's after the dead time
                    if time - last_valid_time < self.config.detection_time:
                        # If it's to close to the last valid time, skip it
                        time_photons_det[i, j] = np.nan  # Set the invalid photon to NaN
                        wavelength_photons_det[i, j] = np.nan  # Set the invalid wavelength to NaN
                    else:
                        last_valid_time = time - pulse_length*self.config.n_pulses  # Update the last valid time'''