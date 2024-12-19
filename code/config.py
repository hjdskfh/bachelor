class SimulationConfig:
    def __init__(self, data, n_samples=10000, n_pulses=4, mean_voltage=1.0, mean_current=0.08, current_amplitude = 0.02,
                 p_z_alice=0.5, p_z_1=0.5, p_decoy=0.1, p_z_bob = 0.5, sampling_rate_FPGA=6.75e9, bandwidth = 4e9, jitter=1e-11, non_signal_voltage = -1, voltage_decoy=0,
                 voltage=0, voltage_decoy_sup=0, voltage_sup=0, 
                 mean_photon_nr=0.7, mean_photon_decoy=0.1, 
                 fiber_attenuation = -3, detector_efficiency = 0.3, dark_count_frequency = 100, detection_time = 1e-9, detector_jitter = 100e-12, last_time_photon = None,
                 mlp = None
                 ):
        # Input data
        self.data = data

        # General simulation parameters
        self.n_samples = n_samples
        self.n_pulses = n_pulses

        # Voltage and amplitude settings
        self.mean_voltage = mean_voltage  # Voltage (in V)
        self.mean_current = mean_current  # current (in A)
        self.current_amplitude = current_amplitude  # in A

        # Probability parameters
        self.p_z_alice = p_z_alice
        self.p_z_1 = p_z_1
        self.p_decoy = p_decoy
        self.p_z_bob = p_z_bob

        # Sampling and frequency
        self.sampling_rate_FPGA= sampling_rate_FPGA # FPGA frequency (Hz)

        # Bandwidth & Timing parameters 
        self.bandwidth = bandwidth
        self.jitter = jitter  # Timing jitter (s)

        # Voltage configurations
        self.non_signal_voltage = non_signal_voltage
        self.voltage_decoy = voltage_decoy #Z-basis decoy state
        self.voltage = voltage #Z-Basis non decoy state
        self.voltage_decoy_sup = voltage_decoy_sup  # Superposition voltage with decoy state
        self.voltage_sup = voltage_sup  # Superposition voltage without decoy state

        # Photon number settings
        self.mean_photon_nr = mean_photon_nr
        self.mean_photon_decoy = mean_photon_decoy

        # attenuation in fiber in dB
        fiber_attenuation = fiber_attenuation

        # detector
        self.detector_efficiency = detector_efficiency
        self.dark_count_frequency = dark_count_frequency #Hz
        self.detection_time = detection_time #s
        self.detector_jitter = detector_jitter #s
        self.last_time_photon = None

        # mlp-style
        self.mlp = mlp

    def to_dict(self):
        """Convert instance parameters to dictionary, excluding non-serializable objects."""
        result = vars(self).copy()
        
        # Remove non-serializable attributes like 'data_manager'
        if 'data' in result:
            del result['data']
        
        return result
        