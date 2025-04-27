import numpy as np

class SimulationConfig:
    def __init__(self, data, round = 1, seed = None, n_samples=20000, n_pulses=4, batchsize = 1000, mean_voltage=-1.755, mean_current=0.082111, voltage_amplitude=0.0011, current_amplitude = 0.00041,
                 p_z_alice=0.5, p_decoy=0.5, p_z_bob = 0.5, sampling_rate_FPGA=6.5e9, bandwidth = 4e9, jitter=0, non_signal_voltage = -1.3, voltage_decoy=0.2,
                 voltage=0.2, voltage_decoy_sup=0.2, voltage_sup=0.2, 
                 mean_photon_nr=0.182, mean_photon_decoy=0.1, 
                 fiber_attenuation = -3, detector_efficiency = 0.3, dark_count_frequency = 1000, detection_time = 25 * 1e-9, detector_jitter = 5e-12, 
                 mlp = "C:\\Users\\leavi\\bachelor\\code\\Presentation_style_1_adjusted_no_grid.mplstyle", 
                 script_name = None, job_id = None
                 ):
        # Input data
        self.data = data 
        self.round = round # count rounds where all setting are the same and eg p_indep_x is not needed

        # random generator / seed
        self.seed = int(seed) if seed is not None else int(np.random.default_rng().integers(1, 1e6))
        self.rng = np.random.default_rng(self.seed)

        # General simulation parameters
        self.n_samples = n_samples
        self.n_pulses = n_pulses
        self.batchsize = batchsize

        # Voltage and amplitude settings
        self.mean_voltage = mean_voltage  # Voltage (in V)
        self.mean_current = mean_current  # current (in A)
        self.voltage_amplitude = voltage_amplitude # in V
        self.current_amplitude = current_amplitude  # in A

        # Probability parameters
        self.p_z_alice = p_z_alice
        self.p_decoy = p_decoy
        self.p_z_bob = p_z_bob

        # Sampling and frequency
        self.sampling_rate_FPGA= sampling_rate_FPGA # FPGA frequency (Hz)

        # Bandwidth & Timing parameters 
        self.bandwidth = bandwidth
        self.jitter = jitter  # Timing jitter in general (s)

        # Voltage configurations: all voltages should be the same and non signal one lower, here 0 and -1 bc of eam transmission curve
        self.non_signal_voltage = non_signal_voltage
        self.voltage_decoy = voltage_decoy #Z-basis decoy state
        self.voltage = voltage #Z-Basis non decoy state
        self.voltage_decoy_sup = voltage_decoy_sup  # Superposition voltage with decoy state
        self.voltage_sup = voltage_sup  # Superposition voltage without decoy state

        # Photon number settings
        self.mean_photon_nr = mean_photon_nr
        self.mean_photon_decoy = mean_photon_decoy

        # attenuation in fiber in dB
        self.fiber_attenuation = fiber_attenuation

        # detector
        self.detector_efficiency = detector_efficiency
        self.dark_count_frequency = dark_count_frequency #Hz
        self.detection_time = detection_time #s
        self.detector_jitter = detector_jitter #s

        # mlp-style
        self.mlp = mlp
        self.script_name = script_name
        self.job_id = job_id

    def to_dict(self):
        """Convert instance parameters to dictionary, excluding non-serializable objects."""
        result = vars(self).copy()
        
        # Remove non-serializable attributes like 'data_manager'
        if 'data' in result:
            del result['data']
        
        return result
    
    def validate_parameters(self):
        """Check if the configuration parameters are within valid ranges."""
        # Example validation checks:
        if self.n_samples % self.batchsize != 0 or self.batchsize > self.n_samples:
            raise ValueError("n_samples must be divisible by batchsize and batchsize must be smaller than n_samples.")
        if self.n_samples <= 0:
            print("Error: n_samples must be positive.")
        if self.n_pulses <= 0 or self.n_pulses % 2 != 0:
            print("Error: n_pulses must be positive and even.")
        if self.current_amplitude < 0:
            print("Error: current_amplitude must be positive.")
        if not (0 <= self.p_z_alice <= 1):
            print("Error: p_z_alice must be between 0 and 1.")
        if not (0 < self.p_z_bob <= 1):
            print("Error: p_z_bob must be between 0 and 1.")
        if self.sampling_rate_FPGA <= 0:
            print("Error: sampling_rate_FPGA must be positive.")
        if self.bandwidth <= 0:
            print("Error: bandwidth must be positive.")
        if self.jitter > 1/self.sampling_rate_FPGA:
            print("Error: jitter must be less than the inverse of the sampling rate.")
        if (self.non_signal_voltage - self.voltage) != -1:
            print("Error: The difference between non_signal_voltage and voltage is not 1")
        if self.voltage != self.voltage_decoy or self.voltage != self.voltage_sup:
            print("Error: The starting voltage values are not the same") 
        if self.mean_photon_nr < 0 or self.mean_photon_decoy < 0 or self.mean_photon_nr < self.mean_photon_decoy or self.mean_photon_nr > 1:
            print("Error: mean_photon_nr and mean_photon_decoy must be between 0 and 1 and mean_photon_nr must be larger than mean_photon_decoy.")
        if self.fiber_attenuation > 0:
            print("Error: fiber_attenuation cannot be positive.")
        if self.detector_efficiency < 0 or self.detector_efficiency > 1:
            print("Error: detector_efficiency must be between 0 and 1.")
        if self.dark_count_frequency < 0:
            print("Error: dark_count_frequency cannot be negative.")
        if self.detection_time < 0:
            print("Error: detection_time must be non negative.")
        if self.detector_jitter < 0:
            print("Error: detector_jitter cannot be negative.")
       
        