from codecs import lookup
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from tqdm import tqdm
import time
import inspect
import gc
import random

from saver import Saver
from simulationengine import SimulationEngine
from simulationsingle import SimulationSingle
from simulationhelper import SimulationHelper
from plotter import Plotter


class SimulationManager:
    def __init__(self, config):
        self.config = config 
        self.simulation_engine = SimulationEngine(config)
        self.simulation_single = SimulationSingle(config)
        self.simulation_helper = SimulationHelper(config)
        self.plotter = Plotter(config)

    def run_simulation_states(self):
        T1_dampening = self.simulation_engine.initialize()        
        
        states = [
                {"title": "State: Z0", "basis": 1, "value": 1, "decoy": 0},
                {"title": "State: Z1", "basis": 1, "value": 0, "decoy": 0},
                {"title": "State: X+", "basis": 0, "value": -1, "decoy": 0},
                {"title": "State: Z0 decoy", "basis": 1, "value": 1, "decoy": 1},
                {"title": "State: Z1 decoy", "basis": 1, "value": 0, "decoy": 1},
                {"title": "State: X+ decoy", "basis": 0, "value": -1, "decoy": 1},
                ]
        
        # Define the states and their corresponding arguments
        other_states = [
            {"title": "State: Z0", "basis": np.array([1]), "value": np.array([1]), "decoy": np.array([0])},
            {"title": "State: Z1", "basis": np.array([1]), "value": np.array([0]), "decoy": np.array([0])},
            {"title": "State: X+", "basis": np.array([0]), "value": np.array([-1]), "decoy": np.array([0])},
            {"title": "State: Z0 decoy", "basis": np.array([1]), "value": np.array([1]), "decoy": np.array([1])},
            {"title": "State: Z1 decoy", "basis": np.array([1]), "value": np.array([0]), "decoy": np.array([1])},
            {"title": "State: X+ decoy", "basis": np.array([0]), "value": np.array([-1]), "decoy": np.array([1])},
        ]
    
        def process_state(state):
            optical_power, peak_wavelength, chosen_voltage, chosen_current = self.simulation_engine.random_laser_output('current_power', 'voltage_shift', fixed = True)

            basis, value, decoy = self.simulation_engine.generate_alice_choices(basis=state["basis"], value=state["value"], decoy=state["decoy"])

            signals, t, square_signals = self.simulation_engine.signal_bandwidth_jitter(basis, value, decoy)
            power_dampened, norm_transmission, calc_mean_photon_nr, energy_per_pulse = self.simulation_engine.eam_transmission(signals, optical_power, T1_dampening, peak_wavelength, t)
            
            return state, t, signals, power_dampened, square_signals


        # Plotting results sequentially to avoid threading issues with Matplotlib
        for state in states:
            state, t, signals, power_dampened, square_signals = process_state(state)
            chosen_index = 10
            fig, ax = plt.subplots(figsize=(8, 5))

            # Plot voltage (left y-axis)
            ax.plot(t * 1e9, signals[chosen_index], color='blue', label='Voltage Signal with Bandwidth')
            ax.set_ylabel('Voltage (V)')

            # Plot square signals (right y-axis)
            ax.plot(t * 1e9, square_signals[chosen_index], color='green', label='Square Voltage Signal')
            ax.set_ylabel('Voltage (V)')

            # Titles and labels
            ax.set_title(state["title"])
            ax.set_xlabel('Time (ns)')
            ax.legend()

            # Save or show the plot
            Saver.save_plot(f"{state['title'].replace(' ', '_').replace(':', '').lower()}_voltage_and_square")

            fig, ax = plt.subplots(figsize=(8, 5)) #8, 5
            ax2 = ax.twinx()  # Create a second y-axis

            # Plot voltage (left y-axis)
            ax.plot(t * 1e9, signals[chosen_index], color='blue', label='Signal incl. BW')
            ax.set_ylabel('Voltage (V)', color='blue')
            ax.tick_params(axis='y', labelcolor='blue')

            # Plot square signals (right y-axis)
            ax.plot(t * 1e9, square_signals[chosen_index], color='lightblue', label='Square Signal')
            ax.set_ylabel('Voltage (V)', color='blue')
            ax.tick_params(axis='y', labelcolor='blue')

            # Plot transmission (right y-axis)
            ax2.plot(t * 1e9, power_dampened[chosen_index], color='red', label='Laser Pulse')
            ax2.set_ylabel('Transmission', color='red')
            ax2.tick_params(axis='y', labelcolor='red')

            # Titles and labels
            ax.set_title(state["title"])
            ax.set_xlabel('Time (ns)')

            # Save or show the plot
            lines, labels = ax.get_legend_handles_labels()  # Get handles and labels from the first axis
            lines2, labels2 = ax2.get_legend_handles_labels()  # Get handles and labels from the second axis
            if state["basis"] == 0:
                ax.legend(lines + lines2, labels + labels2, loc='upper center')  # Combine and add the legend
            else:
                ax.legend(lines + lines2, labels + labels2)
            Saver.save_plot(f"{state['title'].replace(' ', '_').replace(':', '').lower()}_voltage_square_transmission")

    def run_simulation_histograms(self):
        #initialize
        T1_dampening = self.simulation_engine.initialize()

        # Define the states and their corresponding arguments
        states = [
                {"title": "State: Z0", "basis": 1, "value": 1, "decoy": 0},
                {"title": "State: Z1", "basis": 1, "value": 0, "decoy": 0},
                {"title": "State: X+", "basis": 0, "value": -1, "decoy": 0},
                {"title": "State: Z0 decoy", "basis": 1, "value": 1, "decoy": 1},
                {"title": "State: Z1 decoy", "basis": 1, "value": 0, "decoy": 1},
                {"title": "State: X+ decoy", "basis": 0, "value": -1, "decoy": 1},
                ]
        

        for state in states:
            # 0tes Element ist baseline
            if state["decoy"] == 0:
                target_mean_photon_nr = self.config.mean_photon_nr
            else:
                target_mean_photon_nr = self.config.mean_photon_decoy

            optical_power, peak_wavelength, chosen_voltage, chosen_current = self.simulation_engine.random_laser_output('current_power', 'voltage_shift')
            
            # Generate Alice's choices
            basis, value, decoy = self.simulation_engine.generate_alice_choices(basis=state["basis"], value=state["value"], decoy=state["decoy"])
            
            # Simulate signal and transmission

            signals, t, jitter_shifts = self.simulation_engine.signal_bandwidth_jitter(basis, value, decoy)
            Saver.memory_usage("before eam " + str(time.time()))
            power_dampened, transmission, calc_mean_photon_nr, energy_per_pulse = self.simulation_engine.eam_transmission(signals, optical_power, T1_dampening, peak_wavelength, t)
            Saver.memory_usage("before fiber " + str(time.time()))
            power_dampened = self.simulation_engine.fiber_attenuation(power_dampened)
            Saver.memory_usage("before basis " + str(time.time()))
            power_dampened_x, power_dampened_z = self.simulation_engine.basis_selection_bob(power_dampened)
            Saver.memory_usage("before DLI " + str(time.time()))

            # path for X basis
            power_dampened_x = self.simulation_engine.delay_line_interferometer(power_dampened_x, t, peak_wavelength)
            Saver.memory_usage("before choose x " + str(time.time()))
            wavelength_photons_x, time_photons_x, nr_photons_x, index_where_photons_x, all_time_max_nr_photons_x, sum_nr_photons_at_chosen_x = self.simulation_engine.choose_photons(power_dampened_x, 
                                                                                                                                                                                                            transmission, t, 
                                                                                                                                                                                                            peak_wavelength, 
                                                                                                                                                                                                            calc_mean_photon_nr, 
                                                                                                                                                                                                            energy_per_pulse, 
                                                                                                                                                                                                            fixed_nr_photons=None)
            time_photons_det_x, wavelength_photons_det_x, nr_photons_det_x, index_where_photons_det_x = self.simulation_engine.detector(t, wavelength_photons_x, 
                                                                                                                                time_photons_x, nr_photons_x, 
                                                                                                                                index_where_photons_x, 
                                                                                                                                all_time_max_nr_photons_x)
            # path fo Z basis
            wavelength_photons_z, time_photons_z, nr_photons_z, index_where_photons_z, all_time_max_nr_photons_z, sum_nr_photons_at_chosen_z = self.simulation_engine.choose_photons(power_dampened_z, 
                                                                                                                                                                                              transmission, t, 
                                                                                                                                                                                              peak_wavelength, 
                                                                                                                                                                                              calc_mean_photon_nr, 
                                                                                                                                                                                              energy_per_pulse, 
                                                                                                                                                                                              fixed_nr_photons=None)
            time_photons_det_z, wavelength_photons_det_z, nr_photons_det_z, index_where_photons_det_z = self.simulation_engine.detector(t, wavelength_photons_z, 
                                                                                                                                time_photons_z, nr_photons_z, 
                                                                                                                                index_where_photons_z, 
                                                                                                                                all_time_max_nr_photons_z)
            #print(f"print time photon: {time_photons_det_x[:10]}")	


            calc_mean_photon_nr = self.make_data_plottable(calc_mean_photon_nr)
            nr_photons_det_z = self.make_data_plottable(nr_photons_det_z)
            time_photons_det_z = self.make_data_plottable(time_photons_det_z)
            wavelength_photons_det_z = self.make_data_plottable(wavelength_photons_det_z)
            nr_photons_det_x = self.make_data_plottable(nr_photons_det_x)
            time_photons_det_x = self.make_data_plottable(time_photons_det_x)
            wavelength_photons_det_x = self.make_data_plottable(wavelength_photons_det_x)
            
            plt.hist(calc_mean_photon_nr, color = 'green', bins=40, alpha=0.7, label="Mean Photon Number")
            plt.axvline(target_mean_photon_nr, color='darkred', linestyle='--', linewidth=2, label='target mean photon number')
            plt.title(f"{state['title'].replace(':', '').lower()}: mean photon number over {self.config.n_samples} iterations")
            plt.ylabel('iterations')
            plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))
            plt.xlabel(r'$\Delta \langle \mu \rangle$')
            plt.legend()
            plt.tight_layout()
            Saver.save_plot(f"hist_mean_photon_nr_{state['title'].replace(' ', '_').replace(':', '').lower()}")       

            plt.hist(nr_photons_det_x, label = 'X basis', bins=np.arange(0, 11) - 0.5, alpha=0.7)
            plt.hist(nr_photons_det_z, label = 'Z basis', bins=np.arange(0, 11) - 0.5, alpha=0.7)
            plt.title(f"{state['title'].replace(':', '').lower()}: photon number over {self.config.n_samples} iterations")
            plt.xticks(np.arange(0, 11))  # Set x-ticks to be integers 
            plt.ylabel('iterations')
            plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))
            plt.xlabel('photon number')
            plt.legend()
            plt.tight_layout()
            Saver.save_plot(f"hist_nr_photons_{state['title'].replace(' ', '_').replace(':', '').lower()}")   

            plt.hist(time_photons_det_x * 1e9, label = 'X basis', bins=40, alpha=0.7)
            plt.hist(time_photons_det_z * 1e9, label = 'Z basis', bins=40, alpha=0.7)
            plt.title(f"{state['title'].replace(':', '').lower()}: photon time over {self.config.n_samples} iterations")
            plt.ylabel('iterations')
            plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))
            plt.xlabel('photon time (ns)')
            plt.legend()
            plt.tight_layout()
            Saver.save_plot(f"hist_photon_time_{state['title'].replace(' ', '_').replace(':', '').lower()}")
        
            plt.hist(wavelength_photons_det_x * 1e9, label = 'X basis', bins=40, alpha=0.7)
            plt.hist(wavelength_photons_det_z * 1e9, label = 'Z basis', bins=40, alpha=0.7)
            plt.title(f"{state['title'].replace(':', '').lower()}: photon wavelength over {self.config.n_samples} iterations")
            plt.ylabel('iterations')
            plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))
            plt.xlabel('photon wavelength (nm)')
            plt.legend()
            plt.tight_layout()
            Saver.save_plot(f"hist_wavelength_{state['title'].replace(' ', '_').replace(':', '').lower()}")         
        
    def run_simulation_parameter_sweep_amplitude(self):
        #9.12. Parameter sweep von amplitudenschwankung 0,5 mA bis 5 mA für Z0 und Z0 decoy gebe spread in mean photon number wieder

        #initialize
        T1_dampening = self.simulation_engine.initialize()

        # Define the states and their corresponding arguments
        states = [
                {"title": "State: Z0", "basis": 1, "value": 1, "decoy": 0},
                {"title": "State: Z0 decoy", "basis": 1, "value": 1, "decoy": 1},
                ]
        
        parameters_amplitude =  np.linspace(0.0005, 0.005, 20)
        differences_mean_photon_nr = np.empty((len(states), len(parameters_amplitude)))
        
        for index, state in tqdm(enumerate(states), desc= "running simulation for different states", unit=" states", position=0):
            mean_photon_nr_min = 1000
            mean_photon_nr_max = 0
           
            for idx_param, param in tqdm(enumerate(parameters_amplitude), desc="parameters", unit="parameters", leave = False, position=1):
                self.config.current_amplitude = param
            
                mean_of_mean_photon = np.empty(self.config.n_samples)

                for i in range(self.config.n_samples):
                    optical_power, peak_wavelength, chosen_voltage, chosen_current = self.simulation_engine.random_laser_output('current_power', 'voltage_shift')
                    
                    # Generate Alice's choices
                    basis, value, decoy = self.simulation_engine.generate_alice_choices(basis=state["basis"], value=state["value"], decoy=state["decoy"], fixed = True)
                    
                    # Simulate signal and transmission
                    signals, t, jitter_shifts = self.simulation_engine.signal_bandwidth_jitter(basis, value, decoy)
                    power_dampened, transmission = self.simulation_engine.eam_transmission(signals, optical_power, T1_dampening)
                    power_dampened = self.simulation_engine.fiber_attenuation(power_dampened)

                    calc_mean_photon_nr, wavelength_photons, time_photons, nr_photons, index_where_photons, all_time_max_nr_photons, sum_nr_photons_at_chosen = self.simulation_engine.choose_photons(power_dampened, transmission, 
                                                                                                                                                                                t_jitter, peak_wavelength)
            
                    mean_of_mean_photon[i] = calc_mean_photon_nr

                    if mean_photon_nr_min > calc_mean_photon_nr:
                        mean_photon_nr_min = calc_mean_photon_nr
                    if mean_photon_nr_max < calc_mean_photon_nr:
                        mean_photon_nr_max = calc_mean_photon_nr

                differences_mean_photon_nr[index][idx_param] = (mean_photon_nr_max-mean_photon_nr_min) / np.mean(mean_of_mean_photon)

            plt.plot(parameters_amplitude*1e3, differences_mean_photon_nr[index], label= 'for ' + str(state['title'].replace(':', '').lower()))
            
            '''plt.title(f" spread of mean photon number for {state['title'].replace(':', '').lower()} over {self.config.n_samples} iterations")                    
            plt.xlabel(r'$\Delta I \, (\mathrm{mA})$')  # ΔI (mA)
            plt.ylabel(r'$\frac{\Delta \langle \mu \rangle}{\langle \mu \rangle}$')  # Δ⟨μ⟩
            plt.tight_layout()
            plt.legend()
            all_titles = state['title'].replace(' ', '_').replace(':', '').lower()
            save_plot(f"spread_photon_nr_{all_titles}")'''

        plt.title(f" spread of mean photon number over {self.config.n_samples} iterations")
        plt.xlabel(r'$\Delta I \, (\mathrm{mA})$')  # ΔI (mA)
        plt.ylabel(r'$\frac{\Delta \langle \mu \rangle}{\langle \mu \rangle}$')  # Δ⟨μ⟩ / ⟨μ⟩
        plt.tight_layout()
        plt.legend()    
        all_titles = "_".join([state['title'].replace(' ', '_').replace(':', '').lower() for state in states])
        Saver.save_plot(f"spread_photon_nr_{all_titles}_for_4GHz_and_1e-11_jitter")
    
    def run_simulation_after_detector(self):
        T1_dampening = self.simulation_engine.initialize()
        time_in_simulation = 0 
  
        optical_power, peak_wavelength, chosen_voltage, chosen_current = self.simulation_engine.random_laser_output('current_power', 'voltage_shift')
        
        # Generate Alice's choices
        basis, value, decoy = self.simulation_engine.generate_alice_choices()
        Saver.memory_usage("before signal bandwidth jitter")
        # Simulate signal and transmission
        signals, t, _ = self.simulation_engine.signal_bandwidth_jitter(basis, value, decoy)
        Saver.memory_usage("After signal bandwidth jitter")
        power_dampened, transmission, calc_mean_photon_nr, energy_per_pulse = self.simulation_engine.eam_transmission(signals, optical_power, T1_dampening, peak_wavelength, t)
        Saver.memory_usage("After eam transmission")
        power_dampened = self.simulation_engine.fiber_attenuation(power_dampened)
        Saver.memory_usage("After fiber attenuation")

        plt.plot(t * 1e9, power_dampened[0], color='blue', label='0', linestyle='-', marker='o', markersize=1)
        plt.plot(t * 1e9, power_dampened[1], color='green', label='1', linestyle='-', marker='o', markersize=1)
        Saver.save_plot(f"power_fiber")
        
        power_dampened = self.simulation_engine.delay_line_interferometer(power_dampened, t, peak_wavelength)
        Saver.memory_usage("After DLI")
        calc_mean_photon_nr, wavelength_photons, time_photons, nr_photons, index_where_photons, all_time_max_nr_photons, sum_nr_photons_at_chosen = self.simulation_engine.choose_photons(transmission, t, 
                                                                                                                                                        calc_mean_photon_nr, 
                                                                                                                                                        energy_per_pulse, 
                                                                                                                                                        fixed_nr_photons=None)
        Saver.memory_usage("After choose photons")
        time_photons_det, wavelength_photons_det, nr_photons_det, index_where_photons_det = self.simulation_engine.detector(t, wavelength_photons, time_photons, 
                                                                                                                     nr_photons, index_where_photons, all_time_max_nr_photons)
        print(f"shape time_photons_det: {time_photons_det.shape}")
        Saver.memory_usage("After detector")
        dark_count_times, num_dark_counts = self.simulation_engine.darkcount()


        # Labels for the x-axis
        photon_labels = [f"{i}" for i in range(all_time_max_nr_photons + 1)]

        # Plotting
        x = np.arange(len(photon_labels))  # Positions for the bars
        bar_width = 0.35

        # Count occurrences for photons in bins
        fiber_counts = np.bincount(nr_photons, minlength=len(photon_labels))  # Counts of 0, 1, 2, 3 photons in fiber
        detector_counts = np.bincount(nr_photons_det, minlength=len(photon_labels))  # Counts of 0, 1, 2, 3 photons in detector

        # Calculate the number of "no photon" bins
        fiber_no_photon_bins = sum_nr_photons_at_chosen - len(nr_photons)  # For fiber
        detector_no_photon_bins = sum_nr_photons_at_chosen - len(nr_photons_det)  # For detector

        # Add the "no photon" counts to the 0-photon category
        fiber_counts[0] += fiber_no_photon_bins
        detector_counts[0] += detector_no_photon_bins

        plt.bar(x - bar_width/2, fiber_counts, width=bar_width, label="Fiber", color="teal")
        plt.bar(x + bar_width/2, detector_counts, width=bar_width, label="Detector", color="darkred")

        # Add labels, title, and legend
        plt.xlabel("Number of Photons")
        plt.ylabel("Count")
        plt.title("Photon Count Distribution in Fiber and Detector")
        plt.text(
            s="for "  + str(self.config.n_samples) + " symbols with\n"+ str(sum_nr_photons_at_chosen) + " Photons at fiber\n"+ str(nr_photons_det.sum()) + " Photons at detector",
            x=0.79,  # Position the text closer to the right
            y=0.65,   # Position the text in the vertical middle
            transform=plt.gca().transAxes,
            ha='center',
            va='center',
            fontsize=12,
            bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5')  # Box around the text
        )
        plt.xticks(x, photon_labels)  # Set x-axis tick labels
        plt.legend()
        plt.tight_layout()

        # Show the plot
        Saver.save_plot(f"photons_in_fiber_vs_after detector")

    def run_simulation_initialize(self):
        start_time = time.time()  # Record start time

        T1_dampening = self.simulation_engine.initialize()
        print(f"T1 dampening: {T1_dampening}")
        optical_power, peak_wavelength, chosen_voltage, chosen_current = self.simulation_engine.random_laser_output('current_power', 'voltage_shift')
    
        # Generate Alice's choices
        basis, value, decoy = self.simulation_engine.generate_alice_choices()

        # Simulate signal and transmission
        Saver.memory_usage("before simulating signal: " + str(time.time() - start_time))
        signals, t, _ = self.simulation_engine.signal_bandwidth_jitter(basis, value, decoy)

        amount_symbols_in_plot = 3
        pulse_duration = 1 / self.config.sampling_rate_FPGA
        sampling_rate_fft = 100e11
        samples_per_pulse = int(pulse_duration * sampling_rate_fft)
        total_samples = self.config.n_pulses * samples_per_pulse
        t_plot1 = np.linspace(0, amount_symbols_in_plot * self.config.n_pulses * pulse_duration, amount_symbols_in_plot * total_samples, endpoint=False)
        signals_part = signals[:amount_symbols_in_plot]
        flattened_signals = signals_part.reshape(-1)
        plt.plot(t_plot1 * 1e9, flattened_signals)
        plt.title(f"Voltage Signal with Bandwidth and Jitter for {amount_symbols_in_plot} symbols")
        plt.ylabel('Volt (V)')
        plt.xlabel('Time (ns)')
        Saver.save_plot(f"signal_after_bandwidth")
    
    def run_simulation_classificator(self, save_output = True):
        
        start_time = time.time()  # Record start time
        T1_dampening = self.simulation_engine.initialize()
        '''if  self.config.p_indep_x_states_non_dec is None or self.config.p_indep_x_states_dec is None:
            self.config.p_indep_x_states_non_dec, len_ind_has_one_0_and_every_second_symbol_non_dec, len_ind_every_second_symbol_non_dec = self.simulation_engine.find_p_indep_states_x_for_classifier(T1_dampening, simulation_length_factor=1000, is_decoy=False)
            self.config.p_indep_x_states_dec, len_ind_has_one_0_and_every_second_symbol_dec, len_ind_every_second_symbol_dec = self.simulation_engine.find_p_indep_states_x_for_classifier(T1_dampening, simulation_length_factor=1000, is_decoy=True)
            # print(f"len_ind_has_one_0_and_every_second_symbol_non_dec: {len_ind_has_one_0_and_every_second_symbol_non_dec}")
            # print(f"len_ind_has_one_0_and_every_second_symbol_dec: {len_ind_has_one_0_and_every_second_symbol_dec}")
            # print(f"len_ind_every_second_symbol_dec: {len_ind_every_second_symbol_dec}")
            # print(f"len_ind_every_second_symbol_non_dec: {len_ind_every_second_symbol_non_dec}")
        
        else:
            len_ind_has_one_0_and_every_second_symbol_non_dec = -999
            len_ind_has_one_0_and_every_second_symbol_dec = -999
            len_ind_every_second_symbol_dec = -999
            len_ind_every_second_symbol_non_dec = -999
        # print(f"p_indep_x_states_non_dec: {self.config.p_indep_x_states_non_dec}")
        # print(f"p_indep_x_states_dec: {self.config.p_indep_x_states_dec}")'''

        optical_power, peak_wavelength, chosen_voltage, chosen_current = self.simulation_engine.random_laser_output('current_power', 'voltage_shift', fixed = True)
        # Create a histogram
        '''plt.hist(peak_wavelength *1e9, bins=10)  # bins=10 is just an example; adjust as needed
        plt.xlabel('Peak Wavelength (nm)')
        plt.ylabel('Frequency')
        plt.title('Histogram of Peak Wavelengths')
        Saver.save_plot(f"hist_peak_wavelength")'''
    
        # Generate Alice's choices
        # basis, value, decoy = self.simulation_engine.generate_alice_choices(basis=np.array([1, 0, 0, 1, 1, 1, 1, 0, 0, 1]), value=np.array([1, -1, -1, 0, 0, 1, 1, -1, -1, 0]), decoy=np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1]))
        basis, value, decoy = self.simulation_engine.generate_alice_choices(basis=np.array([1, 0, 0, 1, 1, 0, 0, 1]), value=np.array([1, -1, -1, 0, 1, -1, -1, 0]), decoy=np.array([0, 0, 0, 0, 1, 1, 1, 1]))
        # basis, value, decoy = self.simulation_engine.generate_alice_choices(basis=np.array([1,0,1]), value=np.array([1,-1, 0]), decoy=np.array([0,0,0]))
        # basis, value, decoy = self.simulation_engine.generate_alice_choices(basis=np.array([0,1]), value=np.array([-1, 0]), decoy=np.array([0,0]))
        # basis, value, decoy = self.simulation_engine.generate_alice_choices()
        print(f"basis: {basis[:10]}")
        print(f"value: {value[:10]}")
        print(f"decoy: {decoy[:10]}")

        # Simulate signal and transmission
        # Saver.memory_usage("before simulating signal: " + str("{:.3f}".format(time.time() - start_time)))
        signals, t, _ = self.simulation_engine.signal_bandwidth_jitter(basis, value, decoy)

        '''amount_symbols_in_plot = 6
        pulse_duration = 1 / self.config.sampling_rate_FPGA
        sampling_rate_fft = 100e11
        samples_per_pulse = int(pulse_duration * sampling_rate_fft)
        total_samples = self.config.n_pulses * samples_per_pulse
        t_plot1 = np.linspace(0, amount_symbols_in_plot * self.config.n_pulses * pulse_duration, amount_symbols_in_plot * total_samples, endpoint=False)
        signals_part = signals[:amount_symbols_in_plot]
        flattened_signals = signals_part.reshape(-1)
        plt.plot(t_plot1 * 1e9, flattened_signals)
        plt.title(f"Voltage Signal with Bandwidth and Jitter for {amount_symbols_in_plot} symbols")
        plt.ylabel('Volt (V)')
        plt.xlabel('Time (ns)')
        Saver.save_plot(f"signal_after_bandwidth_first_symbols")'''

        '''amount_symbols_in_plot = 20
        pulse_duration = 1 / self.config.sampling_rate_FPGA
        sampling_rate_fft = 100e11
        samples_per_pulse = int(pulse_duration * sampling_rate_fft)
        total_samples = self.config.n_pulses * samples_per_pulse
        t_plot1 = np.linspace(0, amount_symbols_in_plot * self.config.n_pulses * pulse_duration, amount_symbols_in_plot * total_samples, endpoint=False)
        signals_part = signals[100:100 + amount_symbols_in_plot]
        flattened_signals = signals_part.reshape(-1)
        plt.plot(t_plot1 * 1e9, flattened_signals)
        plt.title(f"Voltage Signal with Bandwidth and Jitter for {amount_symbols_in_plot} symbols")
        plt.ylabel('Volt (V)')
        plt.xlabel('Time (ns)')
        Saver.save_plot(f"signal_after_bandwidth_100_symbols_into_batch")'''

        time_simulating_signal = time.time() - start_time
        # Saver.memory_usage("before eam: " + str("{:.3f}".format(time_simulating_signal)))
        power_dampened, norm_transmission,  calc_mean_photon_nr_eam, _ = self.simulation_engine.eam_transmission(signals, optical_power, T1_dampening, peak_wavelength, t)

        # plot so I can delete
        # self.plotter.plot_and_delete_mean_photon_histogram(calc_mean_photon_nr_eam, target_mean_photon_nr = np.array([self.config.mean_photon_nr, self.config.mean_photon_decoy]), 
        #                                         type_photon_nr = "Mean Photon Number at EAM")


        # self.plotter.plot_power(t, power_dampened, amount_symbols_in_plot=5, where_plot_1='after EAM')

        time_eam = time.time() - start_time
        # Saver.memory_usage("before fiber: " + str("{:.3f}".format(time_eam)))
        power_dampened = self.simulation_engine.fiber_attenuation(power_dampened)
        
        # self.plotter.plot_power(t, power_dampened, amount_symbols_in_plot=20, where_plot_1='after fiber')

        # first Z basis bc no interference
        # Saver.memory_usage("before detector z: " + str("{:.3f}".format(time.time() - start_time)))
        power_dampened = power_dampened * self.config.p_z_bob
        time_photons_det_z, wavelength_photons_det_z, nr_photons_det_z, index_where_photons_det_z, \
        calc_mean_photon_nr_detector_z, dark_count_times_z, num_dark_counts_z = self.simulation_engine.detector(t, norm_transmission, peak_wavelength, power_dampened, start_time)
        power_dampened = power_dampened / self.config.p_z_bob
        # Saver.memory_usage("before classificator: " + str("{:.3f}".format(time.time() - start_time)))

        # self.plotter.plot_power(t, power_dampened, amount_symbols_in_plot=20, where_plot_1='after Z det')

        # path for X basis
        # Saver.memory_usage("before DLI: " + str("{:.3f}".format(time.time() - start_time)))
        power_dampened = power_dampened * (1 - self.config.p_z_bob)

        # self.plotter.plot_power(t, power_dampened, amount_symbols_in_plot=20, where_plot_1='bob basis X')

        #plot
        amount_symbols_in_first_part = 30
        first_power = power_dampened[:amount_symbols_in_first_part].copy()
        '''print("len(t):", len(t))
        print("first_power.shape:", first_power.shape)
        print("expected shape:", (amount_symbols_in_first_part, len(t)))
        print("power_dampened.shape:", power_dampened.shape)'''

        # DLI
        power_dampened, f_0 = self.simulation_engine.delay_line_interferometer(power_dampened, t, peak_wavelength, value)

        # plot
        assert power_dampened.shape[1] == len(t), f"Mismatch: power_dampened has {power_dampened.shape[1]} samples per symbol, t has {len(t)}!"
        self.plotter.plot_power(t, power_dampened, amount_symbols_in_plot=amount_symbols_in_first_part, where_plot_1='before DLI',  shortened_first_power=first_power, where_plot_2='after DLI,', title_rest='- in fft, mean_volt: ' + str("{:.4f}".format(self.config.mean_voltage)) + ' voltage: ' + str("{:.4f}".format(chosen_voltage[0])) + ' V and ' + str("{:.8f}".format(peak_wavelength[0])))
        
        # plt.plot(first_power.reshape(-1) * 1e3, color='blue', label='0', linestyle='-', marker='o', markersize=1)
        # Saver.save_plot(f"power_before_DLI_in_mW_outside")

        # Saver.memory_usage("before detector x: " + str(time.time() - start_time))
        time_photons_det_x, wavelength_photons_det_x, nr_photons_det_x, index_where_photons_det_x, calc_mean_photon_nr_detector_x, dark_count_times_x, num_dark_counts_x = self.simulation_engine.detector(t, norm_transmission, peak_wavelength, power_dampened, start_time)        
        np.set_printoptions(threshold=np.inf)  # disable truncation
        print(f"calc_mean_photon_nr_detector_x part: {calc_mean_photon_nr_detector_x[:10]}")
        print(f"calc_mean_photon_nr_detector_z part: {calc_mean_photon_nr_detector_z[:10]}")
        print(f"nr_photons: {len(nr_photons_det_x)} {len(nr_photons_det_z)}")
        # plot so I can delete
        # self.plotter.plot_and_delete_mean_photon_histogram(calc_mean_photon_nr_detector_x, target_mean_photon_nr=None, type_photon_nr="Mean Photon Number at Detector X")
        # self.plotter.plot_and_delete_mean_photon_histogram(calc_mean_photon_nr_detector_z, target_mean_photon_nr=None, type_photon_nr="Mean Photon Number at Detector Z")
        # self.plotter.plot_and_delete_photon_wavelength_histogram_two_diagrams(wavelength_photons_det_x, wavelength_photons_det_z)
        # self.plotter.plot_and_delete_photon_nr_histogram(nr_photons_det_x, nr_photons_det_z)
        
        # get results for both detectors
        '''len_wrong_detections_z, len_wrong_detections_x, total_amount_detections, amount_Z_detections, amount_XP_detections, qber, phase_error_rate, raw_key_rate, gain_XP_norm, gain_XP_dec, gain_Z_norm, gain_Z_dec, detected_indices_x_dec, detected_indices_x_norm, detected_indices_z_dec, detected_indices_z_norm = self.simulation_engine.classificator(t, time_photons_det_x, index_where_photons_det_x, 
                                                                                                            time_photons_det_z, index_where_photons_det_z, 
                                                                                                            basis, value, decoy)
        '''
        p_vacuum_z, vacuum_indices_x_long, len_Z_checked_dec, len_Z_checked_non_dec, \
        gain_Z_non_dec, gain_Z_dec, gain_X_non_dec, gain_X_dec, X_P_calc_non_dec, \
        X_P_calc_dec, wrong_detections_z_dec, wrong_detections_z_non_dec, wrong_detections_x_dec, \
        wrong_detections_x_non_dec, qber_z_dec, qber_z_non_dec, qber_x_dec, qber_x_non_dec, raw_key_rate, \
        total_amount_detections = self.simulation_engine.classificator_new(t, time_photons_det_x, index_where_photons_det_x, time_photons_det_z, index_where_photons_det_z, basis, value, decoy)

        Z0_alice_s = np.where((basis == 1) & (value == 1) & (decoy == 1))[0]  # Indices where Z0 was sent
        XP_alice_s = np.where((basis == 0) & (decoy == 1))[0]  # Indices where XP was sent
        Z0_XP_alice_s = XP_alice_s[np.isin(XP_alice_s - 1, Z0_alice_s)]  # Indices where Z1Z0 was sent
        print(f"calc_mean_photon_nr_detector_x bei index XP erste wenn Z0X+: {calc_mean_photon_nr_detector_x[Z0_XP_alice_s][:10]}")
        # plot so I can delete
        # self.plotter.plot_and_delete_photon_time_histogram(time_photons_det_x, time_photons_det_z)

        # Z1_sent_norm, Z1_sent_dec, Z0_sent_norm, Z0_sent_dec, XP_sent_norm, XP_sent_dec = self.simulation_helper.count_alice_choices(basis, value, decoy)

        #readin time
        end_time_read = time.time()  # Record end time  
        execution_time_run = end_time_read - start_time  # Calculate execution time

        len_wrong_z_dec=len(wrong_detections_z_dec)
        len_wrong_z_non_dec=len(wrong_detections_z_non_dec)
        len_wrong_x_dec=len(wrong_detections_x_dec)
        len_wrong_x_non_dec=len(wrong_detections_x_non_dec)

        '''Saver.save_arrays_to_csv('results', 
                                p_vacuum_z=p_vacuum_z,
                                len_vacuum_indices_x_long=len(vacuum_indices_x_long),
                                len_Z_checked_dec=len_Z_checked_dec,
                                len_Z_checked_non_dec=len_Z_checked_non_dec,
                                XP_calc_non_dec=X_P_calc_non_dec,
                                XP_calc_dec=X_P_calc_dec,
                                gain_Z_non_dec=gain_Z_non_dec,
                                gain_Z_dec=gain_Z_dec,
                                gain_X_non_dec=gain_X_non_dec,
                                gain_X_dec=gain_X_dec,
                                wrong_detections_z_dec=wrong_detections_z_dec,
                                wrong_detections_z_non_dec=wrong_detections_z_non_dec,
                                wrong_detections_x_dec=wrong_detections_x_dec,
                                wrong_detections_x_non_dec=wrong_detections_x_non_dec,
                                )'''

        if save_output == True:
            function_name = inspect.currentframe().f_code.co_name
            Saver.save_results_to_txt(  # Save the results to a text file
                function_used = function_name,
                n_samples=self.config.n_samples,
                seed=self.config.seed,
                non_signal_voltage=self.config.non_signal_voltage,
                voltage_decoy=self.config.voltage_decoy, 
                voltage=self.config.voltage, 
                voltage_decoy_sup=self.config.voltage_decoy_sup, 
                voltage_sup=self.config.voltage_sup,
                p_indep_x_states_non_dec=self.config.p_indep_x_states_non_dec,
                p_indep_x_states_dec=self.config.p_indep_x_states_dec,
                p_vacuum_z=p_vacuum_z,
                len_vacuum_indices_x_long=len(vacuum_indices_x_long),
                len_Z_checked_dec=len_Z_checked_dec,
                len_Z_checked_non_dec=len_Z_checked_non_dec,
                XP_calc_non_dec=X_P_calc_non_dec,
                XP_calc_dec=X_P_calc_dec,       
                gain_Z_non_dec=gain_Z_non_dec,
                gain_Z_dec=gain_Z_dec,
                gain_X_non_dec=gain_X_non_dec,
                gain_X_dec=gain_X_dec,
                wrong_detections_z_dec=wrong_detections_z_dec,
                wrong_detections_z_non_dec=wrong_detections_z_non_dec,
                wrong_detections_x_dec=wrong_detections_x_dec,
                wrong_detections_x_non_dec=wrong_detections_x_non_dec,
                len_wrong_z_dec=len(wrong_detections_z_dec),
                len_wrong_z_non_dec=len(wrong_detections_z_non_dec),
                len_wrong_x_dec=len(wrong_detections_x_dec),
                len_wrong_x_non_dec=len(wrong_detections_x_non_dec),
                qber_z_dec=qber_z_dec,
                qber_z_non_dec=qber_z_non_dec,
                qber_x_dec=qber_x_dec,
                qber_x_non_dec=qber_x_non_dec,
                raw_key_rate=raw_key_rate,
                total_amount_detections=total_amount_detections,
                execution_time_run=execution_time_run
            )
        
        return len_wrong_x_dec, len_wrong_x_non_dec, len_wrong_z_dec, len_wrong_z_non_dec, len_Z_checked_dec, len_Z_checked_non_dec, X_P_calc_non_dec, X_P_calc_dec
        
    def run_simulation_till_DLI(self):
        start_time = time.time()  # Record start time
        T1_dampening = self.simulation_engine.initialize()
        # p_indep_x_states_non_dec = self.simulation_engine.find_p_indep_states_x_for_classifier(T1_dampening, simulation_length_factor=1000, is_decoy=False)
        # p_indep_x_states_dec = self.simulation_engine.find_p_indep_states_x_for_classifier(T1_dampening, simulation_length_factor=1000, is_decoy=True)
        # print(f"p_indep_x_states_non_dec: {p_indep_x_states_non_dec}")
        # print(f"p_indep_x_states_dec: {p_indep_x_states_dec}")

        optical_power, peak_wavelength, chosen_voltage, chosen_current = self.simulation_engine.random_laser_output('current_power', 'voltage_shift', fixed = True)

        # Generate Alice's choices
        basis_array, value_array, decoy_array, lookup_array = self.simulation_helper.create_all_symbol_combinations_for_hist()

        basis, value, decoy = self.simulation_engine.generate_alice_choices(basis=basis_array, value=value_array, decoy=decoy_array)

        # Simulate signal and transmission
        signals, t, _ = self.simulation_engine.signal_bandwidth_jitter(basis, value, decoy)
        print((f"signals: {signals[:10]}"))  # Check the first 10 values
        '''amount_symbols_in_plot = 5
        pulse_duration = 1 / self.config.sampling_rate_FPGA
        sampling_rate_fft = 100e11
        samples_per_pulse = int(pulse_duration * sampling_rate_fft)
        total_samples = self.config.n_pulses * samples_per_pulse
        t_plot1 = np.linspace(0, amount_symbols_in_plot * self.config.n_pulses * pulse_duration, amount_symbols_in_plot * total_samples, endpoint=False)
        signals_part = signals[30:30 + amount_symbols_in_plot]
        flattened_signals = signals_part.reshape(-1)
        plt.plot(t_plot1 * 1e9, flattened_signals)
        plt.title(f"Voltage Signal with Bandwidth and Jitter for {amount_symbols_in_plot} symbols")
        plt.ylabel('Volt (V)')
        plt.xlabel('Time (ns)')
        Saver.save_plot(f"signal_after_bandwidth")'''

        power_dampened, norm_transmission,  calc_mean_photon_nr_eam, _ = self.simulation_engine.eam_transmission(signals, optical_power, T1_dampening, peak_wavelength, t)
        # self.plotter.plot_power(t, power_dampened, amount_symbols_in_plot=10, where_plot_1='after EAM')

        power_dampened = self.simulation_engine.fiber_attenuation(power_dampened)

        # path for X basis
        power_dampened = power_dampened * (1 - self.config.p_z_bob)

        #plot
        amount_symbols_in_first_part = 6
        shift_DLI = 0
        first_power = power_dampened[shift_DLI:shift_DLI + amount_symbols_in_first_part]

        # DLI
        power_dampened, f_0 = self.simulation_engine.delay_line_interferometer(power_dampened, t, peak_wavelength, value)
        # print(f"PHASESHIFT in Grad: {np.angle(phase_shift) / (2 * np.pi) * 360}")
        # print(f"shape of power_dampened after DLI: {power_dampened.shape}")

        # plot
        self.plotter.plot_power(t, power_dampened, amount_symbols_in_plot=amount_symbols_in_first_part, where_plot_1='before DLI',  shortened_first_power=first_power, where_plot_2='after DLI ', title_rest='- in fft, mean_volt: ' + str("{:.5f}".format(self.config.mean_voltage)) + ' voltage: ' + str("{:.5f}".format(chosen_voltage[0])) + ' V and ' + str("{:.5f}".format(peak_wavelength[0])), shift=shift_DLI)

        Saver.memory_usage("after everything: " + str("{:.3f}".format(time.time() - start_time)))
        # print(f"first 10 symbols of basis, value, decoy: {basis[:10]}, {value[:10]}, {decoy[:10]}")
        inverse_voltage = self.simulation_engine.get_interpolated_value(1550.68030 - 1550, 'voltage_shift', inverse_flag=True)
        print(f"inverse voltage: {inverse_voltage}")

    def run_simulation_parameter_sweep_heater_transmission(self):
        #initialize
        T1_dampening = self.simulation_engine.initialize()
        
        # Define the states and their corresponding arguments
        states = [
            {"title": "State: Z0", "basis": 1, "value": 1, "decoy": 0},
            {"title": "State: Z1", "basis": 1, "value": 0, "decoy": 0},
            {"title": "State: X+", "basis": 0, "value": -1, "decoy": 0},
            {"title": "State: Z0 decoy", "basis": 1, "value": 1, "decoy": 1},
            {"title": "State: Z1 decoy", "basis": 1, "value": 0, "decoy": 1},
            {"title": "State: X+ decoy", "basis": 0, "value": -1, "decoy": 1},
        ]
    
        def laser_till_eam(state):
            optical_power, peak_wavelength, chosen_voltage, chosen_current = self.simulation_engine.random_laser_output('current_power', 'voltage_shift')
            basis, value, decoy = self.simulation_engine.generate_alice_choices(basis = 0, value = -1, decoy = 0)

            signals, t, _ = self.simulation_engine.signal_bandwidth_jitter(basis, value, decoy)
            power_dampened, norm_transmission,  calc_mean_photon_nr_eam, _ = self.simulation_engine.eam_transmission(signals, optical_power, T1_dampening, peak_wavelength, t)
            
            return state, power_dampened

        parameters_amplitude =  np.linspace(0.0005, 0.005, 20)
        differences_mean_photon_nr = np.empty((len(states), len(parameters_amplitude)))
        
        for index, state in tqdm(enumerate(states), desc= "running simulation for different states", unit=" states", position=0):
            transmission_min = 1000
            transmission_max = 0
           
            for idx_param, param in tqdm(enumerate(parameters_amplitude), desc="parameters", unit="parameters", leave = False, position=1):
                self.config.current_amplitude = param
            
                mean_of_mean_photon = np.empty(self.config.n_samples)

                for i in range(self.config.n_samples):
                    optical_power, peak_wavelength, chosen_voltage, chosen_current = self.simulation_engine.random_laser_output('current_power', 'voltage_shift')
                    
                    # Generate Alice's choices
                    basis, value, decoy = self.simulation_engine.generate_alice_choices(basis=state["basis"], value=state["value"], decoy=state["decoy"], fixed = True)
                    
                    # Simulate signal and transmission
                    signals, t, jitter_shifts = self.simulation_engine.signal_bandwidth_jitter(basis, value, decoy)
                    power_dampened, transmission = self.simulation_engine.eam_transmission(signals, optical_power, T1_dampening)
                    power_dampened = self.simulation_engine.fiber_attenuation(power_dampened)

                    calc_mean_photon_nr, wavelength_photons, time_photons, nr_photons, index_where_photons, all_time_max_nr_photons, sum_nr_photons_at_chosen = self.simulation_engine.choose_photons(power_dampened, transmission, 
                                                                                                                                                                                t_jitter, peak_wavelength)
            
                    mean_of_mean_photon[i] = calc_mean_photon_nr

                    if mean_photon_nr_min > calc_mean_photon_nr:
                        mean_photon_nr_min = calc_mean_photon_nr
                    if mean_photon_nr_max < calc_mean_photon_nr:
                        mean_photon_nr_max = calc_mean_photon_nr

                differences_mean_photon_nr[index][idx_param] = (mean_photon_nr_max-mean_photon_nr_min) / np.mean(mean_of_mean_photon)

            plt.plot(parameters_amplitude*1e3, differences_mean_photon_nr[index], label= 'for ' + str(state['title'].replace(':', '').lower()))
            
            '''plt.title(f" spread of mean photon number for {state['title'].replace(':', '').lower()} over {self.config.n_samples} iterations")                    
            plt.xlabel(r'$\Delta I \, (\mathrm{mA})$')  # ΔI (mA)
            plt.ylabel(r'$\frac{\Delta \langle \mu \rangle}{\langle \mu \rangle}$')  # Δ⟨μ⟩
            plt.tight_layout()
            plt.legend()
            all_titles = state['title'].replace(' ', '_').replace(':', '').lower()
            save_plot(f"spread_photon_nr_{all_titles}")'''

        plt.title(f" spread of mean photon number over {self.config.n_samples} iterations")
        plt.xlabel(r'$\Delta I \, (\mathrm{mA})$')  # ΔI (mA)
        plt.ylabel(r'$\frac{\Delta \langle \mu \rangle}{\langle \mu \rangle}$')  # Δ⟨μ⟩ / ⟨μ⟩
        plt.tight_layout()
        plt.legend()    
        all_titles = "_".join([state['title'].replace(' ', '_').replace(':', '').lower() for state in states])
        Saver.save_plot(f"spread_photon_nr_{all_titles}_for_4GHz_and_1e-11_jitter")
        
    def run_simulation_det_peak_wave(self):
        
        start_time = time.time()  # Record start time
        T1_dampening = self.simulation_engine.initialize()
        optical_power, peak_wavelength, chosen_voltage, chosen_current = self.simulation_engine.random_laser_output('current_power', 'voltage_shift')
        # Generate Alice's choices
        basis, value, decoy = self.simulation_engine.generate_alice_choices(basis = 0, value = -1, decoy = 0)

        # Simulate signal and transmission
        Saver.memory_usage("before simulating signal: " + str("{:.3f}".format(time.time() - start_time)))
        signals, t, _ = self.simulation_engine.signal_bandwidth_jitter(basis, value, decoy)


        time_simulating_signal = time.time() - start_time
        Saver.memory_usage("before eam: " + str("{:.3f}".format(time_simulating_signal)))
        power_dampened, norm_transmission,  calc_mean_photon_nr_eam, _ = self.simulation_engine.eam_transmission(signals, optical_power, T1_dampening, peak_wavelength, t)

        time_eam = time.time() - start_time
        Saver.memory_usage("before fiber: " + str("{:.3f}".format(time_eam)))
        power_dampened = self.simulation_engine.fiber_attenuation(power_dampened)
        
        # self.plotter.plot_power(t, power_dampened, amount_symbols_in_plot=4, where_plot_1='after fiber')

        # first Z basis bc no interference
        Saver.memory_usage("before detector z: " + str("{:.3f}".format(time.time() - start_time)))
        power_dampened = power_dampened * self.config.p_z_bob
        time_photons_det_z, wavelength_photons_det_z, nr_photons_det_z, index_where_photons_det_z, calc_mean_photon_nr_detector_z, dark_count_times_z, num_dark_counts_z = self.simulation_engine.detector(t, norm_transmission, peak_wavelength, power_dampened, start_time)
        power_dampened = power_dampened / self.config.p_z_bob
        Saver.memory_usage("before classificator: " + str("{:.3f}".format(time.time() - start_time)))

        # path for X basis
        Saver.memory_usage("before DLI: " + str("{:.3f}".format(time.time() - start_time)))
        power_dampened = power_dampened * (1 - self.config.p_z_bob)

        #plot
        amount_symbols_in_first_part = 10
        first_power = power_dampened[:amount_symbols_in_first_part]

        # DLI
        power_dampened, phase_shift = self.simulation_engine.delay_line_interferometer(power_dampened, t, peak_wavelength)
        print(f"PHASESHIFT in Grad: {np.angle(phase_shift) / (2 * np.pi) * 360}")
        print(f"shape of power_dampened after DLI: {power_dampened.shape}")
        # plot
        self.plotter.plot_power(t, power_dampened, amount_symbols_in_plot=amount_symbols_in_first_part, where_plot_1='before DLI',  shortened_first_power=first_power, where_plot_2='after DLI erster port,', title_rest='+ omega 0 for current ' + str(self.config.mean_current) + ' mA')

        Saver.memory_usage("before detector x: " + str(time.time() - start_time))
        time_photons_det_x, wavelength_photons_det_x, nr_photons_det_x, index_where_photons_det_x, calc_mean_photon_nr_detector_x, dark_count_times_x, num_dark_counts_x = self.simulation_engine.detector(t, norm_transmission, peak_wavelength, power_dampened, start_time)        
        
        # get results for both detectors
        len_wrong_detections_z, len_wrong_detections_x, total_amount_detections, amount_Z_detections, amount_XP_detections, qber, phase_error_rate, raw_key_rate, gain_XP_norm, gain_XP_dec, gain_Z_norm, gain_Z_dec, detected_indices_x_dec, detected_indices_x_norm, detected_indices_z_dec, detected_indices_z_norm = self.simulation_engine.classificator(t, time_photons_det_x, index_where_photons_det_x, 
                                                                                                            time_photons_det_z, index_where_photons_det_z, 
                                                                                                            basis, value, decoy)
        
        # plot so I can delete
        self.plotter.plot_and_delete_photon_time_histogram(time_photons_det_x, time_photons_det_z)
        
        # late det in x
        if detected_indices_x_norm.size > 0: 
            condition_norm = np.sum(detected_indices_x_norm == 0, axis=1) == 1
        else:
            # Handle empty case
            condition_norm = np.array([], dtype=bool)  # or handle as needed
        if detected_indices_x_dec.size > 0:
            condition_dec = np.sum(detected_indices_x_dec == 0, axis=1) == 1
        else:
            # Handle empty case
            condition_dec = np.array([], dtype=bool)  # or handle as needed

        # Combine conditions and count how many rows satisfy at least one
        amount_detection_x_late_bin = np.sum(condition_norm | condition_dec)


        #Z1_sent_norm, Z1_sent_dec, Z0_sent_norm, Z0_sent_dec, XP_sent_norm, XP_sent_dec = self.simulation_helper.count_alice_choices(basis, value, decoy)

        #readin time
        end_time_read = time.time()  # Record end time  
        execution_time_run = end_time_read - start_time  # Calculate execution time

        Saver.save_results_to_txt(  # Save the results to a text file
            n_samples=self.config.n_samples,
            seed=self.config.seed,
            non_signal_voltage=self.config.non_signal_voltage,
            voltage_decoy=self.config.voltage_decoy, 
            voltage=self.config.voltage, 
            voltage_decoy_sup=self.config.voltage_decoy_sup, 
            voltage_sup=self.config.voltage_sup,
            len_wrong_detections_z=len_wrong_detections_z,
            len_wrong_detections_x=len_wrong_detections_x,
            total_amount_detections=total_amount_detections,
            amount_Z_detections=amount_Z_detections,
            amount_XP_detections=amount_XP_detections,
            qber=qber,
            phase_error_rate=phase_error_rate,
            raw_key_rate=raw_key_rate,
            gain_XP_norm=gain_XP_norm,
            gain_XP_dec=gain_XP_dec,
            gain_Z_norm=gain_Z_norm,
            gain_Z_dec=gain_Z_dec,
            calc_mean_photon_nr_eam=calc_mean_photon_nr_eam,
            calc_mean_photon_nr_detector_x=calc_mean_photon_nr_detector_x,
            execution_time_run=execution_time_run,
            time_simulating_signal=time_simulating_signal,
            time_eam=time_eam
        )
        '''Z0_sent_norm=Z0_sent_norm,
        Z0_sent_dec=Z0_sent_dec,
        Z1_sent_norm=Z1_sent_norm,
        Z1_sent_dec=Z1_sent_dec,
        XP_sent_norm=XP_sent_norm,
        XP_sent_dec=XP_sent_dec,'''
        return peak_wavelength[0], amount_detection_x_late_bin
        
    def run_simulation_hist_final(self):
        
        start_time = time.time()  # Record start time
        T1_dampening = self.simulation_engine.initialize()

        optical_power, peak_wavelength, chosen_voltage, chosen_current = self.simulation_engine.random_laser_output('current_power', 'voltage_shift', fixed = True)
    
        # Generate Alice's choices
        basis_arr, value_arr, decoy_arr, lookup_arr = self.simulation_helper.create_all_symbol_combinations_for_hist()

        basis, value, decoy = self.simulation_engine.generate_alice_choices(basis=basis_arr, value=value_arr, decoy=decoy_arr)

        # Simulate signal and transmission
        Saver.memory_usage("before simulating signal: " + str("{:.3f}".format(time.time() - start_time)))
        signals, t, _ = self.simulation_engine.signal_bandwidth_jitter(basis, value, decoy)

        time_simulating_signal = time.time() - start_time
        Saver.memory_usage("before eam: " + str("{:.3f}".format(time_simulating_signal)))
        power_dampened, norm_transmission,  calc_mean_photon_nr_eam, _ = self.simulation_engine.eam_transmission(signals, optical_power, T1_dampening, peak_wavelength, t)

        # plot so I can delete
        # self.plotter.plot_and_delete_mean_photon_histogram(calc_mean_photon_nr_eam, target_mean_photon_nr = np.array([self.config.mean_photon_nr, self.config.mean_photon_decoy]), 
        #                                         type_photon_nr = "Mean Photon Number at EAM")


        # self.plotter.plot_power(t, power_dampened, amount_symbols_in_plot=4, where_plot_1='after EAM')

        time_eam = time.time() - start_time
        Saver.memory_usage("before fiber: " + str("{:.3f}".format(time_eam)))
        power_dampened = self.simulation_engine.fiber_attenuation(power_dampened)
        
        # self.plotter.plot_power(t, power_dampened, amount_symbols_in_plot=4, where_plot_1='after fiber')

        # first Z basis bc no interference
        Saver.memory_usage("before detector z: " + str("{:.3f}".format(time.time() - start_time)))
        power_dampened = power_dampened * self.config.p_z_bob
        time_photons_det_z, wavelength_photons_det_z, nr_photons_det_z, index_where_photons_det_z, calc_mean_photon_nr_detector_z, dark_count_times_z, num_dark_counts_z = self.simulation_engine.detector(t, norm_transmission, peak_wavelength, power_dampened, start_time)
        power_dampened = power_dampened / self.config.p_z_bob
        Saver.memory_usage("before classificator: " + str("{:.3f}".format(time.time() - start_time)))

        # path for X basis
        Saver.memory_usage("before DLI: " + str("{:.3f}".format(time.time() - start_time)))
        power_dampened = power_dampened * (1 - self.config.p_z_bob)

        #plot
        first_power = None
        if random.random() < 0.01:
            amount_symbols_in_first_part = 10
            first_power = power_dampened[:amount_symbols_in_first_part]

        # DLI
        power_dampened, _ = self.simulation_engine.delay_line_interferometer(power_dampened, t, peak_wavelength, value)
     
        # plot
        if first_power is not None:
            self.plotter.plot_power(t, power_dampened, amount_symbols_in_plot=amount_symbols_in_first_part, where_plot_1='before DLI',  shortened_first_power=first_power, where_plot_2='after DLI erster port,', title_rest='+ omega 0 for current ' + str(self.config.mean_current) + ' mA')

        Saver.memory_usage("before detector x: " + str(time.time() - start_time))
        time_photons_det_x, wavelength_photons_det_x, nr_photons_det_x, index_where_photons_det_x, calc_mean_photon_nr_detector_x, dark_count_times_x, num_dark_counts_x = self.simulation_engine.detector(t, norm_transmission, peak_wavelength, power_dampened, start_time)        

        # self.plotter.plot_and_delete_photon_time_histogram(time_photons_det_x, time_photons_det_z)   

        # plot so I can delete
        # self.plotter.plot_and_delete_mean_photon_histogram(calc_mean_photon_nr_detector_x, target_mean_photon_nr=None, type_photon_nr="Mean Photon Number at Detector X")
        # self.plotter.plot_and_delete_mean_photon_histogram(calc_mean_photon_nr_detector_z, target_mean_photon_nr=None, type_photon_nr="Mean Photon Number at Detector Z")
        # self.plotter.plot_and_delete_photon_wavelength_histogram_two_diagrams(wavelength_photons_det_x, wavelength_photons_det_z)
        # self.plotter.plot_and_delete_photon_nr_histogram(nr_photons_det_x, nr_photons_det_z)
        
        # get results for both detectors
        # get results for both detectors
        '''function_name = inspect.currentframe().f_code.co_name
        Saver.save_results_to_txt(  # Save the results to a text file 
            function_used = function_name,
            n_samples=self.config.n_samples,
            seed=self.config.seed,
            non_signal_voltage=self.config.non_signal_voltage,
            voltage_decoy=self.config.voltage_decoy, 
            voltage=self.config.voltage, 
            voltage_decoy_sup=self.config.voltage_decoy_sup, 
            voltage_sup=self.config.voltage_sup,
            p_indep_x_states_non_dec=self.config.p_indep_x_states_non_dec,
            p_indep_x_states_dec=self.config.p_indep_x_states_dec)'''
        
        return time_photons_det_x, time_photons_det_z, index_where_photons_det_x, index_where_photons_det_z, t[-1], lookup_arr

    def run_DLI(self):
        #need low n_samples and same batchsize!
        T1_dampening = self.simulation_engine.initialize()

        def laser_till_dli(T1_dampening):
            optical_power, peak_wavelength, chosen_voltage, chosen_current = self.simulation_engine.random_laser_output('current_power', 'voltage_shift', fixed = True)
            basis, value, decoy = self.simulation_engine.generate_alice_choices(basis=np.array([1,0,1]), value=np.array([1,-1, 0]), decoy=np.array([0,0,0]))
            # basis, value, decoy = self.simulation_engine.generate_alice_choices(basis = np.array([0, 0, 0, 1, 1]), value = np.array([-1, -1, -1, 1, 1]), decoy = np.array([0, 0, 0, 0, 0]))
            signals, t, _ = self.simulation_engine.signal_bandwidth_jitter(basis, value, decoy)

            power_dampened, norm_transmission,  calc_mean_photon_nr_eam, _ = self.simulation_engine.eam_transmission(signals, optical_power, T1_dampening, peak_wavelength, t)
            
            '''dt_original = t[1]-t[0]
            num_points = signals.reshape(-1).size 
            print(f"num_points:{num_points}")
            t_original_all_sym = np.arange(t[0], t[0] + num_points * dt_original, dt_original)   
            plt.plot(t_original_all_sym *1e9, power_dampened.reshape(-1) , label='Signal')
            plt.title(f"power Signal after eam with Bandwidth and Jitter")
            plt.ylabel('Power (W)')
            plt.xlabel('Time (ns)')
            plt.legend()
            Saver.save_plot(f"power_after_eam")'''
            
            original_power = power_dampened.reshape(-1).copy()

            destructive_port, f_0= self.simulation_engine.delay_line_interferometer(power_dampened, t, peak_wavelength, value)
            # print(f"shape destructive_port: {destructive_port.shape}")
            destructive_port = destructive_port.reshape(-1)
            return chosen_voltage[0], peak_wavelength[0], destructive_port, original_power, t
        
        voltage_values = np.arange(-2, -1.65, 0.001)  # Example range of mean_voltage
        # voltage_values = np.arange(-2.2, -0.2, 0.002)
        results = []
        wavelengths_nm = []
        
        for voltage in voltage_values:
            self.config.mean_voltage = voltage
            chosen_voltage, peak_wavelength, destructive_port, original_power, t= laser_till_dli(T1_dampening)
            # plt.plot(original_power, color = 'blue', label='Input Power')
            # # plt.plot(destructive_port, color = 'red', label='Destructive Port')
            # plt.show()
            results.append({
                "voltage": chosen_voltage,
                "wavelength": peak_wavelength,
                "destructive_port": destructive_port,
            })
            # print(f"chosen_voltage: {chosen_voltage}, peak_wavelength: {peak_wavelength}, destructive_port: {destructive_port[:5]}")
            # print(f" shape of destructive_port: {destructive_port.shape}, shape chosen_voltage: {chosen_voltage.shape}, shape peak_wavelength: {peak_wavelength.shape}")

        # t = np.arange(len(power_dampened)) * dt  # Time array in seconds
        step_size = t[1] - t[0]
        # Calculate the new length
        new_length = len(t) * self.config.n_samples
        # Generate the new array with the same step size
        t_all_symbols = np.arange(t[0], t[0] + step_size * new_length, step_size)
        print(f"t: {t[:5]} with {self.config.n_samples} samples")
        print(f"t_all_symbols: {t_all_symbols[:5]} with {self.config.n_samples} samples")
        print(f"shape of t: {np.shape(t)}")
        target_time_ns = 1/6.5 * 3  # 1/6.5 is one pulse then multiply by how manyth pulse you want to see
        target_index = np.argmin(np.abs(t_all_symbols * 1e9 - target_time_ns))
        print(f"target_time_ns:{target_time_ns}, target_index: {target_index}, len(t_all_symbols): {len(t_all_symbols)}, len(t): {len(t)}")

        '''for r in results:
            print(f"Shape of r['destructive_port']: {np.shape(r['destructive_port'])}")
            print(f"Value at index {target_index}: {r['destructive_port'][target_index]}")'''

        wavelengths_nm = np.array([r["wavelength"] for r in results])
        print(f"wavelengths_nm: {wavelengths_nm}")
        amplitudes_port1 = np.array([r["destructive_port"][target_index] for r in results])
        print(f"wavelengths_nm shape: {wavelengths_nm.shape}")
        print(f"amplitudes_port1 shape: {amplitudes_port1.shape}")


        # Step 2: Find max, min, and midpoint amplitude locations
        idx_max = np.argmax(amplitudes_port1)
        idx_min = np.argmin(amplitudes_port1)
        print(f"idx_max: {idx_max}, idx_min: {idx_min}")

        # Get mid index between min and max (closest to halfway in wavelength)
        lambda_mid = (wavelengths_nm[idx_max] + wavelengths_nm[idx_min]) / 2
        idx_mid = np.argmin(np.abs(wavelengths_nm - lambda_mid))

        # Gather selected indices and their corresponding data
        selected_indices = [idx_max, idx_min, idx_mid]
        selected_labels = ["Max", "Min", "Mid"]
        selected_data = [results[i] for i in selected_indices]
        print(f"selected_data: {selected_data}")

        for item in selected_data:
            print(item['wavelength'])  # should show a non-zero value

        plt.figure(figsize=(14, 10))
        for i, (label, data) in enumerate(zip(selected_labels, selected_data)):
            plt.subplot(3, 1, i + 1)
            plt.plot(t_all_symbols * 1e9, original_power, label='Input Power', linestyle='--')
            plt.plot(t_all_symbols * 1e9, data["destructive_port"], label='Destructive Port')
            plt.title(f"{label} Interference Case @ {data['wavelength']*1e9:.5f} nm")
            plt.xlabel("Time (ns)")
            plt.ylabel("Power (a.u.)")
            plt.legend()
            plt.grid(True)
        plt.tight_layout()
        Saver.save_plot("DLI_interference_cases")

        # Plot power vs voltage
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.plot(voltage_values, amplitudes_port1, marker='o')
        plt.xlabel("Mean Voltage (V)")
        plt.ylabel("amplitude on special point")
        plt.title("Power vs Mean Voltage")
        plt.grid(True)
        plt.minorticks_on()
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)


        # Plot peak_wavelength vs voltage
        plt.subplot(1, 2, 2)
        plt.plot(voltage_values, wavelengths_nm, marker='x', color='orange')
        plt.xlabel("Mean Voltage (V)")
        plt.ylabel("Peak Wavelength (nm)")
        plt.title("Wavelength vs Mean Voltage")
        plt.grid(True)
        plt.legend()

        plt.tight_layout()
        Saver.save_plot("DLI_power_wavelength_vs_voltage")
        '''for wave in wavelengths_nm:
            inverse_voltage = self.simulation_engine.get_interpolated_value(wave * 1e9 - 1550, 'voltage_shift', inverse_flag=True)
            print(f"inverse voltage: {inverse_voltage}")'''

    def run_simulation_hist_pick_symbols(self):
        start_time = time.time()  # Record start time
        T1_dampening = self.simulation_engine.initialize()

        optical_power, peak_wavelength, chosen_voltage, chosen_current = self.simulation_engine.random_laser_output('current_power', 'voltage_shift', fixed = True)
    
        # Generate Alice's choices
        _, _, _, lookup_array = self.simulation_helper.create_all_symbol_combinations_for_hist()
        basis, value, decoy = self.simulation_engine.generate_alice_choices()

        # Simulate signal and transmission
        Saver.memory_usage("before simulating signal: " + str("{:.3f}".format(time.time() - start_time)))
        signals, t, _ = self.simulation_engine.signal_bandwidth_jitter(basis, value, decoy)

        time_simulating_signal = time.time() - start_time
        Saver.memory_usage("before eam: " + str("{:.3f}".format(time_simulating_signal)))
        power_dampened, norm_transmission,  calc_mean_photon_nr_eam, _ = self.simulation_engine.eam_transmission(signals, optical_power, T1_dampening, peak_wavelength, t)

        # plot so I can delete
        # self.plotter.plot_and_delete_mean_photon_histogram(calc_mean_photon_nr_eam, target_mean_photon_nr = np.array([self.config.mean_photon_nr, self.config.mean_photon_decoy]), 
        #                                         type_photon_nr = "Mean Photon Number at EAM")


        # self.plotter.plot_power(t, power_dampened, amount_symbols_in_plot=4, where_plot_1='after EAM')

        time_eam = time.time() - start_time
        Saver.memory_usage("before fiber: " + str("{:.3f}".format(time_eam)))
        power_dampened = self.simulation_engine.fiber_attenuation(power_dampened)
        
        # self.plotter.plot_power(t, power_dampened, amount_symbols_in_plot=4, where_plot_1='after fiber')

        # first Z basis bc no interference
        Saver.memory_usage("before detector z: " + str("{:.3f}".format(time.time() - start_time)))
        power_dampened = power_dampened * self.config.p_z_bob
        time_photons_det_z, wavelength_photons_det_z, nr_photons_det_z, index_where_photons_det_z, calc_mean_photon_nr_detector_z, dark_count_times_z, num_dark_counts_z = self.simulation_engine.detector(t, norm_transmission, peak_wavelength, power_dampened, start_time)
        power_dampened = power_dampened / self.config.p_z_bob
        Saver.memory_usage("before classificator: " + str("{:.3f}".format(time.time() - start_time)))

        # path for X basis
        Saver.memory_usage("before DLI: " + str("{:.3f}".format(time.time() - start_time)))
        power_dampened = power_dampened * (1 - self.config.p_z_bob)

        #plot
        first_power = None
        if random.random() < 0.01:
            amount_symbols_in_first_part = 10
            first_power = power_dampened[:amount_symbols_in_first_part]

        # DLI
        power_dampened, phase_shift = self.simulation_engine.delay_line_interferometer(power_dampened, t, peak_wavelength, value)

        # plot
        if first_power is not None:
            self.plotter.plot_power(t, power_dampened, amount_symbols_in_plot=amount_symbols_in_first_part, where_plot_1='before DLI',  shortened_first_power=first_power, where_plot_2='after DLI erster port,', title_rest='+ omega 0 for current ' + str(self.config.mean_current) + ' mA')

        Saver.memory_usage("before detector x: " + str(time.time() - start_time))
        time_photons_det_x, wavelength_photons_det_x, nr_photons_det_x, index_where_photons_det_x, calc_mean_photon_nr_detector_x, dark_count_times_x, num_dark_counts_x = self.simulation_engine.detector(t, norm_transmission, peak_wavelength, power_dampened, start_time)        

        # self.plotter.plot_and_delete_photon_time_histogram(time_photons_det_x, time_photons_det_z)   

        # plot so I can delete
        # self.plotter.plot_and_delete_mean_photon_histogram(calc_mean_photon_nr_detector_x, target_mean_photon_nr=None, type_photon_nr="Mean Photon Number at Detector X")
        # self.plotter.plot_and_delete_mean_photon_histogram(calc_mean_photon_nr_detector_z, target_mean_photon_nr=None, type_photon_nr="Mean Photon Number at Detector Z")
        # self.plotter.plot_and_delete_photon_wavelength_histogram_two_diagrams(wavelength_photons_det_x, wavelength_photons_det_z)
        # self.plotter.plot_and_delete_photon_nr_histogram(nr_photons_det_x, nr_photons_det_z)
        
        # get results for both detectors
        # get results for both detectors
        '''function_name = inspect.currentframe().f_code.co_name
        Saver.save_results_to_txt(  # Save the results to a text file 
            function_used = function_name,
            n_samples=self.config.n_samples,
            seed=self.config.seed,
            non_signal_voltage=self.config.non_signal_voltage,
            voltage_decoy=self.config.voltage_decoy, 
            voltage=self.config.voltage, 
            voltage_decoy_sup=self.config.voltage_decoy_sup, 
            voltage_sup=self.config.voltage_sup,
            p_indep_x_states_non_dec=self.config.p_indep_x_states_non_dec,
            p_indep_x_states_dec=self.config.p_indep_x_states_dec)'''
    

        '''# Save the arrays to an NPZ file named "simulation_data.npz"
        np.savez("simulation_data.npz", 
                time_photons_det_z=time_photons_det_z,
                time_photons_det_x=time_photons_det_x, 
                index_where_photons_det_z=index_where_photons_det_z,
                index_where_photons_det_x=index_where_photons_det_x,
                time_one_symbol=t[-1],
                basis=basis,
                value=value,
                decoy=decoy,
                lookup_array=lookup_array)

        print("Data saved to simulation_data.npz")'''
        gc.collect()
        time_one_symbol = t[-1]
        return time_one_symbol, time_photons_det_z, time_photons_det_x, index_where_photons_det_z, index_where_photons_det_x, lookup_array, basis, value, decoy
    
    def lookup(self):
         # Generate Alice's choices
        basis_arr, value_arr, decoy_arr, lookup_arr = self.simulation_helper.create_all_symbol_combinations_for_hist()
        return lookup_arr
    
    def run_simulation_detection_tester(self):
        
        start_time = time.time()  # Record start time
        T1_dampening = self.simulation_engine.initialize()
        optical_power, peak_wavelength, chosen_voltage, chosen_current = self.simulation_engine.random_laser_output('current_power', 'voltage_shift', fixed = True)
        print(f"peak_wavelegth: {peak_wavelength[:10]}")
        # Generate Alice's choices
        basis, value, decoy = self.simulation_engine.generate_alice_choices(basis=np.array([1,0,1]), value=np.array([1,-1, 0]), decoy=np.array([0,0,0]))
        # basis, value, decoy = self.simulation_engine.generate_alice_choices(basis=np.array([0,1]), value=np.array([-1, 0]), decoy=np.array([0,0]))
        # basis, value, decoy = self.simulation_engine.generate_alice_choices(basis=np.array([0]), value=np.array([-1]), decoy=np.array([0]))
        # basis, value, decoy = self.simulation_engine.generate_alice_choices()
        # basis, value, decoy = self.simulation_engine.generate_alice_choices(basis=np.array([1, 0, 0, 1, 1, 0, 0, 1]), value=np.array([1, -1, -1, 0, 1, -1, -1, 0]), decoy=np.array([0, 0, 0, 0, 1, 1, 1, 1]))


        # Simulate signal and transmission
        signals, t, _ = self.simulation_engine.signal_bandwidth_jitter(basis, value, decoy)

        power_dampened, norm_transmission,  calc_mean_photon_nr_eam, _ = self.simulation_engine.eam_transmission(signals, optical_power, T1_dampening, peak_wavelength, t)
        # print(f"mean_photon_nr after EAM: {self.simulation_helper.calculate_mean_photon_number(power_dampened, peak_wavelength, t)[:20]}")
        mean_photon_after_eam = self.simulation_helper.calculate_mean_photon_number(power_dampened, peak_wavelength, t)[:20]
        
        power_dampened = self.simulation_engine.fiber_attenuation(power_dampened)
        print(f"mean_photon_nr after fiber: {self.simulation_helper.calculate_mean_photon_number(power_dampened, peak_wavelength, t)[:20]}")
        mean_photon_after_fiber = self.simulation_helper.calculate_mean_photon_number(power_dampened, peak_wavelength, t)[:20]
        # self.plotter.plot_power(t, power_dampened, amount_symbols_in_plot=20, where_plot_1='after fiber')

        # first Z basis bc no interference
        power_dampened = power_dampened * self.config.p_z_bob
        time_photons_det_z, wavelength_photons_det_z, nr_photons_det_z, index_where_photons_det_z, \
        calc_mean_photon_nr_detector_z, dark_count_times_z, num_dark_counts_z = self.simulation_engine.detector(t, norm_transmission, peak_wavelength, power_dampened, start_time)
        power_dampened = power_dampened / self.config.p_z_bob

        # self.plotter.plot_power(t, power_dampened, amount_symbols_in_plot=20, where_plot_1='after Z det')

        # path for X basis
        power_dampened = power_dampened * (1 - self.config.p_z_bob)
        # print(f"mean_photon_nr after basis choice: {self.simulation_helper.calculate_mean_photon_number(power_dampened, peak_wavelength, t)[:20]}")
        mean_photon_after_basis_choice = self.simulation_helper.calculate_mean_photon_number(power_dampened, peak_wavelength, t)[:20]

        # self.plotter.plot_power(t, power_dampened, amount_symbols_in_plot=20, where_plot_1='bob basis X')

        #plot
        # amount_symbols_in_first_part = 20
        # first_power = power_dampened[:amount_symbols_in_first_part].copy()
        '''print("len(t):", len(t))
        print("first_power.shape:", first_power.shape)
        print("expected shape:", (amount_symbols_in_first_part, len(t)))
        print("power_dampened.shape:", power_dampened.shape)'''

        # DLI
        power_dampened, f_0 = self.simulation_engine.delay_line_interferometer(power_dampened, t, peak_wavelength, value)
        # print(f"mean_photon_nr after DLI: {self.simulation_helper.calculate_mean_photon_number(power_dampened, peak_wavelength, t)[:20]}")
        mean_photon_after_dli = self.simulation_helper.calculate_mean_photon_number(power_dampened, peak_wavelength, t)[:20]

        # plot
        # assert power_dampened.shape[1] == len(t), f"Mismatch: power_dampened has {power_dampened.shape[1]} samples per symbol, t has {len(t)}!"
        # self.plotter.plot_power(t, power_dampened, amount_symbols_in_plot=amount_symbols_in_first_part, where_plot_1='before DLI',  shortened_first_power=first_power, where_plot_2='after DLI,', title_rest='- in fft, mean_volt: ' + str("{:.4f}".format(self.config.mean_voltage)) + ' voltage: ' + str("{:.4f}".format(chosen_voltage[0])) + ' V and ' + str("{:.8f}".format(peak_wavelength[0])))
        
        # plt.plot(first_power.reshape(-1) * 1e3, color='blue', label='0', linestyle='-', marker='o', markersize=1)
        # Saver.save_plot(f"power_before_DLI_in_mW_outside")
        mean_photon_at_x_detector = self.simulation_helper.calculate_mean_photon_number(power_dampened, peak_wavelength, t)
        print(f"len(mean_photon_at_x_detector): {len(mean_photon_at_x_detector)}")
        # Saver.memory_usage("before detector x: " + str(time.time() - start_time))
        time_photons_det_x, wavelength_photons_det_x, nr_photons_det_x, index_where_photons_det_x, calc_mean_photon_nr_detector_x, dark_count_times_x, num_dark_counts_x = self.simulation_engine.detector(t, norm_transmission, peak_wavelength, power_dampened, start_time)        
        # print(f"calc_mean_photon_nr_detector_x: {calc_mean_photon_nr_detector_x[:20]}")
        with np.printoptions(threshold=100):
            Z0_alice_s = np.where((basis == 1) & (value == 1) & (decoy == 0))[0]  # Indices where Z0 was sent
            XP_alice_s = np.where((basis == 0) & (decoy == 0))[0]  # Indices where XP was sent
            Z0_XP_alice_s = XP_alice_s[np.isin(XP_alice_s - 1, Z0_alice_s)]
            print(f"manager len(Z0_alice_s): {len(Z0_alice_s)}, len(XP_alice_s): {len(XP_alice_s)}, len(Z0_XP_alice_s): {len(Z0_XP_alice_s)}")
            Z1_alice_s = np.where((basis == 1) & (value == 0) & (decoy == 1))[0]  # Indices where Z0 was sent
            XP_Z1_alice_s = Z1_alice_s[np.isin(Z1_alice_s - 1, XP_alice_s)]  # Indices where Z1Z0 was sent (index of Z0 used aka the higher index at which time we measure the X+ state)
            print(f"manager len(Z1_alice_s): {len(Z1_alice_s)}, len(XP_Z1_alice_s): {len(XP_Z1_alice_s)}")
            mean_photon_at_detector_of_Z0_XP_symbol_mean_photon_for_whole_symbol = mean_photon_at_x_detector[Z0_XP_alice_s]
            mean_of_Z0_XP = np.mean(mean_photon_at_detector_of_Z0_XP_symbol_mean_photon_for_whole_symbol)

            print(f"nr_photons: {len(nr_photons_det_x)}")

        
        # plot so I can delete
        # self.plotter.plot_and_delete_mean_photon_histogram(calc_mean_photon_nr_detector_x, target_mean_photon_nr=None, type_photon_nr="Mean Photon Number at Detector X")
        # self.plotter.plot_and_delete_mean_photon_histogram(calc_mean_photon_nr_detector_z, target_mean_photon_nr=None, type_photon_nr="Mean Photon Number at Detector Z")
        # self.plotter.plot_and_delete_photon_wavelength_histogram_two_diagrams(wavelength_photons_det_x, wavelength_photons_det_z)
        # self.plotter.plot_and_delete_photon_nr_histogram(nr_photons_det_x, nr_photons_det_z)
        
        # get results for both detectors
       
        p_vacuum_z, vacuum_indices_x_long, len_Z_checked_dec, len_Z_checked_non_dec, \
        gain_Z_non_dec, gain_Z_dec, gain_X_non_dec, gain_X_dec, X_P_calc_non_dec, X_P_calc_dec, \
        wrong_detections_z_dec, wrong_detections_z_non_dec, wrong_detections_x_dec, wrong_detections_x_non_dec, \
        qber_z_dec, qber_z_non_dec, qber_x_dec, qber_x_non_dec, raw_key_rate, \
        total_amount_detections = self.simulation_engine.classificator_new(t, time_photons_det_x, index_where_photons_det_x, 
                                                                           time_photons_det_z, index_where_photons_det_z, 
                                                                           basis, value, decoy)
        
        

        # plot so I can delete
        # self.plotter.plot_and_delete_photon_time_histogram(time_photons_det_x, time_photons_det_z)

        # Z1_sent_norm, Z1_sent_dec, Z0_sent_norm, Z0_sent_dec, XP_sent_norm, XP_sent_dec = self.simulation_helper.count_alice_choices(basis, value, decoy)

        #readin time
        end_time_read = time.time()  # Record end time  
        execution_time_run = end_time_read - start_time  # Calculate execution time

        len_wrong_z_dec=len(wrong_detections_z_dec)
        len_wrong_z_non_dec=len(wrong_detections_z_non_dec)
        len_wrong_x_dec=len(wrong_detections_x_dec)
        len_wrong_x_non_dec=len(wrong_detections_x_non_dec)
        
        '''function_name = inspect.currentframe().f_code.co_name
        Saver.save_results_to_txt(  # Save the results to a text file
            function_used = function_name,
            n_samples=self.config.n_samples,
            seed=self.config.seed,
            non_signal_voltage=self.config.non_signal_voltage,
            voltage_decoy=self.config.voltage_decoy, 
            voltage=self.config.voltage, 
            voltage_decoy_sup=self.config.voltage_decoy_sup, 
            voltage_sup=self.config.voltage_sup,
            p_indep_x_states_non_dec=self.config.p_indep_x_states_non_dec,
            p_indep_x_states_dec=self.config.p_indep_x_states_dec,
            p_vacuum_z=p_vacuum_z,
            len_vacuum_indices_x_long=len(vacuum_indices_x_long),
            len_Z_checked_dec=len_Z_checked_dec,
            len_Z_checked_non_dec=len_Z_checked_non_dec,
            XP_calc_non_dec=X_P_calc_non_dec,
            XP_calc_dec=X_P_calc_dec,       
            gain_Z_non_dec=gain_Z_non_dec,
            gain_Z_dec=gain_Z_dec,
            gain_X_non_dec=gain_X_non_dec,
            gain_X_dec=gain_X_dec,
            wrong_detections_z_dec=wrong_detections_z_dec,
            wrong_detections_z_non_dec=wrong_detections_z_non_dec,
            wrong_detections_x_dec=wrong_detections_x_dec,
            wrong_detections_x_non_dec=wrong_detections_x_non_dec,
            len_wrong_z_dec=len(wrong_detections_z_dec),
            len_wrong_z_non_dec=len(wrong_detections_z_non_dec),
            len_wrong_x_dec=len(wrong_detections_x_dec),
            len_wrong_x_non_dec=len(wrong_detections_x_non_dec),
            qber_z_dec=qber_z_dec,
            qber_z_non_dec=qber_z_non_dec,
            qber_x_dec=qber_x_dec,
            qber_x_non_dec=qber_x_non_dec,
            raw_key_rate=raw_key_rate,
            total_amount_detections=total_amount_detections,
            mean_photon_after_eam=mean_photon_after_eam,
            mean_photon_after_fiber=mean_photon_after_fiber,
            mean_photon_after_basis_choice=mean_photon_after_basis_choice,
            mean_photon_after_dli=mean_photon_after_dli,
            calc_mean_photon_nr_detector_x=calc_mean_photon_nr_detector_x,
            mean_photon_at_detector_of_Z0_XP_symbol_mean_photon_for_whole_symbol=mean_photon_at_detector_of_Z0_XP_symbol_mean_photon_for_whole_symbol,
            mean_of_Z0_XP=mean_of_Z0_XP,
            index_where_photons_det_z=index_where_photons_det_z,
            index_where_photons_det_x=index_where_photons_det_x,
            time_photons_det_x=time_photons_det_x.shape,
            time_photons_det_z=time_photons_det_z.shape,
        )'''
        
        time_one_symbol = t[-1]
        lookup_array = self.simulation_helper.create_all_symbol_combinations_for_hist()[3]
        
        return len_wrong_x_dec, len_wrong_x_non_dec, len_wrong_z_dec, len_wrong_z_non_dec, len_Z_checked_dec, len_Z_checked_non_dec, X_P_calc_non_dec, X_P_calc_dec, time_photons_det_x, time_photons_det_z, time_one_symbol, index_where_photons_det_z, index_where_photons_det_x, \
                basis, value, decoy, lookup_array
        

    def run_simulation_repeat(self, save_output = False):
        
        start_time = time.time()  # Record start time
        T1_dampening = self.simulation_engine.initialize()
     

        optical_power, peak_wavelength, chosen_voltage, chosen_current = self.simulation_engine.random_laser_output('current_power', 'voltage_shift', fixed = True)

    
        # Generate Alice's choices
        # basis, value, decoy = self.simulation_engine.generate_alice_choices(basis=np.array([1, 0, 0, 1, 1, 1, 1, 0, 0, 1]), value=np.array([1, -1, -1, 0, 0, 1, 1, -1, -1, 0]), decoy=np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1]))
        basis, value, decoy = self.simulation_engine.generate_alice_choices(basis=np.array([1, 0, 0, 1, 1, 0, 0, 1]), value=np.array([1, -1, -1, 0, 1, -1, -1, 0]), decoy=np.array([0, 0, 0, 0, 1, 1, 1, 1]))
        # basis, value, decoy = self.simulation_engine.generate_alice_choices(basis=np.array([1,0,1]), value=np.array([1,-1, 0]), decoy=np.array([0,0,0]))
        # basis, value, decoy = self.simulation_engine.generate_alice_choices(basis=np.array([0,1]), value=np.array([-1, 0]), decoy=np.array([0,0]))
        # basis, value, decoy = self.simulation_engine.generate_alice_choices()
        print(f"basis: {basis[:10]}")
        print(f"value: {value[:10]}")
        print(f"decoy: {decoy[:10]}")

        # Simulate signal and transmission
        # Saver.memory_usage("before simulating signal: " + str("{:.3f}".format(time.time() - start_time)))
        signals, t, _ = self.simulation_engine.signal_bandwidth_jitter(basis, value, decoy)

        time_simulating_signal = time.time() - start_time
        # Saver.memory_usage("before eam: " + str("{:.3f}".format(time_simulating_signal)))
        power_dampened, norm_transmission,  calc_mean_photon_nr_eam, _ = self.simulation_engine.eam_transmission(signals, optical_power, T1_dampening, peak_wavelength, t)

        # plot so I can delete
        # self.plotter.plot_and_delete_mean_photon_histogram(calc_mean_photon_nr_eam, target_mean_photon_nr = np.array([self.config.mean_photon_nr, self.config.mean_photon_decoy]), 
        #                                         type_photon_nr = "Mean Photon Number at EAM")


        # self.plotter.plot_power(t, power_dampened, amount_symbols_in_plot=5, where_plot_1='after EAM')

        time_eam = time.time() - start_time
        # Saver.memory_usage("before fiber: " + str("{:.3f}".format(time_eam)))
        power_dampened = self.simulation_engine.fiber_attenuation(power_dampened)
        
        # self.plotter.plot_power(t, power_dampened, amount_symbols_in_plot=20, where_plot_1='after fiber')

        # first Z basis bc no interference
        # Saver.memory_usage("before detector z: " + str("{:.3f}".format(time.time() - start_time)))
        power_dampened = power_dampened * self.config.p_z_bob
        time_photons_det_z, wavelength_photons_det_z, nr_photons_det_z, index_where_photons_det_z, \
        calc_mean_photon_nr_detector_z, dark_count_times_z, num_dark_counts_z = self.simulation_engine.detector(t, norm_transmission, peak_wavelength, power_dampened, start_time)
        power_dampened = power_dampened / self.config.p_z_bob
        # Saver.memory_usage("before classificator: " + str("{:.3f}".format(time.time() - start_time)))

        # self.plotter.plot_power(t, power_dampened, amount_symbols_in_plot=20, where_plot_1='after Z det')

        # path for X basis
        # Saver.memory_usage("before DLI: " + str("{:.3f}".format(time.time() - start_time)))
        power_dampened = power_dampened * (1 - self.config.p_z_bob)

        # self.plotter.plot_power(t, power_dampened, amount_symbols_in_plot=20, where_plot_1='bob basis X')

        #plot
        '''amount_symbols_in_first_part = 30
        first_power = power_dampened[:amount_symbols_in_first_part].copy()'''

        # DLI
        power_dampened, f_0 = self.simulation_engine.delay_line_interferometer(power_dampened, t, peak_wavelength, value)

        # plot
        '''assert power_dampened.shape[1] == len(t), f"Mismatch: power_dampened has {power_dampened.shape[1]} samples per symbol, t has {len(t)}!"
        self.plotter.plot_power(t, power_dampened, amount_symbols_in_plot=amount_symbols_in_first_part, where_plot_1='before DLI',  shortened_first_power=first_power, where_plot_2='after DLI,', title_rest='- in fft, mean_volt: ' + str("{:.4f}".format(self.config.mean_voltage)) + ' voltage: ' + str("{:.4f}".format(chosen_voltage[0])) + ' V and ' + str("{:.8f}".format(peak_wavelength[0])))'''
        
        # plt.plot(first_power.reshape(-1) * 1e3, color='blue', label='0', linestyle='-', marker='o', markersize=1)
        # Saver.save_plot(f"power_before_DLI_in_mW_outside")

        # Saver.memory_usage("before detector x: " + str(time.time() - start_time))
        time_photons_det_x, wavelength_photons_det_x, nr_photons_det_x, index_where_photons_det_x, calc_mean_photon_nr_detector_x, dark_count_times_x, num_dark_counts_x = self.simulation_engine.detector(t, norm_transmission, peak_wavelength, power_dampened, start_time)        
        # np.set_printoptions(threshold=np.inf)  # disable truncation
        # print(f"calc_mean_photon_nr_detector_x part: {calc_mean_photon_nr_detector_x[:10]}")
        # print(f"calc_mean_photon_nr_detector_z part: {calc_mean_photon_nr_detector_z[:10]}")
        # print(f"nr_photons: {len(nr_photons_det_x)} {len(nr_photons_det_z)}")
        # plot so I can delete
        # self.plotter.plot_and_delete_mean_photon_histogram(calc_mean_photon_nr_detector_x, target_mean_photon_nr=None, type_photon_nr="Mean Photon Number at Detector X")
        # self.plotter.plot_and_delete_mean_photon_histogram(calc_mean_photon_nr_detector_z, target_mean_photon_nr=None, type_photon_nr="Mean Photon Number at Detector Z")
        # self.plotter.plot_and_delete_photon_wavelength_histogram_two_diagrams(wavelength_photons_det_x, wavelength_photons_det_z)
        # self.plotter.plot_and_delete_photon_nr_histogram(nr_photons_det_x, nr_photons_det_z)
        
        # get results for both detectors
        p_vacuum_z, vacuum_indices_x_long, len_Z_checked_dec, len_Z_checked_non_dec, \
        gain_Z_non_dec, gain_Z_dec, gain_X_non_dec, gain_X_dec, X_P_calc_non_dec, \
        X_P_calc_dec, wrong_detections_z_dec, wrong_detections_z_non_dec, wrong_detections_x_dec, \
        wrong_detections_x_non_dec, qber_z_dec, qber_z_non_dec, qber_x_dec, qber_x_non_dec, raw_key_rate, \
        total_amount_detections = self.simulation_engine.classificator_new(t, time_photons_det_x, index_where_photons_det_x, time_photons_det_z, index_where_photons_det_z, basis, value, decoy)

        # plot so I can delete
        # self.plotter.plot_and_delete_photon_time_histogram(time_photons_det_x, time_photons_det_z)

        # Z1_sent_norm, Z1_sent_dec, Z0_sent_norm, Z0_sent_dec, XP_sent_norm, XP_sent_dec = self.simulation_helper.count_alice_choices(basis, value, decoy)

        #readin time
        end_time_read = time.time()  # Record end time  
        execution_time_run = end_time_read - start_time  # Calculate execution time

        len_wrong_z_dec=len(wrong_detections_z_dec)
        len_wrong_z_non_dec=len(wrong_detections_z_non_dec)
        len_wrong_x_dec=len(wrong_detections_x_dec)
        len_wrong_x_non_dec=len(wrong_detections_x_non_dec)

        '''Saver.save_arrays_to_csv('results', 
                                p_vacuum_z=p_vacuum_z,
                                len_vacuum_indices_x_long=len(vacuum_indices_x_long),
                                len_Z_checked_dec=len_Z_checked_dec,
                                len_Z_checked_non_dec=len_Z_checked_non_dec,
                                XP_calc_non_dec=X_P_calc_non_dec,
                                XP_calc_dec=X_P_calc_dec,
                                gain_Z_non_dec=gain_Z_non_dec,
                                gain_Z_dec=gain_Z_dec,
                                gain_X_non_dec=gain_X_non_dec,
                                gain_X_dec=gain_X_dec,
                                wrong_detections_z_dec=wrong_detections_z_dec,
                                wrong_detections_z_non_dec=wrong_detections_z_non_dec,
                                wrong_detections_x_dec=wrong_detections_x_dec,
                                wrong_detections_x_non_dec=wrong_detections_x_non_dec,
                                )'''

        if save_output == True:
            function_name = inspect.currentframe().f_code.co_name
            Saver.save_results_to_txt(  # Save the results to a text file
                function_used = function_name,
                n_samples=self.config.n_samples,
                seed=self.config.seed,
                non_signal_voltage=self.config.non_signal_voltage,
                voltage_decoy=self.config.voltage_decoy, 
                voltage=self.config.voltage, 
                voltage_decoy_sup=self.config.voltage_decoy_sup, 
                voltage_sup=self.config.voltage_sup,
                p_indep_x_states_non_dec=self.config.p_indep_x_states_non_dec,
                p_indep_x_states_dec=self.config.p_indep_x_states_dec,
                p_vacuum_z=p_vacuum_z,
                len_vacuum_indices_x_long=len(vacuum_indices_x_long),
                len_Z_checked_dec=len_Z_checked_dec,
                len_Z_checked_non_dec=len_Z_checked_non_dec,
                XP_calc_non_dec=X_P_calc_non_dec,
                XP_calc_dec=X_P_calc_dec,       
                gain_Z_non_dec=gain_Z_non_dec,
                gain_Z_dec=gain_Z_dec,
                gain_X_non_dec=gain_X_non_dec,
                gain_X_dec=gain_X_dec,
                wrong_detections_z_dec=wrong_detections_z_dec,
                wrong_detections_z_non_dec=wrong_detections_z_non_dec,
                wrong_detections_x_dec=wrong_detections_x_dec,
                wrong_detections_x_non_dec=wrong_detections_x_non_dec,
                len_wrong_z_dec=len(wrong_detections_z_dec),
                len_wrong_z_non_dec=len(wrong_detections_z_non_dec),
                len_wrong_x_dec=len(wrong_detections_x_dec),
                len_wrong_x_non_dec=len(wrong_detections_x_non_dec),
                qber_z_dec=qber_z_dec,
                qber_z_non_dec=qber_z_non_dec,
                qber_x_dec=qber_x_dec,
                qber_x_non_dec=qber_x_non_dec,
                raw_key_rate=raw_key_rate,
                total_amount_detections=total_amount_detections,
                execution_time_run=execution_time_run
            )
        
        return len_wrong_x_dec, len_wrong_x_non_dec, len_wrong_z_dec, len_wrong_z_non_dec, len_Z_checked_dec, len_Z_checked_non_dec, X_P_calc_non_dec, X_P_calc_dec
        