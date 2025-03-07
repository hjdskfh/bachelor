import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import time
import os
import datetime

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


    
    def run_simulation_one_state(self):
        
        T1_dampening = self.simulation_engine.initialize()
        optical_power, peak_wavelength = self.simulation_single.random_laser_output_single('current_power','voltage_shift', 'current_wavelength')
        
        basis, value, decoy = self.simulation_single.generate_alice_choices_single(basis = 0, value = 0, decoy = 0)
        signals, t, _ = self.simulation_single.signal_bandwidth_jitter_single(basis, value, decoy)
        _, transmission = self.simulation_single.eam_transmission_single(signals, optical_power, T1_dampening)

        fig, ax = plt.subplots(figsize=(8, 5))
        ax2 = ax.twinx()  # Create a second y-axis
        
        # Plot voltage (left y-axis)
        ax.plot(t *1e9, signals, color='blue', label='Voltage', linestyle='-', marker='o', markersize=1)
        ax.set_ylabel('Voltage (V)', color='blue')
        ax.tick_params(axis='y', labelcolor='blue')

        # Plot transmission (right y-axis)
        ax2.plot(t * 1e9, transmission, color='red', label='Transmission', linestyle='-', marker='o', markersize=1)
        ax2.set_ylabel('Transmission', color='red')
        ax2.tick_params(axis='y', labelcolor='red')
    
        # Titles and labels
        ax.set_title(f"State: Z0")
        ax.set_xlabel('Time in ns')
        ax.grid(True)

        # Save or show the plot
        plt.tight_layout()
        Saver.save_plot('9_12_Z0dec_111aka1000d_voltage_and_transmission_for_4GHz_and_1e-11_jitter')

    def run_simulation_states(self):
        T1_dampening = self.simulation_engine.initialize()
        optical_power, peak_wavelength = self.simulation_single.random_laser_output_single('current_power', 'voltage_shift', 'current_wavelength')
        
        # Define the states and their corresponding arguments
        states = [
            {"title": "State: Z0", "basis": 1, "value": 1, "decoy": 0},
            {"title": "State: Z1", "basis": 1, "value": 0, "decoy": 0},
            {"title": "State: X+", "basis": 0, "value": -1, "decoy": 0},
            {"title": "State: Z0 decoy", "basis": 1, "value": 1, "decoy": 1},
            {"title": "State: Z1 decoy", "basis": 1, "value": 0, "decoy": 1},
            {"title": "State: X+ decoy", "basis": 0, "value": -1, "decoy": 1},
        ]
    
        def process_state(state):
            # Generate Alice's choices
            basis, value, decoy = self.simulation_single.generate_alice_choices_single(basis=state["basis"], value=state["value"], decoy=state["decoy"])
            
            # Simulate signal and transmission
            voltage_signal, t_jitter, signals = self.simulation_single.signal_bandwidth_jitter_single(basis, value, decoy)
            _, transmission = self.simulation_single.eam_transmission_single(voltage_signal, optical_power, T1_dampening)
            
            return state, t_jitter, voltage_signal, transmission, signals

        '''# Use ThreadPoolExecutor to process states in parallel
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(process_state, state) for state in states]
            results = [future.result() for future in futures]
            print(f"results len: {len(results)}")'''

        # Plotting results sequentially to avoid threading issues with Matplotlib
        #for result in enumerate(results): #, desc="parameters", unit="parameters", leave = False, position=0):
        for state in states:
            state, t_jitter, voltage_signal, transmission, signals = process_state(state)
            fig, ax = plt.subplots(figsize=(8, 5))
            ax2 = ax.twinx()  # Create a second y-axis

            # Plot voltage (left y-axis)
            ax.plot(t_jitter * 1e9, voltage_signal, color='blue', label='Voltage')
            ax.set_ylabel('Voltage (V)', color='blue')
            ax.tick_params(axis='y', labelcolor='blue')

            # Plot voltage (left y-axis)
            ax.plot(t_jitter * 1e9, signals, color='green', label='square voltage')
            ax.set_ylabel('Voltage (V)', color='green')
            ax.tick_params(axis='y', labelcolor='green')

            # Plot transmission (right y-axis)
            ax2.plot(t_jitter * 1e9, transmission, color='red', label='Transmission')
            ax2.set_ylabel('Transmission', color='red')
            ax2.tick_params(axis='y', labelcolor='red')


            # Titles and labels
            ax.set_title(state["title"])
            ax.set_xlabel('Time in ns')
            ax.grid(True)

            # Save or show the plot
            plt.tight_layout()
            Saver.save_plot(f"9_12_{state['title'].replace(' ', '_').replace(':', '').lower()}_voltage_and_transmission_for_4GHz_and_1e-11_jitter")

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

            optical_power, peak_wavelength = self.simulation_engine.random_laser_output('current_power', 'voltage_shift', 'current_wavelength')
            
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
                    optical_power, peak_wavelength = self.simulation_engine.random_laser_output('current_power', 'voltage_shift', 'current_wavelength')
                    
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
            '''
            plt.title(f" spread of mean photon number for {state['title'].replace(':', '').lower()} over {self.config.n_samples} iterations")                    
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
  
        optical_power, peak_wavelength = self.simulation_engine.random_laser_output('current_power', 'voltage_shift', 'current_wavelength')
        
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
        optical_power, peak_wavelength = self.simulation_engine.random_laser_output('current_power', 'voltage_shift', 'current_wavelength')
    
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
    
    def run_simulation_classificator(self):
        
        start_time = time.time()  # Record start time
        T1_dampening = self.simulation_engine.initialize()
        optical_power, peak_wavelength = self.simulation_engine.random_laser_output('current_power', 'voltage_shift', 'current_wavelength')

        # Create a histogram
        plt.hist(peak_wavelength *1e9, bins=10)  # bins=10 is just an example; adjust as needed

        # Label axes and give a title
        plt.xlabel('Peak Wavelength (nm)')
        plt.ylabel('Frequency')
        plt.title('Histogram of Peak Wavelengths')
        Saver.save_plot(f"hist_peak_wavelength")
    
        # Generate Alice's choices
        basis, value, decoy = self.simulation_engine.generate_alice_choices()

        # Simulate signal and transmission
        Saver.memory_usage("before simulating signal: " + str(time.time() - start_time))
        signals, t, _ = self.simulation_engine.signal_bandwidth_jitter(basis, value, decoy)

        amount_symbols_in_plot = 4
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

        time_simulating_signal = time.time() - start_time
        Saver.memory_usage("before eam: " + str(time_simulating_signal))
        power_dampened, norm_transmission,  calc_mean_photon_nr_eam, _ = self.simulation_engine.eam_transmission(signals, optical_power, T1_dampening, peak_wavelength, t)

        # plot so I can delete
        self.plotter.plot_and_delete_mean_photon_histogram(calc_mean_photon_nr_eam, target_mean_photon_nr = np.array([self.config.mean_photon_nr, self.config.mean_photon_decoy]), 
                                                type_photon_nr = "Mean Photon Number at EAM")


        self.plotter.plot_power(power_dampened, amount_symbols_in_plot=4, where_plot_1='after EAM')

        time_eam = time.time() - start_time
        Saver.memory_usage("before fiber: " + str(time_eam))
        power_dampened = self.simulation_engine.fiber_attenuation(power_dampened)
        
        self.plotter.plot_power(power_dampened, amount_symbols_in_plot=4, where_plot_1='after fiber')

        # first Z basis bc no interference
        Saver.memory_usage("before detector z: " + str(time.time() - start_time))
        power_dampened = power_dampened * self.config.p_z_bob
        time_photons_det_z, wavelength_photons_det_z, nr_photons_det_z, index_where_photons_det_z, calc_mean_photon_nr_detector_z, dark_count_times_z, num_dark_counts_z = self.simulation_engine.detector(t, norm_transmission, peak_wavelength, power_dampened, start_time)
        power_dampened = power_dampened / self.config.p_z_bob
        Saver.memory_usage("before classificator: " + str(time.time() - start_time))

        # path for X basis
        Saver.memory_usage("before DLI: " + str(time.time() -   start_time))
        power_dampened = power_dampened * (1 - self.config.p_z_bob)

        #plot
        amount_symbols_in_first_part = 6
        first_power = power_dampened[:amount_symbols_in_first_part]

        # DLI
        power_dampened = self.simulation_engine.delay_line_interferometer(power_dampened, t, peak_wavelength)

        # plot
        self.plotter.plot_power(power_dampened, amount_symbols_in_plot=amount_symbols_in_first_part, where_plot_1='before DLI',  shortened_first_power=first_power, where_plot_2='after DLI minus ohne i')

        Saver.memory_usage("before detector x: " + str(time.time() - start_time))
        time_photons_det_x, wavelength_photons_det_x, nr_photons_det_x, index_where_photons_det_x, calc_mean_photon_nr_detector_x, dark_count_times_x, num_dark_counts_x = self.simulation_engine.detector(t, norm_transmission, peak_wavelength, power_dampened, start_time)        

        # plot so I can delete
        self.plotter.plot_and_delete_mean_photon_histogram(calc_mean_photon_nr_detector_x, target_mean_photon_nr=None, type_photon_nr="Mean Photon Number at Detector X")
        self.plotter.plot_and_delete_mean_photon_histogram(calc_mean_photon_nr_detector_z, target_mean_photon_nr=None, type_photon_nr="Mean Photon Number at Detector Z")
        self.plotter.plot_and_delete_photon_wavelength_histogram_two_diagrams(wavelength_photons_det_x, wavelength_photons_det_z)
        self.plotter.plot_and_delete_photon_nr_histogram(nr_photons_det_x, nr_photons_det_z)
        
        # get results for both detectors
        len_wrong_detections_z, len_wrong_detections_x, total_amount_detections, amount_Z_detections, amount_XP_detections, qber, phase_error_rate, raw_key_rate, gain_XP_norm, gain_XP_dec, gain_Z_norm, gain_Z_dec, detected_indices_x_dec, detected_indices_x_norm, detected_indices_z_dec, detected_indices_z_norm = self.simulation_engine.classificator(t, time_photons_det_x, index_where_photons_det_x, 
                                                                                                            time_photons_det_z, index_where_photons_det_z, 
                                                                                                            basis, value, decoy)
        
        # plot so I can delete
        self.plotter.plot_and_delete_photon_time_histogram(time_photons_det_x, time_photons_det_z)
        

        Z1_sent_norm, Z1_sent_dec, Z0_sent_norm, Z0_sent_dec, XP_sent_norm, XP_sent_dec = self.simulation_helper.count_alice_choices(basis, value, decoy)

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
            Z0_sent_norm=Z0_sent_norm,
            Z0_sent_dec=Z0_sent_dec,
            Z1_sent_norm=Z1_sent_norm,
            Z1_sent_dec=Z1_sent_dec,
            XP_sent_norm=XP_sent_norm,
            XP_sent_dec=XP_sent_dec,
            execution_time_run=execution_time_run,
            time_simulating_signal=time_simulating_signal,
            time_eam=time_eam
        )
        '''detected_indices_x_dec=detected_indices_x_dec,
            detected_indices_x_norm=detected_indices_x_norm,
            detected_indices_z_dec=detected_indices_z_dec,
            detected_indices_z_norm=detected_indices_z_norm,'''
        
    def run_test(self):
        pass
        
        
        
        

        
        


