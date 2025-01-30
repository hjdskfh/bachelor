import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

from saver import Saver
from simulationengine import SimulationEngine
from simulationsingle import SimulationSingle

class SimulationManager:
    def __init__(self, config):
        self.config = config 
        self.simulation_engine = SimulationEngine(config)
        self.simulation_single = SimulationSingle(config)
    
    def run_simulation(self):
        
        T1_dampening = self.simulation_engine.initialize()
        optical_power, peak_wavelength = self.simulation_engine.random_laser_output('current_power','voltage_shift', 'current_wavelength')
        
        basis, value, decoy = self.simulation_engine.generate_alice_choices(basis = 0, value = 0, decoy = 0, fixed = True)
        voltage_signal, t_jitter, _, _, _ = self.simulation_engine.signal_bandwidth_jitter(basis, value, decoy)
        _, transmission = self.simulation_engine.eam_transmission(voltage_signal, optical_power, T1_dampening)

        fig, ax = plt.subplots(figsize=(8, 5))
        ax2 = ax.twinx()  # Create a second y-axis
        
        # Plot voltage (left y-axis)
        ax.plot(t_jitter *1e9, voltage_signal, color='blue', label='Voltage', linestyle='-', marker='o', markersize=1)
        ax.set_ylabel('Voltage (V)', color='blue')
        ax.tick_params(axis='y', labelcolor='blue')

        # Plot transmission (right y-axis)
        ax2.plot(t_jitter * 1e9, transmission, color='red', label='Transmission', linestyle='-', marker='o', markersize=1)
        ax2.set_ylabel('Transmission', color='red')
        ax2.tick_params(axis='y', labelcolor='red')
    
        # Titles and labels
        ax.set_title(f"State: Z0 decoy")
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
            voltage_signal, t_jitter, signals, _, _ = self.simulation_single.signal_bandwidth_jitter_single(basis, value, decoy)
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
            ax.plot(t_jitter * 1e9, voltage_signal, color='blue', label='Voltage', lw = 0.1)
            ax.set_ylabel('Voltage (V)', color='blue')
            ax.tick_params(axis='y', labelcolor='blue')

            # Plot voltage (left y-axis)
            ax.plot(t_jitter * 1e9, signals, color='green', label='square voltage', lw = 0.1)
            ax.set_ylabel('Voltage (V)', color='green')
            ax.tick_params(axis='y', labelcolor='green')

            # Plot transmission (right y-axis)
            ax2.plot(t_jitter * 1e9, transmission, color='red', label='Transmission', lw = 0.1)
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
        'nicht fertig umgestellt von single auf double!'
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

            nr_photons_arr = []
            time_arr = []
            wavelength_arr = []
            mean_photon_nr_arr = np.empty(self.config.n_samples + 1)

            # 0tes Element ist baseline
            if state["decoy"] == 0:
                mean_photon_nr_arr[0] = self.config.mean_photon_nr
            else:
                mean_photon_nr_arr[0] = self.config.mean_photon_decoy
            optical_power, peak_wavelength = self.simulation_single.random_laser_output_single('current_power', 'voltage_shift', 'current_wavelength')
            
            # Generate Alice's choices
            basis, value, decoy = self.simulation_single.generate_alice_choices_single(basis=state["basis"], value=state["value"], decoy=state["decoy"], fixed = True)
            
            # Simulate signal and transmission
            voltage_signal, t_jitter, _, _, _ = self.simulation_single.signal_bandwidth_jitter_single(basis, value, decoy)
            power_dampened, transmission = self.simulation_single.eam_transmission_single(voltage_signal, optical_power, T1_dampening)
            calc_mean_photon_nr, wavelength_photons, time_photons, nr_photons, index_where_photons, all_time_max_nr_photons, sum_nr_photons_at_chosen = self.simulation_engine.choose_photons_single(power_dampened, transmission, 
                                                                                                                                                                    t_jitter, peak_wavelength)

            time_photons_det, wavelength_photons_det, nr_photons_det, index_where_photons_det, t_detector_jittered = self.simulation_engine.detector(t_jitter, wavelength_photons, time_photons, 
                                                                                                                    nr_photons, index_where_photons, all_time_max_nr_photons)
            mean_photon_nr_arr[i+1] = calc_mean_photon_nr
            if nr_photons != 0:
                wavelength_arr.extend([wavelength_photons])
                time_arr.extend([time_photons])
                nr_photons_arr.extend([nr_photons])
            else:
                nr_photons_arr.extend([nr_photons])

            plt.hist(mean_photon_nr_arr[1:], bins=40, alpha=0.7, label="Mean Photon Number")
            plt.axvline(mean_photon_nr_arr[0], color='red', linestyle='--', linewidth=2, label='target mean photon number')
            plt.title(f"{state['title'].replace(':', '').lower()}: mean photon number over {self.config.n_samples} iterations")
            plt.ylabel('iterations')
            plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))
            plt.xlabel(r'$\Delta \langle \mu \rangle$')
            plt.legend()
            plt.tight_layout()
            Saver.save_plot(f"hist_mean_photon_nr_{state['title'].replace(' ', '_').replace(':', '').lower()}")       

            plt.hist(nr_photons_arr, bins=np.arange(0, 11) - 0.5, alpha=0.7)
            plt.title(f"{state['title'].replace(':', '').lower()}: photon number over {self.config.n_samples} iterations")
            plt.xticks(np.arange(0, 11))  # Set x-ticks to be integers 
            plt.ylabel('iterations')
            plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))
            plt.xlabel('photon number')
            plt.tight_layout()
            Saver.save_plot(f"hist_nr_photons_{state['title'].replace(' ', '_').replace(':', '').lower()}")   

            if len(time_arr) == 0:
                plt.hist(time_arr, bins=40, alpha=0.7)
                plt.ylim(0, 10) 
            else:
                if len(time_arr) == 1:
                    time_arr = np.concatenate(time_arr)
                else:
                    time_arr = np.concatenate([arr.flatten() for arr in time_arr])
                time_arr = time_arr * 1e9
                plt.hist(time_arr, bins=40, alpha=0.7)
            plt.title(f"{state['title'].replace(':', '').lower()}: photon time over {self.config.n_samples} iterations")
            plt.ylabel('iterations')
            plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))
            plt.xlabel('photon time (ns)')
            plt.tight_layout()
            Saver.save_plot(f"hist_photon_time_{state['title'].replace(' ', '_').replace(':', '').lower()}")
            
            print(wavelength_arr)
            if len(wavelength_arr) == 0:
                plt.hist(wavelength_arr, bins=40, alpha=0.7)
                plt.ylim(0, 10) 
            else:
                if len(wavelength_arr) == 1:
                    wavelength_arr = np.concatenate(wavelength_arr)
                else:
                    wavelength_arr = np.concatenate([arr.flatten() for arr in wavelength_arr])
                wavelength_arr = wavelength_arr * 1e9
                plt.hist(wavelength_arr, bins=40, alpha=0.7)
            plt.title(f"{state['title'].replace(':', '').lower()}: photon wavelength over {self.config.n_samples} iterations")
            plt.ylabel('iterations')
            plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))
            plt.xlabel('photon wavelength (nm)')
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
                    voltage_signal, t_jitter, _, _, _ = self.simulation_engine.signal_bandwidth_jitter(basis, value, decoy)
                    power_dampened, transmission = self.simulation_engine.eam_transmission(voltage_signal, optical_power, T1_dampening)
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

        # Simulate signal and transmission
        voltage_signal, t_jitter, signals, t, jitter_shifts = self.simulation_engine.signal_bandwidth_jitter(basis, value, decoy)
        power_dampened, transmission = self.simulation_engine.eam_transmission(voltage_signal, optical_power, T1_dampening)
        power_dampened = self.simulation_engine.fiber_attenuation(power_dampened)
        power_dampened = self.simulation_engine.shift_jitter_to_bins(power_dampened, t, jitter_shifts, peak_wavelength)

        plt.plot(t * 1e9, power_dampened[0], color='blue', label='0', linestyle='-', marker='o', markersize=1)
        plt.plot(t * 1e9, power_dampened[1], color='green', label='1', linestyle='-', marker='o', markersize=1)
        Saver.save_plot(f"power_fiber")


        power_dampened = self.simulation_engine.delay_line_interferometer(power_dampened, t, peak_wavelength)
        #print(f"power_dampened: {power_dampened.shape()}")
        calc_mean_photon_nr, wavelength_photons, time_photons, nr_photons, index_where_photons, all_time_max_nr_photons, sum_nr_photons_at_chosen = self.simulation_engine.choose_photons(power_dampened, transmission, 
                                                                                                                                                                     t_jitter, peak_wavelength)
    
        time_photons_det, wavelength_photons_det, nr_photons_det, index_where_photons_det, t_detector_jittered = self.simulation_engine.detector(t_jitter, wavelength_photons, time_photons, 
                                                                                                                     nr_photons, index_where_photons, all_time_max_nr_photons)
        dark_count_times, num_dark_counts = self.simulation_engine.darkcount()

            
        # Plot the bar graph
        #plt.bar(index_where_photons - 0.5*bar_width, nr_photons, width=bar_width, label='fiber', color='blue')
        #plt.bar(index_where_photons_det + 0.5*bar_width, nr_photons_det, width=bar_width, label='detector', color='green')

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

        

    def run_simulation_dc(self):
        for i in range(self.config.n_samples):
            _, num_dark_counts = self.simulation_engine.darkcount()
            timer = 0
            if num_dark_counts > 0:
                timer = timer + 1

        print(f"min 1 dark count detected per run: {timer}")

    def run_simulation_initialize(self):
        T1_dampening = self.simulation_engine.initialize()
        print(f"T1 dampening: {T1_dampening}")
    
    def run_simulation_till_signal(self):
        T1_dampening = self.simulation_engine.initialize()
        optical_power, peak_wavelength = self.simulation_engine.random_laser_output('current_power','voltage_shift', 'current_wavelength')
        
        basis, value, decoy = self.simulation_engine.generate_alice_choices(basis = 0, value = 0, decoy = 0, fixed = True)
        filtered_signal, t_jittered, signals, t, jitter_shifts = self.simulation_engine.signal_bandwidth_jitter(basis, value, decoy)

        print(f"second_signal: {signals[0:100]}")
        fig, ax = plt.subplots(figsize=(8, 5))
        ax2 = ax.twinx()  # Create a second y-axis

        # Plot voltage (left y-axis)
        ax.plot(t_jittered *1e9, signals, color='blue', label='np.multi', linestyle='-', marker='o', markersize=1)
        ax.set_ylabel('Voltage (V)', color='blue')
        ax.tick_params(axis='y', labelcolor='blue')

        # Plot transmission (right y-axis)
        ax2.plot(t_jittered * 1e9, filtered_signal, color='red', label='normal multi', linestyle='-', marker='o', markersize=1)
        ax2.set_ylabel('Voltage (V)', color='red')
        ax2.tick_params(axis='y', labelcolor='red')

        # Titles and labels
        ax.set_title(f"State: Z0 decoy")
        ax.set_xlabel('Time in ns')
        ax.grid(True)

        # Save or show the plot
        plt.tight_layout()
        plt.show()
        #Saver.save_plot('9_12_Z0dec_111aka1000d_voltage_and_transmission_for_4GHz_and_1e-11_jitter')