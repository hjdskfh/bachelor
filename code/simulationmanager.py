import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import tqdm

from saver import Saver
from simulationengine import SimulationEngine

class SimulationManager:
    def __init__(self, config):
        self.config = config 
        self.simulation_engine = SimulationEngine(config)      
    
    def run_simulation(self):
        
        T1_dampening = self.simulation_engine.initialize()
        optical_power, peak_wavelength = self.simulation_engine.random_laser_output('current_power','voltage_shift', 'current_wavelength')
        
        basis, value, decoy = self.simulation_engine.generate_alice_choices_fixed(basis = 0, value = 0, decoy = 0)
        voltage_signal, t_jitter, _ = self.simulation_engine.signal_bandwidth_jitter(basis, value, decoy)
        _, _, _, transmission = self.simulation_engine.eam_transmission_1_mean_photon_number(voltage_signal, t_jitter, optical_power, peak_wavelength, T1_dampening, basis, value, decoy)    
        #print(f"wavelength photons: {wavelength_photons}, time photons: {time_photons}")

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
        optical_power, peak_wavelength = self.simulation_engine.random_laser_output('current_power', 'voltage_shift', 'current_wavelength')
        
        # Define the states and their corresponding arguments
        states = [
            {"title": "State: Z0", "basis": 1, "value": 1, "decoy": 0},
            {"title": "State: Z1", "basis": 1, "value": 0, "decoy": 0},
            {"title": "State: X+", "basis": 0, "value": -1, "decoy": 0},
            {"title": "State: Z0 decoy", "basis": 1, "value": 1, "decoy": 1},
            {"title": "State: Z1 decoy", "basis": 1, "value": 0, "decoy": 1},
            {"title": "State: X+ decoy", "basis": 0, "value": -1, "decoy": 1},
            ]
    
        # Iterate over each state
        for state in states:
            # Generate Alice's choices
            basis, value, decoy = self.simulation_engine.generate_alice_choices_fixed(basis=state["basis"], value=state["value"], decoy=state["decoy"])
            
            # Simulate signal and transmission
            voltage_signal, t_jitter, _ = self.simulation_engine.signal_bandwidth_jitter(basis, value, decoy)
            _, _, _, transmission = self.simulation_engine.eam_transmission_1_mean_photon_number(
                voltage_signal, t_jitter, optical_power, peak_wavelength, T1_dampening, basis, value, decoy
            )
            
           # wavelength_photons, time_photons, nr_photons = self.simulation_engine.eam_transmission_2_choose_photons(calc_mean_photon_nr, energy_pp, transmission, t_jitter)

            # Create the plot
            fig, ax = plt.subplots(figsize=(8, 5))
            ax2 = ax.twinx()  # Create a second y-axis

            # Plot voltage (left y-axis)
            ax.plot(t_jitter * 1e9, voltage_signal, color='blue', label='Voltage',linestyle='-', marker='o', markersize=1)
            ax.set_ylabel('Voltage (V)', color='blue')
            ax.tick_params(axis='y', labelcolor='blue')

            # Plot transmission (right y-axis)
            ax2.plot(t_jitter * 1e9, transmission, color='red', label='Transmission',linestyle='--', marker='o', markersize=1)
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

            nr_photons_arr = []
            time_arr = []
            wavelength_arr = []
            mean_photon_nr_arr = np.empty(self.config.n_samples + 1)

            for i in range(self.config.n_samples):
                if state["decoy"] == 0:
                    mean_photon_nr_arr[0] = self.config.mean_photon_nr
                else:
                    mean_photon_nr_arr[0] = self.config.mean_photon_decoy
                optical_power, peak_wavelength = self.simulation_engine.random_laser_output('current_power', 'voltage_shift', 'current_wavelength')
                
                # Generate Alice's choices
                basis, value, decoy = self.simulation_engine.generate_alice_choices_fixed(basis=state["basis"], value=state["value"], decoy=state["decoy"])
                
                # Simulate signal and transmission
                voltage_signal, t_jitter, _ = self.simulation_engine.signal_bandwidth_jitter(basis, value, decoy)
                _, calc_mean_photon_nr, energy_pp, transmission = self.simulation_engine.eam_transmission_1_mean_photon_number(
                    voltage_signal, t_jitter, optical_power, peak_wavelength, T1_dampening, basis, value, decoy
                )
                mean_photon_nr_arr[i+1] = calc_mean_photon_nr
                wavelength_photons, time_photons, nr_photons = self.simulation_engine.eam_transmission_2_choose_photons(calc_mean_photon_nr, energy_pp, transmission, t_jitter)
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
                    basis, value, decoy = self.simulation_engine.generate_alice_choices_fixed(basis=state["basis"], value=state["value"], decoy=state["decoy"])
                    
                    # Simulate signal and transmission
                    voltage_signal, t_jitter, _ = self.simulation_engine.signal_bandwidth_jitter(basis, value, decoy)
                    _, calc_mean_photon_nr, _, _ = self.simulation_engine.eam_transmission_1_mean_photon_number(
                        voltage_signal, t_jitter, optical_power, peak_wavelength, T1_dampening, basis, value, decoy)
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
    