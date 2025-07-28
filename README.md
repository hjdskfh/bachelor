# QKD Simulation and SKR Analysis

## Overview
This project simulates a Quantum Key Distribution (QKD) protocol, processes the resulting data, and analyzes Secret Key Rates (SKR). For full details on the protocol and methodology, see the thesis that is in the folder thesis (Thesis_Lea_Bauer.pdf).

## Full Workflow

1. **Simulation Setup for SKR**
   - For cluster runs:
     - Edit `qkd_simulation_inputs.xlsx` to set the parameters you want to sweep for the QKD simulation.
     - Run `main_repeat_cluster_multiple_jobs` via the provided bash script on the cluster.
   - For local runs (PC):
     - Run `main_repeat_on_PC.py` directly to perform the simulation.

2. **Collect Results**
   - After simulation, results are saved in the cluster or PC. The log files with all the starting parameters are in the logs file, the results in the results folder.
   - Move or copy the results (e.g., `.npz`, `.json` files) into the `stuff_from_cluster/YYYY_MM_DD` folder.
   - You can do this by copying files manually, or by pushing everything on the cluster on GitHub and pulling it locally onto the PC.

3. **Evaluate Secret Key Rates**
   - Input the folder location `stuff_from_cluster/YYYY_MM_DD` in `calculate_evaluate_SKR_from_cluster.py` as input_file.
   - The script matches `.npz` and `.json` files, calculates SKR for each parameter set, and writes results to CSV and TXT files.
   - It also finds the best input parameters (highest SKR) and writes them to a `max_skr_summary_<timestamp>.csv` in the same folder.

4. **Aggregate Results**
   - Manually copy the best results from the `max_skr_summary` CSV into `final_skr_over_attenuation.xlsx` to accumulate and compare results across different runs.

## Project Structure

- `main_repeat_cluster_multiple_jobs.py`: Run QKD simulations on a cluster, sweeping parameters from `qkd_simulation_inputs.xlsx`.
- `main_repeat_on_PC.py`: Run QKD simulations locally on your PC.
- `calculate_evaluate_SKR_from_cluster.py`: Evaluate SKR from simulation results in a specified folder.
- `compare_SKR.py`: Compare SKR values across multiple result files.
- `qkd_simulation_inputs.xlsx`: Input parameters for cluster simulations.
- `final_skr_over_attenuation.xlsx`: Aggregate best SKR results from multiple runs.
- `stuff_from_cluster/`: Folder for storing simulation results.
- Bash scripts: For running jobs on the cluster.

## Usage Tips

- In `stuff_from_cluster` always ensure that the generated data from the simulation (`.npz` file) is numbered and this number matches the corresponding `.json`.
- Use the provided scripts for both cluster and PC runs.
- Aggregate best results manually in the Excel file for long-term comparison.

## File Descriptions

- `calculate_evaluate_SKR_from_cluster.py`: Main script for processing simulation data and calculating SKR.
- `compare_SKR.py`: Script for comparing SKR values across multiple result files.
- `requirements.txt`: Python dependencies.

## Requirements

- Python 3.x
- See `requirements.txt` for dependencies (e.g., numpy, pandas).



