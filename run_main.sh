#!/bin/bash
#SBATCH --job-name=sim_run                # Job name
#SBATCH --output=output_main_%j.log               # Stdout log file
#SBATCH --error=error_main_%j.log                 # Stderr log file
#SBATCH --partition=einc                  # Partition to run on
#SBATCH -c 128                            # Number of cores
#SBATCH --mem=950G                        # Memory you request
#SBATCH --time=04:00:00                   # Time limit (hh:mm:ss)

# Run Python script inside the Singularity container
singularity exec --app dls /containers/stable/2025-02-19_1.img python ~/NeuMoQP/Programm/bachelor/code/main.py
