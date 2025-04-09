#!/bin/bash
#SBATCH -p einc
#SBATCH --job-name=sim_run
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=output_main_%j.log
#SBATCH --error=error_main_%j.log

# Run Python script inside the Singularity container
singularity exec --app dls /containers/stable/2025-02-19_1.img python ~/NeuMoQP/Programm/bachelor/code/main.py

