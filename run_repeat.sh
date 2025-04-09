#!/bin/bash
#SBATCH -p einc
#SBATCH --job-name=sim_run
#SBATCH --mem=256G
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=output_repeat_%j.log
#SBATCH --error=error_repeat_%j.log

# Run Python script inside the Singularity container
singularity exec --app dls /containers/stable/2025-02-19_1.img python ~/NeuMoQP/Programm/bachelor/code/main_repeat_same_cluster.py

