#!/bin/bash -l
#SBATCH --job-name=movie             # Job name
#SBATCH --output=./job%x.%j.out      # Name of stdout output file
#SBATCH --error=./job%x.%j.err       # Name of stderr error file
#SBATCH --partition=cosma8           # Partition (queue) name
#SBATCH --nodes=1                    # One node
#SBATCH --ntasks-per-node=128         # Number of cores (adjust based on num_proc)
#SBATCH --time=02:00:00              # Run time (hh:mm:ss)
#SBATCH --account=dp317              # Project for billing

# Load required modules
module purge
module load python/3.12.4

# Activate your Python environment
source /cosma/apps/dp317/dc-naza3/renv/bin/activate

# Run the movie generation script
cd /cosma8/data/dp317/dc-naza3/gasCloudNfw/plotting
python make_movie_weighted.py
