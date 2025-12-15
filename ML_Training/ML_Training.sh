#!/bin/bash

#SBATCH --job-name=Analyses
#SBATCH --qos=nf
#SBATCH --mem-per-cpu=32Gb
#SBATCH --time=47:00:00
#SBATCH --output=/home/mokr/Loss_Functions_Paper/ML_Training/Analyses.out
#SBATCH --error=/home/mokr/Loss_Functions_Paper/ML_Training/Analyses_error.out

module load python3
python3 /home/mokr/Loss_Functions_Paper/ML_Training/Evaluation_Results.py