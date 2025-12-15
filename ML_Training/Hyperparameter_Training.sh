#!/bin/bash

#SBATCH --qos=ng
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=20:00:00
#SBATCH --job-name=Hyperparameter_Training
#SBATCH --output=/home/mokr/Loss_Functions_Paper/ML_Training/Output_Files/Hyperparameter_Training_5May.out
#SBATCH --error=/home/mokr/Loss_Functions_Paper/ML_Training/Output_Files/Hyperparameter_Training_error_5May.out

module load python3
# python3 /home/mokr/Loss_Functions_Paper/ML_Training/GPU_Training_Binary.py
python3 /home/mokr/Loss_Functions_Paper/ML_Training/Hyperparameter_Training_Binary.py