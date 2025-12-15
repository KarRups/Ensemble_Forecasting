#!/bin/bash

#SBATCH --qos=ng
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=10:30:00
#SBATCH --job-name=Hyperparameter_Training
#SBATCH --output=/home/mokr/Loss_Functions_Paper/ML_Training/Output_Files/NonBinary_Hyperparameter_Training.out
#SBATCH --error=/home/mokr/Loss_Functions_Paper/ML_Training/Output_Files/NonBinary_Hyperparameter_Training_error.out

module load python3

python3 /home/mokr/Loss_Functions_Paper/ML_Training/NonBinary_Hyperparameter_Training.py