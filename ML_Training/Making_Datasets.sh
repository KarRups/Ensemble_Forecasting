#!/bin/bash

#SBATCH --job-name=Making_DataloadersT
#SBATCH --output=/home/mokr/Loss_Functions_Paper/ML_Training/Output_Files/Making_Datasets_T.out
#SBATCH --error=/home/mokr/Loss_Functions_Paper/ML_Training/Output_Files/Making_Datasets_Errors_T.out
#SBATCH --qos=nf
#SBATCH --mem-per-cpu=32Gb
#SBATCH --time=42:00:00

module load python3
python3 /home/mokr/Loss_Functions_Paper/ML_Training/Making_Dataloaders.py