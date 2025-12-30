import numpy as np
# import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

import xarray as xr
import zarr

import random
import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler
from collections import defaultdict
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn as nn


import os
import itertools
import joblib

import sys
from pathlib import Path
functions_path = Path("/home/mokr/Loss_Functions_Paper/ML_Functions/")
sys.path.append(str(functions_path))

import ML_functions
import ML_Models
import ML_Losses
from ML_functions import HydroDataset

def load_and_prepare_data():
    import ML_functions

    # Open Zarr files
    ERA5_zarr_path = '/perm/mokr/Caravans_MultiMet/ERA5_LAND/timeseries.zarr'
    HRES_zarr_path = '/perm/mokr/Caravans_MultiMet/HRES_Jan/HRES/timeseries.zarr'
    
    ERA5_Land = xr.open_zarr(store=ERA5_zarr_path, chunks= "auto")
    HRES = xr.open_zarr(store=HRES_zarr_path, chunks="auto")

    Static_df = pd.read_csv(f'/home/mokr/Loss_Functions_Paper/Scaled/static_caravan_scalers.csv')
    # Get list of viable basins ad split them to training, validation, and test sets
    with open('/home/mokr/Loss_Functions_Paper/Caravans_2016_2019_Basins.txt', 'r') as file:
        Suitable_Basins = file.readlines()
        # This includes newline characters at the end of each line
        Suitable_Basins = [line.strip() for line in Suitable_Basins]  # removes newline characters
    
    Basins = set(path.split('/')[1] for path in Suitable_Basins)
    Basins_List = list(Basins)
    
    # Split into 80% training+val, 20% testing
    train_val_basins, test_basins = train_test_split(Basins_List, test_size=0.2, random_state=42)
    # Second split: Split the train_val set into 87.5% train, 12.5% val, so we get 70,10,20
    train_basins, val_basins = train_test_split(train_val_basins, test_size=0.125, random_state=42)
    
        
    # Static_scalers = joblib.load(f'/home/mokr/Loss_Functions_Paper/Catchment_Averaging/Scalers/static_caravan_scalers.joblib')
    # discharge_scalers = joblib.load(f'/home/mokr/Loss_Functions_Paper/Catchment_Averaging/Scalers/discharge_caravan_scalers.joblib')
    ERA5_Land_scalers = joblib.load(f'/home/mokr/Loss_Functions_Paper/Scalers/ERA5_Land_Scalers.joblib')
    
    Discharge = pd.read_csv('/home/mokr/Loss_Functions_Paper/Scaled/discharge_scaled.csv')
    Discharge = Discharge.set_index('date')
    Discharge.index = pd.to_datetime(Discharge.index)
    
    scalers = ERA5_Land_scalers

    return ERA5_Land, HRES, Static_df, Discharge, scalers, train_basins, val_basins, test_basins, ML_functions


def create_dataloaders(batch_size = 128):
    batch_size = 128

    loaded_traning_data0 = torch.load("/perm/mokr/Loss_Function_Training_Dataset0.pt")
    Training_Dataset0 = HydroDataset(loaded_traning_data0)
    
    # loaded_traning_data1 = torch.load("/perm/mokr/Loss_Function_Training_Dataset1.pt")
    # Training_Dataset1 = HydroDataset(loaded_traning_data1)
    
    # loaded_traning_data2 = torch.load("/perm/mokr/Loss_Function_Training_Dataset2.pt")
    # Training_Dataset2 = HydroDataset(loaded_traning_data2)

    combined_data = []
    datasets = [Training_Dataset0]
    # datasets = [Training_Dataset0, Training_Dataset1, Training_Dataset2]
    # Append data from each dataset to the combined list
    for dataset in datasets:
        combined_data.extend(dataset.data)
    
    Full_Training_Dataset = HydroDataset(combined_data)
    
    Training_Dataloader = DataLoader(Full_Training_Dataset, batch_size=batch_size, shuffle=True, pin_memory=True)

    loaded_data = torch.load("/perm/mokr/Loss_Function_Validation_Dataset.pt")
    Validation_Dataset = ML_functions.HydroDataset(loaded_data)
    Validation_Dataloader = DataLoader(Validation_Dataset, batch_size= 1, shuffle=False, pin_memory=True)


    return Training_Dataloader, Validation_Dataloader

def main():
    """Main function to orchestrate the workflow."""
    # print("Loading data...")
    # ERA5_Land, HRES, Static_df, Discharge, scalers, train_basins, val_basins, test_basins, ML_functions = load_and_prepare_data()

    print("Creating dataloaders...")
    Training_Dataloader, Validation_Dataloader = create_dataloaders()
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize model
    hindcast_hidden_size = 32
    forecast_hidden_size = 16
    handoff_hidden_size = 24
    G_Model = ML_Models.Google_Model_Block(25, 15, 8, 12, hindcast_hidden_size, forecast_hidden_size,  handoff_hidden_size, 1, device)
    optimizer = optim.Adam(G_Model.parameters(), lr=1e-4)
    criterion = ML_Losses.CRPSLoss()
    
    # Train model
    epochs = 20
    print('Starting training...')
    sys.stdout.flush()
    
    ML_functions.train_model_CMAL(G_Model, Training_Dataloader, optimizer, criterion, epochs = epochs, train_mode = True, batch_accumulation = 200, device = device)
    
    print('Training complete.')
    sys.stdout.flush()
    
    # Save model
    save_path = '/home/mokr/Loss_Functions_Paper/Catchment_Averaging/Models/G_Model_Deterministic_GPU_Test.pth'
    torch.save(G_Model, save_path)
    print(f"Model saved to {save_path}")
    
if __name__ == "__main__":
    main()
