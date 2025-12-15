from sklearn.model_selection import train_test_split
import xarray as xr
import zarr
import numpy as np
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
import itertools
import gc
import joblib
import torch
import pickle

import sys
from pathlib import Path
functions_path = Path("/home/mokr/Loss_Functions_Paper/ML_Functions/")
sys.path.append(str(functions_path))

import ML_functions

# Need to do it in 5 chunks so it all actually runs
fifth_length = 490
i = 4


ERA5_zarr_path = '/perm/mokr/Caravans_MultiMet/ERA5_LAND/timeseries.zarr'
HRES_zarr_path = '/perm/mokr/Caravans_MultiMet/HRES_Jan/HRES/timeseries.zarr'

ERA5_Land = xr.open_zarr(store=ERA5_zarr_path, chunks= None)
HRES = xr.open_zarr(store=HRES_zarr_path, chunks= None)


with open("/home/mokr/Loss_Functions_Paper/Catchment_Averaging/ML_basin_splits.pkl", "rb") as f:
    basin_splits = pickle.load(f)

train_basins, val_basins, test_basins = basin_splits['train'], basin_splits['val'], basin_splits['test']

Hind_variables = list(ERA5_Land.data_vars)
Fore_variables = list(HRES.data_vars)

start_date = "2016-01-01"
end_date = "2018-12-20"
valid_start_dates = pd.Series(pd.date_range(
    start=start_date, 
    end=pd.to_datetime(end_date) - pd.Timedelta(days=90), 
    freq='D'
))

# Create a DataLoader
history_sequence_length = 90
forecast_sequence_length = 10


ERA5_Land_scalers = joblib.load(f'/home/mokr/Loss_Functions_Paper/Scalers/ERA5_Land_Scalers.joblib')
scalers = ERA5_Land_scalers

Static_df = pd.read_csv(f'/home/mokr/Loss_Functions_Paper/Scaled/static_caravan_scalers.csv')
Discharge = pd.read_csv('/home/mokr/Loss_Functions_Paper/Scaled/discharge_scaled.csv')
Discharge = Discharge.set_index('date')
Discharge.index = pd.to_datetime(Discharge.index)

train_basins_section = train_basins[fifth_length*i:fifth_length*(i+1)]
# Training_Dataset = ML_functions.Multi_Basins_HF_LSTMDataGenerator(valid_start_dates = valid_start_dates, ERA5_Land = ERA5_Land, 
#                                   HRES=HRES, Static_df = Static_df , Discharge = Discharge, scalers = scalers, basin_indices=train_basins_section, Hind_variables=Hind_variables, Fore_variables = Fore_variables, 
#                                   history_sequence_length=history_sequence_length, forecast_sequence_length= forecast_sequence_length)

# Validation_Dataset = ML_functions.Multi_Basins_HF_LSTMDataGenerator(valid_start_dates = valid_start_dates, ERA5_Land = ERA5_Land, 
#                                   HRES=HRES, Static_df = Static_df , Discharge = Discharge, scalers = scalers, basin_indices=val_basins, Hind_variables=Hind_variables, Fore_variables = Fore_variables, 
#                                   history_sequence_length=history_sequence_length, forecast_sequence_length= forecast_sequence_length)



Training_Dataset = ML_functions.Multi_Basins_HF_LSTMDataGenerator_Binary(valid_start_dates = valid_start_dates, ERA5_Land = ERA5_Land, 
                                  HRES=HRES, Static_df = Static_df , Discharge = Discharge, scalers = scalers, basin_indices= train_basins_section, Hind_variables=Hind_variables, Fore_variables = Fore_variables, 
                                  history_sequence_length=history_sequence_length, forecast_sequence_length= forecast_sequence_length, p = 0.01)



Validation_Dataset_P0 = ML_functions.Multi_Basins_HF_LSTMDataGenerator_Binary(valid_start_dates = valid_start_dates, ERA5_Land = ERA5_Land, 
                                  HRES=HRES, Static_df = Static_df , Discharge = Discharge, scalers = scalers, basin_indices=val_basins, Hind_variables=Hind_variables, Fore_variables = Fore_variables, 
                                  history_sequence_length=history_sequence_length, forecast_sequence_length= forecast_sequence_length, p = 0)

Validation_Dataset_P1 = ML_functions.Multi_Basins_HF_LSTMDataGenerator_Binary(valid_start_dates = valid_start_dates, ERA5_Land = ERA5_Land, 
                                  HRES=HRES, Static_df = Static_df , Discharge = Discharge, scalers = scalers, basin_indices=val_basins, Hind_variables=Hind_variables, Fore_variables = Fore_variables, 
                                  history_sequence_length=history_sequence_length, forecast_sequence_length= forecast_sequence_length, p = 1)

Test_Dataset_P0 = ML_functions.Multi_Basins_HF_LSTMDataGenerator_Binary(valid_start_dates = valid_start_dates, ERA5_Land = ERA5_Land, 
                                  HRES=HRES, Static_df = Static_df , Discharge = Discharge, scalers = scalers, basin_indices=test_basins, Hind_variables=Hind_variables, Fore_variables = Fore_variables, 
                                  history_sequence_length=history_sequence_length, forecast_sequence_length= forecast_sequence_length, p = 0)

Test_Dataset_P1 = ML_functions.Multi_Basins_HF_LSTMDataGenerator_Binary(valid_start_dates = valid_start_dates, ERA5_Land = ERA5_Land, 
                                  HRES=HRES, Static_df = Static_df , Discharge = Discharge, scalers = scalers, basin_indices=test_basins, Hind_variables=Hind_variables, Fore_variables = Fore_variables, 
                                  history_sequence_length=history_sequence_length, forecast_sequence_length= forecast_sequence_length, p = 1)

from tqdm import tqdm

class FixedDataset(torch.utils.data.Dataset):
    def __init__(self, generator):
        self.data = []
        for item in tqdm(generator, desc=f"Processing dataset {i}", mininterval=60.0):
            self.data.append(item)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


Test_Dataset_P1 = FixedDataset(Test_Dataset_P1).data

save_path = "/perm/mokr/10Day_Loss_Function_Test_Dataset_Binary_P1.pt"

torch.save(Test_Dataset_P1, save_path)

# Delete the dataset
del Test_Dataset_P1

# Run garbage collection to free memory
gc.collect()




# Validation_Dataset_P0 = FixedDataset(Validation_Dataset_P0).data

# save_path = "/perm/mokr/10Day_Loss_Function_Validation_Dataset_Binary_P0.pt"
# torch.save(Validation_Dataset_P0, save_path)

# # Delete the dataset
# del Validation_Dataset_P0

# # Run garbage collection to free memory
# gc.collect()


# Validation_Dataset_P1 = FixedDataset(Validation_Dataset_P1).data

# save_path = "/perm/mokr/10Day_Loss_Function_Validation_Dataset_Binary_P1.pt"
# torch.save(Validation_Dataset_P1, save_path)

# # Delete the dataset
# del Validation_Dataset_P1

# # Run garbage collection to free memory
# gc.collect()


# Training_Dataset = FixedDataset(Training_Dataset).data

# save_path = f"/perm/mokr/10Day_Loss_Function_Training_Dataset{i}.pt"
# torch.save(Training_Dataset, save_path)

# print(f"Dataset saved to {save_path}")

# # Delete the dataset
# del Training_Dataset

# Run garbage collection to free memory
# gc.collect()

