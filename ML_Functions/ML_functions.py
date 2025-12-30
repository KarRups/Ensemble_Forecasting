"""
FILE CONTENTS
================================================================================

CLASSES:
--------------------------------------------------------------------------------
 1. HydroDataset
    No description available
    Methods: __init__, __len__, __getitem__
    
 2. Multi_Basins_HF_LSTMDataGenerator
    No description available
    Methods: __init__, __len__, __getitem__
 3. Multi_Basins_HF_LSTMDataGenerator_Binary
    No description available
    Methods: __init__, __len__, __getitem__

4. CRPSLoss_Ensemble
    Continuous Ranked Probability Score (CRPS) implemented as a PyTorch loss func...
    Methods: __init__, smooth_heaviside, forward
5. Discrete_CRPSLoss
    Continuous Ranked Probability Score (CRPS) implemented as a PyTorch loss func...
    Methods: __init__, forward

FUNCTIONS:
--------------------------------------------------------------------------------
 1. load_scalers
    Load saved scalers
 2. load_dataloaders
    No description available
 3. process_era5_land_data
    Selects, converts, and scales ERA5 Land data in a single function.
 4. process_hres_data
    Selects, converts, and scales HRES forecast data in a single function.
 5. prepare_masked_discharge
    Prepare masked discharge with probabilistic masking.
    
    No description available
 7. train_model
    Training function for PyTorch model with gradient accumulation.
 8. train_model_CMAL
    Training function for PyTorch model with gradient accumulation.
 9. train_seeded_model
    Train a model using ensemble predictions and batch accumulation.
    
10. run_ensemble_predictions
    Run model predictions with an additional row of random noise each time.
11. calculate_kge
    Calculate Kling-Gupta Efficiency for timeseries data.
12. calculate_crps
    Computes CRPS from x using y as reference,
13. transform_CMAL_parameters
    Transform a tensor by applying:
14. transform_CMAL_parameters_multi
    Generates predictions from a tensor representing multiple Asymetric Laplacians
15. load_and_unnormalize
    Load scaler and normalize ensemble predictions and true discharge.
16. replace_keys
    Replace keys in a dictionary according to key_map.
17. print_cuda_memory_summary
    No description available
================================================================================
"""

import time
import itertools
import joblib
import sys
import gc
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
from torch.utils.data import Dataset
import warnings
import sys
from pathlib import Path
from ML_Losses import get_member_summaries, compute_crps, get_member_summaries_torch

functions_path = Path("/home/mokr/Loss_Functions_Paper/ML_Functions/")

class HydroDataset(Dataset):
    def __init__(self, data_list):
        self.data = data_list  # List containing [Hist_X_Chunk_Torch, Fore_X_Chunk_Torch, Y_value, end_prediction_date, basin_idx]
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]  # Returns the tuple of 5 elements at position idx


class Multi_Basins_HF_LSTMDataGenerator(Dataset):
    def __init__(self, valid_start_dates, ERA5_Land, HRES, Static_df, Discharge, scalers, basin_indices, Hind_variables, Fore_variables, history_sequence_length, forecast_sequence_length):
        """
        Args:
            valid_start_dates (list or array): List of valid start dates.
            ERA5_Land (xarray.Dataset): ERA5 Land dataset.
            Caravans (pd.DataFrame): DataFrame with additional features.
            Discharge (pd.Series or dict): Discharge values indexed by date.
            basin_index (int): Index of the basin to select.
            variables (list): List of variables to select from ERA5_Land.
            sequence_length (int): Length of the input sequence. Default is 90.
        """
        self.valid_start_dates = valid_start_dates
        self.ERA5_Land = ERA5_Land
        self.HRES = HRES
        self.basin_indices = basin_indices
        self.Hind_variables = Hind_variables
        self.Fore_variables = Fore_variables
        self.history_sequence_length = history_sequence_length
        self.forecast_sequence_length = forecast_sequence_length
        self.scalers = scalers
        self.Static_df = Static_df
        self.Discharge = Discharge

        # Create all possible combinations of start dates and basin indices
        self.combinations = list(itertools.product(self.basin_indices,
            range(len(self.valid_start_dates)) 
        ))

    def __len__(self):
        return len(self.combinations)

    def __getitem__(self, idx):

        start_processing_dates = time.time()
        # Get the date index and basin index from the combinations
        basin_idx, date_idx  = self.combinations[idx]
        location = basin_idx.split('_')[0]

        start_date = self.valid_start_dates.iloc[date_idx]
        end_date = start_date + pd.Timedelta(days=self.history_sequence_length - 1)
        start_prediction_date = end_date + pd.Timedelta(days=1)
        end_prediction_date = start_prediction_date + pd.Timedelta(days=self.forecast_sequence_length - 1)

        ERA5_Land_scalers = self.scalers

        start_era5_processing = time.time()
        
        start_era5_selection = time.time()
        ERA5L_xr = self.ERA5_Land.sel(basin=basin_idx, date=slice(start_date, end_date))
        era5_selection_time = time.time() - start_era5_selection
        
        # Step 2: Converting to numpy array
        start_era5_to_numpy = time.time()
        ERA5L_np = ERA5L_xr[self.Hind_variables].to_array().T.values
        era5_to_numpy_time = time.time() - start_era5_to_numpy
        
        # Step 3: Preparing for scaling
        ERA5L_scaled = np.zeros_like(ERA5L_np)
        
        # Step 4: Applying scalers to each variable
        start_era5_scaling = time.time()
        for i, var in enumerate(self.Hind_variables):
            # Check if a scaler exists for this variable
            if var in ERA5_Land_scalers:
                # Apply scaler to the column
                ERA5L_scaled[:, i] = ERA5_Land_scalers[var].transform(ERA5L_np[:, i].reshape(-1, 1)).flatten()
        era5_scaling_time = time.time() - start_era5_scaling

        era5_processing_time = time.time() - start_era5_processing
        
        start_static_and_discharge = time.time()
        
        Static_df_Row = self.Static_df[self.Static_df['gauge_id'] == basin_idx]
        Static_df_Row = Static_df_Row.drop(axis = 1, columns =  'gauge_id')

        Static_history_array = np.tile(Static_df_Row, (self.history_sequence_length, 1))
        Static_forecast_array = np.tile(Static_df_Row, (self.forecast_sequence_length, 1))

        Discharge = self.Discharge[f'{basin_idx}']
        Discharge_Hist = np.array(Discharge[start_date:end_date])
        Fore_Discharge = Discharge[start_prediction_date : end_prediction_date].values.reshape(-1, 1).flatten()

        static_and_discharge_time = time.time() - start_static_and_discharge
        start_hres_processing = time.time()
        
        # Extract and process HRES data for future dates
        HRES_Basin = self.HRES.sel(basin=basin_idx, date= end_date)
        HRES_xr = HRES_Basin.isel(lead_time = slice(0, self.forecast_sequence_length))
        HRES_np = HRES_xr[self.Fore_variables].to_array().T.values

        HRES_scaled = np.zeros_like(HRES_np)

        for i, var in enumerate(self.Fore_variables):            
            short_var = var.split('_', 1)[1] if '_' in var else var
            # Check if a scaler exists for this variable
            for scaler_key in ERA5_Land_scalers:
                # Extract the part after first underscore from the scaler key
                short_scaler_key = scaler_key.split('_', 1)[1] if '_' in scaler_key else scaler_key
                
                # If the short versions match, we found our scaler
                if short_var == short_scaler_key:
                    matching_key = scaler_key
                    break
                
            # Apply the scaler if found
            if matching_key:
                HRES_scaled[:, i] = ERA5_Land_scalers[matching_key].transform(HRES_np[:, i].reshape(-1, 1)).flatten()

        hres_processing_time = time.time() - start_hres_processing
        start_converting_to_torch = time.time()
        
        Hist_X_Chunk_np = np.concatenate((ERA5L_scaled, Static_history_array), axis=1)
        Hist_X_Chunk_Torch = torch.from_numpy(Hist_X_Chunk_np).float()

        Fore_X_Chunk_np = np.concatenate((HRES_scaled, Static_forecast_array), axis=1)
        Fore_X_Chunk_Torch = torch.from_numpy(Fore_X_Chunk_np).float()
        
        # Get the target discharge value, from Basin_Discharge
        Y_value = torch.tensor(Fore_Discharge).float()
        
        converting_to_torch_time = time.time() - start_converting_to_torch
        total_batch_processing_time = time.time() - start_processing_dates
        
        return Hist_X_Chunk_Torch, Fore_X_Chunk_Torch, Y_value, str(end_prediction_date), basin_idx


class Multi_Basins_HF_LSTMDataGenerator_Binary(Dataset):
    def __init__(self, valid_start_dates, ERA5_Land, HRES, Static_df, Discharge, scalers, basin_indices, Hind_variables, Fore_variables, history_sequence_length, forecast_sequence_length, p = 0.125):
        """
        Args:
            valid_start_dates (list or array): List of valid start dates.
            ERA5_Land (xarray.Dataset): ERA5 Land dataset.
            Caravans (pd.DataFrame): DataFrame with additional features.
            Discharge (pd.Series or dict): Discharge values indexed by date.
            basin_index (int): Index of the basin to select.
            variables (list): List of variables to select from ERA5_Land.
            sequence_length (int): Length of the input sequence. Default is 90.
            p (int 0:1): probability of future discharge being given on a given day
        """
        self.valid_start_dates = valid_start_dates
        self.ERA5_Land = ERA5_Land
        self.HRES = HRES
        self.basin_indices = basin_indices
        self.Hind_variables = Hind_variables
        self.Fore_variables = Fore_variables
        self.history_sequence_length = history_sequence_length
        self.forecast_sequence_length = forecast_sequence_length
        self.scalers = scalers
        self.Static_df = Static_df
        self.Discharge = Discharge
        self.p = p
        
        # Create all possible combinations of start dates and basin indices
        self.combinations = list(itertools.product(self.basin_indices,
            range(len(self.valid_start_dates)) 
        ))

    def __len__(self):
        return len(self.combinations)

    def __getitem__(self, idx):

        start_processing_dates = time.time()
        # Get the date index and basin index from the combinations
        basin_idx, date_idx  = self.combinations[idx]
        location = basin_idx.split('_')[0]

        start_date = self.valid_start_dates.iloc[date_idx]
        end_date = start_date + pd.Timedelta(days=self.history_sequence_length - 1)
        start_prediction_date = end_date + pd.Timedelta(days=1)
        end_prediction_date = start_prediction_date + pd.Timedelta(days=self.forecast_sequence_length - 1)

        ERA5L_scaled = process_era5_land_data(self.ERA5_Land, basin_idx, start_date, end_date, 
            self.Hind_variables, self.scalers)
        
        Static_df_Row = self.Static_df[self.Static_df['gauge_id'] == basin_idx]
        Static_df_Row = Static_df_Row.drop(axis = 1, columns =  'gauge_id')

        Static_history_array = np.tile(Static_df_Row, (self.history_sequence_length, 1))
        Static_forecast_array = np.tile(Static_df_Row, (self.forecast_sequence_length, 1))

        Discharge = self.Discharge[f'{basin_idx}']
        Discharge_Hist = np.array(Discharge[start_date:end_date])
        Fore_Discharge = Discharge[start_prediction_date : end_prediction_date].values.reshape(-1, 1).flatten()
        masked_discharge, mask = prepare_masked_discharge(Fore_Discharge, p= self.p)

        HRES_scaled = process_hres_data(self.HRES, basin_idx, end_date, 
            self.forecast_sequence_length, self.Fore_variables, self.scalers)
                
        Hist_X_Chunk_np = np.concatenate((ERA5L_scaled, Static_history_array), axis=1)
        Hist_X_Chunk_Torch = torch.from_numpy(Hist_X_Chunk_np).float()


        # Concatenate with existing arrays
        Fore_X_Chunk_np = np.concatenate([
            HRES_scaled, 
            Static_forecast_array, 
            masked_discharge.reshape(-1, 1),  # Add masked discharge as a new column
            mask.reshape(-1, 1)               # Add mask as a new column
        ], axis=1)

        Fore_X_Chunk_Torch = torch.from_numpy(Fore_X_Chunk_np).float()
        
        # Get the target discharge value, from Basin_Discharge
        Y_value = torch.tensor(Fore_Discharge).float()
        
        return Hist_X_Chunk_Torch, Fore_X_Chunk_Torch, Y_value, str(end_prediction_date), basin_idx

class CRPSLoss_Ensemble(nn.Module):
    """
    Continuous Ranked Probability Score (CRPS) implemented as a PyTorch loss function.
    
    This loss function evaluates the quality of probabilistic forecasts by comparing
    the forecasted distribution with the observed values.
    
    Lower CRPS values indicate better forecasts.
    """
    
    def __init__(self, clip_value=-0.26787253, smooth_epsilon=0.01):
        """
        Initialize the CRPS loss function.
        
        Args:
            clip_value (float): Value to clip forecasts at. Defaults to -0.26787253.
            smooth_epsilon (float): Parameter for smoothing the heaviside function.
        """
        super(CRPSLoss_Ensemble, self).__init__()
        self.clip_value = clip_value
        self.smooth_epsilon = smooth_epsilon
        
    def smooth_heaviside(self, x, epsilon=0.01):
        """
        Smooth approximation of the heaviside function that maintains gradients.
        
        Args:
            x: Input tensor
            epsilon: Smoothing parameter (smaller = sharper transition)
            
        Returns:
            Smoothed approximation of the heaviside function
        """
        return torch.sigmoid(x / epsilon)        
    def forward(self, forecasts, observation):
        """
        Calculate the Continuous Ranked Probability Score (CRPS).
        
        Args:
            forecasts: Tensor of ensemble forecasts with shape (n_members, batch_size, num_steps)
            observation: Tensor of true observations with shape (batch_size, num_steps)
            
        Returns:
            Tensor of CRPS scores with shape (batch_size, num_steps)
        """
        # Clip forecasts if needed
        forecasts = torch.clamp(forecasts, min=self.clip_value)
        
        # Get dimensions
        n_members, batch_size, num_steps = forecasts.shape
        
        # Reshape forecasts to (batch_size * num_steps, n_members)
        forecasts_reshaped = forecasts.permute(1, 2, 0).reshape(batch_size * num_steps, n_members)
        
        # Sort the forecasts along the ensemble dimension
        sorted_forecasts, _ = torch.sort(forecasts_reshaped, dim=1)
        
        # Reshape observations to match forecasts
        observations_flat = observation.reshape(batch_size * num_steps, 1)
        observations_expanded = observations_flat.expand(-1, n_members)
        
        # Calculate heaviside function (indicator of whether forecast >= observation)
        heaviside  = self.smooth_heaviside(sorted_forecasts - observations_expanded, self.smooth_epsilon)
        # Create ranks and normalize
        ranks = torch.arange(1, n_members + 1, device=forecasts.device, dtype=torch.float) / n_members
        
        # Calculate CRPS
        crps_flat = torch.mean((heaviside - ranks)**2, dim=1)
        
        # Reshape back to (batch_size, num_steps)
        crps_scores = crps_flat.reshape(batch_size, num_steps)
        
        return crps_scores


class Discrete_CRPSLoss(nn.Module):
    """
    Continuous Ranked Probability Score (CRPS) implemented as a PyTorch loss function.
    
    This loss function evaluates the quality of probabilistic forecasts by comparing
    the forecasted distribution with the observed values.
    
    Lower CRPS values indicate better forecasts.
    """
    
    def __init__(self, clip_value=-0.26787253, epsilon= 1e-4):
        """
        Initialize the CRPS loss function.
        
        Args:
            clip_value (float): Value to clip forecasts at. Defaults to -0.26787253.
            smooth_epsilon (float): Parameter for smoothing the heaviside function.
        """
        super(Discrete_CRPSLoss, self).__init__()
        self.clip_value = clip_value
        self.epsilon = epsilon
          
    def forward(self, forecasts, observation):
        """
        Calculate the Continuous Ranked Probability Score (CRPS).
        
        Args:
            forecasts: Tensor of ensemble forecasts with shape (n_members, batch_size, num_steps)
            observation: Tensor of true observations with shape (batch_size, num_steps)
            
        Returns:
            Tensor of CRPS scores with shape (batch_size, num_steps)
        """
        # Clip forecasts if needed
        forecasts = torch.clamp(forecasts, min=self.clip_value)
        
        # Get dimensions
        n_members, batch_size, num_steps = forecasts.shape
        
        # Reshape forecasts to (batch_size * num_steps, n_members)
        forecasts_reshaped = forecasts.permute(1, 2, 0)

        observations_expanded = observation.unsqueeze(-1)
        
        term1 = torch.abs(forecasts_reshaped - observations_expanded).mean(dim= -1)
 
        x_j = forecasts_reshaped.unsqueeze(3)  # (batch_size, num_steps, n_members, 1)
        x_k = forecasts_reshaped.unsqueeze(2)  # (batch_size, num_steps, 1, n_members)
        pairwise_diffs = torch.abs(x_j - x_k)
        term2 = pairwise_diffs.sum(dim=(2, 3)) / (2 * n_members * (n_members - 1))
        crps_scores = term1 - ((1 - self.epsilon)* term2)
        return crps_scores


def load_scalers(output_path = 'scalers'):
    """Load saved scalers"""
    scalers = joblib.load(f'{output_path}.joblib')
    return scalers


def load_dataloaders(batch_size = 128, history_sequence_length = 90, forecast_sequence_length = 10, ML_functions = None):
    batch_size = 128
    
    # loaded_traning_data0 = torch.load("/perm/mokr/10Day_Loss_Function_Training_Dataset0_test.pt")
    # Training_Dataset0 = HydroDataset(loaded_traning_data0)
    
    # loaded_traning_data1 = torch.load("/perm/mokr/10Day_Loss_Function_Training_Dataset1_test.pt")
    # Training_Dataset1 = HydroDataset(loaded_traning_data1)
    
    # loaded_traning_data2 = torch.load("/perm/mokr/10Day_Loss_Function_Training_Dataset2_test.pt")
    # Training_Dataset2 = HydroDataset(loaded_traning_data2)

    # loaded_traning_data3 = torch.load("/perm/mokr/10Day_Loss_Function_Training_Dataset3_test.pt")
    # Training_Dataset3 = HydroDataset(loaded_traning_data3)
    
    loaded_traning_data4 = torch.load("/perm/mokr/10Day_Loss_Function_Training_Dataset4_test.pt")
    Training_Dataset4 = HydroDataset(loaded_traning_data4)

    combined_data = []
    datasets = [Training_Dataset4]
    # datasets = [Training_Dataset0, Training_Dataset1, Training_Dataset2, Training_Dataset3 , Training_Dataset4]

    # Append data from each dataset to the combined list
    for dataset in datasets:
        combined_data.extend(dataset.data)
    
    Full_Training_Dataset = HydroDataset(combined_data)
    
    
    Training_Dataloader = DataLoader(Full_Training_Dataset, batch_size=batch_size, shuffle=True, pin_memory=True)

    loaded_data = torch.load("/perm/mokr/10Day_Loss_Function_Validation_Dataset_Binary_P0_test.pt")
    Validation_Dataset = ML_functions.HydroDataset(loaded_data)
    # Make batch_size bigger later
    Validation_Dataloader = DataLoader(Validation_Dataset, batch_size= batch_size, shuffle=False, pin_memory=True)


    return Training_Dataloader, Validation_Dataloader


def process_era5_land_data(era5_land, basin_idx, start_date, end_date, hind_variables, scalers):
    """
    Selects, converts, and scales ERA5 Land data in a single function.
    
    Args:
        era5_land (xarray.Dataset): Original ERA5 Land dataset
        basin_idx (str): Basin index to select
        start_date (pd.Timestamp): Start date for data selection
        end_date (pd.Timestamp): End date for data selection
        hind_variables (list): List of historical variables to process
        scalers (dict): Dictionary of scalers for each variable
    
    Returns:
        np.ndarray: Scaled ERA5 Land data
    """
    # Select data for specific basin and date range
    ERA5L_xr = era5_land.sel(basin=basin_idx, date=slice(start_date, end_date))
    
    # Convert to numpy array
    ERA5L_np = ERA5L_xr[hind_variables].to_array().T.values
    
    # Prepare scaled array
    ERA5L_scaled = np.zeros_like(ERA5L_np)
    
    # Apply scalers to each variable
    for i, var in enumerate(hind_variables):
        # Check if a scaler exists for this variable
        if var in scalers:
            # Apply scaler to the column
            ERA5L_scaled[:, i] = scalers[var].transform(
                ERA5L_np[:, i].reshape(-1, 1)
            ).flatten()
    
    return ERA5L_scaled


def process_hres_data(hres, basin_idx, end_date, forecast_sequence_length, fore_variables, era5_land_scalers):
    """
    Selects, converts, and scales HRES forecast data in a single function.
    
    Args:
        hres (xarray.Dataset): HRES dataset
        basin_idx (str): Basin index to select
        end_date (pd.Timestamp): End date for historical data
        forecast_sequence_length (int): Length of forecast sequence
        fore_variables (list): List of forecast variables
        era5_land_scalers (dict): Scalers for ERA5 Land variables
    
    Returns:
        np.ndarray: Scaled HRES forecast data
    """
    # Extract HRES data for the specific basin and end date
    HRES_Basin = hres.sel(basin=basin_idx, date=end_date)
    HRES_xr = HRES_Basin.isel(lead_time=slice(0, forecast_sequence_length))
    HRES_np = HRES_xr[fore_variables].to_array().T.values
    
    # Prepare scaled array
    HRES_scaled = np.zeros_like(HRES_np)
    
    # Process and scale each variable
    for i, var in enumerate(fore_variables):            
        # Extract short variable name (part after underscore or full name)
        short_var = var.split('_', 1)[1] if '_' in var else var
        
        # Find matching scaler key
        matching_key = next(
            (scaler_key for scaler_key in era5_land_scalers 
             if (scaler_key.split('_', 1)[1] if '_' in scaler_key else scaler_key) == short_var),
            None
        )
        
        # Apply the scaler if found
        if matching_key:
            HRES_scaled[:, i] = era5_land_scalers[matching_key].transform(
                HRES_np[:, i].reshape(-1, 1)
            ).flatten()
    
    return HRES_scaled


def prepare_masked_discharge(Fore_Discharge, p=0.125):
    """
    Prepare masked discharge with probabilistic masking.
    
    Args:
        Fore_Discharge (np.ndarray): Original forecast discharge values
        p (float): Probability of showing the true discharge value (default 0.125)
    
    Returns:
        tuple: Masked discharge array and mask array
    """
    # Create a random mask based on probability p
    mask = np.random.random(Fore_Discharge.shape) < p
    
    # Create masked discharge array
    masked_discharge = np.where(mask, Fore_Discharge, 0)
    
    return masked_discharge, mask.astype(float)




def train_model(Model, Dataloader, optimizer, criterion, epochs=3, train_mode = True, batch_accumulation=50, device = 'cpu', scheduler = None):
    """
    Training function for PyTorch model with gradient accumulation.
    
    Args:
        G_Model: PyTorch model to train
        Multi_HF_dataloader: DataLoader containing training data
        optimizer: PyTorch optimizer
        criterion: Loss function
        epochs: Number of training epochs
        batch_accumulation: Number of batches to accumulate before updating weights
        max_grad_norm: Maximum gradient norm for gradient clipping (optional)
    
    Returns:
        List of average losses per epoch
    """
    epoch_losses = []
    Model.to(device)
    print(f"Using device: {device}")
    Model.train(train_mode)

    for epoch in range(epochs):
        i = 0
        accumulated_batches = 0
        average_loss = 0

        for Hist_X_Chunk_Torch, Fore_X_Chunk_Torch, Y_value, end_prediction_date, basin_idx in Dataloader:

            i += 1
            accumulated_batches += 1

            
            Hist_X_Chunk_Torch = Hist_X_Chunk_Torch.to(device=device, dtype=torch.float32)
            Fore_X_Chunk_Torch = Fore_X_Chunk_Torch.to(device=device, dtype=torch.float32)
            Y_value = Y_value.to(device=device, dtype=torch.float32)

            # Forward pass
            prediction = Model(
                Hist_X_Chunk_Torch, 
                Fore_X_Chunk_Torch
            )[1]

            loss = criterion(prediction.squeeze(), Y_value)

            optimizer.step()
            optimizer.zero_grad()

            
            average_loss += loss.item()

            if accumulated_batches % batch_accumulation == 0:        
                print(f'Epoch {epoch+1}/{epochs}, Batch {i}/{len(Dataloader)}, '
                      f'Average loss: {average_loss/batch_accumulation:.4f}')

                sys.stdout.flush()
                
                average_loss = 0
                accumulated_batches = 0
        if scheduler is not None:
            scheduler.step()
            
    return epoch_losses


def train_model_CMAL(Model, Dataloader, optimizer, criterion, epochs=3, train_mode = True, batch_accumulation=50, device = 'cpu', scheduler = None, use_observations = True):
    """
    Training function for PyTorch model with gradient accumulation.
    
    Args:
        G_Model: PyTorch model to train
        Multi_HF_dataloader: DataLoader containing training data
        optimizer: PyTorch optimizer
        criterion: Loss function
        epochs: Number of training epochs
        batch_accumulation: Number of batches to accumulate before updating weights
        max_grad_norm: Maximum gradient norm for gradient clipping (optional)
    
    Returns:
        List of average losses per epoch
    """
    epoch_losses = []

    all_conditional_ensemble_summaries = []
    all_greedy_ensemble_summaries = []
    all_discharge_summaries = []
    CRPS_Per_leadtime = []
    

    Model.to(device)
    print(f"Using device: {device}")
    Model.train(train_mode)

    
    for epoch in range(epochs):
        i = 0
        accumulated_batches = 0
        average_loss = 0
        epoch_loss = 0
        for Hist_X_Chunk_Torch, Fore_X_Chunk_Torch, Y_value, end_prediction_date, basin_idx in Dataloader:
            if not use_observations:
                Fore_X_Chunk_Torch = Fore_X_Chunk_Torch[:,:, 0:15]
            i += 1
            accumulated_batches += 1
            loss = 0 
            
            Hist_X_Chunk_Torch = Hist_X_Chunk_Torch.to(device=device, dtype=torch.float32)
            Fore_X_Chunk_Torch = Fore_X_Chunk_Torch.to(device=device, dtype=torch.float32)
            Y_value = Y_value.to(device=device, dtype=torch.float32)

            mask = Fore_X_Chunk_Torch[:,:,-1] != 1
            Y_values_masked = Y_value.clone()
            Y_values_masked[~mask] = float('nan')

            
            prediction = Model(
                Hist_X_Chunk_Torch, 
                Fore_X_Chunk_Torch
            )[1]
            

            predictions = transform_CMAL_parameters_multi(prediction)

            for day in range(len(Y_value[0])):
                if not torch.isnan(Y_values_masked[:, day]).all():

                    
                    prediction = {}
                    prediction = {
                        'mu': predictions[:, day, ::4],   # mu is at indices 0 mod 4
                        'b': predictions[:, day, 1::4],   # b is at indices 1 mod 4 (ensure positive)
                        'tau': predictions[:, day, 2::4], # tau is at indices 2 mod 4 (ensure 0-1)
                        'pi': predictions[:, day, 3::4],  # pi is at indices 3 mod 4
                    }
                    
                    # ground_truth = {}
                    ground_truth = {'y': Y_value[:, day]}
    
                    loss += criterion(prediction, ground_truth)

            if not torch.isnan(loss):
                epoch_loss += loss.item()

                if train_mode:
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()


                else: 
                    # Calculate validation statistics
                    num_members = 11


            
                    
            average_loss += loss.item()    # loss = criterion(prediction.squeeze(), Y_value)
            
            if accumulated_batches % batch_accumulation == 0:        
                print(f'Epoch {epoch+1}/{epochs}, Batch {i}/{len(Dataloader)}, '
                      f'Average loss: {average_loss/batch_accumulation:.4f}')
                
                sys.stdout.flush()
                
                average_loss = 0
                accumulated_batches = 0
            # if i == 1e5:
            #     break

        if scheduler is not None:
            scheduler.step()
        epoch_losses.append(epoch_loss)   

    if train_mode:
        return epoch_losses
    else:
        return epoch_losses #[epoch_losses, all_conditional_ensemble_summaries, all_greedy_ensemble_summaries, all_discharge_summaries, CRPS_Per_leadtime]


def train_seeded_model(model, optimizer, Training_Dataloader, epochs=3, batch_accumulation_size=300, 
              num_ensemble_members=4, noise_scale=1.0,
              criterion=None, scheduler = None, train_mode = True, fixed_noise = False):
    """
    Train a model using ensemble predictions and batch accumulation.
    
    Parameters:
    -----------
    G_Model : torch.nn.Module: The model to train
    Training_Dataloader : torch.utils.data.DataLoader: DataLoader containing training data
    epochs : int: Number of epochs to train for
    batch_accumulation_size : int: Number of batches to accumulate before performing gradient update
    learning_rate : float: Learning rate for optimizer
    num_ensemble_members : int: Number of ensemble members to generate during training
    noise_scale : float: Scale of noise to use in ensemble predictions
    criterion : torch.nn.Module: Loss function to use (defaults to CRPSLoss if None)

    Returns:
    --------
    list
        Training loss history
    """

    # Use provided criterion or default to CRPSLoss
    if criterion is None:
        criterion = CRPSLoss_Ensemble()

    model.train(train_mode)

    
    average_loss = 0
    i = 0
    loss_history = []
    
    for epoch in range(epochs):
        epoch_loss = 0
        batch_count = 0
        
        for Hist_X_Chunk_Torch, Fore_X_Chunk_Torch, Y_value, end_prediction_date, basin_idx in Training_Dataloader:
            i = i + 1
            batch_count += 1

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            Hist_X_Chunk_Torch = Hist_X_Chunk_Torch.to(device=device, dtype=torch.float32)
            Fore_X_Chunk_Torch = Fore_X_Chunk_Torch.to(device=device, dtype=torch.float32)
            Y_value = Y_value.to(device=device, dtype=torch.float32)
       
            # Trim forecast features
            Fore_X_Chunk_Torch = Fore_X_Chunk_Torch[:,:, 0:15]
            
            # Generate ensemble predictions
            if train_mode:
                ensemble_predictions = run_ensemble_predictions(model, 
                    Hist_X_Chunk_Torch.to(torch.float32), Fore_X_Chunk_Torch.to(torch.float32),
                    num_members=num_ensemble_members, noise_scale=noise_scale, fixed_noise = fixed_noise)
                
                # Calculate loss using the ensemble predictions
                batch_loss = criterion(ensemble_predictions.squeeze(), Y_value)
                loss = batch_loss.mean()
    
                # Accumulate loss (without backpropagation yet)
                average_loss += loss
                epoch_loss += loss.item()

                # Backpropagate the accumulated loss
                loss.backward()
                # Update parameters and reset gradients
                optimizer.step()
                optimizer.zero_grad()

            else:
                with torch.no_grad():
                    ensemble_predictions = run_ensemble_predictions(model, 
                    Hist_X_Chunk_Torch.to(torch.float32), Fore_X_Chunk_Torch.to(torch.float32),
                    num_members=num_ensemble_members, noise_scale=noise_scale, fixed_noise = fixed_noise)


                    batch_loss = criterion(ensemble_predictions.squeeze(), Y_value)
                    loss = batch_loss.mean()
                    # Accumulate loss (without backpropagation yet)
                    average_loss += loss
                    epoch_loss += loss.item()


            if i % batch_accumulation_size == 0:
                i = 0
                avg_batch_loss = average_loss / batch_accumulation_size
                print(f'Batch {batch_count}, Average loss: {avg_batch_loss.item():.6f}')
                loss_history.append(avg_batch_loss.item())
                average_loss = 0

        if scheduler is not None:
            scheduler.step()        
    
        print(f'Epoch {epoch+1}/{epochs} complete, average loss: {epoch_loss/batch_count:.6f}')
        # print_cuda_memory_summary()
        sys.stdout.flush()
    
    return loss_history


def run_ensemble_predictions(Model, Hist_X, Fore_X, num_members=4, noise_scale=1.0, fixed_noise = False):
    """
    Run model predictions with an additional row of random noise each time.
    
    Args:
        Model: The model to generate predictions
        Hist_X: Historical data tensor [batch_size, seq_len, features]
        Fore_X: Forecast data tensor [batch_size, seq_len, features]
        num_members: Number of ensemble members to generate
        noise_scale: Scale of random noise to add
        
    Returns:
        Tensor of ensemble predictions [num_members, batch_size, num_steps]
    """
    batch_size = Hist_X.shape[0]
    hist_seq_len = Hist_X.shape[1]
    hist_features = Hist_X.shape[2]
    fore_seq_len = Fore_X.shape[1]
    fore_features = Fore_X.shape[2]
    device = Hist_X.device
    
    # List to store all predictions
    all_predictions = []


    for member_idx in range(num_members):
        # Create noise rows for both historical and forecast data
        with torch.no_grad():
            hist_noise = noise_scale * torch.randn(batch_size, hist_seq_len, 1, device=device)
            fore_noise = noise_scale * torch.randn(batch_size, fore_seq_len, 1, device=device)

            
            if fixed_noise == True:
                # Generate single random noise value per batch item
                batch_noise = noise_scale * torch.randn(batch_size, 1, 1, device=device)
                
                # Repeat this same noise value across the sequence length dimensions
                hist_noise = batch_noise.repeat(1, hist_seq_len, 1)
                fore_noise = batch_noise.repeat(1, fore_seq_len, 1)

        # Concatenate noise as an additional feature
        hist_with_noise = torch.cat([Hist_X, hist_noise], dim=2)
        fore_with_noise = torch.cat([Fore_X, fore_noise], dim=2)
        
        # Run the model with the augmented input
        prediction = Model(hist_with_noise, 
                            fore_with_noise)[1]
        
        # Add this prediction to our collection
        all_predictions.append(prediction)


        del hist_with_noise, fore_with_noise
    
    # Stack all predictions to create a tensor with shape [num_members, batch_size, num_steps]
    ensemble_predictions = torch.stack(all_predictions, dim=0)
    
    return ensemble_predictions


def calculate_kge(observed, simulated):
    """
    Calculate Kling-Gupta Efficiency for timeseries data.
    
    Parameters:
    -----------
    observed : array-like
        Observed timeseries values
    simulated : array-like
        Simulated timeseries values
        
    Returns:
    --------
    float
        KGE value (range: -∞ to 1, where 1 is perfect)
    """
    # Ensure inputs are numpy arrays
    observed = np.array(observed)
    simulated = np.array(simulated)
    
    # Add small constant to avoid division by zero
    eps = 1e-6
    
    # Calculate components
    mean_obs = np.mean(observed)
    mean_sim = np.mean(simulated)
    std_obs = np.std(observed)
    std_sim = np.std(simulated)
    
    # Correlation coefficient
    r = np.corrcoef(observed, simulated)[0, 1]
    
    # Ratio of means (beta)
    beta = mean_sim / (mean_obs + eps)
    
    # Ratio of standard deviations (alpha)
    alpha = std_sim / (std_obs + eps)
    
    # Calculate KGE
    kge = 1 - np.sqrt((r - 1)**2 + (beta - 1)**2 + (alpha - 1)**2)
    
    return kge


def calculate_crps(x, y):
    """
    Computes CRPS from x using y as reference,
    first x dimension must be ensembles, next dimensions can be arbitrary
    x: ensemble data (n_ens, n_points)
    y: observation/analysis data (n_points)
    returns: crps (n_points)
    REFERENCE
      Hersbach, 2000: Decomposition of the Continuous Ranked Probability Score for Ensemble Prediction Systems.
      Weather and Forecasting 15: 559-570.
    """

    # first sort ensemble
    x.sort(axis=0)

    # construct alpha and beta, size nens+1
    n_ens = x.shape[0]
    shape = (n_ens+1,)+x.shape[1:]
    alpha = np.zeros(shape)
    beta = np.zeros(shape)

    # x[i+1]-x[i] and x[i]-y[i] arrays
    diffxy = x-y.reshape(1, *(y.shape))
    diffxx = x[1:]-x[:-1]  # x[i+1]-x[i], size ens-1

    # if i == 0
    alpha[0] = 0
    beta[0] = np.fmax(diffxy[0], 0)  # x(0)-y
    # if i == n_ens
    alpha[-1] = np.fmax(-diffxy[-1], 0)  # y-x(n)
    beta[-1] = 0
    # else
    alpha[1:-1] = np.fmin(diffxx, np.fmax(-diffxy[:-1], 0))  # x(i+1)-x(i) or y-x(i) or 0
    beta[1:-1] = np.fmin(diffxx, np.fmax(diffxy[1:], 0))  # 0 or x(i+1)-y or x(i+1)-x(i)

    # compute crps
    p_exp = (np.arange(n_ens+1)/float(n_ens)).reshape(n_ens+1, *([1]*y.ndim))
    crps = np.sum(alpha*(p_exp**2) + beta*((1-p_exp)**2), axis=0)
    #
    # p = np.arange(n_ens+1)/float(n_ens)
    # alpha_mean = alpha.mean(axis=1)
    # beat_mean = beta.mean(axis=1)
    # crps = alpha_mean*(p**2) + beat_mean*((1-p)**2)
    # crps_mean = crps2.sum()
    #
    # p_exp = np.expand_dims(np.arange(n_ens+1)/float(n_ens), axis=1)
    # crps = np.nansum(alpha*(p_exp**2) + beta*((1-p_exp)**2), axis=0)
    # crps_mean = crps.mean()
    return crps


def transform_CMAL_parameters(tensor):
    """
    Transform a tensor by applying:
    - Softplus to the second element (index 1) of each row
    - Sigmoid to the third element (index 2) of each row
    
    Args:
        tensor: Input tensor of shape [batch_size, features]
        
    Returns:
        Transformed tensor of the same shape
    """
    result = tensor.clone()
    
    # Apply softplus to the second column (index 1)
    result[:, 1] = F.softplus(tensor[:, 1])
    
    # Apply sigmoid to the third column (index 2)
    result[:, 2] = torch.sigmoid(tensor[:, 2])
    
    return result


def transform_CMAL_parameters_multi(tensor):
    '''
    Generates predictions from a tensor representing multiple Asymetric Laplacians 
    '''
    # Input tensor of shape (b, m, 4n), batch, days, number of Asymmetric laplcians
    b, m, n4 = tensor.shape
    n = n4 // 4
    
    # Create a copy of the tensor to modify
    result = tensor.clone()
    
    # Apply softplus to elements with index 1mod4 (positions 1, 5, 9, ...)
    softplus_indices = torch.arange(1, n4, 4)
    result[:, :, softplus_indices] = F.softplus(tensor[:, :, softplus_indices])
    
    # Apply sigmoid to elements with index 2mod4 (positions 2, 6, 10, ...)
    sigmoid_indices = torch.arange(2, n4, 4)
    result[:, :, sigmoid_indices] = torch.sigmoid(tensor[:, :, sigmoid_indices])
    
    # Apply softmax to every 4th element (indices 3mod4)
    # First, reshape to extract these elements
    softmax_indices = torch.arange(3, n4, 4)
    
    # Extract the elements to apply softmax to
    softmax_elements = tensor[:, :, softmax_indices]
    
    # Apply softmax along dimension 1 (across the n elements)
    softmax_result = F.softmax(softmax_elements, dim= -1)
    
    # Put the softmax result back in the original tensor
    result[:, :, softmax_indices] = softmax_result
    
    return result


def load_and_unnormalize(ensemble_predictions, true_discharge, scaler_path='/home/mokr/Loss_Functions_Paper/Scalers/discharge_caravan_scalers.joblib'):
    """
    Load scaler and normalize ensemble predictions and true discharge.
    
    Args:
        ensemble_predictions (torch.Tensor): Ensemble predictions to normalize
        true_discharge (torch.Tensor): True discharge values to normalize
        scaler_path (str): Path to the joblib scaler file
    
    Returns:
        tuple: Normalized ensemble predictions and true discharge
    """
    # Load the scaler
    try:
        scaler = joblib.load(scaler_path)['streamflow']
    except FileNotFoundError:
        print(f"Scaler file not found at {scaler_path}")
        return ensemble_predictions, true_discharge
    except Exception as e:
        print(f"Error loading scaler: {e}")
        return ensemble_predictions, true_discharge
    
    # Convert to numpy if tensor
    if isinstance(ensemble_predictions, torch.Tensor):
        ensemble_predictions = ensemble_predictions.numpy()
    
    if isinstance(true_discharge, torch.Tensor):
        true_discharge = true_discharge.numpy()
    
    # Normalize ensemble predictions
    # Assuming the scaler is a scipy.stats.StandardScaler
    unnormalized_ensemble = (ensemble_predictions*scaler.scale_) + scaler.mean_ 
    
    # Normalize true discharge
    unnormalized_true_discharge = (true_discharge*scaler.scale_) + scaler.mean_
    
    return unnormalized_ensemble, unnormalized_true_discharge


def replace_keys(d, key_map):
    """Replace keys in a dictionary according to key_map."""
    return {key_map.get(k, k): v for k, v in d.items()}


def print_cuda_memory_summary():
    stats = torch.cuda.memory_stats()
    
    def format_bytes(num):
        for unit in ['B', 'KB', 'MB', 'GB']:
            if abs(num) < 1024:
                return f"{num:.1f} {unit}"
            num /= 1024
        return f"{num:.1f} TB"

    summary = {
        "Allocated (current)": format_bytes(stats["allocated_bytes.all.current"]),
        "Allocated (peak)": format_bytes(stats["allocated_bytes.all.peak"]),
        "Reserved (current)": format_bytes(stats["reserved_bytes.all.current"]),
        "Inactive split (current)": format_bytes(stats["inactive_split_bytes.all.current"]),
        "Active blocks": stats["active.all.current"],
        "Segments in use": stats["segment.all.current"],
        "OOMs": stats["num_ooms"],
        "Alloc retries": stats["num_alloc_retries"],
    }

    print("\n CUDA Memory Summary:")
    for k, v in summary.items():
        print(f"  {k:<25}: {v}")


sys.path.append(str(functions_path))

warnings.filterwarnings('ignore', category=FutureWarning, message='Series.__getitem__ treating keys as positions is deprecated')
