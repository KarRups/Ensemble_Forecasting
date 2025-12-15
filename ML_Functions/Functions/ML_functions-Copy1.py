import numpy as np
import pandas as pd
import torch

import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
from torch.utils.data import Dataset

## Making models
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.0, bidirectional=False, Sequence_Target=False):
        super(LSTMModel, self).__init__()
        self.bidirectional = bidirectional
        self.No_Directions = 2 if bidirectional else 1
        self.Sequence_Target = Sequence_Target
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                             batch_first=True, 
                             dropout=dropout, 
                             bidirectional=bidirectional)
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(hidden_size * self.No_Directions, output_size)
    def forward(self, x):
        # Correct initialization of hidden state for batched input
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_layers * self.No_Directions, batch_size, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers * self.No_Directions, batch_size, self.hidden_size).to(x.device)
        
        # Pass initial states to LSTM
        out, _ = self.lstm(x, (h0, c0))
        
        # Apply dropout
        out = self.dropout(out)
        
        # Handle sequence target or last time step
        if not self.Sequence_Target:
            # Take the last time step for each sequence in the batch
            out = out[:, -1, :]  # Shape: [batch_size, hidden_size * No_Directions]
        
        # Final linear layer
        out = self.fc(out)
        return out

class Hindcast_LSTM_Block(nn.Module):
    # This block serves to take in historic data and output the initial memory and hidden 
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout = 0.0, bidirectional = False):
        super(Hindcast_LSTM_Block, self).__init__()
        self.bidirectional = bidirectional  # Store bidirectional as an instance variable
        self.No_Directions = 1 if not bidirectional else 2
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers


        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout= dropout, bidirectional = bidirectional)
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(hidden_size * self.No_Directions, output_size) # If bidirectional is true need the *2

    def forward(self, x):
        # Map H0_sequences and H0_static to the appropriate sizes
        # Is this implementation of history doing anything
        
        # h0 = torch.zeros(self.num_layers * self.No_Directions, self.hidden_size).to(x.device)
        # c0 = torch.zeros(self.num_layers * self.No_Directions, self.hidden_size).to(x.device)

        if len(np.shape(x)) == 3:
            h0 = torch.zeros( self.num_layers * self.No_Directions, x.size(0), self.hidden_size).to(x.device)
            c0 = torch.zeros( self.num_layers * self.No_Directions, x.size(0), self.hidden_size).to(x.device)
        else:
            h0 = torch.zeros( self.num_layers * self.No_Directions, self.hidden_size).to(x.device)
            c0 = torch.zeros(self.num_layers * self.No_Directions, self.hidden_size).to(x.device)
            
        out, (hn, cn) = self.lstm(x, (h0, c0)) 
        out = self.dropout(out)
        out = self.fc(out)  # Take the output from the last time step
        return out, hn, cn

class Forecast_LSTM_Block(nn.Module):
    # This block serves to take in historic data and output the initial memory and hidden 
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout = 0.0, bidirectional = False):
        super(Forecast_LSTM_Block, self).__init__()
        self.bidirectional = bidirectional  # Store bidirectional as an instance variable
        self.No_Directions = 1 if not bidirectional else 2

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers


        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout= dropout, bidirectional = bidirectional)
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(hidden_size * self.No_Directions, output_size) # If bidirectional is true need the *2

    def forward(self, x, h0, c0):
        # Map H0_sequences and H0_static to the appropriate sizes
        # Is this implementation of history doing anything
   
        out, _ = self.lstm(x, (h0, c0)) 
        out = self.dropout(out)
        out = self.fc(out)  # Take the output from the last time step
        return out
    
class Google_LSTMModel(nn.Module):
  def __init__(self, hindcast):
    super(Google_LSTMModel, self).__init__()
    self.hindcast = hindcast
    # self.forecast = forecast
    
  def forward(self, history):
    
    # get states from hindcast model
    # need to decide whether the head recieves the raw history or an encoding of it
    hind_out, hn,cn = self.hindcast(history)
    
    # get forecasts from forecast model
    #out = self.forecast(forecasts, hn,cn)
    return hind_out #, hind_out

def Google_Model_Block(hindcast_input_size, forecast_input_size, hindcast_output_size, forecast_output_size, hidden_size, num_layers, device, dropout = 0.0, bidirectional = False):
    # For now dropout and bidirectional aren't included here, can change that down the line
    # output_size for Hindcast doesn't actually matter
    Hindcast = Hindcast_LSTM_Block(hindcast_input_size, hidden_size, num_layers, hindcast_output_size, dropout = dropout, bidirectional = bidirectional)
    #Forecast = Forecast_LSTM_Block(forecast_input_size, hidden_size, num_layers, forecast_output_size, dropout = dropout, bidirectional = bidirectional)
    Block = Google_LSTMModel(Hindcast)
    Block.to(device)

    return Block
    

# ## Dataloaders


class LSTMDataGenerator(Dataset):
    def __init__(self, valid_start_dates, ERA5_Land, Caravans, Discharge, basin_index, variables, sequence_length=90):
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
        self.Caravans = Caravans
        self.Discharge = Discharge
        self.basin_index = basin_index
        self.variables = variables
        self.sequence_length = sequence_length

    def __len__(self):
        return len(self.valid_start_dates)

    def __getitem__(self, idx):
        # Get the start, end, and prediction dates
        start_date = self.valid_start_dates.iloc[idx]
        end_date = start_date + pd.Timedelta(days=self.sequence_length - 1)
        prediction_date = end_date + pd.Timedelta(days=1)
        # Select the basin-specific ERA5 Land data
        # May also want to load the Caravans dataset here and do everything for that, or do something more clever with loading?
        ERA5_Land_Basin = self.ERA5_Land.isel(basin=self.basin_index)
        ERA5_Land_Basin = (ERA5_Land_Basin - ERA5_Land_Basin.mean()) / ERA5_Land_Basin.std()

        # Extract and preprocess ERA5 Land data
        ERA5L_xr = ERA5_Land_Basin.sel(date=slice(start_date, end_date))
        ERA5L_np = np.array([ERA5L_xr[var] for var in self.variables]).squeeze()
        ERA5L_np = ERA5L_np.transpose()

        # Extract Caravans data for the same date range
        Caravans_np = np.array(self.Caravans[start_date:end_date].iloc[:, :-1])

        # Concatenate ERA5 Land data and Caravans data
        X_Chunk_np = np.concatenate((ERA5L_np, Caravans_np), axis=1)
        X_Chunk_Torch = torch.from_numpy(X_Chunk_np).float()

        # Get the target discharge value
        Y_value = torch.tensor(self.Discharge[prediction_date]).float()

        return X_Chunk_Torch, Y_value
        
class HF_LSTMDataGenerator(Dataset):
    def __init__(self, valid_start_dates, ERA5_Land, Caravans, HRES, Discharge, basin_index, Hind_variables, Fore_variables, history_sequence_length=90, forecast_sequence_length=7):
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
        self.Caravans = Caravans
        self.HRES = HRES
        self.Discharge = Discharge
        self.basin_index = basin_index
        self.Hind_variables = Hind_variables
        self.Fore_variables = Fore_variables
        self.history_sequence_length = history_sequence_length
        self.forecast_sequence_length = forecast_sequence_length

    def __len__(self):
        return len(self.valid_start_dates)

    def __getitem__(self, idx):
        # Get the start, end, and prediction dates
        start_date = pd.Timestamp(self.valid_start_dates.iloc[idx])
        end_date = start_date + pd.Timedelta(days=self.history_sequence_length - 1)
        start_prediction_date = end_date + pd.Timedelta(days=1)
        end_prediction_date = start_prediction_date + pd.Timedelta(days=self.forecast_sequence_length)
        
        # Select the basin-specific ERA5 Land data
        # May also want to load the Caravans dataset here and do everything for that, or do something more clever with loading?
        ERA5_Land_Basin = self.ERA5_Land.isel(basin=self.basin_index)
        ERA5_Land_Basin = (ERA5_Land_Basin - ERA5_Land_Basin.mean()) / ERA5_Land_Basin.std()

        # Extract and preprocess ERA5 Land data
        ERA5L_xr = ERA5_Land_Basin.sel(date=slice(start_date, end_date))
        ERA5L_np = np.array([ERA5L_xr[var] for var in self.Hind_variables]).squeeze()
        ERA5L_np = ERA5L_np.transpose()

        # Extract Caravans data for the same date range
        Caravans_np = np.array(self.Caravans[start_date:end_date].iloc[:, :-1])

        # Extract and process HRES data for future dates
        HRES_Basin = self.HRES.isel(basin=self.basin_index)
        HRES_Basin = (HRES_Basin - HRES_Basin.mean()) / HRES_Basin.std()
        HRES_xr = HRES_Basin.sel(date=slice(start_prediction_date, end_prediction_date))
        HRES_np = np.array([HRES_xr[var] for var in self.Fore_variables]).squeeze()
        HRES_np = HRES_np.transpose()     

    
        # Concatenate ERA5 Land data and Caravans data
        Hist_X_Chunk_np = np.concatenate((ERA5L_np, Caravans_np), axis=1)
        Hist_X_Chunk_Torch = torch.from_numpy(Hist_X_Chunk_np).float()
        Fore_X_Chunk_Torch = torch.from_numpy(HRES_np).float()
        
        # Get the target discharge value
        Y_value = torch.tensor(self.Discharge[start_prediction_date : end_prediction_date]).float()

        return Hist_X_Chunk_Torch, Fore_X_Chunk_Torch, Y_value


class Multi_Basins_HF_LSTMDataGenerator(Dataset):
    def __init__(self, valid_start_dates, ERA5_Land, Caravans, HRES, Discharge, basin_indices, variables, sequence_length=90, forecast_sequence_length=7):
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
        self.Caravans = Caravans
        self.Discharge = Discharge
        self.basin_indices = basin_indices
        self.variables = variables
        self.sequence_length = sequence_length

        # Create all possible combinations of start dates and basin indices
        self.combinations = list(itertools.product(
            range(len(self.valid_start_dates)), 
            self.basin_indices
        ))

    def __len__(self):
        return len(self.combinations)

    def __getitem__(self, idx):
        # Get the date index and basin index from the combinations
        date_idx, basin_idx = self.combinations[idx]
        # print(basin_idx)
        start_date = self.valid_start_dates.iloc[date_idx]
        end_date = start_date + pd.Timedelta(days=self.sequence_length - 1)
        
        start_prediction_date = end_date + pd.Timedelta(days=1)
        end_prediction_date = start_prediction_date + pd.Timedelta(days=self.forecast_sequence_length)

        # Select the basin-specific ERA5 Land data
        ERA5_Land_Basin = self.ERA5_Land.sel(basin=basin_idx)
        ERA5_Land_Basin = (ERA5_Land_Basin - ERA5_Land_Basin.mean()) / ERA5_Land_Basin.std()

        # Extract and preprocess ERA5 Land data
        ERA5L_xr = ERA5_Land_Basin.sel(date=slice(start_date, end_date))
        ERA5L_np = np.array([ERA5L_xr[var] for var in self.variables]).squeeze()
        ERA5L_np = ERA5L_np.transpose()

        # Extract Caravans data for the same date range
        location = basin_idx.split('_')[0]
        Basin_Discharge = pd.read_csv(f'/perm/mokr/Caravans/Caravan/timeseries/csv/{location}/{basin_idx}.csv')
        # Will need to remove surfance net solar min as some catchments have it and some don't
        
        Basin_Caravans = Basin_Discharge.set_index('date')
        if 'surface_net_solar_radiation_min' in Basin_Caravans.columns:
            Basin_Caravans = Basin_Caravans.drop('surface_net_solar_radiation_min', axis=1)
        Basin_Caravans = ((Basin_Caravans - Basin_Caravans.mean())/(Basin_Caravans.std()  + 1e-8) )
        Basin_Caravans.dropna(axis=1, inplace = True)
        Basin_Caravans.index = pd.to_datetime(Basin_Caravans.index)
        Caravans_np = np.array(Basin_Caravans[start_date:end_date].iloc[:, :-1]) # Need to get specific basin

        # Concatenate ERA5 Land data and Caravans data
        Hist_X_Chunk_np = np.concatenate((ERA5L_np, Caravans_np), axis=1)
        Hist_X_Chunk_Torch = torch.from_numpy(Hist_X_Chunk_np).float()

        # Extract and process HRES data for future dates
        HRES_Basin = self.HRES.isel(basin=self.basin_index)
        HRES_Basin = (HRES_Basin - HRES_Basin.mean()) / HRES_Basin.std()
        HRES_xr = HRES_Basin.sel(date=slice(start_prediction_date, end_prediction_date))
        HRES_np = np.array([HRES_xr[var] for var in self.Fore_variables]).squeeze()
        HRES_np = HRES_np.transpose()     

    
        # Concatenate ERA5 Land data and Caravans data
        Hist_X_Chunk_np = np.concatenate((ERA5L_np, Caravans_np), axis=1)
        Hist_X_Chunk_Torch = torch.from_numpy(Hist_X_Chunk_np).float()
        Fore_X_Chunk_Torch = torch.from_numpy(HRES_np).float()
        
        # Get the target discharge value
        Y_value = torch.tensor(self.Discharge[start_prediction_date : end_prediction_date]).float()

        return Hist_X_Chunk_Torch, Fore_X_Chunk_Torch, Y_value, date_idx, basin_idx


class Multi_Basin_LSTMDataGenerator(Dataset):
    def __init__(self, valid_start_dates, ERA5_Land, Caravans, Discharge, basin_indices, variables, sequence_length=90):
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
        self.Caravans = Caravans
        self.Discharge = Discharge
        self.basin_indices = basin_indices
        self.variables = variables
        self.sequence_length = sequence_length

        # Create all possible combinations of start dates and basin indices
        self.combinations = list(itertools.product(
            range(len(self.valid_start_dates)), 
            self.basin_indices
        ))

    def __len__(self):
        return len(self.combinations)

    def __getitem__(self, idx):
        # Get the date index and basin index from the combinations
        date_idx, basin_idx = self.combinations[idx]
        # print(basin_idx)
        start_date = self.valid_start_dates.iloc[date_idx]
        end_date = start_date + pd.Timedelta(days=self.sequence_length - 1)
        prediction_date = end_date + pd.Timedelta(days=1)
        # Select the basin-specific ERA5 Land data
        ERA5_Land_Basin = self.ERA5_Land.sel(basin=basin_idx)
        ERA5_Land_Basin = (ERA5_Land_Basin - ERA5_Land_Basin.mean()) / ERA5_Land_Basin.std()

        # Extract and preprocess ERA5 Land data
        ERA5L_xr = ERA5_Land_Basin.sel(date=slice(start_date, end_date))
        ERA5L_np = np.array([ERA5L_xr[var] for var in self.variables]).squeeze()
        ERA5L_np = ERA5L_np.transpose()

        # Extract Caravans data for the same date range
        location = basin_idx.split('_')[0]
        Basin_Discharge = pd.read_csv(f'/perm/mokr/Caravans/Caravan/timeseries/csv/{location}/{basin_idx}.csv')
        # Will need to remove surfance net solar min as some catchments have it and some don't
        
        Basin_Caravans = Basin_Discharge.set_index('date')
        if 'surface_net_solar_radiation_min' in Basin_Caravans.columns:
            Basin_Caravans = Basin_Caravans.drop('surface_net_solar_radiation_min', axis=1)
        Basin_Caravans = ((Basin_Caravans - Basin_Caravans.mean())/(Basin_Caravans.std()  + 1e-8) )
        Basin_Caravans.dropna(axis=1, inplace = True)
        Basin_Caravans.index = pd.to_datetime(Basin_Caravans.index)
        Caravans_np = np.array(Basin_Caravans[start_date:end_date].iloc[:, :-1]) # Need to get specific basin

        # Concatenate ERA5 Land data and Caravans data
        Hist_X_Chunk_np = np.concatenate((ERA5L_np, Caravans_np), axis=1)
        Hist_X_Chunk_Torch = torch.from_numpy(Hist_X_Chunk_np).float()

    
        # Concatenate ERA5 Land data and Caravans data
        Hist_X_Chunk_np = np.concatenate((ERA5L_np, Caravans_np), axis=1)
        Hist_X_Chunk_Torch = torch.from_numpy(Hist_X_Chunk_np).float()
        
        # Get the target discharge value
        Y_value = torch.tensor(self.Discharge[start_prediction_date : end_prediction_date]).float()

        return Hist_X_Chunk_Torch, Y_value, date_idx, basin_idx



# ## Loss Functions

def calculate_kge(observed, simulated):
    """Calculate Kling-Gupta Efficiency"""
    # Calculate components
    mean_obs = np.mean(observed)
    mean_sim = np.mean(simulated)
    std_obs = np.std(observed)
    std_sim = np.std(simulated)
    
    # Correlation coefficient
    r = np.corrcoef(observed, simulated)[0, 1]
    
    # Bias ratio
    beta = mean_sim / mean_obs
    
    # Variability ratio
    gamma = (std_sim / mean_sim) / (std_obs / mean_obs)
    
    # Calculate KGE
    kge = 1 - np.sqrt((r - 1)**2 + (beta - 1)**2 + (gamma - 1)**2)
    
    return kge

# Need to add dates to x axis

def plot_predictions_vs_observations(model, dataloader, num_samples=365):
    model.eval()  # Set model to evaluation mode
    predictions = []
    observations = []
    
    with torch.no_grad():
        for i, (x, y) in enumerate(dataloader):
            if i >= num_samples:
                break
                
            # Get prediction
            pred = model(x)
            
            # Store prediction and observation
            predictions.append(pred.squeeze().cpu().numpy())
            observations.append(y.squeeze().cpu().numpy())
    
    # Convert to numpy arrays
    predictions = np.array(predictions)
    observations = np.array(observations)
    
    # Create time index
    time_index = np.arange(len(predictions))
    
    # Create the plot
    plt.figure(figsize=(15, 6))
    plt.plot(time_index, observations, label='Observations', color='blue', alpha=0.7)
    plt.plot(time_index, predictions, label='Predictions', color='red', alpha=0.7)
    
    # Add bands for prediction uncertainty if available
    # plt.fill_between(time_index, predictions - std, predictions + std, 
    #                 color='red', alpha=0.2, label='Prediction Uncertainty')
    
    plt.xlabel('Time Steps')
    plt.ylabel('Value')
    plt.title('River Discharge Predictions vs Observations')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Calculate metrics
    # Nash-Sutcliffe Efficiency (NSE)
    nse = 1 - (np.sum((observations - predictions) ** 2) / 
               np.sum((observations - np.mean(observations)) ** 2))
    
    # Kling-Gupta Efficiency (KGE)
    kge = calculate_kge(observations, predictions)
    
    # R-squared (R²)
    r2 = np.corrcoef(observations, predictions)[0, 1]**2
    
    print(f'NSE: {nse:.4f}')
    print(f'KGE: {kge:.4f}')
    print(f'R²: {r2:.4f}')
    plt.show()
    
    return predictions, observations