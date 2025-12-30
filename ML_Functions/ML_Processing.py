"""
FILE CONTENTS
================================================================================

CLASSES:
--------------------------------------------------------------------------------
 1. Multi_Basins_HF_LSTMDataGenerator
    No description available
    Methods: __init__, __len__, __getitem__

 2. Multi_Basin_LSTMDataGenerator
    No description available
    Methods: __init__, __len__, __getitem__

 3. HF_LSTMDataGenerator
    No description available
    Methods: __init__, __len__, __getitem__

 4. LSTMDataGenerator
    No description available
    Methods: __init__, __len__, __getitem__

FUNCTIONS:
--------------------------------------------------------------------------------
 1. check_streamflow_data
    Check CSV files for NaN values in streamflow between specified dates. This me...
 2. check_streamflow_data_recursively
    Recursively check CSV files in all subdirectories for NaN values in streamflo...
 3. create_dataloaders
    No description available
 4. plot_predictions_vs_observations
    No description available
 5. process_ensemble_predictions
    Process ensemble predictions for evaluation metrics in a versatile way.
================================================================================
"""

import os
import numpy as np
import pandas as pd
import glob

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
        

        # print("----- Processing Times -----")
        # print(f"ERA5 Processing: {era5_processing_time:.4f}s ({era5_processing_time/total_batch_processing_time*100:.2f}%)")
        # print(f"Static & Discharge: {static_and_discharge_time:.4f}s ({static_and_discharge_time/total_batch_processing_time*100:.2f}%)")
        # print(f"HRES Processing: {hres_processing_time:.4f}s ({hres_processing_time/total_batch_processing_time*100:.2f}%)")
        # print(f"Converting to Torch: {converting_to_torch_time:.4f}s ({converting_to_torch_time/total_batch_processing_time*100:.2f}%)")
        # print(f"Total Batch Time: {total_batch_processing_time:.4f}s (100%)")
        # print("---------------------------")


        # print("----- ERA5 Processing Breakdown -----")
        # print(f"  - Data Selection: {era5_selection_time:.4f}s ({era5_selection_time/era5_processing_time*100:.2f}% of ERA5)")
        # print(f"  - Convert to NumPy: {era5_to_numpy_time:.4f}s ({era5_to_numpy_time/era5_processing_time*100:.2f}% of ERA5)")
        # print(f"  - Apply Scalers: {era5_scaling_time:.4f}s ({era5_scaling_time/era5_processing_time*100:.2f}% of ERA5)")
        # print(f"Total ERA5 Processing: {era5_processing_time:.4f}s ({era5_processing_time/total_batch_processing_time*100:.2f}% of total)")
        return Hist_X_Chunk_Torch, Fore_X_Chunk_Torch, Y_value, str(end_prediction_date), basin_idx


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
        ERA5_Land_Basin = (ERA5_Land_Basin - ERA5_Land_Basin.mean()) / (ERA5_Land_Basin.std() + 1e-8)

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

        return Hist_X_Chunk_Torch, Y_value, str(end_prediction_date), basin_idx


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


def check_streamflow_data(directory_path, start_date, end_date):
    """
    Check CSV files for NaN values in streamflow between specified dates. This means all the basins have entries in these dates, which means we can randomly chose date-basin combinations in training 
    
    Args:
        directory_path (str): Path to directory containing CSV files
        start_date (str): Start date in format 'YYYY-MM-DD'
        end_date (str): End date in format 'YYYY-MM-DD'
    
    Returns:
        tuple: (valid_files, invalid_files) where each is a list of filenames
    """
    # Get all CSV files in the directory
    csv_files = glob.glob(os.path.join(directory_path, "*.csv"))
    
    valid_files = []
    invalid_files = []
    
    for file_path in csv_files:
        try:
            # Read the CSV file
            # print(file_path)
            df = pd.read_csv(file_path, parse_dates=['date'])
            
            # Filter for date range
            mask = (df['date'] >= start_date) & (df['date'] <= end_date)
            df_filtered = df.loc[mask]
            filename = os.path.splitext(os.path.basename(file_path))[0]
            # Check for NaN values in streamflow column
            if df_filtered['streamflow'].isna().any():
                invalid_files.append(os.path.basename(filename))
            else:
                valid_files.append(os.path.basename(filename))
                # print("found one!")
                
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")
            filename = os.path.splitext(os.path.basename(file_path))[0]
            invalid_files.append(os.path.basename(filename))
    
    return valid_files, invalid_files


def check_streamflow_data_recursively(directory_path, start_date, end_date): # Takes about 25 seconds to process 100 stations, 
    """
    Recursively check CSV files in all subdirectories for NaN values in streamflow between specified dates.
    
    Args:
        directory_path (str): Path to root directory containing CSV files and subdirectories
        start_date (str): Start date in format 'YYYY-MM-DD'
        end_date (str): End date in format 'YYYY-MM-DD'
    
    Returns:
        tuple: (valid_files, invalid_files) where each is a list of filenames with relative paths
    """
    valid_files = []
    invalid_files = []
    
    # Walk through all subdirectories
    for root, dirs, files in os.walk(directory_path):
        counter = 0
        for file in files:
            if file.endswith('.csv'):
                counter += 1
                if counter % 1000 == 0:
                    print("1000 stations processed")
                file_path = os.path.join(root, file)
                try:
                    # Read the CSV file
                    df = pd.read_csv(file_path, parse_dates=['date'])
                    
                    # Filter for date range
                    mask = (df['date'] >= start_date) & (df['date'] <= end_date)
                    df_filtered = df.loc[mask]
                    
                    # Get relative path from the root directory
                    relative_path = os.path.relpath(file_path, directory_path)
                    filename = os.path.splitext(relative_path)[0]
                    
                    # Check for NaN values in streamflow column
                    if df_filtered['streamflow'].isna().any():
                        invalid_files.append(filename)
                    else:
                        valid_files.append(filename)
                        # print("found a suitable basin")
                    
                except Exception as e:
                    print(f"Error processing {file_path}: {str(e)}")
                    relative_path = os.path.relpath(file_path, directory_path)
                    filename = os.path.splitext(relative_path)[0]
                    invalid_files.append(filename)
    
    return valid_files, invalid_files


def create_dataloaders(ERA5_Land, HRES, Static_df, Discharge, scalers, train_basins, val_basins, ML_functions):
    batch_size = 2 #128
    history_sequence_length = 90
    forecast_sequence_length = 7


    start_date = "2016-01-01"
    end_date = "2018-12-24"
    valid_start_dates = pd.Series(pd.date_range(
        start=start_date, 
        end=pd.to_datetime(end_date) - pd.Timedelta(days=90), 
        freq='D'
    ))

    Hind_variables = list(ERA5_Land.data_vars)
    Fore_variables = list(HRES.data_vars)

    loaded_traning_data0 = torch.load("/perm/mokr/Loss_Function_Training_Dataset0.pt")
    Training_Dataset0 = HydroDataset(loaded_traning_data0)
    
    # loaded_traning_data1 = torch.load("/perm/mokr/Loss_Function_Training_Dataset1.pt")
    # Training_Dataset1 = HydroDataset(loaded_traning_data1)
    
    # loaded_traning_data2 = torch.load("/perm/mokr/Loss_Function_Training_Dataset2.pt")
    # Training_Dataset2 = HydroDataset(loaded_traning_data2)

    # loaded_traning_data3 = torch.load("/perm/mokr/Loss_Function_Training_Dataset3.pt")
    # Training_Dataset3 = HydroDataset(loaded_traning_data3)
    
    # loaded_traning_data4 = torch.load("/perm/mokr/Loss_Function_Training_Dataset4.pt")
    # Training_Dataset4 = HydroDataset(loaded_traning_data4)

    combined_data = []
    datasets = [Training_Dataset0]
    # datasets = [Training_Dataset0, Training_Dataset1, Training_Dataset2, Training_Dataset3 , Training_Dataset4]

    # Append data from each dataset to the combined list
    for dataset in datasets:
        combined_data.extend(dataset.data)
    
    Full_Training_Dataset = HydroDataset(combined_data)
    
    
    # ERA5_Land_scalers = joblib.load(f'/home/mokr/Loss_Functions_Paper/Scalers/ERA5_Land_Scalers.joblib')
    
    # scalers = ERA5_Land_scalers
    
    # Static_df = pd.read_csv(f'/home/mokr/Loss_Functions_Paper/Scaled/static_caravan_scalers.csv')
    # Discharge = pd.read_csv('/home/mokr/Loss_Functions_Paper/Scaled/discharge_scaled.csv')
    # Discharge = Discharge.set_index('date')
    # Discharge.index = pd.to_datetime(Discharge.index)

    # Full_Training_Dataset = ML_functions.Multi_Basins_HF_LSTMDataGenerator_Binary(valid_start_dates = valid_start_dates, ERA5_Land = ERA5_Land, HRES=HRES, Static_df = Static_df , Discharge = Discharge, scalers = scalers, basin_indices=train_basins, Hind_variables=Hind_variables, Fore_variables = Fore_variables, 
    #                               history_sequence_length=history_sequence_length, forecast_sequence_length= forecast_sequence_length, p = 0.125)

    Training_Dataloader = DataLoader(Full_Training_Dataset, batch_size=batch_size, shuffle=True, pin_memory=True)

    # loaded_data = torch.load("/perm/mokr/Loss_Function_Validation_Dataset.pt")
    # Validation_Dataset = ML_functions.HydroDataset(loaded_data)
    Validation_Dataloader = DataLoader(Full_Training_Dataset, batch_size= 1, shuffle=False, pin_memory=True)


    return Training_Dataloader, Validation_Dataloader


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


def process_ensemble_predictions(model_predictions, true_discharge, basin_idx,
                               ensemble_summaries, crps_per_leadtime, variogram_scores, stored_forecasts,
                               NSE_scores, KGE_scores, basin_forecasts, variogram_p = 0.5):
    """
    Process ensemble predictions for evaluation metrics in a versatile way.
    
    Args:
        model_predictions: Dictionary with model names as keys and prediction tensors as values
                          e.g., {"CRPS": tensor, "NLL": tensor, "Fixed_Seeded": tensor, ...}
        true_discharge: Ground truth discharge values
        basin_idx: Basin index or list of basin indices
        ensemble_summaries, crps_per_leadtime, etc.: Output dictionaries
    """
    
    # Process each model's predictions
    for model_name, predictions in model_predictions.items():
        if model_name == "Discharge":  # Skip if discharge is accidentally passed as a model
            continue
            
        # Ensemble summaries
        if model_name in ensemble_summaries:
            ensemble_summaries[model_name].append(get_member_summaries_torch(predictions))

        
        # CRPS per leadtime
        if model_name in crps_per_leadtime:
            crps_per_leadtime[model_name].append(compute_crps(predictions, true_discharge))
            variogram_scores[model_name].append(variogram_score_torch(predictions, true_discharge, variogram_p))
        
        # Stored forecasts
        if model_name in stored_forecasts:
            stored_forecasts[model_name].append(predictions)
        
        # NSE scores
        if model_name in NSE_scores:
            NSE_scores[model_name].append(calculate_nse(true_discharge.squeeze(), predictions.squeeze()[0]))
        
        # KGE scores
        if model_name in KGE_scores:
            KGE_scores[model_name].append(calculate_kge(true_discharge.squeeze(), predictions.squeeze()[0]))
        
        # Basin forecasts
        if basin_idx in basin_forecasts and model_name in basin_forecasts[basin_idx]:
            basin_forecasts[basin_idx][model_name].append(predictions[:, 0, :])
    
    # Handle discharge (ground truth) separately
    if "Discharge" in ensemble_summaries:
        ensemble_summaries["Discharge"].append(get_member_summaries_torch(true_discharge.unsqueeze(0)))
    
    if "Discharge" in stored_forecasts:
        stored_forecasts["Discharge"].append(true_discharge)
    
    if basin_idx in basin_forecasts and "Discharge" in basin_forecasts[basin_idx]:
        basin_forecasts[basin_idx]["Discharge"].append(true_discharge[0, :])


v
