import xarray as xr
import pandas as pd
import numpy as np


# 1. Normalize xarray variables using only train_basins
def normalize_xarray(ds, train_basin_list, tol = 1e-8):
    """
    Normalize an xarray dataset using statistics from only the training basins
    
    Parameters:
    -----------
    ds : xarray.Dataset
        The dataset to normalize
    train_basin_list : list
        List of basin IDs in the training set
    
    Returns:
    --------
    xarray.Dataset : Normalized dataset
    dict : Dictionary with normalization parameters (mean and std) for each variable
    """
    # Create a copy of the dataset to store normalized values
    ds_norm = ds.copy()
    
    # Dictionary to store normalization parameters
    norm_params = {}
    
    # Get statistics only from training basins
    ds_train = ds.sel(basin=train_basin_list)
    
    # Normalize each variable
    for var in ds.data_vars:
        # Calculate mean and std only using training basins
        mean = ds_train[var].mean().item()
        std = ds_train[var].std().item()
        
        # Store normalization parameters
        norm_params[var] = {'mean': mean, 'std': std}
        
        # Apply normalization to all basins
        ds_norm[var] = (ds[var] - mean) / (std + 1e-8)
    
    return ds_norm, norm_params

# 2. Normalize CSV files for all basins
def normalize_csv_files(train_basin_list):
    """
    Normalize variables in CSV files using statistics from only the training basins
    
    Parameters:
    -----------
    train_basin_list : list
        List of basin IDs in the training set
    
    Returns:
    --------
    dict : Dictionary with normalization parameters for each variable
    """
    # Dictionary to store all training dataframes
    all_train_dfs = {}
    
    # First, collect all training dataframes
    for basin_idx in train_basin_list:
        location = basin_idx.split('_')[0]
        file_path = f'/perm/mokr/Caravans/Caravan/timeseries/csv/{location}/{basin_idx}.csv'
        df = pd.read_csv(file_path)
        all_train_dfs[basin_idx] = df
    
    # Calculate normalization parameters from training data
    # Combine all training dataframes to calculate stats
    combined_train_df = pd.concat(all_train_dfs.values())
    
    # Get all numeric columns to normalize
    numeric_cols = combined_train_df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Calculate and store normalization parameters
    norm_params = {}
    for col in numeric_cols:
        mean = combined_train_df[col].mean()
        std = combined_train_df[col].std()
        norm_params[col] = {'mean': mean, 'std': std}
    
    # Function to normalize a single CSV file using the parameters
    def normalize_csv(basin_idx, params):
        location = basin_idx.split('_')[0]
        file_path = f'/perm/mokr/Caravans/Caravan/timeseries/csv/{location}/{basin_idx}.csv'
        df = pd.read_csv(file_path)
        
        # Create a copy for normalized data
        df_norm = df.copy()
        
        # Normalize numeric columns
        for col in params:
            if col in df.columns:
                mean = params[col]['mean']
                std = params[col]['std']
                df_norm[col] = (df[col] - mean) / std
        
        # Save normalized CSV
        norm_file_path = f'/perm/mokr/Caravans/Caravan/timeseries/csv_normalized/{location}/{basin_idx}.csv'
        # Make sure the directory exists
        import os
        os.makedirs(os.path.dirname(norm_file_path), exist_ok=True)
        df_norm.to_csv(norm_file_path, index=False)
        
        return df_norm
    
    # Process all basins (including test basins) using train statistics
    all_basin_idx = list(set([basin for basin in ERA5_Land.basin.values]))
    normalized_dfs = {}
    
    for basin_idx in all_basin_idx:
        normalized_dfs[basin_idx] = normalize_csv(basin_idx, norm_params)
    
    return norm_params, normalized_dfs


# 3. Save normalization parameters for later use (e.g., for inference)
def save_normalization_params(era5_params, hres_params, csv_params, output_file='normalization_params.npz'):
    """Save all normalization parameters to a file"""
    np.savez(
        output_file,
        era5_params=era5_params,
        hres_params=hres_params,
        csv_params=csv_params
    )



# Static_df = pd.read_csv(f'/perm/mokr/Caravans/Caravan/attributes/{location}/attributes_caravan_{location}.csv')
