import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
import pandas as pd
import itertools
import pickle

import sys
from pathlib import Path
functions_path = Path("/home/mokr/Loss_Functions_Paper/ML_Functions/")
sys.path.append(str(functions_path))

import ML_functions
from ML_functions import HydroDataset
import ML_Losses

import torch
import torch.nn as nn
from collections import defaultdict
from torch.utils.data import Subset
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from ML_Plots import kl_divergence, CMAL_quantile, transform_CMAL_parameters, draw_from_CMAL_distribution, draw_from_CMAL_distribution_np, load_and_unnormalize, get_ensemble_members_vectorized, plot_ensemble_predictions, generate_greedy_trajectories, plot_dispersion_histograms, plot_uniform_qq, load_and_unnormalize

from ML_Losses import compute_log_likelihood, compute_CDF, crps_loss, compute_crps, calculate_overall_crps, get_member_summaries_torch , compute_nse, compute_kge, process_ensemble_predictions, process_ensemble_predictions_efficient, initialize_storage_efficient

from ML_functions import transform_CMAL_parameters_multi, run_ensemble_predictions





CRPS_Binary_Model =  torch.load('/home/mokr/Loss_Functions_Paper/Models/Binary/hyperparameter_test/Binary_lr_0.001_bi_False_h_256_256_256_0506_30Epochs.pth', weights_only = False, map_location=torch.device('cpu')) # BEST CRPS Model
CMAL_Binary_Model =  torch.load('/home/mokr/Loss_Functions_Paper/Models/Binary/hyperparameter_test/CMAL_Binary_lr_0.001_bi_False_h_256_256_256_0507_10Epochs.pth', weights_only = False, map_location=torch.device('cpu')) # BEST CMAL Modelm loss -15
NonBinary_Model = torch.load('/home/mokr/Loss_Functions_Paper/Models/NonBinary/hyperparameter_test/NonBinary_lr_0.001_bi_False_h_128_128_64_0503_30Epochs.pth', weights_only = False, map_location=torch.device('cpu')) # Best NonBinary models so far
Fixed_Noise_Seeded_Model = torch.load('/home/mokr/Loss_Functions_Paper/Models/Seeded/hyperparameter_test/Seeded_lr_0.001_bi_False_h_256_256_128_0623_8_0.9_5_True_Epochs.pth', weights_only = False, map_location=torch.device('cpu')) # Best Seeded Model so far
Non_Fixed_Noise_Seeded_Model = torch.load('/home/mokr/Loss_Functions_Paper/Models/Seeded/hyperparameter_test/Seeded_lr_0.001_bi_False_h_128_128_64_0526_8_0.85_5_False_Epochs.pth', weights_only = False, map_location=torch.device('cpu')) # 0.06 val loss


scaler_path='/home/mokr/Loss_Functions_Paper/Scalers/discharge_caravan_scalers.joblib'

# loaded_validation_data = torch.load("/perm/mokr/10Day_Loss_Function_Validation_Dataset_Binary_P1.pt")
loaded_validation_data = torch.load("/perm/mokr/10Day_Loss_Function_Test_Dataset_Binary_P1.pt")


nth = 0
print(nth)

target_sample = 0 #num_samples * nth
num_samples = min(int(5e15),  len(loaded_validation_data))

# Create a subset starting from the target sample
target_sample_int = int(target_sample)
end_sample = min(target_sample_int + num_samples, len(loaded_validation_data))
print(target_sample_int, end_sample)
subset_data = Subset(loaded_validation_data, range(target_sample_int, end_sample))

Validation_Dataloader = DataLoader(subset_data, batch_size=1, shuffle=False, pin_memory=True)

# Validation_Dataloader = DataLoader(loaded_validation_data, batch_size= 1, shuffle=False, pin_memory=True)


variogram_p = 0.5

def get_nonbinary_ensemble_members(model, Hist_X_Chunk_Torch, Fore_X_Chunk_Torch, num_members=1, keep_original = False, device='cpu'):
    """
    Generate ensemble predictions for discharge.
    
    Args:
        model (torch.nn.Module): The trained machine learning model
        Hist_X_Chunk_Torch (torch.Tensor): Historical input chunk (batch size, num_steps, num_variables)
        Fore_X_Chunk_Torch (torch.Tensor): Forecast input chunk 
        num_members (int, optional): Number of ensemble members to generate. Defaults to 1.
    
    Returns:
        torch.Tensor: Ensemble predictions of discharge
    """
    
    num_steps = Fore_X_Chunk_Torch.shape[1]
    batch_size = Fore_X_Chunk_Torch.shape[0]

    hist_steps = Hist_X_Chunk_Torch.shape[1]
    
    # Repeat historical input for all ensemble members
    Hist_X_Chunk_Torch_repeated = Hist_X_Chunk_Torch.repeat(num_members, 1, 1, 1)
    Fore_X_Chunk_Torch_repeated = Fore_X_Chunk_Torch.repeat(num_members, 1, 1, 1)

    Original_Predictions = torch.zeros(num_members, batch_size, num_steps, 1, device=device)
    Original_Predictions_Saved = torch.zeros(num_members, batch_size, num_steps, 1, device=device)
    Discharge_Mask = torch.zeros(num_members, batch_size, num_steps, 1, device=device)

          


    #treat the ensembles and batch_size as one big batch set so the dimension is 3 again
    Binary_Fore_X_Chunk_Torch = Fore_X_Chunk_Torch_repeated.reshape(num_members * batch_size, num_steps, -1)
    Hist_X_Chunk_Torch_repeated = Hist_X_Chunk_Torch_repeated.reshape(num_members * batch_size, hist_steps, -1)

    with torch.no_grad():            
        param_predictions = model(
            Hist_X_Chunk_Torch_repeated.to(torch.float32), 
            Binary_Fore_X_Chunk_Torch.to(torch.float32)
        )[1]
        
    for step in range(num_steps):    
        # Transform parameters and draw samples
        param_predictions_step = ML_functions.transform_CMAL_parameters_multi(param_predictions)[:, step]
        
        param_prediction = {
            'mu': param_predictions_step[:, ::4],   # mu is at indices 0 mod 4
            'b': param_predictions_step[:, 1::4],   # b is at indices 1 mod 4 (ensure positive)
            'tau': param_predictions_step[:, 2::4], # tau is at indices 2 mod 4 (ensure 0-1)
            'pi': param_predictions_step[:, 3::4],  # pi is at indices 3 mod 4
        }

    
        # Draw samples for each ensemble member
        samples = draw_from_CMAL_distribution(param_prediction, num_samples=1)
        
        # Update predictions and masks
        Original_Predictions_Saved[:, :, step, :] = samples.clone().detach().reshape(num_members, batch_size, 1)

    return Original_Predictions_Saved.squeeze()

(ensemble_summaries, crps_per_leadtime, variogram_scores, stored_forecasts, NSE_scores, KGE_scores, basin_forecasts, metadata) = initialize_storage_efficient(num_samples)

idx = 0

for i, (Hist_X_Chunk_Torch, Fore_X_Chunk_Torch, Y_value, date_idx, basin_idx) in enumerate(Validation_Dataloader):    
    if torch.isnan(Y_value).any():
        continue   

    # Convert to hashable types
    basin_idx = basin_idx[0]
    date_idx = date_idx[0] 
    
    if basin_idx not in basin_forecasts:
        basin_forecasts[basin_idx] = {"CRPS": [], "Seeded": [], "NonBinary": [], "Discharge": []}
            
    Fore_X_Chunk_Torch_Flagless = Fore_X_Chunk_Torch[:,:, 0:15]
    
    num_members = 11
    num_steps = Fore_X_Chunk_Torch.shape[1]

    # Get the members of the CRPS, CMAL, and NonBinary models, unnormalise them and clip them to only predict positive discharge
    CRPS_ensemble_predictions =  get_ensemble_members_vectorized(CRPS_Binary_Model, Hist_X_Chunk_Torch, Fore_X_Chunk_Torch[:,:,0:15], num_members= num_members, keep_original = False)
    
    NonBinary_ensemble_predictions = get_nonbinary_ensemble_members(NonBinary_Model, Hist_X_Chunk_Torch, Fore_X_Chunk_Torch[:,:,0:15], num_members= num_members, keep_original = False)

    Fixed_Seeded_ensemble_predictions = run_ensemble_predictions(Fixed_Noise_Seeded_Model, Hist_X_Chunk_Torch.to(torch.float32), Fore_X_Chunk_Torch_Flagless.to(torch.float32), num_members=num_members, noise_scale=1.0)
    Non_Fixed_Seeded_ensemble_predictions = run_ensemble_predictions(Non_Fixed_Noise_Seeded_Model, Hist_X_Chunk_Torch.to(torch.float32), Fore_X_Chunk_Torch_Flagless.to(torch.float32), num_members=num_members, noise_scale=1.0)
    
    CRPS_ensemble_predictions, NonBinary_ensemble_predictions = CRPS_ensemble_predictions.unsqueeze(1), NonBinary_ensemble_predictions.unsqueeze(1)
    Fixed_Seeded_ensemble_predictions = Fixed_Seeded_ensemble_predictions.squeeze().unsqueeze(1)
    Non_Fixed_Seeded_ensemble_predictions = Non_Fixed_Seeded_ensemble_predictions.squeeze().unsqueeze(1)

    
    CRPS_ensemble_predictions[CRPS_ensemble_predictions < -0.26787253] = -0.2678725
    NonBinary_ensemble_predictions[NonBinary_ensemble_predictions < -0.26787253] = -0.2678725 
    Fixed_Seeded_ensemble_predictions[Fixed_Seeded_ensemble_predictions < -0.26787253] = -0.2678725
    Non_Fixed_Seeded_ensemble_predictions[Non_Fixed_Seeded_ensemble_predictions < -0.26787253] = -0.2678725
    
    CRPS_ensemble_predictions, true_discharge = load_and_unnormalize(CRPS_ensemble_predictions, Y_value, scaler_path)
    NonBinary_ensemble_predictions, _ = load_and_unnormalize(NonBinary_ensemble_predictions, Y_value, scaler_path)
    Fixed_Seeded_ensemble_predictions, _ = load_and_unnormalize(Fixed_Seeded_ensemble_predictions.detach(), Y_value)   
    Non_Fixed_Seeded_ensemble_predictions, _ = load_and_unnormalize(Non_Fixed_Seeded_ensemble_predictions.detach(), Y_value)

    true_discharge[true_discharge < 0] = 0
    CRPS_ensemble_predictions[CRPS_ensemble_predictions < 0] = 0
    NonBinary_ensemble_predictions[NonBinary_ensemble_predictions < 0] = 0 
    Fixed_Seeded_ensemble_predictions[Fixed_Seeded_ensemble_predictions < 0] = 0
    Non_Fixed_Seeded_ensemble_predictions[Non_Fixed_Seeded_ensemble_predictions < 0] = 0
       
    
    # Convert everything to torch
    CRPS_ensemble_predictions, Fixed_Seeded_ensemble_predictions, Non_Fixed_Seeded_ensemble_predictions, NonBinary_ensemble_predictions, true_discharge = map(
        lambda x: torch.from_numpy(x).float(),
        [CRPS_ensemble_predictions, Fixed_Seeded_ensemble_predictions, Non_Fixed_Seeded_ensemble_predictions, NonBinary_ensemble_predictions, true_discharge]
    )
     

    
    # Store necessary information
    
    # CRPS

    model_preds = {
    "CRPS": CRPS_ensemble_predictions,
    "Fixed_Seeded": Fixed_Seeded_ensemble_predictions,
    "Non_Fixed_Seeded": Non_Fixed_Seeded_ensemble_predictions,
    "NonBinary": NonBinary_ensemble_predictions
    }

    # process_ensemble_predictions(model_preds, true_discharge, basin_idx,
    #                        ensemble_summaries, crps_per_leadtime, variogram_scores, stored_forecasts,
    #                        NSE_scores, KGE_scores, basin_forecasts, variogram_p)

    process_ensemble_predictions_efficient(
            model_preds, true_discharge, basin_idx, idx,
            ensemble_summaries, crps_per_leadtime, variogram_scores, stored_forecasts,
            NSE_scores, KGE_scores, basin_forecasts, variogram_p)


    
    # Store metadata
    metadata[i]['date_idx'] = date_idx
    metadata[i]['basin_idx'] = basin_idx

    idx += 1
    # # Stores dates and basins
    # dates.append(date_idx)
    # basins.append(basin_idx)
    if i % 1e4 == 0:
        print(i, flush=True)

        avg_per_model = {
            model: np.nanmean(tensors, axis=0).squeeze().tolist()
            for model, tensors in crps_per_leadtime.items()
        }
                    
        # Create a DataFrame
        lead_times = [f'Lead {i+1}' for i in range(10)]
        df = pd.DataFrame(avg_per_model, index=lead_times)
        # Transpose for models as rows
        df.index.name = 'Model'
        df.columns.name = 'Lead Time'
    
        print(df, flush=True)

    # have 3.5e6 samples, takes 10 hours for 1e5
    # It probably isn't worth rewiting the code, let's just run the code for 1 day each 8 times



save_dir = "/home/mokr/Loss_Functions_Paper/forecast_results"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

basin_forecasts = dict(basin_forecasts)


# Save each data structure individually
data_objects = {
    "ensemble_summaries": ensemble_summaries,
    "crps_per_leadtime": crps_per_leadtime,
    "variogram_scores": variogram_scores,
    "stored_forecasts": stored_forecasts,
    "KGE_scores": KGE_scores,
    "NSE_scores": NSE_scores,
    "metadata": metadata,
    "basin_forecasts": basin_forecasts
}


# Save all data structures
for name, obj in data_objects.items():
    with open(f"{save_dir}/{name}_Part{nth}_all_test.pkl", "wb") as f:
        pickle.dump(obj, f)
    print(f"Saved {name} to {save_dir}/{name}_Part{nth}_all_test.pkl")

# Also save everything in a single file for convenience
with open(f"{save_dir}/all_forecast_data.pkl_Part{nth}_all_test", "wb") as f:
    pickle.dump(data_objects, f)
    print(f"Saved all data to {save_dir}/all_forecast_data_Part{nth}_all_test.pkl")