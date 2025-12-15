import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
from scipy.special import rel_entr
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
import joblib
import random

import sys
from pathlib import Path
functions_path = Path("/home/mokr/Loss_Functions_Paper/ML_Functions/")
sys.path.append(str(functions_path))

import ML_functions
from ML_Losses import compute_log_likelihood

def plot_ensemble_predictions(ensemble_predictions, true_discharge=None, title='Ensemble Predictions', plot_type='both', unnormalize=True, scaler_path='/home/mokr/Loss_Functions_Paper/Scalers/discharge_caravan_scalers.joblib', ylims=None):
    """
    Plot ensemble predictions with multiple visualization options.
    
    Args:
        ensemble_predictions (torch.Tensor): Tensor of ensemble predictions
        true_discharge (torch.Tensor, optional): True discharge values for comparison
        title (str, optional): Plot title
        plot_type (str, optional): Type of plot to generate
        normalize (bool, optional): Whether to normalize using the scaler
        scaler_path (str, optional): Path to the joblib scaler file
    """
    # Normalize if requested
    if unnormalize:
        ensemble_predictions, true_discharge = load_and_unnormalize(
            ensemble_predictions, 
            true_discharge, 
            scaler_path
        )
        
    # Convert to numpy if tensor
    if isinstance(ensemble_predictions, torch.Tensor):
        ensemble_predictions = ensemble_predictions.numpy()
    
    # Convert true_discharge to numpy if tensor
    if true_discharge is not None and isinstance(true_discharge, torch.Tensor):
        true_discharge = true_discharge.numpy().squeeze()
    
    # Validate plot_type
    valid_plot_types = ['ensemble', 'violin', 'both']
    if plot_type not in valid_plot_types:
        raise ValueError(f"plot_type must be one of {valid_plot_types}")
    
    # Create subplots based on plot type
    if plot_type == 'ensemble':
        fig, ax = plt.subplots(figsize=(14, 7))
        _plot_ensemble(ax, ensemble_predictions, true_discharge, title)
        if ylims:
            ax.set_ylim(ylims)

    elif plot_type == 'violin':
        fig, ax = plt.subplots(figsize=(14, 7))
        _plot_violin(ax, ensemble_predictions, true_discharge, title)
        if ylims:
            ax.set_ylim(ylims)
    else:  # both
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12), sharex=True)
        _plot_ensemble(ax1, ensemble_predictions, true_discharge, f'{title} - Ensemble')
        _plot_violin(ax2, ensemble_predictions, true_discharge, f'{title} - Violin Plot')
        if ylims:
            ax1.set_ylim(ylims)
            ax2.set_ylim(ylims)   
    plt.tight_layout()
    plt.show()

def _plot_ensemble(ax, ensemble_predictions, true_discharge=None, title=None):
    """
    Plot ensemble predictions on a given axis.
    
    Args:
        ax (matplotlib.axes.Axes): Axis to plot on
        ensemble_predictions (np.ndarray): Ensemble predictions
        true_discharge (np.ndarray, optional): True discharge values
        title (str, optional): Plot title
    """
    # Calculate statistics
    mean_predictions = np.mean(ensemble_predictions, axis=0)
    lower_percentile = np.percentile(ensemble_predictions, 10, axis=0)
    upper_percentile = np.percentile(ensemble_predictions, 90, axis=0)
    
    # Plot ensemble members (light blue, low opacity)
    for prediction in ensemble_predictions:
        random_alpha = random.uniform(0.2, 0.7)
        ax.plot(prediction, color='blue', alpha=random_alpha, linewidth=1)
    
    # Plot mean prediction
    ax.plot(mean_predictions, color='blue', linewidth=2, label='Ensemble Mean')
    
    # Plot confidence interval
    ax.fill_between(range(len(mean_predictions)), 
                    lower_percentile, 
                    upper_percentile, 
                    color='grey', 
                    alpha=0.2, 
                    label='80% Confidence Interval')
    
    # Plot true_discharge if provided
    if true_discharge is not None:
        ax.plot(true_discharge, color='black', linewidth=2, label='Observations')

    if true_discharge is not None:
        ax.legend()
        
    ax.set_title(title or 'Ensemble Predictions')
    ax.set_xlabel('Lead Time')
    ax.set_ylabel('Normalised River Discharge')
    ax.legend()

def _plot_violin(ax, ensemble_predictions, true_discharge=None, title=None):
    """
    Create violin plot of ensemble predictions on a given axis.
    
    Args:
        ax (matplotlib.axes.Axes): Axis to plot on
        ensemble_predictions (np.ndarray): Ensemble predictions
        true_discharge (np.ndarray, optional): True discharge values
        title (str, optional): Plot title
    """
    # Prepare data for violin plot
    # Transpose to get distribution for each time step
    ensemble_T = ensemble_predictions
    
    # Create violin plot
    sns.violinplot(data=ensemble_T, ax=ax, color= 'lightcoral', cut=0, log_scale = False)
    
    # Plot true discharge if provided
    if true_discharge is not None:
        ax.plot(range(len(true_discharge)), true_discharge, 
                color='black', linewidth=2, label='Observations', marker='o')
    
    ax.set_title(title or 'Ensemble Predictions - Violin Plot')
    ax.set_xlabel('Lead Time')
    ax.set_ylabel('Normalised River Discharge')
    
    if true_discharge is not None:
        ax.legend()



def CMAL_quantile(alpha, parameters):
    m = parameters[0]
    b = parameters[1]
    tau = parameters[2]
    
    if 0 < alpha <= tau:
        return m + (b/(1-tau)) * np.log(alpha/tau)
    elif tau < alpha < 1:
        return m - (b/tau) * np.log((1-alpha)/(1-tau))
    else:
        return "Alpha must be between 0 and 1"



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

def draw_from_CMAL_distribution(param_predictions, num_samples=1000):
    '''
    Generates samples from a Mixture of Asymmetric Laplacian (MAL) distributions
    
    Parameters:
    -----------
    prediction : dict
        Dictionary containing tensors for mu, b, tau, and pi
        Each tensor has shape (batch_size, num_components)
    
    num_samples : int, optional
        Number of samples to generate (default: 1000)
    
    Returns:
    --------
    numpy.ndarray
        Samples drawn from the mixture distribution
    '''

    
    mu = param_predictions['mu']
    b = param_predictions['b']
    tau = param_predictions['tau']
    pi = param_predictions['pi']
    
    pi = pi / torch.sum(pi, dim=1, keepdim=True)
    
    batch_size, num_components = mu.shape
    
    
    # Prepare output array
    samples = torch.zeros((batch_size, num_samples))
    
    # Sample for each batch
    for batch in range(batch_size):
        # Select component for each sample based on mixing weights
        component_indices = torch.multinomial(pi[batch], num_samples, replacement=True)

        
        # Draw samples from selected components
        for sample_idx, component in enumerate(component_indices):
            mu_comp = mu[batch, component]
            b_comp = b[batch, component]
            tau_comp = tau[batch, component]
            
            # Asymmetric Laplacian sampling
            u = np.random.uniform(0, 1)
                    
            if u <= tau_comp:
                sample = mu_comp + b_comp * torch.log(u / tau_comp) / (1 - tau_comp)
            else:
                sample = mu_comp - b_comp * torch.log((1 - u) / (1 - tau_comp)) / tau_comp

            samples[batch, sample_idx] = sample
    # Squeeze to 1D if batch size is 1
    return samples.squeeze()

def draw_from_CMAL_distribution_np(param_predictions, num_samples=1000):
    '''
    Generates samples from a Mixture of Asymmetric Laplacian (MAL) distributions
    
    Parameters:
    -----------
    prediction : dict
        Dictionary containing tensors for mu, b, tau, and pi
        Each tensor has shape (batch_size, num_components)
    
    num_samples : int, optional
        Number of samples to generate (default: 1000)
    
    Returns:
    --------
    numpy.ndarray
        Samples drawn from the mixture distribution
    '''

    
    
    mu = param_predictions['mu']
    b = param_predictions['b']
    tau = param_predictions['tau']
    pi = param_predictions['pi']
    
    pi = pi / np.sum(pi, axis=1, keepdims=True)
    batch_size, num_components = mu.shape
    
    # Prepare output array
    samples = np.zeros((batch_size, num_samples))
    
    # Sample for each batch
    for batch in range(batch_size):
        # Select component for each sample based on mixing weights
        component_indices = np.random.choice(len(pi[batch]), size=num_samples, p=pi[batch])

        
        # Draw samples from selected components
        for sample_idx, component in enumerate(component_indices):
            mu_comp = mu[batch, component]
            b_comp = b[batch, component]
            tau_comp = tau[batch, component]
            
            # Asymmetric Laplacian sampling
            u = np.random.uniform(0, 1)
                    
            if u <= tau_comp:
                sample = mu_comp + b_comp * np.log(u / tau_comp) / (1 - tau_comp)
            else:
                sample = mu_comp - b_comp * np.log((1 - u) / (1 - tau_comp)) / tau_comp

            samples[batch, sample_idx] = sample
    # Squeeze to 1D if batch size is 1
    return samples.squeeze()

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


def get_ensemble_members_vectorized(model, Hist_X_Chunk_Torch, Fore_X_Chunk_Torch, num_members=1, keep_original = False, device='cpu'):
    # CHECK IF THE RESHAPING RETURNS THNGS BACK WHERE THEY BELONG
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
    
    # Prepare containers for predictions and masks
    Pred_Discharge = torch.zeros(num_members, batch_size, num_steps, 1, device=device)
    Discharge_Mask = torch.zeros(num_members, batch_size, num_steps,  1, device=device)
    
    # Repeat historical input for all ensemble members
    Hist_X_Chunk_Torch_repeated = Hist_X_Chunk_Torch.repeat(num_members, 1, 1, 1)
    Fore_X_Chunk_Torch_repeated = Fore_X_Chunk_Torch.repeat(num_members, 1, 1, 1)

    # Iterate through steps
    for step in range(num_steps):
        # Prepare input by concatenating forecast chunk, current predictions, and mask
        Binary_Fore_X_Chunk_Torch = torch.cat([
            Fore_X_Chunk_Torch_repeated,  
            Pred_Discharge[:, :, :], 
            Discharge_Mask[:, :, :]
        ], dim=-1)


        #treat the ensembles and batch_size as one big batch set so the dimension is 3 again
        Binary_Fore_X_Chunk_Torch = Binary_Fore_X_Chunk_Torch.reshape(num_members * batch_size, num_steps, -1)
        Hist_X_Chunk_Torch_repeated = Hist_X_Chunk_Torch_repeated.reshape(num_members * batch_size, hist_steps, -1)

        
        # Draw samples from the model
        with torch.no_grad():
            param_predictions = model(
                Hist_X_Chunk_Torch_repeated.to(torch.float32), 
                Binary_Fore_X_Chunk_Torch.to(torch.float32)
            )[1]
        
        # Transform parameters and draw samples

        param_predictions_step = ML_functions.transform_CMAL_parameters_multi(param_predictions)[:, step]
        
        param_prediction = {
            'mu': param_predictions_step[:, ::4],   # mu is at indices 0 mod 4
            'b': param_predictions_step[:, 1::4],   # b is at indices 1 mod 4 (ensure positive)
            'tau': param_predictions_step[:, 2::4], # tau is at indices 2 mod 4 (ensure 0-1)
            'pi': param_predictions_step[:, 3::4],  # pi is at indices 3 mod 4
        }
        
        # Draw samples for each ensemble member
        samples = draw_from_CMAL_distribution(param_prediction, num_samples=1).clone().detach()
        # Update predictions and masks
        Pred_Discharge[:, :, step, :] = samples.clone().detach().reshape(num_members, batch_size, 1)
        Discharge_Mask[:, :, step, :] = 1

    if keep_original:
        Original_Predictions = torch.zeros(num_members, batch_size, num_steps, 1, device=device)
        Original_Predictions_Saved = torch.zeros(num_members, batch_size, num_steps, 1, device=device)
        Discharge_Mask = torch.zeros(num_members, batch_size, num_steps, 1, device=device)

        Binary_Fore_X_Chunk_Torch = torch.cat([
            Fore_X_Chunk_Torch_repeated,  
            Original_Predictions[:, :, :], 
            Discharge_Mask[:, :, :]
        ], dim=-1)


        #treat the ensembles and batch_size as one big batch set so the dimension is 3 again
        Binary_Fore_X_Chunk_Torch = Binary_Fore_X_Chunk_Torch.reshape(num_members * batch_size, num_steps, -1)
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
        
        
        return Pred_Discharge.squeeze(), Original_Predictions_Saved.squeeze()
    else:
        return Pred_Discharge.squeeze()

    

def plot_dispersion_histograms(dispersion_df):
    '''
    Plots histogram of truth percentile ranks for each statistic.

    Parameters:
    dispersion_df: DataFrame as returned by get_dispersion_calculations
                   Rows are percentile bins, columns are statistics
    '''
    ax = dispersion_df.plot(kind='bar', figsize=(12, 6), width=0.8)

    expected_value = 100 / len(dispersion_df)
    ax.axhline(y=expected_value, color='r', linestyle='--', alpha=0.7, 
               label=f'Expected (uniform): {expected_value:.2f}%')
    
    ax.set_xlabel("Percentile Bin")
    ax.set_ylabel("Percentage of Samples")
    ax.set_title("Truth Percentile Rank Distribution per Statistic")
    ax.legend(title="Statistic", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.ylim(0, 100)
    plt.tight_layout()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()



