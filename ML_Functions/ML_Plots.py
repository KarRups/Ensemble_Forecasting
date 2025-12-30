"""
FILE CONTENTS
================================================================================

FUNCTIONS:
--------------------------------------------------------------------------------

Helper Plotting functions 
 1. _plot_ensemble
    Plot ensemble predictions on a given axis.
 2. _plot_violin
    Create violin plot of ensemble predictions on a given axis.

Core Plotting functions 
 3. plot_ensemble_predictions
    Plot ensemble predictions with multiple visualization options.
 4. plot_ensemble_forecasts
    Plot 2x2 subplot of ensemble forecasts from different models with truth overlay.
 5. plot_dispersion_histograms
    Plots histogram of truth percentile ranks for each statistic.
 6. plot_hydrograph
    Plot hydrograph for a specific basin, model, and leadtime
 7. plot_hydrograph_grid
    Plot a 2x2 grid of hydrographs with NSE and KGE metrics
 8. generate_random_plots
    Generate multiple random ensemble forecast plots for selection.

Distribution Related functions
 9. CMAL_quantile
    No description available
10. draw_from_CMAL_distribution
    Generates samples from a Mixture of Asymmetric Laplacian (MAL) distributions
11. draw_from_CMAL_distribution_np
    Generates samples from a Mixture of Asymmetric Laplacian (MAL) distributions

Ensemble generation functions
12. get_ensemble_members_vectorized
    Generate ensemble predictions for discharge.
13. get_nonbinary_ensemble_members
    Generate ensemble predictions for discharge.

Utility Functions for analyzing CRPS
14. get_basin_name
    No description available
15. create_metric_crps_df
    No description available
16. create_basin_crps_tables
    Create tables showing average CRPS at each leadtime for each basin.
17. print_basin_tables
    Print the basin tables in a nice format.
18. process_crps_df
    No description available
19. add_coordinates_to_basin_table
    Add latitude and longitude columns to the basin tables DataFrame by reading l...
20. create_model_dict
    Create a dictionary splitting dataframe by Model.
21. flatten_batch_dict_of_lists
    Flatten batch dimension in a dictionary of lists of tensors or dictionaries o...

================================================================================
"""

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
import pandas as pd
from collections import defaultdict
import os
import gc
from datetime import datetime, timedelta
from scipy import stats
import sys
from pathlib import Path
import ML_functions
from ML_fucntions import transform_CMAL_parameters
from ML_Losses import compute_log_likelihood, get_member_summaries_torch 
from ML_Metrics import calculate_nse, calculate_kge, compute_crps, compute_crps_np, variogram_score, variogram_score_torch
from ML_Processing import process_ensemble_predictions

functions_path = Path("/home/mokr/Loss_Functions_Paper/ML_Functions/")



"""
Helper Plotting Functions
"""
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



"""
Plotting Functions
"""

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


def plot_ensemble_forecasts(forecasts_by_model_and_basin, truth_by_basin, metadata_by_basin, 
                           basin_idx, sample_idx, models=None, colors=None, average_crps_by_basin=None):
    """
    Plot 2x2 subplot of ensemble forecasts from different models with truth overlay.
    
    Parameters:
    -----------
    forecasts_by_model_and_basin : dict
        Dictionary with structure [model][basin] containing forecast arrays
    truth_by_basin : dict
        Dictionary with structure [basin] containing truth arrays (995, 10)
    metadata_by_basin : dict
        Dictionary containing metadata including dates
    basin_idx : str
        Basin identifier
    sample_idx : int
        Sample index to plot
    models : list, optional
        List of 4 model names. If None, uses first 4 models from dict
    colors : list, optional
        List of 4 colors for each model. If None, uses default colors
    average_crps_by_basin : dict, optional
        Dictionary with structure [model][basin] containing average CRPS values
    """
    
    # Get models if not provided
    if models is None:
        models = list(forecasts_by_model_and_basin.keys())[:4]
    
    if len(models) != 4:
        raise ValueError("Exactly 4 models required for 2x2 plot")
    
    # Default colors if not provided
    if colors is None:
        colors = ['#2E86AB', '#F18F01', '#06A77D', '#A23B72']
    
    # Get basin name and date
    basin_name = get_basin_name(basin_idx)
    date_str = str(metadata_by_basin[basin_idx][sample_idx]['date_idx']).split()[0]
    
    # Get truth data for this sample: (10,)
    truth = truth_by_basin[basin_idx][sample_idx, :]
    
    # Create figure with larger size for paper
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    # Calculate global y limits across all models
    y_min, y_max = np.inf, -np.inf
    for model in models:
        forecast = forecasts_by_model_and_basin[model][basin_idx]
        sample_forecast = forecast[sample_idx, :, :]
        y_min = min(y_min, sample_forecast.min())
        y_max = max(y_max, sample_forecast.max())
    
    # Also consider truth in y limits
    y_min = min(y_min, truth.min())
    y_max = max(y_max, truth.max())
    
    # Add 5% padding to y limits
    y_range = y_max - y_min
    y_min -= 0.05 * y_range
    y_max += 0.05 * y_range
    
    # Plot each model
    for idx, (ax, model, color) in enumerate(zip(axes, models, colors)):
        # Get forecast data: (995, 11, 10)
        forecast = forecasts_by_model_and_basin[model][basin_idx]
        
        # Extract data for this sample: (11, 10)
        sample_forecast = forecast[sample_idx, :, :]
        
        # Days (x-axis)
        days = np.arange(1, 11)
        
        # Plot each ensemble member
        for member in range(sample_forecast.shape[0]):
            ax.plot(days, sample_forecast[member, :], alpha=0.4, linewidth=1.5, 
                   color=color, label='Ensemble' if member == 0 else '')
        
        # Plot truth as bold black line
        ax.plot(days, truth, color='black', linewidth=3, label='Truth', zorder=100)
        
        # Formatting
        ax.set_xlabel('Forecast Day', fontsize=16, fontweight='bold')
        ax.set_ylabel(f'Specific Discharge ($mm/Day$)', fontsize=16, fontweight='bold')
        
        # Add CRPS to title if available
        if average_crps_by_basin is not None:
            crps_value = average_crps_by_basin[model][basin_idx][sample_idx]
            title_text = f'{model}\nAvg CRPS: {crps_value:.3f}'
        else:
            title_text = model
        
        ax.set_title(title_text, fontsize=20, fontweight='bold', pad=10)
        ax.tick_params(labelsize=16)
        ax.grid(True, alpha=0.3, linewidth=0.8)
        ax.set_xlim(1, 10)
        ax.set_ylim(y_min, y_max)
        
        # Add legend only to first subplot
        ax.legend(fontsize=14, loc='upper right', framealpha=0.92)
    
    # Overall title
    fig.suptitle(f'{basin_name} \nDate: {date_str}', 
                 fontsize=22, fontweight='bold', y=0.92)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    return fig


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


def plot_hydrograph(forecasts_by_model_and_basin, truth_by_basin, dates_by_basin, basin, model, leadtime_idx):
    """
    Plot hydrograph for a specific basin, model, and leadtime
    
    Parameters:
    -----------
    forecasts_by_model_and_basin : dict
        Dictionary with structure [model][basin] containing forecast arrays
    truth_by_basin : dict
        Dictionary with structure [basin] containing truth arrays
    dates_by_basin : dict
        Dictionary containing dates for each basin
    basin : str
        Basin ID (e.g., 'hysets_07311900')
    model : str
        Model name (e.g., 'Conditional')
    leadtime_idx : int
        Index of leadtime (0-9 for your 10 leadtimes)
    """
    basin_name = get_basin_name(basin)
    print(basin_name)
    
    # Get forecasts for this basin and model
    forecasts = forecasts_by_model_and_basin[model][basin]  # shape: (n_samples, 11, 10)
    
    # Get observations (truth) for this basin
    observations = truth_by_basin[basin][:, leadtime_idx]  # shape: (n_samples,)
    print(np.mean(observations))
    
    # Get dates for this basin
    init_dates = dates_by_basin[basin]
    
    # Calculate target dates (initialization date + leadtime)
    target_dates = init_dates + timedelta(days=leadtime_idx + 1)
    
    # Extract forecasts for this leadtime
    forecasts_at_leadtime = forecasts[:, :, leadtime_idx]  # shape: (n_samples, 11)
    
    # Calculate ensemble mean and spread
    ensemble_mean = np.mean(forecasts_at_leadtime, axis=1)
    ensemble_min = np.min(forecasts_at_leadtime, axis=1)
    ensemble_max = np.max(forecasts_at_leadtime, axis=1)
    
    # Set larger font sizes for publication
    plt.rcParams.update({
        'font.size': 16,
        'axes.labelsize': 18,
        'axes.titlesize': 20,
        'xtick.labelsize': 16,
        'ytick.labelsize': 16,
        'legend.fontsize': 16
    })
    
    # Plot
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Plot observations
    ax.plot(target_dates, observations, 'k-', linewidth=2, label='Observations', zorder=3)
    
    # Plot ensemble forecast
    ax.plot(target_dates, ensemble_mean, 'b-', linewidth=2, label='Ensemble Mean', zorder=2)
    ax.fill_between(target_dates, 
                     ensemble_min, 
                     ensemble_max,
                     alpha=0.3, color='blue', label='Ensemble Range', zorder=1)
    
    ax.set_xlabel('Initialisation Date', fontsize=18, fontweight='bold')
    ax.set_ylabel('Specific Discharge (mm/day)', fontsize=18, fontweight='bold')
    ax.set_title(f'{basin_name} - {model} - Leadtime {leadtime_idx + 1}', 
                 fontsize=20, fontweight='bold', pad=20)
    ax.legend(loc='best', frameon=True, fancybox=True, shadow=True)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()


def plot_hydrograph_grid(forecasts_by_model_and_basin, truth_by_basin, dates_by_basin, basins, model, leadtimes):
    """Plot a 2x2 grid of hydrographs with NSE and KGE metrics
    
    Parameters:
    -----------
    forecasts_by_model_and_basin : dict
        Dictionary with structure [model][basin] containing forecast arrays
    truth_by_basin : dict
        Dictionary with structure [basin] containing truth arrays
    dates_by_basin : dict
        Dictionary containing dates for each basin
    basins : list
        List of 2 basin IDs
    model : str
        Model name (e.g., 'Conditional')
    leadtimes : list
        List of 2 leadtime indices
    """
    # Set larger font sizes for publication
    plt.rcParams.update({
        'font.size': 14,
        'axes.labelsize': 16,
        'axes.titlesize': 16,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12
    })
    
    fig, axes = plt.subplots(2, 2, figsize=(20, 12))
    
    for row_idx, basin_id in enumerate(basins):
        
        basin_name = get_basin_name(basin_id)
        print(basin_name)
        
        for col_idx, leadtime_idx in enumerate(leadtimes):
            ax = axes[row_idx, col_idx]
            
            # Get forecasts for this basin and model
            forecasts = forecasts_by_model_and_basin[model][basin_id]
            
            # Get observations (truth) for this basin
            observations = truth_by_basin[basin_id][:, leadtime_idx]
            
            # Get dates for this basin
            init_dates = dates_by_basin[basin_id]
            target_dates = init_dates + timedelta(days=leadtime_idx + 1)
            
            # Extract forecasts for this leadtime
            forecasts_at_leadtime = forecasts[:, :, leadtime_idx]
            
            # Calculate ensemble mean and spread
            ensemble_mean = np.mean(forecasts_at_leadtime, axis=1)
            ensemble_min = np.min(forecasts_at_leadtime, axis=1)
            ensemble_max = np.max(forecasts_at_leadtime, axis=1)

            # Calculate metrics
            nse_value = calculate_nse(observations, ensemble_mean)
            kge_value = calculate_kge(observations, ensemble_mean)
            
            # Plot observations
            ax.plot(target_dates, observations, 'k-', linewidth=2, 
                   label='Observations', zorder=3)
            
            # Plot ensemble forecast
            ax.plot(target_dates, ensemble_mean, 'b-', linewidth=2, 
                   label='Ensemble Mean', zorder=2)
            ax.fill_between(target_dates, 
                           ensemble_min, 
                           ensemble_max,
                           alpha=0.3, color='blue', label='Ensemble Range', zorder=1)
            
            # Add metrics to title
            title = f'{basin_name}\nLeadtime {leadtime_idx + 1} | NSE: {nse_value:.3f} | KGE: {kge_value:.3f}'
            ax.set_title(title, fontsize=18, fontweight='bold', pad=10)
            
            ax.set_xlabel('Date', fontsize=16, fontweight='bold')
            ax.set_ylabel('Specific Discharge (mm/day)', fontsize=16, fontweight='bold')
            ax.legend(loc='best', frameon=True, fancybox=True, shadow=True, fontsize=14)
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.tick_params(axis='both', which='major', labelsize=14)
            
            # Rotate x-axis labels
            for label in ax.get_xticklabels():
                label.set_rotation(45)
                label.set_ha('right')
        
        del basin_name, basin_id
        gc.collect()
    
    plt.tight_layout()
    plt.savefig(f'hydrograph_grid_{model}.png', dpi=300, bbox_inches='tight')
    plt.show()


def generate_random_plots(forecasts_by_model_and_basin, truth_by_basin, metadata_by_basin,
                         models=None, colors=None, n_plots=5, basin_idx=None, sample_idx=None,
                         seed=None, average_crps_by_basin=None):
    """
    Generate multiple random ensemble forecast plots for selection.
    
    Parameters:
    -----------
    forecasts_by_model_and_basin : dict
        Dictionary with structure [model][basin] containing forecast arrays
    truth_by_basin : dict
        Dictionary with structure [basin] containing truth arrays (995, 10)
    metadata_by_basin : dict
        Dictionary containing metadata including dates
    models : list, optional
        List of 4 model names
    colors : list, optional
        List of 4 colors for each model
    n_plots : int, default=5
        Number of random plots to generate
    basin_idx : str, optional
        If provided, fix the basin and only randomize sample_idx
    sample_idx : int, optional
        If provided, fix the sample index and only randomize basin_idx
    seed : int, optional
        Random seed for reproducibility
    average_crps_by_basin : dict, optional
        Dictionary with structure [model][basin] containing average CRPS values
    
    Returns:
    --------
    list of tuples : [(fig, basin_idx, sample_idx), ...]
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Get available basins from first model
    if models is None:
        models = ['Conditional', 'Seeded (Static)', 'Seeded (Variable)', 'Probabilistic']
    
    first_model = list(forecasts_by_model_and_basin.keys())[0]
    available_basins = list(forecasts_by_model_and_basin[first_model].keys())
    
    # Get number of samples (assuming all basins have same number)
    n_samples = forecasts_by_model_and_basin[first_model][available_basins[0]].shape[0]
    
    plots = []
    
    for i in range(n_plots):
        # Select basin and sample
        if basin_idx is None:
            selected_basin = np.random.choice(available_basins)
        else:
            selected_basin = basin_idx
        
        if sample_idx is None:
            selected_sample = np.random.randint(0, n_samples)
        else:
            selected_sample = sample_idx
        
        # Generate plot
        fig = plot_ensemble_forecasts(
            forecasts_by_model_and_basin,
            truth_by_basin,
            metadata_by_basin,
            selected_basin,
            selected_sample,
            models=models,
            colors=colors,
            average_crps_by_basin=average_crps_by_basin
        )
        
        plots.append((fig, selected_basin, selected_sample))
        
        print(f"Plot {i+1}: Basin = {selected_basin}, Sample = {selected_sample}")
    
    return plots
    

"""
Distribution Related Functions
"""


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


"""
Ensemble Generation Functions
"""

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


"""
Utility Functions for analyzing CRPS (Continuous Ranked Probability Score) by basin.
"""


def get_basin_name(basin_idx, csv_base_path="/perm/mokr/Caravans/Caravan/attributes/"):
    # Split by underscore and take the first part as location
    location = basin_idx.split('_')[0]
    
    csv_filename = f"{location}/attributes_other_{location}.csv"
    csv_path = os.path.join(csv_base_path, csv_filename)
    df = pd.read_csv(csv_path, engine='python')
    gauge_name = df[df['gauge_id'] == basin_idx]['gauge_name'].values[0]

    return gauge_name


def create_metric_crps_df(metric, model_names, display_names, model_crps_per_example):
    per_example_data = {}
    for model_name in model_names:
        if model_name in model_crps_per_example:
            display_name = display_names[model_names.index(model_name)] if model_names.index(model_name) < len(display_names) else model_name
            per_example_data[display_name] = model_crps_per_example[model_name][metric]
    
    metric_crps_df = pd.DataFrame(per_example_data)
    print(f"{metric} CRPS DataFrame shape: {metric_crps_df.shape}")
    
    return metric_crps_df


def create_basin_crps_tables(crps_per_leadtime, basins):
    """
    Create tables showing average CRPS at each leadtime for each basin.
    
    Parameters:
    - crps_per_leadtime: dict with keys ['Conditional', 'Seeded (Static)', 'Seeded (Variable)', 'Probabilistic']
    - basins: list of basin identifiers corresponding to each forecast
    
    Returns:
    - Dictionary of DataFrames, one for each model type
    """
    
    # Get the model types (excluding non-CRPS metrics if present)
    model_types = [key for key in crps_per_leadtime.keys() if key != "NLL"] 
    # Dictionary to store results for each model
    basin_tables = {}
    
    for model_type in model_types:
        print(f"\nProcessing {model_type}...")
        
        # Get CRPS data for this model
        crps_data = crps_per_leadtime[model_type]
        
        # Convert to numpy array if it's a list of tensors
        if isinstance(crps_data[0], torch.Tensor):
            crps_array = torch.stack(crps_data).cpu().numpy().squeeze()
        else:
            crps_array = np.array(crps_data).squeeze()

        # crps_array should be shape (n_forecasts, n_leadtimes)
        n_forecasts, n_leadtimes = crps_array.shape
        
        # Create a dictionary to group by basin
        basin_crps = defaultdict(list)
        
        # Group CRPS scores by basin
        basins_array = np.array(basins)
        unique_basins = np.unique(basins_array)

        
        # Method 1: Using broadcasting and boolean masking (fastest for most cases)
        basin_avg_crps = {}
        for basin in unique_basins:
            mask = basins_array == basin 
            mask = mask.squeeze()  # (shape, n_forecasts,)
            basin_forecasts = crps_array[mask]  # Get all forecasts for this basin
            basin_avg_crps[basin] = np.mean(basin_forecasts, axis=0) 

        
        
        # Create DataFrame
        leadtime_columns = [f'Leadtime_{i+1}' for i in range(n_leadtimes)]
        
        # Convert to DataFrame
        df_data = []
        for basin, avg_crps in basin_avg_crps.items():
            row = [basin] + list(avg_crps)
            df_data.append(row)
        
        df = pd.DataFrame(df_data, columns=['Basin'] + leadtime_columns)
        df = df.sort_values('Basin').reset_index(drop=True)
        
        
        basin_tables[model_type] = df
        
        print(f"{model_type} table created:")
        print(f"- {len(basin_avg_crps)} unique basins")
        print(f"- {n_leadtimes} leadtimes")
        print(f"- Total forecasts: {n_forecasts}")
    
    return basin_tables


def print_basin_tables(basin_tables, max_rows=None):
    """
    Print the basin tables in a nice format.
    
    Parameters:
    - basin_tables: Dictionary of DataFrames from create_basin_crps_tables
    - max_rows: Maximum number of rows to display (None for all)
    """
    
    for model_type, df in basin_tables.items():
        print(f"\n{'='*60}")
        print(f"AVERAGE CRPS BY LEADTIME - {model_type.upper()}")
        print(f"{'='*60}")
        
        if max_rows is not None and len(df) > max_rows:
            print(df.head(max_rows))
            print(f"... ({len(df) - max_rows} more rows)")
        else:
            print(df)
        
        avg_crps_by_basin = df[df.columns[1:]].mean(axis=1)
        
        min_idx = avg_crps_by_basin.idxmin()
        max_idx = avg_crps_by_basin.idxmax()
        
        print(f"\nBasin with lowest average CRPS: {df.loc[min_idx, 'Basin']} (CRPS: {avg_crps_by_basin[min_idx]:.4f})")
        print(f"Basin with highest average CRPS: {df.loc[max_idx, 'Basin']} (CRPS: {avg_crps_by_basin[max_idx]:.4f})")


def process_crps_df(crps_df, basins, metric_name):
    crps_df['Basin'] = basins    
    grouped = crps_df.groupby('Basin', as_index=False).agg(lambda x: np.mean(np.stack(x), axis=0))
    long_df = grouped.melt(id_vars='Basin', var_name='Model', value_name='CRPS').explode('CRPS')
    long_df['Leadtime'] = long_df.groupby(['Basin', 'Model']).cumcount() + 1
    long_df['CRPS'] = long_df['CRPS'].astype(float)
        
    return long_df


def add_coordinates_to_basin_table(basin_tables_df, csv_base_path="/perm/mokr/Caravans/Caravan/attributes/"):
    """
    Add latitude and longitude columns to the basin tables DataFrame by reading location CSV files.
    
    Parameters:
    - basin_tables_df: DataFrame with Basin column in format {location}_{gauge_id}
    - csv_base_path: Base path to the CSV files
    
    Returns:
    - DataFrame with added 'Latitude' and 'Longitude' columns
    """
    
    # Extract unique locations from basin names
    print("Extracting unique locations from basin names...")
    locations = set()
    basin_to_location = {}
    
    for basin in basin_tables_df['Basin']:

        # Split by underscore and take the first part as location
        location = basin.split('_')[0]
        locations.add(location)
        basin_to_location[basin] = location
    
    print(f"Found {len(locations)} unique locations: {sorted(locations)}")
    
    # Dictionary to store coordinates for each location
    location_coordinates = {}
    
    # Read CSV files for each location
    for location in locations:
        csv_filename = f"{location}/attributes_other_{location}.csv"
        csv_path = os.path.join(csv_base_path, csv_filename)
        

        df_coords = pd.read_csv(csv_path)
        location_coordinates[location] = df_coords

    # Add coordinate columns to the basin tables
    print("\nAdding coordinates to basin table...")
    
    # Initialize new columns
    basin_tables_df['Latitude'] = np.nan
    basin_tables_df['Longitude'] = np.nan
    basin_tables_df['Location'] = None
    
    # Iterate through each row and add coordinates
    for idx, row in basin_tables_df.iterrows():
        basin = row['Basin']
        location = basin_to_location.get(basin)
        
        if location is None:
            continue
            
        basin_tables_df.at[idx, 'Location'] = location

        gauge_id = basin  # Everything after first underscore
        
        # Look up coordinates for this gauge_id in the location's CSV
        coord_df = location_coordinates.get(location)
        if coord_df is None:
            continue
        
        gauge_col = 'gauge_id'
                
        # Find matching row in coordinates CSV
        matching_rows = coord_df[coord_df[gauge_col].astype(str) == str(gauge_id)]

        
        # Extract coordinates
        coord_row = matching_rows.iloc[0]
        
        # Try different possible column names for latitude/longitude
        lat_col = 'gauge_lat'
        lon_col = 'gauge_lon'
        
        lat_val = coord_row[lat_col]
        lon_val = coord_row[lon_col]

        basin_tables_df.at[idx, 'Latitude'] = float(lat_val)
        basin_tables_df.at[idx, 'Longitude'] = float(lon_val)
    
    # Summary statistics
    total_basins = len(basin_tables_df)
    basins_with_coords = basin_tables_df[['Latitude', 'Longitude']].notna().all(axis=1).sum()
    
    print(f"\n{'='*50}")
    print(f"COORDINATE MATCHING SUMMARY")
    print(f"{'='*50}")
    print(f"Total basins: {total_basins}")
    print(f"Basins with coordinates: {basins_with_coords}")
    print(f"Success rate: {basins_with_coords/total_basins*100:.1f}%")
    
    # Reorder columns to put coordinates near Basin
    cols = basin_tables_df.columns.tolist()
    # Move Location, Latitude, Longitude to after Basin
    basin_idx = cols.index('Basin')
    new_cols = cols[:basin_idx+1] + ['Latitude', 'Longitude'] + [c for c in cols if c not in ['Basin', 'Latitude', 'Longitude']]
    basin_tables_df = basin_tables_df[new_cols]
    
    return basin_tables_df


def create_model_dict(df):
    """Create a dictionary splitting dataframe by Model."""
    return {model: df[df['Model'] == model] for model in df['Model'].unique()}


def flatten_batch_dict_of_lists(ensemble_summaries, nested_dicts=True, batch_dim = 1):
    """
    Flatten batch dimension in a dictionary of lists of tensors or dictionaries of tensors.

    Args:
        ensemble_summaries (dict): The input dictionary. Each value is a list of items,
                                   where each item is either a tensor or a dictionary of tensors.
        nested_dicts (bool): If True, expects each list item to be a dict of tensors.
                             If False, expects each list item to be a tensor.

    Returns:
        dict: A new dictionary with the batch dimension flattened into the list level.
    """
    new_ensemble = {}

    for key, lst in ensemble_summaries.items():
        new_list = []

        for entry in lst:
            if nested_dicts:
                batch_size = next(iter(entry.values())).shape[batch_dim]
                for i in range(batch_size):
                    new_entry = {k: v[..., i].clone() for k, v in entry.items()}
                    new_list.append(new_entry)
            else:
                batch_size = entry.shape[batch_dim]
                for i in range(batch_size):
                    new_list.append(entry.select(batch_dim, i).clone())

        new_ensemble[key] = new_list

    return new_ensemble


sys.path.append(str(functions_path))

