
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



def _plot_boxplot(ax, ensemble_predictions, true_discharge=None, title=None):
    """
    Create a detailed boxplot with multiple quantiles for ensemble predictions.
    
    Args:
        ax (matplotlib.axes.Axes): Axis to plot on
        ensemble_predictions (np.ndarray): Ensemble predictions
        true_discharge (np.ndarray, optional): True discharge values
        title (str, optional): Plot title
    """
    # Transpose to get distribution for each time step
    ensemble_T = ensemble_predictions.T
    
    # Calculate quantiles
    quantiles = [0.05, 0.25, 0.5, 0.75, 0.95]
    boxplot_data = []
    
    # Compute quantiles for each time step
    for time_step in ensemble_T:
        step_quantiles = np.percentile(time_step, [q * 100 for q in quantiles])
        boxplot_data.append(step_quantiles)
    
    # Convert to numpy array for easier manipulation
    boxplot_data = np.array(boxplot_data)
    
    # Custom boxplot with multiple quantiles
    box_width = 0.6
    whisker_width = 0.3
    
    # Plot boxes
    for i, (q1, q2, median, q3, q4) in enumerate(boxplot_data):
        # Body of the box (50% interval)
        ax.add_patch(plt.Rectangle((i - box_width/2, q2), box_width, q3 - q2, 
                                   facecolor='lightblue', edgecolor='blue', alpha=0.5))
        
        # Median line
        ax.plot([i - box_width/2, i + box_width/2], [median, median], color='blue', linewidth=2)
        
        # Lower whisker (5%)
        ax.plot([i, i], [q1, q2], color='blue', linewidth=1)
        ax.plot([i - whisker_width/2, i + whisker_width/2], [q1, q1], color='blue', linewidth=1)
        
        # Upper whisker (95%)
        ax.plot([i, i], [q3, q4], color='blue', linewidth=1)
        ax.plot([i - whisker_width/2, i + whisker_width/2], [q4, q4], color='blue', linewidth=1)
    
    # Plot true discharge if provided
    if true_discharge is not None:
        ax.plot(range(len(true_discharge)), true_discharge, 
                color='black', linewidth=2, label='Observations', marker='o')
    
    ax.set_title(title or 'Ensemble Predictions - Detailed Boxplot')
    ax.set_xlabel('Lead Time')
    ax.set_ylabel('Normalised River Discharge')
    ax.set_xticks(range(len(ensemble_T)))
    
    if true_discharge is not None:
        ax.legend()


def generate_greedy_trajectories(ensemble_predictions, param_predictions, model, Fore_X_Chunk_Torch, Hist_X_Chunk_Torch):
    """
    Generate greedy trajectories by selecting the next timestep value that maximizes the PDF.
    Optimized for GPU performance.
    
    Args:
        ensemble_predictions (torch.Tensor): Shape (num_members, batch_size, num_steps)
        param_predictions (dict): Contains 'mu', 'b', 'tau', 'pi' for each timestep
        model: The model used for predictions
        Fore_X_Chunk_Torch (torch.Tensor): Forecast features tensor
        Hist_X_Chunk_Torch (torch.Tensor): Historical features tensor
    
    Returns:
        torch.Tensor: Shape (num_members, batch_size, num_steps), greedy trajectories
    """
    num_members, batch_size, num_steps = ensemble_predictions.shape
    device = ensemble_predictions.device
    
    # Pre-allocate memory for trajectories and selected indices
    trajectories = torch.zeros((num_members, batch_size, num_steps), device=device)
    selected_indices = torch.full((num_members, batch_size, num_steps), -1, dtype=torch.int64, device=device)
    
    # Process all batches in parallel for the first timestep
    if num_steps > 0:
        step_values = ensemble_predictions[:, :, 0]  # (num_members, batch_size)
        
        # Generate permutations for all batches in one go
        # For each batch, we need a different permutation of indices
        batch_permutations = torch.stack([torch.randperm(num_members, device=device) 
                                         for _ in range(batch_size)], dim=1)  # (num_members, batch_size)
        
        # Use advanced indexing to assign values efficiently
        batch_indices = torch.arange(batch_size, device=device)
        for m in range(num_members):
            member_indices = batch_permutations[m]  # (batch_size,)
            trajectories[m, batch_indices, 0] = step_values[member_indices, batch_indices]
            selected_indices[m, batch_indices, 0] = member_indices
    
    # Pre-allocate the modified forecast features for all members at once
    all_fore_chunks = Fore_X_Chunk_Torch.clone().unsqueeze(0).expand(num_members, -1, -1, -1).contiguous()
    batched_hist_chunks = Hist_X_Chunk_Torch.repeat(num_members, 1, 1)    
    
    # For steps beyond the first one
    for step in range(1, num_steps):
        step_values = ensemble_predictions[:, :, step]  # (num_members, batch_size)
        
        # Create batch input for the model - much more efficient than running per member
        # Update forecast chunks for all members simultaneously
        for m in range(num_members):
            all_fore_chunks[m, :, 0:step, -1] = 1
            all_fore_chunks[m, :, 0:step, -2] = trajectories[m, :, 0:step]
        
        # Reshape for batch processing
        batched_fore_chunks = all_fore_chunks.reshape(-1, all_fore_chunks.shape[2], all_fore_chunks.shape[3])
        # batched_hist_chunks = Hist_X_Chunk_Torch.repeat(num_members, 1, 1)
        
        # Run a single forward pass with all members batched together
        with torch.no_grad():
            batched_predictions = model(batched_hist_chunks.to(torch.float32), 
                                      batched_fore_chunks.to(torch.float32))[1][:, step, :]
        
        # Reshape back to per-member format
        new_predictions = batched_predictions.reshape(num_members, batch_size, -1)
        
        # Transform parameters once for all members
        all_param_predictions = ML_functions.transform_CMAL_parameters_multi(new_predictions)
        
        all_param_dictionaries = {
            'mu': all_param_predictions[:, :, ::4],   # mu is at indices 0 mod 4
            'b': all_param_predictions[:, :, 1::4],   # b is at indices 1 mod 4
            'tau': all_param_predictions[:, :, 2::4], # tau is at indices 2 mod 4
            'pi': all_param_predictions[:, :, 3::4],  # pi is at indices 3 mod 4
        }
        
        # Create mask to avoid selecting duplicate indices
        available_mask = torch.ones((num_members, batch_size), dtype=torch.bool, device=device)
        
        # Process each member sequentially (this part is harder to parallelize due to dependencies)
        for m in range(num_members):
            member_params = {
                'mu': all_param_dictionaries['mu'][m].unsqueeze(0),
                'b': all_param_dictionaries['b'][m].unsqueeze(0),
                'tau': all_param_dictionaries['tau'][m].unsqueeze(0),
                'pi': all_param_dictionaries['pi'][m].unsqueeze(0)
            }
            
            # Compute log-likelihoods for all possible choices
            log_likelihoods = compute_log_likelihood(step_values, member_params)  # Shape: (num_members, batch_size)
            
            # Apply mask to prevent duplicate selections
            masked_log_likelihoods = log_likelihoods.clone()
            masked_log_likelihoods[~available_mask] = float('-inf')
            
            # Get the best indices
            chosen_indices = torch.argmax(masked_log_likelihoods, dim=0)  # (batch_size,)
            
            # Update available mask to mark indices as used
            for b in range(batch_size):
                available_mask[chosen_indices[b], b] = False
            
            # Store chosen values
            batch_indices = torch.arange(batch_size, device=device)
            trajectories[m, batch_indices, step] = step_values[chosen_indices, batch_indices]
            selected_indices[m, batch_indices, step] = chosen_indices
    
    return trajectories
def kl_divergence(p_samples, q_samples, bins=10):
    # Create histograms for both distributions
    p_counts, bin_edges = np.histogram(p_samples, bins=bins, density=True)
    q_counts, _ = np.histogram(q_samples, bins=bin_edges, density=True)
    
    # Add small epsilon to avoid division by zero or log(0)
    epsilon = 1e-10
    p_counts = p_counts + epsilon
    q_counts = q_counts + epsilon


def plot_uniform_qq(dispersion_df, title = 'Q-Q Plot: All Statistics vs Uniform'):
    '''
    Creates Quantile-Quantile plots comparing each statistic's distribution 
    to a Uniform distribution.
    
    Parameters:
    dispersion_df: DataFrame where rows are percentile bins, columns are statistics
                   Values should be percentage of samples in each bin
    '''
    import matplotlib.pyplot as plt
    import numpy as np
    import scipy.stats as stats
    from matplotlib.gridspec import GridSpec
    
    # Get list of statistics (column names)
    statistics = dispersion_df.columns
    n_stats = len(statistics)
    
    
    # For each statistic
    for i, stat in enumerate(statistics):
        # Get the data - convert percentages to proportions
        data = dispersion_df[stat].values / 100.0
        
        # Convert from bin percentages to individual data points
        # We'll replicate each bin value according to its percentage
        bin_edges = np.linspace(0, 1, len(data) + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        # Generate synthetic data points from bin frequencies
        synthetic_data = []
        for j, bin_center in enumerate(bin_centers):
            # Number of points to generate for this bin
            n_points = int(round(data[j] * 100))
            # Add points at bin center with small jitter
            if n_points > 0:
                jitter = np.random.uniform(-0.1/len(data), 0.1/len(data), n_points)
                synthetic_data.extend([bin_center + j for j in jitter])

        
        # Generate uniform theoretical quantiles
        uniform_quantiles = np.linspace(0, 1, len(synthetic_data))
        
        # Sort our data to get empirical quantiles
        empirical_quantiles = np.sort(synthetic_data)


    # Optional: Also create a single-plot version with all statistics
    plt.figure(figsize=(10, 10))
    for stat in statistics:
        data = dispersion_df[stat].values / 100.0
        
        # Generate synthetic data points from bin frequencies
        synthetic_data = []
        for j, bin_center in enumerate(bin_centers):
            n_points = int(round(data[j] * 100))
            if n_points > 0:
                jitter = np.random.uniform(-0.1/len(data), 0.1/len(data), n_points)
                synthetic_data.extend([bin_center + j for j in jitter])
        
        # Generate uniform theoretical quantiles
        uniform_quantiles = np.linspace(0, 1, len(synthetic_data))
        
        # Sort our data to get empirical quantiles
        empirical_quantiles = np.sort(synthetic_data)
        
        # Plot this statistic
        plt.scatter(uniform_quantiles, empirical_quantiles, alpha=0.5, label=stat)
        
    # Add diagonal line
    plt.plot([0, 1], [0, 1], 'r--', linewidth=3, label='Uniform')  # Thicker line
    plt.title(f'{title}', fontsize=20)  # Larger title
    plt.xlabel('Uniform Theoretical Quantiles', fontsize=18)
    plt.ylabel('Empirical Quantiles', fontsize=18)
    plt.grid(True, alpha=0.3)
    
    # Make tick marks bigger
    plt.tick_params(axis='both', which='major', width=2, length=8, labelsize=16)
    
    # Make legend bigger with larger markers
    legend = plt.legend(prop={'size': 14}, frameon=True, markerscale=2)

    plt.tight_layout()

    plt.show()
    # Normalize
    p_counts = p_counts / np.sum(p_counts)
    q_counts = q_counts / np.sum(q_counts)
    
    # Calculate KL divergence using scipy.special.rel_entr
    kl_div = np.sum(rel_entr(p_counts, q_counts))
    return kl_div




def get_ensemble_members(model, Hist_X_Chunk_Torch, Fore_X_Chunk_Torch, num_members=1):
    """
    Generate ensemble predictions for discharge.
    
    Args:
        model (torch.nn.Module): The trained machine learning model
        Hist_X_Chunk_Torch (torch.Tensor): Historical input chunk
        Fore_X_Chunk_Torch (torch.Tensor): Forecast input chunk
        num_members (int, optional): Number of ensemble members to generate. Defaults to 1.
    
    Returns:
        torch.Tensor: Ensemble predictions of discharge
    """
    
    num_steps = len(Fore_X_Chunk_Torch[0])
    ensemble_predictions = []
    
    for member in range(num_members):
        # Initialize tensors for predictions and mask
        Pred_Discharge = torch.zeros(num_steps)
        Discharge_Mask = torch.zeros(num_steps)
        
        for step in range(num_steps):
            # Prepare input by concatenating forecast chunk, current predictions, and mask
            Binary_Fore_X_Chunk_Torch = torch.cat([Fore_X_Chunk_Torch,  
            Pred_Discharge.unsqueeze(0).unsqueeze(-1), Discharge_Mask.unsqueeze(0).unsqueeze(-1)
                                                  ], dim= -1)
            
            # Draw a sample from the model
            with torch.no_grad():
                param_predictions = model(
                    Hist_X_Chunk_Torch.to(torch.float32), 
                    Binary_Fore_X_Chunk_Torch.to(torch.float32)
                )[1]
            
            # Transform parameters and draw sample (assuming external functions)
            param_predictions_step = ML_functions.transform_CMAL_parameters_multi(param_predictions)[:,step]
    
            param_prediction = {
                'mu': param_predictions_step[:, ::4],   # mu is at indices 0 mod 4
                'b': param_predictions_step[:, 1::4],   # b is at indices 1 mod 4 (ensure positive)
                'tau': param_predictions_step[:, 2::4], # tau is at indices 2 mod 4 (ensure 0-1)
                'pi': param_predictions_step[:, 3::4],  # pi is at indices 3 mod 4
            }

           
            sample = draw_from_CMAL_distribution_np(param_prediction, num_samples=1)
            
            # Update predictions and mask
            Pred_Discharge[step] = torch.tensor(sample)
            Discharge_Mask[step] = 1
        
        ensemble_predictions.append(Pred_Discharge)

    
    return torch.stack(ensemble_predictions)

    