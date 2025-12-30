"""
FILE CONTENTS
================================================================================

FUNCTIONS:
--------------------------------------------------------------------------------
# Distribution Utilities
 1. compute_log_likelihood
    Compute the log likelihood for each ensemble member at a given timestep.
 2. compute_CDF
    Compute CDF of predicted distribution
 3. crps_loss
    Calculate CRPS loss for a batch of predictions

# Efficient Processing
 4. initialize_storage_efficient
    Pre-allocate numpy arrays instead of empty lists.
 5. process_ensemble_predictions_efficient
    Efficient version that stores directly to pre-allocated arrays.

CLASSES:
--------------------------------------------------------------------------------
 1. CMALLoss
    No description available
    Methods:
      - __init__: No description available
      - forward: No description available

 2. CRPSLoss
    PyTorch module implementing CRPS loss for a mixture of asymmetric Laplacian d...
    Methods:
      - __init__: Initialize the CRPS loss function
      - forward: Calculate CRPS loss for a batch of predictions
      - _compute_mixture_cdf: Compute CDF of a mixture of asymmetric Laplacian distributions

 3. KGELoss
    Kling-Gupta Efficiency loss for batched predictions with multiple timesteps.
    Methods:
      - __init__: No description available
      - forward: simulated (torch.Tensor): Simulated values of shape [batch_size, timesteps]

================================================================================
"""

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from collections import defaultdict
import sys
from pathlib import Path


functions_path = Path("/home/mokr/Loss_Functions_Paper/ML_Functions/")
sys.path.append(str(functions_path))

import ML_functions
from ML_functions import load_and_unnormalize
from ML_Metrics import calculate_nse, calculate_kge, compute_crps, compute_crps_np, calculate_overall_crps, calculate_model_crps, variogram_score, variogram_score_torch, get_member_summaries, get_member_summaries_torch
from ML_Processing import process_ensemble_predictions

def compute_log_likelihood(y, param_predictions):
    """
    Compute the log likelihood for each ensemble member at a given timestep.
    
    Args:
        y (np.ndarray): Target value (shape: (num_trajectories, ))
        param_predictions (dict): Dictionary with 'mu', 'b', 'tau', 'pi' arrays (shape: (num_members, ))
    
    Returns:
        np.ndarray: Log-likelihood values for each ensemble member (shape: (num_trajectories, num_members))
    """
    m = param_predictions['mu']  # Mean
    b = param_predictions['b']   # Scale (>0)
    t = param_predictions['tau']  # Asymmetry (0-1)
    p = param_predictions['pi']  # Mixture weights

    y = y.unsqueeze(-1)  # Shape (num_trajectories, 1)
    m = m.clone().detach()   # (num_members,)
    b = b.clone().detach()   # (num_members,)
    t = t.clone().detach()   # (num_members,)
    p = p.clone().detach()   # (num_members,)

    error = y - m
    log_like = (torch.log(t + 1e-8) + 
                torch.log(1.0 - t + 1e-8) - 
                torch.log(b + 1e-8) - 
                torch.max(t * error, (t - 1.0) * error) / b)

    log_weights = torch.log(p + 1e-8)  # Prevent log(0)
    result = torch.logsumexp(log_weights + log_like, dim=-1)  # Shape: (num_trajectories, num_members)

    
    return result  


def compute_CDF(y, param_predictions):
    """
    Compute CDF of predicted distribution
    
    Args:
        y (torch.Tensor): Points at which to evaluate the CDF (shape: (batch_size, num_points))
        param_predictions (dict): Dictionary with 'mu', 'b', 'tau', 'pi' arrays
            mu (torch.Tensor): Mean parameters (shape: (batch_size, num_components))
            b (torch.Tensor): Scale parameters (shape: (batch_size, num_components))
            tau (torch.Tensor): Shape parameters (shape: (batch_size, num_components))
            pi (torch.Tensor): Mixture weights (shape: (batch_size, num_components))
    
    Returns:
        torch.Tensor: CDF values at the points y (shape: (batch_size, num_points))
    """
    m = param_predictions['mu']  # (batch_size, num_components)
    b = param_predictions['b']   # (batch_size, num_components)
    t = param_predictions['tau'] # (batch_size, num_components)
    pi = param_predictions['pi'] # (batch_size, num_components)
    
    # Ensure pi sums to 1 along component dimension
    pi = pi / torch.sum(pi, dim=1, keepdim=True)
    
    # Expand dimensions for broadcasting
    # y: (batch_size, num_points, 1)
    # m: (batch_size, 1, num_components)
    y_expanded = y.unsqueeze(-1)  
    m_expanded = m.unsqueeze(1)
    b_expanded = b.unsqueeze(1)
    t_expanded = t.unsqueeze(1)
    pi_expanded = pi.unsqueeze(1)
    
    # Calculate normalized error
    error = y_expanded - m_expanded  # (batch_size, num_points, num_components)
    
    # Calculate CDF value based on error sign
    positive_mask = error > 0
    negative_mask = ~positive_mask
    
    # Initialize CDF tensor
    component_cdfs = torch.zeros_like(error)
    
    # Handle positive errors
    if torch.any(positive_mask):
        component_cdfs[positive_mask] = t_expanded.expand_as(error)[positive_mask] * torch.exp(
            (1 - t_expanded.expand_as(error)[positive_mask]) * error[positive_mask] / b_expanded.expand_as(error)[positive_mask]
        )
    
    # Handle negative errors
    if torch.any(negative_mask):
        component_cdfs[negative_mask] = 1 - (1 - t_expanded.expand_as(error)[negative_mask]) * torch.exp(
            (-t_expanded.expand_as(error)[negative_mask]) * error[negative_mask] / b_expanded.expand_as(error)[negative_mask]
        )
    
    # Apply mixture weights and sum over components
    cdf_values = torch.sum(pi_expanded.expand_as(component_cdfs) * component_cdfs, dim=2)
    
    return cdf_values


def crps_loss(param_predictions, observations, num_integration_points=100):
    """
    Calculate CRPS loss for a batch of predictions
    
    Args:
        param_predictions (dict): Dictionary with distribution parameters
        observations (torch.Tensor): Observed values (shape: (batch_size,))
        num_integration_points (int): Number of points to use for numerical integration
        
    Returns:
        torch.Tensor: CRPS loss (shape: (batch_size,))
    """
    batch_size = observations.shape[0]
    
    # Define integration range
    # We need to consider the support of the distribution
    std_estimate = torch.mean(param_predictions['b'])
    mean_estimate = torch.mean(param_predictions['mu'])
    
    # Define integration range: mean ± 5*std should cover most of the distribution
    lower_bound = (mean_estimate - 5 * std_estimate).item()
    upper_bound = (mean_estimate + 5 * std_estimate).item()
    
    # Create integration points
    integration_points = torch.linspace(lower_bound, upper_bound, num_integration_points)
    integration_points = integration_points.repeat(batch_size, 1)  # (batch_size, num_integration_points)
    
    # Compute CDF at integration points
    cdf_values = compute_CDF(integration_points, param_predictions)  # (batch_size, num_integration_points)
    
    # Calculate heaviside step function (1 if integration_point >= observation, 0 otherwise)
    # First, expand observations to match integration points shape
    observations_expanded = observations.unsqueeze(1).expand_as(integration_points)
    heaviside = (integration_points >= observations_expanded).float()
    
    # Calculate squared difference
    squared_diff = (cdf_values - heaviside) ** 2
    
    # Numerical integration using trapezoidal rule
    dx = (upper_bound - lower_bound) / (num_integration_points - 1)
    integrated_values = 0.5 * dx * (squared_diff[:, :-1] + squared_diff[:, 1:])
    crps = torch.sum(integrated_values, dim=1)  # Sum over integration points
    
    return crps


def initialize_storage_efficient(num_samples, num_members = 11, num_leadtimes=10):
    """
    Pre-allocate numpy arrays instead of empty lists.
    Use float32 to save 50% memory compared to float64.
    """
    
    model_names = ["CRPS", "Fixed_Seeded", "Non_Fixed_Seeded", "NonBinary"]
    
    # Initialize dictionaries with pre-allocated arrays
    summary_keys = ['total_flow', 'variance', 'gamma', 'autoregression', 'num_rise', 'fdc_curve']

    ensemble_summaries = {
        name: {
            key: np.full((num_samples, num_members), np.nan, dtype=np.float32)
            for key in summary_keys
        }
        for name in model_names + ["Discharge"]
    }
    
    crps_per_leadtime = {
        name: np.full((num_samples, num_leadtimes), np.nan, dtype=np.float32)
        for name in model_names
    }
    
    variogram_scores = {
        name: np.full(num_samples, np.nan, dtype=np.float32)
        for name in ["CRPS", "NLL", "Fixed_Seeded", "Non_Fixed_Seeded", "NonBinary"]
    }
    
    NSE_scores = {
        name: np.full(num_samples, np.nan, dtype=np.float32)
        for name in model_names
    }
    
    KGE_scores = {
        name: np.full(num_samples, np.nan, dtype=np.float32)
        for name in model_names
    }

    stored_forecasts = {
        name: np.full((num_samples, num_members, num_leadtimes), np.nan, dtype=np.float16)
        for name in model_names + ["Discharge"]
    }
    
    basin_forecasts = defaultdict(lambda: {
        name: [] for name in model_names + ["Discharge"]
    })
    
    # Metadata
    metadata = np.zeros(num_samples, dtype=[
        ('date_idx', object),
        ('basin_idx', object),
    ])
    
    return (ensemble_summaries, crps_per_leadtime, variogram_scores, 
            stored_forecasts, NSE_scores, KGE_scores, basin_forecasts, metadata)


def process_ensemble_predictions_efficient(
    model_predictions, true_discharge, basin_idx, idx,
    ensemble_summaries, crps_per_leadtime, variogram_scores, stored_forecasts,
    NSE_scores, KGE_scores, basin_forecasts, variogram_p):
    """
    Efficient version that stores directly to pre-allocated arrays.
    
    Args:
        model_predictions: Dictionary with model names as keys and prediction tensors as values
        true_discharge: Ground truth discharge values
        basin_idx: Basin index
        idx: Current sample index (for array storage)
        ... (other args same as before)
        variogram_p: Parameter for variogram score
    """
    
    # Process each model's predictions
    for model_name, predictions in model_predictions.items():
        if model_name == "Discharge":
            continue
        
        # Convert to numpy and float32 immediately (avoid keeping tensors in memory)
        predictions_np = predictions.cpu().numpy().astype(np.float32) if torch.is_tensor(predictions) else predictions.astype(np.float32)
        true_discharge_np = true_discharge.cpu().numpy().astype(np.float32) if torch.is_tensor(true_discharge) else true_discharge.astype(np.float32)
        
        # Ensemble summaries - store single float value
        if model_name in ensemble_summaries:
            summary = get_member_summaries_torch(predictions)
            # summary is a dict like {'total_flow': tensor[num_members, num_batches], ...}
            # We store all member values, squeezing out the batch dimension
            for key, value in summary.items():
                if key in ensemble_summaries[model_name]:
                    if torch.is_tensor(value):
                        # Shape is [num_members, num_batches], squeeze batch dim and convert
                        val = value.squeeze().cpu().numpy().astype(np.float32)
                    else:
                        val = np.squeeze(value).astype(np.float32)
                    ensemble_summaries[model_name][key][idx] = val

        
        # CRPS per leadtime - store array of values
        if model_name in crps_per_leadtime:
            crps = compute_crps(predictions, true_discharge)
            if torch.is_tensor(crps):
                crps_per_leadtime[model_name][idx] = crps.cpu().numpy().astype(np.float32)
            else:
                crps_per_leadtime[model_name][idx] = crps.astype(np.float32)
            
            # Variogram scores
            vario = variogram_score_torch(predictions, true_discharge, variogram_p)
            variogram_scores[model_name][idx] = np.float32(vario.item() if torch.is_tensor(vario) else vario)
        
        # Stored forecasts
        stored_forecasts[model_name][idx] = predictions_np.squeeze()
        
        # NSE scores
        if model_name in NSE_scores:
            nse = calculate_nse(true_discharge.squeeze(), predictions.squeeze()[0])
            NSE_scores[model_name][idx] = np.float32(nse.item() if torch.is_tensor(nse) else nse)
        
        # KGE scores
        if model_name in KGE_scores:
            kge = calculate_kge(true_discharge.squeeze(), predictions.squeeze()[0])
            KGE_scores[model_name][idx] = np.float32(kge.item() if torch.is_tensor(kge) else kge)
        
        # Basin forecasts - keep as lists for sparse storage
        if basin_idx in basin_forecasts and model_name in basin_forecasts[basin_idx]:
            # Store only first timestep and convert to float32
            basin_forecasts[basin_idx][model_name].append(
                predictions[:, 0, :].cpu().numpy().astype(np.float32) if torch.is_tensor(predictions) 
                else predictions[:, 0, :].astype(np.float32)
            )
    
    # Handle discharge (ground truth) separately
    if "Discharge" in ensemble_summaries:
        summary = get_member_summaries_torch(true_discharge.unsqueeze(0))
        for key, value in summary.items():
            if key in ensemble_summaries["Discharge"]:
                if torch.is_tensor(value):
                    # Shape is [num_members, num_batches], squeeze and convert
                    val = value.squeeze().cpu().numpy().astype(np.float32)
                else:
                    val = np.squeeze(value).astype(np.float32)
                ensemble_summaries["Discharge"][key][idx] = val
    
    if stored_forecasts is not None and "Discharge" in stored_forecasts:
        true_discharge_np = true_discharge.cpu().numpy().astype(np.float32) if torch.is_tensor(true_discharge) else true_discharge.astype(np.float32)
        
        if isinstance(stored_forecasts["Discharge"], dict):
            stored_forecasts["Discharge"]['median'][idx] = true_discharge_np.squeeze()
            stored_forecasts["Discharge"]['q10'][idx] = true_discharge_np.squeeze()
            stored_forecasts["Discharge"]['q90'][idx] = true_discharge_np.squeeze()
        else:
            stored_forecasts["Discharge"][idx] = true_discharge_np
    
    if basin_idx in basin_forecasts and "Discharge" in basin_forecasts[basin_idx]:
        basin_forecasts[basin_idx]["Discharge"].append(
            true_discharge[0, :].cpu().numpy().astype(np.float32) if torch.is_tensor(true_discharge)
            else true_discharge[0, :].astype(np.float32)
        )


class CRPSLoss(nn.Module):
    """
    PyTorch module implementing CRPS loss for a mixture of asymmetric Laplacian distributions
    
    This loss can be used directly in standard PyTorch training loops.
    """
    def __init__(self, num_integration_points=100):
        """
        Initialize the CRPS loss function
        
        Args:
            num_integration_points (int): Number of points to use for numerical integration
        """
        super(CRPSLoss, self).__init__()
        self.num_integration_points = num_integration_points
    def forward(self, param_predictions, observations):
        """
        Calculate CRPS loss for a batch of predictions
        
        Args:
            param_predictions (dict): Dictionary with distribution parameters
                                     'mu': location parameters (batch_size, num_components)
                                     'b': scale parameters (batch_size, num_components)
                                     'tau': asymmetry parameters (batch_size, num_components)
                                     'pi': mixture weights (batch_size, num_components)
            observations (torch.Tensor): Observed values (shape: (batch_size,))
            
        Returns:
            torch.Tensor: CRPS loss (shape: (batch_size,))
        """
        observations = observations['y']
        batch_size = observations.shape[0]
        
        # Extract parameters for asymmetric Laplacian mixture
        m = param_predictions['mu']  # (batch_size, num_components)
        b = param_predictions['b']   # (batch_size, num_components)
        t = param_predictions['tau'] # (batch_size, num_components)
        pi = param_predictions['pi'] # (batch_size, num_components)
        
        # Ensure pi sums to 1 along component dimension
        pi = pi / torch.sum(pi, dim=1, keepdim=True)
        
        # Calculate proper mean and standard deviation for each component
        # Mean = mu + b * (1 - 2*tau) / (tau * (1 - tau))
        component_means = m + b * (1 - 2*t) / (t * (1 - t))
        
        # Variance = b^2 * (1 - 2*tau + 2*tau^2) / (tau^2 * (1 - tau)^2)
        component_vars = b**2 * (1 - 2*t + 2*t**2) / (t**2 * (1 - t)**2)
        component_stds = torch.sqrt(component_vars)
        
        # Calculate mixture mean (weighted average of component means)
        mixture_mean = torch.sum(pi * component_means, dim=1)  # (batch_size,)
        
        # Calculate mixture variance: E[(X-μ)²] = Σπ_i[(μ_i-μ)² + σ_i²]
        mean_variance = torch.sum(pi * (component_means - mixture_mean.unsqueeze(1))**2, dim=1)
        avg_variance = torch.sum(pi * component_vars, dim=1)
        mixture_var = mean_variance + avg_variance
        mixture_std = torch.sqrt(mixture_var)
        
        mean_avg = torch.mean(mixture_mean)
        std_avg = torch.mean(mixture_std)
        
        # Use mean±5*std 
        lower_bound = (mean_avg - 4 * std_avg).item()
        upper_bound = (mean_avg + 6 * std_avg).item()
        
        # Create integration points
        integration_points = torch.linspace(lower_bound, upper_bound, self.num_integration_points, device=observations.device)
        integration_points = integration_points.repeat(batch_size, 1)  # (batch_size, num_integration_points)
        
        # Compute CDF at integration points
        cdf_values = self._compute_mixture_cdf(integration_points, param_predictions)
        
        # Calculate heaviside step function (1 if integration_point >= observation, 0 otherwise)
        observations_expanded = observations.unsqueeze(1).expand_as(integration_points)
        heaviside = (integration_points >= observations_expanded).float()
        
        # Calculate squared difference
        squared_diff = (cdf_values - heaviside) ** 2
        
        # Numerical integration using trapezoidal rule
        dx = (upper_bound - lower_bound) / (self.num_integration_points - 1)
        integrated_values = 0.5 * dx * (squared_diff[:, :-1] + squared_diff[:, 1:])
        crps = torch.sum(integrated_values, dim=1)  # Sum over integration points
        crps = torch.sum(crps)
        return crps
    
    def _compute_mixture_cdf(self, integration_points, param_predictions):
        """
        Compute CDF of a mixture of asymmetric Laplacian distributions
        
        Args:
            x (torch.Tensor): Points to evaluate CDF at (shape: (batch_size, num_points))
            params (dict): Dictionary with distribution parameters
        
        Returns:
            torch.Tensor: CDF values (shape: (batch_size, num_points))
        """
        # Get parameters
        m = param_predictions['mu']  # (batch_size, num_components)
        b = param_predictions['b']   # (batch_size, num_components)
        t = param_predictions['tau'] # (batch_size, num_components)
        pi = param_predictions['pi'] # (batch_size, num_components)
        
        # Ensure pi sums to 1 along component dimension
        pi = pi / torch.sum(pi, dim=1, keepdim=True)
        
        # Expand dimensions for broadcasting
        # y: (batch_size, num_points, 1)
        # m: (batch_size, 1, num_components)
        integration_points_expanded = integration_points.unsqueeze(-1)  
        m_expanded = m.unsqueeze(1)
        b_expanded = b.unsqueeze(1)
        t_expanded = t.unsqueeze(1)
        pi_expanded = pi.unsqueeze(1)
        
        # Calculate normalized error
        error = integration_points_expanded - m_expanded  # (batch_size, num_points, num_components)
        
        # Calculate CDF value based on error sign
        greater_mask = error > 0
        less_mask = ~greater_mask
        
        # Initialize CDF tensor
        component_cdfs = torch.zeros_like(error)
        
        # Handle positive errors
        if torch.any(less_mask):
            component_cdfs[less_mask] = t_expanded.expand_as(error)[less_mask] * torch.exp(
                (1 - t_expanded.expand_as(error)[less_mask]) * error[less_mask] / b_expanded.expand_as(error)[less_mask]
            )
        
        # Handle negative errors
        if torch.any(greater_mask):
            component_cdfs[greater_mask] = 1 - (1 - t_expanded.expand_as(error)[greater_mask]) * torch.exp(
                (-t_expanded.expand_as(error)[greater_mask]) * error[greater_mask] / b_expanded.expand_as(error)[greater_mask]
            )
        
        # Apply mixture weights and sum over components
        cdf_values = torch.sum(pi_expanded.expand_as(component_cdfs) * component_cdfs, dim=2)
        
        return cdf_values


class CMALLoss(nn.Module):
    # Negative log-likelihood for a single asymmetric Laplacian distribution
    # Taken from Neural Hydrology, uses Alternative parameterization for Bayesian quantile regression (wikipedia)
    def __init__(self, eps=1e-8):
        super(CMALLoss, self).__init__()
        self.eps = eps  # Small constant for numerical stability

    def forward(self, prediction: dict[str, torch.Tensor], ground_truth: dict[str, torch.Tensor]):
        mask = ~torch.isnan(ground_truth['y'])
        
        # Apply mask to all relevant tensors
        y = ground_truth['y'][mask]
        m = prediction['mu'][mask]
        b = prediction['b'][mask]
        t = prediction['tau'][mask]
        p = prediction['pi'][mask]    
        error = y.unsqueeze(-1) - m
        log_like = (torch.log(t  + self.eps) + 
                    torch.log(1.0 - t  + self.eps) - 
                    torch.log(b  + self.eps) - 
                    torch.max(t * error, (t - 1.0) * error) / (b + self.eps))
        
        log_weights = torch.log(p + self.eps)  # Prevent log(0) issues
    
        result = torch.logsumexp(log_weights + log_like, dim=-1)  # Use dim=-1 for flexibility
        
        result = -torch.mean(result, dim=-1)
        return result


class KGELoss(nn.Module):
    """
    Kling-Gupta Efficiency loss for batched predictions with multiple timesteps.
    KGE = 1 - sqrt((r - 1)^2 + (alpha - 1)^2 + (beta - 1)^2)
    where:
    r = correlation coefficient
    alpha = std(sim)/std(obs)
    beta = mean(sim)/mean(obs)
    
    Handles inputs of shape [batch_size, timesteps]
    """
    def __init__(self, reduction='mean'):
        super(KGELoss, self).__init__()
        self.reduction = reduction
        
    def forward(self, simulated, observed):
        """
        Args:
            simulated (torch.Tensor): Simulated values of shape [batch_size, timesteps]
            observed (torch.Tensor): Observed values of shape [batch_size, timesteps]
        Returns:
            torch.Tensor: KGE loss
        """
        # Ensure the tensors are floating point
        sim = simulated.float()
        obs = observed.float()
        
        # Handle potential NaN values
        eps = 1e-12
        
        # Calculate means along timestep dimension
        sim_mean = torch.mean(sim, dim=1)  # Shape: [batch_size]
        obs_mean = torch.mean(obs, dim=1)  # Shape: [batch_size]
        
        # Calculate standard deviations along timestep dimension
        sim_std = torch.std(sim, dim=1)  # Shape: [batch_size]
        obs_std = torch.std(obs, dim=1)  # Shape: [batch_size]
        
        # Calculate correlation coefficient for each sample in the batch
        sim_norm = sim - sim_mean.unsqueeze(1)  # Shape: [batch_size, timesteps]
        obs_norm = obs - obs_mean.unsqueeze(1)  # Shape: [batch_size, timesteps]
        
        # Calculate numerator and denominator for correlation
        numerator = torch.sum(sim_norm * obs_norm, dim=1)  # Shape: [batch_size]
        sim_norm_sum = torch.sum(sim_norm ** 2, dim=1)  # Shape: [batch_size]
        obs_norm_sum = torch.sum(obs_norm ** 2, dim=1)  # Shape: [batch_size]
        
        # Add epsilon for numerical stability
        denominator = torch.sqrt(sim_norm_sum * obs_norm_sum + eps)
        r = numerator / denominator  # Shape: [batch_size]
        
        # Calculate alpha (ratio of standard deviations)
        alpha = sim_std / (obs_std + eps)  # Shape: [batch_size]
        
        # Calculate beta (ratio of means)
        # Handle the case where observed mean is zero or very small
        beta = sim_mean / (obs_mean + eps)
        # Calculate KGE for each sample in the batch
        kge = 1 - torch.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)  # Shape: [batch_size]
        
        # Convert to loss (1 - KGE, since KGE's optimal value is 1)
        loss = 1 - kge  # Shape: [batch_size]
        
        if self.reduction == 'mean':
            return torch.mean(loss)
        elif self.reduction == 'sum':
            return torch.sum(loss)
        else:  # 'none'
            return loss

  
class NSELoss(nn.Module):
    """
    Nash-Sutcliffe Efficiency loss.
    """
    def __init__(self, reduction='mean'):
        super(NSELoss, self).__init__()
        self.reduction = reduction

    def forward(self, simulated, observed):
        """
        Args:
            simulated (torch.Tensor): Simulated values
            observed (torch.Tensor): Observed values
        Returns:
            torch.Tensor: NSE loss
        """
        # Ensure the tensors are floating point
        sim = simulated.float()
        obs = observed.float()

        # Calculate mean of observed values
        obs_mean = torch.mean(obs, dim=-1, keepdim=True)

        # Calculate numerator (mean squared error)
        numerator = torch.sum((sim - obs) ** 2, dim=-1)

        # Calculate denominator (variance of observed values)
        denominator = torch.sum((obs - obs_mean) ** 2, dim=-1)

        # Calculate NSE
        nse = 1 - numerator / (denominator + 1e-6)

        # Convert to loss (1 - NSE, since NSE's optimal value is 1)
        loss = 1 - nse

        if self.reduction == 'mean':
            return torch.mean(loss)
        elif self.reduction == 'sum':
            return torch.sum(loss)
        else:  # 'none'
            return loss

