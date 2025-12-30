import numpy as np
# import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
import pickle 

import random
import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler
from collections import defaultdict
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn as nn


import os
import itertools
import joblib
from datetime import datetime


import sys
from pathlib import Path
functions_path = Path("/home/mokr/Loss_Functions_Paper/ML_Functions/")
sys.path.append(str(functions_path))

import ML_functions
import ML_Losses
import ML_Models
from ML_functions import HydroDataset


def load_dataloaders(batch_size = 128, history_sequence_length = 90, forecast_sequence_length = 10, ML_functions = ML_functions):
    batch_size = 128
    
    loaded_traning_data0 = torch.load("/perm/mokr/10Day_Loss_Function_Training_Dataset0_test.pt")
    Training_Dataset0 = HydroDataset(loaded_traning_data0)
    
    loaded_traning_data1 = torch.load("/perm/mokr/10Day_Loss_Function_Training_Dataset1_test.pt")
    Training_Dataset1 = HydroDataset(loaded_traning_data1)
    
    loaded_traning_data2 = torch.load("/perm/mokr/10Day_Loss_Function_Training_Dataset2_test.pt")
    Training_Dataset2 = HydroDataset(loaded_traning_data2)

    loaded_traning_data3 = torch.load("/perm/mokr/10Day_Loss_Function_Training_Dataset3_test.pt")
    Training_Dataset3 = HydroDataset(loaded_traning_data3)
    
    loaded_traning_data4 = torch.load("/perm/mokr/10Day_Loss_Function_Training_Dataset4_test.pt")
    Training_Dataset4 = HydroDataset(loaded_traning_data4)

    combined_data = []
    # datasets = [Training_Dataset4]
    datasets = [Training_Dataset0, Training_Dataset1, Training_Dataset2, Training_Dataset3 , Training_Dataset4]

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


def run_hyperparameter_test():
    Training_Dataloader, Validation_Dataloader = load_dataloaders(batch_size = 128, history_sequence_length = 90, forecast_sequence_length = 10, ML_functions = ML_functions)

    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Define hyperparameter grid
    lr = 1e-3  #[1e-2, 1e-3, 1e-4]
    bidirectional = False #[False, True]
    hidden_sizes = [
        (256, 256, 256)
        
    ]
    
    
    # Fixed parameters
    hindcast_input_size = 25
    forecast_input_size = 17
    hindcast_output_size = 8
    forecast_output_size = 12
    
    batch_accumulation = 6000
    gammas = [0.97]
    
    # Create directory for models if it doesn't exist
    model_dir = '/home/mokr/Loss_Functions_Paper/Models/Binary/hyperparameter_test'
    os.makedirs(model_dir, exist_ok=True)
    
    # DataFrame to store results
    results = []
    
    # Timestamp for this test run
    timestamp = datetime.now().strftime("%m%d")
    
    # Loop through hyperparameter combinations
    num_epochs = [10]
    for epochs in num_epochs:
        for hidden_size in hidden_sizes:
            for gamma in gammas:
            
                
                hindcast_hidden_size, forecast_hidden_size, handoff_hidden_size = hidden_size
        
                # Model configuration description
                config_desc = f"lr_{lr}_bi_{bidirectional}_h_{hindcast_hidden_size}_{forecast_hidden_size}_{handoff_hidden_size}"
                print(f"\n{'='*80}\nTraining model with: {config_desc}\n{'='*80}")
                sys.stdout.flush()
                
                # Initialize model
                model = ML_Models.Google_Model_Block(
                    hindcast_input_size, forecast_input_size,
                    hindcast_output_size, forecast_output_size,
                    hindcast_hidden_size, forecast_hidden_size, handoff_hidden_size,
                    1, device, bidirectional=bidirectional
                )


                # Setup optimizer and criterion
                optimizer = optim.Adam(model.parameters(), lr=lr)

                scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma)
                
                # criterion = ML_Losses.CMALLoss()
                criterion = ML_Losses.CRPSLoss()
                # Train model
                print('Starting training...')
                sys.stdout.flush()
                
                train_loss = ML_functions.train_model_CMAL(
                    model, Training_Dataloader, optimizer, criterion,
                    epochs=30, train_mode=True,
                    batch_accumulation=batch_accumulation, device=device, scheduler = scheduler, use_observations = True
                )

                print('Training complete.')
                sys.stdout.flush()


                # Save model
                model_path = os.path.join(model_dir, f"CRPS_Binary_{config_desc}_{timestamp}_10Epochs.pth")
                torch.save(model, model_path)
                print(f"Model saved to {model_path}")
                
                val_metrics = ML_functions.train_model_CMAL(
                    model, Validation_Dataloader, optimizer, criterion,
                    epochs= 1, train_mode=False,
                    batch_accumulation= batch_accumulation, device=device,
                )

                # Epochs 11 - 25                    
                train_loss = ML_functions.train_model_CMAL(
                    model, Training_Dataloader, optimizer, criterion,
                    epochs=30, train_mode=True,
                    batch_accumulation=batch_accumulation, device=device, scheduler = scheduler, use_observations = True
                )

                model_path = os.path.join(model_dir, f"CRPS_Binary_{config_desc}_{timestamp}_30Epochs.pth")
                torch.save(model, model_path)
                print(f"Model saved to {model_path}")
                
                val_metrics = ML_functions.train_model_CMAL(
                    model, Validation_Dataloader, optimizer, criterion,
                    epochs= 1, train_mode=False,
                    batch_accumulation= batch_accumulation, device=device, use_observations = True
                )

                # Epochs 26 - 50                    
                train_loss = ML_functions.train_model_CMAL(
                    model, Training_Dataloader, optimizer, criterion,
                    epochs= 40, train_mode=True,
                    batch_accumulation=batch_accumulation, device=device, scheduler = scheduler, use_observations = True
                )

                model_path = os.path.join(model_dir, f"CRPS_Binary_{config_desc}_{timestamp}_50Epochs.pth")
                torch.save(model, model_path)
                print(f"Model saved to {model_path}")
                
                val_metrics = ML_functions.train_model_CMAL(
                    model, Validation_Dataloader, optimizer, criterion,
                    epochs= 1, train_mode=False,
                    batch_accumulation= batch_accumulation, device=device, use_observations = True
                )
                
                
                
                metrics_path = os.path.join(model_dir, f"val_metrics_{config_desc}_{timestamp}.pkl")
                with open(metrics_path, 'wb') as f:
                    pickle.dump(val_metrics, f)
                    
                # Store results
                results.append({
                    'learning_rate': lr,
                    'num_epochs': epochs,
                    'bidirectional': bidirectional,
                    'hindcast_hidden_size': hindcast_hidden_size,
                    'forecast_hidden_size': forecast_hidden_size,
                    'handoff_hidden_size': handoff_hidden_size,
                    'final_train_loss': train_loss[-1] if train_loss else None,
                    'final_val_loss': val_metrics[-1] if val_metrics else None,
                    'model_path': model_path
                })
            
            # Save results to CSV
            results_df = pd.DataFrame(results)
            results_path = os.path.join(model_dir, f"CRPS_hyperparameter_results_{timestamp}.csv")
            results_df.to_csv(results_path, index=False)
            print(f"Results saved to {results_path}")
            
            # Find and print best model
            if results_df['final_val_loss'].notna().any():
                best_idx = results_df['final_val_loss'].idxmin()
                best_model = results_df.iloc[best_idx]
                print("\n" + "="*80)
                print("Best model configuration:")
                print(f"Learning rate: {best_model['learning_rate']}")
                print(f"Bidirectional: {best_model['bidirectional']}")
                print(f"Hidden sizes: ({best_model['hindcast_hidden_size']}, {best_model['forecast_hidden_size']}, {best_model['handoff_hidden_size']})")
                print(f"Validation loss: {best_model['final_val_loss']}")
                print(f"Model path: {best_model['model_path']}")
                print("="*80)
    
                sys.stdout.flush()
if __name__ == "__main__":
    run_hyperparameter_test()
