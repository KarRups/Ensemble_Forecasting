import pickle 

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
import pandas as pd

import sys
from pathlib import Path
functions_path = Path("/home/mokr/Loss_Functions_Paper/ML_Functions/")
sys.path.append(str(functions_path))

import ML_functions
import ML_Models
import ML_Losses
from ML_functions import HydroDataset, CRPSLoss_Ensemble, train_seeded_model, Discrete_CRPSLoss #, load_dataloaders


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
    learning_rates = [1e-3] 
    bidirectional = False
    num_layers = [1,2]
    hidden_sizes = [      
        (256, 256, 128),
    ]
    num_ensemble_members_list = [16] 
    

    gammas = [0.3]
    fixed_noises = [True, False]
    
    # Fixed parameters
    hindcast_input_size = 26
    forecast_input_size = 16
    hindcast_output_size = 8
    forecast_output_size = 1
    
    batch_accumulation = 6000
    
    # Create directory for models if it doesn't exist
    model_dir = '/home/mokr/Loss_Functions_Paper/Models/Seeded/hyperparameter_test'
    os.makedirs(model_dir, exist_ok=True)
    
    # DataFrame to store results
    results = []
    
    # Timestamp for this test run
    timestamp = datetime.now().strftime("%m%d")
    
    # Loop through hyperparameter combinations
    noise_scales = [1, 0.1]
    for noise_scale in noise_scales:
        for hidden_size in hidden_sizes:
            for num_ensemble_members in num_ensemble_members_list:
                for lr in learning_rates:   
                    for gamma in gammas:
                        for fixed_noise in fixed_noises:
                    
                            
                            hindcast_hidden_size, forecast_hidden_size, handoff_hidden_size = hidden_size
                    
                            # Model configuration description
                            config_desc = f"lr_{lr}_bi_{noise_scale}_h_{hindcast_hidden_size}_{forecast_hidden_size}_{handoff_hidden_size}"
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
        
                            criterion = Discrete_CRPSLoss()
                            # Train model
                            print('Starting training...')
                            sys.stdout.flush()
                            
                            train_seeded_model(model, optimizer, Training_Dataloader, epochs=5, batch_accumulation_size=batch_accumulation, 
                                          num_ensemble_members=num_ensemble_members, noise_scale= noise_scale,
                                          criterion = criterion, scheduler = scheduler, train_mode = True, fixed_noise = fixed_noise)
                            print('Training complete.')
                            sys.stdout.flush()
        
        
                            # Save model
                            model_path = os.path.join(model_dir, f"Seeded_{config_desc}_{timestamp}_{num_ensemble_members}_{gamma}_5_{fixed_noise}_Epochs.pth")
                            torch.save(model, model_path)
                            print(f"Model saved to {model_path}")
        
                            
                            val_metrics = train_seeded_model(model, optimizer, Validation_Dataloader, epochs=1, batch_accumulation_size=batch_accumulation, 
                                          num_ensemble_members=11, noise_scale= noise_scale,
                                          criterion= criterion, scheduler = None, train_mode = False, fixed_noise = fixed_noise)
        
        
        
                            
                            train_seeded_model(model, optimizer, Training_Dataloader, epochs=5, batch_accumulation_size=batch_accumulation, 
                                          num_ensemble_members=num_ensemble_members, noise_scale=noise_scale,
                                          criterion= criterion, scheduler = scheduler, train_mode = True, fixed_noise = fixed_noise)
                            print('Training complete.')
                            sys.stdout.flush()
        
        
                            # Save model
                            model_path = os.path.join(model_dir, f"Seeded_{config_desc}_{timestamp}_{num_ensemble_members}_{gamma}_10_{fixed_noise}_Epochs.pth")
                            torch.save(model, model_path)
                            print(f"Model saved to {model_path}")
        
                            
                            val_metrics = train_seeded_model(model, optimizer, Validation_Dataloader, epochs=1, batch_accumulation_size=batch_accumulation, 
                                          num_ensemble_members= 11, noise_scale=noise_scale,
                                          criterion=None, scheduler = None, train_mode = False, fixed_noise = fixed_noise)
        
        
                            # train_seeded_model(model, optimizer, Training_Dataloader, epochs=15, batch_accumulation_size=batch_accumulation, 
                            #               num_ensemble_members= num_ensemble_members, noise_scale=noise_scale,
                            #               criterion= criterion, scheduler = scheduler, train_mode = True, fixed_noise = fixed_noise)
                            
                            # print('Training complete.')
                            # sys.stdout.flush()
        
        
                            # # Save model
                            # model_path = os.path.join(model_dir, f"Seeded_{config_desc}_{timestamp}_{num_ensemble_members}__{gamma}_25_{fixed_noise}_Epochs.pth")
                            # torch.save(model, model_path)
                            # print(f"Model saved to {model_path}")
        
                            
                            # val_metrics = train_seeded_model(model, optimizer, Validation_Dataloader, epochs=1, batch_accumulation_size=batch_accumulation, 
                            #               num_ensemble_members= 11, noise_scale=noise_scale,
                            #               criterion=None, scheduler = None, train_mode = False, fixed_noise = fixed_noise)
                            
                            
                            metrics_path = os.path.join(model_dir, f"val_metrics_{config_desc}_{timestamp}.pkl")
                            with open(metrics_path, 'wb') as f:
                                pickle.dump(val_metrics, f)
                                
                            # Store results
                            results.append({
                                'learning_rate': lr,
                                'bidirectional': bidirectional,
                                'hindcast_hidden_size': hindcast_hidden_size,
                                'forecast_hidden_size': forecast_hidden_size,
                                'handoff_hidden_size': handoff_hidden_size,
                                # 'final_train_loss': train_loss[-1] if train_loss else None,
                                # 'final_val_loss': val_metrics[-1] if val_metrics else None,
                                'model_path': model_path
                            })
                
                # Save results to CSV
                results_df = pd.DataFrame(results)
                results_path = os.path.join(model_dir, f"Seeded_hyperparameter_results_{timestamp}.csv")
                results_df.to_csv(results_path, index=False)
                print(f"Results saved to {results_path}")
 
if __name__ == "__main__":
    run_hyperparameter_test()


