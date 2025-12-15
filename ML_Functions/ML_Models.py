import random
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler
from torch.cuda.amp import autocast, GradScaler


from collections import defaultdict


import sys
sys.path.append('/data/Hydra_Work/Competition_Functions') 
from Processing_Functions import process_forecast_date, process_seasonal_forecasts, fit_fourier_to_h0, Get_History_Statistics


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


def Specific_Heads(basins, hindcast_input_size, forecast_input_size, hindcast_output_size, forecast_output_size, hidden_size, num_layers, device, dropout = 0.0, bidirectional = False):
    model_heads = {}
    for basin in basins:
        basin_hindcast = Hindcast_LSTM_Block(hindcast_input_size, hidden_size, num_layers, hindcast_output_size, dropout = dropout, bidirectional = bidirectional)
        #basin_forecast = Forecast_LSTM_Block(forecast_input_size, hidden_size, num_layers, forecast_output_size, dropout = dropout, bidirectional = bidirectional)
    
        model_heads[f'{basin}'] = Google_LSTMModel(basin_hindcast)
        model_heads[f'{basin}'].to(device)
    return model_heads



    
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout = 0.0, bidirectional = False, Sequence_Target = False):
        self.bidirectional = bidirectional  # Store bidirectional as an instance variable
        self.No_Directions = 1 if not bidirectional else 2
        self.Sequence_Target = Sequence_Target

        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout= dropout, bidirectional = bidirectional)
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(hidden_size * self.No_Directions, output_size) # If bidirectional is true need the *2


    def forward(self, x):
        
        h0 = torch.zeros(self.num_layers * self.No_Directions, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers * self.No_Directions, self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))

        out = self.dropout(out)
        if not self.Sequence_Target:
            out = out[-1,:]  # Take the output from the last time step
        out = self.fc(out) 
        return out

class SumPinballLoss(nn.Module):
    def __init__(self, quantiles = [0.1,0.5,0.9]):
        super(SumPinballLoss, self).__init__()
        self.quantiles = quantiles

    def forward(self, observed, modeled):
        # Initialize a list to store losses for each output
        output_losses = []
        observed = observed.squeeze()
        # modeled = torch.sum(modeled, dim = 1)

 
        # Calculate the quantile loss for each output and quantile
        for i, quantile in enumerate(self.quantiles):
            modeled_quantile = modeled[...,i]
            errors = torch.max(quantile * (observed - modeled_quantile), (quantile - 1) * (observed - modeled_quantile))
            loss = torch.mean(errors)
        
            output_losses.append(loss)

        # Sum the losses for each output and quantile
        overall_loss = sum(output_losses) 
        
        return overall_loss
    
class ChangingPinballLoss(nn.Module): # 4 differnt trials to run to show the effect of loss function to seasnoal flow prediction when using LSTMs/Transformers
    def __init__(self, quantiles = [0.1,0.5,0.9]):
        super(SumPinballLoss, self).__init__()
        self.quantiles = quantiles

    def forward(self, observed, modeled, multiplier, constant = 1):
        # Initialize a list to store losses for each output
        output_losses = []
        observed = observed.squeeze()

        #modeled = torch.sum(modeled, dim = 1)
 
        # Calculate the quantile loss for each output and quantile
        for i, quantile in enumerate(self.quantiles):

            modeled_quantile = modeled[...,i]
            loss = torch.nanmean(torch.max(quantile * torch.nansum(observed - modeled_quantile), (quantile - 1) * torch.nansum(observed - modeled_quantile) ))

            length = loss.size(0)
            if multiplier == 'Linear':
                multipliers = torch.arange(length) * constant
            elif multiplier == 'exponential':
                multipliers = torch.exp(torch.arange(length) * constant)
            elif multiplier == "quadratic":
                multipliers = torch.clamp(constant[1]*(torch.arange(length) - constant[0]) ** 2, min=0)

        
            loss = loss*multipliers
            output_losses.append(loss)
            
        # Sum the losses for each output and quantile
        overall_loss = sum(output_losses)
        
        return overall_loss

class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


# Do we want hindcast and forecast num-layers to be different?
def define_models(hindcast_input_size, forecast_input_size, hidden_size, num_layers, dropout, bidirectional, learning_rate, copies = 3, forecast_output_size = 3, device = 'cpu'):
    models = {}
    params_to_optimize = {}
    optimizers = {}
    schedulers = {}
    
    hindcast_output_size = forecast_output_size
    for copy in range(copies):
        models[copy] = Google_Model_Block(hindcast_input_size, forecast_input_size, hindcast_output_size, forecast_output_size, hidden_size, num_layers, device, dropout, bidirectional)
        
        models[copy].to(device)
        params_to_optimize[copy] = list(models[copy].parameters())
        # Probably should be doing 1e-2 and 10
        optimizers[copy] = torch.optim.Adam(params_to_optimize[copy], lr= learning_rate, weight_decay = 1e-4)
        schedulers[copy] = lr_scheduler.StepLR(optimizers[copy], step_size= 5000, gamma=0.95) #.CosineAnnealingLR(optimizers[copy], T_max = 100000,)

        

    return models, params_to_optimize, optimizers, schedulers

# Model Running Code
# Key difference here is that we won't need to restrict the history length to 90, it can be variable


