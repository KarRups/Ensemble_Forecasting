"""
FILE CONTENTS
================================================================================

CLASSES:
--------------------------------------------------------------------------------
 1. LSTMModel
    No description available
    Methods: __init__, forward

 2. Hindcast_LSTM_Block
    No description available
    Methods: __init__, forward
 3. Forecast_LSTM_Block
    No description available
    Methods: __init__, forward
 4. Google_LSTMModel
    No description available
    Methods: __init__, forward
 5. NN
    No description available
    Methods: __init__, forward
 6. EarlyStopper
    No description available
    Methods: __init__, early_stop

FUNCTIONS:
--------------------------------------------------------------------------------
 1. Google_Model_Block
    No description available
3. define_models
    No description available
================================================================================
"""

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


class Hindcast_LSTM_Block(nn.Module):
    # This block serves to take in historic data and output the initial memory and hidden 
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout = 0.0, bidirectional = False, eval_dropout = True):
        super(Hindcast_LSTM_Block, self).__init__()
        self.bidirectional = bidirectional  # Store bidirectional as an instance variable
        self.No_Directions = 1 if not bidirectional else 2
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.dropout = dropout
        self.eval_dropout = eval_dropout
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout= dropout, bidirectional = bidirectional)
        # self.dropout = nn.Dropout(p=dropout)
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
            
        out, (hn, cn) = self.lstm(x, (h0.contiguous(), c0.contiguous())) 
        # out = self.dropout(out)
        out = F.dropout(out, p= self.dropout, training= self.eval_dropout)
        out = self.fc(out)  # Take the output from the last time step
        return out, hn, cn


class Forecast_LSTM_Block(nn.Module):
    # This block serves to take in historic data and output the initial memory and hidden 
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout = 0.0, bidirectional = False, eval_dropout = True):
        super(Forecast_LSTM_Block, self).__init__()
        self.bidirectional = bidirectional  # Store bidirectional as an instance variable
        self.No_Directions = 1 if not bidirectional else 2

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.dropout = dropout
        self.eval_dropout = eval_dropout

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout= dropout, bidirectional = bidirectional)
        # self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(hidden_size * self.No_Directions, output_size) # If bidirectional is true need the *2

    def forward(self, x, h0, c0):
        # Map H0_sequences and H0_static to the appropriate sizes
        # Is this implementation of history doing anything
   
        out, _ = self.lstm(x, (h0.contiguous(), c0.contiguous())) 
        # out = self.dropout(out)
        out = F.dropout(out, p= self.dropout, training= self.eval_dropout)
        out = self.fc(out)  # Take the output from the last time step
        return out


class NN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.output_size = output_size  # Store output_size for slicing


    def forward(self, x):
        # x has shape [num_layers, batch_size, input_size]
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)

        # Split along the last dimension
        split_size = self.output_size // 2  # Ensure even split
        hn_new, cn_new = x[..., :split_size], x[..., split_size:]

        return hn_new, cn_new
        
class Google_LSTMModel(nn.Module):
  def __init__(self, hindcast, forecast, handover):
    super(Google_LSTMModel, self).__init__()
    self.hindcast = hindcast
    self.forecast = forecast
    self.handover = handover
      
  def forward(self, history, forecasts):
    
    # get states from hindcast model
    # need to decide whether the head recieves the raw history or an encoding of it
    hind_out, hn,cn = self.hindcast(history)
    # add a neural network to convert one states fro history to an initialisation for forecasts
    hn_new, cn_new = self.handover(torch.cat((hn, cn), dim=-1))
    # get forecasts from forecast model

    # Ensure contiguity and correct shape for LSTM input
    hn_new = hn_new.contiguous()
    cn_new = cn_new.contiguous()
      
    out = self.forecast(forecasts, hn_new,cn_new)
    return hind_out , out

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



def Google_Model_Block(hindcast_input_size, forecast_input_size, hindcast_output_size, forecast_output_size, hindcast_hidden_size, forecast_hidden_size, handoff_hidden_size, num_layers, device, dropout = 0.0, bidirectional = False, eval_dropout = False):
    Hindcast = Hindcast_LSTM_Block(hindcast_input_size, hindcast_hidden_size, num_layers, hindcast_output_size, dropout = dropout, bidirectional = bidirectional, eval_dropout = eval_dropout)
    Forecast = Forecast_LSTM_Block(forecast_input_size, forecast_hidden_size, num_layers, forecast_output_size, dropout = dropout, bidirectional = bidirectional, eval_dropout = eval_dropout)

    size_multiplier = 2

    Handover = NN(input_size= hindcast_hidden_size*2, hidden_size= handoff_hidden_size, output_size = forecast_hidden_size*size_multiplier)
    Block = Google_LSTMModel(Hindcast, Forecast, Handover)
    Block.to(device)

    return Block

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



