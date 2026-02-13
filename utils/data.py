from typing import Optional
import numpy as np
import pandas as pd
import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

def add_time_dimension(data: torch.Tensor, t: Optional[torch.Tensor] = None):
    """
    Adds a time dimension to the input tensor by concatenating repeated time steps.

    Args:
        data (torch.Tensor): Input tensor of shape (N, T, D).
        t (Optional[torch.Tensor]): 1D tensor or numpy array of time steps of shape (T,). If None, uses linspace from 0 to 1.

    Returns:
        torch.Tensor: Tensor of shape (N, T, D+1) with time as the first feature.
    """
    N, T, _ = data.shape
    if t is None:
        time_steps = np.linspace(0, 1, T)
    else:
        time_steps = t.cpu().numpy() if isinstance(t, torch.Tensor) else t
    time_steps_repeated = np.tile(time_steps, (N, 1)).reshape(N, T, 1)
    data_with_time = torch.cat([torch.tensor(time_steps_repeated, dtype=data.dtype, device=data.device), data], dim=-1)
    return data_with_time

def batch_lead_lag_transform(data: torch.Tensor, t:torch.Tensor, lead_lag: int|list[int]=1):
    '''
    Transform data to lead-lag format
    data is of shape (seq_len, seq_dim)
    '''
    assert data.ndim == 3 and t.ndim == 3, 'data and t must be of shape (batch_size, seq_len, seq_dim)'
    assert data.shape[1] == t.shape[1], 'data and df_index must have the same length'
    if isinstance(lead_lag, int):
        if lead_lag <= 0: raise ValueError('lead_lag must be a positive integer')
    else:
        for lag in lead_lag:
            if lag <= 0: raise ValueError('lead_lag must be a positive integer')

    # get shape of output
    batch_size = data.shape[0]
    seq_len = data.shape[1]
    seq_dim = data.shape[2]
    shape = list(data.shape)
    if isinstance(lead_lag, int):
        lead_lag = [lead_lag]
    max_lag = max(lead_lag)
    shape[1] = shape[1] + max_lag
    shape[2] = (len(lead_lag) + 1) * seq_dim

    # create time dimension t.shape = (batch_size, seq_len, 1)
    # pad latter values with last value, shape (seq_len + max_lag, 1)
    t = torch.cat([t, (torch.ones(batch_size, max_lag, 1, dtype=t.dtype, device=t.device, requires_grad=False) * t[:,-1:,:])], dim=1)

    # create lead-lag series
    lead_lag_data = torch.empty(shape, dtype=data.dtype, device=t.device, requires_grad=False) # shape (seq_len + max_lag, seq_dim * (len(lead_lag) + 1))
    lead_lag_data[:, :seq_len, :seq_dim] = data # fill in original sequence
    lead_lag_data[:, seq_len:, :seq_dim] = data[:,-1:,:] # pad latter values with last value
    for i, lag in enumerate(lead_lag):
        i = i + 1 # skip first seq_dim columns
        lead_lag_data[:, :lag, i*seq_dim:(i+1)*seq_dim] = 0.0 # pad initial values with zeros
        lead_lag_data[:, lag:lag+seq_len, i*seq_dim:(i+1)*seq_dim] = data
        lead_lag_data[:, lag+seq_len-1:, i*seq_dim:(i+1)*seq_dim] = data[:,-1:,:] # pad latter values with last value
    return torch.cat([t, lead_lag_data], axis=2)