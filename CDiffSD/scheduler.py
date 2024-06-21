import pandas as pd
import numpy as np
import random
import torch
from tqdm.auto import tqdm
import torch.nn.functional as F
import utils.utils_diff as u

# Set the device to GPU if available, otherwise use CPU
device = torch.device(f"cuda:1" if torch.cuda.is_available() else "cpu")

def get_index_from_list(vals, t, x_shape):
    """ 
    Returns a specific index t of a passed list of values vals
    while considering the batch dimension.
    
    Args:
        vals (torch.Tensor): The list of values.
        t (torch.Tensor): The indices to gather.
        x_shape (tuple): The shape of the input tensor.
        
    Returns:
        torch.Tensor: The gathered values reshaped to match the batch dimension.
    """
    batch_size = t.shape[0]
    out = vals.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

def compute_beta_schedule(args):
    """
    Computes the beta schedule based on the specified scheduler type.
    
    Args:
        args (argparse.Namespace): The arguments containing scheduler type and parameters.
        
    Returns:
        tuple: sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod
    """
    def linear_beta_schedule(timesteps):
        """Creates a linear beta schedule."""
        scale = 1000 / timesteps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float64)

    def cosine_beta_schedule(timesteps, s):
        """Creates a cosine beta schedule."""
        beta_start = 0.0001
        beta_end = 0.02
        steps = torch.arange(timesteps, dtype=torch.float64)
        beta_t = torch.cos(((steps / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
        beta_t = beta_t * (beta_end - beta_start) + beta_start  # Scale the beta values according to the specification
        return beta_t

    # Select the beta schedule based on the scheduler type
    if args.scheduler_type == 'linear':
        betas = linear_beta_schedule(args.T)
    elif args.scheduler_type == 'cosine':
        betas = cosine_beta_schedule(args.T, args.s)
    else:
        raise ValueError("Unsupported scheduler type. Choose 'linear' or 'cosine'.")

    alphas = 1. - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
    return sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod

def forward_diffusion_sample(x_start, x_end, t, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod):
    """
    Generates a forward diffusion sample by interpolating using alphas.
    
    Args:
        x_start (torch.Tensor): The initial input tensor.
        x_end (torch.Tensor): The final input tensor.
        t (torch.Tensor): The time steps.
        sqrt_alphas_cumprod (torch.Tensor): The cumulative product of alphas.
        sqrt_one_minus_alphas_cumprod (torch.Tensor): The square root of one minus the cumulative product of alphas.
        
    Returns:
        torch.Tensor: The interpolated tensor.
    """
    return (get_index_from_list(sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            get_index_from_list(sqrt_one_minus_alphas_cumprod, t, x_start.shape) * x_end)
