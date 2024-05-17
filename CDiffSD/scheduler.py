import pandas as pd
import numpy as np
import random
import torch
from tqdm.auto import tqdm
import torch.nn.functional as F
import utils.utils_diff as u

device = torch.device(f"cuda:1" if torch.cuda.is_available() else "cpu")

def get_index_from_list(vals, t, x_shape):
    """ 
    Returns a specific index t of a passed list of values vals
    while considering the batch dimension.
    """
    batch_size = t.shape[0]
    out = vals.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)


def compute_beta_schedule(args):
    def linear_beta_schedule(timesteps):
        scale = 1000 / timesteps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float64)

    def cosine_beta_schedule(timesteps, s):
        beta_start = 0.0001
        beta_end = 0.02
        steps = torch.arange(timesteps, dtype=torch.float64)
        beta_t = torch.cos(((steps / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
        beta_t = beta_t * (beta_end - beta_start) + beta_start  # Scala i valori di beta secondo la specifica
        return beta_t

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

    # simply use the alphas to interpolate
    return (get_index_from_list(sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            get_index_from_list(sqrt_one_minus_alphas_cumprod, t, x_start.shape) * (x_end))