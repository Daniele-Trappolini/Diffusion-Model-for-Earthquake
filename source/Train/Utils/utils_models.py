import numpy as np 
import torch
from torch import nn
import math
import scipy 


def positionalEncoding(x, batch_size, device, d_model=32):
    max_sequence_length=x.shape[2]
    x_in=x.clone()
    even_i = torch.arange(0, d_model, 2).float()
    denominator = torch.pow(10000, even_i/d_model)
    position = torch.arange(max_sequence_length).reshape(max_sequence_length, 1)
    even_PE = torch.sin(position / denominator)
    odd_PE = torch.cos(position / denominator)
    stacked = torch.stack([even_PE, odd_PE], dim=2)
    pe = torch.flatten(stacked, start_dim=1, end_dim=2).repeat(batch_size,1,1)
    x_in=x_in.repeat(1,d_model,1)+pe.permute(0,2,1).to(device)
    return x_in



class BlockPosEnc(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim, up=False):
        super().__init__()
        self.time_mlp =  nn.Linear(time_emb_dim, out_ch)
        if up:
            self.conv1 = nn.Conv1d(2*in_ch, out_ch, 3, padding=1)
            self.transform = nn.ConvTranspose1d(out_ch, out_ch, 3, 2, 1, 1)
        else:
            self.conv1 = nn.Conv1d(in_ch, out_ch, 3, padding=1)
            self.transform = nn.Conv1d(out_ch, out_ch, 3, 2, 1)
        self.conv2 = nn.Conv1d(out_ch, out_ch, 3, padding=1)
        self.bnorm1 = nn.BatchNorm1d(out_ch)
        self.bnorm2 = nn.BatchNorm1d(out_ch)
        self.relu  = nn.ReLU()
        
    def forward(self, x, t):
        # First Conv
        h = self.bnorm1(self.relu(self.conv1(x)))
        # Time embedding
        time_emb = self.relu(self.time_mlp(t))
        # Extend last 2 dimensions
        time_emb = time_emb[(..., ) + (None, )]
        # Add time channel
        h = h + time_emb
        # Second Conv
        h = self.bnorm2(self.relu(self.conv2(h)))
        # Down or Upsample
        return self.transform(h)


class SinusoidalPositionEmbeddingsPosEnc(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class SimpleUnetPosEnc(nn.Module):
    """
    A simplified variant of the Unet architecture.
    """
    def __init__(self):
        super().__init__()
        earthqk_channels = 32 
        down_channels = (64, 128, 256, 512, 1024, 2048)
        up_channels = (2048,1024, 512, 256, 128, 64)
        out_dim = 2 # the model should output the eq and the real noise
        time_emb_dim = 32

        # Time embedding
        self.time_mlp = nn.Sequential(
                SinusoidalPositionEmbeddingsPosEnc(time_emb_dim),
                nn.Linear(time_emb_dim, time_emb_dim),
                nn.ReLU()
            )
        
        # Initial projection
        self.conv0 = nn.Conv1d(earthqk_channels, down_channels[0], 3, padding=1)

        # Downsample
        self.downs = nn.ModuleList([BlockPosEnc(down_channels[i], down_channels[i+1], \
                                    time_emb_dim) \
                    for i in range(len(down_channels)-1)])
        # Upsample
        self.ups = nn.ModuleList([BlockPosEnc(up_channels[i], up_channels[i+1], \
                                        time_emb_dim, up=True) \
                    for i in range(len(up_channels)-1)])

        self.output = nn.Conv1d(up_channels[-1], out_dim, 1)

    def forward(self, x, timestep):
        # Embedd time
        t = self.time_mlp(timestep)
        # Initial conv
        x=positionalEncoding(x,batch_size=x.shape[0], device=x.device)
        x = self.conv0(x)
        # Unet
        residual_inputs = []
        for down in self.downs:
            x = down(x, t)
            residual_inputs.append(x)
        for up in self.ups:
            residual_x = residual_inputs.pop()
            # Add residual x as additional channels
            x = torch.cat((x, residual_x), dim=1)
            x = up(x, t)
            
        x=self.output(x)
        return x
    


# This is for DeepDenoiser
def create_mask(noise_in,eq_in):
#   noise_mean=2
#   noise_std=1
  nfft_param= 60 
  nperseg_param=30 
  tmp_noisy_signal_all=[]
  mask_eq_all=[]
  mask_noise_all=[]
  t_signal_all=[]
  f_signal_all=[]

  for i in range(noise_in.shape[0]):
    eq=eq_in[i].flatten()
    eq -= torch.mean(eq)

    f_signal, t_signal, tmp_signal = scipy.signal.stft(eq,fs=100,nperseg=nperseg_param,nfft=nfft_param, boundary='zeros')
    
    noise=noise_in[i].flatten()
    noise -= torch.mean(noise)
  
    f_noise, t_noise, tmp_noise = scipy.signal.stft(noise, fs=100, nperseg=nperseg_param, nfft=nfft_param, boundary='zeros')
    ratio = 1 #noise_mean + np.random.randn() * noise_std
    tmp_noisy_signal = tmp_signal + ratio * tmp_noise
    noisy_signal = np.stack([tmp_noisy_signal.real, tmp_noisy_signal.imag], axis=-1)
    noisy_signal = noisy_signal #/ np.std(noisy_signal)
    tmp_mask = np.abs(tmp_signal) / (np.abs(tmp_signal) + np.abs(ratio * tmp_noise) + 1e-4)
    tmp_mask[tmp_mask >= 1] = 1
    tmp_mask[tmp_mask <= 0] = 0
    mask = np.zeros([tmp_mask.shape[0], tmp_mask.shape[1], 2])
    mask[:, :, 0] = tmp_mask
    mask[:, :, 1] = 1 - tmp_mask

    tmp_noisy_signal_all.append(noisy_signal)
    mask_eq_all.append(mask[:, :, 0])
    mask_noise_all.append(mask[:, :, 1])
    t_signal_all.append(t_signal)
    f_signal_all.append(f_signal)

  return torch.Tensor(t_signal_all), torch.Tensor(f_signal_all),torch.Tensor(tmp_noisy_signal_all).permute(0,3,1,2), torch.Tensor(mask_eq_all), torch.Tensor(mask_noise_all)






class Block_unsupervised_newUnet(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim, up=False):
        super().__init__()
        self.time_mlp =  nn.Linear(time_emb_dim, out_ch)
        if up:
            self.conv1 = nn.Conv1d(2*in_ch, out_ch, 3, padding=1)
            self.transform = nn.ConvTranspose1d(out_ch, out_ch, 3, 2, 1, 1)
        else:
            self.conv1 = nn.Conv1d(in_ch, out_ch, 3, padding=1)
            self.transform = nn.Conv1d(out_ch, out_ch, 3, 2, 1)
        self.conv2 = nn.Conv1d(out_ch, out_ch, 3, padding=1)
        self.bnorm1 = nn.BatchNorm1d(out_ch)
        self.bnorm2 = nn.BatchNorm1d(out_ch)
        self.relu  = nn.ReLU()
        
    def forward(self, x, t):
        # First Conv
        h = self.bnorm1(self.relu(self.conv1(x)))
        # Time embedding
        time_emb = self.relu(self.time_mlp(t))
        # Extend last 2 dimensions
        time_emb = time_emb[(..., ) + (None, )]
        # Add time channel
        h = h + time_emb
        # Second Conv
        h = self.bnorm2(self.relu(self.conv2(h)))
        # Down or Upsample
        return self.transform(h)


class SinusoidalPositionEmbeddings_unsupervised_newUnet(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

def positionalEncoding_unsupervised_newUnet(x, batch_size, device, d_model=32):
    max_sequence_length=x.shape[2]
    x_in=x.clone()
    even_i = torch.arange(0, d_model, 2).float()
    denominator = torch.pow(10000, even_i/d_model)
    position = torch.arange(max_sequence_length).reshape(max_sequence_length, 1)
    even_PE = torch.sin(position / denominator)
    odd_PE = torch.cos(position / denominator)
    stacked = torch.stack([even_PE, odd_PE], dim=2)
    pe = torch.flatten(stacked, start_dim=1, end_dim=2).repeat(batch_size,1,1)
    x_in=x_in.repeat(1,d_model,1)+pe.permute(0,2,1).to(device)
    return x_in

class SimpleUnet_unsupervised_newUnet(nn.Module):
    """
    A simplified variant of the Unet architecture.
    """
    def __init__(self):
        super().__init__()
        earthqk_channels = 32 # 1 channel in input
        down_channels = (64, 128, 256, 512, 1024, 2048)
        up_channels = (2048, 1024, 512, 256, 128, 64)
        out_dim = 1 # the model should output the gaussian noise
        time_emb_dim = 32

        # Time embedding
        self.time_mlp = nn.Sequential(
                SinusoidalPositionEmbeddings_unsupervised_newUnet(time_emb_dim),
                nn.Linear(time_emb_dim, time_emb_dim),
                nn.ReLU()
            )
        
        # Initial projection
        self.conv0 = nn.Conv1d(earthqk_channels, down_channels[0], 3, padding=1)

        # Downsample
        self.downs = nn.ModuleList([Block_unsupervised_newUnet(down_channels[i], down_channels[i+1], \
                                    time_emb_dim) \
                    for i in range(len(down_channels)-1)])
        # Upsample
        self.ups = nn.ModuleList([Block_unsupervised_newUnet(up_channels[i], up_channels[i+1], \
                                        time_emb_dim, up=True) \
                    for i in range(len(up_channels)-1)])

        self.output = nn.Conv1d(up_channels[-1], out_dim, 1)

    def forward(self, x, timestep):
        # Embedd time
        t = self.time_mlp(timestep)
        # Initial conv
        x=positionalEncoding_unsupervised_newUnet(x,batch_size=x.shape[0], device=x.device)
        x = self.conv0(x)
        # Unet
        residual_inputs = []
        for down in self.downs:
            x = down(x, t)
            residual_inputs.append(x)
        for up in self.ups:
            residual_x = residual_inputs.pop()
            # Add residual x as additional channels
            x = torch.cat((x, residual_x), dim=1)  
            x = up(x, t)
            
        x=self.output(x)
        return x