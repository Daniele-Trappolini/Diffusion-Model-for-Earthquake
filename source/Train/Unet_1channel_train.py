import pandas as pd
import h5py
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns
import gc
import random
from datetime import datetime
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torch.optim import Adam
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import os
from obspy.imaging.spectrogram import spectrogram
from torchmetrics import ScaleInvariantSignalDistortionRatio
from torchmetrics import SignalNoiseRatio
from torchmetrics.audio import SignalDistortionRatio
import torch.nn.functional as F
import scipy
import seisbench.models as sbm
from obspy import Stream,Trace
import argparse

import Utils.utils_diff as u
import Utils.utils_models as um

### Args Parser ###
def read_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset_path",
                        default='c:\\Users\\dantr\\Desktop\\Github\\Dataset\\Train\\')
    
    parser.add_argument("--checkpoint_path",
                        default='C:\\Users\\dantr\\Desktop\\Github\\source\\Test\\Checkpoint\\Unet_1Channel\\')
    
    parser.add_argument("--ch",
                        default=0,
                        type=int,
                        help="number of channel")
    
    parser.add_argument("--TRACE_SIZE",
                        default=2496,
                        type=int,
                        help="trace size (default: 3000)")
    
        
    parser.add_argument("--T",
                        default=300,
                        type=int,
                        help="Timesteps (default: 300)")
    
    parser.add_argument("--batch_size",
                        default=16,
                        type=int,
                        help="Batch size (default: 16)")

    parser.add_argument("--signal_start",
                        default=700,
                        type=int,
                        help="signal_start (default: 700)")
    
    parser.add_argument("--lr",
                        default=0.0001,
                        type=float,
                        help="learning rate (default: 0.0001)")
    
    parser.add_argument("--train_percentage",
                        default=0.90,
                        type=float,
                        help="train_percentage (default: 0.95)")
    
    parser.add_argument("--val_percentage",
                        default=0.05,
                        type=float,
                        help="val_percentage (default: 0.025)")
    
    parser.add_argument("--test_percentage",
                        default=0.05,
                        type=float,
                        help="test_percentage (default: 0.025)")
    
    parser.add_argument("--seed",
                        default=1234,
                        type=int,
                        help="seed (default: 1234)")
    
    parser.add_argument("--epochs",
                        default=200,
                        type=int,
                        help="epochs (default: 200)")
    
    # args = parser.parse_args()
    args, unknown = parser.parse_known_args() # Questo solo per jupyter notebook 
    
    return args

args = read_args()


print(torch.cuda.is_available())
if torch.cuda.is_available():  
    dev = "cuda" 
    map_location=None
else:  
    dev = "cpu"  
    map_location='cpu'
device = torch.device(dev)

u.seed_everything(args.seed)
force_traces_in_test=[]
num_classes=2

df = pd.read_pickle(args.dataset_path+"df_train.csv")
df = df[:500]
df=df.drop(columns=["level_0"])
df_noise = pd.read_pickle(args.dataset_path+"df_noise_train.csv")
df_noise = df_noise[:500]
print("len(df noise_train)",len(df_noise),"len(df train)",len(df))


df, X_train, index_train, X_val, index_val, X_test, index_test= u.train_val_test_split(df, signal_start=args.signal_start, signal_end=args.signal_start+args.TRACE_SIZE, train_percentage=args.train_percentage, val_percentage=args.val_percentage, test_percentage=args.test_percentage,force_in_test=force_traces_in_test)
tr_dl = u.create_dataloader(X=X_train, y=X_train, index=index_train,target_dataset="train_dataset", batch_size=args.batch_size,normalize_data=True)
val_dl = u.create_dataloader(X=X_val, y=X_val, index=index_val,target_dataset="val_dataset", batch_size=args.batch_size,normalize_data=True)
test_dl = u.create_dataloader(X=X_test, y=X_test, index=index_test,target_dataset="test_dataset", batch_size=args.batch_size,normalize_data=True)
print("data samples in tr_dl: ", len(tr_dl.dataset))

df_noise, X_train_noise, index_train_noise, X_val_noise, index_val_noise, X_test_noise, index_test_noise=u.train_val_test_split(df_noise, signal_start=args.signal_start, signal_end=args.signal_start+args.TRACE_SIZE, train_percentage=args.train_percentage, val_percentage=args.val_percentage, test_percentage=args.test_percentage,force_in_test=[])
tr_dl_noise = u.create_dataloader(X=X_train_noise, y=X_train_noise, index=index_train_noise,target_dataset="train_dataset", batch_size=args.batch_size)
val_dl_noise = u.create_dataloader(X=X_val_noise, y=X_val_noise, index=index_val_noise,target_dataset="val_dataset", batch_size=args.batch_size)
test_dl_noise = u.create_dataloader(X=X_test_noise, y=X_test_noise, index=index_test_noise,target_dataset="test_dataset", batch_size=args.batch_size)
print("data samples in tr_dl: ", len(tr_dl_noise.dataset))

# Define beta schedule
betas = u.linear_beta_schedule(timesteps=args.T)

# Pre-calculate different terms for closed form
alphas = 1. - betas
alphas_cumprod = torch.cumprod(alphas, axis=0)
alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

model = um.SimpleUnetPosEnc()
model.to(device)
optimizer = Adam(model.parameters(), lr=args.lr)
min_loss = np.Inf
max_si_sdr = -np.Inf
si_sdr = ScaleInvariantSignalDistortionRatio()

for epoch in range(args.epochs):
    model.train() 
    for step, (batch, noise_in) in tqdm(enumerate(zip(tr_dl, tr_dl_noise)),total = len(tr_dl)):
        t=torch.Tensor([0]).type(torch.int64)
        model.zero_grad()  #aggiunto
        optimizer.zero_grad()
        noise_in = noise_in[0].permute(0,2,1).float()[:,args.ch,:].reshape(args.batch_size,1,args.TRACE_SIZE).to(device)
        x=batch[0].permute(0,2,1).float()[:,args.ch,:].reshape(args.batch_size,1,args.TRACE_SIZE).to(device)
        reduce_noise = random.randint(40, 65)*0.01
        noise_in=noise_in*reduce_noise
        earthqk_noise=x+noise_in
        out = model(earthqk_noise.to(device), t.to(device))
        noise_in_pred=out[:,0,:]
        eq_pred=out[:,1,:]
        loss_noise_in=F.l1_loss(noise_in, noise_in_pred.reshape(args.batch_size,1,args.TRACE_SIZE))
        loss_eq= F.l1_loss(x, eq_pred.reshape(args.batch_size,1,args.TRACE_SIZE))
        loss_mix= F.l1_loss(earthqk_noise, eq_pred.reshape(args.batch_size,1,args.TRACE_SIZE)+noise_in_pred.reshape(args.batch_size,1,args.TRACE_SIZE))
        loss=loss_eq+loss_noise_in+loss_mix
          
        loss.backward()
        optimizer.step()

    print("loss train", loss.item())
    model.eval()
    with torch.no_grad():
        sum_si_sdr_val = 0
        sum_val_loss=0
        for step, batch in tqdm(enumerate(val_dl), total=len(val_dl)):
            t=torch.Tensor([0]).type(torch.int64)
            noise_in = next(iter(val_dl_noise))[0].permute(0,2,1).float()[:,args.ch,:].reshape(args.batch_size,1,args.TRACE_SIZE).to(device)
            x=batch[0].permute(0,2,1).float()[:,args.ch,:].reshape(args.batch_size,1,args.TRACE_SIZE).to(device)
            reduce_noise=random.randint(40, 65)*0.01
            noise_in=noise_in*reduce_noise
            earthqk_noise=x+noise_in
            out = model(earthqk_noise.to(device), t.to(device))
            noise_in_pred=out[:,0,:]
            eq_pred=out[:,1,:]
            sum_si_sdr_val += si_sdr(x.cpu(), eq_pred.reshape(args.batch_size,1,args.TRACE_SIZE).cpu())
            curr_si_sdr= sum_si_sdr_val/len(val_dl)
            print("curr_si_sdr",curr_si_sdr," max_si_sdr",max_si_sdr)
            loss_noise_in=F.l1_loss(noise_in, noise_in_pred.reshape(args.batch_size,1,args.TRACE_SIZE))
            loss_eq= F.l1_loss(x, eq_pred.reshape(args.batch_size,1,args.TRACE_SIZE))
            loss_mix= F.l1_loss(earthqk_noise, eq_pred.reshape(args.batch_size,1,args.TRACE_SIZE)+noise_in_pred.reshape(args.batch_size,1,args.TRACE_SIZE))
            loss=loss_eq+loss_noise_in+loss_mix
            print("loss val", loss.item())
            if curr_si_sdr > max_si_sdr:
                print("Saving best epoch: ", epoch)
                max_si_sdr = curr_si_sdr
                torch.save(model.state_dict(), args.checkpoint_path+"final_epoch"+str(epoch)+"UnetPosEmb.pt")
            print("model saved")
       