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
                        default='C:\\Users\\dantr\\Desktop\\Github\\source\\Test\\Checkpoint\\DeepDenoiser\\')
    
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
    
    args = parser.parse_args()
    #args, unknown = parser.parse_known_args() # Questo solo per jupyter notebook 
    
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


#### Load the Dataset ################

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


#### Training part        ################################
model = sbm.DeepDenoiser()
model.to(device)
optimizer = Adam(model.parameters(), lr=args.lr)
min_loss = np.Inf
max_si_sdr = -np.Inf
si_sdr = ScaleInvariantSignalDistortionRatio()

for epoch in range(args.epochs):
    model.train() 
    for step, (batch, noise_in) in tqdm(enumerate(zip(tr_dl, tr_dl_noise)),total = len(tr_dl)):
        model.zero_grad()
        optimizer.zero_grad()
        noise_in = noise_in[0].permute(0,2,1).float()[:,args.ch,:].reshape(args.batch_size,1,args.TRACE_SIZE).to(device)
        x=batch[0].permute(0,2,1).float()[:,args.ch,:].reshape(args.batch_size,1,args.TRACE_SIZE).to(device)
        reduce_noise = random.randint(40, 65)*0.01
        noise_in = noise_in*reduce_noise
        f_signal, t_signal, tmp_noisy_signal,y_eq, y_noise = um.create_mask(noise_in.cpu(),x.cpu())
        out = model(tmp_noisy_signal.to(device))
        eq_mask_pred=out[:,0,:]
        noise_mask_pred=out[:,1,:]
        mask = torch.stack([y_eq,y_noise],axis = 1)
        loss=F.cross_entropy(out.to(device),mask.to(device))
        
        loss.backward()
        optimizer.step()

    print("loss train", loss.item())

    model.eval()
    with torch.no_grad():
        sum_si_sdr_val = 0
        sum_val_loss=0
        for step, batch in tqdm(enumerate(val_dl), total=len(val_dl)):
            noise_in = next(iter(val_dl_noise))[0].permute(0,2,1).float()[:,args.ch,:].reshape(args.batch_size,1,args.TRACE_SIZE).to(device)
            x=batch[0].permute(0,2,1).float()[:,args.ch,:].reshape(args.batch_size,1,args.TRACE_SIZE).to(device)
            reduce_noise=random.randint(40, 65)*0.01
            noise_in=noise_in*reduce_noise
            f_signal, t_signal, tmp_noisy_signal,y_eq, y_noise=um.create_mask(noise_in.cpu(),x.cpu())
            out = model(tmp_noisy_signal.to(device))
            eq_mask_pred=out[:,0,:]
            noise_mask_pred=out[:,1,:]
            mask = torch.stack([y_eq,y_noise],axis = 1)
            sum_val_loss+=F.cross_entropy(out.to(device),mask.to(device)).item()*args.batch_size
            curr_val_loss = sum_val_loss/(len(val_dl)*args.batch_size)
        print("curr_val_loss",curr_val_loss," min_loss",min_loss) 
        if curr_val_loss < min_loss:
            min_loss = curr_val_loss
            torch.save(model.state_dict(), args.checkpoint_path+"final_epoch"+str(epoch)+"DeepDen.pt")
            print("Best Epoch:", epoch+1)

    print("model saved")