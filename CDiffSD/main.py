import config_parser as cp
from train_validate import train_model, test_model
import pandas as pd
from scheduler import *

import optuna
import argparse
import json
import os

# Configure arguments
old_args = cp.configure_args()
device = torch.device(f"cuda:{old_args.gpu}" if torch.cuda.is_available() else "cpu")

# Load datasets
df = pd.read_pickle(old_args.dataset_path + "df_train.csv")
df_noise = pd.read_pickle(old_args.dataset_path + "df_noise_train.csv")

print('Uploading data')

args = cp.configure_args()

# Create dataloaders
tr_dl, val_dl, test_dl, index_train = u.create_dataloader(df, batch_size=args.batch_size, is_noise=False, train_frac=args.train_percentage, val_frac=args.val_percentage, test_frac=args.test_percentage)
tr_dl_noise, val_dl_noise, test_dl_noise, index_noise = u.create_dataloader(df_noise, batch_size=args.batch_size, is_noise=True, train_frac=args.train_percentage, val_frac=args.val_percentage, test_frac=args.test_percentage)

# Create dataloaders for testing with batch size 1
_, _, test_dl, index_train = u.create_dataloader(df, batch_size=1, is_noise=False, train_frac=args.train_percentage, val_frac=args.val_percentage, test_frac=args.test_percentage)
_, _, test_dl_noise, index_noise = u.create_dataloader(df_noise, batch_size=1, is_noise=True, train_frac=args.train_percentage, val_frac=args.val_percentage, test_frac=args.test_percentage)

# Function to save trial results to a JSON file
def save_results_to_json(file_name, trial_data):
    # If the file already exists, read the existing data and add the new trial
    if os.path.exists(file_name):
        with open(file_name, 'r') as file:
            data = json.load(file)
        data.update(trial_data)
    else:
        data = trial_data
    
    # Write the updated data to the JSON file
    with open(file_name, 'w') as file:
        json.dump(data, file, indent=4)

# Objective function for Optuna optimization
def objective(trial):
    args = argparse.Namespace(**vars(old_args))
    T = trial.suggest_categorical("T", [20, 50, 100, 150, 300])
    scheduler_type = trial.suggest_categorical("scheduler_type", ["linear", "cosine"])
    s = trial.suggest_categorical("s", [0.0001, 0.0008, 0.001])
    Range_RNF = trial.suggest_categorical('Range_RNF', [(50, 75), (30, 80), (40, 65), (10, 90)])

    print(f"Trial {trial.number}: T={T}, scheduler_type={scheduler_type}, s={s}, Range_RNF={Range_RNF}")

    # Example of updating arguments with the values suggested by Optuna
    setattr(args, 'T', T)
    setattr(args, 'scheduler_type', scheduler_type)
    setattr(args, 's', s)
    setattr(args, 'Range_RNF', Range_RNF)  # Assuming you can handle the tuple directly

    # Train the model with the current parameters
    model_performance = train_model(args, tr_dl, tr_dl_noise, val_dl, val_dl_noise)

    trial_data = {
        f"Trial {trial.number}": {
            'T': T,
            'scheduler_type': scheduler_type,
            's': s,
            'Range_RNF': str(Range_RNF),  # Convert the tuple to a string
            'model_performance': model_performance
        }
    }

    # Save the results to the JSON file after each trial
    save_results_to_json('trial_results.json', trial_data)
    return model_performance

# Create an Optuna study and optimize
if args.training:
    if args.tuning:
        study = optuna.create_study(direction='minimize')  # or 'maximize' depending on your metric
        study.optimize(objective, n_trials=30)
        print("Best hyperparameters:", study.best_params)
    else:
        train_model(old_args, tr_dl, tr_dl_noise, val_dl, val_dl_noise)
else:
    test_model(old_args, test_dl, test_dl_noise)
