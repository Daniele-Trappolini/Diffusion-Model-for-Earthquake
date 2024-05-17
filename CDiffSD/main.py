import config_parser as cp
from train_validate import train_model
import pandas as pd
from scheduler import *
import pdb
import matplotlib.pyplot as plt
import optuna
import argparse
import json
import os
# Configura gli argomenti
old_args = cp.configure_args()
device = torch.device(f"cuda:{old_args.gpu}" if torch.cuda.is_available() else "cpu")
  
df = pd.read_pickle(old_args.dataset_path + "df_train.csv")
df_noise = pd.read_pickle(old_args.dataset_path + "df_noise_train.csv")

print('Uploading data')
args = cp.configure_args()    
tr_dl, val_dl, test_dl, index_train = u.create_dataloader(df, batch_size = args.batch_size, is_noise = False, train_frac=args.train_percentage, val_frac=args.val_percentage, test_frac=args.test_percentage)
tr_dl_noise, val_dl_noise, test_dl_noise, index_noise = u.create_dataloader(df, batch_size = args.batch_size, is_noise = True, train_frac=args.train_percentage, val_frac=args.val_percentage, test_frac=args.test_percentage)

# pdb.set_trace()

# Allenamento del modello
# train_model(args, tr_dl, tr_dl_noise, val_dl, val_dl_noise)
def save_results_to_json(file_name, trial_data):
    # Se il file esiste già, leggi i dati esistenti e aggiungi il nuovo trial
    if os.path.exists(file_name):
        with open(file_name, 'r') as file:
            data = json.load(file)
        data.update(trial_data)
    else:
        data = trial_data
    
    # Scrivi i dati aggiornati nel file JSON
    with open(file_name, 'w') as file:
        json.dump(data, file, indent=4)


def objective(trial):
    args = argparse.Namespace(**vars(old_args))
    T = trial.suggest_categorical("T", [20,50,100,150,300])
    scheduler_type = trial.suggest_categorical("scheduler_type", ["linear", "cosine"])
    s = trial.suggest_categorical("s", [0.0001, 0.0008, 0.001])
    Range_RNF = trial.suggest_categorical('Range_RNF', [(50, 75), (30, 80), (40, 65), (10, 90)])

    print(f"Trial {trial.number}: T={T}, scheduler_type = {scheduler_type}, s={s}, Range_RNF={Range_RNF}")

    # Esempio di aggiornamento degli argomenti con i valori suggeriti da Optuna
    setattr(args, 'T', T)
    setattr(args, 'scheduler_type', scheduler_type )
    setattr(args, 's', s)
    setattr(args, 'Range_RNF', Range_RNF)  # Assumendo che tu possa gestire la tupla direttamente

    
    # Esegui la funzione di addestramento del modello con i parametri attuali
    model_performance = train_model(args, tr_dl, tr_dl_noise, val_dl, val_dl_noise)

    trial_data = {
        f"Trial {trial.number}": {
            'T': T,
            'scheduler_type': scheduler_type,
            's': s,
            'Range_RNF': str(Range_RNF),  # La tupla viene convertita in stringa
            'model_performance': model_performance
        }
    }

    # Salva i risultati nel file JSON dopo ogni trial
    save_results_to_json('trial_results.json', trial_data)
    return model_performance

# Crea uno studio Optuna e ottimizza
if args.tuning == True:
    study = optuna.create_study(direction='minimize')  # o 'maximize' a seconda della tua metrica
    study.optimize(objective, n_trials=30)
    print("Migliori iperparametri:", study.best_params)
else:
    train_model(old_args, tr_dl, tr_dl_noise, val_dl, val_dl_noise)

