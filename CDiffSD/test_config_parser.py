import argparse
import os
def configure_args(T=300, channel_type=0, dataset_type="stead"):
    parser = argparse.ArgumentParser()

    # Dataset type
    parser.add_argument("--dataset",
                        default=dataset_type,
                        choices="stead",
                        type=str,
                        help="Choose the dataset type. Options: [stead, instance]")

      # Path to the regular dataset
    parser.add_argument("--train_data_folder",
                        default='./CDiff_DAILY_GNSS/Training_Target', # GNSS_Giacomo/TIM_Data/GNSS_TIM/Train_H0
                        type=str,
                        help="Path to the regular dataset directory") 

    # Path to the noise dataset
    parser.add_argument("--val_data_folder",
                        default='./CDiff_DAILY_GNSS/Validation_Target', # GNSS_Giacomo/TIM_Data/GNSS_TIM/Test_H0
                        type=str,
                        help="Path to the noise dataset directory")
    
    # Path to the regular dataset
    parser.add_argument("--noise_train_data_folder",
                        default='./CDiff_DAILY_GNSS/Training_Noise', # GNSS_Giacomo/TIM_Data/GNSS_TIM/Train_Noise_H0
                        type=str,
                        help="Path to the regular dataset directory")

    # Path to the noise dataset
    parser.add_argument("--noise_val_data_folder",
                        default='./CDiff_DAILY_GNSS/Validation_Noise', # 
                        type=str,
                        help="Path to the noise dataset directory") 
    
    # Path to the noise dataset
    parser.add_argument("--test_data_folder",
                        default='/media/work/danieletrappolini/GNSS_Giacomo/TIM_Data/GNSS_TIM/Test_H0', # GNSS_Giacomo/TIM_Data/GNSS_TIM/Test_Noise_H0
                        type=str,
                        help="Path to the noise dataset directory") 

    # Path to the regular dataset
    parser.add_argument("--noise_test_data_folder",
                        default='/media/work/danieletrappolini/GNSS_Giacomo/TIM_Data/GNSS_TIM/Test_Noise_H0', # GNSS_Giacomo/TIM_Data/GNSS_TIM/Train_Noise_H0
                        type=str,
                        help="Path to the regular dataset directory")


    # GPU

    parser.add_argument("--gpu",
                        default=0,
                        type=int,
                        help="")
    
    # Learning rate
    parser.add_argument("--lr",
                    default=0.0001,
                    type=float,
                    help="Learning rate for the optimizer")
    # Batch size
    parser.add_argument("--batch_size",
                        default=16,
                        type=int,
                        help="Batch size (default: 32)")

    # Epochs
    parser.add_argument("--epochs",
                        default=100,
                        type=int,
                        help="Number of training epochs")

    # Train, validation, test percentages
    parser.add_argument("--train_percentage",
                        default=0.9,
                        type=float,
                        help="Percentage of data for training (default: 0.9)")
    
    parser.add_argument("--val_percentage",
                        default=0.05,
                        type=float,
                        help="Percentage of data for validation (default: 0.05)")
    
    parser.add_argument("--test_percentage",
                        default=0.05,
                        type=float,
                        help="Percentage of data for testing (default: 0.05)")
    # Trace parameters
    parser.add_argument("--trace_size",
                        default=1000,
                        type=int,
                        help="trace size (default: 3000)")
    parser.add_argument("--signal_start",
                        type=int,
                        help="signal start based on dataset type")

    # Channels
    parser.add_argument("--number_channels",
                        default=1,
                        type=int,
                        help="Number of channels")
    
    parser.add_argument("--channel_type",
                        default=channel_type,
                        type=int,
                        help="Type of channel (0, 1, 2)")

    # Diffusion model sceduler
    parser.add_argument("--T",
                        default=300,
                        type=int,
                        help=f"Timesteps for diffusion model (default: {T})")
    
    parser.add_argument("--scheduler_type",
                        default="linear",
                        type=str,
                        choices=['linear', 'cosine'],
                        help="Tipo di scheduler per il modello di diffusione (default: linear)")
    
    parser.add_argument("--s",
                        default=0.008,
                        type=int,
                        help=f"Timesteps for diffusion model (default: {T})")
    
    # RNF
    parser.add_argument("--Range_RNF",
                        default=(10,90),  # Sostituisci con il tuo default se necessario
                        type=tuple,
                        help="Range RNF come tupla di due interi (min,max). Esempio: --Range_RNF 10,20")


    # Penalization
    parser.add_argument("--penalization",
                        default=3.0,
                        type=float,
                        help="Penalization value (default: 3.0)")

    # wandb configurations
    parser.add_argument("--wandb_project",
                        default="Cold_diffusion_1",
                        type=str,
                        help="Wandb project name")
    
    parser.add_argument("--wandb_entity",
                        default="geoscience_ai_sapienza",
                        type=str,
                        help="Wandb entity name")

    # Construct names based on the passed arguments
            
  
    parser.add_argument("--wandb_name_project",
                        default='base_name',
                        type=str,
                        help="Project name for wandb")
    
    
    parser.add_argument("--tuning",
                        default=True,
                        type=bool,
                        help="If you want to apply the iperparameter tuning")

   
    parser.add_argument("--iswandb",
                        default=False,
                        type=bool,
                        help="If you want to log in wandb")
    
    parser.add_argument("--checkpoint_path",
                        default="./Checkpoint/Channel_0/",
                        type=str)
    

        
    parser.add_argument("--path_model",
                        default="",
                        type=str)
    
    
     
    parser.add_argument("--file_name",
                        default="ColdDiffusion",
                        type=str)
    
  
    args = parser.parse_args()
    
    args.signal_start = 0

    if not os.path.exists(args.checkpoint_path):
        os.makedirs(args.checkpoint_path)
    
    return args
