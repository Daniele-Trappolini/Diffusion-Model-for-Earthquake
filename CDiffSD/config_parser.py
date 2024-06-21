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

    parser.add_argument("--dataset_path",
                        default="/media/work/danieletrappolini/CDiffSD/Dataset/Train/",
                        type=str,
                        help="Path to the dataset")

    # GPU
    parser.add_argument("--gpu",
                        default=0,
                        type=int,
                        help="GPU index to use (default: 0)")

    # Learning rate
    parser.add_argument("--lr",
                        default=0.0001,
                        type=float,
                        help="Learning rate for the optimizer (default: 0.0001)")

    # Batch size
    parser.add_argument("--batch_size",
                        default=16,
                        type=int,
                        help="Batch size (default: 16)")

    # Epochs
    parser.add_argument("--epochs",
                        default=30,
                        type=int,
                        help="Number of training epochs (default: 30)")

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
                        default=3000,
                        type=int,
                        help="Size of the trace (default: 3000)")

    parser.add_argument("--signal_start",
                        type=int,
                        help="Signal start based on dataset type")

    # Channels
    parser.add_argument("--number_channels",
                        default=1,
                        type=int,
                        help="Number of channels (default: 1)")

    parser.add_argument("--channel_type",
                        default=channel_type,
                        type=int,
                        help="Type of channel (0, 1, 2)")

    # Diffusion model scheduler
    parser.add_argument("--T",
                        default=T,
                        type=int,
                        help=f"Timesteps for diffusion model (default: {T})")

    parser.add_argument("--scheduler_type",
                        default="linear",
                        type=str,
                        choices=['linear', 'cosine'],
                        help="Type of scheduler for the diffusion model (default: linear)")

    parser.add_argument("--s",
                        default=0.008,
                        type=int,
                        help="Scheduler parameter s (default: 0.008)")

    # RNF
    parser.add_argument("--Range_RNF",
                        default=(40, 65),
                        type=tuple,
                        help="Range RNF as a tuple of two integers (min, max). Example: --Range_RNF 10, 20")

    # Penalization
    parser.add_argument("--penalization",
                        default=3.0,
                        type=float,
                        help="Penalization value (default: 3.0)")

    # wandb configurations
    parser.add_argument("--wandb_project",
                        default="Cold_diffusion_1",
                        type=str,
                        help="Wandb project name (default: Cold_diffusion_1)")

    parser.add_argument("--wandb_entity",
                        default="geoscience_ai_sapienza",
                        type=str,
                        help="Wandb entity name (default: geoscience_ai_sapienza)")

    parser.add_argument("--wandb_name_project",
                        default='base_name',
                        type=str,
                        help="Project name for wandb (default: base_name)")

    parser.add_argument("--tuning",
                        default=True,
                        type=bool,
                        help="Whether to apply hyperparameter tuning (default: True)")

    parser.add_argument("--training",
                        default=False,
                        type=bool,
                        help="Whether to train or test the model (default: False)")

    parser.add_argument("--iswandb",
                        default=False,
                        type=bool,
                        help="Whether to log in wandb (default: False)")

    parser.add_argument("--checkpoint_path",
                        default="./Checkpoint/Channel_E/",
                        type=str,
                        help="Path to save checkpoints (default: ./Checkpoint/Channel_E/)")

    parser.add_argument("--file_name",
                        default="ColdDiffusion",
                        type=str,
                        help="File name for saved models (default: ColdDiffusion)")

    parser.add_argument("--path_model",
                        default="/media/work/danieletrappolini/CDiffSD/CDiffSD/Checkpoint/Channel_E/epoch_0_20_cosine_(40, 65)_ColdDiffusion",
                        type=str,
                        help="Path to the model weights (default: full path)")

    # Parse arguments
    args = parser.parse_args()

    # Set signal start
    args.signal_start = 700

    # Create checkpoint directory if it doesn't exist
    if not os.path.exists(args.checkpoint_path):
        os.makedirs(args.checkpoint_path)

    return args
