import torch
from tqdm.auto import tqdm
import wandb
import torch.nn.functional as F
import pdb
from scheduler import *
from torch.optim import Adam
from utils.model import Unet1D
import torch.optim.lr_scheduler as lr_scheduler
import utils.testing as testing

### Model Parameters 
def create_model_and_optimizer(args):
    '''
    Creates the Unet1D model and the Adam optimizer using the given arguments.
    
    Args:
        args (argparse.Namespace): The arguments containing model and optimizer parameters.
        
    Returns:
        model (Unet1D): The instantiated model.
        optimizer (Adam): The instantiated optimizer.
    '''
    model = Unet1D(
        dim = 8,
        dim_mults = (1, 2, 4, 8),
        channels = args.number_channels
    )
    optimizer = Adam(model.parameters(), lr= args.lr)
    return model, optimizer

def load_model_and_weights(path_model):
    '''
    Loads the Unet1D model and its weights from the specified path.
    
    Args:
        path_model (str): The path to the model weights.
        
    Returns:
        model (Unet1D): The model with loaded weights.
    '''
    model = Unet1D(dim=8, dim_mults=(1, 2, 4, 8), channels=1)
    model.load_state_dict(torch.load(path_model, map_location=device))
    return model

### Loss
def p_losses(args, denoise_model, eq_in, noise_real, t, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod, device, loss_type="l1"):
    '''
    Computes the loss for the denoising model.
    
    Args:
        args (argparse.Namespace): The arguments containing training parameters.
        denoise_model (Unet1D): The denoising model.
        eq_in (torch.Tensor): The input tensor.
        noise_real (torch.Tensor): The real noise tensor.
        t (torch.Tensor): The time steps tensor.
        sqrt_alphas_cumprod (torch.Tensor): The cumulative product of alphas.
        sqrt_one_minus_alphas_cumprod (torch.Tensor): The square root of one minus the cumulative product of alphas.
        device (torch.device): The device to run the model on.
        loss_type (str): The type of loss to use ('l1', 'l2', 'huber').
        
    Returns:
        final_loss (torch.Tensor): The computed loss.
    '''
    x_start = eq_in
    x_end = eq_in + noise_real
    x_noisy = forward_diffusion_sample(x_start, x_end, t, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod).to(torch.float32)
    predicted_eq = denoise_model(x_noisy.to(torch.float32), t.to(torch.float32))
    x = x_noisy
    new_x_start = predicted_eq
    predicted_noise = ((x - get_index_from_list(sqrt_alphas_cumprod, t, x.shape) * predicted_eq) / get_index_from_list(sqrt_one_minus_alphas_cumprod, t, x.shape))
    new_x_end = predicted_noise + predicted_eq
    new_t = torch.randint(0, args.T, (x.shape[0],), device=device).long()
    new_x_noisy = forward_diffusion_sample(new_x_start, new_x_end, new_t, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod).to(device)
    new_predicted_eq = denoise_model(new_x_noisy.to(torch.float32), new_t.to(torch.float32))

    if loss_type == 'l1':
        loss = F.l1_loss(eq_in, predicted_eq)
        loss2 = F.l1_loss(eq_in, new_predicted_eq)
        final_loss = (loss + args.penalization * loss2)
    elif loss_type == 'l2':
        loss = F.mse_loss(eq_in, predicted_eq)
        loss2 = F.mse_loss(eq_in, new_predicted_eq)
        final_loss = (loss + args.penalization * loss2)
    elif loss_type == "huber":
        loss = F.smooth_l1_loss(eq_in, predicted_eq)
        loss2 = F.smooth_l1_loss(eq_in, new_predicted_eq)
        final_loss = (loss + args.penalization * loss2)
    else:
        raise NotImplementedError()

    return final_loss

def train_one_epoch(model, optimizer, tr_dl, tr_dl_noise, args, device):
    '''
    Trains the model for one epoch.
    
    Args:
        model (Unet1D): The model to train.
        optimizer (Adam): The optimizer.
        tr_dl (DataLoader): The training data loader.
        tr_dl_noise (DataLoader): The noisy training data loader.
        args (argparse.Namespace): The arguments containing training parameters.
        device (torch.device): The device to run the model on.
        
    Returns:
        curr_train_loss (float): The average training loss for the epoch.
    '''
    model.train()
    sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod = compute_beta_schedule(args)
    sum_train_loss = 0
    for step, (eq_in, noise_in) in tqdm(enumerate(zip(tr_dl, tr_dl_noise)), total=len(tr_dl)):
        optimizer.zero_grad()
        eq_in = eq_in[1][:, args.channel_type, :].unsqueeze(dim=1).to(device)
        reduce_noise = random.randint(*args.Range_RNF) * 0.01
        noise_real = (noise_in[1][:, args.channel_type, :].unsqueeze(dim=1) * reduce_noise).to(device)
        t = torch.randint(0, args.T, (eq_in.shape[0],), device=device).long()
        loss = p_losses(args, model.to(device), eq_in, noise_real, t, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod, device)
        sum_train_loss += loss.item()
        loss.backward()
        optimizer.step()
    curr_train_loss = sum_train_loss / len(tr_dl)
    return curr_train_loss

def validate_model(model, val_dl, val_dl_noise, args, device=device):
    '''
    Validates the model on the validation dataset.
    
    Args:
        model (Unet1D): The model to validate.
        val_dl (DataLoader): The validation data loader.
        val_dl_noise (DataLoader): The noisy validation data loader.
        args (argparse.Namespace): The arguments containing validation parameters.
        device (torch.device): The device to run the model on.
        
    Returns:
        curr_val_loss (float): The average validation loss.
    '''
    model.eval()
    sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod = compute_beta_schedule(args)
    sum_val_loss = 0
    with torch.no_grad():
        for step, (eq_in, noise_in) in tqdm(enumerate(zip(val_dl, val_dl_noise)), total=len(val_dl)):
            eq_in = eq_in[1][:, args.channel_type, :].unsqueeze(dim=1).to(device)
            reduce_noise = random.randint(*args.Range_RNF) * 0.01
            noise_real = (noise_in[1][:, args.channel_type, :].unsqueeze(dim=1) * reduce_noise).to(device)
            t = torch.randint(0, args.T, (eq_in.shape[0],), device=device).long()
            loss = p_losses(args, model.to(device), eq_in, noise_real, t, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod, device)
            sum_val_loss += loss.item()
        curr_val_loss = sum_val_loss / len(val_dl)
    return curr_val_loss

def train_model(args, tr_dl, tr_dl_noise, val_dl, val_dl_noise):
    '''
    Trains the model for multiple epochs and validates it.
    
    Args:
        args (argparse.Namespace): The arguments containing training parameters.
        tr_dl (DataLoader): The training data loader.
        tr_dl_noise (DataLoader): The noisy training data loader.
        val_dl (DataLoader): The validation data loader.
        val_dl_noise (DataLoader): The noisy validation data loader.
        
    Returns:
        min_loss (float): The minimum validation loss achieved during training.
    '''
    print(f"Trial: T={args.T}, scheduler_type = {args.scheduler_type}, s={args.s}, Range_RNF={args.Range_RNF}")
    if args.iswandb:
        wandb.login()
        wandb.init(project=args.wandb_project, entity=args.wandb_entity, name=args.wandb_name_project)    
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    
    model, optimizer = create_model_and_optimizer(args)
    model = model.to(device)
    min_loss = np.inf
    scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    for epoch in range(args.epochs):
        train_loss = train_one_epoch(model, optimizer, tr_dl, tr_dl_noise, args, device)
        print(f'Epoch: {epoch}, Train Loss: {train_loss}')
        if args.iswandb:
            wandb.log({"Train Loss": train_loss}, step=epoch)  # Log train loss to wandb
        val_loss = validate_model(model, val_dl, val_dl_noise, args, device)
        print(f'Epoch: {epoch}, Val Loss: {val_loss}')
        if args.iswandb:
            wandb.log({"Val Loss": val_loss}, step=epoch)  # Log validation loss to wandb
        scheduler.step()
        if val_loss < min_loss:
            min_loss = val_loss
            save_path = f'{args.checkpoint_path}epoch_{epoch}_{args.T}_{args.scheduler_type}_{args.Range_RNF}_{args.file_name}' 
            torch.save(model.state_dict(), save_path)
            if args.iswandb:
                wandb.save(save_path)  # Log the model checkpoint to wandb
            print(f"Best Epoch (so far): {epoch+1}")
    if args.iswandb:
        wandb.finish()  # End the wandb run after training is complete
    return min_loss

def test_model(args, test_loader, noise_test_loader):
    '''
    Tests the model on the test dataset and saves the results.
    
    Args:
        args (argparse.Namespace): The arguments containing test parameters.
        test_loader (DataLoader): The test data loader.
        noise_test_loader (DataLoader): The noisy test data loader.
    '''
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    testing.initialize_parameters(args.T)
    model = load_model_and_weights(args.path_model)
    model = model.to(device)

    Original, restored_direct, restored_sampling, Noised = [], [], [], []
    
    T = args.T

    with torch.no_grad():
        model.eval()
        for eq_in, noise_in in tqdm(zip(test_loader, noise_test_loader), total=len(test_loader)):
            eq_in = eq_in[1][:,args.channel_type,:].unsqueeze(dim=1).to(device)
            reduce_noise = random.randint(*args.Range_RNF) * 0.01
            noise_real = (noise_in[1][:,args.channel_type,:].unsqueeze(dim=1) * reduce_noise).to(device)
            signal_noisy = eq_in + noise_real
            t = torch.Tensor([T-1]).long().to(device)
            
            restored_ch1 = testing.direct_denoising(model, signal_noisy.to(device).float().reshape(-1,1,args.trace_size), t)
            restored_direct.extend([x[0].cpu().numpy() for x in restored_ch1])

            t = T-1
            restored_sample = testing.sample(
                                            model,
                                            signal_noisy.float().reshape(-1, 1, args.trace_size),
                                            t,
                                            batch_size=signal_noisy.shape[0]
                                            )
            restored_sampling.extend([x[0].cpu().numpy() for x in restored_sample[-1]])
            Original.extend(eq_in.squeeze().cpu().numpy())
            Noised.extend(signal_noisy.squeeze().cpu().numpy())

    np.save(f"./Restored/Restored_direct_0.npy", np.array(restored_direct))
    np.save(f"./Restored/Restored_sampling_0.npy", np.array(restored_sampling))
    np.save(f"./Restored/Original.npy", np.array(Original))
    np.save(f"./Restored/Noised.npy", np.array(Noised))
