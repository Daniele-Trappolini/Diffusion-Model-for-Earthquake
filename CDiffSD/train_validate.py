import torch
from tqdm.auto import tqdm
import wandb
import torch.nn.functional as F
import pdb
from scheduler import *
from torch.optim import Adam
from utils.model import Unet1D
import torch.optim.lr_scheduler as lr_scheduler

### Model Parameters 
def create_model_and_optimizer(args):
    model = Unet1D(
        dim = 8,
        dim_mults = (1, 2, 4, 8),
        channels = args.number_channels
    )
    optimizer = Adam(model.parameters(), lr= args.lr)
    return model, optimizer

### Loss
def p_losses(args,denoise_model, eq_in, noise_real, t, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod,device,loss_type="l1"):
    x_start = eq_in
    x_end = eq_in + noise_real
    x_noisy = forward_diffusion_sample(x_start, x_end, t, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod).to(torch.float32)
    predicted_eq = denoise_model(x_noisy.to(torch.float32), t.to(torch.float32))
    x = x_noisy
    new_x_start = predicted_eq
    predicted_noise = ((x - get_index_from_list(sqrt_alphas_cumprod, t, x.shape) * predicted_eq) /get_index_from_list(sqrt_one_minus_alphas_cumprod, t, x.shape))
    new_x_end = predicted_noise + predicted_eq
    new_t = torch.randint(0, args.T, (x.shape[0],), device=device).long()
    new_x_noisy = forward_diffusion_sample(new_x_start,new_x_end,new_t,sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod).to(device)
    new_predicted_eq = denoise_model(new_x_noisy.to(torch.float32), new_t.to(torch.float32))

    if loss_type == 'l1':
        loss = F.l1_loss(eq_in, predicted_eq)
        loss2 = F.l1_loss(eq_in, new_predicted_eq)
        final_loss = (loss + args.penalization*loss2)
    elif loss_type == 'l2':
        loss = F.mse_loss(eq_in, predicted_eq)
        loss2 = F.mse_loss(eq_in, new_predicted_eq)
        final_loss = (loss + args.penalization*loss2)
    elif loss_type == "huber":
        loss = F.smooth_l1_loss(eq_in, predicted_eq)
        loss2 = F.smooth_l1_loss(eq_in, new_predicted_eq)
        final_loss = (loss + args.penalization*loss2)
    else:
        raise NotImplementedError()

    return final_loss

def train_one_epoch(model, optimizer, tr_dl, tr_dl_noise, args,device):
    model.train()
    sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod = compute_beta_schedule(args)
    sum_train_loss = 0
    for step, (eq_in, noise_in) in tqdm(enumerate(zip(tr_dl, tr_dl_noise)), total=len(tr_dl)):
        optimizer.zero_grad()
        #pdb.set_trace()
        ## Input 1D
        eq_in = eq_in[1][:, args.channel_type, :].unsqueeze(dim=1).to(device)
    
        reduce_noise = random.randint(*args.Range_RNF) * 0.01
        noise_real = (noise_in[1][:,args.channel_type, :].unsqueeze(dim=1) * reduce_noise).to(device)

        t = torch.randint(0, args.T, (eq_in.shape[0],), device=device).long()

        loss = p_losses(args,model.to(device), eq_in, noise_real, t, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod, device)
        sum_train_loss += loss.item()

        loss.backward()
        optimizer.step()

    curr_train_loss = sum_train_loss / len(tr_dl)
    return curr_train_loss

def validate_model(model, val_dl, val_dl_noise, args, device=device):
    model.eval()
    sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod = compute_beta_schedule(args)
    sum_val_loss = 0
    with torch.no_grad():
        for step, (eq_in, noise_in) in tqdm(enumerate(zip(val_dl, val_dl_noise)), total=len(val_dl)):
            ## Input 1D
            eq_in = eq_in[1][:,args.channel_type,:].unsqueeze(dim=1).to(device)
            reduce_noise = random.randint(*args.Range_RNF) * 0.01
            noise_real = (noise_in[1][:,args.channel_type,:].unsqueeze(dim=1) * reduce_noise).to(device)

            t = torch.randint(0, args.T, (eq_in.shape[0],), device=device).long()

            loss = p_losses(args,model.to(device), eq_in, noise_real, t, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod, device)
            sum_val_loss += loss.item()

        curr_val_loss = sum_val_loss / len(val_dl)
    return curr_val_loss

def train_model(args, tr_dl, tr_dl_noise, val_dl, val_dl_noise):
    # Initialize wandb
    print(f"Trial: T={args.T}, scheduler_type = {args.scheduler_type}, s={args.s}, Range_RNF={args.Range_RNF}")
    if args.iswandb == True:
        wandb.login()
        wandb.init(project=args.wandb_project,entity = args.wandb_entity,name = args.wandb_name_project)    
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    
    model, optimizer = create_model_and_optimizer(args)
    model = model.to(device)
    min_loss = np.inf
        
    scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1 )
    for epoch in range(args.epochs):
        train_loss = train_one_epoch(model, optimizer, tr_dl, tr_dl_noise, args, device)
        print(f'Epoch: {epoch}, Train Loss: {train_loss}')
        if args.iswandb == True:
            wandb.log({"Train Loss": train_loss}, step=epoch)  # Log train loss to wandb

        val_loss = validate_model(model, val_dl, val_dl_noise, args, device)
        print(f'Epoch: {epoch}, Val Loss: {val_loss}')
        if args.iswandb == True:
            wandb.log({"Val Loss": val_loss}, step=epoch)  # Log validation loss to wandb
        
        scheduler.step()
        # Save the model based on validation loss
        if val_loss < min_loss:
            min_loss = val_loss
            save_path = f'{args.checkpoint_path}epoch_{epoch}_{args.T}_{args.scheduler_type}_{args.Range_RNF}_{args.file_name}' 
            torch.save(model.state_dict(), save_path)
            if args.iswandb == True:
                wandb.save(save_path)  # Log the model checkpoint to wandb
            print(f"Best Epoch (so far): {epoch+1}")
    

    if args.iswandb == True:
        wandb.finish()  # End the wandb run after training is complete

    return min_loss