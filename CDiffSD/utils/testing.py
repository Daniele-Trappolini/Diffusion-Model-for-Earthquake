import torch 
import torch.nn.functional as F
import config_parser as cp


old_args = cp.configure_args()
device = torch.device(f"cuda:{old_args.gpu}" if torch.cuda.is_available() else "cpu")

def linear_beta_schedule(timesteps, start=0.0001, end=0.02):
    return torch.linspace(start, end, timesteps)

def get_index_from_list(vals, t, x_shape):
    """ 
    Returns a specific index t of a passed list of values vals
    while considering the batch dimension.
    """
    batch_size = t.shape[0]
    out = vals.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)


def forward_diffusion_sample(x_start, x_end, t):

    # simply use the alphas to interpolate
    return (get_index_from_list(sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            get_index_from_list(sqrt_one_minus_alphas_cumprod, t, x_start.shape) * (x_end))

def get_index_from_list(vals, t, x_shape):
    """ 
    Returns a specific index t of a passed list of values vals
    while considering the batch dimension.
    """
    batch_size = t.shape[0]
    out = vals.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

# Qui non inizializzi T e le sue variabili correlate
T = None
betas = None
alphas = None
alphas_cumprod = None
alphas_cumprod_prev = None
sqrt_recip_alphas = None
sqrt_alphas_cumprod = None
sqrt_one_minus_alphas_cumprod = None
posterior_variance = None

def initialize_parameters(timesteps):
    global T, betas, alphas, alphas_cumprod, alphas_cumprod_prev, sqrt_recip_alphas, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod, posterior_variance
    
    T = timesteps
    betas = linear_beta_schedule(timesteps=T)
    alphas = 1. - betas
    alphas_cumprod = torch.cumprod(alphas, axis=0)
    alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
    sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
    posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)


## direct non utilizza i parametri di sopra
@torch.no_grad()
def direct_denoising(model,img,t):
    model.eval()
    model_mean = model(img, t)
    return model_mean
    

@torch.no_grad()
def sample(model,img,t,batch_size = 4):

    model.eval()
 
    while (t):
        xt = img
        step = torch.full((batch_size,), t - 1, dtype=torch.long).cuda() # Prendo T
        x1_bar = model(img,step) # Model mean da inserire qui 
        x2_bar = get_x2_bar_from_xt(x1_bar, img, step) # vedi sotto

        xt_bar = x1_bar 
        if t != 0:
            xt_bar = forward_diffusion_sample(x_start=xt_bar, x_end=x2_bar, t=step)

        
        xt_sub1_bar = x1_bar

        # Questa è la parte vitale dove riaggiung D(x,s-1)
        if t - 1 != 0:
            step2 = torch.full((batch_size,), t - 2, dtype = torch.long).cuda()
            xt_sub1_bar = forward_diffusion_sample(x_start = xt_sub1_bar, x_end = x2_bar, t = step2)

        x = img - xt_bar + xt_sub1_bar
        img = x
        t = t - 1
        
    return xt, img

### Bisogna trovare il modo di fare entrare quello anche qua dentro 

def get_x2_bar_from_xt(x1_bar, xt, t):
        return ((xt - (get_index_from_list(sqrt_alphas_cumprod, t, x1_bar.shape) * x1_bar) - (get_index_from_list(sqrt_one_minus_alphas_cumprod, t, x1_bar.shape)*x1_bar)) /
                get_index_from_list(sqrt_one_minus_alphas_cumprod, t, x1_bar.shape))