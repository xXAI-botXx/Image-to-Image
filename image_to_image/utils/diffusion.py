"""
Helper functions for Diffusion Model.

For adding noise to a input and performing a reverse denoising step.

Functions:
- get_alphas_cumprod
- make_one_denoising_step
- p_sample

By Tobia Ippolito
"""
# ---------------------------
#         > Import <
# ---------------------------
import torch



# ---------------------------
#        > Functions <
# ---------------------------
def get_alphas_cumprod(timesteps, beta_start=0.0001, beta_end=0.02):
    """
    Compute the cumulative product of alphas for the diffusion process.

    Parameters:
    - timesteps (int): Number of timesteps in the diffusion process.
    - beta_start (float): Starting value of beta.
    - beta_end (float): Ending value of beta.

    Returns:
    - torch.tensor: Cumulative product of alphas of shape [timesteps].
    """
    betas = torch.linspace(beta_start, beta_end, timesteps)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    return alphas_cumprod




def add_noise_step(x0, t, alphas_cumprod, noise=None):
    """
    Add noise to x0 at timestep t using the closed-form forward process.

    'q_sample' in classic DDPM.

    One forward noising step.

    Parameters:
    - x0: [B, C, H, W]
    - t:  [B]
    - alphas_cumprod: [T]
    - noise: [B, C, H, W] or None
    """
    if noise is None:
        noise = torch.randn_like(x0)

    t = t.to(x0.device)
    noise = noise.to(x0.device)
    alphas_cumprod = alphas_cumprod.to(x0.device)
    

    # Extract sqrt(alpha_bar_t) and sqrt(1 - alpha_bar_t)
    sqrt_alpha_bar = torch.sqrt(alphas_cumprod[t]).view(-1, 1, 1, 1)
    sqrt_one_minus_alpha_bar = torch.sqrt(1 - alphas_cumprod[t]).view(-1, 1, 1, 1)

    return sqrt_alpha_bar * x0 + sqrt_one_minus_alpha_bar * noise



def remove_noise_step(epsilon_theta, x_t, t, schedule):
    """
    Perform one reverse diffusion step: x_t -> x_{t-1}

    'p_sample' in classic DDPM.

    One reverse diffusion step

    Without prediction of the noise this is muste be made outside of this function.
    """
    t = t.to(x_t.device)
    schedule = schedule.to(x_t.device)

    alpha_t = schedule[t].view(-1, 1, 1, 1)
    alpha_prev = schedule[torch.clamp(t-1, min=0)].view(-1, 1, 1, 1)

    # prediction of original image
    x0_pred = (x_t - torch.sqrt(1 - alpha_t) * epsilon_theta) / torch.sqrt(alpha_t)

    # posterior mean
    # mean = (1 / alpha_t.sqrt()) * (x_t - ((1 - alpha_t)/ (1 - alpha_prev)).sqrt() * epsilon_theta)
    mean = torch.sqrt(alpha_prev) * x0_pred + torch.sqrt(1 - alpha_prev) * epsilon_theta

    # add noise if t > 0
    # make boolean in image shape + use with multiplication not branching
    is_not_zero = (t > 0).float().view(-1, 1, 1, 1)  # 1 or 0 as tensor

    z = torch.randn_like(x_t)
    beta_t = 1 - alpha_t / alpha_prev
    sigma_t = torch.sqrt(beta_t)
    # sigma_t = ((1 - alpha_prev) / (1 - alpha_t) * (1 - alpha_t / alpha_prev)).sqrt()

    x_prev = mean + is_not_zero * sigma_t * z
    return x_prev


