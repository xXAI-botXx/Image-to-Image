"""
Module to define a Wrapper for a U-ViT Model for image to image.

Classes:
- UViT

By Tobia Ippolito
"""
# ---------------------------
#        > Imports <
# ---------------------------
import torch
import torch.nn as nn

from ..utils.diffusion import get_alphas_cumprod, remove_noise_step, add_noise_step

# import the model from the official repo
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from physics_u_vit.libs.uvit import UViT as _OfficialUViT
# from .uvit import UViT as _OfficialUViT



# ---------------------------
#         > U-ViT <
# ---------------------------
class UViT(nn.Module):
    """
    Wrapper around U-ViT from https://github.com/baofff/U-ViT
    """

    def __init__(self,
                 input_channels=1,
                 hidden_channels=64,
                 output_channels=1,
                 image_size=256,
                 timesteps=1000):
        """
        Parameters:
        - input_channels (int): number of input channels (default: 1)
        - hidden_channels (int): model hidden dimension (default: 64)
        - output_channels (int): number of output channels (default: 1)
        - image_size (int): resolution of input images (default: 256)
        """

        super().__init__()

        self.input_channels = input_channels
        self.output_channels = output_channels
        self.hidden_channels = hidden_channels
        self.image_size = image_size
        self.timesteps = timesteps

        # Instantiate official U-ViT
        self.model = _OfficialUViT(
            img_size=image_size,
            in_chans=input_channels + input_channels,  # because [Noise, Input-Image]
            embed_dim=hidden_channels,
            mlp_ratio=4.0,
        )

        self.schedule_alphas_cumprod = get_alphas_cumprod(timesteps=timesteps, beta_start=0.0001, beta_end=0.02)

        self.combine_net = nn.Sequential(
            nn.Conv2d(input_channels * 2, hidden_channels//2, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_channels//2, hidden_channels//2, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_channels//2, output_channels, kernel_size=1)
        )
    
    def get_input_channels(self):
        """
        Returns the number of input channels used by the model.

        Returns:
        - int: 
            Number of input channels expected by the model.
        """
        return self.input_channels

    def get_output_channels(self):
        """
        Returns the number of output channels produced by the model.

        Returns:
        - int: 
            Number of output channels the model generates
        """
        return self.output_channels
    
    def dummy_forward(self, x, timestep=None):
        """
        Forward pass used only for torchinfo summary.
        Only passes through the core model without diffusion loop.
        """
        B, C, H, W = x.shape
        device = x.device

        # Create a dummy timestep tensor if needed
        if timestep is None:
            timestep = torch.zeros(B, dtype=torch.long, device=device)

        # Pass through U-ViT core model
        x_t_with_cond = torch.cat([x, x], dim=1)  # simulate concatenated input
        epsilon_theta_raw = self.model(x_t_with_cond, timestep)
        epsilon_theta = self.combine_net(epsilon_theta_raw)

        return epsilon_theta

    def sample_forward(self, x, timestep=None):
        B, C, H, W = x.shape
        device = x.device

        with torch.no_grad():
            # determine starting timestep
            t_start = self.timesteps - 1 if timestep is None else timestep
            if isinstance(t_start, int):
                t_start = torch.full((B,), t_start, dtype=torch.long, device=device)

            # Add noise to original image
            noise = torch.randn_like(x)
            x_t = add_noise_step(x, t_start, self.schedule_alphas_cumprod, noise)

            # reverse diffusion process 
            for t_inv in reversed(range(t_start[0].item() + 1)):
                t_tensor = torch.full((B,), t_inv, dtype=torch.long, device=device)

                # concatenate condition (input image) to the noisy image
                x_t_with_cond = torch.cat([x_t, x], dim=1)

                # predict noise
                epsilon_theta_raw = self.model(x_t_with_cond, t_tensor)
                # self.combine_net.to(device)
                epsilon_theta = self.combine_net(epsilon_theta_raw)

                # compute posterior mean, means we go one step back in time
                x_t = remove_noise_step(epsilon_theta, x_t, t_tensor, self.schedule_alphas_cumprod)

            y = torch.clamp(x_t, 0.0, 1.0)

        return y

    def forward(self, x, timestep=None, inference=True, dummy_pass=False):
        """
        Forward pass through U-ViT
        Expects x: [B, C, H, W]
        Returns: same spatial size, output clamped between [0,1].
        """
        # get dimensions
        B, C, H, W = x.shape
        device = x.device

        if dummy_pass:
            return self.dummy_forward(x, timestep)
        elif inference:
            y = self.sample_forward(x, timestep)
        else:
            if timestep is None:
                timestep = torch.zeros(B, dtype=torch.long, device=x.device)
            elif isinstance(timestep, int):
                timestep = torch.full((B,), timestep, dtype=torch.long, device=x.device)

            y = self.model(x, timestep)
            y = self.combine_net(y)
            y = torch.clamp(y, 0.0, 1.0)

        return y









