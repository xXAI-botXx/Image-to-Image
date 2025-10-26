"""
Module to define a simple CNN Model for image to image. 

Classes:
- ResidualBlock
- ResFCN

By Tobia Ippolito
"""
# ---------------------------
#        > Imports <
# ---------------------------
import torch
import torch.nn as nn



# ---------------------------
#      > Residual FCN <
# ---------------------------
class ResidualBlock(nn.Module):
    """
    A simple residual block using fully convolutional layers.

    The block applies two convolutional layers with an activation in between
    and adds the input to the output (skip connection) to preserve features.
    """
    def __init__(self, channels, kernel_size=3, padding=1):
        """
        Initializes the ResidualBlock.

        Parameters:
        - channels (int): 
            Number of input and output channels.
        - kernel_size (int): 
            Kernel size for the convolutional layers (default=3).
        - padding (int): 
            Padding for the convolutional layers to maintain spatial dimensions (default=1).
        """
        super().__init__()
        self.pre_fcn = nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=padding)
        self.activation = nn.GELU()
        self.post_fcn = nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=padding)

    def forward(self, x):
        """
        Forward pass through the ResidualBlock.

        Parameters:
        - x (torch.tensor): 
            Input tensor of shape [B, C, H, W].

        Returns:
        - torch.tensor: Output tensor after residual connection, same shape as input.
        """
        fcn_x = self.pre_fcn(x)
        fcn_x = self.activation(fcn_x)
        fcn_x = self.post_fcn(fcn_x)
        return x + fcn_x



# ---------------------------
#        > ResFCN <
# ---------------------------
class ResFCN(nn.Module):
    """
    A simple image-to-image model using Fully Convolutional Networks (FCN) with residual connections.

    Architecture:
    - Initial convolution layer to increase channel depth.
    - Sequence of ResidualBlocks with varying kernel sizes.
    - Final convolution layer to project back to output channels.
    - Output is clamped between 0.0 and 1.0.
    """
    def __init__(self, input_channels=1, hidden_channels=64, output_channels=1, num_blocks=64):
        """
        Initializes the ResFCN model.

        Parameters:
        - input_channels (int): 
            Number of channels in the input image (default=1).
        - hidden_channels (int): 
            Number of channels in hidden layers / residual blocks (default=64).
        - output_channels (int): 
            Number of channels in the output image (default=1).
        - num_blocks (int): 
            Number of residual blocks to apply (default=64).
        """
        super().__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels

        self.pre_layer = nn.Conv2d(input_channels, hidden_channels, kernel_size=7, padding=3)

        kernel_sizes = [1, 3, 5, 7, 9]
        residual_fcn_layers = []
        for i in range(num_blocks):
            cur_kernel_size = kernel_sizes[i%(len(kernel_sizes)-1)]
            residual_fcn_layers += [ResidualBlock(channels=hidden_channels, 
                                                  kernel_size=cur_kernel_size,
                                                  padding=cur_kernel_size // 2
                                                 )
                                   ]
        self.residual_fcn_layers = nn.Sequential(*residual_fcn_layers)

        self.post_layer = nn.Conv2d(hidden_channels, output_channels, kernel_size=3, padding=1)

    
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

    def forward(self, x):
        """
        Forward pass through the ResFCN model.

        Parameters:
        - x (torch.tensor): 
            Input image tensor of shape [B, C, H, W].

        Returns:
        - torch.tensor: Output image tensor, same spatial size as input, values clamped between 0.0 and 1.0.
        """
        x = self.pre_layer(x)

        x = self.residual_fcn_layers(x)

        x = self.post_layer(x)

        return torch.clamp(x, 0.0, 1.0)

