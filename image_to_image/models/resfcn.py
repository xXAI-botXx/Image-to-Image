# ---------------------------
#        > Imports <
# ---------------------------
import torch
import torch.nn as nn



# ---------------------------
#      > Residual FCN <
# ---------------------------
class ResidualBlock(nn.Module):
    def __init__(self, channels, kernel_size=3, padding=1):
        super().__init__()
        self.pre_fcn = nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=padding)
        self.activation = nn.GELU()
        self.post_fcn = nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=padding)

    def forward(self, x):
        fcn_x = self.pre_fcn(x)
        fcn_x = self.activation(fcn_x)
        fcn_x = self.post_fcn(fcn_x)
        return x + fcn_x



# ---------------------------
#        > ResFCN <
# ---------------------------
class ResFCN(nn.Module):
    """
    This is a very simple Image to Image model.

    It uses Fully Convolutional Network (FCN) and Residual Connections.
    """
    def __init__(self, input_channels=1, hidden_channels=64, output_channels=1, num_blocks=64):
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
        return self.input_channels

    def get_output_channels(self):
        return self.output_channels

    def forward(self, x):
        x = self.pre_layer(x)

        x = self.residual_fcn_layers(x)

        x = self.post_layer(x)

        return torch.clamp(x, 0.0, 1.0)

