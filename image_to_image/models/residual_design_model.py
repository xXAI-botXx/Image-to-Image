"""
Module to define a Residual-Design Model.<br>
A model which consists of 2 models.

The data is (should) be splitted up in 2 parts (sub-problems) 
for example in Physgen Dataset, the base-propagation and 
the complex-propagation.

Classes:
- CombineNet
- ResidualDesignModel

By Tobia Ippolito
"""
# ---------------------------
#        > Imports <
# ---------------------------
import torch
import torch.nn as nn
import torch.optim as optim



# ---------------------------
#         > Helper <
# ---------------------------
class CombineNet(nn.Module):
    """
    Helper network for combining two images. 
    The two outputs of the submodels of the ResidualDesignModel. 

    The CombineNet is a lightweight convolutional neural network that takes 
    two input tensors, merges them channel-wise, and learns to predict a 
    combined output representation. It can be used for post-processing, 
    fusion of multiple model outputs, or blending of different feature spaces.

    Architecture Overview:
    - 3 convolutional layers with batch normalization and GELU activation.
    - Final sigmoid activation to normalize outputs between [0, 1].
    - Optimized using L1 loss (Mean Absolute Error).
    """
    def __init__(self, input_channels=1, output_channels=1, hidden_channels=32):
        """
        Init of the CombineNet model.

        Parameter:
        - input_channels (int): 
            Number of channels for each of the two input tensors 
            (e.g., 1 for grayscale, 3 for RGB).
        - output_channels (int): 
            Number of output channels of the combined result.
        - hidden_channels (int): 
            Number of feature channels in the hidden layers (internal representation).
        """
        super().__init__()

        self.model = nn.Sequential(
            nn.Conv2d(input_channels, hidden_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.GELU(),

            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.GELU(),

            nn.Conv2d(hidden_channels, output_channels, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )

        self.loss = torch.nn.L1Loss()
        self.last_loss = float("inf")
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.0001, betas=(0.5, 0.999))


    def forward(self, x, y):
        """
        Forward pass of the CombineNet.

        Parameter:
        - x (torch.tensor): 
            First input tensor of shape (batch_size, input_channels, height, width).
        - y (torch.tensor): 
            Second input tensor of shape (batch_size, input_channels, height, width).

        Returns:
        - torch.tensor: 
            Combined output tensor of shape (batch_size, output_channels, height, width).
        """
        return self.model(torch.cat([x, y], dim=1))
    

    def backward(self, y_base, y_complex, y):
        """
        Backward pass (training step) for the CombineNet.

        Parameter:
        - y_base (torch.tensor): 
            First input tensor (e.g., base model output).
        - y_complex (torch.tensor): 
            Second input tensor (e.g., complex or refined prediction).
        - y (torch.tensor): 
            Ground truth tensor (target output for supervision).

        Returns:
        - float: 
            The scalar loss value (L1 loss) from the current optimization step.
        """
        self.optimizer.zero_grad()
        y_pred = self.forward(y_base, y_complex)
        loss = self.loss(y, y_pred)
        loss.backward()
        self.last_loss = loss.item()
        self.optimizer.step()
        return self.last_loss



# ---------------------------
#  > Residual Design Model <
# ---------------------------
class ResidualDesignModel(nn.Module):
    """
    Residual Design Model for combining predictions from a base and a complex model.

    The ResidualDesignModel enables two modes of combination:
    1. **Mathematical Residual (`math`)**:
       - Computes a weighted sum of the base and complex model outputs.
       - The weight `alpha` is learnable and optimized via L1 loss.
    2. **Neural Network Fusion (`nn`)**:
       - Uses a small CNN (`CombineNet`) to learn a nonlinear combination of the outputs.
    """
    def __init__(self,
                 base_model: nn.Module,
                 complex_model: nn.Module,
                 combine_mode="math"):  # math or nn
        """
        Init of the ResidualDesignModel.

        Parameter:
        - base_model (nn.Module): Pretrained or instantiated base model.
        - complex_model (nn.Module): Pretrained or instantiated complex model.
        - combine_mode (str, default='math'): Mode for combining outputs. Options:
            - 'math': Weighted residual combination with learnable alpha.
            - 'nn': Nonlinear fusion using a small CombineNet.
        """
        super().__init__()

        self.base_model = base_model
        self.complex_model = complex_model
        self.combine_mode = combine_mode

        self.input_channels = (self.base_model.get_input_channels(), self.complex_model.get_input_channels())  # max(self.base_model.get_input_channels(), self.complex_model.get_input_channels())
        self.output_channels = min(self.base_model.get_output_channels(), self.complex_model.get_output_channels())

        self.combine_net = CombineNet(input_channels=self.base_model.get_output_channels() + self.complex_model.get_output_channels(), 
                                      output_channels=self.output_channels, hidden_channels=32)

        self.alpha = nn.Parameter(torch.tensor(0.5))
        self.alpha_optimizer = optim.Adam([self.alpha], lr=1e-5)
        self.alpha_criterion = nn.L1Loss()

        self.last_base_loss = float('nan')
        self.last_complex_loss = float('nan')
        self.last_combined_loss = float('nan')
        self.last_combined_math_loss = float('nan')


    def get_input_channels(self):
        """
        Returns a tuple with the input channels of the base and complex models.

        Returns:
        - tuple: (base_model_input_channels, complex_model_input_channels)
        """
        return self.input_channels
    

    def get_output_channels(self):
        """
        Returns the number of output channels for the combined prediction.

        Returns:
        - int: Minimum of base_model and complex_model output channels.
        """
        return self.output_channels


    def forward(self, x_base, x_complex):
        """
        Forward pass of the ResidualDesignModel.

        Parameter:
        - x_base (torch.tensor): Input tensor for the base model.
        - x_complex (torch.tensor): Input tensor for the complex model.

        Returns:
        - torch.tensor: Combined prediction, either via weighted residual or CombineNet.
        """
        y_base = self.base_model(x_base)
        y_complex = self.complex_model(x_complex)

        if self.combine_mode == 'math':
            # y = (y_complex*(-0.5)) + y_base
            y = y_base + self.alpha * y_complex
            if len(y.shape) == 4:
                y = y.squeeze(1)
            return y
        else:
            return self.combine_net(x_base, x_complex)
        
    def backward(self, y_base, y_complex, y):
        """
        Backward pass to optimize the alpha parameter for mathematical residual combination.

        Parameter:
        - y_base (torch.tensor): 
            Output of the base model.
        - y_complex (torch.tensor): 
            Output of the complex model.
        - y (torch.tensor): 
            Ground truth tensor.
        """
        self.alpha_optimizer.zero_grad()
        y_pred = y_base + self.alpha * y_complex
        combine_loss = self.alpha_criterion(y_pred, y)
        combine_loss.backward()
        self.last_combined_math_loss = combine_loss.item()
        self.alpha_optimizer.step()
        
    def get_dict(self):
        """
        Returns a dictionary with the most recent loss values.

        Returns:
        - dict: Loss components (base, complex).

        Notes:
        - Useful for logging or monitoring training progress.
        """
        return {
                f"loss_base": self.last_base_loss, 
                f"loss_complex": self.last_complex_loss, 
                f"loss_combined_net": self.combine_net.last_loss,
                f"loss_combined_math": self.last_combined_math_loss
               }



