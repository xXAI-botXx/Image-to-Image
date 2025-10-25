# ---------------------------
#        > Imports <
# ---------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from ..data.physgen import PhysGenDataset 



# ---------------------------
#         > Helper <
# ---------------------------
class CombineNet(nn.Module):
    def __init__(self, input_channels=1, output_channels=1, hidden_channels=32):
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
        return self.model(torch.cat([x, y], dim=1))
    

    def backward(self, y_base, y_complex, y):
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
    def __init__(self,
                 base_model: nn.Module,
                 complex_model: nn.Module,
                 combine_mode="math"):  # math or nn
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
        return self.input_channels
    

    def get_output_channels(self):
        return self.output_channels


    def forward(self, x_base, x_complex):
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
        self.alpha_optimizer.zero_grad()
        y_pred = y_base + self.alpha * y_complex
        combine_loss = self.alpha_criterion(y_pred, y)
        combine_loss.backward()
        self.last_combined_math_loss = combine_loss.item()
        self.alpha_optimizer.step()
        
    def get_dict(self):

        return {
                f"loss_base": self.last_base_loss, 
                f"loss_complex": self.last_complex_loss, 
                f"loss_combined_net": self.combine_net.last_loss,
                f"loss_combined_math": self.last_combined_math_loss
               }



