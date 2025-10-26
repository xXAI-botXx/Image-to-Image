"""
Module to define a Pix2Pix Model. 
A UNet CNN Generator combined with a small generative loss.

Functions:
- unet_down_block
- unet_up_block

Classes:
- MMC
- UNetGenerator
- Discriminator
- Pix2Pix

By Tobia Ippolito
"""
# ---------------------------
#        > Imports <
# ---------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast



# ---------------------------
#       > Generator <
# ---------------------------
class MMC(nn.Module):  # MinMaxClamping
    """
    Min-Max Clamping Module.

    Clamps input tensor values between a specified minimum and maximum.

    Parameter:
    - min (float): 
        Minimum allowed value (default=0.0).
    - max (float): 
        Maximum allowed value (default=1.0).

    Usage:
    - Can be used at the output layer of a generator to ensure predictions remain in a valid range.
    """
    def __init__(self, min=0.0, max=1.0):
        super().__init__()
        self.min = min
        self.max = max

    def forward(self, x):
        """
        Forward pass.

        Parameter:
        - x (torch.tensor): 
            Input tensor.

        Returns:
        - torch.tensor: Clamped tensor with values between `min` and `max`.
        """
        return torch.clamp(x, self.min, self.max)

def unet_down_block(in_channels=1, out_channels=1, normalize=True):
    """
    Creates a U-Net downsampling block.

    Parameter:
    - in_channels (int): 
        Number of input channels.
    - out_channels (int): 
        Number of output channels.
    - normalize (bool): 
        Whether to apply instance normalization.

    Returns:
    - nn.Sequential: Convolutional downsampling block with LeakyReLU activation.
    """
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False)]
    if normalize:
        # layers += [nn.BatchNorm2d(out_channels)]
        layers += [nn.InstanceNorm2d(out_channels, affine=True)]
    layers += [nn.LeakyReLU(0.2, inplace=True)]
    return nn.Sequential(*layers)

def unet_up_block(in_channels=1, out_channels=1, dropout=0.0):
    """
    Creates a U-Net upsampling block.

    Parameter:
    - in_channels (int): 
        Number of input channels.
    - out_channels (int): 
        Number of output channels.
    - dropout (float): 
        Dropout probability (default=0).

    Returns:
    - nn.Sequential: Transposed convolutional block with ReLU activation and optional dropout.
    """
    layers = [
        nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False),
        # nn.BatchNorm2d(out_channels),
        nn.InstanceNorm2d(out_channels, affine=True),
        nn.ReLU(inplace=True)
    ]
    
    if dropout:
        layers += [nn.Dropout(dropout)]
    return nn.Sequential(*layers)

class UNetGenerator(nn.Module):
    """
    U-Net Generator for image-to-image translation.

    Architecture:
    - 8 downsampling blocks (encoder)
    - 8 upsampling blocks (decoder) with skip connections
    - Sigmoid activation at output for [0,1] pixel normalization

    Parameter:
    - input_channels (int): 
        Number of input image channels.
    - output_channels (int): 
        Number of output image channels.
    - hidden_channels (int): 
        Base hidden channels for first layer.
    """
    def __init__(self, input_channels=1, output_channels=1, hidden_channels=64):
        super().__init__()
        # Encoder
        self.down1 = unet_down_block(input_channels, hidden_channels, normalize=False) # 128
        self.down2 = unet_down_block(hidden_channels, hidden_channels*2)                    # 64
        self.down3 = unet_down_block(hidden_channels*2, hidden_channels*4)                  # 32
        self.down4 = unet_down_block(hidden_channels*4, hidden_channels*8)                  # 16
        self.down5 = unet_down_block(hidden_channels*8, hidden_channels*8)                  # 8
        self.down6 = unet_down_block(hidden_channels*8, hidden_channels*8)                  # 4
        self.down7 = unet_down_block(hidden_channels*8, hidden_channels*8)                  # 2
        self.down8 = unet_down_block(hidden_channels*8, hidden_channels*8, normalize=False) # 1

        # Decoder
        self.up1 = unet_up_block(hidden_channels*8, hidden_channels*8, dropout=0.5)
        self.up2 = unet_up_block(hidden_channels*16, hidden_channels*8, dropout=0.5)
        self.up3 = unet_up_block(hidden_channels*16, hidden_channels*8, dropout=0.5)
        self.up4 = unet_up_block(hidden_channels*16, hidden_channels*8)
        self.up5 = unet_up_block(hidden_channels*16, hidden_channels*4)
        self.up6 = unet_up_block(hidden_channels*8, hidden_channels*2)
        self.up7 = unet_up_block(hidden_channels*4, hidden_channels)
        self.up8 = nn.Sequential(
            nn.ConvTranspose2d(hidden_channels*2, output_channels, 4, 2, 1),
            # nn.Tanh()
            ## MMC(min=0.0, max=1.0)
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        Forward pass of the U-Net generator.

        Parameter:
        - x (torch.tensor): 
            Input image tensor (batch_size, input_channels, H, W).

        Returns:
        - torch.tensor: Generated output tensor (batch_size, output_channels, H, W).
        """
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)
        d8 = self.down8(d7)

        u1 = self.up1(d8)
        u2 = self.up2(torch.cat([u1, d7], 1))
        u3 = self.up3(torch.cat([u2, d6], 1))
        u4 = self.up4(torch.cat([u3, d5], 1))
        u5 = self.up5(torch.cat([u4, d4], 1))
        u6 = self.up6(torch.cat([u5, d3], 1))
        u7 = self.up7(torch.cat([u6, d2], 1))
        u8 = self.up8(torch.cat([u7, d1], 1))

        return u8



# ---------------------------
#      > Discriminator <
# ---------------------------
# PatchGAN
class Discriminator(nn.Module):
    """
    PatchGAN Discriminator for Pix2Pix GAN.

    Parameter:
    - input_channels (int): 
        Number of input channels (typically input + target channels concatenated).
    - hidden_channels (int): 
        Base hidden channels for first layer.

    Architecture:
    - 5 convolutional blocks with LeakyReLU and batch normalization.
    - Outputs a 2D patch map of predictions.
    """
    def __init__(self, input_channels=6, hidden_channels=64):
        """
        Initializes a PatchGAN discriminator.

        The discriminator evaluates input-target image pairs to determine
        if they are real or generated (fake). It progressively downsamples
        the spatial dimensions while increasing the number of feature channels.

        Parameters:
        - input_channels (int): 
            Number of input channels, typically input + target concatenated (default=6).
        - hidden_channels (int): 
            Number of channels in the first convolutional layer; doubled in subsequent layers (default=64).
        """
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(input_channels, hidden_channels, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(hidden_channels, hidden_channels*2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels*2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(hidden_channels*2, hidden_channels*4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels*4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(hidden_channels*4, hidden_channels*8, kernel_size=4, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels*8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(hidden_channels*8, 1, kernel_size=4, stride=1, padding=1)
        )

    def forward(self, x, y):
        """
        Forward pass of the discriminator.

        Parameter:
        - x (torch.tensor): 
            Input image tensor.
        - y (torch.tensor): 
            Target or generated image tensor.

        Returns:
        - torch.tensor: PatchGAN output tensor predicting real/fake for each patch.
        """
        # concatenate input and target channels
        return self.model(torch.cat([x, y], dim=1))



# ---------------------------
#         > Pix2Pix <
# ---------------------------
class Pix2Pix(nn.Module):
    """
    Pix2Pix GAN for image-to-image translation.

    Components:
    - Generator: U-Net generator producing synthetic images.
    - Discriminator: PatchGAN discriminator evaluating real vs fake images.
    - Adversarial loss: Binary cross-entropy.
    - Optional second loss for pixel-wise supervision.

    Parameter:
    - input_channels (int): 
        Number of input channels.
    - output_channels (int): 
        Number of output channels.
    - hidden_channels (int): 
        Base hidden channels for both generator and discriminator.
    - second_loss (nn.Module): 
        Optional secondary loss (default: L1Loss).
    - lambda_second (float): 
        Weight for secondary loss in generator optimization.
    """
    def __init__(self, input_channels=1, output_channels=1, hidden_channels=64, 
                 second_loss=nn.L1Loss(), lambda_second=100):
        """
        Initializes the Pix2Pix GAN model.

        Components:
        - Generator: U-Net architecture for producing synthetic images.
        - Discriminator: PatchGAN for evaluating real vs. fake images.
        - Adversarial loss: Binary cross-entropy to train the generator to fool the discriminator.
        - Optional secondary loss: Pixel-wise supervision (default: L1Loss).

        Parameter:
        - input_channels (int): 
            Number of channels in the input images (default=1).
        - output_channels (int): 
            Number of channels in the output images (default=1).
        - hidden_channels (int): 
            Base number of hidden channels in the generator and discriminator (default=64).
        - second_loss (nn.Module): 
            Optional secondary loss for the generator (default: nn.L1Loss()).
        - lambda_second (float): 
            Weight applied to the secondary loss in generator optimization (default=100).
        """
        super().__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels

        self.generator = UNetGenerator(input_channels=input_channels, 
                                       output_channels=output_channels, 
                                       hidden_channels=hidden_channels)
        self.discriminator = Discriminator(input_channels=input_channels+output_channels, hidden_channels=hidden_channels)

        self.adversarial_loss = nn.BCEWithLogitsLoss()
        self.second_loss = second_loss
        self.lambda_second = lambda_second

        self.last_generator_loss = float("inf")
        self.last_generator_adversarial_loss = float("inf")
        self.last_generator_second_loss = float("inf")
        self.last_discriminator_loss = float("inf")

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

    def get_dict(self):
        """
        Returns a dictionary with the most recent loss values.

        Returns:
        - dict: Loss components (base, complex).

        Notes:
        - Useful for logging or monitoring training progress.
        """
        return {
                f"loss_generator": self.last_generator_loss, 
                f"loss_generator_adversarial": self.last_generator_adversarial_loss, 
                f"loss_generator_second": self.last_generator_second_loss,
                f"loss_discriminator": self.last_discriminator_loss
               }

    def forward(self, x):
        """
        Forward pass through the generator.

        Parameter:
        - x (torch.tensor): 
            Input tensor.

        Returns:
        - torch.tensor: Generated output image.
        """
        return self.generator(x)

    def generator_step(self, x, y, optimizer, amp_scaler, device, gradient_clipping_threshold):
        """
        Performs a single optimization step for the generator.

        This includes:
        - Forward pass through the generator and discriminator.
        - Computing adversarial loss (generator tries to fool the discriminator).
        - Computing optional secondary loss (e.g., L1 or MSE).
        - Backpropagation and optimizer step, optionally with AMP and gradient clipping.

        Parameters:
        - x (torch.tensor): 
            Input tensor for the generator (e.g., source image).
        - y (torch.tensor): 
            Target tensor for supervised secondary loss.
        - optimizer (torch.optim.Optimizer): 
            Optimizer for the generator parameters.
        - amp_scaler (torch.cuda.amp.GradScaler or None): 
            Automatic mixed precision scaler.
        - device (torch.device): 
            Device for AMP autocast.
        - gradient_clipping_threshold (float or None): 
            Max norm for gradient clipping; if None, no clipping.

        Returns:
        - tuple(torch.tensor, torch.tensor, torch.tensor):
            - Total generator loss (adversarial + secondary).
            - Adversarial loss component.
            - Secondary loss component (weighted by `lambda_second`).

        Notes:
        - If AMP is enabled, gradients are scaled and unscaled appropriately.
        - `last_generator_loss`, `last_generator_adversarial_loss`, and `last_generator_second_loss` are updated.
        """
        if amp_scaler:
            with autocast(device_type=device.type):
                # make predictions
                fake_y = self.generator(x)

                discriminator_fake = self.discriminator(x, fake_y)

                # calc loss -> discriminator thinks it is real?
                loss_adversarial = self.adversarial_loss(discriminator_fake, torch.ones_like(discriminator_fake))
                loss_second = self.second_loss(fake_y, y) * self.lambda_second
                loss_total = loss_adversarial + loss_second

            # backward pass -> calc gradients and change the weights towards the opposite of gradients via optimizer
    
            optimizer.zero_grad(set_to_none=True)
            amp_scaler.scale(loss_total).backward()
            if gradient_clipping_threshold:
                # Unscale first!
                amp_scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(self.generator.parameters(), max_norm=gradient_clipping_threshold)
            amp_scaler.step(optimizer)
            amp_scaler.update()
        else:
            # make predictions
            fake_y = self.generator(x)

            discriminator_fake = self.discriminator(x, fake_y)

            # calc loss -> discriminator thinks it is real?
            loss_adversarial = self.adversarial_loss(discriminator_fake, torch.ones_like(discriminator_fake))
            loss_second = self.second_loss(fake_y, y) * self.lambda_second
            loss_total = loss_adversarial + loss_second
            optimizer.zero_grad()
            loss_total.backward()
            if gradient_clipping_threshold:
                torch.nn.utils.clip_grad_norm_(self.generator.parameters(), max_norm=gradient_clipping_threshold)
            optimizer.step()

        self.last_generator_loss = loss_total.item()
        self.last_generator_adversarial_loss = loss_adversarial.item()
        self.last_generator_second_loss = loss_second.item()

        return loss_total, loss_adversarial, loss_second

    def discriminator_step(self, x, y, optimizer, amp_scaler, device, gradient_clipping_threshold):
        """
        Performs a single optimization step for the discriminator.

        This includes:
        - Forward pass through the discriminator for both real and fake samples.
        - Computing adversarial loss (binary cross-entropy) for real vs fake patches.
        - Backpropagation and optimizer step, optionally with AMP and gradient clipping.

        Parameters:
        - x (torch.tensor): 
            Input tensor (e.g., source image).
        - y (torch.tensor): 
            Target tensor (real image) for the discriminator.
        - optimizer (torch.optim.Optimizer): 
            Optimizer for the discriminator parameters.
        - amp_scaler (torch.cuda.amp.GradScaler or None): 
            Automatic mixed precision scaler.
        - device (torch.device): 
            Device for AMP autocast.
        - gradient_clipping_threshold (float or None): 
            Max norm for gradient clipping; if None, no clipping.

        Returns:
        - tuple(torch.tensor, torch.tensor, torch.tensor):
            - Total discriminator loss (mean of real and fake losses).
            - Loss for real samples.
            - Loss for fake samples.

        Notes:
        - Fake images are detached from the generator to prevent updating its weights.
        - `last_discriminator_loss` is updated.
        - Supports AMP and optional gradient clipping for stability.
        """
        if amp_scaler:
            with autocast(device_type=device.type): 
                # make predictions
                fake_y = self.generator(x).detach()  # don't update generator!!

                discriminator_real = self.discriminator(x, y)
                discriminator_fake = self.discriminator(x, fake_y)

                # calc loss -> 1: predictions = real, 0: predictions = fake
                loss_real = self.adversarial_loss(discriminator_real, torch.ones_like(discriminator_real))  # torch.full_like(discriminator_real, 0.9)
                loss_fake = self.adversarial_loss(discriminator_fake, torch.zeros_like(discriminator_fake))
                loss_total = (loss_real + loss_fake) * 0.5

            # backward pass -> calc gradients and change the weights towards the opposite of gradients via optimizer
            optimizer.zero_grad(set_to_none=True)
            amp_scaler.scale(loss_total).backward()
            if gradient_clipping_threshold:
                # Unscale first!
                amp_scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), max_norm=gradient_clipping_threshold)
            amp_scaler.step(optimizer)
            amp_scaler.update()
        else:
            # make predictions
            fake_y = self.generator(x).detach()

            discriminator_real = self.discriminator(x, y)
            discriminator_fake = self.discriminator(x, fake_y)

            # calc loss -> 1: predictions = real, 0: predictions = fake
            loss_real = self.adversarial_loss(discriminator_real, torch.ones_like(discriminator_real))  # torch.full_like(discriminator_real, 0.9)
            loss_fake = self.adversarial_loss(discriminator_fake, torch.zeros_like(discriminator_fake))
            loss_total = (loss_real + loss_fake) * 0.5
            optimizer.zero_grad()
            loss_total.backward()
            if gradient_clipping_threshold:
                torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), max_norm=gradient_clipping_threshold)
            optimizer.step()

        self.last_discriminator_loss = loss_total.item()

        return loss_total, loss_real, loss_fake









