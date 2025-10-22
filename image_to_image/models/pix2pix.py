# ---------------------------
#        > Imports <
# ---------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F



# ---------------------------
#       > Generator <
# ---------------------------
class MMC(nn.Module):  # MinMaxClamping
    def __init__(self, min=0.0, max=1.0):
        super().__init__()
        self.min = min
        self.max = max

    def forward(self, x):
        return torch.clamp(x, self.min, self.max)

def unet_down_block(in_channels=1, out_channels=1, normalize=True):
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False)]
    if normalize:
        layers += [nn.BatchNorm2d(out_channels)]
    layers += [nn.LeakyReLU(0.2, inplace=True)]
    return nn.Sequential(*layers)

def unet_up_block(in_channels=1, out_channels=1, dropout=0.0):
    layers = [
        nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
        #   # better? -> may make an module/activation
    ]
    
    if dropout:
        layers += [nn.Dropout(dropout)]
    return nn.Sequential(*layers)

class UNetGenerator(nn.Module):
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
            MMC(min=0.0, max=1.0)
        )

    def forward(self, x):
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
    def __init__(self, input_channels=6, hidden_channels=64):
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
        # concatenate input and target channels
        return self.model(torch.cat([x, y], dim=1))



# ---------------------------
#         > Pix2Pix <
# ---------------------------
class Pix2Pix(nn.Module):
    def __init__(self, input_channels=1, output_channels=1, hidden_channels=64, 
                 second_loss=nn.L1Loss(), lambda_second=100):
        super().__init__()
        self.generator = UNetGenerator(input_channels=input_channels, 
                                       output_channels=output_channels, 
                                       hidden_channels=hidden_channels)
        self.discriminator = Discriminator(input_channels=input_channels+output_channels, hidden_channels=hidden_channels)

        self.adversarial_loss = nn.BCEWithLogitsLoss()
        self.second_loss = second_loss
        self.lambda_second = lambda_second

        self.last_generator_loss = float("nan")
        self.last_generator_adversarial_loss = float("nan")
        self.last_generator_second_loss = float("nan")
        self.last_discriminator_loss = float("nan")

    def get_dict(self):
        return {
                f"loss_generator": self.last_generator_loss, 
                f"loss_generator_adversarial": self.last_generator_adversarial_loss, 
                f"loss_generator_second": self.last_generator_second_loss,
                f"loss_discriminator": self.last_discriminator_loss
               }

    def forward(self, x):
        return self.generator(x)

    def generator_step(self, x, y, optimizer, scaler):
        # make predictions
        fake_y = self.generator(x)

        discriminator_fake = self.discriminator(x, fake_y)

        # calc loss -> discriminator thinks it is real?
        loss_adversarial = self.adversarial_loss(discriminator_fake, torch.ones_like(discriminator_fake))
        loss_second = self.second_loss(fake_y, y) * self.lambda_second
        loss_total = loss_adversarial + loss_second

        # backward pass -> calc gradients and change the weights towards the opposite of gradients via optimizer
        if scaler:
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss_total).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.zero_grad()
            loss_total.backward()
            optimizer.step()

        self.last_generator_loss = loss_total.item()
        self.last_generator_adversarial_loss = loss_adversarial.item()
        self.last_generator_second_loss = loss_second.item()

        return loss_total, loss_adversarial, loss_second

    def discriminator_step(self, x, y, optimizer, scaler):
        # make predictions
        fake_y = self.generator(x).detach()

        discriminator_real = self.discriminator(x, y)
        discriminator_fake = self.discriminator(x, fake_y)

        # calc loss -> 1: predictions = real, 0: predictions = fake
        loss_real = self.adversarial_loss(discriminator_real, torch.ones_like(discriminator_real))
        loss_fake = self.adversarial_loss(discriminator_fake, torch.zeros_like(discriminator_fake))
        loss_total = (loss_real + loss_fake) * 0.5

        # backward pass -> calc gradients and change the weights towards the opposite of gradients via optimizer
        if scaler:
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss_total).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.zero_grad()
            loss_total.backward()
            optimizer.step()

        self.last_discriminator_loss = loss_total.item()

        return loss_total, loss_real, loss_fake









