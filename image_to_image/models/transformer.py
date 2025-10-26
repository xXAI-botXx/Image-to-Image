# ---------------------------
#        > Imports <
# ---------------------------
import torch
import torch.nn as nn

# ---------------------------
#        > Patching <
# ---------------------------
# Converting an Input Image into Patches ('Tokens' in NLP)

class PatchEmbedding(nn.Module):
    """
    Converts an Image into Patches.

    Image size must be: H x W x C

    Patch size must be: P x P 
    """
    def __init__(self, img_size, patch_size=16, input_channels=1, embedded_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_channels = input_channels
        self.embedded_dim = embedded_dim

        self.projection = nn.Conv2d(input_channels, embedded_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # Input:                     (batch_size, in_channels, height, width)
        x = self.projection(x)     # (batch_size, embedded_size, num_patches/2, num_patches/2)
        B, C, H, W = x.shape
        x = x.flatten(2)           # (batch_size, embedded_size, num_patches)
        x = x.transpose(1, 2)   # (batch_size, num_patches, embedded_size)
        return x, (H, W)



# ---------------------------
#   > Positional Encoding <
# ---------------------------
# Add learnable parameters which adds positional information of the patches 
# the position of a patch is important, because a picture makes only sense if the other of sub pictures (patches)
# is right.
class PositionalEncoding(nn.Module):
    def __init__(self, num_patches, embedded_dim=768):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.zeros(1, num_patches, embedded_dim))

    def forward(self, x):
        # use broadcast addition
        return x + self.positional_embedding


# ---------------------------
#       > Attention <
# ---------------------------
# Basic element of Transformer are the attention-layer.
# Attention layer computes relations to all patches.
# This is done by calculating the similarity between 2 learnable vectors ! and K.
class Attention(nn.Module):
    def __init__(self, embedded_dim, num_heads):
        super().__init__()
        assert embedded_dim % num_heads == 0, \
            f"embedded_dim ({embedded_dim}) must be divisible by num_heads ({num_heads})"

        self.num_heads = num_heads
        self.head_dim = embedded_dim // num_heads
        self.scale = self.head_dim**-0.5  # factor for scaling
        self.qkv = nn.Linear(embedded_dim, embedded_dim*3)
        self.fc = nn.Linear(embedded_dim, embedded_dim)

    def forward(self, x):
        batch_size, num_patches, embedded_dim = x.shape
        qkv = self.qkv(x)  # (batch_size, num_patches, embedded_dim*3)
        qkv = qkv.reshape(batch_size, num_patches, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # -(3, batch_size, num_heads, num_patches, head_dim)
        q, k, v = qkv.unbind(dim=0)

        # compute scaled dot-product
        attention_scores = (q@k.transpose(-2, -1)) * self.scale  # (batch_size, num_heads, num_patches, num_patches)
        attention_weights = attention_scores.softmax(dim=-1)
        attention_output = (attention_weights@v).reshape(batch_size, num_patches, embedded_dim)

        return self.fc(attention_output)



# ---------------------------
# > Transformer Encoder Block <
# ---------------------------
# A Transformer Encoder Block consists of self-attention layer
# followed by a feedforward layer (mlp = multilayerperceptron) 
# + (layer) normalization
# and with residual connections / skip connections
class TransformerEncoderBlock(nn.Module):
    def __init__(self, embedded_dim, num_heads, mlp_dim, dropout=0.1):
        super().__init__()
        self.norm_1 = nn.LayerNorm(embedded_dim)
        self.attention = Attention(embedded_dim=embedded_dim, num_heads=num_heads)

        self.norm_2 = nn.LayerNorm(embedded_dim)
        # hidden_dim = int(embedded_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embedded_dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, embedded_dim),
            nn.Dropout(dropout)
        )

    
    def forward(self, x):
        # self attention with skip connection
        x = self.norm_1(x)
        x = x + self.attention(x)

        # MLP with skip connection
        x = self.norm_2(x)
        x = x + self.mlp(x)

        return x



# ---------------------------
#     > CNN Refinement <
# ---------------------------
class CNNRefinement(nn.Module):
    def __init__(self, input_channels=1, hidden_channels=64, output_channels=1, skip_connection=True):
        super().__init__()

        self.conv_1 = nn.Conv2d(input_channels, hidden_channels, kernel_size=3, stride=1, padding=1)
        self.activation_1 = nn.ReLU()
        
        self.conv_2 = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, stride=1, padding=1)
        self.activation_2 = nn.ReLU()

        self.conv_3 = nn.Conv2d(hidden_channels, output_channels, kernel_size=3, stride=1, padding=1)

        # or just:
        # nn.Sequential(
        #     nn.Conv2d(output_channels, 64, 3, padding=1),
        #     nn.ReLU(),
        #     nn.Conv2d(64, 64, 3, padding=1),
        #     nn.ReLU(),
        #     nn.Conv2d(64, output_channels, 3, padding=1)
        # )

        self.skip_connection = skip_connection

    
    def forward(self, x, original=None):
        x = self.conv_1(x)
        x = self.activation_1(x)

        x = self.conv_2(x)
        x = self.activation_2(x)

        x = self.conv_3(x)

        if self.skip_connection and original is not None:
            # apply correction to image
            x = x + original

        return x

# ---------------------------
#   > Img2Img Transformer <
# ---------------------------
# The whole model consists of:
#   - patching (tokenizing)
#   - add positional encoding
#   - Transformer Blocks
class PhysicFormer(nn.Module):
    """
    y_pred = Transformer(x) + CNN(Transformer(x)) + x_input

    CNN(Transformer(x)) = Pure Transformation
    CNN(Transformer(x)) + x_input = residual refinement
    Transformer(x) + CNN(Transformer(x)) + x_input = global field + local correction + geometry/residual
    """
    def __init__(self, input_channels=1, output_channels=1, 
                 img_size=256, patch_size=4, 
                 embedded_dim=1026, num_blocks=8,
                 heads=16, mlp_dim=2048, dropout=0.1):
        super().__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.patch_size = patch_size

        self.patch_embedding = PatchEmbedding(img_size=img_size,
                                              patch_size=patch_size,
                                              input_channels=input_channels,
                                              embedded_dim=embedded_dim)

        num_patches = (img_size//patch_size) * (img_size//patch_size)
        self.positional_encoding = PositionalEncoding(num_patches=num_patches, embedded_dim=embedded_dim)

        self.dropout = nn.Dropout(dropout)

        blocks = []
        for _ in range(num_blocks):
            blocks += [TransformerEncoderBlock(embedded_dim=embedded_dim, num_heads=heads, mlp_dim=mlp_dim, dropout=dropout)]
        self.transformer_blocks = nn.ModuleList(blocks)

        self.to_img = nn.Sequential(
            nn.Linear(embedded_dim, patch_size*patch_size*output_channels)
        )

        self.norm = nn.LayerNorm(embedded_dim)

        self.refinement = CNNRefinement(input_channels=output_channels, hidden_channels=64, output_channels=output_channels, skip_connection=True)


    def get_input_channels(self):
        return self.input_channels

    def get_output_channels(self):
        return self.output_channels


    def forward(self, x):
        x_input = x

        # patch embedding / tokenization
        x, (height, width) = self.patch_embedding(x)

        # encoding / add positional information
        x = self.positional_encoding(x)

        x = self.dropout(x)
        
        # transformer blocks
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x)

        x = self.norm(x)

        # translation to image
        x = self.to_img(x)

        # return it in the right format: [B, C, H, W]
        x = x.transpose(1, 2).reshape(x.shape[0], self.output_channels, self.patch_size*height, self.patch_size*width)
        # when you call .view() right after .transpose(), PyTorch can’t reinterpret the data layout -> this is an error.

        # refinement
        x = self.refinement(x, original=x_input)

        # Other version:
        # refined = self.refinement(x)

        # # Combine contributions (global + local + input)
        # x = x + refined + x_input

        return torch.sigmoid(x)  # between 0.0 and 1.0 -> alt: torch.clamp(x, 0.0, 1.0)


