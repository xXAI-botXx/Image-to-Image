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
    def __init__(self, img_size, patch_size=16, in_channels=1, embedded_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embedded_dim = embedded_dim

        self.projection = nn.Conv2d(in_channels, embedded_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # Input:                     (batch_size, in_channels, height, width)
        x = self.projection(x)     # (batch_size, embedded_size, num_patches/2, num_patches/2)
        B, C, H, W = x.shape
        x = x.flatten(2)           # (batch_size, embedded_size, num_patches)
        x = x.transpose(0, 1, 2)   # (batch_size, num_patches, embedded_size)
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
        self.num_heads = num_heads
        self.head_dim = embedded_dim // num_heads
        self.scale = self.head_dim**-0.5  # factor for scaling
        self.qkv = nn.Linear(embedded_dim, embedded_dim*3)
        self.fc = nn.Linear(embedded_dim, embedded_dim)

    def forward(self, x):
        batch_size, num_patches, embedded_dim = x.shape
        qkv = self.qkv(x)  # (batch_size, num_patches, embedded_dim*3)
        qkv = qkv.reshape(batch_size, num_patches, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)

        # compute scaled dot-product
        attention_scores = (q@k.transpose(-2, -1)) * self.scale  # (batch_size, num_heads, num_patches, num_patches)
        attention_weights = attention_scores.softmax(dim=-1)
        attention_output = (attention_weights@v).reshape(batch_size, num_patches, embedded_dim)

        return self.fc(attention_output)

