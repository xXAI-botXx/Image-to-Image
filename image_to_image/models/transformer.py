"""
Module to define basic Transformer parts.<br>
Also defines a Transformer model for image-to-image tasks, the PhysicsFormer.

Classes:
- PatchEmbedding
- PositionalEncoding
- Attention
- TransformerEncoderBlock
- PhysicsFormer

By Tobia Ippolito
"""
# ---------------------------
#        > Imports <
# ---------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F



# ---------------------------
#        > Patching <
# ---------------------------
class PatchEmbedding(nn.Module):
    """
    Converts an Image into Patches ('Tokens' in NLP).

    Image size must be: H x W x C

    Patch size must be: P x P 
    """
    def __init__(self, img_size, patch_size=16, input_channels=1, embedded_dim=768):
        """
        Init of Patching.

        Each filter creates one new value for each patch 
        and this with embedded_dim-filters. <br>
        So one patch is projected into a 'embedded_dim'-vector. <br>
        For example the value at [0, 0] on each channel in the embedded image is together 
        the embedded vector of the first patch.

        Parameter:
        - img_size (int): 
            Size (width or height) of your image. Your image must have the same width and height.
        - patch_size (int, default=16): 
            Size (width or height) of one patch.
        - input_channels (int): 
            Number of Input Channels to be expected.
        - embedded_dim (int): 
            Output channels / channels of the embedding.
        """
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_channels = input_channels
        self.embedded_dim = embedded_dim

        self.projection = nn.Conv2d(input_channels, embedded_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        """
        Forward pass of patching.

        Parameter:
        - x (torch.tensor): 
            Input Image(s).

        Returns:
        - tuple(torch.tensor, tuple(int, int)): 
            The Embedded image with the height and width.
        """
        # Input:                     (batch_size, in_channels, height, width)
        x = self.projection(x)     # (batch_size, embedded_size, num_patches/2, num_patches/2)
        B, C, H, W = x.shape
        x = x.flatten(2)           # (batch_size, embedded_size, num_patches)
        x = x.transpose(1, 2)   # (batch_size, num_patches, embedded_size)
        return x, (H, W)



# ---------------------------
#   > Positional Encoding <
# ---------------------------
class PositionalEncoding(nn.Module):
    """
    Add learnable parameters which adds positional information of the patches 
    the position of a patch is important, because a picture makes only sense 
    if the other of sub pictures (patches) is right.
    """
    def __init__(self, num_patches, embedded_dim=768):
        """
        Init of Positonal Encoding.

        Parameter:
        - num_patches (int): 
            Amount of Patches ('Tokens').
        - embedded_dim (int, default=768): 
            Get amount of the embedding channels.
        """
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.zeros(1, num_patches, embedded_dim))

    def forward(self, x):
        """
        Forward pass of positional information adding.

        Parameter:
        - x (torch.tensor): 
            Patch Embedded Image(s).

        Returns:
        - torch.tensor: 
            The Embedded image(s) with positional encoding added.
        """
        # use broadcast addition
        return x + self.positional_embedding



# ---------------------------
#       > Attention <
# ---------------------------
class Attention(nn.Module):
    """
    Basic element of Transformer are the attention-layer. <br>
    Attention layer computes relations to all patches. <br>
    This is done by calculating the similarity between 2 learnable vectors ! and K.
    """
    def __init__(self, embedded_dim, num_heads):
        """
        Init of Attention Layer.

        Parameter:
        - embedded_dim (int): 
            Patch Embedding Channels.
        - num_heads (int): 
            Number of parallel attention computations.
        """
        super().__init__()
        assert embedded_dim % num_heads == 0, \
            f"embedded_dim ({embedded_dim}) must be divisible by num_heads ({num_heads})"

        self.num_heads = num_heads
        self.head_dim = embedded_dim // num_heads
        self.scale = self.head_dim**-0.5  # factor for scaling
        self.qkv = nn.Linear(embedded_dim, embedded_dim*3)
        self.fc = nn.Linear(embedded_dim, embedded_dim)

    def forward(self, x):
        """
        Forward pass of Attention Layer.

        softmax(QK^T)V

        Parameter:
        - x (torch.tensor): 
            Patch Embedded Image(s) with positional encoding added.

        Returns:
        - torch.tensor: 
            The attention cores passed through fully connected layer.
        """
        batch_size, num_patches, embedded_dim = x.shape
        qkv = self.qkv(x)  # (batch_size, num_patches, embedded_dim*3)
        qkv = qkv.reshape(batch_size, num_patches, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # -(3, batch_size, num_heads, num_patches, head_dim)
        q, k, v = qkv.unbind(dim=0)

        # compute scaled dot-product
        # attention_scores = (q@k.transpose(-2, -1)) * self.scale  # (batch_size, num_heads, num_patches, num_patches)
        # attention_weights = attention_scores.softmax(dim=-1)
        # attention_output = (attention_weights@v).reshape(batch_size, num_patches, embedded_dim)

        # -> Memory-efficient attention score version (uses flash-attention when available)
        attention_output = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
        attention_output = attention_output.transpose(1, 2).reshape(
            batch_size, num_patches, embedded_dim
        )

        return self.fc(attention_output)



# ---------------------------
# > Transformer Encoder Block <
# ---------------------------
class TransformerEncoderBlock(nn.Module):
    """
    A Transformer Encoder Block consists of self-attention layer 
    followed by a feedforward layer (mlp = multilayerperceptron) 
    + (layer) normalization 
    and with residual connections / skip connections.
    """
    def __init__(self, embedded_dim, num_heads, mlp_dim, dropout=0.1):
        """
        Init of a Transformer Block.

        Parameter:
        - embedded_dim (int): 
            Patch Embedding Channels.
        - num_heads (int): 
            Number of parallel attention computations.
        - mlp_dim (int): 
            Hidden/Feature dimension of the multi layer perceptron layer.
        - dropout (float): 
            Propability of droput regulization.
        """
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
        """
        Forward pass of Transformer Block.

        Parameter:
        - x (torch.tensor): 
            Patch Embedded Image(s) with positional encoding added.

        Returns:
        - torch.tensor: 
            The attention cores passed through fully connected layer and a multilayer perceptron with layer normalization.
        """
        # self attention with skip connection
        y = x + self.attention(self.norm_1(x))

        # MLP with skip connection
        y = y + self.mlp(self.norm_2(y))

        return y



# ---------------------------
#     > CNN Refinement <
# ---------------------------
class CNNRefinement(nn.Module):
    """
    Refinement Network to remove transformer artefacts.
    """
    def __init__(self, input_channels=1, hidden_channels=64, output_channels=1, skip_connection=True):
        """
        Init of a CNN Refinement network.

        Parameter:
        - input_channels (int): 
            Number of input channels.
        - hidden_channels (int): 
            Number of hidden/feature channels.
        - output_channels (int): 
            Number of output channels.
        - skip_connection (bool): 
            Should a skip connection be used? Means if a second input (the original image) should be added to the output. 
            That changes the CNN network to learning a correction which will be applied to the original image.
        """
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
        """
        Forward pass of CNN Refinement network.

        Parameter:
        - x (torch.tensor): 
            Patch Embedded Image(s) with positional encoding added.
        - original (torch.tensor or None, default=None): 
            Original image which will be added at the end if skip_connection is set to true.

        Returns:
        - torch.tensor: 
            The refined image.
        """
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
    Image-to-Image Transformer.

    The whole model consists of:
    - Patching (tokenizing)
    - Add positional encoding
    - Transformer Blocks (Attention + MLP)
    - Image Reconstruction/Remapping -> Embedded Space to Pixel Space
    - CNN Refinement

    Model logic:<br>
    - `CNN(Transformer(x))` = Pure Transformation (skip connection = false)
    - `CNN(Transformer(x)) + x_input` = residual refinement (skip connection = true)
    - `Transformer(x) + CNN(Transformer(x)) + x_input` = global field + local correction + geometry/residual (not available yet)
    """
    def __init__(self, input_channels=1, output_channels=1, 
                 img_size=256, patch_size=4, 
                 embedded_dim=1026, num_blocks=8,
                 heads=16, mlp_dim=2048, dropout=0.1,
                 is_train=False):
        """
        Init of the PhysicFormer model.

        Parameter:
        - input_channels (int): 
            Number of input image channels (e.g., 1 for grayscale, 3 for RGB).
        - output_channels (int): 
            Number of output image channels.
        - img_size (int): 
            Size (height and width) of the input image in pixels.
        - patch_size (int): 
            Size of each square patch to split the image into. 
            The image must be divisible by this size.
        - embedded_dim (int): 
            Dimension of the patch embedding (feature space size per token).
        - num_blocks (int): 
            Number of Transformer Encoder blocks.
        - heads (int): 
            Number of attention heads per Attention layer.
        - mlp_dim (int): 
            Hidden dimension of the feed-forward (MLP) layers within each Transformer block.
        - dropout (float): 
            Dropout probability for regularization applied after positional encoding and inside MLP.
        """
        super().__init__()
        self.is_train = is_train
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

        self.refinement = CNNRefinement(input_channels=output_channels, hidden_channels=64, output_channels=output_channels, skip_connection=False)


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
        Forward pass of the PhysicFormer network.

        Parameter:
        - x (torch.tensor): 
            Input image tensor of shape (batch_size, input_channels, height, width).

        Returns:
        - torch.tensor: 
            Refined output image tensor of shape (batch_size, output_channels, height, width), 
            with values normalized to [0.0, 1.0].

        Notes:
        - The output passes through a `sigmoid()` activation, ensuring all pixel values ∈ [0, 1].
        - Designed for physics-informed or visual reconstruction tasks where local and global consistency are important.
        """
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

        if self.is_train:
            return x  # torch.sigmoid(x)  # between 0.0 and 1.0 -> alt: torch.clamp(x, 0.0, 1.0)
        else:
            return torch.clamp(x, 0.0, 1.0)


