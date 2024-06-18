"""
Implementation of the UNet model from paper: "Denoising Diffusion Probabilistic Models" by Jonathan Ho, Ajay Jain, Pieter Abbeel
"""

import math
from typing import Optional, Tuple, Union, List

import torch
from torch import nn


class Swish(nn.Module):
    """
    Swish activation function
    """
    def forward(self, x):
        return x * torch.sigmoid(x)


class TimeEmbedding(nn.Module):
    """
    Time Embedding layer, this can be seen as a positional encoding layer.
    But instead of using sinusoidal positional encoding, we use a multi-layer perceptron (MLP) to generate embeddings.
    """

    def __init__(self, 
                 model_dim: int):
        super().__init__()
        self.model_dim = model_dim
        self.fc1 = nn.Linear(self.model_dim // 4, self.model_dim)
        self.fc2 = nn.Linear(self.model_dim, self.model_dim)
        self.activation = Swish()

    def forward(self, t: torch.Tensor):
        """
        This forward function will first implement the sinusoidal positional encoding and project it with an MLP
        """
        half_dim = self.model_dim // 8
        emb = math.log(10_000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=2)

        # Project it into a different dimension by using the MLP
        emb = self.activation(self.fc1(emb))
        emb = self.fc2(emb)

        return emb


class ResidualBlock(nn.Module):
    """
    A residual block has two convolution layers with group normalization.
    Each resolution is processed with two residual blocks.
    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        time_channels: Number of channels in the time step embeddings
        n_groups: Number of groups for group normalization
        dropout: Dropout rate
    """

    def __init__(self, 
                 in_channels: int, 
                 out_channels: int, 
                 time_channels: int,
                 n_groups: int = 32, 
                 dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.GroupNorm(n_groups, in_channels)
        self.act1 = Swish()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), padding=(1, 1))
        self.norm2 = nn.GroupNorm(n_groups, out_channels)
        self.act2 = Swish()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), padding=(1, 1))

        # Skip connection, similar to resnet
        if in_channels != out_channels:
            self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1))
        else:
            self.skip = nn.Identity()

        # Linear layer for time embeddings
        self.time_emb = nn.Linear(time_channels, out_channels)
        self.time_act = Swish()

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        """
        Args:
            x: Input tensor with shape `[batch_size, in_channels, height, width]
            t: Time step embeddings with shape `[batch_size, time_channels]
        """
        h = self.conv1(self.act1(self.norm1(x)))
        # Add time embeddings
        h += self.time_emb(self.time_act(t)).squeeze(0)[:, :, None, None]
        h = self.conv2(self.dropout(self.act2(self.norm2(h))))

        # Add skip connection
        result = h + self.skip(x)
        return result


class AttentionBlock(nn.Module):
    """
    Implementation of the attention block as described in the paper:
    "Denoising Diffusion Probabilistic Models" by Jonathan Ho, Ajay Jain, Pieter Abbeel
    This would be the same as "attention is all you need" paper by Vaswani et al.
    Args:
        n_channels: Number of input channels
        n_heads: Number of heads in the multi-head attention
        d_k: Number of dimensions in the key and query vectors
        n_groups: Number of groups for group normalization
    """
    def __init__(self, 
                 n_channels: int, 
                 n_heads: int = 8, 
                 d_k: int = None, 
                 n_groups: int = 32):
        super().__init__()
        if d_k is None:
            d_k = n_channels
        # Normalization layer
        self.norm = nn.GroupNorm(n_groups, n_channels)
        # Linear layer for query, key, and values
        self.projection = nn.Linear(n_channels, n_heads * d_k * 3)
        # Output linear layer
        self.output = nn.Linear(n_heads * d_k, n_channels)
        # Scale factor for dot-product attention
        self.scale = d_k ** -0.5
        self.n_heads = n_heads
        self.d_k = d_k

    def forward(self, x: torch.Tensor, t: Optional[torch.Tensor] = None):
        """
        Args:
            x: Input tensor with shape `[batch_size, in_channels, height, width]`
            t: Time step embeddings with shape `[batch_size, time_channels]`
        """
        batch, channels, h, w = x.shape
        x = x.view(batch, channels, -1).permute(0, 2, 1)
        qkv = self.projection(x).view(batch, -1, self.n_heads, 3 * self.d_k)
        q, k, v = torch.chunk(qkv, 3, dim=-1)
        attn = torch.einsum('bihd,bjhd->bijh', q, k) * self.scale
        attn = attn.softmax(dim=2)
        res = torch.einsum('bijh,bjhd->bihd', attn, v)
        res = res.reshape(batch, -1, self.n_heads * self.d_k)
        res = self.output(res)

        res += x

        res = res.permute(0, 2, 1).view(batch, channels, h, w)

        return res


class DownBlock(nn.Module):
    """
    DownSample Block in the UNet
    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        time_channels: Number of channels in the time step embeddings
        has_attn: Whether to use attention block, not every block has attention, check the paper for more details
    """
    def __init__(self, 
                 in_channels: int, 
                 out_channels: int, 
                 time_channels: int, 
                 has_attn: bool):
        super().__init__()
        self.res = ResidualBlock(in_channels, out_channels, time_channels)
        if has_attn:
            self.attn = AttentionBlock(out_channels)
        else:
            self.attn = nn.Identity()

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        x = self.res(x, t)
        x = self.attn(x)
        return x


class UpBlock(nn.Module):
    """
    UpSample Block in the UNet
    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        time_channels: Number of channels in the time step embeddings
        has_attn: Whether to use attention block, not every block has attention, check the paper for more details
    """
    def __init__(self, 
                 in_channels: int, 
                 out_channels: int, 
                 time_channels: int, 
                 has_attn: bool):
        super().__init__()
        self.res = ResidualBlock(in_channels + out_channels, out_channels, time_channels)
        if has_attn:
            self.attn = AttentionBlock(out_channels)
        else:
            self.attn = nn.Identity()

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        x = self.res(x, t)
        x = self.attn(x)
        return x


class MiddleBlock(nn.Module):
    """
    Middle Block in the UNet
    Args:
        n_channels: Number of input channels
        time_channels: Number of channels in the time step embeddings
    """

    def __init__(self, 
                 n_channels: int, 
                 time_channels: int):
        super().__init__()
        self.res1 = ResidualBlock(n_channels, n_channels, time_channels)
        self.attn = AttentionBlock(n_channels)
        self.res2 = ResidualBlock(n_channels, n_channels, time_channels)

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        x = self.res1(x, t)
        x = self.attn(x)
        x = self.res2(x, t)
        return x


class Upsample(nn.Module):
    """
    Upsample Layer in the UNet
    Args:
        n_channels: Number of input channels
    """

    def __init__(self, n_channels):
        super().__init__()
        self.conv = nn.ConvTranspose2d(n_channels, n_channels, (4, 4), (2, 2), (1, 1))

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        return self.conv(x)


class Downsample(nn.Module):
    """
    Downsample Layer in the UNet
    Args:
        n_channels: Number of input channels
    """

    def __init__(self, n_channels):
        super().__init__()
        self.conv = nn.Conv2d(n_channels, n_channels, (3, 3), (2, 2), (1, 1))

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        return self.conv(x)


class DiffusionUNet(nn.Module):
    """
    The whole UNet Model for DDPM
    It has the following structure:
        4 Donwsample Blocks
        2 Middle Blocks
        4 Upsample Blocks
    Args:
        image_channels: Number of channels in the image
        n_channels: Number of channels in feature map
        is_attn: Whether to use attention block at each resolution
        n_blocks: Number of blocks at each resolution
    """

    def __init__(self, 
                 image_channels: int = 3, 
                 n_channels: int = 64,
                 channel_multipliers: Union[Tuple[int, ...], List[int]] = (1, 2, 2, 4),
                 is_attn: Union[Tuple[bool, ...], List[bool]] = (False, False, True, True),
                 n_blocks: int = 2):
        super().__init__()
        
        # Different feature map resolutions
        n_factor = len(channel_multipliers)

        # project the image into high dimensional feature map space
        self.image_proj = nn.Conv2d(image_channels, n_channels, kernel_size=(3, 3), padding=(1, 1))
        
        # time embedding, the only main difference between an attention model and the DDPM model
        self.time_emb = TimeEmbedding(n_channels * 4)

        # Downsampling Blocks
        down = []
        # Number of channels
        out_channels = n_channels
        in_channels = n_channels
        for i in range(n_factor):
            out_channels = in_channels * channel_multipliers[i]
            for _ in range(n_blocks):
                down.append(DownBlock(in_channels, out_channels, n_channels * 4, is_attn[i]))
                in_channels = out_channels
            # the last block will not downsample the image
            if i < n_factor - 1:
                down.append(Downsample(in_channels))
        
        # The Downsample blocks are created on the fly
        self.down = nn.ModuleList(down)
         
        # Bottleneck Blocks 
        self.middle = MiddleBlock(out_channels, n_channels * 4, )

        # Upsampling Blocks
        up = []
        # Number of channels
        in_channels = out_channels
        # For each resolution, use the reversed n_factor to upsample the images resolution
        for i in reversed(range(n_factor)):
            out_channels = in_channels
            for _ in range(n_blocks):
                up.append(UpBlock(in_channels, out_channels, n_channels * 4, is_attn[i]))
            out_channels = in_channels // channel_multipliers[i]
            up.append(UpBlock(in_channels, out_channels, n_channels * 4, is_attn[i]))
            in_channels = out_channels
            # the last block will not upsample the image
            if i > 0:
                up.append(Upsample(in_channels))

        # The Upsample blocks are created on the fly
        self.up = nn.ModuleList(up)

        self.norm = nn.GroupNorm(8, n_channels)
        self.act = Swish()
        self.final = nn.Conv2d(in_channels, image_channels, kernel_size=(3, 3), padding=(1, 1))

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        t = self.time_emb(t)
        x = self.image_proj(x)

        h = [x]
        for m in self.down:
            x = m(x, t)
            h.append(x)

        x = self.middle(x, t)

        for m in self.up:
            if isinstance(m, Upsample):
                x = m(x, t)
            else:
                s = h.pop()
                x = torch.cat((x, s), dim=1)
                x = m(x, t)

        result = self.final(self.act(self.norm(x)))
        return result
    

if __name__ == "__main__":
    # Test the UNet model
    model = DiffusionUNet()
    x = torch.randn(1, 3, 64, 64)
    t = torch.randn(1, 1)
    with torch.no_grad():
        out = model(x, t)
        print(out.shape)