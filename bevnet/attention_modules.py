import torch
import torch.nn as nn
import torch.nn.functional as F


class SEBlock(nn.Module):
    """Squeeze-and-Excitation block"""
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class CBAM(nn.Module):
    """Convolutional Block Attention Module"""
    def __init__(self, channel, reduction=16, kernel_size=7):
        super(CBAM, self).__init__()
        # Channel attention
        self.channel_attention = ChannelAttention(channel, reduction)
        # Spatial attention
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // reduction, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return x * self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        x_cat = self.conv1(x_cat)
        return x * self.sigmoid(x_cat)


class SelfAttention2D(nn.Module):
    """Self-attention module for 2D feature maps"""
    def __init__(self, in_channels, reduction=8):
        super(SelfAttention2D, self).__init__()
        self.in_channels = in_channels
        self.query_conv = nn.Conv2d(in_channels, in_channels // reduction, 1)
        self.key_conv = nn.Conv2d(in_channels, in_channels // reduction, 1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
        inputs:
            x : input feature maps (B, C, H, W)
        returns:
            out : self attention value + input feature
            attention: B x N x N (N is Height*Width)
        """
        B, C, H, W = x.size()
        
        # Project to query, key, value
        proj_query = self.query_conv(x).view(B, -1, H * W).permute(0, 2, 1)  # B x N x C'
        proj_key = self.key_conv(x).view(B, -1, H * W)  # B x C' x N
        proj_value = self.value_conv(x).view(B, -1, H * W)  # B x C x N
        
        # Compute attention
        attention = torch.bmm(proj_query, proj_key)  # B x N x N
        attention = self.softmax(attention)
        
        # Apply attention to values
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(B, C, H, W)
        
        # Add residual connection with learnable weight
        out = self.gamma * out + x
        return out


class PositionalEncoding2D(nn.Module):
    """2D positional encoding for attention mechanisms"""
    def __init__(self, channels):
        super(PositionalEncoding2D, self).__init__()
        self.channels = channels
        inv_freq = 1. / (10000 ** (torch.arange(0, channels, 2).float() / channels))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, tensor):
        """
        :param tensor: A 4d tensor of size (batch_size, channels, height, width)
        :return: Positional encoding of size (batch_size, channels, height, width)
        """
        if len(tensor.shape) != 4:
            raise RuntimeError("The input tensor has to be 4d!")
        
        batch_size, _, height, width = tensor.shape
        pos_x = torch.arange(height, device=tensor.device).type(self.inv_freq.type())
        pos_y = torch.arange(width, device=tensor.device).type(self.inv_freq.type())
        
        sin_inp_x = torch.einsum('i,j->ij', pos_x, self.inv_freq)
        sin_inp_y = torch.einsum('i,j->ij', pos_y, self.inv_freq)
        
        emb_x = torch.cat([sin_inp_x.sin(), sin_inp_x.cos()], dim=-1).unsqueeze(1)
        emb_y = torch.cat([sin_inp_y.sin(), sin_inp_y.cos()], dim=-1).unsqueeze(0)
        
        emb = torch.zeros(height, width, self.channels, device=tensor.device)
        emb[:, :, :self.channels//2] = emb_x
        emb[:, :, self.channels//2:] = emb_y
        
        return emb.permute(2, 0, 1).unsqueeze(0).repeat(batch_size, 1, 1, 1)


class TransformerBlock2D(nn.Module):
    """Transformer block for 2D feature maps"""
    def __init__(self, dim, num_heads=8, mlp_ratio=4., qkv_bias=False, 
                 drop=0., attn_drop=0., drop_path=0.):
        super(TransformerBlock2D, self).__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadAttention2D(dim, num_heads, qkv_bias, attn_drop, drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)

    def forward(self, x):
        # x shape: (B, C, H, W)
        B, C, H, W = x.shape
        
        # Reshape to (B, H*W, C) for transformer operations
        x_flat = x.flatten(2).transpose(1, 2)
        
        # Self-attention
        x_flat = x_flat + self.drop_path(self.attn(self.norm1(x_flat)))
        
        # MLP
        x_flat = x_flat + self.drop_path(self.mlp(self.norm2(x_flat)))
        
        # Reshape back to (B, C, H, W)
        x = x_flat.transpose(1, 2).reshape(B, C, H, W)
        return x


class MultiHeadAttention2D(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks)."""
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        output = x.div(keep_prob) * random_tensor
        return output