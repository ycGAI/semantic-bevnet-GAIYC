import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class DeformableConv2d(nn.Module):
    """基础的Deformable Convolution实现"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, 
                 padding=1, bias=True, modulated=True):
        super(DeformableConv2d, self).__init__()
        
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.modulated = modulated
        
        # 偏移量预测卷积
        self.offset_conv = nn.Conv2d(
            in_channels, 
            2 * kernel_size * kernel_size,  # 2D偏移量(x,y)
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=True
        )
        
        # 如果使用调制(DCNv2)，添加调制权重预测
        if modulated:
            self.modulator_conv = nn.Conv2d(
                in_channels,
                kernel_size * kernel_size,  # 每个采样点的权重
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=True
            )
        
        # 主卷积层
        self.regular_conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias
        )
        
        # 初始化
        self._init_weights()
    
    def _init_weights(self):
        # 偏移量初始化为0
        nn.init.constant_(self.offset_conv.weight, 0)
        nn.init.constant_(self.offset_conv.bias, 0)
        
        if self.modulated:
            # 调制权重初始化为0
            nn.init.constant_(self.modulator_conv.weight, 0)
            nn.init.constant_(self.modulator_conv.bias, 0)
        
        # 主卷积正常初始化
        nn.init.kaiming_normal_(self.regular_conv.weight, mode='fan_out', nonlinearity='relu')
        if self.regular_conv.bias is not None:
            nn.init.constant_(self.regular_conv.bias, 0)
    
    def forward(self, x):
        # 预测偏移量
        offset = self.offset_conv(x)
        
        if self.modulated:
            # 预测调制权重
            modulator = torch.sigmoid(self.modulator_conv(x))
            
            # 应用deformable卷积（使用grid_sample进行采样）
            x_deformed = self._deform_conv(x, offset, modulator)
        else:
            x_deformed = self._deform_conv(x, offset, None)
        
        # 应用主卷积
        out = self.regular_conv(x_deformed)
        return out
    
    def _deform_conv(self, x, offset, modulator=None):
        """简化的deformable采样实现"""
        b, c, h, w = x.shape
        
        # 创建基础网格
        grid_y, grid_x = torch.meshgrid(
            torch.arange(h, dtype=torch.float32, device=x.device),
            torch.arange(w, dtype=torch.float32, device=x.device),
            indexing='ij'  # 添加这个参数避免警告
        )
        grid = torch.stack([grid_x, grid_y], dim=-1)  # h x w x 2
        grid = grid.unsqueeze(0).expand(b, -1, -1, -1)  # b x h x w x 2
        
        # 重塑偏移量
        offset = offset.permute(0, 2, 3, 1)  # b x h x w x (2*k*k)
        offset_x = offset[..., ::2]
        offset_y = offset[..., 1::2]
        
        # 应用偏移量（简化版本，只使用中心偏移）
        # 添加一个缩放因子来控制偏移量的大小
        offset_scale = 0.1  # 限制偏移量的范围
        grid_offset = grid + offset_scale * torch.stack([
            offset_x[..., self.kernel_size*self.kernel_size//2],
            offset_y[..., self.kernel_size*self.kernel_size//2]
        ], dim=-1)
        
        # 归一化到[-1, 1]
        grid_offset[..., 0] = 2.0 * grid_offset[..., 0] / max(w - 1, 1) - 1.0
        grid_offset[..., 1] = 2.0 * grid_offset[..., 1] / max(h - 1, 1) - 1.0
        
        # 使用grid_sample进行采样
        x_sampled = F.grid_sample(x, grid_offset, mode='bilinear', padding_mode='zeros', align_corners=False)
        
        if modulator is not None:
            # 应用调制权重
            modulator = modulator.mean(dim=1, keepdim=True)  # 简化：平均所有kernel位置
            x_sampled = x_sampled * (1 + modulator)  # 使用残差形式
        
        return x_sampled