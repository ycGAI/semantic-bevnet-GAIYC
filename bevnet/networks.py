import functools
import torch
import torch.nn as nn
import torch.nn.functional as F
import spconv
from torchvision.models.resnet import resnet18

# Import the attention modules from the previous file
from attention_modules import SEBlock, CBAM, SelfAttention2D, TransformerBlock2D


class SpMiddleNoDownsampleXYWithAttention(nn.Module):
    """
    SpMiddleNoDownsampleXY with SE attention blocks
    """
    def __init__(self,
                 output_shape,
                 num_input_features,
                 use_attention='se',  # 'se', 'cbam', 'self', or None
                 attention_reduction=16):
        super(SpMiddleNoDownsampleXYWithAttention, self).__init__()

        BatchNorm1d = functools.partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)
        SpConv3d = functools.partial(spconv.SparseConv3d, bias=False)
        SubMConv3d = functools.partial(spconv.SubMConv3d, bias=False)

        import numpy as np
        sparse_shape = np.array(output_shape[1:4]) + [1, 0, 0]
        self.sparse_shape = sparse_shape
        self.voxel_output_shape = output_shape
        
        self.use_attention = use_attention
        
        # Build the sparse convolution layers
        self.middle_conv = spconv.SparseSequential(
            SubMConv3d(num_input_features, 32, 3, indice_key="subm0"),
            BatchNorm1d(32),
            nn.ReLU(),
            SubMConv3d(32, 32, 3, indice_key="subm0"),
            BatchNorm1d(32),
            nn.ReLU(),
            SpConv3d(32, 64, 3, (2, 1, 1), padding=[1, 1, 1]),
            BatchNorm1d(64),
            nn.ReLU(),
            SubMConv3d(64, 64, 3, indice_key="subm1"),
            BatchNorm1d(64),
            nn.ReLU(),
            SubMConv3d(64, 64, 3, indice_key="subm1"),
            BatchNorm1d(64),
            nn.ReLU(),
            SubMConv3d(64, 64, 3, indice_key="subm1"),
            BatchNorm1d(64),
            nn.ReLU(),
            SpConv3d(64, 64, 3, (2, 1, 1), padding=[0, 1, 1]),
            BatchNorm1d(64),
            nn.ReLU(),
            SubMConv3d(64, 64, 3, indice_key="subm2"),
            BatchNorm1d(64),
            nn.ReLU(),
            SubMConv3d(64, 64, 3, indice_key="subm2"),
            BatchNorm1d(64),
            nn.ReLU(),
            SubMConv3d(64, 64, 3, indice_key="subm2"),
            BatchNorm1d(64),
            nn.ReLU(),
            SpConv3d(64, 64, (3, 1, 1), (2, 1, 1)),
            BatchNorm1d(64),
            nn.ReLU(),
        )
        
        # Add attention modules after converting to dense
        if use_attention == 'se':
            self.attention = SEBlock(192, reduction=attention_reduction)  # 64 * 3 = 192 channels after reshape
        elif use_attention == 'cbam':
            self.attention = CBAM(192, reduction=attention_reduction)
        elif use_attention == 'self':
            self.attention = SelfAttention2D(192, reduction=8)
        elif use_attention == 'transformer':
            self.attention = TransformerBlock2D(192, num_heads=8)
        else:
            self.attention = None

    def forward(self, voxel_features, coors, batch_size):
        coors = coors.int()
        ret = spconv.SparseConvTensor(voxel_features, coors, self.sparse_shape, batch_size)
        ret = self.middle_conv(ret)
        ret = ret.dense()
        N, C, D, H, W = ret.shape
        ret = ret.view(N, C * D, H, W)
        
        # Apply attention if enabled
        if self.attention is not None:
            ret = self.attention(ret)
        
        return ret


class InpaintingResNet18WithAttention(nn.Module):
    """ResNet18 with attention modules for BEV semantic segmentation"""
    def __init__(self, num_input_features, num_class, 
                 use_attention='se',  # 'se', 'cbam', 'self', 'transformer', or None
                 attention_positions=['after_layer1', 'after_layer2', 'after_layer3'],
                 attention_reduction=16):
        super(InpaintingResNet18WithAttention, self).__init__()

        trunk = resnet18(pretrained=False, zero_init_residual=True)
        self.conv1 = nn.Conv2d(
            num_input_features, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = trunk.bn1
        self.relu = trunk.relu

        self.layer1 = trunk.layer1
        self.layer2 = trunk.layer2
        self.layer3 = trunk.layer3

        self.use_attention = use_attention
        self.attention_positions = attention_positions
        
        # Add attention modules at specified positions
        if use_attention and 'after_layer1' in attention_positions:
            if use_attention == 'se':
                self.attention1 = SEBlock(64, reduction=attention_reduction)
            elif use_attention == 'cbam':
                self.attention1 = CBAM(64, reduction=attention_reduction)
            elif use_attention == 'self':
                self.attention1 = SelfAttention2D(64, reduction=8)
            elif use_attention == 'transformer':
                self.attention1 = TransformerBlock2D(64, num_heads=4)
        else:
            self.attention1 = None
            
        if use_attention and 'after_layer2' in attention_positions:
            if use_attention == 'se':
                self.attention2 = SEBlock(128, reduction=attention_reduction)
            elif use_attention == 'cbam':
                self.attention2 = CBAM(128, reduction=attention_reduction)
            elif use_attention == 'self':
                self.attention2 = SelfAttention2D(128, reduction=8)
            elif use_attention == 'transformer':
                self.attention2 = TransformerBlock2D(128, num_heads=8)
        else:
            self.attention2 = None
            
        if use_attention and 'after_layer3' in attention_positions:
            if use_attention == 'se':
                self.attention3 = SEBlock(256, reduction=attention_reduction)
            elif use_attention == 'cbam':
                self.attention3 = CBAM(256, reduction=attention_reduction)
            elif use_attention == 'self':
                self.attention3 = SelfAttention2D(256, reduction=8)
            elif use_attention == 'transformer':
                self.attention3 = TransformerBlock2D(256, num_heads=8)
        else:
            self.attention3 = None

        self.up1 = Up(64+256, 256, scale_factor=4)
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(256, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, num_class, kernel_size=1, padding=0),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x1 = self.layer1(x)
        if self.attention1 is not None:
            x1 = self.attention1(x1)
            
        x = self.layer2(x1)
        if self.attention2 is not None:
            x = self.attention2(x)
            
        x = self.layer3(x)
        if self.attention3 is not None:
            x = self.attention3(x)

        x = self.up1(x, x1)
        x = self.up2(x)

        return dict(bev_preds=x)


class Up(nn.Module):
    def __init__(self, inC, outC, scale_factor=2):
        super().__init__()

        self.up = nn.Upsample(
            scale_factor=scale_factor, mode='bilinear', align_corners=False
        )

        self.conv = nn.Sequential(
            nn.Conv2d(inC, outC, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(outC),
            nn.ReLU(inplace=True),
            nn.Conv2d(outC, outC, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(outC),
            nn.ReLU(inplace=True)
        )

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x1 = torch.cat([x2, x1], dim=1)
        return self.conv(x1)


class MergeUnitWithAttention(nn.Module):
    """MergeUnit with attention mechanism for temporal feature fusion"""
    def __init__(self,
            input_channels,
            rnn_input_channels=None,
            rnn_config=None,
            costmap_pose_name=None,
            use_attention='se',  # Add attention type
            attention_reduction=16):
        super(MergeUnitWithAttention, self).__init__()

        if rnn_input_channels is None:
            self.pre_rnn_conv = None
            rnn_input_channels = input_channels
        else:
            # Import fchardnet ConvLayer if needed
            import fchardnet
            self.pre_rnn_conv = fchardnet.ConvLayer(input_channels,
                                                    rnn_input_channels,
                                                    kernel=1,
                                                    bn=True)

        self.costmap_pose_name = costmap_pose_name
        
        # Add attention before RNN
        if use_attention == 'se':
            self.pre_attention = SEBlock(rnn_input_channels, reduction=attention_reduction)
        elif use_attention == 'cbam':
            self.pre_attention = CBAM(rnn_input_channels, reduction=attention_reduction)
        elif use_attention == 'self':
            self.pre_attention = SelfAttention2D(rnn_input_channels, reduction=8)
        elif use_attention == 'transformer':
            self.pre_attention = TransformerBlock2D(rnn_input_channels, num_heads=8)
        else:
            self.pre_attention = None
            
        if rnn_config is None:
            self.rnn = None
        else:
            self.groups = rnn_config.get('groups', 1)
            hidden_dims = rnn_config['hidden_dims']

            if rnn_input_channels % self.groups:
                raise Exception(f'RNN input channels {rnn_input_channels}'
                                 ' is not divisible by groups!')
            if any([d % self.groups for d in hidden_dims]):
                raise Exception(f'Not all the hidden_dims are divisible by groups!')

            rnn_input_channels //= self.groups
            hidden_dims = [h//self.groups for h in hidden_dims]

            import convgru
            self.rnn = convgru.ConvGRU(input_size=rnn_config['input_size'],
                                       input_dim=rnn_input_channels,
                                       hidden_dim=hidden_dims,
                                       kernel_size=rnn_config.get('kernel_size', (3,3)),
                                       num_layers=len(hidden_dims),
                                       dtype=torch.cuda.FloatTensor,
                                       batch_first=True,
                                       bias=True,
                                       return_all_layers=True,
                                       noisy_pose=rnn_config.get('noisy_pose', False),
                                       cell_type=rnn_config.get('cell_type', 'standard'))
        
        # Add attention after RNN
        if use_attention and rnn_config is not None:
            output_channels = hidden_dims[-1] * self.groups if self.groups > 1 else hidden_dims[-1]
            if use_attention == 'se':
                self.post_attention = SEBlock(output_channels, reduction=attention_reduction)
            elif use_attention == 'cbam':
                self.post_attention = CBAM(output_channels, reduction=attention_reduction)
            elif use_attention == 'self':
                self.post_attention = SelfAttention2D(output_channels, reduction=8)
            elif use_attention == 'transformer':
                self.post_attention = TransformerBlock2D(output_channels, num_heads=8)
        else:
            self.post_attention = None

    def forward(self, x, t=1, bos=None, pose=None):
        if self.pre_rnn_conv is not None:
            x = self.pre_rnn_conv(x)
        
        # Apply pre-attention if enabled
        if self.pre_attention is not None:
            x = self.pre_attention(x)

        if self.rnn is not None:
            assert(bos is not None and pose is not None)

            ### reshape (bt, c, h, w) --> (b, t, c, h, w)
            bt, c, h, w = x.shape
            b = bt//t
            bos = bos.reshape(b, t)
            pose = pose.reshape(b, t, 3, 3)

            if self.groups > 1:
                bg = b * self.groups
                assert(c % self.groups == 0)
                x = x.reshape(b, t, self.groups, c//self.groups, h, w)
                x = x.transpose(1, 2)
                x = x.reshape(bg, t, c//self.groups, h , w)
                bos = bos.repeat(self.groups, 1)
                pose = pose.repeat(self.groups, 1 , 1, 1)
            else:
                x = x.reshape(b, t, c, h, w)

            assert(torch.all(torch.all(bos, axis=0) ^ torch.all(~bos, axis=0)))
            assert(torch.any(bos[0, 1:]) == False)

            if bos[0, 0]:
                self.hidden_state = None

            pose = pose[:, :, None].expand(-1, -1, self.rnn.num_layers, -1, -1)
            layer_output_list, last_state_list = self.rnn(x, pose,
                                                          hidden_state=self.hidden_state)

            self.hidden_state = []
            for state in last_state_list:
                assert(len(state) == 1)
                dstate = state[0].detach()
                dstate.requires_grad = True
                self.hidden_state.append(dstate)

            x = layer_output_list[-1]

            if self.groups > 1:
                x = x.reshape(b, self.groups, t, c//self.groups, h, w)
                x = x.transpose(1, 2)

            x = x.reshape(bt, c, h, w)
        
        # Apply post-attention if enabled
        if self.post_attention is not None:
            x = self.post_attention(x)

        return x