import numpy as np
from bevnet.attention_modules import SEBlock, CBAM, SelfAttention2D, TransformerBlock2D
from bevnet.fchardnet import HardNet1024Skip, ConvLayer, HarDBlock
import functools
import torch
import torch.nn as nn
import torch.nn.functional as F
import spconv
from torchvision.models.resnet import resnet18
import fchardnet
import convgru


class SpMiddleNoDownsampleXYWithSE(nn.Module):
    """SpMiddleNoDownsampleXY with SE attention"""
    def __init__(self,
                 output_shape,
                 num_input_features):
        super(SpMiddleNoDownsampleXYWithSE, self).__init__()
        
        # 复制原始SpMiddleNoDownsampleXY的所有初始化代码
        BatchNorm1d = functools.partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)
        SpConv3d = functools.partial(spconv.SparseConv3d, bias=False)
        SubMConv3d = functools.partial(spconv.SubMConv3d, bias=False)

        sparse_shape = np.array(output_shape[1:4]) + [1, 0, 0]
        self.sparse_shape = sparse_shape
        self.voxel_output_shape = output_shape
        
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
        
        # 添加SE注意力模块
        self.se = SEBlock(192, reduction=16)  # 64 * 3 = 192

    def forward(self, voxel_features, coors, batch_size):
        coors = coors.int()
        ret = spconv.SparseConvTensor(voxel_features, coors, self.sparse_shape, batch_size)
        ret = self.middle_conv(ret)
        ret = ret.dense()
        N, C, D, H, W = ret.shape
        ret = ret.view(N, C * D, H, W)
        
        # 应用SE注意力
        ret = self.se(ret)
        
        return ret


class InpaintingResNet18WithSE(nn.Module):
    """ResNet18 with SE attention blocks"""
    def __init__(self, num_input_features, num_class):
        super(InpaintingResNet18WithSE, self).__init__()

        trunk = resnet18(pretrained=False, zero_init_residual=True)
        self.conv1 = nn.Conv2d(
            num_input_features, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = trunk.bn1
        self.relu = trunk.relu

        self.layer1 = trunk.layer1
        self.layer2 = trunk.layer2
        self.layer3 = trunk.layer3


        self.se1 = SEBlock(64, reduction=16)
        self.se2 = SEBlock(128, reduction=16)
        self.se3 = SEBlock(256, reduction=16)

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
        x1 = self.se1(x1)  
        
        x = self.layer2(x1)
        x = self.se2(x)    
        
        x = self.layer3(x)
        x = self.se3(x)    

        x = self.up1(x, x1)
        x = self.up2(x)

        return dict(bev_preds=x)
    
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
class VoxelFeatureExtractorV3(nn.Module):
    def __init__(self):
        super(VoxelFeatureExtractorV3, self).__init__()

    def forward(self, features, num_voxels):
        # import ipdb; ipdb.set_trace()
        # features: [concated_num_points, num_voxel_size, n_dim]
        # num_voxels: [concated_num_points]
        points_mean = features.sum(
            dim=1, keepdim=False) / num_voxels.type_as(features).view(-1, 1)
        return points_mean.contiguous()


class VoxelFeatureExtractorV3MultiStep(nn.Module):
    def __init__(self):
        super(VoxelFeatureExtractorV3MultiStep, self).__init__()

    def forward(self, features, num_voxels):
        # features: T x [concated_num_points, num_voxel_size, 3(4)]
        # num_voxels: T x [concated_num_points]
        # returns list of T x [num_voxels]
        t = len(features)
        output = []
        for i in range(t):
            features_single = features[i]
            num_single = num_voxels[i]
            points_mean = features_single.sum(
                dim=1, keepdim=False) / num_single.type_as(features_single).view(-1, 1)
            output.append(points_mean.contiguous())
        return output


class SpMiddleNoDownsampleXY(nn.Module):
    """
    Only downsample z. Do not downsample X and Y.
    """
    def __init__(self,
                 output_shape,
                 num_input_features):
        super(SpMiddleNoDownsampleXY, self).__init__()

        BatchNorm1d = functools.partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)
        SpConv3d = functools.partial(spconv.SparseConv3d, bias=False)
        SubMConv3d = functools.partial(spconv.SubMConv3d, bias=False)

        sparse_shape = np.array(output_shape[1:4]) + [1, 0, 0]
        self.sparse_shape = sparse_shape
        self.voxel_output_shape = output_shape
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

    def forward(self, voxel_features, coors, batch_size):
        # import ipdb; ipdb.set_trace()
        coors = coors.int()
        ret = spconv.SparseConvTensor(voxel_features, coors, self.sparse_shape, batch_size)
        ret = self.middle_conv(ret)
        ret = ret.dense()
        N, C, D, H, W = ret.shape
        ret = ret.view(N, C * D, H, W)
        return ret


class SpMiddleNoDownsampleXYMultiStep(SpMiddleNoDownsampleXY):
    """
    No gradients!
    """
    def __init__(self, *args, **kwargs):
        super(SpMiddleNoDownsampleXYMultiStep, self).__init__(*args, **kwargs)

    def forward(self, voxel_features, coors, batch_size):
        self.eval()
        with torch.no_grad():
            t = len(voxel_features)
            output = []
            for i in range(t):
                voxel_features_i = voxel_features[i]
                coors_i = coors[i]
                coors_i = coors_i.int()
                # import ipdb; ipdb.set_trace()
                ret = spconv.SparseConvTensor(voxel_features_i, coors_i, self.sparse_shape, batch_size)
                ret = self.middle_conv(ret)
                ret = ret.dense()
                N, C, D, H, W = ret.shape
                ret = ret.view(N, C * D, H, W)
                output.append(ret.detach())
            return output


class SpMiddleNoDownsampleXYNoExpand(nn.Module):
    """
    Only downsample z. Do not downsample X and Y.
    """
    def __init__(self,
                 output_shape,
                 num_input_features):
        super(SpMiddleNoDownsampleXYNoExpand, self).__init__()

        BatchNorm1d = functools.partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)
        SubMConv3d = functools.partial(spconv.SubMConv3d, bias=False)

        sparse_shape = np.array(output_shape[1:4]) + [1, 0, 0]
        self.sparse_shape = sparse_shape
        self.voxel_output_shape = output_shape
        self.middle_conv = spconv.SparseSequential(
            SubMConv3d(num_input_features, 32, 3, indice_key="subm0"),
            BatchNorm1d(32),
            nn.ReLU(),
            SubMConv3d(32, 64, 3, indice_key="subm0"),
            BatchNorm1d(64),
            nn.ReLU(),
            # SubMConv3d(32, 64, 3, (2, 1, 1), padding=[1, 1, 1]),
            spconv.SparseMaxPool3d(3, (2, 1, 1), padding=[1, 1, 1]),
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
            # SubMConv3d(64, 64, 3, (2, 1, 1), padding=[0, 1, 1]),
            spconv.SparseMaxPool3d(3, (2, 1, 1), padding=[0, 1, 1]),
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
            # SubMConv3d(64, 64, (3, 1, 1), (2, 1, 1)),
            spconv.SparseMaxPool3d((3, 1, 1), (2, 1, 1)),
            BatchNorm1d(64),
            nn.ReLU(),
        )

    def forward(self, voxel_features, coors, batch_size):
        coors = coors.int()
        ret = spconv.SparseConvTensor(voxel_features, coors, self.sparse_shape, batch_size)
        ret = self.middle_conv(ret)
        ret = ret.dense()
        N, C, D, H, W = ret.shape
        ret = ret.view(N, C * D, H, W)
        return ret


class SpMiddleNoDownsampleXYNoExpandMultiStep(SpMiddleNoDownsampleXYNoExpand):
    """
    No gradients!
    """
    def __init__(self, *args, **kwargs):
        super(SpMiddleNoDownsampleXYNoExpandMultiStep, self).__init__(*args, **kwargs)

    def forward(self, voxel_features, coors, batch_size):
        self.eval()
        with torch.no_grad():
            t = len(voxel_features)
            output = []
            for i in range(t):
                voxel_features_i = voxel_features[i]
                coors_i = coors[i]
                coors_i = coors_i.int()
                ret = spconv.SparseConvTensor(voxel_features_i, coors_i, self.sparse_shape, batch_size)
                ret = self.middle_conv(ret)
                ret = ret.dense()
                N, C, D, H, W = ret.shape
                ret = ret.view(N, C * D, H, W)
                output.append(ret.detach())
            return output


class MiddleNoDownsampleXY(nn.Module):
    """
    Only downsample z. Do not downsample X and Y.
    """
    def __init__(self, output_shape, num_input_features):
        super(MiddleNoDownsampleXY, self).__init__()
        Conv3d = functools.partial(nn.Conv3d, bias=True)
        sparse_shape = np.array(output_shape[1:4]) + [1, 0, 0]
        self.sparse_shape = sparse_shape
        self.voxel_output_shape = output_shape
        self.middle_conv = nn.Sequential(
            Conv3d(num_input_features, 32, 3, padding=1),
            nn.ReLU(),
            Conv3d(32, 64, 3, (2, 1, 1), padding=1),  # Downsample z
            nn.ReLU(),
            Conv3d(64, 64, 3, stride=(2, 1, 1), padding=[0, 1, 1]),  # Downsample z
            nn.ReLU(),
            Conv3d(64, 64, kernel_size=(3, 1, 1), stride=(2, 1, 1)),  # Downsample z
            nn.ReLU(),
        )

    def forward(self, voxel_features, coors, batch_size):
        inputs = spconv.SparseConvTensor(voxel_features, coors, self.sparse_shape, batch_size).dense()
        ret = self.middle_conv(inputs)
        N, C, D, H, W = ret.shape
        ret = ret.view(N, C * D, H, W)
        return ret


class InpaintingFCHardnetRecurrentBase(object):
    def __init__(self,
                 aggregation_type='pre',
                 gru_input_size=(256, 256),
                 gru_input_dim=448,
                 gru_hidden_dims=[448],
                 gru_cell_type='standard',
                 noisy_pose=False, **kwargs):
        super(InpaintingFCHardnetRecurrentBase, self).__init__(**kwargs)

        assert aggregation_type in ['pre', 'post', 'none'], aggregation_type
        self.aggregation_type = aggregation_type

        if aggregation_type != 'none':
            ### Amirreza: GRU parameters are hardcoded for now
            self.gru = convgru.ConvGRU(input_size=gru_input_size,
                                       input_dim=gru_input_dim,
                                       hidden_dim=gru_hidden_dims,
                                       kernel_size=(3, 3),
                                       num_layers=len(gru_hidden_dims),
                                       dtype=torch.cuda.FloatTensor,
                                       batch_first=True,
                                       bias=True,
                                       return_all_layers=True,
                                       noisy_pose=noisy_pose,
                                       cell_type=gru_cell_type)

            def get_poses(input_pose):
                # convert to matrix
                mat = torch.zeros(input_pose.shape[0], # batch_size
                                  input_pose.shape[1], # t
                                  3, 3, dtype=input_pose.dtype,
                                  device=input_pose.device)

                mat[:, :, 0] = input_pose[:, :, :3]
                mat[:, :, 1] = input_pose[:, :, 3:6]
                mat[:, :, 2, 2] = 1.0

                # We are using two GRU cells with the same poses
                return mat[:, :, None]

            self.get_poses = get_poses

    def forward(self, x, seq_start=None, input_pose=None):
        n, c, h, w = x[0].shape
        t = len(x)

        if isinstance(x, list):
            x = torch.cat(x, dim=0)
        elif isinstance(x, torch.Tensor):
            x = x.view((-1,) + x.size()[2:])  # Fuse dim 0 and 1

        if self.aggregation_type != 'none':
            if seq_start is None:
                self.hidden_state = None
            else:
                # sanity check: only the first index can be True
                assert(torch.any(seq_start[1:]) == False)

                if seq_start[0]:  # start of a new sequence
                    self.hidden_state = None
        if self.aggregation_type == 'pre':
            layer_output_list, last_state_list = self.gru(x[None],
                                                          self.get_poses(input_pose[None]),
                                                          hidden_state=self.hidden_state)
            x = layer_output_list[-1].squeeze(0)

        out = self.fchardnet(x)

        if self.aggregation_type == 'post':
            layer_output_list, last_state_list = self.gru(out[None],
                                                          self.get_poses(input_pose[None]),
                                                          hidden_state=self.hidden_state)
            out = layer_output_list[-1].squeeze(0)

        if self.aggregation_type != 'none':
            self.hidden_state = []
            for state in last_state_list:
                dstate = state[0].detach()
                dstate.requires_grad = True
                self.hidden_state.append(dstate)

        num_class = out.shape[1]
        out = out.reshape((t, n, num_class, h, w))
        ret_dict = {
            "bev_preds": out,
        }
        return ret_dict


class InpaintingFCHardNetSkip1024(nn.Module):
    def __init__(self,
                 num_class=2,
                 num_input_features=128):
        super(InpaintingFCHardNetSkip1024, self).__init__()
        self.fchardnet = fchardnet.HardNet1024Skip(num_input_features, num_class)

    def forward(self, x, *args, **kwargs):
        # import ipdb; ipdb.set_trace()
        out = self.fchardnet(x)
        ret_dict = {
            "bev_preds": out,
        }
        return ret_dict
class InpaintingFCHardNetSkipGRU512(InpaintingFCHardnetRecurrentBase, InpaintingFCHardNetSkip1024):
    pass

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


class HarDBlockWithSE(nn.Module):
    """HarDBlock with SE attention after concatenation"""
    def __init__(self, in_channels, growth_rate, grmul, n_layers, 
                 keepBase=False, residual_out=False, bn=False):
        super().__init__()
        self.hardblock = HarDBlock(in_channels, growth_rate, grmul, n_layers, 
                                  keepBase, residual_out, bn)
        self.se = SEBlock(self.hardblock.get_out_ch())
        
    def get_out_ch(self):
        return self.hardblock.get_out_ch()
    
    def forward(self, x):
        out = self.hardblock(x)
        out = self.se(out)
        return out


class InpaintingFCHardNetSkip1024WithSE(nn.Module):
    """HardNet with SE blocks integrated"""
    def __init__(self, num_input_features, num_class):
        super(InpaintingFCHardNetSkip1024WithSE, self).__init__()
        
        ch_list = [32, 64, 96, 128, 160]
        grmul = 1.7
        gr = [10, 16, 18, 24, 32]
        n_layers = [4, 4, 8, 8, 8]
        
        blks = len(n_layers)
        self.shortcut_layers = []
        
        self.base = nn.ModuleList([])
        self.predownsample_ch = 32
        self.base.append(ConvLayer(in_channels=num_input_features,
                                  out_channels=self.predownsample_ch, 
                                  kernel=3, stride=2))
        
        skip_connection_channel_counts = []
        ch = 32
        for i in range(blks):
            # 使用带SE的HarDBlock
            if i >= 2:  # 只在后面的块中添加SE（可调整）
                blk = HarDBlockWithSE(ch, gr[i], grmul, n_layers[i])
            else:
                blk = HarDBlock(ch, gr[i], grmul, n_layers[i])
            
            ch = blk.get_out_ch()
            skip_connection_channel_counts.append(ch)
            self.base.append(blk)
            
            if i < blks - 1:
                self.shortcut_layers.append(len(self.base) - 1)
            
            self.base.append(ConvLayer(ch, ch_list[i], kernel=1))
            ch = ch_list[i]
            
            if i < blks - 1:
                self.base.append(nn.AvgPool2d(kernel_size=2, stride=2))
        
        cur_channels_count = ch
        prev_block_channels = ch
        n_blocks = blks - 1
        self.n_blocks = n_blocks
        
        # 上采样路径
        self.transUpBlocks = nn.ModuleList([])
        self.denseBlocksUp = nn.ModuleList([])
        self.conv1x1_up = nn.ModuleList([])
        
        for i in range(n_blocks - 1, -1, -1):
            from bevnet.fchardnet import TransitionUp
            self.transUpBlocks.append(TransitionUp(prev_block_channels, prev_block_channels))
            cur_channels_count = prev_block_channels + skip_connection_channel_counts[i]
            
            # 在上采样路径也添加SE
            conv_with_se = nn.Sequential(
                ConvLayer(cur_channels_count, cur_channels_count // 2, kernel=1),
                SEBlock(cur_channels_count // 2) if i < 2 else nn.Identity()
            )
            self.conv1x1_up.append(conv_with_se)
            cur_channels_count = cur_channels_count // 2
            
            blk = HarDBlock(cur_channels_count, gr[i], grmul, n_layers[i])
            self.denseBlocksUp.append(blk)
            prev_block_channels = blk.get_out_ch()
            cur_channels_count = prev_block_channels
        
        # 最终上采样
        self.final_upsample = nn.ConvTranspose2d(cur_channels_count, cur_channels_count, 3,
                                                stride=2, padding=1)
        
        # 最终卷积前添加SE
        self.pre_final_se = SEBlock(cur_channels_count + num_input_features)
        self.finalConv = nn.Conv2d(in_channels=cur_channels_count + num_input_features,
                                  out_channels=num_class, kernel_size=1, stride=1,
                                  padding=0, bias=True)
    
    def forward(self, x):
        skip_connections = []
        size_in = x.size()
        inputs = x
        
        # 编码器路径
        for i in range(len(self.base)):
            x = self.base[i](x)
            if i in self.shortcut_layers:
                skip_connections.append(x)
        out = x
        
        # 解码器路径
        for i in range(self.n_blocks):
            skip = skip_connections.pop()
            out = self.transUpBlocks[i](out, skip, True)
            out = self.conv1x1_up[i](out)
            out = self.denseBlocksUp[i](out)
        
        out = torch.relu(self.final_upsample(out, output_size=(out.size(0), out.size(1)) + size_in[2:4]))
        out = torch.cat([inputs, out], dim=1)
        
        # 应用最终SE块
        out = self.pre_final_se(out)
        out = self.finalConv(out)
        
        return dict(bev_preds=out)

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
        batch_size, C, H, W = x.size()
        
        # 为了减少计算量，可以先下采样
        if H * W > 4096:  # 如果特征图太大，先进行下采样
            x_down = F.adaptive_avg_pool2d(x, (H//2, W//2))
            proj_query = self.query_conv(x_down).view(batch_size, -1, (H//2) * (W//2)).permute(0, 2, 1)
            proj_key = self.key_conv(x_down).view(batch_size, -1, (H//2) * (W//2))
            proj_value = self.value_conv(x_down).view(batch_size, -1, (H//2) * (W//2))
        else:
            proj_query = self.query_conv(x).view(batch_size, -1, H * W).permute(0, 2, 1)
            proj_key = self.key_conv(x).view(batch_size, -1, H * W)
            proj_value = self.value_conv(x).view(batch_size, -1, H * W)
        
        attention = torch.bmm(proj_query, proj_key)
        attention = self.softmax(attention)
        
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        
        if H * W > 4096:
            out = out.view(batch_size, C, H//2, W//2)
            out = F.interpolate(out, size=(H, W), mode='bilinear', align_corners=False)
        else:
            out = out.view(batch_size, C, H, W)
        
        out = self.gamma * out + x
        return out


class InpaintingFCHardNetSkip1024WithTransformer(nn.Module):
    """HardNet with Transformer attention blocks integrated at strategic positions"""
    def __init__(self, num_input_features, num_class, 
                 num_heads=8, mlp_ratio=4.0, drop_path=0.1,
                 attention_positions=['after_layer2', 'after_layer3']):
        super(InpaintingFCHardNetSkip1024WithTransformer, self).__init__()
        
        ch_list = [32, 64, 96, 128, 160]
        grmul = 1.7
        gr = [10, 16, 18, 24, 32]
        n_layers = [4, 4, 8, 8, 8]
        
        blks = len(n_layers)
        self.shortcut_layers = []
        self.attention_positions = attention_positions
        
        self.base = nn.ModuleList([])
        self.predownsample_ch = 32
        self.base.append(ConvLayer(in_channels=num_input_features,
                                  out_channels=self.predownsample_ch, 
                                  kernel=3, stride=2))
        
        skip_connection_channel_counts = []
        ch = 32
        
        # Encoder path with transformer blocks
        for i in range(blks):
            blk = HarDBlock(ch, gr[i], grmul, n_layers[i])
            ch = blk.get_out_ch()
            skip_connection_channel_counts.append(ch)
            self.base.append(blk)
            
            if i < blks - 1:
                self.shortcut_layers.append(len(self.base) - 1)
            
            self.base.append(ConvLayer(ch, ch_list[i], kernel=1))
            ch = ch_list[i]
            
            # Add transformer attention after specific layers
            if f'after_layer{i}' in attention_positions and i < blks - 1:
                # 计算合适的注意力头数
                suitable_heads = self._get_suitable_heads(ch, num_heads)
                self.base.append(TransformerBlock2D(
                    dim=ch, 
                    num_heads=suitable_heads,
                    mlp_ratio=mlp_ratio,
                    drop_path=drop_path
                ))
            
            if i < blks - 1:
                self.base.append(nn.AvgPool2d(kernel_size=2, stride=2))
        
        cur_channels_count = ch
        prev_block_channels = ch
        n_blocks = blks - 1
        self.n_blocks = n_blocks
        
        # Decoder path
        self.transUpBlocks = nn.ModuleList([])
        self.denseBlocksUp = nn.ModuleList([])
        self.conv1x1_up = nn.ModuleList([])
        self.up_transformers = nn.ModuleList([])
        
        for i in range(n_blocks - 1, -1, -1):
            from bevnet.fchardnet import TransitionUp
            self.transUpBlocks.append(TransitionUp(prev_block_channels, prev_block_channels))
            cur_channels_count = prev_block_channels + skip_connection_channel_counts[i]
            self.conv1x1_up.append(ConvLayer(cur_channels_count, cur_channels_count // 2, kernel=1))
            cur_channels_count = cur_channels_count // 2
            
            # Add transformer in decoder if specified
            if f'decoder_layer{i}' in attention_positions:
                suitable_heads = self._get_suitable_heads(cur_channels_count, num_heads)
                self.up_transformers.append(TransformerBlock2D(
                    dim=cur_channels_count,
                    num_heads=suitable_heads,
                    mlp_ratio=mlp_ratio,
                    drop_path=drop_path
                ))
            else:
                self.up_transformers.append(nn.Identity())
            
            blk = HarDBlock(cur_channels_count, gr[i], grmul, n_layers[i])
            self.denseBlocksUp.append(blk)
            prev_block_channels = blk.get_out_ch()
            cur_channels_count = prev_block_channels
        
        # Final upsampling
        self.final_upsample = nn.ConvTranspose2d(cur_channels_count, cur_channels_count, 3,
                                                stride=2, padding=1)
        
        # Optional: Add final transformer before output
        if 'before_output' in attention_positions:
            final_ch = cur_channels_count + num_input_features
            suitable_heads = self._get_suitable_heads(final_ch, num_heads)
            self.final_transformer = TransformerBlock2D(
                dim=final_ch,
                num_heads=suitable_heads,
                mlp_ratio=mlp_ratio,
                drop_path=drop_path
            )
        else:
            self.final_transformer = nn.Identity()
            
        self.finalConv = nn.Conv2d(in_channels=cur_channels_count + num_input_features,
                                  out_channels=num_class, kernel_size=1, stride=1,
                                  padding=0, bias=True)
    
    def _get_suitable_heads(self, channels, desired_heads):
        """找到能整除通道数的最接近期望值的注意力头数"""
        for heads in [desired_heads, 8, 6, 4, 2, 1]:
            if channels % heads == 0:
                return heads
        return 1
    
    def forward(self, x):
        skip_connections = []
        size_in = x.size()
        inputs = x
        
        # Encoder path
        for i in range(len(self.base)):
            x = self.base[i](x)
            if i in self.shortcut_layers:
                skip_connections.append(x)
        out = x
        
        # Decoder path
        for i in range(self.n_blocks):
            skip = skip_connections.pop()
            out = self.transUpBlocks[i](out, skip, True)
            out = self.conv1x1_up[i](out)
            out = self.up_transformers[i](out)  # Apply transformer if present
            out = self.denseBlocksUp[i](out)
        
        out = F.relu(self.final_upsample(out, output_size=(out.size(0), out.size(1)) + size_in[2:4]))
        out = torch.cat([inputs, out], dim=1)
        
        # Apply final transformer if specified
        out = self.final_transformer(out)
        out = self.finalConv(out)
        
        return dict(bev_preds=out)


class TransformerBlock2D(nn.Module):
    """Efficient Transformer block for 2D feature maps with spatial downsampling"""
    def __init__(self, dim, num_heads=8, mlp_ratio=4., qkv_bias=False, 
                 drop=0., attn_drop=0., drop_path=0., 
                 downsample_ratio=4):  # Spatial downsampling for efficiency
        super(TransformerBlock2D, self).__init__()
        
        # 确保num_heads能整除dim
        assert dim % num_heads == 0, f"dim {dim} must be divisible by num_heads {num_heads}"
        
        self.downsample_ratio = downsample_ratio
        
        # Efficient spatial downsampling for attention computation
        if downsample_ratio > 1:
            self.down_conv = nn.Conv2d(dim, dim, kernel_size=downsample_ratio, 
                                      stride=downsample_ratio, groups=dim)
        else:
            self.down_conv = nn.Identity()
            
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadAttention2D(dim, num_heads, qkv_bias, attn_drop, drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)
        
        # Learnable upsampling parameters
        if downsample_ratio > 1:
            self.up_proj = nn.Conv2d(dim, dim, kernel_size=1)

    def forward(self, x):
        # x shape: (B, C, H, W)
        B, C, H, W = x.shape
        identity = x
        
        # Downsample for attention computation
        if self.downsample_ratio > 1:
            x_down = self.down_conv(x)
            _, _, H_d, W_d = x_down.shape
            x_flat = x_down.flatten(2).transpose(1, 2)  # (B, H_d*W_d, C)
        else:
            x_flat = x.flatten(2).transpose(1, 2)  # (B, H*W, C)
            H_d, W_d = H, W
        
        # Self-attention on downsampled features
        x_flat = x_flat + self.drop_path(self.attn(self.norm1(x_flat)))
        
        # MLP
        x_flat = x_flat + self.drop_path(self.mlp(self.norm2(x_flat)))
        
        # Reshape back
        x_down = x_flat.transpose(1, 2).reshape(B, C, H_d, W_d)
        
        # Upsample and combine with identity
        if self.downsample_ratio > 1:
            x_up = F.interpolate(x_down, size=(H, W), mode='bilinear', align_corners=False)
            x_up = self.up_proj(x_up)
            out = identity + x_up
        else:
            out = x_down
            
        return out


class MultiHeadAttention2D(nn.Module):
    """Efficient multi-head attention for 2D feature maps"""
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} must be divisible by num_heads {num_heads}"
        
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
    """Feed-forward network in transformer"""
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
    """Drop paths (Stochastic Depth) per sample"""
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