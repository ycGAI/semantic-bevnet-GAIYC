import numpy as np
import torch
import yaml

from spconv.utils import VoxelGenerator
from bevnet import networks
from bevnet.utils import pprint_dict
from bevnet.human_safety import SimpleHumanDetector
from bevnet.train_fixture_utils import make_label_vis, get_colormap


def make_nets(config, device):
    ret = {}
    for net_name, spec in config.items():
        net_class = getattr(networks, spec['class'])
        net_args = spec.get('net_kwargs', {})
        net = net_class(**net_args).to(device)
        ret[net_name] = net
    return ret


class BEVNetBase(object):
    from functools import partial
    import spconv
    print('patch spconv to increase the allowable z range. This will not affect the point cloud range.')
    spconv.utils.points_to_voxel = partial(spconv.utils.points_to_voxel,
                                           height_threshold=-4.0,
                                           height_high_threshold=5.0)

    def __init__(self, weights_file, device='cuda'):
        self.weights_file = weights_file
        if weights_file:
            self._load(weights_file, device)

        self.device = device
        self.h = None

    def _load(self, weights_file, device):
        state_dict = torch.load(weights_file, map_location='cpu')
        print('loaded %s' % weights_file)
        g = state_dict.get('global_args', {})
        print('global args:')
        print(pprint_dict(g))

        self.g = g
        self.weights_file = weights_file

        if isinstance(g.model_config, dict):
            nets = make_nets(g.model_config, device)
        else:
            nets = make_nets(yaml.load(open(g.model_config).read(),
                                       Loader=yaml.SafeLoader), device)

        for name, net in nets.items():
            net.load_state_dict(state_dict['nets'][name])
            net.train(False)
        self.nets = nets

        self.voxelizer = self._make_voxelizer(self.g.voxelizer)

    def _make_voxelizer(self, cfg):
        return VoxelGenerator(
            voxel_size=list(cfg['voxel_size']),
            point_cloud_range=list(cfg['point_cloud_range']),
            max_num_points=cfg['max_number_of_points_per_voxel'],
            full_mean=cfg['full_mean'],
            max_voxels=cfg['max_voxels'])

    def _as_tensor(self, data, dtype=None):
        return torch.as_tensor(data, dtype=dtype).to(device=self.device, non_blocking=True)


class BEVNetSingle(BEVNetBase):
    def predict(self, points):
        with torch.no_grad():
            # import ipdb; ipdb.set_trace()
            points_with_idx = np.concatenate([
                points, np.arange(len(points))[:, None].astype(points.dtype)], axis=-1)
            voxels, coords, num_points = self.voxelizer.generate(points_with_idx, max_voxels=90000)
            voxel_point_idxs = voxels[:, :, -1].astype(np.int32)  # num_voxels x max_num_points_per_voxel
            voxels = voxels[:, :, :-1]

            # Insert the batch dim
            coords = np.pad(coords, ((0, 0), (1, 0)), mode='constant', constant_values=0)

            nets = self.nets

            voxels_th = self._as_tensor(voxels)
            num_points_th = self._as_tensor(num_points)
            coords_th = self._as_tensor(coords)

            voxel_features = nets['VoxelFeatureEncoder'](voxels_th, num_points_th)
            features = nets['MiddleSparseEncoder'](voxel_features, coords_th, 1)
            preds = nets['BEVClassifier'](features)['bev_preds']
            return preds


class BEVNetRecurrent(BEVNetBase):
    def __init__(self, *args, **kwargs):
        super(BEVNetRecurrent, self).__init__(*args, **kwargs)
        self.seq_start = None
        self.reset()

    def reset(self):
        self.seq_start = True

    def predict(self, points, pose):
        with torch.no_grad():
            # import ipdb; ipdb.set_trace()
            points_with_idx = np.concatenate([
                points, np.arange(len(points))[:, None].astype(points.dtype)], axis=-1)
            voxels, coords, num_points = self.voxelizer.generate(points_with_idx, max_voxels=90000)
            voxel_point_idxs = voxels[:, :, -1].astype(np.int32)  # num_voxels x max_num_points_per_voxel
            voxels = voxels[:, :, :-1]

            # Insert the batch dim
            coords = np.pad(coords, ((0, 0), (1, 0)), mode='constant', constant_values=0)

            nets = self.nets

            voxels_th = self._as_tensor(voxels)
            num_points_th = self._as_tensor(num_points)
            coords_th = self._as_tensor(coords)
            pose_th = self._as_tensor(pose)

            voxel_features = nets['VoxelFeatureEncoder']([voxels_th], [num_points_th])
            # print(voxel_features.size(), coords_th.size()); exit(0)
            features = nets['MiddleSparseEncoder'](voxel_features, [coords_th], 1)
            preds = nets['BEVClassifier'](features,
                                          seq_start=torch.tensor([self.seq_start]),
                                          input_pose=pose_th[None])['bev_preds']
            preds = preds.squeeze(1)

            self.seq_start = False

            return preds


class BEVNetSingleWithSafety(BEVNetSingle):
    """带人员安全检测后处理的BEVNet推理类"""
    
    def __init__(self, weights_file, device='cuda', human_detection_config=None, safety_config=None):
        """
        初始化带安全检测的BEVNet模型
        
        Args:
            weights_file: 模型权重文件路径
            device: 计算设备 ('cuda' 或 'cpu')
            human_detection_config: 人员检测配置字典（可选）
            safety_config: 安全区域配置字典（可选）
        """
        super(BEVNetSingleWithSafety, self).__init__(weights_file, device)
        
        # 初始化人员检测器
        self.human_detector = None
        
        # 默认安全配置
        self.safety_config = {
            'safety_radius': 1.5,        # 安全半径（米）
            'human_confidence': 10.0,    # 人员区域置信度
            'human_class': None,         # 人员类别索引（None表示使用最后一个类别）
        }
        
        # 更新安全配置
        if safety_config:
            self.safety_config.update(safety_config)
        
        # 初始化人员检测器
        if human_detection_config is None:
            # 尝试从模型配置中获取
            if hasattr(self, 'g') and self.g is not None:
                human_detection_config = self.g.get('human_detection', None)
        
        if human_detection_config and human_detection_config.get('enabled', False):
            self.human_detector = SimpleHumanDetector(human_detection_config)
            print('Human detection enabled')
        else:
            print('Human detection disabled')
    
    def predict(self, points, return_human_positions=False):
        """
        预测并添加人员安全后处理
        
        Args:
            points: 输入点云 (N, 4) - (x, y, z, intensity)
            return_human_positions: 是否返回检测到的人员位置
            
        Returns:
            如果 return_human_positions=True: (preds, human_positions)
            否则: preds
        """
        # 调用父类的预测方法获取BEV预测
        with torch.no_grad():
            preds = super(BEVNetSingleWithSafety, self).predict(points)
        
        # 初始化人员位置列表
        human_positions = []
        
        # 如果启用了人员检测，进行检测和后处理
        if self.human_detector is not None:
            try:
                # 检测人员
                human_positions = self.human_detector.detect_humans(points)
                
                if human_positions:
                    # 在BEV上标记人员
                    preds = self._add_humans_to_bev(preds, human_positions)
                    print(f'Detected {len(human_positions)} humans')
                        
            except Exception as e:
                print(f'Human detection failed: {e}')
        
        # 根据参数返回结果
        if return_human_positions:
            return preds, human_positions
        else:
            return preds
    
    def _add_humans_to_bev(self, bev_pred, human_positions):
        """
        在BEV预测结果上标记人员位置
        
        Args:
            bev_pred: BEV预测张量 [n, C, H, W] 其中n=1, C=5(类别数)
            human_positions: 人员位置列表 [(x, y), ...]
            
        Returns:
            修改后的BEV预测
        """
        if len(human_positions) == 0:
            return bev_pred
        
        # 获取BEV网格参数
        voxelizer_cfg = self.g.voxelizer
        pc_range = voxelizer_cfg['point_cloud_range']
        voxel_size = voxelizer_cfg['voxel_size']
        
        min_x, min_y = pc_range[0], pc_range[1]
        resolution_x = voxel_size[0]
        resolution_y = voxel_size[1]
        
        # 获取安全参数
        safety_radius = self.safety_config['safety_radius']
        human_confidence = self.safety_config['human_confidence']
        
        # 获取BEV维度
        n, num_classes, h, w = bev_pred.shape
        
        # 确定人员类别索引（障碍物类别）
        human_class = self.safety_config['human_class']
        if human_class is None:
            # 默认使用最后一个类别作为障碍物/高成本类别
            human_class = num_classes - 2  # 如果有5个类别，这将是索引3
        
        # 创建人员掩码（内圈）和缓冲区掩码（外圈）
        human_mask = torch.zeros((h, w), dtype=torch.bool, device=bev_pred.device)
        buffer_mask = torch.zeros((h, w), dtype=torch.bool, device=bev_pred.device)
        
        # 计算外圈半径（增加20%）
        buffer_radius = safety_radius * 1.8
        
        # 为每个检测到的人员创建掩码
        for (x, y) in human_positions:
            # 世界坐标转BEV像素坐标
            pixel_x = int((x - min_x) / resolution_x)
            pixel_y = int((y - min_y) / resolution_y)
            
            # 检查是否在有效范围内
            if 0 <= pixel_x < w and 0 <= pixel_y < h:
                # 计算安全半径对应的像素数
                radius_pixels = max(1, int(safety_radius / resolution_x))
                buffer_radius_pixels = max(1, int(buffer_radius / resolution_x))
                
                # 创建圆形安全区域和缓冲区
                for dx in range(-buffer_radius_pixels, buffer_radius_pixels + 1):
                    for dy in range(-buffer_radius_pixels, buffer_radius_pixels + 1):
                        px, py = pixel_x + dx, pixel_y + dy
                        
                        # 检查像素是否在边界内
                        if (0 <= px < w and 0 <= py < h):
                            dist_sq = dx * dx + dy * dy
                            
                            # 内圈（人员区域）
                            if dist_sq <= radius_pixels * radius_pixels:
                                human_mask[py, px] = True
                            # 外圈（缓冲区域）
                            elif dist_sq <= buffer_radius_pixels * buffer_radius_pixels:
                                buffer_mask[py, px] = True
        
        # 修改BEV预测
        # 复制原始预测以避免in-place修改
        bev_pred = bev_pred.clone()
        
        # 定义置信度阈值，用于判断类别3是否已经有高置信度
        high_confidence_threshold = 5.0  # 可以根据需要调整
        medium_confidence_boost = 15.0    # 类别2的置信度提升值
        
        # 在人员位置设置高置信度
        # 对于batch中的每个样本（通常n=1）
        for batch_idx in range(n):
            # 1. 处理内圈（人员区域）
            # 在障碍物类别通道上设置高置信度
            bev_pred[batch_idx, human_class, human_mask] = human_confidence
            
            # 降低其他类别的置信度
            for c in range(num_classes):
                if c != human_class:
                    bev_pred[batch_idx, c, human_mask] *= 0.1
            
            # 2. 处理外圈（缓冲区域）
            # 获取缓冲区域中类别3的置信度
            class3_confidence_in_buffer = bev_pred[batch_idx, 3, buffer_mask]
            
            # 创建一个掩码，标记缓冲区中类别3置信度不高的位置
            low_class3_mask = buffer_mask.clone()
            if class3_confidence_in_buffer.numel() > 0:
                # 找出缓冲区中类别3置信度较低的像素
                buffer_pixels = torch.where(buffer_mask)
                for i in range(len(buffer_pixels[0])):
                    y, x = buffer_pixels[0][i], buffer_pixels[1][i]
                    if bev_pred[batch_idx, 3, y, x] >= high_confidence_threshold:
                        low_class3_mask[y, x] = False
            
            # 在类别3置信度不高的缓冲区域，提高类别2的置信度
            if low_class3_mask.any():
                bev_pred[batch_idx, 2, low_class3_mask] += medium_confidence_boost
                
                # 可选：稍微降低其他类别（除了2和3）的置信度
                # for c in range(num_classes):
                #     if c not in [2, 3, human_class]:
                #         bev_pred[batch_idx, c, low_class3_mask] *= 0.7
        
        return bev_pred
    
    def set_safety_params(self, safety_radius=None, human_confidence=None, human_class=None):
        """
        动态设置安全参数
        
        Args:
            safety_radius: 人员周围的安全半径（米）
            human_confidence: 人员检测的置信度
            human_class: 人员类别索引
        """
        if safety_radius is not None:
            self.safety_config['safety_radius'] = safety_radius
        if human_confidence is not None:
            self.safety_config['human_confidence'] = human_confidence
        if human_class is not None:
            self.safety_config['human_class'] = human_class
            
        print(f'Updated safety parameters: {self.safety_config}')
    
    def set_detection_params(self, **kwargs):
        """
        动态更新检测参数
        
        Args:
            **kwargs: 检测参数，如 height_range, width_range, min_points 等
        """
        if self.human_detector is not None:
            self.human_detector.set_params(**kwargs)
        else:
            print('Human detector not initialized')
    
    def enable_human_detection(self, config=None):
        """动态启用人员检测"""
        if self.human_detector is None:
            if config is None:
                config = {'enabled': True}
            self.human_detector = SimpleHumanDetector(config)
        else:
            self.human_detector.enable()
        print('Human detection enabled')
    
    def disable_human_detection(self):
        """动态禁用人员检测"""
        if self.human_detector is not None:
            self.human_detector.disable()
        print('Human detection disabled')
    


# 工厂函数
def create_bevnet_model(weights_file, device='cuda', with_safety=False, 
                       human_detection_config=None, safety_config=None, model_type='single'):
    """
    创建BEVNet模型的工厂函数
    
    Args:
        weights_file: 模型权重文件路径
        device: 计算设备 ('cuda' 或 'cpu')
        with_safety: 是否启用人员安全检测
        human_detection_config: 人员检测配置字典（可选）
        safety_config: 安全区域配置字典（可选）
        model_type: 模型类型 ('single' 或 'recurrent')
        
    Returns:
        BEVNet模型实例
        
    Example:
        # 创建普通模型
        model = create_bevnet_model('model.pth')
        
        # 创建带安全检测的模型
        model = create_bevnet_model('model.pth', with_safety=True)
        
        # 创建自定义配置的安全模型
        human_config = {'enabled': True, 'min_points': 30}
        safety_config = {'safety_radius': 2.0, 'human_confidence': 15.0}
        model = create_bevnet_model('model.pth', with_safety=True, 
                                  human_detection_config=human_config,
                                  safety_config=safety_config)
    """
    if model_type == 'single':
        if with_safety:
            # 确保人员检测配置启用
            if human_detection_config is None:
                human_detection_config = {'enabled': True}
            else:
                human_detection_config['enabled'] = True
                
            return BEVNetSingleWithSafety(
                weights_file, 
                device, 
                human_detection_config,
                safety_config
            )
        else:
            return BEVNetSingle(weights_file, device)
            
    elif model_type == 'recurrent':
        if with_safety:
            raise NotImplementedError("BEVNetRecurrentWithSafety not implemented yet")
        else:
            return BEVNetRecurrent(weights_file, device)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

if __name__ == '__main__':
    import os
    import matplotlib
    matplotlib.use('Agg')  # 设置为非交互式后端
    import matplotlib.pyplot as plt
    import numpy as np
    
    # 创建带安全检测的模型
    human_detection_config = {
        'enabled': True,
        'height_range': (1.2, 2.2),
        'width_range': (0.3, 1.0),
        'min_points': 50,
        'ground_threshold': 0.3,
        'clustering_eps': 0.5,
        'clustering_min_samples': 10
    }
    
    # 配置安全参数
    safety_config = {
        'safety_radius': 1.5,
        'human_confidence': 10.0,
        'human_class': None  # 使用最后一个类别
    }
    
    # 创建带安全检测的模型
    model = BEVNetSingleWithSafety(
        '/workspace/bevnet/experiments/rellis4_100/single/default---batch_size=1-logs/best.pth.8',
        device='cuda',
        human_detection_config=human_detection_config,
        safety_config=safety_config
    )
    
    # 加载点云数据
    scan = np.fromfile('/workspace/data/rellis_3d/dataset/sequences/00000/velodyne/000000_with_humans.bin', dtype=np.float32)
    scan = scan.reshape(-1, 4)
    
    # 获取预测结果
    logits = model.predict(scan)
    
    # 可视化预测结果
    # logits shape: [1, num_classes, H, W]
    # 获取最可能的类别
    pred = torch.argmax(logits[0], dim=0)  # [H, W]
    
    # 获取colormap
    cmap = get_colormap(model.g.dataset_type)
    
    # 转换预测为可视化图像
    pred_vis = make_label_vis(pred.cpu().numpy(), cmap)
    
    # 创建图形
    plt.figure(figsize=(10, 10))
    plt.imshow(pred_vis)
    plt.title('BEV Prediction with Human Safety Detection', fontsize=16)
    plt.axis('off')
    
    # 保存图像
    save_path = os.path.join(os.getcwd(), 'bev_prediction_with_safety.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Visualization saved to: {save_path}")
    
    # 可选：打印一些统计信息
    unique_classes, counts = torch.unique(pred, return_counts=True)
    print("\nClass distribution:")
    for cls, count in zip(unique_classes.cpu().numpy(), counts.cpu().numpy()):
        print(f"  Class {cls}: {count} pixels ({count/pred.numel()*100:.2f}%)")