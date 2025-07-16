import numpy as np
import torch
import yaml

from spconv.utils import VoxelGenerator
from bevnet import networks
from bevnet.utils import pprint_dict
from bevnet.human_safety import SimpleHumanDetector


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
        
# 在文件顶部添加导入
from bevnet.human_safety import SimpleHumanDetector

# 在文件末尾添加新类
class BEVNetSingleWithSafety(BEVNetSingle):
    """带人员安全检测后处理的BEVNet推理类"""
    
    def __init__(self, weights_file, device='cuda', human_detection_config=None):
        super(BEVNetSingleWithSafety, self).__init__(weights_file, device)
        
        # 初始化人员检测器
        if human_detection_config is None:
            # 尝试从模型配置中获取
            human_detection_config = self.g.get('human_detection', None) if hasattr(self, 'g') else None
        
        if human_detection_config and human_detection_config.get('enabled', False):
            self.human_detector = SimpleHumanDetector(human_detection_config)
            print('Human detection enabled with config:', human_detection_config)
        else:
            self.human_detector = None
            print('Human detection disabled')
    
    def predict(self, points):
        """
        预测并添加人员安全后处理
        
        Args:
            points: 输入点云 (N, 4) - (x, y, z, intensity)
            
        Returns:
            preds: 处理后的BEV预测结果
            human_positions: 检测到的人员位置列表
        """
        # 调用父类的预测方法
        preds = super(BEVNetSingleWithSafety, self).predict(points)
        
        # 初始化人员位置列表
        human_positions = []
        
        # 如果启用了人员检测，进行后处理
        if self.human_detector is not None:
            # 检测人员
            human_positions = self.human_detector.detect_humans(points)
            
            if human_positions:
                # 在BEV上标记人员
                preds = self._add_humans_to_bev(preds, human_positions)
                print(f'Detected {len(human_positions)} humans')
        
        # 可以选择返回预测结果和人员位置
        if hasattr(self, 'return_human_positions') and self.return_human_positions:
            return preds, human_positions
        else:
            return preds
    
    def _add_humans_to_bev(self, bev_pred, human_positions):
        """
        在BEV预测结果上标记人员位置
        
        Args:
            bev_pred: BEV预测张量 [C, H, W]
            human_positions: 人员位置列表 [(x, y), ...]
            
        Returns:
            修改后的BEV预测
        """
        # 获取BEV参数
        voxelizer_cfg = self.g.voxelizer
        pc_range = voxelizer_cfg['point_cloud_range']
        voxel_size = voxelizer_cfg['voxel_size']
        
        min_x, min_y = pc_range[0], pc_range[1]
        resolution_x = voxel_size[0]
        resolution_y = voxel_size[1]
        
        # 安全参数
        safety_radius = 1.5  # 米
        human_confidence = 10.0
        
        # 获取维度
        num_classes, h, w = bev_pred.shape
        
        # 创建人员掩码
        human_mask = torch.zeros((h, w), dtype=torch.bool, device=bev_pred.device)
        
        for (x, y) in human_positions:
            # 世界坐标转BEV像素坐标
            pixel_x = int((x - min_x) / resolution_x)
            pixel_y = int((y - min_y) / resolution_y)
            
            # 检查边界
            if 0 <= pixel_x < w and 0 <= pixel_y < h:
                # 标记圆形区域
                radius_pixels = int(safety_radius / resolution_x)
                
                for dx in range(-radius_pixels, radius_pixels + 1):
                    for dy in range(-radius_pixels, radius_pixels + 1):
                        px, py = pixel_x + dx, pixel_y + dy
                        if (0 <= px < w and 0 <= py < h and 
                            dx*dx + dy*dy <= radius_pixels*radius_pixels):
                            human_mask[py, px] = True
        
        # 修改BEV预测
        # 根据类别数选择策略
        if num_classes >= 4:
            # 假设最后一个类别是障碍物/高成本
            obstacle_class = num_classes - 1
            bev_pred[obstacle_class, human_mask] = human_confidence
            # 降低其他类别的置信度
            for c in range(obstacle_class):
                bev_pred[c, human_mask] *= 0.1
        else:
            # 对于二分类或三分类，使用最后一个类别
            obstacle_class = num_classes - 1
            bev_pred[obstacle_class, human_mask] = human_confidence
            for c in range(obstacle_class):
                bev_pred[c, human_mask] *= 0.1
        
        return bev_pred
    
    def set_safety_params(self, safety_radius=1.5, human_confidence=10.0):
        """
        设置安全参数
        
        Args:
            safety_radius: 人员周围的安全半径（米）
            human_confidence: 人员检测的置信度
        """
        self.safety_radius = safety_radius
        self.human_confidence = human_confidence


# 为了向后兼容，也可以创建一个工厂函数
def create_bevnet_model(weights_file, device='cuda', with_safety=False, human_detection_config=None):
    """
    创建BEVNet模型
    
    Args:
        weights_file: 模型权重文件路径
        device: 计算设备
        with_safety: 是否启用人员安全检测
        human_detection_config: 人员检测配置
        
    Returns:
        BEVNet模型实例
    """
    if with_safety:
        return BEVNetSingleWithSafety(weights_file, device, human_detection_config)
    else:
        return BEVNetSingle(weights_file, device)


if __name__ == '__main__':
    model = BEVNetSingle('../experiments/kitti4_100/single/include_unknown/default-logs/model.pth.4')
    scan = np.fromfile('../data/semantic_kitti_4class_100x100/sequences/valid/velodyne/00000.bin', dtype=np.float32)
    scan = scan.reshape(-1, 4)
    pred = model.predict(scan)
    print(pred.size())
