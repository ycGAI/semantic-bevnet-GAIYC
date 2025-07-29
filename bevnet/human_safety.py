import numpy as np
from sklearn.cluster import DBSCAN

class SimpleHumanDetector:
    def __init__(self, config=None):
        """
        初始化人员检测器
        
        Args:
            config: 检测配置字典
        """
        # 默认参数
        default_config = {
            'enabled': True,
            'height_range': (1.2, 2.2),      # 人体高度范围（米）
            'width_range': (0.3, 1.0),       # 人体宽度范围（米）
            'min_points': 50,                # 最少点数
            'ground_threshold': 0.3,         # 地面高度阈值（米）
            'clustering_eps': 0.5,           # DBSCAN聚类半径（米）
            'clustering_min_samples': 10,    # DBSCAN最小样本数
        }
        
        # 合并用户配置
        if config:
            default_config.update(config)
        
        self.config = default_config
        self.enabled = self.config.get('enabled', True)
        
    def detect_humans(self, points):
        """
        从点云中检测人员
        
        Args:
            points: Nx4 数组 (x, y, z, intensity)
            
        Returns:
            human_positions: 检测到的人的位置列表 [(x,y), ...]
        """
        if not self.enabled or len(points) == 0:
            return []
            
        # 1. 简单地面去除 - 保留高度在阈值以上的点
        above_ground = points[points[:, 2] > self.config['ground_threshold']]
        
        if len(above_ground) < self.config['min_points']:
            return []
            
        # 2. DBSCAN聚类
        try:
            clustering = DBSCAN(
                eps=self.config['clustering_eps'], 
                min_samples=self.config['clustering_min_samples']
            ).fit(above_ground[:, :3])
            
            labels = clustering.labels_
        except Exception as e:
            print(f"Clustering failed: {e}")
            return []
        
        # 3. 检查每个聚类是否符合人体特征
        human_positions = []
        unique_labels = set(labels)
        
        for label in unique_labels:
            if label == -1:  # 跳过噪声点
                continue
                
            # 获取属于当前聚类的点
            cluster_mask = labels == label
            cluster_points = above_ground[cluster_mask]
            
            if len(cluster_points) < self.config['min_points']:
                continue
            
            # 计算聚类的几何尺寸
            z_min, z_max = cluster_points[:, 2].min(), cluster_points[:, 2].max()
            x_min, x_max = cluster_points[:, 0].min(), cluster_points[:, 0].max()
            y_min, y_max = cluster_points[:, 1].min(), cluster_points[:, 1].max()
            
            height = z_max - z_min
            width_x = x_max - x_min
            width_y = y_max - y_min
            max_width = max(width_x, width_y)
            
            # 判断是否符合人体尺寸
            height_range = self.config['height_range']
            width_range = self.config['width_range']
            
            if (height_range[0] <= height <= height_range[1] and
                width_range[0] <= max_width <= width_range[1]):
                
                # 计算聚类中心位置
                center_x = cluster_points[:, 0].mean()
                center_y = cluster_points[:, 1].mean()
                human_positions.append((center_x, center_y))
                
        return human_positions
    
    def set_params(self, **kwargs):
        """
        动态更新检测参数
        
        Args:
            **kwargs: 检测参数，如 height_range, width_range, min_points 等
        """
        for key, value in kwargs.items():
            if key in self.config:
                self.config[key] = value
                print(f'Updated detection parameter: {key} = {value}')
    
    def enable(self):
        """启用检测"""
        self.enabled = True
        
    def disable(self):
        """禁用检测"""
        self.enabled = False