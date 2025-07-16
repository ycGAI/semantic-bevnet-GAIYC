import numpy as np
import torch
from sklearn.cluster import DBSCAN

class SimpleHumanDetector:
    def __init__(self, config=None):
        # 默认参数
        default_config = {
            'height_range': (1.2, 2.2),
            'width_range': (0.3, 1.0),
            'min_points': 50,
            'ground_threshold': 0.3,
            'clustering_eps': 0.5,
            'clustering_min_samples': 10
        }
        
        self.config = config if config else default_config
        
    def detect_humans(self, points):
        """
        输入: points - Nx4 数组 (x, y, z, intensity)
        输出: human_positions - 检测到的人的位置列表 [(x,y), ...]
        """
        if len(points) == 0:
            return []
            
        # 1. 简单地面去除
        above_ground = points[points[:, 2] > self.config['ground_threshold']]
        
        if len(above_ground) < self.config['min_points']:
            return []
            
        # 2. 聚类
        clustering = DBSCAN(
            eps=self.config['clustering_eps'], 
            min_samples=self.config['clustering_min_samples']
        ).fit(above_ground[:, :3])
        
        labels = clustering.labels_
        
        # 3. 检查每个聚类
        human_positions = []
        for label in set(labels):
            if label == -1:  # 噪声点
                continue
                
            cluster_points = above_ground[labels == label]
            
            # 计算聚类尺寸
            height = cluster_points[:, 2].max() - cluster_points[:, 2].min()
            width_x = cluster_points[:, 0].max() - cluster_points[:, 0].min()
            width_y = cluster_points[:, 1].max() - cluster_points[:, 1].min()
            max_width = max(width_x, width_y)
            
            # 判断是否为人
            height_range = self.config['height_range']
            width_range = self.config['width_range']
            
            if (height_range[0] <= height <= height_range[1] and
                width_range[0] <= max_width <= width_range[1] and
                len(cluster_points) >= self.config['min_points']):
                
                # 记录中心位置
                center_x = cluster_points[:, 0].mean()
                center_y = cluster_points[:, 1].mean()
                human_positions.append((center_x, center_y))
                
        return human_positions