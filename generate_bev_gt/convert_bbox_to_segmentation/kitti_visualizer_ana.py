#!/usr/bin/env python3
"""
KITTI数据集可视化工具
支持点云、图像、3D标注框的可视化
"""

import os
import sys
import numpy as np

# 完全禁用OpenCV的GUI功能
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '0'
os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = ''

# 设置matplotlib后端，避免Qt冲突
import matplotlib
matplotlib.use('TkAgg')  # 使用Tkinter后端

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import argparse
from pathlib import Path

try:
    import open3d as o3d
    OPEN3D_AVAILABLE = True
except ImportError:
    OPEN3D_AVAILABLE = False
    print("Warning: Open3D not available. 3D visualization will be limited.")

# 完全禁用OpenCV避免Qt冲突
CV2_AVAILABLE = False
print("Note: OpenCV disabled to avoid Qt conflicts. Using matplotlib for image loading.")


class KITTIVisualizer:
    def __init__(self, data_path):
        self.data_path = Path(data_path)
        self.current_frame = 0
        
    def load_velodyne_points(self, frame_id):
        """加载Velodyne点云数据"""
        velo_file = self.data_path / 'velodyne' / f'{frame_id:06d}.bin'
        if not velo_file.exists():
            raise FileNotFoundError(f"Velodyne file not found: {velo_file}")
        
        points = np.fromfile(str(velo_file), dtype=np.float32)
        points = points.reshape(-1, 4)  # [x, y, z, intensity]
        return points
    
    def load_image(self, frame_id, camera='image_2'):
        """加载相机图像"""
        img_file = self.data_path / camera / f'{frame_id:06d}.png'
        if not img_file.exists():
            img_file = self.data_path / camera / f'{frame_id:06d}.jpg'
        
        if not img_file.exists():
            raise FileNotFoundError(f"Image file not found: {img_file}")
        
        # 使用matplotlib读取图像，避免OpenCV的Qt冲突
        image = plt.imread(str(img_file))
        return image
    
    def load_labels(self, frame_id):
        """加载标注数据（支持检测和分割标签）"""
        detection_labels = []
        
        # 尝试不同的标签文件格式
        possible_label_files = [
            self.data_path / 'label_2' / f'{frame_id:06d}.txt',   # 标准txt格式
            self.data_path / 'label_2' / f'{frame_id:06d}.label', # 你的数据集格式
        ]
        
        for label_file in possible_label_files:
            if label_file.exists():
                print(f"Loading labels from: {label_file}")
                try:
                    with open(label_file, 'r') as f:
                        for line in f.readlines():
                            parts = line.strip().split()
                            if len(parts) >= 15:
                                # 标准KITTI 3D检测格式
                                obj = {
                                    'type': parts[0],
                                    'truncated': float(parts[1]),
                                    'occluded': int(parts[2]),
                                    'alpha': float(parts[3]),
                                    'bbox': [float(x) for x in parts[4:8]],  # 2D bbox
                                    'dimensions': [float(x) for x in parts[8:11]],  # h, w, l
                                    'location': [float(x) for x in parts[11:14]],  # x, y, z
                                    'rotation_y': float(parts[14])
                                }
                                detection_labels.append(obj)
                                print(f"Loaded object: {obj['type']} at {obj['location']}")
                except Exception as e:
                    print(f"Error loading labels from {label_file}: {e}")
                break
        
        # 尝试加载点云分割标签（如果存在）
        seg_labels = self.load_segmentation_labels(frame_id)
        
        return detection_labels, seg_labels
    
    def load_segmentation_labels(self, frame_id):
        """加载点云分割标签"""
        print(f"Looking for segmentation labels for frame {frame_id}...")
        
        # 尝试不同的分割标签文件格式和路径
        possible_seg_files = [
            # SemanticKITTI格式 - 二进制
            self.data_path / 'labels' / f'{frame_id:06d}.label',
            self.data_path / 'segmentation' / f'{frame_id:06d}.label',
            # 二进制格式
            self.data_path / 'labels' / f'{frame_id:06d}.bin', 
            self.data_path / 'segmentation' / f'{frame_id:06d}.bin',
            # 文本格式 - 每行一个标签
            self.data_path / 'labels' / f'{frame_id:06d}.txt',
            self.data_path / 'segmentation' / f'{frame_id:06d}.txt',
            # 特殊命名格式
            self.data_path / 'label_2' / f'{frame_id:06d}_seg.txt',
            self.data_path / 'label_2' / f'{frame_id:06d}_seg.label',
            self.data_path / 'label_2' / f'{frame_id:06d}_labels.txt',
        ]
        
        for seg_file in possible_seg_files:
            print(f"Checking: {seg_file}")
            if seg_file.exists():
                print(f"Found potential segmentation file: {seg_file}")
                try:
                    if seg_file.suffix == '.label' and seg_file.parent.name != 'label_2':
                        # 只有不在label_2目录下的.label文件才尝试作为二进制分割标签
                        # SemanticKITTI格式：uint32标签
                        labels = np.fromfile(str(seg_file), dtype=np.uint32)
                        labels = labels & 0xFFFF  # 取低16位作为语义标签
                        print(f"Loaded {len(labels)} labels from binary .label file")
                        return labels
                    elif seg_file.suffix == '.bin':
                        # 尝试不同的数据类型
                        for dtype in [np.int32, np.uint32, np.int16, np.uint16]:
                            try:
                                labels = np.fromfile(str(seg_file), dtype=dtype)
                                if len(labels) > 0:
                                    print(f"Loaded {len(labels)} labels from .bin file (dtype: {dtype})")
                                    return labels
                            except:
                                continue
                    elif seg_file.suffix == '.txt' and 'seg' in seg_file.name:
                        # 文本格式分割标签
                        try:
                            # 尝试作为整数数组
                            labels = np.loadtxt(str(seg_file), dtype=np.int32)
                            if labels.size > 0:
                                if labels.ndim == 0:  # 单个值
                                    labels = np.array([labels])
                                print(f"Loaded {len(labels)} labels from segmentation .txt file")
                                return labels
                        except:
                            try:
                                # 尝试逐行读取
                                with open(seg_file, 'r') as f:
                                    lines = f.readlines()
                                labels = []
                                for line in lines:
                                    parts = line.strip().split()
                                    if parts:
                                        # 取第一个数字作为标签
                                        labels.append(int(parts[0]))
                                if labels:
                                    labels = np.array(labels)
                                    print(f"Loaded {len(labels)} labels from segmentation text file (line by line)")
                                    return labels
                            except Exception as e:
                                print(f"Failed to parse segmentation text file: {e}")
                                continue
                        
                except Exception as e:
                    print(f"Error loading segmentation file {seg_file}: {e}")
                    continue
        
        print("No segmentation labels found - will visualize 3D detection boxes only")
        return None
    
    def load_calibration(self):
        """加载标定参数"""
        calib_file = self.data_path / 'calib' / '000000.txt'
        if not calib_file.exists():
            # 尝试其他可能的标定文件位置
            calib_file = self.data_path / 'calib.txt'
        
        if not calib_file.exists():
            print("Warning: Calibration file not found. Using default values.")
            return None
        
        calib = {}
        with open(calib_file, 'r') as f:
            for line in f.readlines():
                if ':' in line:
                    key, value = line.split(':', 1)
                    calib[key.strip()] = np.array([float(x) for x in value.split()])
        
        return calib
    
    def get_3d_box_corners(self, obj):
        """计算3D框的8个角点"""
        h, w, l = obj['dimensions']
        x, y, z = obj['location']
        ry = obj['rotation_y']
        
        # 3D bounding box corners
        x_corners = [l/2, l/2, -l/2, -l/2, l/2, l/2, -l/2, -l/2]
        y_corners = [0, 0, 0, 0, -h, -h, -h, -h]
        z_corners = [w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2, w/2]
        
        # Rotate around Y-axis
        corners = np.array([x_corners, y_corners, z_corners])
        rotation_matrix = np.array([
            [np.cos(ry), 0, np.sin(ry)],
            [0, 1, 0],
            [-np.sin(ry), 0, np.cos(ry)]
        ])
        
        corners = rotation_matrix @ corners
        corners[0, :] += x
        corners[1, :] += y
        corners[2, :] += z
        
        return corners.T
    
    def visualize_2d(self, frame_id):
        """2D可视化：图像 + 鸟瞰图 + 前视图 + 分割可视化"""
        try:
            points = self.load_velodyne_points(frame_id)
            detection_labels, seg_labels = self.load_labels(frame_id)
            
            # 根据是否有分割标签调整布局
            if seg_labels is not None:
                fig = plt.figure(figsize=(20, 12))
                subplot_config = [(2, 3, i) for i in range(1, 7)]
            else:
                fig = plt.figure(figsize=(18, 6))
                subplot_config = [(1, 3, i) for i in range(1, 4)]
            
            plot_idx = 0
            
            # 尝试加载图像
            try:
                image = self.load_image(frame_id)
                ax1 = fig.add_subplot(*subplot_config[plot_idx])
                ax1.imshow(image)
                ax1.set_title(f'Camera Image - Frame {frame_id}')
                ax1.axis('off')
                plot_idx += 1
            except FileNotFoundError:
                print(f"Image not found for frame {frame_id}")
            
            # 鸟瞰图 - 原始点云
            ax2 = fig.add_subplot(*subplot_config[plot_idx])
            mask = (points[:, 0] > -50) & (points[:, 0] < 50) & \
                   (points[:, 1] > -50) & (points[:, 1] < 50)
            filtered_points = points[mask]
            
            if seg_labels is not None and len(seg_labels) == len(points):
                # 使用分割标签着色
                filtered_labels = seg_labels[mask]
                colors = self.get_segmentation_colors(filtered_labels)
                scatter = ax2.scatter(filtered_points[:, 0], filtered_points[:, 1], 
                                    c=colors, s=0.5, alpha=0.8)
                ax2.set_title('Bird Eye View (Segmentation)')
            else:
                # 使用高度着色
                scatter = ax2.scatter(filtered_points[:, 0], filtered_points[:, 1], 
                                    c=filtered_points[:, 2], s=0.5, cmap='viridis', alpha=0.6)
                ax2.set_title('Bird Eye View (Height)')
                plt.colorbar(scatter, ax=ax2, shrink=0.6)
            
            # 绘制3D框在鸟瞰图上的投影
            for obj in detection_labels:
                if isinstance(obj, dict) and 'dimensions' in obj and obj['type'] in ['Car', 'Van', 'Truck', 'Pedestrian', 'Cyclist']:
                    corners = self.get_3d_box_corners(obj)
                    bottom_corners = corners[:4, [0, 2]]  # x, z coordinates
                    for i in range(4):
                        j = (i + 1) % 4
                        ax2.plot([bottom_corners[i, 0], bottom_corners[j, 0]], 
                                [bottom_corners[i, 1], bottom_corners[j, 1]], 
                                'r-', linewidth=2)
            
            ax2.set_xlabel('X (m)')
            ax2.set_ylabel('Y (m)')
            ax2.axis('equal')
            plot_idx += 1
            
            # 前视图
            ax3 = fig.add_subplot(*subplot_config[plot_idx])
            front_mask = (points[:, 0] > 0) & (points[:, 0] < 70)
            front_points = points[front_mask]
            
            if seg_labels is not None and len(seg_labels) == len(points):
                front_labels = seg_labels[front_mask]
                colors = self.get_segmentation_colors(front_labels)
                scatter2 = ax3.scatter(front_points[:, 0], front_points[:, 2], 
                                     c=colors, s=0.5, alpha=0.8)
                ax3.set_title('Front View (Segmentation)')
            else:
                scatter2 = ax3.scatter(front_points[:, 0], front_points[:, 2], 
                                     c=front_points[:, 1], s=0.5, cmap='plasma', alpha=0.6)
                ax3.set_title('Front View')
                plt.colorbar(scatter2, ax=ax3, shrink=0.6)
            
            ax3.set_xlabel('X (m)')
            ax3.set_ylabel('Z (m)')
            plot_idx += 1
            
            # 如果有分割标签，添加额外的可视化
            if seg_labels is not None:
                # 分割标签统计
                ax4 = fig.add_subplot(*subplot_config[plot_idx])
                unique_labels, counts = np.unique(seg_labels, return_counts=True)
                
                # 创建标签名称映射
                label_names = {
                            1: 'traversable', 2: 'mid-cost', 3: 'high-cost', 4: 'barrier',
                }
                
                # 只显示前10个最常见的标签
                sorted_indices = np.argsort(counts)[::-1][:10]
                top_labels = unique_labels[sorted_indices]
                top_counts = counts[sorted_indices]
                
                colors_bar = [np.array(self.get_segmentation_colors(np.array([label]))[0]) 
                             for label in top_labels]
                
                bars = ax4.bar(range(len(top_labels)), top_counts, color=colors_bar)
                ax4.set_xlabel('Segmentation Classes')
                ax4.set_ylabel('Point Count')
                ax4.set_title('Segmentation Statistics')
                
                # 设置x轴标签
                labels_text = [label_names.get(label, f'class_{label}') for label in top_labels]
                ax4.set_xticks(range(len(top_labels)))
                ax4.set_xticklabels(labels_text, rotation=45, ha='right')
                
                # 在条形图上显示数值
                for bar, count in zip(bars, top_counts):
                    height = bar.get_height()
                    ax4.text(bar.get_x() + bar.get_width()/2., height,
                            f'{count}', ha='center', va='bottom')
                
                plot_idx += 1
                
                # 3D分割可视化（下采样）
                ax5 = fig.add_subplot(*subplot_config[plot_idx], projection='3d')
                
                # 下采样以提高显示速度
                step = max(1, len(points) // 20000)
                sampled_indices = np.arange(0, len(points), step)
                sampled_points = points[sampled_indices]
                sampled_labels = seg_labels[sampled_indices]
                
                # 过滤显示范围
                display_mask = (sampled_points[:, 0] > -30) & (sampled_points[:, 0] < 30) & \
                              (sampled_points[:, 1] > -15) & (sampled_points[:, 1] < 15) & \
                              (sampled_points[:, 2] > -2) & (sampled_points[:, 2] < 3)
                
                display_points = sampled_points[display_mask]
                display_labels = sampled_labels[display_mask]
                
                colors_3d = self.get_segmentation_colors(display_labels)
                
                ax5.scatter(display_points[:, 0], display_points[:, 1], display_points[:, 2],
                           c=colors_3d, s=0.5, alpha=0.8)
                
                ax5.set_xlabel('X (m)')
                ax5.set_ylabel('Y (m)')
                ax5.set_zlabel('Z (m)')
                ax5.set_title('3D Segmentation')
                plot_idx += 1
                
                # 分割标签图例
                if plot_idx < len(subplot_config):
                    ax6 = fig.add_subplot(*subplot_config[plot_idx])
                    ax6.axis('off')
                    
                    legend_text = "Segmentation Legend:\n\n"
                    for i, (label, count) in enumerate(zip(top_labels, top_counts)):
                        name = label_names.get(label, f'class_{label}')
                        percentage = (count / len(seg_labels)) * 100
                        legend_text += f"{label}: {name} ({percentage:.1f}%)\n"
                    
                    ax6.text(0.1, 0.9, legend_text, transform=ax6.transAxes, 
                            fontsize=10, verticalalignment='top', fontfamily='monospace')
            
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            print(f"Error in 2D visualization: {e}")
            import traceback
            traceback.print_exc()
    
    def visualize_3d_matplotlib(self, frame_id):
        """使用matplotlib进行3D可视化"""
        try:
            points = self.load_velodyne_points(frame_id)
            labels = self.load_labels(frame_id)
            
            fig = plt.figure(figsize=(12, 8))
            ax = fig.add_subplot(111, projection='3d')
            
            # 下采样点云以提高显示速度
            step = max(1, len(points) // 50000)
            sampled_points = points[::step]
            
            # 过滤显示范围
            mask = (sampled_points[:, 0] > -50) & (sampled_points[:, 0] < 50) & \
                   (sampled_points[:, 1] > -25) & (sampled_points[:, 1] < 25) & \
                   (sampled_points[:, 2] > -3) & (sampled_points[:, 2] < 5)
            
            filtered_points = sampled_points[mask]
            
            scatter = ax.scatter(filtered_points[:, 0], filtered_points[:, 1], filtered_points[:, 2],
                               c=filtered_points[:, 3], s=0.5, cmap='viridis', alpha=0.6)
            
            # 绘制3D标注框
            for obj in labels:
                if obj['type'] in ['Car', 'Van', 'Truck', 'Pedestrian', 'Cyclist']:
                    corners = self.get_3d_box_corners(obj)
                    self.draw_3d_box(ax, corners, color='red')
            
            ax.set_xlabel('X (m)')
            ax.set_ylabel('Y (m)')
            ax.set_zlabel('Z (m)')
            ax.set_title(f'3D Point Cloud - Frame {frame_id}')
            
            plt.colorbar(scatter, shrink=0.5)
            plt.show()
            
        except Exception as e:
            print(f"Error in 3D matplotlib visualization: {e}")
    
    def visualize_3d_open3d(self, frame_id):
        """使用Open3D进行3D可视化，包含分割标签"""
        if not OPEN3D_AVAILABLE:
            print("Open3D not available. Please install: pip install open3d")
            return
        
        try:
            points = self.load_velodyne_points(frame_id)
            detection_labels, seg_labels = self.load_labels(frame_id)
            
            # 创建点云对象
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points[:, :3])
            
            # 根据分割标签或强度值着色
            if seg_labels is not None and len(seg_labels) == len(points):
                colors = self.get_segmentation_colors(seg_labels)
                pcd.colors = o3d.utility.Vector3dVector(colors)
                print(f"Visualizing with segmentation labels ({len(np.unique(seg_labels))} classes)")
            else:
                # 使用强度值着色
                intensities = points[:, 3]
                normalized_intensities = (intensities - intensities.min()) / (intensities.max() - intensities.min())
                colors = plt.cm.viridis(normalized_intensities)[:, :3]
                pcd.colors = o3d.utility.Vector3dVector(colors)
                print("Visualizing with intensity values")
            
            # 创建3D框
            boxes = []
            for obj in detection_labels:
                if isinstance(obj, dict) and 'dimensions' in obj and obj['type'] in ['Car', 'Van', 'Truck', 'Pedestrian', 'Cyclist']:
                    corners = self.get_3d_box_corners(obj)
                    box = self.create_open3d_box(corners)
                    boxes.append(box)
            
            # 可视化
            geometries = [pcd] + boxes
            
            # 添加坐标系
            coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=2.0, origin=[0, 0, 0])
            geometries.append(coordinate_frame)
            
            o3d.visualization.draw_geometries(geometries, 
                                            window_name=f"KITTI Frame {frame_id} - Segmentation",
                                            width=1200, height=800)
            
        except Exception as e:
            print(f"Error in Open3D visualization: {e}")
            import traceback
            traceback.print_exc()
    
    def get_segmentation_colors(self, labels):
        """为不同的分割类别分配颜色"""
        # 常见的语义分割类别颜色映射
        color_map = {
            0: [0, 0, 0],        # unlabeled - 黑色
            1: [128, 64, 128],   # road - 紫色
            2: [244, 35, 232],   # sidewalk - 粉色
            3: [70, 70, 70],     # building - 灰色
            4: [102, 102, 156],  # wall - 浅蓝紫色
            5: [190, 153, 153],  # fence - 浅棕色
            6: [153, 153, 153],  # pole - 中灰色
            7: [250, 170, 30],   # traffic light - 橙色
            8: [220, 220, 0],    # traffic sign - 黄色
            9: [107, 142, 35],   # vegetation - 橄榄绿
            10: [152, 251, 152], # terrain - 浅绿色
            11: [70, 130, 180],  # sky - 钢蓝色
            12: [220, 20, 60],   # person - 深红色
            13: [255, 0, 0],     # rider - 红色
            14: [0, 0, 142],     # car - 深蓝色
            15: [0, 0, 70],      # truck - 深蓝色
            16: [0, 60, 100],    # bus - 深蓝绿色
            17: [0, 80, 100],    # train - 深青色
            18: [0, 0, 230],     # motorcycle - 蓝色
            19: [119, 11, 32],   # bicycle - 深红棕色
        }
        
        # 为每个点分配颜色
        colors = np.zeros((len(labels), 3))
        unique_labels = np.unique(labels)
        
        for label in unique_labels:
            mask = labels == label
            if label in color_map:
                colors[mask] = np.array(color_map[label]) / 255.0
            else:
                # 为未知标签生成随机颜色
                np.random.seed(int(label))  # 确保同一标签总是相同颜色
                colors[mask] = np.random.rand(3)
        
        return colors
        """在matplotlib 3D图中绘制3D框"""
        # 定义12条边
        edges = [
            [0, 1], [1, 2], [2, 3], [3, 0],  # 底面
            [4, 5], [5, 6], [6, 7], [7, 4],  # 顶面
            [0, 4], [1, 5], [2, 6], [3, 7]   # 竖直边
        ]
        
        for edge in edges:
            points = corners[edge]
            ax.plot3D(*points.T, color=color, linewidth=2)
    
    def draw_3d_box(self, ax, corners, color='red'):
        """创建Open3D线框"""
        # 定义12条边
        lines = [
            [0, 1], [1, 2], [2, 3], [3, 0],  # 底面
            [4, 5], [5, 6], [6, 7], [7, 4],  # 顶面
            [0, 4], [1, 5], [2, 6], [3, 7]   # 竖直边
        ]
        
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(corners)
        line_set.lines = o3d.utility.Vector2iVector(lines)
        # 设置为红色
        colors = [[1, 0, 0] for i in range(len(lines))]
        line_set.colors = o3d.utility.Vector3dVector(colors)
        
        return line_set
    
    def interactive_visualizer(self):
        """交互式可视化"""
        print("Interactive KITTI Visualizer")
        print("Commands:")
        print("  n/next: Next frame")
        print("  p/prev: Previous frame")
        print("  g/goto: Go to specific frame")
        print("  2d: 2D visualization")
        print("  3d: 3D visualization (matplotlib)")
        print("  o3d: 3D visualization (Open3D)")
        print("  q/quit: Quit")
        
        while True:
            cmd = input(f"\nFrame {self.current_frame} > ").strip().lower()
            
            if cmd in ['q', 'quit']:
                break
            elif cmd in ['n', 'next']:
                self.current_frame += 1
                self.visualize_2d(self.current_frame)
            elif cmd in ['p', 'prev']:
                self.current_frame = max(0, self.current_frame - 1)
                self.visualize_2d(self.current_frame)
            elif cmd in ['g', 'goto']:
                try:
                    frame = int(input("Enter frame number: "))
                    self.current_frame = frame
                    self.visualize_2d(self.current_frame)
                except ValueError:
                    print("Invalid frame number")
            elif cmd == '2d':
                self.visualize_2d(self.current_frame)
            elif cmd == '3d':
                self.visualize_3d_matplotlib(self.current_frame)
            elif cmd == 'o3d':
                self.visualize_3d_open3d(self.current_frame)
            else:
                print("Unknown command")


def main():
    parser = argparse.ArgumentParser(description='KITTI Dataset Visualizer')
    parser.add_argument('data_path', help='Path to KITTI dataset directory')
    parser.add_argument('--frame', type=int, default=0, help='Frame number to visualize')
    parser.add_argument('--mode', choices=['2d', '3d', 'o3d', 'interactive'], 
                       default='interactive', help='Visualization mode')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.data_path):
        print(f"Error: Data path {args.data_path} does not exist")
        return
    
    visualizer = KITTIVisualizer(args.data_path)
    
    if args.mode == 'interactive':
        visualizer.current_frame = args.frame
        visualizer.interactive_visualizer()
    elif args.mode == '2d':
        visualizer.visualize_2d(args.frame)
    elif args.mode == '3d':
        visualizer.visualize_3d_matplotlib(args.frame)
    elif args.mode == 'o3d':
        visualizer.visualize_3d_open3d(args.frame)


if __name__ == "__main__":
    main()