#!/usr/bin/env python3
"""
地面穿越性数据集可视化器
可视化KITTI格式的地面穿越性标注数据集，包括点云和3D边界框
标签：1=traversable, 2=mid-cost, 3=high-cost, 4=barrier
"""

import numpy as np
import open3d as o3d
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits.mplot3d import Axes3D
import argparse
import os
from pathlib import Path
import glob

# 地面穿越性类别映射
TRAVERSABILITY_CLASSES = {
    1: {'name': 'Traversable', 'color': [0.2, 0.8, 0.2], 'description': '可通行区域'},      # 绿色
    2: {'name': 'Mid-cost', 'color': [1.0, 0.8, 0.2], 'description': '中等代价区域'},        # 橙色
    3: {'name': 'High-cost', 'color': [1.0, 0.4, 0.2], 'description': '高代价区域'},        # 橙红色
    4: {'name': 'Barrier', 'color': [0.8, 0.2, 0.2], 'description': '障碍物/不可通行'}       # 红色
}

class TraversabilityVisualizer:
    def __init__(self, dataset_path, sequence="00"):
        """
        初始化地面穿越性数据集可视化器
        
        Args:
            dataset_path: 数据集根目录或序列目录
            sequence: 序列号，如"00"
        """
        self.dataset_path = Path(dataset_path)
        
        # 智能检测路径结构
        if (self.dataset_path / "sequences" / sequence).exists():
            # 传入的是数据集根目录
            self.sequence_path = self.dataset_path / "sequences" / sequence
        elif (self.dataset_path / "velodyne").exists() or (self.dataset_path / "labels").exists():
            # 传入的直接是序列目录
            self.sequence_path = self.dataset_path
        else:
            raise FileNotFoundError(f"无法找到有效的数据集结构: {self.dataset_path}")
        
        self.sequence = sequence
        
        print(f"🎯 加载地面穿越性数据集: {self.sequence_path}")
        
        # 打印类别信息
        print("\n📊 穿越性类别:")
        for class_id, info in TRAVERSABILITY_CLASSES.items():
            print(f"   {class_id}: {info['name']} - {info['description']}")
        
    def load_point_cloud(self, frame_id):
        """加载指定帧的点云数据"""
        velodyne_file = self.sequence_path / "velodyne" / f"{frame_id:06d}.bin"
        
        if not velodyne_file.exists():
            print(f"⚠️  点云文件不存在: {velodyne_file}")
            return None
        
        try:
            # 读取KITTI点云格式 (N x 4: x, y, z, intensity)
            points = np.fromfile(velodyne_file, dtype=np.float32).reshape(-1, 4)
            print(f"📊 加载点云: {len(points):,} 个点")
            return points
        except Exception as e:
            print(f"❌ 加载点云失败: {e}")
            return None
    
    def load_labels(self, frame_id):
        """加载指定帧的穿越性标注"""
        label_file = self.sequence_path / "labels" / f"{frame_id:06d}.txt"
        
        if not label_file.exists():
            print(f"⚠️  标注文件不存在: {label_file}")
            return []
        
        labels = []
        try:
            with open(label_file, 'r') as f:
                for line_num, line in enumerate(f, 1):
                    parts = line.strip().split()
                    if len(parts) >= 15:
                        try:
                            traversability_type = int(parts[0])
                            if traversability_type not in TRAVERSABILITY_CLASSES:
                                print(f"⚠️  第{line_num}行: 未知的穿越性类别 {traversability_type}")
                                continue
                            
                            label = {
                                'type': traversability_type,
                                'truncated': float(parts[1]),
                                'occluded': int(parts[2]),
                                'alpha': float(parts[3]),
                                'bbox_2d': [float(parts[4]), float(parts[5]), 
                                           float(parts[6]), float(parts[7])],
                                'h': float(parts[8]),
                                'w': float(parts[9]),
                                'l': float(parts[10]),
                                'x': float(parts[11]),
                                'y': float(parts[12]),
                                'z': float(parts[13]),
                                'ry': float(parts[14])
                            }
                            labels.append(label)
                        except ValueError as e:
                            print(f"⚠️  第{line_num}行解析错误: {e}")
                            continue
                    else:
                        print(f"⚠️  第{line_num}行字段不足: {len(parts)}/15")
        except Exception as e:
            print(f"❌ 读取标注文件失败: {e}")
            return []
        
        print(f"📊 加载标注: {len(labels)} 个区域")
        
        # 统计各类别数量
        if labels:
            class_counts = {}
            for label in labels:
                class_type = label['type']
                class_counts[class_type] = class_counts.get(class_type, 0) + 1
            
            for class_id, count in class_counts.items():
                class_name = TRAVERSABILITY_CLASSES[class_id]['name']
                print(f"   {class_name}: {count} 个区域")
        
        return labels
    
    def load_image(self, frame_id, camera="image_0"):
        """加载指定帧的图像"""
        for camera_dir in ["image_0", "image_2"]:
            image_file = self.sequence_path / camera_dir / f"{frame_id:06d}.png"
            if image_file.exists():
                try:
                    image = cv2.imread(str(image_file))
                    if image is not None:
                        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        return image
                except Exception as e:
                    print(f"⚠️  读取图像失败: {e}")
        
        print(f"⚠️  图像文件不存在: {frame_id:06d}.png")
        return None
    
    def create_bbox_lines(self, bbox):
        """创建3D边界框的线条（适配CVAT/Datumaro格式）"""
        x, y, z = bbox['x'], bbox['y'], bbox['z']
        h, w, l = bbox['h'], bbox['w'], bbox['l']
        ry = bbox['ry']
        
        # 边界框的8个顶点（在物体坐标系中）
        vertices = np.array([
            [-w/2, -l/2, 0],     # 0: 左前下
            [ w/2, -l/2, 0],     # 1: 右前下
            [ w/2,  l/2, 0],     # 2: 右后下
            [-w/2,  l/2, 0],     # 3: 左后下
            [-w/2, -l/2, h],     # 4: 左前上
            [ w/2, -l/2, h],     # 5: 右前上
            [ w/2,  l/2, h],     # 6: 右后上
            [-w/2,  l/2, h]      # 7: 左后上
        ])
        
        # 旋转矩阵（绕Y轴旋转）
        cos_ry = np.cos(ry)
        sin_ry = np.sin(ry)
        R = np.array([
            [cos_ry, 0, sin_ry],
            [0, 1, 0],
            [-sin_ry, 0, cos_ry]
        ])
        
        # 应用旋转和平移（CVAT格式：Y是深度，Z是高度）
        vertices_rotated = vertices @ R.T
        # 注意：bbox的z是底面高度，不需要额外添加h/2
        vertices_world = vertices_rotated + np.array([x, y, z])  # CVAT格式：直接使用z作为底面高度
        
        # 定义边界框的12条边
        lines = [
            [0, 1], [1, 2], [2, 3], [3, 0],  # 底面
            [4, 5], [5, 6], [6, 7], [7, 4],  # 顶面
            [0, 4], [1, 5], [2, 6], [3, 7]   # 垂直边
        ]
        
        return vertices_world, lines
    
    def visualize_with_open3d(self, frame_id, point_size=3.0, max_points=None, show_ground_plane=False):
        """使用Open3D可视化点云和穿越性标注"""
        print(f"\n🎬 可视化帧 {frame_id:06d}")
        
        # 加载数据
        points = self.load_point_cloud(frame_id)
        labels = self.load_labels(frame_id)
        
        if points is None:
            print("❌ 无法加载点云数据")
            return
        
        # 创建可视化器
        vis = o3d.visualization.Visualizer()
        vis.create_window(f"地面穿越性可视化 - 帧 {frame_id:06d}", width=1400, height=900)
        
        # 点云降采样（如果需要）
        if max_points and len(points) > max_points:
            print(f"🔽 点云降采样: {len(points):,} → {max_points:,}")
            indices = np.random.choice(len(points), max_points, replace=False)
            points = points[indices]
        
        # 创建点云对象
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points[:, :3])
        
        # 根据强度着色点云
        if points.shape[1] >= 4:
            intensities = points[:, 3]
            # 归一化强度到[0,1]
            if intensities.max() > intensities.min():
                intensities_norm = (intensities - intensities.min()) / (intensities.max() - intensities.min())
            else:
                intensities_norm = np.ones_like(intensities) * 0.5
            
            # 使用viridis颜色图
            colors = plt.cm.viridis(intensities_norm)[:, :3]
            pcd.colors = o3d.utility.Vector3dVector(colors)
        else:
            # 默认颜色
            pcd.paint_uniform_color([0.7, 0.7, 0.7])
        
        vis.add_geometry(pcd)
        
        # 添加地面平面（可选）
        if show_ground_plane:
            ground_plane = o3d.geometry.TriangleMesh.create_box(width=50, height=0.01, depth=50)
            ground_plane.translate([-25, -0.005, -25])
            ground_plane.paint_uniform_color([0.3, 0.3, 0.3])
            vis.add_geometry(ground_plane)
        
        # 添加3D边界框
        for i, bbox in enumerate(labels):
            vertices, lines = self.create_bbox_lines(bbox)
            
            # 获取类别信息
            class_info = TRAVERSABILITY_CLASSES.get(bbox['type'], 
                                                   {'name': 'Unknown', 'color': [0.5, 0.5, 0.5]})
            
            # 创建线框
            line_set = o3d.geometry.LineSet()
            line_set.points = o3d.utility.Vector3dVector(vertices)
            line_set.lines = o3d.utility.Vector2iVector(lines)
            line_set.colors = o3d.utility.Vector3dVector([class_info['color']] * len(lines))
            
            vis.add_geometry(line_set)
            
            print(f"  📦 边界框 {i+1}: {class_info['name']} at ({bbox['x']:.1f}, {bbox['y']:.1f}, {bbox['z']:.1f})")
        
        # 添加坐标轴
        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=3.0)
        vis.add_geometry(coord_frame)
        
        # 设置渲染选项
        render_option = vis.get_render_option()
        render_option.point_size = point_size
        render_option.background_color = np.array([0.05, 0.05, 0.05])
        render_option.point_show_normal = False
        render_option.show_coordinate_frame = True
        
        # 设置相机视角
        ctr = vis.get_view_control()
        ctr.set_front([0.5, -0.3, -0.8])
        ctr.set_lookat([0, 0, 0])
        ctr.set_up([0, 0, 1])
        ctr.set_zoom(0.4)
        
        print("\n🎮 控制说明:")
        print("   鼠标左键拖拽: 旋转视角")
        print("   鼠标右键拖拽: 平移视角")
        print("   滚轮: 缩放")
        print("   Q 或 ESC: 退出")
        
        # 运行可视化
        vis.run()
        vis.destroy_window()
    
    def visualize_with_matplotlib(self, frame_id, view_range=30, point_size=2.0):
        """使用Matplotlib可视化（鸟瞰图）"""
        print(f"\n🎬 可视化帧 {frame_id:06d} (Matplotlib)")
        
        # 加载数据
        points = self.load_point_cloud(frame_id)
        labels = self.load_labels(frame_id)
        
        if points is None:
            print("❌ 无法加载点云数据")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 9))
        
        # 过滤点云范围
        mask = (np.abs(points[:, 0]) < view_range) & (np.abs(points[:, 1]) < view_range)
        filtered_points = points[mask]
        print(f"📊 显示点云: {len(filtered_points):,} 个点 (范围: ±{view_range}m)")
        
        # 左图：点云鸟瞰图
        ax1.set_title(f'点云鸟瞰图 - 帧 {frame_id:06d}', fontsize=14, fontweight='bold')
        
        # 绘制点云
        if len(filtered_points) > 0:
            scatter = ax1.scatter(filtered_points[:, 0], filtered_points[:, 1], 
                                c=filtered_points[:, 3], cmap='viridis', s=point_size, alpha=0.7)
            plt.colorbar(scatter, ax=ax1, label='Intensity', shrink=0.8)
        
        ax1.set_xlabel('X (m)', fontsize=12)
        ax1.set_ylabel('Y (m)', fontsize=12)
        ax1.set_xlim([-view_range, view_range])
        ax1.set_ylim([-view_range, view_range])
        ax1.set_aspect('equal')
        ax1.grid(True, alpha=0.3)
        
        # 右图：穿越性标注鸟瞰图
        ax2.set_title(f'穿越性标注 - 帧 {frame_id:06d}', fontsize=14, fontweight='bold')
        
        # 绘制点云背景
        if len(filtered_points) > 0:
            ax2.scatter(filtered_points[:, 0], filtered_points[:, 1], 
                       c='lightgray', s=point_size*0.3, alpha=0.4)
        
        # 绘制边界框
        for i, bbox in enumerate(labels):
            vertices, _ = self.create_bbox_lines(bbox)
            # 获取底面的4个顶点（XZ平面用于鸟瞰图）
            bottom_vertices = vertices[:4]
            
            class_info = TRAVERSABILITY_CLASSES[bbox['type']]
            
            # 创建多边形
            polygon = patches.Polygon(bottom_vertices[:, [0, 2]], # 使用X和Z坐标（鸟瞰图）
                                    linewidth=2, 
                                    edgecolor=class_info['color'],
                                    facecolor=class_info['color'],
                                    alpha=0.4)
            ax2.add_patch(polygon)
            
            # 添加标签
            center_x = bbox['x']
            center_y = bbox['y']
            ax2.text(center_x, center_y, f"{class_info['name'][:4]}\n({bbox['type']})", 
                    ha='center', va='center', fontsize=9, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        
        ax2.set_xlabel('X (m)', fontsize=12)
        ax2.set_ylabel('Y (m)', fontsize=12)
        ax2.set_xlim([-view_range, view_range])
        ax2.set_ylim([-view_range, view_range])
        ax2.set_aspect('equal')
        ax2.grid(True, alpha=0.3)
        
        # 添加图例
        legend_elements = []
        for class_id, info in TRAVERSABILITY_CLASSES.items():
            legend_elements.append(patches.Patch(color=info['color'], 
                                                label=f"{class_id}: {info['name']}"))
        ax2.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=10)
        
        plt.tight_layout()
        plt.show()
    
    def analyze_dataset(self):
        """分析整个数据集的统计信息"""
        print("\n📊 数据集分析")
        print("=" * 60)
        
        # 统计文件数量
        velodyne_files = list(self.sequence_path.glob("velodyne/*.bin"))
        label_files = list(self.sequence_path.glob("labels/*.txt"))
        
        image_files = []
        for img_dir in ["image_0", "image_2"]:
            img_path = self.sequence_path / img_dir
            if img_path.exists():
                image_files.extend(list(img_path.glob("*.png")))
        
        print(f"📄 点云文件: {len(velodyne_files)}")
        print(f"📋 标注文件: {len(label_files)}")
        print(f"🖼️  图像文件: {len(image_files)}")
        
        if not label_files:
            print("⚠️  没有找到标注文件")
            return
        
        # 分析标注统计
        total_annotations = 0
        class_counts = {i: 0 for i in TRAVERSABILITY_CLASSES.keys()}
        frame_counts = {}
        
        for label_file in label_files:
            frame_id = int(label_file.stem)
            frame_annotations = 0
            
            try:
                with open(label_file, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 15:
                            try:
                                class_type = int(parts[0])
                                if class_type in TRAVERSABILITY_CLASSES:
                                    class_counts[class_type] += 1
                                    total_annotations += 1
                                    frame_annotations += 1
                            except ValueError:
                                continue
                
                frame_counts[frame_id] = frame_annotations
            except Exception as e:
                print(f"⚠️  读取 {label_file.name} 失败: {e}")
        
        print(f"\n📊 总标注数量: {total_annotations:,}")
        print("📈 各类别分布:")
        for class_id, count in class_counts.items():
            class_info = TRAVERSABILITY_CLASSES[class_id]
            percentage = (count / total_annotations * 100) if total_annotations > 0 else 0
            print(f"   {class_id}: {class_info['name']:<12} - {count:>6} ({percentage:>5.1f}%)")
        
        if frame_counts:
            avg_annotations = np.mean(list(frame_counts.values()))
            max_annotations = max(frame_counts.values())
            min_annotations = min(frame_counts.values())
            
            print(f"\n📈 标注分布:")
            print(f"   平均每帧: {avg_annotations:.1f} 个")
            print(f"   最多: {max_annotations} 个")
            print(f"   最少: {min_annotations} 个")


def main():
    parser = argparse.ArgumentParser(description="地面穿越性数据集可视化器")
    parser.add_argument("dataset_path", help="数据集根目录或序列目录路径")
    parser.add_argument("-s", "--sequence", default="00", help="序列号 (默认: 00)")
    parser.add_argument("-f", "--frame", type=int, default=0, help="帧ID (默认: 0)")
    parser.add_argument("--mode", choices=["o3d", "plt", "analyze"], default="o3d",
                       help="可视化模式: o3d=Open3D, plt=Matplotlib, analyze=数据分析")
    parser.add_argument("--point-size", type=float, default=3.0, help="点云大小 (默认: 3.0)")
    parser.add_argument("--max-points", type=int, help="最大点云数量（降采样）")
    parser.add_argument("--view-range", type=int, default=30, help="Matplotlib视图范围(米)")
    parser.add_argument("--show-ground", action="store_true", help="显示地面平面（默认不显示）")
    
    args = parser.parse_args()
    
    try:
        visualizer = TraversabilityVisualizer(args.dataset_path, args.sequence)
        
        if args.mode == "analyze":
            visualizer.analyze_dataset()
        elif args.mode == "o3d":
            visualizer.visualize_with_open3d(
                args.frame, 
                point_size=args.point_size,
                max_points=args.max_points,
                show_ground_plane=args.show_ground
            )
        elif args.mode == "plt":
            visualizer.visualize_with_matplotlib(
                args.frame, 
                args.view_range, 
                point_size=args.point_size
            )
            
    except KeyboardInterrupt:
        print("\n🛑 用户中断")
    except Exception as e:
        print(f"❌ 错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()