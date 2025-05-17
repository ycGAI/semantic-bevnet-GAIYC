#!/usr/bin/env python3
"""
可视化转换后的点云分割结果
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import argparse
import os

def visualize_labeled_pointcloud(velodyne_path, label_path, max_points=10000):
    """
    可视化带标签的点云
    """
    # 读取点云
    if not os.path.exists(velodyne_path):
        print(f"Point cloud file not found: {velodyne_path}")
        return
    
    points = np.fromfile(velodyne_path, dtype=np.float32).reshape(-1, 4)
    xyz = points[:, :3]
    
    # 读取标签
    labels = None
    if os.path.exists(label_path):
        labels = np.fromfile(label_path, dtype=np.uint32)
        if len(labels) != len(xyz):
            print(f"Warning: Labels length ({len(labels)}) doesn't match points ({len(xyz)})")
            labels = None
    
    # 如果点太多，随机采样
    if len(xyz) > max_points:
        indices = np.random.choice(len(xyz), max_points, replace=False)
        xyz = xyz[indices]
        if labels is not None:
            labels = labels[indices]
    
    # 创建图形
    fig = plt.figure(figsize=(15, 10))
    
    # 3D散点图
    ax1 = fig.add_subplot(221, projection='3d')
    if labels is not None:
        # 根据标签着色
        unique_labels = np.unique(labels)
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
        
        for i, label in enumerate(unique_labels):
            mask = labels == label
            ax1.scatter(xyz[mask, 0], xyz[mask, 1], xyz[mask, 2], 
                       c=[colors[i]], label=f'Class {label}', s=1)
        ax1.legend()
    else:
        ax1.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c='blue', s=1)
    
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_title('3D Point Cloud with Labels')
    
    # 鸟瞰图 (X-Z平面)
    ax2 = fig.add_subplot(222)
    if labels is not None:
        for i, label in enumerate(unique_labels):
            mask = labels == label
            ax2.scatter(xyz[mask, 0], xyz[mask, 2], 
                       c=[colors[i]], label=f'Class {label}', s=1)
        ax2.legend()
    else:
        ax2.scatter(xyz[:, 0], xyz[:, 2], c='blue', s=1)
    
    ax2.set_xlabel('X')
    ax2.set_ylabel('Z')
    ax2.set_title('Bird\'s Eye View (X-Z)')
    ax2.axis('equal')
    
    # 侧视图 (X-Y平面)
    ax3 = fig.add_subplot(223)
    if labels is not None:
        for i, label in enumerate(unique_labels):
            mask = labels == label
            ax3.scatter(xyz[mask, 0], xyz[mask, 1], 
                       c=[colors[i]], label=f'Class {label}', s=1)
    else:
        ax3.scatter(xyz[:, 0], xyz[:, 1], c='blue', s=1)
    
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.set_title('Side View (X-Y)')
    
    # 标签分布直方图
    ax4 = fig.add_subplot(224)
    if labels is not None:
        unique, counts = np.unique(labels, return_counts=True)
        ax4.bar(unique, counts)
        ax4.set_xlabel('Class Label')
        ax4.set_ylabel('Number of Points')
        ax4.set_title('Label Distribution')
        
        # 添加数值标签
        for i, (label, count) in enumerate(zip(unique, counts)):
            ax4.text(label, count + max(counts)*0.01, str(count), 
                    ha='center', va='bottom')
    else:
        ax4.text(0.5, 0.5, 'No labels available', 
                transform=ax4.transAxes, ha='center', va='center')
        ax4.set_title('No Labels Available')
    
    plt.tight_layout()
    plt.show()

def compare_before_after(original_velodyne, filtered_velodyne, label_path):
    """
    比较过滤前后的点云
    """
    # 读取原始点云
    if os.path.exists(original_velodyne):
        original_points = np.fromfile(original_velodyne, dtype=np.float32).reshape(-1, 4)
        original_xyz = original_points[:, :3]
    else:
        print(f"Original point cloud not found: {original_velodyne}")
        return
    
    # 读取过滤后点云
    if os.path.exists(filtered_velodyne):
        filtered_points = np.fromfile(filtered_velodyne, dtype=np.float32).reshape(-1, 4)
        filtered_xyz = filtered_points[:, :3]
    else:
        print(f"Filtered point cloud not found: {filtered_velodyne}")
        return
    
    # 读取标签
    labels = None
    if os.path.exists(label_path):
        labels = np.fromfile(label_path, dtype=np.uint32)
    
    # 创建比较图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 原始点云（鸟瞰图）
    ax1.scatter(original_xyz[:, 0], original_xyz[:, 2], c='gray', s=0.5, alpha=0.6)
    ax1.set_xlabel('X')
    ax1.set_ylabel('Z')
    ax1.set_title(f'Original Point Cloud\n({len(original_xyz)} points)')
    ax1.axis('equal')
    
    # 过滤后点云（鸟瞰图）
    if labels is not None:
        unique_labels = np.unique(labels)
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
        
        for i, label in enumerate(unique_labels):
            mask = labels == label
            ax2.scatter(filtered_xyz[mask, 0], filtered_xyz[mask, 2], 
                       c=[colors[i]], label=f'Class {label}', s=0.5)
        ax2.legend()
    else:
        ax2.scatter(filtered_xyz[:, 0], filtered_xyz[:, 2], c='blue', s=0.5)
    
    ax2.set_xlabel('X')
    ax2.set_ylabel('Z')
    ax2.set_title(f'Filtered Point Cloud with Labels\n({len(filtered_xyz)} points)')
    ax2.axis('equal')
    
    plt.tight_layout()
    plt.show()
    
    # 打印统计信息
    print(f"Original points: {len(original_xyz)}")
    print(f"Filtered points: {len(filtered_xyz)}")
    print(f"Retention rate: {len(filtered_xyz)/len(original_xyz)*100:.1f}%")
    
    if labels is not None:
        unique_labels, counts = np.unique(labels, return_counts=True)
        print("\nLabel distribution:")
        for label, count in zip(unique_labels, counts):
            print(f"  Class {label}: {count} points ({count/len(filtered_xyz)*100:.1f}%)")

def main():
    parser = argparse.ArgumentParser(description="Visualize converted point cloud segmentation")
    parser.add_argument("--dataset_root", type=str, required=True,
                      help="Path to dataset root directory")
    parser.add_argument("--sequence", type=str, default="00",
                      help="Sequence ID")
    parser.add_argument("--frame", type=str, default="000000",
                      help="Frame ID (e.g., 000000)")
    parser.add_argument("--compare", action="store_true",
                      help="Compare original and filtered point clouds")
    
    args = parser.parse_args()
    
    sequence_path = os.path.join(args.dataset_root, "sequences", args.sequence)
    
    # 文件路径
    filtered_velodyne = os.path.join(sequence_path, "velodyne_filtered", f"{args.frame}.bin")
    label_file = os.path.join(sequence_path, "labels_segmentation", f"{args.frame}.label")
    
    if args.compare:
        original_velodyne = os.path.join(sequence_path, "velodyne", f"{args.frame}.bin")
        compare_before_after(original_velodyne, filtered_velodyne, label_file)
    else:
        visualize_labeled_pointcloud(filtered_velodyne, label_file)

if __name__ == "__main__":
    main()
"""
python3 visualize_segmentation.py \
    --dataset_root "/media/gyc/Backup Plus3/gyc/thesis/raw_demo_rosbag/dataset" \
    --sequence "00" \
    --frame "000000" \
    --compare
"""