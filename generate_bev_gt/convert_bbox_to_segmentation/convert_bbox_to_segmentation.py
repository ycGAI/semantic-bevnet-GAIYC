#!/usr/bin/env python3
"""
将3D边界框标注转换为点云分割标注，并过滤掉bbox外的点云
"""

import os
import numpy as np
import glob
from pathlib import Path
import argparse

def parse_annotation_line(line):
    """
    解析标注文件中的一行
    KITTI 3D Object格式: type truncated occluded alpha bbox_2d dimensions location rotation_y [score]
    
    对于您的格式，假设是：
    class_id truncated occluded alpha x1 y1 x2 y2 ? h w l x y z ry [score]
    """
    parts = line.strip().split()
    if len(parts) < 15:
        return None
    
    annotation = {
        'class_id': int(parts[0]),  # 类别ID
        'dimensions': [float(parts[8]), float(parts[9]), float(parts[10])],  # h, w, l
        'location': [float(parts[11]), float(parts[12]), float(parts[13])],   # x, y, z
        'rotation_y': float(parts[14])  # 绕Y轴旋转角度
    }
    return annotation

def rotation_matrix_y(angle):
    """绕Y轴旋转矩阵"""
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)
    return np.array([
        [cos_a,  0, sin_a],
        [0,      1, 0],
        [-sin_a, 0, cos_a]
    ])

def point_in_bbox_3d(point, bbox_center, bbox_size, rotation_y):
    """
    检查点是否在3D边界框内
    
    Args:
        point: 点坐标 [x, y, z]
        bbox_center: 边界框中心 [x, y, z]  
        bbox_size: 边界框尺寸 [h, w, l]
        rotation_y: 绕Y轴旋转角度
    """
    # 将点转换到bbox的局部坐标系
    rel_point = point - np.array(bbox_center)
    
    # 应用逆旋转
    R_inv = rotation_matrix_y(-rotation_y)
    local_point = R_inv @ rel_point
    
    # 检查是否在边界框内
    h, w, l = bbox_size
    return (abs(local_point[0]) <= l/2 and 
            abs(local_point[1]) <= h/2 and 
            abs(local_point[2]) <= w/2)

def process_pointcloud_with_labels(velodyne_path, label_path, output_velodyne_path, output_label_path):
    """
    处理单个点云文件，生成分割标注和过滤后的点云
    """
    # 读取点云数据
    if not os.path.exists(velodyne_path):
        print(f"Warning: Point cloud file not found: {velodyne_path}")
        return
    
    points = np.fromfile(velodyne_path, dtype=np.float32).reshape(-1, 4)
    xyz_points = points[:, :3]
    
    # 初始化标签（0表示无标签/背景）
    point_labels = np.zeros(len(xyz_points), dtype=np.uint32)
    
    # 读取边界框标注
    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            for line in f:
                annotation = parse_annotation_line(line)
                if annotation is None:
                    continue
                
                # 为每个点检查是否在当前边界框内
                for i, point in enumerate(xyz_points):
                    if point_in_bbox_3d(point, 
                                       annotation['location'], 
                                       annotation['dimensions'], 
                                       annotation['rotation_y']):
                        point_labels[i] = annotation['class_id']
    
    # 过滤点云：只保留有标签的点
    valid_mask = point_labels > 0
    filtered_points = points[valid_mask]
    filtered_labels = point_labels[valid_mask]
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_velodyne_path), exist_ok=True)
    os.makedirs(os.path.dirname(output_label_path), exist_ok=True)
    
    # 保存过滤后的点云
    filtered_points.tofile(output_velodyne_path)
    
    # 保存标签文件（KITTI格式）
    filtered_labels.astype(np.uint32).tofile(output_label_path)
    
    print(f"Processed: {os.path.basename(velodyne_path)} - "
          f"Original: {len(points)} points, "
          f"Filtered: {len(filtered_points)} points "
          f"({len(filtered_points)/len(points)*100:.1f}%)")

def convert_dataset(dataset_root, sequence_id="00"):
    """
    转换整个序列的数据
    """
    sequence_path = os.path.join(dataset_root, "sequences", sequence_id)
    
    # 输入路径
    velodyne_dir = os.path.join(sequence_path, "velodyne")
    labels_dir = os.path.join(sequence_path, "labels")
    
    # 输出路径
    output_velodyne_dir = os.path.join(sequence_path, "velodyne_filtered")
    output_labels_dir = os.path.join(sequence_path, "labels_segmentation")
    
    # 获取所有点云文件
    velodyne_files = sorted(glob.glob(os.path.join(velodyne_dir, "*.bin")))
    
    print(f"Found {len(velodyne_files)} point cloud files in sequence {sequence_id}")
    
    for velodyne_file in velodyne_files:
        # 获取对应的标注文件
        base_name = os.path.splitext(os.path.basename(velodyne_file))[0]
        label_file = os.path.join(labels_dir, base_name + ".txt")
        
        # 输出文件路径
        output_velodyne_file = os.path.join(output_velodyne_dir, base_name + ".bin")
        output_label_file = os.path.join(output_labels_dir, base_name + ".label")
        
        # 处理文件
        process_pointcloud_with_labels(
            velodyne_file, label_file, 
            output_velodyne_file, output_label_file
        )

def main():
    parser = argparse.ArgumentParser(description="Convert 3D bbox annotations to point cloud segmentation")
    parser.add_argument("--dataset_root", type=str, required=True,
                      help="Path to dataset root directory")
    parser.add_argument("--sequence", type=str, default="00", 
                      help="Sequence ID to process (default: 00)")
    parser.add_argument("--all_sequences", action="store_true",
                      help="Process all sequences in the dataset")
    
    args = parser.parse_args()
    
    if args.all_sequences:
        # 处理所有序列
        sequences_dir = os.path.join(args.dataset_root, "sequences")
        if os.path.exists(sequences_dir):
            sequences = [d for d in os.listdir(sequences_dir) 
                        if os.path.isdir(os.path.join(sequences_dir, d))]
            sequences.sort()
            
            for seq in sequences:
                print(f"\n=== Processing sequence {seq} ===")
                convert_dataset(args.dataset_root, seq)
        else:
            print(f"Sequences directory not found: {sequences_dir}")
    else:
        # 处理单个序列
        print(f"=== Processing sequence {args.sequence} ===")
        convert_dataset(args.dataset_root, args.sequence)
    
    print("\n=== Conversion completed ===")

if __name__ == "__main__":
    main()