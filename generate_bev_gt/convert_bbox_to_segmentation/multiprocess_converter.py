#!/usr/bin/env python3
"""
多进程版本的3D边界框转点云分割工具
支持并行处理，显著提高处理速度
"""

import os
import numpy as np
import glob
import json
import logging
from pathlib import Path
import argparse
import yaml
from datetime import datetime
import multiprocessing as mp
from functools import partial
import time

def process_single_file(args):
    """
    处理单个文件的函数（用于多进程）
    """
    velodyne_file, label_file, output_velodyne_file, output_label_file, config = args
    
    try:
        # 解析配置
        annotation_format = config['annotation_format']
        filtering = config['filtering']
        class_mapping = config.get('class_mapping')
        
        # 读取点云数据
        if not os.path.exists(velodyne_file):
            return {
                'success': False,
                'file': os.path.basename(velodyne_file),
                'error': 'Point cloud file not found'
            }
        
        points = np.fromfile(velodyne_file, dtype=np.float32).reshape(-1, 4)
        
        # 过滤无效点（NaN和Inf）
        valid_mask = np.isfinite(points).all(axis=1)
        invalid_count = 0
        if not valid_mask.all():
            invalid_count = (~valid_mask).sum()
            points = points[valid_mask]
        
        xyz_points = points[:, :3]
        original_count = len(xyz_points)
        
        # 初始化标签
        point_labels = np.zeros(len(xyz_points), dtype=np.uint32)
        
        # 处理边界框标注
        bbox_count = 0
        if os.path.exists(label_file):
            with open(label_file, 'r') as f:
                for line in f:
                    annotation = parse_annotation_line(line, annotation_format, class_mapping)
                    if annotation is None:
                        continue
                    
                    bbox_count += 1
                    points_in_bbox = 0
                    
                    # 为每个点检查是否在当前边界框内
                    for i, point in enumerate(xyz_points):
                        if point_in_bbox_3d(point, 
                                           annotation['location'], 
                                           annotation['dimensions'], 
                                           annotation['rotation_y'],
                                           filtering['bbox_expansion_factor']):
                            point_labels[i] = annotation['class_id']
                            points_in_bbox += 1
                    
                    # 检查最小点数阈值
                    if points_in_bbox < filtering['min_points_per_bbox']:
                        pass  # 可以记录日志，但在多进程中暂时忽略
        
        # 过滤点云
        if filtering['keep_only_labeled_points']:
            valid_mask = point_labels > 0
            filtered_points = points[valid_mask]
            filtered_labels = point_labels[valid_mask]
        else:
            filtered_points = points
            filtered_labels = point_labels
        
        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_velodyne_file), exist_ok=True)
        os.makedirs(os.path.dirname(output_label_file), exist_ok=True)
        
        # 保存文件
        filtered_points.tofile(output_velodyne_file)
        filtered_labels.astype(np.uint32).tofile(output_label_file)
        
        # 统计信息
        unique_labels, counts = np.unique(filtered_labels[filtered_labels > 0], return_counts=True)
        class_distribution = {}
        for label, count in zip(unique_labels, counts):
            class_distribution[int(label)] = int(count)
        
        return {
            'success': True,
            'file': os.path.basename(velodyne_file),
            'original_count': original_count,
            'filtered_count': len(filtered_points),
            'invalid_count': invalid_count,
            'bbox_count': bbox_count,
            'class_distribution': class_distribution
        }
        
    except Exception as e:
        return {
            'success': False,
            'file': os.path.basename(velodyne_file),
            'error': str(e)
        }

def parse_annotation_line(line, annotation_format, class_mapping):
    """解析标注文件中的一行"""
    parts = line.strip().split()
    
    if len(parts) < annotation_format['min_fields']:
        return None
    
    try:
        annotation = {
            'class_id': int(parts[annotation_format['class_id_index']]),
            'dimensions': [
                float(parts[annotation_format['height_index']]),
                float(parts[annotation_format['width_index']]),
                float(parts[annotation_format['length_index']])
            ],
            'location': [
                float(parts[annotation_format['x_index']]),
                float(parts[annotation_format['y_index']]),
                float(parts[annotation_format['z_index']])
            ],
            'rotation_y': float(parts[annotation_format['rotation_y_index']])
        }
        
        # 应用类别映射
        if class_mapping:
            original_class = annotation['class_id']
            if original_class in class_mapping:
                annotation['class_id'] = class_mapping[original_class]
            else:
                return None  # 忽略未映射的类别
        
        return annotation
    except (ValueError, IndexError):
        return None

def rotation_matrix_y(angle):
    """绕Y轴旋转矩阵"""
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)
    return np.array([
        [cos_a,  0, sin_a],
        [0,      1, 0],
        [-sin_a, 0, cos_a]
    ])

def point_in_bbox_3d(point, bbox_center, bbox_size, rotation_y, expansion_factor=1.0):
    """检查点是否在3D边界框内"""
    # 应用边界框扩展因子
    bbox_size = [s * expansion_factor for s in bbox_size]
    
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

class MultiprocessConverter:
    def __init__(self, config_path=None, num_processes=None):
        """初始化多进程转换器"""
        # 加载配置
        self.config = self.load_config(config_path)
        
        # 设置进程数
        if num_processes is None:
            num_processes = self.config.get('performance', {}).get('num_processes', 0)
        
        if num_processes <= 0:
            num_processes = mp.cpu_count()
        
        self.num_processes = min(num_processes, mp.cpu_count())
        
        # 设置日志
        self.setup_logging()
        
        # 统计信息
        self.stats = {
            'total_files_processed': 0,
            'total_original_points': 0,
            'total_filtered_points': 0,
            'total_invalid_points': 0,
            'class_distribution': {},
            'processing_time': 0,
            'start_time': datetime.now().isoformat(),
            'num_processes_used': self.num_processes
        }
    
    def load_config(self, config_path):
        """加载配置文件"""
        # 默认配置
        default_config = {
            'class_mapping': None,
            'annotation_format': {
                'class_id_index': 0,
                'height_index': 8,
                'width_index': 9,
                'length_index': 10,
                'x_index': 11,
                'y_index': 12,
                'z_index': 13,
                'rotation_y_index': 14,
                'min_fields': 15
            },
            'filtering': {
                'keep_only_labeled_points': True,
                'bbox_expansion_factor': 1.0,
                'min_points_per_bbox': 10
            },
            'output': {
                'filtered_pointcloud_dir': 'velodyne_filtered',
                'segmentation_labels_dir': 'labels_segmentation',
                'save_statistics': True,
                'statistics_file': 'conversion_stats.json'
            },
            'logging': {
                'level': 'INFO',
                'save_to_file': True,
                'log_file': 'conversion.log',
                'progress_interval': 100
            },
            'performance': {
                'use_multiprocessing': True,
                'num_processes': 0
            }
        }
        
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                user_config = yaml.safe_load(f)
            # 递归更新配置
            self.update_dict(default_config, user_config)
        
        return default_config
    
    def update_dict(self, base_dict, update_dict):
        """递归更新字典"""
        for key, value in update_dict.items():
            if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                self.update_dict(base_dict[key], value)
            else:
                base_dict[key] = value
    
    def setup_logging(self):
        """设置日志"""
        log_config = self.config['logging']
        log_level = getattr(logging, log_config['level'].upper())
        
        # 创建logger
        self.logger = logging.getLogger('MultiprocessConverter')
        self.logger.setLevel(log_level)
        
        # 清除现有handlers
        self.logger.handlers.clear()
        
        # 创建formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # 控制台handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
        # 文件handler
        if log_config['save_to_file']:
            file_handler = logging.FileHandler(log_config['log_file'])
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
    
    def convert_sequence(self, dataset_root, sequence_id):
        """转换单个序列（多进程版本）"""
        sequence_path = os.path.join(dataset_root, "sequences", sequence_id)
        
        # 输入目录
        velodyne_dir = os.path.join(sequence_path, "velodyne")
        labels_dir = os.path.join(sequence_path, "labels")
        
        # 输出目录
        output_config = self.config['output']
        output_velodyne_dir = os.path.join(sequence_path, output_config['filtered_pointcloud_dir'])
        output_labels_dir = os.path.join(sequence_path, output_config['segmentation_labels_dir'])
        
        # 获取所有点云文件
        velodyne_files = sorted(glob.glob(os.path.join(velodyne_dir, "*.bin")))
        
        if not velodyne_files:
            self.logger.warning(f"No point cloud files found in sequence {sequence_id}")
            return
        
        self.logger.info(f"Processing sequence {sequence_id}: {len(velodyne_files)} files using {self.num_processes} processes")
        
        # 准备任务列表
        tasks = []
        for velodyne_file in velodyne_files:
            base_name = os.path.splitext(os.path.basename(velodyne_file))[0]
            label_file = os.path.join(labels_dir, base_name + ".txt")
            output_velodyne_file = os.path.join(output_velodyne_dir, base_name + ".bin")
            output_label_file = os.path.join(output_labels_dir, base_name + ".label")
            
            tasks.append((velodyne_file, label_file, output_velodyne_file, output_label_file, self.config))
        
        # 多进程处理
        start_time = time.time()
        
        # 进度报告间隔
        progress_interval = max(1, len(tasks) // 20)  # 每5%报告一次进度
        
        success_count = 0
        with mp.Pool(processes=self.num_processes) as pool:
            # 创建进度条
            results = []
            for i, result in enumerate(pool.imap(process_single_file, tasks)):
                results.append(result)
                
                if result['success']:
                    success_count += 1
                    # 更新统计信息
                    self.update_statistics_from_result(result)
                else:
                    self.logger.error(f"Failed to process {result['file']}: {result.get('error', 'Unknown error')}")
                
                # 进度报告
                if (i + 1) % progress_interval == 0 or i == len(tasks) - 1:
                    elapsed = time.time() - start_time
                    progress = (i + 1) / len(tasks) * 100
                    speed = (i + 1) / elapsed
                    eta = (len(tasks) - i - 1) / speed if speed > 0 else 0
                    self.logger.info(f"Progress: {i + 1}/{len(tasks)} ({progress:.1f}%) - "
                                   f"Speed: {speed:.1f} files/s - ETA: {eta:.0f}s")
        
        processing_time = time.time() - start_time
        self.logger.info(f"Sequence {sequence_id} completed in {processing_time:.1f}s: "
                        f"{success_count}/{len(velodyne_files)} files processed successfully")
    
    def update_statistics_from_result(self, result):
        """从单个文件的处理结果更新统计信息"""
        self.stats['total_files_processed'] += 1
        self.stats['total_original_points'] += result['original_count']
        self.stats['total_filtered_points'] += result['filtered_count']
        self.stats['total_invalid_points'] += result.get('invalid_count', 0)
        
        # 更新类别分布
        for class_id, count in result.get('class_distribution', {}).items():
            if class_id not in self.stats['class_distribution']:
                self.stats['class_distribution'][class_id] = 0
            self.stats['class_distribution'][class_id] += count
    
    def save_statistics(self, output_dir):
        """保存统计信息"""
        if not self.config['output']['save_statistics']:
            return
        
        # 完成统计信息
        self.stats['end_time'] = datetime.now().isoformat()
        self.stats['retention_rate'] = (
            self.stats['total_filtered_points'] / self.stats['total_original_points'] 
            if self.stats['total_original_points'] > 0 else 0
        )
        
        # 转换numpy数据类型为Python原生类型（修复JSON序列化问题）
        stats_to_save = {}
        for key, value in self.stats.items():
            if isinstance(value, dict):
                # 转换字典中的值
                stats_to_save[key] = {k: int(v) if isinstance(v, (np.integer, np.int64)) else v 
                                    for k, v in value.items()}
            elif isinstance(value, (np.integer, np.int64)):
                stats_to_save[key] = int(value)
            elif isinstance(value, (np.floating, np.float64)):
                stats_to_save[key] = float(value)
            else:
                stats_to_save[key] = value
        
        # 保存到文件
        stats_file = os.path.join(output_dir, self.config['output']['statistics_file'])
        os.makedirs(os.path.dirname(stats_file), exist_ok=True)
        
        with open(stats_file, 'w') as f:
            json.dump(stats_to_save, f, indent=2)
        
        self.logger.info(f"Statistics saved to: {stats_file}")
    
    def convert_dataset(self, dataset_root, sequence_ids=None):
        """转换整个数据集或指定序列"""
        start_time = datetime.now()
        
        if sequence_ids is None:
            # 获取所有序列
            sequences_dir = os.path.join(dataset_root, "sequences")
            if not os.path.exists(sequences_dir):
                self.logger.error(f"Sequences directory not found: {sequences_dir}")
                return
            
            sequence_ids = [d for d in os.listdir(sequences_dir) 
                           if os.path.isdir(os.path.join(sequences_dir, d))]
            sequence_ids.sort()
        
        self.logger.info(f"Starting conversion of {len(sequence_ids)} sequences using {self.num_processes} processes")
        
        # 处理每个序列
        for seq_id in sequence_ids:
            self.convert_sequence(dataset_root, seq_id)
        
        # 计算处理时间
        end_time = datetime.now()
        self.stats['processing_time'] = (end_time - start_time).total_seconds()
        
        # 保存统计信息
        self.save_statistics(dataset_root)
        
        # 打印最终统计
        self.print_final_statistics()
    
    def print_final_statistics(self):
        """打印最终统计信息"""
        stats = self.stats
        
        self.logger.info("=== Conversion Summary ===")
        self.logger.info(f"Files processed: {stats['total_files_processed']}")
        self.logger.info(f"Original points: {stats['total_original_points']:,}")
        self.logger.info(f"Filtered points: {stats['total_filtered_points']:,}")
        self.logger.info(f"Invalid points: {stats['total_invalid_points']:,}")
        self.logger.info(f"Retention rate: {stats.get('retention_rate', 0)*100:.1f}%")
        self.logger.info(f"Processing time: {stats['processing_time']:.1f} seconds")
        self.logger.info(f"Average speed: {stats['total_files_processed']/stats['processing_time']:.1f} files/s")
        self.logger.info(f"Processes used: {stats['num_processes_used']}")
        
        if stats['class_distribution']:
            self.logger.info("Class distribution:")
            for class_id, count in sorted(stats['class_distribution'].items()):
                percentage = count / stats['total_filtered_points'] * 100
                self.logger.info(f"  Class {class_id}: {count:,} points ({percentage:.1f}%)")

def main():
    parser = argparse.ArgumentParser(description="Multiprocess 3D bbox to point cloud segmentation converter")
    parser.add_argument("--dataset_root", type=str, required=True,
                      help="Path to dataset root directory")
    parser.add_argument("--sequence", type=str,
                      help="Process specific sequence")
    parser.add_argument("--all_sequences", action="store_true",
                      help="Process all sequences in the dataset")
    parser.add_argument("--config", type=str,
                      help="Path to configuration file")
    parser.add_argument("--processes", type=int,
                      help="Number of processes to use (default: CPU count)")
    
    args = parser.parse_args()
    
    # 创建转换器
    converter = MultiprocessConverter(args.config, args.processes)
    
    # 确定要处理的序列
    if args.all_sequences:
        sequence_ids = None  # 处理所有序列
    elif args.sequence:
        sequence_ids = [args.sequence]
    else:
        # 默认处理序列00
        sequence_ids = ["00"]
        converter.logger.info("No sequence specified, processing sequence 00")
    
    # 执行转换
    converter.convert_dataset(args.dataset_root, sequence_ids)

if __name__ == "__main__":
    main()