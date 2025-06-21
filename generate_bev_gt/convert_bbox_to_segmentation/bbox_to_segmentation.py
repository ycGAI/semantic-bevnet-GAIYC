#!/usr/bin/env python3
"""
3D边界框转点云分割标签生成器
将KITTI格式的3D边界框标注转换为点云级别的分割标签
同时生成过滤后的点云数据（只保留bbox内的点）
"""

import numpy as np
import argparse
import os
from pathlib import Path
import struct
from tqdm import tqdm

# 地面穿越性类别映射
TRAVERSABILITY_CLASSES = {
    1: {'name': 'Traversable', 'label': 1},      # 可通行区域
    2: {'name': 'Mid-cost', 'label': 2},         # 中等代价区域  
    3: {'name': 'High-cost', 'label': 3},        # 高代价区域
    4: {'name': 'Barrier', 'label': 4}           # 障碍物/不可通行
}

class BBoxToSegmentationConverter:
    def __init__(self, input_path, output_path, sequence="00"):
        """
        初始化转换器
        
        Args:
            input_path: 输入数据集路径
            output_path: 输出数据集路径  
            sequence: 序列号
        """
        self.input_path = Path(input_path)
        self.output_path = Path(output_path)
        self.sequence = sequence
        
        # 输入路径
        if (self.input_path / "sequences" / sequence).exists():
            self.input_sequence_path = self.input_path / "sequences" / sequence
        else:
            self.input_sequence_path = self.input_path
            
        # 输出路径
        self.output_sequence_path = self.output_path / "sequences" / sequence
        
        # 创建输出目录
        self.setup_output_directories()
        
        print(f"🎯 输入路径: {self.input_sequence_path}")
        print(f"📁 输出路径: {self.output_sequence_path}")
        
    def setup_output_directories(self):
        """创建输出目录结构"""
        directories = [
            "velodyne",           # 过滤后的点云
            "labels_needed",             # bbox内点云对应的标签
            "semantic_labels",    # 全部的点云标签
        ]
        
        for dir_name in directories:
            (self.output_sequence_path / dir_name).mkdir(parents=True, exist_ok=True)
        
        print(f"✅ 创建输出目录: {self.output_sequence_path}")
    
    def load_point_cloud(self, frame_id):
        """加载点云数据"""
        velodyne_file = self.input_sequence_path / "velodyne" / f"{frame_id:06d}.bin"
        
        if not velodyne_file.exists():
            return None
        
        try:
            points = np.fromfile(velodyne_file, dtype=np.float32).reshape(-1, 4)
            return points
        except Exception as e:
            print(f"❌ 加载点云失败 {frame_id:06d}: {e}")
            return None
    
    def load_bboxes(self, frame_id):
        """加载3D边界框标注"""
        label_file = self.input_sequence_path / "labels" / f"{frame_id:06d}.txt"
        
        if not label_file.exists():
            return []
        
        bboxes = []
        try:
            with open(label_file, 'r') as f:
                for line_num, line in enumerate(f, 1):
                    parts = line.strip().split()
                    if len(parts) >= 15:
                        try:
                            bbox = {
                                'type': int(parts[0]),
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
                            
                            if bbox['type'] in TRAVERSABILITY_CLASSES:
                                bboxes.append(bbox)
                            
                        except ValueError as e:
                            print(f"⚠️  第{line_num}行解析错误: {e}")
                            continue
                            
        except Exception as e:
            print(f"❌ 读取标注文件失败: {e}")
            return []
        
        return bboxes
    
    def expand_bbox_vertically(self, bbox, expand_up=0.5, expand_down=0.3):
        """
        在垂直方向上扩大边界框
        
        Args:
            bbox: 原始边界框
            expand_up: 向上扩展距离(米)
            expand_down: 向下扩展距离(米)
        """
        expanded_bbox = bbox.copy()
        expanded_bbox['h'] = bbox['h'] + expand_up + expand_down
        expanded_bbox['z'] = bbox['z'] - expand_down  # 底面向下移动
        
        return expanded_bbox
    
    def point_in_bbox(self, points, bbox):
        """
        检查点是否在3D边界框内
        
        Args:
            points: 点云数组 (N, 3) 或 (N, 4)
            bbox: 边界框参数字典
            
        Returns:
            mask: 布尔数组，True表示点在边界框内
        """
        # 提取点的3D坐标
        if points.shape[1] >= 3:
            pts = points[:, :3]
        else:
            raise ValueError("点云数据至少需要3个坐标")
        
        # 边界框参数
        x, y, z = bbox['x'], bbox['y'], bbox['z']
        h, w, l = bbox['h'], bbox['w'], bbox['l']
        ry = bbox['ry']
        
        # 将点转换到边界框坐标系
        # 1. 平移到边界框中心
        pts_translated = pts - np.array([x, y, z + h/2])
        
        # 2. 绕Y轴旋转（CVAT格式）
        cos_ry = np.cos(-ry)  # 注意：这里用负角度进行逆旋转
        sin_ry = np.sin(-ry)
        
        # 旋转矩阵（绕Y轴）
        R_inv = np.array([
            [cos_ry, 0, sin_ry],
            [0, 1, 0],
            [-sin_ry, 0, cos_ry]
        ])
        
        pts_rotated = pts_translated @ R_inv.T
        
        # 3. 检查是否在边界框范围内
        mask_x = np.abs(pts_rotated[:, 0]) <= w / 2
        mask_y = np.abs(pts_rotated[:, 1]) <= l / 2  
        mask_z = np.abs(pts_rotated[:, 2]) <= h / 2
        
        return mask_x & mask_y & mask_z
    
    def process_frame(self, frame_id, expand_up=0.5, expand_down=0.3, priority_order=None, filter_background=True):
        """
        处理单帧数据，生成分割标签和过滤点云
        
        Args:
            frame_id: 帧ID
            expand_up: 向上扩展距离
            expand_down: 向下扩展距离  
            priority_order: 类别优先级顺序（解决重叠问题）
            filter_background: 是否过滤背景点
        """
        # 加载数据
        points = self.load_point_cloud(frame_id)
        bboxes = self.load_bboxes(frame_id)
        
        if points is None:
            print(f"⚠️  跳过帧 {frame_id:06d}: 无点云数据")
            return False, {}
            
        if not bboxes:
            print(f"⚠️  跳过帧 {frame_id:06d}: 无边界框标注")
            return False, {}
        
        print(f"🔄 处理帧 {frame_id:06d}: {len(points):,} 点, {len(bboxes)} 个边界框")
        
        # 初始化标签（0表示背景/未标注）
        labels = np.zeros(len(points), dtype=np.int32)
        in_any_bbox = np.zeros(len(points), dtype=bool)
        
        # 设置优先级顺序：1 < 2 < 3 < 4 （数字越大优先级越高）
        if priority_order is None:
            priority_order = [1, 2, 3, 4]
        
        print(f"   🏆 优先级顺序: {' < '.join(map(str, priority_order))} (右侧优先级更高)")
        print(f"   🗂️  过滤模式: {'过滤背景点' if filter_background else '保留所有点'}")
        
        # 统计每个边界框包含的点数（用于重叠分析）
        bbox_point_counts = {}
        overlap_stats = {}
        
        # 按优先级从低到高处理边界框（高优先级会覆盖低优先级）
        sorted_bboxes = sorted(bboxes, key=lambda x: priority_order.index(x['type']) if x['type'] in priority_order else -1)
        
        for i, bbox in enumerate(sorted_bboxes):
            if bbox['type'] not in priority_order:
                continue
                
            # 扩展边界框
            expanded_bbox = self.expand_bbox_vertically(bbox, expand_up, expand_down)
            
            # 找到在边界框内的点
            mask = self.point_in_bbox(points, expanded_bbox)
            point_count = np.sum(mask)
            
            # 统计与已有标签的重叠
            already_labeled_mask = (labels > 0) & mask
            overlap_count = np.sum(already_labeled_mask)
            new_points = point_count - overlap_count
            
            # 记录重叠信息
            if overlap_count > 0:
                overlapped_labels = np.unique(labels[already_labeled_mask])
                overlap_info = []
                for prev_label in overlapped_labels:
                    if prev_label > 0:
                        prev_count = np.sum((labels == prev_label) & mask)
                        prev_name = TRAVERSABILITY_CLASSES[prev_label]['name']
                        overlap_info.append(f"{prev_count} from {prev_label}({prev_name})")
                
                overlap_stats[bbox['type']] = {
                    'total_overlap': overlap_count,
                    'details': overlap_info
                }
            
            # 分配标签（高优先级会覆盖低优先级）
            labels[mask] = bbox['type']
            in_any_bbox[mask] = True
            
            bbox_point_counts[bbox['type']] = bbox_point_counts.get(bbox['type'], 0) + point_count
            
            class_name = TRAVERSABILITY_CLASSES[bbox['type']]['name']
            priority_level = priority_order.index(bbox['type'])
            
            print(f"   📦 边界框 {i+1}: 类别 {bbox['type']} ({class_name}), 优先级 {priority_level}")
            print(f"      包含点数: {point_count:,}")
            if overlap_count > 0:
                print(f"      重叠覆盖: {overlap_count:,} 个点 ({', '.join(overlap_stats[bbox['type']]['details'])})")
                print(f"      新增点数: {new_points:,}")
        
        # 统计最终标签分布
        unique_labels, counts = np.unique(labels, return_counts=True)
        total_labeled = np.sum(in_any_bbox)
        
        frame_stats = {}
        
        print(f"\n   📊 最终标签统计:")
        for label, count in zip(unique_labels, counts):
            if label == 0:
                if not filter_background:  # 只有保留背景时才统计
                    print(f"      背景: {count:,} 个点 ({count/len(points)*100:.1f}%)")
                    frame_stats[0] = count
            else:
                class_name = TRAVERSABILITY_CLASSES.get(label, {}).get('name', f'Class_{label}')
                priority_level = priority_order.index(label) if label in priority_order else -1
                print(f"      {label} ({class_name}): {count:,} 个点 ({count/len(points)*100:.1f}%) [优先级: {priority_level}]")
                frame_stats[label] = count
        
        print(f"   📈 总标注覆盖率: {total_labeled:,}/{len(points):,} ({total_labeled/len(points)*100:.1f}%)")
        
        # 重叠统计摘要
        if overlap_stats:
            print(f"\n   🔄 重叠处理摘要:")
            for class_id, stats in overlap_stats.items():
                class_name = TRAVERSABILITY_CLASSES[class_id]['name']
                print(f"      {class_id} ({class_name}) 覆盖了 {stats['total_overlap']:,} 个低优先级点")
        
        # 保存分割标签
        if filter_background:
            # 只保存bbox内的点和标签
            filtered_points = points[in_any_bbox]
            filtered_labels = labels[in_any_bbox]
            
            if len(filtered_points) > 0:
                self.save_filtered_pointcloud(frame_id, filtered_points)
                self.save_filtered_labels(frame_id, filtered_labels)
                self.save_segmentation_labels(frame_id, filtered_labels)  # 保存过滤后的标签
                
                print(f"   💾 保存过滤后点云: {len(filtered_points):,} 个点 (所有点都有标签)")
            else:
                print(f"   ⚠️  没有点在边界框内")
        else:
            # 保存完整点云和对应标签
            self.save_filtered_pointcloud(frame_id, points)
            self.save_segmentation_labels(frame_id, labels)  # 包含背景标签
            
            # 同时保存仅标注点的版本
            if total_labeled > 0:
                labeled_points = points[in_any_bbox]
                labeled_labels = labels[in_any_bbox]
                self.save_filtered_labels(frame_id, labeled_labels)
                
                print(f"   💾 保存完整点云: {len(points):,} 个点 (包含 {len(points)-total_labeled:,} 个背景点)")
                print(f"   💾 保存标注点: {total_labeled:,} 个点 (无背景)")
            else:
                print(f"   ⚠️  没有点在边界框内")
        
        return True, frame_stats
    
    def save_segmentation_labels(self, frame_id, labels):
        """保存完整点云的分割标签"""
        label_file = self.output_sequence_path / "semantic_labels" / f"{frame_id:06d}.label"
        
        # 转换为uint32格式保存
        labels_uint32 = labels.astype(np.uint32)
        labels_uint32.tofile(label_file)
    
    def save_filtered_pointcloud(self, frame_id, points):
        """保存过滤后的点云"""
        velodyne_file = self.output_sequence_path / "velodyne" / f"{frame_id:06d}.bin"
        points.astype(np.float32).tofile(velodyne_file)
    
    def save_filtered_labels(self, frame_id, labels):
        """保存过滤后点云对应的标签"""
        label_file = self.output_sequence_path / "labels" / f"{frame_id:06d}.label"
        labels.astype(np.uint32).tofile(label_file)
    
    def convert_dataset(self, expand_up=0.5, expand_down=0.3, priority_order=None, filter_background=True):
        """转换整个数据集"""
        print(f"🚀 开始转换数据集")
        print(f"📏 垂直扩展: 向上 {expand_up}m, 向下 {expand_down}m")
        print(f"🗂️  过滤模式: {'过滤背景点' if filter_background else '保留所有点'}")
        
        if priority_order is None:
            priority_order = [1, 2, 3, 4]
            
        print(f"🏆 类别优先级: {' < '.join(map(str, priority_order))} (数字越大优先级越高)")
        
        # 找到所有点云文件
        velodyne_files = list(self.input_sequence_path.glob("velodyne/*.bin"))
        velodyne_files.sort()
        
        if not velodyne_files:
            print("❌ 没有找到点云文件")
            return
        
        print(f"📄 找到 {len(velodyne_files)} 个点云文件")
        
        success_count = 0
        failed_count = 0
        
        # 全局统计
        global_stats = {0: 0}  # 背景
        for class_id in TRAVERSABILITY_CLASSES.keys():
            global_stats[class_id] = 0
        
        total_points = 0
        total_labeled_points = 0
        
        # 处理每一帧
        for velodyne_file in tqdm(velodyne_files, desc="转换进度"):
            frame_id = int(velodyne_file.stem)
            
            try:
                success, frame_stats = self.process_frame(frame_id, expand_up, expand_down, priority_order, filter_background)
                if success:
                    success_count += 1
                    # 累加统计
                    for label, count in frame_stats.items():
                        global_stats[label] = global_stats.get(label, 0) + count
                        if label > 0:
                            total_labeled_points += count
                        total_points += count
                else:
                    failed_count += 1
            except Exception as e:
                print(f"❌ 处理帧 {frame_id:06d} 失败: {e}")
                failed_count += 1
        
        print(f"\n✅ 转换完成!")
        print(f"   成功: {success_count} 帧")
        print(f"   失败: {failed_count} 帧")
        print(f"   输出目录: {self.output_sequence_path}")
        
        # 打印全局统计
        print(f"\n📊 全数据集统计:")
        print(f"   总点数: {total_points:,}")
        print(f"   标注点数: {total_labeled_points:,} ({total_labeled_points/total_points*100:.1f}%)")
        if not filter_background:
            print(f"   背景点数: {global_stats[0]:,} ({global_stats[0]/total_points*100:.1f}%)")
        
        print(f"\n🏷️  各类别点数统计:")
        for class_id in sorted(TRAVERSABILITY_CLASSES.keys()):
            count = global_stats.get(class_id, 0)
            class_name = TRAVERSABILITY_CLASSES[class_id]['name']
            percentage = count / total_points * 100 if total_points > 0 else 0
            labeled_percentage = count / total_labeled_points * 100 if total_labeled_points > 0 else 0
            priority_level = priority_order.index(class_id) if class_id in priority_order else -1
            
            print(f"   {class_id} ({class_name}): {count:,} 个点")
            print(f"      占总点数: {percentage:.2f}%")
            print(f"      占标注点数: {labeled_percentage:.2f}%")
            print(f"      优先级: {priority_level}")
            print()
        
        # 生成统计报告
        self.generate_summary_report(global_stats, total_points, total_labeled_points, priority_order, filter_background)
        
        return global_stats
    
    def generate_summary_report(self, global_stats, total_points, total_labeled_points, priority_order, filter_background):
        """生成转换总结报告"""
        report_file = self.output_sequence_path / "conversion_report.txt"
        
        # 统计输出文件
        velodyne_files = list((self.output_sequence_path / "velodyne").glob("*.bin"))
        semantic_files = list((self.output_sequence_path / "semantic_labels").glob("*.label"))
        filtered_files = list((self.output_sequence_path / "labels").glob("*.label"))
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("3D边界框到点云分割转换报告\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"输入路径: {self.input_sequence_path}\n")
            f.write(f"输出路径: {self.output_sequence_path}\n")
            f.write(f"序列号: {self.sequence}\n\n")
            
            f.write("输出文件统计:\n")
            f.write(f"  过滤后点云: {len(velodyne_files)} 个文件\n")
            f.write(f"  完整分割标签: {len(semantic_files)} 个文件\n")
            f.write(f"  过滤分割标签: {len(filtered_files)} 个文件\n\n")
            
            f.write("处理模式:\n")
            f.write(f"  过滤背景点: {'是' if filter_background else '否'}\n")
            f.write(f"  输出类型: {'仅标注点' if filter_background else '完整点云+背景'}\n\n")
            
            f.write("优先级设置:\n")
            f.write(f"  优先级顺序: {' < '.join(map(str, priority_order))} (数字越大优先级越高)\n")
            f.write(f"  重叠处理: 高优先级标签覆盖低优先级标签\n\n")
            
            f.write("类别映射:\n")
            for class_id, info in TRAVERSABILITY_CLASSES.items():
                priority_level = priority_order.index(class_id) if class_id in priority_order else -1
                f.write(f"  {class_id}: {info['name']} (优先级: {priority_level})\n")
            f.write(f"  0: 背景/未标注\n\n")
            
            f.write("全数据集点云统计:\n")
            f.write(f"  总点数: {total_points:,}\n")
            f.write(f"  标注点数: {total_labeled_points:,} ({total_labeled_points/total_points*100:.2f}%)\n")
            if not filter_background:
                f.write(f"  背景点数: {global_stats[0]:,} ({global_stats[0]/total_points*100:.2f}%)\n")
            f.write("\n")
            
            f.write("各类别详细统计:\n")
            for class_id in sorted(TRAVERSABILITY_CLASSES.keys()):
                count = global_stats.get(class_id, 0)
                class_name = TRAVERSABILITY_CLASSES[class_id]['name']
                percentage = count / total_points * 100 if total_points > 0 else 0
                labeled_percentage = count / total_labeled_points * 100 if total_labeled_points > 0 else 0
                priority_level = priority_order.index(class_id) if class_id in priority_order else -1
                
                f.write(f"  类别 {class_id} ({class_name}):\n")
                f.write(f"    点数: {count:,}\n")
                f.write(f"    占总点数: {percentage:.2f}%\n")
                f.write(f"    占标注点数: {labeled_percentage:.2f}%\n")
                f.write(f"    优先级: {priority_level}\n\n")
            
            f.write("数据使用说明:\n")
            if filter_background:
                f.write("  1. semantic_labels/*.label - 过滤后点云的分割标签 (无背景)\n")
                f.write("  2. velodyne/*.bin - 过滤后的点云 (只包含bbox内的点)\n")
                f.write("  3. labels/*.label - 过滤后点云对应的标签 (与semantic_labels相同)\n")
                f.write("  4. 所有保留的点都有非零标签 (1-4)\n")
            else:
                f.write("  1. semantic_labels/*.label - 完整点云的分割标签 (包含背景)\n")
                f.write("  2. velodyne/*.bin - 完整的原始点云\n")
                f.write("  3. labels/*.label - 过滤后点云对应的标签 (只有bbox内的点)\n")
                f.write("  4. 标签格式: 0=背景, 1-4=穿越性类别\n")
            f.write("  5. 标签格式: uint32, 每个点一个标签值\n")
        
        print(f"📝 生成详细报告: {report_file}")
        
        # 同时生成CSV格式的统计文件
        csv_file = self.output_sequence_path / "class_statistics.csv"
        with open(csv_file, 'w') as f:
            f.write("class_id,class_name,point_count,total_percentage,labeled_percentage,priority_level\n")
            if not filter_background:
                f.write(f"0,Background,{global_stats[0]},{global_stats[0]/total_points*100:.2f},-,-\n")
            
            for class_id in sorted(TRAVERSABILITY_CLASSES.keys()):
                count = global_stats.get(class_id, 0)
                class_name = TRAVERSABILITY_CLASSES[class_id]['name']
                percentage = count / total_points * 100 if total_points > 0 else 0
                labeled_percentage = count / total_labeled_points * 100 if total_labeled_points > 0 else 0
                priority_level = priority_order.index(class_id) if class_id in priority_order else -1
                
                f.write(f"{class_id},{class_name},{count},{percentage:.2f},{labeled_percentage:.2f},{priority_level}\n")
        
        print(f"📊 生成CSV统计: {csv_file}")


def main():
    parser = argparse.ArgumentParser(description="3D边界框转点云分割标签生成器")
    parser.add_argument("input_path", help="输入数据集路径")
    parser.add_argument("output_path", help="输出数据集路径")
    parser.add_argument("-s", "--sequence", default="00", help="序列号 (默认: 00)")
    parser.add_argument("--expand-up", type=float, default=0.5, 
                       help="边界框向上扩展距离(米) (默认: 0.5)")
    parser.add_argument("--expand-down", type=float, default=0.3,
                       help="边界框向下扩展距离(米) (默认: 0.3)")
    parser.add_argument("--priority", nargs='+', type=int, default=[1, 2, 3, 4],
                       help="类别优先级顺序 (默认: 1 2 3 4)")
    parser.add_argument("--filter-background", action="store_true", default=True,
                       help="过滤背景点，只保留bbox内的点 (默认)")
    parser.add_argument("--keep-all", action="store_true",
                       help="保留所有点，包括背景点")
    
    args = parser.parse_args()
    
    # 处理互斥参数
    if args.keep_all:
        args.filter_background = False
    
    try:
        converter = BBoxToSegmentationConverter(
            args.input_path, 
            args.output_path, 
            args.sequence
        )
        
        converter.convert_dataset(
            expand_up=args.expand_up,
            expand_down=args.expand_down,
            priority_order=args.priority,
            filter_background=args.filter_background
        )
        
    except KeyboardInterrupt:
        print("\n🛑 用户中断转换")
    except Exception as e:
        print(f"❌ 转换失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()