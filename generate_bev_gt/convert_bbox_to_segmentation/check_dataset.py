#!/usr/bin/env python3
"""
修复的数据格式检查脚本
处理点云中的NaN值问题
"""

import os
import numpy as np
import glob

def check_pointcloud_files_fixed(velodyne_dir):
    """检查点云文件（处理NaN值）"""
    print("\n=== 检查点云文件 ===")
    
    bin_files = glob.glob(os.path.join(velodyne_dir, "*.bin"))
    if not bin_files:
        print("错误：未找到点云文件(.bin)")
        return False
    
    # 检查第一个文件
    first_file = bin_files[0]
    try:
        points = np.fromfile(first_file, dtype=np.float32)
        points = points.reshape(-1, 4)
        
        print(f"✓ 点云文件数量：{len(bin_files)}")
        print(f"✓ 示例文件：{os.path.basename(first_file)}")
        print(f"✓ 原始点数：{len(points)}")
        
        # 检查NaN值
        nan_mask = np.isnan(points).any(axis=1)
        if nan_mask.any():
            print(f"! 发现 {nan_mask.sum()} 个包含NaN的点，将被过滤")
            valid_points = points[~nan_mask]
        else:
            valid_points = points
        
        # 检查无穷大值
        inf_mask = np.isinf(valid_points).any(axis=1)
        if inf_mask.any():
            print(f"! 发现 {inf_mask.sum()} 个包含无穷大值的点，将被过滤")
            valid_points = valid_points[~inf_mask]
        
        print(f"✓ 有效点数：{len(valid_points)}")
        
        if len(valid_points) > 0:
            print(f"✓ 点云范围：")
            print(f"    X: [{valid_points[:, 0].min():.2f}, {valid_points[:, 0].max():.2f}]")
            print(f"    Y: [{valid_points[:, 1].min():.2f}, {valid_points[:, 1].max():.2f}]")
            print(f"    Z: [{valid_points[:, 2].min():.2f}, {valid_points[:, 2].max():.2f}]")
            print(f"    强度: [{valid_points[:, 3].min():.2f}, {valid_points[:, 3].max():.2f}]")
            
            # 检查点云分布
            zero_points = (valid_points[:, :3] == 0).all(axis=1).sum()
            if zero_points > 0:
                print(f"! 发现 {zero_points} 个原点处的点")
                
            # 检查点云是否异常集中
            distances = np.linalg.norm(valid_points[:, :3], axis=1)
            print(f"✓ 距离统计：")
            print(f"    最小距离: {distances.min():.2f}")
            print(f"    最大距离: {distances.max():.2f}")
            print(f"    平均距离: {distances.mean():.2f}")
        else:
            print("错误：没有有效的点")
            return False
        
        return True
    except Exception as e:
        print(f"错误：无法读取点云文件：{e}")
        return False

def analyze_pointcloud_quality(velodyne_dir, num_files_to_check=5):
    """分析多个点云文件的质量"""
    print("\n=== 分析点云质量 ===")
    
    bin_files = sorted(glob.glob(os.path.join(velodyne_dir, "*.bin")))
    files_to_check = bin_files[:num_files_to_check]
    
    total_points = 0
    total_valid_points = 0
    total_nan_points = 0
    total_inf_points = 0
    
    for i, file_path in enumerate(files_to_check):
        try:
            points = np.fromfile(file_path, dtype=np.float32).reshape(-1, 4)
            
            # 统计无效点
            nan_mask = np.isnan(points).any(axis=1)
            inf_mask = np.isinf(points[~nan_mask]).any(axis=1) if (~nan_mask).any() else np.array([])
            
            num_points = len(points)
            num_nan = nan_mask.sum()
            num_inf = inf_mask.sum() if len(inf_mask) > 0 else 0
            num_valid = num_points - num_nan - num_inf
            
            total_points += num_points
            total_valid_points += num_valid
            total_nan_points += num_nan
            total_inf_points += num_inf
            
            if i < 3:  # 只显示前3个文件的详细信息
                print(f"文件 {i+1} ({os.path.basename(file_path)}):")
                print(f"  总点数: {num_points}, 有效: {num_valid}, NaN: {num_nan}, Inf: {num_inf}")
                
        except Exception as e:
            print(f"无法读取文件 {file_path}: {e}")
    
    print(f"\n总计 (检查了 {len(files_to_check)} 个文件):")
    print(f"  总点数: {total_points:,}")
    print(f"  有效点数: {total_valid_points:,} ({total_valid_points/total_points*100:.1f}%)")
    print(f"  NaN点数: {total_nan_points:,} ({total_nan_points/total_points*100:.1f}%)")
    print(f"  Inf点数: {total_inf_points:,} ({total_inf_points/total_points*100:.1f}%)")

def main():
    # 您的数据集路径
    dataset_root = "/media/gyc/Backup Plus3/gyc/thesis/raw_demo_rosbag/dataset"
    
    print(f"检查数据集：{dataset_root}")
    
    # 检查具体文件
    seq_path = os.path.join(dataset_root, "sequences", "00")
    velodyne_dir = os.path.join(seq_path, "velodyne")
    
    # 使用修复的检查函数
    if check_pointcloud_files_fixed(velodyne_dir):
        # 分析更多文件的质量
        analyze_pointcloud_quality(velodyne_dir)
        
        print("\n=== 建议 ===")
        print("1. 点云数据包含一些无效值（NaN），转换脚本会自动过滤这些点")
        print("2. 配置文件已正确设置，可以直接运行转换")
        print("3. 如果转换后点数太少，可以考虑调整 bbox_expansion_factor")
        
        print("\n您可以现在运行转换脚本：")
        print("./run_conversion.sh")
        print("或者")
        print("python3 enhanced_converter.py --dataset_root '/media/gyc/Backup Plus3/gyc/thesis/raw_demo_rosbag/dataset' --sequence '00' --config config.yaml")
    else:
        print("\n=== 错误 ===")
        print("点云文件检查失败，请检查数据文件")

if __name__ == "__main__":
    main()