#!/usr/bin/env python3
"""
KITTI bin格式点云转PCD格式转换器
支持批量转换和多种PCD格式选项
"""

import numpy as np
import argparse
import os
from pathlib import Path
from tqdm import tqdm
import struct

class BinToPCDConverter:
    def __init__(self, input_path, output_path, pcd_format='ascii'):
        """
        初始化转换器
        
        Args:
            input_path: 输入路径（文件或目录）
            output_path: 输出路径（文件或目录）
            pcd_format: PCD格式 ('ascii', 'binary', 'binary_compressed')
        """
        self.input_path = Path(input_path)
        self.output_path = Path(output_path)
        self.pcd_format = pcd_format.lower()
        
        # 验证PCD格式
        valid_formats = ['ascii', 'binary', 'binary_compressed']
        if self.pcd_format not in valid_formats:
            raise ValueError(f"不支持的PCD格式: {pcd_format}. 支持的格式: {valid_formats}")
    
    def load_kitti_bin(self, bin_file):
        """
        加载KITTI bin格式点云
        
        Args:
            bin_file: bin文件路径
            
        Returns:
            points: numpy数组 (N, 4) [x, y, z, intensity]
        """
        try:
            # KITTI bin格式：N×4的float32数组 [x, y, z, intensity]
            points = np.fromfile(bin_file, dtype=np.float32).reshape(-1, 4)
            return points
        except Exception as e:
            print(f"❌ 加载bin文件失败 {bin_file}: {e}")
            return None
    
    def save_pcd_ascii(self, points, pcd_file):
        """保存ASCII格式PCD文件"""
        num_points = len(points)
        
        with open(pcd_file, 'w') as f:
            # PCD文件头
            f.write("# .PCD v0.7 - Point Cloud Data file format\n")
            f.write("VERSION 0.7\n")
            f.write("FIELDS x y z intensity\n")
            f.write("SIZE 4 4 4 4\n")
            f.write("TYPE F F F F\n")
            f.write("COUNT 1 1 1 1\n")
            f.write(f"WIDTH {num_points}\n")
            f.write("HEIGHT 1\n")
            f.write("VIEWPOINT 0 0 0 1 0 0 0\n")
            f.write(f"POINTS {num_points}\n")
            f.write("DATA ascii\n")
            
            # 点云数据
            for point in points:
                f.write(f"{point[0]:.6f} {point[1]:.6f} {point[2]:.6f} {point[3]:.6f}\n")
    
    def save_pcd_binary(self, points, pcd_file):
        """保存二进制格式PCD文件"""
        num_points = len(points)
        
        with open(pcd_file, 'wb') as f:
            # PCD文件头（ASCII部分）
            header = f"""# .PCD v0.7 - Point Cloud Data file format
VERSION 0.7
FIELDS x y z intensity
SIZE 4 4 4 4
TYPE F F F F
COUNT 1 1 1 1
WIDTH {num_points}
HEIGHT 1
VIEWPOINT 0 0 0 1 0 0 0
POINTS {num_points}
DATA binary
"""
            f.write(header.encode('ascii'))
            
            # 点云数据（二进制）
            points_binary = points.astype(np.float32).tobytes()
            f.write(points_binary)
    
    def save_pcd_binary_compressed(self, points, pcd_file):
        """保存压缩二进制格式PCD文件"""
        try:
            import lzf  # 需要安装：pip install lzf
        except ImportError:
            print("⚠️  lzf库未安装，使用普通二进制格式代替")
            print("   安装方法: pip install lzf")
            self.save_pcd_binary(points, pcd_file)
            return
        
        num_points = len(points)
        
        with open(pcd_file, 'wb') as f:
            # 压缩点云数据
            points_binary = points.astype(np.float32).tobytes()
            compressed_data = lzf.compress(points_binary)
            compressed_size = len(compressed_data)
            uncompressed_size = len(points_binary)
            
            # PCD文件头
            header = f"""# .PCD v0.7 - Point Cloud Data file format
VERSION 0.7
FIELDS x y z intensity
SIZE 4 4 4 4
TYPE F F F F
COUNT 1 1 1 1
WIDTH {num_points}
HEIGHT 1
VIEWPOINT 0 0 0 1 0 0 0
POINTS {num_points}
DATA binary_compressed
"""
            f.write(header.encode('ascii'))
            
            # 压缩信息
            f.write(struct.pack('<I', compressed_size))
            f.write(struct.pack('<I', uncompressed_size))
            
            # 压缩数据
            f.write(compressed_data)
    
    def save_pcd(self, points, pcd_file):
        """根据指定格式保存PCD文件"""
        if self.pcd_format == 'ascii':
            self.save_pcd_ascii(points, pcd_file)
        elif self.pcd_format == 'binary':
            self.save_pcd_binary(points, pcd_file)
        elif self.pcd_format == 'binary_compressed':
            self.save_pcd_binary_compressed(points, pcd_file)
    
    def convert_single_file(self, bin_file, pcd_file):
        """转换单个文件"""
        # 加载bin文件
        points = self.load_kitti_bin(bin_file)
        if points is None:
            return False
        
        # 创建输出目录
        pcd_file.parent.mkdir(parents=True, exist_ok=True)
        
        # 保存PCD文件
        try:
            self.save_pcd(points, pcd_file)
            return True
        except Exception as e:
            print(f"❌ 保存PCD文件失败 {pcd_file}: {e}")
            return False
    
    def convert_batch(self):
        """批量转换"""
        if self.input_path.is_file():
            # 单文件转换
            if self.input_path.suffix != '.bin':
                print(f"❌ 输入文件不是bin格式: {self.input_path}")
                return
            
            # 确定输出文件名
            if self.output_path.is_dir():
                output_file = self.output_path / f"{self.input_path.stem}.pcd"
            else:
                output_file = self.output_path
                if output_file.suffix != '.pcd':
                    output_file = output_file.with_suffix('.pcd')
            
            print(f"🔄 转换单个文件:")
            print(f"   输入: {self.input_path}")
            print(f"   输出: {output_file}")
            print(f"   格式: {self.pcd_format}")
            
            success = self.convert_single_file(self.input_path, output_file)
            if success:
                print(f"✅ 转换成功!")
                self.print_file_info(self.input_path, output_file)
            else:
                print(f"❌ 转换失败!")
        
        elif self.input_path.is_dir():
            # 批量转换
            bin_files = list(self.input_path.glob("*.bin"))
            if not bin_files:
                print(f"❌ 在目录中没有找到bin文件: {self.input_path}")
                return
            
            print(f"🚀 批量转换:")
            print(f"   输入目录: {self.input_path}")
            print(f"   输出目录: {self.output_path}")
            print(f"   格式: {self.pcd_format}")
            print(f"   文件数量: {len(bin_files)}")
            
            # 创建输出目录
            self.output_path.mkdir(parents=True, exist_ok=True)
            
            success_count = 0
            failed_count = 0
            
            for bin_file in tqdm(bin_files, desc="转换进度"):
                pcd_file = self.output_path / f"{bin_file.stem}.pcd"
                
                if self.convert_single_file(bin_file, pcd_file):
                    success_count += 1
                else:
                    failed_count += 1
            
            print(f"✅ 批量转换完成:")
            print(f"   成功: {success_count} 个文件")
            print(f"   失败: {failed_count} 个文件")
            print(f"   输出目录: {self.output_path}")
            
            # 显示文件大小统计
            if success_count > 0:
                self.print_batch_info(bin_files, success_count)
        
        else:
            print(f"❌ 输入路径不存在: {self.input_path}")
    
    def print_file_info(self, bin_file, pcd_file):
        """打印单个文件转换信息"""
        try:
            bin_size = bin_file.stat().st_size
            pcd_size = pcd_file.stat().st_size
            
            # 加载点云获取点数
            points = self.load_kitti_bin(bin_file)
            num_points = len(points) if points is not None else 0
            
            print(f"📊 文件信息:")
            print(f"   点云数量: {num_points:,}")
            print(f"   bin大小: {bin_size / 1024:.1f} KB")
            print(f"   pcd大小: {pcd_size / 1024:.1f} KB")
            print(f"   压缩比: {pcd_size / bin_size:.2f}")
            
        except Exception as e:
            print(f"⚠️  无法获取文件信息: {e}")
    
    def print_batch_info(self, bin_files, success_count):
        """打印批量转换统计信息"""
        try:
            # 计算文件大小
            total_bin_size = sum(f.stat().st_size for f in bin_files[:success_count])
            
            pcd_files = list(self.output_path.glob("*.pcd"))
            total_pcd_size = sum(f.stat().st_size for f in pcd_files)
            
            # 计算总点数（采样几个文件）
            sample_files = bin_files[:min(10, len(bin_files))]
            total_points = 0
            for bin_file in sample_files:
                points = self.load_kitti_bin(bin_file)
                if points is not None:
                    total_points += len(points)
            
            avg_points = total_points / len(sample_files) if sample_files else 0
            estimated_total_points = avg_points * success_count
            
            print(f"📊 批量转换统计:")
            print(f"   估计总点数: {estimated_total_points:,.0f}")
            print(f"   平均每文件点数: {avg_points:,.0f}")
            print(f"   总bin大小: {total_bin_size / 1024 / 1024:.1f} MB")
            print(f"   总pcd大小: {total_pcd_size / 1024 / 1024:.1f} MB")
            print(f"   平均压缩比: {total_pcd_size / total_bin_size:.2f}")
            
        except Exception as e:
            print(f"⚠️  无法计算统计信息: {e}")

def main():
    parser = argparse.ArgumentParser(description="KITTI bin格式点云转PCD格式转换器")
    parser.add_argument("input", help="输入bin文件或包含bin文件的目录")
    parser.add_argument("output", help="输出pcd文件或输出目录")
    parser.add_argument("-f", "--format", choices=['ascii', 'binary', 'binary_compressed'], 
                       default='ascii', help="PCD格式 (默认: ascii)")
    parser.add_argument("--sample", action="store_true", help="只转换前10个文件（用于测试）")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"❌ 输入路径不存在: {args.input}")
        return
    
    try:
        converter = BinToPCDConverter(args.input, args.output, args.format)
        
        # 如果是测试模式，只处理前几个文件
        if args.sample and Path(args.input).is_dir():
            input_path = Path(args.input)
            bin_files = sorted(list(input_path.glob("*.bin")))[:10]
            
            print(f"🧪 测试模式：只转换前{len(bin_files)}个文件")
            
            output_path = Path(args.output)
            output_path.mkdir(parents=True, exist_ok=True)
            
            for bin_file in bin_files:
                pcd_file = output_path / f"{bin_file.stem}.pcd"
                converter.convert_single_file(bin_file, pcd_file)
                
        else:
            converter.convert_batch()
            
    except KeyboardInterrupt:
        print("\n🛑 用户中断转换")
    except Exception as e:
        print(f"❌ 转换失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()