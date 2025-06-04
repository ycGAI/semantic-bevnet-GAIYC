#!/usr/bin/env python3
"""
从MCAP文件中提取相机标定和传感器变换信息
生成真实的KITTI格式标定文件
"""

import numpy as np
import argparse
from pathlib import Path
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
import rosbag2_py
import yaml

class CalibrationExtractor:
    def __init__(self, mcap_file):
        """
        初始化标定提取器
        
        Args:
            mcap_file: MCAP文件路径
        """
        self.mcap_file = Path(mcap_file)
        self.camera_info = {}
        self.tf_tree = {}
        
        print(f"🎯 分析MCAP文件: {self.mcap_file}")
        
    def extract_calibration_data(self):
        """提取标定数据"""
        reader = rosbag2_py.SequentialReader()
        reader.open(
            rosbag2_py.StorageOptions(uri=str(self.mcap_file), storage_id="mcap"),
            rosbag2_py.ConverterOptions(
                input_serialization_format="cdr", 
                output_serialization_format="cdr"
            ),
        )

        topic_types = reader.get_all_topics_and_types()
        type_map = {topic.name: topic.type for topic in topic_types}
        
        print("\n📋 搜索标定相关话题...")
        
        # 查找相关话题
        calibration_topics = []
        for topic_name in type_map.keys():
            if any(keyword in topic_name.lower() for keyword in 
                   ['camera_info', 'calibration', 'tf', 'tf_static']):
                calibration_topics.append(topic_name)
                print(f"  📄 找到: {topic_name} ({type_map[topic_name]})")
        
        if not calibration_topics:
            print("⚠️  没有找到标定相关话题")
            return False
        
        message_count = 0
        max_messages = 1000  # 限制读取消息数量
        
        while reader.has_next() and message_count < max_messages:
            topic, data, timestamp = reader.read_next()
            message_count += 1
            
            if topic not in calibration_topics:
                continue
                
            try:
                msg_type = get_message(type_map[topic])
                msg = deserialize_message(data, msg_type)
                
                # 处理CameraInfo消息
                if 'camera_info' in topic.lower():
                    self.extract_camera_info(topic, msg)
                
                # 处理TF消息
                elif topic in ['/tf', '/tf_static']:
                    self.extract_tf_transforms(msg)
                    
            except Exception as e:
                print(f"⚠️  处理消息失败 {topic}: {e}")
                continue
        
        del reader
        return True
    
    def extract_camera_info(self, topic, msg):
        """提取相机信息"""
        if hasattr(msg, 'k') and hasattr(msg, 'p'):
            camera_name = self.get_camera_name(topic)
            
            self.camera_info[camera_name] = {
                'topic': topic,
                'frame_id': getattr(msg, 'header', {}).frame_id if hasattr(msg, 'header') else 'unknown',
                'width': getattr(msg, 'width', 0),
                'height': getattr(msg, 'height', 0),
                'K': np.array(msg.k).reshape(3, 3) if hasattr(msg, 'k') else None,
                'D': np.array(msg.d) if hasattr(msg, 'd') else None,
                'R': np.array(msg.r).reshape(3, 3) if hasattr(msg, 'r') else None,
                'P': np.array(msg.p).reshape(3, 4) if hasattr(msg, 'p') else None,
                'distortion_model': getattr(msg, 'distortion_model', 'unknown')
            }
            
            print(f"  📷 相机信息: {camera_name}")
            print(f"     分辨率: {msg.width}x{msg.height}")
            print(f"     畸变模型: {getattr(msg, 'distortion_model', 'unknown')}")
    
    def get_camera_name(self, topic):
        """从话题名推断相机名称"""
        topic_lower = topic.lower()
        if 'left' in topic_lower or 'image_0' in topic_lower:
            return 'camera_left'
        elif 'right' in topic_lower or 'image_1' in topic_lower:
            return 'camera_right'
        elif 'color' in topic_lower or 'image_2' in topic_lower:
            return 'camera_color'
        else:
            # 从话题路径提取
            parts = topic.split('/')
            for part in parts:
                if 'camera' in part:
                    return part
            return 'camera_unknown'
    
    def extract_tf_transforms(self, msg):
        """提取TF变换"""
        if hasattr(msg, 'transforms'):
            for transform in msg.transforms:
                parent = transform.header.frame_id
                child = transform.child_frame_id
                
                # 提取变换矩阵
                t = transform.transform.translation
                r = transform.transform.rotation
                
                # 四元数转旋转矩阵
                from scipy.spatial.transform import Rotation
                rotation = Rotation.from_quat([r.x, r.y, r.z, r.w])
                R_matrix = rotation.as_matrix()
                
                # 构建4x4变换矩阵
                T = np.eye(4)
                T[:3, :3] = R_matrix
                T[:3, 3] = [t.x, t.y, t.z]
                
                self.tf_tree[f"{parent}->{child}"] = {
                    'parent': parent,
                    'child': child,
                    'translation': [t.x, t.y, t.z],
                    'rotation_quat': [r.x, r.y, r.z, r.w],
                    'matrix': T
                }
                
                print(f"  🔄 TF变换: {parent} -> {child}")
                print(f"     平移: [{t.x:.3f}, {t.y:.3f}, {t.z:.3f}]")
    
    def generate_kitti_calibration(self, output_file):
        """生成KITTI格式的标定文件"""
        print(f"\n📝 生成KITTI标定文件: {output_file}")
        
        with open(output_file, 'w') as f:
            f.write("# KITTI Calibration File\n")
            f.write("# Generated from MCAP bag file\n")
            f.write(f"# Source: {self.mcap_file}\n\n")
            
            # 写入相机投影矩阵
            camera_mapping = {
                'camera_left': 'P0',
                'camera_right': 'P1', 
                'camera_color': 'P2',
                'camera_unknown': 'P0'
            }
            
            for i, (camera_name, camera_data) in enumerate(self.camera_info.items()):
                if camera_data['P'] is not None:
                    p_name = camera_mapping.get(camera_name, f'P{i}')
                    p_matrix = camera_data['P'].flatten()
                    f.write(f"# {camera_name} ({camera_data['topic']})\n")
                    f.write(f"# Resolution: {camera_data['width']}x{camera_data['height']}\n")
                    f.write(f"{p_name}: {' '.join(f'{val:.6e}' for val in p_matrix)}\n\n")
            
            # 写入传感器变换矩阵
            f.write("# Sensor transformations\n")
            
            # 查找激光雷达到相机的变换
            print("🔍 计算传感器变换...")
            lidar_to_cam = self.find_transform_chain(['velodyne', 'lidar', 'laser'], 
                                                   ['camera', 'cam', 'optical'])
            
            if lidar_to_cam is not None:
                tr_matrix = lidar_to_cam[:3, :].flatten()  # 取前3行
                f.write(f"# Velodyne to Camera transformation (computed from TF tree)\n")
                f.write(f"Tr: {' '.join(f'{val:.6e}' for val in tr_matrix)}\n\n")
                print("✅ 使用计算得到的真实变换")
            else:
                f.write("# Velodyne to Camera transformation (template - please update)\n")
                f.write("Tr: 4.276802385584e-04 -9.999672484946e-01 -8.084491683471e-03 -1.198459927713e-02 -7.210626507497e-03 8.081198471645e-03 -9.999413164504e-01 -5.403984729748e-02 9.999738645903e-01 4.859485810390e-04 -7.206933692422e-03 -2.921968648686e-01\n\n")
                print("⚠️  使用模板变换")
            
            # 添加相机投影矩阵模板（因为没有真实相机标定）
            f.write("# Camera projection matrices (TEMPLATE - please update with real calibration)\n")
            f.write("# These are placeholder values and should be replaced with actual camera calibration\n")
            f.write("P0: 7.215377e+02 0.000000e+00 6.095593e+02 0.000000e+00 0.000000e+00 7.215377e+02 1.728540e+02 0.000000e+00 0.000000e+00 0.000000e+00 1.000000e+00 0.000000e+00\n")
            f.write("P1: 7.215377e+02 0.000000e+00 6.095593e+02 -3.875744e+02 0.000000e+00 7.215377e+02 1.728540e+02 0.000000e+00 0.000000e+00 0.000000e+00 1.000000e+00 0.000000e+00\n")
            f.write("P2: 7.215377e+02 0.000000e+00 6.095593e+02 0.000000e+00 0.000000e+00 7.215377e+02 1.728540e+02 0.000000e+00 0.000000e+00 0.000000e+00 1.000000e+00 0.000000e+00\n")
            f.write("P3: 7.215377e+02 0.000000e+00 6.095593e+02 -3.875744e+02 0.000000e+00 7.215377e+02 1.728540e+02 0.000000e+00 0.000000e+00 0.000000e+00 1.000000e+00 0.000000e+00\n\n")
            
            # 写入内参矩阵
            for camera_name, camera_data in self.camera_info.items():
                if camera_data['K'] is not None:
                    f.write(f"# {camera_name} intrinsic matrix\n")
                    k_matrix = camera_data['K'].flatten()
                    f.write(f"K_{camera_name}: {' '.join(f'{val:.6e}' for val in k_matrix)}\n")
                    
                    if camera_data['D'] is not None:
                        f.write(f"# {camera_name} distortion coefficients\n") 
                        d_coeffs = camera_data['D']
                        f.write(f"D_{camera_name}: {' '.join(f'{val:.6e}' for val in d_coeffs)}\n")
                    f.write("\n")
        
        print("✅ KITTI标定文件生成完成")
    
    def find_transform_chain(self, source_keywords, target_keywords):
        """查找从源到目标的变换链"""
        # 基于你的TF树计算 velodyne -> camera_link 变换
        
        # 从TF树中找到关键变换
        velodyne_to_sensor_rack = None
        sensor_rack_to_base_link = None  
        base_link_to_camera = None
        
        for transform_name, transform_data in self.tf_tree.items():
            # base_link_sensor_rack -> velodyne
            if (transform_data['parent'] == 'base_link_sensor_rack' and 
                transform_data['child'] == 'velodyne'):
                # 需要取逆变换 velodyne -> base_link_sensor_rack
                T_inv = np.linalg.inv(transform_data['matrix'])
                velodyne_to_sensor_rack = T_inv
                print(f"  找到: velodyne -> base_link_sensor_rack")
                
            # base_link -> base_link_sensor_rack  
            elif (transform_data['parent'] == 'base_link' and 
                  transform_data['child'] == 'base_link_sensor_rack'):
                # 需要取逆变换 base_link_sensor_rack -> base_link
                T_inv = np.linalg.inv(transform_data['matrix'])
                sensor_rack_to_base_link = T_inv
                print(f"  找到: base_link_sensor_rack -> base_link")
                
            # base_link -> camera_link
            elif (transform_data['parent'] == 'base_link' and 
                  transform_data['child'] == 'camera_link'):
                base_link_to_camera = transform_data['matrix']
                print(f"  找到: base_link -> camera_link")
        
        # 计算完整变换链: velodyne -> base_link_sensor_rack -> base_link -> camera_link
        if (velodyne_to_sensor_rack is not None and 
            sensor_rack_to_base_link is not None and 
            base_link_to_camera is not None):
            
            # 链式变换
            velodyne_to_camera = base_link_to_camera @ sensor_rack_to_base_link @ velodyne_to_sensor_rack
            
            print(f"  ✅ 计算得到 velodyne -> camera_link 变换")
            return velodyne_to_camera
        
        print(f"  ⚠️  无法构建完整变换链")
        return None
    
    def save_detailed_report(self, output_dir):
        """保存详细的标定分析报告"""
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        # 保存相机信息
        camera_report = output_dir / "camera_calibration.yaml"
        with open(camera_report, 'w') as f:
            yaml.dump({
                'cameras': {name: {
                    'topic': data['topic'],
                    'frame_id': data['frame_id'],
                    'resolution': [data['width'], data['height']],
                    'distortion_model': data['distortion_model'],
                    'K': data['K'].tolist() if data['K'] is not None else None,
                    'D': data['D'].tolist() if data['D'] is not None else None,
                    'P': data['P'].tolist() if data['P'] is not None else None,
                } for name, data in self.camera_info.items()}
            }, f, default_flow_style=False)
        
        # 保存TF树
        tf_report = output_dir / "tf_transforms.yaml"
        with open(tf_report, 'w') as f:
            yaml.dump({
                'transforms': {name: {
                    'parent_frame': data['parent'],
                    'child_frame': data['child'],
                    'translation': data['translation'],
                    'rotation_quaternion': data['rotation_quat'],
                } for name, data in self.tf_tree.items()}
            }, f, default_flow_style=False)
        
        print(f"📄 详细报告保存到: {output_dir}")
    
    def print_summary(self):
        """打印提取摘要"""
        print(f"\n📊 标定数据提取摘要:")
        print(f"   相机数量: {len(self.camera_info)}")
        print(f"   TF变换数量: {len(self.tf_tree)}")
        
        if self.camera_info:
            print(f"\n📷 相机信息:")
            for name, data in self.camera_info.items():
                print(f"   {name}: {data['width']}x{data['height']} ({data['frame_id']})")
        
        if self.tf_tree:
            print(f"\n🔄 TF变换:")
            for name, data in self.tf_tree.items():
                print(f"   {data['parent']} -> {data['child']}")


def main():
    parser = argparse.ArgumentParser(description="从MCAP文件提取相机标定信息")
    parser.add_argument("mcap_file", help="输入MCAP文件路径")
    parser.add_argument("-o", "--output", default="calib.txt", help="输出标定文件路径")
    parser.add_argument("--report-dir", help="详细报告输出目录")
    
    args = parser.parse_args()
    
    try:
        extractor = CalibrationExtractor(args.mcap_file)
        
        if extractor.extract_calibration_data():
            extractor.print_summary()
            extractor.generate_kitti_calibration(args.output)
            
            if args.report_dir:
                extractor.save_detailed_report(args.report_dir)
        else:
            print("❌ 没有找到标定数据")
            
    except Exception as e:
        print(f"❌ 提取失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
#LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 python extract_calibration.py "/media/gyc/Backup Plus4/gyc/thesis/raw_demo_rosbag/row_coverage_1_0.mcap" -o ./calib.txt