#!/usr/bin/env python3
"""
MCAP to KITTI Odometry Dataset Converter with Multiprocessing, Integrated Calibration and Time Synchronization
Converts ROS2 bag data to KITTI Odometry format with synchronized timestamps and real calibration data
Based on the original version with added time synchronization to handle different sensor frequencies
"""

import argparse
import os
import cv2
import numpy as np
from multiprocessing import Pool, Manager, cpu_count
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import time
from datetime import datetime
from scipy.spatial.transform import Rotation as R
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
import rosbag2_py
from tqdm import tqdm
import struct
import bisect

class MCAPToKITTIOdometryConverter:
    def __init__(self, input_bag, output_dir, sequence_id="00", num_processes=None):
        self.input_bag = input_bag
        self.output_dir = Path(output_dir)
        self.sequence_id = sequence_id
        self.sequence_dir = self.output_dir / "sequences" / sequence_id
        self.num_processes = num_processes or max(1, cpu_count() - 1)
        
        # KITTI Odometry目录结构
        self.setup_directories()
        
        # 原版数据存储
        self.poses_data = []
        self.camera_data = []
        self.lidar_data = []
        
        # 标定数据存储
        self.camera_info = {}
        self.tf_tree = {}
        
        # 时间同步相关
        self.time_tolerance_ns = 50_000_000  # 50ms容差
        self.pose_timestamp_map = {}  # timestamp -> pose_data 映射
        
    def setup_directories(self):
        """创建KITTI Odometry标准目录结构"""
        directories = [
            f'sequences/{self.sequence_id}/image_0',   # 左相机（灰度）
            f'sequences/{self.sequence_id}/image_1',   # 右相机（灰度）
            f'sequences/{self.sequence_id}/image_2',   # 左相机（彩色）
            f'sequences/{self.sequence_id}/image_3',   # 右相机（彩色）
            f'sequences/{self.sequence_id}/velodyne',  # 激光雷达点云
            f'sequences/{self.sequence_id}/labels',    # 语义分割标签（可选）
            'poses'  # 位姿文件目录
        ]
        
        for dir_name in directories:
            (self.output_dir / dir_name).mkdir(parents=True, exist_ok=True)
        
        print(f"✅ 创建KITTI Odometry目录结构: {self.output_dir}")
        print(f"📁 序列目录: {self.sequence_dir}")

    def read_bag_messages(self):
        """读取bag文件中的所有消息"""
        reader = rosbag2_py.SequentialReader()
        reader.open(
            rosbag2_py.StorageOptions(uri=self.input_bag, storage_id="mcap"),
            rosbag2_py.ConverterOptions(
                input_serialization_format="cdr", 
                output_serialization_format="cdr"
            ),
        )

        topic_types = reader.get_all_topics_and_types()
        type_map = {topic.name: topic.type for topic in topic_types}

        messages = []
        print("📖 读取bag文件消息...")
        
        while reader.has_next():
            topic, data, timestamp = reader.read_next()
            messages.append((topic, data, timestamp, type_map.get(topic)))
        
        del reader
        print(f"✅ 读取完成，共 {len(messages)} 条消息")
        return messages

    def extract_calibration_data(self):
        """提取标定数据"""
        print("\n🎯 提取标定数据...")
        
        reader = rosbag2_py.SequentialReader()
        reader.open(
            rosbag2_py.StorageOptions(uri=self.input_bag, storage_id="mcap"),
            rosbag2_py.ConverterOptions(
                input_serialization_format="cdr", 
                output_serialization_format="cdr"
            ),
        )

        topic_types = reader.get_all_topics_and_types()
        type_map = {topic.name: topic.type for topic in topic_types}
        
        # 查找相关话题
        calibration_topics = []
        for topic_name in type_map.keys():
            if any(keyword in topic_name.lower() for keyword in 
                   ['camera_info', 'calibration', 'tf', 'tf_static']):
                calibration_topics.append(topic_name)
                print(f"  📄 找到标定话题: {topic_name} ({type_map[topic_name]})")
        
        if not calibration_topics:
            print("⚠️  没有找到标定相关话题")
            del reader
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
                print(f"⚠️  处理标定消息失败 {topic}: {e}")
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
                rotation = R.from_quat([r.x, r.y, r.z, r.w])
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

    def find_nearest_timestamp(self, target_timestamp, timestamp_list, tolerance_ns):
        """
        在时间戳列表中找到最接近目标时间戳的条目
        使用二分查找提高效率
        """
        if not timestamp_list:
            return None
            
        # 使用二分查找找到插入位置
        idx = bisect.bisect_left(timestamp_list, target_timestamp)
        
        candidates = []
        
        # 检查左边的时间戳
        if idx > 0:
            candidates.append(timestamp_list[idx - 1])
        
        # 检查右边的时间戳  
        if idx < len(timestamp_list):
            candidates.append(timestamp_list[idx])
        
        # 找到时间差最小且在容差范围内的
        best_timestamp = None
        min_diff = float('inf')
        
        for ts in candidates:
            diff = abs(ts - target_timestamp)
            if diff <= tolerance_ns and diff < min_diff:
                min_diff = diff
                best_timestamp = ts
        
        return best_timestamp

    def process_message_batch(self, message_batch):
        """处理一批消息（多进程调用）"""
        poses = []
        cameras = []
        lidars = []
        pose_timestamp_map = {}  # 局部位姿时间戳映射
        
        for topic, data, timestamp, msg_type_name in message_batch:
            if not msg_type_name:
                continue
                
            try:
                msg_type = get_message(msg_type_name)
                msg = deserialize_message(data, msg_type)
                
                # 处理里程计数据（poses）
                if topic == '/odometry/filtered/global':
                    pose_data = self.extract_pose_data(msg, timestamp)
                    if pose_data:
                        poses.append(pose_data)
                        pose_timestamp_map[timestamp] = pose_data
                
                # 处理相机数据
                elif any(keyword in topic.lower() for keyword in ['image', 'camera']):
                    camera_data = self.extract_camera_data(msg, timestamp, topic)
                    if camera_data:
                        cameras.append(camera_data)
                
                # 处理激光雷达数据
                elif any(keyword in topic.lower() for keyword in ['velodyne', 'lidar', 'pointcloud', 'points']):
                    lidar_data = self.extract_lidar_data(msg, timestamp)
                    if lidar_data:
                        lidars.append(lidar_data)
                        
            except Exception as e:
                print(f"⚠️  处理消息失败 {topic}: {e}")
                continue
        
        return poses, cameras, lidars, pose_timestamp_map

    def extract_pose_data(self, msg, timestamp):
        """提取位姿数据"""
        try:
            pos = msg.pose.pose.position
            rot = msg.pose.pose.orientation
            
            # 四元数转旋转矩阵
            rotation = R.from_quat([rot.x, rot.y, rot.z, rot.w])
            rot_matrix = rotation.as_matrix()
            
            # 构建4x4变换矩阵
            transform = np.eye(4)
            transform[:3, :3] = rot_matrix
            transform[:3, 3] = [pos.x, pos.y, pos.z]
            
            # KITTI格式：前3行展开为12个数值
            kitti_pose = transform[:3, :].flatten()
            
            return {
                'timestamp': timestamp,
                'pose': kitti_pose,
                'position': (pos.x, pos.y, pos.z),
                'orientation': (rot.x, rot.y, rot.z, rot.w)
            }
        except Exception as e:
            print(f"⚠️  提取位姿数据失败: {e}")
            return None

    def extract_camera_data(self, msg, timestamp, topic):
        """提取相机数据"""
        try:
            # 处理压缩图像格式 (CompressedImage)
            if hasattr(msg, 'format') and hasattr(msg, 'data'):
                return {
                    'timestamp': timestamp,
                    'topic': topic,
                    'format': msg.format,  # 压缩格式 (jpeg, png等)
                    'data': bytes(msg.data),
                    'is_compressed': True
                }
            # 处理原始图像格式 (Image)
            elif hasattr(msg, 'width') and hasattr(msg, 'height'):
                return {
                    'timestamp': timestamp,
                    'topic': topic,
                    'width': msg.width,
                    'height': msg.height,
                    'encoding': msg.encoding,
                    'data': bytes(msg.data),
                    'step': getattr(msg, 'step', msg.width * 3),
                    'is_compressed': False
                }
        except Exception as e:
            print(f"⚠️  提取相机数据失败: {e}")
            return None

    def extract_lidar_data(self, msg, timestamp):
        """提取激光雷达数据"""
        try:
            if hasattr(msg, 'data'):
                return {
                    'timestamp': timestamp,
                    'width': getattr(msg, 'width', 1),
                    'height': getattr(msg, 'height', len(msg.data) // msg.point_step if hasattr(msg, 'point_step') else 1),
                    'point_step': getattr(msg, 'point_step', 16),
                    'row_step': getattr(msg, 'row_step', len(msg.data)),
                    'data': bytes(msg.data),
                    'fields': [(f.name, f.offset, f.datatype, f.count) for f in msg.fields] if hasattr(msg, 'fields') else []
                }
        except Exception as e:
            print(f"⚠️  提取激光雷达数据失败: {e}")
            return None

    def synchronize_with_lidar(self):
        """
        以LiDAR为基准进行时间同步
        确保位姿数据与LiDAR帧一一对应
        """
        print(f"\n⏱️  开始时间同步 (容差: {self.time_tolerance_ns/1e6:.0f}ms)...")
        
        # 获取所有位姿时间戳并排序
        pose_timestamps = sorted(self.pose_timestamp_map.keys())
        
        # 获取LiDAR时间戳并排序  
        lidar_timestamps = [ld['timestamp'] for ld in self.lidar_data]
        lidar_timestamps.sort()
        
        print(f"   位姿数据: {len(pose_timestamps)} 个")
        print(f"   LiDAR数据: {len(lidar_timestamps)} 个")
        
        # 为每个LiDAR帧寻找最近的位姿
        synchronized_poses = []
        pose_miss_count = 0
        
        for lidar_ts in lidar_timestamps:
            # 查找最近的位姿时间戳
            nearest_pose_ts = self.find_nearest_timestamp(
                lidar_ts, pose_timestamps, self.time_tolerance_ns
            )
            
            if nearest_pose_ts is not None:
                # 找到匹配的位姿
                pose_data = self.pose_timestamp_map[nearest_pose_ts]
                synchronized_poses.append(pose_data)
            else:
                # 没有找到匹配的位姿，使用单位矩阵
                identity_pose = np.eye(4)[:3, :].flatten()
                synchronized_poses.append({
                    'timestamp': lidar_ts,
                    'pose': identity_pose,
                    'position': (0.0, 0.0, 0.0),
                    'orientation': (0.0, 0.0, 0.0, 1.0)
                })
                pose_miss_count += 1
        
        # 更新位姿数据为同步后的数据
        self.poses_data = synchronized_poses
        
        print(f"✅ 时间同步完成:")
        print(f"   同步位姿: {len(synchronized_poses)}")
        print(f"   缺失位姿: {pose_miss_count} ({pose_miss_count/len(lidar_timestamps)*100:.1f}%)")
        
        # 分析同步质量
        if pose_miss_count < len(lidar_timestamps):
            matched_count = len(lidar_timestamps) - pose_miss_count
            print(f"   匹配成功: {matched_count} ({matched_count/len(lidar_timestamps)*100:.1f}%)")

    def save_camera_image(self, camera_data, index):
        """保存相机图像到KITTI Odometry格式"""
        try:
            # 处理压缩图像
            if camera_data.get('is_compressed', False):
                # 解码压缩图像
                np_arr = np.frombuffer(camera_data['data'], np.uint8)
                img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                
                if img is None:
                    print(f"⚠️  无法解码压缩图像，格式: {camera_data.get('format', 'unknown')}")
                    return False
                
                # 转换为灰度和彩色
                if len(img.shape) == 3:
                    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    color_img = img
                else:
                    gray_img = img
                    color_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            
            # 处理原始图像
            else:
                width = camera_data['width']
                height = camera_data['height']
                encoding = camera_data['encoding']
                data = camera_data['data']
                
                if encoding == 'bgr8':
                    img = np.frombuffer(data, np.uint8).reshape((height, width, 3))
                    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    color_img = img
                elif encoding == 'rgb8':
                    img = np.frombuffer(data, np.uint8).reshape((height, width, 3))
                    gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                    color_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                elif encoding == 'mono8':
                    gray_img = np.frombuffer(data, np.uint8).reshape((height, width))
                    color_img = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR)
                elif encoding == 'mono16':
                    img_16 = np.frombuffer(data, np.uint16).reshape((height, width))
                    gray_img = (img_16 / 256).astype(np.uint8)  # 转换为8位
                    color_img = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR)
                else:
                    print(f"⚠️  不支持的图像编码: {encoding}")
                    return False
            
            # 根据话题名确定是左相机还是右相机
            is_right_camera = any(keyword in camera_data['topic'].lower() 
                                for keyword in ['right', 'image_01', 'image_1', 'cam1'])
            
            # 保存灰度图像（image_0 和 image_1）
            if is_right_camera:
                gray_path = self.sequence_dir / 'image_1' / f'{index:06d}.png'
                color_path = self.sequence_dir / 'image_3' / f'{index:06d}.png'
            else:
                gray_path = self.sequence_dir / 'image_0' / f'{index:06d}.png'
                color_path = self.sequence_dir / 'image_2' / f'{index:06d}.png'
            
            # 保存图像
            cv2.imwrite(str(gray_path), gray_img)
            cv2.imwrite(str(color_path), color_img)
            
            return True
            
        except Exception as e:
            print(f"⚠️  保存图像失败: {e}")
            import traceback
            traceback.print_exc()
            return False

    def save_lidar_points(self, lidar_data, index):
        """保存激光雷达点云为KITTI格式"""
        try:
            data = lidar_data['data']
            point_step = lidar_data['point_step']
            fields = lidar_data['fields']
            
            # 解析点云字段
            field_map = {}
            for field_name, offset, datatype, count in fields:
                field_map[field_name] = (offset, datatype, count)
            
            # 计算点的数量
            num_points = len(data) // point_step
            points = []
            
            for i in range(num_points):
                point_data = data[i * point_step:(i + 1) * point_step]
                
                # 提取 x, y, z
                x = struct.unpack('f', point_data[0:4])[0] if len(point_data) >= 4 else 0.0
                y = struct.unpack('f', point_data[4:8])[0] if len(point_data) >= 8 else 0.0
                z = struct.unpack('f', point_data[8:12])[0] if len(point_data) >= 12 else 0.0
                
                # 提取强度信息（如果有）
                if 'intensity' in field_map and len(point_data) >= 16:
                    intensity = struct.unpack('f', point_data[12:16])[0]
                else:
                    intensity = 0.0
                
                points.append([x, y, z, intensity])
            
            if points:
                points_array = np.array(points, dtype=np.float32)
                
                # 保存为KITTI格式（二进制，N×4的float32数组）
                output_path = self.sequence_dir / 'velodyne' / f'{index:06d}.bin'
                points_array.tofile(output_path)
                return True
            
            return False
            
        except Exception as e:
            print(f"⚠️  保存点云失败: {e}")
            return False

    def save_poses(self):
        """保存位姿文件"""
        if not self.poses_data:
            print("⚠️  没有找到位姿数据")
            return
        
        # 按时间戳排序（已经是同步后的数据）
        self.poses_data.sort(key=lambda x: x['timestamp'])
        
        # 保存到 poses/序列号.txt
        poses_file = self.output_dir / 'poses' / f'{self.sequence_id}.txt'
        poses_file.parent.mkdir(exist_ok=True)
        
        with open(poses_file, 'w') as f:
            for pose_data in self.poses_data:
                pose_str = ' '.join(f'{val:.6f}' for val in pose_data['pose'])
                f.write(pose_str + '\n')
        
        # 同时在序列目录下也保存一份
        sequence_poses_file = self.sequence_dir / 'poses.txt'
        with open(sequence_poses_file, 'w') as f:
            for pose_data in self.poses_data:
                pose_str = ' '.join(f'{val:.6f}' for val in pose_data['pose'])
                f.write(pose_str + '\n')
        
        print(f"✅ 保存 {len(self.poses_data)} 个位姿到:")
        print(f"   {poses_file}")
        print(f"   {sequence_poses_file}")

    def save_times(self):
        """保存时间戳文件"""
        if not self.lidar_data:
            print("⚠️  没有找到LiDAR数据用于时间戳")
            return
        
        # 使用LiDAR时间戳作为基准
        lidar_timestamps = [ld['timestamp'] for ld in self.lidar_data]
        lidar_timestamps.sort()
        
        # 保存到 sequences/序列号/times.txt
        times_file = self.sequence_dir / 'times.txt'
        
        with open(times_file, 'w') as f:
            for timestamp in lidar_timestamps:
                # 转换纳秒时间戳为秒（浮点数）
                timestamp_sec = timestamp / 1e9
                f.write(f'{timestamp_sec:.6f}\n')
        
        print(f"✅ 保存 {len(lidar_timestamps)} 个时间戳到 {times_file}")

    def create_calib_file(self):
        """创建标定文件（使用提取的真实标定数据）"""
        calib_file = self.sequence_dir / 'calib.txt'
        
        print(f"📝 生成KITTI标定文件: {calib_file}")
        
        with open(calib_file, 'w') as f:

            
            # 写入相机投影矩阵
            camera_mapping = {
                'camera_left': 'P0',
                'camera_right': 'P1', 
                'camera_color': 'P2',
                'camera_unknown': 'P0'
            }
            
            # 如果有真实的相机标定数据，使用它们
            if self.camera_info:
                for i, (camera_name, camera_data) in enumerate(self.camera_info.items()):
                    if camera_data['P'] is not None:
                        p_name = camera_mapping.get(camera_name, f'P{i}')
                        p_matrix = camera_data['P'].flatten()
                        f.write(f"{p_name}: {' '.join(f'{val:.6e}' for val in p_matrix)}\n\n")
                
                print(f"✅ 使用提取的 {len(self.camera_info)} 个相机标定")
            else:
                # 没有真实标定数据时使用模板
                f.write("P0: 7.215377e+02 0.000000e+00 6.095593e+02 0.000000e+00 0.000000e+00 7.215377e+02 1.728540e+02 0.000000e+00 0.000000e+00 0.000000e+00 1.000000e+00 0.000000e+00\n")
                f.write("P1: 7.215377e+02 0.000000e+00 6.095593e+02 -3.875744e+02 0.000000e+00 7.215377e+02 1.728540e+02 0.000000e+00 0.000000e+00 0.000000e+00 1.000000e+00 0.000000e+00\n")
                f.write("P2: 7.215377e+02 0.000000e+00 6.095593e+02 0.000000e+00 0.000000e+00 7.215377e+02 1.728540e+02 0.000000e+00 0.000000e+00 0.000000e+00 1.000000e+00 0.000000e+00\n")
                f.write("P3: 7.215377e+02 0.000000e+00 6.095593e+02 -3.875744e+02 0.000000e+00 7.215377e+02 1.728540e+02 0.000000e+00 0.000000e+00 0.000000e+00 1.000000e+00 0.000000e+00\n\n")
                print("⚠️  没有找到相机标定，使用模板数据")
            
            
            # 查找激光雷达到相机的变换
            if self.tf_tree:
                print("🔍 计算传感器变换...")
                lidar_to_cam = self.find_transform_chain(['velodyne', 'lidar', 'laser'], 
                                                       ['camera', 'cam', 'optical'])
                
                if lidar_to_cam is not None:
                    tr_matrix = lidar_to_cam[:3, :].flatten()  # 取前3行
                    f.write(f"Tr: {' '.join(f'{val:.6e}' for val in tr_matrix)}\n\n")
                    print("✅ 使用计算得到的真实变换")
                else:
                    f.write("Tr: 4.276802385584e-04 -9.999672484946e-01 -8.084491683471e-03 -1.198459927713e-02 -7.210626507497e-03 8.081198471645e-03 -9.999413164504e-01 -5.403984729748e-02 9.999738645903e-01 4.859485810390e-04 -7.206933692422e-03 -2.921968648686e-01\n\n")
                    print("⚠️  TF链不完整，使用模板变换")
            else:
                f.write("Tr: 4.276802385584e-04 -9.999672484946e-01 -8.084491683471e-03 -1.198459927713e-02 -7.210626507497e-03 8.081198471645e-03 -9.999413164504e-01 -5.403984729748e-02 9.999738645903e-01 4.859485810390e-04 -7.206933692422e-03 -2.921968648686e-01\n\n")
                print("⚠️  没有找到TF数据，使用模板变换")
            
            # 写入内参矩阵（如果有）
            for camera_name, camera_data in self.camera_info.items():
                if camera_data['K'] is not None:
                    # f.write(f"# {camera_name} intrinsic matrix\n")
                    k_matrix = camera_data['K'].flatten()
                    f.write(f"K_{camera_name}: {' '.join(f'{val:.6e}' for val in k_matrix)}\n")
                    
                    if camera_data['D'] is not None:
                        # f.write(f"# {camera_name} distortion coefficients\n") 
                        d_coeffs = camera_data['D']
                        f.write(f"D_{camera_name}: {' '.join(f'{val:.6e}' for val in d_coeffs)}\n")
                    f.write("\n")
        
        print(f"✅ KITTI标定文件生成完成: {calib_file}")

    def print_calibration_summary(self):
        """打印标定数据摘要"""
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

    def convert(self):
        """主转换函数"""
        start_time = time.time()
        print(f"🚀 开始转换 MCAP 到 KITTI Odometry 格式（带时间同步）")
        print(f"📁 输入: {self.input_bag}")
        print(f"📁 输出: {self.output_dir}")
        print(f"🔢 序列号: {self.sequence_id}")
        print(f"⚙️  使用 {self.num_processes} 个进程")
        
        # 首先提取标定数据
        self.extract_calibration_data()
        self.print_calibration_summary()
        
        # 读取所有消息
        all_messages = self.read_bag_messages()
        
        # 将消息分批处理
        batch_size = max(100, len(all_messages) // self.num_processes)
        message_batches = [
            all_messages[i:i+batch_size] 
            for i in range(0, len(all_messages), batch_size)
        ]
        
        print(f"📦 分为 {len(message_batches)} 批处理，每批约 {batch_size} 条消息")
        
        # 多进程处理消息
        with Pool(self.num_processes) as pool:
            results = list(tqdm(
                pool.imap(self.process_message_batch, message_batches),
                total=len(message_batches),
                desc="处理消息"
            ))
        
        # 合并结果
        for poses, cameras, lidars, pose_ts_map in results:
            self.poses_data.extend(poses)
            self.camera_data.extend(cameras)
            self.lidar_data.extend(lidars)
            # 合并位姿时间戳映射
            self.pose_timestamp_map.update(pose_ts_map)
        
        print(f"📊 原始数据统计:")
        print(f"   位姿: {len(self.poses_data)}")
        print(f"   相机: {len(self.camera_data)}")
        print(f"   激光雷达: {len(self.lidar_data)}")
        
        # 时间同步处理
        self.synchronize_with_lidar()
        
        print(f"📊 同步后数据统计:")
        print(f"   位姿: {len(self.poses_data)} (与LiDAR帧数一致)")
        print(f"   相机: {len(self.camera_data)}")
        print(f"   激光雷达: {len(self.lidar_data)}")
        
        # 按时间戳对所有数据排序
        self.camera_data.sort(key=lambda x: x['timestamp'])
        self.lidar_data.sort(key=lambda x: x['timestamp'])
        
        # 保存数据
        print("💾 保存数据文件...")
        
        # 保存位姿和时间戳
        self.save_poses()
        self.save_times()
        
        # 多线程保存图像和点云
        with ThreadPoolExecutor(max_workers=self.num_processes) as executor:
            # 保存相机图像
            camera_futures = []
            for i, camera_data in enumerate(self.camera_data):
                future = executor.submit(self.save_camera_image, camera_data, i)
                camera_futures.append(future)
            
            # 保存激光雷达点云
            lidar_futures = []
            for i, lidar_data in enumerate(self.lidar_data):
                future = executor.submit(self.save_lidar_points, lidar_data, i)
                lidar_futures.append(future)
            
            # 等待完成
            if camera_futures:
                for future in tqdm(camera_futures, desc="保存图像"):
                    future.result()
            
            if lidar_futures:
                for future in tqdm(lidar_futures, desc="保存点云"):
                    future.result()
        
        # 创建标定文件（使用提取的真实数据）
        self.create_calib_file()
        
        elapsed_time = time.time() - start_time
        print(f"✅ 转换完成！用时 {elapsed_time:.2f} 秒")
        print(f"📁 输出目录结构:")
        print(f"   {self.output_dir}/sequences/{self.sequence_id}/")
        print(f"   ├── image_0/      # 左相机灰度图")
        print(f"   ├── image_2/      # 左相机彩色图")
        print(f"   ├── velodyne/     # 激光雷达点云")
        print(f"   ├── poses.txt     # 位姿文件（与LiDAR帧同步）")
        print(f"   ├── times.txt     # 时间戳文件（LiDAR基准）")
        print(f"   └── calib.txt     # 标定文件（含真实标定数据）")


def main():
    parser = argparse.ArgumentParser(
        description="Convert MCAP ROS2 bag to KITTI Odometry dataset format with time synchronization"
    )
    parser.add_argument("input", help="输入MCAP文件路径")
    parser.add_argument("-o", "--output", required=True, help="输出KITTI数据集目录")
    parser.add_argument("-s", "--sequence", default="00", help="序列号 (默认: 00)")
    parser.add_argument("-j", "--jobs", type=int, help="并行进程数（默认：CPU核心数-1）")
    parser.add_argument("-t", "--tolerance", type=int, default=50, help="时间同步容差(ms)")
    parser.add_argument("--dry-run", action="store_true", help="仅分析数据，不执行转换")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"❌ 输入文件不存在: {args.input}")
        return
    
    if args.dry_run:
        print("🔍 仅分析模式，不执行实际转换")
        # 这里可以添加数据分析代码
        return
    
    converter = MCAPToKITTIOdometryConverter(
        input_bag=args.input,
        output_dir=args.output,
        sequence_id=args.sequence,
        num_processes=args.jobs
    )
    
    # 设置时间容差
    converter.time_tolerance_ns = args.tolerance * 1_000_000  # ms to ns
    
    try:
        converter.convert()
    except KeyboardInterrupt:
        print("\n❌ 用户中断转换")
    except Exception as e:
        print(f"❌ 转换失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()