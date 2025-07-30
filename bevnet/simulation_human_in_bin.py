import numpy as np

def generate_human_point_cloud(center_x, center_y, num_points=200):
    """
    生成一个人形点云
    
    Args:
        center_x: 人的中心x坐标
        center_y: 人的中心y坐标
        num_points: 点云数量
    
    Returns:
        points: Nx4 array (x, y, z, intensity)
    """
    points = []
    
    # 人体参数
    height = np.random.uniform(1.6, 1.8)  # 身高
    shoulder_width = np.random.uniform(0.4, 0.5)  # 肩宽
    body_depth = np.random.uniform(0.2, 0.3)  # 身体厚度
    
    # 1. 头部（球形）
    head_radius = 0.12
    head_center_z = height - 0.15
    n_head = num_points // 5
    
    for i in range(n_head):
        theta = np.random.uniform(0, 2*np.pi)
        phi = np.random.uniform(0, np.pi)
        r = head_radius * (0.8 + 0.2 * np.random.rand())
        
        x = center_x + r * np.sin(phi) * np.cos(theta)
        y = center_y + r * np.sin(phi) * np.sin(theta)
        z = head_center_z + r * np.cos(phi)
        intensity = np.random.uniform(10, 30)
        
        points.append([x, y, z, intensity])
    
    # 2. 躯干（圆柱形）
    trunk_top = height - 0.3
    trunk_bottom = 0.7
    n_trunk = num_points // 2
    
    for i in range(n_trunk):
        z = np.random.uniform(trunk_bottom, trunk_top)
        theta = np.random.uniform(0, 2*np.pi)
        
        # 椭圆形横截面
        r_x = shoulder_width/2 * (1 - 0.3 * (z - trunk_bottom)/(trunk_top - trunk_bottom))
        r_y = body_depth/2
        
        x = center_x + r_x * np.cos(theta)
        y = center_y + r_y * np.sin(theta)
        intensity = np.random.uniform(10, 30)
        
        points.append([x, y, z, intensity])
    
    # 3. 双腿（两个圆柱）
    leg_top = 0.8
    leg_bottom = 0.0
    leg_radius = 0.08
    n_legs = num_points - n_head - n_trunk
    
    for i in range(n_legs // 2):
        # 左腿
        z = np.random.uniform(leg_bottom, leg_top)
        theta = np.random.uniform(0, 2*np.pi)
        r = leg_radius * (0.8 + 0.2 * np.random.rand())
        
        x = center_x - 0.15 + r * np.cos(theta)
        y = center_y + r * np.sin(theta)
        intensity = np.random.uniform(10, 30)
        
        points.append([x, y, z, intensity])
        
        # 右腿
        z = np.random.uniform(leg_bottom, leg_top)
        theta = np.random.uniform(0, 2*np.pi)
        r = leg_radius * (0.8 + 0.2 * np.random.rand())
        
        x = center_x + 0.15 + r * np.cos(theta)
        y = center_y + r * np.sin(theta)
        intensity = np.random.uniform(10, 30)
        
        points.append([x, y, z, intensity])
    
    return np.array(points)

def add_humans_to_point_cloud(original_points, human_positions, num_points_per_human=200):
    """
    在原始点云中添加人形点云
    
    Args:
        original_points: 原始点云 Nx4
        human_positions: 人的位置列表 [(x, y), ...]
        num_points_per_human: 每个人的点数
    
    Returns:
        combined_points: 合并后的点云
    """
    all_points = [original_points]
    
    for x, y in human_positions:
        human_points = generate_human_point_cloud(x, y, num_points_per_human)
        all_points.append(human_points)
        print(f"Generated human at ({x:.1f}, {y:.1f}) with {len(human_points)} points")
    
    combined_points = np.vstack(all_points)
    return combined_points

# 主程序
if __name__ == '__main__':
    # 1. 加载原始点云
    original_file = '/workspace/data/raw_demo_rosbag/bev_res_yc_fin_sl50tr1/sequences/train/velodyne/00000.bin'
    original_points = np.fromfile(original_file, dtype=np.float32).reshape(-1, 4)
    print(f"Original point cloud: {len(original_points)} points")
    
    # 2. 定义人的位置（前方10米和后方10米）
    human_positions = [
        (10.0, 0.0),   # 前方10米
        (-10.0, 0.0),  # 后方10米
    ]
    
    # 3. 生成带人的点云
    combined_points = add_humans_to_point_cloud(
        original_points, 
        human_positions, 
        num_points_per_human=300  # 每个人300个点
    )
    
    print(f"Combined point cloud: {len(combined_points)} points")
    
    # 4. 保存新的点云文件
    output_file = '/workspace/data/raw_demo_rosbag/bev_res_yc_fin_sl50tr1/sequences/train/velodyne/000000_with_humans.bin'
    combined_points.astype(np.float32).tofile(output_file)
    print(f"Saved to: {output_file}")
    
    # 5. 打印统计信息
    print(f"\nAdded {len(human_positions)} humans")
    print(f"Total new points: {len(combined_points) - len(original_points)}")