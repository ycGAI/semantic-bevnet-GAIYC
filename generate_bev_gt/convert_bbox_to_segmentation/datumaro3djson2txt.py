import json
import os

def json_to_kitti(json_path, output_dir):

    with open(json_path, 'r') as f:
        data = json.load(f)

    labels = data['categories']['label']['labels']

    os.makedirs(output_dir, exist_ok=True)

    # 遍历每一帧
    for item in data['items']:
        item_id = item['id']  # 使用 JSON 中的 'id' 值
        annotations = item['annotations']

        # 输出 KITTI 格式文件的路径，使用 'id' 命名
        output_path = f"{output_dir}/{item_id}.txt"
        
        with open(output_path, 'w') as f_out:
            # 遍历每个标注
            for annotation in annotations:
                label_id = annotation['label_id']
                label_name = labels[label_id]['name']
                
                # 提取 3D 立方体信息
                position = annotation['position']
                rotation = annotation['rotation']
                scale = annotation['scale']
                
                # KITTI 格式字段
                truncated = 0  # 默认为 0，因为未提供截断信息
                occluded = 1 if annotation['attributes']['occluded'] else 0
                alpha = rotation[2]  # 使用 Z 轴的旋转角作为方向角
                bbox_left = 0.0  # 2D 边界框位置，点云标注中通常为 0
                bbox_top = 0.0
                bbox_right = 0.0
                bbox_bottom = 0.0
                height = scale[2]  # 物体高度
                width = scale[0]   # 物体宽度
                length = scale[1]  # 物体长度
                x = position[0]    # 物体在相机坐标系中的 x 坐标
                y = position[1]    # 物体在相机坐标系中的 y 坐标
                z = position[2]    # 物体在相机坐标系中的 z 坐标
                rotation_y = rotation[2]  # KITTI 中物体绕 Y 轴的旋转角度

                # 将数据写入到 KITTI 格式文件
                f_out.write(f"{label_name} {truncated} {occluded} {alpha} "
                            f"{bbox_left} {bbox_top} {bbox_right} {bbox_bottom} "
                            f"{height} {width} {length} {x} {y} {z} {rotation_y}\n")


json_to_kitti('/workspace/data/raw_demo_rosbag/dataset_yc/sequences/00/labels_fin/annotations/default.json', 
              '/workspace/data/raw_demo_rosbag/dataset_yc/sequences/00/labels_fin')
