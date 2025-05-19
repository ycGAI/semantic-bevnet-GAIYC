import argparse
import os
from collections import deque
import time
import yaml
from PIL import Image
import multiprocessing
from copy import deepcopy
import numpy as np
import tqdm
import torch
from common import parse_calibration, parse_poses, make_color_map
device = None
"""
1. 主程序流程

参数解析：解析命令行参数，包括配置文件路径、工作进程数、GPU设备和内存分配比例等。
加载配置：读取YAML配置文件，获取数据集路径、输出路径、序列长度和步长等信息。
为每个分割集（训练/验证）执行处理：

创建必要的输出目录结构
对每个序列文件夹进行处理



2. 序列处理
对于每个序列文件夹：

获取点云文件：列出所有激光雷达扫描文件（.bin文件）
读取校准数据和位姿：

解析校准文件（calib.txt）获取传感器内外参
解析位姿文件（poses.txt）并将位姿从相机坐标系转换到激光雷达坐标系


为每个时间点准备处理作业：

收集前后一段时间的点云和标签文件（由sequence_length决定）
创建作业参数列表


并行处理点云序列：

使用多进程池执行gen_costmap函数
保存每个处理结果



3. 鸟瞰图生成（gen_costmap函数）
每个工作进程执行以下步骤：

加载历史点云：

读取一系列时间点的点云和标签数据
转换为齐次坐标表示
将数据保存到历史队列中


选择关键帧：选择历史队列中间的点云作为参考帧
移除移动物体：根据配置中定义的移动类别（如行人、车辆）移除移动物体
合并点云：

使用join_pointclouds函数将历史点云变换到关键帧坐标系下
合并所有时间点的点云


创建成本地图：

调用create_costmap函数将合并的点云转换为BEV语义地图
可选地应用凸包计算限制地图范围


生成可视化图像：

利用调色板（颜色映射）将语义标签转换为RGB图像
同时为单帧点云生成BEV图


返回处理结果：

返回处理后的点云
原始和后处理的标签
BEV语义地图（多帧和单帧版本）
位姿信息



4. 保存结果
对于每个处理结果：

保存处理后的点云数据（velodyne文件夹）
保存原始标签（labels文件夹）
保存后处理标签（postprocessed_labels文件夹）
保存BEV语义地图（bev_labels文件夹）
保存单帧BEV地图（bev_1step_labels文件夹）
保存位姿和BEV投影位姿信息

5. 关键技术点

坐标转换：使用校准矩阵将位姿从相机坐标系转换到激光雷达坐标系
多帧融合：通过合并多个时间点的点云提高BEV地图的密度和质量
语义投影：将3D点云的语义标签投影到2D地图上
并行处理：利用多进程并行处理，提高生成效率
可视化处理：使用颜色映射将语义标签转换为直观的彩色图像
"""


# This is called inside the worker process.
def init(queue):
    global device
    device = queue.get()


def gen_costmap(kwargs, mem_frac):
    from common import (
        remove_moving_objects,
        join_pointclouds,
        create_costmap,
        compute_convex_hull,
    )

    global device
    if device is None:
        # 如果 device 未初始化，使用传入的第一个设备
        import os
        if 'CUDA_VISIBLE_DEVICES' in os.environ:
            device = f"cuda:{os.environ['CUDA_VISIBLE_DEVICES'].split(',')[0]}"
        else:
            device = "cuda:0"
        print(f"初始化设备: {device}")
    
    if torch.cuda.is_available():
        torch.cuda.set_device(device)
        torch.cuda.set_per_process_memory_fraction(mem_frac)
    # torch.cuda.set_device(device)
    # torch.cuda.set_per_process_memory_fraction(mem_frac)

    cfg, cmap, scan_files, label_files, poses = [kwargs[_] for _ in [
        'cfg', 'cmap', 'scan_files', 'label_files', 'poses'
    ]]

    assert len(scan_files) == len(label_files)
    assert len(scan_files) == len(poses)

    history = deque()
    for i in range(len(scan_files)):
        scan = np.fromfile(scan_files[i], dtype=np.float32)
        scan = scan.reshape((-1, 4))#x y z f
        labels = np.fromfile(label_files[i], dtype=np.uint32)
        labels = labels.reshape((-1))

        # convert points to homogenous coordinates (x, y, z, 1)
        points = np.ones((scan.shape))
        points[:, 0:3] = scan[:, 0:3]
        remissions = scan[:, 3]

        scan_ground_frame = scan.copy()
        scan_ground_frame[:, 2] += cfg.get('lidar_height')

        # append current data to history queue.
        history.appendleft({
            "scan": scan.copy(),
            "scan_ground_frame": scan_ground_frame,
            "points": points,
            "labels": labels,
            "remissions": remissions,
            "pose": poses[i].copy(),
            "filename": scan_files[i]
        })
    key_scan_id = len(history) // 2
    key_scan = history[key_scan_id]

    # new_history = remove_moving_objects(history,
    #                                     cfg["moving_classes"],
    #                                     key_scan_id)
    #临时
    new_history = history
    #将历史点云数据序列合并到统一的坐标系下（以关键帧的坐标系为参考）
    cat_points, cat_labels, pc_ids = join_pointclouds(new_history, key_scan["pose"])

    # Create costmap
    #将带有语义标签的3D点云投影到2D栅格地图上
    import ipdb; ipdb.set_trace()
    (costmap, key_scan_postprocess_labels,
     costmap_pose) = create_costmap(cat_points, cat_labels, cfg,
                                    pose=key_scan["pose"],
                                    pc_ids=pc_ids,
                                    key_scan_id=key_scan_id)

    if cfg.get('convex_hull', False):
        cvx_hull = compute_convex_hull(costmap)
        costmap[np.logical_not(cvx_hull)] = 4

    costimg = Image.fromarray(costmap, mode='P')
    costimg.putpalette(cmap)

    # Just the current snapshot
    points_1step, labels_1step, _ = join_pointclouds([key_scan], key_scan["pose"])
    costmap_1step, _, _ = create_costmap(points_1step, labels_1step, cfg,
                                         force_return_cmap=True)

    costimg_1step = Image.fromarray(costmap_1step, mode='P')
    costimg_1step.putpalette(cmap)

    return {
        'scan_ground_frame': key_scan['scan_ground_frame'],
        'labels': key_scan['labels'],
        'postprocessed_labels': key_scan_postprocess_labels,
        'costimg': costimg,
        'costimg_1step': costimg_1step,
        'pose': poses[key_scan_id],
        'costmap_pose': costmap_pose,
    }
# def gen_costmap(kwargs, mem_frac):
#     from common import (
#         remove_moving_objects,
#         join_pointclouds,
#         create_costmap,
#         compute_convex_hull,
#     )

#     global device
#     torch.cuda.set_device(device)
#     torch.cuda.set_per_process_memory_fraction(mem_frac)

#     cfg, cmap, scan_files, label_files, poses = [kwargs[_] for _ in [
#         'cfg', 'cmap', 'scan_files', 'label_files', 'poses'
#     ]]

#     assert len(scan_files) == len(label_files)
#     assert len(scan_files) == len(poses)

#     history = deque()

#     for i in range(len(scan_files)):
#         scan = np.fromfile(scan_files[i], dtype=np.float32)
#         scan = scan.reshape((-1, 4))
#         labels = np.fromfile(label_files[i], dtype=np.uint32)
#         labels = labels.reshape((-1))

#         # 转换点为齐次坐标 (x, y, z, 1)
#         points = np.ones((scan.shape))
#         points[:, 0:3] = scan[:, 0:3]
#         remissions = scan[:, 3]

#         scan_ground_frame = scan.copy()
#         scan_ground_frame[:, 2] += cfg.get('lidar_height')

#         # 将当前数据添加到历史队列
#         history.appendleft({
#             "scan": scan.copy(),
#             "scan_ground_frame": scan_ground_frame,
#             "points": points,
#             "labels": labels,
#             "remissions": remissions,
#             "pose": poses[i].copy(),
#             "filename": scan_files[i]
#         })

#     key_scan_id = len(history) // 2
#     key_scan = history[key_scan_id]

#     # new_history = remove_moving_objects(history,
#     #                                   cfg["moving_classes"],
#     #                                   key_scan_id)
#     new_history = history
#     cat_points, cat_labels, pc_ids = join_pointclouds(new_history, key_scan["pose"])

#     # 创建 costmap
#     (costmap, key_scan_postprocess_labels,
#      costmap_pose) = create_costmap(cat_points, cat_labels, cfg,
#                                   pose=key_scan["pose"],
#                                   pc_ids=pc_ids,
#                                   key_scan_id=key_scan_id,
#                                   force_return_cmap=True)  # 添加这个参数强制返回 costmap

#     # 检查 costmap 是否为 None
#     if costmap is None:
#         # 如果仍然是 None，创建一个空的 costmap
#         map_cfg = cfg["costmap"]
#         h = int(np.floor((map_cfg["maxy"] - map_cfg["miny"]) / map_cfg["gridh"]))
#         w = int(np.floor((map_cfg["maxx"] - map_cfg["minx"]) / map_cfg["gridw"]))
#         costmap = np.full((h, w), 255, dtype=np.uint8)  # 全部填充为 255（未知）

#     if cfg.get('convex_hull', False):
#         cvx_hull = compute_convex_hull(costmap)
#         costmap[np.logical_not(cvx_hull)] = 4

#     costimg = Image.fromarray(costmap, mode='P')
#     costimg.putpalette(cmap)

#     # 只处理当前快照
#     points_1step, labels_1step, _ = join_pointclouds([key_scan], key_scan["pose"])
#     costmap_1step, _, _ = create_costmap(points_1step, labels_1step, cfg,
#                                        force_return_cmap=True)

#     # 确保 costmap_1step 不为 None
#     if costmap_1step is None:
#         map_cfg = cfg["costmap"]
#         h = int(np.floor((map_cfg["maxy"] - map_cfg["miny"]) / map_cfg["gridh"]))
#         w = int(np.floor((map_cfg["maxx"] - map_cfg["minx"]) / map_cfg["gridw"]))
#         costmap_1step = np.full((h, w), 255, dtype=np.uint8)

#     costimg_1step = Image.fromarray(costmap_1step, mode='P')
#     costimg_1step.putpalette(cmap)

#     # 如果 key_scan_postprocess_labels 为 None，创建一个默认值
#     if key_scan_postprocess_labels is None:
#         key_scan_postprocess_labels = np.full_like(key_scan['labels'], 255, dtype=np.uint32)

#     return {
#         'scan_ground_frame': key_scan['scan_ground_frame'],
#         'labels': key_scan['labels'],
#         'postprocessed_labels': key_scan_postprocess_labels,
#         'costimg': costimg,
#         'costimg_1step': costimg_1step,
#         'pose': poses[key_scan_id],
#         'costmap_pose': costmap_pose if costmap_pose is not None else np.eye(3),
#     }

def gen_voxel_costmap(kwargs, mem_frac):
    from common import (
        remove_moving_objects,
        join_pointclouds,
        create_costmap,
        compute_convex_hull,
    )

    global device
    torch.cuda.set_device(device)
    torch.cuda.set_per_process_memory_fraction(mem_frac)

    cfg, cmap, scan_files, label_files, poses = [kwargs[_] for _ in [
        'cfg', 'cmap', 'scan_files', 'label_files', 'poses'
    ]]

    assert len(scan_files) == len(label_files)
    assert len(scan_files) == len(poses)

    history = deque()

    for i in range(len(scan_files)):
        scan = np.fromfile(scan_files[i], dtype=np.float32)
        scan = scan.reshape((-1, 4))
        labels = np.fromfile(label_files[i], dtype=np.uint32)
        labels = labels.reshape((-1))

        # convert points to homogenous coordinates (x, y, z, 1)
        points = np.ones((scan.shape))
        points[:, 0:3] = scan[:, 0:3]
        remissions = scan[:, 3]

        scan_ground_frame = scan.copy()
        scan_ground_frame[:, 2] += cfg.get('lidar_height')

        # append current data to history queue.
        history.appendleft({
            "scan": scan.copy(),
            "scan_ground_frame": scan_ground_frame,
            "points": points,
            "labels": labels,
            "remissions": remissions,
            "pose": poses[i].copy(),
            "filename": scan_files[i]
        })

    key_scan_id = len(history) // 2
    key_scan = history[key_scan_id]

    # new_history = remove_moving_objects(history,
    #                                     cfg["moving_classes"],
    #                                     key_scan_id)
    #临时
    new_history = history
    cat_points, cat_labels, pc_ids = join_pointclouds(new_history, key_scan["pose"])

    # Create costmap
    (costmap, key_scan_postprocess_labels,
     costmap_pose) = create_costmap(cat_points, cat_labels, cfg,
                                    pose=key_scan["pose"],
                                    pc_ids=pc_ids,
                                    key_scan_id=key_scan_id)

    if cfg.get('convex_hull', False):
        cvx_hull = compute_convex_hull(costmap)
        costmap[np.logical_not(cvx_hull)] = 4

    costimg = Image.fromarray(costmap, mode='P')
    costimg.putpalette(cmap)

    # Just the current snapshot
    points_1step, labels_1step, _ = join_pointclouds([key_scan], key_scan["pose"])
    costmap_1step, _, _ = create_costmap(points_1step, labels_1step, cfg,
                                         force_return_cmap=True)

    costimg_1step = Image.fromarray(costmap_1step, mode='P')
    costimg_1step.putpalette(cmap)

    return {
        'scan_ground_frame': key_scan['scan_ground_frame'],
        'labels': key_scan['labels'],
        'postprocessed_labels': key_scan_postprocess_labels,
        'costimg': costimg,
        'costimg_1step': costimg_1step,
        'pose': poses[key_scan_id],
        'costmap_pose': costmap_pose,
    }


if __name__ == '__main__':
    def run():
        start_time = time.time()

        parser = argparse.ArgumentParser("./generate_parallel.py")

        parser.add_argument(
            '--config',
            '-c',
            required=True,
            help='path to the config file')

        parser.add_argument(
            '--n_worker',
            type=int,
            required=True,
            help='Number of workers.')

        parser.add_argument(
            '--devices',
            type=str,
            required=False,
            default='cuda',
            help='A comma-separated list of cuda devices.'
        )

        parser.add_argument(
            '--mem_frac',
            type=float,
            required=True,
            help='GPU memory fraction per worker. Used to avoid out-of-memory issue.')

        FLAGS, unparsed = parser.parse_known_args()

        with open(FLAGS.config, 'r') as stream:
            cfg = yaml.safe_load(stream)

        FLAGS.dataset = cfg["input"]
        FLAGS.output = cfg["output"]

        sequence_length = cfg["sequence_length"]
        stride = cfg.get('stride', 1)

        sequences_dir = os.path.join(FLAGS.dataset, "sequences")
        cmap = make_color_map(cfg)
        """
        首先创建一个256×3大小的零矩阵semantic_cmap，用于存储RGB颜色值
        对0到255的每个标签值进行遍历（加1是为了匹配KITTI数据集的标签规范）
        检查该标签值是否存在于配置文件的learning_map_inv（逆映射表）中
        如果存在，从配置文件的color_map中获取对应的BGR颜色值
        将BGR颜色转换为RGB颜色（通过[::-1]反转顺序）
        将RGB颜色值存储到颜色映射表中对应的位置
        最后为未分类的区域（标签值255）设置一个特定的颜色值(255, 250, 230)，通常是浅米色

        这个函数在数据集处理过程中非常重要，因为语义分割的结果通常是灰度图像，其中每个像素的值代表一个类别。
        通过颜色映射，这些灰度图像可以转换为彩色图像，使得不同的语义类别（如道路、植被、建筑物等）能够直观地区分开来，方便研究人员进行分析和评估。
        这里的颜色是指bev分割图中的颜色，而不是点云的颜色。

        256×3只是调色板的大小，不是最终BEV图像的尺寸

        """

        for split in cfg["split"]:
            sequence_folders = cfg["split"][split]

            # Output directories
            output_folder = os.path.join(FLAGS.output, "sequences", split)
            velodyne_folder = os.path.join(output_folder, "velodyne")
            velodyne_labels_folder = os.path.join(output_folder, "labels")
            velodyne_pp_labels_folder = os.path.join(output_folder, "postprocessed_labels")
            labels_folder = os.path.join(output_folder, "bev_labels")
            labels_1step_folder = os.path.join(output_folder, "bev_1step_labels")

            os.makedirs(output_folder, exist_ok=True)
            os.makedirs(velodyne_folder, exist_ok=True)
            os.makedirs(velodyne_labels_folder, exist_ok=True)
            os.makedirs(velodyne_pp_labels_folder, exist_ok=True)
            os.makedirs(labels_folder, exist_ok=True)
            os.makedirs(labels_1step_folder, exist_ok=True)

            counters = dict()
            all_poses = []
            all_costmap_poses = []

            counter = 0

            for folder in sequence_folders:
                input_folder = os.path.join(sequences_dir, folder)
                scan_files = [
                    f for f in sorted(os.listdir(os.path.join(input_folder,
                                                              'velodyne')))
                    if f.endswith(".bin")
                ]

                calibration = parse_calibration(os.path.join(input_folder, 'calib.txt'))
                """
                P0、P1、P2、P3
                这些是四个相机的投影矩阵（3×4矩阵），用于将3D点投影到对应相机的图像平面上：

                P0: 左侧灰度相机的投影矩阵
                P1: 右侧灰度相机的投影矩阵
                P2: 左侧彩色相机的投影矩阵
                P3: 右侧彩色相机的投影矩阵

                每个投影矩阵包含相机的内参和外参信息：

                前3×3部分包含焦距和主点坐标
                第四列包含平移信息

                Tr
                这是激光雷达与相机之间的变换矩阵，也称为"外参"。它定义了从激光雷达坐标系到相机坐标系的变换（通常是到相机0的变换）。
                这个3×4矩阵包含：

                3×3旋转矩阵
                最后一列为平移向量
"""
                poses = parse_poses(os.path.join(input_folder, 'poses.txt'), calibration)
                """
                这个函数通过结合校准数据（calibration中的Tr参数），将位姿从相机坐标系转换到激光雷达坐标系。
                这一步很重要，因为在KITTI数据集中，poses.txt记录的是相机坐标系下的位姿，而我们需要在激光雷达坐标系下处理数据。
                """

                start_counter = counter
                print("Processing {} ".format(folder), end="", flush=True)

                job_args = []

                for i in range(len(scan_files)):
                    start_idx = i - (sequence_length // 2 * stride)
                    history = []

                    data_idxs = [start_idx + _ * stride for _ in range(sequence_length)]
                    if data_idxs[0] < 0 or data_idxs[-1] >= len(scan_files):
                        # Out of range
                        continue
                    
                    for data_idx in data_idxs:
                        scan_file = scan_files[data_idx]
                        basename = os.path.splitext(scan_file)[0]
                        scan_path = os.path.join(input_folder, "velodyne", scan_file)
                        label_path = os.path.join(input_folder, "labels", basename + ".label")
                        history.append((scan_path, label_path, poses[data_idx]))
                    history = history[::-1]

                    if len(history) < sequence_length:
                        continue
                    assert len(history) == sequence_length

                    hist_scan_files, hist_label_files, hist_poses = zip(*list(history))
                    job_args.append({
                        'cfg': cfg,
                        'cmap': cmap,
                        'scan_files': hist_scan_files,
                        'label_files': hist_label_files,
                        'poses': hist_poses,
                    })

                # devices = FLAGS.devices.split(',')
                # manager = multiprocessing.Manager()
                # worker_init_queue = manager.Queue()
                # for i in range(FLAGS.n_worker):
                #     worker_init_queue.put(devices[i % len(devices)])

                # ctx = multiprocessing.get_context('spawn')
                # with ctx.Pool(FLAGS.n_worker, initializer=init, initargs=(worker_init_queue,)) as pool:
                #     async_results = [pool.apply_async(gen_costmap, (job, FLAGS.mem_frac)) for job in job_args]
                #     for future in tqdm.tqdm(async_results):
                #         ret = future.get()
                #         ret['scan_ground_frame'].tofile(
                #             os.path.join(velodyne_folder, '{:05d}.bin'.format(counter)))
                #         ret["labels"].tofile(
                #             os.path.join(velodyne_labels_folder, '{:05d}.label'.format(counter)))
                #         ret['postprocessed_labels'].tofile(
                #             os.path.join(velodyne_pp_labels_folder, '{:05d}.label'.format(counter)))
                #         ret['costimg'].save(
                #             os.path.join(labels_folder, "{:05d}.png".format(counter)))
                #         ret['costimg_1step'].save(
                #             os.path.join(labels_1step_folder, "{:05d}.png".format(counter)))

                #         all_poses.append(ret['pose'][:3])
                #         all_costmap_poses.append(ret['costmap_pose'][:2])
                #         counter += 1

                #     counters[folder] = [start_counter, counter]
                # 单线程调试版本
                devices = FLAGS.devices.split(',')
                device = devices[0]

                # 设置 CUDA 设备
                if 'cuda' in device:
                    torch.cuda.set_device(device)
                    torch.cuda.set_per_process_memory_fraction(FLAGS.mem_frac)

                # 处理任务
                for idx, job in enumerate(tqdm.tqdm(job_args, desc="处理")):
                    print(f"\n处理任务 {idx+1}/{len(job_args)}")
                    
                    # 设置断点 - 在这里可以调试 gen_costmap 函数
                    # breakpoint()  # Python 3.7+
                    # 或者使用 import pdb; pdb.set_trace()
                    
                    # 直接调用 gen_costmap 函数
                    ret = gen_costmap(job, FLAGS.mem_frac)
                    
                    # 保存结果
                    ret['scan_ground_frame'].tofile(
                        os.path.join(velodyne_folder, '{:05d}.bin'.format(counter)))
                    ret["labels"].tofile(
                        os.path.join(velodyne_labels_folder, '{:05d}.label'.format(counter)))
                    ret['postprocessed_labels'].tofile(
                        os.path.join(velodyne_pp_labels_folder, '{:05d}.label'.format(counter)))
                    ret['costimg'].save(
                        os.path.join(labels_folder, "{:05d}.png".format(counter)))
                    ret['costimg_1step'].save(
                        os.path.join(labels_1step_folder, "{:05d}.png".format(counter)))
                    
                    all_poses.append(ret['pose'][:3])
                    all_costmap_poses.append(ret['costmap_pose'][:2])
                    counter += 1

                counters[folder] = [start_counter, counter]

            # Save metadatas.
            yaml.dump(counters, open(os.path.join(output_folder, 'counters.yaml'), 'w'))
            # Flatten 4x4 matrix to 1D
            if len(all_poses) > 0:
                all_poses = np.array(all_poses)
                all_poses = all_poses.reshape((all_poses.shape[0], -1))
                np.savetxt(os.path.join(output_folder, 'poses.txt'), all_poses, fmt='%.8e')

            if len(all_costmap_poses) > 0:
                all_costmap_poses = np.array(all_costmap_poses)
                all_costmap_poses = all_costmap_poses.reshape((all_costmap_poses.shape[0], -1))
                np.savetxt(os.path.join(output_folder, 'costmap_poses.txt'),
                           all_costmap_poses, fmt='%.8e')

        print("execution time: {}".format(time.time() - start_time))

    run()
