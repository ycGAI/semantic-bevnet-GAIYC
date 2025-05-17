#!/bin/bash

# 针对您的数据集的转换脚本
# 数据路径：/media/gyc/Backup Plus3/gyc/thesis/raw_demo_rosbag/dataset

# 设置变量
DATASET_ROOT="/media/gyc/Backup Plus3/gyc/thesis/raw_demo_rosbag/dataset"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE="${SCRIPT_DIR}/config.yaml"

# 检查路径是否存在
if [ ! -d "$DATASET_ROOT" ]; then
    echo "错误：数据集路径不存在：$DATASET_ROOT"
    exit 1
fi

# 检查配置文件
if [ ! -f "$CONFIG_FILE" ]; then
    echo "错误：配置文件不存在：$CONFIG_FILE"
    echo "请先创建 config.yaml 文件"
    exit 1
fi

# 显示信息
echo "=== 开始转换 ==="
echo "数据集路径：$DATASET_ROOT"
echo "配置文件：$CONFIG_FILE"
echo ""

# 检查序列00是否存在
SEQUENCE_PATH="$DATASET_ROOT/sequences/00"
if [ ! -d "$SEQUENCE_PATH" ]; then
    echo "错误：序列00不存在：$SEQUENCE_PATH"
    exit 1
fi

# 检查必要的目录
VELODYNE_DIR="$SEQUENCE_PATH/velodyne"
LABELS_DIR="$SEQUENCE_PATH/labels"

if [ ! -d "$VELODYNE_DIR" ]; then
    echo "错误：点云目录不存在：$VELODYNE_DIR"
    exit 1
fi

if [ ! -d "$LABELS_DIR" ]; then
    echo "错误：标注目录不存在：$LABELS_DIR"
    exit 1
fi

# 统计文件数量
VELODYNE_COUNT=$(find "$VELODYNE_DIR" -name "*.bin" | wc -l)
LABELS_COUNT=$(find "$LABELS_DIR" -name "*.txt" | wc -l)

echo "找到 $VELODYNE_COUNT 个点云文件"
echo "找到 $LABELS_COUNT 个标注文件"
echo ""

# 运行转换（使用多进程版本）
echo "=== 开始处理序列00（多进程模式） ==="
python3 "${SCRIPT_DIR}/multiprocess_converter.py" \
    --dataset_root "$DATASET_ROOT" \
    --sequence "00" \
    --config "$CONFIG_FILE"

# 检查转换结果
if [ $? -eq 0 ]; then
    echo ""
    echo "=== 转换完成 ==="
    
    # 检查输出目录
    FILTERED_DIR="$SEQUENCE_PATH/velodyne_filtered"
    SEGMENTATION_DIR="$SEQUENCE_PATH/labels_segmentation"
    
    if [ -d "$FILTERED_DIR" ]; then
        FILTERED_COUNT=$(find "$FILTERED_DIR" -name "*.bin" | wc -l)
        echo "生成了 $FILTERED_COUNT 个过滤后的点云文件"
    fi
    
    if [ -d "$SEGMENTATION_DIR" ]; then
        SEGMENTATION_COUNT=$(find "$SEGMENTATION_DIR" -name "*.label" | wc -l)
        echo "生成了 $SEGMENTATION_COUNT 个分割标注文件"
    fi
    
    # 检查统计文件
    STATS_FILE="$DATASET_ROOT/conversion_stats.json"
    if [ -f "$STATS_FILE" ]; then
        echo "统计信息已保存到：$STATS_FILE"
    fi
    
    # 检查日志文件
    LOG_FILE="$DATASET_ROOT/conversion.log"
    if [ -f "$LOG_FILE" ]; then
        echo "日志文件已保存到：$LOG_FILE"
    fi
    
    echo ""
    echo "=== 开始可视化检查 ==="
    
    # 找到第一个可用的帧进行可视化
    if [ -d "$FILTERED_DIR" ]; then
        FIRST_FRAME=$(ls "$FILTERED_DIR"/*.bin 2>/dev/null | head -1)
        if [ -n "$FIRST_FRAME" ]; then
            FRAME_NAME=$(basename "$FIRST_FRAME" .bin)
            echo "可视化帧：$FRAME_NAME"
            
            python3 "${SCRIPT_DIR}/visualize_segmentation.py" \
                --dataset_root "$DATASET_ROOT" \
                --sequence "00" \
                --frame "$FRAME_NAME" \
                --compare
        fi
    fi
    
else
    echo ""
    echo "=== 转换失败 ==="
    echo "请检查日志文件获取更多信息"
fi