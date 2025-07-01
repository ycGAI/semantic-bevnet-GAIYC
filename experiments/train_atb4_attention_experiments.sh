#!/bin/bash
# 设置基础路径
EXPERIMENT_DIR="/workspace/bevnet/experiments/atb4/single"
DATASET_CONFIG="../dataset_configs/atb4_100x100_unknown_single.yaml" 
DATASET_PATH="/workspace/data/raw_demo_rosbag/bev_res_yc_fin_sl50_str1/sequences"

# 实验1: 基线模型（无注意力）
# echo "Training baseline model without attention..."
# bash train_atb4-unknown_single.sh ${EXPERIMENT_DIR}/baseline_no_attention.yaml baseline

# 实验2: 轻量级SE（只在分类器）
echo "Training with SE attention in classifier only..."
bash train_atb4-unknown_single.sh ${EXPERIMENT_DIR}/se_classifier_only.yaml se_light

# 实验3: 完整SE注意力
echo "Training with full SE attention..."
bash train_atb4-unknown_single.sh ${EXPERIMENT_DIR}/se_attention.yaml se_full

# 实验4: CBAM注意力
# echo "Training with CBAM attention..."
# bash train_atb4-unknown_single.sh ${EXPERIMENT_DIR}/cbam_attention.yaml cbam

# 实验5: Self-Attention（可选，计算量大）
# echo "Training with self-attention..."
# bash train_atb4-unknown_single.sh ${EXPERIMENT_DIR}/self_attention.yaml self_att

# 实验6: 混合注意力
# echo "Training with mixed attention..."
# bash train_atb4-unknown_single.sh ${EXPERIMENT_DIR}/mixed_attention.yaml mixed

---
#!/bin/bash
# 文件名: experiments/train_atb4_single_attention.sh
# 单个实验的训练脚本

CONFIG_FILE=$1
TAG=$2

if [ -z "$CONFIG_FILE" ] || [ -z "$TAG" ]; then
    echo "Usage: $0 <config_file> <tag>"
    echo "Example: $0 atb4/single/se_attention.yaml se_test"
    exit 1
fi

# 打印配置信息
echo "================================"
echo "Training ATB4 with Attention"
echo "Config: $CONFIG_FILE"
echo "Tag: $TAG"
echo "================================"

# 运行训练
python ../bevnet/train.py \
    --model_config="$CONFIG_FILE" \
    --dataset_config="../dataset_configs/atb4.yaml" \
    --dataset_path="../data/atb4_dataset/" \
    --output="${CONFIG_FILE%.*}-$TAG-logs" \
    --batch_size=4 \
    --log_interval=50 \
    --epochs=15 \
    "${@:3}"

# 训练完成后评估
echo "Training completed. Running evaluation..."
python ../bevnet/train.py \
    --model_config="$CONFIG_FILE" \
    --dataset_config="../dataset_configs/atb4.yaml" \
    --dataset_path="../data/atb4_dataset/" \
    --output="${CONFIG_FILE%.*}-$TAG-logs" \
    --test \
    --resume="${CONFIG_FILE%.*}-$TAG-logs/best.pth"