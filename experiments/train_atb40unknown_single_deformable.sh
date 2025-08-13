#!/usr/bin/env bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
if [ "$PWD" != "$DIR" ]; then
    echo "Please run the script in the script's residing directory"
    exit 0
fi

model_config=$1
tag=${2:-"deform_10k"}

out_dir="${model_config%.*}-$tag-logs"

echo "======================================"
echo "Training Deformable FChardNet (10k frames)"
echo "Config: $model_config"
echo "Output: $out_dir"
echo "======================================"

python ../bevnet/train_single.py \
    --model_config="$model_config" \
    --dataset_config="../dataset_configs/atb4_100x100_unknown_single.yaml" \
    --dataset_path="/workspace/data/rellis_3d/rellis_4class_100x100_2_sl50tr1/sequences" \
    --output="$out_dir" \
    --batch_size=2 \
    --include_unknown \
    --epochs=60 \
    --log_interval=100 \
    --lr=5e-4 \
    --lr_decay_epoch=15 \
    --lr_decay=0.8 \
    --progressive_deformable \
    --warmup_epochs=5 \
    --deform_lr_factor=0.1 \
    --grad_clip=5.0 \
    --save_checkpoint_every=5 \
    "${@:3}"