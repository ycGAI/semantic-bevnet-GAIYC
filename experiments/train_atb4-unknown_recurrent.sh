#!/usr/bin/env bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

if [ "$PWD" != "$DIR" ]; then
    echo "Please run the script in the script's residing directory"
    exit 0
fi


model_config=$1
tag=$2

if [ "$tag" != "" ]; then
    out_dir="${model_config%.*}-$tag-logs"
else
    out_dir="${model_config%.*}-logs"
fi


python ../bevnet/train_recurrent.py \
    --model_config="$model_config" \
    --dataset_config="../dataset_configs/atb4_100x100_unknown_recurrent.yaml" \
    --dataset_path="/workspace/data/raw_demo_rosbag/bev_res_yc_fin_sl50_str1/sequences" \
    --output="$out_dir" \
    --batch_size=1 \
    --include_unknown \
    --log_interval=50 \
    "${@:3}"
# bash train_atb4-unknown_recurrent.sh atb4/recurrent/default.yaml --batch_size=1 --n_frame=2 --seq_len=5 --frame_strides 1 2 3