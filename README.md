# BEVNet

Source code for our work **"Semantic Terrain Classification for Off-Road Autonomous Driving"**

[website](https://sites.google.com/view/terrain-traversability/home)
![Alt Text](figs/canal.gif)
Our BEVNet-R on completly unseen data/envrionment. 

## TODOs
- [x] source code upload
- [ ] model weights upload
- [ ] dataset upload
- [ ] Instructions on dataset generation
- [ ] Instructions on inference
- [ ] experiment results
- [ ] arxiv link

## Setup (Incomplete)

Our setup runs python3.6+, the easiest way to setup would be to use [Anaconda](https://www.anaconda.com/):
```
conda create -n bevnet python=3.7
conda activate bevnet
# pip usually prevents issues with compiling spconv.
# please make sure you're installing the version that matches your CUDA env.
pip install torch==1.8.0+cu111 torchvision==0.9.0+cu111 torchaudio==0.8.0 -f https://download.pytorch.org/whl/torch_stable.html
# rest may be installed with this
pip install -r requirements.txt
```

### SpConv: PyTorch Spatially Sparse Convolution Library
We utlize spconv for our 3D convolution network. To install:
```
# 进入 bevnet 目录
cd /workspace/bevnet

# 克隆 spconv 源码
git clone https://github.com/traveller59/spconv.git --recursive
cd spconv
git checkout v1.2.1

# 初始化子模块（重要！）
git submodule update --init --recursive

# 设置 CUDA 架构（根据你的 GPU 调整）
# RTX 3090/3080/3070 (计算能力 8.6)
export TORCH_CUDA_ARCH_LIST="8.6"

# RTX 2080 Ti/2080/2070 (计算能力 7.5)
# export TORCH_CUDA_ARCH_LIST="7.5"

# V100 (计算能力 7.0)
# export TORCH_CUDA_ARCH_LIST="7.0"

# 多架构支持（编译时间较长但兼容性更好）
# export TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6"

# 设置 CUDA 路径
export CUDA_HOME=/usr/local/cuda

# 编译
python setup.py bdist_wheel

# 离开源码目录（重要！避免导入错误）
cd ..

# 安装编译好的 wheel
pip install spconv/dist/spconv-1.2.1-cp38-cp38-linux_x86_64.whl
```

## Datasets
Datasets should be put inside `data/`. For example, `data/semantic_kitti_4class_100x100`.

## Training

### BEVNet-S
Example:
```
cd experiments
bash train_kitti4-unknown_single.sh kitti4_100/single/include_unknown/default.yaml <tag> arg1 arg2 ...
```
Logs and model weights will be stored in a subdirectory of the config file like this: 
`experiments/kitti4_100/single/include_unknown/default-<tag>-logs/`
* `<tag>` is useful when you want to use the same config file but different hyperparameters. For example, if you
  want to do some debugging you can use set `<tag>` to `debug`.
* `arg1 arg2 ...` are command line arguments supported by `train_single.py`. For example, you can pass 
  `--batch_size=4 --log_interval=100`, etc.


### BEVNet-R
The command line formats are the same as BEVNet-S
Example:
```
cd experiments
bash train_kitti4-unknown_recurrent.sh kitti4_100/recurrent/include_unknown/default.yaml <tag> \
--n_frame=6 --seq_len=20 --frame_strides 1 10 20 \
--resume kitti4_100/single/include_unknown/default-logs/model.pth.4 \
--resume_epoch 0
```
Logs and model weights will be stored in a subdirectory of the config file 
`experiments/kitti4_100/recurrent/include_unknown/default-<tag>-logs/`.
