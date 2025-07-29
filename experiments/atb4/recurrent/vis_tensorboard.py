#!/usr/bin/env python3
"""
BEVNet单个模型训练可视化
可视化单个模型的训练过程，显示epoch与acc/iou的变化
"""

import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import os
from glob import glob

def read_tensorboard_log(log_dir):
    """
    读取TensorBoard日志数据
    """
    # 查找事件文件
    event_files = glob(os.path.join(log_dir, 'events.out.tfevents.*'))
    if not event_files:
        print(f"错误：在 {log_dir} 中未找到TensorBoard事件文件")
        return None
    
    # 如果有多个事件文件，使用最新的
    event_file = max(event_files, key=os.path.getmtime)
    print(f"读取事件文件: {os.path.basename(event_file)}")
    
    try:
        # 创建事件累加器
        ea = EventAccumulator(event_file)
        ea.Reload()
        
        # 打印可用的标签
        print("\n可用的标量标签:")
        for tag in ea.Tags()['scalars']:
            print(f"  - {tag}")
        
        # 读取数据
        data = {}
        
        # 读取验证集指标
        for metric in ['Val/acc', 'Val/iou', 'Val/loss']:
            if metric in ea.Tags()['scalars']:
                events = ea.Scalars(metric)
                steps = [e.step for e in events]
                values = [e.value for e in events]
                data[metric] = (steps, values)
                print(f"\n{metric}: {len(values)} 个数据点")
        
        # 读取训练集指标（如果有）
        for metric in ['Train/loss']:
            if metric in ea.Tags()['scalars']:
                events = ea.Scalars(metric)
                steps = [e.step for e in events]
                values = [e.value for e in events]
                data[metric] = (steps, values)
        
        return data
        
    except Exception as e:
        print(f"读取日志文件时出错: {e}")
        return None

def plot_training_curves(data, model_name, save_dir='.'):
    """
    绘制训练曲线
    """
    # 创建图表
    fig = plt.figure(figsize=(15, 10))
    
    # 1. Accuracy曲线
    ax1 = plt.subplot(2, 2, 1)
    if 'Val/acc' in data:
        steps, values = data['Val/acc']
        epochs = range(1, len(values) + 1)
        ax1.plot(epochs, values, 'b-', linewidth=2.5, marker='o', 
                markersize=6, markevery=max(1, len(epochs)//20))
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Validation Accuracy', fontsize=12)
        ax1.set_title('Validation Accuracy vs Epoch', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # 标注最高点
        max_acc = max(values)
        max_epoch = values.index(max_acc) + 1
        ax1.annotate(f'Max: {max_acc:.4f}\nEpoch: {max_epoch}',
                    xy=(max_epoch, max_acc),
                    xytext=(max_epoch + len(epochs)*0.1, max_acc - 0.02),
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    
    # 2. IoU曲线
    ax2 = plt.subplot(2, 2, 2)
    if 'Val/iou' in data:
        steps, values = data['Val/iou']
        epochs = range(1, len(values) + 1)
        ax2.plot(epochs, values, 'g-', linewidth=2.5, marker='s',
                markersize=6, markevery=max(1, len(epochs)//20))
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('Validation IoU', fontsize=12)
        ax2.set_title('Validation IoU vs Epoch', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # 标注最高点
        max_iou = max(values)
        max_epoch = values.index(max_iou) + 1
        ax2.annotate(f'Max: {max_iou:.4f}\nEpoch: {max_epoch}',
                    xy=(max_epoch, max_iou),
                    xytext=(max_epoch + len(epochs)*0.1, max_iou - 0.02),
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.7),
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    
    # 3. Acc和IoU组合图
    ax3 = plt.subplot(2, 2, 3)
    if 'Val/acc' in data and 'Val/iou' in data:
        _, acc_values = data['Val/acc']
        _, iou_values = data['Val/iou']
        epochs = range(1, len(acc_values) + 1)
        
        # 双Y轴
        ax3_twin = ax3.twinx()
        
        line1 = ax3.plot(epochs, acc_values, 'b-', linewidth=2.5, label='Accuracy')
        line2 = ax3_twin.plot(epochs, iou_values, 'g-', linewidth=2.5, label='IoU')
        
        ax3.set_xlabel('Epoch', fontsize=12)
        ax3.set_ylabel('Accuracy', fontsize=12, color='b')
        ax3_twin.set_ylabel('IoU', fontsize=12, color='g')
        ax3.tick_params(axis='y', labelcolor='b')
        ax3_twin.tick_params(axis='y', labelcolor='g')
        
        # 合并图例
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax3.legend(lines, labels, loc='lower right')
        
        ax3.set_title('Accuracy & IoU vs Epoch', fontsize=14, fontweight='bold')
        ax3.grid(True, alpha=0.3)
    
    # 4. Loss曲线（如果有）
    ax4 = plt.subplot(2, 2, 4)
    has_loss = False
    if 'Val/loss' in data:
        steps, values = data['Val/loss']
        epochs = range(1, len(values) + 1)
        ax4.plot(epochs, values, 'r-', linewidth=2.5, label='Val Loss',
                marker='o', markersize=6, markevery=max(1, len(epochs)//20))
        has_loss = True
    
    if 'Train/loss' in data:
        steps, values = data['Train/loss']
        # 可能训练loss的记录频率更高，需要降采样
        if len(values) > 100:
            # 每个epoch取平均
            epoch_size = len(values) // len(epochs) if 'Val/loss' in data else len(values) // 20
            averaged_values = []
            for i in range(0, len(values), epoch_size):
                averaged_values.append(np.mean(values[i:i+epoch_size]))
            values = averaged_values
        
        epochs_train = range(1, len(values) + 1)
        ax4.plot(epochs_train, values, 'b--', linewidth=2, label='Train Loss', alpha=0.7)
        has_loss = True
    
    if has_loss:
        ax4.set_xlabel('Epoch', fontsize=12)
        ax4.set_ylabel('Loss', fontsize=12)
        ax4.set_title('Training Loss vs Epoch', fontsize=14, fontweight='bold')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
    else:
        ax4.text(0.5, 0.5, 'No Loss Data Available', 
                ha='center', va='center', transform=ax4.transAxes,
                fontsize=14, color='gray')
        ax4.set_xticks([])
        ax4.set_yticks([])
    
    # 总标题
    plt.suptitle(f'{model_name} - Training Progress', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # 保存图表
    output_path = os.path.join(save_dir, f'{model_name}_training_curves.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n图表已保存: {output_path}")
    
    plt.show()

def print_statistics(data, model_name):
    """
    打印训练统计信息
    """
    print("\n" + "="*60)
    print(f"{model_name} - 训练统计")
    print("="*60)
    
    if 'Val/acc' in data:
        _, acc_values = data['Val/acc']
        print(f"\nAccuracy统计:")
        print(f"  - 总Epochs: {len(acc_values)}")
        print(f"  - 最终值: {acc_values[-1]:.4f}")
        print(f"  - 最高值: {max(acc_values):.4f} (Epoch {acc_values.index(max(acc_values)) + 1})")
        print(f"  - 最低值: {min(acc_values):.4f}")
        print(f"  - 平均值: {np.mean(acc_values):.4f}")
        print(f"  - 标准差: {np.std(acc_values):.4f}")
        
        # 最后10个epoch的稳定性
        if len(acc_values) >= 10:
            last_10_std = np.std(acc_values[-10:])
            print(f"  - 最后10 epochs标准差: {last_10_std:.4f}")
    
    if 'Val/iou' in data:
        _, iou_values = data['Val/iou']
        print(f"\nIoU统计:")
        print(f"  - 总Epochs: {len(iou_values)}")
        print(f"  - 最终值: {iou_values[-1]:.4f}")
        print(f"  - 最高值: {max(iou_values):.4f} (Epoch {iou_values.index(max(iou_values)) + 1})")
        print(f"  - 最低值: {min(iou_values):.4f}")
        print(f"  - 平均值: {np.mean(iou_values):.4f}")
        print(f"  - 标准差: {np.std(iou_values):.4f}")
        
        # 最后10个epoch的稳定性
        if len(iou_values) >= 10:
            last_10_std = np.std(iou_values[-10:])
            print(f"  - 最后10 epochs标准差: {last_10_std:.4f}")
    
    print("="*60)

def main():
    """
    主函数
    """
    # 设置要分析的日志目录
    log_dir = "/workspace/bevnet/experiments/atb4/recurrent/default---n_frame=2-logs"
    model_name = "BEVNet-R_Default_nframe2"
    
    print(f"分析模型: {model_name}")
    print(f"日志目录: {log_dir}")
    
    # 检查目录是否存在
    if not os.path.exists(log_dir):
        print(f"错误：目录不存在 - {log_dir}")
        return
    
    # 读取数据
    data = read_tensorboard_log(log_dir)
    
    if data:
        # 打印统计信息
        print_statistics(data, model_name)
        
        # 绘制图表
        plot_training_curves(data, model_name)
        
        # 导出数据到CSV（可选）
        export_to_csv(data, model_name)
    else:
        print("无法读取数据！")

def export_to_csv(data, model_name):
    """
    将数据导出到CSV文件
    """
    import pandas as pd
    
    # 准备数据
    max_length = max(len(data[key][1]) for key in data if key in ['Val/acc', 'Val/iou'])
    
    df_data = {'epoch': list(range(1, max_length + 1))}
    
    if 'Val/acc' in data:
        _, values = data['Val/acc']
        df_data['val_acc'] = values[:max_length]
    
    if 'Val/iou' in data:
        _, values = data['Val/iou']
        df_data['val_iou'] = values[:max_length]
    
    if 'Val/loss' in data:
        _, values = data['Val/loss']
        df_data['val_loss'] = values[:max_length]
    
    # 创建DataFrame并保存
    df = pd.DataFrame(df_data)
    csv_path = f'{model_name}_training_data.csv'
    df.to_csv(csv_path, index=False)
    print(f"\n数据已导出到: {csv_path}")

if __name__ == "__main__":
    main()