#!/usr/bin/env python3
"""
BEVNet训练数据拼接脚本
将同一配置下的多次训练数据拼接成连续的曲线
"""

import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import os
from glob import glob
from datetime import datetime
import pandas as pd

def read_all_runs(log_dir):
    """
    读取一个日志目录下的所有训练运行，并按时间顺序排序
    """
    event_files = glob(os.path.join(log_dir, 'events.out.tfevents.*'))
    if not event_files:
        return None
    
    # 按文件创建时间排序
    event_files.sort(key=os.path.getctime)
    
    all_runs = []
    
    for event_file in event_files:
        try:
            ea = EventAccumulator(event_file)
            ea.Reload()
            
            run_data = {}
            
            # 读取所有感兴趣的标签
            for tag in ['Val/acc', 'Val/iou', 'Train/loss', 'Val/loss']:
                if tag in ea.Tags()['scalars']:
                    events = ea.Scalars(tag)
                    if events:
                        run_data[tag] = {
                            'steps': [e.step for e in events],
                            'values': [e.value for e in events],
                            'wall_times': [e.wall_time for e in events]
                        }
            
            if run_data:  # 如果有数据
                all_runs.append({
                    'file': event_file,
                    'data': run_data,
                    'start_time': os.path.getctime(event_file)
                })
                
        except Exception as e:
            print(f"读取 {event_file} 时出错: {e}")
    
    return all_runs

def concatenate_runs(all_runs):
    """
    将多次训练运行拼接成连续的数据
    """
    if not all_runs:
        return None
    
    concatenated_data = {}
    
    # 获取所有可用的标签
    all_tags = set()
    for run in all_runs:
        all_tags.update(run['data'].keys())
    
    # 对每个标签进行拼接
    for tag in all_tags:
        all_steps = []
        all_values = []
        current_offset = 0
        
        for i, run in enumerate(all_runs):
            if tag in run['data']:
                steps = run['data'][tag]['steps']
                values = run['data'][tag]['values']
                
                if i > 0:
                    # 计算步骤偏移量（基于前一次运行的最后步骤）
                    current_offset = all_steps[-1] + 1 if all_steps else 0
                
                # 添加偏移后的步骤
                adjusted_steps = [s + current_offset for s in steps]
                
                all_steps.extend(adjusted_steps)
                all_values.extend(values)
        
        if all_steps:
            concatenated_data[tag] = (all_steps, all_values)
    
    return concatenated_data

def analyze_training_continuity(all_runs):
    """
    分析训练的连续性，打印详细信息
    """
    print("\n训练运行分析:")
    print("-" * 60)
    
    for i, run in enumerate(all_runs):
        file_name = os.path.basename(run['file'])
        file_time = datetime.fromtimestamp(run['start_time'])
        
        print(f"\n运行 {i+1}: {file_name}")
        print(f"  开始时间: {file_time}")
        
        for tag, data in run['data'].items():
            if data['steps']:
                print(f"  {tag}: {len(data['steps'])} 个数据点, "
                      f"步骤 {min(data['steps'])}-{max(data['steps'])}, "
                      f"值范围 {min(data['values']):.4f}-{max(data['values']):.4f}")

def plot_concatenated_comparison():
    """
    主函数：读取所有模型数据并生成拼接后的对比图
    """
    # 模型配置
    models = {
        'default': {
            'dir': 'default---batch_size=1-logs',
            'color': '#808080',
            'marker': 'o',
            'name': 'Default (Baseline)'
        },
        'se_classifier_only': {
            'dir': 'se_classifier_only---batch_size=1-logs',
            'color': '#3498db',
            'marker': 's',
            'name': 'SE Classifier Only'
        },
        'se_attention': {
            'dir': 'se_attention---batch_size=1-logs',
            'color': '#e74c3c',
            'marker': '^',
            'name': 'SE Attention'
        },
        'self_attention': {
            'dir': 'self_attention---batch_size=1-logs',
            'color': '#2ecc71',
            'marker': 'D',
            'name': 'Self Attention'
        },
        'cbam_attention': {
            'dir': 'cbam_attention---batch_size=1-logs',
            'color': '#f39c12',
            'marker': 'v',
            'name': 'CBAM Attention'
        },
        'se_self_attention': {
            'dir': 'se_self_attention---batch_size=1-logs',
            'color': '#9b59b6',
            'marker': '*',
            'name': 'SE + Self Attention'
        }
    }
    
    # 读取并拼接所有模型的数据
    print("="*60)
    print("BEVNet 训练数据拼接分析")
    print("="*60)
    
    model_data = {}
    
    for model_name, config in models.items():
        if os.path.exists(config['dir']):
            print(f"\n处理模型: {model_name}")
            print(f"目录: {config['dir']}")
            
            # 读取所有运行
            all_runs = read_all_runs(config['dir'])
            
            if all_runs:
                print(f"找到 {len(all_runs)} 次训练运行")
                
                # 分析训练连续性
                analyze_training_continuity(all_runs)
                
                # 拼接数据
                concatenated = concatenate_runs(all_runs)
                
                if concatenated:
                    model_data[model_name] = concatenated
                    
                    # 打印拼接结果
                    print("\n拼接后的数据:")
                    for tag, (steps, values) in concatenated.items():
                        print(f"  {tag}: {len(values)} 个数据点")
            else:
                print("  未找到训练数据")
    
    if not model_data:
        print("\n错误：没有找到任何可用数据！")
        return
    
    # 创建对比图
    create_comparison_plots(model_data, models)
    
    # 保存详细统计
    save_performance_summary(model_data, models)

def create_comparison_plots(model_data, models):
    """
    创建各种对比图
    """
    # 1. 综合对比图（Accuracy和IoU并排）
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    
    # Accuracy子图
    for model_name, data in model_data.items():
        if 'Val/acc' in data:
            steps, values = data['Val/acc']
            epochs = range(1, len(values) + 1)
            
            ax1.plot(epochs, values,
                    label=models[model_name]['name'],
                    color=models[model_name]['color'],
                    marker=models[model_name]['marker'],
                    linewidth=2.5,
                    markersize=7,
                    markevery=max(1, len(epochs)//15),
                    alpha=0.9)
    
    ax1.set_xlabel('Epoch', fontsize=13)
    ax1.set_ylabel('Accuracy', fontsize=13)
    ax1.set_title('Validation Accuracy (Concatenated)', fontsize=15, fontweight='bold')
    ax1.legend(loc='lower right', fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(bottom=0)
    
    # IoU子图
    for model_name, data in model_data.items():
        if 'Val/iou' in data:
            steps, values = data['Val/iou']
            epochs = range(1, len(values) + 1)
            
            ax2.plot(epochs, values,
                    label=models[model_name]['name'],
                    color=models[model_name]['color'],
                    marker=models[model_name]['marker'],
                    linewidth=2.5,
                    markersize=7,
                    markevery=max(1, len(epochs)//15),
                    alpha=0.9)
    
    ax2.set_xlabel('Epoch', fontsize=13)
    ax2.set_ylabel('IoU', fontsize=13)
    ax2.set_title('Validation IoU (Concatenated)', fontsize=15, fontweight='bold')
    ax2.legend(loc='lower right', fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(bottom=0)
    
    plt.suptitle('BEVNet Models - Concatenated Training Runs', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('bevnet_concatenated_comparison.png', dpi=300, bbox_inches='tight')
    print("\n已保存: bevnet_concatenated_comparison.png")
    plt.show()
    
    # 2. 单独的大尺寸Accuracy图
    plt.figure(figsize=(14, 9))
    
    for model_name, data in model_data.items():
        if 'Val/acc' in data:
            steps, values = data['Val/acc']
            epochs = range(1, len(values) + 1)
            
            plt.plot(epochs, values,
                    label=models[model_name]['name'],
                    color=models[model_name]['color'],
                    marker=models[model_name]['marker'],
                    linewidth=3,
                    markersize=8,
                    markevery=max(1, len(epochs)//15),
                    alpha=0.9)
    
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Validation Accuracy', fontsize=14)
    plt.title('BEVNet Models - Accuracy Comparison (All Training Runs)', fontsize=16, fontweight='bold')
    plt.legend(loc='lower right', fontsize=12, framealpha=0.95)
    plt.grid(True, alpha=0.3)
    plt.grid(True, which='minor', alpha=0.1, linestyle=':')
    plt.minorticks_on()
    plt.tight_layout()
    plt.savefig('bevnet_concatenated_accuracy.png', dpi=300, bbox_inches='tight')
    print("已保存: bevnet_concatenated_accuracy.png")
    plt.show()
    
    # 3. 单独的大尺寸IoU图
    plt.figure(figsize=(14, 9))
    
    for model_name, data in model_data.items():
        if 'Val/iou' in data:
            steps, values = data['Val/iou']
            epochs = range(1, len(values) + 1)
            
            plt.plot(epochs, values,
                    label=models[model_name]['name'],
                    color=models[model_name]['color'],
                    marker=models[model_name]['marker'],
                    linewidth=3,
                    markersize=8,
                    markevery=max(1, len(epochs)//15),
                    alpha=0.9)
    
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Validation IoU', fontsize=14)
    plt.title('BEVNet Models - IoU Comparison (All Training Runs)', fontsize=16, fontweight='bold')
    plt.legend(loc='lower right', fontsize=12, framealpha=0.95)
    plt.grid(True, alpha=0.3)
    plt.grid(True, which='minor', alpha=0.1, linestyle=':')
    plt.minorticks_on()
    plt.tight_layout()
    plt.savefig('bevnet_concatenated_iou.png', dpi=300, bbox_inches='tight')
    print("已保存: bevnet_concatenated_iou.png")
    plt.show()

def save_performance_summary(model_data, models):
    """
    保存详细的性能统计
    """
    summary = []
    
    for model_name, data in model_data.items():
        row = {
            'Model': models[model_name]['name'],
            'Total Epochs': 0
        }
        
        # Accuracy统计
        if 'Val/acc' in data:
            _, values = data['Val/acc']
            row['Total Epochs'] = len(values)
            row['Final Acc'] = f"{values[-1]:.4f}"
            row['Best Acc'] = f"{max(values):.4f}"
            row['Best Acc Epoch'] = values.index(max(values)) + 1
            row['Avg Last 10'] = f"{np.mean(values[-10:]):.4f}" if len(values) >= 10 else "N/A"
        
        # IoU统计
        if 'Val/iou' in data:
            _, values = data['Val/iou']
            row['Final IoU'] = f"{values[-1]:.4f}"
            row['Best IoU'] = f"{max(values):.4f}"
            row['Best IoU Epoch'] = values.index(max(values)) + 1
        
        summary.append(row)
    
    # 创建DataFrame并保存
    df = pd.DataFrame(summary)
    
    # 按最佳IoU排序
    if 'Best IoU' in df.columns:
        df['Best IoU (numeric)'] = df['Best IoU'].astype(float)
        df = df.sort_values('Best IoU (numeric)', ascending=False)
        df = df.drop('Best IoU (numeric)', axis=1)
    
    # 打印到控制台
    print("\n" + "="*80)
    print("性能统计摘要（拼接后的完整训练）")
    print("="*80)
    print(df.to_string(index=False))
    print("="*80)
    
    # 保存到CSV
    df.to_csv('bevnet_concatenated_summary.csv', index=False)
    print(f"\n性能摘要已保存到: bevnet_concatenated_summary.csv")

if __name__ == "__main__":
    plot_concatenated_comparison()