import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime

# 创建pic文件夹（如果不存在）
if not os.path.exists('pic'):
    os.makedirs('pic')
    print("Created 'pic' folder")

# 读取CSV文件
csv_file = 'training_history.csv'
df = pd.read_csv(csv_file)

print(f"Successfully loaded {len(df)} episodes")
print(f"Columns: {df.columns.tolist()}")

# 定义参考值
reference_values = {
    'reward': -20.874,
    'task_energy': 35.212,
    'transmission_energy': 4.218,
    'movement_energy': 75424.882,
    'uav_delays': [45.65089790818104, 33.088746927539475]
}

# 创建图表 - 3行2列的布局
fig = plt.figure(figsize=(15, 14))
fig.suptitle('DDPG Multi-UAV Training Results', fontsize=16, fontweight='bold')

# 1. 绘制 Reward
ax1 = plt.subplot(3, 2, 1)
ax1.plot(df['episode'], df['reward'], linewidth=2, color='#2E86AB', marker='o', markersize=3)
ax1.axhline(y=reference_values['reward'], color='red', linestyle='--', linewidth=2, 
            label=f"Reference: {reference_values['reward']:.3f}")
ax1.set_xlabel('Episode', fontsize=12)
ax1.set_ylabel('Reward', fontsize=12)
ax1.set_title('Reward', fontsize=13, fontweight='bold')
ax1.legend(loc='best', fontsize=9)
ax1.grid(True, alpha=0.3)

# 2. 绘制 Task Energy
ax2 = plt.subplot(3, 2, 2)
ax2.plot(df['episode'], df['task_energy'], linewidth=2, color='#A23B72', marker='s', markersize=3)
ax2.axhline(y=reference_values['task_energy'], color='red', linestyle='--', linewidth=2,
            label=f"Reference: {reference_values['task_energy']:.3f}")
ax2.set_xlabel('Episode', fontsize=12)
ax2.set_ylabel('Task Energy', fontsize=12)
ax2.set_title('Task Energy', fontsize=13, fontweight='bold')
ax2.legend(loc='best', fontsize=9)
ax2.grid(True, alpha=0.3)

# 3. 绘制 Transmission Energy
ax3 = plt.subplot(3, 2, 3)
ax3.plot(df['episode'], df['transmission_energy'], linewidth=2, color='#F18F01', marker='^', markersize=3)
ax3.axhline(y=reference_values['transmission_energy'], color='red', linestyle='--', linewidth=2,
            label=f"Reference: {reference_values['transmission_energy']:.3f}")
ax3.set_xlabel('Episode', fontsize=12)
ax3.set_ylabel('Transmission Energy', fontsize=12)
ax3.set_title('Transmission Energy', fontsize=13, fontweight='bold')
ax3.legend(loc='best', fontsize=9)
ax3.grid(True, alpha=0.3)

# 4. 绘制 Movement Energy
ax4 = plt.subplot(3, 2, 4)
ax4.plot(df['episode'], df['movement_energy'], linewidth=2, color='#06D6A0', marker='d', markersize=3)
ax4.axhline(y=reference_values['movement_energy'], color='red', linestyle='--', linewidth=2,
            label=f"Reference: {reference_values['movement_energy']:.3f}")
ax4.set_xlabel('Episode', fontsize=12)
ax4.set_ylabel('Movement Energy', fontsize=12)
ax4.set_title('Movement Energy', fontsize=13, fontweight='bold')
ax4.legend(loc='best', fontsize=9)
ax4.grid(True, alpha=0.3)

# 5. 绘制各UAV的时延对比（占据底部两个子图的位置）
ax5 = plt.subplot(3, 1, 3)

# 自动检测有多少个UAV
uav_columns = [col for col in df.columns if col.startswith('uav_') and col.endswith('_delay')]
num_uavs = len(uav_columns)

# 使用不同颜色绘制每个UAV的时延
colors = ['#1B9AAA', '#EF476F', '#06D6A0', '#FFD166', '#118AB2', '#073B4C']

for i, uav_col in enumerate(uav_columns):
    uav_id = uav_col.split('_')[1]
    ax5.plot(df['episode'], df[uav_col], 
             linewidth=2.5, 
             color=colors[i % len(colors)], 
             marker='o', 
             markersize=4,
             label=f'UAV {uav_id}',
             alpha=0.8)

# 添加UAV延迟的参考线
for i, delay_value in enumerate(reference_values['uav_delays']):
    if i < len(uav_columns):
        ax5.axhline(y=delay_value, color='red', linestyle='--', linewidth=2, alpha=0.7,
                   label=f"UAV {i} Ref: {delay_value:.3f}")

ax5.set_xlabel('Episode', fontsize=12)
ax5.set_ylabel('Delay', fontsize=12)
ax5.set_title('UAV Delay Comparison', fontsize=13, fontweight='bold')
ax5.legend(loc='best', fontsize=10, ncol=2)
ax5.grid(True, alpha=0.3)

# 调整布局
plt.tight_layout()

# 保存图片到pic文件夹
current_time = datetime.now().strftime('%Y%m%d_%H%M')
output_file = f'pic/training_results_{current_time}.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"\nFigure saved to: {output_file}")

# 不显示图表
plt.close()

# 打印统计信息
print("\n=== Training Statistics ===")
print(f"Total Episodes: {len(df)}")
print(f"\nReward - Min: {df['reward'].min():.4f}, Max: {df['reward'].max():.4f}, Mean: {df['reward'].mean():.4f}")
print(f"Task Energy - Min: {df['task_energy'].min():.4f}, Max: {df['task_energy'].max():.4f}, Mean: {df['task_energy'].mean():.4f}")
print(f"Transmission Energy - Min: {df['transmission_energy'].min():.4f}, Max: {df['transmission_energy'].max():.4f}, Mean: {df['transmission_energy'].mean():.4f}")
print(f"Movement Energy - Min: {df['movement_energy'].min():.4f}, Max: {df['movement_energy'].max():.4f}, Mean: {df['movement_energy'].mean():.4f}")

print("\nUAV Delay Statistics:")
for uav_col in uav_columns:
    uav_id = uav_col.split('_')[1]
    print(f"UAV {uav_id} - Min: {df[uav_col].min():.6f}, Max: {df[uav_col].max():.6f}, Mean: {df[uav_col].mean():.6f}")

print("\n=== Reference Values ===")
print(f"Reward: {reference_values['reward']}")
print(f"Task Energy: {reference_values['task_energy']}")
print(f"Transmission Energy: {reference_values['transmission_energy']}")
print(f"Movement Energy: {reference_values['movement_energy']}")
print(f"UAV Delays: {reference_values['uav_delays']}")