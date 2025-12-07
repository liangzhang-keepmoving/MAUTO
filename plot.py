import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime

# --- 配置 ---
# 创建pic文件夹（如果不存在）
if not os.path.exists('pic'):
    os.makedirs('pic')
    print("Created 'pic' folder")

# 读取CSV文件
csv_file = 'training_history.csv'
try:
    df = pd.read_csv(csv_file)
except FileNotFoundError:
    print(f"Error: CSV file '{csv_file}' not found.")
    exit()

print(f"Successfully loaded {len(df)} episodes")
print(f"Columns: {df.columns.tolist()}")

# --- 绘图逻辑 ---

# 创建图表 - 3行1列的布局 (共 3 个子图)
fig = plt.figure(figsize=(10, 15))
fig.suptitle('DDPG Multi-UAV Training Results', fontsize=18, fontweight='bold', y=0.98)

# 1. 绘制 Reward
ax1 = plt.subplot(3, 1, 1)
ax1.plot(df['episode'], df['reward'], linewidth=2.5, color='#2E86AB', marker='o', markersize=3, alpha=0.8)
ax1.set_xlabel('Episode', fontsize=12)
ax1.set_ylabel('Reward', fontsize=12)
ax1.set_title('Reward', fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.4)

# 2. 绘制 Avg Delay
ax2 = plt.subplot(3, 1, 2)
ax2.plot(df['episode'], df['avg_delay'], linewidth=2.5, color='#EF476F', marker='s', markersize=3, alpha=0.8)
ax2.set_xlabel('Episode', fontsize=12)
ax2.set_ylabel('Avg Delay (s)', fontsize=12)
ax2.set_title('Average Delay', fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.4)

# 3. 绘制 Total Energy
ax3 = plt.subplot(3, 1, 3)
ax3.plot(df['episode'], df['total_energy'], linewidth=2.5, color='#06D6A0', marker='d', markersize=3, alpha=0.8)
ax3.set_xlabel('Episode', fontsize=12)
ax3.set_ylabel('Total Energy (J)', fontsize=12)
ax3.set_title('Total Energy', fontsize=14, fontweight='bold')
ax3.grid(True, alpha=0.4)

# 调整布局
plt.tight_layout(rect=[0, 0, 1, 0.96]) # 调整以容纳 suptitle

# 保存图片到pic文件夹
current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
output_file = f'pic/training_results_{current_time}.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"\nFigure saved to: {output_file}")

# 不显示图表
plt.close()

# --- 打印统计信息 ---
print("\n=== Training Statistics ===")
print(f"Total Episodes: {len(df)}")
print(f"\nReward - Min: {df['reward'].min():.4f}, Max: {df['reward'].max():.4f}, Mean: {df['reward'].mean():.4f}")
if 'avg_delay' in df.columns:
    print(f"Avg Delay - Min: {df['avg_delay'].min():.6f}, Max: {df['avg_delay'].max():.6f}, Mean: {df['avg_delay'].mean():.6f}")
if 'total_energy' in df.columns:
    print(f"Total Energy - Min: {df['total_energy'].min():.4f}, Max: {df['total_energy'].max():.4f}, Mean: {df['total_energy'].mean():.4f}")
