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

# 创建图表 - 3行2列的布局 (共 6 个子图)
fig = plt.figure(figsize=(16, 18))
fig.suptitle('DDPG Multi-UAV Training Results (Simplified Metrics)', fontsize=18, fontweight='bold', y=1.01)

# 确保索引从 1 开始
plot_index = 1

# 1. 绘制 Reward
ax1 = plt.subplot(3, 2, plot_index); plot_index += 1
ax1.plot(df['episode'], df['reward'], linewidth=2.5, color='#2E86AB', marker='o', markersize=3, alpha=0.8)
ax1.set_xlabel('Episode', fontsize=12)
ax1.set_ylabel('Reward', fontsize=12)
ax1.set_title('Reward', fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.4)

# 2. 绘制 Average Delay Sum (avg_delay_sum)
ax2 = plt.subplot(3, 2, plot_index); plot_index += 1
ax2.plot(df['episode'], df['avg_delay_sum'], linewidth=2.5, color='#EF476F', marker='s', markersize=3, alpha=0.8)
ax2.set_xlabel('Episode', fontsize=12)
ax2.set_ylabel('Avg Delay Sum', fontsize=12)
ax2.set_title('Average Delay Per Step', fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.4)

# 3. 绘制 Max Delay Sum (max_delay_sum)
ax3 = plt.subplot(3, 2, plot_index); plot_index += 1
ax3.plot(df['episode'], df['max_delay_sum'], linewidth=2.5, color='#FFC43A', marker='^', markersize=3, alpha=0.8)
ax3.set_xlabel('Episode', fontsize=12)
ax3.set_ylabel('Max Delay Sum', fontsize=12)
ax3.set_title('Max Delay Across Users', fontsize=14, fontweight='bold')
ax3.grid(True, alpha=0.4)

# 4. 绘制 Task Energy (task_energy)
ax4 = plt.subplot(3, 2, plot_index); plot_index += 1
ax4.plot(df['episode'], df['task_energy'], linewidth=2.5, color='#06D6A0', marker='d', markersize=3, alpha=0.8)
ax4.set_xlabel('Episode', fontsize=12)
ax4.set_ylabel('Task Energy (J)', fontsize=12)
ax4.set_title('Total Task Energy', fontsize=14, fontweight='bold')
ax4.grid(True, alpha=0.4)

# 5. 绘制 Movement Energy (move_energy)
ax5 = plt.subplot(3, 2, plot_index); plot_index += 1
ax5.plot(df['episode'], df['move_energy'], linewidth=2.5, color='#A23B72', marker='*', markersize=4, alpha=0.8)
ax5.set_xlabel('Episode', fontsize=12)
ax5.set_ylabel('Movement Energy (J)', fontsize=12)
ax5.set_title('Total Movement Energy', fontsize=14, fontweight='bold')
ax5.grid(True, alpha=0.4)

# --- 6. 绘制各用户（User）的平均延迟对比 ---
ax6 = plt.subplot(3, 2, plot_index); plot_index += 1

# 自动检测有多少个 User 延迟列
user_columns = [col for col in df.columns if col.startswith('user_') and col.endswith('_delay')]
num_users = len(user_columns)

# 使用一套区分度高的颜色
colors = ['#1B9AAA', '#EF476F', '#06D6A0', '#FFD166', '#118AB2', '#073B4C', '#A23B72']

print(f"Found {num_users} user delay columns for plotting.")

if num_users > 0:
    for i, user_col in enumerate(user_columns):
        user_id = user_col.split('_')[1]
        ax6.plot(df['episode'], df[user_col], 
                 linewidth=2, 
                 color=colors[i % len(colors)], 
                 alpha=0.8,
                 label=f'User {user_id} Avg Delay')

    ax6.set_ylabel('Average Actual Delay', fontsize=12)
    ax6.set_xlabel('Episode', fontsize=12)
    ax6.set_title('User Average Actual Delay Comparison', fontsize=14, fontweight='bold')
    ax6.legend(loc='best', fontsize=10)
    ax6.grid(True, alpha=0.4)
else:
    ax6.set_title('User Delay Comparison (No Data Found)', fontsize=14, fontweight='bold')
    print("Warning: No 'user_N_delay' columns found in the CSV.")


# 调整布局
plt.tight_layout(rect=[0, 0, 1, 0.98]) # 调整以容纳 suptitle

# 保存图片到pic文件夹
current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
output_file = f'pic/training_simplified_results_{current_time}.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"\nFigure saved to: {output_file}")

# 不显示图表
plt.close()

# --- 打印统计信息 ---
print("\n=== Training Statistics ===")
print(f"Total Episodes: {len(df)}")
print(f"\nReward - Min: {df['reward'].min():.4f}, Max: {df['reward'].max():.4f}, Mean: {df['reward'].mean():.4f}")
print(f"Avg Delay Sum - Min: {df['avg_delay_sum'].min():.6f}, Max: {df['avg_delay_sum'].max():.6f}, Mean: {df['avg_delay_sum'].mean():.6f}")
print(f"Max Delay Sum - Min: {df['max_delay_sum'].min():.6f}, Max: {df['max_delay_sum'].max():.6f}, Mean: {df['max_delay_sum'].mean():.6f}")
print(f"Task Energy - Min: {df['task_energy'].min():.4f}, Max: {df['task_energy'].max():.4f}, Mean: {df['task_energy'].mean():.4f}")
print(f"Move Energy - Min: {df['move_energy'].min():.4f}, Max: {df['move_energy'].max():.4f}, Mean: {df['move_energy'].mean():.4f}")

print("\nUser Average Delay Statistics:")
for user_col in user_columns:
    user_id = user_col.split('_')[1]
    print(f"User {user_id} - Min: {df[user_col].min():.6f}, Max: {df[user_col].max():.6f}, Mean: {df[user_col].mean():.6f}")