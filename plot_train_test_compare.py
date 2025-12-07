import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime
import re

# --- 创建输出文件夹 ---
if not os.path.exists('train_test_compare'):
    os.makedirs('train_test_compare')
    print("Created 'train_test_compare' folder")

# --- 读取训练历史 ---
csv_file = 'training_history.csv'
try:
    df = pd.read_csv(csv_file)
except FileNotFoundError:
    print(f"Error: CSV file '{csv_file}' not found.")
    raise SystemExit(1)

print(f"Successfully loaded {len(df)} episodes")
print(f"Columns: {df.columns.tolist()}")

# -----------------------------------------------------
# 读取评估结果（每个模型一条基线）
# -----------------------------------------------------
eval_csv = "eval_results.csv"
df_eval = None
model_color = {}
if os.path.exists(eval_csv):
    df_eval = pd.read_csv(eval_csv)
    if 'model' not in df_eval.columns:
        print("Warning: eval_results.csv lacks 'model' column. Baselines disabled.")
        df_eval = None
    else:
        # 统一提取模型名的简短“标签”（例如 episode_100）
        def short_label(name):
            m = re.search(r'(episode_\d+)', str(name))
            return m.group(1) if m else os.path.splitext(os.path.basename(str(name)))[0]

        df_eval['model_label'] = df_eval['model'].apply(short_label)

        # 为每个模型分配固定颜色（跨子图一致）
        unique_labels = df_eval['model_label'].unique().tolist()
        cmap = plt.cm.get_cmap('tab20', max(20, len(unique_labels)))  # 至少 20 色
        for i, lab in enumerate(unique_labels):
            model_color[lab] = cmap(i / max(1, len(unique_labels)-1))

        print(f"Loaded {len(df_eval)} evaluated models for baselines: {', '.join(unique_labels[:10])}{' ...' if len(unique_labels)>10 else ''}")
else:
    print("No eval_results.csv found -> skip baselines.")

def add_model_baselines(ax, metric_key, ylabel=None):
    """
    在子图 ax 上为每个模型画一条横向基线：
    y = df_eval[metric_key], x 跨越训练 episode 的范围
    """
    if df_eval is None or metric_key not in df_eval.columns:
        return

    x_min = df['episode'].min()
    x_max = df['episode'].max()

    # 画每条基线
    for _, row in df_eval.iterrows():
        lab = row['model_label']
        yv = float(row[metric_key])
        ax.hlines(y=yv, xmin=x_min, xmax=x_max,
                  colors=[model_color[lab]], linestyles='--', linewidth=1.5, alpha=0.9,
                  label=lab)

    # 合并图例（避免重复）
    handles, labels = ax.get_legend_handles_labels()
    # 按 label 去重并保持顺序
    seen = set()
    uniq = [(h, l) for h, l in zip(handles, labels) if (l not in seen and not seen.add(l))]
    # 如果模型太多，图例会挤；把图例放到外侧
    ax.legend(*zip(*uniq), loc='upper left', bbox_to_anchor=(1.01, 1.0), borderaxespad=0., fontsize=9, title="Baselines")

    if ylabel:
        ax.set_ylabel(ylabel)

# --- 绘图 ---
fig = plt.figure(figsize=(18, 18))
fig.suptitle('DDPG Multi-UAV Training Results (with Per-Model Baselines)', fontsize=18, fontweight='bold', y=1.02)

plot_index = 1

# 1. Reward
ax1 = plt.subplot(3, 2, plot_index); plot_index += 1
ax1.plot(df['episode'], df['reward'], linewidth=2.5, color='#2E86AB', marker='o', markersize=3, alpha=0.8, label='Training')
ax1.set_xlabel('Episode', fontsize=12)
ax1.set_ylabel('Reward', fontsize=12)
ax1.set_title('Reward', fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.4)
add_model_baselines(ax1, 'reward')  # <- 每个模型一条横线基线

# 2. Avg Delay Sum
ax2 = plt.subplot(3, 2, plot_index); plot_index += 1
ax2.plot(df['episode'], df['avg_delay_sum'], linewidth=2.5, color='#EF476F', marker='s', markersize=3, alpha=0.8, label='Training')
ax2.set_xlabel('Episode', fontsize=12)
ax2.set_ylabel('Avg Delay Sum', fontsize=12)
ax2.set_title('Average Delay Per Step', fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.4)
add_model_baselines(ax2, 'avg_delay_sum')

# 3. Max Delay Sum
ax3 = plt.subplot(3, 2, plot_index); plot_index += 1
ax3.plot(df['episode'], df['max_delay_sum'], linewidth=2.5, color='#FFC43A', marker='^', markersize=3, alpha=0.8, label='Training')
ax3.set_xlabel('Episode', fontsize=12)
ax3.set_ylabel('Max Delay Sum', fontsize=12)
ax3.set_title('Max Delay Across Users', fontsize=14, fontweight='bold')
ax3.grid(True, alpha=0.4)
add_model_baselines(ax3, 'max_delay_sum')

# 4. Task Energy
ax4 = plt.subplot(3, 2, plot_index); plot_index += 1
ax4.plot(df['episode'], df['task_energy'], linewidth=2.5, color='#06D6A0', marker='d', markersize=3, alpha=0.8, label='Training')
ax4.set_xlabel('Episode', fontsize=12)
ax4.set_ylabel('Task Energy (J)', fontsize=12)
ax4.set_title('Total Task Energy', fontsize=14, fontweight='bold')
ax4.grid(True, alpha=0.4)
add_model_baselines(ax4, 'task_energy')

# 5. Movement Energy
ax5 = plt.subplot(3, 2, plot_index); plot_index += 1
ax5.plot(df['episode'], df['move_energy'], linewidth=2.5, color='#A23B72', marker='*', markersize=4, alpha=0.8, label='Training')
ax5.set_xlabel('Episode', fontsize=12)
ax5.set_ylabel('Movement Energy (J)', fontsize=12)
ax5.set_title('Total Movement Energy', fontsize=14, fontweight='bold')
ax5.grid(True, alpha=0.4)
add_model_baselines(ax5, 'move_energy')

# 6. 用户延迟对比（此图不添加基线，因为 eval 不含 per-user）
ax6 = plt.subplot(3, 2, plot_index); plot_index += 1
user_columns = [col for col in df.columns if col.startswith('user_') and col.endswith('_delay')]
colors = ['#1B9AAA', '#EF476F', '#06D6A0', '#FFD166', '#118AB2', '#073B4C', '#A23B72']
print(f"Found {len(user_columns)} user delay columns for plotting.")
if user_columns:
    for i, user_col in enumerate(user_columns):
        user_id = user_col.split('_')[1]
        ax6.plot(df['episode'], df[user_col],
                 linewidth=2, color=colors[i % len(colors)], alpha=0.8,
                 label=f'User {user_id} Avg Delay')
    ax6.set_ylabel('Average Actual Delay', fontsize=12)
    ax6.set_xlabel('Episode', fontsize=12)
    ax6.set_title('User Average Actual Delay Comparison', fontsize=14, fontweight='bold')
    ax6.legend(loc='best', fontsize=10)
    ax6.grid(True, alpha=0.4)
else:
    ax6.set_title('User Delay Comparison (No Data Found)', fontsize=14, fontweight='bold')
    print("Warning: No 'user_N_delay' columns found in the CSV.")

# 调整布局并保存
plt.tight_layout(rect=[0, 0, 0.85, 0.98])  # 预留右侧图例空间
current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
output_file = f'train_test_compare/training_simplified_results_{current_time}.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight')
plt.close()
print(f"\nFigure saved to: {output_file}")

# 打印简单统计
print("\n=== Training Statistics ===")
print(f"Total Episodes: {len(df)}")
for k in ['reward', 'avg_delay_sum', 'max_delay_sum', 'task_energy', 'move_energy']:
    print(f"{k} - Min: {df[k].min():.6f}, Max: {df[k].max():.6f}, Mean: {df[k].mean():.6f}")
