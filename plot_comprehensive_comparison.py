import pandas as pd
import json
import matplotlib.pyplot as plt
import os

# Set paths
base_dir = r"D:\code\1206"
ddpg_csv_1 = os.path.join(base_dir, "modele_valuation", "run_20251207_132139_wdelay0.6_wenergy0.4.csv")
ddpg_csv_2 = os.path.join(base_dir, "modele_valuation", "run_20251207_131945_wdelay0.5_wenergy0.5.csv")
llm_json = os.path.join(base_dir, "llm_test_20251207_152451.json")
llm_ddpg_json = os.path.join(base_dir, "llm_test_with_ddpg_knowledge_20251207_154256.json")

# Load Data
print("Loading data...")
try:
    df_ddpg_1 = pd.read_csv(ddpg_csv_1)
    df_ddpg_2 = pd.read_csv(ddpg_csv_2)
    
    with open(llm_json, 'r') as f:
        data_llm = json.load(f)
    
    with open(llm_ddpg_json, 'r') as f:
        data_llm_ddpg = json.load(f)
except Exception as e:
    print(f"Error loading files: {e}")
    exit(1)

# Process DDPG Data
# We want to see the best performance achieved during training/evaluation
# We'll use the episode with the highest reward as the representative
def get_best_metrics(df, label):
    best_idx = df['reward'].idxmax()
    best_row = df.iloc[best_idx]
    print(f"{label} Best Episode: {best_row['episode']}, Reward: {best_row['reward']:.4f}")
    return best_row['avg_delay'], best_row['total_energy']

ddpg_1_delay, ddpg_1_energy = get_best_metrics(df_ddpg_1, "DDPG (0.6/0.4)")
ddpg_2_delay, ddpg_2_energy = get_best_metrics(df_ddpg_2, "DDPG (0.5/0.5)")

# LLM Data
llm_delay = data_llm['total_delay']
llm_energy = data_llm['total_energy']

llm_ddpg_delay = data_llm_ddpg['total_delay']
llm_ddpg_energy = data_llm_ddpg['total_energy']

# Visualization
plt.style.use('ggplot')
fig = plt.figure(figsize=(16, 10))
fig.suptitle('MAUTO: LLM vs DDPG Performance Comparison', fontsize=16)

# 1. DDPG Training Reward Curve
ax1 = plt.subplot(2, 2, 1)
ax1.plot(df_ddpg_1['episode'], df_ddpg_1['reward'], 'o-', label='DDPG (Delay 0.6, Energy 0.4)')
ax1.plot(df_ddpg_2['episode'], df_ddpg_2['reward'], 's-', label='DDPG (Delay 0.5, Energy 0.5)')
ax1.set_title('DDPG Training Reward Convergence')
ax1.set_xlabel('Episode')
ax1.set_ylabel('Reward')
ax1.legend()

# 2. DDPG Delay vs Energy Trade-off Curve
ax2 = plt.subplot(2, 2, 2)
sc1 = ax2.scatter(df_ddpg_1['avg_delay'], df_ddpg_1['total_energy'], c=df_ddpg_1['episode'], cmap='Blues', marker='o', label='DDPG (0.6/0.4)')
sc2 = ax2.scatter(df_ddpg_2['avg_delay'], df_ddpg_2['total_energy'], c=df_ddpg_2['episode'], cmap='Reds', marker='s', label='DDPG (0.5/0.5)')
ax2.set_title('Pareto Front Approximation: Delay vs Energy')
ax2.set_xlabel('Total Delay (s)')
ax2.set_ylabel('Total Energy (J)')
plt.colorbar(sc1, ax=ax2, label='Episode (Blue)')
# plt.colorbar(sc2, ax=ax2, label='Episode (Red)') # Dual colorbar is tricky, skip for now
ax2.legend()

# Add LLM points to the scatter plot
ax2.scatter([llm_delay], [llm_energy], color='green', marker='*', s=200, label='Pure LLM')
ax2.scatter([llm_ddpg_delay], [llm_ddpg_energy], color='purple', marker='*', s=200, label='LLM + DDPG')
ax2.text(llm_delay, llm_energy, '  Pure LLM', verticalalignment='bottom')
ax2.text(llm_ddpg_delay, llm_ddpg_energy, '  LLM + DDPG', verticalalignment='bottom')

# 3. Bar Chart: Delay Comparison
ax3 = plt.subplot(2, 2, 3)
methods = ['Pure LLM', 'LLM + DDPG', 'DDPG (0.6/0.4)', 'DDPG (0.5/0.5)']
delays = [llm_delay, llm_ddpg_delay, ddpg_1_delay, ddpg_2_delay]
colors = ['#2ca02c', '#9467bd', '#1f77b4', '#d62728'] # Green, Purple, Blue, Red

bars = ax3.bar(methods, delays, color=colors, alpha=0.8)
ax3.set_title('Total Delay Comparison (Lower is Better)')
ax3.set_ylabel('Time (s)')
ax3.bar_label(bars, fmt='%.1f')
ax3.grid(axis='y')

# 4. Bar Chart: Energy Comparison
ax4 = plt.subplot(2, 2, 4)
energies = [llm_energy, llm_ddpg_energy, ddpg_1_energy, ddpg_2_energy]
bars = ax4.bar(methods, energies, color=colors, alpha=0.8)
ax4.set_title('Total Energy Comparison (Lower is Better)')
ax4.set_ylabel('Energy (J)')
ax4.bar_label(bars, fmt='%.1f')
ax4.grid(axis='y')

plt.tight_layout()
output_file = os.path.join(base_dir, 'experiment_comparison.png')
plt.savefig(output_file, dpi=300)
print(f"Comparison plot saved to: {output_file}")

# Print Summary Table
print("\n=== Performance Summary ===")
summary_df = pd.DataFrame({
    'Method': methods,
    'Total Delay (s)': delays,
    'Total Energy (J)': energies
})
print(summary_df.to_string(index=False))
