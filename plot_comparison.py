import pandas as pd
import json
import matplotlib.pyplot as plt
import os
import glob
import numpy as np

# 设置字体以支持中文显示 (尝试常见的中文字体)
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

def load_json_metrics(file_path, label):
    """从JSON文件加载LLM测试指标"""
    if not os.path.exists(file_path):
        print(f"警告: 文件未找到 {file_path}")
        return None
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        total_steps = data.get('total_steps', 1)
        if total_steps == 0: total_steps = 1
        
        # 计算平均时延
        total_delay = data.get('total_delay', 0)
        avg_delay = total_delay
        
        total_energy = data.get('total_energy', 0)
        
        return {
            'label': label,
            'avg_delay': avg_delay,
            'total_delay': total_delay,
            'total_energy': total_energy
        }
    except Exception as e:
        print(f"读取JSON文件出错 {file_path}: {e}")
        return None

def plot_for_csv(csv_file, llm_metrics_list, output_dir):
    """读取CSV并绘制对比图"""
    try:
        df = pd.read_csv(csv_file)
    except Exception as e:
        print(f"读取CSV文件出错 {csv_file}: {e}")
        return

    # 准备DDPG数据
    # 假设CSV中有 'episode', 'avg_delay', 'total_energy' 列
    labels = [f"DDPG-Ep{ep}" for ep in df['episode']]
    
    # 处理CSV数据中的单位不一致问题 (Heuristic: < 20 为平均时延, > 20 为总时延)
    raw_delays = df['avg_delay'].tolist()
    steps_list = df['steps'].tolist() if 'steps' in df.columns else [15] * len(df)
    total_delays = []
    
    for i, val in enumerate(raw_delays):
        if val < 20: # 认为是平均时延
            total_delays.append(val * steps_list[i])
        else: # 认为是总时延 (虽然列名是avg_delay)
            total_delays.append(val)
            
    total_energies = df['total_energy'].tolist()
    
    # 颜色列表
    colors = ['skyblue'] * len(df)
    
    # 添加LLM数据
    for llm in llm_metrics_list:
        if llm:
            labels.append(llm['label'])
            total_delays.append(llm['total_delay'])
            total_energies.append(llm['total_energy'])
            colors.append('salmon' if 'Knowledge' not in llm['label'] else 'lightgreen')
            
    # 创建输出目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    base_name = os.path.splitext(os.path.basename(csv_file))[0]
    
    # --- 绘制总时延对比图 ---
    plt.figure(figsize=(14, 7))
    bars = plt.bar(labels, total_delays, color=colors)
    plt.xlabel('模型 (Model)')
    plt.ylabel('总时延 (Total Delay) [s]')
    plt.title(f'总时延对比 - {base_name}')
    plt.xticks(rotation=45, ha='right')
    
    # 添加数值标签
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'{height:.1f}',
                 ha='center', va='bottom')
    
    plt.tight_layout()
    delay_plot_path = os.path.join(output_dir, f'{base_name}_delay_comparison.png')
    plt.savefig(delay_plot_path)
    plt.close()
    
    # --- 绘制总能耗对比图 ---
    plt.figure(figsize=(14, 7))
    bars = plt.bar(labels, total_energies, color=colors)
    plt.xlabel('模型 (Model)')
    plt.ylabel('总能耗 (Total Energy) [J]')
    plt.title(f'总能耗对比 - {base_name}')
    plt.xticks(rotation=45, ha='right')
    
    # 添加数值标签
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'{height:.1f}',
                 ha='center', va='bottom')
                 
    plt.tight_layout()
    energy_plot_path = os.path.join(output_dir, f'{base_name}_energy_comparison.png')
    plt.savefig(energy_plot_path)
    plt.close()
    
    print(f"已生成图表: \n  - {delay_plot_path}\n  - {energy_plot_path}")

def main():
    # 定义路径
    csv_dir = r"D:\code\1206\modele_valuation"
    llm_file1 = r"D:\code\1206\llm_test_20251207_152451.json"
    llm_file2 = r"D:\code\1206\llm_test_with_ddpg_knowledge_20251207_154256.json"
    
    print("开始处理数据并绘图...")
    
    # 加载LLM指标
    llm_metrics = []
    # Gemini Pure
    llm_metrics.append(load_json_metrics(llm_file1, "LLM (Gemini)"))
    # Gemini + DDPG Knowledge
    llm_metrics.append(load_json_metrics(llm_file2, "LLM + Knowledge"))
    
    # 查找所有CSV文件
    csv_files = glob.glob(os.path.join(csv_dir, "*.csv"))
    
    if not csv_files:
        print(f"在 {csv_dir} 未找到CSV文件")
        return

    print(f"找到 {len(csv_files)} 个CSV文件。")
    
    # 对每个CSV文件生成图表
    for csv_file in csv_files:
        plot_for_csv(csv_file, llm_metrics, csv_dir)
        
    print("\n所有图表绘制完成！")

if __name__ == "__main__":
    main()
