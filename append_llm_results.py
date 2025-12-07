# append_llm_results.py
"""
读取LLM测试结果JSON文件，汇总数据并追加到eval_results.csv
"""
import json
import csv
from pathlib import Path


def read_and_summarize_json(json_file):
    """
    读取JSON文件并汇总数据
    
    Args:
        json_file: JSON文件路径
    
    Returns:
        dict: 汇总结果
    """
    with open(json_file, 'r', encoding='utf-8') as f:
        steps = json.load(f)
    
    # 累加所有步骤的数据
    total_reward = 0.0
    total_avg_delay = 0.0
    total_max_delay = 0.0
    total_task_energy = 0.0
    total_move_energy = 0.0
    num_steps = len(steps)
    
    for step in steps:
        total_reward += step['reward']
        total_avg_delay += step['avg_delay']
        total_max_delay += step['max_delay']
        total_task_energy += step['task_energy']
        total_move_energy += step['move_energy']
    
    return {
        'model': 'LLM',
        'episode': 0,  # LLM没有episode概念，设为0
        'reward': round(total_reward, 4),
        'avg_delay_sum': round(total_avg_delay, 6),
        'max_delay_sum': round(total_max_delay, 6),
        'task_energy': round(total_task_energy, 2),
        'move_energy': round(total_move_energy, 2),
        'steps': num_steps
    }


def append_to_csv(csv_file, data):
    """
    追加数据到CSV文件
    
    Args:
        csv_file: CSV文件路径
        data: 要追加的数据字典
    """
    # 检查CSV是否存在
    file_exists = Path(csv_file).exists()
    
    fieldnames = ['model', 'episode', 'reward', 'avg_delay_sum', 'max_delay_sum', 
                  'task_energy', 'move_energy', 'steps']
    
    with open(csv_file, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        
        # 如果文件不存在，写入表头
        if not file_exists:
            writer.writeheader()
        
        # 追加数据
        writer.writerow(data)


def main(json_file, csv_file):
    """
    主函数
    
    Args:
        json_file: JSON输入文件路径
        csv_file: CSV输出文件路径
    """
    print("="*70)
    print("LLM结果追加到CSV")
    print("="*70)
    
    # 读取并汇总JSON
    print(f"\n[1/2] 读取JSON文件: {json_file}")
    summary = read_and_summarize_json(json_file)
    
    print(f"✓ 汇总完成:")
    print(f"  模型: {summary['model']}")
    print(f"  总奖励: {summary['reward']}")
    print(f"  平均时延累计: {summary['avg_delay_sum']:.6f}s")
    print(f"  最大时延累计: {summary['max_delay_sum']:.6f}s")
    print(f"  任务能耗累计: {summary['task_energy']:.2f}J")
    print(f"  移动能耗累计: {summary['move_energy']:.2f}J")
    print(f"  步数: {summary['steps']}")
    
    # 追加到CSV
    print(f"\n[2/2] 追加到CSV文件: {csv_file}")
    append_to_csv(csv_file, summary)
    print("✓ 追加完成")
    
    print("\n" + "="*70)
    print("完成!")
    print("="*70)


if __name__ == "__main__":
    # 配置文件路径
    json_file = "result.json"  # 你的JSON文件路径
    csv_file = "eval_results.csv"
    
    main(json_file, csv_file)