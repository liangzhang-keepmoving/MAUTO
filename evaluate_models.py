# evaluate_ddpg.py
"""
评估DDPG训练保存的模型性能
"""
import os
import re
import csv
import json
import torch
import numpy as np
from datetime import datetime
from pathlib import Path

from DDPG import DDPGAgent
from env_test import TestMultiUAVEnvironment


def convert_to_serializable(obj):
    """将numpy/torch类型转换为Python原生类型"""
    if isinstance(obj, (np.integer, torch.LongTensor)):
        return int(obj)
    elif isinstance(obj, (np.floating, torch.FloatTensor)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, torch.Tensor):
        return obj.cpu().numpy().tolist()
    elif isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    else:
        return obj


def convert_soft_to_hard_allocation(allocation_soft, num_uavs, num_users):
    """
    将软分配 [N, M] 转为硬分配 [N, M]（每个用户只选一个无人机）
    与 train_ddpg.py 中的逻辑一致
    """
    allocation_hard = torch.zeros_like(allocation_soft)
    for user_id in range(num_users):
        user_probs = allocation_soft[:, user_id]
        best_uav = torch.argmax(user_probs)
        allocation_hard[best_uav, user_id] = 1.0
    return allocation_hard


def evaluate_one_episode(env, agent, max_steps=15, save_details=False):
    """
    评估一个episode
    
    Args:
        env: 环境实例
        agent: DDPG代理
        max_steps: 最大步数
        save_details: 是否保存详细步骤信息
    
    Returns:
        dict: 评估结果
    """
    state = env.reset()
    
    episode_reward = 0.0
    episode_steps = 0
    episode_avg_delay = 0.0
    episode_total_energy = 0.0
    
    steps_detail = [] if save_details else None
    
    # 评估时不加噪声
    agent.actor.eval()
    agent.critic.eval()
    
    for step in range(max_steps):
        # 生成软分配，转为硬分配执行
        allocation_soft, offloading, motion = agent.select_action(
            state, add_noise=False
        )
        allocation = convert_soft_to_hard_allocation(
            allocation_soft, env.num_uavs, env.num_users
        )
        
        # 组装环境动作
        actions = {}
        # 保持与 train_ddpg.py 一致的解析逻辑
        offloading_np = offloading.cpu().numpy() # [N_Users]
        
        for uav_id in range(env.num_uavs):
            # 获取动作分量
            vx = motion[uav_id, 0].item() # [-1, 1]
            vy = motion[uav_id, 1].item() # [-1, 1]
            
            # 转换为笛卡尔坐标系的移动向量 (dx, dy)
            # max_distance 在这里作为单步最大移动距离 (20m)
            dx = vx * 70
            dy = vy * 70
            print(f"UAV {uav_id} 移动向量: ({dx:.2f}, {dy:.2f})")
            
            actions[f'uav_{uav_id}'] = {
                'user_competition_probs': allocation[uav_id].cpu().numpy(),
                'offloading_ratios': offloading_np, # 传递完整向量，由calculator处理
                'move_vector': (dx, dy)
            }
        
        next_state, reward, done, info = env.step(actions)

        
        # 累计指标
        comps = info['reward_components']
        episode_reward += float(reward)
        episode_steps += 1
        episode_avg_delay += float(comps['avg_delay'])
        episode_total_energy += float(comps['total_energy'])
        
        # 保存步骤详情 (只保留奖励、时延、能耗)
        if save_details:
            step_info = {
                'step': step,
                'reward': float(reward),
                'avg_delay': float(comps['avg_delay']),
                'total_energy': float(comps['total_energy'])
            }
            steps_detail.append(step_info)
        
        state = next_state
        
        if done:
            break
    
    
    result = {
        'reward': round(episode_reward, 4),
        'avg_delay': round(episode_avg_delay, 6) if episode_steps > 0 else 0, # 平均时延
        'total_energy': round(episode_total_energy, 2),
        'steps': episode_steps
    }
    
    if save_details:
        result['steps_detail'] = steps_detail
    
    return result


def evaluate_models(
    base_models_dir="saved_models",
    output_json_dir="eval_details",
    trajectory_file="user_trajectories.json",
    max_distance=30,
    max_steps=40,
    save_details=True
):
    """
    遍历 saved_models 下的每个子目录，分别为每个 run 生成一个独立的 CSV 评估报告
    """
    print("="*70)
    print("DDPG模型批量评估")
    print("="*70)
    
    # 1. 初始化环境 (只需初始化一次)
    print("\n[1/3] 初始化环境...")
    env = TestMultiUAVEnvironment(
        num_uavs=2,
        num_users=5,
        trajectory_file=trajectory_file,
    )
    print("✓ 环境创建成功")
    
    # 2. 初始化Agent
    print("\n[2/3] 初始化DDPG代理...")
    agent = DDPGAgent(
        n_uavs=env.num_uavs,
        n_users=env.num_users,
        lr_actor=1e-4,
        lr_critic=1e-3,
        gamma=0.99,
        tau=0.005,
        max_distance=max_distance,
    )
    print("✓ DDPG代理创建成功")

    # 3. 遍历子目录
    print("\n[3/3] 开始遍历子目录...")
    
    if not os.path.isdir(base_models_dir):
         # 尝试使用当前目录下的saved_models
         print(f"警告：未找到指定目录 {base_models_dir}")
         base_models_dir = os.path.join(os.getcwd(), "saved_models")
         print(f"尝试搜索: {base_models_dir}")
    
    if not os.path.isdir(base_models_dir):
        raise FileNotFoundError(f"找不到模型根目录：{base_models_dir}")

    # 获取所有一级子目录 (即各个 run_xxx 文件夹)
    subdirs = [
        d for d in os.listdir(base_models_dir) 
        if os.path.isdir(os.path.join(base_models_dir, d)) and d.startswith("run_")
    ]
    
    if not subdirs:
        print(f"警告：在 {base_models_dir} 下未找到以 'run_' 开头的子目录")
        return

    # 创建JSON详情根目录
    if save_details:
        Path(output_json_dir).mkdir(exist_ok=True)

    # 创建CSV输出目录
    output_csv_dir = "modele_valuation"
    Path(output_csv_dir).mkdir(exist_ok=True)

    # 逐个子目录处理
    for subdir_name in subdirs:
        subdir_path = os.path.join(base_models_dir, subdir_name)
        csv_filename = os.path.join(output_csv_dir, f"{subdir_name}.csv")
        print(f"\n>> 正在处理: {subdir_name} -> {csv_filename}")
        
        # 查找该子目录下的所有模型文件
        model_files = [
            f for f in os.listdir(subdir_path)
            if f.endswith("_model.pth") and f.startswith("episode_")
        ]
        
        if not model_files:
            print(f"   [跳过] 该目录下无模型文件")
            continue
            
        # 按episode排序
        def parse_episode(filename):
            m = re.search(r"episode_(\d+)_model\.pth", filename)
            return int(m.group(1)) if m else 0
        
        model_files.sort(key=parse_episode)
        
        # 准备写入CSV
        fieldnames = ['model_path', 'episode', 'reward', 'avg_delay', 'total_energy', 'steps']
        
        with open(csv_filename, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for idx, model_file in enumerate(model_files, 1):
                model_path = os.path.join(subdir_path, model_file)
                episode_num = parse_episode(model_file)
                
                # 加载模型
                agent.load(model_path)
                
                # 评估
                result = evaluate_one_episode(env, agent, max_steps=max_steps, save_details=save_details)
                
                # 写入CSV
                csv_row = {
                    'model_path': model_file, # 只保留文件名，例如 episode_250_model.pth
                    'episode': episode_num,
                    **{k: v for k, v in result.items() if k != 'steps_detail'}
                }
                writer.writerow(csv_row)
                
                print(f"   [{idx}/{len(model_files)}] Ep {episode_num}: "
                      f"R={result['reward']:.2f}, D={result['avg_delay']:.4f}, E={result['total_energy']:.1f}")
                
                # 保存详细JSON
                if save_details and 'steps_detail' in result:
                    safe_name = f"{subdir_name}_ep{episode_num}"
                    json_file = f"{output_json_dir}/{safe_name}_detail.json"
                    
                    json_output = {
                        'model': model_path,
                        'episode': episode_num,
                        'timestamp': datetime.now().isoformat(),
                        'summary': {k: v for k, v in result.items() if k != 'steps_detail'},
                        'steps': result['steps_detail']
                    }
                    
                    with open(json_file, 'w', encoding='utf-8') as jf:
                        json.dump(json_output, jf, indent=2, ensure_ascii=False)
        
        print(f"   ✓ 结果已保存至 {csv_filename}")

    print("\n" + "="*70)
    print("所有评估任务完成!")
    print("="*70)


if __name__ == "__main__":
    evaluate_models(
        base_models_dir="saved_models",
        output_json_dir="eval_details",
        trajectory_file="user_trajectories.json", 
        max_distance=60,
        max_steps=30,
        save_details=True
    )
