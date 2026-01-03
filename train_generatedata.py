"""
修改后的DDPG训练脚本 - 自动收集LLM训练数据
Modified DDPG Training - Auto-collect LLM Training Data

新增功能：
1. 训练过程中保存每个回合的状态-动作对
2. 自动选择最佳回合保存为LLM训练数据
3. 支持保存最后N个回合的数据
"""

import os
import time
import numpy as np
import torch
import matplotlib.pyplot as plt
import json
import csv
from datetime import datetime
from collections import deque

# 导入自定义模块
from DDPG import DDPGAgent
from env_train import TrainingMultiUAVEnvironment


def convert_soft_to_hard_allocation(allocation_soft, num_uavs, num_users):
    """将软分配转换为硬分配"""
    allocation_hard = torch.zeros_like(allocation_soft)
    uav_loads = np.zeros(num_uavs, dtype=int)
    tie_margin = 0.05
    for user_id in range(num_users):
        user_probs = allocation_soft[:, user_id]
        probs_np = user_probs.detach().cpu().numpy()
        max_prob = float(np.max(probs_np))
        if num_uavs >= 2:
            sorted_probs = np.sort(probs_np)
            second_prob = float(sorted_probs[-2])
        else:
            second_prob = float("-inf")

        if max_prob - second_prob <= tie_margin:
            candidates = np.flatnonzero(probs_np >= (max_prob - tie_margin))
            min_load = int(np.min(uav_loads[candidates])) if candidates.size > 0 else int(np.min(uav_loads))
            best_candidates = candidates[uav_loads[candidates] == min_load] if candidates.size > 0 else np.flatnonzero(uav_loads == min_load)
            best_uav = int(np.random.choice(best_candidates))
        else:
            best_uav = int(torch.argmax(user_probs).item())
        allocation_hard[best_uav, user_id] = 1.0
        uav_loads[best_uav] += 1
    return allocation_hard


def save_actions_to_json(episode, actions_history, initial_positions, save_dir='action_logs', run_id=None):
    """保存动作和位置数据"""
    if run_id:
        save_dir = f'{save_dir}/run_{run_id}'
    
    os.makedirs(save_dir, exist_ok=True)
    
    filename = f'episode_{episode}_actions.json'
    filepath = os.path.join(save_dir, filename)
    
    # 保存数据
    save_data = {
        'episode': episode,
        'total_steps': len(actions_history),
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'initial_positions': initial_positions,
        'actions': actions_history
    }
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(save_data, f, indent=2, ensure_ascii=False)
    
    print(f"  ✓ 动作已保存到: {filepath}")


# ============ 新增：LLM数据收集函数 ============

def tensor_to_list(tensor):
    """将Tensor转换为列表"""
    if isinstance(tensor, torch.Tensor):
        return tensor.cpu().numpy().tolist()
    elif isinstance(tensor, np.ndarray):
        return tensor.tolist()
    else:
        return tensor


def save_episode_for_llm(episode, episode_data, save_dir='llm_training_data'):
    """
    保存单个episode的LLM训练数据
    
    Args:
        episode: episode编号
        episode_data: episode数据列表，每个元素是一个step的数据
        save_dir: 保存目录
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # 保存为JSONL格式（每行一个样本）
    filename = f'episode_{episode:04d}.jsonl'
    filepath = os.path.join(save_dir, filename)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        for sample in episode_data:
            json.dump(sample, f, ensure_ascii=False)
            f.write('\n')
    
    return filepath


def create_llm_sample(episode, step, state, actions, reward, next_state, done, num_uavs):
    """
    创建LLM训练样本
    
    Args:
        episode: episode编号
        step: 步数
        state: 当前状态 {'uav_pos': tensor, 'user_pos': tensor, 'user_tasks': tensor}
        actions: 动作字典 {'uav_0': {...}, 'uav_1': {...}}
        reward: 奖励值
        next_state: 下一个状态
        done: 是否结束
        num_uavs: UAV数量
    
    Returns:
        dict: LLM训练样本
    """
    sample = {
        'episode': int(episode),
        'step': int(step),
        'state': {
            'uav_pos': tensor_to_list(state['uav_pos']),
            'user_pos': tensor_to_list(state['user_pos']),
            'user_tasks': tensor_to_list(state['user_tasks'])
        },
        'action': {
            f'uav_{uav_id}': {
                'user_assignments': tensor_to_list(actions[f'uav_{uav_id}']['user_competition_probs']),
                'offloading_ratios': tensor_to_list(actions[f'uav_{uav_id}']['offloading_ratios']),
                'movement_direction': float(actions[f'uav_{uav_id}']['movement_direction']),
                'movement_distance': float(actions[f'uav_{uav_id}']['movement_distance'])
            }
            for uav_id in range(num_uavs)
        },
        'reward': float(reward),
        'next_state': {
            'uav_pos': tensor_to_list(next_state['uav_pos']),
            'user_pos': tensor_to_list(next_state['user_pos']),
            'user_tasks': tensor_to_list(next_state['user_tasks'])
        },
        'done': bool(done)
    }
    
    return sample


def select_and_save_best_episodes(episode_rewards, all_episode_data, save_dir='llm_training_data', top_k=20):
    """
    选择并保存最佳的K个episode
    
    Args:
        episode_rewards: episode奖励列表 [(episode_id, total_reward, episode_data), ...]
        all_episode_data: 所有episode数据
        save_dir: 保存目录
        top_k: 保存前K个最佳episode
    
    Returns:
        list: 最佳样本列表
    """
    print("\n" + "="*70)
    print(f"选择Top-{top_k}最佳Episode用于LLM训练")
    print("="*70)
    
    if len(episode_rewards) == 0:
        print("✗ 没有episode数据")
        return []
    
    # ⭐ 创建保存目录（如果不存在）
    os.makedirs(save_dir, exist_ok=True)
    print(f"✓ 确保目录存在: {save_dir}")
    
    # 按奖励排序（降序）
    sorted_episodes = sorted(episode_rewards, key=lambda x: x[1], reverse=True)
    best_episodes = sorted_episodes[:min(top_k, len(sorted_episodes))]
    
    # 合并最佳episode的所有样本
    all_best_samples = []
    
    print(f"\n最佳{len(best_episodes)}个Episode:")
    for rank, (ep_id, ep_reward, ep_data) in enumerate(best_episodes):
        print(f"  Rank {rank+1}: Episode {ep_id}, Reward={ep_reward:.3f}, Samples={len(ep_data)}")
        all_best_samples.extend(ep_data)
    
    # 保存合并后的最佳样本
    output_file = os.path.join(save_dir, 'best_samples.jsonl')
    with open(output_file, 'w', encoding='utf-8') as f:
        for sample in all_best_samples:
            json.dump(sample, f, ensure_ascii=False)
            f.write('\n')
    
    # 保存元数据
    metadata = {
        'num_best_episodes': len(best_episodes),
        'total_samples': len(all_best_samples),
        'best_episode_ids': [ep[0] for ep in best_episodes],
        'best_episode_rewards': [ep[1] for ep in best_episodes],
        'avg_reward': np.mean([ep[1] for ep in best_episodes]),
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    metadata_file = os.path.join(save_dir, 'metadata.json')
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ 保存了 {len(all_best_samples)} 个样本到: {output_file}")
    print(f"✓ 元数据保存到: {metadata_file}")
    print(f"平均奖励: {metadata['avg_reward']:.3f}")
    print("="*70)
    
    return all_best_samples


# ============ 修改后的训练函数 ============

def train_ddpg(config):
    """主训练函数（带LLM数据收集）"""
    print("=== 初始化多无人机DDPG训练 ===")
    print("⭐ 新增：自动收集LLM训练数据")
    
    # 创建环境
    env = TrainingMultiUAVEnvironment(
        num_uavs=config['num_uavs'], 
        num_users=config['num_users'],
        trajectory_file="user_trajectories.json"
    )
    
    # 创建智能体
    agent = DDPGAgent(
        n_uavs=config['num_uavs'],
        n_users=config['num_users'],
        lr_actor=config['lr_actor'],
        lr_critic=config['lr_critic'],
        gamma=config['gamma'],
        tau=config['tau'],
        max_distance=config['max_distance']
    )
    
    print(f"环境: {config['num_uavs']}无人机, {config['num_users']}用户")
    print(f"训练episodes: {config['max_episodes']}")
    print(f"设备: {agent.device}")
    
    # 训练循环
    start_time = time.time()
    
    # 初始化CSV文件
    csv_file = 'training_history.csv'
    fieldnames = ['episode', 'reward', 'task_energy','transmission_energy', 'movement_energy', 'steps']
    for uav_id in range(config['num_uavs']):
        fieldnames.append(f'uav_{uav_id}_delay')
    
    with open(csv_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
    print(f"创建新的训练记录文件: {csv_file}")
    
    # ⭐ 新增：用于存储episode数据
    episode_rewards = []  # 存储 (episode_id, total_reward, episode_data)
    save_last_n_episodes = config.get('save_last_n_episodes', 50)  # 保存最后N个episode
    
    for episode in range(config['max_episodes']):
        # 重置环境
        state = env.reset()
        episode_reward = 0
        episode_steps = 0
        episode_task_energy = 0
        episode_transmission_energy = 0
        episode_movement_energy = 0
        actions_history = []
        uav_delays = [0] * config['num_uavs']
        
        # ⭐ 新增：存储本episode的LLM训练数据
        episode_llm_data = []
        
        # 重置噪声
        agent.reset_noise()

        if episode % 10 == 0:
            initial_positions = {
                f'uav_{uav_id}': [
                    float(env.uav_states[uav_id, 0]),
                    float(env.uav_states[uav_id, 1])
                ]
                for uav_id in range(env.num_uavs)
            }

        for step in range(config['max_steps_per_episode']):
            # 选择动作
            allocation, offloading, motion = agent.select_action(
                state, 
                add_noise=True,
                hard=False
            )
            allocation_hard = convert_soft_to_hard_allocation(allocation, env.num_uavs, env.num_users)
            
            # 构建环境动作格式
            actions = {}
            for uav_id in range(env.num_uavs):
                actions[f'uav_{uav_id}'] = {
                    'user_competition_probs': allocation_hard[uav_id].cpu().numpy(),
                    'offloading_ratios': offloading[uav_id].cpu().numpy(),
                    'movement_direction': motion[uav_id, 1].item() * np.pi,
                    'movement_distance': motion[uav_id, 0].item() * agent.max_distance
                }
            
            # 执行动作
            next_state, rewards, done, info = env.step(actions)

            # ⭐ 新增：创建并保存LLM训练样本
            llm_sample = create_llm_sample(
                episode=episode,
                step=step,
                state=state,
                actions=actions,
                reward=rewards,
                next_state=next_state,
                done=done,
                num_uavs=config['num_uavs']
            )
            episode_llm_data.append(llm_sample)

            # 原有的动作保存逻辑
            if episode % 10 == 0:
                step_data = {
                    'step': step,
                    'actions': {}
                }
                
                for uav_id in range(env.num_uavs):
                    uav_key = f'uav_{uav_id}'
                    step_data['actions'][uav_key] = {
                        'user_competition_probs': actions[uav_key]['user_competition_probs'].tolist(),
                        'offloading_ratios': actions[uav_key]['offloading_ratios'].tolist(),
                        'movement_direction': float(actions[uav_key]['movement_direction']),
                        'movement_distance': float(actions[uav_key]['movement_distance']),
                        'position': [
                            float(env.uav_states[uav_id, 0]),
                            float(env.uav_states[uav_id, 1])
                        ]
                    }
                
                actions_history.append(step_data)
            
            # 原有的环境步进逻辑
            raw_metrics = info.get('raw_metrics', {})
            
            episode_task_energy += sum([raw_metrics[f'uav_{i}']['actual_task_energy_raw'] for i in range(env.num_uavs)])
            episode_transmission_energy += sum([raw_metrics[f'uav_{i}']['actual_transmission_energy_raw'] for i in range(env.num_uavs)])
            episode_movement_energy += sum([raw_metrics[f'uav_{i}']['uav_movement_energy_raw'] for i in range(env.num_uavs)])
            
            for uav_id in range(env.num_uavs):
                uav_delays[uav_id] += raw_metrics[f'uav_{uav_id}']['delay_raw']
            
            total_reward = rewards
            episode_reward += total_reward
            episode_steps += 1
            
            # 存储经验
            action_tensors = (
                allocation.detach(),
                offloading.detach(), 
                motion.detach()
            )
            
            agent.replay_buffer.push(
                state, action_tensors, total_reward, next_state, done
            )
            
            # 训练
            if len(agent.replay_buffer) >= config['batch_size']:
                actor_loss, critic_loss = agent.train_step(config['batch_size'])
            
            state = next_state
            
            if done:
                break
        
        agent.decay_noise()
        
        # ⭐ 新增：保存本episode的LLM数据
        episode_rewards.append((episode, episode_reward, episode_llm_data))
        
        # 只保留最后N个episode的数据（节省内存）
        if len(episode_rewards) > save_last_n_episodes:
            episode_rewards.pop(0)
        
        # 保存动作日志
        if episode % 10 == 0:
            print(f"\n保存Episode {episode}的动作数据...")
            save_actions_to_json(episode, actions_history, initial_positions)
        
        # 保存CSV数据
        row_data = {
            'episode': episode,
            'reward': round(episode_reward, 4),
            'task_energy': round(episode_task_energy, 4),
            'movement_energy': round(episode_movement_energy, 4),
            'transmission_energy': round(episode_transmission_energy, 4),
            'steps': episode_steps
        }
        
        for uav_id in range(config['num_uavs']):
            row_data[f'uav_{uav_id}_delay'] = round(uav_delays[uav_id], 6)
        
        with open(csv_file, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writerow(row_data)
        
        # 打印信息
        uav_delays_str = ", ".join([f"UAV{i}={uav_delays[i]:.4f}" 
                                     for i in range(config['num_uavs'])])
        print(f"第{episode}回合：奖励={episode_reward:.2f}, "
              f"任务能耗={episode_task_energy:.2f}, "
              f"移动能耗={episode_movement_energy:.2f}")
        print(f"  各UAV总时延: {uav_delays_str}")
    
    print(f"\n训练完成！数据已保存到 {csv_file}")
    
    # ⭐ 新增：训练结束后，选择并保存最佳episode
    print("\n" + "="*70)
    print("开始选择并保存LLM训练数据...")
    print("="*70)
    
    best_samples = select_and_save_best_episodes(
        episode_rewards=episode_rewards,
        all_episode_data=None,
        save_dir='llm_training_data',
        top_k=config.get('save_top_k_episodes', 20)
    )
    
    print(f"\n✓ LLM训练数据准备完成！")
    print(f"  - 最佳样本数: {len(best_samples)}")
    print(f"  - 保存位置: llm_training_data/best_samples.jsonl")
    print(f"  - 可直接用于训练LLM控制器")
    
    return agent


# ============ 主函数 ============

def main():
    """主函数"""
    
    # 训练配置
    config = {
        # 环境参数
        'num_uavs': 2,
        'num_users': 5,
        
        # 训练参数
        'max_episodes': 300,
        'max_steps_per_episode': 40,
        'batch_size': 64,
        
        # 网络参数
        'lr_actor': 1e-4,
        'lr_critic': 1e-3,
        'gamma': 0.99,
        'tau': 0.005,
        'max_distance': 30,
        
        # ⭐ 新增：LLM数据收集参数
        'save_last_n_episodes': 50,    # 保存最后50个episode
        'save_top_k_episodes': 1,     # 从中选择最好的20个
    }
    
    print("开始DDPG多无人机训练...")
    print(f"配置: {json.dumps(config, indent=2, ensure_ascii=False)}")
    
    # 开始训练
    trained_agent = train_ddpg(config)
    
    # 保存最终模型
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_path = f'ddpg_model_{timestamp}.pth'
    trained_agent.save(model_path)
    print(f"\n✓ 最终模型已保存: {model_path}")


if __name__ == "__main__":
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 开始训练
    main()
