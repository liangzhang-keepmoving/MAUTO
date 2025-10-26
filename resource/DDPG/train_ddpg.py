import os
import time
import numpy as np
import torch
import matplotlib.pyplot as plt
from collections import deque
import json
import csv
from datetime import datetime

# 导入自定义模块
from DDPG import DDPGAgent
from env_simplified import SimplifiedMultiUAVEnvironment

def convert_soft_to_hard_allocation(allocation_soft, num_uavs, num_users):
    """将软分配转换为硬分配"""
    allocation_hard = torch.zeros_like(allocation_soft)
    for user_id in range(num_users):
        user_probs = allocation_soft[:, user_id]
        best_uav = torch.argmax(user_probs)
        allocation_hard[best_uav, user_id] = 1.0
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
        'initial_positions': initial_positions,  # 🆕 添加初始位置
        'actions': actions_history
    }
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(save_data, f, indent=2, ensure_ascii=False)
    
    print(f"  ✓ 动作已保存到: {filepath}")

def train_ddpg(config):
    """主训练函数"""
    print("=== 初始化多无人机DDPG训练 ===")
    
    # 创建环境
    env = SimplifiedMultiUAVEnvironment(
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
    
    # 初始化CSV文件（每次训练都创建新文件）
    csv_file = 'training_history.csv'
    
    # 构建CSV列名
    fieldnames = ['episode', 'reward', 'task_energy','transmission_energy', 'movement_energy', 'steps']
    
    # 为每个无人机添加列
    for uav_id in range(config['num_uavs']):
        fieldnames.append(f'uav_{uav_id}_delay')
    
    # 创建新文件并写入表头
    with open(csv_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
    print(f"创建新的训练记录文件: {csv_file}")

    
    for episode in range(config['max_episodes']):
        # 重置环境
        state = env.reset()
        episode_reward = 0
        episode_steps = 0
        episode_task_energy = 0
        episode_transmission_energy = 0
        episode_movement_energy = 0
        actions_history = []
        # 为每个无人机单独累加时延
        uav_delays = [0] * config['num_uavs']
        
        # 重置噪声
        agent.reset_noise()  # 新增

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
                hard=False  # 训练时使用软分配
            )
            allocation_hard = convert_soft_to_hard_allocation(allocation, env.num_uavs, env.num_users)
            # 构建环境动作格式
            actions = {}
            for uav_id in range(env.num_uavs):
                actions[f'uav_{uav_id}'] = {
                    'user_competition_probs': allocation_hard[uav_id].cpu().numpy(),
                    'offloading_ratios': offloading[uav_id].cpu().numpy(),
                    'movement_direction': motion[uav_id, 1].item() * np.pi,  # 转换为弧度
                    'movement_distance': motion[uav_id, 0].item() * agent.max_distance
                }
            
            next_state, rewards, done, info = env.step(actions)

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
                        # ✅ 保存移动后的位置
                        'position': [
                            float(env.uav_states[uav_id, 0]),
                            float(env.uav_states[uav_id, 1])
                        ]
                    }
                
                actions_history.append(step_data)
            
            # 环境步进

            raw_metrics = info.get('raw_metrics', {})
            
            # 累加总体能耗
            episode_task_energy += sum([raw_metrics[f'uav_{i}']['actual_task_energy_raw'] for i in range(env.num_uavs)])
            episode_transmission_energy += sum([raw_metrics[f'uav_{i}']['actual_transmission_energy_raw'] for i in range(env.num_uavs)])
            episode_movement_energy += sum([raw_metrics[f'uav_{i}']['uav_movement_energy_raw'] for i in range(env.num_uavs)])
            
            
            # 为每个无人机单独累加时延（任务时延+移动时延）
            for uav_id in range(env.num_uavs):
                uav_delays[uav_id] += raw_metrics[f'uav_{uav_id}']['delay_raw']
            
            # 计算总奖励
            total_reward = rewards
            episode_reward += total_reward
            episode_steps += 1
            
            # 存储经验
            # 将动作转换为tensor格式用于存储
            action_tensors = (
                allocation.detach(),
                offloading.detach(), 
                motion.detach()
            )
            
            agent.replay_buffer.push(
                state, action_tensors, total_reward, next_state, done
            )
            
            # 训练
            actor_loss, critic_loss = None, None
            if len(agent.replay_buffer) >= config['batch_size']:
                actor_loss, critic_loss = agent.train_step(config['batch_size'])
            
            state = next_state
            
            if done:
                break
        agent.decay_noise() 
        # 构建本episode的数据行
        if episode % 10 == 0:
            print(f"\n保存Episode {episode}的动作数据...")
            save_actions_to_json(episode, actions_history,initial_positions)
        row_data = {
            'episode': episode,
            'reward': round(episode_reward, 4),
            'task_energy': round(episode_task_energy, 4),
            'movement_energy': round(episode_movement_energy, 4),
            'transmission_energy': round(episode_transmission_energy, 4),
            'steps': episode_steps
        }
        
        # 添加每个无人机的总时延
        for uav_id in range(config['num_uavs']):
            row_data[f'uav_{uav_id}_delay'] = round(uav_delays[uav_id], 6)
        
        # 直接追加到CSV文件
        with open(csv_file, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writerow(row_data)
        
        # 打印信息（包含每个无人机的时延）
        uav_delays_str = ", ".join([f"UAV{i}={uav_delays[i]:.4f}" 
                                     for i in range(config['num_uavs'])])
        print(f"第{episode}回合：奖励={episode_reward:.2f}, "
              f"任务能耗={episode_task_energy:.2f}, "
              f"移动能耗={episode_movement_energy:.2f}")
        print(f"  各UAV总时延: {uav_delays_str}")
    
    print(f"\n训练完成！数据已保存到 {csv_file}")

def generate_ddpg_random_actions_hard_and_convert(n_uavs, n_users):
    """
    生成符合 DDPG Actor 输出约束的随机**硬分配 (0/1)**动作张量，并转换为原始字典格式。
    
    Returns:
        actions_dict: 原始字典格式的动作
    """
    
    # === 1. 硬分配 (列和为 1, 0/1 约束) ===
    # 为每个用户 (M) 随机选择一个无人机 (N)
    chosen_uavs = np.random.randint(0, n_uavs, size=n_users)
    
    # 构建硬分配矩阵 [N, M]
    allocation_hard = np.zeros((n_uavs, n_users), dtype=float) # 使用 float 类型以便后续乘法
    allocation_hard[chosen_uavs, np.arange(n_users)] = 1.0  # [N, M] - 列和为 1，元素为 0 或 1
    
    # === 2. 原始卸载比例 (连续值) ===
    # 随机原始比例 [N, M] in [0, 1] 
    offloading_ratio_raw = np.random.uniform(0, 1, size=(n_uavs, n_users))
    
    # 最终卸载比例：耦合硬分配 (非零比例只出现在被分配的 UAV-User 对上)
    offloading_final = allocation_hard * offloading_ratio_raw
    
    # === 3. 运动控制 (归一化) ===
    # distance [N, 1] in [0, 1] (模仿 Sigmoid)
    distance = np.random.uniform(0, 1, size=(n_uavs, 1))
    # angle [N, 1] in [-1, 1] (模仿 Tanh)
    angle = np.random.uniform(-1, 1, size=(n_uavs, 1))
    
    # === 4. 转换为原始字典格式 (使用硬分配结果进行填充) ===
    
    # 假设运动参数的原始范围是：
    MAX_MOVE_DISTANCE = 30  # 假设原始代码中的 20 是最大距离
    
    actions_dict = {}
    for uav_id in range(n_uavs):
        
        # 妥协 1: 用户竞争/分配 ('user_competition_probs')
        # 在硬分配模式下，每个 UAV i 仅对那些分配给它的用户 j 具有 '1' 的值。
        # 这里使用 allocation_hard 的行作为填充。
        user_competition_probs_proxy = allocation_hard[uav_id, :] 
        
        # 妥协 2: 卸载比例 ('offloading_ratios')
        # 使用最终耦合后的卸载比例 (offloading_final) 的行。
        offloading_ratios_proxy = offloading_final[uav_id, :]
        
        # 运动参数转换
        movement_distance_real = distance[uav_id, 0] * MAX_MOVE_DISTANCE 
        movement_direction_real = (angle[uav_id, 0] + 1) / 2 * (2 * np.pi) # [-1, 1] -> [0, 2*pi]
        
        actions_dict[f'uav_{uav_id}'] = {
            'user_competition_probs': user_competition_probs_proxy, # 包含 0 或 1
            'offloading_ratios': offloading_ratios_proxy,             # 包含 0 或 [0, 1] 的值
            'movement_direction': movement_direction_real,
            'movement_distance': movement_distance_real
        }
        
    return actions_dict

# =========================================================================

def estimate_reference_values(config, num_episodes=20000):
    env = SimplifiedMultiUAVEnvironment(
        num_uavs=config['num_uavs'], 
        num_users=config['num_users'],
        trajectory_file="user_trajectories.json"
    )
    """估计归一化参考值"""
    energies = []
    delays = []
    transmission_energies = []
    movement_energies = []
    task_energies = []
    movement_delays = []
    transmission_delays = []
    
    print("正在估计参考值...")
    
    for ep in range(num_episodes):
        state = env.reset()
       
        for step in range(40):
            # 🚀 关键修改: 生成 DDPG 硬分配动作并转换回原始字典格式
            actions = generate_ddpg_random_actions_hard_and_convert(env.num_uavs, env.num_users)
            
            
            
            next_state, reward, done, info = env.step(actions)
            
            # 累加
            raw_metrics = info.get('raw_metrics', {})

            task_energies.append(raw_metrics[f'uav_{0}']['actual_task_energy_raw'])
            task_energies.append(raw_metrics[f'uav_{1}']['actual_task_energy_raw'])
            transmission_energies.append(raw_metrics[f'uav_{0}']['actual_transmission_energy_raw'])
            transmission_energies.append(raw_metrics[f'uav_{1}']['actual_transmission_energy_raw'])
            movement_energies.append(raw_metrics[f'uav_{0}']['uav_movement_energy_raw'])
            movement_energies.append(raw_metrics[f'uav_{1}']['uav_movement_energy_raw'])
            delays.append(raw_metrics[f'uav_{0}']['delay_raw'])
            delays.append(raw_metrics[f'uav_{1}']['delay_raw'])
            movement_delays.append(raw_metrics[f'uav_{0}']['movement_delay_raw'])
            movement_delays.append(raw_metrics[f'uav_{1}']['movement_delay_raw'])
            transmission_delays.append(raw_metrics[f'uav_{0}']['transmission_delay_raw'])
            transmission_delays.append(raw_metrics[f'uav_{1}']['transmission_delay_raw'])
          
            if done:
                break
        
       
    
    # 统计

    mean_task_energy = np.mean(task_energies)
    std_task_energy = np.std(task_energies)
    mean_transmission_energy = np.mean(transmission_energies)
    std_transmission_energy = np.std(transmission_energies)
    mean_movement_energy = np.mean(movement_energies)
    std_movement_energy = np.std(movement_energies)
    mean_delay = np.mean(delays)
    std_delay = np.std(delays)
    mean_movement_delay = np.mean(movement_delays)
    std_movement_delay = np.std(movement_delays)
    mean_transmission_delay = np.mean(transmission_delays)
    std_transmission_delay = np.std(transmission_delays)
    energy_ref = mean_task_energy + std_task_energy
    transmission_energy_ref = mean_transmission_energy + std_transmission_energy
    movement_energy_ref = mean_movement_energy + std_movement_energy
    delay_ref = mean_delay + std_delay
    movement_delay_ref = mean_movement_delay + std_movement_delay
    transmission_delay_ref = mean_transmission_delay + std_transmission_delay
    max_task_energy_ref = max(task_energies)
    max_transmission_energy_ref = max(transmission_energies)
    max_movement_energy_ref = max(movement_energies)
    max_delay_ref = max(delays)
    max_movement_delay_ref = max(movement_delays)
    max_transmission_delay_ref = max(transmission_delays)
    min_task_energy_ref = min(task_energies)
    min_transmission_energy_ref = min(transmission_energies)
    min_movement_energy_ref = min(movement_energies)
    min_delay_ref = min(delays)
    min_movement_delay_ref = min(movement_delays)
    min_transmission_delay_ref = min(transmission_delays)
    # 建议参考值（均值 + 1倍标准差）

    print(f"\n推荐参考值:")
    print(f"  MAX_TASK_ENERGY_REF = {max_task_energy_ref:.1f}  # J")
    print(f"  MAX_TRANSMISSION_ENERGY_REF = {max_transmission_energy_ref:.1f}  # J")
    print(f"  MAX_MOVEMENT_ENERGY_REF = {max_movement_energy_ref:.1f}  # J")
    print(f"  MAX_DELAY_REF = {max_delay_ref:.2f}   # s")
    print(f"  MAX_MOVEMENT_DELAY_REF = {max_movement_delay_ref:.2f}   # s")
    print(f"  MAX_TRANSMISSION_DELAY_REF = {max_transmission_delay_ref:.2f}   # s")
    print(f"  MIN_TASK_ENERGY_REF = {min_task_energy_ref:.1f}  # J")
    print(f"  MIN_TRANSMISSION_ENERGY_REF = {min_transmission_energy_ref:.1f}  # J")
    print(f"  MIN_MOVEMENT_ENERGY_REF = {min_movement_energy_ref:.1f}  # J")
    print(f"  MIN_DELAY_REF = {min_delay_ref:.2f}   # s")
    print(f"  MIN_MOVEMENT_DELAY_REF = {min_movement_delay_ref:.2f}   # s")
    print(f"  MIN_TRANSMISSION_DELAY_REF = {min_transmission_delay_ref:.2f}   # s")
    print(f"  ENERGY_REF = {energy_ref:.1f}  # J")
    print(f"  DELAY_REF = {delay_ref:.2f}   # s")
    print(f"  TRANSMISSION_ENERGY_REF = {transmission_energy_ref:.1f}  # J")
    print(f"  MOVEMENT_ENERGY_REF = {movement_energy_ref:.1f}  # J")
    print(f"  MOVEMENT_DELAY_REF = {movement_delay_ref:.2f}   # s")
    print(f"  TRANSMISSION_DELAY_REF = {transmission_delay_ref:.2f}   # s")

    
    return energy_ref, delay_ref












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
    }
    
    print("开始DDPG多无人机训练...")
    print(f"配置: {json.dumps(config, indent=2, ensure_ascii=False)}")
    
    # 开始训练
    train_ddpg(config)
    #estimate_reference_values(config)



if __name__ == "__main__":
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 开始训练
    main()