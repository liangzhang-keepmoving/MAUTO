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
def save_uav_trajectories_to_json(episode, trajectories, initial_positions,
                                  save_dir='uav_trajectories', run_id=None):
    """
    保存 UAV 轨迹到 JSON（不包含 actions）
    trajectories: dict[uav_id] -> list[[x,y], ...]  （含起点与每步后的新位置）
    initial_positions: { 'uav_0': [x,y], ... }
    """
    if run_id:
        save_dir = f'{save_dir}/run_{run_id}'
    os.makedirs(save_dir, exist_ok=True)

    filename = f'episode_{episode}_trajectories.json'
    filepath = os.path.join(save_dir, filename)

    # 统一转为可序列化
    traj_serializable = {
        f'uav_{uid}': [[float(p[0]), float(p[1])] for p in traj_list]
        for uid, traj_list in trajectories.items()
    }

    save_data = {
        'episode': episode,
        'total_steps': max(len(v) for v in trajectories.values()) - 1,  # 不含起点的数量
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'initial_positions': initial_positions,
        'trajectories': traj_serializable  # 每架UAV的完整轨迹（起点 + 每步新位置）
    }

    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(save_data, f, indent=2, ensure_ascii=False)

    print(f"  ✓ 轨迹已保存到: {filepath}")


def convert_soft_to_hard_allocation(allocation_soft, num_uavs, num_users):
    """将软分配转换为硬分配"""
    allocation_hard = torch.zeros_like(allocation_soft)
    for user_id in range(num_users):
        user_probs = allocation_soft[:, user_id]
        best_uav = torch.argmax(user_probs)
        allocation_hard[best_uav, user_id] = 1.0
    return allocation_hard


def train_ddpg(config):

    run_id = time.strftime('%Y%m%d_%H%M%S')  # 本次训练唯一标识

    # 创建环境
    env = SimplifiedMultiUAVEnvironment(
        num_uavs=config['num_uavs'], 
        num_users=config['num_users'],
        trajectory_file="user_trajectories.json",
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
    
    # 初始化CSV文件（每次训练都创建新文件）
    csv_file = 'training_history.csv'
    fieldnames = ['episode', 'reward', 'avg_delay_sum', 'max_delay_sum', 
                  'task_energy', 'move_energy', 'steps',
                  'w_avg_delay', 'w_task_energy', 'w_move_energy']
    for user_id in range(config['num_users']):
        fieldnames.append(f'user_{user_id}_delay')
    with open(csv_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

    for episode in range(config['max_episodes']):
        # 重置环境
        state = env.reset()
        episode_reward = 0
        episode_steps = 0
        
        # 累加指标
        episode_avg_delay = 0
        episode_max_delay = 0
        episode_task_energy = 0
        episode_move_energy = 0
        user_delays = [0] * config['num_users']

        # ====== 轨迹收集：起点 ======
        # env.uav_states 预期是 (num_uavs, 2) 的 numpy 数组
        uav_states_np = np.asarray(env.uav_states, dtype=float)
        assert uav_states_np.shape[1] >= 2, f"env.uav_states shape invalid: {uav_states_np.shape}"

        initial_positions = {
            f'uav_{uav_id}': [float(uav_states_np[uav_id, 0]), float(uav_states_np[uav_id, 1])]
            for uav_id in range(env.num_uavs)
        }
        trajectories = {
            uav_id: [[float(uav_states_np[uav_id, 0]), float(uav_states_np[uav_id, 1])]]
            for uav_id in range(env.num_uavs)
        }
        
        # 重置噪声
        agent.reset_noise()

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
            
            next_state, reward, done, info = env.step(actions)

            # ====== 轨迹收集：用 info['uav_position'] 追加下一时刻位置 ======
            uav_pos = info['uav_position']     # 直接就是 (num_uavs, 2) 的 numpy

            for uav_id in range(env.num_uavs):
                x, y = uav_pos[uav_id]
                trajectories[uav_id].append([float(x), float(y)])


            # 累加指标
            reward_comp = info['reward_components']
            episode_avg_delay += reward_comp['avg_delay']
            episode_max_delay += reward_comp['max_delay']
            episode_task_energy += reward_comp['total_task_energy']
            episode_move_energy += reward_comp['total_move_energy']
            for user_id in range(config['num_users']):
                user_delays[user_id] += info['raw_metrics'][user_id]['user_actual_delay']

            episode_reward += reward
            episode_steps += 1
            
            # 存储经验
            action_tensors = (
                allocation.detach(),
                offloading.detach(), 
                motion.detach()
            )
            agent.replay_buffer.push(
                state, action_tensors, reward, next_state, done
            )
            
            # 训练
            actor_loss, critic_loss = None, None
            if len(agent.replay_buffer) >= config['batch_size']:
                actor_loss, critic_loss = agent.train_step(config['batch_size'])
            
            state = next_state
            
            if done:
                break

        agent.decay_noise()

        # ====== 每个 episode 保存轨迹（不再保存 action_logs） ======
        if episode%50 ==0:
            save_uav_trajectories_to_json(
                episode=episode,
                trajectories=trajectories,
                initial_positions=initial_positions,
                save_dir='uav_trajectories',
                run_id=run_id
            )

        env.reward_system.on_episode_end(episode_reward)
        
        # 写入CSV
        current_weights = env.reward_system.weights 
        row_data = {
            'episode': episode,
            'reward': round(episode_reward, 4),
            'avg_delay_sum': round(episode_avg_delay / episode_steps, 6),
            'max_delay_sum': round(episode_max_delay, 6),
            'task_energy': round(episode_task_energy, 2),
            'move_energy': round(episode_move_energy, 2),
            'steps': episode_steps,
            'w_avg_delay': round(current_weights['w_avg_delay'], 4),
            'w_task_energy': round(current_weights['w_task_energy'], 4),
            'w_move_energy': round(current_weights['w_move_energy'], 4)
        }
        for user_id in range(config['num_users']):
            row_data[f'user_{user_id}_delay'] = round(user_delays[user_id], 6)

        with open(csv_file, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writerow(row_data)
        
        # 打印进度
        if episode % 10 == 0:
            print(f"\nEpisode {episode}:")
            print(f"  Reward: {episode_reward:.2f}")
            print(f"  Avg Delay: {episode_avg_delay/episode_steps:.4f}")
            print(f"  Task Energy: {episode_task_energy:.2f}, Move Energy: {episode_move_energy:.2f}")
            print(f"  Weights: delay={current_weights['w_avg_delay']:.3f}, "
                  f"task_e={current_weights['w_task_energy']:.3f}, "
                  f"move_e={current_weights['w_move_energy']:.3f}")
            diagnostics = env.reward_system.get_diagnostics()
            if diagnostics:
                print(f"  Variance Ratios: delay/energy={diagnostics['variance_ratios']['delay_vs_energy']:.2f}")
    
    print(f"\n训练完成！数据已保存到 {csv_file}")


def main():
    """主函数"""
    config = {
        # 环境参数
        'num_uavs': 2,
        'num_users': 6,
        
        # 训练参数
        'max_episodes': 800,
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
    train_ddpg(config)


if __name__ == "__main__":
    main()
