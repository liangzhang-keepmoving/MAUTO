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
from env_train import TrainingMultiUAVEnvironment
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


def convert_soft_to_hard_allocation(allocation_soft, num_uavs, num_users, epsilon=0.0):
    """
    将软分配转换为硬分配
    """
    allocation_hard = torch.zeros_like(allocation_soft)
    for user_id in range(num_users):
        user_probs = allocation_soft[:, user_id]
        best_uav = torch.argmax(user_probs)
        
    allocation_hard[best_uav, user_id] = 1.0
        
    return allocation_hard


def train_ddpg(config):
    
    # 创建环境
    env = TrainingMultiUAVEnvironment(
        num_uavs=config['num_uavs'], 
        num_users=config['num_users'],
        trajectory_file="user_trajectories_hot.json",
    )
    
    # 获取奖励权重用于命名
    weights = env.reward_system.weights
    w_delay = weights.get('w_delay', 0.0)
    w_task_energy = weights.get('w_task_energy', 0.0)
    w_move_energy = weights.get('w_move_energy', 0.0)
    
    run_id = time.strftime('%Y%m%d_%H%M%S')  # 本次训练唯一标识
    # 创建模型保存目录，加上权重信息
    model_save_dir = f'saved_models/run_{run_id}_wdelay{w_delay}_wtask{w_task_energy}_wmove{w_move_energy}'
    os.makedirs(model_save_dir, exist_ok=True)
    # 创建智能体SSS
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
    fieldnames = ['episode', 'reward', 'avg_delay', 'total_energy', 'steps']
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
        episode_total_energy = 0
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
                add_noise=True
            )
            
            # 使用 epsilon-greedy 策略进行用户分配探索
            # epsilon 随 agent.noise_scale 衰减
            #epsilon_allocation = 0.3 * agent.noise_scale 
            epsilon_allocation = 0
            allocation_hard = convert_soft_to_hard_allocation(
                allocation, env.num_uavs, env.num_users, epsilon=epsilon_allocation
            )

            # 构建环境动作格式
            actions = {}
            
            # offloading is now [N_Users] tensor
            # We need to pass this to the environment in a way that it understands
            # The environment expects per-UAV actions, but the offloading ratio is per-user
            # We will duplicate the offloading ratio for all UAVs or modify environment to handle global user actions
            # For compatibility with current environment structure (which iterates over UAVs), 
            # we can pass the full offloading vector to each UAV, and let the environment pick the right one based on user ID.
            
            offloading_np = offloading.cpu().numpy() # [N_Users]
            
            for uav_id in range(env.num_uavs):
                # 在 step 函数或 train_ddpg.py 的循环中
                vx = motion[uav_id, 0].item() # 范围 [-1, 1]
                vy = motion[uav_id, 1].item() # 范围 [-1, 1]

                # 计算新位置 (直接在笛卡尔坐标系更新)
                # agent.max_distance 在这里理解为 "最大单步移动距离" (例如 30m)
                dx = vx * config['max_distance']
                dy = vy * config['max_distance']
                
                actions[f'uav_{uav_id}'] = {
                    'user_competition_probs': allocation_hard[uav_id].cpu().numpy(),
                    'offloading_ratios': offloading_np, # Pass the full vector, calculator will handle indexing
                    'move_vector': (dx, dy) # 传递向量而不是角度/距离
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
            episode_total_energy += reward_comp['total_energy']
            
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
        if episode%20 ==0:
            save_uav_trajectories_to_json(
                episode=episode,
                trajectories=trajectories,
                initial_positions=initial_positions,
                save_dir='uav_trajectories',
                run_id=run_id
            )
            # 保存模型
            model_path = os.path.join(model_save_dir, f'episode_{episode}_model.pth')
            agent.save(model_path)

        env.reward_system.on_episode_end(episode_reward)
        
        # 写入CSV
        current_weights = env.reward_system.weights 
        row_data = {
            'episode': episode,
            'reward': round(episode_reward, 4),
            'avg_delay': round(episode_avg_delay, 6),
            'total_energy': round(episode_total_energy, 2),
            'steps': episode_steps
        }

        with open(csv_file, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writerow(row_data)
        
        # 打印进度
        if episode % 10 == 0:
            print(f"\nEpisode {episode}:")
            print(f"  Reward: {episode_reward:.2f}")
            print(f"  Avg Delay: {episode_avg_delay:.4f}")
            print(f"  Total Energy: {episode_total_energy:.2f}")
            
            diagnostics = env.reward_system.get_diagnostics()
            if diagnostics and 'variance_ratios' in diagnostics:
                print(f"  Variance Ratios: delay/energy={diagnostics['variance_ratios']['delay_vs_energy']:.2f}")

        
    
    print(f"\n训练完成！数据已保存到 {csv_file}")

def main():
    """主函数"""
    config = {
        # 环境参数
        'num_uavs': 2,
        'num_users': 5,
        
        # 训练参数
        'max_episodes': 500,
        'max_steps_per_episode': 40,
        'batch_size': 256,
        
        # 网络参数
        'lr_actor': 1e-5,
        'lr_critic': 1e-4,
        'gamma': 0.99,
        'tau': 0.001,
        'max_distance': 30,
    }
    
    print("开始DDPG多无人机训练...")
    print(f"配置: {json.dumps(config, indent=2, ensure_ascii=False)}")
    np.random.seed(42)      
    torch.manual_seed(42)
    train_ddpg(config)


if __name__ == "__main__":
    main()
