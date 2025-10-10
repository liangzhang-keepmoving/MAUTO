#!/usr/bin/env python3
"""
完整的多无人机DDPG训练脚本
包含训练循环、监控、评估和可视化功能
"""

import os
import time
import numpy as np
import torch
import matplotlib.pyplot as plt
from collections import deque
import json
from datetime import datetime

# 导入自定义模块
from DDPG import DDPGAgent
from env_simplified import SimplifiedMultiUAVEnvironment

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
    
    # 创建监控器
    monitor = TrainingMonitor(config['log_dir'])
    
    print(f"环境: {config['num_uavs']}无人机, {config['num_users']}用户")
    print(f"训练episodes: {config['max_episodes']}")
    print(f"设备: {agent.device}")
    
    # 训练循环
    start_time = time.time()
    training_history = {
    'episode_rewards': [],
    'episode_task_energy': [],
    'episode_movement_energy': [],
    'episode_task_delays': [],
    'episode_movement_delays': []
    }
    
    for episode in range(config['max_episodes']):
        # 重置环境
        state = env.reset()
        episode_reward = 0
        episode_steps = 0
        episode_task_energy = 0
        episode_movement_energy = 0
        episode_task_delays = 0
        episode_movement_delays = 0
        
        # 重置噪声
        agent.noise.reset()
        
        for step in range(config['max_steps_per_episode']):
            # 选择动作
            allocation, offloading, motion = agent.select_action(
                state, 
                add_noise=True,
                hard=True  # 训练时使用软分配
            )
            
            # 构建环境动作格式
            actions = {}
            for uav_id in range(env.num_uavs):
                actions[f'uav_{uav_id}'] = {
                    'user_competition_probs': allocation[uav_id].cpu().numpy(),
                    'offloading_ratios': offloading[uav_id].cpu().numpy(),
                    'movement_direction': motion[uav_id, 1].item() * np.pi,  # 转换为弧度
                    'movement_distance': motion[uav_id, 0].item() * agent.max_distance
                }
            
            # 环境步进
            next_state, rewards, done, info = env.step(actions)
            if(episode%10==0):
                print(actions)

            raw_metrics = info.get('raw_metrics', {})
            episode_task_energy += sum([raw_metrics[f'uav_{i}']['actual_total_energy_raw'] for i in range(env.num_uavs)])
            episode_movement_energy += sum([raw_metrics[f'uav_{i}']['uav_movement_energy_raw'] for i in range(env.num_uavs)])
            episode_task_delays += sum([raw_metrics[f'uav_{i}']['delay_raw'] for i in range(env.num_uavs)])
            episode_movement_delays += sum([raw_metrics[f'uav_{i}']['movement_delay_raw'] for i in range(env.num_uavs)])
            
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
        
        training_history['episode_rewards'].append(episode_reward)
        training_history['episode_task_energy'].append(episode_task_energy)
        training_history['episode_movement_energy'].append(episode_movement_energy)
        training_history['episode_task_delays'].append(episode_task_delays)
        training_history['episode_movement_delays'].append(episode_movement_delays)
        print(f"第{episode}回合：奖励={episode_reward:.2f}, "
          f"任务能耗={episode_task_energy:.2f}, "
          f"移动能耗={episode_movement_energy:.2f}, "
          f"任务延迟={episode_task_delays:.4f}, "
          f"移动延迟={episode_movement_delays:.4f}")
        
    
    import json
    import pickle
    import pandas as pd
    with open('training_history.json', 'w') as f:
        json.dump(training_history, f, indent=4)
 


def main():
    """主函数"""
    # 训练配置
    config = {
        # 环境参数
        'num_uavs': 2,
        'num_users': 5,
        
        # 训练参数
        'max_episodes': 2000,
        'max_steps_per_episode': 40,
        'batch_size': 64,
        
        # 网络参数
        'lr_actor': 1e-4,
        'lr_critic': 1e-3,
        'gamma': 0.99,
        'tau': 0.005,
        'max_distance': 30,
        
        # 监控参数
        'print_interval': 50,
        'eval_interval': 200,
        'log_dir': f'./training_logs_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    }
    
    print("开始DDPG多无人机训练...")
    print(f"配置: {json.dumps(config, indent=2, ensure_ascii=False)}")
    
    # 开始训练
    train_ddpg(config)
    
    print("\n训练完成！")
    print(f"日志保存在: {config['log_dir']}")
    
    


if __name__ == "__main__":
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 开始训练
    main()
