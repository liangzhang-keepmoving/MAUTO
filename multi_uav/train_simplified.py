#!/usr/bin/env python3
"""
简化版MADDPG训练脚本
专注于能耗优化：UAV能耗 + 用户能耗
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import json
from datetime import datetime
from env_simplified import SimplifiedMultiUAVEnvironment
from maddpg_trainer_simplified import SimplifiedMADDPGTrainer

def train():
    # 创建简化环境
    env = SimplifiedMultiUAVEnvironment(num_uavs=3, num_users=6)
    
    # 创建简化训练器
    trainer = SimplifiedMADDPGTrainer(env, lr_actor=2e-4, lr_critic=2e-3)  # 稍微提高学习率
    
    # 训练参数
    num_episodes = 5000  # 先用较少episode测试
    save_interval = 200
    eval_interval = 100
    
    # 记录训练过程
    episode_rewards = []
    episode_lengths = []
    episode_reward_details = []
    
    # 创建保存目录
    os.makedirs('models_simplified', exist_ok=True)
    os.makedirs('logs_simplified', exist_ok=True)
    
    # 创建训练记录文件
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    reward_log_file = f'logs_simplified/training_rewards_{timestamp}.json'
    csv_log_file = f'logs_simplified/training_rewards_{timestamp}.csv'
    detailed_rewards_file = f'logs_simplified/detailed_rewards_{timestamp}.json'
    detailed_rewards_csv = f'logs_simplified/detailed_rewards_{timestamp}.csv'
    
    
    for episode in range(num_episodes):

        # 训练一个episode
        total_reward, episode_length, reward_breakdown = trainer.train_episode()
        
        episode_rewards.append(total_reward)
        episode_lengths.append(episode_length)
        episode_reward_details.append(reward_breakdown)
        
        # 计算平均奖励
        avg_reward = np.mean(episode_rewards[-10:]) if len(episode_rewards) >= 10 else np.mean(episode_rewards)
        avg_length = np.mean(episode_lengths[-10:]) if len(episode_lengths) >= 10 else np.mean(episode_lengths)
        
        # 详细输出（每个episode）
        print(f"Episode {episode:3d} | Reward: {total_reward:8.2f} | Avg-10: {avg_reward:8.2f} | Length: {episode_length:3d} | Noise: {trainer.noise_scale:.4f}")
        print(f"  📊 奖励分解 (用户视角归一化: 能耗累加 + 时延):")
        print(f"    🔋 归一化能耗惩罚:    {reward_breakdown['cumulative_total_energy_penalty']:8.2f}")
        print(f"    ⏱️  归一化时延惩罚:    {reward_breakdown['cumulative_delay_penalty']:8.2f}")
        print(f"    🔋 实际UAV能耗相加:    {reward_breakdown['cumulative_actual_total_energy_raw']:8.2f}")
        print(f"    ⏱️  实际UAV时延相加:    {reward_breakdown['cumulative_actual_uav_delay_raw']:8.2f}")
        print(f"    ➕ 累计总奖励:        {reward_breakdown['cumulative_total_reward']:8.2f}")
        
        # 验证一致性
        manual_total = (reward_breakdown['cumulative_total_energy_penalty'] + 
                       reward_breakdown['cumulative_delay_penalty'])
        
        if abs(manual_total - reward_breakdown['cumulative_total_reward']) > 0.01:
            print(f"    ⚠️  一致性检查: 手动={manual_total:.2f} vs 记录={reward_breakdown['cumulative_total_reward']:.2f}")
        
        print()  # 空行
        
        # 保存当前episode数据
        episode_data = {
            'episode': episode,
            'reward': float(total_reward),
            'actual_total_energy_raw': float(reward_breakdown['cumulative_actual_total_energy_raw']),
            'actual_uav_delay_raw': float(reward_breakdown['cumulative_actual_uav_delay_raw']),
            'length': int(episode_length),
            'avg_10_reward': float(avg_reward),
            'noise_scale': float(trainer.noise_scale),
            'timestamp': datetime.now().isoformat()
        }
        
        # 保存详细奖励分解数据
        detailed_episode_data = {
            'episode': episode,
            'step_count': reward_breakdown['step_count'],
            'cumulative_total_energy_penalty': float(reward_breakdown['cumulative_total_energy_penalty']),
            'cumulative_delay_penalty': float(reward_breakdown['cumulative_delay_penalty']),
            'cumulative_total_reward': float(reward_breakdown['cumulative_total_reward']),
            'cumulative_actual_total_energy_raw': float(reward_breakdown['cumulative_actual_total_energy_raw']),
            'cumulative_actual_uav_delay_raw': float(reward_breakdown['cumulative_actual_uav_delay_raw']),
            'total_reward_check': float(total_reward),
            'timestamp': datetime.now().isoformat()
        }
        
        # 实时保存到JSON文件
        with open(reward_log_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(episode_data, ensure_ascii=False) + '\n')
        
        with open(detailed_rewards_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(detailed_episode_data, ensure_ascii=False) + '\n')
        
        # 实时保存到CSV文件
        if episode == 0:
            # 基本CSV头部
            with open(csv_log_file, 'w', encoding='utf-8') as f:
                f.write('episode,reward,length,avg_10_reward,noise_scale,timestamp\n')
            
            # 详细奖励CSV头部
            with open(detailed_rewards_csv, 'w', encoding='utf-8') as f:
                f.write('episode,step_count,total_energy_penalty,delay_penalty,total_reward,timestamp\n')
        
        with open(csv_log_file, 'a', encoding='utf-8') as f:
            f.write(f"{episode},{total_reward:.2f},{episode_length},{avg_reward:.2f},{trainer.noise_scale:.4f},{datetime.now().isoformat()}\n")
        
        with open(detailed_rewards_csv, 'a', encoding='utf-8') as f:
            f.write(f"{episode},{reward_breakdown['step_count']},{reward_breakdown['cumulative_total_energy_penalty']:.2f},{reward_breakdown['cumulative_delay_penalty']:.2f},{reward_breakdown['cumulative_total_reward']:.2f},{datetime.now().isoformat()}\n")
        
        # 保存模型
        if episode % save_interval == 0 and episode > 0:
            trainer.save_models(f'models_simplified/simplified_maddpg_episode_{episode}.pth')
            print(f"💾 模型已保存: episode {episode}")
        
        # 评估模型
        if episode % eval_interval == 0 and episode > 0:
            eval_reward = evaluate_model(trainer, env, num_eval_episodes=3)
            print(f"📊 评估结果 (Episode {episode}): 平均奖励 = {eval_reward:.2f}")
    
    # 保存最终模型
    trainer.save_models('models_simplified/simplified_maddpg_final.pth')
    
    # 保存完整的训练数据汇总
    training_summary = {
        'training_config': {
            'model_type': 'Simplified_MADDPG',
            'reward_components': ['UAV_energy_penalty', 'user_energy_penalty'],
            'num_episodes': num_episodes,
            'num_uavs': env.num_uavs,
            'num_users': env.num_users,
            'target_episode_steps': env.target_episode_steps,
            'total_target_task_size': env.total_target_task_size,
            'lr_actor': 2e-4,
            'lr_critic': 2e-3,
            'buffer_capacity': 50000,
            'update_frequency': trainer.update_frequency,
            'training_date': timestamp
        },
        'final_stats': {
            'final_avg_reward': float(np.mean(episode_rewards[-10:]) if len(episode_rewards) >= 10 else np.mean(episode_rewards)),
            'best_reward': float(max(episode_rewards)),
            'worst_reward': float(min(episode_rewards)),
            'total_episodes': len(episode_rewards),
            'final_noise_scale': float(trainer.noise_scale),
            'avg_episode_length': float(np.mean(episode_lengths))
        },
        'all_episode_rewards': [float(r) for r in episode_rewards],
        'all_episode_lengths': [int(l) for l in episode_lengths]
    }
    
    summary_file = f'logs_simplified/training_summary_{timestamp}.json'
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(training_summary, f, indent=2, ensure_ascii=False)
    
    # 绘制训练曲线
    
    print("🎉 训练完成!")
    print(f"📁 训练数据已保存到:")
    print(f"  - 实时记录(JSON): {reward_log_file}")
    print(f"  - 实时记录(CSV): {csv_log_file}")
    print(f"  - 详细奖励(JSON): {detailed_rewards_file}")
    print(f"  - 详细奖励(CSV): {detailed_rewards_csv}")
    print(f"  - 训练汇总: {summary_file}")
    print(f"  - 训练曲线: logs_simplified/training_curves_{timestamp}.png")
    
    # 显示最终结果
    print(f"\n📊 最终训练结果:")
    print(f"  🏆 最佳奖励: {max(episode_rewards):.2f}")
    print(f"  📉 最差奖励: {min(episode_rewards):.2f}")
    print(f"  📈 最终平均奖励: {training_summary['final_stats']['final_avg_reward']:.2f}")
    print(f"  ⏱️  平均episode长度: {training_summary['final_stats']['avg_episode_length']:.1f}步")

def evaluate_model(trainer, env, num_eval_episodes=5):
    """评估模型性能"""
    eval_rewards = []
    
    for _ in range(num_eval_episodes):
        obs = env.reset()
        total_reward = 0
        
        step = 0
        max_eval_steps = 200
        while step < max_eval_steps:
            # 不使用探索噪声
            actions, action_details = trainer.select_actions(obs, training=False)
            next_obs, rewards, done, info = env.step(action_details)
            
            total_reward += np.mean(list(rewards.values())) if rewards else 0
            obs = next_obs
            step += 1
            
            if done:
                break
        
        eval_rewards.append(total_reward)
    
    return np.mean(eval_rewards)



if __name__ == "__main__":
    # 测试环境
    # test_environment()
    
    print("\n" + "="*60)
    
    # 开始训练
    train()
