"""
train_ddpg_save_llm_data.py - 训练DDPG的同时保存LLM训练数据
修改自你的train_ddpg.py，增加了完整的数据保存功能
"""

import os
import time
import numpy as np
import torch
import json
from datetime import datetime
from collections import deque

from DDPG import DDPGAgent
from env_train import TrainingMultiUAVEnvironment


class LLMDataCollector:
    """LLM训练数据收集器"""
    
    def __init__(self, save_dir='llm_training_data', buffer_size=10000):
        """
        Args:
            save_dir: 数据保存目录
            buffer_size: 内存缓冲区大小（达到后写入文件）
        """
        self.save_dir = save_dir
        self.buffer_size = buffer_size
        self.buffer = []
        
        # 创建保存目录
        os.makedirs(save_dir, exist_ok=True)
        
        # 统计信息
        self.total_samples = 0
        self.episodes_saved = []
        
        print(f"✓ LLM数据收集器初始化: {save_dir}")
    
    @staticmethod
    def _convert_to_native_types(obj):
        """
        递归转换为Python原生类型,确保JSON可序列化
        """
        if isinstance(obj, dict):
            return {k: LLMDataCollector._convert_to_native_types(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [LLMDataCollector._convert_to_native_types(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, torch.Tensor):
            return obj.cpu().numpy().tolist()
        elif isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
            return float(obj)
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        else:
            return obj
    def collect_step(self, episode, step, state, actions, reward, next_state, done, info):
        """
        收集单个时间步的数据
        
        Args:
            episode: 当前episode编号
            step: 当前step编号
            state: 当前状态 (dict with 'uav_pos', 'user_pos', 'user_tasks')
            actions: 动作字典 (原始环境格式)
            reward: 奖励值
            next_state: 下一状态
            done: 是否结束
            info: 环境返回的额外信息
        """
        # 转换为LLM友好的格式
        sample = {
            'episode': int(episode),
            'step': int(step),
            'timestamp': datetime.now().isoformat(),
            
            # === 状态信息 ===
            'state': {
                'uav_pos': self._tensor_to_list(state['uav_pos']),      # [N, 2]
                'user_pos': self._tensor_to_list(state['user_pos']),    # [M, 2]
                'user_tasks': self._tensor_to_list(state['user_tasks']) # [M, 1]
            },
            
            # === 动作信息 ===
            'action': self._format_actions(actions),
            
            # === 奖励和结果 ===
            'reward': float(reward),
            'done': bool(done),
            
            # === 下一状态 ===
            'next_state': {
                'uav_pos': self._tensor_to_list(next_state['uav_pos']),
                'user_pos': self._tensor_to_list(next_state['user_pos']),
                'user_tasks': self._tensor_to_list(next_state['user_tasks'])
            },
            
            # === 额外信息（可选，用于分析）===
            'info': {
                'reward_components': info.get('reward_components', {}),
                'user_assignments': info.get('user_assignments', {}),
            }
        }
        
        # 添加到缓冲区
        self.buffer.append(sample)
        self.total_samples += 1
        
        # 达到缓冲区大小时写入文件
        if len(self.buffer) >= self.buffer_size:
            self._flush_buffer()
    
    def _tensor_to_list(self, tensor):
        """将Tensor转为嵌套列表"""
        if isinstance(tensor, torch.Tensor):
            return tensor.cpu().numpy().tolist()
        elif isinstance(tensor, np.ndarray):
            return tensor.tolist()
        else:
            return tensor
    
    def _format_actions(self, actions):
        """格式化动作为LLM友好格式"""
        formatted = {}
        
        for uav_id, action in actions.items():
            formatted[uav_id] = {
                'user_assignments': self._tensor_to_list(action['user_competition_probs']),
                'offloading_ratios': self._tensor_to_list(action['offloading_ratios']),
                'movement_direction': float(action['movement_direction']),
                'movement_distance': float(action['movement_distance'])
            }
        
        return formatted
    
    
    def _flush_buffer(self):
        """将缓冲区数据写入文件"""
        if not self.buffer:
            return
        
        filename = f'samples_{datetime.now().strftime("%Y%m%d_%H%M%S")}.jsonl'
        filepath = os.path.join(self.save_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            for sample in self.buffer:
                # 转换所有数据为Python原生类型
                clean_sample = self._convert_to_native_types(sample)
                f.write(json.dumps(clean_sample, ensure_ascii=False) + '\n')
        
        print(f"  💾 已保存 {len(self.buffer)} 个样本到 {filename}")
        self.buffer.clear()
    def on_episode_end(self, episode, episode_reward, episode_samples):
        """
        Episode结束时的处理
        
        Args:
            episode: episode编号
            episode_reward: 总奖励
            episode_samples: 本episode的所有样本
        """
        self.episodes_saved.append({
            'episode': episode,
            'reward': episode_reward,
            'num_samples': len(episode_samples)
        })
        
        # 强制写入当前缓冲区（确保episode数据完整）
        if self.buffer:
            self._flush_buffer()
    
    def finalize(self):
        """训练结束时的最终处理"""
        # 1. 写入剩余缓冲区数据
        if self.buffer:
            self._flush_buffer()
        
        # 2. 合并所有文件为单个文件
        print("\n正在合并所有训练数据...")
        self._merge_all_samples()
        
        # 3. 选择最佳样本
        print("\n正在选择最佳样本...")
        self._select_best_samples()
        
        # 4. 生成元数据
        self._save_metadata()
        
        print(f"\n✅ LLM训练数据收集完成！")
        print(f"   总样本数: {self.total_samples}")
        print(f"   保存目录: {self.save_dir}")
    
    def _merge_all_samples(self):
        """合并所有样本文件"""
        all_samples = []
        
        # 读取所有jsonl文件
        for filename in os.listdir(self.save_dir):
            if filename.startswith('samples_') and filename.endswith('.jsonl'):
                filepath = os.path.join(self.save_dir, filename)
                with open(filepath, 'r', encoding='utf-8') as f:
                    for line in f:
                        all_samples.append(json.loads(line))
        
        # 写入合并文件
        merged_file = os.path.join(self.save_dir, 'all_samples.jsonl')
        with open(merged_file, 'w', encoding='utf-8') as f:
            for sample in all_samples:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')
        
        print(f"  ✓ 合并完成: {len(all_samples)} 个样本 -> all_samples.jsonl")
    
    def _select_best_samples(self, top_k=20, percentile=90):
        """
        选择最佳样本
        
        Args:
            top_k: 保留前K个最佳episode
            percentile: 保留奖励前X%的样本
        """
        # 读取所有样本
        all_samples = []
        merged_file = os.path.join(self.save_dir, 'all_samples.jsonl')
        
        with open(merged_file, 'r', encoding='utf-8') as f:
            for line in f:
                all_samples.append(json.loads(line))
        
        if not all_samples:
            return
        
        # 方法1: 按episode总奖励选择
        episode_rewards = {}
        episode_samples = {}
        
        for sample in all_samples:
            ep = sample['episode']
            if ep not in episode_rewards:
                episode_rewards[ep] = 0
                episode_samples[ep] = []
            episode_rewards[ep] += sample['reward']
            episode_samples[ep].append(sample)
        
        # 选择奖励最高的top_k个episode
        sorted_episodes = sorted(episode_rewards.items(), 
                                key=lambda x: x[1], reverse=True)
        top_episodes = [ep for ep, _ in sorted_episodes[:top_k]]
        
        best_by_episode = []
        for ep in top_episodes:
            best_by_episode.extend(episode_samples[ep])
        
        # 保存
        output_file = os.path.join(self.save_dir, f'best_episodes_top{top_k}.jsonl')
        with open(output_file, 'w', encoding='utf-8') as f:
            for sample in best_by_episode:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')
        
        print(f"  ✓ 最佳episode: {len(best_by_episode)} 个样本 -> best_episodes_top{top_k}.jsonl")
        
        # 方法2: 按单步奖励选择
        all_rewards = [s['reward'] for s in all_samples]
        threshold = np.percentile(all_rewards, percentile)
        
        best_by_reward = [s for s in all_samples if s['reward'] >= threshold]
        
        output_file = os.path.join(self.save_dir, f'best_samples_p{percentile}.jsonl')
        with open(output_file, 'w', encoding='utf-8') as f:
            for sample in best_by_reward:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')
        
        print(f"  ✓ 高奖励样本: {len(best_by_reward)} 个样本 -> best_samples_p{percentile}.jsonl")
    
    def _save_metadata(self):
        """保存元数据"""
        metadata = {
            'collection_date': datetime.now().isoformat(),
            'total_samples': self.total_samples,
            'total_episodes': len(self.episodes_saved),
            'episode_summary': self.episodes_saved,
            'reward_statistics': self._compute_reward_stats()
        }
        
        output_file = os.path.join(self.save_dir, 'metadata.json')
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        print(f"  ✓ 元数据已保存 -> metadata.json")
    
    def _compute_reward_stats(self):    
        """计算奖励统计"""
        if not self.episodes_saved:
            return {}
        
        rewards = [ep['reward'] for ep in self.episodes_saved]
        return {
            'mean': float(np.mean(rewards)),
            'std': float(np.std(rewards)),
            'min': float(np.min(rewards)),
            'max': float(np.max(rewards)),
            'median': float(np.median(rewards))
        }


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


def train_ddpg_with_llm_data_collection(config):
    """
    训练DDPG并同时收集LLM训练数据
    """
    run_id = time.strftime('%Y%m%d_%H%M%S')
    
    print("="*60)
    print("DDPG Training with LLM Data Collection")
    print("="*60)
    print(f"Run ID: {run_id}")
    print(f"Config: {json.dumps(config, indent=2)}\n")
    
    # 创建环境
    env = TrainingMultiUAVEnvironment(
        num_uavs=config['num_uavs'], 
        num_users=config['num_users'],
        trajectory_file=config.get('trajectory_file', "user_trajectories.json"),
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
    
    # === 创建LLM数据收集器 ===
    llm_collector = LLMDataCollector(
        save_dir=config.get('llm_data_dir', 'llm_training_data'),
        buffer_size=config.get('buffer_size', 1000)
    )
    
    # 创建CSV记录
    import csv
    csv_file = 'training_history_collect_data.csv'
    fieldnames = ['episode', 'reward', 'avg_delay_sum', 'max_delay_sum', 
                  'task_energy', 'move_energy', 'steps']
    
    with open(csv_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
    
    # 训练循环
    for episode in range(config['max_episodes']):
        state = env.reset()
        episode_reward = 0
        episode_steps = 0
        
        # 用于存储本episode的样本
        episode_samples = []
        
        # 累加指标
        episode_avg_delay = 0
        episode_max_delay = 0
        episode_task_energy = 0
        episode_move_energy = 0
        
        # 重置噪声
        agent.reset_noise()
        
        for step in range(config['max_steps_per_episode']):
            # 选择动作
            allocation, offloading, motion = agent.select_action(
                state, 
                add_noise=True,
                hard=False
            )
            allocation_hard = convert_soft_to_hard_allocation(
                allocation, env.num_uavs, env.num_users
            )
            
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
            next_state, reward, done, info = env.step(actions)
            
            # === 收集LLM训练数据 ===
            llm_collector.collect_step(
                episode=episode,
                step=step,
                state=state,
                actions=actions,
                reward=reward,
                next_state=next_state,
                done=done,
                info=info
            )
            episode_samples.append((state, actions, reward))
            
            # 累加指标
            reward_comp = info['reward_components']
            episode_avg_delay += reward_comp['avg_delay']
            episode_max_delay += reward_comp['max_delay']
            episode_task_energy += reward_comp['total_task_energy']
            episode_move_energy += reward_comp['total_move_energy']
            
            episode_reward += reward
            episode_steps += 1
            
            # 存储经验到DDPG回放缓冲区
            action_tensors = (
                allocation.detach(),
                offloading.detach(), 
                motion.detach()
            )
            agent.replay_buffer.push(
                state, action_tensors, reward, next_state, done
            )
            
            # 训练DDPG
            if len(agent.replay_buffer) >= config['batch_size']:
                actor_loss, critic_loss = agent.train_step(config['batch_size'])
            
            state = next_state
            
            if done:
                break
        
        # Episode结束处理
        agent.decay_noise()
        
        # 通知LLM收集器episode结束
        llm_collector.on_episode_end(episode, episode_reward, episode_samples)
        
        # 记录到CSV
        row_data = {
            'episode': episode,
            'reward': round(episode_reward, 4),
            'avg_delay_sum': round(episode_avg_delay / episode_steps, 6),
            'max_delay_sum': round(episode_max_delay, 6),
            'task_energy': round(episode_task_energy, 2),
            'move_energy': round(episode_move_energy, 2),
            'steps': episode_steps
        }
        
        with open(csv_file, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writerow(row_data)
        
        # 打印进度
        if episode % 10 == 0:
            print(f"\nEpisode {episode}:")
            print(f"  Reward: {episode_reward:.2f}")
            print(f"  Avg Delay: {episode_avg_delay/episode_steps:.4f}")
            print(f"  Steps: {episode_steps}")
            print(f"  LLM Samples: {llm_collector.total_samples}")
    
    # 训练结束
    print("\n" + "="*60)
    print("Training Completed!")
    print("="*60)
    
    # === 最终化LLM数据 ===
    print("\n处理LLM训练数据...")
    llm_collector.finalize()
    
    print(f"\n训练完成！")
    print(f"  DDPG训练历史: {csv_file}")
    print(f"  LLM训练数据: {llm_collector.save_dir}/")


def main():
    """主函数"""
    config = {
        # 环境参数
        'num_uavs': 2,
        'num_users': 6,
        'trajectory_file': 'user_trajectories.json',
        
        # 训练参数
        'max_episodes': 100,  # 先测试100个episode
        'max_steps_per_episode': 40,
        'batch_size': 64,
        
        # DDPG参数
        'lr_actor': 1e-4,
        'lr_critic': 1e-3,
        'gamma': 0.99,
        'tau': 0.005,
        'max_distance': 30,
        
        # === LLM数据收集参数 ===
        'llm_data_dir': 'llm_training_data',  # 数据保存目录
        'buffer_size': 1000,  # 缓冲区大小（每1000个样本写入一次）
    }
    
    print("开始训练（带LLM数据收集）...")
    train_ddpg_with_llm_data_collection(config)


if __name__ == "__main__":
    main()
