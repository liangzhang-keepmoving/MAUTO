# ddpg_inference_and_collect.py
"""
DDPG模型推理和数据收集
加载训练好的DDPG模型，与环境交互，收集决策数据并转换为LLM可理解的格式
"""
import os
import torch
import numpy as np
import json
from datetime import datetime
from DDPG import DDPGAgent
from env_simplified import SimplifiedMultiUAVEnvironment


class DDPGInferenceCollector:
    """DDPG推理数据收集器"""
    
    def __init__(self, model_path, env_config, save_dir='llm_training_data'):
        """
        初始化
        
        Args:
            model_path: DDPG模型路径
            env_config: 环境配置
            save_dir: 数据保存目录
        """
        self.model_path = model_path
        self.env_config = env_config
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # 创建环境
        self.env = SimplifiedMultiUAVEnvironment(
            num_uavs=env_config['num_uavs'],
            num_users=env_config['num_users'],
            trajectory_file=env_config.get('trajectory_file', 'user_trajectories_hot.json')
        )
        
        # 创建并加载DDPG智能体
        self.agent = DDPGAgent(
            n_uavs=env_config['num_uavs'],
            n_users=env_config['num_users'],
            lr_actor=1e-4,  # 推理时学习率无关紧要
            lr_critic=1e-3,
            gamma=0.99,
            tau=0.005,
            max_distance=30
        )
        
        # 加载模型
        self.agent.load(model_path)
        print(f"✓ 已加载DDPG模型: {model_path}")
        
        # 数据收集缓冲区
        self.collected_samples = []
        
    def collect_episodes(self, num_episodes=10, max_steps_per_episode=40):
        """
        收集多个episode的数据
        
        Args:
            num_episodes: 收集的episode数量
            max_steps_per_episode: 每个episode的最大步数
        """
        print(f"\n开始收集数据: {num_episodes}个episodes...")
        print("="*70)
        
        for episode in range(num_episodes):
            print(f"\nEpisode {episode+1}/{num_episodes}")
            episode_data = self._collect_single_episode(episode, max_steps_per_episode)
            self.collected_samples.extend(episode_data)
            
            print(f"  收集了 {len(episode_data)} 个样本")
        
        print(f"\n✓ 总共收集 {len(self.collected_samples)} 个样本")
        return self.collected_samples
    
    def _collect_single_episode(self, episode_num, max_steps):
        """收集单个episode的数据"""
        state = self.env.reset()
        episode_samples = []
        
        for step in range(max_steps):
            # DDPG决策（不添加噪声，使用确定性策略）
            allocation, offloading, motion = self.agent.select_action(
                state, 
                add_noise=False,  # 推理时不添加噪声
                hard=False
            )
            
            # 转换为硬分配
            allocation_hard = self._convert_soft_to_hard(allocation)
            
            # 构建环境动作
            actions = self._build_env_actions(allocation_hard, offloading, motion)
            
            # 环境步进
            next_state, reward, done, info = self.env.step(actions)
            
            # 收集样本（转换为LLM可理解的格式）
            sample = self._create_llm_sample(
                state, allocation_hard, offloading, motion, 
                actions, reward, info, episode_num, step
            )
            episode_samples.append(sample)
            
            # 打印进度
            if step % 10 == 0:
                print(f"    步骤 {step}: reward={reward:.4f}, "
                      f"delay={info['reward_components']['avg_delay']:.4f}s")
            
            state = next_state
            if done:
                break
        
        return episode_samples
    
    def _convert_soft_to_hard(self, allocation_soft):
        """将软分配转换为硬分配"""
        allocation_hard = torch.zeros_like(allocation_soft)
        for user_id in range(self.env.num_users):
            user_probs = allocation_soft[:, user_id]
            best_uav = torch.argmax(user_probs)
            allocation_hard[best_uav, user_id] = 1.0
        return allocation_hard
    
    def _build_env_actions(self, allocation, offloading, motion):
        """构建环境动作格式"""
        actions = {}
        offloading_np = offloading.cpu().numpy()
        
        for uav_id in range(self.env.num_uavs):
            vx = motion[uav_id, 0].item()
            vy = motion[uav_id, 1].item()
            dx = vx * 20
            dy = vy * 20
            
            actions[f'uav_{uav_id}'] = {
                'user_competition_probs': allocation[uav_id].cpu().numpy(),
                'offloading_ratios': offloading_np,
                'move_vector': (dx, dy)
            }
        
        return actions
    
    def _create_llm_sample(self, state, allocation, offloading, motion, 
                          actions, reward, info, episode, step):
        """
        创建LLM可理解的样本
        
        将DDPG的数值决策转换为自然语言描述
        """
        # 提取状态信息（反归一化）
        uav_pos = state['uav_pos'].cpu().numpy()
        user_pos = state['user_pos'].cpu().numpy()
        user_tasks = state['user_tasks'].cpu().numpy()
        
        # UAV位置
        uav_positions = {}
        for i in range(self.env.num_uavs):
            uav_positions[i] = {
                'x': float(uav_pos[i][0] * self.env.area_length),
                'y': float(uav_pos[i][1] * self.env.area_width)
            }
        
        # 用户信息
        users = {}
        for i in range(self.env.num_users):
            users[i] = {
                'x': float(user_pos[i][0] * self.env.area_length),
                'y': float(user_pos[i][1] * self.env.area_width),
                'task_size': float(user_tasks[i][0] * self.env.max_task_size)
            }
        
        # 提取决策
        user_assignments = info['user_assignments']
        
        offloading_ratios = {}
        for user_id in range(self.env.num_users):
            offloading_ratios[user_id] = float(offloading[user_id].item())
        
        uav_movements = {}
        for uav_id in range(self.env.num_uavs):
            dx, dy = actions[f'uav_{uav_id}']['move_vector']
            uav_movements[uav_id] = {'dx': float(dx), 'dy': float(dy)}
        
        # 构建样本
        sample = {
            'metadata': {
                'episode': episode,
                'step': step,
                'timestamp': datetime.now().isoformat()
            },
            
            'state': {
                'uav_positions': uav_positions,
                'users': users
            },
            
            'ddpg_decision': {
                'user_assignments': {int(k): int(v) for k, v in user_assignments.items()},
                'offloading_ratios': {int(k): float(v) for k, v in offloading_ratios.items()},
                'uav_movements': {int(k): v for k, v in uav_movements.items()}
            },
            
            'outcome': {
                'reward': float(reward),
                'avg_delay': float(info['reward_components']['avg_delay']),
                'total_energy': float(info['reward_components']['total_energy'])
            }
        }
        
        return sample
    
    def save_json_data(self, filename='ddpg_expert_data.json'):
        """保存原始JSON数据"""
        filepath = os.path.join(self.save_dir, filename)
        
        output = {
            'metadata': {
                'model_path': self.model_path,
                'timestamp': datetime.now().isoformat(),
                'num_samples': len(self.collected_samples),
                'env_config': self.env_config
            },
            'samples': self.collected_samples
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        
        print(f"\n✓ JSON数据已保存: {filepath}")
        return filepath
    
    def save_llm_prompt_data(self, filename='ddpg_llm_training_prompts.json'):
        """
        保存转换为LLM训练格式的数据
        每个样本转换为 (input_prompt, expected_output) 对
        """
        filepath = os.path.join(self.save_dir, filename)
        
        llm_training_data = []
        
        for sample in self.collected_samples:
            # 生成输入提示词
            input_prompt = self._generate_input_prompt(sample)
            
            # 生成期望输出（DDPG的决策）
            expected_output = self._generate_expected_output(sample)
            
            llm_training_data.append({
                'input': input_prompt,
                'output': expected_output,
                'metadata': sample['metadata'],
                'outcome': sample['outcome']
            })
        
        output = {
            'metadata': {
                'model_path': self.model_path,
                'timestamp': datetime.now().isoformat(),
                'num_training_pairs': len(llm_training_data),
                'description': 'DDPG expert demonstrations for LLM training'
            },
            'training_data': llm_training_data
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        
        print(f"✓ LLM训练数据已保存: {filepath}")
        return filepath
    
    def _generate_input_prompt(self, sample):
        """生成LLM输入提示词（场景描述）"""
        state = sample['state']
        
        prompt = "**当前系统状态:**\n\n"
        
        # UAV位置
        prompt += "**UAV位置:**\n"
        for uav_id, pos in state['uav_positions'].items():
            prompt += f"- UAV_{uav_id}: ({pos['x']:.1f}m, {pos['y']:.1f}m)\n"
        
        prompt += "\n**用户信息:**\n"
        
        # 用户信息（包含到各UAV的距离）
        for user_id, user_info in state['users'].items():
            # 计算到各UAV的3D距离
            distances = []
            for uav_id, uav_pos in state['uav_positions'].items():
                d_2d = np.sqrt((user_info['x'] - uav_pos['x'])**2 + 
                             (user_info['y'] - uav_pos['y'])**2)
                d_3d = np.sqrt(d_2d**2 + self.env.uav_height**2)
                distances.append((uav_id, d_3d))
            
            dist_str = ", ".join([f"到UAV_{uid}: {d:.1f}m" for uid, d in distances])
            prompt += f"- 用户_{user_id}: 位置({user_info['x']:.1f}m, {user_info['y']:.1f}m), "
            prompt += f"任务{user_info['task_size']:.2f}MB, {dist_str}\n"
        
        # 注意: Few-shot示例中不包含决策请求指令
        # 只在测试时（test_llm_agent）才需要请求LLM做出决策
        
        return prompt
    
    def _generate_expected_output(self, sample):
        """生成期望输出（DDPG的决策，JSON格式）"""
        decision = sample['ddpg_decision']
        
        output = {
            "user_assignments": {str(k): int(v) for k, v in decision['user_assignments'].items()},
            "offloading_ratios": {str(k): round(float(v), 3) for k, v in decision['offloading_ratios'].items()},
            "uav_movements": {
                str(k): {
                    "dx": round(float(v['dx']), 2),
                    "dy": round(float(v['dy']), 2)
                }
                for k, v in decision['uav_movements'].items()
            }
        }
        
        return json.dumps(output, ensure_ascii=False)
    
    def get_statistics(self):
        """获取数据统计"""
        if not self.collected_samples:
            return {}
        
        rewards = [s['outcome']['reward'] for s in self.collected_samples]
        delays = [s['outcome']['avg_delay'] for s in self.collected_samples]
        energies = [s['outcome']['total_energy'] for s in self.collected_samples]
        
        return {
            'num_samples': len(self.collected_samples),
            'reward': {
                'mean': np.mean(rewards),
                'std': np.std(rewards),
                'min': np.min(rewards),
                'max': np.max(rewards)
            },
            'avg_delay': {
                'mean': np.mean(delays),
                'std': np.std(delays),
                'min': np.min(delays),
                'max': np.max(delays)
            },
            'total_energy': {
                'mean': np.mean(energies),
                'std': np.std(energies),
                'min': np.min(energies),
                'max': np.max(energies)
            }
        }


def main():
    """主函数"""
    # 配置
    MODEL_PATH = "/home/niuma008/zsz/1206/saved_models/run_20251207_131732_wdelay0.4_wenergy0.6/episode_500_model.pth"
    
    ENV_CONFIG = {
        'num_uavs': 2,
        'num_users': 5,
        'trajectory_file': 'user_trajectories_hot.json'
    }
    
    NUM_EPISODES = 20  # 收集20个episodes的数据
    MAX_STEPS = 40
    
    print("="*70)
    print("DDPG模型推理和数据收集")
    print("="*70)
    print(f"模型路径: {MODEL_PATH}")
    print(f"环境配置: {ENV_CONFIG}")
    print(f"收集episodes: {NUM_EPISODES}")
    print("="*70)
    
    # 创建收集器
    collector = DDPGInferenceCollector(
        model_path=MODEL_PATH,
        env_config=ENV_CONFIG,
        save_dir='llm_training_data'
    )
    
    # 收集数据
    collector.collect_episodes(
        num_episodes=NUM_EPISODES,
        max_steps_per_episode=MAX_STEPS
    )
    
    # 保存数据
    collector.save_json_data('ddpg_expert_data.json')
    collector.save_llm_prompt_data('ddpg_llm_training_prompts.json')
    
    # 打印统计
    stats = collector.get_statistics()
    print("\n" + "="*70)
    print("数据统计:")
    print("="*70)
    print(f"总样本数: {stats['num_samples']}")
    print(f"\n奖励统计:")
    print(f"  平均: {stats['reward']['mean']:.4f}")
    print(f"  标准差: {stats['reward']['std']:.4f}")
    print(f"  范围: [{stats['reward']['min']:.4f}, {stats['reward']['max']:.4f}]")
    print(f"\n平均时延:")
    print(f"  平均: {stats['avg_delay']['mean']:.4f}s")
    print(f"  范围: [{stats['avg_delay']['min']:.4f}, {stats['avg_delay']['max']:.4f}]s")
    print(f"\n总能耗:")
    print(f"  平均: {stats['total_energy']['mean']:.2f}J")
    print(f"  范围: [{stats['total_energy']['min']:.2f}, {stats['total_energy']['max']:.2f}]J")
    print("="*70)
    print("\n✓ 数据收集完成！")
    print("  生成的文件:")
    print("  1. llm_training_data/ddpg_expert_data.json - 原始数据")
    print("  2. llm_training_data/ddpg_llm_training_prompts.json - LLM训练格式")


if __name__ == "__main__":
    main()