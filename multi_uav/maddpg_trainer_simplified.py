import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from env_simplified import SimplifiedMultiUAVEnvironment
from collections import deque

class ActorNetwork(nn.Module):
    """简化的Actor网络"""
    def __init__(self, obs_size, num_users, max_speed=15.0, time_step=5.0, hidden_size=128):
        super(ActorNetwork, self).__init__()
        self.num_users = num_users
        # 更保守的最大移动距离
        self.max_distance = max_speed * time_step * 0.3  # 最大移动距离 = 15*5*0.3 = 22.5米
        
        # 共享特征提取层
        self.shared_layers = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
        
        # 用户竞争概率头
        self.user_competition_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size//2),
            nn.ReLU(),
            nn.Linear(hidden_size//2, num_users),
            nn.Sigmoid()
        )
        
        # 卸载比例头
        self.offloading_ratio_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size//2),
            nn.ReLU(),
            nn.Linear(hidden_size//2, num_users),
            nn.Sigmoid()
        )
        
        # 移动方向头
        self.movement_direction_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size//4),
            nn.ReLU(),
            nn.Linear(hidden_size//4, 1),
            nn.Sigmoid()
        )
        
        # 移动距离头
        self.movement_distance_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size//4),
            nn.ReLU(),
            nn.Linear(hidden_size//4, 1),
            nn.Sigmoid()
        )
        
    def forward(self, obs):
        """前向传播"""
        shared_features = self.shared_layers(obs)
        
        user_competition_probs = self.user_competition_head(shared_features)
        offloading_ratios = self.offloading_ratio_head(shared_features)
        movement_direction_normalized = self.movement_direction_head(shared_features)
        movement_distance_normalized = self.movement_distance_head(shared_features)
        
        return {
            'user_competition_probs': user_competition_probs,
            'offloading_ratios': offloading_ratios,
            'movement_direction_normalized': movement_direction_normalized,
            'movement_distance_normalized': movement_distance_normalized
        }
    
    def select_complete_action(self, obs, threshold=0.5, noise=0.1):
        """选择完整动作 - 支持批处理输入"""
        # 处理输入维度
        original_shape = obs.shape
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)  # [20] -> [1, 20]
            single_input = True
        else:
            single_input = False    # 已经是 [batch_size, 20]
        
        batch_size = obs.shape[0]
        
        # 前向传播
        action_outputs = self.forward(obs)
        
        # 用户竞争概率
        user_competition_probs = action_outputs['user_competition_probs']
        if noise > 0:
            noise_tensor = torch.normal(0, noise, size=user_competition_probs.shape)
            user_competition_probs = torch.clamp(user_competition_probs + noise_tensor, 0, 1)
        
        # 卸载比例
        offloading_ratios = action_outputs['offloading_ratios']
        if noise > 0:
            ratio_noise = torch.normal(0, noise * 0.1, size=offloading_ratios.shape)
            offloading_ratios = torch.clamp(offloading_ratios + ratio_noise, 0, 1)
        
        # 移动方向
        movement_direction_norm = action_outputs['movement_direction_normalized']
        if noise > 0:
            direction_noise = torch.normal(0, noise * 0.2, size=movement_direction_norm.shape)
            movement_direction_norm = torch.clamp(movement_direction_norm + direction_noise, 0, 1)
        
        movement_direction = movement_direction_norm * 2 * np.pi
        
        # 移动距离
        movement_distance_norm = action_outputs['movement_distance_normalized']
        if noise > 0:
            distance_noise = torch.normal(0, noise * 0.1, size=movement_distance_norm.shape)
            movement_distance_norm = torch.clamp(movement_distance_norm + distance_noise, 0, 1)
        
        movement_distance = movement_distance_norm * self.max_distance
        
        # 组合完整动作
        complete_action = {
            'user_competition_probs': user_competition_probs,
            'offloading_ratios': offloading_ratios,
            'movement_direction': movement_direction,
            'movement_distance': movement_distance
        }
        
        # 返回展平的动作向量
        flattened_action = torch.cat([
            user_competition_probs,
            offloading_ratios,
            movement_direction_norm,
            movement_distance_norm
        ], dim=1)  # [batch_size, 14]
        
        # 如果原始输入是单个样本，压缩输出
        if single_input:
            flattened_action = flattened_action.squeeze(0)  # [14]
            complete_action = {
                'user_competition_probs': user_competition_probs.squeeze(0),
                'offloading_ratios': offloading_ratios.squeeze(0),
                'movement_direction': movement_direction.squeeze(0),
                'movement_distance': movement_distance.squeeze(0)
            }
            user_competition_probs = user_competition_probs.squeeze(0)
        
        return flattened_action, complete_action, user_competition_probs

class CriticNetwork(nn.Module):
    """简化的Critic网络"""
    def __init__(self, total_obs_size, total_action_size, hidden_size=256):
        super(CriticNetwork, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(total_obs_size + total_action_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size//2),
            nn.ReLU(),
            nn.Linear(hidden_size//2, 1)
        )
        
    def forward(self, global_obs, global_actions):
        x = torch.cat([global_obs, global_actions], dim=-1)
        return self.network(x)

class ReplayBuffer:
    """经验回放缓冲区"""
    def __init__(self, capacity=50000):  # 减小buffer容量
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        return (torch.FloatTensor(state),
                torch.FloatTensor(action), 
                torch.FloatTensor(reward),
                torch.FloatTensor(next_state),
                torch.BoolTensor(done))
    
    def __len__(self):
        return len(self.buffer)

class SimplifiedMADDPGTrainer:
    """简化的MADDPG训练器 - 专注于能耗优化"""
    def __init__(self, env, lr_actor=1e-4, lr_critic=1e-3, gamma=0.99, tau=0.01):
        self.env = env
        self.num_uavs = env.num_uavs
        self.num_users = env.num_users
        self.gamma = gamma
        self.tau = tau
        
        # 获取观察空间大小
        self.obs_size = self._get_obs_size()
        self.total_obs_size = (self.num_uavs * 2) + (self.num_users * 3)
        self.action_size_per_uav = self.num_users * 2 + 2
        self.total_action_size = self.action_size_per_uav * self.num_uavs
        
        # 创建Actor网络
        self.actors = {}
        self.actor_targets = {}
        self.actor_optimizers = {}
        
        for i in range(self.num_uavs):
            self.actors[f'uav_{i}'] = ActorNetwork(self.obs_size, self.num_users, 
                                                 max_speed=env.uav_max_speed, 
                                                 time_step=env.time_step)
            self.actor_targets[f'uav_{i}'] = ActorNetwork(self.obs_size, self.num_users,
                                                        max_speed=env.uav_max_speed,
                                                        time_step=env.time_step)
            self.actor_optimizers[f'uav_{i}'] = optim.Adam(self.actors[f'uav_{i}'].parameters(), lr=lr_actor)
            
            # 初始化target网络
            self.actor_targets[f'uav_{i}'].load_state_dict(self.actors[f'uav_{i}'].state_dict())
        
        # 创建Critic网络
        self.critic = CriticNetwork(self.total_obs_size, self.total_action_size)
        self.critic_target = CriticNetwork(self.total_obs_size, self.total_action_size)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)
        
        # 初始化target网络
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        # 经验回放
        self.replay_buffer = ReplayBuffer()
        
        # 探索噪声（更慢的衰减）
        self.noise_scale = 0.2  # 增加初始噪声
        self.noise_decay = 0.9995  # 更慢的衰减
        
        # 更新计数器
        self.update_count = 0
        self.update_frequency = 5  # 每5步更新一次
        
    def _get_obs_size(self):
        return self.env.get_observation_space_size()
    
    def select_actions(self, observations, training=True, threshold=None):
        """选择动作"""
        actions = {}
        action_details = {}
        
        if threshold is None:
            threshold = 0.3  # 固定阈值
        
        for uav_id in range(self.num_uavs):
            #obs = torch.FloatTensor(observations[f'uav_{uav_id}']).unsqueeze(0)
            obs = torch.FloatTensor(observations[f'uav_{uav_id}'])

            
            noise = self.noise_scale if training else 0.0
            flattened_action, complete_action, user_probs = self.actors[f'uav_{uav_id}'].select_complete_action(
                obs.squeeze(), threshold=threshold, noise=noise
            )
            
            actions[f'uav_{uav_id}'] = flattened_action.detach().numpy()
            
            action_details[f'uav_{uav_id}'] = {
                'user_competition_probs': complete_action['user_competition_probs'].detach().numpy(),
                'offloading_ratios': complete_action['offloading_ratios'].detach().numpy(),
                'movement_direction': complete_action['movement_direction'].detach().numpy().item(),
                'movement_distance': complete_action['movement_distance'].detach().numpy().item(),
                'user_probs': user_probs.detach().numpy()
            }
        
        return actions, action_details
    
    def _process_observations(self, obs_dict):
        global_obs = []
        
        # 1. 提取所有UAV的位置信息（每个UAV 2维）
        for uav_id in range(self.num_uavs):
            uav_obs = obs_dict[f'uav_{uav_id}']
            # 只取UAV的位置信息（前2维）
            global_obs.extend(uav_obs[:2])
        
        # 2. 提取用户信息（只取一次，避免重复）
        # 从第一个UAV的观察中提取用户信息（第3维开始）
        first_uav_obs = obs_dict['uav_0']
        user_info = first_uav_obs[2:]  # 用户信息部分
        global_obs.extend(user_info)
        
        return np.array(global_obs)
    
    def _process_actions(self, action_dict):
        """将动作字典转换为全局动作向量"""
        global_actions = []
        for uav_id in range(self.num_uavs):
            global_actions.extend(action_dict[f'uav_{uav_id}'])
        return np.array(global_actions)

    def _extract_uav_observation_from_global(self, global_state, uav_id):
        """从全局状态中提取单个UAV的观察"""
        batch_size = global_state.shape[0]
        
        # 提取该UAV的位置 (2维)
        uav_pos_start = uav_id * 2
        uav_pos_end = (uav_id + 1) * 2
        uav_position = global_state[:, uav_pos_start:uav_pos_end]
        
        # 提取所有用户信息 (从第6位开始，因为前6位是3个UAV的位置)
        all_uav_positions_size = self.num_uavs * 2  # 6
        user_info = global_state[:, all_uav_positions_size:]  # 18维
        
        # 拼接该UAV的完整观察
        uav_observation = torch.cat([uav_position, user_info], dim=1)
        
        return uav_observation
    
    def update(self, batch_size=64):
        """更新网络"""
        if len(self.replay_buffer) < batch_size:
            return
        
        # 采样经验
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)
        
        # 更新Critic网络
        with torch.no_grad():
            next_actions = []
            for uav_id in range(self.num_uavs):
                # 提取UAV观察 [batch_size, 20]
                uav_next_obs = self._extract_uav_observation_from_global(next_states, uav_id)
                
                # 批量生成动作 [batch_size, 14]
                next_action, _, _ = self.actor_targets[f'uav_{uav_id}'].select_complete_action(
                    uav_next_obs, threshold=0.3, noise=0
                )
                next_actions.append(next_action)
            
            next_actions = torch.cat(next_actions, dim=1)  # [batch_size, 42]
            target_q = self.critic_target(next_states, next_actions)
            
            if rewards.dim() > 1:
                reward_mean = rewards.mean(dim=1, keepdim=True)
            else:
                reward_mean = rewards.unsqueeze(1)
            target_q = reward_mean + self.gamma * target_q * (~dones.unsqueeze(1))
        
        current_q = self.critic(states, actions)
        critic_loss = F.mse_loss(current_q, target_q)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # 更新Actor网络
        for uav_id in range(self.num_uavs):
            current_actions = actions.clone()
            
            # 提取UAV观察 [batch_size, 20]
            uav_obs = self._extract_uav_observation_from_global(states, uav_id)
            
            # 批量生成新动作 [batch_size, 14]
            new_action, _, _ = self.actors[f'uav_{uav_id}'].select_complete_action(
                uav_obs, threshold=0.3, noise=0
            )
            
            # 替换该UAV的动作
            action_start = uav_id * self.action_size_per_uav
            action_end = (uav_id + 1) * self.action_size_per_uav
            current_actions[:, action_start:action_end] = new_action
            
            actor_loss = -self.critic(states, current_actions).mean()
            
            self.actor_optimizers[f'uav_{uav_id}'].zero_grad()
            actor_loss.backward()
            self.actor_optimizers[f'uav_{uav_id}'].step()
        
        # 软更新target网络
        self._soft_update()
        
        # 缓慢衰减噪声
        self.noise_scale *= self.noise_decay
        
        return critic_loss.item(), actor_loss.item()
    
    def _soft_update(self):
        """软更新target网络"""
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        for uav_id in [f'uav_{i}' for i in range(self.num_uavs)]:
            for param, target_param in zip(self.actors[uav_id].parameters(), 
                                         self.actor_targets[uav_id].parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
    
    def train_episode(self):
        """训练一个episode"""
        obs = self.env.reset()
        total_reward = 0
        episode_length = 0
        
        # 记录详细奖励信息
        episode_reward_breakdown = {
            'total_energy_penalties': [],  # 系统总能耗惩罚
            'delay_penalties': [],         # 时延惩罚
            'total_rewards': [],
            'actual_uav_delay_raw': [],
            'actual_total_energy_raw':[],
            'step_count': 0
        }
        
        step = 0
        max_steps_safety = 200  # 安全上限（增加到200）
        while step < max_steps_safety:
            # 选择动作
            actions, action_details = self.select_actions(obs, training=True)
            
            # 执行动作
            next_obs, rewards, done, info = self.env.step(action_details)
            
            # 收集详细奖励信息
            if 'reward_breakdown' in info:
                step_breakdown = info['reward_breakdown']
                
                step_total_energy = sum([step_breakdown[f'uav_{i}']['total_energy_penalty'] for i in range(self.num_uavs)])
                step_delay = sum([step_breakdown[f'uav_{i}']['delay_penalty'] for i in range(self.num_uavs)])
                step_total_reward = sum([rewards[f'uav_{i}'] for i in range(self.num_uavs)])
                step_actual_uav_delay_raw = sum([step_breakdown[f'uav_{i}']['actual_uav_delay_raw'] for i in range(self.num_uavs)])
                step_actual_total_energy_raw = sum([step_breakdown[f'uav_{i}']['actual_total_energy_raw'] for i in range(self.num_uavs)])
                
                episode_reward_breakdown['total_energy_penalties'].append(step_total_energy)
                episode_reward_breakdown['delay_penalties'].append(step_delay)
                episode_reward_breakdown['total_rewards'].append(step_total_reward)
                episode_reward_breakdown['actual_uav_delay_raw'].append(step_actual_uav_delay_raw)
                episode_reward_breakdown['actual_total_energy_raw'].append(step_actual_total_energy_raw)
            
            # 处理数据格式
            global_obs = self._process_observations(obs)
            global_actions = self._process_actions(actions)
            global_next_obs = self._process_observations(next_obs)
            
            # 将每个UAV的奖励组成向量
            reward_vector = []
            for uav_id in range(self.num_uavs):
                uav_key = f'uav_{uav_id}'
                reward_vector.append(rewards.get(uav_key, 0.0))
            
            # 存储经验
            self.replay_buffer.push(global_obs, global_actions, reward_vector, 
                                  global_next_obs, done)
            
            # 控制更新频率
            if len(self.replay_buffer) > 500 and step % self.update_frequency == 0:  # 降低启动阈值
                critic_loss, actor_loss = self.update()
            
            # 计算平均奖励
            avg_reward = np.mean(reward_vector) if reward_vector else 0
            total_reward += sum([rewards[f'uav_{i}'] for i in range(self.num_uavs)])
            episode_length += 1
            obs = next_obs
            step += 1
            
            if done:
                break
        
        # 安全检查
        if step >= max_steps_safety:
            print(f"⚠️  警告: Episode达到安全上限 {max_steps_safety} 步，强制结束")
        
        # 计算episode总累计奖励
        episode_reward_breakdown['step_count'] = episode_length
        episode_reward_breakdown['cumulative_total_energy_penalty'] = sum(episode_reward_breakdown['total_energy_penalties'])
        episode_reward_breakdown['cumulative_delay_penalty'] = sum(episode_reward_breakdown['delay_penalties'])
        episode_reward_breakdown['cumulative_total_reward'] = sum(episode_reward_breakdown['total_rewards'])
        episode_reward_breakdown['cumulative_actual_uav_delay_raw'] = sum(episode_reward_breakdown['actual_uav_delay_raw'])
        episode_reward_breakdown['cumulative_actual_total_energy_raw'] = sum(episode_reward_breakdown['actual_total_energy_raw'])
        
        return total_reward, episode_length, episode_reward_breakdown
    
    def save_models(self, filepath):
        """保存模型"""
        torch.save({
            'actors': {uav_id: actor.state_dict() for uav_id, actor in self.actors.items()},
            'critic': self.critic.state_dict(),
        }, filepath)
    
    def load_models(self, filepath):
        """加载模型"""
        checkpoint = torch.load(filepath)
        for uav_id, actor in self.actors.items():
            actor.load_state_dict(checkpoint['actors'][uav_id])
        self.critic.load_state_dict(checkpoint['critic'])

# 使用示例
if __name__ == "__main__":
    from env_simplified import SimplifiedMultiUAVEnvironment
    
    # 创建环境和训练器
    env = SimplifiedMultiUAVEnvironment(num_uavs=3, num_users=6)
    trainer = SimplifiedMADDPGTrainer(env)
    
    print("=== 简化MADDPG训练器 ===")
    print("目标: 最小化UAV和用户能耗")
    print(f"Buffer容量: {len(trainer.replay_buffer.buffer)}")
    print(f"更新频率: 每{trainer.update_frequency}步")
