from env_simplified import SimplifiedMultiUAVEnvironment
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import deque
import random

# ============================================================================
# 1. Actor 网络 - 单一网络输出所有决策
# ============================================================================

class Actor(nn.Module):
    """Actor网络 - 统一处理所有无人机和用户"""
    def __init__(self, n_uavs, n_users, uav_feature_dim=2, user_feature_dim=3):
        """
        Args:
            n_uavs: 无人机数量
            n_users: 用户数量
            uav_feature_dim: 无人机特征维度 (默认2: x,y)
            user_feature_dim: 用户特征维度 (默认3: x,y,task_size)
        """
        super(Actor, self).__init__()
        self.n_uavs = n_uavs
        self.n_users = n_users
        
        # 无人机特征编码器 (2D位置)
        self.uav_encoder = nn.Sequential(
            nn.Linear(uav_feature_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128)
        )
        
        # 用户特征编码器 (2D位置 + 任务大小)
        self.user_encoder = nn.Sequential(
            nn.Linear(user_feature_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128)
        )
        
        # 全局特征融合
        self.global_fusion = nn.Sequential(
            nn.Linear(128 * (n_uavs + n_users), 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU()
        )
        
        # === 输出头1: 用户分配 ===
        # 输出 [N × M] 的logits
        self.allocation_head = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, n_uavs * n_users)
        )
        
        # === 输出头2: 卸载比例 ===
        self.offloading_head = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, n_users)
        )

        # === 输出头3: 运动控制 (XY速度) ===
        self.motion_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, n_uavs * 2) # 输出 vx, vy
        )
        
    def forward(self, uav_pos, user_pos, user_tasks, hard=False):

        batch_size = uav_pos.shape[0]
        
        # === 编码特征 ===
        uav_features = self.uav_encoder(uav_pos)  # [batch, N, 128]
        user_input = torch.cat([user_pos, user_tasks], dim=-1)  # [batch, M, 3]
        user_features = self.user_encoder(user_input)  # [batch, M, 128]
        
        # === 展平并融合所有特征 ===
        combined = torch.cat([
            uav_features.view(batch_size, -1),      # [batch, N*128]
            user_features.view(batch_size, -1)      # [batch, M*128]
        ], dim=1)  # [batch, (N+M)*128]
        
        global_features = self.global_fusion(combined)  # [batch, 256]
        
        # === 输出1: 用户分配 ===
        allocation_logits = self.allocation_head(global_features)  # [batch, N*M]
        allocation_logits = allocation_logits.view(batch_size, self.n_uavs, self.n_users)
        
        # 在无人机维度(dim=1)做softmax，确保每个用户选择概率和为1
        allocation_soft = F.softmax(allocation_logits, dim=1)  # [batch, N, M]
        
        if hard:
            # 硬分配：每个用户选择概率最高的无人机
            allocation = self._hard_allocation(allocation_soft)
        else:
            allocation = allocation_soft
        
        # 2. 卸载比例
        offloading_raw = self.offloading_head(global_features) # [batch, M]
        offloading_final = torch.sigmoid(offloading_raw)       # 范围 [0, 1]
        
        # 3. 运动控制 (XY 速度)
        motion_raw = self.motion_head(global_features).view(batch_size, self.n_uavs, 2)
        motion = torch.tanh(motion_raw) # 范围 [-1, 1]，代表归一化的 vx, vy
        
        return allocation, offloading_final, motion
    
    def _hard_allocation(self, allocation_soft):
        """将软分配转换为硬分配 (0/1)"""
        max_indices = allocation_soft.argmax(dim=1, keepdim=True)  # [batch, 1, M]
        allocation_hard = torch.zeros_like(allocation_soft)
        allocation_hard.scatter_(1, max_indices, 1.0)
        return allocation_hard


# ============================================================================
# 2. Critic 网络 - 评估状态-动作价值
# ============================================================================

class Critic(nn.Module):
    """Critic网络 - Q(s, a)"""
    def __init__(self, n_uavs, n_users, state_dim):
        """
        Args:
            n_uavs: 无人机数量
            n_users: 用户数量
            state_dim: 状态总维度 (N*2 + M*3)
        """
        super(Critic, self).__init__()
        
        # 状态编码器
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )
        
        # 分配动作编码器
        self.allocation_encoder = nn.Sequential(
            nn.Linear(n_uavs * n_users, 128),
            nn.ReLU()
        )
        
        # 卸载比例编码器
        self.offloading_encoder = nn.Sequential(
            nn.Linear(n_users, 128), # 修改：与 Actor 输出对齐，只输入 M 个值
            nn.ReLU()
        )
        
        # 运动动作编码器
        self.motion_encoder = nn.Sequential(
            nn.Linear(n_uavs * 2, 64),
            nn.ReLU()
        )
        
        # Q值输出
        self.q_net = nn.Sequential(
            nn.Linear(128 + 128 + 128 + 64, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
    def forward(self, state, allocation, offloading, motion):
        """
        Args:
            state: [batch, state_dim]
            allocation: [batch, N, M]
            offloading: [batch, M]  # Modified: Now only N_Users vector
            motion: [batch, N, 2]
        
        Returns:
            q_value: [batch, 1]
        """
        # 编码状态
        state_feat = self.state_encoder(state)
        
        # 编码动作 (展平)
        allocation_flat = allocation.view(allocation.shape[0], -1)
        allocation_feat = self.allocation_encoder(allocation_flat)
        
        # offloading is now [batch, M]
        offloading_feat = self.offloading_encoder(offloading)
        
        motion_flat = motion.view(motion.shape[0], -1)
        motion_feat = self.motion_encoder(motion_flat)
        
        # 融合所有特征
        combined = torch.cat([
            state_feat, allocation_feat, offloading_feat, motion_feat
        ], dim=-1)
        
        # 输出Q值
        q_value = self.q_net(combined)
        
        return q_value


# ============================================================================
# 3. Ornstein-Uhlenbeck 噪声 (用于探索)
# ============================================================================

class OUNoise:
    """Ornstein-Uhlenbeck过程噪声"""
    def __init__(self, action_dim, mu=0.0, theta=0.15, sigma=0.2):
        self.action_dim = action_dim
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.action_dim) * self.mu
        self.reset()
    
    def reset(self):
        self.state = np.ones(self.action_dim) * self.mu
    
    def sample(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.action_dim)
        self.state = x + dx
        return torch.FloatTensor(self.state)


# ============================================================================
# 4. 经验回放缓冲区
# ============================================================================

class ReplayBuffer:
    """经验回放缓冲区"""
    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        """
        存储经验
        state: dict with keys 'uav_pos', 'user_pos', 'user_tasks'
        action: tuple (allocation, offloading, motion)
        """
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        """采样一批经验"""
        batch = random.sample(self.buffer, batch_size)
        
        # 解包
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # 转换为tensor
        state_batch = self._collate_states(states)
        next_state_batch = self._collate_states(next_states)
        
        allocation_batch = torch.stack([a[0] for a in actions])
        offloading_batch = torch.stack([a[1] for a in actions])
        motion_batch = torch.stack([a[2] for a in actions])
        
        reward_batch = torch.FloatTensor(rewards).unsqueeze(1)
        done_batch = torch.FloatTensor(dones).unsqueeze(1)
        
        return (state_batch, 
                (allocation_batch, offloading_batch, motion_batch),
                reward_batch, 
                next_state_batch, 
                done_batch)
    
    def _collate_states(self, states):
        """将状态列表转换为batch tensor"""
        # Use torch.stack to properly stack tensors
        uav_pos = torch.stack([s['uav_pos'] if isinstance(s['uav_pos'], torch.Tensor) else torch.tensor(s['uav_pos']) for s in states])
        user_pos = torch.stack([s['user_pos'] if isinstance(s['user_pos'], torch.Tensor) else torch.tensor(s['user_pos']) for s in states])
        user_tasks = torch.stack([s['user_tasks'] if isinstance(s['user_tasks'], torch.Tensor) else torch.tensor(s['user_tasks']) for s in states])
        return {'uav_pos': uav_pos, 'user_pos': user_pos, 'user_tasks': user_tasks}
    
    def __len__(self):
        return len(self.buffer)


# ============================================================================
# 5. DDPG Agent
# ============================================================================

class DDPGAgent:
    """DDPG智能体"""
    def __init__(self, n_uavs, n_users, lr_actor=1e-4, lr_critic=1e-3, 
                 gamma=0.99, tau=0.005, max_distance=25.0):
        """
        Args:
            n_uavs: 无人机数量
            n_users: 用户数量
            lr_actor: Actor学习率
            lr_critic: Critic学习率
            gamma: 折扣因子
            tau: 软更新系数
            max_distance: 最大飞行距离
        """
        self.n_uavs = n_uavs
        self.n_users = n_users
        self.gamma = gamma
        self.tau = tau
        self.max_distance = max_distance
        
        state_dim = n_uavs * 2 + n_users * 3  # 无人机2D位置 + 用户2D位置和任务
        
        # 创建网络
        self.actor = Actor(n_uavs, n_users, uav_feature_dim=2, user_feature_dim=3)
        self.actor_target = Actor(n_uavs, n_users, uav_feature_dim=2, user_feature_dim=3)
        self.actor_target.load_state_dict(self.actor.state_dict())
        
        self.critic = Critic(n_uavs, n_users, state_dim)
        self.critic_target = Critic(n_uavs, n_users, state_dim)
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        # 优化器
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr_critic)
        
        # 噪声
        # 
        self.noise = OUNoise(action_dim=n_uavs * 2)
        self.noise_scale = 1.0      # 新增: 噪声比例
        self.noise_decay = 0.995    # 新增: 衰减率
        self.min_noise_scale = 0.01 # 新增: 最小噪声
        
        # 经验回放
        self.replay_buffer = ReplayBuffer()
        
        # 设备
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)
    
    def to(self, device):
        """将所有网络移到指定设备"""
        self.actor.to(device)
        self.actor_target.to(device)
        self.critic.to(device)
        self.critic_target.to(device)
        self.device = device
    
    def select_action(self, state, add_noise=True, hard=False):
        """
        选择动作
        Args:
            state: dict with 'uav_pos' [N,2], 'user_pos' [M,2], 'user_tasks' [M,1]
            add_noise: 是否添加探索噪声
            hard: 是否使用硬分配
        
        Returns:
            allocation, offloading, motion (all as tensors)
        """
        with torch.no_grad():
            # 转换到正确的设备和维度
            uav_pos = state['uav_pos'].unsqueeze(0).to(self.device)
            user_pos = state['user_pos'].unsqueeze(0).to(self.device)
            user_tasks = state['user_tasks'].unsqueeze(0).to(self.device)
            
            allocation, offloading, motion = self.actor(
                uav_pos, user_pos, user_tasks, hard=hard
            )
            
            if add_noise:
                # 1. 为运动添加噪声 (OU Noise)
                # 减小噪声强度
                noise = self.noise.sample().view(1, self.n_uavs, 2).to(self.device)
                motion = motion + noise * self.noise_scale * 0.5 # 降低运动噪声影响
                motion = torch.clamp(motion, -1, 1)
                
                # 2. 为卸载比例添加噪声 (Gaussian Noise)
                # offloading shape: [1, n_users]
                # 降低卸载决策的噪声
                off_noise = torch.randn_like(offloading) * 0.05 * self.noise_scale 
                offloading = offloading + off_noise
                offloading = torch.clamp(offloading, 0, 1)
        
        return allocation.squeeze(0), offloading.squeeze(0), motion.squeeze(0)

    def decay_noise(self):
        """
        每个episode结束后调用，衰减噪声
        """
        self.noise_scale = max(
            self.min_noise_scale, 
            self.noise_scale * self.noise_decay
        )
        
    
    def reset_noise(self):
        """
        重置OU噪声的内部状态（每个episode开始时调用）
        """
        self.noise.reset()
    
    def get_noise_scale(self):
        """
        获取当前噪声比例（用于监控）
        """
        return self.noise_scale
    def train_step(self, batch_size=64):
        """执行一步训练"""
        if len(self.replay_buffer) < batch_size:
            return None, None
        
        # 采样
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = \
            self.replay_buffer.sample(batch_size)
        
        # 移到设备
        state_flat = self._flatten_state(state_batch).to(self.device)
        next_state_flat = self._flatten_state(next_state_batch).to(self.device)
        
        allocation_batch = action_batch[0].to(self.device)
        offloading_batch = action_batch[1].to(self.device)
        motion_batch = action_batch[2].to(self.device)
        
        # 归一化奖励: 将奖励缩放到一定范围，减少方差
        reward_batch = reward_batch.to(self.device)
        # 奖励缩放 (Reward Scaling)
        # 假设奖励主要在 [-10, 0] 之间，可以除以一个常数，例如 10.0
        # 或者使用批次归一化 (但要注意 Critic 的目标值稳定性)
        # reward_batch = reward_batch * 0.1 

        done_batch = done_batch.to(self.device)
        
        # ===== 更新Critic =====
        with torch.no_grad():
            # 目标网络预测下一个动作
            next_allocation, next_offloading, next_motion = self.actor_target(
                next_state_batch['uav_pos'].to(self.device),
                next_state_batch['user_pos'].to(self.device),
                next_state_batch['user_tasks'].to(self.device),
                hard=False
            )
            
            # 目标Q值
            target_q = self.critic_target(
                next_state_flat, next_allocation, next_offloading, next_motion
            )
            y = reward_batch + self.gamma * (1 - done_batch) * target_q
        
        # 当前Q值
        current_q = self.critic(state_flat, allocation_batch, offloading_batch, motion_batch)
        
        # Critic损失
        critic_loss = F.mse_loss(current_q, y)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_optimizer.step()
        
        # ===== 更新Actor =====
        # 注意：必须用soft分配以保证梯度流动
        pred_allocation, pred_offloading, pred_motion = self.actor(
            state_batch['uav_pos'].to(self.device),
            state_batch['user_pos'].to(self.device),
            state_batch['user_tasks'].to(self.device),
            hard=False  # 训练时必须是soft
        )
        
        actor_loss = -self.critic(
            state_flat, pred_allocation, pred_offloading, pred_motion
        ).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_optimizer.step()
        
        # ===== 软更新目标网络 =====
        self._soft_update(self.actor, self.actor_target)
        self._soft_update(self.critic, self.critic_target)
        
        return critic_loss.item(), actor_loss.item()
    
    def _flatten_state(self, state_dict):
        """将状态字典展平为向量"""
        batch_size = state_dict['uav_pos'].shape[0]
        return torch.cat([
            state_dict['uav_pos'].view(batch_size, -1),
            state_dict['user_pos'].view(batch_size, -1),
            state_dict['user_tasks'].view(batch_size, -1)
        ], dim=1)
    
    def _soft_update(self, source, target):
        """软更新目标网络"""
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.tau) + param.data * self.tau
            )
    
    def save(self, filepath):
        """保存模型"""
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'actor_target': self.actor_target.state_dict(),
            'critic_target': self.critic_target.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
        }, filepath)
        print(f"模型已保存到 {filepath}")
    
    def load(self, filepath):
        """加载模型"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.actor_target.load_state_dict(checkpoint['actor_target'])
        self.critic_target.load_state_dict(checkpoint['critic_target'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
        print(f"模型已从 {filepath} 加载")

