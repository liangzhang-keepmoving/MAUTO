"""
PyTorch implementation of Deep Deterministic Policy Gradient (DDPG)
Training module only - no plotting dependencies

Usage:
    python DDPG_train.py           # Train model
    python DDPG_train.py test      # Test model
    python DDPG_train.py demo      # Demo single episode
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import time
import random
from collections import deque
from UAV_env import UAVEnv
from state_normalization import StateNormalization

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

#####################  Hyper Parameters  ####################
MAX_EPISODES = 2000      # 增加训练轮数
LR_A = 0.0001           # 提高actor学习率
LR_C = 0.0003          # 适当提高critic学习率
GAMMA = 0.99            # 保持不变
TAU = 0.005             # 保持不变
VAR_MIN = 0.02          # 提高最小探索噪声
VAR_DECAY = 0.999       # 减慢衰减速度
MEMORY_CAPACITY = 15000 # 增加经验回放容量
BATCH_SIZE = 64         # 保持不变


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, action_bound):
        super(Actor, self).__init__()
        self.action_bound = action_bound
        
        # 简化结构，移除BatchNorm和Dropout
        self.fc1 = nn.Linear(state_dim, 400)
        self.fc2 = nn.Linear(400, 300)
        self.fc3 = nn.Linear(300, 64)  # 保持去瓶颈的改进
        self.fc4 = nn.Linear(64, action_dim)
        
        self.init_weights()
    
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if m == self.fc4:  # 输出层小初始化
                    nn.init.uniform_(m.weight, -0.003, 0.003)
                    nn.init.uniform_(m.bias, -0.003, 0.003)
                else:
                    nn.init.kaiming_normal_(m.weight)
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = torch.tanh(self.fc4(x))
        return x


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        
        # First layer processes state and action separately
        self.fc1_s = nn.Linear(state_dim, 400)
        self.fc1_a = nn.Linear(action_dim, 400)
        
        self.fc2 = nn.Linear(400, 300)
        self.fc3 = nn.Linear(300, 64)    # 增加容量，去除瓶颈
        self.fc4 = nn.Linear(64, 1)      # 对应调整
        
        # Initialize weights
        self.init_weights()
    
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # 使用He初始化
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)
    
    def forward(self, state, action):
        # Combine state and action in first layer
        s = self.fc1_s(state)
        a = self.fc1_a(action)
        x = F.relu(s + a)        # 使用标准relu
        
        x = F.relu(self.fc2(x))  # 使用标准relu
        x = F.relu(self.fc3(x))
        q_value = self.fc4(x)
        return q_value


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state):
        self.buffer.append((state, action, reward, next_state))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state = map(np.stack, zip(*batch))
        return state, action, reward, next_state
    
    def __len__(self):
        return len(self.buffer)


class DDPG:
    def __init__(self, state_dim, action_dim, action_bound):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_bound = action_bound
        
        # Initialize networks
        self.actor = Actor(state_dim, action_dim, action_bound).to(device)
        self.critic = Critic(state_dim, action_dim).to(device)
        self.target_actor = Actor(state_dim, action_dim, action_bound).to(device)
        self.target_critic = Critic(state_dim, action_dim).to(device)
        
        # Initialize target networks with same weights
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())
        
        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=LR_A)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=LR_C)
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer(MEMORY_CAPACITY)
        
        # Loss function
        self.criterion = nn.MSELoss()
    
    def choose_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            action = self.actor(state).cpu().numpy()[0]
        return action
    
    def store_transition(self, state, action, reward, next_state):
        self.replay_buffer.push(state, action, reward, next_state)
    
    def learn(self):
        if len(self.replay_buffer) < BATCH_SIZE:
            return
        
        # Sample from replay buffer
        states, actions, rewards, next_states = self.replay_buffer.sample(BATCH_SIZE)
        
        # Convert to tensors
        states = torch.FloatTensor(states).to(device)
        actions = torch.FloatTensor(actions).to(device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(device)
        next_states = torch.FloatTensor(next_states).to(device)
        
        # Critic loss (TD error)
        with torch.no_grad():
            next_actions = self.target_actor(next_states)
            target_q = rewards + GAMMA * self.target_critic(next_states, next_actions)
        
        current_q = self.critic(states, actions)
        critic_loss = self.criterion(current_q, target_q)
        
        # Update critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_optimizer.step()
        
        # Actor loss (maximize Q value)
        actor_actions = self.actor(states)
        actor_loss = -self.critic(states, actor_actions).mean()
        
        # Update actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_optimizer.step()
        
        # Soft update target networks (every step)
        self.soft_update(self.target_actor, self.actor, TAU)
        self.soft_update(self.target_critic, self.critic, TAU)
    
    def soft_update(self, target, source, tau):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - tau) + param.data * tau
            )


def eval_policy(ddpg, eval_episodes=10):
    eval_env = UAVEnv()
    avg_reward = 0.
    
    for i in range(eval_episodes):
        state = eval_env.reset()
        # Use slot_num for fixed number of steps
        for j in range(eval_env.slot_num):
            action = ddpg.choose_action(state)
            action = np.clip(action, *ddpg.action_bound)
            state, reward, is_terminal, step_redo, _, _ = eval_env.step(action)
            
            if step_redo:
                continue
                
            avg_reward += reward
            
            if is_terminal:
                break
    
    avg_reward /= eval_episodes
    print("---------------------------------------")
    print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
    print("---------------------------------------")
    return avg_reward


def load_model(ddpg, model_path):
    """Load trained model from checkpoint"""
    try:
        checkpoint = torch.load(model_path, map_location=device)
        ddpg.actor.load_state_dict(checkpoint['actor_state_dict'])
        ddpg.critic.load_state_dict(checkpoint['critic_state_dict'])
        ddpg.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        ddpg.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        print(f"Model loaded successfully from {model_path}")
        return True
    except Exception as e:
        print(f"Error loading model: {e}")
        return False


def test_trained_model(model_path, test_episodes=10):
    """Test the trained model"""
    env = UAVEnv()
    state_dim = env.state_dim
    action_dim = env.action_dim
    action_bound = env.action_bound
    
    # Initialize DDPG agent
    ddpg = DDPG(state_dim, action_dim, action_bound)
    
    # Load trained model
    if not load_model(ddpg, model_path):
        return
    
    # Set to evaluation mode
    ddpg.actor.eval()
    ddpg.critic.eval()
    
    s_normal = StateNormalization()
    test_rewards = []
    
    print(f"Testing trained model for {test_episodes} episodes...")
    
    for episode in range(test_episodes):
        state = env.reset()
        ep_reward = 0
        step = 0
        
        print(f"\n=== Episode {episode} ===")
        action_1_values = []  # 记录action[1]的值
        
        while step < env.slot_num:
            # Choose action without exploration noise
            with torch.no_grad():
                action = ddpg.choose_action(s_normal.state_normal(state))
                action = np.clip(action, *action_bound)
            
            action_1_values.append(action[1])
            
            if step < 5:  # 打印前5步的action
                print(f"Step {step}: action = {action}")
            
            next_state, reward, is_terminal, step_redo, offloading_ratio_change, reset_dist = env.step(action)
            
            if step_redo:
                continue
            
            state = next_state
            ep_reward += reward
            
            if step == env.slot_num - 1 or is_terminal:
                print(f'Test Episode: {episode:2d} | Steps: {step:2d} | Reward: {ep_reward:7.2f}')
                print(f'Action[1] stats: mean={np.mean(action_1_values):.3f}, std={np.std(action_1_values):.3f}, range=[{np.min(action_1_values):.3f}, {np.max(action_1_values):.3f}]')
                test_rewards.append(ep_reward)
                break
            
            step += 1
    
    # Statistics
    avg_reward = np.mean(test_rewards)
    std_reward = np.std(test_rewards)
    print(f"\nTest Results:")
    print(f"Average Reward: {avg_reward:.3f}")
    print(f"Std Deviation: {std_reward:.3f}")
    print(f"Max Reward: {np.max(test_rewards):.3f}")
    print(f"Min Reward: {np.min(test_rewards):.3f}")
    
    # Save test results
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    np.save(f'test_rewards_{timestamp}.npy', test_rewards)
    print(f"Test rewards saved to 'test_rewards_{timestamp}.npy'")
    
    return test_rewards


def run_single_episode_demo(model_path):
    """Run a single episode with detailed output for demonstration"""
    env = UAVEnv()
    state_dim = env.state_dim
    action_dim = env.action_dim
    action_bound = env.action_bound
    
    # Initialize DDPG agent
    ddpg = DDPG(state_dim, action_dim, action_bound)
    
    # Load trained model
    if not load_model(ddpg, model_path):
        return
    
    # Set to evaluation mode
    ddpg.actor.eval()
    
    s_normal = StateNormalization()
    state = env.reset()
    ep_reward = 0
    step = 0
    
    print("=== Running Demo Episode ===")
    print(f"Initial state: UAV battery: {env.e_battery_uav}, UAV location: {env.loc_uav}")
    print(f"UE locations: {env.loc_ue_list}")
    print(f"UE tasks: {env.task_list}")
    print(f"Block flags: {env.block_flag_list}")
    print()
    
    while step < env.slot_num:
        # Choose action
        with torch.no_grad():
            action = ddpg.choose_action(s_normal.state_normal(state))
            action = np.clip(action, *action_bound)
        
        print(f"Step {step}: Action = {action}")
        
        next_state, reward, is_terminal, step_redo, offloading_ratio_change, reset_dist = env.step(action)
        
        if step_redo:
            print("  -> Step redo required")
            continue
        
        if reset_dist:
            print("  -> Distance reset due to boundary violation")
        
        if offloading_ratio_change:
            print("  -> Offloading ratio changed due to energy constraint")
        
        print(f"  -> Reward: {reward:.3f}, Terminal: {is_terminal}")
        print(f"  -> UAV battery: {env.e_battery_uav:.0f}, UAV location: {env.loc_uav}")
        print()
        
        state = next_state
        ep_reward += reward
        
        if step == env.slot_num - 1 or is_terminal:
            print(f"Episode finished at step {step}")
            print(f"Total reward: {ep_reward:.3f}")
            break
        
        step += 1


def main():
    # Set random seeds for reproducibility
    np.random.seed(1)
    torch.manual_seed(1)
    random.seed(1)
    
    # Initialize environment
    env = UAVEnv()
    MAX_EP_STEPS = env.slot_num
    state_dim = env.state_dim
    action_dim = env.action_dim
    action_bound = env.action_bound  # [-1, 1]
    
    # Initialize DDPG agent
    ddpg = DDPG(state_dim, action_dim, action_bound)
    
    # Training parameters - 使用高初始探索噪声
    var = 0.15  # 高初始探索噪声
    ep_reward_list = []
    s_normal = StateNormalization()
    
    print(f"State dimension: {state_dim}")
    print(f"Action dimension: {action_dim}")
    print(f"Action bound: {action_bound}")
    print(f"Max episodes: {MAX_EPISODES}")
    print(f"Max steps per episode: {MAX_EP_STEPS}")
    print(f"Initial exploration noise: {var}")
    print(f"GAMMA: {GAMMA}, TAU: {TAU}, LR_A: {LR_A}, LR_C: {LR_C}")
    
    start_time = time.time()
    
    for episode in range(MAX_EPISODES):
        state = env.reset()
        ep_reward = 0
        step = 0
        
        # 记录本episode的action[1]值
        episode_action_1_values = []
        
        while step < MAX_EP_STEPS:
            # Choose action with exploration noise
            action = ddpg.choose_action(s_normal.state_normal(state))
            
            # 添加基础探索噪声
            action = np.clip(np.random.normal(action, var), *action_bound)
            
            # 针对action[1]添加额外探索（前70%训练期）
            if episode < MAX_EPISODES * 0.7:
                direction_noise = np.random.normal(0, 0.15)
                action[1] = np.clip(action[1] + direction_noise, -1, 1)
            
            episode_action_1_values.append(action[1])
            
            # 保存原始动作（关键修复点）
            original_action = action.copy()
            
            # Take action
            next_state, reward, is_terminal, step_redo, offloading_ratio_change, reset_dist = env.step(action)
            
            if step_redo:
                continue
            
            # 关键修复：存储原始动作，不做任何修改
            ddpg.store_transition(
                s_normal.state_normal(state), 
                original_action,  # 存储原始动作，不是修改后的
                reward, 
                s_normal.state_normal(next_state)
            )
            
            # Learn from experience (更早开始学习)
            if len(ddpg.replay_buffer) > BATCH_SIZE * 5:
                ddpg.learn()
                # Decay exploration noise
                if var > VAR_MIN:
                    var *= VAR_DECAY
            
            state = next_state
            ep_reward += reward
            
            if step == MAX_EP_STEPS - 1 or is_terminal:
                # 计算action[1]的统计信息
                action_1_std = np.std(episode_action_1_values)
                action_1_range = np.max(episode_action_1_values) - np.min(episode_action_1_values)
                
                print(f'Episode: {episode:4d} | Steps: {step:2d} | Reward: {ep_reward:7.2f} | Explore: {var:.4f} | Action[1] std: {action_1_std:.3f} | range: {action_1_range:.3f}')
                ep_reward_list.append(ep_reward)
                
                # Write to file
                file_name = 'output.txt'
                with open(file_name, 'a') as file_obj:
                    file_obj.write(f"\n======== Episode {episode} done, Reward: {ep_reward:.2f} ========")
                break
            
            step += 1
        
        # Evaluate periodically
        if (episode + 1) % 100 == 0:
            eval_policy(ddpg, 5)
    
    print(f'Training completed in {time.time() - start_time:.2f} seconds')
    
    # Save training rewards to file with timestamp
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    reward_file = f'training_rewards_{timestamp}.npy'
    np.save(reward_file, ep_reward_list)
    print(f"Training rewards saved to '{reward_file}'")
    
    # Also save to default filename for convenience
    np.save('training_rewards.npy', ep_reward_list)
    
    # Save training log
    log_file = f'training_log_{timestamp}.txt'
    with open(log_file, 'w') as f:
        f.write(f"Training completed at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total episodes: {len(ep_reward_list)}\n")
        f.write(f"Training time: {time.time() - start_time:.2f} seconds\n")
        f.write(f"Final reward: {ep_reward_list[-1]:.2f}\n")
        f.write(f"Best reward: {max(ep_reward_list):.2f}\n")
        f.write(f"Average reward: {np.mean(ep_reward_list):.2f}\n")
        f.write(f"Hyperparameters:\n")
        f.write(f"  GAMMA: {GAMMA}\n")
        f.write(f"  TAU: {TAU}\n")
        f.write(f"  LR_A: {LR_A}\n")
        f.write(f"  LR_C: {LR_C}\n")
        f.write(f"  VAR_MIN: {VAR_MIN}\n")
        f.write(f"  VAR_DECAY: {VAR_DECAY}\n")
        f.write(f"  BATCH_SIZE: {BATCH_SIZE}\n")
        f.write(f"  MEMORY_CAPACITY: {MEMORY_CAPACITY}\n")
        f.write(f"\nEpisode rewards:\n")
        for i, reward in enumerate(ep_reward_list):
            f.write(f"Episode {i}: {reward:.2f}\n")
    
    print(f"Training log saved to '{log_file}'")
    
    # Save model
    model_file = f'ddpg_model_fixed_{timestamp}.pth'
    torch.save({
        'actor_state_dict': ddpg.actor.state_dict(),
        'critic_state_dict': ddpg.critic.state_dict(),
        'actor_optimizer_state_dict': ddpg.actor_optimizer.state_dict(),
        'critic_optimizer_state_dict': ddpg.critic_optimizer.state_dict(),
        'episode_rewards': ep_reward_list,
        'training_time': time.time() - start_time,
        'timestamp': timestamp,
        'hyperparameters': {
            'GAMMA': GAMMA,
            'TAU': TAU,
            'LR_A': LR_A,
            'LR_C': LR_C,
            'VAR_MIN': VAR_MIN,
            'VAR_DECAY': VAR_DECAY,
            'BATCH_SIZE': BATCH_SIZE,
            'MEMORY_CAPACITY': MEMORY_CAPACITY
        }
    }, model_file)
    
    # Also save to default filename
    torch.save({
        'actor_state_dict': ddpg.actor.state_dict(),
        'critic_state_dict': ddpg.critic.state_dict(),
        'actor_optimizer_state_dict': ddpg.actor_optimizer.state_dict(),
        'critic_optimizer_state_dict': ddpg.critic_optimizer.state_dict(),
        'episode_rewards': ep_reward_list,
        'training_time': time.time() - start_time,
        'timestamp': timestamp,
        'hyperparameters': {
            'GAMMA': GAMMA,
            'TAU': TAU,
            'LR_A': LR_A,
            'LR_C': LR_C,
            'VAR_MIN': VAR_MIN,
            'VAR_DECAY': VAR_DECAY,
            'BATCH_SIZE': BATCH_SIZE,
            'MEMORY_CAPACITY': MEMORY_CAPACITY
        }
    }, 'ddpg_model.pth')
    
    print(f"Model saved as '{model_file}' and 'ddpg_model.pth'")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == "test":
            # Test mode
            model_path = "ddpg_model.pth"
            if len(sys.argv) > 2:
                model_path = sys.argv[2]
            test_trained_model(model_path)
            
        elif command == "demo":
            # Demo mode
            model_path = "ddpg_model.pth"
            if len(sys.argv) > 2:
                model_path = sys.argv[2]
            run_single_episode_demo(model_path)
            
        else:
            print("Unknown command. Available commands:")
            print("  python DDPG_train.py           - Training mode")
            print("  python DDPG_train.py test [model.pth] - Test trained model")
            print("  python DDPG_train.py demo [model.pth] - Demo single episode")
    else:
        # Training mode
        main()