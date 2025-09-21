"""
UAV环境LLM训练数据生成器
按照论文要求使用训练好的DDPG模型生成高质量状态-动作对数据
"""

import numpy as np
import json
import torch
import time
import os
from UAV_env import UAVEnv
from DDPG import DDPG, load_model
from state_normalization import StateNormalization

class UAVDataGenerator:
    def __init__(self, model_path="ddpg_model.pth"):
        self.env = UAVEnv()
        self.state_normalizer = StateNormalization()
        self.model_path = model_path
        self.ddpg = None
        self.load_trained_model()
        
    def load_trained_model(self):
        """加载训练好的DDPG模型"""
        try:
            self.ddpg = DDPG(self.env.state_dim, self.env.action_dim, self.env.action_bound)
            if load_model(self.ddpg, self.model_path):
                self.ddpg.actor.eval()
                print(f"Successfully loaded trained model from {self.model_path}")
                return True
            else:
                print(f"Failed to load model from {self.model_path}")
                return False
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def normalize_state_for_llm(self, state):
        """将状态归一化到[0,1]范围，便于LLM理解"""
        # 定义状态各分量的最大值
        max_values = [
            500000,           # UAV电池容量
            100, 100,         # UAV x,y 坐标
            100 * 1048576,    # 剩余任务大小
            # UE位置 (4个UE，每个2个坐标)
            100, 100, 100, 100, 100, 100, 100, 100,
            # UE任务大小 (4个UE)
            3145729, 3145729, 3145729, 3145729,
            # UE遮挡标志 (4个UE)
            1, 1, 1, 1
        ]
        
        normalized = []
        for i, (val, max_val) in enumerate(zip(state, max_values)):
            norm_val = min(max(float(val) / max_val, 0), 1)  # 确保转换为Python float
            normalized.append(round(norm_val, 3))
        
        return normalized
    
    def action_to_llm_format(self, action):
        """将动作从[-1,1]转换到[0,1]范围"""
        action_01 = (np.array(action) + 1) / 2
        return [round(float(x), 3) for x in action_01]
    
    def generate_episode_data(self, episode_id, add_exploration=False):
        """生成单个episode的数据"""
        state = self.env.reset()
        episode_data = []
        step = 0
        
        while step < self.env.slot_num:
            # 获取动作
            if self.ddpg is not None:
                # 使用训练好的模型
                normalized_state = self.state_normalizer.state_normal(state)
                action = self.ddpg.choose_action(normalized_state)
                
                # 可选：添加少量探索噪声增加数据多样性
                if add_exploration:
                    noise = np.random.normal(0, 0.05, self.env.action_dim)
                    action = np.clip(action + noise, -1, 1)
            else:
                # 如果模型加载失败，使用随机动作
                action = np.random.uniform(-1, 1, self.env.action_dim)
            
            # 执行动作
            next_state, reward, is_terminal, step_redo, offloading_change, reset_dist = self.env.step(action)
            
            # 跳过需要重做的步骤
            if step_redo:
                continue
            
            # 记录数据点
            data_point = {
                "state": self.normalize_state_for_llm(state),
                "action": self.action_to_llm_format(action),
                "reward": round(float(reward), 3),
                "episode": episode_id,
                "step": step,
                "terminal": is_terminal,
                # 额外信息（用于分析，不用于LLM训练）
                "meta": {
                    "uav_battery": float(self.env.e_battery_uav),
                    "uav_location": self.env.loc_uav.copy(),
                    "offloading_change": offloading_change,
                    "reset_dist": reset_dist,
                    "raw_state": state.tolist(),
                    "raw_action": action.tolist()
                }
            }
            
            episode_data.append(data_point)
            state = next_state
            step += 1
            
            if is_terminal:
                break
        
        return episode_data
    
    def generate_training_data(self, episodes=200, add_exploration_ratio=0.3):
        """
        生成LLM训练数据
        
        Args:
            episodes: 生成的episode数量
            add_exploration_ratio: 添加探索噪声的episode比例
        """
        if self.ddpg is None:
            print("Warning: No trained model loaded. Using random policy.")
        
        all_training_data = []
        successful_episodes = 0
        
        print(f"Starting data generation for {episodes} episodes...")
        start_time = time.time()
        
        for episode in range(episodes):
            # 决定是否添加探索噪声
            add_exploration = np.random.random() < add_exploration_ratio
            
            try:
                episode_data = self.generate_episode_data(episode, add_exploration)
                
                if len(episode_data) > 0:
                    all_training_data.extend(episode_data)
                    successful_episodes += 1
                    
                    if episode % 20 == 0:
                        avg_reward = np.mean([d["reward"] for d in episode_data])
                        print(f"Episode {episode}: {len(episode_data)} samples, "
                              f"avg_reward: {avg_reward:.3f}, "
                              f"exploration: {'Yes' if add_exploration else 'No'}")
                
            except Exception as e:
                print(f"Error in episode {episode}: {e}")
                continue
        
        elapsed_time = time.time() - start_time
        print(f"\nData generation completed!")
        print(f"Time elapsed: {elapsed_time:.2f} seconds")
        print(f"Successful episodes: {successful_episodes}/{episodes}")
        print(f"Total samples generated: {len(all_training_data)}")
        
        return all_training_data
    
    def analyze_data_quality(self, training_data):
        """分析生成数据的质量"""
        if not training_data:
            print("No data to analyze")
            return
        
        rewards = [d["reward"] for d in training_data]
        states = np.array([d["state"] for d in training_data])
        actions = np.array([d["action"] for d in training_data])
        
        print("\n=== Data Quality Analysis ===")
        print(f"Total samples: {len(training_data)}")
        print(f"Episodes represented: {len(set(d['episode'] for d in training_data))}")
        
        print(f"\nReward Statistics:")
        print(f"  Mean: {np.mean(rewards):.3f}")
        print(f"  Std: {np.std(rewards):.3f}")
        print(f"  Min: {np.min(rewards):.3f}")
        print(f"  Max: {np.max(rewards):.3f}")
        print(f"  25th percentile: {np.percentile(rewards, 25):.3f}")
        print(f"  75th percentile: {np.percentile(rewards, 75):.3f}")
        
        print(f"\nAction Distribution (each should be roughly [0,1]):")
        action_names = ["UE_selection", "Flight_direction", "Flight_distance", "Offloading_ratio"]
        for i in range(actions.shape[1]):
            print(f"  {action_names[i]}: mean={np.mean(actions[:, i]):.3f}, "
                  f"std={np.std(actions[:, i]):.3f}, "
                  f"range=[{np.min(actions[:, i]):.3f}, {np.max(actions[:, i]):.3f}]")
        
        print(f"\nState Statistics (first 4 dimensions):")
        state_names = ["UAV_battery", "UAV_x", "UAV_y", "Remaining_tasks"]
        for i in range(min(4, states.shape[1])):
            print(f"  {state_names[i]}: mean={np.mean(states[:, i]):.3f}, "
                  f"std={np.std(states[:, i]):.3f}")
    
    def filter_quality_data(self, training_data, reward_threshold=-50, min_steps=5):
        """过滤高质量数据"""
        filtered_data = []
        
        for data in training_data:
            if (data["reward"] > reward_threshold and 
                data["step"] >= min_steps and
                not data["meta"]["offloading_change"]):
                filtered_data.append(data)
        
        print(f"Filtered data: {len(filtered_data)}/{len(training_data)} samples kept")
        return filtered_data
    
    def save_data(self, training_data, base_filename="uav_llm_training_data"):
        """保存训练数据到多种格式"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        # 1. 保存LLM格式数据（简化版，只包含state, action, reward）
        llm_filename = f"{base_filename}_{timestamp}.txt"
        with open(llm_filename, 'w') as f:
            for data in training_data:
                llm_sample = {
                    "state": data["state"],
                    "action": data["action"],
                    "reward": data["reward"]
                }
                f.write(json.dumps(llm_sample) + '\n')
        
        # 2. 保存完整数据（包含所有信息）
        full_filename = f"{base_filename}_full_{timestamp}.json"
        with open(full_filename, 'w') as f:
            json.dump(training_data, f, indent=2)
        
        # 3. 保存数据统计
        stats_filename = f"{base_filename}_stats_{timestamp}.txt"
        with open(stats_filename, 'w') as f:
            f.write(f"Data Generation Report\n")
            f.write(f"Generated at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Model used: {self.model_path}\n")
            f.write(f"Total samples: {len(training_data)}\n")
            f.write(f"Episodes: {len(set(d['episode'] for d in training_data))}\n")
            
            rewards = [d["reward"] for d in training_data]
            f.write(f"\nReward Statistics:\n")
            f.write(f"  Mean: {np.mean(rewards):.3f}\n")
            f.write(f"  Std: {np.std(rewards):.3f}\n")
            f.write(f"  Min: {np.min(rewards):.3f}\n")
            f.write(f"  Max: {np.max(rewards):.3f}\n")
        
        print(f"\nData saved to:")
        print(f"  LLM format: {llm_filename}")
        print(f"  Full data: {full_filename}")
        print(f"  Statistics: {stats_filename}")
        
        return llm_filename, full_filename, stats_filename
    
    def generate_sample_prompt(self, training_data, num_examples=5):
        """生成示例LLM提示词"""
        sample_data = training_data[:num_examples]
        
        prompt = """You are an intelligent UAV edge computing decision system.
Your task is to predict optimal actions for UAV task offloading based on current state.

Environment Description:
- State: [uav_battery, uav_x, uav_y, remaining_tasks, ue_locations..., ue_tasks..., ue_blocks...]
- Action: [ue_selection_ratio, flight_direction, flight_distance, offloading_ratio]
- All values are normalized to [0,1] range

Training Examples:
"""
        
        for data in sample_data:
            state_str = str(data["state"])
            action_str = str(data["action"])
            prompt += f"state: {state_str}, action: {action_str}\n"
        
        prompt += """\nBased on these patterns, predict the action for the given state.
Output only the action array in format: [value1, value2, value3, value4]

Current state: [0.456, 0.23, 0.78, 0.12, 0.45, 0.67, 0.89, 0.34, 0.56, 0.78, 0.23, 0.45, 0.234, 0.567, 0.789, 0.123, 0, 1, 0, 1]
Predict action:"""
        
        print("\n=== Sample LLM Prompt ===")
        print(prompt)
        
        return prompt


def main():
    """主函数"""
    # 使用您的训练模型路径
    model_path = "/home/niuma008/zsz/UAV-DDPG/DDPG_pytorch/ddpg_model_fixed_20250920_210625.pth"
    
    if not os.path.exists(model_path):
        print(f"Error: Model file {model_path} not found!")
        print("Please check the model path.")
        return
    
    print(f"Using trained model: {model_path}")
    
    # 初始化数据生成器
    generator = UAVDataGenerator(model_path)
    
    # 生成训练数据
    print("Generating training data...")
    training_data = generator.generate_training_data(
        episodes=200,           # 生成200个episode
        add_exploration_ratio=0.2  # 20%的episode添加探索噪声
    )
    
    if not training_data:
        print("No training data generated!")
        return
    
    # 分析数据质量
    generator.analyze_data_quality(training_data)
    
    # 可选：过滤高质量数据
    # training_data = generator.filter_quality_data(training_data)
    
    # 保存数据
    llm_file, full_file, stats_file = generator.save_data(training_data)
    
    # 生成示例提示词
    generator.generate_sample_prompt(training_data)
    
    print(f"\n数据生成完成！共生成 {len(training_data)} 个训练样本")
    print(f"LLM训练数据文件: {llm_file}")


if __name__ == "__main__":
    main()