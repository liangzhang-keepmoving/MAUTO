"""
LLM-based UAV Edge Computing Controller
基于大语言模型的无人机边缘计算控制器
"""

import json
import numpy as np
import re
from openai import OpenAI
from UAV_env import UAVEnv
from state_normalization import StateNormalization


class LLMUAVController:
    def __init__(self, api_key, model_name="google/gemini-2.5-pro", training_data_file=None):
        """
        初始化LLM控制器

        Args:
            api_key: OpenRouter API密钥
            model_name: 使用的模型名称
            training_data_file: 训练数据文件路径
        """
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",  # 添加OpenRouter端点
            api_key=api_key
        )
        self.model_name = model_name
        self.env = UAVEnv()
        self.state_normalizer = StateNormalization()

        # 加载训练数据
        self.training_data = []
        if training_data_file:
            self.load_training_data(training_data_file)

        # 构建上下文模板
        self.context_template = self.build_context_template()

    def load_training_data(self, file_path):
        """加载训练数据"""
        try:
            with open(file_path, 'r') as f:
                for line in f:
                    if line.strip():
                        data = json.loads(line.strip())
                        self.training_data.append(data)
            print(f"Loaded {len(self.training_data)} training samples")
        except FileNotFoundError:
            print(f"Training data file {file_path} not found")
        except Exception as e:
            print(f"Error loading training data: {e}")

    def build_context_template(self):
        """构建LLM上下文模板"""
        # 选择代表性的训练样本
        selected_samples = self.select_representative_samples()

        examples_text = ""
        for i, sample in enumerate(selected_samples):
            state_str = str(sample['state'])
            action_str = str(sample['action'])
            examples_text += f"state: {state_str}, action: {action_str}\n"

        template = f"""You are an intelligent UAV edge computing decision system.
Your task is to predict optimal actions for UAV task offloading based on current state.

Environment Description:
- Single UAV serves 4 UEs (User Equipment) sequentially, one at a time
- UAV selects one UE to serve in each time step, then flies to optimal position
- Each UE has computing tasks that can be processed locally or offloaded to UAV server
- UAV has limited battery and must manage energy consumption for flight and computation

State Format (20 dimensions):
- uav_battery: Current UAV battery level [0,1]
- uav_x, uav_y: UAV position coordinates [0,1] 
- remaining_tasks: Total remaining task size across all UEs [0,1]
- ue_locations: 8 values (x,y coordinates for 4 UEs) [0,1]
- ue_tasks: 4 values (task sizes for each UE) [0,1]
- ue_blocks: 4 values (LOS/NLOS flags for each UE, 0 or 1)

Action Format (4 dimensions):
- ue_selection_ratio: Which UE to serve (0-1 maps to UE 0-3)
- flight_direction: UAV flight direction (0-1 maps to 0-2π radians)
- flight_distance: How far to fly (0-1 maps to max flight distance)
- offloading_ratio: Fraction of selected UE's task to offload to UAV (0-1)

Constraints:
- UAV can only serve one UE per time step
- Flight consumes battery energy
- Task processing consumes battery energy
- Communication quality depends on distance and LOS/NLOS conditions
- Goal: Minimize total task completion delay across all UEs

Training Examples:
{examples_text}

Based on these patterns and constraints, predict the optimal action for the current state.
Output ONLY the action array in format: [ue_selection, flight_direction, flight_distance, offloading_ratio]
"""
        return template

    def select_representative_samples(self, num_samples=70):
        """选择代表性的训练样本"""
        if not self.training_data:
            return []

        # 按奖励分层选择样本
        sorted_data = sorted(self.training_data, key=lambda x: x['reward'], reverse=True)

        # 选择不同性能水平的样本
        selected = []
        total_samples = len(sorted_data)

        if total_samples == 0:
            return []

        # 高性能样本 (前20%)
        high_end = max(1, total_samples // 5)
        high_perf_samples = sorted_data[:high_end]
        high_count = min(6, len(high_perf_samples))
        if high_count > 0:
            indices = np.random.choice(len(high_perf_samples), high_count, replace=False)
            selected.extend([high_perf_samples[i] for i in indices])

        # 中等性能样本 (中间40%)
        mid_start = max(1, total_samples // 5)
        mid_end = max(mid_start + 1, 3 * total_samples // 5)
        mid_perf_samples = sorted_data[mid_start:mid_end]
        if mid_perf_samples:
            mid_count = min(6, len(mid_perf_samples))
            indices = np.random.choice(len(mid_perf_samples), mid_count, replace=False)
            selected.extend([mid_perf_samples[i] for i in indices])

        # 较低性能样本 (后40%)
        low_start = max(1, 3 * total_samples // 5)
        low_perf_samples = sorted_data[low_start:]
        if low_perf_samples:
            low_count = min(3, len(low_perf_samples))
            indices = np.random.choice(len(low_perf_samples), low_count, replace=False)
            selected.extend([low_perf_samples[i] for i in indices])

        return selected[:num_samples]

    def normalize_state_for_llm(self, state):
        """将环境状态归一化为LLM输入格式"""
        max_values = [500000, 100, 100, 100 * 1048576] + [100] * 8 + [3145729] * 4 + [1] * 4

        normalized = []
        for i, (val, max_val) in enumerate(zip(state, max_values)):
            norm_val = min(max(float(val) / max_val, 0), 1)
            normalized.append(round(norm_val, 3))

        return normalized

    def parse_llm_output(self, llm_output):
        """解析LLM输出为动作"""
        try:
            # 使用正则表达式提取数组
            pattern = r'\[([^\]]+)\]'
            matches = re.findall(pattern, llm_output)

            if matches:
                # 取最后一个匹配的数组（通常是最终答案）
                action_str = '[' + matches[-1] + ']'
                action = json.loads(action_str)

                # 确保动作维度正确
                if len(action) >= 4:
                    action = action[:4]
                else:
                    action = action + [0.5] * (4 - len(action))

                # 确保在[0,1]范围内
                action = np.clip(action, 0, 1)

                # 转换为环境需要的[-1,1]范围
                action = np.array(action) * 2 - 1

                return action
            else:
                raise ValueError("No valid action array found in LLM output")

        except Exception as e:
            print(f"Error parsing LLM output: {e}")
            print(f"LLM output: {llm_output}")
            # 返回中性动作
            return np.array([0.0, 0.0, 0.0, 0.0])

    def get_action(self, state):
        """获取LLM决策的动作"""
        # 归一化状态
        normalized_state = self.normalize_state_for_llm(state)

        # 构建提示词
        prompt = f"""{self.context_template}

Current state: {normalized_state}
Predict action:"""

        try:
            # 调用LLM
            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                top_p=0.7,
                max_tokens=4000,
                stream=False
            )

            llm_output = completion.choices[0].message.content
            print(f"LLM output: {llm_output}")  # 调试输出
            action = self.parse_llm_output(llm_output)

            return action

        except Exception as e:
            print(f"Error calling LLM: {e}")
            # 返回随机动作作为备选
            return np.random.uniform(-1, 1, 4)

    def run_single_episode(self, verbose=True):
        """运行单个episode"""
        state = self.env.reset()
        total_reward = 0
        step = 0

        if verbose:
            print(f"Initial UAV battery: {self.env.e_battery_uav:.0f}")
            print(f"Initial UAV location: {self.env.loc_uav}")
            print(f"UE locations: {self.env.loc_ue_list}")
            print(f"UE tasks: {self.env.task_list}")

        while step < self.env.slot_num:
            # LLM决策
            action = self.get_action(state)

            if verbose:
                print(f"\nStep {step}: LLM Action = {action}")

            # 执行动作
            next_state, reward, is_terminal, step_redo, offloading_change, reset_dist = self.env.step(action)

            if step_redo:
                if verbose:
                    print("  -> Step redo required")
                continue

            if reset_dist:
                if verbose:
                    print("  -> Distance reset due to boundary violation")

            if offloading_change:
                if verbose:
                    print("  -> Offloading ratio changed due to energy constraint")

            if verbose:
                print(f"  -> Reward: {reward:.3f}")
                print(f"  -> UAV battery: {self.env.e_battery_uav:.0f}")
                print(f"  -> UAV location: {self.env.loc_uav}")

            total_reward += reward
            state = next_state
            step += 1

            if is_terminal:
                if verbose:
                    print(f"Episode terminated at step {step}")
                break

        if verbose:
            print(f"Episode total reward: {total_reward:.3f}")

        return total_reward, step

    def evaluate(self, episodes=10):
        """评估LLM控制器性能"""
        print(f"Evaluating LLM controller for {episodes} episodes...")

        rewards = []
        steps = []

        for episode in range(episodes):
            print(f"\n=== Episode {episode + 1} ===")
            reward, step_count = self.run_single_episode(verbose=False)
            rewards.append(reward)
            steps.append(step_count)

            print(f"Episode {episode + 1}: Reward = {reward:.3f}, Steps = {step_count}")

        # 统计结果
        avg_reward = np.mean(rewards)
        std_reward = np.std(rewards)
        avg_steps = np.mean(steps)

        print(f"\n=== Evaluation Results ===")
        print(f"Average reward: {avg_reward:.3f}")
        print(f"Std deviation: {std_reward:.3f}")
        print(f"Max reward: {np.max(rewards):.3f}")
        print(f"Min reward: {np.min(rewards):.3f}")
        print(f"Average steps: {avg_steps:.1f}")

        return {
            'avg_reward': avg_reward,
            'std_reward': std_reward,
            'max_reward': np.max(rewards),
            'min_reward': np.min(rewards),
            'avg_steps': avg_steps,
            'all_rewards': rewards
        }


def compare_llm_vs_ddpg(api_key, training_data_file, episodes=10):
    """比较LLM和DDPG的性能"""
    from DDPG_train import test_trained_model

    print("=== LLM vs DDPG Performance Comparison ===")

    # 测试LLM性能
    print("\n1. Testing LLM Controller...")
    llm_controller = LLMUAVController(api_key, training_data_file=training_data_file)
    llm_results = llm_controller.evaluate(episodes)

    # 测试DDPG性能
    print("\n2. Testing DDPG Controller...")
    model_path = "ddpg_model_20250920_153558.pth"
    ddpg_results = test_trained_model(model_path, episodes)

    # 比较结果
    print(f"\n=== Comparison Results ===")
    print(f"LLM  - Average Reward: {llm_results['avg_reward']:.3f}")
    print(f"DDPG - Average Reward: {np.mean(ddpg_results):.3f}")
    print(f"Difference (LLM - DDPG): {llm_results['avg_reward'] - np.mean(ddpg_results):.3f}")

    if llm_results['avg_reward'] > np.mean(ddpg_results):
        print("LLM outperforms DDPG!")
    else:
        print("DDPG outperforms LLM")


def demo_llm_controller(api_key, training_data_file):
    """演示LLM控制器"""
    print("=== LLM Controller Demo ===")

    # 初始化控制器
    controller = LLMUAVController(api_key, training_data_file=training_data_file)

    # 运行演示
    controller.run_single_episode(verbose=True)


if __name__ == "__main__":
    # 配置参数 - 请将API密钥移到环境变量中
    import os

    API_KEY = os.getenv('OPENROUTER_API_KEY',
                        "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")

    # 自动查找训练数据文件
    import glob

    data_files = glob.glob("uav_llm_training_data_*.txt")

    TRAINING_DATA_FILE = "uav_llm_training_data_20250920_210918.txt"  # 备用文件名

    # 演示LLM控制器
    demo_llm_controller(API_KEY, TRAINING_DATA_FILE)