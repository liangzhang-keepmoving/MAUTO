"""
基于大语言模型的多无人机边缘计算控制器（修改版 - 加载单个回合数据）
LLM-based Multi-UAV Edge Computing Controller (Modified - Load Single Episode)
"""

import json
import numpy as np
import re
import os
from openai import OpenAI
from env_simplified import SimplifiedMultiUAVEnvironment


class LLMMultiUAVController:
    """LLM多无人机控制器"""

    def __init__(self, api_key, num_uavs=2, num_users=5,
                 model_name="google/gemini-2.0-flash-exp:free",
                 training_data_file=None):
        """
        初始化LLM控制器

        Args:
            api_key: OpenRouter API密钥
            num_uavs: 无人机数量
            num_users: 用户数量
            model_name: 使用的模型名称
            training_data_file: 训练数据文件路径（单个回合的JSONL文件）
        """
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key
        )
        self.model_name = model_name
        self.num_uavs = num_uavs
        self.num_users = num_users

        # 创建环境
        self.env = SimplifiedMultiUAVEnvironment(
            num_uavs=num_uavs,
            num_users=num_users,
            trajectory_file="user_trajectories.json"
        )

        # 加载训练数据
        self.training_data = []
        if training_data_file:
            self.load_training_data(training_data_file)

        # 构建上下文模板
        self.context_template = self.build_context_template()

    def load_training_data(self, file_path):
        """
        加载单个回合的JSONL训练数据
        每一行是一个step的数据
        """
        try:
            with open(file_path, 'r') as f:
                for line in f:
                    if line.strip():
                        data = json.loads(line.strip())
                        self.training_data.append(data)

            print(f"✓ 加载了 {len(self.training_data)} 条训练样本")

            # 显示回合信息
            if len(self.training_data) > 0:
                episode_id = self.training_data[0].get('episode', 'unknown')
                print(f"  Episode: {episode_id}")
                print(f"  步数: {len(self.training_data)}")

                # 显示奖励范围
                rewards = [s.get('reward', 0) for s in self.training_data]
                print(f"  奖励范围: [{min(rewards):.3f}, {max(rewards):.3f}]")
                print(f"  平均奖励: {np.mean(rewards):.3f}")

        except FileNotFoundError:
            print(f"✗ 训练数据文件 {file_path} 未找到")
        except Exception as e:
            print(f"✗ 加载训练数据时出错: {e}")

    def build_context_template(self):
        """构建增强版LLM上下文模板（基于实际环境参数）"""

        # 使用所有样本（如果样本太多，随机采样）
        if len(self.training_data) > 20:
            indices = np.random.choice(len(self.training_data), 20, replace=False)
            selected_samples = [self.training_data[i] for i in sorted(indices)]
        else:
            selected_samples = self.training_data

        # 构建轨迹示例
        examples_text = self._build_trajectory_examples(selected_samples)

        template = f"""You are an intelligent multi-UAV edge computing decision system controller.
Your goal is to minimize total cost (task energy + transmission energy + delay) while serving all users.

# ==================== ENVIRONMENT PARAMETERS ====================

## Physical Setup:
- Map size: 400m × 400m
- UAV altitude: 50m (fixed)
- Time step: 5.0 seconds per s
- Coordinates: Normalized [0,1] (multiply by 400 for actual meters)
- {self.num_uavs} UAVs, {self.num_users} users

## UAV Specifications:
- CPU frequency: 0.8 GHz (800 MHz)
- CPU power: (0.8)³ = 0.512 W
- CPU cycles per MB: 1000 Megacycles
- Max speed: 15 m/s
- Max movement per step: 15 m/s × 2s = 30m (but normalized to [0,1])

## User Specifications:
- CPU frequency: 0.4 GHz (400 MHz) - **Half of UAV power**
- CPU power: (0.4)³ = 0.064 W
- Task size: 0.5 - 1.0 MB per task
- Transmission power: 0.1 W
- **Movement: FIXED predictable trajectories** (critical insight!)

## Channel Model:
- Bandwidth: 3 MHz
- Transmission power: 0.1 W
- Path loss exponent: 2 (urban environment)
- Reference distance: 1.0 m
- Reference path loss: 1e-5
- Noise power: 1e-10 W (very low noise)

## Normalization Ranges (IMPORTANT for understanding costs):
**Task Energy:**
- Min: 0 W, Max: 3.5 W
- Normalized formula: (actual - 0) / 3.5

**Transmission Energy:**
- Min: 0 W, Max: 0.6 W  
- Normalized formula: (actual - 0) / (0.6 × 5) = actual / 3.0
- Note: Max is multiplied by 5 in normalization

**Movement Energy:**
- Min: 1737 W, Max: 2171.2 W (very high!)
- Normalized formula: (actual - 1737) / (2171.2 - 1737) = (actual - 1737) / 434.2
- **Key insight: Movement is VERY expensive** (~2000W vs <5W for tasks)
- Note: Movement energy NOT directly included in reward (affects future steps)

**Computation Delay:**
- Min: 0s, Max: 8.37s
- Normalized formula: (actual - 0) / 8.37

**Transmission Delay:**
- Min: 0s, Max: 2.87s
- Normalized formula: (actual - 0) / 2.87

**Flight Delay:**
- Min: 0s, Max: 2s
- Normalized formula: (actual - 0) / 2.0

# ==================== REWARD FUNCTION (CRITICAL!) ====================

## How Reward is Calculated:

**For EACH UAV, calculate 3 normalized components:**

1. **normalized_task_energy** = (task_energy - 0) / 3.5
   - task_energy includes both local processing (user) and remote processing (UAV)
   - Range: [0, 1]

2. **normalized_transmission_energy** = (transmission_energy - 0) / 3.0
   - Energy for transmitting offloaded data from user to UAV
   - Range: [0, 1]
   - Note: Normalized by 3.0 (= 0.6 × 5), not by 0.6

3. **normalized_delay** = (total_delay - 0) / 8.37
   - total_delay = computation_delay + transmission_delay
   - Range: [0, 1]

**Sum these 3 components for each UAV:**
```
uav_cost = normalized_task_energy + normalized_transmission_energy + normalized_delay
```

**Total reward (for all UAVs):**
```
reward = -(uav_0_cost + uav_1_cost + ... + uav_N_cost)
```

**Important:** The reward is NEGATIVE (cost minimization)
- Better performance → reward closer to 0 (e.g., -0.5)
- Worse performance → more negative reward (e.g., -2.5)

## What This Means for Decision-Making:

**Each normalized component has equal weight in the final reward:**
- Task energy contributes: [0, 1]
- Transmission energy contributes: [0, 1]  
- Delay contributes: [0, 1]
- Total cost per UAV: [0, 3]

**Example calculation:**
```
UAV 0:
  - Task energy: 2.0W → normalized: 2.0/3.5 = 0.571
  - Transmission: 0.3W → normalized: 0.3/3.0 = 0.100
  - Delay: 4.0s → normalized: 4.0/8.37 = 0.478
  - Total: 0.571 + 0.100 + 0.478 = 1.149

UAV 1:
  - Task energy: 1.5W → normalized: 1.5/3.5 = 0.429
  - Transmission: 0.2W → normalized: 0.2/3.0 = 0.067
  - Delay: 3.5s → normalized: 3.5/8.37 = 0.418
  - Total: 0.429 + 0.067 + 0.418 = 0.914

Final Reward: -(1.149 + 0.914) = -2.063
```

## Optimization Strategy Based on Reward Function:

Since all 3 components are equally weighted (each [0,1]), you should:

1. **Balance across all metrics** - Don't optimize just one at the expense of others
2. **Task energy optimization:**
   - Balance local vs UAV processing
   - Consider that UAV CPU is 2x faster but 8x more power-hungry
   - Typical sweet spot: 0.4-0.6 offload ratio

3. **Transmission energy optimization:**
   - Keep reasonable distance from assigned users
   - Path loss ∝ distance² (not too sensitive)
   - 100-200m distance is acceptable

4. **Delay optimization:**
   - Faster UAV processing helps (2x speedup)
   - But transmission adds delay (depends on distance)
   - Balance processing location vs transmission overhead
1. **Task Processing Energy**: 
   - User processes (1 - offload_ratio) locally: 0.064W CPU power
   - UAV processes offload_ratio remotely: 0.512W CPU power (8x more powerful!)
   - Energy = CPU_power × processing_time
   - Range: 0 - 3.5W per UAV

2. **Transmission Energy**:
   - User transmits offloaded data to UAV
   - Energy = transmission_power × transmission_time
   - Depends on distance (path loss = distance²)
   - Range: 0 - 0.6W per UAV
   - **Normalized by factor of 5** (max can be 3.0W)

3. **Delay**:
   - Computation delay: task_size / CPU_frequency
   - Transmission delay: task_size / data_rate (depends on distance)
   - Range: 0 - 8.37s computation, 0 - 2.87s transmission

4. **Movement Energy (NOT directly in reward, but affects future steps):**
   - Range: 1737 - 2171.2W (EXTREMELY HIGH!)
   - Movement is about **500x more expensive** than task processing
   - **Critical strategy: Minimize movement!**

# ==================== KEY INSIGHTS ====================

## Cost Comparison (Relative Scale):
```
Movement:     1737-2171W  (×500 baseline)  ← AVOID excessive movement
Task:         0-3.5W      (×1 baseline)
Transmission: 0-0.6W      (×0.17 baseline)
Delay:        Weighted equally with energy in final reward
```

## Strategic Principles:

1. **Movement Strategy (CRITICAL):**
   - Movement is ~500x more expensive than task processing!
   - **DO NOT move unless absolutely necessary**
   - Small movements (5-10m) are OK, large movements (>20m) are very costly
   - Early steps (0-3): Move to establish good position (10-20m max)
   - Later steps (4+): **Stay still or move minimally** (<10m)
   - Typical pattern: Move early, stabilize quickly

2. **User Assignment:**
   - Assign nearby users to minimize transmission distance
   - Each user → EXACTLY ONE UAV (sum of assignments = 1)
   - Balance load: 2-3 users per UAV for {self.num_uavs} UAVs, {self.num_users} users
   - Prefer stable assignments (don't swap frequently)

3. **Offloading Decision:**
   - UAV CPU is 2x faster (0.8 GHz vs 0.4 GHz)
   - UAV CPU power is 8x higher (0.512W vs 0.064W)
   - Tradeoff: Faster processing vs higher power consumption
   - **Typical sweet spot: 0.4-0.6 offload ratio**
   - If user very close (<50m): Can offload more (0.5-0.7)
   - If user far (>150m): Offload less (0.3-0.5) due to transmission cost

4. **Proactive Positioning (Users have FIXED paths):**
   - Users follow predictable trajectories
   - Position UAVs where users WILL BE, not where they ARE
   - But remember: movement is expensive, so don't chase constantly
   - Find a compromise position that works for multiple steps

## Typical Good Episode Pattern:
- **Rewards: -0.5 to -2.5 per step** (lower = better, closer to 0)
- **Total episode reward: -10 to -50** for 20 steps
- **Movement pattern:**
  - Step 0-3: Establish position (10-20m moves)
  - Step 4-10: Minimal adjustment (5-10m moves)
  - Step 11+: Nearly stationary (0-5m moves)
- **Offload ratios: 0.45-0.60 range** (balanced approach)
- **User assignments: Stable** (same users throughout)

## Common Mistakes to Avoid:
❌ Moving too much (movement >> task energy)
❌ Chasing user movements every step
❌ Extreme offload ratios (0.0 or 1.0)
❌ Frequently changing user assignments

✅ Establish good position early, then stay put
✅ Accept slightly suboptimal distances to avoid movement
✅ Balanced offload ratios (0.4-0.6)
✅ Stable user assignments

# ==================== DEMONSTRATED TRAJECTORY ====================

{examples_text}

# ==================== YOUR TASK ====================

## Output Format (JSON ONLY):
{{
  "uav_0": {{
    "user_assignments": [{self.num_users} binary values: 0 or 1],
    "offloading_ratios": [{self.num_users} floats: 0.0-1.0],
    "movement_direction": float (radians 0-6.28),
    "movement_distance": float (0.0-1.0, will be scaled by actual movement constraints)
  }},
  "uav_1": {{
    "user_assignments": [{self.num_users} binary values: 0 or 1],
    "offloading_ratios": [{self.num_users} floats: 0.0-1.0],
    "movement_direction": float (radians 0-6.28),
    "movement_distance": float (0.0-1.0)
  }}
}}

## Critical Rules:
1. ✓ Each user → EXACTLY ONE UAV: Σ(assignments) = 1 per user
2. ✓ Only offload for assigned users: assignment[i]=0 → offload[i]=0
3. ✓ **MINIMIZE MOVEMENT** - Movement costs 500x more than processing!
4. ✓ Consider FUTURE positions (users follow fixed paths)
5. ✓ Balanced offload ratios: typically 0.4-0.6

## Decision Guidelines:
- **Early steps (0-3)?** → Move to good position (10-20m), establish user assignments
- **Later steps (4+)?** → **STAY PUT** (0-5m movements), maintain assignments
- **Users clustered?** → Assign cluster to UAV, position at cluster center, then stay
- **Users dispersed?** → Assign 2-3 per UAV, find compromise position, then stay
- **Movement cost reminders:** 
  - 0m move = 1737W (baseline hovering)
  - 10m move ≈ 1850W (+113W)
  - 20m move ≈ 2000W (+263W)
  - 30m move ≈ 2171W (+434W, MAX)
- **Offload ratio:** Start with 0.5, adjust ±0.1 based on distance

**KEY STRATEGY: Find good position early, then minimize movement!**

Predict actions now:
"""
        return template

    def _build_trajectory_examples(self, selected_samples):
        """构建轨迹示例文本"""
        examples_text = ""

        # 按step分组
        steps_by_num = {}
        for sample in selected_samples:
            step_num = sample.get('step', 0)
            if step_num not in steps_by_num:
                steps_by_num[step_num] = []
            steps_by_num[step_num].append(sample)

        # 如果是完整轨迹
        if len(steps_by_num) > 5:
            examples_text += "Complete Episode (showing sequential patterns):\n\n"

            for step_num in sorted(steps_by_num.keys())[:12]:  # 前12步
                sample = steps_by_num[step_num][0]
                state = sample['state']
                action = sample['action']
                reward = sample.get('reward', 0)

                uav_pos = np.array(state['uav_pos'])
                user_pos = np.array(state['user_pos'])

                examples_text += f"Step {step_num} (R={reward:.3f}): "

                # UAV位置
                for i, p in enumerate(uav_pos):
                    examples_text += f"UAV{i}@({p[0] * 400:.0f},{p[1] * 400:.0f})m "

                # 动作摘要
                for uav_id in range(len(action)):
                    uav_key = f'uav_{uav_id}'
                    if uav_key in action:
                        uav_act = action[uav_key]
                        assigned = [i for i, v in enumerate(uav_act['user_assignments']) if v > 0.5]
                        dist = uav_act['movement_distance']
                        examples_text += f"| UAV{uav_id}→users{assigned}, move{dist:.0f}m "

                examples_text += "\n"

            examples_text += "\nPattern: UAVs move to good positions early, then track users with moderate movements.\n\n"

        else:
            # 独立样本
            for i, sample in enumerate(selected_samples[:8]):
                state_str = self.format_state_for_prompt(sample['state'])
                action_str = self.format_action_for_prompt(sample['action'])
                examples_text += f"Example {i + 1}: State={state_str} → Action={action_str}\n"

        return examples_text

    def format_state_for_prompt(self, state):
        """将状态格式化为提示词字符串"""
        if isinstance(state, dict):
            # 从字典提取状态
            uav_pos = state['uav_pos']
            user_pos = state['user_pos']
            user_tasks = state['user_tasks']

            state_list = []
            # UAV位置
            for i in range(len(uav_pos)):
                state_list.extend([round(float(uav_pos[i][0]), 3),
                                   round(float(uav_pos[i][1]), 3)])
            # 用户位置和任务
            for i in range(len(user_pos)):
                state_list.extend([round(float(user_pos[i][0]), 3),
                                   round(float(user_pos[i][1]), 3),
                                   round(float(user_tasks[i][0]), 3)])

            return str(state_list)
        else:
            # 已经是列表或数组
            return str([round(float(x), 3) for x in state])

    def format_action_for_prompt(self, action):
        """将动作格式化为提示词字符串"""
        if isinstance(action, dict):
            # 动作已经是字典格式
            formatted = {}
            for uav_key, uav_action in action.items():
                formatted[uav_key] = {
                    'user_assignments': [round(float(x), 2) for x in uav_action['user_assignments']],
                    'offloading_ratios': [round(float(x), 2) for x in uav_action['offloading_ratios']],
                    'movement_direction': round(float(uav_action['movement_direction']), 2),
                    'movement_distance': round(float(uav_action['movement_distance']), 2)
                }
            return json.dumps(formatted, ensure_ascii=False)
        else:
            return str(action)

    def normalize_state_for_llm(self, state):
        """将环境状态归一化（状态已在环境中归一化）"""
        return self.format_state_for_prompt(state)

    def parse_llm_output(self, llm_output):
        """解析LLM输出为环境动作"""
        try:
            # 提取JSON部分
            json_match = re.search(r'\{.*\}', llm_output, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                actions_dict = json.loads(json_str)

                # 转换为环境格式
                env_actions = {}
                for uav_id in range(self.num_uavs):
                    uav_key = f'uav_{uav_id}'
                    if uav_key in actions_dict:
                        uav_action = actions_dict[uav_key]

                        # 提取并转换动作
                        user_assignments = np.array(uav_action['user_assignments'], dtype=np.float32)
                        offloading_ratios = np.array(uav_action['offloading_ratios'], dtype=np.float32)
                        movement_direction = float(uav_action['movement_direction'])
                        movement_distance = float(uav_action['movement_distance'])

                        # 限制范围
                        user_assignments = np.clip(user_assignments, 0, 1)
                        offloading_ratios = np.clip(offloading_ratios, 0, 1)
                        movement_direction = np.clip(movement_direction, 0, 2 * np.pi)
                        movement_distance = np.clip(movement_distance, 0, 1)

                        env_actions[uav_key] = {
                            'user_competition_probs': user_assignments,
                            'offloading_ratios': offloading_ratios,
                            'movement_direction': movement_direction,
                            'movement_distance': movement_distance * 30  # 转换为实际距离
                        }

                # 检查是否所有UAV都有动作
                if len(env_actions) == self.num_uavs:
                    return env_actions
                else:
                    raise ValueError(f"缺少某些UAV的动作")

            else:
                raise ValueError("未找到有效的JSON格式")

        except Exception as e:
            print(f"✗ 解析LLM输出失败: {e}")
            print(f"LLM输出: {llm_output[:500]}")
            # 返回默认动作
            return self._get_default_actions()

    def _get_default_actions(self):
        """获取默认动作（当解析失败时使用）"""
        env_actions = {}
        for uav_id in range(self.num_uavs):
            uav_key = f'uav_{uav_id}'

            # 简单策略：每个UAV负责连续的用户
            users_per_uav = self.num_users // self.num_uavs
            start_user = uav_id * users_per_uav
            end_user = start_user + users_per_uav if uav_id < self.num_uavs - 1 else self.num_users

            user_assignments = np.zeros(self.num_users)
            user_assignments[start_user:end_user] = 1.0

            env_actions[uav_key] = {
                'user_competition_probs': user_assignments,
                'offloading_ratios': np.ones(self.num_users) * 0.5,
                'movement_direction': 0.0,
                'movement_distance': 15.0
            }

        return env_actions

    def get_action(self, state):
        """获取LLM决策的动作"""
        # 格式化状态
        state_str = self.normalize_state_for_llm(state)

        # 构建提示词
        prompt = f"""{self.context_template}

Current State: {state_str}

Predict the optimal actions for all {self.num_uavs} UAVs:"""

        try:
            # 调用LLM
            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                top_p=0.8,
                max_tokens=10000,
                stream=False
            )

            llm_output = completion.choices[0].message.content
            print(f"\n--- LLM输出 ---")
            print(llm_output[:300] + "..." if len(llm_output) > 300 else llm_output)

            # 解析输出
            actions = self.parse_llm_output(llm_output)
            return actions

        except Exception as e:
            print(f"✗ 调用LLM时出错: {e}")
            return self._get_default_actions()

    def run_single_episode(self, verbose=True, max_steps=25):
        """运行单个episode"""
        state = self.env.reset()
        total_reward = 0
        step = 0
        episode_task_energy = 0
        episode_transmission_energy = 0
        episode_movement_energy = 0
        uav_delays = [0] * 2
        if verbose:
            print(f"\n{'=' * 60}")
            print(f"开始新的Episode")
            print(f"{'=' * 60}")
            print(f"初始UAV位置: {self.env.uav_states}")
            print(f"初始用户位置和任务: {self.env.user_states}")


        while step < max_steps:
            if verbose:
                print(f"\n--- Step {step} ---")

            # LLM决策
            actions = self.get_action(state)

            # 执行动作
            next_state, reward, done, info = self.env.step(actions)

            raw_metrics = info.get('raw_metrics', {})

            # 累加总体能耗
            episode_task_energy += sum([raw_metrics[f'uav_{i}']['actual_task_energy_raw'] for i in range(2)])
            episode_transmission_energy += sum(
                [raw_metrics[f'uav_{i}']['actual_transmission_energy_raw'] for i in range(2)])
            episode_movement_energy += sum(
                [raw_metrics[f'uav_{i}']['uav_movement_energy_raw'] for i in range(2)])
            for uav_id in range(2):
                uav_delays[uav_id] += raw_metrics[f'uav_{uav_id}']['delay_raw']


            if verbose:
                print(f"奖励: {reward:.3f}")
                print(f"UAV位置: {self.env.uav_states}")
                print(f"用户分配: {info.get('user_assignments', {})}")

            total_reward += reward
            state = next_state
            step += 1

            if done:
                if verbose:
                    print(f"\nEpisode在第{step}步结束")
                break

        if verbose:
            print(f"\n{'=' * 60}")
            print(f"Episode总奖励: {total_reward:.3f}")
            print(f"{'=' * 60}")

        return total_reward, step ,episode_task_energy,episode_transmission_energy,episode_movement_energy,uav_delays

    def evaluate(self, episodes=5):
        """评估LLM控制器性能"""
        print(f"\n{'=' * 60}")
        print(f"评估LLM控制器 - {episodes} episodes")
        print(f"{'=' * 60}")

        rewards = []
        steps_list = []

        for episode in range(episodes):
            print(f"\n{'=' * 60}")
            print(f"Episode {episode + 1}/{episodes}")
            print(f"{'=' * 60}")

            reward, steps,episode_task_energy,episode_transmission_energy,episode_movement_energy,uav_delays = self.run_single_episode(verbose=True)
            rewards.append(reward)
            steps_list.append(steps)

            print(f"Reward: {reward:.3f}, Steps: {steps}, "
                  f"Task Energy: {episode_task_energy:.3f}, "
                  f"Transmission Energy: {episode_transmission_energy:.3f}, "
                  f"Movement Energy: {episode_movement_energy:.3f}, "
                  f"UAV Delays: {uav_delays}")

        # 统计结果
        avg_reward = np.mean(rewards)
        std_reward = np.std(rewards)
        avg_steps = np.mean(steps_list)

        print(f"\n{'=' * 60}")
        print(f"评估结果汇总")
        print(f"{'=' * 60}")
        print(f"平均奖励: {avg_reward:.3f}")
        print(f"标准差: {std_reward:.3f}")
        print(f"最大奖励: {np.max(rewards):.3f}")
        print(f"最小奖励: {np.min(rewards):.3f}")
        print(f"平均步数: {avg_steps:.1f}")
        print(f"{'=' * 60}")

        return {
            'avg_reward': avg_reward,
            'std_reward': std_reward,
            'max_reward': np.max(rewards),
            'min_reward': np.min(rewards),
            'avg_steps': avg_steps,
            'all_rewards': rewards
        }


def demo_llm_controller():
    """演示LLM控制器"""
    print("\n" + "=" * 60)
    print("LLM多无人机控制器演示（加载单个回合数据）")
    print("=" * 60)

    # API密钥（建议使用环境变量）
    API_KEY = os.getenv('OPENROUTER_API_KEY', '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')

    if API_KEY == 'your-api-key-here':
        print("⚠️  警告：请设置OPENROUTER_API_KEY环境变量")
        return

    # 训练数据文件（修改为你的文件名）
    training_data_file = 'best_samples.jsonl'  # ⭐ 修改为你的文件名

    # 检查文件是否存在
    if not os.path.exists(training_data_file):
        print(f"✗ 训练数据文件不存在: {training_data_file}")
        print(f"当前目录: {os.getcwd()}")
        print(f"请确保文件在当前目录下")
        return

    # 初始化控制器
    controller = LLMMultiUAVController(
        api_key=API_KEY,
        num_uavs=2,
        num_users=5,
        model_name="google/gemini-2.5-pro",
        training_data_file=training_data_file  # ⭐ 加载单个回合的数据
    )

    # 评估性能
    results = controller.evaluate(episodes=1)

    return results


if __name__ == "__main__":
    # 运行演示
    demo_llm_controller()