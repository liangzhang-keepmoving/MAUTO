# 多无人机边缘计算系统 DDPG 建模文档

------

## 1. 系统概述

### 1.1 系统架构

系统现在是有2个无人机和五个地面用户组成，所有的地面用户都会被无人机选中，一个无人机回复无语多个用户，无人机的计算能力会被用户平分，用户轨迹和在每个系统时刻的任务量大小是预先生成的

------

## 2. 状态空间建模

### 2.1 状态表示

系统状态由**无人机状态**和**用户状态**共同组成：

```python
state = {
    'uav_pos': Tensor[N, 2],      # N个无人机的2D位置
    'user_pos': Tensor[M, 2],     # M个用户的2D位置
    'user_tasks': Tensor[M, 1]    # M个用户的当前任务大小
}
```

### 2.2 状态维度说明

| 状态分量     | 维度  | 取值范围        | 物理含义                        |
| ------------ | ----- | --------------- | ------------------------------- |
| `uav_pos`    | N × 2 | [0, 1] (归一化) | 无人机在150m×150m区域的水平坐标 |
| `user_pos`   | M × 2 | [0, 1] (归一化) | 用户在地面的2D坐标              |
| `user_tasks` | M × 1 | [0, 1] (归一化) | 用户当前任务大小（0.5-1.0 MB）  |

**总状态维度**：`N × 2 + M × 3 = 2N + 3M`

### 2.3 归一化策略

```python
# 位置归一化
uav_pos_normalized = uav_pos / [area_length, area_width]
user_pos_normalized = user_pos / [area_length, area_width]

# 任务大小归一化
user_tasks_normalized = user_tasks / max_tas_size
```

------

## 3. 动作空间建模

### 3.1 动作输出结构

Actor网络输出**三类连续动作**：

```python
allocation, offloading, motion = actor(state)
```

### 3.2 动作详细说明

#### (1) 用户分配矩阵 `allocation`

**形状**：`[N, M]`
 **含义**：每个无人机对每个用户的竞争概率

```python
# 软分配（训练时）
allocation_soft = softmax(allocation_logits, dim=1)  # 在UAV维度

# 硬分配（执行时）
assigned_uav = argmax(allocation_soft, dim=1)  # 每列选最大
```

**分配规则**：每个用户被分配给概率最高的无人机。

------

#### (2) 卸载比例矩阵 `offloading`

**形状**：`[N, M]`
 **含义**：每个用户任务的卸载比例（0-1之间）

```python
offloading_ratio = sigmoid(offloading_raw)  # ∈ [0, 1]

# 实际卸载比例（考虑分配）
offloading_final = allocation × offloading_ratio
```

**任务分割**：

- 本地处理：`local_task = task_size × (1 - offloading_ratio)`
- 卸载处理：`offload_task = task_size × offloading_ratio`

------

#### (3) 运动控制 `motion`

**形状**：`[N, 2]`
 **含义**：每个无人机的移动参数 `[距离, 角度]`

```python
distance = sigmoid(motion[:, 0]) × max_speed × time_step  # ∈ [0, 25m]
angle = tanh(motion[:, 1]) × π                             # ∈ [0, 2π]

# 新位置计算
new_x = current_x + distance × cos(angle)
new_y = current_y + distance × sin(angle)
```

**边界处理**：使用clip函数限制在有效区域内。

------

### 3.3 动作空间总结

| 动作类型 | 维度  | 取值范围              | 激活函数       | 决策类型       |
| -------- | ----- | --------------------- | -------------- | -------------- |
| 用户分配 | N × M | [0, 1] (概率)         | Softmax        | 离散（硬分配） |
| 卸载比例 | N × M | [0, 1]                | Sigmoid        | 连续           |
| 运动控制 | N × 2 | 距离[0,1], 角度[-1,1] | Sigmoid + Tanh | 连续           |

**总动作维度**：`2NM + 2N`

------

## 4. 奖励函数设计

### 4.1 奖励函数结构

```python
reward = -(归一化能耗 + 归一化延迟)
```

**目标**：最小化系统总成本（能耗和延迟的加权和）

------

### 4.2 能耗建模

#### (1) 任务处理能耗

```python
task_energy = user_local_energy + user_transmission_energy + uav_computation_energy
```

**组成部分**：

| 能耗类型     | 计算公式            | 物理参数                                 |
| ------------ | ------------------- | ---------------------------------------- |
| 用户本地计算 | `P_local × T_local` | 功率=0.064W, 时间=任务量/用户的CPU频率   |
| 用户传输     | `P_tx × T_tx`       | 功率=0.1W, 时间=任务量/传输速率          |
| UAV计算      | `P_uav × T_uav`     | 功率=1.728W, 时间=任务量/平分后的UAV频率 |

**详细计算**：

```python
# 1. 本地计算能耗
T_local = task_size × cpu_cycles_per_mb / (user_cpu_frequency × 1000)
E_local = user_cpu_power × T_local  # = 0.4³ × T_local

# 2. 传输能耗
SNR = P_tx × L_0 / (distance³ × N_0)
Rate = bandwidth × log₂(1 + SNR)
T_tx = task_size / Rate
E_tx = user_transmission_power × T_tx  # = 0.1 × T_tx

# 3. UAV计算能耗
T_uav = task_size × cpu_cycles_per_mb / (uav_cpu_frequency × 1000)
E_uav = uav_cpu_power × T_uav  # = 1.2³ × T_uav
```

------

#### (2) UAV移动能耗

```python
movement_energy = 0.5 × m × v² × flight_time
```

**参数**：

- 质量 `m = 9.65 kg`
- 最大速度 `v = 20 m/s`
- 飞行时间 `flight_time = actual_distance / v`

**归一化基准**：2895 J

------

### 4.3 延迟建模

#### (1) 任务处理延迟

```python
total_delay = max(local_delay, offload_delay)  # 并行处理
```

**组成部分**：

```python
# 本地处理延迟
local_delay = local_task_size × cpu_cycles_per_mb / user_cpu_frequency

# 卸载延迟（串行）
offload_delay = transmission_delay + uav_computation_delay
```

**传输延迟**（Shannon公式）：

```python
SNR = P_tx × L_0 / (d² × N_0)
Rate = B × log₂(1 + SNR)
T_tx = task_size / Rate
```

**归一化基准**：4.6秒

------

#### (2) UAV移动延迟

```python
movement_delay = actual_distance / max_speed
```

**归一化基准**：1.6秒

------

### 4.4 总奖励公式

```python
reward = - (
    (task_energy / 5.0) + 
    (movement_energy / 2895.0) + 
    (max_delay / 4.6) + 
    (movement_delay / 1.6)
)
```

**权重设计**：通过归一化基准隐式平衡各项。

------

## 5. 环境动态建模

### 5.1 状态转移流程

```
当前状态 s_t
    ↓
执行动作 a_t (allocation, offloading, motion)
    ↓
┌──────────────────────────────────────┐
│ 1. 用户分配（竞争分配机制）           │
│ 2. 任务卸载（计算能耗和延迟）         │
│ 3. UAV移动（更新位置，计算移动能耗）  │
│ 4. 用户移动（随机游走/预定义轨迹）    │
│ 5. 生成新任务                        │
└──────────────────────────────────────┘
    ↓
下一状态 s_{t+1} + 奖励 r_t + 终止标志 done
```

------

### 5.2 关键约束

####  通信模型

**路径损耗模型**（简化）：

```python
# 三维距离
d_3d = sqrt(d_horizontal² + h_uav²)

# 信噪比
SNR = P_tx × L_0 / (d_3d² × N_0)

# 传输速率（Shannon容量）
Rate = B × log₂(1 + SNR)
```

**参数**：

- 发射功率：`P_tx = 0.1 W`
- 参考路径损耗：`L_0 = 1e-4`
- 噪声功率：`N_0 = 1e-13 W`
- 带宽：`B = 1 MHz`

------

### 5.3 用户移动模型

#### 模式1：随机游走

```python
angle = random.uniform(0, 2π)
distance = random.uniform(8, 12) m
new_pos = current_pos + distance × [cos(angle), sin(angle)]
```

#### 模式2：预定义轨迹

```python
# 从JSON文件加载
trajectory_data = load_json("user_trajectories.json")
user_pos[t] = trajectory_data[user_id][step]
```

------

### 5.4 终止条件

```python
done = (completed_task_size >= total_target_task_size)
```

**默认设置**：

- 目标任务总量：20步 × 5用户 × 7.5MB = 750 MB
- 最大步数：25步

------

## 6. DDPG网络架构

### 6.1 Actor网络

#### 网络结构

```
输入: state (uav_pos, user_pos, user_tasks)
    ↓
┌─────────────────────────────────────┐
│ UAV特征编码器                        │
│ Linear(2 → 64) → ReLU → Linear(64 → 128) │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│ User特征编码器                       │
│ Linear(3 → 64) → ReLU → Linear(64 → 128) │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│ 全局特征融合                         │
│ Concat → Linear(256 → 512) → ReLU   │
│        → Linear(512 → 256) → ReLU   │
└─────────────────────────────────────┘
    ↓
┌──────────┬──────────┬──────────┐
│ 分配头    │ 卸载头    │ 运动头    │
│ 256→256  │ 256→256  │ 256→128  │
│ →ReLU    │ →ReLU    │ →ReLU    │
│ →N×M     │ →N×M     │ →N×2     │
│ (Softmax)│ (Sigmoid)│(Sig+Tanh)│
└─────────┴──────────┴──────────┘
```

------

### 6.2 Critic网络

#### 网络结构

```
输入: state + actions (allocation, offloading, motion)
    ↓
┌────────────────────────────────┐
│ 状态编码器                      │
│ Linear(state_dim → 256) → ReLU │
│ → Linear(256 → 128)            │
└────────────────────────────────┘
    ↓
┌────────────────────────────────┐
│ 动作编码器（三个分支）          │
│ allocation: Linear(N×M → 128)  │
│ offloading: Linear(N×M → 128)  │
│ motion: Linear(N×2 → 64)       │
└────────────────────────────────┘
    ↓
┌────────────────────────────────┐
│ Q值融合网络                     │
│ Concat(128+128+128+64) → 448   │
│ → Linear(448 → 256) → ReLU     │
│ → Linear(256 → 128) → ReLU     │
│ → Linear(128 → 1)  [Q值输出]   │
└────────────────────────────────┘
```



------

### 6.3 目标网络

使用**软更新**机制：

```python
θ_target ← τ × θ + (1 - τ) × θ_target
```

**参数**：`τ = 0.005`（更新速度）

------

## 7. 训练算法流程

### 7.1 DDPG算法伪代码

```
初始化:
    - Actor网络 μ(s|θ^μ) 和目标网络 μ'
    - Critic网络 Q(s,a|θ^Q) 和目标网络 Q'
    - 经验回放缓冲区 D (容量100,000)
    - OU噪声过程 N

For episode = 1 to M:
    重置环境状态 s_1
    重置噪声 N
    
    For t = 1 to T:
        # 1. 选择动作（探索）
        a_t = μ(s_t|θ^μ) + N_t
        
        # 2. 执行动作
        执行 a_t, 观察 r_t 和 s_{t+1}
        
        # 3. 存储经验
        将 (s_t, a_t, r_t, s_{t+1}) 存入 D
        
        # 4. 训练网络（采样批次）
        If |D| >= batch_size:
            从 D 随机采样 minibatch
            
            # 更新Critic
            y_i = r_i + γ × Q'(s_{i+1}, μ'(s_{i+1}|θ^{μ'})|θ^{Q'})
            L = (1/N) Σ(y_i - Q(s_i, a_i|θ^Q))²
            最小化 L 更新 θ^Q
            
            # 更新Actor
            ∇_θ^μ J ≈ (1/N) Σ ∇_a Q(s,a|θ^Q)|_{a=μ(s)} × ∇_θ^μ μ(s|θ^μ)
            梯度上升更新 θ^μ
            
            # 软更新目标网络
            θ^{Q'} ← τθ^Q + (1-τ)θ^{Q'}
            θ^{μ'} ← τθ^μ + (1-τ)θ^{μ'}
```

------

### 7.2 关键训练技巧

#### (1) OU噪声探索

```python
class OUNoise:
    def sample(self):
        dx = θ × (μ - x) + σ × randn()
        x = x + dx
        return x
```

**参数**：`θ=0.15`, `σ=0.2`

------

#### (2) 经验回放

```python
class ReplayBuffer:
    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)
    
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
```

**作用**：打破数据相关性，提高样本利用率。

------

#### (3) 梯度裁剪

```python
torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
torch.nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
```

**目的**：防止梯度爆炸，稳定训练。

------

### 7.3 超参数配置

| 参数         | 值    | 单位 | 说明         |
| ------------ | ----- | ---- | ------------ |
| 带宽         | 1     | MHz  | 信道带宽     |
| 发射功率     | 0.1   | W    | 用户上行功率 |
| 参考路径损耗 | 1e-4  | -    | 参考距离损耗 |
| 噪声功率     | 1e-13 | W    | 背景噪声     |
| 路径损耗指数 | 2.0   | -    | 简化模型     |

------

### 9.5 任务参数

| 参数         | 值   | 单位          | 说明       |
| ------------ | ---- | ------------- | ---------- |
| CPU周期数    | 1000 | Megacycles/MB | 计算复杂度 |
| 最小任务大小 | 0.5  | MB            | 任务下界   |
| 最大任务大小 | 1.0  | MB            | 任务上界   |
| 目标总任务量 | 15   | MB            | 终止条件   |



### 12.1 环境配置

```bash
# 1. 安装依赖
pip install torch numpy matplotlib

# 2. 项目结构
multi_uav_latest/
├── DDPG.py                    # DDPG智能体
├── env_simplified.py          # 环境模块
├── delay_calculator.py        # 延迟计算
├── task_energy_calculator.py  # 能耗计算
├── uav_movement.py            # UAV移动
├── user_allocation.py         # 用户分配
├── generate_trajectory.py     # 轨迹生成
├── train_ddpg.py              # 训练脚本
└── user_trajectories.json     # 预生成轨迹
```

------

### 

### 8.1 用户分配模块 (`user_allocation.py`)

**功能**：处理多UAV竞争分配机制

```python
class UserAllocationManager:
    def process_uav_actions_with_conflict_resolution(self, actions):
        # 构建竞争矩阵 [N, M]
        competition_matrix = extract_probs(actions)
        
        # 每个用户选择概率最高的UAV
        for user_id in range(M):
            assigned_uav = argmax(competition_matrix[:, user_id])
            user_assignments[user_id] = assigned_uav
        
        return user_assignments
```

------

### 8.2 延迟计算模块 (`delay_calculator.py`)

**功能**：计算任务处理延迟

```python
class DelayCalculator:
    def calculate_task_delays(self, actions, user_states, uav_states, user_assignments):
        # 1. 计算本地处理延迟
        local_delay = task_size × cycles / cpu_freq
        
        # 2. 计算卸载延迟
        transmission_delay = task_size / transmission_rate
        uav_computation_delay = task_size × cycles / uav_freq
        offload_delay = transmission_delay + uav_computation_delay
        
        # 3. 总延迟（并行）
        total_delay = max(local_delay, offload_delay)
        
        return delays
```

------

### 8.3 能耗计算模块 (`task_energy_calculator.py`)

**功能**：计算任务处理能耗

```python
class TaskEnergyCalculator:
    def calculate_task_processing_energy(self, actions, user_states, uav_states, user_assignments):
        # 1. 用户本地处理能耗
        local_energy = user_cpu_power × local_computation_time
        
        # 2. 用户传输能耗
        transmission_energy = tx_power × transmission_time
        
        # 3. UAV计算能耗
        uav_energy = uav_cpu_power × uav_computation_time
        
        return energies
```

------

### 8.4 UAV移动模块 (`uav_movement.py`)

**功能**：处理UAV移动、边界约束和碰撞检测

```python
class UAVMovementManager:
    def process_uav_movements(self, actions, uav_states):
        # 1. 解析移动动作
        direction, distance = extract_motion(actions)
        
        # 2. 计算新位置
        new_pos = current_pos + distance × [cos(θ), sin(θ)]
        
        # 3. 边界约束
        new_pos = clip(new_pos, bounds)
        
        # 4. 碰撞检测
        collision_penalty = check_collisions(uav_states)
        
        # 5. 计算移动能耗
        energy = 0.5 × m × v² × time
        
        return penalties, energies
```

------

### 8.5 环境模块 (`env_simplified.py`)

**功能**：封装完整的MDP环境

```python
class SimplifiedMultiUAVEnvironment:
    def step(self, actions):
        # 1. 用户分配
        user_assignments = self.allocate_users(actions)
        
        # 2. UAV移动
        penalties, movement_energy = self.move_uavs(actions)
        
        # 3. 计算延迟和能耗
        delays = self.calculate_delays(actions, user_assignments)
        task_energy = self.calculate_task_energy(actions, user_assignments)
        
        # 4. 计算奖励
        reward = self.compute_reward(delays, task_energy, movement_energy)
        
        # 5. 更新用户状态
        self.update_users()
        
        # 6. 判断终止
        done = self.check_termination()
        
        return next_state, reward, done, info
```

------

## 9. 系统参数配置

### 9.1 环境参数

| 参数     | 值      | 单位 | 说明         |
| -------- | ------- | ---- | ------------ |
| 区域大小 | 150×150 | 米   | 二维平面区域 |
| UAV高度  | 100     | 米   | 固定飞行高度 |
| UAV数量  | 2-3     | 个   | 可配置       |
| 用户数量 | 5-6     | 个   | 可配置       |
| 时间步长 | 5.0     | 秒   | 决策间隔     |

------

### 9.2 UAV参数

| 参数     | 值    | 单位 | 说明         |
| -------- | ----- | ---- | ------------ |
| CPU频率  | 1.2   | GHz  | 计算能力     |
| CPU功耗  | 1.728 | W    | 1.2³         |
| 最大速度 | 20    | m/s  | 飞行速度     |
| 质量     | 9.65  | kg   | 用于能耗计算 |
| 电池容量 | 10000 | J    | 能量上限     |

------

### 9.3 用户参数

| 参数     | 值      | 单位 | 说明         |
| -------- | ------- | ---- | ------------ |
| CPU频率  | 0.4     | GHz  | 本地计算能力 |
| CPU功耗  | 0.064   | W    | 0.4³         |
| 传输功率 | 0.1     | W    | 上行传输     |
| 最大速度 | 2.0     | m/s  | 移动速度     |
| 任务大小 | 0.5-1.0 | MB   | 随机生成     |

