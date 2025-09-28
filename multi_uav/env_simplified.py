import math
import random
import numpy as np
from typing import Dict, List

# 导入模块化组件
from user_allocation import UserAllocationManager
from uav_movement import UAVMovementManager
from delay_calculator import DelayCalculator
from task_energy_calculator import TaskEnergyCalculator



class SimplifiedMultiUAVEnvironment:
    """简化的多无人机多用户边缘计算环境 - 只考虑能耗奖励"""
    
    def __init__(self, num_uavs=3, num_users=6):
        """
        初始化简化多无人机环境
        
        Args:
            num_uavs: 无人机数量
            num_users: 用户数量
        """
        self.max_delay_test = 0
        self.max_energy_test = 0

        # 基础参数
        self.num_uavs = num_uavs
        self.num_users = num_users
        self.current_step = 0
        
        # 区域参数
        self.area_length = 500  # 米
        self.area_width = 500   # 米
        self.uav_height = 50    # 无人机飞行高度 米
        
        # 时间参数
        self.time_step = 5.0    # 时间步长 秒
        
        # ==================== 无人机参数 ====================
        self.uav_cpu_frequency = 5       # UAV CPU频率 GHz
        self.uav_memory = 8.0              # 内存 GB
        self.cpu_cycles_per_mb = 1000      # 每MB任务的CPU周期数 (Megacycles)
        
        # 移动参数
        self.uav_max_speed = 15.0          # 最大飞行速度 m/s
        self.uav_min_speed = 0.0           # 最小飞行速度 m/s
        
        # 通信参数（优化过的）
        self.bandwidth = 10               # 带宽 MHz（增加带宽）
        self.transmission_power = 20.0      # 发射功率 dBm（增加功率）
        
        # 无线信道模型参数
        self.path_loss_exponent = 3.2      # 路径损耗指数（降低）
        self.reference_distance = 1.0      # 参考距离 (m)
        self.reference_path_loss = 46    # 参考路径损耗 (dB)（降低）
        self.noise_power = -104           # 噪声功率 (dBm)（降低噪声）
        
        # 能耗参数
        self.uav_battery_capacity = 10000  # 电池容量
        self.hover_power = 100             # 悬停功耗
        self.move_power = 150              # 移动功耗
        
        # 任务处理功耗参数
        self.user_cpu_power = 2.0          # 用户CPU功耗 (W)
        self.uav_cpu_power = 10.0          # UAV CPU功耗 (W)
        self.user_transmission_power = 0.1 # 用户传输功耗 (W)
        
        # 简化的奖励函数权重（三个组件：能耗 + 时延）
        self.uav_energy_weight = -1     # UAV能耗惩罚权重（调小）
        self.user_energy_weight = -1    # 用户能耗惩罚权重（调小）
        self.delay_weight = -1          # 时延惩罚权重（很小的权重）
        
        # ==================== 用户参数 ====================
        self.user_cpu_frequency = 1.5     # 用户设备CPU频率 GHz
        self.user_memory = 4.0             # 内存 GB
        self.user_max_speed = 2.0          # 用户移动速度 m/s
        
        # 任务参数
        self.task_arrival_rate = 1.0       # 任务到达率 (每秒)
        self.min_task_size = 5.0           # 最小任务大小 (MB)
        self.max_task_size = 10.0          # 最大任务大小 (MB)
        
        # 结束条件参数
        self.avg_task_size = (self.min_task_size + self.max_task_size) / 2  # 平均任务大小
        self.target_episode_steps = 30     # 预期每个episode的步数
        self.target_task_size_per_user = self.avg_task_size * self.target_episode_steps  # 每个用户目标任务大小
        self.total_target_task_size = self.num_users * self.target_task_size_per_user  # 总目标任务大小(MB)
        self.completed_task_size = 0.0     # 已完成任务总大小计数器(MB)
        
        # 状态空间初始化
        self.uav_states = np.zeros((self.num_uavs, 2))    # [x, y]
        self.user_states = np.zeros((self.num_users, 3))   # [x, y, current_task_size]
        
        # 性能统计
        self.stats = {
            'total_tasks_generated': 0,
            'total_tasks_completed': 0,
            'total_uav_energy_consumed': 0.0,
            'total_user_energy_consumed': 0.0
        }
        
        # 初始化模块化组件
        self._initialize_modules()
        
        # 初始化位置
        self._initialize_positions()
    
    def _initialize_modules(self):
        """初始化模块化组件"""
        # 用户分配管理器
        self.user_allocation_manager = UserAllocationManager(
            num_uavs=self.num_uavs,
            num_users=self.num_users,
            max_task_size=self.max_task_size
        )
        
        # UAV移动管理器
        self.uav_movement_manager = UAVMovementManager(
            num_uavs=self.num_uavs,
            area_length=self.area_length,
            area_width=self.area_width,
            uav_max_speed=self.uav_max_speed,
            time_step=self.time_step,
            move_power=self.move_power,
            hover_power=self.hover_power,
            uav_battery_capacity=self.uav_battery_capacity
        )
        
        # 时延计算器（简化版不用于奖励，但保留用于统计）
        self.delay_calculator = DelayCalculator(
            uav_height=self.uav_height,
            uav_cpu_frequency=self.uav_cpu_frequency,
            user_cpu_frequency=self.user_cpu_frequency,
            cpu_cycles_per_mb=self.cpu_cycles_per_mb,
            bandwidth=self.bandwidth,
            transmission_power=self.transmission_power,
            path_loss_exponent=self.path_loss_exponent,
            reference_distance=self.reference_distance,
            reference_path_loss=self.reference_path_loss,
            noise_power=self.noise_power
        )
        
        # 任务处理能耗计算器（核心组件）
        self.task_energy_calculator = TaskEnergyCalculator(
            user_cpu_frequency=self.user_cpu_frequency,
            uav_cpu_frequency=self.uav_cpu_frequency,
            cpu_cycles_per_mb=self.cpu_cycles_per_mb,
            user_cpu_power=self.user_cpu_power,
            uav_cpu_power=self.uav_cpu_power,
            user_transmission_power=self.user_transmission_power,
            bandwidth=self.bandwidth,
            transmission_power=self.transmission_power,
            path_loss_exponent=self.path_loss_exponent,
            reference_distance=self.reference_distance,
            reference_path_loss=self.reference_path_loss,
            noise_power=self.noise_power,
            uav_height=self.uav_height
        )
    
    def _initialize_positions(self):
        """初始化UAV和用户位置"""
        self._initialize_uavs()
        self._initialize_users()
    
    def _initialize_uavs(self):
        """初始化无人机状态"""
        for i in range(self.num_uavs):
            # 随机位置，避免太靠近边界
            x = random.uniform(50, self.area_length - 50)
            y = random.uniform(50, self.area_width - 50)
            
            self.uav_states[i] = [x, y]
    
    def _initialize_users(self):
        """初始化用户状态"""
        for i in range(self.num_users):
            # 随机位置
            x = random.uniform(0, self.area_length)
            y = random.uniform(0, self.area_width)
            task_size = random.uniform(self.min_task_size, self.max_task_size)
            
            self.user_states[i] = [x, y, task_size]
    
    def _update_users(self):
        """更新用户位置和生成新任务"""
        for i in range(self.num_users):
            # 1. 更新用户位置（随机游走）
            angle = random.uniform(0, 2 * math.pi)
            distance = random.uniform(0, self.user_max_speed * self.time_step)
            
            # 计算新位置
            new_x = self.user_states[i, 0] + distance * math.cos(angle)
            new_y = self.user_states[i, 1] + distance * math.sin(angle)
            
            # 边界处理
            self.user_states[i, 0] = max(0, min(self.area_length, new_x))
            self.user_states[i, 1] = max(0, min(self.area_width, new_y))
            
            # 2. 生成新任务
            new_task_size = random.uniform(self.min_task_size, self.max_task_size)
            self.user_states[i, 2] = new_task_size
            self.stats['total_tasks_generated'] += 1

    def step(self, actions=None):
        """简化的环境步进函数 - 只计算能耗奖励"""
        # 增加步数
        self.current_step += 1
        
        # 1. 处理无人机动作和用户分配
        _, user_assignments = self.user_allocation_manager.process_uav_actions_with_conflict_resolution(actions, self.user_states)
        
        # 2. 处理无人机移动（更新位置）并计算移动相关能耗
        _, movement_energy_costs = self.uav_movement_manager.process_uav_movements(actions, self.uav_states)
        
        # 3. 计算任务时延（新增）
        delay_results = self.delay_calculator.calculate_task_delays(actions, user_assignments, self.user_states, self.uav_states)
        
        uav_delays = delay_results['actual_delays']
        max_local_delays = delay_results['max_local_delays']
        max_offload_delays = delay_results['max_offload_delays']
        
        # 4. 计算任务处理能耗（核心，包含最大值）
        task_energy_breakdown = self.task_energy_calculator.calculate_task_processing_energy(actions, user_assignments, self.user_states, self.uav_states)
        
        # 4. 更新已完成任务大小
        completed_size_this_step = 0.0
        for user_id in user_assignments:
            task_size = self.user_states[user_id, 2]
            completed_size_this_step += task_size
        
        self.completed_task_size += completed_size_this_step
        self.stats['total_tasks_completed'] += len(user_assignments)
        
        # 5. 计算简化的奖励（三个组件：能耗 + 时延）
        total_rewards = {}
        reward_breakdown = {}
        raw_metrics = {}
        
        # 计算每个UAV负责的用户能耗累加（基于新的用户视角数据）
        uav_actual_total_energy = {uav_id: 0.0 for uav_id in range(self.num_uavs)}
        uav_max_total_energy = {uav_id: 0.0 for uav_id in range(self.num_uavs)}
        
        for user_id, uav_id in user_assignments.items():
            # 用户实际总能耗（直接从task_energy_breakdown获取）
            user_actual_energy = task_energy_breakdown['user_total_energy'].get(user_id, 0.0)
            uav_actual_total_energy[uav_id] += user_actual_energy
            
            # 用户最大能耗（直接从task_energy_breakdown获取）
            user_max_energy = task_energy_breakdown['user_max_energy'].get(user_id, 0.0)
            uav_max_total_energy[uav_id] += user_max_energy
        
        for uav_id in range(self.num_uavs):
            uav_key = f'uav_{uav_id}'
            
            # 获取UAV的飞行能耗（独立计算）
            uav_movement_energy_raw = movement_energy_costs.get(uav_key, 0.0)
            
            # 获取该UAV负责的用户能耗累加
            actual_total_energy_raw = uav_actual_total_energy[uav_id]  # 该UAV服务的所有用户实际总能耗之和
            max_total_energy_raw = uav_max_total_energy[uav_id]        # 该UAV服务的所有用户最大能耗之和
            
            # UAV能耗归一化：(user_total_energy1 + user_total_energy2) / (user_max_energy1 + user_max_energy2)
            if max_total_energy_raw > 0:
                if actual_total_energy_raw > self.max_energy_test:
                    self.max_energy_test = actual_total_energy_raw
                normalized_total_energy = actual_total_energy_raw / 25
            else:
                normalized_total_energy = 0.0
            
            # 获取原始时延值
            uav_delay_raw = uav_delays.get(uav_id, 0.0)
            max_local_delay_raw = max_local_delays.get(uav_id, 0.0)
            max_offload_delay_raw = max_offload_delays.get(uav_id, 0.0)
            
            # 计算每个UAV的最大时延（本地和卸载的最大值）
            max_delay_for_uav = max(max_local_delay_raw, max_offload_delay_raw)
            
            # 计算归一化时延奖励：真实时延 / 最大时延
            if max_delay_for_uav > 0:
                if uav_delay_raw > self.max_delay_test:
                    self.max_delay_test = uav_delay_raw
                normalized_delay = uav_delay_raw / 20
            else:
                normalized_delay = 0.0
            
            # 计算加权后的惩罚（两个组件：系统总能耗 + 时延）
            # 注意：现在只有一个能耗组件（系统总能耗），不再分UAV和用户
            total_energy_penalty = self.uav_energy_weight * normalized_total_energy  # 使用归一化系统总能耗
            delay_penalty = self.delay_weight * normalized_delay                     # 使用归一化时延
            
            # 收集该UAV服务的所有用户的详细能耗信息
            uav_served_users = [user_id for user_id, assigned_uav in user_assignments.items() if assigned_uav == uav_id]
            
            # 计算该UAV服务的用户能耗总和
            total_user_local_energy = 0.0
            total_user_offload_energy = 0.0
            
            for user_id in uav_served_users:
                total_user_local_energy += task_energy_breakdown['user_local_energy'].get(user_id, 0.0)
                total_user_offload_energy += task_energy_breakdown['user_offload_energy'].get(user_id, 0.0)
            
            # 保存原始值（包含所有基准值和归一化信息）
            raw_metrics[uav_key] = {
                # 原始能耗值（用户视角累加）
                'actual_total_energy_raw': actual_total_energy_raw,      # 该UAV服务的所有用户实际总能耗之和
                'max_total_energy_raw': max_total_energy_raw,            # 该UAV服务的所有用户最大能耗之和
                'uav_movement_energy_raw': uav_movement_energy_raw,      # UAV飞行能耗（独立）
                # 用户能耗累加总和
                'total_user_local_energy': total_user_local_energy,     # 该UAV服务的所有用户本地能耗总和
                'total_user_offload_energy': total_user_offload_energy, # 该UAV服务的所有用户卸载能耗总和
                'served_users': uav_served_users,                       # 该UAV服务的用户列表
                # 归一化能耗
                'normalized_total_energy': normalized_total_energy,      # 归一化能耗：实际累加/最大累加
                # 原始时延值
                'delay_raw': uav_delay_raw,
                'max_local_delay_raw': max_local_delay_raw,
                'max_offload_delay_raw': max_offload_delay_raw,
                'max_delay_for_uav': max_delay_for_uav,
                # 归一化时延
                'normalized_delay': normalized_delay
            }
            
            # 记录奖励分解
            reward_breakdown[uav_key] = {
                'total_energy_penalty': total_energy_penalty,             # 系统总能耗惩罚
                'delay_penalty': delay_penalty,                           # 时延惩罚
                'normalized_total_energy_used': normalized_total_energy,  # 记录使用的归一化系统总能耗
                'normalized_delay_used': normalized_delay,                 # 记录使用的归一化时延
                'actual_total_energy_raw':actual_total_energy_raw,
                'actual_uav_delay_raw':uav_delay_raw
                
            }
            
            # 总奖励（两个组件）
            total_rewards[uav_key] = total_energy_penalty + delay_penalty
        
        # 更新统计
        total_system_energy = sum([raw_metrics[f'uav_{i}']['actual_total_energy_raw'] for i in range(self.num_uavs)])
        total_movement_energy = sum([raw_metrics[f'uav_{i}']['uav_movement_energy_raw'] for i in range(self.num_uavs)])
        self.stats['total_system_energy_consumed'] = self.stats.get('total_system_energy_consumed', 0.0) + total_system_energy
        self.stats['total_movement_energy_consumed'] = self.stats.get('total_movement_energy_consumed', 0.0) + total_movement_energy
        
        # 更新用户
        self._update_users()
        
        # 获取观察
        observations = self._get_observations()
        
        # 判断是否结束（只基于任务完成）
        done = self.completed_task_size >= self.total_target_task_size
        # if done:
        #     print(f"max_energy_test: {self.max_energy_test}")
        #     print(f"max_delay_test: {self.max_delay_test}")
        
        # 信息字典
        info = {
            'step': self.current_step,
            'stats': self.stats.copy(),
            'user_assignments': user_assignments,
            'uav_delays': uav_delays,
            'max_local_delays': max_local_delays,
            'max_offload_delays': max_offload_delays,
            'task_energy_breakdown': task_energy_breakdown,
            'movement_energy_costs': movement_energy_costs,
            'reward_breakdown': reward_breakdown,
            'raw_metrics': raw_metrics,
            'completed_task_size': self.completed_task_size,
            'total_target_task_size': self.total_target_task_size,
            'completed_size_this_step': completed_size_this_step,
            'done_by_task_size': done
        }
        
        return observations, total_rewards, done, info
    
    def _get_observations(self):
        """获取所有无人机的观察"""
        observations = {}
        for uav_id in range(self.num_uavs):
            observations[f'uav_{uav_id}'] = self._get_uav_observation(uav_id)
        return observations
    
    def _get_uav_observation(self, uav_id):
        """获取单个无人机的观察（归一化处理）"""
        # 当前无人机状态
        uav_state = self.uav_states[uav_id]
        uav_obs = [
            uav_state[0] / self.area_length,    # 归一化x坐标
            uav_state[1] / self.area_width,     # 归一化y坐标  
        ]
        
        # 用户信息（位置 + 任务大小）
        user_info = []
        for user_id in range(self.num_users):
            user_state = self.user_states[user_id]
            # 计算到用户的距离
            distance = self._calculate_distance_2d(uav_state[:2], user_state[:2])
            max_distance = math.sqrt(self.area_length**2 + self.area_width**2)
            
            user_info.extend([
                user_state[0] / self.area_length,     # 归一化用户x
                user_state[1] / self.area_width,      # 归一化用户y
                user_state[2] / self.max_task_size   # 归一化任务大小
            ])
        
        
        # 合并所有观察
        full_observation = uav_obs + user_info 
        return np.array(full_observation, dtype=np.float32)
    
    def reset(self):
        """重置环境"""
        self.current_step = 0
        
        # 重新初始化模块化组件
        self._initialize_modules()
        
        # 重新初始化位置
        self._initialize_positions()
        
        # 重置统计
        self.stats = {
            'total_tasks_generated': 0,
            'total_tasks_completed': 0,
            'total_uav_energy_consumed': 0.0,
            'total_user_energy_consumed': 0.0
        }
        
        # 重置任务完成计数器
        self.completed_task_size = 0.0
        
        return self._get_observations()
    
    def _calculate_distance_2d(self, pos1, pos2):
        """计算两点间的2D距离"""
        return math.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
    
    def get_observation_space_size(self):
        """获取观察空间大小"""
        # UAV自身: 2个特征 (x, y)
        uav_features = 2
        # 每个用户: 4个特征 (x, y, task_size)
        user_features = self.num_users * 3
        
        total = uav_features + user_features
        return total
    
    def get_action_space_info(self):
        """获取动作空间信息"""
        return {
            'user_competition_probs': self.num_users,  # 竞争概率 [0,1]
            'offloading_ratios': self.num_users,       # 卸载比例 [0,1]
            'movement_direction': 1,                   # 移动方向 [0, 2π]
            'movement_distance': 1                     # 移动距离 [0, max_distance]
        }

# 使用示例
if __name__ == "__main__":
    print("\n=== 简化版多无人机环境 ===")
    print("奖励函数: UAV能耗惩罚 + 用户能耗惩罚")
    print("优化目标: 最小化总能耗")
    
    env = SimplifiedMultiUAVEnvironment(num_uavs=3, num_users=6)
    print(f"总目标任务大小: {env.total_target_task_size} MB")
    print(f"预期episode步数: {env.target_episode_steps}")
