import math
import random
import numpy as np
import torch
from typing import Dict, List

# 导入模块化组件
from user_allocation import UserAllocationManager
from uav_movement import UAVMovementManager
from delay_calculator import DelayCalculator
from task_energy_calculator import TaskEnergyCalculator



class SimplifiedMultiUAVEnvironment:
    """简化的多无人机多用户边缘计算环境 - 只考虑能耗奖励"""
    
    def __init__(self, num_uavs=2, num_users=5,trajectory_file=None):
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
        self.area_length = 150  # 米
        self.area_width = 150   # 米
        self.uav_height = 100    # 无人机飞行高度 米
        
        # 时间参数
        self.time_step = 5.0    # 时间步长 秒
        
        # ==================== 无人机参数 ====================
        self.uav_cpu_frequency = 1.2       # UAV CPU频率 GHz
        self.cpu_cycles_per_mb = 1000      # 每MB任务的CPU周期数 (Megacycles)
        
        # 移动参数
        self.uav_max_speed = 20          # 最大飞行速度 m/s
        
        # 通信参数（优化过的）
        self.bandwidth = 1               # 带宽 MHz（增加带宽）
        self.transmission_power = 0.1     
        
        # 无线信道模型参数
        self.path_loss_exponent = 2      # 路径损耗指数（降低）
        self.reference_distance = 1.0      # 参考距离 (m)
        self.reference_path_loss = 1e-4    # 参考路径损耗 (dB)（降低）
        self.noise_power = 1e-13            # 噪声功率 (dBm)（降低噪声）
        
        # 任务处理功耗参数
        self.user_transmission_power = 0.1 # 用户传输功耗 (W)
        
        # ==================== 用户参数 ====================
        self.user_cpu_frequency = 0.4     # 用户设备CPU频率 GHz
        self.uav_cpu_power = (self.uav_cpu_frequency)**3          # UAV CPU功耗 (W)
        self.user_cpu_power = self.user_cpu_frequency**3          # 用户CPU功耗 (W)

        self.user_max_speed = 2.0          # 用户移动速度 m/s
        
        # 任务参数
        self.min_task_size = 5           # 最小任务大小 (MB)
        self.max_task_size = 10          # 最大任务大小 (MB)
        
        # 结束条件参数
        self.avg_task_size = (self.min_task_size + self.max_task_size) / 2  # 平均任务大小
        self.target_episode_steps = 20     # 预期每个episode的步数
        self.target_task_size_per_user = self.avg_task_size * self.target_episode_steps  # 每个用户目标任务大小
        self.total_target_task_size = self.num_users * self.target_task_size_per_user  # 总目标任务大小(MB)
        self.completed_task_size = 0.0     # 已完成任务总大小计数器(MB)
        self.uav_battery_capacity = 10000
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

        self.trajectory_file = trajectory_file
        self.use_predefined_trajectory = trajectory_file is not None
        self.trajectory_data = None
        self.max_trajectory_steps = None

        if self.use_predefined_trajectory:
            self._load_trajectory_file()
        
        # 初始化模块化组件
        self._initialize_modules()
        
        # 初始化位置
        self._initialize_positions()
    
    def _load_trajectory_file(self):
        """从JSON文件加载预生成的轨迹"""
        import json
        
        with open(self.trajectory_file, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
        
        # 加载轨迹数据
        self.trajectory_data = {}
        for user_id_str, traj_list in json_data['trajectories'].items():
            user_id = int(user_id_str)
            self.trajectory_data[user_id] = np.array(traj_list)
        
        self.max_trajectory_steps = self.trajectory_data[0].shape[0]
        
        print(f"✓ 成功加载轨迹: {self.num_users}个用户, {self.max_trajectory_steps}步")
    
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
            x = random.uniform(0, self.area_length)
            y = random.uniform(0, self.area_width)
            
            self.uav_states[i] = [x, y]
    
    def _initialize_users(self):
        """初始化用户状态"""
        if self.use_predefined_trajectory:
           
            # 使用轨迹文件的第0步
            for i in range(self.num_users):
                x = self.trajectory_data[i][0, 0]
                y = self.trajectory_data[i][0, 1]
                task_size = self.trajectory_data[i][0, 2]
                self.user_states[i] = [x, y, task_size]
        else:
            # 原有的随机初始化代码
            for i in range(self.num_users):
                x = random.uniform(0, self.area_length)
                y = random.uniform(0, self.area_width)
                task_size = random.uniform(self.min_task_size, self.max_task_size)
                self.user_states[i] = [x, y, task_size]
    
    def _update_users(self):
        """更新用户位置和生成新任务"""
        if self.use_predefined_trajectory:
            # 使用预定义轨迹
            if self.current_step < self.max_trajectory_steps:
                for i in range(self.num_users):
                    self.user_states[i, 0] = self.trajectory_data[i][self.current_step, 0]
                    self.user_states[i, 1] = self.trajectory_data[i][self.current_step, 1]
                    self.user_states[i, 2] = self.trajectory_data[i][self.current_step, 2]
                    self.stats['total_tasks_generated'] += 1
        else:
            # 原有的随机移动代码（保持不变）
            for i in range(self.num_users):
                angle = random.uniform(0, 2 * math.pi)
                distance = random.uniform(0, self.user_max_speed * self.time_step)
                new_x = self.user_states[i, 0] + distance * math.cos(angle)
                new_y = self.user_states[i, 1] + distance * math.sin(angle)
                self.user_states[i, 0] = max(0, min(self.area_length, new_x))
                self.user_states[i, 1] = max(0, min(self.area_width, new_y))
                new_task_size = random.uniform(self.min_task_size, self.max_task_size)
                self.user_states[i, 2] = new_task_size
                self.stats['total_tasks_generated'] += 1

    def step(self, actions=None):
        """简化的环境步进函数 - 只计算能耗奖励"""
        # 增加步数
        self.current_step += 1
        
        # 1. 处理无人机动作和用户分配
        user_assignments = self.user_allocation_manager.process_uav_actions_with_conflict_resolution(actions)
        
        # 2. 处理无人机移动（更新位置）并计算移动相关能耗
        _, movement_energy_costs, uav_movement_actual_delays = self.uav_movement_manager.process_uav_movements(actions, self.uav_states)
        
        # 3. 计算任务时延（新增）
        uav_delays = self.delay_calculator.calculate_task_delays(actions, self.user_states, self.uav_states,user_assignments)['actual_delays']
        max_uav_computation_delays = self.delay_calculator.calculate_task_delays(actions, self.user_states, self.uav_states,user_assignments)['max_uav_computation_delays']
        
        # 4. 计算任务处理能耗（核心，包含最大值）
        uav_actual_total_energy = self.task_energy_calculator.calculate_task_processing_energy(actions, self.user_states, self.uav_states,user_assignments,max_uav_computation_delays)['uav_actual_total_energy']
        
        # 4. 更新已完成任务大小
        completed_size_this_step = 0.0
        for user_id in range(self.num_users):
            task_size = self.user_states[user_id, 2]
            completed_size_this_step += task_size
        
        self.completed_task_size += completed_size_this_step
        self.stats['total_tasks_completed'] += len(user_assignments)
        
        # 5. 计算简化的奖励（三个组件：能耗 + 时延）
        total_rewards = 0
        reward_breakdown = {}
        raw_metrics = {}
        max_delay = 0
        for uav_id in range(self.num_uavs):
            uav_key = f'uav_{uav_id}'
            
            normalized_movement_energy = movement_energy_costs.get(uav_key, 0.0) / 2895
            
            # 获取该UAV负责的能耗累加
            normalized_task_energy = uav_actual_total_energy[uav_id] / 5
            
            # 获取原始时延值
            uav_delay_raw = uav_delays.get(uav_id, 0.0)
            normalized_delay = uav_delay_raw / 3

            uav_movement_actual_delay = uav_movement_actual_delays.get(uav_key, 0.0)
            normalized_movement_delay = uav_movement_actual_delay / 1.6
            
            # 计算加权后的惩罚（两个组件：系统总能耗 + 时延）
            reward_energy = -(normalized_movement_energy + normalized_task_energy)
            reward_delay = (uav_delay_raw + uav_movement_actual_delay)
            if(uav_delay_raw > max_delay):
                max_delay = uav_delay_raw
            # reward = -movement_energy_costs.get(uav_key, 0.0)*uav_delays.get(uav_id, 0.0)
            
            # 保存原始值（包含所有基准值和归一化信息）
            raw_metrics[uav_key] = {
                # 原始能耗值（用户视角累加）
                'actual_total_energy_raw': uav_actual_total_energy[uav_id],      # 该UAV服务的所有用户实际总能耗之和
                'uav_movement_energy_raw': movement_energy_costs.get(uav_key, 0.0) ,      # UAV飞行能耗（独立）
                # 归一化能耗
                
                # 原始时延值
                'delay_raw': uav_delay_raw,
                'movement_delay_raw': uav_movement_actual_delay
            }
            
            
            # 总奖励（两个组件）
            total_rewards += reward_energy
        total_rewards += (-max_delay/4.6)
        # print(raw_metrics)
        # 更新统计
        total_system_energy = sum([raw_metrics[f'uav_{i}']['actual_total_energy_raw'] for i in range(self.num_uavs)])
        total_movement_energy = sum([raw_metrics[f'uav_{i}']['uav_movement_energy_raw'] for i in range(self.num_uavs)])
       
        # 更新用户
        self._update_users()
        
        # 获取观察
        observations = self._get_observations()
        
        # 判断是否结束（只基于任务完成）
        done = self.completed_task_size >= self.total_target_task_size
     
        
        # 信息字典
        info = {
            'step': self.current_step,
            'stats': self.stats.copy(),
            'user_assignments': user_assignments,
            'uav_delays': uav_delays,
            'movement_energy_costs': movement_energy_costs,
            'uav_movement_actual_delays': uav_movement_actual_delays,
            'raw_metrics': raw_metrics,
            'completed_task_size': self.completed_task_size,
            'total_target_task_size': self.total_target_task_size,
            'completed_size_this_step': completed_size_this_step,
            'done_by_task_size': done
        }
        
        return observations, total_rewards, done, info
    
    def _get_observations(self):
       
        """
        获取适配神经网络的状态格式（归一化）
        
        Returns:
            state: dict with keys 'uav_pos', 'user_pos', 'user_tasks'
        """
        # 归一化无人机位置
        uav_pos_normalized = torch.FloatTensor([
            [
                self.uav_states[i, 0] / self.area_length,
                self.uav_states[i, 1] / self.area_width
            ]
            for i in range(self.num_uavs)
        ])  # [N, 2]
        
        # 归一化用户位置
        user_pos_normalized = torch.FloatTensor([
            [
                self.user_states[i, 0] / self.area_length,
                self.user_states[i, 1] / self.area_width
            ]
            for i in range(self.num_users)
        ])  # [M, 2]
        
        # 归一化用户任务大小
        user_tasks_normalized = torch.FloatTensor([
            [self.user_states[i, 2] / self.max_task_size]
            for i in range(self.num_users)
        ])  # [M, 1]
        
        return {
            'uav_pos': uav_pos_normalized,      # [N, 2], 归一化到[0, 1]
            'user_pos': user_pos_normalized,    # [M, 2], 归一化到[0, 1]
            'user_tasks': user_tasks_normalized # [M, 1], 归一化到[0, 1]
        }
    
    
    
    def reset(self):
        """重置环境"""
        self.current_step = 0
        self.max_delay_test = 0
        self.max_energy_test = 0
        
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
