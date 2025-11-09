"""
奖励系统模块 - 包含自适应权重和完整的奖励函数
"""
import numpy as np
from collections import deque
import json
import os

class AdaptiveRewardSystem:
    """自适应奖励系统 - 动态调整各指标权重"""
    
    def __init__(self, 
                 num_uavs,
                 num_users,
                 enable_adaptive=True,
                 history_size=1000,
                 min_samples_for_adaptation=200,
                 log_dir='reward_logs'):
        """
        初始化奖励系统
        
        Args:
            num_uavs: 无人机数量
            num_users: 用户数量
            enable_adaptive: 是否启用自适应权重
            history_size: 历史数据保存大小
            min_samples_for_adaptation: 启用自适应的最小样本数
            log_dir: 日志保存目录
        """
        self.num_uavs = num_uavs
        self.num_users = num_users
        self.enable_adaptive = enable_adaptive
        self.min_samples_for_adaptation = min_samples_for_adaptation
        self.log_dir = log_dir
        
        # 创建日志目录
        os.makedirs(log_dir, exist_ok=True)
        
        # ========== 历史数据队列 ==========
        self.avg_delay_history = deque(maxlen=history_size)
        self.max_delay_history = deque(maxlen=history_size)
        self.task_energy_history = deque(maxlen=history_size)
        self.move_energy_history = deque(maxlen=history_size)
        self.load_variance_history = deque(maxlen=history_size)
        self.movement_benefit_history = deque(maxlen=history_size)
        
        # ========== 初始权重配置 ==========
        self.weights = {
            'w_avg_delay': 0.25,
            'w_max_delay': 0.2,
            'w_task_energy': 0.3,
            'w_move_energy': 0.05,
            'w_load_balance': 0.05,
            'w_movement_benefit': 0.15
        }
        
        # ========== 归一化参数（需要从环境获取）==========
        self.norm_params = {
            'min_avg_delay': None,
            'max_avg_delay': None,
            'min_max_delay': None,
            'max_max_delay': None,
            'min_task_energy': None,
            'max_task_energy': None,
            'min_move_energy': None,
            'max_move_energy': None,
            'min_load_variance': 0.0,
            'max_load_variance': None,
            'max_movement_benefit': None
        }
        
        # ========== 统计信息 ==========
        self.episode = 0
        self.total_rewards = []
        self.weight_history = []
        
    def set_normalization_params(self, params):
        """
        设置归一化参数
        
        Args:
            params: dict, 包含各指标的min/max值
        """
        self.norm_params.update(params)
        print(f"✓ 归一化参数已设置")
    
    def calculate_reward(self, raw_metrics, movement_energy_costs, 
                        uav_movement_delays, info):
        """
        计算综合奖励（核心方法）
        
        Args:
            raw_metrics: dict, 每个用户的原始指标
            movement_energy_costs: dict, 每个UAV的移动能耗
            uav_movement_delays: dict, 每个UAV的移动时延
            info: dict, 包含user_assignments等信息
            
        Returns:
            reward: float, 综合奖励值
            reward_components: dict, 各组成部分的详细信息
        """
        # ========== 1. 提取原始指标 ==========
        user_delays = []
        user_local_energies = []
        user_transmission_energies = []
        user_uav_energies = []
        
        for user_id in range(self.num_users):
            if user_id in raw_metrics:
                user_delays.append(raw_metrics[user_id]['user_actual_delay'])
                user_local_energies.append(raw_metrics[user_id]['user_local_computation_energy'])
                user_transmission_energies.append(raw_metrics[user_id]['user_transmission_energy'])
                user_uav_energies.append(raw_metrics[user_id]['user_uav_computation_energy'])
        
        # ========== 2. 计算聚合指标 ==========
        avg_delay = np.mean(user_delays)
        max_delay = np.max(user_delays)
        
        total_task_energy = (sum(user_local_energies) + 
                            sum(user_transmission_energies) + 
                            sum(user_uav_energies))
        
        total_move_energy = sum(movement_energy_costs.values())
        
        # 负载均衡
        uav_loads = [0] * self.num_uavs
        user_assignments = info.get('user_assignments', {})
        for user_id, uav_id in user_assignments.items():
            uav_loads[uav_id] += 1
        load_variance = np.var(uav_loads)
        
        # 移动收益
        movement_benefit = self._calculate_movement_benefit(
            user_assignments, info.get('uav_states'), info.get('user_states')
        )
        
        # ========== 3. 归一化 ==========
        norm_avg_delay = self._normalize(
            avg_delay, 
            self.norm_params['min_avg_delay'], 
            self.norm_params['max_avg_delay']
        )
        
        norm_max_delay = self._normalize(
            max_delay,
            self.norm_params['min_max_delay'],
            self.norm_params['max_max_delay']
        )
        
        norm_task_energy = self._normalize(
            total_task_energy,
            self.norm_params['min_task_energy'],
            self.norm_params['max_task_energy']
        )
        
        norm_move_energy = self._normalize(
            total_move_energy,
            self.norm_params['min_move_energy'],
            self.norm_params['max_move_energy']
        )
        
        norm_load_variance = self._normalize(
            load_variance,
            self.norm_params['min_load_variance'],
            self.norm_params.get('max_load_variance', self.num_users ** 2)
        )
        
        norm_movement_benefit = self._normalize(
            movement_benefit,
            0.0,
            self.norm_params.get('max_movement_benefit', self.num_uavs * 0.5)
        )
        
        # ========== 4. 记录历史（用于自适应） ==========
        self.avg_delay_history.append(norm_avg_delay)
        self.max_delay_history.append(norm_max_delay)
        self.task_energy_history.append(norm_task_energy)
        self.move_energy_history.append(norm_move_energy)
        self.load_variance_history.append(norm_load_variance)
        self.movement_benefit_history.append(norm_movement_benefit)
        
        # ========== 5. 更新权重（自适应） ==========
        # if self.enable_adaptive and len(self.avg_delay_history) >= self.min_samples_for_adaptation:
        #     self._update_adaptive_weights()
        
        # ========== 6. 计算加权奖励 ==========
        reward = -(
            self.weights['w_avg_delay'] * norm_avg_delay +
            self.weights['w_max_delay'] * norm_max_delay +
            self.weights['w_task_energy'] * norm_task_energy +
            self.weights['w_move_energy'] * norm_move_energy +
            self.weights['w_load_balance'] * norm_load_variance
        ) + self.weights['w_movement_benefit'] * norm_movement_benefit
        
        # ========== 7. 记录奖励组成 ==========
        reward_components = {
            # 原始值
            'avg_delay': avg_delay,
            'max_delay': max_delay,
            'total_task_energy': total_task_energy,
            'total_move_energy': total_move_energy,
            'load_variance': load_variance,
            'movement_benefit': movement_benefit,
            
            # 归一化值
            'norm_avg_delay': norm_avg_delay,
            'norm_max_delay': norm_max_delay,
            'norm_task_energy': norm_task_energy,
            'norm_move_energy': norm_move_energy,
            'norm_load_variance': norm_load_variance,
            'norm_movement_benefit': norm_movement_benefit,
            
            # 加权贡献
            'delay_contribution': -(self.weights['w_avg_delay'] * norm_avg_delay + 
                                   self.weights['w_max_delay'] * norm_max_delay),
            'energy_contribution': -(self.weights['w_task_energy'] * norm_task_energy + 
                                    self.weights['w_move_energy'] * norm_move_energy),
            'balance_contribution': -self.weights['w_load_balance'] * norm_load_variance,
            'benefit_contribution': self.weights['w_movement_benefit'] * norm_movement_benefit,
            
            # 当前权重
            'weights': self.weights.copy(),
            
            # 总奖励
            'total_reward': reward
        }
        
        return reward, reward_components
    
    def _normalize(self, value, min_val, max_val):
        """归一化到 [0, 1]"""
        if min_val is None or max_val is None:
            print("使用原值了，很危险********************************")
            return value  # 如果没有设置范围，返回原值
        
        if max_val - min_val < 1e-8:
            return 0.0
        
        return np.clip((value - min_val) / (max_val - min_val), 0.0, 1.0)
    
    def _calculate_movement_benefit(self, user_assignments, uav_states, user_states):
        """
        计算移动收益：UAV越靠近其服务的用户，收益越高
        
        Returns:
            float: 移动收益值
        """
        if uav_states is None or user_states is None:
            return 0.0
        
        total_benefit = 0.0
        
        for uav_id in range(self.num_uavs):
            # 找到该UAV服务的所有用户
            served_users = [uid for uid, assigned_uav in user_assignments.items() 
                           if assigned_uav == uav_id]
            
            if len(served_users) == 0:
                continue
            
            # 计算到服务用户的平均距离
            distances = []
            for user_id in served_users:
                dx = uav_states[uav_id, 0] - user_states[user_id, 0]
                dy = uav_states[uav_id, 1] - user_states[user_id, 1]
                distance = np.sqrt(dx**2 + dy**2)
                distances.append(distance)
            
            avg_distance = np.mean(distances)
            
            # 距离越近，收益越大（使用倒数关系）
            # 加1避免除零，系数可调
            benefit = 1.0 / (avg_distance / 100.0 + 1.0)  # 归一化到合理范围
            total_benefit += benefit
        
        return total_benefit
    
    def _update_adaptive_weights(self):
        print("正在更新参数")
      
        # 计算各指标的标准差
        std_avg_delay = np.std(self.avg_delay_history)
        std_max_delay = np.std(self.max_delay_history)
        std_task_energy = np.std(self.task_energy_history)
        std_move_energy = np.std(self.move_energy_history)
        std_load_variance = np.std(self.load_variance_history)
        std_movement_benefit = np.std(self.movement_benefit_history)
        
        # 避免除零
        std_avg_delay = max(std_avg_delay, 1e-6)
        std_max_delay = max(std_max_delay, 1e-6)
        std_task_energy = max(std_task_energy, 1e-6)
        std_move_energy = max(std_move_energy, 1e-6)
        std_load_variance = max(std_load_variance, 1e-6)
        std_movement_benefit = max(std_movement_benefit, 1e-6)
        
        # 计算原始权重（方差的倒数）
        raw_w_avg_delay = 1.0 / std_avg_delay
        raw_w_max_delay = 1.0 / std_max_delay
        raw_w_task_energy = 1.0 / std_task_energy
        raw_w_move_energy = 1.0 / std_move_energy
        raw_w_load_balance = 1.0 / std_load_variance
        raw_w_movement_benefit = 1.0 / std_movement_benefit
        
        # 归一化（惩罚项和奖励项分别归一化）
        total_penalty = (raw_w_avg_delay + raw_w_max_delay + raw_w_task_energy + 
                        raw_w_move_energy + raw_w_load_balance)
        
        # 更新权重（使用指数移动平均，避免剧烈波动）
        alpha = 0.1  # 平滑系数
        
        new_weights = {
            'w_avg_delay': raw_w_avg_delay / total_penalty * 0.95,  # 95%分配给惩罚项
            'w_max_delay': raw_w_max_delay / total_penalty * 0.95,
            'w_task_energy': raw_w_task_energy / total_penalty * 0.95,
            'w_move_energy': raw_w_move_energy / total_penalty * 0.95,
            'w_load_balance': raw_w_load_balance / total_penalty * 0.95,
            'w_movement_benefit': 0.05  # 固定5%给奖励项
        }
        
        # 指数移动平均更新
        for key in self.weights:
            self.weights[key] = (1 - alpha) * self.weights[key] + alpha * new_weights[key]
        
        # 记录权重历史
        self.weight_history.append(self.weights.copy())
    
    def on_episode_end(self, episode_reward):

        self.episode += 1

        self.total_rewards.append(episode_reward)

        # 每100个episode保存一次统计
        if self.episode % 100 == 0:
            self.save_statistics()
    
    def save_statistics(self):
        """保存统计信息到文件"""
        stats = {
            'episode_count': self.episode,
            'current_weights': self.weights,
            'avg_reward_last_100': np.mean(self.total_rewards[-100:]) if len(self.total_rewards) >= 100 else 0,
            'weight_history': self.weight_history[-100:],  # 只保存最近100个
            'variance_stats': {
                'avg_delay_std': np.std(self.avg_delay_history),
                'task_energy_std': np.std(self.task_energy_history),
                'move_energy_std': np.std(self.move_energy_history)
            }
        }
        
        filepath = os.path.join(self.log_dir, f'reward_stats_ep{self.episode}.json')
        with open(filepath, 'w') as f:
            json.dump(stats, f, indent=2)
        
        print(f"  ✓ 奖励统计已保存: {filepath}")
    
    def get_diagnostics(self):
        """
        获取诊断信息（用于监控）
        
        Returns:
            dict: 包含各项诊断指标
        """
        if len(self.avg_delay_history) < 10:
            return None
        
        diagnostics = {
            'sample_count': len(self.avg_delay_history),
            'current_weights': self.weights.copy(),
            'variance_ratios': {
                'delay_vs_energy': (np.var(self.avg_delay_history) / 
                                   np.var(self.task_energy_history)) if np.var(self.task_energy_history) > 0 else 0,
                'task_vs_move': (np.var(self.task_energy_history) / 
                                np.var(self.move_energy_history)) if np.var(self.move_energy_history) > 0 else 0
            },
            'value_ranges': {
                'norm_avg_delay': [np.min(self.avg_delay_history), np.max(self.avg_delay_history)],
                'norm_task_energy': [np.min(self.task_energy_history), np.max(self.task_energy_history)],
                'norm_move_energy': [np.min(self.move_energy_history), np.max(self.move_energy_history)]
            }
        }
        
        return diagnostics