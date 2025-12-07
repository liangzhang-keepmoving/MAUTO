"""
奖励系统模块 - 简化版
只考虑归一化后的时延和能耗
"""
import numpy as np
import os

class AdaptiveRewardSystem:
    """简化版奖励系统 - 固定权重，仅关注时延和能耗"""
    
    def __init__(self, 
                 num_uavs,
                 num_users,
                 enable_adaptive=False, # 保留接口但不使用
                 log_dir='reward_logs'):
        """
        初始化奖励系统
        
        Args:
            num_uavs: 无人机数量
            num_users: 用户数量
            enable_adaptive: 是否启用自适应权重 (本版本忽略)
            log_dir: 日志保存目录
        """
        self.num_uavs = num_uavs
        self.num_users = num_users
        self.log_dir = log_dir
        
        # 固定权重设置
        # 你可以根据需要调整这些权重
        self.weights = {
            'w_delay': 0.6,   # 时延权重
            'w_energy': 0.4   # 能耗权重
        }
        
        # ========== 归一化参数 ==========
        # 只需要这些核心参数
        self.norm_params = {
            'min_avg_delay': 0.0,
            'max_avg_delay': 1.0,
            'min_task_energy': 0.0,
            'max_task_energy': 1.0,
            'min_move_energy': 0.0,
            'max_move_energy': 1.0
        }
        
        # 统计极值
        self.stats = {
            'min_avg_delay': float('inf'),
            'max_avg_delay': float('-inf'),
            'min_total_task_energy': float('inf'),
            'max_total_task_energy': float('-inf')
        }
        
    def set_normalization_params(self, params):
        """
        设置归一化参数
        """
        # 安全更新，防止 KeyError
        for key in self.norm_params:
            if key in params:
                self.norm_params[key] = params[key]
        print(f"✓ 归一化参数已设置: {self.norm_params}")
    
    def calculate_reward(self, raw_metrics, movement_energy_costs, 
                        uav_movement_delays, info):
        """
        计算简化版奖励
        Reward = - (w_delay * norm_delay + w_energy * norm_energy)
        """
        # ========== 1. 提取并聚合指标 ==========
        user_delays = []
        user_total_energies = [] # 包含本地计算、传输、UAV计算
        
        for user_id in range(self.num_users):
            if user_id in raw_metrics:
                metrics = raw_metrics[user_id]
                # ================= 修改开始 =================
                # 强制清洗数据：将可能的 numpy.float32 或 tensor 转为 python float
                clean_delay = float(metrics['user_actual_delay'])
                user_delays.append(clean_delay)
                
                # 用户总能耗 = 本地 + 传输 + UAV端 (全部清洗)
                clean_local_e = float(metrics['user_local_computation_energy'])
                clean_trans_e = float(metrics['user_transmission_energy'])
                clean_uav_e = float(metrics['user_uav_computation_energy'])
                
                u_energy = (clean_local_e + clean_trans_e + clean_uav_e)
                user_total_energies.append(u_energy)
                # ================= 修改结束 =================
                # user_delays.append(metrics['user_actual_delay'])
                
                # # 用户总能耗 = 本地 + 传输 + UAV端
                # u_energy = (metrics['user_local_computation_energy'] + 
                #            metrics['user_transmission_energy'] + 
                #            metrics['user_uav_computation_energy'])
                # user_total_energies.append(u_energy)
        
        # 聚合指标
        avg_delay = np.mean(user_delays) if user_delays else 0.0
        
        # 总任务能耗
        total_task_energy = sum(user_total_energies)
        
        # 更新统计极值
        self.stats['min_avg_delay'] = min(self.stats['min_avg_delay'], avg_delay)
        self.stats['max_avg_delay'] = max(self.stats['max_avg_delay'], avg_delay)
        self.stats['min_total_task_energy'] = min(self.stats['min_total_task_energy'], total_task_energy)
        self.stats['max_total_task_energy'] = max(self.stats['max_total_task_energy'], total_task_energy)
        
        # 系统总能耗 (只考虑任务能耗，忽略飞行能耗)
        total_system_energy = total_task_energy
        
        # ========== 2. 归一化 ==========
        # 归一化时延
        norm_delay = self._normalize(
            avg_delay, 
            self.norm_params.get('min_avg_delay', 0), 
            self.norm_params.get('max_avg_delay', 1.0)
        )
        
        # 归一化能耗 (只归一化任务能耗)
        norm_task_energy = self._normalize(
            total_task_energy,
            self.norm_params.get('min_task_energy', 0),
            self.norm_params.get('max_task_energy', 1.0)
        )
        
        # 综合能耗得分 = 归一化后的任务能耗
        norm_energy_score = norm_task_energy

        
        # ========== 3. 计算最终奖励 ==========
        # 核心公式：Reward = - (w_d * D + w_e * E)
        penalty = (
            self.weights['w_delay'] * norm_delay + 
            self.weights['w_energy'] * norm_energy_score
        )
        
        reward = -penalty
        
        # ========== 4. 返回详情 ==========
        reward_components = {
            'avg_delay': avg_delay,
            'total_energy': total_system_energy,
            'norm_delay': norm_delay,
            'norm_energy': norm_energy_score,
            'reward': reward
        }
        
        return reward, reward_components
    
    def _normalize(self, value, min_val, max_val):
        """安全归一化"""
        if min_val is None: 
            min_val = 0.0
            print("未找到归一化值*************************")
        if max_val is None or max_val == 0: 
            print("未找到归一化值*************************")
            max_val = 1.0
        
        if max_val - min_val < 1e-8:
            return 0.0
            
        return np.clip((value - min_val) / (max_val - min_val), 0.0, 1.0)
        
    def update(self, *args, **kwargs):
        """保留接口以兼容旧代码调用，但不做任何事"""
        pass
    
    def on_episode_end(self, episode_reward):
        """保留接口以兼容train_ddpg.py调用，但不做任何事"""
        pass

    def get_diagnostics(self):
        """返回简单的诊断信息"""
        return {
            'weights': self.weights,
            'stats': self.stats
        }
