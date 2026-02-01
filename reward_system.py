"""
奖励系统模块 - 简化版
只考虑归一化后的时延和能耗
"""
import numpy as np

class AdaptiveRewardSystem:
    
    def __init__(self, 
                 num_uavs,
                 num_users,
                 log_dir='reward_logs'):
        """
        初始化奖励系统
        
        Args:
            num_uavs: 无人机数量
            num_users: 用户数量
            log_dir: 日志保存目录
        """
        self.num_uavs = num_uavs
        self.num_users = num_users
        self.log_dir = log_dir
        
        # 固定权重设置
        # 你可以根据需要调整这些权重
        self.weights = {
            'w_delay': 0.2,
            'w_distance': 0.3,
            'w_load': 0.2,
            'w_task_energy': 0.2,
            'w_move_energy': 0,
            'w_boundary': 0.1
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
                        uav_movement_delays, info, boundary_violation_penalty=0.0):
            # ========== 1. 提取指标 ==========
        delays = [raw_metrics[i]['user_actual_delay'] for i in range(self.num_users)]
        avg_delay = np.mean(delays)

        task_energies = [
            raw_metrics[i]['user_local_computation_energy'] +
            raw_metrics[i]['user_transmission_energy'] +
            raw_metrics[i]['user_uav_computation_energy']
            for i in range(self.num_users)
        ]
        total_task_energy = sum(task_energies)
        total_move_energy = sum(movement_energy_costs.values())
        total_system_energy = total_task_energy + total_move_energy

        # ========== 2. 归一化 ==========
        norm_delay = min(avg_delay / self.norm_params['max_avg_delay'], 1.0)
        norm_task_energy = min(total_task_energy / self.norm_params['max_task_energy'], 1.0)
        norm_move_energy = min(total_move_energy / self.norm_params['max_move_energy'], 1.0)
        norm_boundary = min(boundary_violation_penalty, 1.0)
        boundary_violation_penalty_raw = boundary_violation_penalty

        # ========== 3. 距离计算 ==========
        dists = [
            np.linalg.norm(
                info['uav_states'][info['user_assignments'][i]][:2] - 
                info['user_states'][i][:2]
            )
            for i in range(self.num_users)
        ]
        avg_distance = np.mean(dists)
        area_length = info.get('area_length', 1.0)
        area_width = info.get('area_width', 1.0)
        max_dist = np.sqrt(area_length**2 + area_width**2)
        norm_distance = min(avg_distance / max_dist, 1.0)

        # ========== 4. 负载均衡 ==========
        load_per_uav = np.zeros(self.num_uavs)
        for i in range(self.num_users):
            uav_id = info['user_assignments'][i]
            load_per_uav[uav_id] += info['user_states'][i, 2]  # task size
        total_load = load_per_uav.sum()
        if total_load > 0:
            norm_load_imbalance = (load_per_uav.max() - load_per_uav.min()) / total_load
        else:
            norm_load_imbalance = 0.0
        norm_load_imbalance = min(norm_load_imbalance, 1.0)

        # ========== 5. 奖励计算 ==========
        penalty = (
            self.weights['w_delay'] * norm_delay +
            self.weights['w_task_energy'] * norm_task_energy +
            self.weights['w_move_energy'] * norm_move_energy +
            self.weights['w_distance'] * norm_distance +
            self.weights['w_load'] * norm_load_imbalance +
            self.weights['w_boundary'] * norm_boundary
        )
        reward = -penalty

        # ========== 6. 返回组件==========
        reward_components = {
            'avg_delay': float(avg_delay),
            'total_energy': float(total_system_energy),
            'total_task_energy': float(total_task_energy),
            'total_move_energy': float(total_move_energy),
            'norm_delay': float(norm_delay),
            'norm_task_energy': float(norm_task_energy),
            'norm_move_energy': float(norm_move_energy),
            'avg_distance': float(avg_distance),
            'norm_distance': float(norm_distance),
            'load_per_uav': [float(v) for v in load_per_uav],
            'norm_load_imbalance': float(norm_load_imbalance),
            'boundary_violation_penalty_raw': float(boundary_violation_penalty_raw),
            'norm_boundary_violation': float(norm_boundary),
            'reward': float(reward)
        }
        
        return reward, reward_components
    
    
        
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
        }
