"""
奖励系统模块 - 简化版
只考虑归一化后的时延和能耗
"""
import numpy as np

class AdaptiveRewardSystem:
    """简化版奖励系统 - 固定权重，仅关注时延和能耗"""
    
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
            'w_delay': 0.3,
            'w_distance': 0.2,
            'w_load': 0.2,
            'w_task_energy': 0.3,
            'w_move_energy': 0.3,
            'w_boundary': 0.2
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
        """
        计算简化版奖励
        Reward = - (w_delay * norm_delay + w_task_energy * norm_task_energy + w_move_energy * norm_move_energy)
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
        
        # 聚合指标
        avg_delay = np.mean(user_delays) if user_delays else 0.0
        
        # 总任务能耗
        total_task_energy = sum(user_total_energies)

        total_move_energy = 0.0
        try:
            if isinstance(movement_energy_costs, dict):
                total_move_energy = float(sum(float(v) for v in movement_energy_costs.values()))
            elif isinstance(movement_energy_costs, (list, tuple, np.ndarray)):
                total_move_energy = float(np.sum(np.asarray(movement_energy_costs, dtype=float)))
            elif movement_energy_costs is not None:
                total_move_energy = float(movement_energy_costs)
        except Exception:
            total_move_energy = 0.0
        
        total_system_energy = total_task_energy + total_move_energy
        
        norm_delay = self._normalize(
            avg_delay, 
            0.0,
            self.norm_params.get('max_avg_delay', 1.0)
        )
        
        norm_task_energy = self._normalize(
            total_task_energy,
            0.0,
            self.norm_params.get('max_task_energy', 1.0)
        )

        norm_move_energy = self._normalize(
            total_move_energy,
            0.0,
            self.norm_params.get('max_move_energy', 1.0)
        )
        
        w_task_energy = float(self.weights.get('w_task_energy', 0.0))
        w_move_energy = float(self.weights.get('w_move_energy', 0.0))
        norm_energy = w_task_energy * norm_task_energy + w_move_energy * norm_move_energy

        boundary_violation_penalty_raw = 0.0
        try:
            boundary_violation_penalty_raw = float(boundary_violation_penalty)
        except Exception:
            boundary_violation_penalty_raw = 0.0
        norm_boundary = float(np.clip(boundary_violation_penalty_raw, 0.0, 1.0))
        
        avg_distance = 0.0
        norm_distance = 0.0
        load_per_uav = [0.0 for _ in range(self.num_uavs)]
        norm_load_imbalance = 0.0
        try:
            assignments = info.get('user_assignments', {}) if isinstance(info, dict) else {}
            uav_states = np.asarray(info.get('uav_states', []), dtype=float) if isinstance(info, dict) else np.asarray([], dtype=float)
            user_states = np.asarray(info.get('user_states', []), dtype=float) if isinstance(info, dict) else np.asarray([], dtype=float)
            
            dists = []
            if uav_states.ndim == 2 and user_states.ndim == 2 and len(assignments) > 0:
                area_length = float(info.get('area_length', 0.0)) if isinstance(info, dict) else 0.0
                area_width = float(info.get('area_width', 0.0)) if isinstance(info, dict) else 0.0
                if area_length > 0.0 and area_width > 0.0:
                    max_dist = float(np.sqrt(area_length * area_length + area_width * area_width))
                else:
                    max_axis = 1.0
                    if uav_states.size > 0:
                        max_axis = max(max_axis, float(np.max(uav_states[:, :2])))
                    if user_states.size > 0:
                        max_axis = max(max_axis, float(np.max(user_states[:, :2])))
                    max_dist = float(np.sqrt((max_axis ** 2) * 2.0))
                
                for user_id in range(self.num_users):
                    uav_id = assignments.get(user_id, assignments.get(str(user_id), None))
                    if uav_id is None:
                        continue
                    try:
                        uav_id = int(uav_id)
                    except Exception:
                        continue
                    if uav_id < 0 or uav_id >= uav_states.shape[0] or user_id >= user_states.shape[0]:
                        continue
                    dx = float(uav_states[uav_id, 0] - user_states[user_id, 0])
                    dy = float(uav_states[uav_id, 1] - user_states[user_id, 1])
                    dists.append(float(np.sqrt(dx * dx + dy * dy)))
                
                if dists:
                    avg_distance = float(np.mean(dists))
                    if max_dist > 1e-8:
                        norm_distance = float(np.clip(avg_distance / max_dist, 0.0, 1.0))

                if user_states.ndim == 2 and user_states.shape[1] >= 3:
                    for user_id in range(self.num_users):
                        uav_id = assignments.get(user_id, assignments.get(str(user_id), None))
                        if uav_id is None:
                            continue
                        try:
                            uav_id = int(uav_id)
                        except Exception:
                            continue
                        if uav_id < 0 or uav_id >= self.num_uavs or user_id >= user_states.shape[0]:
                            continue
                        load_per_uav[uav_id] += float(user_states[user_id, 2])

                    total_load = float(np.sum(load_per_uav))
                    if total_load > 1e-8:
                        max_load = float(np.max(load_per_uav))
                        min_load = float(np.min(load_per_uav))
                        norm_load_imbalance = float(np.clip((max_load - min_load) / (total_load + 1e-8), 0.0, 1.0))
        except Exception:
            avg_distance = 0.0
            norm_distance = 0.0
            load_per_uav = [0.0 for _ in range(self.num_uavs)]
            norm_load_imbalance = 0.0

        penalty = (
            float(self.weights.get('w_delay', 0.0)) * norm_delay +
            float(self.weights.get('w_task_energy', 0.0)) * norm_task_energy +
            float(self.weights.get('w_move_energy', 0.0)) * norm_move_energy +
            float(self.weights.get('w_distance', 0.0)) * norm_distance +
            float(self.weights.get('w_load', 0.0)) * norm_load_imbalance +
            float(self.weights.get('w_boundary', 0.0)) * norm_boundary
        )
        
        reward = -penalty
        
        # ========== 4. 返回详情 ==========
        reward_components = {
            'avg_delay': avg_delay,
            'total_energy': total_system_energy,
            'total_task_energy': total_task_energy,
            'total_move_energy': total_move_energy,
            'norm_delay': norm_delay,
            'norm_energy': norm_energy,
            'norm_task_energy': norm_task_energy,
            'norm_move_energy': norm_move_energy,
            'avg_distance': avg_distance,
            'norm_distance': norm_distance,
            'load_per_uav': [float(v) for v in load_per_uav],
            'norm_load_imbalance': norm_load_imbalance,
            'boundary_violation_penalty_raw': boundary_violation_penalty_raw,
            'norm_boundary_violation': norm_boundary,
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
        }
