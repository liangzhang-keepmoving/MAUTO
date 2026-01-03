"""
UAV移动模块 - 处理UAV移动和能耗计算（基于 Zeng et al. 2019 模型）
"""
import numpy as np

# ----------------------------
# UAV 能耗模型（Zeng et al. 2019）
# ----------------------------
class UAVEnergyModel:
    def __init__(self):
        # 参数来自: Y. Zeng et al., "Energy Minimization for Wireless Communication with Rotary-Wing UAV", IEEE TWC 2019
        self.P0 = 79.86      # Blade profile power (W)
        self.Pi = 88.63      # Induced power in hover (W)
        self.U_tip = 120     # Rotor tip speed (m/s)
        self.v0 = 4.03       # Mean rotor induced velocity in hover (m/s)
        self.d0 = 0.6        # Fuselage drag ratio
        self.rho = 1.225     # Air density (kg/m^3)
        self.s = 0.05        # Rotor solidity
        self.A = 0.503       # Rotor disc area (m^2)

    def calculate_power(self, velocity):
        """
        计算给定速度下的总飞行功率（W）
        Args:
            velocity (float or np.ndarray): 飞行速度 (m/s)
        Returns:
            power (float or np.ndarray): 总功率 (W)
        """
        V = np.maximum(velocity, 0.0)  # 防止负速度
        
        # 1. Blade profile power
        part1 = self.P0 * (1 + (3 * V**2) / (self.U_tip**2))
        
        # 2. Induced power
        # 避免除零或负数开方（数值稳定）
        ind_inner = 1 + (V**4) / (4 * self.v0**4) - (V**2) / (2 * self.v0**2)
        # 理论上 ind_inner >= 0，但浮点误差可能造成微小负值
        ind_inner = np.maximum(ind_inner, 0.0)
        part2 = self.Pi * np.sqrt(np.sqrt(1 + (V**4) / (4 * self.v0**4)) - (V**2) / (2 * self.v0**2))
        
        # 3. Parasite power
        part3 = 0.5 * self.d0 * self.rho * self.s * self.A * (V**3)
        
        return part1 + part2 + part3


# ----------------------------
# UAV 移动管理器
# ----------------------------
class UAVMovementManager:
    """UAV移动管理器 - 基于固定时间步长和 Zeng 能耗模型"""

    def __init__(self, num_uavs, area_length, area_width, max_flight_distance, time_step=1.0):
        self.num_uavs = num_uavs
        self.area_length = area_length
        self.area_width = area_width
        self.max_flight_distance = max_flight_distance
        self.time_step = time_step  # 固定时间步长（秒）

        # 初始化能耗模型
        self.energy_model = UAVEnergyModel()

    def process_uav_movements(self, actions, uav_states):
        # 处理 actions 为 None 的情况（返回默认值 + 当前位置）
        if actions is None:
            dummy_pos = np.stack([uav_states[i].copy() for i in range(self.num_uavs)], axis=0)
            zero_dict = {i: 0.0 for i in range(self.num_uavs)}
            return zero_dict, zero_dict, dummy_pos, 0.0

        movement_energy_costs = {}
        movement_delays = {}
        uav_position = {}
        total_theoretical_distance = 0.0
        total_diff_distance = 0.0

        for uav_id in range(self.num_uavs):
            uav_key = f'uav_{uav_id}'
            current_x, current_y = uav_states[uav_id, 0], uav_states[uav_id, 1]

            if uav_key in actions and actions[uav_key] is not None and 'move_vector' in actions[uav_key]:
                dx, dy = actions[uav_key]['move_vector']

                # === 限制单步最大飞行距离 ===
                distance = np.sqrt(dx**2 + dy**2)
                if distance > self.max_flight_distance:
                    scale = self.max_flight_distance / distance
                    dx *= scale
                    dy *= scale
                    distance = self.max_flight_distance
                theoretical_distance = float(distance)
                total_theoretical_distance += theoretical_distance

                # === 计算新位置并应用边界约束 ===
                desired_x = current_x + dx
                desired_y = current_y + dy
                new_x = np.clip(desired_x, 0, self.area_length)
                new_y = np.clip(desired_y, 0, self.area_width)

                # 更新状态
                uav_states[uav_id, 0] = new_x
                uav_states[uav_id, 1] = new_y
                uav_position[uav_id] = uav_states[uav_id].copy()

                # === 计算实际移动距离 ===
                actual_distance = np.sqrt((new_x - current_x)**2 + (new_y - current_y)**2)
                diff_distance = max(theoretical_distance - float(actual_distance), 0.0)
                total_diff_distance += diff_distance

                # === 计算速度（基于固定时间步）===
                velocity = actual_distance / self.time_step if self.time_step > 0 else 0.0

                # === 计算功率和能耗 ===
                power = self.energy_model.calculate_power(velocity)
                energy_consumed = power * self.time_step
                movement_energy_costs[uav_id] = energy_consumed

                # === 移动时延：在固定时间步模型中，每步耗时 time_step ===
                movement_delays[uav_id] = self.time_step

            else:
                # 无移动动作：悬停
                uav_position[uav_id] = uav_states[uav_id].copy()
                movement_delays[uav_id] = self.time_step
                # 悬停速度 = 0
                power = self.energy_model.calculate_power(0.0)
                energy_consumed = power * self.time_step
                movement_energy_costs[uav_id] = energy_consumed

        uav_pos_array = np.stack([uav_position[i] for i in range(self.num_uavs)], axis=0)
        if total_theoretical_distance > 1e-8:
            boundary_violation_penalty = float(np.clip(total_diff_distance / total_theoretical_distance, 0.0, 1.0))
        else:
            boundary_violation_penalty = 0.0
        return movement_energy_costs, movement_delays, uav_pos_array, boundary_violation_penalty
