"""
UAV移动模块 - 处理UAV移动、边界约束和碰撞检测
"""
import numpy as np

class UAVMovementManager:
    """UAV移动管理器 - 处理移动、边界检查和碰撞检测"""
    
    def __init__(self, num_uavs, area_length, area_width, uav_max_speed, time_step,uav_battery_capacity):
        """
        初始化UAV移动管理器
        
        Args:
            num_uavs: UAV数量
            area_length: 区域长度 (米)
            area_width: 区域宽度 (米)
            uav_max_speed: UAV最大速度 (m/s)
            time_step: 时间步长 (秒)
            move_power: 移动功耗
            hover_power: 悬停功耗
            uav_battery_capacity: 电池容量
        """
        self.num_uavs = num_uavs
        self.area_length = area_length
        self.area_width = area_width
        self.uav_max_speed = uav_max_speed
        self.time_step = time_step
        self.uav_battery_capacity = uav_battery_capacity
        
        # 碰撞检测参数
        self.min_safe_distance = 5.0   # 最小安全距离 (米)
        self.critical_distance = 2.0   # 严重危险距离 (米)
        self.hover_power = 0.5*9.65*self.uav_max_speed*self.uav_max_speed*0.8
        
    def process_uav_movements(self, actions, uav_states):
        """
        处理UAV移动动作
        
        Args:
            actions: 动作字典，格式为 {f'uav_{uav_id}': {'movement_direction': float, 'movement_distance': float, ...}}
            uav_states: UAV状态数组 [N, 2] 只包含 [x, y] 位置信息
            
        Returns:
            tuple: (movement_penalties, movement_energy_costs)
                - movement_penalties: UAV移动惩罚字典 {f'uav_{uav_id}': penalty} (包含边界惩罚和碰撞惩罚)
                - movement_energy_costs: UAV移动能耗字典 {f'uav_{uav_id}': energy_consumed}
        """
        if actions is None:
            return {f'uav_{i}': 0.0 for i in range(self.num_uavs)}, {f'uav_{i}': 0.0 for i in range(self.num_uavs)}
        
        boundary_penalties = {}
        movement_energy_costs = {}
        uav_movement_actual_delays = {}
        
        # 存储原始位置用于计算实际移动距离
        original_positions = uav_states.copy()
            
        for uav_id in range(self.num_uavs):
            uav_key = f'uav_{uav_id}'
            boundary_penalties[uav_key] = 0.0  # 初始化移动惩罚
            movement_energy_costs[uav_key] = 0.0  # 初始化移动能耗
            
            if uav_key in actions and actions[uav_key] is not None:
                action = actions[uav_key]
                
                # 检查动作是否包含移动参数
                if 'movement_direction' in action and 'movement_distance' in action:
                    direction = action['movement_direction']  # 弧度 [0, 2π]
                    distance = action['movement_distance']    # 米 [0, max_distance]
                    
                    # 限制移动距离在合理范围内
                    distance = np.clip(distance, 0, self.uav_max_speed * self.time_step)
                    
                    # 获取当前位置
                    current_x = uav_states[uav_id, 0]
                    current_y = uav_states[uav_id, 1]
                    
                    # 计算期望的新位置
                    desired_x = current_x + distance * np.cos(direction)
                    desired_y = current_y + distance * np.sin(direction)
                    
                    # 检查是否超出边界并计算惩罚
                    boundary_penalty = self._calculate_boundary_penalty(desired_x, desired_y)
                    boundary_penalties[uav_key] = boundary_penalty
                    
                    # 应用边界约束
                    new_x = np.clip(desired_x, 0, self.area_length)
                    new_y = np.clip(desired_y, 0, self.area_width)
                    
                    # 更新UAV位置
                    uav_states[uav_id, 0] = new_x
                    uav_states[uav_id, 1] = new_y
                    
                    
                    
                    # 计算实际移动距离
                    actual_distance = np.sqrt((new_x - current_x)**2 + (new_y - current_y)**2)
                    uav_movement_actual_delays[uav_key] = actual_distance/self.uav_max_speed
                    # 计算移动能耗
                    energy_consumed = self._calculate_energy_consumption(actual_distance)
                    movement_energy_costs[uav_key] = energy_consumed
                else:
                    # 如果没有移动参数，UAV保持原位（悬停）
                    energy_consumed = self._calculate_energy_consumption(0.0)  # 悬停能耗
                    movement_energy_costs[uav_key] = energy_consumed
            else:
                # 如果没有对应UAV的动作，保持原位（悬停）
                energy_consumed = self._calculate_energy_consumption(0.0)  # 悬停能耗
                movement_energy_costs[uav_key] = energy_consumed
        
        # 检查UAV间碰撞并计算惩罚
        collision_penalties = self._check_uav_collisions(uav_states)
        
        # 合并边界惩罚和碰撞惩罚
        movement_penalties = {}
        for uav_id in range(self.num_uavs):
            uav_key = f'uav_{uav_id}'
            boundary_penalty = boundary_penalties.get(uav_key, 0.0)
            collision_penalty = collision_penalties.get(uav_key, 0.0)
            movement_penalties[uav_key] = boundary_penalty + collision_penalty
        
        return movement_penalties, movement_energy_costs, uav_movement_actual_delays
    
    def _calculate_boundary_penalty(self, desired_x, desired_y):
        """
        计算边界违规惩罚
        
        Args:
            desired_x: 期望的X坐标
            desired_y: 期望的Y坐标
            
        Returns:
            float: 边界惩罚值 (负数)
        """
        boundary_violation_distance = 0.0
        out_of_bounds = False
        
        if desired_x < 0:
            boundary_violation_distance += abs(desired_x)
            out_of_bounds = True
        elif desired_x > self.area_length:
            boundary_violation_distance += (desired_x - self.area_length)
            out_of_bounds = True
            
        if desired_y < 0:
            boundary_violation_distance += abs(desired_y)
            out_of_bounds = True
        elif desired_y > self.area_width:
            boundary_violation_distance += (desired_y - self.area_width)
            out_of_bounds = True
        
        # 计算边界惩罚
        if out_of_bounds:
            # 惩罚 = 基础惩罚 + 违规距离惩罚
            base_penalty = -1.0  # 基础边界违规惩罚
            distance_penalty = -0.1 * boundary_violation_distance  # 每米超出惩罚0.1
            return base_penalty + distance_penalty
        
        return 0.0
    
    def _calculate_energy_consumption(self, actual_distance):
        """
        计算能耗（基于实际飞行时间）
        
        Args:
            actual_distance: 实际移动距离
            
        Returns:
            float: 能耗值
        """
    
        # 计算实际飞行时间
        flight_time = actual_distance / self.uav_max_speed
        # 剩余时间为悬停
        hover_time = 2 - flight_time
        
       
        energy_consumed = flight_time*0.5*9.65*self.uav_max_speed*self.uav_max_speed+hover_time*self.hover_power
    
        return energy_consumed
    
    def _check_uav_collisions(self, uav_states):
        """
        检查UAV间碰撞并计算惩罚
        
        Args:
            uav_states: UAV状态数组
            
        Returns:
            dict: 碰撞惩罚字典 {f'uav_{uav_id}': penalty}
        """
        collision_penalties = {f'uav_{i}': 0.0 for i in range(self.num_uavs)}
        
        # 检查所有UAV对的距离
        for i in range(self.num_uavs):
            for j in range(i + 1, self.num_uavs):
                # 计算两个UAV之间的水平距离
                uav1_pos = uav_states[i, :2]  # [x, y]
                uav2_pos = uav_states[j, :2]  # [x, y]
                distance = np.sqrt((uav1_pos[0] - uav2_pos[0])**2 + (uav1_pos[1] - uav2_pos[1])**2)
                
                # 计算碰撞惩罚
                if distance < self.critical_distance:
                    # 严重碰撞风险
                    penalty = -5.0 - (self.critical_distance - distance) * 2.0  # 基础惩罚 + 距离惩罚
                elif distance < self.min_safe_distance:
                    # 一般安全距离违规
                    penalty = -2.0 - (self.min_safe_distance - distance) * 0.5
                else:
                    # 安全距离，无惩罚
                    penalty = 0.0
                
                # 对两个UAV都施加惩罚
                if penalty < 0:
                    collision_penalties[f'uav_{i}'] += penalty
                    collision_penalties[f'uav_{j}'] += penalty
        
        return collision_penalties
