"""
UAV移动模块 - 处理UAV移动和能耗计算
"""
import numpy as np

class UAVMovementManager:
    """UAV移动管理器 - 处理移动和能耗计算"""
    
    def __init__(self, num_uavs, area_length, area_width, uav_max_speed,max_flight_distance):

        self.num_uavs = num_uavs
        self.area_length = area_length
        self.area_width = area_width
        self.uav_max_speed = uav_max_speed
        self.max_flight_distance = max_flight_distance
        
        # 悬停功耗计算
        self.hover_power = 0.5 * 9.65 * self.uav_max_speed * self.uav_max_speed * 0.8
        
    def process_uav_movements(self, actions, uav_states):
        if actions is None:
            return (
                {i: 0.0 for i in range(self.num_uavs)}, 
                {i: 0.0 for i in range(self.num_uavs)}
            )
        
        movement_energy_costs = {}
        movement_delays = {}
        uav_position = {}
        
        # 存储原始位置用于计算实际移动距离
        original_positions = uav_states.copy()
            
        for uav_id in range(self.num_uavs):
            uav_key = f'uav_{uav_id}'
            
            if uav_key in actions and actions[uav_key] is not None:
                action = actions[uav_key]
                
                # 检查动作是否包含移动参数 (使用笛卡尔坐标系: move_vector (dx, dy))
                if 'move_vector' in action:
                    dx, dy = action['move_vector']
                    
                    # 限制单步移动距离在最大飞行距离内 (虽然train_ddpg已经做了，这里作为双重保险)
                    # 注意：dx, dy 是已经乘过 max_distance 的实际位移
                    distance = np.sqrt(dx**2 + dy**2)
                    if distance > self.max_flight_distance:
                        scale = self.max_flight_distance / distance
                        dx *= scale
                        dy *= scale
                        distance = self.max_flight_distance
                    
                    # 获取当前位置
                    current_x = uav_states[uav_id, 0]
                    current_y = uav_states[uav_id, 1]
                    
                    # 计算期望的新位置
                    desired_x = current_x + dx
                    desired_y = current_y + dy
                    
                    # 应用边界约束（自动限制在边界内）
                    new_x = np.clip(desired_x, 0, self.area_length)
                    new_y = np.clip(desired_y, 0, self.area_width)
                    
                    # 更新UAV位置
                    uav_states[uav_id, 0] = new_x
                    uav_states[uav_id, 1] = new_y
                    uav_position[uav_id] = uav_states[uav_id].copy()
                    
                    # 计算实际移动距离
                    actual_distance = np.sqrt((new_x - current_x)**2 + (new_y - current_y)**2)
                    
                    # 计算移动时延
                    if actual_distance > 0:
                        movement_delays[uav_id] = actual_distance / self.uav_max_speed
                    else:
                        movement_delays[uav_id] = 0.0
                    
                    # 计算移动能耗
                    energy_consumed = self._calculate_energy_consumption(actual_distance)
                    movement_energy_costs[uav_id] = energy_consumed
                else:
                    # 兼容旧代码或无移动参数情况
                    # 如果没有移动参数，UAV保持原位（悬停）
                    uav_position[uav_id] = uav_states[uav_id].copy()
                    movement_delays[uav_id] = 0.0
                    energy_consumed = self._calculate_energy_consumption(0.0)  # 悬停能耗
                    movement_energy_costs[uav_id] = energy_consumed
            else:
                # 如果没有对应UAV的动作，保持原位（悬停）
                uav_position[uav_id] = uav_states[uav_id].copy()
                movement_delays[uav_id] = 0.0
                energy_consumed = self._calculate_energy_consumption(0.0)  # 悬停能耗
                movement_energy_costs[uav_id] = energy_consumed
        uav_pos_array = np.stack([uav_position[i] for i in range(self.num_uavs)], axis=0)

        return movement_energy_costs, movement_delays,uav_pos_array
    
    def _calculate_energy_consumption(self, actual_distance):
      
        flight_time = actual_distance / self.uav_max_speed
        # 剩余时间为悬停
        hover_time = self.max_flight_distance/self.uav_max_speed - flight_time
        # 飞行能耗 + 悬停能耗
        flight_power = 0.5 * 9.65 * self.uav_max_speed * self.uav_max_speed
        energy_consumed = flight_time * flight_power + hover_time * self.hover_power
      
        
        return energy_consumed

