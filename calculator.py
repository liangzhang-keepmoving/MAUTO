"""
时延计算模块 - 处理任务时延的详细计算
"""

import numpy as np

class Calculator:
    """时延计算器 - 计算任务的本地时延、传输时延和UAV计算时延"""
    
    def __init__(self, uav_height=50, uav_cpu_frequency=7.5, user_cpu_frequency=1.0, 
                 cpu_cycles_per_mb=1000, bandwidth=5.0, transmission_power=20.0,
                 path_loss_exponent=2.8, reference_distance=1.0, reference_path_loss=50.0, noise_power=-90.0):
        """
        初始化时延计算器
        
        Args:
            uav_height: UAV飞行高度 (米)
            uav_cpu_frequency: UAV CPU频率 (GHz)
            user_cpu_frequency: 用户CPU频率 (GHz)
            cpu_cycles_per_mb: 每MB任务的CPU周期数 (Megacycles)
            bandwidth: 带宽 (MHz)
            transmission_power: 发射功率 (dBm)
            path_loss_exponent: 路径损耗指数
            reference_distance: 参考距离 (m)
            reference_path_loss: 参考路径损耗 (dB)
            noise_power: 噪声功率 (dBm)
        """
        self.uav_height = uav_height
        self.uav_cpu_frequency = uav_cpu_frequency
        self.user_cpu_frequency = user_cpu_frequency
        self.cpu_cycles_per_mb = cpu_cycles_per_mb
        self.bandwidth = bandwidth
        self.transmission_power = transmission_power
        self.path_loss_exponent = path_loss_exponent
        self.reference_distance = reference_distance
        self.reference_path_loss = reference_path_loss
        self.noise_power = noise_power
        
    def calculate_task(self, actions, user_states, uav_states, user_assignments):
        # 1. 按UAV分组用户
        uav_user_groups = {}
        for user_id, uav_id in user_assignments.items():
            if uav_id not in uav_user_groups:
                uav_user_groups[uav_id] = []
            uav_user_groups[uav_id].append(user_id)
        
        # 2. 初始化用户视角的时延字典
        user_local_computation_delays = {}
        user_local_computation_energys = {}
        
        user_transmission_delays = {}
        user_transmission_energys = {}

        user_uav_computation_delays = {}
        user_uav_computation_energys = {}
        
        total_offload_delays = {}

        user_actual_delays = {}
        
        # 4. 为每个UAV计算时延
        for uav_id, served_users in uav_user_groups.items():
          
            # UAV的总计算能力
            uav_total_cpu_frequency = self.uav_cpu_frequency  # 使用环境参数
            uav_total_bandwidth = self.bandwidth
            
            # 计算该UAV服务的用户数量，平均分配计算资源
            num_users_served = len(served_users)
            uav_cpu_per_user = uav_total_cpu_frequency / num_users_served if num_users_served > 0 else uav_total_cpu_frequency
            uav_bandwidth_per_user = uav_total_bandwidth / num_users_served if num_users_served > 0 else uav_total_bandwidth
            # uav_bandwidth_per_user =1
            # 为每个用户计算时延
            for user_id in served_users:
                task_size = user_states[user_id, 2]
                uav_key = f'uav_{uav_id}'
                
                # 获取卸载比例
                offloading_ratio = 0.5  # 默认值
                if actions and uav_key in actions and 'offloading_ratios' in actions[uav_key]:
                    offloading_ratio = actions[uav_key]['offloading_ratios'][user_id]
                offloading_ratio = np.clip(offloading_ratio, 0.0, 1.0)
                
                # 计算本地和卸载的任务大小
                local_task_size = task_size * (1 - offloading_ratio)
                offload_task_size = task_size * offloading_ratio
                
                # === 用户视角：计算本地处理时延 ===
                local_delay, local_energy = self._calculate_local_computation(local_task_size, user_id)
                user_local_computation_delays[user_id] = local_delay
                user_local_computation_energys[user_id] = local_energy
                
                # === 用户视角：计算传输时延和UAV计算时延 ===
                if offload_task_size > 0:
                    total_offload_delay, uav_computation_delay, transmission_delay, uav_energy, transmission_energy = self._calculate_offloading(
                        offload_task_size, user_id, uav_id, uav_cpu_per_user, uav_bandwidth_per_user,
                        user_states, uav_states, return_uav_delay=True
                    )
                    user_transmission_delays[user_id] = transmission_delay
                    user_transmission_energys[user_id] = transmission_energy

                    user_uav_computation_delays[user_id] = uav_computation_delay
                    user_uav_computation_energys[user_id] = uav_energy

                    total_offload_delays[user_id] = total_offload_delay
                    
                else:
                    total_offload_delay = 0.0
                    user_transmission_delays[user_id] = 0.0
                    user_transmission_energys[user_id] = 0.0

                    user_uav_computation_delays[user_id] = 0.0
                    user_uav_computation_energys[user_id] = 0.0

                    total_offload_delay[user_id] = 0.0
                    
                
                # 用户总时延 = max(本地时延, 卸载时延)（并行处理）
                user_actual_delay = max(local_delay, total_offload_delay)
                user_actual_delays[user_id] = user_actual_delay
        
        # 5. 确保所有用户都有时延记录（未分配到UAV的用户）
        num_users = len(user_states)
        for user_id in range(num_users):
            if user_id not in user_local_computation_delays:
                # 未分配的用户，假设全部本地处理
                task_size = user_states[user_id, 2]
                local_delay, local_energy = self._calculate_local_computation_delay(task_size, user_id)
                user_local_computation_delays[user_id] = local_delay
                user_local_computation_energys[user_id] = local_energy
                user_transmission_delays[user_id] = 0.0
                user_uav_computation_delays[user_id] = 0.0
                total_offload_delays[user_id] = 0.0
                user_actual_delays[user_id] = local_delay
        
        # 7. 返回包含用户视角和UAV视角的时延数据
        return {
            'user_local_computation_delays': user_local_computation_delays,
            'user_local_computation_energy': user_local_computation_energys,
            'user_transmission_delays': user_transmission_delays,
            'user_transmission_energy': user_transmission_energys,
            'user_uav_computation_delays': user_uav_computation_delays,
            'user_uav_computation_energy': user_uav_computation_energys,
            'total_offload_delay': total_offload_delays,
            'user_actual_delays': user_actual_delays
            }
    
    def _calculate_local_computation(self, task_size, user_id):
        """计算用户本地计算时延"""
        if task_size <= 0:
            return 0.0,0.0
        
        # 用户本地计算能力参数（使用环境参数）
        user_cpu_frequency_ghz = self.user_cpu_frequency  # 用户设备CPU频率 (GHz)
        cpu_cycles_per_mb = self.cpu_cycles_per_mb         # 每MB任务需要的CPU周期数 (Megacycles)
        
        # 单位转换：GHz -> MHz，以匹配Megacycles
        user_cpu_frequency_mhz = user_cpu_frequency_ghz * 1000  # 转换为MHz
        
        # 计算时延 = 任务大小 * 每MB周期数 / CPU频率
        local_delay = task_size * cpu_cycles_per_mb / user_cpu_frequency_mhz  # 秒

        local_energy = task_size * self.user_cpu_frequency *self.user_cpu_frequency
        
        return local_delay, local_energy
    
    def _calculate_offloading(self, task_size, user_id, uav_id, uav_cpu_per_user, uav_bandwidth_per_user,
                                   user_states, uav_states, return_uav_delay=False):
        """计算卸载时延（传输时延 + UAV计算时延）
        
        Args:
            task_size: 任务大小
            user_id: 用户ID
            uav_id: UAV ID
            uav_cpu_per_user: 每个用户分配的UAV CPU频率
            user_states: 用户状态数组
            uav_states: UAV状态数组
            return_uav_delay: 是否返回UAV计算时延
            
        Returns:
            如果return_uav_delay=True: (总卸载时延, UAV计算时延)
            如果return_uav_delay=False: 总卸载时延
        """
        
        
        # 1. 传输时延
        transmission_delay = self._calculate_transmission_delay(task_size, user_id, uav_id, user_states, uav_states,uav_bandwidth_per_user)
        transmission_energy = self.transmission_power * transmission_delay
        
        # 2. UAV计算时延
        uav_computation_delay, uav_energy = self._calculate_uav_computation(task_size, uav_cpu_per_user)
        
        # 总卸载时延 = 传输时延 + UAV计算时延（串行）
        total_offload_delay = transmission_delay + uav_computation_delay
        
        
        return total_offload_delay, uav_computation_delay, transmission_delay, uav_energy, transmission_energy
        
    
    def _calculate_transmission_delay(self, task_size, user_id, uav_id, user_states, uav_states,uav_bandwidth_per_user):
        """计算传输时延"""
        # 计算用户到UAV的三维距离
        user_pos = user_states[user_id, :2]   # [x, y] 用户在地面，z=0
        uav_pos = uav_states[uav_id, :2]      # [x, y] UAV的水平位置
        
        # 使用三维距离计算函数
        distance_3d = self._calculate_distance_3d(uav_pos, user_pos)
        
        # 使用环境的无线信道模型参数
        path_loss_exponent = self.path_loss_exponent      # 路径损耗指数
       
        reference_path_loss = self.reference_path_loss    # 参考路径损耗 (dB)
        noise_power = self.noise_power                    # 噪声功率 (dBm)
        
        # 计算路径损耗 (dB) - 使用三维距离
        
        # 计算接收功率 (dBm)
        tx_power = self.transmission_power  # 使用环境参数
        
        
        # 计算信噪比 (dB)
        snr_db = tx_power*reference_path_loss/distance_3d/distance_3d/noise_power
        
        # 计算传输速率 (Shannon公式)
        bandwidth = uav_bandwidth_per_user  # 使用环境参数 MHz
        transmission_rate = bandwidth * np.log2(1 + snr_db)  # Mbps
        
        # 计算传输时延
        transmission_delay = task_size / transmission_rate  # 秒
        
        return transmission_delay
    
    def _calculate_uav_computation(self, task_size, uav_cpu_frequency):
        """计算UAV计算时延"""
        if task_size <= 0:
            return 0.0
        
        # UAV计算能力参数（使用环境参数）
        cpu_cycles_per_mb = self.cpu_cycles_per_mb  # 每MB任务需要的CPU周期数 (Megacycles)
        
        # 单位转换：GHz -> MHz，以匹配Megacycles  
        uav_cpu_frequency_mhz = uav_cpu_frequency * 1000  # 转换为MHz
        
        # 计算时延 = 任务大小 * 每MB周期数 / 分配的CPU频率
        uav_delay = task_size * cpu_cycles_per_mb / uav_cpu_frequency_mhz  # 秒
        uav_energy = task_size * uav_cpu_frequency * uav_cpu_frequency
        
        return uav_delay, uav_energy
    
    def _calculate_distance_3d(self, uav_pos, user_pos):
        """计算UAV和用户之间的三维距离
        
        Args:
            uav_pos: UAV的水平位置 [x, y]
            user_pos: 用户的水平位置 [x, y] (z=0，在地面)
            
        Returns:
            float: 三维距离 (米)
        """
        horizontal_distance = np.sqrt((uav_pos[0] - user_pos[0])**2 + (uav_pos[1] - user_pos[1])**2)
        distance_3d = np.sqrt(horizontal_distance**2 + self.uav_height**2)
        return distance_3d
    

