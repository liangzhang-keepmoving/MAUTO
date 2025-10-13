"""
时延计算模块 - 处理任务时延的详细计算
"""

import numpy as np

class DelayCalculator:
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
        
    def calculate_task_delays(self, actions, user_states, uav_states, user_assignments):

        """
        计算任务时延（基于actions中的用户竞争概率直接计算分配）
        
        Args:
            actions: 动作字典，格式为 {f'uav_{uav_id}': {'user_competition_probs': array, 'offloading_ratios': array, ...}}
            user_states: 用户状态数组 [N_users, 3] 包含 [x, y, task_size]
            uav_states: UAV状态数组 [N_uavs, 2] 包含 [x, y]
            user_assignments: 用户分配结果 {user_id: uav_id}
            
        Returns:
            dict: 包含实际时延、最大本地时延、最大卸载时延以及最大UAV计算时延的字典
            {
                'actual_delays': {uav_id: delay},
                'max_local_delays': {uav_id: max_local_delay},
                'max_offload_delays': {uav_id: max_offload_delay},
                'max_uav_computation_delays': {uav_id: max_uav_comp_delay}
            }
        """
        # 1. 按UAV分组用户
        uav_user_groups = {}
        for user_id, uav_id in user_assignments.items():
            if uav_id not in uav_user_groups:
                uav_user_groups[uav_id] = []
            uav_user_groups[uav_id].append(user_id)
        
        # 2. 为每个UAV计算时延（实际、最大本地、最大卸载）
        actual_delays = {}
        max_local_delays = {}
        max_offload_delays = {}
        max_uav_computation_delays = {}  # 新增：保存每个UAV的最大计算时延
        
        for uav_id, served_users in uav_user_groups.items():
            user_actual_delays = {}
            user_uav_computation_delays = {}  # 临时存储该UAV为每个用户的计算时延
            
            # UAV的总计算能力
            uav_total_cpu_frequency = self.uav_cpu_frequency  # 使用环境参数
            
            # 计算该UAV服务的用户数量，平均分配计算资源
            num_users_served = len(served_users)
            uav_cpu_per_user = uav_total_cpu_frequency / num_users_served if num_users_served > 0 else uav_total_cpu_frequency
            
            # 为每个用户计算时延
            for user_id in served_users:
               
                task_size = user_states[user_id, 2]
                uav_key = f'uav_{uav_id}'
                
                # 获取卸载比例
                offloading_ratio = 0.5  # 默认值
                if actions and uav_key in actions and 'offloading_ratios' in actions[uav_key]:
                    offloading_ratio = actions[uav_key]['offloading_ratios'][user_id]
                offloading_ratio = np.clip(offloading_ratio, 0.0, 1.0)
                
                # === 1. 实际时延计算（基于真实卸载比例）===
                local_task_size = task_size * (1 - offloading_ratio)
                offload_task_size = task_size * offloading_ratio
                
                
                local_delay = self._calculate_local_computation_delay(local_task_size, user_id)
               
                
                # 计算卸载时延并保存UAV计算时延
                if offload_task_size > 0:
                    offload_delay, uav_comp_delay = self._calculate_offloading_delay(
                        offload_task_size, user_id, uav_id, uav_cpu_per_user, 
                        user_states, uav_states, return_uav_delay=True
                    )
                    user_uav_computation_delays[user_id] = uav_comp_delay
                else:
                    offload_delay = 0.0
                    user_uav_computation_delays[user_id] = 0.0
                
                # 用户总时延 = max(本地时延, 卸载时延)（并行处理）
                user_actual_delay = max(local_delay, offload_delay)
               
                user_actual_delays[user_id] = user_actual_delay
            
            # UAV的时延 = 服务的所有用户中的最大时延
            actual_delays[uav_id] = max(user_actual_delays.values()) if user_actual_delays else 0.0
            
            # 保存该UAV的最大计算时延（所有用户中处理时间最长的）
            max_uav_computation_delays[uav_id] = max(user_uav_computation_delays.values()) if user_uav_computation_delays else 0.0
        
        # 确保所有UAV都有时延记录
        num_uavs = len(uav_states)
        for uav_id in range(num_uavs):
            if uav_id not in actual_delays:
                actual_delays[uav_id] = 0.0
                max_local_delays[uav_id] = 0.0  
                max_offload_delays[uav_id] = 0.0
                max_uav_computation_delays[uav_id] = 0.0
        
        # 返回包含四种时延的字典
        return {
            'actual_delays': actual_delays,
            'max_uav_computation_delays': max_uav_computation_delays  # 新增：每个UAV的最大计算时延
        }
    
    def _calculate_local_computation_delay(self, task_size, user_id):
        """计算用户本地计算时延"""
        if task_size <= 0:
            return 0.0
        
        # 用户本地计算能力参数（使用环境参数）
        user_cpu_frequency_ghz = self.user_cpu_frequency  # 用户设备CPU频率 (GHz)
        cpu_cycles_per_mb = self.cpu_cycles_per_mb         # 每MB任务需要的CPU周期数 (Megacycles)
        
        # 单位转换：GHz -> MHz，以匹配Megacycles
        user_cpu_frequency_mhz = user_cpu_frequency_ghz * 1000  # 转换为MHz
        
        # 计算时延 = 任务大小 * 每MB周期数 / CPU频率
        local_delay = task_size * cpu_cycles_per_mb / user_cpu_frequency_mhz  # 秒
        
        return local_delay
    
    def _calculate_offloading_delay(self, task_size, user_id, uav_id, uav_cpu_per_user, 
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
        transmission_delay = self._calculate_transmission_delay(task_size, user_id, uav_id, user_states, uav_states)
        
        # 2. UAV计算时延
        uav_computation_delay = self._calculate_uav_computation_delay(task_size, uav_cpu_per_user)
        
        # 总卸载时延 = 传输时延 + UAV计算时延（串行）
        total_offload_delay = transmission_delay + uav_computation_delay
        
        
        return total_offload_delay, uav_computation_delay
        
    
    def _calculate_transmission_delay(self, task_size, user_id, uav_id, user_states, uav_states):
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
        bandwidth = self.bandwidth  # 使用环境参数 MHz
        transmission_rate = bandwidth * np.log2(1 + snr_db)  # Mbps
        
        # 计算传输时延
        transmission_delay = task_size / transmission_rate  # 秒
        
        return transmission_delay
    
    def _calculate_uav_computation_delay(self, task_size, uav_cpu_frequency):
        """计算UAV计算时延"""
        if task_size <= 0:
            return 0.0
        
        # UAV计算能力参数（使用环境参数）
        cpu_cycles_per_mb = self.cpu_cycles_per_mb  # 每MB任务需要的CPU周期数 (Megacycles)
        
        # 单位转换：GHz -> MHz，以匹配Megacycles  
        uav_cpu_frequency_mhz = uav_cpu_frequency * 1000  # 转换为MHz
        
        # 计算时延 = 任务大小 * 每MB周期数 / 分配的CPU频率
        uav_delay = task_size * cpu_cycles_per_mb / uav_cpu_frequency_mhz  # 秒
        
        return uav_delay
    
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
    

# DelayCalculator = DelayCalculator()
# actions = {
#     'uav_0': {
#         'user_competition_probs': np.array([1., 1., 1., 0., 0.], dtype=np.float32), 
#         'offloading_ratios': np.array([0.1, 0.2, 0.3, 0., 0.], dtype=np.float32)
#     },
#     'uav_1': {
#         'user_competition_probs': np.array([0., 0., 0., 1., 1.], dtype=np.float32),
#         'offloading_ratios': np.array([0., 0., 0., 0.5, 0.5], dtype=np.float32)
#     }
# }
# user_states = np.array([[0.1, 0.1, 10], [0.2, 0.2, 10], [0.3, 0.3, 10], [0.4, 0.4, 10], [0.5, 0.5, 10]])
# uav_states = np.array([[0.1, 0.1], [0.2, 0.2]])
# user_assignments = {0: 0, 1: 0, 2: 0, 3: 1, 4: 1}
# print(DelayCalculator.calculate_task_delays(actions, user_states, uav_states, user_assignments))