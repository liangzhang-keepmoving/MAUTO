"""
任务处理能耗计算模块 - 计算任务处理过程中的各种能耗
"""
import numpy as np

class TaskEnergyCalculator:
    """任务处理能耗计算器 - 计算用户本地处理、传输和UAV处理的能耗"""
    
    def __init__(self, user_cpu_frequency=1.0, uav_cpu_frequency=5.0, cpu_cycles_per_mb=1000,
                 user_cpu_power=2.0, uav_cpu_power=10.0, user_transmission_power=0.1,
                 bandwidth=5.0, transmission_power=20.0, path_loss_exponent=2.8, 
                 reference_distance=1.0, reference_path_loss=50.0, noise_power=-90.0, uav_height=50):
        """
        初始化任务处理能耗计算器
        
        Args:
            user_cpu_frequency: 用户CPU频率 (GHz)
            uav_cpu_frequency: UAV CPU频率 (GHz)
            cpu_cycles_per_mb: 每MB任务的CPU周期数 (Megacycles)
            user_cpu_power: 用户CPU功耗 (W)
            uav_cpu_power: UAV CPU功耗 (W)
            user_transmission_power: 用户传输功耗 (W)
            bandwidth: 带宽 (MHz)
            transmission_power: 发射功率 (dBm)
            path_loss_exponent: 路径损耗指数
            reference_distance: 参考距离 (m)
            reference_path_loss: 参考路径损耗 (dB)
            noise_power: 噪声功率 (dBm)
            uav_height: UAV飞行高度 (米)
        """
        # CPU参数
        self.user_cpu_frequency = user_cpu_frequency
        self.uav_cpu_frequency = uav_cpu_frequency
        self.cpu_cycles_per_mb = cpu_cycles_per_mb
        
        # 功耗参数
        self.user_cpu_power = user_cpu_power           # 用户CPU功耗 (W)
        self.uav_cpu_power = uav_cpu_power             # UAV CPU功耗 (W)
        self.user_transmission_power = user_transmission_power  # 用户传输功耗 (W)
        
        # 通信参数
        self.bandwidth = bandwidth
        self.transmission_power = transmission_power
        self.path_loss_exponent = path_loss_exponent
        self.reference_distance = reference_distance
        self.reference_path_loss = reference_path_loss
        self.noise_power = noise_power
        self.uav_height = uav_height
        
    def calculate_task_processing_energy(self, actions, user_assignments, user_states, uav_states):
        """
        计算任务处理能耗
        
        Args:
            actions: 动作字典
            user_assignments: 用户分配结果 {user_id: uav_id}
            user_states: 用户状态数组 [x, y, task_size]
            uav_states: UAV状态数组 [x, y, battery_level]
            
        Returns:
            dict: 用户视角的任务处理能耗字典
                {
                    'user_local_energy': {user_id: energy},      # 用户X任务在本地处理的实际能耗
                    'user_offload_energy': {user_id: energy},    # 用户X任务卸载处理的实际能耗（传输+UAV）
                    'user_total_energy': {user_id: energy},      # 用户X任务的总能耗（本地+卸载）
                    'max_local_energy': {user_id: energy},       # 用户X任务全本地处理的能耗
                    'max_offload_energy': {user_id: energy},     # 用户X任务全卸载处理的能耗（传输+UAV）
                    'user_max_energy': {user_id: energy}         # 用户X任务的最大处理能耗 = max(全本地, 全卸载)
                }
        """
        num_users = len(user_states)
        num_uavs = len(uav_states)
        
        # 初始化用户视角的能耗字典
        user_local_energy = {user_id: 0.0 for user_id in range(num_users)}     # 用户本地处理能耗
        user_offload_energy = {user_id: 0.0 for user_id in range(num_users)}   # 用户卸载处理能耗（传输+UAV）
        user_total_energy = {user_id: 0.0 for user_id in range(num_users)}     # 用户总能耗
        
        # 初始化最大能耗基准值
        max_local_energy = {user_id: 0.0 for user_id in range(num_users)}      # 全本地处理能耗
        max_offload_energy = {user_id: 0.0 for user_id in range(num_users)}    # 全卸载处理能耗
        user_max_energy = {user_id: 0.0 for user_id in range(num_users)}       # 用户最大处理能耗
        
        # 按UAV分组用户，计算每个UAV的CPU分配
        uav_user_groups = {}
        for user_id, uav_id in user_assignments.items():
            if uav_id not in uav_user_groups:
                uav_user_groups[uav_id] = []
            uav_user_groups[uav_id].append(user_id)
        
        # 为每个用户计算任务处理能耗
        for user_id in range(num_users):
            if user_id not in user_assignments:
                continue
                
            uav_id = user_assignments[user_id]
            task_size = user_states[user_id, 2]
            #print(f"task_size: {task_size}")
            
            # 获取卸载比例
            uav_key = f'uav_{uav_id}'
            if actions and uav_key in actions and 'offloading_ratios' in actions[uav_key]:
                offloading_ratio = actions[uav_key]['offloading_ratios'][user_id]
            else:
                offloading_ratio = 0.5  # 默认50%卸载
            offloading_ratio = np.clip(offloading_ratio, 0.0, 1.0)
            
            # 任务分割
            local_task_size = task_size * (1 - offloading_ratio)
            #print(f"local_task_size: {local_task_size}")
            offload_task_size = task_size * offloading_ratio
            #print(f"offload_task_size: {offload_task_size}")
            
            # === 1. 实际能耗计算（基于用户的卸载策略）===
            # 1.1 用户本地处理能耗
            user_local_energy[user_id] = self._calculate_user_local_processing_energy(local_task_size)
            #print(f"user_local_energy: {user_local_energy[user_id]}")
            
            # 1.2 用户卸载能耗 = 传输能耗 + UAV处理能耗
            if offload_task_size > 0:
                # 传输能耗
                transmission_energy = self._calculate_user_transmission_energy(
                    offload_task_size, user_id, uav_id, user_states, uav_states
                )
                #print(f"transmission_energy: {transmission_energy}")
                # UAV处理能耗
                uav_cpu_per_user = self.uav_cpu_frequency / len(uav_user_groups[uav_id])
                uav_processing_energy = self._calculate_uav_processing_energy(offload_task_size, uav_cpu_per_user)
                #print(f"uav_processing_energy: {uav_processing_energy}")
                # 卸载总能耗
                user_offload_energy[user_id] = transmission_energy + uav_processing_energy
                #print(f"user_offload_energy: {user_offload_energy[user_id]}")
            else:
                user_offload_energy[user_id] = 0.0
            
            # 1.3 用户总能耗 = 本地 + 卸载
            user_total_energy[user_id] = user_local_energy[user_id] + user_offload_energy[user_id]
            #print(f"user_total_energy: {user_total_energy[user_id]}")
            
            # === 2. 最大能耗基准值计算 ===
            # 2.1 用户全本地处理能耗
            max_local_energy[user_id] = self._calculate_user_local_processing_energy(task_size)
            
            # 2.2 用户全卸载处理能耗（传输 + UAV处理）
            max_transmission_energy = self._calculate_user_transmission_energy(
                task_size, user_id, uav_id, user_states, uav_states
            )
            max_uav_processing_energy = self._calculate_uav_processing_energy(
                task_size, self.uav_cpu_frequency / len(uav_user_groups[uav_id])
            )
            max_offload_energy[user_id] = max_transmission_energy + max_uav_processing_energy
            
            # 2.3 用户最大处理能耗 = max(全本地, 全卸载)
            user_max_energy[user_id] = max(max_local_energy[user_id], max_offload_energy[user_id])
        
        return {
            'user_local_energy': user_local_energy,        # 用户X任务在本地处理的实际能耗
            'user_offload_energy': user_offload_energy,    # 用户X任务卸载处理的实际能耗（传输+UAV）
            'user_total_energy': user_total_energy,        # 用户X任务的总能耗（本地+卸载）
            'max_local_energy': max_local_energy,          # 用户X任务全本地处理的能耗
            'max_offload_energy': max_offload_energy,      # 用户X任务全卸载处理的能耗（传输+UAV）
            'user_max_energy': user_max_energy             # 用户X任务的最大处理能耗 = max(全本地, 全卸载)
        }
    
    def _calculate_user_local_processing_energy(self, task_size):
        """
        计算用户本地处理能耗
        
        Args:
            task_size: 本地处理任务大小 (MB)
            
        Returns:
            float: 本地处理能耗 (J - 焦耳)
        """
        if task_size <= 0:
            return 0.0
        
        # 计算处理时间
        user_cpu_frequency_mhz = self.user_cpu_frequency * 1000  # GHz -> MHz
        processing_time = task_size * self.cpu_cycles_per_mb / user_cpu_frequency_mhz  # 秒
        
        # 能耗 = 功率 × 时间
        energy = self.user_cpu_power * processing_time  # 焦耳
        
        return energy
    
    def _calculate_user_transmission_energy(self, task_size, user_id, uav_id, user_states, uav_states):
        """
        计算用户传输能耗
        
        Args:
            task_size: 传输任务大小 (MB)
            user_id: 用户ID
            uav_id: UAV ID
            user_states: 用户状态数组
            uav_states: UAV状态数组
            
        Returns:
            float: 传输能耗 (J - 焦耳)
        """
        if task_size <= 0:
            return 0.0
        
        # 计算传输时间
        transmission_time = self._calculate_transmission_time(task_size, user_id, uav_id, user_states, uav_states)
        
        # 能耗 = 传输功率 × 传输时间
        energy = self.user_transmission_power * transmission_time  # 焦耳
        
        return energy
    
    def _calculate_uav_processing_energy(self, task_size, uav_cpu_per_user):
        """
        计算UAV处理能耗
        
        Args:
            task_size: UAV处理任务大小 (MB)
            uav_cpu_per_user: 分配给该用户的UAV CPU频率 (GHz)
            
        Returns:
            float: UAV处理能耗 (J - 焦耳)
        """
        if task_size <= 0:
            return 0.0
        
        # 计算处理时间
        uav_cpu_frequency_mhz = uav_cpu_per_user * 1000  # GHz -> MHz
        processing_time = task_size * self.cpu_cycles_per_mb / uav_cpu_frequency_mhz  # 秒
        
        # 能耗 = 功率 × 时间
        energy = self.uav_cpu_power * processing_time  # 焦耳
        
        return energy
    
    def _calculate_transmission_time(self, task_size, user_id, uav_id, user_states, uav_states):
        """
        计算传输时间（复用DelayCalculator的逻辑）
        
        Args:
            task_size: 传输任务大小 (MB)
            user_id: 用户ID
            uav_id: UAV ID
            user_states: 用户状态数组
            uav_states: UAV状态数组
            
        Returns:
            float: 传输时间 (秒)
        """
        # 计算用户到UAV的三维距离
        user_pos = user_states[user_id, :2]   # [x, y] 用户在地面，z=0
        uav_pos = uav_states[uav_id, :2]      # [x, y] UAV的水平位置
        
        # 使用三维距离计算
        distance_3d = self._calculate_distance_3d(uav_pos, user_pos)
        
        # 计算路径损耗 (dB) - 使用三维距离
        if distance_3d < self.reference_distance:
            distance_3d = self.reference_distance
        path_loss = self.reference_path_loss + 10 * self.path_loss_exponent * np.log10(distance_3d / self.reference_distance)
        
        # 计算接收功率 (dBm)
        rx_power = self.transmission_power - path_loss
        
        # 计算信噪比 (dB)
        snr_db = rx_power - self.noise_power
        snr_linear = 10 ** (snr_db / 10)
        
        # 计算传输速率 (Shannon公式)
        transmission_rate = self.bandwidth * np.log2(1 + snr_linear)  # Mbps
        
        # 计算传输时间
        transmission_time = task_size / transmission_rate  # 秒
        
        return transmission_time
    
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
