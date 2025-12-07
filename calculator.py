"""
时延计算模块 - 处理任务时延的详细计算
"""

import numpy as np

class Calculator:
    """时延计算器 - 计算任务的本地时延、传输时延和UAV计算时延"""
    
    def __init__(self, uav_height=50,uav_cpu_frequency = 6e9,user_cpu_frequency = 2e9, 
                 cpu_cycles_per_bit=1000, kappa_uav = 1e-28,kappa_user = 1e-28,bandwidth=2, transmission_power=20.0,
                 path_loss_exponent=2.8, reference_distance=1.0, reference_path_loss=1e-5, noise_power=1e-13):
    
        self.uav_height = uav_height
        self.uav_cpu_frequency = uav_cpu_frequency
        self.user_cpu_frequency = user_cpu_frequency
        self.cpu_cycles_per_bit = cpu_cycles_per_bit
        self.kappa_uav=kappa_uav
        self.kappa_user=kappa_user
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
                # 修改：从actions中获取卸载比例，注意现在的结构
                # 假设 offloading_ratios 现在是一个 vector (num_users,)，而不是 per-UAV
                # 但是为了兼容性，我们需要看actions是怎么传进来的
                # 在 train_ddpg.py 中，actions[f'uav_{uav_id}']['offloading_ratios'] 
                # 如果网络输出是 vector (num_users,)，那么这里需要适配
                
                offloading_ratio = 0.5  # 默认值
                if actions and uav_key in actions and 'offloading_ratios' in actions[uav_key]:
                    # 假设 actions[uav_key]['offloading_ratios'] 是一个 scalar 或者 array
                    # 如果是 array，对应 index user_id
                    ratios = actions[uav_key]['offloading_ratios']
                    if np.isscalar(ratios):
                         offloading_ratio = ratios
                    elif isinstance(ratios, (list, np.ndarray)):
                        if len(ratios) > user_id:
                            offloading_ratio = ratios[user_id]
                        else:
                            # Fallback if index out of bounds (shouldn't happen if logic is correct)
                            offloading_ratio = ratios[0] 
                    else:
                         offloading_ratio = float(ratios)
                         
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

                    total_offload_delays[user_id] = 0.0
                    
                
                # 用户总时延 = max(本地时延, 卸载时延)（并行处理）
                user_actual_delay = max(local_delay, total_offload_delay)
                user_actual_delays[user_id] = user_actual_delay
        
        # 5. 确保所有用户都有时延记录（未分配到UAV的用户）
        num_users = len(user_states)
        for user_id in range(num_users):
            if user_id not in user_local_computation_delays:
                # 未分配的用户，假设全部本地处理
                task_size = user_states[user_id, 2]
                local_delay, local_energy = self._calculate_local_computation(task_size, user_id)
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
        
        # Step 1: 转换为 bits（MEC 标准单位）
        task_size_bits = task_size * 1e6  # Mbits → bits
        # Step 2: 总计算量 F = C * D
        total_cycles = self.cpu_cycles_per_bit * task_size_bits  # [cycles]
        # Step 3: 本地计算时延 T = F / f
        local_delay = total_cycles / self.user_cpu_frequency   # [seconds]
        # Step 4: 本地计算能耗 E = κ * F * f²
        local_energy = (
        self.kappa_user 
        * total_cycles 
        * (self.user_cpu_frequency ** 2)
        )  # [Joules]

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
        # 1.计算用户到UAV的三维距离
        user_pos = user_states[user_id, :2]   # [x, y] 用户在地面，z=0
        uav_pos = uav_states[uav_id, :2]      # [x, y] UAV的水平位置
        distance_3d = self._calculate_distance_3d(uav_pos, user_pos)
        
        # 2. 路径损耗 L(d) = beta_0 * d^(-alpha)
        path_loss = self.reference_path_loss / (distance_3d ** self.path_loss_exponent)
        
        # 3. 接收功率 P_rx = P_tx * L(d)
        received_power = self.transmission_power * path_loss
        
        # 4. SNR = P_rx / N0 （线性域）
        snr = received_power / self.noise_power
        
        # 5. 带宽转换：MHz → Hz
        bandwidth_hz = uav_bandwidth_per_user * 1e6  # MHz to Hz
        
        # 6. 香农速率 R = B * log2(1 + SNR) [bps]
        transmission_rate_bps = bandwidth_hz * np.log2(1 + snr)
        
        # 7. 任务大小：Mbits → bits
        task_size_bits = task_size * 1e6
        
        # 8. 传输时延 = 任务大小 (bits) / 速率 (bps)
        transmission_delay = task_size_bits / transmission_rate_bps
        
        return transmission_delay
    
    def _calculate_uav_computation(self, task_size, uav_cpu_frequency):
        """计算UAV计算时延"""
        if task_size <= 0:
            return 0.0
        
        # Step 1: 转换为 bits（MEC 标准单位）
        task_size_bits = task_size * 1e6  # Mbits → bits
        # Step 2: 总计算量 F = C * D
        total_cycles = self.cpu_cycles_per_bit * task_size_bits  # [cycles]
        # Step 3: 本地计算时延 T = F / f
        uav_delay = total_cycles / uav_cpu_frequency   # [seconds]
        # Step 4: 本地计算能耗 E = κ * F * f²
        uav_energy = (
        self.kappa_uav 
        * total_cycles 
        * (uav_cpu_frequency ** 2)
        )  # [Joules]
        
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
    

