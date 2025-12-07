"""
estimate_norm_params.py - 修正版
用于估计归一化参数，适配最新的 Offloading (N_Users vector) 和 Movement (Cartesian Vx, Vy) 格式。
"""
import numpy as np
import json
from env_simplified import SimplifiedMultiUAVEnvironment

def generate_ddpg_random_actions_hard_and_convert(n_uavs, n_users, max_move_distance=30.0):
    """
    生成符合 DDPG Actor 输出约束的随机**硬分配 (0/1)**动作张量，并转换为原始字典格式。
    
    Args:
        n_uavs: 无人机数量
        n_users: 用户数量
        max_move_distance: 最大飞行距离 (用于归一化)
    
    Returns:
        actions_dict: 原始字典格式的动作
    """
    
    # === 1. 硬分配 (Hard Allocation) ===
    # 为每个用户 (M) 随机选择一个无人机 (N)
    chosen_uavs = np.random.randint(0, n_uavs, size=n_users)
    
    # 构建硬分配矩阵 [N, M]
    allocation_hard = np.zeros((n_uavs, n_users), dtype=float)
    allocation_hard[chosen_uavs, np.arange(n_users)] = 1.0
    
    # === 2. 卸载比例 (Offloading Ratio) ===
    # 新逻辑：生成 [N_Users] 的向量，而不是 [N_UAVs, N_Users]
    # 每个用户决定自己的卸载比例，无论连哪个 UAV
    offloading_vector = np.random.uniform(0, 1, size=(n_users,))
    
    # === 3. 运动控制 (UAV Movement) ===
    # 新逻辑：生成 [N_UAVs, 2] 的 (vx, vy) 向量，范围 [-1, 1]
    # 模拟 Actor 输出 tanh 后的结果
    motion_vector = np.random.uniform(-1, 1, size=(n_uavs, 2))
    
    # === 4. 转换为环境交互格式 ===
    
    actions_dict = {}
    for uav_id in range(n_uavs):
        
        # A. 用户竞争/分配 ('user_competition_probs')
        # 使用硬分配结果
        user_competition_probs_proxy = allocation_hard[uav_id, :] 
        
        # B. 卸载比例 ('offloading_ratios')
        # 注意：所有 UAV 共享同一个全量用户卸载向量
        # calculator.py 会根据 user_id 自行索引
        offloading_ratios_proxy = offloading_vector
        
        # C. 运动参数 ('move_vector')
        # 将 [-1, 1] 的网络输出映射到 [-max_dist, +max_dist] 的物理位移
        vx = motion_vector[uav_id, 0] * max_move_distance
        vy = motion_vector[uav_id, 1] * max_move_distance
        move_vector_real = (vx, vy)
        
        actions_dict[f'uav_{uav_id}'] = {
            'user_competition_probs': user_competition_probs_proxy,
            'offloading_ratios': offloading_ratios_proxy,
            'move_vector': move_vector_real  # 新格式：使用 Cartesian 向量
        }
        
    return actions_dict

def estimate_normalization_parameters(config, num_episodes=20000, save_path='norm_params.json'):
    """
    估计归一化参数（修正版）
    """
    print("=" * 60)
    print("开始估计归一化参数（适配新Action格式）...")
    print(f"  Episodes: {num_episodes}")
    print("=" * 60)
    
    env = SimplifiedMultiUAVEnvironment(
        num_uavs=config['num_uavs'],
        num_users=config['num_users'],
        trajectory_file="user_trajectories_hot.json",
        max_flight_distance=config.get('max_distance', 20)
    )
    
    # ========== 收集数据 ==========
    min_all_delay = float("INF")
    max_all_delay = float("-INF")
    min_avg_delay = float("INF")
    max_avg_delay = float("-INF")
    min_max_delay = float("INF")
    max_max_delay= float("-INF")
    min_task_energy = float("INF")
    max_task_energy = float("-INF")
    min_transmission_energy = float("INF")
    max_transmission_energy = float("-INF")
    min_move_energy = float("INF")
    max_move_energy = float("-INF")
        
    
    for ep in range(num_episodes):
        state = env.reset()
        
        for step in range(40):
            actions = generate_ddpg_random_actions_hard_and_convert(
                config['num_uavs'], 
                config['num_users'],
                config.get('max_distance', 20)
            )
            
            next_state, reward, done, info = env.step(actions)
            
            raw_metrics = info.get('raw_metrics', {})
            
            # 收集用户级时延
            step_delays = []
            step_task_energy = 0
            step_transmission_energy = 0
            
            for user_id in range(config['num_users']):
                if user_id in raw_metrics:
                    # ==================== 修改开始 ====================
                    # 1. 强制转换为 Python float，切断底层依赖
                    delay = float(raw_metrics[user_id]['user_actual_delay'])
                    
                    if(delay < min_all_delay):
                        min_all_delay = delay
                    if(delay > max_all_delay):
                        max_all_delay = delay
                    step_delays.append(delay)
                    
                    # 2. 能耗部分也全部强转 float
                    e_local = float(raw_metrics[user_id]['user_local_computation_energy'])
                    e_trans = float(raw_metrics[user_id]['user_transmission_energy'])
                    e_uav   = float(raw_metrics[user_id]['user_uav_computation_energy'])
                    
                    step_task_energy += (e_local + e_trans + e_uav)
                    step_transmission_energy += e_trans
                    # ==================== 修改结束 ====================
                    # # 时延
                    # delay = float(raw_metrics[user_id]['user_actual_delay'])
                    # if(delay<min_all_delay):
                    #     min_all_delay = delay
                    # if(delay>max_all_delay):
                    #     max_all_delay = delay
                    # step_delays.append(delay)
                    
                    # # 能耗 (修正：包含本地计算、传输、UAV计算)
                    # step_task_energy += (
                    #     raw_metrics[user_id]['user_local_computation_energy'] +
                    #     raw_metrics[user_id]['user_transmission_energy'] +
                    #     raw_metrics[user_id]['user_uav_computation_energy']
                    # )
                    # step_transmission_energy += raw_metrics[user_id]['user_transmission_energy']
            
            # 聚合指标
            if len(step_delays) > 0:
                avg_delay = np.mean(step_delays)
                max_delay = np.max(step_delays)
                if(avg_delay<min_avg_delay):
                    min_avg_delay = avg_delay
                if(avg_delay>max_avg_delay):
                    max_avg_delay = avg_delay
                    
                if(max_delay<min_max_delay):
                    min_max_delay = max_delay
                if(max_delay>max_max_delay):
                    max_max_delay = max_delay
            
            if(step_task_energy<min_task_energy):
                min_task_energy = step_task_energy
            if(step_task_energy>max_task_energy):
                max_task_energy = step_task_energy
            if(step_transmission_energy<min_transmission_energy):
                min_transmission_energy = step_transmission_energy
            if(step_transmission_energy>max_transmission_energy):
                max_transmission_energy = step_transmission_energy
            
            
            
            # 移动能耗
            step_move_energy = sum(info.get('movement_energy_costs', {}).values())
            
            if(step_move_energy<min_move_energy):
                min_move_energy = step_move_energy
            if(step_move_energy>max_move_energy):
                max_move_energy = step_move_energy
            
            
            if done:
                break
        
        if (ep + 1) % 1000 == 0:
            print(f"  进度: {ep + 1}/{num_episodes}")
    
    # 统计参数（扁平结构）
    params = {
        # 全局时延
        'min_all_delay': min_all_delay,
        'max_all_delay': max_all_delay,
        
        # 平均时延
        'min_avg_delay': min_avg_delay,
        'max_avg_delay': max_avg_delay,
        
        # 最大时延
        'min_max_delay': min_max_delay,
        'max_max_delay': max_max_delay,
        
        # 任务能耗
        'min_task_energy': min_task_energy,
        'max_task_energy': max_task_energy,
        
        'min_transmission_energy': min_transmission_energy,
        'max_transmission_energy': max_transmission_energy,
        
        # 移动能耗
        'min_move_energy': min_move_energy,
        'max_move_energy': max_move_energy,
        
        # 负载方差 (保留以防万一)
        'min_load_variance': 0.0,
        'max_load_variance': float(config['num_users'] ** 2),
        
        # 移动收益 (保留以防万一)
        'max_movement_benefit': float(config['num_uavs'] * 1.0)
    }
    
    # 保存
    with open('norm_params.json', 'w') as f:
        json.dump(params, f, indent=2)
    
    return params

if __name__ == "__main__":
    config = {
        'num_uavs': 2,
        'num_users': 5,
        'max_distance':20
    }
    
    estimate_normalization_parameters(config, num_episodes=50000)
