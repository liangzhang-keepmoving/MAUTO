"""
estimate_norm_params.py - 修正版
"""
import numpy as np
import json
from env_simplified import SimplifiedMultiUAVEnvironment

def generate_ddpg_random_actions_hard_and_convert(n_uavs, n_users):
    """
    生成符合 DDPG Actor 输出约束的随机**硬分配 (0/1)**动作张量，并转换为原始字典格式。
    
    Returns:
        actions_dict: 原始字典格式的动作
    """
    
    # === 1. 硬分配 (列和为 1, 0/1 约束) ===
    # 为每个用户 (M) 随机选择一个无人机 (N)
    chosen_uavs = np.random.randint(0, n_uavs, size=n_users)
    
    # 构建硬分配矩阵 [N, M]
    allocation_hard = np.zeros((n_uavs, n_users), dtype=float) # 使用 float 类型以便后续乘法
    allocation_hard[chosen_uavs, np.arange(n_users)] = 1.0  # [N, M] - 列和为 1，元素为 0 或 1
    
    # === 2. 原始卸载比例 (连续值) ===
    # 随机原始比例 [N, M] in [0, 1] 
    offloading_ratio_raw = np.random.uniform(0, 1, size=(n_uavs, n_users))
    
    # 最终卸载比例：耦合硬分配 (非零比例只出现在被分配的 UAV-User 对上)
    offloading_final = allocation_hard * offloading_ratio_raw
    
    # === 3. 运动控制 (归一化) ===
    # distance [N, 1] in [0, 1] (模仿 Sigmoid)
    distance = np.random.uniform(0, 1, size=(n_uavs, 1))
    # angle [N, 1] in [-1, 1] (模仿 Tanh)
    angle = np.random.uniform(-1, 1, size=(n_uavs, 1))
    
    # === 4. 转换为原始字典格式 (使用硬分配结果进行填充) ===
    
    # 假设运动参数的原始范围是：
    MAX_MOVE_DISTANCE = 30  # 假设原始代码中的 20 是最大距离
    
    actions_dict = {}
    for uav_id in range(n_uavs):
        
        # 妥协 1: 用户竞争/分配 ('user_competition_probs')
        # 在硬分配模式下，每个 UAV i 仅对那些分配给它的用户 j 具有 '1' 的值。
        # 这里使用 allocation_hard 的行作为填充。
        user_competition_probs_proxy = allocation_hard[uav_id, :] 
        
        # 妥协 2: 卸载比例 ('offloading_ratios')
        # 使用最终耦合后的卸载比例 (offloading_final) 的行。
        offloading_ratios_proxy = offloading_final[uav_id, :]
        
        # 运动参数转换
        movement_distance_real = distance[uav_id, 0] * MAX_MOVE_DISTANCE 
        movement_direction_real = (angle[uav_id, 0] + 1) / 2 * (2 * np.pi) # [-1, 1] -> [0, 2*pi]
        
        actions_dict[f'uav_{uav_id}'] = {
            'user_competition_probs': user_competition_probs_proxy, # 包含 0 或 1
            'offloading_ratios': offloading_ratios_proxy,             # 包含 0 或 [0, 1] 的值
            'movement_direction': movement_direction_real,
            'movement_distance': movement_distance_real
        }
        
    return actions_dict

def estimate_normalization_parameters(config, num_episodes=20000, save_path='norm_params.json'):
    """
    估计归一化参数（修正版）
    """
    print("=" * 60)
    print("开始估计归一化参数（修正版）...")
    print(f"  Episodes: {num_episodes}")
    print("=" * 60)
    
    env = SimplifiedMultiUAVEnvironment(
        num_uavs=config['num_uavs'],
        num_users=config['num_users'],
        trajectory_file="user_trajectories.json",
        max_flight_distance=config.get('max_distance', 30)
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
    min_move_energy = float("INF")
    max_move_energy = float("-INF")
        
    
    for ep in range(num_episodes):
        state = env.reset()
        
        for step in range(40):
            actions = generate_ddpg_random_actions_hard_and_convert(
                config['num_uavs'], 
                config['num_users']
            )
            
            next_state, reward, done, info = env.step(actions)
            
            raw_metrics = info.get('raw_metrics', {})
            
            # 收集用户级时延
            step_delays = []
            step_task_energy = 0
            
            for user_id in range(config['num_users']):
                if user_id in raw_metrics:
                    # 时延
                    delay = raw_metrics[user_id]['user_actual_delay']
                    if(delay<min_all_delay):
                        min_all_delay = delay
                    if(delay>max_all_delay):
                        max_all_delay = delay
                    step_delays.append(delay)
                    
                    # 能耗
                    step_task_energy += (
                        raw_metrics[user_id]['user_local_computation_energy'] +
                        raw_metrics[user_id]['user_transmission_energy'] +
                        raw_metrics[user_id]['user_uav_computation_energy']
                    )
            
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
        
        # 移动能耗
        'min_move_energy': min_move_energy,
        'max_move_energy': max_move_energy,
        
        # 负载方差
        'min_load_variance': 0.0,
        'max_load_variance': float(config['num_users'] ** 2),
        
        # 移动收益
        'max_movement_benefit': float(config['num_uavs'] * 1.0)
    }
    
    # 保存
    with open('norm_params.json', 'w') as f:
        json.dump(params, f, indent=2)
    
    return params

if __name__ == "__main__":
    config = {
        'num_uavs': 2,
        'num_users': 6,
        'max_distance': 30
    }
    
    estimate_normalization_parameters(config, num_episodes=100000)