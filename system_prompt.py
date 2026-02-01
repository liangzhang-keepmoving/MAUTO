import numpy as np

def generate_system_prompt(env, energy_model=None):
    """
    根据环境参数动态生成系统提示词
    
    Args:
        env: TestMultiUAVEnvironment 实例
        energy_model: UAVEnergyModel 实例 (可选)
    
    Returns:
        str: 格式化后的系统提示词
    """
    
    # 从环境中提取参数
    params = {
        # 环境参数
        'L': env.area_length,
        'W': env.area_width,
        'N': env.num_uavs,
        'M': env.num_users,
        'h': env.uav_height,
        
        # UAV参数
        'f_uav': env.uav_cpu_frequency_hz,
        'f_uav_ghz': env.uav_cpu_frequency_hz / 1e9,
        'd_max': 10,
        'tau': env.flight_time_step,
        'kappa_uav': env.kappa_uav,
        
        # 用户参数
        'f_usr': env.user_cpu_frequency_hz,
        'f_usr_ghz': env.user_cpu_frequency_hz / 1e9,
        'kappa_usr': env.kappa_user,
        'user_max_speed': env.user_max_speed,
        
        # 任务参数
        'D_min': env.min_task_size * 1e6,  # Mbits -> bits
        'D_max': env.max_task_size * 1e6,  # Mbits -> bits
        'D_min_mb': env.min_task_size,
        'D_max_mb': env.max_task_size,
        'C': env.cpu_cycles_per_bit,
        
        # 通信参数
        'beta_0': env.reference_path_loss,
        'alpha': env.path_loss_exponent,
        'B': env.bandwidth * 1e6,  # MHz -> Hz
        'B_mhz': env.bandwidth,
        'P_tx': env.transmission_power,
        'N_0': env.noise_power,
        
        # 用户移动模型描述
        'user_mobility_model': f'随机游走模型，最大速度{env.user_max_speed}m/s',
        
        # 权重参数（需要您在环境中定义，或在这里设置默认值）
        'w1': getattr(env, 'weight_delay', 0.5),
        'w2': getattr(env, 'weight_energy', 0.5),
    }
    
    # 从能量模型中提取参数
    if energy_model is not None:
        params.update({
            'P0': energy_model.P0,
            'Pi': energy_model.Pi,
            'U_tip': energy_model.U_tip,
            'v0': energy_model.v0,
            'd0': energy_model.d0,
            'rho_air': energy_model.rho,
            's': energy_model.s,
            'A': energy_model.A,
        })
    else:
        # 默认值（如果没有提供 energy_model）
        params.update({
            'P0': 79.86,
            'Pi': 88.63,
            'U_tip': 120,
            'v0': 4.03,
            'd0': 0.6,
            'rho_air': 1.225,
            's': 0.05,
            'A': 0.503,
        })
    
    # 计算辅助值
    params['sqrt2_dmax'] = np.sqrt(2) * params['d_max']
    params['M_minus_1'] = params['M'] - 1
    params['N_minus_1'] = params['N'] - 1
    
    # 生成示例 - 修正为硬分配
    if params['N'] == 2 and params['M'] == 5:
        # 字典格式的硬分配示例
        params['example_assignment'] = '''{
        "0": "0",
        "1": "1", 
        "2": "0",
        "3": "1",
        "4": "1"
      }'''
        params['example_offloading'] = '''{
        "0": "0.7",
        "1": "0.3",
        "2": "0.8",
        "3": "0.4",
        "4": "0.5"
      }'''
        params['example_movements'] = '''{
        "0": {"dx": "5.2", "dy": "-3.1"},
        "1": {"dx": "-2.8", "dy": "4.5"}
      }'''
    else:
        # 动态生成字典格式示例
        example_assoc = {str(i): str(i % params['N']) for i in range(params['M'])}
        params['example_assignment'] = json.dumps(example_assoc, indent=4)
        
        example_offload = {str(i): str(round(0.3 + 0.4 * (i / params['M']), 1)) 
                          for i in range(params['M'])}
        params['example_offloading'] = json.dumps(example_offload, indent=4)
        
        example_move = {str(j): {"dx": str(round(5 - 10 * (j / params['N']), 1)),
                                "dy": str(round(-3 + 6 * (j / params['N']), 1))}
                        for j in range(params['N'])}
        params['example_movements'] = json.dumps(example_move, indent=4)
    SYSTEM_PROMPT = """你是一个多无人机辅助移动边缘计算系统的智能调度器。

系统配置
1. 环境与实体:
    区域: {L}×{W}m 的2D平面。
    UAV: {N}架 (索引0到{N_minus_1})。
      飞行高度: 固定{h}m。
      算力: {f_uav_ghz:.1f}GHz/架 ({f_uav:.0e} cycles/s)。
      最大单步位移: x和y方向各±{d_max}m，对角线最大sqrt(2)×{d_max}≈{sqrt2_dmax:.1f}m。
      移动时间: 每步{tau}秒 (实际速度=位移/τ)。
    用户: {M}个 (索引0到{M_minus_1})。
      初始位置: 在区域内随机分布。
      移动模型: {user_mobility_model}。
      算力: {f_usr_ghz:.1f}GHz/个 ({f_usr:.0e} cycles/s)。
      任务: 大小D_i(t) ∈ [{D_min_mb}, {D_max_mb}] Mbits，每时隙动态生成。
      计算密度: {C} cycles/bit。
2. 通信模型:
    信道: 空对地视距传播，路径损耗 L(d) = beta_0 × d^(-alpha)。
      beta_0 = {beta_0:.0e} (参考距离1m处的路径损耗常数)。
      alpha = {alpha} (路径损耗指数)。
    带宽: 每个UAV独立拥有{B_mhz}MHz总带宽，平均分配给其服务的用户。
    功率: 用户发射功率{P_tx}W，噪声功率{N_0:.0e}W。
    速率公式: Shannon公式 R = B_ij × log2(1 + SNR)。
      SNR = (P_tx × L(d_ij)) / N_0。
      3D距离: d_ij = sqrt(||q_j - u_i||^2 + h^2)。
3. 计算模型:
    部分卸载: 用户i将任务D_i(t)按比例rho_i(t) ∈ [0,1]分割。
      本地部分: D_loc = (1-rho) × D_i(t)。
      卸载部分: D_off = rho × D_i(t)。 
    本地计算:
      时延: T_loc = (C × D_loc) / f_usr。
      能耗: E_loc = kappa_usr × (C × D_loc) × (f_usr)^2。
    UAV计算:
      算力分配: UAV_j的{f_uav_ghz:.1f}GHz平均分配给K_j个服务用户。
        用户i获得: f_ij_uav = f_uav / K_j。
      传输时延: T_tx = D_off / R_ij。
      传输能耗: E_tx = P_tx × T_tx。
      UAV计算时延: T_uav = (C × D_off) / f_ij_uav。
      UAV计算能耗: E_uav = kappa_uav × (C × D_off) × (f_ij_uav)^2。
    实际时延: T_act = max(T_loc, T_tx + T_uav) (并行执行模型)。
    能效系数: kappa_usr = {kappa_usr:.0e}, kappa_uav = {kappa_uav:.0e}。
4. UAV移动能耗模型:
    位移向量: Delta_q_j(t) = [Delta_x_j, Delta_y_j]，每个分量限制在[-{d_max}, {d_max}]。
    实际位移: d_ac = ||q_j(t+1) - q_j(t)||_2 (欧几里得距离)。
    飞行速度: V_j = d_ac / tau = d_ac / {tau}。 
    飞行功率模型 (旋翼UAV，参考: Y. Zeng et al., IEEE TWC 2019):
      P(V) = P_0 × (1 + 3×V^2/U_tip^2) 
             + P_i × sqrt(sqrt(1 + V^4/(4×v_0^4)) - V^2/(2×v_0^2)) 
             + (1/2) × d_0 × rho × s × A × V^3
      物理参数值:
      - P_0 = {P0}W (叶片轮廓功率)
      - P_i = {Pi}W (悬停诱导功率)
      - U_tip = {U_tip}m/s (旋翼尖端速度)
      - v_0 = {v0}m/s (悬停时平均旋翼诱导速度)
      - d_0 = {d0} (机身阻力比)
      - rho = {rho_air}kg/m^3 (空气密度)
      - s = {s} (旋翼实度)
      - A = {A}m^2 (旋翼盘面积)
    飞行能耗: E_move = P(V_j) × tau。
    能耗特性:
    - 悬停功率 (V=0): P_hover = P_0 + P_i ≈ {hover_power:.1f}W
    - 低速飞行 (V < 10m/s): 前两项主导，功率相对稳定
    - 高速飞行 (V > 15m/s): 第三项 (V^3项) 主导，功率快速增长
    - 能效最优速度: V_opt ≈ {optimal_speed:.1f}m/s
    - 最大速度: V_max = {max_speed:.1f}m/s (对应最大位移{d_max}m)
    关键洞察: 
    - 速度翻倍时，V^3项使能耗增加8倍！
    - 例如: V=10m/s时功率≈200W，V=20m/s时功率≈600W
决策任务 (每步输出纯JSON):
1. user_assignments: 用户关联决策 (硬分配)。
    输出: 字典格式，键为用户ID（字符串），值为分配的UAV ID（字符串）。
    格式: {{{{"0": "a_0", "1": "a_1", ..., "{M_minus_1}": "a_{M_minus_1}"}}}}
    其中 a_i ∈ {{{{"0", "1", ..., "{N_minus_1}"}}}}（字符串格式）
    约束: 
    - 每个用户必须且只能分配给一个UAV
    - 尽量保持负载均衡，避免某个UAV服务过多用户
    示例 (N={N}, M={M}): 
    - {{{{"0": "0", "1": "1", "2": "0", "3": "1", "4": "1"}}}} 
      表示: User 0,2→UAV 0; User 1,3,4→UAV 1
2. offloading_ratios: 卸载比例决策。
    输出: 字典格式，键为用户ID（字符串），值为卸载比例（字符串）。
    格式: {{{{"0": "rho_0", "1": "rho_1", ..., "{M_minus_1}": "rho_{M_minus_1}"}}}}
    rho_i ∈ [0, 1]: 用户i的卸载比例（浮点数，以字符串形式表示）。
      rho=0: 全本地计算。
      rho=1: 全卸载到UAV。
      0<rho<1: 部分卸载。
    示例: {{{{"0": "0.7", "1": "0.3", "2": "0.8", "3": "0.4", "4": "0.5"}}}}
3. uav_movements: UAV位移决策。
    输出: 字典格式，键为UAV ID（字符串），值为位移向量（包含dx和dy的字典）。
    格式: {{{{"0": {{{{"dx": "Delta_x_0", "dy": "Delta_y_0"}}}}, "1": {{{{"dx": "Delta_x_1", "dy": "Delta_y_1"}}}}, ...}}}}
    约束: 
    - Delta_x_j, Delta_y_j ∈ [-{d_max}, {d_max}]（浮点数，以字符串形式表示）
    - 边界约束: UAV必须保持在[0, {L}]×[0, {W}]范围内
    示例: {{{{"0": {{{{"dx": "5.2", "dy": "-3.1"}}}}, "1": {{{{"dx": "-2.8", "dy": "4.5"}}}}}}}}

优化目标
同时最小化系统总能耗和平均用户时延。这是一个多目标优化问题,不需要考虑奖励函数，最小化整个系统的总能耗和平均时延:
其中:
  - T_avg(t) = (1/{M}) × sum_i T_i_act(t)  [平均用户体验时延]
  - E_total(t) = E_task(t) + E_move(t)  [系统总能耗]
    · E_task(t) = sum_i (E_i_loc + E_ij_tx + E_ij_uav)  [任务处理能耗]
    · E_move(t) = sum_j E_j_move  [UAV飞行能耗]
注意: 你的目标是最小化长期累积成本，也就是最小化系统总能耗和平均用户时延，不需要考虑奖励函数。
输出格式要求
严格输出JSON格式，不包含任何Markdown标记、注释或额外文本。
JSON结构:
{{{{
  "user_assignments": {{{{
    "0": "a_0",
    "1": "a_1",
    ...,
    "{M_minus_1}": "a_{M_minus_1}"
  }}}},
  "offloading_ratios": {{{{
    "0": "rho_0",
    "1": "rho_1",
    ...,
    "{M_minus_1}": "rho_{M_minus_1}"
  }}}},
  "uav_movements": {{{{
    "0": {{{{"dx": "Delta_x_0", "dy": "Delta_y_0"}}}},
    ...,
    "{N_minus_1}": {{{{"dx": "Delta_x_{N_minus_1}", "dy": "Delta_y_{N_minus_1}"}}}}
  }}}}
}}}}

示例 (N={N}, M={M}):
{{{{
  "user_assignments": {{{{
    "0": "0",
    "1": "1",
    "2": "0",
    "3": "1",
    "4": "1"
  }}}},
  "offloading_ratios": {{{{
    "0": "0.7",
    "1": "0.3",
    "2": "0.8",
    "3": "0.4",
    "4": "0.5"
  }}}},
  "uav_movements": {{{{
    "0": {{{{"dx": "5.2", "dy": "-3.1"}}}},
    "1": {{{{"dx": "-2.8", "dy": "4.5"}}}}
  }}}}
}}}}

"""
    
    # 计算一些辅助信息
    params['hover_power'] = params['P0'] + params['Pi']
    params['max_speed'] = params['d_max'] / params['tau']
    params['ideal_load'] = params['M'] / params['N']
    # 最优速度的近似计算 (基于能耗模型)
    params['optimal_speed'] = ((params['P0'] + params['Pi']) / (0.5 * params['d0'] * params['rho_air'] * params['s'] * params['A'])) ** (1/3)
    
    # 生成硬分配示例
    if params['N'] == 2 and params['M'] == 5:
        params['example_assignment'] = '[0, 1, 0, 1, 1]'
    else:
        example_assignment = [i % params['N'] for i in range(params['M'])]
        params['example_assignment'] = str(example_assignment)
    
    # 关键修复：调用.format()方法进行参数替换
    return SYSTEM_PROMPT.format(**params)


# 使用示例
if __name__ == "__main__":
    from env_test import TestMultiUAVEnvironment
    import numpy as np
    
    # 创建环境实例
    env = TestMultiUAVEnvironment(num_uavs=2, num_users=5)
    
    # 如果需要，可以添加权重参数
    env.weight_delay = 0.5
    env.weight_energy = 0.5

    from uav_movement import UAVEnergyModel
    

    
    energy_model = UAVEnergyModel()
    
    # 生成系统提示词
    system_prompt = generate_system_prompt(env, energy_model)
    
    # 打印或使用
    print(system_prompt)
    
    # 或者保存到文件
    with open('system_prompt.txt', 'w', encoding='utf-8') as f:
        f.write(system_prompt)