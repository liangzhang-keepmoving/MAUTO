# llm_utils.py
"""
LLM工具函数 - 处理状态和动作的转换
"""
import numpy as np
import json
import re


def state_to_prompt(state, env):
    """
    将环境状态转换为LLM可读的提示词
    
    Args:
        state: 环境状态字典 {'uav_pos', 'user_pos', 'user_tasks'}
        env: 环境实例 (用于获取参数)
    
    Returns:
        str: 格式化的提示词
    """
    uav_pos = state['uav_pos'].cpu().numpy()  # [N, 2]
    user_pos = state['user_pos'].cpu().numpy()  # [M, 2]
    user_tasks = state['user_tasks'].cpu().numpy()  # [M, 1]
    
    prompt = "**当前系统状态:**\n\n"
    
    # UAV信息
    prompt += "**UAV位置:**\n"
    for i in range(env.num_uavs):
        x = uav_pos[i][0] * env.area_length
        y = uav_pos[i][1] * env.area_width
        prompt += f"- UAV_{i}: ({x:.1f}m, {y:.1f}m)\n"
    
    prompt += "\n**用户信息:**\n"
    
    # 用户信息（包含到每个UAV的距离）
    for i in range(env.num_users):
        x = user_pos[i][0] * env.area_length
        y = user_pos[i][1] * env.area_width
        task = user_tasks[i][0] * env.max_task_size
        
        # 计算到各UAV的3D距离（考虑UAV高度）
        distances = []
        for j in range(env.num_uavs):
            uav_x = uav_pos[j][0] * env.area_length
            uav_y = uav_pos[j][1] * env.area_width
            # 水平距离
            d_2d = np.sqrt((x - uav_x)**2 + (y - uav_y)**2)
            # 3D距离（考虑UAV高度50m）
            d_3d = np.sqrt(d_2d**2 + env.uav_height**2)
            distances.append(d_3d)
        
        dist_str = ", ".join([f"到UAV_{j}: {d:.1f}m" for j, d in enumerate(distances)])
        prompt += f"- 用户_{i}: 位置({x:.1f}m, {y:.1f}m), 任务{task:.2f}MB, {dist_str}\n"
    
    return prompt


def create_decision_prompt(state, env, step_num, ddpg_suggestion=None):
    """
    创建完整的决策提示词
    
    Args:
        state: 环境状态
        env: 环境实例
        step_num: 当前步数
        ddpg_suggestion: (可选) DDPG专家给出的建议动作字典
    
    Returns:
        str: 完整提示词
    """
    state_desc = state_to_prompt(state, env)
    
    suggestion_text = ""
    if ddpg_suggestion:
        suggestion_text = f"""
**DDPG专家建议:**
根据当前状态，专家模型建议采取以下行动（仅供参考）：
- 用户分配: {json.dumps(ddpg_suggestion['user_assignments'])}
- 卸载比例: {json.dumps(ddpg_suggestion['offloading_ratios'])}
- UAV移动: {json.dumps(ddpg_suggestion['uav_movements'])}
"""

    prompt = f"""
{state_desc}

**当前时刻:** 第{step_num}步
{suggestion_text}
**请做出以下决策:**

1. **用户分配**: 每个用户应该由哪个UAV服务？
2. **卸载比例**: 每个用户的任务中多少比例卸载到UAV？(0表示全本地，1表示全卸载)
3. **UAV移动**: 每个UAV应该移动的方向和距离

**输出要求:**
请严格按照以下JSON格式输出，不要包含任何其他文字：

{{
  "user_assignments": {{
    "0": 0,
    "1": 1,
    "2": 0,
    "3": 1,
    "4": 0
  }},
  "offloading_ratios": {{
    "0": 0.7,
    "1": 0.6,
    "2": 0.8,
    "3": 0.5,
    "4": 0.7
  }},
  "uav_movements": {{
    "0": {{"dx": 5.0, "dy": 0.0}},
    "1": {{"dx": -5.0, "dy": 5.0}}
  }}
}}

**注意:**
- user_assignments: 值必须是0或1（对应UAV_0或UAV_1）
- offloading_ratios: 值范围[0, 1]
- uav_movements的dx, dy: 值范围[-20, 20]米
"""
    
    return prompt


def llm_decision_to_env_actions(llm_output, env):
    """
    将LLM的JSON决策转换为环境可执行的动作格式
    
    Args:
        llm_output: LLM输出的JSON字符串或字典
        env: 环境实例
    
    Returns:
        dict: 环境动作格式 {'uav_0': {...}, 'uav_1': {...}}
    """
    # 如果是字符串，先解析为字典
    if isinstance(llm_output, str):
        try:
            # 清理输出（移除markdown标记）
            cleaned = llm_output.replace('```json', '').replace('```', '').strip()
            decision = json.loads(cleaned)
        except json.JSONDecodeError:
            # 如果解析失败，尝试提取JSON部分
            json_match = re.search(r'\{.*\}', llm_output, re.DOTALL)
            if json_match:
                decision = json.loads(json_match.group())
            else:
                raise ValueError("无法解析LLM输出为JSON")
    else:
        decision = llm_output
    
    # 提取决策
    user_assignments = decision['user_assignments']
    offloading_ratios = decision['offloading_ratios']
    uav_movements = decision['uav_movements']
    
    # 构建环境动作格式
    actions = {}
    
    for uav_id in range(env.num_uavs):
        # 初始化该UAV的动作
        uav_key = f'uav_{uav_id}'
        
        # 1. 用户竞争概率（硬分配）
        user_competition_probs = np.zeros(env.num_users)
        for user_id_str, assigned_uav in user_assignments.items():
            user_id = int(user_id_str)
            if assigned_uav == uav_id:
                user_competition_probs[user_id] = 1.0
        
        # 2. 卸载比例
        offloading_ratios_array = np.array([
            float(offloading_ratios[str(i)]) 
            for i in range(env.num_users)
        ])
        
        # 3. 移动参数
        movement = uav_movements[str(uav_id)]
        
        # 直接使用 dx, dy
        dx = float(movement.get('dx', 0.0))
        dy = float(movement.get('dy', 0.0))
        
        # 强制截断到 [-20, 20] 以防 LLM 输出越界
        dx = np.clip(dx, -20.0, 20.0)
        dy = np.clip(dy, -20.0, 20.0)
        
        actions[uav_key] = {
            'user_competition_probs': user_competition_probs,
            'offloading_ratios': offloading_ratios_array,
            'move_vector': (dx, dy)  
        }
    
    return actions


# ★★★ 添加函数别名，兼容两种导入方式 ★★★
parse_llm_output = llm_decision_to_env_actions


def get_default_action(env):
    """
    当LLM失败时使用的默认动作
    
    Args:
        env: 环境实例
    
    Returns:
        dict: 默认动作
    """
    actions = {}
    
    for uav_id in range(env.num_uavs):
        uav_key = f'uav_{uav_id}'
        
        # 简单策略：轮流分配用户，中等卸载比例，不移动
        user_competition_probs = np.zeros(env.num_users)
        for user_id in range(env.num_users):
            if user_id % env.num_uavs == uav_id:
                user_competition_probs[user_id] = 1.0
        
        actions[uav_key] = {
            'user_competition_probs': user_competition_probs,
            'offloading_ratios': np.full(env.num_users, 0.6),
            'move_vector': (0.0, 0.0)
        }
    
    return actions