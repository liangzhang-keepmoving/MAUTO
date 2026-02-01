import numpy as np
import json
import re


def create_prompt(state, env, step_num, history=None):
    """
    创建LLM决策提示词
    
    Args:
        state: 环境状态字典 {'uav_pos', 'user_pos', 'user_tasks'}
        env: 环境实例
        step_num: 当前步数
        history: 历史记录列表 (可选)
    
    Returns:
        str: 完整提示词
    """
    uav_pos = state['uav_pos'].cpu().numpy()
    user_pos = state['user_pos'].cpu().numpy()
    user_tasks = state['user_tasks'].cpu().numpy()
    
    # 构造UAV信息
    uav_info = ""
    for i in range(env.num_uavs):
        x = uav_pos[i][0] * env.area_length
        y = uav_pos[i][1] * env.area_width
        uav_info += f"- UAV_{i}: ({x:.1f}m, {y:.1f}m)\n"
    
    # 构造用户信息
    user_info = ""
    for i in range(env.num_users):
        x = user_pos[i][0] * env.area_length
        y = user_pos[i][1] * env.area_width
        task = user_tasks[i][0] * env.max_task_size
        
        # 计算到各UAV的3D距离
        distances = []
        for j in range(env.num_uavs):
            uav_x = uav_pos[j][0] * env.area_length
            uav_y = uav_pos[j][1] * env.area_width
            d_2d = np.sqrt((x - uav_x)**2 + (y - uav_y)**2)
            d_3d = np.sqrt(d_2d**2 + env.uav_height**2)
            distances.append(d_3d)
        
        dist_str = ", ".join([f"到UAV_{j}: {d:.1f}m" for j, d in enumerate(distances)])
        user_info += f"- 用户_{i}: 位置({x:.1f}m, {y:.1f}m), 任务{task:.2f}MB, {dist_str}\n"

    # 构造历史信息
    history_str = ""
    if history and len(history) > 0:
        history_str = "**历史决策回顾 (Context):**\n为了帮助你做出更好的决策，以下是历史系统状态和执行结果回顾：\n\n"
        # 使用全部历史数据
        for h in history:
            step_num = h['step']
            assignments = h['user_assignments']
            avg_offloading = h['avg_offloading']
            delay = h['delay']
            energy = h['energy']
            
            history_str += f"### 第{step_num}步:\n"
            
            # 状态描述
            state_desc_parts = []
            if 'uav_states' in h and h['uav_states'] is not None:
                uav_desc = ", ".join([f"UAV{i}位于({p[0]:.0f}, {p[1]:.0f})" for i, p in enumerate(h['uav_states'])])
                state_desc_parts.append(f"无人机位置: {uav_desc}")
            
            if 'user_states' in h and h['user_states'] is not None:
                user_desc = ", ".join([f"用户{i}(任务{s[2]:.1f}MB)" for i, s in enumerate(h['user_states'])])
                state_desc_parts.append(f"用户状态: {user_desc}")
            
            if state_desc_parts:
                history_str += f"- **场景状态**: {'; '.join(state_desc_parts)}。\n"
            
            # 决策描述
            # 将字典转为自然语言
            uav0_users = [k for k, v in assignments.items() if str(v) == '0' or v == 0]
            uav1_users = [k for k, v in assignments.items() if str(v) == '1' or v == 1]
            
            action_desc = "采取了以下调度策略："
            if uav0_users:
                action_desc += f"UAV0服务用户[{', '.join(map(str, uav0_users))}]"
            else:
                action_desc += "UAV0未服务任何用户"
            
            if uav1_users:
                action_desc += f"，UAV1服务用户[{', '.join(map(str, uav1_users))}]"
            else:
                action_desc += "，UAV1未服务任何用户"
                
            action_desc += f"。所有用户的平均任务卸载比例设定为 {avg_offloading:.0%}。"
            history_str += f"- **执行动作**: {action_desc}\n"
            
            # 结果描述
            history_str += f"- **执行结果**: 该决策导致了 {delay:.3f}秒 的平均时延和 {energy:.2f}焦耳 的总能耗。\n"
            history_str += "\n"
    
    # 返回完整提示词
    return f"""**当前系统状态:**
**UAV位置:**
{uav_info.strip()}
**用户信息:**
{user_info.strip()}
{history_str.strip()}
**当前时刻:** 第{step_num}步
**请做出以下决策:**
1. **用户分配**: 每个用户应该由哪个UAV服务？
2. **卸载比例**: 每个用户的任务中多少比例卸载到UAV？(0表示全本地，1表示全卸载)
3. **UAV移动**: 每个UAV应该移动的方向和距离
"""


def parse_llm_response(llm_output, env):
    """
    解析LLM输出并转换为环境动作
    
    Args:
        llm_output: LLM输出的JSON字符串或字典
        env: 环境实例
    
    Returns:
        dict: 环境动作格式 {'uav_0': {...}, 'uav_1': {...}}
    """
    # 解析JSON
    if isinstance(llm_output, str):
        try:
            # 移除 <think> 标签及其内容
            cleaned = re.sub(r'<think>.*?</think>', '', llm_output, flags=re.DOTALL)
            
            # 移除Markdown代码块标记
            cleaned = re.sub(r'```json\s*', '', cleaned)
            cleaned = re.sub(r'```\s*', '', cleaned)
            cleaned = cleaned.strip()
            
            # 尝试直接解析
            try:
                decision = json.loads(cleaned)
            except json.JSONDecodeError:
                # 寻找第一个 { 和最后一个 }
                start = cleaned.find('{')
                end = cleaned.rfind('}')
                if start != -1 and end != -1 and end > start:
                    json_str = cleaned[start:end+1]
                    decision = json.loads(json_str)
                else:
                    raise ValueError("未找到有效的JSON对象")
            
        except (json.JSONDecodeError, ValueError) as e:
            # 打印原始输出用于调试
            print(f"\n⚠️  JSON解析失败，原始LLM输出:")
            print("="*70)
            print(llm_output[:500])  # 只打印前500字符
            print("="*70)
            raise ValueError(f"无法解析LLM输出为JSON: {e}")
    else:
        decision = llm_output
    
    # 提取决策
    user_assignments = decision['user_assignments']
    offloading_ratios = decision['offloading_ratios']
    uav_movements = decision['uav_movements']
    
    # 转换为环境动作格式
    actions = {}
    for uav_id in range(env.num_uavs):
        # 用户分配（硬分配）
        user_probs = np.zeros(env.num_users)
        for user_id_str, assigned_uav in user_assignments.items():
            if int(assigned_uav) == uav_id:
                user_probs[int(user_id_str)] = 1.0
        
        # 卸载比例
        offload_ratios = np.array([
            float(offloading_ratios[str(i)]) 
            for i in range(env.num_users)
        ])
        
        # UAV移动（限制范围）
        movement = uav_movements[str(uav_id)]
        max_dist = float(getattr(env, 'max_flight_distance', 20.0))
        dx = np.clip(float(movement.get('dx', 0.0)), -max_dist, max_dist)
        dy = np.clip(float(movement.get('dy', 0.0)), -max_dist, max_dist)
        
        actions[f'uav_{uav_id}'] = {
            'user_competition_probs': user_probs,
            'offloading_ratios': offload_ratios,
            'move_vector': (dx, dy)
        }
    
    return actions


def get_default_action(env):
    """
    LLM失败时的默认动作
    
    Args:
        env: 环境实例
    
    Returns:
        dict: 默认动作
    """
    actions = {}
    for uav_id in range(env.num_uavs):
        user_probs = np.zeros(env.num_users)
        for user_id in range(env.num_users):
            if user_id % env.num_uavs == uav_id:
                user_probs[user_id] = 1.0
        
        actions[f'uav_{uav_id}'] = {
            'user_competition_probs': user_probs,
            'offloading_ratios': np.full(env.num_users, 0.6),
            'move_vector': (0.0, 0.0)
        }
    
    return actions
