# test_llm.py
"""测试LLM决策"""
import json
import numpy as np
from datetime import datetime
from env_test import TestMultiUAVEnvironment
from llm_agent import LLMAgent


def convert_to_serializable(obj):
    """
    将numpy类型转换为Python原生类型
    
    Args:
        obj: 任意对象
    
    Returns:
        可JSON序列化的对象
    """
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    else:
        return obj


def test_llm(num_steps=30, save_result=True):
    """
    测试LLM决策
    
    Args:
        num_steps: 测试步数
        save_result: 是否保存结果
    """
    print("="*70)
    print("LLM无人机边缘计算决策测试")
    print("="*70)
    
    # 创建环境
    print("\n[1/3] 初始化环境...")
    env = TestMultiUAVEnvironment(
        num_uavs=2,
        num_users=5,
        trajectory_file="user_trajectories.json"
    )
    print("✓ 环境创建成功")
    
    # 创建LLM代理
    print("\n[2/3] 初始化LLM代理...")
    agent = LLMAgent(model="deepseek-reasoner",api_key="sk-2d58dde01aa94f64a1b886765fd10305")
    print("✓ LLM代理创建成功")
    
    # 运行测试
    print(f"\n[3/3] 开始运行 {num_steps} 步...")
    print("-"*70)
    
    state = env.reset()
    total_reward = 0
    results = []
    
    for step in range(num_steps):
        # LLM决策
        actions = agent.get_action(state, env, step)
        
        # 环境步进
        next_state, reward, done, info = env.step(actions)
        
        # 记录决策结果到Agent历史
        offloading_ratios = actions['uav_0']['offloading_ratios']
        agent.record_result(
            step=step,
            user_assignments=convert_to_serializable(info['user_assignments']),
            offloading_ratios=offloading_ratios,
            delay=float(info['reward_components']['avg_delay']),
            energy=float(info['reward_components']['total_energy']),
            uav_states=convert_to_serializable(info['uav_states']),
            user_states=convert_to_serializable(info['user_states'])
        )
        
        # 记录（转换为可序列化类型）
        step_info = {
            'step': step,
            'reward': float(reward),
            'user_assignments': convert_to_serializable(info['user_assignments']),
            'avg_delay': float(info['reward_components']['avg_delay']),
            'total_energy': float(info['reward_components']['total_energy'])
        }
        results.append(step_info)
        
        # 打印进度
        total_reward += reward
        print(f"步骤 {step:2d}: 奖励={reward:7.4f}, 分配={info['user_assignments']}, "
              f"时延={info['reward_components']['avg_delay']:.3f}s")
        
        state = next_state
        
        if done:
            print(f"\n✓ Episode在第{step+1}步结束")
            break
    
    # 统计结果
    print("\n" + "="*70)
    print("测试结果:")
    print("="*70)
    print(f"总步数: {len(results)}")
    print(f"总奖励: {total_reward:.4f}")
    print(f"平均奖励: {total_reward/len(results):.4f}")
    print(results)
    
    # 性能指标
    total_delay = sum(r['avg_delay'] for r in results)
    total_energy = sum(r['total_energy'] for r in results)
    
    print(f"\n性能指标:")
    print(f"  累积平均时延: {total_delay:.4f}s")
    print(f"  累积总能耗: {total_energy:.2f}J")
    
    # LLM统计
    stats = agent.get_stats()
    print(f"\nLLM统计:")
    print(f"  调用次数: {stats['total_calls']}")
    print(f"  失败次数: {stats['failed_calls']}")
    print(f"  成功率: {stats['success_rate']*100:.1f}%")
    print(f"  总Token: {stats['total_tokens']:,}")
    print(f"  平均Token/步: {stats['avg_tokens_per_call']:.0f}")
    
    # 估算成本（Gemini Flash: ~$0.15/M tokens）
    cost = stats['total_tokens'] * 0.15 / 1_000_000
    print(f"  估算成本: ${cost:.4f}")
    
    # 保存结果
    if save_result:
        output = {
            'timestamp': datetime.now().isoformat(),
            'model': agent.model,
            'env_config': {
                'num_uavs': env.num_uavs,
                'num_users': env.num_users,
                'area_size': f"{env.area_length}x{env.area_width}",
            },
            'total_steps': len(results),
            'total_reward': float(total_reward),
            'avg_reward': float(total_reward / len(results)),
            'total_delay': float(total_delay),
            'total_energy': float(total_energy),
            'steps': results,
            'llm_stats': stats
        }
        filename = f"llm_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        print(f"\n✓ 结果已保存到: {filename}")
    
    print("="*70)
    
    return total_reward, len(results)


if __name__ == "__main__":
    # 运行测试
    test_llm(num_steps=30, save_result=True)
