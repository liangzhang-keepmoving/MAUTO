# test_llm_agent_with_ddpg_knowledge.py
"""
测试学习了DDPG知识的LLM Agent
加载DDPG专家数据作为Few-shot示例，然后与环境交互
"""
import json
import os
import numpy as np
from datetime import datetime
from openai import OpenAI
from env_test import TestMultiUAVEnvironment
from llm_utils import parse_llm_output, get_default_action, state_to_prompt


class LLMAgentWithDDPGKnowledge:
    """学习了DDPG知识的LLM Agent"""
    
    def __init__(self, 
                 model="google/gemini-3-pro-preview",  # 与原版llm_agent.py保持一致
                 api_key=None,
                 ddpg_data_path='llm_training_data/llm_few_shot_training.txt',
                 use_few_shot=True):
        """
        初始化
        
        Args:
            model: LLM模型名称
            api_key: API密钥
            ddpg_data_path: DDPG few-shot数据路径
            use_few_shot: 是否使用few-shot学习
        """
        self.model = model
        self.use_few_shot = use_few_shot
        
        # 初始化OpenRouter客户端
        if api_key is None:
            api_key = os.getenv("OPENROUTER_API_KEY")
            if not api_key:
                raise ValueError("请设置OPENROUTER_API_KEY环境变量")
        
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key
        )
        
        # 加载DDPG few-shot示例
        self.system_prompt = ""
        if use_few_shot and os.path.exists(ddpg_data_path):
            with open(ddpg_data_path, 'r', encoding='utf-8') as f:
                self.system_prompt = f.read()
            print(f"✓ 已加载DDPG Few-shot知识: {ddpg_data_path}")
            print(f"  提示词长度: {len(self.system_prompt)} 字符")
        else:
            print(f"⚠️  未加载DDPG知识，使用纯提示词模式")
            # 使用基础提示词
            self.system_prompt = self._get_basic_prompt()
        
        # 统计
        self.total_calls = 0
        self.failed_calls = 0
        self.total_tokens = 0
        
        print(f"✓ LLM Agent初始化: {model}")
        print(f"  模式: {'Few-shot学习(DDPG知识)' if use_few_shot else '纯提示词'}")
    
    def _get_basic_prompt(self):
        """基础系统提示词（不包含DDPG示例）"""
        return """你是2架UAV边缘计算系统的智能调度器。

**系统配置**
- 区域: 400×400m
- UAV: 2架 (UAV_0, UAV_1)，飞行高度50m，最大速度20m/s
- 用户: 5个，任务15-20Mbits
- 优化目标: 最小化时延和能耗

**决策输出**（严格JSON格式，不要包含任何Markdown标记）:
{
  "user_assignments": {"0": 0, "1": 1, "2": 0, "3": 1, "4": 0},
  "offloading_ratios": {"0": 0.7, "1": 0.6, "2": 0.8, "3": 0.5, "4": 0.7},
  "uav_movements": {"0": {"dx": 5.0, "dy": 0.0}, "1": {"dx": -5.0, "dy": 5.0}}
}
"""
    
    def get_action(self, state, env, step_num):
        """
        获取LLM决策
        
        Args:
            state: 环境状态
            env: 环境实例
            step_num: 当前步数
        
        Returns:
            dict: 环境动作格式
        """
        try:
            # 创建用户提示词
            user_prompt = self._create_user_prompt(state, env, step_num)
            
            # 调用LLM API
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=20000  # 与原版llm_agent.py保持一致
            )
            
            # 统计token
            if hasattr(response, 'usage') and response.usage:
                tokens = response.usage.prompt_tokens + response.usage.completion_tokens
                self.total_tokens += tokens
            
            # 解析输出
            llm_output = response.choices[0].message.content
            actions = parse_llm_output(llm_output, env)
            
            self.total_calls += 1
            return actions
            
        except Exception as e:
            print(f"⚠️  步骤{step_num} LLM调用失败: {e}")
            self.failed_calls += 1
            self.total_calls += 1
            return get_default_action(env)
    
    def _create_user_prompt(self, state, env, step_num):
        """创建用户提示词（当前场景描述）"""
        # 使用llm_utils中的统一状态描述函数
        state_desc = state_to_prompt(state, env)
        
        prompt = f"**当前时刻: 第{step_num}步**\n\n"
        prompt += state_desc
        
        prompt += "\n**请做出调度决策:**\n"
        prompt += "严格按照JSON格式输出，不要包含任何Markdown标记。\n"
        
        return prompt
    
    def get_stats(self):
        """获取统计信息"""
        success_rate = (self.total_calls - self.failed_calls) / max(self.total_calls, 1)
        avg_tokens = self.total_tokens / max(self.total_calls, 1)
        
        return {
            'agent_type': 'llm_with_ddpg_knowledge' if self.use_few_shot else 'llm_vanilla',
            'total_calls': self.total_calls,
            'failed_calls': self.failed_calls,
            'success_rate': success_rate,
            'total_tokens': self.total_tokens,
            'avg_tokens_per_call': avg_tokens
        }


def convert_to_serializable(obj):
    """将numpy类型转换为Python原生类型"""
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


def test_llm_agent(agent, env, num_steps=30, save_result=True):
    """
    测试LLM Agent
    
    Args:
        agent: LLM Agent实例
        env: 环境实例
        num_steps: 测试步数
        save_result: 是否保存结果
    
    Returns:
        tuple: (总奖励, 总步数)
    """
    print("\n" + "="*70)
    print("开始测试LLM Agent")
    print("="*70)
    
    state = env.reset()
    total_reward = 0
    results = []
    
    for step in range(num_steps):
        # LLM决策
        actions = agent.get_action(state, env, step)
        
        # 环境步进
        next_state, reward, done, info = env.step(actions)
        
        # 记录
        step_info = {
            'step': step,
            'reward': float(reward),
            'user_assignments': convert_to_serializable(info['user_assignments']),
            'avg_delay': float(info['reward_components']['avg_delay']),
            'total_energy': float(info['reward_components']['total_energy'])
        }
        results.append(step_info)
        
        total_reward += reward
        
        # 打印进度
        if step % 5 == 0 or step < 3:
            print(f"步骤 {step:2d}: 奖励={reward:7.4f}, "
                  f"分配={info['user_assignments']}, "
                  f"时延={info['reward_components']['avg_delay']:.4f}s, "
                  f"能耗={info['reward_components']['total_energy']:.2f}J")
        
        state = next_state
        
        if done:
            print(f"\n✓ Episode在第{step+1}步结束")
            break
    
    # 统计结果
    print("\n" + "="*70)
    print("测试结果:")
    print("="*70)
    
    avg_reward = total_reward / len(results)
    total_delay = sum(r['avg_delay'] for r in results)
    total_energy = sum(r['total_energy'] for r in results)
    avg_delay = total_delay / len(results)
    avg_energy = total_energy / len(results)
    
    print(f"总步数: {len(results)}")
    print(f"总奖励: {total_reward:.4f}")
    print(f"平均奖励: {avg_reward:.4f}")
    print(f"累积平均时延: {total_delay:.4f}s")
    print(f"平均时延: {avg_delay:.4f}s")
    print(f"累积总能耗: {total_energy:.2f}J")
    print(f"平均能耗: {avg_energy:.2f}J")
    
    # LLM统计
    stats = agent.get_stats()
    print(f"\nLLM统计:")
    print(f"  模式: {stats['agent_type']}")
    print(f"  调用次数: {stats['total_calls']}")
    print(f"  失败次数: {stats['failed_calls']}")
    print(f"  成功率: {stats['success_rate']*100:.1f}%")
    print(f"  总Token: {stats['total_tokens']:,}")
    print(f"  平均Token/步: {stats['avg_tokens_per_call']:.0f}")
    
    # 估算成本（Gemini Flash: ~$0.15/M tokens）
    cost = stats['total_tokens'] * 0.15 / 1_000_000
    print(f"  估算成本: ${cost:.4f}")
    
    # 保存结果（与原始test_llm_agent.py完全相同的格式）
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
            'avg_reward': float(avg_reward),
            'total_delay': float(total_delay),
            'total_energy': float(total_energy),
            'steps': results,  # 与原版相同的字段名
            'llm_stats': stats
        }
        
        # 根据agent类型命名文件
        agent_type = 'with_ddpg_knowledge' if agent.use_few_shot else 'vanilla'
        filename = f"llm_test_{agent_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        
        print(f"\n✓ 结果已保存到: {filename}")
    
    print("="*70)
    
    return total_reward, len(results)


def main():
    """主函数"""
    print("="*70)
    print("LLM Agent测试 - 学习DDPG知识版本")
    print("="*70)
    
    # 检查DDPG数据是否存在
    ddpg_data_path = 'llm_training_data/llm_few_shot_training.txt'
    
    if not os.path.exists(ddpg_data_path):
        print(f"\n⚠️  未找到DDPG Few-shot数据: {ddpg_data_path}")
        print("请先运行:")
        print("  1. python ddpg_inference_and_collect.py")
        print("  2. python generate_llm_training_prompts.py")
        print("\n将使用纯提示词模式运行...")
        use_few_shot = False
    else:
        use_few_shot = True
    
    # 创建环境
    print("\n[1/3] 初始化环境...")
    trajectory_file = "user_trajectories_hot.json"
    if not os.path.exists(trajectory_file):
        print(f"⚠️  警告: 未找到轨迹文件 {trajectory_file}")
        print("    将使用随机生成的用户轨迹。")
        trajectory_file = None
        
    env = TestMultiUAVEnvironment(
        num_uavs=2,
        num_users=5,
        trajectory_file=trajectory_file
    )
    print("✓ 环境创建成功")
    
    # 创建LLM Agent
    print("\n[2/3] 初始化LLM Agent...")
    
    # 优先从环境变量获取API Key，如果未设置则使用默认（测试用）Key
    api_key = os.getenv("OPENROUTER_API_KEY") or "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"
    
    agent = LLMAgentWithDDPGKnowledge(
        model="google/gemini-3-pro-preview",  # 与原版test_llm_agent.py保持一致
        api_key=api_key,
        ddpg_data_path=ddpg_data_path,
        use_few_shot=use_few_shot
    )
    print("✓ LLM Agent创建成功")
    
    # 运行测试
    print(f"\n[3/3] 开始测试 (30步)...")
    print("-"*70)
    
    total_reward, total_steps = test_llm_agent(
        agent=agent,
        env=env,
        num_steps=15,
        save_result=True
    )
    
    print("\n✓ 测试完成！")


if __name__ == "__main__":
    main()
