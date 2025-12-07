# hybrid_llm_agent.py
"""
混合型LLM代理 - 结合DDPG专家的建议与LLM的推理能力
"""
import os
import torch
import numpy as np
from openai import OpenAI
from llm_prompts import SYSTEM_PROMPT
from llm_utils import create_decision_prompt, parse_llm_output, get_default_action
from DDPG import DDPGAgent


class HybridLLMAgent:
    """混合型LLM代理 (LLM + DDPG Expert)"""
    
    def __init__(self, model="google/gemini-flash-1.5-8b", api_key=None, 
                 ddpg_model_path=None, env_config=None):
        """
        初始化
        
        Args:
            model: OpenRouter模型
            api_key: API密钥
            ddpg_model_path: 预训练的DDPG模型路径
            env_config: 环境配置 (用于初始化DDPG)
        """
        self.model = model
        
        # 初始化 LLM 客户端
        if api_key is None:
            api_key = os.getenv("OPENROUTER_API_KEY")
            if not api_key:
                raise ValueError("请设置OPENROUTER_API_KEY环境变量")
        
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key
        )
        
        # 初始化 DDPG 专家
        self.ddpg_agent = None
        if ddpg_model_path:
            if not os.path.exists(ddpg_model_path):
                raise ValueError(f"DDPG模型文件不存在: {ddpg_model_path}")
                
            if not env_config:
                # 默认配置
                env_config = {
                    'num_uavs': 2,
                    'num_users': 5,
                    'max_distance': 30
                }
            
            print(f"正在加载 DDPG 专家模型: {ddpg_model_path}...")
            self.ddpg_agent = DDPGAgent(
                n_uavs=env_config.get('num_uavs', 2),
                n_users=env_config.get('num_users', 5),
                max_distance=env_config.get('max_distance', 30)
            )
            self.ddpg_agent.load(ddpg_model_path)
            
            # 设为评估模式
            self.ddpg_agent.actor.eval()
            self.ddpg_agent.critic.eval()
            print("✓ DDPG 专家加载完成")
        else:
            print("⚠️ 警告: 未提供 DDPG 模型路径，将回退到纯 LLM 模式")

        # 统计
        self.total_calls = 0
        self.failed_calls = 0
        self.total_tokens = 0
        
        print(f"✓ Hybrid LLM代理初始化: {model}")
    
    def _get_ddpg_suggestion(self, state, env):
        """
        获取 DDPG 专家的建议动作
        """
        if not self.ddpg_agent:
            return None
            
        # 使用 DDPG 选择动作 (不加噪声，硬分配)
        allocation_soft, offloading, motion = self.ddpg_agent.select_action(
            state, add_noise=False, hard=False
        )
        
        # 解析动作
        # 1. 用户分配 (选择概率最大的)
        user_assignments = {}
        allocation_np = allocation_soft.cpu().numpy() # [N_UAV, N_USER]
        for u in range(env.num_users):
            uav_idx = np.argmax(allocation_np[:, u])
            user_assignments[str(u)] = int(uav_idx)
            
        # 2. 卸载比例
        offloading_ratios = {}
        offloading_np = offloading.cpu().numpy() # [N_USER]
        for u in range(env.num_users):
            offloading_ratios[str(u)] = float(offloading_np[u])
            
        # 3. 移动向量
        uav_movements = {}
        motion_np = motion.cpu().numpy() # [N_UAV, 2]
        for i in range(env.num_uavs):
            # DDPG输出的是 [-1, 1] 的归一化值，需要映射到实际距离
            # 注意：env.uav_max_speed = 20
            vx, vy = motion_np[i]
            dx = vx * 20.0
            dy = vy * 20.0
            uav_movements[str(i)] = {"dx": float(dx), "dy": float(dy)}
            
        return {
            "user_assignments": user_assignments,
            "offloading_ratios": offloading_ratios,
            "uav_movements": uav_movements
        }

    def get_action(self, state, env, step_num):
        """获取决策"""
        try:
            # 1. 获取 DDPG 建议
            ddpg_suggestion = self._get_ddpg_suggestion(state, env)
            
            # 2. 创建提示词 (包含 DDPG 建议)
            # 注意：我们需要修改 create_decision_prompt 来支持 ddpg_suggestion 参数
            # 或者在这里手动拼接建议
            user_prompt = create_decision_prompt(state, env, step_num, ddpg_suggestion)
            
            # 3. 调用 LLM
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=20000
            )
            
            # 统计token
            if hasattr(response, 'usage') and response.usage:
                tokens = response.usage.prompt_tokens + response.usage.completion_tokens
                self.total_tokens += tokens
            
            # 4. 解析输出
            llm_output = response.choices[0].message.content
            actions = parse_llm_output(llm_output, env)
            
            self.total_calls += 1
            return actions
            
        except Exception as e:
            print(f"⚠️  步骤{step_num} Hybrid LLM失败: {e}")
            self.failed_calls += 1
            self.total_calls += 1
            return get_default_action(env)
