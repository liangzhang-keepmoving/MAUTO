# llm_agent.py
"""LLM决策代理"""
import os
from openai import OpenAI
from llm_prompts import SYSTEM_PROMPT
from llm_utils import create_decision_prompt, parse_llm_output, get_default_action


class LLMAgent:
    """LLM决策代理"""
    
    def __init__(self, model="google/gemini-flash-1.5-8b", api_key=None):
        """
        初始化
        
        Args:
            model: OpenRouter模型
            api_key: API密钥（或从环境变量读取）
        """
        self.model = model
        
        # 初始化客户端
        if api_key is None:
            api_key = os.getenv("OPENROUTER_API_KEY")
            if not api_key:
                raise ValueError("请设置OPENROUTER_API_KEY环境变量")
        
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key
        )
        
        # 统计
        self.total_calls = 0
        self.failed_calls = 0
        self.total_tokens = 0
        
        print(f"✓ LLM代理初始化: {model}")
    
    def get_action(self, state, env, step_num):
        """获取决策"""
        try:
            # 创建提示词
            user_prompt = create_decision_prompt(state, env, step_num)
            
            # 调用API
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
            
            # 解析输出
            llm_output = response.choices[0].message.content
            actions = parse_llm_output(llm_output, env)
            
            self.total_calls += 1
            return actions
            
        except Exception as e:
            print(f"⚠️  步骤{step_num} LLM失败: {e}")
            self.failed_calls += 1
            self.total_calls += 1
            return get_default_action(env)
    
    def get_stats(self):
        """获取统计"""
        success_rate = (self.total_calls - self.failed_calls) / max(self.total_calls, 1)
        avg_tokens = self.total_tokens / max(self.total_calls, 1)
        
        return {
            'total_calls': self.total_calls,
            'failed_calls': self.failed_calls,
            'success_rate': success_rate,
            'total_tokens': self.total_tokens,
            'avg_tokens_per_call': avg_tokens
        }