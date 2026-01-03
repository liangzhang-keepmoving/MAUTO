import os
from openai import OpenAI
from llm_prompts import SYSTEM_PROMPT
from llm_utils import create_prompt, parse_llm_response, get_default_action


class LLMAgent:
    """LLM决策代理"""
    
    def __init__(self, model="deepseek-reasoner", api_key=None):
        """初始化LLM代理"""
        self.model = model
        
        # 初始化API客户端
        if api_key is None:
            api_key = os.getenv("DEEPSEEK_API_KEY")
            if not api_key:
                raise ValueError("请设置DEEPSEEK_API_KEY环境变量")
        
        self.client = OpenAI(
            base_url="https://api.deepseek.com/v1",
            api_key=api_key
        )
        
        # 统计信息
        self.total_calls = 0
        self.failed_calls = 0
        self.total_tokens = 0   
        
        # 历史记录
        self.history = []
        
        print(f"✓ LLM代理初始化: {model}")
    
    def record_result(self, step, user_assignments, offloading_ratios, delay, energy, uav_states=None, user_states=None):
        """记录一步的决策和结果"""
        avg_offloading = sum(offloading_ratios) / len(offloading_ratios) if len(offloading_ratios) > 0 else 0
        
        record =    {
            'step': step,
            'user_assignments': user_assignments,
            'avg_offloading': avg_offloading,
            'delay': delay,
            'energy': energy
        }
        
        if uav_states is not None:
            record['uav_states'] = uav_states
        if user_states is not None:
            record['user_states'] = user_states
            
        self.history.append(record)
    
    def get_action(self, state, env, step_num):
        """
        获取决策动作
        
        Args:
            state: 环境状态
            env: 环境实例
            step_num: 当前步数
        
        Returns:
            dict: 环境动作
        """
        try:
            # 创建提示词
            prompt = create_prompt(state, env, step_num, history=self.history)
            
            # 调用LLM API
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=20000
            )
            
            # 统计token使用
            if hasattr(response, 'usage') and response.usage:
                self.total_tokens += (response.usage.prompt_tokens + 
                                     response.usage.completion_tokens)
            
            # 解析LLM输出
            llm_output = response.choices[0].message.content
            
            # 保存LLM输出到文件
            os.makedirs("llm_logs", exist_ok=True)
            with open(f"llm_logs/step_{step_num}.txt", "w", encoding="utf-8") as f:
                f.write(llm_output)
            
            actions = parse_llm_response(llm_output, env)
            
            self.total_calls += 1
            return actions
            
        except Exception as e:
            print(f"⚠️  步骤{step_num} LLM调用失败: {e}")
            self.failed_calls += 1
            self.total_calls += 1
            return get_default_action(env)
    
    def get_stats(self):
        """获取统计信息"""
        success_rate = (self.total_calls - self.failed_calls) / max(self.total_calls, 1)
        avg_tokens = self.total_tokens / max(self.total_calls, 1)
        
        return {
            'total_calls': self.total_calls,
            'failed_calls': self.failed_calls,
            'success_rate': success_rate,
            'total_tokens': self.total_tokens,
            'avg_tokens_per_call': avg_tokens
        }