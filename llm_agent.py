import os
import json
import traceback
from openai import OpenAI
from system_prompt import generate_system_prompt
from llm_utils import create_prompt, parse_llm_response, get_default_action
from uav_movement import UAVEnergyModel


class LLMAgent:
    """LLM决策代理 - 支持Google Gemini推理模式"""
    
    def __init__(self, model="google/gemini-3-flash-preview", api_key=None, enable_reasoning=True):
        """初始化LLM代理"""
        self.model = model
        self.enable_reasoning = enable_reasoning  # ✅ 确保设置
        
        # 初始化API客户端
        if api_key is None:
            api_key = os.getenv("OPENROUTER_API_KEY")
            if not api_key:
                raise ValueError("请设置OPENROUTER_API_KEY环境变量")
        
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key
        )
        
        # 统计信息
        self.total_calls = 0
        self.failed_calls = 0
        self.total_tokens = 0   
        
        # 历史记录
        self.history = []
        
        print(f"✓ LLM代理初始化: {model}")
        print(f"  推理模式: {'启用' if self.enable_reasoning else '禁用'}")
    
    def record_result(self, step, user_assignments, offloading_ratios, delay, energy, uav_states=None, user_states=None):
        """记录一步的决策和结果"""
        avg_offloading = sum(offloading_ratios) / len(offloading_ratios) if len(offloading_ratios) > 0 else 0
        
        record = {
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
        """获取决策动作"""
        try:
            print(f"\n{'='*60}")
            print(f"🚀 步骤 {step_num}: 开始LLM决策")
            print(f"{'='*60}")
            
            # ✅ 步骤1: 创建提示词
            print("📝 [1/4] 生成提示词...")
            try:
                prompt = create_prompt(state, env, step_num, history=self.history)
                energy_model = UAVEnergyModel()
                system_prompt = generate_system_prompt(env, energy_model)
                
                print(f"  ✓ System Prompt: {len(system_prompt)} 字符")
                print(f"  ✓ User Prompt: {len(prompt)} 字符")
                
            except Exception as e:
                print(f"  ❌ 提示词生成失败: {e}")
                raise
            
            # ✅ 步骤2: 构建API参数
            print("🔧 [2/4] 构建API参数...")
            api_params = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": 20000
            }
            
            if self.enable_reasoning:
                api_params["extra_body"] = {"reasoning": {"enabled": True}}
                print(f"  ✓ 推理模式已启用")
            
            # ✅ 步骤3: 调用API
            print(f"🔄 [3/4] 调用API (模型: {self.model})...")
            try:
                response = self.client.chat.completions.create(**api_params)
                print(f"  ✓ API调用成功")
                
            except Exception as api_error:
                print(f"  ❌ API调用失败: {api_error}")
                raise
            
            # ✅ 步骤4: 解析响应
            print("📥 [4/4] 解析响应...")
            
            # 验证响应结构
            if not response:
                raise ValueError("API返回None")
            
            if not hasattr(response, 'choices'):
                raise ValueError(f"响应缺少choices属性, 响应类型: {type(response)}")
            
            if len(response.choices) == 0:
                raise ValueError("响应的choices列表为空")
            
            # 提取消息
            message = response.choices[0].message
            
            if not message:
                raise ValueError("消息对象为空")
            
            # 提取内容
            llm_output = message.content or ""
            reasoning_details = getattr(message, 'reasoning_details', None)
            
            print(f"  ✓ 输出长度: {len(llm_output)} 字符")
            print(f"  ✓ 包含推理过程: {reasoning_details is not None}")
            
            # 保存日志
            os.makedirs("llm_logs", exist_ok=True)
            log_path = f"llm_logs/step_{step_num}.txt"
            
            with open(log_path, "w", encoding="utf-8") as f:
                if reasoning_details:
                    f.write("=" * 60 + "\n")
                    f.write("推理过程 (Reasoning Details)\n")
                    f.write("=" * 60 + "\n")
                    try:
                        f.write(json.dumps(reasoning_details, indent=2, ensure_ascii=False))
                    except:
                        f.write(str(reasoning_details))
                    f.write("\n\n")
                
                f.write("=" * 60 + "\n")
                f.write("最终答案 (Final Answer)\n")
                f.write("=" * 60 + "\n")
                f.write(llm_output)
            
            print(f"  ✓ 日志已保存: {log_path}")
            
            # 解析动作
            actions = parse_llm_response(llm_output, env)
            
            # 更新统计
            if hasattr(response, 'usage') and response.usage:
                tokens = response.usage.total_tokens
                self.total_tokens += tokens
                print(f"  ✓ Token使用: {tokens}")
            
            self.total_calls += 1
            print(f"✅ 步骤 {step_num} 完成\n")
            
            return actions
            
        except Exception as e:
            print(f"\n{'='*60}")
            print(f"❌ 步骤 {step_num} 失败")
            print(f"{'='*60}")
            print(f"错误类型: {type(e).__name__}")
            print(f"错误信息: {str(e)}")
            print(f"\n完整堆栈追踪:")
            traceback.print_exc()
            print(f"{'='*60}\n")
            
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