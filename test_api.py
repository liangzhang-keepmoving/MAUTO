# test_api.py
import os
from openai import OpenAI

api_key="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"
print(f"✓ API Key已设置: {api_key[:10]}..." if api_key else "❌ API Key未设置")

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key
)

try:
    print("\n测试API调用...")
    response = client.chat.completions.create(
        model="google/gemini-3-flash-preview",
        messages=[
            {"role": "user", "content": "Hello, say 'API works!'"}
        ],
        max_tokens=100,
        extra_body={"reasoning": {"enabled": True}}
    )
    
    print("✓ API调用成功!")
    print(f"响应: {response.choices[0].message.content}")
    
except Exception as e:
    print(f"❌ API调用失败: {e}")
    import traceback
    traceback.print_exc()