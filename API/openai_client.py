import os
import sys
from dotenv import load_dotenv
from openai import OpenAI

# 获取.env文件,用绝对路径
load_dotenv(r"D:\Mypython\.env")

try:
    client = OpenAI(
        # 若没有配置环境变量，请用阿里云百炼API Key将下行替换为: api_key="sk-xxx",
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        # 以下为华北2（北京）地域的URL，各地域的URL不同。调用时请将WorkspaceId替换为真实的业务空间ID。
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )

    completion = client.chat.completions.create(
        model="qwen-plus",  # 模型列表: https://help.aliyun.com/model-studio/getting-started/models
        messages=[
            {'role': 'system', 'content': 'You are a helpful assistant.'},
            {'role': 'user', 'content': '你是谁？'}
        ]
    )
    print(completion.choices[0].message.content)
except Exception as e:
    print(f"错误信息：{e}")
    print("请参考文档：https://help.aliyun.com/model-studio/developer-reference/error-code")