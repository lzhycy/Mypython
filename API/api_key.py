from dotenv import load_dotenv
import os
import sys

# 方法一：用相对路径（推荐！）
# 当前脚本在 API/ 目录，.env 在上一级目录
#load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

# 方法二：用绝对路径（更保险，适合调试）
load_dotenv(r"D:\Mypython\.env")  # 注意：Windows 路径用 r"" 或双反斜杠 \\

# 后续代码不变
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    print("❌ 未找到 API Key，请检查 .env 文件")
else:
    print(f"✅ Key 加载成功：{api_key[:8]}...")