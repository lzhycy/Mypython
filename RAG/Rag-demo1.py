from dotenv import load_dotenv
import os
import sys

# 方法一：用相对路径（推荐！）
# 当前脚本在 API/ 目录，.env 在上一级目录
#load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

# 方法二：用绝对路径（更保险，适合调试）
load_dotenv(r"D:\Mypython\.env")  # 注意：Windows 路径用 r"" 或双反斜杠 \\


# 2. 导入核心组件 (注意：这里没有引入任何 community 包)
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
import dashscope
from dashscope import TextEmbedding, Generation

# ==========================================
# 自定义 DashScope Embeddings (替代不稳定的社区版)
# ==========================================
class DashScopeEmbeddings:
    def __init__(self, model="text-embedding-v3"):
        self.model = model

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """批量向量化文档"""
        resp = TextEmbedding.call(model=self.model, input=texts)
        if resp.status_code != 200:
            raise Exception(f"Embedding Error: {resp.code} - {resp.message}")
        return [item["embedding"] for item in resp.output["embeddings"]]

    def embed_query(self, text: str) -> list[float]:
        """向量化单条查询"""
        return self.embed_documents([text])[0]

# ==========================================
# RAG 核心流程
# ==========================================
def run_rag():
    # 1. 准备模拟数据 (实际使用时可替换为 PyPDFLoader 读取本地PDF)
    raw_docs = [
        Document(page_content="LangChain v0.3 采用了全新的架构设计，将核心功能拆分到了 langchain-core 中。"),
        Document(page_content="ChromaDB 是一个开源的向量数据库，非常适合用于构建本地的 RAG 应用。"),
        Document(page_content="阿里云 DashScope 提供了通义千问(Qwen)系列大模型和高质量的文本向量模型。"),
    ]

    # 2. 文档切分
    splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)
    chunks = splitter.split_documents(raw_docs)
    print(f"✅ 文档切分完成，共得到 {len(chunks)} 个片段")

    # 3. 向量化并存入 ChromaDB (纯内存模式，无需启动额外服务)
    embeddings = DashScopeEmbeddings()
    vectorstore = Chroma.from_documents(
        documents=chunks, 
        embedding=embeddings,
        collection_name="my_rag_demo"
    )
    print("✅ 向量入库完成")

    # 4. 语义检索
    query = "LangChain 的新架构有什么特点？"
    retrieved_docs = vectorstore.similarity_search(query, k=1)
    context = retrieved_docs[0].page_content
    print(f"\n🔍 检索到的上下文:\n{context}")

    # 5. 调用 Qwen 大模型生成回答
    prompt = f"""请根据以下参考资料回答问题。如果资料中没有相关信息，请如实告知。
参考资料：{context}
问题：{query}
回答："""

    response = Generation.call(
        model="qwen-plus",
        messages=[{"role": "user", "content": prompt}],
        result_format="message"
    )
    
    answer = response.output.choices[0].message.content
    print(f"\n🤖 AI 回答:\n{answer}")

if __name__ == "__main__":
    run_rag()