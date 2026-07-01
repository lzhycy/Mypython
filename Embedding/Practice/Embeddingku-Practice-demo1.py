import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from sentence_transformers import SentenceTransformer
import chromadb

# ================= 1. 初始化模型与数据库 =================
model = SentenceTransformer('BAAI/bge-small-zh-v1.5')

#创建库客户端
client = chromadb.Client()

# 获取或创建一个名为 "faq_knowledge" 的集合
collection = client.get_or_create_collection(
    name="faq_knowledge",
    metadata={"hnsw:space": "cosine"}
)

#================= 2. 准备要存入的数据 =================
docs = [
    "买的東西坏了想退钱怎么办",
    "快递三天了还没到，去哪查物流",
    "账号密码忘了怎么重置"
]

#打标签
metadatas = [
    {"category": "refund", "priority": "high"},
    {"category": "logistics", "priority": "normal"},
    {"category": "account", "priority": "low"}
]

# 唯一ID列表
ids = ["doc_001", "doc_002", "doc_003"]

# ================= 3. 生成向量并写入数据库 =================
vectors = model.encode(docs, normalize_embeddings=True,max_length=512).tolist()

#存入数据库
collection.upsert(
    ids=ids,
    embeddings=vectors,
    documents=docs,
    metadatas=metadatas
)

#查询
query = "买的东西坏了想退钱怎么办"

#================= 4. 查询数据库 =================
# 生成查询向量
query_vector = model.encode([query], normalize_embeddings=True,max_length=512).tolist()

# 执行查询，返回最相似的前3条结果
results = collection.query(
    query_embeddings=query_vector,
    n_results=3
)