# ================= 导入依赖库 =================
# SentenceTransformer: 用于把文字变成向量（你之前学过的 Embedding）
from sentence_transformers import SentenceTransformer
# chromadb: 轻量级向量数据库，专门用来存和查向量
import chromadb

# ================= 1. 初始化模型与数据库 =================
# 加载 BGE 中文小模型，这个模型会把中文文本转成 512 维的向量
model = SentenceTransformer('BAAI/bge-small-zh-v1.5')

# 创建 Chroma 客户端
# 注意：这里用的是内存模式，程序一关数据就没了，仅适合学习测试
# 如果要持久化存储，应改为: chromadb.PersistentClient(path="./chroma_data")
client = chromadb.Client()

# 获取或创建一个名为 "faq_knowledge" 的集合
# 集合相当于关系型数据库里的"表"，是存取数据的基本单位
collection = client.get_or_create_collection(
    name="faq_knowledge",
    # 👇 关键配置：指定向量相似度计算方式为余弦相似度
    # 因为你之前做了 normalize_embeddings=True，必须配 cosine 才准确
    # 如果不写这个，Chroma 默认用 L2 距离，会导致搜索结果完全错误
    metadata={"hnsw:space": "cosine"}
)

# ================= 2. 准备要存入的数据 =================
# 原始文本列表：这是用户可能会问的问题原文
docs = [
    "买的東西坏了想退钱怎么办",
    "快递三天了还没到，去哪查物流",
    "账号密码忘了怎么重置"
]

# 元数据列表：给每条文本打的标签，后续可以用 where 条件过滤
# 元数据必须是字典格式，支持字符串、数字、布尔值
metadatas = [
    {"category": "refund", "priority": "high"},      # 退款类，高优先级
    {"category": "logistics", "priority": "normal"}, # 物流类，普通优先级
    {"category": "account", "priority": "low"}       # 账号类，低优先级
]

# 唯一ID列表：每条数据的身份证号，Upsert 时靠它判断是新增还是更新
# 必须是字符串类型，不能重复
ids = ["doc_001", "doc_002", "doc_003"]

# ================= 3. 生成向量并写入数据库 =================
# 批量把文本转成向量
# normalize_embeddings=True: 归一化，让向量长度为1，适配余弦相似度
# .tolist(): 把 numpy 数组转成 Python 原生列表，Chroma 只接受原生列表
vectors = model.encode(docs, normalize_embeddings=True).tolist()

# 💾 核心写入操作：Upsert = Update + Insert
# 如果 id 已存在 → 更新该条数据的向量、原文、元数据
# 如果 id 不存在 → 新增一条完整数据
collection.upsert(
    ids=ids,              # 必填：数据唯一标识
    embeddings=vectors,   # 必填：文本对应的向量
    documents=docs,       # 可选但强烈建议：存原文，查询时直接返回，不用再去别处找
    metadatas=metadatas   # 可选但强烈建议：存标签，用于后续过滤筛选
)
print(f"✅ 成功存入 {len(docs)} 条数据")

# ================= 4. 构造查询请求 =================
# 用户的实际提问
query = "东西坏了能退款吗"

# 把查询文本也转成向量
# 注意：查询时也要用同一个模型、同样的归一化参数，否则维度/空间不一致会报错
# [query]: encode 接受列表，所以要把单个字符串包在列表里
# [0]: 取返回列表的第一个元素，因为查询只有一条文本
query_vector = model.encode([query], normalize_embeddings=True).tolist()[0]

# ================= 5. 执行向量检索 =================
# 🔍 核心查询操作：在集合中找最相似的 Top-K 条数据
results = collection.query(
    query_embeddings=[query_vector],  # 必填：查询向量，必须是列表格式（支持批量查询）
    n_results=3,                      # 返回最相似的前 3 条结果
    # 👇 元数据过滤：只在 category=refund 的数据里搜索
    # 这相当于 SQL 的 WHERE category='refund'
    # 先缩小范围再算相似度，既提升速度又避免跨类别误召回
    where={"category": "refund"},
    # 指定返回内容：documents=原文, distances=距离分数, metadatas=标签
    # 不指定的话默认只返回 id 和 distance
    include=["documents", "distances", "metadatas"]
)

# ================= 6. 解析并展示结果 =================
print(f"\n🔎 查询: '{query}'")
print("="*40)

# results 的结构是嵌套列表：results['ids'][0] 表示第一条查询的结果列表
# 因为我们只传了 1 个查询向量，所以固定取 [0]
for i in range(len(results['ids'][0])):
    # 取出第 i 条结果的原文
    doc = results['documents'][0][i]
    # 取出第 i 条结果的距离分数
    dist = results['distances'][0][i]
    # 取出第 i 条结果的元数据标签
    meta = results['metadatas'][0][i]

    # ⚠️ 重要转换：Chroma 在 cosine 空间下返回的是"距离"
    # 距离越小越相似，但我们习惯用"相似度"（越大越像）
    # 公式：similarity = 1 - distance
    similarity = 1 - dist

    print(f"Top-{i+1} | 相似度: {similarity:.4f}")
    print(f"   内容: {doc}")
    print(f"   标签: {meta}")
    print("-"*40)