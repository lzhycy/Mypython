# 题目：构建一个“智能客服工单自动路由”原型
# 🎯 业务背景
# 某电商公司的客服系统每天收到大量用户反馈。现在需要写一个脚本，当新工单进来时，自动计算它与各个“标准问题模板”的语义相似度，并将其分配给最匹配的部门处理。
# 🛠️ 任务要求
# 请使用 uv 运行环境 + BAAI/bge-small-zh-v1.5 模型完成以下功能：
# 定义标准模板库（作为知识库）：
# 退款售后：“如何申请退货退款？”
# 物流查询：“我的快递到哪了？”
# 账号安全：“怎么修改登录密码？”
# 接收一条新工单文本：
# "买的东西坏了想退钱怎么办"
# 核心计算：
# 将模板库和新工单分别转向量（注意归一化）。
# 计算新工单与每一个模板的余弦相似度。
# 输出结果：
# 打印出新工单与每个模板的相似度分数（保留4位小数）。
# 找出相似度最高的模板，打印出匹配结果，格式如：🎯 最佳匹配：[退款售后] (相似度: 0.xx)
# 💡 提示与考察点
# 考察点1：你是否记得在 encode() 时加上 normalize_embeddings=True？
# 考察点2：你是否能正确处理“1条新工单 vs N条模板”的批量相似度计算，而不是写低效的 for 循环逐个算？
# 考察点3：你是否能从二维相似度矩阵中正确提取出最大值及其对应的索引？


import os 
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'  

from sentence_transformers import SentenceTransformer #文本模型
from sklearn.metrics.pairwise import cosine_similarity #向量计算

model = SentenceTransformer('BAAI/bge-small-zh-v1.5') #加载模型

texts = [
    "退款售后：“如何申请退货退款？”",
    "物流查询：“我的快递到哪了？”",
    "账号安全：“怎么修改登录密码？”"
]

#批量转向量
zxl=model.encode(texts,normalize_embeddings=True,max_length=512)

#新的文本
texts_1 = [
    "买的东西坏了想退钱怎么办"
]
xzxl = model.encode(texts_1,normalize_embeddings=True,max_length=512)
                    
                    
                    

#计算向量相似
similarity_scores = cosine_similarity(xzxl, zxl)[0]

# 1. 找出最高相似度的索引和分数
best_idx = similarity_scores.argmax()     # 获取最大值的下标 (0, 1, 或 2)
best_score = similarity_scores[best_idx]  # 获取对应的相似度分数

print("best_idx:", best_idx)
print("best_score:", best_score)
# 2. 格式化打印所有相似度
print("📊 各模板相似度得分：")
for i, text in enumerate(texts):
    print(f"   {text} -> {similarity_scores[i]:.4f}")

# 3. 输出最终匹配结果
# 提取部门名称（去掉冒号后面的具体问题，让输出更干净）
dept_name = texts[best_idx].split("：“")[0]
print(f"\n🎯 最佳匹配：[{dept_name}] (相似度: {best_score:.4f})")

