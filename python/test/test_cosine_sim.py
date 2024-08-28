import numpy as np

def cosine_similarity(vector1, vector2):
    # 计算点积
    dot_product = np.dot(vector1, vector2)
    
    # 计算两个向量的范数
    norm1 = np.linalg.norm(vector1)
    norm2 = np.linalg.norm(vector2)
    
    # 计算余弦相似度
    cosine_sim = dot_product / (norm1 * norm2)
    print(cosine_sim)

    return cosine_sim

# 示例向量
vec1 = np.array([1, 2, 3])
vec2 = np.array([4, 5, 6])

# 计算并打印余弦相似度
similarity = cosine_similarity(vec1, vec2)
print(f"余弦相似度: {similarity}")