from sentence_transformers import SentenceTransformer
import numpy as np

"""
可能需要开启系统代理来进行运算.
"""
model = SentenceTransformer('BAAI/bge-large-en-v1.5', device='cuda')

class Embedder:
    def __init__(self) -> None:
        self.model = SentenceTransformer('BAAI/bge-lagre-en-v1.5', device='cuda')
    

def embeddingForList(texts:list[str]) -> np.ndarray:
    """
    对一个list[str]来进行embedding操作
    """
    embeddings = model.encode(texts, normalize_embeddings=True)
    return embeddings

def embeddingForOne(text:str) -> np.ndarray:
    """
    对某一个str来进行embedding操作
    """
    embeddings = model.encode([text], normalize_embeddings=True)
    return embeddings[0]

def caculateSimilarity(a:np.ndarray, b:np.ndarray):
    """
    计算两个embedding的相似度
    """
    return a @ b.T
    