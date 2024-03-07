from sentence_transformers import SentenceTransformer
import numpy as np

"""
可能需要开启系统代理来进行运算.
"""
class Embedder:
    
    model = SentenceTransformer('BAAI/bge-large-en-v1.5', device='cuda')

    def __init__(self) -> None:
        print("Successful Load BAAI Model For Embedding.")
    
    def embeddingForList(embedder:SentenceTransformer,texts:list[str]) -> np.ndarray:
        """
        对一个list[str]来进行embedding操作
        """
        embeddings = embedder.encode(texts, normalize_embeddings=True)
        return embeddings

    def embeddingForOne(embedder:SentenceTransformer,text:str) -> np.ndarray:
        """
        对某一个str来进行embedding操作
        """
        embeddings = embedder.encode([text], normalize_embeddings=True)
        return embeddings[0]

    def caculateSimilarity(a:np.ndarray, b:np.ndarray):
        """
        计算两个embedding的相似度
        """
        return a @ b.T
    