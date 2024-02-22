from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer('BAAI/bge-large-en-v1.5')

def embeddingForList(texts:list[str]) -> np.ndarray:
    embeddings = model.encode(texts, normalize_embeddings=True)
    return embeddings

def embeddingForOne(text:str) -> np.ndarray:
    embeddings = model.encode([text], normalize_embeddings=True)
    return embeddings[0]

def caculateSimilarity(a:np.ndarray, b:np.ndarray):
    return a @ b.T
    