#Extract posts and items' embeddings with SentenceTransformers 

from sentence_transformers import SentenceTransformer, util
import random

set_seed(42)

model = SentenceTransformer('sentence-transformers/msmarco-MiniLM-L6-cos-v5')

def get_embedding(text, model):
    return model.encode(text)

