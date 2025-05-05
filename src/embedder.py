import os
from sentence_transformers import SentenceTransformer

class Embedder():
    
    model = SentenceTransformer("all-MiniLM-L6-v2")
    
    def __init__(self):
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        self.dimension = 384 
    
    def generate_embedding(self, content: str):
        return self.model.encode(content).tolist()
    
    @staticmethod
    def get_dimension():
        return 384