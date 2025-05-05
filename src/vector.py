
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct, Filter, FieldCondition, Match
from src.config import Config
from src.embedder import Embedder
from src.parser import parse_pdf
import logging

class VectorDB():
    def __init__(self):        
        self.client = QdrantClient(
            url=f"http://{Config.QDRANT_HOST}:{Config.QDRANT_PORT}",
            api_key=Config.QDRANT_API_KEY
        )
        logging.info(f"Connected to Qdrant at {Config.QDRANT_HOST}:{Config.QDRANT_PORT}")
        self.collection_name = Config.COLLECTION_NAME
        
        if not self.client.collection_exists(collection_name=self.collection_name):
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=Embedder.get_dimension(),
                    distance=Distance.COSINE
                ),
            )
        logging.info(f"Collection '{self.collection_name}' is ready.")

    def insert(self):
        documents = parse_pdf(file_path=Config.DATA_PATH)
        
        points = [
            PointStruct(
                id=doc.id_,
                vector=doc.embedding,
                payload={"metadata": doc.metadata, "text": doc.text},
            )
            for doc in documents
        ]
        
        self.client.upload_points(
            collection_name=self.collection_name, 
            points=points, 
            batch_size=64, 
            parallel=4
        )
        logging.info(f"Inserted {len(points)} points into the collection '{self.collection_name}'.")
        
    def query(self, query: str, top_k: int = Config.RETRIEVE_TOP_K):
        """
        Query the vector database for similar documents.
        """
        query_vector = Embedder.generate_embedding(query)
        
        results = self.client.query_points(
            collection_name=self.collection_name,
            query_vector=query_vector,
            query_filter=Filter(must=[FieldCondition(key='filename',range=Match(value="computer_science"))]),
            limit=top_k,
            with_payload=True,
            with_vectors=False,     
        )
        
        return results