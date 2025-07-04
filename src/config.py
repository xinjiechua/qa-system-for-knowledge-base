import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "gemini-2.0-flash-lite")

    QDRANT_HOST =  os.getenv("MILVUS_HOST", "localhost")
    QDRANT_PORT = int(os.getenv("MILVUS_PORT", "6333"))
    QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
    COHERE_API_KEY= os.getenv("COHERE_API_KEY")
    COLLECTION_NAME = os.getenv("COLLECTION_NAME", "Repository")
    
    RETRIEVE_TOP_K = int(os.getenv("RETRIEVE_TOP_K", 8))
    RERANK_TOP_P = int(os.getenv("RERANK_TOP_P", 5))
    
    DATA_PATH = os.getenv("DATA_PATH", "./data")
    GUARDRAIL_PATH = os.getenv("GUARDRAIL_PATH", "src/prompts/guardrail_prompt.txt")
    QA_PROMPT_PATH = os.getenv("QA_PROMPT_PATH", "src/prompts/qa_prompt.txt")
    REFORMULATE_PROMPT_PATH = os.getenv("REFORMULATE_PROMPT_PATH", "src/prompts/reformulate_prompt.txt")
    
    COURSE_TO_FILE_MAP = {
        "Computer Science": "computer_science.pdf",
        "Electrical Engineering": "electrical_engineering.pdf",
        "Medicine": "medicine.pdf",
        "Pharmacy": "pharmacy.pdf",
        "Creative Arts": "creative_arts.pdf"
    }