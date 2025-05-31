import os
import logging
import json
import asyncio
from pathlib import Path
from llama_index.core import Document
from llama_cloud_services import LlamaParse
from src.embedder import Embedder

logger = logging.getLogger(__name__)
embedder = Embedder()

def save_documents_to_json(documents, output_path):
    json_data = []
    for doc in documents:
        json_data.append({
            "text": doc.text,
            "metadata": doc.metadata,
        })
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, ensure_ascii=False, indent=2)
       
async def llama_parse(filepath):
    result = await LlamaParse().aparse(str(filepath))
    markdown_nodes = await result.aget_markdown_nodes(split_by_page=True)
    nodes_data = [node.to_dict() if hasattr(node, "to_dict") else dict(node) for node in markdown_nodes]
    return nodes_data
     
async def process_file(filepath):
    nodes_data = await llama_parse(filepath)
    filename = os.path.basename(filepath)
    text_documents = [
        Document(text=node["text"], metadata={**node.get("metadata", {}), "filename": filename})
        for node in nodes_data
    ]
    for doc in text_documents:
        doc.embedding = embedder.generate_embedding(doc.text)
    return text_documents

async def parse_pdf(file_path):
    """
    Parses all PDF files in a directory concurrently and returns their content as documents.
    """
    documents = []
    logger.info(f"Parsing text from {file_path}")

    filepaths = [p for p in Path(file_path).iterdir() if p.suffix.lower() == ".pdf"]
    tasks = [process_file(filepath) for filepath in filepaths]
    results = await asyncio.gather(*tasks)
    for docs in results:
        documents.extend(docs)

    save_documents_to_json(documents, "parsed_documents.json")
    return documents