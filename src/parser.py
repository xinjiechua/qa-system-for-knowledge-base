import os
from pathlib import Path
from llama_index.core import Document
from unstructured.partition.pdf import partition_pdf
from src.embedder import Embedder
import logging
import json
embedder = Embedder()

def save_documents_to_json(documents, output_path):
    json_data = []
    for doc in documents:
        json_data.append({
            "text": doc.text,
            "metadata": doc.metadata,
            "embedding": doc.embedding
        })
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, ensure_ascii=False, indent=2)
        
def parse_pdf(file_path):
    """
    Parses a PDF file and returns its content as document.
    """
    documents = []
    logging.info(f"Parsing text from {file_path}")

    for filepath in Path(file_path).iterdir():
        raw_elements = []
        raw_elements = partition_pdf(
            filename=str(filepath),
            infer_table_structure=True,
            chunking_strategy="by_title",
            max_characters=1000,
            combine_under_n_chars=800,
            overlap=300,
            strategy="hi_res",
            include_metadata=True,
            multipage_sections=True,
            extract_images_in_pdf=False,
        )
        texts = []
        for element in raw_elements:
            if "unstructured.documents.elements.Table" in str(type(element)):
                texts.append(str(element))
            elif "unstructured.documents.elements.CompositeElement" in str(type(element)):
                texts.append(str(element))
            else:
                texts.append(str(element))
    
    filename = os.path.basename(filepath)
    text_documents = [Document(text=text, metadata={'filename': filename}) for text in texts]
    for doc in text_documents:
        doc.embedding = embedder.generate_embedding(doc.text)
    documents.extend(text_documents)
    save_documents_to_json(documents, "parsed_documents.json")
    return documents       



