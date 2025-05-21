import json
import logging
from llama_index.postprocessor.cohere_rerank import CohereRerank
from llama_index.core.schema import NodeWithScore, TextNode
from src.llm import GeminiLLM
from src.config import Config
from src.vector import VectorDB

llm = GeminiLLM()
vectordb = VectorDB()

class RAG():
    def __init__(self):
        self.reranker = CohereRerank(top_n=Config.RERANK_TOP_P)
        with open(Config.REFORMULATE_PROMPT_PATH, "r") as file:
            self.reformulate_prompt = file.read()
        with open(Config.QA_PROMPT_PATH, "r") as file:
            self.qa_prompt = file.read()
        
    def reformulate_query(self, history, query) -> str:
        """
        Reformulate query based on past message history 
        """
        prompt = self.reformulate_prompt.format(
            chat_history=history,
            latest_message=query
        )
    
        reformulated_query = llm.complete(messages=prompt)
        reformulated_query_dict = json.loads(reformulated_query)
        print("Reformulated query: ", reformulated_query)
        return reformulated_query_dict.get("reformulated_message", "")

    def rerank_chunks(self, chunks, query: str) -> list:
        """
        Rerank the retrieved chunks using Cohere reranker
        """
        nodes = [
            NodeWithScore(
                node=TextNode(
                    text=chunk.payload["text"],
                    metadata=chunk.payload["metadata"],
                    id_=chunk.id
                ),
                score=chunk.score
            )
            for chunk in chunks
        ]
        reranked_chunks = self.reranker.postprocess_nodes(nodes=nodes, query_str=query)
        filtered_chunks = [node for node in reranked_chunks if node.score >= 0.3]
        return filtered_chunks

    def get_response(self, user_message, history, selected_course) -> str:
        """
        Retreive relevant chunks from the document and generate a response using the LLM.
        """
        reformulated_query = self.reformulate_query(history=history,query=user_message)
        retrieved_chunks = vectordb.query(query=reformulated_query, selected_course=selected_course)
        print("Retrieved chunks: ", retrieved_chunks)
        
        if hasattr(retrieved_chunks, 'points') and retrieved_chunks.points: 
            chunks = retrieved_chunks.points
            filtered_chunks = self.rerank_chunks(chunks, reformulated_query)
            print("Filtered chunks: ", filtered_chunks)
            context_str = "\n\n".join([f"Context {i + 1}:\n{content.node.text}" for i, content in enumerate(filtered_chunks)])
        else:
            context_str="No relevant context found."
            
        prompt = self.qa_prompt.format(context_str=context_str)
        messages = [{"role": "model", "parts": [{"text": prompt}]}]
        for user_turn, bot_turn in history:
            messages.append({"role": "user", "parts": [{"text": user_turn}]})
            messages.append({"role": "model", "parts": [{"text": bot_turn}]})
        messages.append({"role": "user", "parts": [{"text": user_message}]})
        
        print("Messages: ", messages)
        response = llm.complete(messages=messages)
        response_dict = json.loads(response)
        print("LLM response: ", response)
        return response_dict.get("message", "")