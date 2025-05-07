import json
import logging
from llama_index.postprocessor.cohere_rerank import CohereRerank
from llama_index.core.schema import NodeWithScore
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
        print("Reformulated query: %s", reformulated_query)
        return reformulated_query_dict.get("reformulated_message", "")

    def get_response(self, user_message, history, selected_course) -> str:
        """
        Retreive relevant chunks from the document and generate a response using the LLM.
        """
        reformulated_query = self.reformulate_query(history=history,query=user_message)
        retrieved_chunks = vectordb.query(query=reformulated_query, selected_course=selected_course)
        print("Retrieved chunks:", retrieved_chunks)
        
        if retrieved_chunks.points: 
            if isinstance(retrieved_chunks, tuple):
                retrieved_chunks = retrieved_chunks[0]
            context_str = "\n\n".join([f"Context {i + 1}:\n{content.payload['text']}" for i, content in enumerate(retrieved_chunks)])
            # nodes = [
            #     NodeWithScore(id=chunk.id, score=chunk.score, payload=chunk.payload)
            #     for chunk in retrieved_chunks
            # ]
            # reranked_chunks = self.reranker.postprocess_nodes(nodes=nodes, query_str=reformulated_query)
            # filtered_chunks = [node for node in reranked_chunks if node.score >= 0.3]
            # print("Filtered chunks: ", filtered_chunks)
            # context_str = "\n\n".join([f"Context {i + 1}:\n{content.payload['text']}" for i, content in enumerate(filtered_chunks)])
        else:
            context_str="No relevant context found."
            
        # prompt = self.qa_prompt.format(context_str=context_str)
        # messages = [{"role": "system", "content": prompt}]
        # for user_turn, bot_turn in history:
        #     messages.append({"role": "user", "content": user_turn})
        #     messages.append({"role": "assistant", "content": bot_turn}) 
        # messages.append({"role": "user", "content": user_message})
        
        prompt = self.qa_prompt.format(user_message=user_message, context_str=context_str)

        response = llm.complete(prompt)
        response_dict = json.loads(response)
        print("LLM response: %s", response)
        return response_dict.get("message", "")