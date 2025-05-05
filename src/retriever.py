from llama_index.postprocessor.cohere_rerank import CohereRerank
from src.llm import GeminiLLM
from src.config import Config
import logging

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
    
        reformulated_query = GeminiLLM.complete(prompt)
        logging.debug("Reformulated query: %s", reformulated_query)
        return reformulated_query.get("reformulated_message", "")

    def get_response(self, user_message, history):
        """
        Retreive relevant chunks from the document and generate a response using the LLM.
        """
        reformulated_query = self.reformulate_query(history=history,query=user_message)
        retrieved_chunks = self.retriever.retrieve(reformulated_query) 
        reranked_chunks = self.reranker.postprocess_nodes(retrieved_chunks, query_str=reformulated_query)
        filtered_chunks = [node for node in reranked_chunks if node.score >= 0.3]
        
        context_str = "\n\n".join([f"Context {i + 1}:\n{content['text']}" for i, content in enumerate(filtered_chunks)])
        prompt = self.qa_prompt.format(context_str=context_str)
        
        messages = [{"role": "system", "content": prompt}]
        for user_turn, bot_turn in history:
            messages.append({"role": "user", "content": user_turn})
            messages.append({"role": "assistant", "content": bot_turn}) 
        messages.append({"role": "user", "content": user_message})
        
        response = GeminiLLM.complete_chat(messages)
        logging.debug("LLM response: %s", response)
        return response.get("message", "")