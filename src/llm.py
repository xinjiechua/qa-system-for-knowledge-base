import logging
from typing import Dict, List, Optional
from google import genai
from google.genai import types
from src.config import Config

logger = logging.getLogger(__name__)
config = Config()

class GeminiLLM:
    def __init__(self):
        """
        Initialize the Gemini client with API key.
        """
        self._client = genai.Client(api_key=config.GEMINI_API_KEY)
        self.model = config.LLM_MODEL_NAME
        
    def complete(
        self,
        messages: List[Dict],
        temperature: float = 0,
        max_tokens: int = 2048,
        response_mime_type: str = "application/json",
    ) -> Optional[str]:
        """
        Send a request to Gemini for chat completion.
        """
        
        config = types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
            response_mime_type=response_mime_type,
        )
           
        response = self._client.models.generate_content(
            model=self.model,
            contents=messages,
            config=config,
        )
        
        if response.candidates:
            if response.candidates[0].content and response.candidates[0].content.parts:
                return response.candidates[0].content.parts[0].text
        return None