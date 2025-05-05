
from typing import Dict, List, Optional
from google import genai
import src.config as Config
import logging


class GeminiLLM:
    def __init__(self):
        """
        Initialize the Gemini client with API key.
        """
        self._client = genai.Client(api_key=Config.LLM_API_KEY)
        self.model = Config.LLM_MODEL_NAME
        
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
        response = self._client.models.generate_content(
            model=self.model,
            contents=messages,
            temperature=temperature,
            max_output_tokens=max_tokens,
            response_mime_type=response_mime_type,
        )

        logging.debug("Gemini response: %s", response)
        return response.text