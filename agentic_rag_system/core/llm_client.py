import requests
import json
from typing import List, Dict, Any, Optional
from config import Config

class LMStudioClient:
    """Client for interacting with LMStudio for chat completions and embeddings"""
    
    def __init__(self):
        self.base_url = Config.LMSTUDIO_BASE_URL
        self.api_key = Config.LMSTUDIO_API_KEY
        self.chat_model = Config.CHAT_MODEL
        self.embedding_model = Config.EMBEDDING_MODEL
        self.temperature = Config.TEMPERATURE
        self.max_tokens = Config.MAX_TOKENS
        
    def _make_request(self, endpoint: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Make HTTP request to LMStudio"""
        url = f"{self.base_url}/{endpoint}"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=60)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error making request to LMStudio: {e}")
            raise
    
    def chat_completion(self, messages: List[Dict[str, str]], 
                       temperature: Optional[float] = None,
                       max_tokens: Optional[int] = None) -> str:
        """Generate chat completion using LMStudio"""
        payload = {
            "model": self.chat_model,
            "messages": messages,
            "temperature": temperature or self.temperature,
            "max_tokens": max_tokens or self.max_tokens,
            "stream": False
        }
        
        try:
            response = self._make_request("v1/chat/completions", payload)
            return response["choices"][0]["message"]["content"]
        except Exception as e:
            print(f"Error in chat completion: {e}")
            return "I apologize, but I encountered an error processing your request."
    
    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using LMStudio"""
        if not texts:
            return []
            
        payload = {
            "model": self.embedding_model,
            "input": texts
        }
        
        try:
            response = self._make_request("v1/embeddings", payload)
            return [item["embedding"] for item in response["data"]]
        except Exception as e:
            print(f"Error generating embeddings: {e}")
            # Return zero embeddings as fallback
            return [[0.0] * 384 for _ in texts]  # Assuming 384-dim embeddings
    
    def get_single_embedding(self, text: str) -> List[float]:
        """Generate embedding for a single text"""
        embeddings = self.get_embeddings([text])
        return embeddings[0] if embeddings else [0.0] * 384
    
    def test_connection(self) -> bool:
        """Test connection to LMStudio"""
        try:
            test_messages = [{"role": "user", "content": "Hello"}]
            response = self.chat_completion(test_messages)
            return bool(response and len(response) > 0)
        except Exception as e:
            print(f"LMStudio connection test failed: {e}")
            return False