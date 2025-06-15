import requests
import json
import re
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
    
    def _strip_thinking_tags(self, text: str) -> str:
        """Remove thinking content from LLM response"""
        if not text:
            return text
            
        # Remove content between <think> and </think> tags (case insensitive)
        # This handles both single line and multiline thinking blocks
        cleaned_text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL | re.IGNORECASE)
        
        # Also handle <thinking> tags in case the model uses different format
        cleaned_text = re.sub(r'<thinking>.*?</thinking>', '', cleaned_text, flags=re.DOTALL | re.IGNORECASE)
        
        # Clean up extra whitespace and newlines that might be left
        cleaned_text = re.sub(r'\n\s*\n\s*\n', '\n\n', cleaned_text)  # Replace multiple newlines
        cleaned_text = cleaned_text.strip()
        
        return cleaned_text
    
    def _extract_final_answer(self, text: str) -> str:
        """Extract the final answer portion if explicitly marked"""
        if not text:
            return text
            
        # Look for common patterns that indicate the start of the actual answer
        answer_patterns = [
            r'\*\*Answer:\*\*\s*(.*)',  # **Answer:** pattern
            r'Answer:\s*(.*)',          # Answer: pattern
            r'\*\*Response:\*\*\s*(.*)', # **Response:** pattern
            r'Response:\s*(.*)',        # Response: pattern
        ]
        
        for pattern in answer_patterns:
            match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        # If no explicit answer marker found, return the cleaned text
        return text
    
    def _clean_llm_response(self, raw_response: str) -> str:
        """Clean LLM response by removing thinking content and extracting final answer"""
        if not raw_response:
            return "I apologize, but I received an empty response."
        
        # Step 1: Remove thinking tags
        cleaned_response = self._strip_thinking_tags(raw_response)
        
        # Step 2: Extract final answer if explicitly marked
        final_response = self._extract_final_answer(cleaned_response)
        
        # Step 3: Final cleanup
        final_response = final_response.strip()
        
        # Ensure we have some content
        if not final_response or len(final_response.strip()) < 10:
            # Fallback: try to extract any meaningful content from original response
            fallback_clean = re.sub(r'<[^>]+>.*?</[^>]+>', '', raw_response, flags=re.DOTALL)
            final_response = fallback_clean.strip() if fallback_clean.strip() else raw_response
        
        return final_response
    
    def chat_completion(self, messages: List[Dict[str, str]], 
                       temperature: Optional[float] = None,
                       max_tokens: Optional[int] = None,
                       strip_thinking: bool = True) -> str:
        """Generate chat completion using LMStudio with optional thinking content removal"""
        payload = {
            "model": self.chat_model,
            "messages": messages,
            "temperature": temperature or self.temperature,
            "max_tokens": max_tokens or self.max_tokens,
            "stream": False
        }
        
        try:
            response = self._make_request("v1/chat/completions", payload)
            raw_content = response["choices"][0]["message"]["content"]
            
            # Clean the response if strip_thinking is enabled (default)
            if strip_thinking:
                return self._clean_llm_response(raw_content)
            else:
                return raw_content
                
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
            response = self.chat_completion(test_messages, strip_thinking=False)  # Don't strip for test
            return bool(response and len(response) > 0)
        except Exception as e:
            print(f"LMStudio connection test failed: {e}")
            return False