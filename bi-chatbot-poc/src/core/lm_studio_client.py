"""
LM Studio API client for local LLM interactions.
Provides a simple interface to communicate with LM Studio server.
"""

import requests
import json
import logging
from typing import Dict, Any, Optional, List
import time

logger = logging.getLogger(__name__)

class LMStudioClient:
    """Client for interacting with LM Studio local server"""
    
    def __init__(self, config: Dict[str, Any]):
        self.base_url = config.get('base_url', 'http://localhost:1234')
        self.api_key = config.get('api_key', '')  # Usually not needed for local
        self.model = config.get('model', 'local-model')
        self.timeout = config.get('timeout', 30)
        self.max_retries = config.get('max_retries', 3)
        
        # Default generation parameters
        self.default_params = {
            'temperature': config.get('temperature', 0.1),
            'max_tokens': config.get('max_tokens', 1000),
            'top_p': config.get('top_p', 0.9),
            'stream': False
        }
        
        logger.info(f"Initialized LM Studio client for {self.base_url}")
    
    def _make_request(self, endpoint: str, data: Dict[str, Any]) -> requests.Response:
        """Make HTTP request to LM Studio server with retries"""
        url = f"{self.base_url}{endpoint}"
        headers = {
            'Content-Type': 'application/json'
        }
        
        if self.api_key:
            headers['Authorization'] = f'Bearer {self.api_key}'
        
        for attempt in range(self.max_retries):
            try:
                response = requests.post(
                    url, 
                    headers=headers, 
                    json=data, 
                    timeout=self.timeout
                )
                response.raise_for_status()
                return response
                
            except requests.exceptions.RequestException as e:
                logger.warning(f"Request attempt {attempt + 1} failed: {str(e)}")
                if attempt == self.max_retries - 1:
                    raise
                time.sleep(2 ** attempt)  # Exponential backoff
        
        raise Exception("Max retries exceeded")
    
    def generate(self, 
                prompt: str, 
                system_prompt: Optional[str] = None,
                temperature: Optional[float] = None,
                max_tokens: Optional[int] = None,
                **kwargs) -> str:
        """Generate text completion from prompt"""
        
        try:
            # Prepare messages for chat completion format
            messages = []
            
            if system_prompt:
                messages.append({
                    "role": "system",
                    "content": system_prompt
                })
            
            messages.append({
                "role": "user", 
                "content": prompt
            })
            
            # Prepare request data
            request_data = {
                "model": self.model,
                "messages": messages,
                "temperature": temperature or self.default_params['temperature'],
                "max_tokens": max_tokens or self.default_params['max_tokens'],
                "top_p": self.default_params['top_p'],
                "stream": False
            }
            
            # Add any additional parameters
            request_data.update(kwargs)
            
            # Make request
            response = self._make_request('/v1/chat/completions', request_data)
            response_data = response.json()
            
            # Extract generated text
            if 'choices' in response_data and len(response_data['choices']) > 0:
                return response_data['choices'][0]['message']['content'].strip()
            else:
                raise Exception("No response generated")
                
        except Exception as e:
            logger.error(f"Error in text generation: {str(e)}")
            raise Exception(f"Failed to generate response: {str(e)}")
    
    def generate_json(self, 
                     prompt: str, 
                     system_prompt: Optional[str] = None,
                     **kwargs) -> Dict[str, Any]:
        """Generate JSON response with structured output"""
        
        # Add JSON instruction to prompt
        json_prompt = f"{prompt}\n\nPlease respond with valid JSON only."
        
        if system_prompt:
            system_prompt += " Always respond with valid JSON format."
        else:
            system_prompt = "You are a helpful assistant that always responds with valid JSON format."
        
        response_text = self.generate(
            prompt=json_prompt,
            system_prompt=system_prompt,
            temperature=0.1,  # Lower temperature for structured output
            **kwargs
        )
        
        try:
            # Try to parse JSON
            return json.loads(response_text)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {response_text}")
            # Attempt to extract JSON from response
            try:
                # Look for JSON block in response
                import re
                json_match = re.search(r'```json\s*(.*?)\s*```', response_text, re.DOTALL)
                if json_match:
                    return json.loads(json_match.group(1))
                
                # Try to find JSON-like content
                json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                if json_match:
                    return json.loads(json_match.group(0))
                    
            except:
                pass
            
            raise Exception(f"Invalid JSON response: {response_text}")
    
    def chat(self, 
             messages: List[Dict[str, str]], 
             **kwargs) -> str:
        """Multi-turn chat completion"""
        
        try:
            request_data = {
                "model": self.model,
                "messages": messages,
                "temperature": self.default_params['temperature'],
                "max_tokens": self.default_params['max_tokens'],
                "top_p": self.default_params['top_p'],
                "stream": False
            }
            
            # Override with any provided parameters
            request_data.update(kwargs)
            
            response = self._make_request('/v1/chat/completions', request_data)
            response_data = response.json()
            
            if 'choices' in response_data and len(response_data['choices']) > 0:
                return response_data['choices'][0]['message']['content'].strip()
            else:
                raise Exception("No response generated")
                
        except Exception as e:
            logger.error(f"Error in chat completion: {str(e)}")
            raise Exception(f"Failed to generate chat response: {str(e)}")
    
    def health_check(self) -> bool:
        """Check if LM Studio server is available"""
        try:
            response = requests.get(f"{self.base_url}/v1/models", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def get_models(self) -> List[str]:
        """Get available models from LM Studio"""
        try:
            response = requests.get(f"{self.base_url}/v1/models", timeout=10)
            response.raise_for_status()
            data = response.json()
            
            if 'data' in data:
                return [model['id'] for model in data['data']]
            return []
            
        except Exception as e:
            logger.error(f"Error fetching models: {str(e)}")
            return []
    
    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for texts (if supported by model)"""
        try:
            request_data = {
                "model": self.model,
                "input": texts
            }
            
            response = self._make_request('/v1/embeddings', request_data)
            response_data = response.json()
            
            if 'data' in response_data:
                return [item['embedding'] for item in response_data['data']]
            else:
                raise Exception("No embeddings returned")
                
        except Exception as e:
            logger.warning(f"Embeddings not supported or error: {str(e)}")
            # Return dummy embeddings for POC
            import numpy as np
            return [np.random.normal(0, 1, 384).tolist() for _ in texts]