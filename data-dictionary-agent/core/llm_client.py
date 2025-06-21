"""
LMStudio Client for LLM and Embedding operations
Handles communication with LMStudio server
"""
import json
import asyncio
import aiohttp
from typing import Dict, List, Any, Optional, Union
import logging

logger = logging.getLogger(__name__)


class LMStudioClient:
    """Client for interacting with LMStudio API"""
    
    def __init__(self, base_url: str = "http://localhost:1234", 
                 model: str = "local-model", api_key: Optional[str] = None):
        """
        Initialize LMStudio client
        
        Args:
            base_url: LMStudio server URL
            model: Model name to use
            api_key: API key if required
        """
        self.base_url = base_url.rstrip('/')
        self.model = model
        self.api_key = api_key
        self.session = None
        
        # API endpoints
        self.chat_endpoint = f"{self.base_url}/v1/chat/completions"
        self.embeddings_endpoint = f"{self.base_url}/v1/embeddings"
        self.models_endpoint = f"{self.base_url}/v1/models"
    
    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
    
    async def _ensure_session(self):
        """Ensure session is available"""
        if not self.session:
            self.session = aiohttp.ClientSession()
    
    async def generate_response(self, messages: List[Dict[str, str]], 
                              temperature: float = 0.7, 
                              max_tokens: int = 2000,
                              system_prompt: Optional[str] = None) -> str:
        """
        Generate response from LLM
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            system_prompt: Optional system prompt
            
        Returns:
            Generated response text
        """
        try:
            await self._ensure_session()
            
            # Prepare messages
            formatted_messages = []
            
            # Add system prompt if provided
            if system_prompt:
                formatted_messages.append({
                    "role": "system",
                    "content": system_prompt
                })
            
            # Add user messages
            formatted_messages.extend(messages)
            
            # Prepare request payload
            payload = {
                "model": self.model,
                "messages": formatted_messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "stream": False
            }
            
            # Prepare headers
            headers = {
                "Content-Type": "application/json"
            }
            
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            
            # Make request
            async with self.session.post(
                self.chat_endpoint,
                json=payload,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=60)
            ) as response:
                
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"LMStudio API error: {response.status} - {error_text}")
                    raise Exception(f"API request failed: {response.status}")
                
                result = await response.json()
                
                # Extract response
                if 'choices' in result and len(result['choices']) > 0:
                    return result['choices'][0]['message']['content']
                else:
                    logger.error(f"Unexpected response format: {result}")
                    raise Exception("No response generated")
        
        except asyncio.TimeoutError:
            logger.error("Request timeout")
            raise Exception("Request timeout - LMStudio server may be overloaded")
        except aiohttp.ClientError as e:
            logger.error(f"Client error: {str(e)}")
            raise Exception(f"Connection error: {str(e)}")
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            raise
    
    async def generate_embeddings(self, texts: Union[str, List[str]]) -> List[List[float]]:
        """
        Generate embeddings for text(s)
        
        Args:
            texts: Single text or list of texts
            
        Returns:
            List of embedding vectors
        """
        try:
            await self._ensure_session()
            
            # Ensure texts is a list
            if isinstance(texts, str):
                texts = [texts]
            
            # Prepare request payload
            payload = {
                "model": self.model,
                "input": texts
            }
            
            # Prepare headers
            headers = {
                "Content-Type": "application/json"
            }
            
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            
            # Make request
            async with self.session.post(
                self.embeddings_endpoint,
                json=payload,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"Embeddings API error: {response.status} - {error_text}")
                    # Return empty embeddings on error for PoC
                    return [[0.0] * 384 for _ in texts]  # Default embedding size
                
                result = await response.json()
                
                # Extract embeddings
                if 'data' in result:
                    return [item['embedding'] for item in result['data']]
                else:
                    logger.warning("No embeddings in response, returning empty vectors")
                    return [[0.0] * 384 for _ in texts]
        
        except Exception as e:
            logger.error(f"Error generating embeddings: {str(e)}")
            # Return empty embeddings for PoC
            return [[0.0] * 384 for _ in (texts if isinstance(texts, list) else [texts])]
    
    async def check_health(self) -> bool:
        """
        Check if LMStudio server is healthy
        
        Returns:
            True if server is responding
        """
        try:
            await self._ensure_session()
            
            async with self.session.get(
                self.models_endpoint,
                timeout=aiohttp.ClientTimeout(total=10)
            ) as response:
                return response.status == 200
        
        except Exception as e:
            logger.error(f"Health check failed: {str(e)}")
            return False
    
    async def list_models(self) -> List[str]:
        """
        List available models
        
        Returns:
            List of model names
        """
        try:
            await self._ensure_session()
            
            async with self.session.get(
                self.models_endpoint,
                timeout=aiohttp.ClientTimeout(total=10)
            ) as response:
                
                if response.status != 200:
                    return []
                
                result = await response.json()
                
                if 'data' in result:
                    return [model['id'] for model in result['data']]
                
                return []
        
        except Exception as e:
            logger.error(f"Error listing models: {str(e)}")
            return []
    
    async def analyze_with_context(self, prompt: str, context_data: Dict[str, Any], 
                                 temperature: float = 0.3) -> str:
        """
        Analyze data with context using LLM
        
        Args:
            prompt: Analysis prompt
            context_data: Dictionary containing context information
            temperature: Sampling temperature
            
        Returns:
            Analysis result
        """
        try:
            # Format context data
            context_str = json.dumps(context_data, indent=2, default=str)
            
            # Create analysis message
            messages = [
                {
                    "role": "user",
                    "content": f"""
Context Data:
{context_str}

Task: {prompt}

Please provide a comprehensive analysis based on the context data provided.
Focus on actionable insights and practical recommendations.
"""
                }
            ]
            
            # Generate response
            response = await self.generate_response(
                messages=messages,
                temperature=temperature,
                max_tokens=1500
            )
            
            return response
        
        except Exception as e:
            logger.error(f"Error in context analysis: {str(e)}")
            return f"Analysis failed: {str(e)}"
    
    async def extract_insights(self, column_profiles: List[Dict[str, Any]]) -> List[str]:
        """
        Extract insights from column profiles using LLM
        
        Args:
            column_profiles: List of column profile dictionaries
            
        Returns:
            List of insight strings
        """
        try:
            # Prepare summary of profiles
            summary = {
                'total_columns': len(column_profiles),
                'data_types': {},
                'quality_issues': [],
                'high_null_columns': []
            }
            
            for profile in column_profiles:
                # Count data types
                dtype = profile.get('data_type', 'unknown')
                summary['data_types'][dtype] = summary['data_types'].get(dtype, 0) + 1
                
                # Collect quality issues
                issues = profile.get('quality_issues', [])
                summary['quality_issues'].extend(issues)
                
                # High null columns
                null_pct = profile.get('statistics', {}).get('null_percentage', 0)
                if null_pct > 20:
                    summary['high_null_columns'].append({
                        'column': profile.get('name'),
                        'null_percentage': null_pct
                    })
            
            # Generate insights
            insights_prompt = f"""
Based on the following data profile summary, provide 3-5 key insights about the dataset:

{json.dumps(summary, indent=2)}

Focus on:
1. Data quality observations
2. Potential data issues
3. Recommendations for data usage
4. Notable patterns or anomalies

Provide insights as a simple list of actionable statements.
"""
            
            response = await self.analyze_with_context(
                prompt=insights_prompt,
                context_data=summary
            )
            
            # Parse response into list
            insights = []
            for line in response.split('\n'):
                line = line.strip()
                if line and (line.startswith('-') or line.startswith('•') or line.startswith('*')):
                    insights.append(line.lstrip('-•* '))
                elif line and len(line) > 10:  # Likely an insight
                    insights.append(line)
            
            return insights[:5]  # Return top 5 insights
        
        except Exception as e:
            logger.error(f"Error extracting insights: {str(e)}")
            return ["Insight extraction failed due to technical issues"]
    
    async def close(self):
        """Close the client session"""
        if self.session:
            await self.session.close()
            self.session = None