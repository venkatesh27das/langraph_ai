"""
General Agent for handling general knowledge queries.
Uses LLM for broad questions not related to company-specific data.
"""

from typing import Dict, List, Any
import logging
from ..core.lm_studio_client import LMStudioClient

logger = logging.getLogger(__name__)

class GeneralAgent:
    """Agent for handling general knowledge queries"""
    
    def __init__(self, config: Dict[str, Any], llm_client: LMStudioClient):
        self.config = config
        self.llm_client = llm_client
        
    def process(self, user_input: str, conversation_history: List[Dict[str, str]]) -> str:
        """Process general knowledge query"""
        try:
            response = self._generate_response(user_input, conversation_history)
            return response
            
        except Exception as e:
            logger.error(f"Error in general processing: {str(e)}")
            return "I apologize, but I encountered an error while processing your question. Please try again."
    
    def _generate_response(self, user_input: str, history: List[Dict[str, str]]) -> str:
        """Generate response for general knowledge queries"""
        
        # Build conversation context
        conversation_context = ""
        if history:
            recent_history = history[-3:]  # Last 3 exchanges
            conversation_context = "\n".join([
                f"User: {h.get('user', '')}\nAssistant: {h.get('bot', '')}" 
                for h in recent_history
            ])
        
        prompt = f"""You are a helpful BI assistant. While your main expertise is in business intelligence and data analysis, you can also help with general questions.

Previous conversation:
{conversation_context}

User's question: {user_input}

Instructions:
1. Answer the user's question clearly and helpfully
2. If the question relates to BI, data analysis, or business concepts, provide detailed insights
3. For other general questions, give accurate and concise answers
4. If you're not sure about something, acknowledge the uncertainty
5. Keep your tone professional but friendly
6. If the question could relate to data analysis, suggest how it might be relevant to BI
7. Don't make up specific facts or statistics

Response:"""

        try:
            response = self.llm_client.generate(
                prompt, 
                max_tokens=self.config.get('max_response_tokens', 400),
                temperature=0.4
            )
            
            return response.strip()
            
        except Exception as e:
            logger.error(f"Error generating general response: {str(e)}")
            return "I'm having trouble generating a response right now. Could you please try rephrasing your question?"
    
    def is_bi_related(self, user_input: str) -> bool:
        """Check if the query might be BI-related for potential routing suggestions"""
        bi_keywords = [
            'data', 'analytics', 'report', 'dashboard', 'metric', 'kpi', 
            'business intelligence', 'visualization', 'chart', 'graph',
            'sales', 'revenue', 'profit', 'customer', 'performance',
            'trend', 'analysis', 'insight', 'database', 'query'
        ]
        
        user_lower = user_input.lower()
        return any(keyword in user_lower for keyword in bi_keywords)
    
    def suggest_bi_alternative(self, user_input: str) -> str:
        """Suggest how the query might be answered with BI data"""
        if not self.is_bi_related(user_input):
            return ""
        
        return "\n\nNote: If you're looking for specific data or metrics from your company's database, please let me know and I can help analyze that information for you."