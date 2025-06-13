"""
Clarification Agent for handling vague queries and asking follow-up questions.
Helps users refine their queries for better results.
"""

from typing import Dict, List, Any, Optional
import logging
from ..core.lm_studio_client import LMStudioClient

logger = logging.getLogger(__name__)

class ClarificationAgent:
    """Agent for handling vague queries and providing clarifications"""
    
    def __init__(self, config: Dict[str, Any], llm_client: LMStudioClient):
        self.config = config
        self.llm_client = llm_client
        self.vague_patterns = [
            'what', 'how', 'why', 'tell me about', 'show me', 'help', 
            'information', 'data', 'report', 'analysis'
        ]
        
    def process(self, user_input: str, conversation_history: List[Dict[str, str]], 
                clarification_questions: Optional[List[str]] = None) -> str:
        """Process vague query and generate clarification questions"""
        try:
            # Analyze the vague query
            analysis = self._analyze_vague_query(user_input, conversation_history)
            
            # Generate clarification response
            response = self._generate_clarification_response(
                user_input, 
                analysis, 
                conversation_history,
                clarification_questions
            )
            
            return response
            
        except Exception as e:
            logger.error(f"Error in clarification processing: {str(e)}")
            return "I'd like to help you better. Could you please provide more specific details about what you're looking for?"
    
    def _analyze_vague_query(self, user_input: str, history: List[Dict[str, str]]) -> Dict[str, Any]:
        """Analyze what makes the query vague and what clarifications are needed"""
        
        # Build context from conversation
        context = ""
        if history:
            context = "\n".join([f"User: {h.get('user', '')}\nBot: {h.get('bot', '')}" for h in history[-2:]])
        
        prompt = f"""Analyze this user query to understand what clarifications are needed.

Previous conversation:
{context}

Current user query: "{user_input}"

Analyze:
1. What type of information is the user likely looking for? (documents, data analysis, general knowledge)
2. What specific details are missing?
3. What clarification questions would help?

Respond in JSON format:
{{
    "likely_intent": "rag|sql|general",
    "missing_details": ["detail1", "detail2"],
    "suggested_questions": ["question1", "question2", "question3"],
    "vagueness_level": "high|medium|low",
    "context_available": true/false
}}
"""

        try:
            response = self.llm_client.generate(prompt, max_tokens=300, temperature=0.3)
            
            # Parse JSON response (simplified)
            import json
            try:
                analysis = json.loads(response.strip())
                return analysis
            except:
                # Fallback analysis
                return {
                    "likely_intent": self._guess_intent(user_input),
                    "missing_details": ["specific topic", "time period", "data type"],
                    "suggested_questions": [
                        "What specific information are you looking for?",
                        "Are you asking about company data or general information?",
                        "What time period are you interested in?"
                    ],
                    "vagueness_level": "high",
                    "context_available": len(history) > 0
                }
                
        except Exception as e:
            logger.error(f"Error analyzing vague query: {str(e)}")
            return {
                "likely_intent": "general",
                "missing_details": ["specifics"],
                "suggested_questions": ["Could you be more specific?"],
                "vagueness_level": "high",
                "context_available": False
            }
    
    def _guess_intent(self, user_input: str) -> str:
        """Simple intent guessing based on keywords"""
        user_lower = user_input.lower()
        
        if any(word in user_lower for word in ['document', 'report', 'file', 'pdf']):
            return 'rag'
        elif any(word in user_lower for word in ['data', 'sales', 'revenue', 'customer', 'analytics']):
            return 'sql'
        else:
            return 'general'
    
    def _generate_clarification_response(self, user_input: str, analysis: Dict[str, Any], 
                                       history: List[Dict[str, str]], 
                                       predefined_questions: Optional[List[str]] = None) -> str:
        """Generate helpful clarification response"""
        
        likely_intent = analysis.get('likely_intent', 'general')
        suggested_questions = predefined_questions or analysis.get('suggested_questions', [])
        
        # Build conversation context
        conversation_context = ""
        if history:
            conversation_context = "\n".join([
                f"User: {h.get('user', '')}\nAssistant: {h.get('bot', '')}" 
                for h in history[-2:]
            ])
        
        prompt = f"""You are a helpful BI assistant. The user has asked a vague question that needs clarification.

Previous conversation:
{conversation_context}

User's vague query: "{user_input}"

Analysis shows:
- Likely intent: {likely_intent}
- Missing details: {', '.join(analysis.get('missing_details', []))}

Suggested clarification questions:
{chr(10).join(f"- {q}" for q in suggested_questions[:3])}

Instructions:
1. Acknowledge the user's request warmly
2. Explain briefly why you need more information
3. Ask 2-3 specific clarification questions
4. Provide examples if helpful
5. Be encouraging and helpful
6. If you can guess the intent, mention what you can help with

Generate a friendly clarification response:"""

        try:
            response = self.llm_client.generate(
                prompt, 
                max_tokens=self.config.get('max_response_tokens', 300),
                temperature=0.4
            )
            
            return response.strip()
            
        except Exception as e:
            logger.error(f"Error generating clarification response: {str(e)}")
            
            # Fallback clarification response
            return self._generate_fallback_clarification(user_input, likely_intent)
    
    def _generate_fallback_clarification(self, user_input: str, likely_intent: str) -> str:
        """Generate a simple fallback clarification response"""
        
        base_response = "I'd be happy to help you with that! To give you the most relevant information, I need a bit more detail."
        
        if likely_intent == 'rag':
            specific_questions = [
                "What specific topic or document are you looking for?",
                "Are you looking for information from reports, policies, or other documents?"
            ]
        elif likely_intent == 'sql':
            specific_questions = [
                "What specific data or metrics are you interested in?",
                "Are you looking for sales data, customer information, or something else?",
                "What time period should I focus on?"
            ]
        else:
            specific_questions = [
                "What specific information are you looking for?",
                "Are you asking about company data or general information?"
            ]
        
        questions_text = "\n".join(f"• {q}" for q in specific_questions[:2])
        
        examples = self._get_example_queries(likely_intent)
        
        return f"""{base_response}

{questions_text}

{examples}"""
    
    def _get_example_queries(self, intent: str) -> str:
        """Get example queries for different intents"""
        
        examples = {
            'rag': """For example, you could ask:
• "What does our employee handbook say about vacation policies?"
• "Show me information about our Q3 performance report"
• "What are the key findings from the market research document?"
""",
            'sql': """For example, you could ask:
• "What were our total sales last month?"
• "Show me the top 5 customers by revenue"
• "How many new customers did we acquire this quarter?"
""",
            'general': """For example, you could ask:
• "What is business intelligence and how does it work?"
• "Explain the difference between KPIs and metrics"
• "What are best practices for data visualization?"
"""
        }
        
        return examples.get(intent, "Let me know what specific information you're looking for!")
    
    def is_query_vague(self, user_input: str) -> bool:
        """Determine if a query is too vague and needs clarification"""
        user_lower = user_input.lower().strip()
        
        # Very short queries
        if len(user_input.split()) <= 2:
            return True
        
        # Contains only very general terms
        general_only_patterns = [
            'what', 'how', 'why', 'tell me about', 'show me', 'help me with',
            'i want', 'i need', 'can you', 'give me', 'find', 'search'
        ]
        
        starts_with_general = any(user_lower.startswith(pattern) for pattern in general_only_patterns)
        
        # Check if it contains specific enough content
        specific_indicators = [
            'sales', 'revenue', 'customer', 'product', 'report', 'document',
            'last month', 'this year', 'quarterly', 'top', 'best', 'worst',
            'how many', 'how much', 'what is the', 'show me the'
        ]
        
        has_specifics = any(indicator in user_lower for indicator in specific_indicators)
        
        # Query is vague if it starts with general terms but lacks specifics
        return starts_with_general and not has_specifics
    
    def extract_intent_clues(self, user_input: str) -> Dict[str, Any]:
        """Extract clues about user intent from vague queries"""
        user_lower = user_input.lower()
        
        clues = {
            'keywords': [],
            'possible_intent': 'general',
            'confidence': 0.5
        }
        
        # Document/RAG keywords
        rag_keywords = ['document', 'report', 'file', 'policy', 'handbook', 'paper', 'study']
        found_rag = [kw for kw in rag_keywords if kw in user_lower]
        
        # Data/SQL keywords  
        sql_keywords = ['data', 'sales', 'revenue', 'customer', 'metric', 'analytics', 'number']
        found_sql = [kw for kw in sql_keywords if kw in user_lower]
        
        if found_rag:
            clues['possible_intent'] = 'rag'
            clues['keywords'] = found_rag
            clues['confidence'] = 0.7
        elif found_sql:
            clues['possible_intent'] = 'sql'
            clues['keywords'] = found_sql
            clues['confidence'] = 0.7
        
        return clues