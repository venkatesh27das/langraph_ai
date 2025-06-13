"""
RAG Agent for handling unstructured data queries.
Uses document retrieval and generation for answering questions.
"""

from typing import Dict, List, Any
import logging
from ..core.rag_pipeline import RAGPipeline
from ..core.lm_studio_client import LMStudioClient

logger = logging.getLogger(__name__)

class RAGAgent:
    """Agent for handling unstructured data queries using RAG"""
    
    def __init__(self, config: Dict[str, Any], llm_client: LMStudioClient):
        self.config = config
        self.llm_client = llm_client
        self.rag_pipeline = RAGPipeline(config['rag'])
        
    def process(self, user_input: str, conversation_history: List[Dict[str, str]]) -> Dict[str, Any]:
        """Process user query using RAG pipeline"""
        try:
            # Get relevant documents
            retrieved_docs = self.rag_pipeline.retrieve(user_input, top_k=5)
            
            if not retrieved_docs:
                return {
                    "response": "I couldn't find any relevant documents for your query. Please try rephrasing your question or check if the documents are properly indexed.",
                    "context": {"retrieved_docs": 0}
                }
            
            # Build context from retrieved documents
            context = self._build_context(retrieved_docs)
            
            # Generate response using LLM
            response = self._generate_response(user_input, context, conversation_history)
            
            return {
                "response": response,
                "context": {
                    "retrieved_docs": len(retrieved_docs),
                    "sources": [doc.get("source", "Unknown") for doc in retrieved_docs],
                    "snippets": [doc.get("content", "")[:200] + "..." for doc in retrieved_docs]
                }
            }
            
        except Exception as e:
            logger.error(f"Error in RAG processing: {str(e)}")
            return {
                "response": "I encountered an error while searching through the documents. Please try again.",
                "context": {"error": str(e)}
            }
    
    def _build_context(self, retrieved_docs: List[Dict[str, Any]]) -> str:
        """Build context string from retrieved documents"""
        context_parts = []
        
        for i, doc in enumerate(retrieved_docs, 1):
            source = doc.get("source", "Unknown")
            content = doc.get("content", "")
            score = doc.get("score", 0.0)
            
            context_part = f"""
Document {i} (Source: {source}, Relevance: {score:.2f}):
{content}
---
"""
            context_parts.append(context_part)
        
        return "\n".join(context_parts)
    
    def _generate_response(self, user_input: str, context: str, history: List[Dict[str, str]]) -> str:
        """Generate response using LLM with retrieved context"""
        
        # Build conversation context
        conversation_context = ""
        if history:
            recent_history = history[-3:]  # Last 3 exchanges
            conversation_context = "\n".join([
                f"User: {h.get('user', '')}\nAssistant: {h.get('bot', '')}" 
                for h in recent_history
            ])
        
        prompt = f"""You are a helpful BI assistant that answers questions based on company documents and data. 

Previous conversation:
{conversation_context}

User's current question: {user_input}

Relevant documents:
{context}

Instructions:
1. Answer the user's question based on the provided documents
2. Be specific and cite which documents you're referencing
3. If the documents don't contain enough information, say so clearly
4. Keep your response conversational but professional
5. If relevant, mention document sources
6. If the user is asking a follow-up question, connect it to the previous conversation

Response:"""

        try:
            response = self.llm_client.generate(
                prompt, 
                max_tokens=self.config.get('max_response_tokens', 500),
                temperature=0.3
            )
            
            return response.strip()
            
        except Exception as e:
            logger.error(f"Error generating RAG response: {str(e)}")
            return "I apologize, but I encountered an error while generating a response based on the documents."
    
    def add_documents(self, documents: List[Dict[str, Any]]) -> bool:
        """Add new documents to the RAG pipeline"""
        try:
            return self.rag_pipeline.add_documents(documents)
        except Exception as e:
            logger.error(f"Error adding documents: {str(e)}")
            return False
    
    def get_document_stats(self) -> Dict[str, Any]:
        """Get statistics about indexed documents"""
        try:
            return self.rag_pipeline.get_stats()
        except Exception as e:
            logger.error(f"Error getting document stats: {str(e)}")
            return {"error": str(e)}