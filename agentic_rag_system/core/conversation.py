from typing import List, Dict, Any, Optional
from datetime import datetime
import json

class ConversationHandler:
    """Handles multi-turn conversations and context management"""
    
    def __init__(self, max_history: int = 10):
        self.max_history = max_history
        self.conversation_history: List[Dict[str, Any]] = []
        self.context_memory: Dict[str, Any] = {}
        self.session_id = self._generate_session_id()
    
    def _generate_session_id(self) -> str:
        """Generate a unique session ID"""
        return f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    def add_message(self, role: str, content: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Add a message to conversation history"""
        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {}
        }
        
        self.conversation_history.append(message)
        
        # Keep only the most recent messages
        if len(self.conversation_history) > self.max_history * 2:  # *2 for user+assistant pairs
            self.conversation_history = self.conversation_history[-(self.max_history * 2):]
    
    def get_conversation_context(self) -> str:
        """Get formatted conversation context for the agent"""
        if not self.conversation_history:
            return ""
        
        context_parts = []
        for msg in self.conversation_history[-6:]:  # Last 3 exchanges
            role = msg["role"].upper()
            content = msg["content"]
            context_parts.append(f"{role}: {content}")
        
        return "\n".join(context_parts)
    
    def get_messages_for_llm(self) -> List[Dict[str, str]]:
        """Get messages formatted for LLM consumption"""
        messages = []
        
        # Add recent conversation history
        for msg in self.conversation_history[-8:]:  # Last 4 exchanges
            messages.append({
                "role": msg["role"],
                "content": msg["content"]
            })
        
        return messages
    
    def update_context_memory(self, key: str, value: Any) -> None:
        """Update context memory with key-value pairs"""
        self.context_memory[key] = value
    
    def get_context_memory(self, key: str) -> Any:
        """Retrieve value from context memory"""
        return self.context_memory.get(key)
    
    def get_recent_user_queries(self, count: int = 3) -> List[str]:
        """Get recent user queries for context"""
        user_messages = [
            msg["content"] for msg in self.conversation_history 
            if msg["role"] == "user"
        ]
        return user_messages[-count:]
    
    def has_context_for_clarification(self) -> bool:
        """Check if there's enough context to ask clarifying questions"""
        user_messages = [msg for msg in self.conversation_history if msg["role"] == "user"]
        return len(user_messages) > 0
    
    def extract_entities_and_topics(self) -> Dict[str, List[str]]:
        """Extract entities and topics from conversation for context"""
        entities = set()
        topics = set()
        
        for msg in self.conversation_history:
            if msg["role"] == "user":
                content = msg["content"].lower()
                
                # Simple entity extraction (can be enhanced)
                words = content.split()
                
                # Look for potential entities (capitalized words, technical terms)
                for word in words:
                    if len(word) > 3 and (word[0].isupper() or word in self.context_memory.get("domain_terms", [])):
                        entities.add(word)
                
                # Look for question words to identify topics
                question_indicators = ["what", "how", "why", "when", "where", "which", "who"]
                for indicator in question_indicators:
                    if indicator in content:
                        topics.add(indicator)
        
        return {
            "entities": list(entities),
            "topics": list(topics)
        }
    
    def is_follow_up_question(self, current_query: str) -> bool:
        """Determine if current query is a follow-up to previous questions"""
        if not self.conversation_history:
            return False
        
        # Simple heuristics for follow-up detection
        follow_up_indicators = [
            "also", "additionally", "furthermore", "moreover", "besides",
            "what about", "how about", "can you also", "tell me more",
            "explain further", "elaborate", "continue", "more details"
        ]
        
        current_lower = current_query.lower()
        
        # Check for follow-up indicators
        for indicator in follow_up_indicators:
            if indicator in current_lower:
                return True
        
        # Check for pronouns that might refer to previous context
        pronouns = ["it", "this", "that", "they", "them", "these", "those"]
        for pronoun in pronouns:
            if f" {pronoun} " in f" {current_lower} ":
                return True
        
        return False
    
    def get_conversation_summary(self) -> str:
        """Generate a summary of the conversation"""
        if not self.conversation_history:
            return "No conversation history available."
        
        user_messages = [msg["content"] for msg in self.conversation_history if msg["role"] == "user"]
        assistant_messages = [msg["content"] for msg in self.conversation_history if msg["role"] == "assistant"]
        
        summary = f"Conversation Summary:\n"
        summary += f"- Total exchanges: {len(user_messages)}\n"
        summary += f"- Session ID: {self.session_id}\n"
        
        if user_messages:
            summary += f"- Recent topics: {', '.join(user_messages[-3:])}\n"
        
        return summary
    
    def clear_history(self) -> None:
        """Clear conversation history"""
        self.conversation_history.clear()
        self.context_memory.clear()
        self.session_id = self._generate_session_id()
    
    def export_conversation(self) -> Dict[str, Any]:
        """Export conversation for analysis or storage"""
        return {
            "session_id": self.session_id,
            "history": self.conversation_history,
            "context_memory": self.context_memory,
            "created_at": datetime.now().isoformat()
        }
    
    def import_conversation(self, data: Dict[str, Any]) -> bool:
        """Import conversation from exported data"""
        try:
            self.session_id = data.get("session_id", self.session_id)
            self.conversation_history = data.get("history", [])
            self.context_memory = data.get("context_memory", {})
            return True
        except Exception as e:
            print(f"Error importing conversation: {e}")
            return False