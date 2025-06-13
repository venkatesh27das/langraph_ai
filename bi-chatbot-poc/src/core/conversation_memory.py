"""
Conversation memory management for multi-turn conversations.
Handles session storage and conversation context.
"""

import json
import os
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from collections import defaultdict

logger = logging.getLogger(__name__)

class ConversationMemory:
    """Manages conversation history and context for multi-turn interactions"""
    
    def __init__(self, storage_path: str = "data/conversations", max_sessions: int = 100):
        self.storage_path = storage_path
        self.max_sessions = max_sessions
        self.max_history_length = 20  # Maximum turns per conversation
        self.session_timeout = timedelta(hours=24)  # Sessions expire after 24 hours
        
        # In-memory storage for active sessions
        self.sessions: Dict[str, Dict[str, Any]] = {}
        
        # Ensure storage directory exists
        os.makedirs(storage_path, exist_ok=True)
        
        # Load existing sessions
        self._load_sessions()
        
        logger.info(f"Initialized conversation memory with {len(self.sessions)} sessions")
    
    def _load_sessions(self):
        """Load existing sessions from storage"""
        try:
            for filename in os.listdir(self.storage_path):
                if filename.endswith('.json'):
                    session_id = filename[:-5]  # Remove .json extension
                    file_path = os.path.join(self.storage_path, filename)
                    
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            session_data = json.load(f)
                        
                        # Check if session is not expired
                        last_activity = datetime.fromisoformat(session_data.get('last_activity', ''))
                        if datetime.now() - last_activity < self.session_timeout:
                            self.sessions[session_id] = session_data
                        else:
                            # Remove expired session file
                            os.remove(file_path)
                            
                    except Exception as e:
                        logger.warning(f"Error loading session {session_id}: {str(e)}")
                        
        except Exception as e:
            logger.error(f"Error loading sessions: {str(e)}")
    
    def _save_session(self, session_id: str):
        """Save session to persistent storage"""
        try:
            if session_id in self.sessions:
                file_path = os.path.join(self.storage_path, f"{session_id}.json")
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(self.sessions[session_id], f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Error saving session {session_id}: {str(e)}")
    
    def create_session(self, session_id: str, user_info: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Create a new conversation session"""
        session_data = {
            'session_id': session_id,
            'created_at': datetime.now().isoformat(),
            'last_activity': datetime.now().isoformat(),
            'conversation_history': [],
            'context': {},
            'user_info': user_info or {},
            'session_stats': {
                'total_messages': 0,
                'user_messages': 0,
                'bot_messages': 0,
                'intents': defaultdict(int)
            }
        }
        
        self.sessions[session_id] = session_data
        self._save_session(session_id)
        
        logger.info(f"Created new session: {session_id}")
        return session_data
    
    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get session data"""
        if session_id in self.sessions:
            # Update last activity
            self.sessions[session_id]['last_activity'] = datetime.now().isoformat()
            return self.sessions[session_id]
        return None
    
    def get_conversation(self, session_id: str) -> List[Dict[str, Any]]:
        """Get conversation history for a session"""
        session = self.get_session(session_id)
        if session:
            return session.get('conversation_history', [])
        return []
    
    def add_message(self, 
                   session_id: str, 
                   role: str, 
                   content: str, 
                   intent: Optional[str] = None,
                   metadata: Optional[Dict[str, Any]] = None):
        """Add a message to conversation history"""
        
        # Create session if it doesn't exist
        if session_id not in self.sessions:
            self.create_session(session_id)
        
        session = self.sessions[session_id]
        
        message = {
            'role': role,
            'content': content,
            'timestamp': datetime.now().isoformat(),
            'intent': intent,
            'metadata': metadata or {}
        }
        
        # Add to conversation history
        session['conversation_history'].append(message)
        
        # Trim history if too long
        if len(session['conversation_history']) > self.max_history_length:
            session['conversation_history'] = session['conversation_history'][-self.max_history_length:]
        
        # Update session stats
        session['session_stats']['total_messages'] += 1
        if role == 'user':
            session['session_stats']['user_messages'] += 1
        elif role == 'assistant':
            session['session_stats']['bot_messages'] += 1
        
        if intent:
            session['session_stats']['intents'][intent] += 1
        
        # Update last activity
        session['last_activity'] = datetime.now().isoformat()
        
        # Save session
        self._save_session(session_id)
        
        logger.debug(f"Added {role} message to session {session_id}")
    
    def update_conversation(self, session_id: str, conversation_history: List[Dict[str, Any]]):
        """Update entire conversation history"""
        if session_id not in self.sessions:
            self.create_session(session_id)
        
        self.sessions[session_id]['conversation_history'] = conversation_history[-self.max_history_length:]
        self.sessions[session_id]['last_activity'] = datetime.now().isoformat()
        
        # Update stats
        stats = self.sessions[session_id]['session_stats']
        stats['total_messages'] = len(conversation_history)
        stats['user_messages'] = sum(1 for msg in conversation_history if msg.get('role') == 'user')
        stats['bot_messages'] = sum(1 for msg in conversation_history if msg.get('role') == 'assistant')
        
        # Count intents
        intents = defaultdict(int)
        for msg in conversation_history:
            if msg.get('intent'):
                intents[msg['intent']] += 1
        stats['intents'] = dict(intents)
        
        self._save_session(session_id)
    
    def get_context(self, session_id: str) -> Dict[str, Any]:
        """Get conversation context for a session"""
        session = self.get_session(session_id)
        if session:
            return session.get('context', {})
        return {}
    
    def update_context(self, session_id: str, context_updates: Dict[str, Any]):
        """Update conversation context"""
        if session_id not in self.sessions:
            self.create_session(session_id)
        
        session = self.sessions[session_id]
        session['context'].update(context_updates)
        session['last_activity'] = datetime.now().isoformat()
        
        self._save_session(session_id)
    
    def get_recent_messages(self, session_id: str, count: int = 5) -> List[Dict[str, Any]]:
        """Get recent messages from conversation"""
        conversation = self.get_conversation(session_id)
        return conversation[-count:] if conversation else []
    
    def get_conversation_summary(self, session_id: str) -> Dict[str, Any]:
        """Get summary of conversation"""
        session = self.get_session(session_id)
        if not session:
            return {}
        
        conversation = session.get('conversation_history', [])
        stats = session.get('session_stats', {})
        
        # Get most common intent
        intents = stats.get('intents', {})
        primary_intent = max(intents.items(), key=lambda x: x[1])[0] if intents else None
        
        # Get recent topics (simple keyword extraction)
        recent_content = ' '.join([msg['content'] for msg in conversation[-5:]])
        
        return {
            'session_id': session_id,
            'message_count': stats.get('total_messages', 0),
            'primary_intent': primary_intent,
            'created_at': session.get('created_at'),
            'last_activity': session.get('last_activity'),
            'recent_topics': recent_content[:200] + '...' if len(recent_content) > 200 else recent_content
        }
    
    def clear_conversation(self, session_id: str):
        """Clear conversation history for a session"""
        if session_id in self.sessions:
            self.sessions[session_id]['conversation_history'] = []
            self.sessions[session_id]['context'] = {}
            self.sessions[session_id]['session_stats'] = {
                'total_messages': 0,
                'user_messages': 0,
                'bot_messages': 0,
                'intents': defaultdict(int)
            }
            self.sessions[session_id]['last_activity'] = datetime.now().isoformat()
            self._save_session(session_id)
            
            logger.info(f"Cleared conversation for session: {session_id}")
    
    def delete_session(self, session_id: str):
        """Delete a session completely"""
        if session_id in self.sessions:
            del self.sessions[session_id]
            
            # Remove file
            file_path = os.path.join(self.storage_path, f"{session_id}.json")
            if os.path.exists(file_path):
                os.remove(file_path)
            
            logger.info(f"Deleted session: {session_id}")
    
    def cleanup_expired_sessions(self):
        """Remove expired sessions"""
        current_time = datetime.now()
        expired_sessions = []
        
        for session_id, session_data in self.sessions.items():
            last_activity = datetime.fromisoformat(session_data.get('last_activity', ''))
            if current_time - last_activity > self.session_timeout:
                expired_sessions.append(session_id)
        
        for session_id in expired_sessions:
            self.delete_session(session_id)
        
        logger.info(f"Cleaned up {len(expired_sessions)} expired sessions")
    
    def get_all_sessions(self) -> List[Dict[str, Any]]:
        """Get summary of all active sessions"""
        return [self.get_conversation_summary(session_id) for session_id in self.sessions.keys()]
    
    def get_timestamp(self) -> str:
        """Get current timestamp in ISO format"""
        return datetime.now().isoformat()