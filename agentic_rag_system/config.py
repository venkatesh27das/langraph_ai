import os
from typing import Dict, Any

class Config:
    """Configuration settings for the Agentic RAG System"""
    
    # LMStudio Configuration
    LMSTUDIO_BASE_URL = os.getenv("LMSTUDIO_BASE_URL", "http://localhost:1234")
    LMSTUDIO_API_KEY = os.getenv("LMSTUDIO_API_KEY", "lm-studio")
    
    # Model Configuration
    CHAT_MODEL = os.getenv("CHAT_MODEL", "qwen/qwen3-4b")
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-nomic-embed-text-v1.5")
    
    # ChromaDB Configuration
    CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "./data/chroma_db")
    COLLECTION_NAME = os.getenv("COLLECTION_NAME", "rag_documents")
    
    # RAG Configuration
    TOP_K_RESULTS = int(os.getenv("TOP_K_RESULTS", "5"))
    SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.7"))
    
    # Agent Configuration
    MAX_ITERATIONS = int(os.getenv("MAX_ITERATIONS", "5"))
    TEMPERATURE = float(os.getenv("TEMPERATURE", "0.7"))
    MAX_TOKENS = int(os.getenv("MAX_TOKENS", "2048"))
    
    # Document Processing
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))

    # Clarification Control Settings
    ENABLE_CLARIFICATION = True  # Master switch to enable/disable clarifications
    CLARIFICATION_THRESHOLD = 0.7  # Similarity distance threshold for poor results
    MIN_QUERY_LENGTH = 2  # Minimum words before clarification
    MAX_CLARIFICATIONS_PER_SESSION = 3  # Limit clarifications per conversation
    
    # Response Quality Settings
    MIN_RESPONSE_CONFIDENCE = 0.3  # Minimum confidence for direct response
    REQUIRE_MULTIPLE_SOURCES = False  # Whether to require multiple document sources
    
    # Debug Settings
    DEBUG_CLARIFICATION = True  # Show clarification decision reasons
    LOG_RETRIEVAL_SCORES = False  # Log document similarity scores

     # Thinking Model Configuration
    USE_THINKING_MODEL = True  # Set to False to disable thinking model features
    THINKING_MODEL_PROMPT_ADDITION = "Think through this step by step, then provide your final answer."
    
    # Thinking Model Debug Settings
    SAVE_RAW_RESPONSES = False  # Set to True to save raw responses with thinking content for debugging
    DEBUG_THINKING_CONTENT = False  # Set to True to print thinking content to console for debugging
    
    # Advanced Thinking Model Settings
    THINKING_MODEL_TEMPERATURE = 0.7  # Temperature for thinking model reasoning
    THINKING_MODEL_MAX_TOKENS = 3000   # Higher token limit for thinking models
    
    # Fallback behavior when thinking content removal fails
    FALLBACK_ON_THINKING_FAILURE = True  # Return cleaned response even if thinking removal partially fails
    
    # Paths
    DOCUMENTS_PATH = os.getenv("DOCUMENTS_PATH", "./data/documents")
    
    @classmethod
    def to_dict(cls) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {
            key: value for key, value in cls.__dict__.items()
            if not key.startswith('_') and not callable(value)
        }
    
    @classmethod
    def validate(cls) -> bool:
        """Validate configuration settings"""
        try:
            # Check if required directories exist
            os.makedirs(cls.DOCUMENTS_PATH, exist_ok=True)
            os.makedirs(cls.CHROMA_DB_PATH, exist_ok=True)
            return True
        except Exception as e:
            print(f"Configuration validation failed: {e}")
            return False