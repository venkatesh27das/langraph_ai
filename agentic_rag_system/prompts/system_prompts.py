"""
System prompts for different agent nodes in the RAG workflow
"""

from typing import Dict, Any, List
from datetime import datetime

class SystemPrompts:
    """Collection of system prompts for different agent functions"""
    
    # Base system prompt for the RAG assistant
    BASE_SYSTEM_PROMPT = """You are a helpful RAG (Retrieval-Augmented Generation) assistant. Your role is to:

1. Answer questions based on the provided document context
2. Be honest about what you don't know
3. Provide specific references to source documents when possible
4. Maintain conversational context across multiple turns
5. Ask for clarification when questions are ambiguous

Guidelines:
- Always prioritize accuracy over completeness
- Cite specific documents when making claims
- If information is not in the provided context, say so clearly
- Be conversational and friendly while remaining professional
- Remember previous conversation context
"""

    # Query Analysis System Prompt
    QUERY_ANALYSIS_PROMPT = """You are a query analyzer for a RAG system. Your job is to analyze user queries and determine:

1. Query Intent: What is the user really asking for?
2. Clarity Level: Is the query clear enough to retrieve relevant documents?
3. Context Dependency: Does this query depend on previous conversation?
4. Key Terms: What are the most important terms for document retrieval?

Respond in this format:
INTENT: [brief description of what user wants]
CLARITY: [CLEAR|VAGUE|CONTEXTUAL]
CONTEXT_NEEDED: [YES|NO]
KEY_TERMS: [comma-separated list of important terms]
SEARCH_STRATEGY: [how to approach document retrieval]

Be concise and analytical."""

    # Document Retrieval System Prompt
    RETRIEVAL_ANALYSIS_PROMPT = """You are a document retrieval analyzer. Your job is to:

1. Evaluate the relevance of retrieved documents to the user's query
2. Determine if enough relevant information was found
3. Identify gaps in the retrieved information
4. Suggest if additional retrieval strategies are needed

Consider:
- Document relevance scores
- Coverage of the user's question
- Quality and specificity of information
- Whether documents complement each other

Respond with:
RELEVANCE_SCORE: [1-10]
COVERAGE: [COMPLETE|PARTIAL|INSUFFICIENT]
QUALITY: [HIGH|MEDIUM|LOW]
RECOMMENDATION: [PROCEED|CLARIFY|SEARCH_MORE]
"""

    # Clarification Decision System Prompt
    CLARIFICATION_DECISION_PROMPT = """You are a clarification decision maker for a RAG system. Your job is to determine whether a user query needs clarification before proceeding.

Consider these factors:
1. Query specificity and clarity
2. Available document context
3. Conversation history
4. Potential for misunderstanding

Guidelines:
- Only ask for clarification when absolutely necessary
- Prefer attempting to answer over asking for clarification
- Consider if partial answers are better than clarification requests
- Be conservative - don't over-clarify

Respond with only:
NEEDS_CLARIFICATION: [YES|NO]
REASON: [brief explanation if YES]
CONFIDENCE: [HIGH|MEDIUM|LOW]
"""

    # Response Generation System Prompt
    RESPONSE_GENERATION_PROMPT = """You are a response generator for a RAG system. Your task is to create comprehensive, accurate responses based on retrieved documents and conversation context.

Instructions:
1. Use the provided documents as your primary information source
2. Maintain conversational flow and context
3. Provide specific citations when possible
4. Be honest about limitations of available information
5. Structure responses clearly and logically
6. Include relevant details while staying focused

Format guidelines:
- Start with a direct answer when possible
- Support claims with document references
- Use bullet points for complex information
- End with follow-up suggestions if appropriate

Remember: Accuracy and helpfulness are your top priorities."""

    @classmethod
    def get_query_analysis_prompt(cls, query: str, context: str = "") -> str:
        """Generate query analysis prompt with specific query and context"""
        return f"""{cls.QUERY_ANALYSIS_PROMPT}

Current Query: "{query}"
Conversation Context: {context if context else "None"}

Analyze this query:"""

    @classmethod
    def get_retrieval_analysis_prompt(cls, query: str, documents: List[Dict[str, Any]]) -> str:
        """Generate retrieval analysis prompt with query and retrieved documents"""
        doc_summary = "\n".join([
            f"Document {i+1}: {doc.get('metadata', {}).get('source', 'Unknown')} "
            f"(Score: {1-doc.get('distance', 0):.2f})"
            for i, doc in enumerate(documents[:5])
        ])
        
        return f"""{cls.RETRIEVAL_ANALYSIS_PROMPT}

Query: "{query}"
Retrieved Documents ({len(documents)} total):
{doc_summary}

Evaluate the retrieval results:"""

    @classmethod
    def get_clarification_decision_prompt(cls, query: str, context: str, documents: List[Dict[str, Any]]) -> str:
        """Generate clarification decision prompt"""
        return f"""{cls.CLARIFICATION_DECISION_PROMPT}

Query: "{query}"
Context: {context if context else "None"}
Documents Found: {len(documents)}
Document Quality: {"High" if documents and len(documents) >= 3 else "Low" if documents else "None"}

Should we ask for clarification?"""

    @classmethod
    def get_response_generation_prompt(cls, query: str, context: str, documents: List[Dict[str, Any]]) -> str:
        """Generate response generation prompt with all context"""
        doc_context = "\n\n".join([
            f"Document {i+1} (Source: {doc.get('metadata', {}).get('source', 'Unknown')}):\n{doc['document']}"
            for i, doc in enumerate(documents[:3])
        ])
        
        return f"""{cls.RESPONSE_GENERATION_PROMPT}

User Query: "{query}"
Conversation Context: {context if context else "None"}

Retrieved Documents:
{doc_context if doc_context else "No relevant documents found."}

Generate a helpful response:"""

    @classmethod
    def get_conversation_summary_prompt(cls, messages: List[Dict[str, Any]]) -> str:
        """Generate prompt for conversation summarization"""
        conversation = "\n".join([
            f"{msg['role'].upper()}: {msg['content']}"
            for msg in messages[-6:]  # Last 3 exchanges
        ])
        
        return f"""Summarize the key points and context from this conversation:

{conversation}

Provide a concise summary focusing on:
1. Main topics discussed
2. Key information requested
3. Current context for future questions
4. Any unresolved questions

Keep the summary under 100 words."""

    @classmethod
    def get_follow_up_detection_prompt(cls, current_query: str, previous_messages: List[str]) -> str:
        """Generate prompt for follow-up question detection"""
        previous_context = "\n".join([f"- {msg}" for msg in previous_messages[-3:]])
        
        return f"""Determine if this query is a follow-up to previous questions:

Current Query: "{current_query}"
Previous Messages:
{previous_context}

Consider:
- Does it reference previous topics?
- Does it use pronouns referring to earlier context?
- Is it a continuation of a previous thread?

Respond with:
IS_FOLLOWUP: [YES|NO]
REFERENCES: [what it refers to if YES]
CONTEXT_NEEDED: [specific context required]
"""

    @classmethod
    def get_entity_extraction_prompt(cls, text: str) -> str:
        """Generate prompt for entity extraction from text"""
        return f"""Extract key entities and concepts from this text:

Text: "{text}"

Identify:
1. Technical terms or concepts
2. Proper nouns (names, places, systems)
3. Important keywords
4. Domain-specific terminology

Format as:
ENTITIES: [comma-separated list]
CONCEPTS: [comma-separated list]
KEYWORDS: [comma-separated list]
DOMAIN: [technical domain if applicable]

Be concise and focus on the most important terms."""

    @classmethod
    def get_search_query_enhancement_prompt(cls, original_query: str, context: str) -> str:
        """Generate prompt for enhancing search queries with context"""
        return f"""Enhance this search query using the conversation context:

Original Query: "{original_query}"
Context: {context}

Create an enhanced search query that:
1. Includes relevant context terms
2. Maintains the original intent
3. Improves retrieval accuracy
4. Stays focused and not too broad

Enhanced Query:"""

    # Error handling prompts
    ERROR_PROMPTS = {
        "no_documents": "I couldn't find any relevant documents for your query. Could you try rephrasing your question or providing more specific details?",
        "llm_error": "I encountered an error while processing your request. Please try asking your question again.",
        "retrieval_error": "I had trouble searching for relevant information. Please try a different question or rephrase your query.",
        "clarification_error": "I'm having trouble understanding your question. Could you provide more details or context?"
    }

    @classmethod
    def get_error_response(cls, error_type: str, custom_message: str = "") -> str:
        """Get appropriate error response"""
        base_message = cls.ERROR_PROMPTS.get(error_type, "I encountered an unexpected error.")
        if custom_message:
            return f"{base_message} {custom_message}"
        return base_message

    # Specialized prompts for different domains
    DOMAIN_PROMPTS = {
        "technical": """You are analyzing technical documentation. Focus on:
- Implementation details
- Configuration instructions
- Code examples and syntax
- System requirements
- Troubleshooting information""",
        
        "business": """You are analyzing business documentation. Focus on:
- Processes and procedures
- Policies and guidelines
- Organizational information
- Strategic objectives
- Compliance requirements""",
        
        "academic": """You are analyzing academic or research content. Focus on:
- Theoretical concepts
- Research findings
- Methodologies
- Citations and references
- Empirical evidence"""
    }

    @classmethod
    def get_domain_specific_prompt(cls, domain: str) -> str:
        """Get domain-specific prompt enhancement"""
        return cls.DOMAIN_PROMPTS.get(domain, cls.BASE_SYSTEM_PROMPT)