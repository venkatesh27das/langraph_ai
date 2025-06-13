"""
Clarification templates and strategies for the Agentic RAG System
"""

from typing import Dict, List, Any, Optional
import random

class ClarificationTemplates:
    """Templates and strategies for generating clarification questions"""
    
    # Base clarification templates
    BASE_TEMPLATES = {
        "vague_query": [
            "I'd be happy to help! Could you provide more specific details about what you're looking for?",
            "To give you the most accurate answer, could you clarify what specific aspect interests you?",
            "I want to make sure I understand correctly. Could you elaborate on what you'd like to know?",
        ],
        
        "multiple_topics": [
            "I found information on several related topics. Which specific area would you like me to focus on?",
            "Your question touches on multiple areas. Could you help me understand your main priority?",
            "I can help with several aspects of this topic. What's your primary interest?",
        ],
        
        "ambiguous_context": [
            "I want to give you the most relevant information. Could you provide more context about your situation?",
            "To better assist you, could you share more details about what you're trying to accomplish?",
            "Understanding your specific use case would help me provide a more targeted answer.",
        ],
        
        "insufficient_information": [
            "I found limited information on this topic. Could you try rephrasing your question or provide additional context?",
            "The available information seems incomplete. Could you help me understand what specific details you need?",
            "I'd like to help, but I need more information to give you a comprehensive answer.",
        ]
    }
    
    # Context-specific clarification templates
    CONTEXT_TEMPLATES = {
        "technical": {
            "implementation": [
                "Are you looking for implementation details, configuration steps, or conceptual overview?",
                "Which programming language or technology stack are you working with?",
                "Are you seeking troubleshooting help or implementation guidance?",
            ],
            "configuration": [
                "Which specific configuration aspect would you like me to focus on?",
                "Are you looking for initial setup or advanced configuration options?",
                "What's your target environment or use case?",
            ],
            "troubleshooting": [
                "What specific issue or error are you encountering?",
                "What steps have you already tried?",
                "What's your current setup or configuration?",
            ]
        },
        
        "process": {
            "workflow": [
                "Which part of the process would you like me to explain?",
                "Are you looking for a high-level overview or detailed steps?",
                "What's your role or perspective in this process?",
            ],
            "procedure": [
                "Which specific procedure are you interested in?",
                "Do you need the complete process or just certain steps?",
                "What's your current situation or starting point?",
            ]
        },
        
        "conceptual": {
            "explanation": [
                "Would you like a basic explanation or more detailed technical information?",
                "Which aspect of this concept interests you most?",
                "Are you looking for examples or theoretical understanding?",
            ],
            "comparison": [
                "What would you like me to compare this with?",
                "Which specific features or aspects should I focus on?",
                "What's your evaluation criteria or use case?",
            ]
        }
    }
    
    # Follow-up question templates
    FOLLOW_UP_TEMPLATES = {
        "clarify_scope": [
            "Just to clarify, are you asking about {topic} in general or {specific_aspect}?",
            "When you mention {term}, do you mean {interpretation1} or {interpretation2}?",
            "Are you looking for information about {scope1} or would you like me to cover {scope2} as well?",
        ],
        
        "confirm_understanding": [
            "Let me make sure I understand: you want to know about {summary}. Is that correct?",
            "So you're specifically interested in {focus_area}, right?",
            "Am I understanding correctly that you need help with {task}?",
        ],
        
        "offer_alternatives": [
            "I can help you with {option1}, {option2}, or {option3}. Which would be most useful?",
            "Would you prefer information about {approach1} or {approach2}?",
            "I could focus on {aspect1} or {aspect2}. What would be more helpful?",
        ]
    }
    
    # Question type identification and templates
    QUESTION_TYPES = {
        "what": {
            "clarifications": [
                "What specific aspect of {topic} would you like me to explain?",
                "Are you looking for a definition, explanation, or examples of {topic}?",
                "What level of detail would be most helpful for you?",
            ]
        },
        
        "how": {
            "clarifications": [
                "Are you looking for step-by-step instructions or a general approach?",
                "What's your current level of experience with {topic}?",
                "Are you trying to {action1} or {action2}?",
            ]
        },
        
        "why": {
            "clarifications": [
                "Are you interested in the technical reasons or business rationale?",
                "Would you like me to explain the background or focus on current implications?",
                "Are you looking for causes, benefits, or decision factors?",
            ]
        },
        
        "when": {
            "clarifications": [
                "Are you asking about timing, scheduling, or historical context?",
                "Do you need information about past events or future planning?",
                "Are you looking for deadlines or general timeframes?",
            ]
        },
        
        "where": {
            "clarifications": [
                "Are you looking for physical locations or system/process locations?",
                "Do you need configuration locations or documentation sources?",
                "Are you asking about organizational or technical placement?",
            ]
        }
    }

    @classmethod
    def generate_clarification(cls, query: str, context: str = "", 
                             clarification_type: str = "vague_query",
                             retrieved_docs: List[Dict[str, Any]] = None) -> str:
        """Generate appropriate clarification question based on context"""
        
        # Analyze query for better clarification
        query_analysis = cls._analyze_query_for_clarification(query, context, retrieved_docs)
        
        # Get base template
        templates = cls.BASE_TEMPLATES.get(clarification_type, cls.BASE_TEMPLATES["vague_query"])
        base_clarification = random.choice(templates)
        
        # Enhance with context-specific information
        enhanced_clarification = cls._enhance_clarification(
            base_clarification, query_analysis, retrieved_docs
        )
        
        return enhanced_clarification

    @classmethod
    def _analyze_query_for_clarification(cls, query: str, context: str, 
                                       retrieved_docs: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Analyze query to determine best clarification strategy"""
        analysis = {
            "query_length": len(query.split()),
            "question_words": [],
            "domain_indicators": [],
            "ambiguous_terms": [],
            "potential_topics": []
        }
        
        # Identify question words
        question_words = ["what", "how", "why", "when", "where", "which", "who"]
        for word in question_words:
            if word in query.lower():
                analysis["question_words"].append(word)
        
        # Identify domain indicators
        technical_indicators = ["configure", "implement", "install", "setup", "debug", "code", "api"]
        process_indicators = ["process", "workflow", "procedure", "steps", "method"]
        conceptual_indicators = ["explain", "understand", "concept", "theory", "principle"]
        
        query_lower = query.lower()
        if any(indicator in query_lower for indicator in technical_indicators):
            analysis["domain_indicators"].append("technical")
        if any(indicator in query_lower for indicator in process_indicators):
            analysis["domain_indicators"].append("process")
        if any(indicator in query_lower for indicator in conceptual_indicators):
            analysis["domain_indicators"].append("conceptual")
        
        # Identify potential topics from retrieved documents
        if retrieved_docs:
            for doc in retrieved_docs[:3]:
                source = doc.get("metadata", {}).get("source", "")
                if source:
                    analysis["potential_topics"].append(source)
        
        return analysis

    @classmethod
    def _enhance_clarification(cls, base_clarification: str, analysis: Dict[str, Any], 
                             retrieved_docs: List[Dict[str, Any]] = None) -> str:
        """Enhance clarification with specific context"""
        
        # Add topic-specific options if documents are available
        if retrieved_docs and len(retrieved_docs) > 1:
            topics = []
            for doc in retrieved_docs[:3]:
                source = doc.get("metadata", {}).get("source", "")
                if source and source not in topics:
                    topics.append(source.replace(".txt", "").replace(".md", "").replace("_", " "))
            
            if topics:
                topic_options = ", ".join(topics[:-1]) + f", or {topics[-1]}" if len(topics) > 1 else topics[0]
                enhanced = f"{base_clarification}\n\nI found information related to: {topic_options}. Which area interests you most?"
                return enhanced
        
        # Add question-type specific clarification
        if analysis["question_words"]:
            primary_question = analysis["question_words"][0]
            if primary_question in cls.QUESTION_TYPES:
                specific_clarifications = cls.QUESTION_TYPES[primary_question]["clarifications"]
                additional_clarification = random.choice(specific_clarifications)
                return f"{base_clarification}\n\n{additional_clarification.replace('{topic}', 'this topic')}"
        
        return base_clarification

    @classmethod
    def get_progressive_clarification(cls, attempt_number: int, query: str, 
                                    previous_clarifications: List[str] = None) -> str:
        """Generate progressive clarification that becomes more specific"""
        
        if attempt_number == 1:
            return cls.generate_clarification(query, clarification_type="vague_query")
        elif attempt_number == 2:
            return cls._generate_specific_clarification(query)
        else:
            return cls._generate_alternative_approach(query)

    @classmethod
    def _generate_specific_clarification(cls, query: str) -> str:
        """Generate more specific clarification for second attempt"""
        templates = [
            f"I want to make sure I give you exactly what you need. Could you tell me more about your specific goal or what you're trying to achieve?",
            f"To provide the most helpful answer, what's the context or situation that prompted this question?",
            f"I'd like to focus my response better. What would be most valuable for you to know about this topic?"
        ]
        return random.choice(templates)

    @classmethod
    def _generate_alternative_approach(cls, query: str) -> str:
        """Generate alternative approach for persistent clarification needs"""
        templates = [
            "Let me try a different approach. Could you give me an example of what you're looking for or describe a similar situation?",
            "Maybe I can help by asking: what would you do with this information once you have it?",
            "Let's break this down differently. What's the first thing you'd like to understand about this topic?"
        ]
        return random.choice(templates)

    @classmethod
    def create_multiple_choice_clarification(cls, options: List[str], query: str) -> str:
        """Create multiple choice clarification"""
        if len(options) <= 1:
            return cls.generate_clarification(query)
        
        intro = "I found information on several related topics. Which one would be most helpful?"
        
        choice_list = "\n".join([f"{i+1}. {option}" for i, option in enumerate(options)])
        
        return f"{intro}\n\n{choice_list}\n\nPlease let me know which option interests you, or if you'd like information on multiple topics."

    @classmethod
    def create_contextual_clarification(cls, query: str, context_type: str, 
                                      available_info: Dict[str, Any]) -> str:
        """Create clarification based on available contextual information"""
        
        if context_type not in cls.CONTEXT_TEMPLATES:
            return cls.generate_clarification(query)
        
        context_templates = cls.CONTEXT_TEMPLATES[context_type]
        
        # Choose appropriate sub-context
        if "implementation" in query.lower() and "technical" in context_templates:
            templates = context_templates["technical"]["implementation"]
        elif "process" in query.lower() and "workflow" in context_templates:
            templates = context_templates["process"]["workflow"]
        elif "explain" in query.lower() and "conceptual" in context_templates:
            templates = context_templates["conceptual"]["explanation"]
        else:
            # Use first available template set
            first_category = list(context_templates.keys())[0]
            first_subcategory = list(context_templates[first_category].keys())[0]
            templates = context_templates[first_category][first_subcategory]
        
        return random.choice(templates)

    # Validation and quality checks
    @classmethod
    def validate_clarification_quality(cls, clarification: str, query: str) -> Dict[str, Any]:
        """Validate the quality of generated clarification"""
        quality_metrics = {
            "is_question": "?" in clarification,
            "is_specific": len(clarification.split()) > 10,
            "offers_options": any(word in clarification.lower() for word in ["or", "either", "which", "option"]),
            "is_helpful": any(word in clarification.lower() for word in ["help", "assist", "clarify", "understand"]),
            "appropriate_length": 20 <= len(clarification.split()) <= 50
        }
        
        quality_score = sum(quality_metrics.values()) / len(quality_metrics)
        
        return {
            "score": quality_score,
            "metrics": quality_metrics,
            "passed": quality_score >= 0.6
        }

    @classmethod
    def get_fallback_clarification(cls) -> str:
        """Get fallback clarification when all else fails"""
        return ("I want to help you find the right information. Could you try asking your question "
                "in a different way, or let me know what specific problem you're trying to solve?")