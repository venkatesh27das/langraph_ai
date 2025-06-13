"""
Main orchestrator using LangGraph for BI chatbot workflow.
Determines which agent to route queries to and manages conversation flow.
"""

from typing import TypedDict, List, Any, Dict
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolExecutor
import logging

from .rag_agent import RAGAgent
from .sql_agent import SQLAgent
from .general_agent import GeneralAgent
from .clarification_agent import ClarificationAgent
from ..core.conversation_memory import ConversationMemory
from ..core.lm_studio_client import LMStudioClient

logger = logging.getLogger(__name__)

class ChatState(TypedDict):
    """State object for LangGraph workflow"""
    user_input: str
    conversation_history: List[Dict[str, str]]
    current_intent: str
    needs_clarification: bool
    clarification_questions: List[str]
    response: str
    context_data: Dict[str, Any]
    visualization_data: Dict[str, Any]
    error: str

class BiChatbotOrchestrator:
    """Main orchestrator class using LangGraph for workflow management"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.llm_client = LMStudioClient(config['lm_studio'])
        self.memory = ConversationMemory()
        
        # Initialize agents
        self.rag_agent = RAGAgent(config, self.llm_client)
        self.sql_agent = SQLAgent(config, self.llm_client)
        self.general_agent = GeneralAgent(config, self.llm_client)
        self.clarification_agent = ClarificationAgent(config, self.llm_client)
        
        # Build the workflow graph
        self.workflow = self._build_workflow()
        
    def _build_workflow(self) -> StateGraph:
        """Build the LangGraph workflow"""
        workflow = StateGraph(ChatState)
        
        # Add nodes
        workflow.add_node("classify_intent", self._classify_intent)
        workflow.add_node("clarification", self._handle_clarification)
        workflow.add_node("rag_processing", self._handle_rag)
        workflow.add_node("sql_processing", self._handle_sql)
        workflow.add_node("general_processing", self._handle_general)
        workflow.add_node("finalize_response", self._finalize_response)
        
        # Set entry point
        workflow.set_entry_point("classify_intent")
        
        # Add conditional edges
        workflow.add_conditional_edges(
            "classify_intent",
            self._route_after_classification,
            {
                "clarification": "clarification",
                "rag": "rag_processing",
                "sql": "sql_processing",
                "general": "general_processing"
            }
        )
        
        # Add edges from processing nodes to finalization
        workflow.add_edge("clarification", "finalize_response")
        workflow.add_edge("rag_processing", "finalize_response")
        workflow.add_edge("sql_processing", "finalize_response")
        workflow.add_edge("general_processing", "finalize_response")
        
        # End after finalization
        workflow.add_edge("finalize_response", END)
        
        return workflow.compile()
    
    def _classify_intent(self, state: ChatState) -> ChatState:
        """Classify user intent to determine routing"""
        try:
            user_input = state["user_input"]
            history = state.get("conversation_history", [])
            
            # Get classification from LLM
            classification_result = self._get_intent_classification(user_input, history)
            
            state["current_intent"] = classification_result["intent"]
            state["needs_clarification"] = classification_result["needs_clarification"]
            state["clarification_questions"] = classification_result.get("clarification_questions", [])
            
            logger.info(f"Classified intent: {classification_result['intent']}")
            
        except Exception as e:
            logger.error(f"Error in intent classification: {str(e)}")
            state["error"] = f"Classification error: {str(e)}"
            state["current_intent"] = "general"
            
        return state
    
    def _get_intent_classification(self, user_input: str, history: List[Dict]) -> Dict[str, Any]:
        """Use LLM to classify user intent"""
        context = ""
        if history:
            context = "\n".join([f"User: {h.get('user', '')}\nBot: {h.get('bot', '')}" for h in history[-3:]])
        
        prompt = f"""
        Analyze the user's query and classify it into one of these categories:
        1. "rag" - Questions about documents, reports, unstructured data
        2. "sql" - Questions about data analysis, metrics, structured data queries
        3. "general" - General knowledge questions not related to company data
        4. "clarification" - Vague queries that need clarification
        
        Context from previous conversation:
        {context}
        
        Current user query: "{user_input}"
        
        Respond with JSON format:
        {{
            "intent": "rag|sql|general|clarification",
            "confidence": 0.8,
            "needs_clarification": false,
            "clarification_questions": [],
            "reasoning": "brief explanation"
        }}
        """
        
        response = self.llm_client.generate(prompt, max_tokens=200)
        
        # Parse JSON response (simplified - in production, use proper JSON parsing)
        try:
            import json
            result = json.loads(response.strip())
            return result
        except:
            # Fallback if JSON parsing fails
            if "rag" in response.lower() or "document" in response.lower():
                return {"intent": "rag", "needs_clarification": False}
            elif "sql" in response.lower() or "data" in response.lower():
                return {"intent": "sql", "needs_clarification": False}
            elif "clarification" in response.lower() or "vague" in response.lower():
                return {"intent": "clarification", "needs_clarification": True}
            else:
                return {"intent": "general", "needs_clarification": False}
    
    def _route_after_classification(self, state: ChatState) -> str:
        """Determine which node to route to after classification"""
        if state["needs_clarification"]:
            return "clarification"
        
        intent = state["current_intent"]
        if intent == "rag":
            return "rag"
        elif intent == "sql":
            return "sql"
        else:
            return "general"
    
    def _handle_clarification(self, state: ChatState) -> ChatState:
        """Handle clarification requests"""
        try:
            response = self.clarification_agent.process(
                state["user_input"],
                state["conversation_history"],
                state.get("clarification_questions", [])
            )
            state["response"] = response
        except Exception as e:
            logger.error(f"Error in clarification processing: {str(e)}")
            state["error"] = f"Clarification error: {str(e)}"
            state["response"] = "I need more information to help you better. Could you please provide more details about what you're looking for?"
        
        return state
    
    def _handle_rag(self, state: ChatState) -> ChatState:
        """Handle RAG processing for unstructured data"""
        try:
            result = self.rag_agent.process(
                state["user_input"],
                state["conversation_history"]
            )
            state["response"] = result["response"]
            state["context_data"] = result.get("context", {})
        except Exception as e:
            logger.error(f"Error in RAG processing: {str(e)}")
            state["error"] = f"RAG processing error: {str(e)}"
            state["response"] = "I encountered an error while searching through the documents. Please try rephrasing your question."
        
        return state
    
    def _handle_sql(self, state: ChatState) -> ChatState:
        """Handle SQL processing for structured data"""
        try:
            result = self.sql_agent.process(
                state["user_input"],
                state["conversation_history"]
            )
            state["response"] = result["response"]
            state["visualization_data"] = result.get("visualization", {})
            state["context_data"] = result.get("data", {})
        except Exception as e:
            logger.error(f"Error in SQL processing: {str(e)}")
            state["error"] = f"SQL processing error: {str(e)}"
            state["response"] = "I couldn't process your data query. Please check if your question relates to available data sources."
        
        return state
    
    def _handle_general(self, state: ChatState) -> ChatState:
        """Handle general knowledge queries"""
        try:
            response = self.general_agent.process(
                state["user_input"],
                state["conversation_history"]
            )
            state["response"] = response
        except Exception as e:
            logger.error(f"Error in general processing: {str(e)}")
            state["error"] = f"General processing error: {str(e)}"
            state["response"] = "I encountered an error while processing your question. Please try again."
        
        return state
    
    def _finalize_response(self, state: ChatState) -> ChatState:
        """Finalize the response and update conversation memory"""
        try:
            # Update conversation history
            conversation_entry = {
                "user": state["user_input"],
                "bot": state["response"],
                "intent": state["current_intent"],
                "timestamp": self.memory.get_timestamp()
            }
            
            # Add context data if available
            if state.get("context_data"):
                conversation_entry["context"] = state["context_data"]
            if state.get("visualization_data"):
                conversation_entry["visualization"] = state["visualization_data"]
            
            # Update memory
            updated_history = state.get("conversation_history", [])
            updated_history.append(conversation_entry)
            state["conversation_history"] = updated_history[-10:]  # Keep last 10 exchanges
            
            logger.info(f"Finalized response for intent: {state['current_intent']}")
            
        except Exception as e:
            logger.error(f"Error in response finalization: {str(e)}")
        
        return state
    
    def chat(self, user_input: str, session_id: str = "default") -> Dict[str, Any]:
        """Main chat interface"""
        try:
            # Get conversation history
            history = self.memory.get_conversation(session_id)
            
            # Initialize state
            initial_state = ChatState(
                user_input=user_input,
                conversation_history=history,
                current_intent="",
                needs_clarification=False,
                clarification_questions=[],
                response="",
                context_data={},
                visualization_data={},
                error=""
            )
            
            # Run the workflow
            final_state = self.workflow.invoke(initial_state)
            
            # Update memory with final state
            self.memory.update_conversation(session_id, final_state["conversation_history"])
            
            # Return response
            return {
                "response": final_state["response"],
                "intent": final_state["current_intent"],
                "visualization": final_state.get("visualization_data"),
                "context": final_state.get("context_data"),
                "error": final_state.get("error")
            }
            
        except Exception as e:
            logger.error(f"Error in chat processing: {str(e)}")
            return {
                "response": "I encountered an error processing your request. Please try again.",
                "intent": "error",
                "error": str(e)
            }
    
    def reset_conversation(self, session_id: str = "default"):
        """Reset conversation for a session"""
        self.memory.clear_conversation(session_id)