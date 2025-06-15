from typing import Dict, Any, List, Optional, TypedDict
from langgraph.graph import StateGraph, END
from langgraph.graph.state import CompiledStateGraph
import re

# Import other components
from .llm_client import LMStudioClient
from .vector_store import VectorStore
from .conversation import ConversationHandler
from config import Config

# Import prompt modules
from prompts.system_prompts import SystemPrompts
from prompts.clarification import ClarificationTemplates

class AgentState(TypedDict):
    """State structure for the agent graph"""
    user_query: str
    conversation_context: str
    retrieved_documents: List[Dict[str, Any]]
    needs_clarification: bool
    clarification_question: str
    final_response: str
    iteration_count: int
    search_results: List[Dict[str, Any]]
    analysis_results: Optional[Dict[str, Any]]

class AgentGraph:
    """LangGraph-based agentic workflow for RAG system"""
    
    def __init__(self, llm_client: LMStudioClient, vector_store: VectorStore, 
                 conversation_handler: ConversationHandler):
        self.llm_client = llm_client
        self.vector_store = vector_store
        self.conversation_handler = conversation_handler
        self.graph = self._build_graph()
        
        # Clarification thresholds
        self.MIN_DOCS_FOR_RESPONSE = 1  # Minimum docs needed to avoid clarification
        self.MIN_SIMILARITY_SCORE = 0.3  # Minimum similarity score (lower distance = higher similarity)
        self.MIN_QUERY_WORDS = 2  # Minimum words in query without context
    
    def _build_graph(self) -> CompiledStateGraph:
        """Build the LangGraph workflow"""
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("analyze_query", self._analyze_query)
        workflow.add_node("document_retrieval", self._retrieve_documents)
        workflow.add_node("clarification_check", self._check_clarification_needed)
        workflow.add_node("generate_clarification", self._generate_clarification)
        workflow.add_node("generate_response", self._generate_response)
        
        # Add edges
        workflow.set_entry_point("analyze_query")
        
        workflow.add_edge("analyze_query", "document_retrieval")
        workflow.add_edge("document_retrieval", "clarification_check")
        
        # Conditional edges
        workflow.add_conditional_edges(
            "clarification_check",
            self._should_clarify,
            {
                "clarify": "generate_clarification",
                "respond": "generate_response"
            }
        )
        
        workflow.add_edge("generate_clarification", END)
        workflow.add_edge("generate_response", END)
        
        return workflow.compile()
    
    def _analyze_query(self, state: AgentState) -> Dict[str, Any]:
        """Analyze the user query for intent and context"""
        query = state["user_query"]
        context = self.conversation_handler.get_conversation_context()
        
        # Enhanced query analysis using structured prompts
        analysis_prompt = SystemPrompts.get_query_analysis_prompt(query, context)
        
        try:
            analysis_response = self.llm_client.chat_completion([
                {"role": "system", "content": SystemPrompts.BASE_SYSTEM_PROMPT},
                {"role": "user", "content": analysis_prompt}
            ])
            
            # Parse analysis response
            query_analysis = self._parse_query_analysis(analysis_response)
            
        except Exception as e:
            print(f"Error in query analysis: {e}")
            query_analysis = {
                "intent": "general_question",
                "clarity": "CLEAR",
                "context_needed": False,
                "key_terms": query.split(),
                "search_strategy": "similarity_search"
            }
        
        return {
            **state,
            "conversation_context": context,
            "iteration_count": state.get("iteration_count", 0) + 1,
            "search_results": [],
            "query_analysis": query_analysis
        }
    
    def _parse_query_analysis(self, analysis_text: str) -> Dict[str, Any]:
        """Parse the structured query analysis response"""
        analysis = {
            "intent": "general_question",
            "clarity": "CLEAR",
            "context_needed": False,
            "key_terms": [],
            "search_strategy": "similarity_search"
        }
        
        lines = analysis_text.split('\n')
        for line in lines:
            if line.startswith("INTENT:"):
                analysis["intent"] = line.replace("INTENT:", "").strip()
            elif line.startswith("CLARITY:"):
                analysis["clarity"] = line.replace("CLARITY:", "").strip()
            elif line.startswith("CONTEXT_NEEDED:"):
                analysis["context_needed"] = "YES" in line.upper()
            elif line.startswith("KEY_TERMS:"):
                terms = line.replace("KEY_TERMS:", "").strip()
                analysis["key_terms"] = [term.strip() for term in terms.split(",")]
            elif line.startswith("SEARCH_STRATEGY:"):
                analysis["search_strategy"] = line.replace("SEARCH_STRATEGY:", "").strip()
        
        return analysis
    
    def _retrieve_documents(self, state: AgentState) -> Dict[str, Any]:
        """Retrieve relevant documents from vector store"""
        query = state["user_query"]
        context = state.get("conversation_context", "")
        query_analysis = state.get("query_analysis", {})
        
        # Enhance query with context for better retrieval
        enhanced_query = query
        if context and self.conversation_handler.is_follow_up_question(query):
            # Use key terms from analysis if available
            if query_analysis.get("key_terms"):
                key_terms = " ".join(query_analysis["key_terms"])
                enhanced_query = f"{key_terms} {query}"
            else:
                recent_queries = self.conversation_handler.get_recent_user_queries(2)
                enhanced_query = f"{' '.join(recent_queries)} {query}"
        
        # Retrieve documents
        try:
            search_results = self.vector_store.similarity_search(
                enhanced_query, 
                k=Config.TOP_K_RESULTS
            )
            
            # Filter results by similarity threshold
            filtered_results = [
                result for result in search_results 
                if result.get("distance", 1.0) >= (Config.SIMILARITY_THRESHOLD)
            ]
            
        except Exception as e:
            print(f"Error in document retrieval: {e}")
            filtered_results = []
        
        return {
            **state,
            "retrieved_documents": filtered_results,
            "search_results": search_results
        }
    
    def _check_clarification_needed(self, state: AgentState) -> Dict[str, Any]:
        """IMPROVED: Check if clarification is needed with stricter conditions"""
        query = state["user_query"]
        retrieved_docs = state["retrieved_documents"]
        context = state.get("conversation_context", "")
        query_analysis = state.get("query_analysis", {})
        
        needs_clarification = False
        
        # CONDITION 1: No documents found at all
        if len(retrieved_docs) == 0:
            needs_clarification = True
            print("🔍 Clarification needed: No relevant documents found")
        
        # CONDITION 2: Very short query with no context (but be more lenient)
        elif len(query.split()) <= self.MIN_QUERY_WORDS and not context:
            # Only clarify for single-word queries or very ambiguous ones
            ambiguous_single_words = ["help", "info", "what", "how", "explain", "tell", "show"]
            if len(query.split()) == 1 and query.lower() in ambiguous_single_words:
                needs_clarification = True
                print("🔍 Clarification needed: Query too vague")
        
        # CONDITION 3: Poor quality results (all documents have low similarity)
        elif retrieved_docs:
            # Check if all retrieved documents have poor similarity scores
            poor_results = all(doc.get("distance", 0) < Config.SIMILARITY_THRESHOLD for doc in retrieved_docs)
            if poor_results:
                needs_clarification = True
                print("🔍 Clarification needed: Poor quality search results")
        
        # CONDITION 4: Only ask LLM for clarification decision in edge cases
        # Remove the automatic LLM call that was causing unnecessary clarifications
        
        # CONDITION 5: Special case - if user explicitly asks for clarification
        clarification_requests = ["can you clarify", "what do you mean", "i don't understand", 
                                "be more specific", "explain better"]
        if any(phrase in query.lower() for phrase in clarification_requests):
            needs_clarification = True
            print("🔍 Clarification needed: User requested clarification")
        
        return {
            **state,
            "needs_clarification": needs_clarification
        }
    
    def _generate_clarification(self, state: AgentState) -> Dict[str, Any]:
        """Generate clarification questions using templates"""
        query = state["user_query"]
        retrieved_docs = state["retrieved_documents"]
        context = state.get("conversation_context", "")
        query_analysis = state.get("query_analysis", {})
        
        # Determine clarification type based on analysis
        clarification_type = "vague_query"
        if len(retrieved_docs) == 0:
            clarification_type = "insufficient_information"
        elif len(retrieved_docs) > 3:
            clarification_type = "multiple_topics"
        elif query_analysis.get("clarity") == "CONTEXTUAL":
            clarification_type = "ambiguous_context"
        
        try:
            # Use clarification templates for better consistency
            if len(retrieved_docs) > 1:
                # Create multiple choice clarification
                doc_topics = []
                for doc in retrieved_docs[:3]:
                    source = doc.get("metadata", {}).get("source", "")
                    if source:
                        topic = source.replace(".txt", "").replace(".md", "").replace("_", " ")
                        if topic not in doc_topics:
                            doc_topics.append(topic)
                
                if len(doc_topics) > 1:
                    clarification = ClarificationTemplates.create_multiple_choice_clarification(
                        doc_topics, query
                    )
                else:
                    clarification = ClarificationTemplates.generate_clarification(
                        query, context, clarification_type, retrieved_docs
                    )
            else:
                clarification = ClarificationTemplates.generate_clarification(
                    query, context, clarification_type, retrieved_docs
                )
            
            # Validate clarification quality
            quality_check = ClarificationTemplates.validate_clarification_quality(clarification, query)
            if not quality_check["passed"]:
                clarification = ClarificationTemplates.get_fallback_clarification()
                
        except Exception as e:
            print(f"Error generating clarification: {e}")
            clarification = ClarificationTemplates.get_fallback_clarification()
        
        return {
            **state,
            "final_response": clarification,
            "clarification_question": clarification
        }
    
    def _generate_response(self, state: AgentState) -> Dict[str, Any]:
        """Generate final response using retrieved documents and structured prompts"""
        query = state["user_query"]
        retrieved_docs = state["retrieved_documents"]
        context = state.get("conversation_context", "")
        
        if not retrieved_docs:
            # Use error prompt from SystemPrompts
            no_docs_response = SystemPrompts.get_error_response(
                "no_documents", 
                "You might try rephrasing your question or adding more context."
            )
            return {
                **state,
                "final_response": no_docs_response
            }
        
        # Use structured response generation prompt
        rag_prompt = SystemPrompts.get_response_generation_prompt(
            query, context, retrieved_docs
        )
        
        try:
            response = self.llm_client.chat_completion([
                {"role": "system", "content": SystemPrompts.BASE_SYSTEM_PROMPT},
                {"role": "user", "content": rag_prompt}
            ])
            
            # Enhance response with source citations
            response = self._add_source_citations(response, retrieved_docs)
            
        except Exception as e:
            print(f"Error generating response: {e}")
            response = SystemPrompts.get_error_response("llm_error")
        
        return {
            **state,
            "final_response": response
        }
    
    def _add_source_citations(self, response: str, documents: List[Dict[str, Any]]) -> str:
        """Add source citations to the response"""
        if not documents:
            return response
        
        # Add sources at the end
        sources = []
        for i, doc in enumerate(documents[:3], 1):
            source = doc.get("metadata", {}).get("source", f"Document {i}")
            if source not in sources:
                sources.append(f"{i}. {source}")
        
        if sources:
            response += f"\n\n📚 **Sources:**\n" + "\n".join(sources)
        
        return response
    
    def _should_clarify(self, state: AgentState) -> str:
        """Determine whether to clarify or respond"""
        return "clarify" if state.get("needs_clarification", False) else "respond"
    
    async def process_query(self, user_query: str) -> Dict[str, Any]:
        """Process a user query through the agent graph"""
        initial_state = AgentState(
            user_query=user_query,
            conversation_context="",
            retrieved_documents=[],
            needs_clarification=False,
            clarification_question="",
            final_response="",
            iteration_count=0,
            search_results=[],
            query_analysis=None
        )
        
        try:
            # Run the graph
            result = await self.graph.ainvoke(initial_state)
            return result
        except Exception as e:
            print(f"Error processing query through graph: {e}")
            return {
                **initial_state,
                "final_response": SystemPrompts.get_error_response("llm_error")
            }
    
    def process_query_sync(self, user_query: str) -> Dict[str, Any]:
        """Synchronous version of process_query"""
        initial_state = AgentState(
            user_query=user_query,
            conversation_context="",
            retrieved_documents=[],
            needs_clarification=False,
            clarification_question="",
            final_response="",
            iteration_count=0,
            search_results=[],
            query_analysis=None
        )
        
        try:
            # Run the graph synchronously
            result = self.graph.invoke(initial_state)
            return result
        except Exception as e:
            print(f"Error processing query through graph: {e}")
            return {
                **initial_state,
                "final_response": SystemPrompts.get_error_response("llm_error")
            }
    
    def get_graph_visualization(self) -> str:
        """Get a text representation of the graph structure"""
        return """
        Agent Graph Structure:
        
        query_analysis → document_retrieval → clarification_check
                                                     ↓
                              generate_clarification ← needs_clarification?
                                     ↓                        ↓
                                   END                 generate_response
                                                             ↓
                                                           END
        
        Nodes:
        - query_analysis: Analyzes user intent and context using structured prompts
        - document_retrieval: Retrieves relevant documents with enhanced queries
        - clarification_check: Uses stricter conditions to determine clarification needs
        - generate_clarification: Creates clarification using templates and validation
        - generate_response: Generates final RAG response with source citations
        """