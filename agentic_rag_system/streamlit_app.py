#!/usr/bin/env python3
"""
Streamlit Web Interface for Agentic RAG System
"""

import streamlit as st
import os
import sys
from pathlib import Path
import time
from typing import Optional, Dict, Any, List
import tempfile
import shutil

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from config import Config
from core import AgentGraph, LMStudioClient, VectorStore, ConversationHandler

# Page configuration
st.set_page_config(
    page_title="Agentic RAG System",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        border-left: 4px solid #1f77b4;
    }
    .user-message {
        background-color: #f0f8ff;
        border-left-color: #4CAF50;
    }
    .assistant-message {
        background-color: #f8f9fa;
        border-left-color: #1f77b4;
    }
    .system-info {
        background-color: #e8f4fd;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #1f77b4;
    }
    .error-message {
        background-color: #ffebee;
        color: #c62828;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #f44336;
    }
    .success-message {
        background-color: #e8f5e8;
        color: #2e7d32;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #4caf50;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
</style>
""", unsafe_allow_html=True)

class StreamlitRAGApp:
    """Streamlit application for the Agentic RAG System"""
    
    def __init__(self):
        self.initialize_session_state()
    
    def initialize_session_state(self):
        """Initialize Streamlit session state"""
        if 'initialized' not in st.session_state:
            st.session_state.initialized = False
            st.session_state.llm_client = None
            st.session_state.vector_store = None
            st.session_state.conversation_handler = None
            st.session_state.agent_graph = None
            st.session_state.chat_history = []
            st.session_state.system_status = "Not initialized"
            st.session_state.collection_info = {}
            st.session_state.doc_stats = {}
    
    def initialize_system(self) -> bool:
        """Initialize the RAG system components"""
        if st.session_state.initialized:
            return True
        
        with st.spinner("🚀 Initializing Agentic RAG System..."):
            try:
                # Validate configuration
                if not Config.validate():
                    st.error("❌ Configuration validation failed!")
                    return False
                
                # Initialize LLM client
                st.session_state.llm_client = LMStudioClient()
                
                if not st.session_state.llm_client.test_connection():
                    st.error("❌ Failed to connect to LMStudio. Please ensure it's running on http://localhost:1234")
                    return False
                
                # Initialize vector store
                st.session_state.vector_store = VectorStore(st.session_state.llm_client)
                
                # Initialize conversation handler
                st.session_state.conversation_handler = ConversationHandler()
                
                # Initialize agent graph
                st.session_state.agent_graph = AgentGraph(
                    st.session_state.llm_client,
                    st.session_state.vector_store,
                    st.session_state.conversation_handler
                )
                
                st.session_state.initialized = True
                st.session_state.system_status = "Initialized"
                
                # Get system info
                self.update_system_info()
                
                return True
                
            except Exception as e:
                st.error(f"❌ Initialization failed: {e}")
                st.session_state.system_status = f"Error: {e}"
                return False
    
    def update_system_info(self):
        """Update system information in session state"""
        if st.session_state.vector_store:
            st.session_state.collection_info = st.session_state.vector_store.get_collection_info()
            st.session_state.doc_stats = st.session_state.vector_store.get_document_statistics()
    
    def render_header(self):
        """Render the application header"""
        st.markdown('<h1 class="main-header">🤖 Agentic RAG System</h1>', unsafe_allow_html=True)
        st.markdown("---")
    
    def render_sidebar(self):
        """Render the sidebar with system information and controls"""
        with st.sidebar:
            st.header("📊 System Status")
            
            # System status indicator
            if st.session_state.initialized:
                st.success(f"✅ {st.session_state.system_status}")
            else:
                st.warning(f"⚠️ {st.session_state.system_status}")
            
            if st.button("🔄 Initialize System", disabled=st.session_state.initialized):
                self.initialize_system()
                st.rerun()
            
            if st.session_state.initialized:
                st.markdown("---")
                
                # Collection information
                st.subheader("📚 Document Collection")
                collection_info = st.session_state.collection_info
                doc_stats = st.session_state.doc_stats
                
                if collection_info:
                    st.info(f"""
                    **Collection:** {collection_info.get('name', 'Unknown')}
                    **Total Chunks:** {collection_info.get('document_count', 0)}
                    **Files Indexed:** {len(doc_stats.get('files', {}))}
                    """)
                
                # Model information
                st.subheader("🤖 Model Configuration")
                st.info(f"""
                **Chat Model:** {Config.CHAT_MODEL}
                **Embedding Model:** {Config.EMBEDDING_MODEL}
                **Top-K Results:** {Config.TOP_K_RESULTS}
                **Chunk Size:** {Config.CHUNK_SIZE}
                """)
                
                st.markdown("---")
                
                # Quick actions
                st.subheader("⚡ Quick Actions")
                
                if st.button("🗑️ Clear Chat History"):
                    st.session_state.chat_history = []
                    if st.session_state.conversation_handler:
                        st.session_state.conversation_handler.clear_history()
                    st.success("Chat history cleared!")
                    st.rerun()
                
                if st.button("🔄 Refresh System Info"):
                    self.update_system_info()
                    st.success("System info refreshed!")
                    st.rerun()
    
    def render_chat_interface(self):
        """Render the main chat interface"""
        st.header("💬 Chat with Your Documents")
        
        if not st.session_state.initialized:
            st.warning("⚠️ Please initialize the system first using the sidebar.")
            return
        
        # Chat history display
        chat_container = st.container()
        
        with chat_container:
            for i, message in enumerate(st.session_state.chat_history):
                if message["role"] == "user":
                    st.markdown(f"""
                    <div class="chat-message user-message">
                        <strong>🤔 You:</strong><br>
                        {message["content"]}
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="chat-message assistant-message">
                        <strong>🤖 Assistant:</strong><br>
                        {message["content"]}
                    </div>
                    """, unsafe_allow_html=True)
        
        # Chat input
        with st.form("chat_form", clear_on_submit=True):
            user_input = st.text_area(
                "Ask a question about your documents:",
                placeholder="What would you like to know about your documents?",
                height=100
            )
            
            col1, col2 = st.columns([1, 4])
            with col1:
                submit_button = st.form_submit_button("Send 📤", use_container_width=True)
            
            if submit_button and user_input.strip():
                self.process_user_message(user_input.strip())
                st.rerun()
    
    def process_user_message(self, user_input: str):
        """Process user message and generate response"""
        # Add user message to chat history
        st.session_state.chat_history.append({
            "role": "user",
            "content": user_input
        })
        
        # Show processing indicator
        with st.spinner("🤖 Thinking..."):
            try:
                # Add user message to conversation handler
                st.session_state.conversation_handler.add_message("user", user_input)
                
                # Process through agent graph
                result = st.session_state.agent_graph.process_query_sync(user_input)
                
                # Get response
                response = result.get("final_response", "No response generated")
                
                # Add assistant response to conversation handler
                st.session_state.conversation_handler.add_message("assistant", response)
                
                # Update context memory if needed
                if result.get("retrieved_documents"):
                    st.session_state.conversation_handler.update_context_memory(
                        "last_search_results", 
                        len(result["retrieved_documents"])
                    )
                
                # Add assistant message to chat history
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": response
                })
                
            except Exception as e:
                error_message = f"Error processing query: {e}"
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": f"❌ {error_message}"
                })
    
    def render_document_manager(self):
        """Render document management interface"""
        st.header("📁 Document Management")
        
        if not st.session_state.initialized:
            st.warning("⚠️ Please initialize the system first.")
            return
        
        # Current document statistics
        doc_stats = st.session_state.doc_stats
        collection_info = st.session_state.collection_info
        
        if doc_stats.get('total_chunks', 0) > 0:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Chunks", doc_stats['total_chunks'])
            with col2:
                st.metric("Files Indexed", len(doc_stats.get('files', {})))
            with col3:
                st.metric("File Types", len(doc_stats.get('file_types', {})))
            
            # File breakdown
            if doc_stats.get('files'):
                st.subheader("📋 Indexed Files")
                file_data = []
                for filename, chunks in doc_stats['files'].items():
                    file_data.append({
                        "Filename": filename,
                        "Chunks": chunks,
                        "Type": filename.split('.')[-1].upper() if '.' in filename else "Unknown"
                    })
                st.dataframe(file_data, use_container_width=True)
            
            # File type breakdown
            if doc_stats.get('file_types'):
                st.subheader("📊 File Type Distribution")
                file_type_data = []
                total_chunks = doc_stats['total_chunks']
                for file_type, count in doc_stats['file_types'].items():
                    percentage = (count / total_chunks) * 100
                    file_type_data.append({
                        "File Type": file_type.upper(),
                        "Chunks": count,
                        "Percentage": f"{percentage:.1f}%"
                    })
                st.dataframe(file_type_data, use_container_width=True)
        
        st.markdown("---")
        
        # Document upload section
        st.subheader("📤 Upload Documents")
        
        # File uploader
        uploaded_files = st.file_uploader(
            "Choose files to upload",
            type=['txt', 'md', 'py', 'js', 'json', 'csv', 'pdf'],
            accept_multiple_files=True,
            help="Supported formats: TXT, MD, PY, JS, JSON, CSV, PDF"
        )
        
        if uploaded_files:
            if st.button("🚀 Process Uploaded Files"):
                self.process_uploaded_files(uploaded_files)
        
        # Directory path input
        st.markdown("**Or load from directory:**")
        directory_path = st.text_input(
            "Directory Path:",
            value=Config.DOCUMENTS_PATH,
            help="Enter the path to a directory containing documents"
        )
        
        if st.button("📁 Load from Directory"):
            if directory_path and os.path.exists(directory_path):
                self.load_documents_from_directory(directory_path)
            else:
                st.error("❌ Invalid directory path")
        
        st.markdown("---")
        
        # Database management
        st.subheader("🗄️ Database Management")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Refresh Statistics"):
                self.update_system_info()
                st.success("Statistics refreshed!")
                st.rerun()
        
        with col2:
            if st.button("🗑️ Clear Database", type="secondary"):
                if st.session_state.get('confirm_clear', False):
                    if st.session_state.vector_store.clear_collection():
                        st.success("🗑️ Database cleared successfully!")
                        self.update_system_info()
                        st.session_state.confirm_clear = False
                        st.rerun()
                    else:
                        st.error("❌ Failed to clear database")
                else:
                    st.session_state.confirm_clear = True
                    st.warning("⚠️ Click again to confirm database clearing")
                    st.rerun()
    
    def process_uploaded_files(self, uploaded_files):
        """Process uploaded files"""
        with st.spinner("📤 Processing uploaded files..."):
            try:
                # Create temporary directory
                with tempfile.TemporaryDirectory() as temp_dir:
                    # Save uploaded files to temp directory
                    for uploaded_file in uploaded_files:
                        file_path = os.path.join(temp_dir, uploaded_file.name)
                        with open(file_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())
                    
                    # Load documents from temp directory
                    success = st.session_state.vector_store.load_documents_from_directory(temp_dir)
                    
                    if success:
                        self.update_system_info()
                        st.success(f"✅ Successfully processed {len(uploaded_files)} file(s)!")
                        st.rerun()
                    else:
                        st.error("❌ Failed to process uploaded files")
                        
            except Exception as e:
                st.error(f"❌ Error processing files: {e}")
    
    def load_documents_from_directory(self, directory_path: str):
        """Load documents from a directory"""
        with st.spinner(f"📁 Loading documents from {directory_path}..."):
            try:
                success = st.session_state.vector_store.load_documents_from_directory(directory_path)
                
                if success:
                    self.update_system_info()
                    st.success("✅ Documents loaded successfully!")
                    st.rerun()
                else:
                    st.error("❌ Failed to load documents")
                    
            except Exception as e:
                st.error(f"❌ Error loading documents: {e}")
    
    def render_system_settings(self):
        """Render system settings interface"""
        st.header("⚙️ System Settings")
        
        # Configuration display
        st.subheader("📋 Current Configuration")
        
        config_data = {
            "Setting": [
                "Chat Model",
                "Embedding Model", 
                "LMStudio URL",
                "Top-K Results",
                "Similarity Threshold",
                "Chunk Size",
                "Chunk Overlap",
                "Temperature",
                "Max Tokens"
            ],
            "Value": [
                Config.CHAT_MODEL,
                Config.EMBEDDING_MODEL,
                Config.LMSTUDIO_BASE_URL,
                Config.TOP_K_RESULTS,
                Config.SIMILARITY_THRESHOLD,
                Config.CHUNK_SIZE,
                Config.CHUNK_OVERLAP,
                Config.TEMPERATURE,
                Config.MAX_TOKENS
            ]
        }
        
        st.dataframe(config_data, use_container_width=True)
        
        st.markdown("---")
        
        # System diagnostics
        st.subheader("🔍 System Diagnostics")
        
        if st.button("🧪 Run System Tests"):
            self.run_system_diagnostics()
    
    def run_system_diagnostics(self):
        """Run system diagnostics"""
        st.subheader("🔍 Diagnostic Results")
        
        # Test LMStudio connection
        with st.spinner("Testing LMStudio connection..."):
            if st.session_state.llm_client and st.session_state.llm_client.test_connection():
                st.success("✅ LMStudio connection: OK")
            else:
                st.error("❌ LMStudio connection: FAILED")
        
        # Test vector store
        with st.spinner("Testing vector store..."):
            try:
                collection_info = st.session_state.vector_store.get_collection_info()
                st.success(f"✅ Vector store: OK ({collection_info.get('document_count', 0)} chunks)")
            except Exception as e:
                st.error(f"❌ Vector store: FAILED ({e})")
        
        # Test conversation handler
        with st.spinner("Testing conversation handler..."):
            try:
                history_count = len(st.session_state.conversation_handler.conversation_history)
                st.success(f"✅ Conversation handler: OK ({history_count} messages)")
            except Exception as e:
                st.error(f"❌ Conversation handler: FAILED ({e})")
        
        # Test agent graph
        with st.spinner("Testing agent graph..."):
            try:
                test_query = "test query"
                result = st.session_state.agent_graph.process_query_sync(test_query)
                if result.get("final_response"):
                    st.success("✅ Agent graph: OK")
                else:
                    st.warning("⚠️ Agent graph: Response generation issue")
            except Exception as e:
                st.error(f"❌ Agent graph: FAILED ({e})")
    
    def run(self):
        """Main application runner"""
        self.render_header()
        self.render_sidebar()
        
        # Main content area with tabs
        tab1, tab2, tab3 = st.tabs(["💬 Chat", "📁 Documents", "⚙️ Settings"])
        
        with tab1:
            self.render_chat_interface()
        
        with tab2:
            self.render_document_manager()
        
        with tab3:
            self.render_system_settings()

def main():
    """Main entry point"""
    app = StreamlitRAGApp()
    app.run()

if __name__ == "__main__":
    main()