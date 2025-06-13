"""
Streamlit app for BI Chatbot POC
Main interface for interacting with the agentic BI assistant
"""

import streamlit as st
import yaml
import logging
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, Any, List
import os
import sys
from datetime import datetime

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.agents.orchestrator import BiChatbotOrchestrator
from src.utils import load_config, setup_logging

# Configure logging
setup_logging()
logger = logging.getLogger(__name__)

# Page config
st.set_page_config(
    page_title="BI Chatbot POC",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 0.5rem;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        border-left: 4px solid #1f77b4;
    }
    .user-message {
        background-color: #f0f2f6;
        border-left-color: #ff7f0e;
    }
    .bot-message {
        background-color: #e8f4fd;
        border-left-color: #1f77b4;
    }
    .intent-badge {
        display: inline-block;
        padding: 0.25rem 0.5rem;
        border-radius: 0.25rem;
        font-size: 0.75rem;
        font-weight: bold;
        margin-left: 0.5rem;
    }
    .rag-badge { background-color: #d4edda; color: #155724; }
    .sql-badge { background-color: #cce5ff; color: #004085; }
    .general-badge { background-color: #fff3cd; color: #856404; }
    .clarification-badge { background-color: #f8d7da; color: #721c24; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def initialize_chatbot():
    """Initialize the chatbot orchestrator"""
    try:
        config = load_config()
        orchestrator = BiChatbotOrchestrator(config)
        return orchestrator, None
    except Exception as e:
        logger.error(f"Failed to initialize chatbot: {str(e)}")
        return None, str(e)

def display_chat_message(message: Dict[str, Any], is_user: bool = False):
    """Display a chat message with proper styling"""
    if is_user:
        st.markdown(f"""
        <div class="chat-message user-message">
            <strong>You:</strong> {message.get('content', message.get('user', ''))}
        </div>
        """, unsafe_allow_html=True)
    else:
        intent = message.get('intent', 'general')
        intent_class = f"{intent}-badge" if intent != 'error' else 'clarification-badge'
        
        st.markdown(f"""
        <div class="chat-message bot-message">
            <strong>Assistant:</strong> {message.get('content', message.get('bot', ''))}
            <span class="intent-badge {intent_class}">{intent.upper()}</span>
        </div>
        """, unsafe_allow_html=True)

def display_visualization(viz_data: Dict[str, Any]):
    """Display visualization if available"""
    if not viz_data:
        return
    
    try:
        chart_type = viz_data.get('type', 'bar')
        data = viz_data.get('data', {})
        
        if not data:
            return
            
        df = pd.DataFrame(data)
        
        if chart_type == 'bar':
            if len(df.columns) >= 2:
                fig = px.bar(df, x=df.columns[0], y=df.columns[1], 
                           title=viz_data.get('title', 'Data Visualization'))
        elif chart_type == 'line':
            if len(df.columns) >= 2:
                fig = px.line(df, x=df.columns[0], y=df.columns[1], 
                            title=viz_data.get('title', 'Trend Analysis'))
        elif chart_type == 'scatter':
            if len(df.columns) >= 2:
                fig = px.scatter(df, x=df.columns[0], y=df.columns[1], 
                               title=viz_data.get('title', 'Scatter Plot'))
        else:
            # Default to bar chart
            fig = px.bar(df, x=df.columns[0], y=df.columns[1] if len(df.columns) > 1 else df.columns[0])
        
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error displaying visualization: {str(e)}")

def display_data_table(data: Dict[str, Any]):
    """Display data table if available"""
    if not data or 'results' not in data:
        return
    
    results = data['results']
    if not results:
        return
    
    try:
        df = pd.DataFrame(results)
        if not df.empty:
            st.subheader("Query Results")
            st.dataframe(df, use_container_width=True)
            
            # Show summary stats
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Rows", len(df))
            with col2:
                st.metric("Columns", len(df.columns))
            with col3:
                if data.get('sql'):
                    st.code(data['sql'], language='sql')
                    
    except Exception as e:
        st.error(f"Error displaying data: {str(e)}")

def display_context_info(context: Dict[str, Any]):
    """Display context information"""
    if not context:
        return
    
    with st.expander("Context Information", expanded=False):
        if 'retrieved_docs' in context:
            st.write(f"**Retrieved Documents:** {context['retrieved_docs']}")
            
            if 'sources' in context:
                st.write("**Sources:**")
                for source in context['sources'][:3]:  # Show first 3 sources
                    st.write(f"- {source}")
                    
        if 'sql' in context:
            st.write("**SQL Query:**")
            st.code(context['sql'], language='sql')

def main():
    """Main Streamlit app"""
    
    # Header
    st.markdown('<h1 class="main-header">🤖 BI Chatbot POC</h1>', unsafe_allow_html=True)
    st.markdown("Ask questions about your data, documents, or general business intelligence topics!")
    
    # Initialize chatbot
    orchestrator, init_error = initialize_chatbot()
    
    if init_error:
        st.error(f"Failed to initialize chatbot: {init_error}")
        st.info("Please check your configuration and ensure all dependencies are properly set up.")
        return
    
    # Sidebar
    with st.sidebar:
        st.header("Settings & Info")
        
        # Session management
        if st.button("🔄 New Conversation"):
            if 'session_id' in st.session_state:
                orchestrator.reset_conversation(st.session_state.session_id)
            st.session_state.messages = []
            st.rerun()
        
        # Example queries
        st.subheader("Example Queries")
        examples = {
            "📊 Data Analysis": [
                "What were our total sales last month?",
                "Show me the top 5 customers by revenue",
                "How many new customers this quarter?"
            ],
            "📄 Document Search": [
                "What does our policy say about remote work?",
                "Show me the Q3 performance highlights",
                "Find information about our product features"
            ],
            "❓ General Questions": [
                "What is a KPI?",
                "Explain data visualization best practices",
                "How does business intelligence work?"
            ]
        }
        
        for category, queries in examples.items():
            with st.expander(category):
                for query in queries:
                    if st.button(query, key=f"example_{hash(query)}"):
                        st.session_state.example_query = query
        
        # Debug info
        with st.expander("Debug Info", expanded=False):
            st.write("**Session ID:**", st.session_state.get('session_id', 'Not set'))
            st.write("**Messages:**", len(st.session_state.get('messages', [])))
    
    # Initialize session state
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    if 'session_id' not in st.session_state:
        st.session_state.session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Display chat history
    chat_container = st.container()
    
    with chat_container:
        for i, message in enumerate(st.session_state.messages):
            if message['role'] == 'user':
                display_chat_message({'content': message['content']}, is_user=True)
            else:
                display_chat_message({
                    'content': message['content'],
                    'intent': message.get('intent', 'general')
                })
                
                # Display additional data if available
                if message.get('visualization'):
                    display_visualization(message['visualization'])
                
                if message.get('data'):
                    display_data_table(message['data'])
                
                if message.get('context'):
                    display_context_info(message['context'])
    
    # Chat input
    user_input = st.chat_input("Ask me anything about your data or documents...")
    
    # Handle example query selection
    if 'example_query' in st.session_state:
        user_input = st.session_state.example_query
        del st.session_state.example_query
    
    if user_input:
        # Add user message to chat
        st.session_state.messages.append({
            'role': 'user',
            'content': user_input
        })
        
        # Show user message immediately
        with chat_container:
            display_chat_message({'content': user_input}, is_user=True)
        
        # Process with chatbot
        with st.spinner("Thinking..."):
            try:
                response = orchestrator.chat(user_input, st.session_state.session_id)
                
                # Add bot response to chat
                bot_message = {
                    'role': 'assistant',
                    'content': response['response'],
                    'intent': response.get('intent', 'general')
                }
                
                # Add additional data if available
                if response.get('visualization'):
                    bot_message['visualization'] = response['visualization']
                
                if response.get('context'):
                    bot_message['context'] = response['context']
                
                # For SQL queries, include data
                if response.get('intent') == 'sql' and 'context' in response:
                    context = response['context']
                    if isinstance(context, dict) and 'results' in context:
                        bot_message['data'] = context
                
                st.session_state.messages.append(bot_message)
                
                # Display bot response
                with chat_container:
                    display_chat_message({
                        'content': response['response'],
                        'intent': response.get('intent', 'general')
                    })
                    
                    # Display additional data
                    if bot_message.get('visualization'):
                        display_visualization(bot_message['visualization'])
                    
                    if bot_message.get('data'):
                        display_data_table(bot_message['data'])
                    
                    if bot_message.get('context'):
                        display_context_info(bot_message['context'])
                
            except Exception as e:
                st.error(f"Error processing your request: {str(e)}")
                logger.error(f"Chat processing error: {str(e)}")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; font-size: 0.8rem;'>
        BI Chatbot POC - Powered by LangGraph & LM Studio
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()