#!/usr/bin/env python3
"""
Agentic RAG System - Terminal Interface
"""

import os
import sys
import asyncio
from typing import Optional
import argparse
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from config import Config
from core import AgentGraph, LMStudioClient, VectorStore, ConversationHandler

class RAGTerminalInterface:
    """Terminal interface for the Agentic RAG System"""
    
    def __init__(self):
        self.llm_client: Optional[LMStudioClient] = None
        self.vector_store: Optional[VectorStore] = None
        self.conversation_handler: Optional[ConversationHandler] = None
        self.agent_graph: Optional[AgentGraph] = None
        self.initialized = False
    
    def initialize(self) -> bool:
        """Initialize all components"""
        print("🚀 Initializing Agentic RAG System...")
        
        # Validate configuration
        if not Config.validate():
            print("❌ Configuration validation failed!")
            return False
        
        try:
            # Initialize LLM client
            print("📡 Connecting to LMStudio...")
            self.llm_client = LMStudioClient()
            
            if not self.llm_client.test_connection():
                print("❌ Failed to connect to LMStudio. Please ensure it's running on http://localhost:1234")
                return False
            
            print("✅ LMStudio connection successful")
            
            # Initialize vector store
            print("🗄️  Initializing ChromaDB...")
            self.vector_store = VectorStore(self.llm_client)
            
            # Initialize conversation handler
            print("💬 Setting up conversation handler...")
            self.conversation_handler = ConversationHandler()
            
            # Initialize agent graph
            print("🤖 Building agent graph...")
            self.agent_graph = AgentGraph(
                self.llm_client,
                self.vector_store,
                self.conversation_handler
            )
            
            self.initialized = True
            print("✅ System initialized successfully!")
            
            # Display system info
            self._display_system_info()
            
            return True
            
        except Exception as e:
            print(f"❌ Initialization failed: {e}")
            return False
    
    def _display_system_info(self):
        """Display system information"""
        print("\n" + "="*60)
        print("📊 SYSTEM INFORMATION")
        print("="*60)
        
        # Vector store info
        collection_info = self.vector_store.get_collection_info()
        print(f"📚 Document Collection: {collection_info['name']}")
        print(f"📄 Total Chunks Indexed: {collection_info['document_count']}")
        print(f"💾 Storage Path: {collection_info['path']}")
        
        # Get detailed document statistics
        doc_stats = self.vector_store.get_document_statistics()
        if doc_stats['total_chunks'] > 0:
            print(f"\n📋 Document Breakdown:")
            for file_type, count in doc_stats['file_types'].items():
                print(f"   {file_type}: {count} chunks")
            
            print(f"\n📁 Files Indexed:")
            for filename, chunks in doc_stats['files'].items():
                print(f"   {filename}: {chunks} chunks")
        
        # Model info
        print(f"\n🤖 Chat Model: {Config.CHAT_MODEL}")
        print(f"🔍 Embedding Model: {Config.EMBEDDING_MODEL}")
        print(f"🌐 LMStudio URL: {Config.LMSTUDIO_BASE_URL}")
        
        # Chunk settings
        print(f"📏 Chunk Size: {Config.CHUNK_SIZE}")
        print(f"🔄 Chunk Overlap: {Config.CHUNK_OVERLAP}")
        print(f"🔍 Top-K Results: {Config.TOP_K_RESULTS}")
        
        print("="*60)
    
    def load_documents(self, directory_path: str) -> bool:
        """Load documents from a directory"""
        if not self.initialized:
            print("❌ System not initialized!")
            return False
        
        print(f"📁 Loading documents from: {directory_path}")
        
        if not os.path.exists(directory_path):
            print(f"❌ Directory not found: {directory_path}")
            return False
        
        # Check for PDF files and warn about dependencies
        pdf_files = [f for f in os.listdir(directory_path) if f.lower().endswith('.pdf')]
        if pdf_files:
            try:
                import PyPDF2
                import fitz
                print(f"📄 Found {len(pdf_files)} PDF file(s). PDF processing is available.")
            except ImportError:
                print("⚠️  Warning: PDF files found but PDF processing libraries not installed.")
                print("   Install with: pip install PyPDF2 PyMuPDF")
                print("   PDF files will be skipped.")
        
        success = self.vector_store.load_documents_from_directory(directory_path)
        
        if success:
            # Update system info
            collection_info = self.vector_store.get_collection_info()
            doc_stats = self.vector_store.get_document_statistics()
            print(f"\n✅ Documents loaded successfully!")
            print(f"📊 Total chunks: {collection_info['document_count']}")
            print(f"📁 Files processed: {len(doc_stats['files'])}")
        else:
            print("❌ Failed to load documents")
        
        return success
    
    def process_query(self, query: str) -> str:
        """Process a user query"""
        if not self.initialized:
            return "❌ System not initialized!"
        
        try:
            # Add user message to conversation
            self.conversation_handler.add_message("user", query)
            
            # Process through agent graph
            result = self.agent_graph.process_query_sync(query)
            
            # Get response
            response = result.get("final_response", "No response generated")
            
            # Add assistant response to conversation
            self.conversation_handler.add_message("assistant", response)
            
            # Update context memory if needed
            if result.get("retrieved_documents"):
                self.conversation_handler.update_context_memory(
                    "last_search_results", 
                    len(result["retrieved_documents"])
                )
            
            return response
            
        except Exception as e:
            error_msg = f"Error processing query: {e}"
            print(f"❌ {error_msg}")
            return error_msg
    
    def run_interactive_mode(self):
        """Run interactive terminal mode"""
        if not self.initialized:
            print("❌ System not initialized!")
            return
        
        print("\n🎯 INTERACTIVE MODE")
        print("="*60)
        print("Type your questions below. Commands:")
        print("  /help     - Show available commands")
        print("  /info     - Show system information")
        print("  /history  - Show conversation history")
        print("  /clear    - Clear conversation history")
        print("  /load     - Load documents from directory")
        print("  /stats    - Show detailed document statistics")
        print("  /reset    - Clear vector database")
        print("  /quit     - Exit the system")
        print("="*60)
        
        while True:
            try:
                # Get user input
                user_input = input("\n🤔 You: ").strip()
                
                if not user_input:
                    continue
                
                # Handle commands
                if user_input.startswith('/'):
                    if user_input == '/quit':
                        print("👋 Goodbye!")
                        break
                    elif user_input == '/help':
                        self._show_help()
                    elif user_input == '/info':
                        self._display_system_info()
                    elif user_input == '/history':
                        self._show_conversation_history()
                    elif user_input == '/clear':
                        self._clear_conversation()
                    elif user_input == '/load':
                        self._interactive_load_documents()
                    elif user_input == '/stats':
                        self._show_detailed_stats()
                    elif user_input == '/reset':
                        self._reset_vector_database()
                    else:
                        print("❌ Unknown command. Type /help for available commands.")
                    continue
                
                # Process query
                print("🤖 Assistant: ", end="", flush=True)
                response = self.process_query(user_input)
                print(response)
                
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
    
    def _show_help(self):
        """Show help information"""
        help_text = """
📖 AGENTIC RAG SYSTEM HELP

Available Commands:
  /help     - Show this help message
  /info     - Display system information and statistics
  /history  - Show recent conversation history
  /clear    - Clear conversation history and start fresh
  /load     - Load documents from a directory
  /stats    - Show detailed document statistics
  /reset    - Clear vector database (WARNING: Deletes all indexed documents)
  /quit     - Exit the system

Usage Tips:
• Ask questions about your loaded documents
• The system will ask for clarification if your question is too vague
• Use follow-up questions to dive deeper into topics
• The system remembers context from previous questions

Supported Document Types:
• Text files (.txt)
• Markdown files (.md)
• Python files (.py)
• JavaScript files (.js)
• JSON files (.json)
• CSV files (.csv)
• PDF files (.pdf) - requires PyPDF2 and PyMuPDF

Example Questions:
• "What is the main topic of the documentation?"
• "How do I configure the system?"
• "Can you explain the architecture?"
• "What are the key features mentioned?"
• "Summarize the content of the PDF files"
        """
        print(help_text)
    
    def _show_conversation_history(self):
        """Show conversation history"""
        if not self.conversation_handler.conversation_history:
            print("📝 No conversation history available.")
            return
        
        print("\n📝 CONVERSATION HISTORY")
        print("="*40)
        
        for i, msg in enumerate(self.conversation_handler.conversation_history[-10:], 1):
            role = msg["role"].title()
            content = msg["content"][:100] + "..." if len(msg["content"]) > 100 else msg["content"]
            timestamp = msg["timestamp"][:16]  # Show date and time only
            
            print(f"{i}. [{timestamp}] {role}: {content}")
        
        print("="*40)
    
    def _clear_conversation(self):
        """Clear conversation history"""
        self.conversation_handler.clear_history()
        print("🗑️  Conversation history cleared!")
    
    def _interactive_load_documents(self):
        """Interactive document loading"""
        default_path = Config.DOCUMENTS_PATH
        path = input(f"📁 Enter documents directory path (default: {default_path}): ").strip()
        
        if not path:
            path = default_path
        
        self.load_documents(path)
    
    def _show_detailed_stats(self):
        """Show detailed document statistics"""
        doc_stats = self.vector_store.get_document_statistics()
        
        print("\n📊 DETAILED DOCUMENT STATISTICS")
        print("="*50)
        print(f"Total Chunks: {doc_stats['total_chunks']}")
        
        if doc_stats['file_types']:
            print("\nBy File Type:")
            for file_type, count in sorted(doc_stats['file_types'].items()):
                percentage = (count / doc_stats['total_chunks']) * 100
                print(f"  {file_type}: {count} chunks ({percentage:.1f}%)")
        
        if doc_stats['files']:
            print(f"\nBy File ({len(doc_stats['files'])} files):")
            for filename, chunks in sorted(doc_stats['files'].items()):
                print(f"  {filename}: {chunks} chunks")
        
        print("="*50)
    
    def _reset_vector_database(self):
        """Reset the vector database"""
        confirm = input("⚠️  WARNING: This will delete ALL indexed documents. Type 'yes' to confirm: ")
        if confirm.lower() == 'yes':
            if self.vector_store.clear_collection():
                print("🗑️  Vector database cleared successfully!")
            else:
                print("❌ Failed to clear vector database")
        else:
            print("Operation cancelled.")

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Agentic RAG System")
    parser.add_argument("--load-docs", type=str, help="Load documents from directory")
    parser.add_argument("--query", type=str, help="Process a single query")
    parser.add_argument("--interactive", action="store_true", default=True, help="Run in interactive mode")
    
    args = parser.parse_args()
    
    # Create terminal interface
    rag_system = RAGTerminalInterface()
    
    # Initialize system
    if not rag_system.initialize():
        print("❌ Failed to initialize system. Exiting.")
        sys.exit(1)
    
    # Load documents if specified
    if args.load_docs:
        rag_system.load_documents(args.load_docs)
    
    # Process single query if specified
    if args.query:
        response = rag_system.process_query(args.query)
        print(f"\n🤖 Response: {response}")
        return
    
    # Run interactive mode
    if args.interactive:
        rag_system.run_interactive_mode()

if __name__ == "__main__":
    main()