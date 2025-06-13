#!/bin/bash

# Create the root directory
mkdir -p agentic_rag_system

# Create root level files
touch agentic_rag_system/README.md
touch agentic_rag_system/requirements.txt
touch agentic_rag_system/.env.example
touch agentic_rag_system/.gitignore
touch agentic_rag_system/main.py
touch agentic_rag_system/config.py

# Create core directory and files
mkdir -p agentic_rag_system/core
touch agentic_rag_system/core/__init__.py
touch agentic_rag_system/core/agent_graph.py
touch agentic_rag_system/core/vector_store.py
touch agentic_rag_system/core/llm_client.py
touch agentic_rag_system/core/conversation.py

# Create prompts directory and files
mkdir -p agentic_rag_system/prompts
touch agentic_rag_system/prompts/__init__.py
touch agentic_rag_system/prompts/system_prompts.py
touch agentic_rag_system/prompts/clarification.py

# Create data directories
mkdir -p agentic_rag_system/data/documents
mkdir -p agentic_rag_system/data/chroma_db

echo "Agentic RAG System folder structure created successfully!"