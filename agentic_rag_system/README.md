# Agentic RAG System

A proof-of-concept Retrieval-Augmented Generation (RAG) system using LangGraph for agentic workflows, LMStudio for local LLMs, and ChromaDB for vector storage. The system supports multi-turn conversations and intelligent clarification when queries are vague.

## 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Terminal UI   │◄──►│   Agent Graph    │◄──►│   LLM Client    │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │                        │
                                ▼                        ▼
                       ┌──────────────────┐    ┌─────────────────┐
                       │  Conversation    │    │    LMStudio     │
                       │    Handler       │    │  (localhost)    │
                       └──────────────────┘    └─────────────────┘
                                │
                                ▼
                       ┌──────────────────┐
                       │  Vector Store    │
                       │   (ChromaDB)     │
                       └──────────────────┘
```

## 🚀 Features

- **Agentic Workflow**: LangGraph-based intelligent decision making
- **Multi-turn Conversations**: Context-aware dialogue management
- **Smart Clarification**: Automatically asks clarifying questions for vague queries
- **Local LLMs**: Uses LMStudio for complete privacy and control
- **Vector Search**: ChromaDB for efficient document retrieval
- **Terminal Interface**: Clean command-line interaction
- **Document Support**: Multiple file formats (txt, md, py, js, json, csv)

## 📋 Prerequisites

1. **Python 3.9+** installed on your Mac
2. **LMStudio** downloaded and running
3. At least one **chat model** and **embedding model** loaded in LMStudio

### LMStudio Setup

1. Download LMStudio from [https://lmstudio.ai/](https://lmstudio.ai/)
2. Install and launch LMStudio
3. Download models (recommended):
   - **Chat Model**: `llama-3.2-3b-instruct` or similar
   - **Embedding Model**: `nomic-embed-text-v1.5` or similar
4. Start the local server (usually on `http://localhost:1234`)

## 🛠️ Installation

1. **Clone or create the project structure**:
```bash
mkdir agentic_rag_system
cd agentic_rag_system
```

2. **Create the folder structure**:
```bash
mkdir -p core prompts data/documents data/chroma_db
```

3. **Set up Python virtual environment**:
```bash
python -m venv venv
source venv/bin/activate  # On Mac/Linux
```

4. **Install dependencies**:
```bash
pip install -r requirements.txt
```

5. **Create environment file**:
```bash
cp .env.example .env
# Edit .env with your specific model names and settings
```

## 🎯 Usage

### Basic Usage

1. **Start the system**:
```bash
python main.py
```

2. **Load documents** (first time):
```bash
# In interactive mode, use:
/load
# Then enter your documents directory path
```

3. **Ask questions**:
```bash
🤔 You: What is the main topic of the documentation?
🤖 Assistant: Based on the documents I found...
```

### Command Line Options

```bash
# Load documents and run interactively
python main.py --load-docs ./data/documents

# Process a single query
python main.py --query "What is the system architecture?"

# Just run interactively (default)
python main.py --interactive
```

### Interactive Commands

- `/help` - Show available commands
- `/info` - Display system information
- `/history` - Show conversation history
- `/clear` - Clear conversation history
- `/load` - Load documents from directory
- `/quit` - Exit the system

## 📁 Adding Documents

1. Place your documents in the `data/documents/` directory
2. Supported formats: `.txt`, `.md`, `.py`, `.js`, `.json`, `.csv`
3. Use the `/load` command or restart with `--load-docs`

Example document structure:
```
data/documents/
├── api_documentation.md
├── setup_guide.txt
├── configuration.json
└── examples/
    ├── example1.py
    └── example2.js
```

## 🔧 Configuration

Edit `.env` file to customize:

```bash
# Model names (must match LMStudio loaded models)
CHAT_MODEL=llama-3.2-3b-instruct
EMBEDDING_MODEL=nomic-embed-text-v1.5

# RAG parameters
TOP_K_RESULTS=5
SIMILARITY_THRESHOLD=0.7
CHUNK_SIZE=1000

# Agent behavior
TEMPERATURE=0.7
MAX_TOKENS=2048
```

## 🧠 How It Works

### Agent Graph Workflow

1. **Query Analysis**: Analyzes user intent and context
2. **Document Retrieval**: Searches vector store for relevant documents
3. **Clarification Check**: Determines if query needs clarification
4. **Response Generation**: Creates contextual responses using retrieved documents

### Multi-turn Conversation

The system maintains conversation context and can:
- Handle follow-up questions
- Reference previous queries
- Maintain topic continuity
- Ask clarifying questions when needed

### Smart Clarification

The system asks for clarification when:
- No relevant documents are found
- Query is too vague or short
- Multiple interpretations are possible
- Context is insufficient

## 🐛 Troubleshooting

### LMStudio Connection Issues

```bash
# Check if LMStudio is running
curl http://localhost:1234/v1/models

# Verify model names in LMStudio match your .env file
```

### ChromaDB Issues

```bash
# Clear ChromaDB if corrupted
rm -rf data/chroma_db
# Restart the system to recreate
```

### Common Errors

1. **"Failed to connect to LMStudio"**
   - Ensure LMStudio is running
   - Check the server URL in `.env`
   - Verify models are loaded

2. **"No documents found"**
   - Check document directory path
   - Ensure supported file formats
   - Use `/load` command to reload

3. **"Embedding generation failed"**
   - Verify embedding model is loaded in LMStudio
   - Check model name in `.env`

## 🎨 Customization

### Adding New Document Types

Edit `vector_store.py`:
```python
supported_extensions = {'.txt', '.md', '.py', '.js', '.json', '.csv', '.your_ext'}
```

### Modifying Agent Behavior

Edit `agent_graph.py` to:
- Add new nodes to the workflow
- Modify clarification logic
- Enhance response generation

### Custom Prompts

Create new prompt templates in `prompts/` directory and import them in the agent graph.

## 📊 System Monitoring

The system provides:
- Document count and storage info
- Conversation history tracking
- Search result statistics
- Error logging

## 🔒 Privacy & Security

- All processing happens locally
- No data sent to external APIs
- Documents stored locally in ChromaDB
- LMStudio runs on your machine

## 🤝 Contributing

This is a proof-of-concept system. Feel free to extend it with:
- Additional document parsers
- Enhanced agent workflows
- Better clarification strategies
- UI improvements
- Production-ready features

## 📝 License

This project is for educational and proof-of-concept purposes.

---

**Need help?** Check the `/help` command in the interactive mode or review the troubleshooting section above.