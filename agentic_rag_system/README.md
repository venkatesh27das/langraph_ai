# Agentic RAG System

A sophisticated Retrieval-Augmented Generation (RAG) system built with LangGraph for intelligent agentic workflows, LMStudio for local LLMs, and ChromaDB for vector storage. Features multi-turn conversations, smart clarification handling, and both terminal and web interfaces.

## 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│ Terminal/Web UI │◄──►│   Agent Graph    │◄──►│   LLM Client    │
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

### Core Capabilities
- **🤖 Agentic Workflow**: LangGraph-based intelligent decision making with multi-step reasoning
- **💬 Multi-turn Conversations**: Context-aware dialogue management with conversation history
- **🧠 Smart Clarification**: Automatically asks clarifying questions for vague or ambiguous queries
- **🔒 Complete Privacy**: Local LLMs via LMStudio - no data sent to external APIs
- **🔍 Intelligent Retrieval**: ChromaDB vector search with similarity thresholds and filtering
- **🖥️ Dual Interface**: Clean terminal interface + modern Streamlit web UI
- **📄 Multi-format Support**: txt, md, py, js, json, csv, pdf documents
- **🎯 Thinking Model Support**: Enhanced reasoning with configurable thinking model behavior

### Advanced Features
- **📚 Document Statistics**: Comprehensive analytics on indexed documents
- **🔄 Real-time Updates**: Live system status and collection information
- **⚡ Quick Actions**: Easy document management and system controls
- **🛠️ System Diagnostics**: Built-in health checks and troubleshooting
- **📤 File Upload**: Drag-and-drop document uploading in web interface
- **🎨 Modern UI**: Responsive design with intuitive navigation

## 📋 Prerequisites

1. **Python 3.9+** installed on your system
2. **LMStudio** downloaded and running locally
3. At least one **chat model** and **embedding model** loaded in LMStudio

### LMStudio Setup

1. Download LMStudio from [https://lmstudio.ai/](https://lmstudio.ai/)
2. Install and launch LMStudio
3. Download and load models (see tested models below)
4. Start the local server (default: `http://localhost:1234`)

### ✅ Tested Models

The system has been thoroughly tested with these model combinations:

#### **Chat Models (LLMs)**
- **Gemma-3-12B** - Excellent performance, great reasoning
- **Qwen-3-14B** - Strong multilingual support, detailed responses  
- **DeepSeek-R1-Qwen3-8B** - Fast inference, good for thinking workflows
- **Qwen-3-4B** - Lightweight option, good for resource-constrained systems

#### **Embedding Models**
- **Nomic-embed-v1.5** - Recommended, excellent retrieval quality
- **all-MiniLM-v6** - Faster alternative, good for smaller documents

> **💡 Recommendation**: For best results, use **Qwen-3-14B** with **Nomic-embed-v1.5**

## 🛠️ Installation

### Quick Setup

1. **Clone the repository**:
```bash
git clone <repository-url>
cd agentic_rag_system
```

2. **Create virtual environment**:
```bash
python -m venv venv
source venv/bin/activate  # On Mac/Linux
# or
venv\Scripts\activate     # On Windows
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

4. **Configure environment**:
```bash
cp .env.example .env
# Edit .env with your model names
```

### Project Structure
```
agentic_rag_system/
├── core/                   # Core system components
│   ├── __init__.py
│   ├── agent_graph.py     # LangGraph workflow
│   ├── llm_client.py      # LMStudio client
│   ├── vector_store.py    # ChromaDB integration
│   └── conversation.py    # Conversation management
├── prompts/               # Prompt templates
│   ├── system_prompts.py
│   └── clarification.py
├── data/
│   ├── documents/         # Your documents here
│   └── chroma_db/        # Vector database (auto-created)
├── config.py             # Configuration settings
├── main.py              # Terminal interface
├── streamlit_app.py     # Web interface
└── requirements.txt
```

## 🎯 Usage

### Web Interface (Recommended)

1. **Start the web interface**:
```bash
streamlit run streamlit_app.py
```

2. **Open your browser** to `http://localhost:8501`

3. **Initialize the system** using the sidebar button

4. **Upload documents** via the Documents tab or load from directory

5. **Start chatting** with your documents!

### Terminal Interface

1. **Start the terminal interface**:
```bash
python main.py
```

2. **Load documents** (first time):
```bash
# In interactive mode
/load
# Enter your documents directory path when prompted
```

3. **Ask questions**:
```bash
🤔 You: What is the main architecture of this system?
🤖 Assistant: Based on the documentation, the system uses a multi-component architecture...
```

### Command Line Options

```bash
# Load documents and run interactively
python main.py --load-docs ./data/documents

# Process a single query
python main.py --query "Explain the agent graph workflow"

# Interactive mode (default)
python main.py --interactive
```

### Interactive Commands (Terminal)

- `/help` - Show all available commands
- `/info` - Display system information and statistics
- `/history` - Show conversation history
- `/clear` - Clear conversation history
- `/load` - Load documents from directory
- `/quit` - Exit the system

## 📁 Document Management

### Supported Formats
- **Text**: `.txt`, `.md`
- **Code**: `.py`, `.js`, `.json`
- **Data**: `.csv`
- **Documents**: `.pdf` (via web interface)

### Adding Documents

#### Option 1: Directory Loading
```bash
data/documents/
├── api_documentation.md
├── setup_guide.txt
├── configuration.json
├── code_examples/
│   ├── example1.py
│   └── example2.js
└── data_files/
    └── sample_data.csv
```

#### Option 2: Web Upload (Streamlit)
- Use the **Documents** tab in the web interface
- Drag and drop files or click to browse
- Supports batch uploads

## 🔧 Configuration

### Environment Variables (`.env`)

```bash
# Model Configuration (must match LMStudio loaded models)
CHAT_MODEL=qwen-3-14b-instruct
EMBEDDING_MODEL=nomic-embed-text-v1.5

# LMStudio Settings
LMSTUDIO_BASE_URL=http://localhost:1234

# RAG Parameters
TOP_K_RESULTS=5
SIMILARITY_THRESHOLD=0.7
CHUNK_SIZE=1000
CHUNK_OVERLAP=200

# Generation Settings
TEMPERATURE=0.7
MAX_TOKENS=2048

# Thinking Model Features
USE_THINKING_MODEL=true
THINKING_MODEL_PROMPT_ADDITION="Think through this step by step, then provide your final answer."

# Paths
DOCUMENTS_PATH=./data/documents
CHROMA_DB_PATH=./data/chroma_db
```

### Model-Specific Optimizations

#### For Gemma-3-12B:
```bash
TEMPERATURE=0.6
MAX_TOKENS=4096
CHUNK_SIZE=1200
```

#### For Qwen-3-4B (Resource Constrained):
```bash
TEMPERATURE=0.8
MAX_TOKENS=1024
CHUNK_SIZE=800
TOP_K_RESULTS=3
```

#### For DeepSeek-R1-Qwen3-8B (Thinking Model):
```bash
USE_THINKING_MODEL=true
TEMPERATURE=0.5
MAX_TOKENS=3072
```

## 🧠 How It Works

### Agent Graph Workflow

The system uses a sophisticated LangGraph workflow:

1. **📝 Query Analysis**: Analyzes user intent, context, and clarity
2. **🔍 Document Retrieval**: Enhanced similarity search with context awareness
3. **❓ Clarification Check**: Intelligent decision on whether clarification is needed
4. **💡 Response Generation**: Context-aware response with source citations
5. **🔄 Conversation Management**: Maintains context across multiple turns

### Multi-turn Conversation Features

- **Context Continuity**: References previous questions and answers
- **Follow-up Handling**: Understands related questions without re-explanation
- **Memory Management**: Efficient context window management
- **Topic Tracking**: Maintains conversation topics across sessions

### Smart Clarification System

The system asks for clarification when:
- **🔍 No Documents Found**: Query doesn't match any indexed content
- **❓ Ambiguous Intent**: Multiple possible interpretations exist
- **📝 Insufficient Context**: Query is too vague or incomplete
- **🎯 User Request**: Explicit requests for clarification or explanation

### Thinking Model Integration

When enabled, the system:
- **🧠 Enhanced Reasoning**: Uses step-by-step thinking for complex queries
- **🎯 Better Analysis**: Improved query intent understanding
- **💭 Transparent Process**: Shows reasoning in development mode
- **⚡ Automatic Cleanup**: Removes thinking content from final responses

## 🐛 Troubleshooting

### Common Issues

#### 🔌 LMStudio Connection
```bash
# Test connection
curl http://localhost:1234/v1/models

# Common fixes:
- Ensure LMStudio is running
- Check if models are loaded
- Verify port in .env matches LMStudio
- Restart LMStudio if needed
```

#### 🗄️ ChromaDB Issues
```bash
# Clear corrupted database
rm -rf data/chroma_db
# System will recreate on next startup
```

#### 📄 Document Loading
```bash
# Check file permissions
ls -la data/documents/

# Verify supported formats
# Supported: .txt, .md, .py, .js, .json, .csv, .pdf
```

#### 🤖 Model Performance
- **Slow responses**: Try smaller models (Qwen-3-4B)
- **Poor quality**: Increase temperature, try larger models
- **Out of memory**: Reduce MAX_TOKENS and CHUNK_SIZE

### Error Messages

| Error | Cause | Solution |
|-------|-------|----------|
| "Failed to connect to LMStudio" | LMStudio not running | Start LMStudio and load models |
| "No documents found" | Empty or inaccessible directory | Check document path and permissions |
| "Embedding generation failed" | Wrong embedding model name | Verify model name in LMStudio |
| "Context length exceeded" | Query too long | Reduce MAX_TOKENS or clear history |

## 📊 System Monitoring

### Terminal Interface
- Real-time document count and statistics
- Conversation history tracking
- Search result quality metrics
- Error logging with timestamps

### Web Interface
- **📊 System Dashboard**: Live status indicators
- **📈 Document Analytics**: File type distribution, chunk statistics
- **🔍 Diagnostic Tools**: Built-in system health checks
- **⚡ Performance Metrics**: Response times and success rates

## 🎨 Customization

### Adding New Document Types

Edit `core/vector_store.py`:
```python
supported_extensions = {
    '.txt', '.md', '.py', '.js', '.json', '.csv', '.pdf',
    '.yaml', '.yml', '.xml'  # Add new types
}
```

### Custom Prompts

Create new templates in `prompts/`:
```python
# prompts/custom_prompts.py
class CustomPrompts:
    SPECIALIZED_PROMPT = """
    Your custom prompt template here...
    """
```

### Agent Workflow Modifications

Extend `core/agent_graph.py`:
```python
# Add new nodes
workflow.add_node("custom_analysis", self._custom_analysis)

# Add new conditional logic
workflow.add_conditional_edges(
    "custom_analysis",
    self._custom_condition,
    {"option1": "node1", "option2": "node2"}
)
```

## 🔒 Privacy & Security

- **🏠 Fully Local**: All processing happens on your machine
- **🚫 No External APIs**: No data sent to cloud services  
- **🔐 Data Control**: Complete ownership of your documents and conversations
- **🛡️ Secure Storage**: Local ChromaDB with no external dependencies

## 🚀 Performance Tips

### For Large Document Collections (1000+ files):
```bash
# Optimize chunking
CHUNK_SIZE=800
CHUNK_OVERLAP=100
TOP_K_RESULTS=3

# Use faster embedding model
EMBEDDING_MODEL=all-minilm-v6
```

### For Better Response Quality:
```bash
# Use larger chat model
CHAT_MODEL=qwen-3-14b-instruct

# Enable thinking model
USE_THINKING_MODEL=true

# Increase context
TOP_K_RESULTS=7
SIMILARITY_THRESHOLD=0.6
```

### For Faster Responses:
```bash
# Use smaller models
CHAT_MODEL=qwen-3-4b-instruct
EMBEDDING_MODEL=all-minilm-v6

# Reduce generation length
MAX_TOKENS=1024
TOP_K_RESULTS=3
```

## 🤝 Contributing

This system is designed to be extensible. Areas for contribution:

- **🔌 New Integrations**: Additional LLM providers, vector stores
- **📄 Document Parsers**: Support for more file formats
- **🎨 UI Enhancements**: Better visualizations, mobile support
- **🧠 Agent Improvements**: More sophisticated reasoning workflows
- **📊 Analytics**: Advanced performance monitoring
- **🌐 Deployment**: Docker, cloud deployment options

## 📚 Advanced Usage

### Batch Processing
```python
# Process multiple queries programmatically
from core import AgentGraph, LMStudioClient, VectorStore, ConversationHandler

# Initialize system
client = LMStudioClient()
store = VectorStore(client)
handler = ConversationHandler()
agent = AgentGraph(client, store, handler)

# Process queries
queries = ["Query 1", "Query 2", "Query 3"]
for query in queries:
    result = agent.process_query_sync(query)
    print(f"Q: {query}\nA: {result['final_response']}\n")
```

### Custom Evaluation
```python
# Evaluate system performance
def evaluate_responses(test_queries, expected_responses):
    scores = []
    for query, expected in zip(test_queries, expected_responses):
        result = agent.process_query_sync(query)
        # Add your evaluation logic here
        score = calculate_similarity(result['final_response'], expected)
        scores.append(score)
    return scores
```

## 📝 License

This project is open source and available for everyone.

---

## 🎯 Quick Start Checklist

- [ ] Install Python 3.9+
- [ ] Download and setup LMStudio
- [ ] Load compatible models (see tested models above)
- [ ] Clone repository and install dependencies
- [ ] Configure `.env` file with your model names
- [ ] Add documents to `data/documents/`
- [ ] Run `streamlit run streamlit_app.py` or `python main.py`
- [ ] Initialize system and start chatting!

**Need help?** Check the troubleshooting section or run system diagnostics in the web interface.