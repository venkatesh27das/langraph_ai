# BI Chatbot POC

A proof-of-concept **agentic AI chatbot** for Business Intelligence that can answer queries from three knowledge sources: unstructured documents (RAG), structured data (SQL), and general knowledge (LLM). Built with **LangGraph** and **LM Studio**.

## 🚀 Quick Start

1. **Setup LM Studio**
   ```bash
   # Download and start LM Studio
   # Load a model (e.g., Llama 2 7B, Mistral 7B)
   # Start the local server on http://localhost:1234
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure**
   ```bash
   # Edit config.yaml with your settings
   cp config.yaml.example config.yaml
   ```

4. **Setup Data**
   ```bash
   python scripts/setup.py
   python scripts/generate_data_dict.py
   ```

5. **Run the App**
   ```bash
   streamlit run app.py
   ```

## 🏗️ Architecture

```
User Query → Intent Classification → Route to Agent → Generate Response
                    ↓
            [RAG | SQL | General | Clarification]
                    ↓
            Response + Visualizations (if applicable)
```

### Agents
- **RAG Agent**: Searches documents using vector similarity
- **SQL Agent**: Converts natural language to SQL queries
- **General Agent**: Uses LLM for general knowledge questions
- **Clarification Agent**: Handles vague queries and asks follow-ups

## 📁 Project Structure

```
bi-chatbot-poc/
├── app.py                          # Main Streamlit interface
├── config.yaml                     # Configuration settings
├── requirements.txt                 # Python dependencies
│
├── src/
│   ├── agents/                     # LangGraph agents
│   │   ├── orchestrator.py         # Main workflow orchestrator
│   │   ├── rag_agent.py            # Document search & retrieval
│   │   ├── sql_agent.py            # SQL generation & execution
│   │   ├── general_agent.py        # General knowledge queries
│   │   └── clarification_agent.py  # Query clarification
│   │
│   ├── core/                       # Core components
│   │   ├── rag_pipeline.py         # Document processing & embeddings
│   │   ├── sql_generator.py        # Natural language to SQL
│   │   ├── data_dictionary.py      # Auto-generated schema info
│   │   ├── lm_studio_client.py     # LM Studio API client
│   │   ├── chart_generator.py      # Data visualization
│   │   └── conversation_memory.py  # Multi-turn conversations
│   │
│   ├── prompts.py                  # LLM prompt templates
│   └── utils.py                    # Helper functions
│
├── data/
│   ├── documents/                  # PDF, TXT, MD files
│   ├── sample_db.sqlite           # Sample database
│   └── embeddings/                # Vector store
│
├── scripts/
│   ├── setup.py                   # Initial setup
│   └── generate_data_dict.py      # Auto data dictionary
│
└── notebooks/
    └── experimentation.ipynb      # Testing & exploration
```

## 🔧 Configuration

Edit `config.yaml`:

```yaml
lm_studio:
  base_url: "http://localhost:1234/v1"
  model: "local-model"

database:
  path: "data/sample_db.sqlite"
  
rag:
  documents_path: "data/documents"
  embeddings_path: "data/embeddings"
  chunk_size: 1000
```

## 💡 Example Queries

### 📄 Document Questions (RAG)
- "What does our employee handbook say about vacation policies?"
- "Show me the Q3 performance highlights"
- "What are the key findings from the market research?"

### 📊 Data Questions (SQL)
- "What were our total sales last month?"
- "Show me the top 5 customers by revenue"
- "How many new customers did we acquire this quarter?"

### ❓ General Questions
- "What is business intelligence?"
- "Explain KPIs vs metrics"
- "Best practices for data visualization"

## 🎯 Key Features

- **Multi-source Intelligence**: RAG + SQL + General Knowledge
- **Agentic Architecture**: LangGraph-powered workflow
- **Local LLM**: Privacy-first with LM Studio
- **Smart Routing**: Automatic intent classification
- **Clarification Handling**: Asks follow-ups for vague queries
- **Multi-turn Conversations**: Maintains context across exchanges
- **Auto Data Dictionary**: LLM-generated schema documentation
- **Visual Insights**: Automatic chart generation from SQL results
- **Interactive UI**: Streamlit-based chat interface

## 🔄 Workflow

1. **User Input** → Intent classification
2. **Route to Agent** based on query type
3. **Process Query**:
   - RAG: Retrieve relevant documents → Generate answer
   - SQL: Generate SQL → Execute → Create insights + charts
   - General: Direct LLM response
   - Clarification: Ask follow-up questions
4. **Return Response** with context and visualizations

## 🛠️ Setup Details

### Data Dictionary Generation
```bash
python scripts/generate_data_dict.py
# Automatically analyzes your database schema
# Generates descriptions and relationships using LLM
```

### Adding Documents
```bash
# Place PDF, TXT, or MD files in data/documents/
# Run setup to process and embed documents
python scripts/setup.py --rebuild-embeddings
```

### Custom Database
```bash
# Replace data/sample_db.sqlite with your database
# Update config.yaml with the new path
# Regenerate data dictionary
```

## 🚨 Limitations (POC)

- Single SQLite database support
- Basic error handling
- Limited visualization types
- No user authentication
- Local deployment only
- Simplified intent classification

## 📋 Requirements

- Python 3.8+
- LM Studio with local model
- 4GB+ RAM for embeddings
- SQLite database

## 🤝 Contributing

This is a proof-of-concept. Feel free to:
- Test different LLM models
- Add new data sources
- Improve visualization types
- Enhance error handling
- Experiment with prompt engineering

## 📄 License

MIT License - See LICENSE file for details