# Data Dictionary Agent 📊

An AI-powered tool that automatically generates comprehensive data dictionaries from CSV and Excel files using LangGraph workflows and local LLMs via LMStudio.

## 🎯 Overview

This agent analyzes structured data files and creates detailed data dictionaries including:
- Column descriptions and semantic types
- Data quality assessments
- Business context inference
- Usage recommendations
- Compliance considerations
- Data lineage insights

## 🏗️ Architecture

Built using:
- **LangGraph**: Workflow orchestration and state management
- **LMStudio**: Local LLM integration for privacy and control
- **Pydantic**: Type-safe data models and validation
- **Pandas**: Data loading and profiling

## 🚀 Quick Start

### Prerequisites

1. **LMStudio Setup**
   - Download and install [LMStudio](https://lmstudio.ai/)
   - Start the local server (default: http://localhost:1234)

2. **Recommended Models** (tested with):
   - `qwen2.5-3b-instruct` / `qwen2.5-14b-instruct`
   - `deepseek-r1-distilled-qwen-8b`
   - `gemma-2-9b-it` / `gemma-2-27b-it`

### Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd data-dictionary-agent

# Install dependencies
pip install -r requirements.txt

# Create necessary directories
mkdir -p sample_data outputs
```

### Configuration

Edit `config.py` to match your LMStudio setup:

```python
# LMStudio Configuration
LMSTUDIO_BASE_URL = "http://localhost:1234/v1"
LMSTUDIO_MODEL = "qwen2.5-3b-instruct"  # Change to your loaded model
LMSTUDIO_API_KEY = "lm-studio"
```

### Basic Usage

```bash
# Analyze a CSV file
python main.py sample_data/customers.csv

# Specify output format and location
python main.py data.xlsx --format markdown --output my_dictionary.md

# Adjust sample size for large datasets
python main.py large_dataset.csv --sample 5000
```

## 📋 Command Line Options

```bash
python main.py <file_path> [options]

Options:
  --output, -o     Output file path (optional)
  --format, -f     Output format: json, yaml, markdown (default: json)
  --sample, -s     Sample size for analysis (default: 1000)
```

## 📁 Project Structure

```
data-dictionary-agent/
├── README.md
├── requirements.txt
├── config.py                    # Configuration settings
├── main.py                      # Entry point
├── agents/
│   ├── analyzer.py              # Data analysis agent
│   ├── generator.py             # Dictionary generation agent
│   └── workflow.py              # LangGraph workflow orchestration
├── core/
│   ├── data_loader.py           # CSV/Excel loading + profiling
│   ├── llm_client.py            # LMStudio integration
│   └── models.py                # Pydantic data models
├── prompts/
│   ├── analysis.py              # Data analysis prompts
│   └── generation.py            # Dictionary generation prompts
├── sample_data/
│   ├── customers.csv
│   └── sales.xlsx
└── outputs/                     # Generated dictionaries
    └── .gitkeep
```

## 🤖 Tested Models

### Performance Notes

| Model | Size | Speed | Quality | Memory | Notes |
|-------|------|-------|---------|---------|-------|
| Qwen3-4B | 4B | ⚡⚡⚡ | ⭐⭐⭐ | 4GB RAM | Fast, good for quick prototyping |
| Qwen3-14B | 14B | ⚡⚡ | ⭐⭐⭐⭐⭐ | 16GB RAM | Best balance of speed/quality |
| DeepSeek-R1-Qwen-8B | 8B | ⚡⚡ | ⭐⭐⭐⭐ | 10GB RAM | Excellent reasoning capabilities |
| Gemma-3-12B | 12B | ⚡⚡ | ⭐⭐⭐⭐ | 12GB RAM | Good instruction following |

### Model-Specific Tips

**Qwen Models:**
- Excellent for semantic type classification
- Good at inferring business context
- Works well with technical terminology

**DeepSeek-R1:**
- Superior reasoning for data quality assessment
- Excellent at identifying anomalies and patterns
- Good for complex business rule inference

**Gemma Models:**
- Strong instruction following for structured outputs
- Good for generating readable descriptions
- Reliable for compliance and documentation tasks

## 🛠️ Workflow Steps

The agent follows a 4-step LangGraph workflow:

1. **📊 Data Loading**: Load and validate CSV/Excel files
2. **🔍 Data Analysis**: Profile columns, detect patterns, assess quality
3. **📝 Dictionary Generation**: Create comprehensive documentation using LLM
4. **🎨 Output Formatting**: Format results in JSON, YAML, or Markdown

## 📄 Sample Output

### JSON Format
```json
{
  "metadata": {
    "source_file": "customers.csv",
    "total_rows": 10000,
    "total_columns": 15,
    "generated_at": "2024-01-15T10:30:00"
  },
  "dataset_description": "Customer database containing demographic and transaction information...",
  "column_definitions": [
    {
      "name": "customer_id",
      "description": "Unique identifier for each customer account",
      "data_type": "int64",
      "semantic_type": "identifier",
      "nullable": false,
      "sample_values": ["1001", "1002", "1003"],
      "business_context": "Primary key used across all customer-related systems..."
    }
  ],
  "data_quality": {
    "overall_score": 8.5,
    "completeness": 0.95,
    "issues": ["Missing values in phone_number column"],
    "recommendations": ["Implement phone number validation"]
  }
}
```

### Markdown Format
```markdown
# Data Dictionary

## Dataset Information
- **Source File**: customers.csv
- **Total Rows**: 10,000
- **Total Columns**: 15

## Column Definitions

### customer_id
**Description**: Unique identifier for each customer account
**Data Type**: int64
**Semantic Type**: identifier
```

## 🔧 Configuration Options

### LMStudio Settings
```python
# config.py
LMSTUDIO_BASE_URL = "http://localhost:1234/v1"
LMSTUDIO_MODEL = "your-model-name"
LMSTUDIO_API_KEY = "lm-studio"

# Analysis settings
DEFAULT_SAMPLE_SIZE = 1000
MAX_SAMPLE_VALUES = 10
SUPPORTED_FORMATS = ['.csv', '.xlsx', '.xls']
```

### Model-Specific Configurations

For different models, you may want to adjust:
- **Temperature**: 0.1-0.3 for consistent outputs
- **Max Tokens**: 2048-4096 depending on model
- **Context Window**: Ensure your prompts fit within model limits

## 🚨 Troubleshooting

### Common Issues

**LMStudio Connection Failed**
```bash
# Check if LMStudio server is running
curl http://localhost:1234/v1/models

# Verify model is loaded in LMStudio UI
```

**Memory Issues with Large Models**
- Use smaller models (3B-8B) for development
- Reduce sample size with `--sample` parameter
- Consider quantized model versions

**Poor Output Quality**
- Try different models (DeepSeek-R1 for reasoning, Qwen for general tasks)
- Adjust temperature in LMStudio (0.1-0.3 recommended)
- Ensure model is properly loaded and not just cached

**Large File Processing**
```bash
# Process large files with smaller samples
python main.py large_file.csv --sample 2000

# Or process in chunks (modify data_loader.py)
```

## 🤝 Contributing

This is a proof-of-concept project. Feel free to:
- Test with different models and share results
- Improve prompts for better outputs
- Add support for more file formats
- Enhance the LangGraph workflow

## 📋 Requirements

See `requirements.txt` for full dependencies:
- `langgraph` - Workflow orchestration
- `pandas` - Data manipulation
- `pydantic` - Data validation
- `openai` - LMStudio API client
- `openpyxl` - Excel file support
- `PyYAML` - YAML output format

## 🎯 Use Cases

- **Data Governance**: Automated documentation for compliance
- **Data Discovery**: Understanding new datasets quickly
- **Migration Projects**: Documenting legacy systems
- **Quality Assurance**: Identifying data issues proactively
- **Onboarding**: Helping new team members understand data

## 📈 Future Enhancements

- [ ] Support for database connections
- [ ] Interactive web interface
- [ ] Multi-language support
- [ ] Advanced statistical analysis
- [ ] Integration with data catalogs
- [ ] Automated data lineage detection

## 📄 License

This project is for educational and proof-of-concept purposes.

---

**Note**: This tool uses local LLMs for privacy and control. All data processing happens locally on your machine.