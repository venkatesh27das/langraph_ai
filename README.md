# LangGraph AI Projects

This repository contains a collection of AI projects built using LangGraph, a framework for building stateful, multi-actor applications with LLMs.

## Projects

### 1. SQL Generator Agent (`query_to_insights/`)

A powerful natural language to SQL conversion agent that uses LM Studio for generating SQL queries from natural language input. This project demonstrates how to:

- Convert natural language queries to SQL
- Interact with SQLite databases
- Use LM Studio for local LLM inference
- Implement a LangGraph-based agent workflow

For detailed documentation and setup instructions, see the [SQL Generator Agent README](query_to_insights/README.md).

## Getting Started

Each project in this repository has its own setup instructions and requirements. Please refer to the individual project READMEs for specific setup steps.

## Requirements

- Python 3.8+
- LangGraph
- LM Studio (for SQL Generator Agent)
- Other dependencies as specified in individual project requirements

## Project Structure

```
langraph_ai/
├── query_to_insights/     # SQL Generator Agent project
│   ├── src/              # Source code
│   ├── examples/         # Example scripts and data
│   ├── main.py          # Main entry point
│   └── README.md        # Project documentation
└── README.md            # This file
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.