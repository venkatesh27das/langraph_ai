"""
Configuration file for Data Dictionary Agent
"""
import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent
SAMPLE_DATA_DIR = PROJECT_ROOT / "sample_data"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
PROMPTS_DIR = PROJECT_ROOT / "prompts"

# Ensure output directory exists
OUTPUTS_DIR.mkdir(exist_ok=True)

# LMStudio Configuration
LMSTUDIO_BASE_URL = os.getenv("LMSTUDIO_BASE_URL", "http://localhost:1234/v1")
LMSTUDIO_MODEL = os.getenv("LMSTUDIO_MODEL", "local-model")  # Replace with your actual model name
LMSTUDIO_API_KEY = os.getenv("LMSTUDIO_API_KEY", "lm-studio")  # LMStudio typically uses this

# Data Analysis Configuration
MAX_UNIQUE_VALUES_DISPLAY = 20  # Max unique values to show in analysis
SAMPLE_SIZE_FOR_ANALYSIS = 1000  # Sample size for large datasets
MIN_CORRELATION_THRESHOLD = 0.3  # Minimum correlation to report

# LLM Configuration
MAX_TOKENS = 2048
TEMPERATURE = 0.1  # Low temperature for consistent outputs
TIMEOUT = 60  # Request timeout in seconds

# Supported file formats
SUPPORTED_FORMATS = ['.csv', '.xlsx', '.xls']

# Output formats
OUTPUT_FORMATS = {
    'json': 'application/json',
    'yaml': 'text/yaml',
    'markdown': 'text/markdown'
}

# Default output format
DEFAULT_OUTPUT_FORMAT = 'json'