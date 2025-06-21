"""
Prompts package for Data Dictionary Agent
Contains all LLM prompts for analysis and generation
"""

from .analysis import DataAnalysisPrompts
from .generation import DictionaryGenerationPrompts

__all__ = ["DataAnalysisPrompts", "DictionaryGenerationPrompts"]