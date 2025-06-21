"""
Pydantic models for Data Dictionary Agent
Defines all data structures used throughout the application
"""
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Union
from datetime import datetime
from enum import Enum


class SemanticType(str, Enum):
    """Semantic types for columns"""
    CATEGORICAL = "categorical"
    NUMERIC_CONTINUOUS = "numeric_continuous"
    NUMERIC_INTEGER = "numeric_integer"
    CATEGORICAL_NUMERIC = "categorical_numeric"
    IDENTIFIER = "identifier"
    TEXT = "text"
    TEXT_LONG = "text_long"
    TEXT_IDENTIFIER = "text_identifier"
    EMAIL = "email"
    PHONE = "phone"
    URL = "url"
    DATETIME = "datetime"
    MOSTLY_NULL = "mostly_null"
    UNKNOWN = "unknown"


class DataType(str, Enum):
    """Basic data types"""
    INTEGER = "int64"
    FLOAT = "float64"
    STRING = "object"
    BOOLEAN = "bool"
    DATETIME = "datetime64"
    UNKNOWN = "unknown"


class ColumnProfile(BaseModel):
    """Profile of a single column"""
    name: str = Field(..., description="Column name")
    data_type: str = Field(..., description="Pandas data type")
    semantic_type: str = Field(..., description="Inferred semantic type")
    total_count: int = Field(..., description="Total number of rows")
    null_count: int = Field(..., description="Number of null values")
    non_null_count: int = Field(..., description="Number of non-null values")
    unique_count: int = Field(..., description="Number of unique values")
    unique_ratio: float = Field(..., description="Ratio of unique values to total")
    sample_values: List[Any] = Field(default_factory=list, description="Sample values from column")
    statistics: Dict[str, Any] = Field(default_factory=dict, description="Statistical measures")
    quality_issues: List[str] = Field(default_factory=list, description="Identified quality issues")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class DataAnalysisResult(BaseModel):
    """Result of data analysis"""
    file_path: str = Field(..., description="Path to analyzed file")
    total_rows: int = Field(..., description="Total number of rows")
    total_columns: int = Field(..., description="Total number of columns")
    column_profiles: List[ColumnProfile] = Field(..., description="Individual column profiles")
    dataset_summary: Dict[str, Any] = Field(default_factory=dict, description="Overall dataset summary")
    analysis_timestamp: datetime = Field(default_factory=datetime.now, description="When analysis was performed")
    sample_size: int = Field(default=1000, description="Sample size used for analysis")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class ColumnDefinition(BaseModel):
    """Definition of a column for the data dictionary"""
    name: str = Field(..., description="Column name")
    description: str = Field(..., description="Human-readable description")
    data_type: str = Field(..., description="Technical data type")
    semantic_type: str = Field(..., description="Business/semantic type")
    nullable: bool = Field(..., description="Whether column can contain null values")
    sample_values: List[str] = Field(default_factory=list, description="Example values")
    constraints: List[str] = Field(default_factory=list, description="Data constraints or validation rules")
    business_context: Optional[str] = Field(None, description="Business context and usage")
    source_notes: Optional[str] = Field(None, description="Notes about data source")
    quality_score: Optional[float] = Field(None, description="Data quality score (0-1)")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class DataQuality(BaseModel):
    """Data quality assessment"""
    overall_score: float = Field(..., description="Overall quality score (0-1)")
    completeness: float = Field(..., description="Completeness score (0-1)")
    consistency: float = Field(..., description="Consistency score (0-1)")
    validity: float = Field(..., description="Validity score (0-1)")
    issues: List[str] = Field(default_factory=list, description="Identified quality issues")
    recommendations: List[str] = Field(default_factory=list, description="Quality improvement recommendations")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class DatasetMetadata(BaseModel):
    """Metadata about the dataset"""
    source_file: str = Field(..., description="Original file path")
    file_size_bytes: Optional[int] = Field(None, description="File size in bytes")
    generated_at: datetime = Field(default_factory=datetime.now, description="When dictionary was generated")
    generated_by: str = Field(default="Data Dictionary Agent", description="Tool that generated this")
    version: str = Field(default="1.0", description="Dictionary version")
    total_rows: int = Field(..., description="Total number of rows")
    total_columns: int = Field(..., description="Total number of columns")
    sample_size: int = Field(..., description="Sample size used for analysis")
    processing_time_seconds: Optional[float] = Field(None, description="Time taken to process")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class DataDictionary(BaseModel):
    """Complete data dictionary"""
    metadata: DatasetMetadata = Field(..., description="Dataset metadata")
    dataset_description: str = Field(..., description="Overall description of the dataset")
    column_definitions: List[ColumnDefinition] = Field(..., description="Definitions for each column")
    data_quality: DataQuality = Field(..., description="Data quality assessment")
    insights: List[str] = Field(default_factory=list, description="Key insights about the dataset")
    usage_recommendations: List[str] = Field(default_factory=list, description="Recommendations for using this data")
    glossary: Dict[str, str] = Field(default_factory=dict, description="Business terms and definitions")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class ValidationResult(BaseModel):
    """Result of data validation"""
    is_valid: bool = Field(..., description="Whether data passes validation")
    errors: List[str] = Field(default_factory=list, description="Validation errors")
    warnings: List[str] = Field(default_factory=list, description="Validation warnings")
    column_results: Dict[str, Dict[str, Any]] = Field(default_factory=dict, description="Per-column validation results")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class AnalysisConfig(BaseModel):
    """Configuration for data analysis"""
    sample_size: int = Field(default=1000, description="Number of rows to sample")
    max_unique_values: int = Field(default=50, description="Max unique values to consider as categorical")
    null_threshold: float = Field(default=0.5, description="Threshold for considering column mostly null")
    outlier_threshold: float = Field(default=0.05, description="Threshold for outlier detection")
    text_length_threshold: int = Field(default=20, description="Threshold for long text classification")
    enable_llm_analysis: bool = Field(default=True, description="Whether to use LLM for analysis")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class WorkflowState(BaseModel):
    """State object for workflow tracking"""
    step: str = Field(..., description="Current workflow step")
    status: str = Field(..., description="Current status")
    progress: float = Field(default=0.0, description="Progress percentage (0-1)")
    message: Optional[str] = Field(None, description="Status message")
    error: Optional[str] = Field(None, description="Error message if any")
    data: Dict[str, Any] = Field(default_factory=dict, description="Step-specific data")
    timestamp: datetime = Field(default_factory=datetime.now, description="Last update timestamp")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class ColumnInsight(BaseModel):
    """Insight about a specific column"""
    column_name: str = Field(..., description="Name of the column")
    insight_type: str = Field(..., description="Type of insight")
    description: str = Field(..., description="Description of the insight")
    severity: str = Field(default="info", description="Severity level: info, warning, error")
    recommendation: Optional[str] = Field(None, description="Recommended action")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class DatasetInsight(BaseModel):
    """Overall dataset insight"""
    insight_type: str = Field(..., description="Type of insight")
    title: str = Field(..., description="Insight title")
    description: str = Field(..., description="Detailed description")
    affected_columns: List[str] = Field(default_factory=list, description="Columns affected by this insight")
    impact: str = Field(default="medium", description="Impact level: low, medium, high")
    recommendation: Optional[str] = Field(None, description="Recommended action")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class ProfilerSettings(BaseModel):
    """Settings for data profiling"""
    calculate_correlations: bool = Field(default=False, description="Whether to calculate column correlations")
    detect_pii: bool = Field(default=True, description="Whether to detect potential PII")
    generate_histograms: bool = Field(default=False, description="Whether to generate distribution data")
    include_value_counts: bool = Field(default=True, description="Whether to include value frequency counts")
    max_categorical_cardinality: int = Field(default=20, description="Max cardinality for categorical analysis")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }