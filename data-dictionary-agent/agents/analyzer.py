"""
Data Analysis Agent
Responsible for analyzing the structure and content of uploaded data
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import json

from core.models import ColumnProfile, DataProfile
from core.llm_client import LMStudioClient
from prompts.analysis import COLUMN_ANALYSIS_PROMPT, DATA_OVERVIEW_PROMPT
from config import MAX_UNIQUE_VALUES_DISPLAY, MIN_CORRELATION_THRESHOLD


@dataclass
class AnalysisResult:
    """Result of data analysis"""
    column_profiles: Dict[str, ColumnProfile]
    data_profile: DataProfile
    insights: List[str]
    correlations: Dict[str, Any]


class DataAnalyzer:
    """Agent responsible for analyzing data structure and generating insights"""
    
    def __init__(self, llm_client: LMStudioClient):
        self.llm_client = llm_client
    
    async def analyze(self, df: pd.DataFrame, sample_size: Optional[int] = None) -> AnalysisResult:
        """
        Analyze the dataframe and generate comprehensive insights
        """
        print("🔍 Starting data analysis...")
        
        # Sample data if it's too large
        if sample_size and len(df) > sample_size:
            df_sample = df.sample(n=sample_size, random_state=42)
            print(f"📊 Sampled {sample_size} rows from {len(df)} total rows")
        else:
            df_sample = df
        
        # Generate column profiles
        column_profiles = await self._analyze_columns(df_sample)
        
        # Generate data profile
        data_profile = self._generate_data_profile(df, df_sample)
        
        # Generate insights using LLM
        insights = await self._generate_insights(df_sample, column_profiles, data_profile)
        
        # Calculate correlations
        correlations = self._calculate_correlations(df_sample)
        
        return AnalysisResult(
            column_profiles=column_profiles,
            data_profile=data_profile,
            insights=insights,
            correlations=correlations
        )
    
    async def _analyze_columns(self, df: pd.DataFrame) -> Dict[str, ColumnProfile]:
        """Analyze each column in the dataframe"""
        profiles = {}
        
        for column in df.columns:
            print(f"  📋 Analyzing column: {column}")
            profile = await self._analyze_single_column(df, column)
            profiles[column] = profile
        
        return profiles
    
    async def _analyze_single_column(self, df: pd.DataFrame, column: str) -> ColumnProfile:
        """Analyze a single column"""
        series = df[column]
        
        # Basic statistics
        total_count = len(series)
        null_count = series.isnull().sum()
        unique_count = series.nunique()
        
        # Data type detection
        dtype = str(series.dtype)
        inferred_type = self._infer_semantic_type(series)
        
        # Unique values (limited)
        unique_values = series.dropna().unique()
        if len(unique_values) > MAX_UNIQUE_VALUES_DISPLAY:
            sample_unique = list(unique_values[:MAX_UNIQUE_VALUES_DISPLAY])
            sample_unique.append(f"... and {len(unique_values) - MAX_UNIQUE_VALUES_DISPLAY} more")
        else:
            sample_unique = list(unique_values)
        
        # Statistics for numeric columns
        statistics = {}
        if pd.api.types.is_numeric_dtype(series):
            statistics = {
                'mean': float(series.mean()) if not series.isna().all() else None,
                'median': float(series.median()) if not series.isna().all() else None,
                'std': float(series.std()) if not series.isna().all() else None,
                'min': float(series.min()) if not series.isna().all() else None,
                'max': float(series.max()) if not series.isna().all() else None,
                'quantiles': {
                    '25%': float(series.quantile(0.25)) if not series.isna().all() else None,
                    '75%': float(series.quantile(0.75)) if not series.isna().all() else None
                }
            }
        
        # Value patterns for string columns
        patterns = []
        if dtype == 'object' and not series.dropna().empty:
            patterns = self._detect_patterns(series.dropna())
        
        # Generate column description using LLM
        description = await self._generate_column_description(column, series, statistics, patterns)
        
        return ColumnProfile(
            name=column,
            data_type=dtype,
            inferred_type=inferred_type,
            total_count=total_count,
            null_count=null_count,
            unique_count=unique_count,
            sample_values=sample_unique,
            statistics=statistics,
            patterns=patterns,
            description=description
        )
    
    def _infer_semantic_type(self, series: pd.Series) -> str:
        """Infer semantic type of a column"""
        if pd.api.types.is_numeric_dtype(series):
            if series.dtype == 'int64' or series.dtype == 'int32':
                # Check if it could be an ID
                if series.nunique() == len(series.dropna()):
                    return "identifier"
                elif series.min() >= 0 and series.max() <= 1:
                    return "binary"
                else:
                    return "integer"
            else:
                return "numeric"
        elif pd.api.types.is_datetime64_any_dtype(series):
            return "datetime"
        elif series.dtype == 'bool':
            return "boolean"
        else:
            # String analysis
            sample_values = series.dropna().astype(str)
            if sample_values.empty:
                return "text"
            
            # Check for email patterns
            if sample_values.str.contains('@').any() and sample_values.str.contains(r'\.').any():
                return "email"
            
            # Check for URL patterns
            if sample_values.str.startswith(('http://', 'https://')).any():
                return "url"
            
            # Check for categorical (low cardinality)
            if series.nunique() / len(series.dropna()) < 0.05 and series.nunique() < 50:
                return "categorical"
            
            return "text"
    
    def _detect_patterns(self, series: pd.Series) -> List[str]:
        """Detect common patterns in string data"""
        patterns = []
        sample_str = series.astype(str)
        
        # Length patterns
        lengths = sample_str.str.len()
        if lengths.nunique() == 1:
            patterns.append(f"Fixed length: {lengths.iloc[0]} characters")
        
        # Common formats
        if sample_str.str.match(r'^\d{4}-\d{2}-\d{2}$').any():
            patterns.append("Date format: YYYY-MM-DD")
        
        if sample_str.str.match(r'^[A-Z]{2,3}\d+$').any():
            patterns.append("Code format: Letters followed by numbers")
        
        if sample_str.str.match(r'^\d+\.\d+$').any():
            patterns.append("Decimal format")
        
        return patterns
    
    async def _generate_column_description(self, column_name: str, series: pd.Series, 
                                         statistics: Dict, patterns: List[str]) -> str:
        """Generate column description using LLM"""
        try:
            # Prepare context for LLM
            context = {
                "column_name": column_name,
                "data_type": str(series.dtype),
                "total_values": len(series),
                "null_values": series.isnull().sum(),
                "unique_values": series.nunique(),
                "sample_values": list(series.dropna().head(5).astype(str)),
                "statistics": statistics,
                "patterns": patterns
            }
            
            prompt = COLUMN_ANALYSIS_PROMPT.format(
                column_name=column_name,
                context=json.dumps(context, indent=2)
            )
            
            response = await self.llm_client.generate(prompt)
            return response.strip()
        
        except Exception as e:
            print(f"Warning: Could not generate LLM description for column {column_name}: {e}")
            return f"Column containing {str(series.dtype)} data with {series.nunique()} unique values"
    
    def _generate_data_profile(self, df_full: pd.DataFrame, df_sample: pd.DataFrame) -> DataProfile:
        """Generate overall data profile"""
        return DataProfile(
            total_rows=len(df_full),
            total_columns=len(df_full.columns),
            sample_size=len(df_sample),
            data_types=df_full.dtypes.astype(str).to_dict(),
            missing_values_count=df_full.isnull().sum().to_dict(),
            memory_usage=df_full.memory_usage(deep=True).to_dict()
        )
    
    async def _generate_insights(self, df: pd.DataFrame, column_profiles: Dict[str, ColumnProfile], 
                               data_profile: DataProfile) -> List[str]:
        """Generate data insights using LLM"""
        try:
            # Prepare summary for LLM
            summary = {
                "dataset_shape": f"{data_profile.total_rows} rows × {data_profile.total_columns} columns",
                "column_types": data_profile.data_types,
                "missing_data": {k: v for k, v in data_profile.missing_values_count.items() if v > 0},
                "high_cardinality_columns": [name for name, profile in column_profiles.items() 
                                           if profile.unique_count > profile.total_count * 0.8],
                "low_cardinality_columns": [name for name, profile in column_profiles.items() 
                                          if profile.unique_count < 10],
            }
            
            prompt = DATA_OVERVIEW_PROMPT.format(
                summary=json.dumps(summary, indent=2)
            )
            
            response = await self.llm_client.generate(prompt)
            
            # Parse response into list of insights
            insights = [insight.strip() for insight in response.split('\n') if insight.strip()]
            return insights
        
        except Exception as e:
            print(f"Warning: Could not generate LLM insights: {e}")
            return ["Data analysis completed with basic profiling"]
    
    def _calculate_correlations(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate correlations for numeric columns"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) < 2:
            return {"message": "Not enough numeric columns for correlation analysis"}
        
        try:
            corr_matrix = df[numeric_cols].corr()
            
            # Find high correlations
            high_correlations = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    corr_val = corr_matrix.iloc[i, j]
                    if abs(corr_val) >= MIN_CORRELATION_THRESHOLD:
                        high_correlations.append({
                            "column1": corr_matrix.columns[i],
                            "column2": corr_matrix.columns[j],
                            "correlation": float(corr_val)
                        })
            
            return {
                "correlation_matrix": corr_matrix.to_dict(),
                "high_correlations": high_correlations,
                "numeric_columns": list(numeric_cols)
            }
        
        except Exception as e:
            return {"error": f"Could not calculate correlations: {str(e)}"}