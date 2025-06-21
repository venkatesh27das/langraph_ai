"""
Data Loader for CSV/Excel files with basic profiling
Handles file loading and initial data profiling
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
import logging
from datetime import datetime

from .models import ColumnProfile

logger = logging.getLogger(__name__)


class DataLoader:
    """Loads and profiles CSV/Excel data files"""
    
    def __init__(self):
        self.supported_formats = ['.csv', '.xlsx', '.xls']
    
    async def load_data(self, file_path: str) -> Optional[pd.DataFrame]:
        """
        Load data from CSV or Excel file
        
        Args:
            file_path: Path to the data file
            
        Returns:
            pandas DataFrame or None if loading fails
        """
        try:
            file_path = Path(file_path)
            
            if not file_path.exists():
                logger.error(f"File not found: {file_path}")
                return None
            
            if file_path.suffix.lower() not in self.supported_formats:
                logger.error(f"Unsupported format: {file_path.suffix}")
                return None
            
            # Load based on file type
            if file_path.suffix.lower() == '.csv':
                df = pd.read_csv(file_path, encoding='utf-8-sig')
            else:  # Excel files
                df = pd.read_excel(file_path)
            
            logger.info(f"Loaded {len(df)} rows and {len(df.columns)} columns from {file_path}")
            return df
            
        except Exception as e:
            logger.error(f"Error loading file {file_path}: {str(e)}")
            return None
    
    def profile_dataframe(self, df: pd.DataFrame, sample_size: int = 1000) -> List[ColumnProfile]:
        """
        Profile all columns in the dataframe
        
        Args:
            df: Input dataframe
            sample_size: Number of rows to sample for analysis
            
        Returns:
            List of ColumnProfile objects
        """
        try:
            # Sample data if too large
            if len(df) > sample_size:
                df_sample = df.sample(n=sample_size, random_state=42)
                logger.info(f"Sampling {sample_size} rows from {len(df)} total rows")
            else:
                df_sample = df.copy()
            
            profiles = []
            
            for column in df.columns:
                profile = self._profile_column(df[column], df_sample[column])
                profiles.append(profile)
            
            return profiles
            
        except Exception as e:
            logger.error(f"Error profiling dataframe: {str(e)}")
            return []
    
    def _profile_column(self, full_column: pd.Series, sample_column: pd.Series) -> ColumnProfile:
        """
        Profile a single column
        
        Args:
            full_column: Full column data for counts
            sample_column: Sampled column data for analysis
            
        Returns:
            ColumnProfile object
        """
        try:
            # Basic info
            column_name = full_column.name
            total_count = len(full_column)
            null_count = full_column.isnull().sum()
            non_null_count = total_count - null_count
            
            # Data type detection
            inferred_type = str(sample_column.dtype)
            semantic_type = self._infer_semantic_type(sample_column)
            
            # Sample values (non-null)
            sample_values = sample_column.dropna().head(10).tolist()
            
            # Unique values info
            unique_count = sample_column.nunique()
            unique_ratio = unique_count / len(sample_column) if len(sample_column) > 0 else 0
            
            # Statistics based on data type
            statistics = self._calculate_column_statistics(sample_column)
            
            # Data quality indicators
            quality_issues = self._identify_quality_issues(sample_column)
            
            return ColumnProfile(
                name=column_name,
                data_type=inferred_type,
                semantic_type=semantic_type,
                total_count=total_count,
                null_count=null_count,
                non_null_count=non_null_count,
                unique_count=unique_count,
                unique_ratio=unique_ratio,
                sample_values=sample_values,
                statistics=statistics,
                quality_issues=quality_issues
            )
            
        except Exception as e:
            logger.error(f"Error profiling column {full_column.name}: {str(e)}")
            # Return basic profile on error
            return ColumnProfile(
                name=full_column.name,
                data_type="unknown",
                semantic_type="unknown",
                total_count=len(full_column),
                null_count=full_column.isnull().sum(),
                non_null_count=len(full_column) - full_column.isnull().sum(),
                unique_count=0,
                unique_ratio=0.0,
                sample_values=[],
                statistics={},
                quality_issues=[]
            )
    
    def _infer_semantic_type(self, column: pd.Series) -> str:
        """
        Infer semantic type of column based on content
        
        Args:
            column: Pandas series
            
        Returns:
            Semantic type string
        """
        try:
            # Skip if mostly null
            if column.isnull().sum() / len(column) > 0.9:
                return "mostly_null"
            
            # Check for datetime
            if pd.api.types.is_datetime64_any_dtype(column):
                return "datetime"
            
            # Check for numeric types
            if pd.api.types.is_numeric_dtype(column):
                # Check if it could be categorical (few unique values)
                if column.nunique() <= 20 and column.nunique() / len(column) < 0.1:
                    return "categorical_numeric"
                
                # Check if integers could be IDs
                if pd.api.types.is_integer_dtype(column):
                    if column.nunique() / len(column) > 0.8:  # High uniqueness
                        return "identifier"
                    return "numeric_integer"
                
                return "numeric_continuous"
            
            # String/object types
            if pd.api.types.is_object_dtype(column):
                non_null = column.dropna()
                
                if len(non_null) == 0:
                    return "text"
                
                # Check for email patterns
                if non_null.astype(str).str.contains(r'@.*\.', regex=True).any():
                    return "email"
                
                # Check for phone patterns
                if non_null.astype(str).str.contains(r'[\d\-\(\)\+\s]{7,}', regex=True).sum() > len(non_null) * 0.5:
                    return "phone"
                
                # Check for URL patterns
                if non_null.astype(str).str.contains(r'http[s]?://', regex=True).any():
                    return "url"
                
                # Check if categorical (limited unique values)
                unique_ratio = column.nunique() / len(column)
                if unique_ratio < 0.1 and column.nunique() <= 50:
                    return "categorical"
                
                # Check for high cardinality text (could be names, descriptions)
                if unique_ratio > 0.8:
                    avg_length = non_null.astype(str).str.len().mean()
                    if avg_length > 20:
                        return "text_long"
                    else:
                        return "text_identifier"
                
                return "text"
            
            return "unknown"
            
        except Exception as e:
            logger.error(f"Error inferring semantic type: {str(e)}")
            return "unknown"
    
    def _calculate_column_statistics(self, column: pd.Series) -> Dict[str, Any]:
        """
        Calculate relevant statistics for a column
        
        Args:
            column: Pandas series
            
        Returns:
            Dictionary of statistics
        """
        stats = {}
        
        try:
            # Basic stats for all types
            stats['null_percentage'] = (column.isnull().sum() / len(column)) * 100
            stats['unique_count'] = column.nunique()
            stats['most_frequent'] = column.mode().iloc[0] if not column.mode().empty else None
            stats['most_frequent_count'] = column.value_counts().iloc[0] if len(column.value_counts()) > 0 else 0
            
            # Numeric statistics
            if pd.api.types.is_numeric_dtype(column):
                non_null = column.dropna()
                if len(non_null) > 0:
                    stats.update({
                        'mean': float(non_null.mean()),
                        'median': float(non_null.median()),
                        'std': float(non_null.std()),
                        'min': float(non_null.min()),
                        'max': float(non_null.max()),
                        'q25': float(non_null.quantile(0.25)),
                        'q75': float(non_null.quantile(0.75))
                    })
            
            # String statistics
            elif pd.api.types.is_object_dtype(column):
                non_null = column.dropna().astype(str)
                if len(non_null) > 0:
                    lengths = non_null.str.len()
                    stats.update({
                        'avg_length': float(lengths.mean()),
                        'min_length': int(lengths.min()),
                        'max_length': int(lengths.max()),
                        'empty_strings': int((non_null == '').sum())
                    })
            
            # Datetime statistics
            elif pd.api.types.is_datetime64_any_dtype(column):
                non_null = column.dropna()
                if len(non_null) > 0:
                    stats.update({
                        'earliest': str(non_null.min()),
                        'latest': str(non_null.max()),
                        'date_range_days': (non_null.max() - non_null.min()).days
                    })
            
        except Exception as e:
            logger.error(f"Error calculating statistics: {str(e)}")
            stats['error'] = str(e)
        
        return stats
    
    def _identify_quality_issues(self, column: pd.Series) -> List[str]:
        """
        Identify potential data quality issues
        
        Args:
            column: Pandas series
            
        Returns:
            List of quality issue descriptions
        """
        issues = []
        
        try:
            # High null percentage
            null_pct = (column.isnull().sum() / len(column)) * 100
            if null_pct > 50:
                issues.append(f"High null percentage: {null_pct:.1f}%")
            elif null_pct > 20:
                issues.append(f"Moderate null percentage: {null_pct:.1f}%")
            
            # Check for potential duplicates in identifier columns
            if column.nunique() / len(column) > 0.8:  # Likely identifier
                duplicates = len(column) - column.nunique()
                if duplicates > 0:
                    issues.append(f"Potential duplicate identifiers: {duplicates} duplicates")
            
            # Check for inconsistent formatting in text columns
            if pd.api.types.is_object_dtype(column):
                non_null = column.dropna().astype(str)
                
                # Check for mixed case
                if len(non_null) > 0:
                    has_lower = non_null.str.islower().any()
                    has_upper = non_null.str.isupper().any()
                    has_title = non_null.str.istitle().any()
                    
                    if sum([has_lower, has_upper, has_title]) > 1:
                        issues.append("Inconsistent text casing")
                
                # Check for leading/trailing whitespace
                has_whitespace = non_null.str.strip().ne(non_null).any()
                if has_whitespace:
                    issues.append("Contains leading/trailing whitespace")
            
            # Check for outliers in numeric columns
            if pd.api.types.is_numeric_dtype(column):
                non_null = column.dropna()
                if len(non_null) > 10:  # Need sufficient data
                    q1, q3 = non_null.quantile([0.25, 0.75])
                    iqr = q3 - q1
                    lower_bound = q1 - 1.5 * iqr
                    upper_bound = q3 + 1.5 * iqr
                    
                    outliers = ((non_null < lower_bound) | (non_null > upper_bound)).sum()
                    if outliers > 0:
                        outlier_pct = (outliers / len(non_null)) * 100
                        if outlier_pct > 5:
                            issues.append(f"Potential outliers detected: {outliers} values ({outlier_pct:.1f}%)")
        
        except Exception as e:
            logger.error(f"Error identifying quality issues: {str(e)}")
            issues.append("Error during quality assessment")
        
        return issues