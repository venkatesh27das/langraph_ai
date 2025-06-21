"""
Data Analysis Prompts for LLM-based column profiling and insights
"""
from typing import Dict, Any, List


class DataAnalysisPrompts:
    """Collection of prompts for data analysis tasks"""
    
    @staticmethod
    def semantic_type_classification(column_name: str, data_type: str, sample_values: List[Any], 
                                   unique_count: int, total_count: int, null_count: int) -> str:
        """Prompt for classifying semantic type of a column"""
        
        sample_str = ", ".join([str(v) for v in sample_values[:10] if v is not None])
        unique_ratio = unique_count / total_count if total_count > 0 else 0
        null_ratio = null_count / total_count if total_count > 0 else 0
        
        return f"""
You are a data analyst. Classify the semantic type of this column based on the provided information.

Column Name: {column_name}
Data Type: {data_type}
Sample Values: {sample_str}
Unique Count: {unique_count}
Total Count: {total_count}
Null Count: {null_count}
Unique Ratio: {unique_ratio:.3f}
Null Ratio: {null_ratio:.3f}

Possible semantic types:
- categorical: Limited set of categories/labels
- numeric_continuous: Continuous numerical measurements
- numeric_integer: Integer counts or discrete numbers
- categorical_numeric: Numbers used as categories (like zip codes)
- identifier: Unique identifiers (IDs, keys)
- text: Short text fields
- text_long: Long text or descriptions
- text_identifier: Text-based identifiers
- email: Email addresses
- phone: Phone numbers
- url: Web URLs
- datetime: Date/time values
- mostly_null: Column with >50% null values
- unknown: Cannot determine type

Analyze the column and respond with ONLY the semantic type name (lowercase, underscore-separated).
Consider the column name, sample values, and statistics to make your decision.

Semantic Type:"""

    @staticmethod
    def column_description_generation(column_name: str, semantic_type: str, sample_values: List[Any],
                                    statistics: Dict[str, Any], data_type: str) -> str:
        """Prompt for generating human-readable column descriptions"""
        
        sample_str = ", ".join([str(v) for v in sample_values[:5] if v is not None])
        stats_summary = []
        
        if statistics.get('mean') is not None:
            stats_summary.append(f"Average: {statistics['mean']:.2f}")
        if statistics.get('min') is not None:
            stats_summary.append(f"Range: {statistics['min']} to {statistics['max']}")
        if statistics.get('unique_count'):
            stats_summary.append(f"Unique values: {statistics['unique_count']}")
            
        stats_str = ", ".join(stats_summary) if stats_summary else "No statistics available"
        
        return f"""
You are a data documentation expert. Generate a clear, concise description for this database column.

Column Information:
- Name: {column_name}
- Data Type: {data_type}
- Semantic Type: {semantic_type}
- Sample Values: {sample_str}
- Statistics: {stats_str}

Write a 1-2 sentence description that explains:
1. What this column represents
2. The type of data it contains
3. Any relevant business context you can infer

Keep it professional and clear. Avoid technical jargon.

Description:"""

    @staticmethod
    def business_context_inference(column_name: str, semantic_type: str, sample_values: List[Any],
                                 dataset_context: str = "") -> str:
        """Prompt for inferring business context and usage patterns"""
        
        sample_str = ", ".join([str(v) for v in sample_values[:8] if v is not None])
        
        return f"""
You are a business analyst. Infer the business context and potential usage of this data column.

Column: {column_name}
Type: {semantic_type}
Sample Values: {sample_str}
Dataset Context: {dataset_context if dataset_context else "General business dataset"}

Consider:
- What business process might generate this data?
- How might analysts or business users utilize this column?
- What decisions could be made using this information?
- Are there any compliance or privacy considerations?

Provide a brief business context (2-3 sentences) focusing on practical usage and business value.

Business Context:"""

    @staticmethod
    def data_quality_assessment(column_name: str, null_ratio: float, unique_ratio: float,
                              quality_issues: List[str], sample_values: List[Any]) -> str:
        """Prompt for assessing data quality issues"""
        
        issues_str = ", ".join(quality_issues) if quality_issues else "None detected"
        sample_str = ", ".join([str(v) for v in sample_values[:5] if v is not None])
        
        return f"""
You are a data quality analyst. Assess the quality of this column and provide actionable insights.

Column: {column_name}
Null Ratio: {null_ratio:.3f} ({null_ratio*100:.1f}% missing)
Unique Ratio: {unique_ratio:.3f}
Detected Issues: {issues_str}
Sample Values: {sample_str}

Evaluate:
1. Overall data quality (scale 1-10)
2. Specific quality concerns
3. Impact on analysis/reporting
4. Recommendations for improvement

Provide a concise quality assessment with a numerical score and key recommendations.

Quality Assessment:"""

    @staticmethod
    def dataset_overview_analysis(total_rows: int, total_columns: int, column_types: Dict[str, int],
                                missing_data_summary: Dict[str, Any], file_name: str) -> str:
        """Prompt for generating overall dataset insights"""
        
        type_summary = ", ".join([f"{k}: {v}" for k, v in column_types.items()])
        
        return f"""
You are a senior data analyst. Provide a comprehensive overview of this dataset.

Dataset: {file_name}
Rows: {total_rows:,}
Columns: {total_columns}
Column Types: {type_summary}
Missing Data: {missing_data_summary.get('total_missing', 0):,} values across {missing_data_summary.get('columns_with_missing', 0)} columns

Analyze:
1. Dataset size and scope
2. Data completeness and quality patterns
3. Potential use cases and applications
4. Key strengths and limitations
5. Recommendations for users

Provide a professional summary (3-4 sentences) that gives stakeholders a clear understanding of this dataset's characteristics and potential value.

Dataset Overview:"""

    @staticmethod
    def correlation_insights(correlation_pairs: List[tuple], column_info: Dict[str, Any]) -> str:
        """Prompt for analyzing column relationships and correlations"""
        
        if not correlation_pairs:
            return "No significant correlations found in the dataset."
        
        corr_text = []
        for col1, col2, corr_value in correlation_pairs[:5]:  # Top 5 correlations
            corr_text.append(f"{col1} & {col2}: {corr_value:.3f}")
        
        correlations_str = "\n".join(corr_text)
        
        return f"""
You are a data scientist analyzing relationships between variables.

Top Correlations Found:
{correlations_str}

Column Information Available: {len(column_info)} columns analyzed

Provide insights about:
1. Strongest relationships and their business meaning
2. Potential causation vs correlation considerations
3. How these relationships might impact analysis
4. Recommendations for further investigation

Keep insights actionable and business-focused.

Correlation Insights:"""

    @staticmethod
    def anomaly_detection_analysis(anomalies: Dict[str, List[Any]], column_profiles: Dict[str, Any]) -> str:
        """Prompt for analyzing detected anomalies and outliers"""
        
        if not anomalies:
            return "No significant anomalies detected in the dataset."
        
        anomaly_summary = []
        for col, values in anomalies.items():
            anomaly_summary.append(f"{col}: {len(values)} anomalies detected")
        
        summary_str = "\n".join(anomaly_summary[:5])  # Top 5 columns with anomalies
        
        return f"""
You are a data quality specialist analyzing anomalies and outliers.

Anomalies Detected:
{summary_str}

Analyze these anomalies and provide:
1. Potential causes (data entry errors, system issues, legitimate edge cases)
2. Impact on data analysis and reporting
3. Recommended actions (investigate, clean, flag, or keep)
4. Prevention strategies for future data collection

Focus on actionable recommendations for data stewards.

Anomaly Analysis:"""

    @staticmethod
    def column_constraint_inference(column_name: str, semantic_type: str, statistics: Dict[str, Any],
                                  sample_values: List[Any]) -> str:
        """Prompt for inferring data constraints and validation rules"""
        
        sample_str = ", ".join([str(v) for v in sample_values[:10] if v is not None])
        
        constraints_context = []
        if statistics.get('min') is not None:
            constraints_context.append(f"Min: {statistics['min']}")
        if statistics.get('max') is not None:
            constraints_context.append(f"Max: {statistics['max']}")
        if statistics.get('unique_count'):
            constraints_context.append(f"Unique values: {statistics['unique_count']}")
            
        context_str = ", ".join(constraints_context)
        
        return f"""
You are a database designer. Infer appropriate data constraints and validation rules for this column.

Column: {column_name}
Type: {semantic_type}
Sample Values: {sample_str}
Statistics: {context_str}

Consider:
- Data type constraints (length, format, range)
- Business rules (valid values, relationships)
- Quality constraints (not null, unique, etc.)
- Pattern validation (email format, phone format)

List specific, implementable constraints that would ensure data quality.

Constraints:"""

    @staticmethod
    def usage_recommendations(column_profiles: List[Dict[str, Any]], dataset_summary: Dict[str, Any]) -> str:
        """Prompt for generating usage recommendations for the entire dataset"""
        
        high_quality_cols = [col['name'] for col in column_profiles if col.get('quality_score', 0) > 0.8]
        low_quality_cols = [col['name'] for col in column_profiles if col.get('quality_score', 0) < 0.5]
        
        return f"""
You are a data strategy consultant. Provide usage recommendations for this dataset.

Dataset Summary:
- Total Columns: {len(column_profiles)}
- High Quality Columns: {len(high_quality_cols)} ({', '.join(high_quality_cols[:5])})
- Low Quality Columns: {len(low_quality_cols)} ({', '.join(low_quality_cols[:3])})
- Total Rows: {dataset_summary.get('total_rows', 'Unknown')}

Provide specific recommendations for:
1. Primary use cases and applications
2. Columns best suited for analysis/reporting
3. Data preparation steps needed
4. Potential limitations and caveats
5. Integration with other datasets

Focus on actionable guidance for data consumers.

Usage Recommendations:"""