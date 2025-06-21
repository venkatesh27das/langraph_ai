"""
Dictionary Generation Agent
Responsible for generating comprehensive data dictionaries
"""
import json
from typing import Dict, Any, List
from datetime import datetime
from dataclasses import asdict

from core.models import DataDictionary, ColumnDefinition
from core.llm_client import LMStudioClient
from agents.analyzer import AnalysisResult
from prompts.generation import DICTIONARY_GENERATION_PROMPT, COLUMN_DEFINITION_PROMPT


class DictionaryGenerator:
    """Agent responsible for generating data dictionaries from analysis results"""
    
    def __init__(self, llm_client: LMStudioClient):
        self.llm_client = llm_client
    
    async def generate(self, analysis_result: AnalysisResult, 
                      file_path: str, output_format: str = 'json') -> DataDictionary:
        """
        Generate comprehensive data dictionary from analysis results
        """
        print("📝 Generating data dictionary...")
        
        # Generate column definitions
        column_definitions = await self._generate_column_definitions(analysis_result)
        
        # Generate overall dataset description
        dataset_description = await self._generate_dataset_description(analysis_result)
        
        # Generate data quality assessment
        quality_assessment = self._generate_quality_assessment(analysis_result)
        
        # Generate usage recommendations
        usage_recommendations = await self._generate_usage_recommendations(analysis_result)
        
        # Create data dictionary
        dictionary = DataDictionary(
            metadata={
                "source_file": file_path,
                "generated_at": datetime.now().isoformat(),
                "total_rows": analysis_result.data_profile.total_rows,
                "total_columns": analysis_result.data_profile.total_columns,
                "sample_size": analysis_result.data_profile.sample_size
            },
            dataset_description=dataset_description,
            column_definitions=column_definitions,
            data_quality=quality_assessment,
            correlations=analysis_result.correlations,
            insights=analysis_result.insights,
            usage_recommendations=usage_recommendations,
            generated_by="Data Dictionary Agent v1.0"
        )
        
        print("✅ Data dictionary generated successfully!")
        return dictionary
    
    async def _generate_column_definitions(self, analysis_result: AnalysisResult) -> List[ColumnDefinition]:
        """Generate detailed column definitions"""
        definitions = []
        
        for column_name, profile in analysis_result.column_profiles.items():
            print(f"  📋 Generating definition for: {column_name}")
            
            # Generate enhanced description using LLM
            enhanced_description = await self._enhance_column_description(profile, analysis_result)
            
            # Determine constraints and validation rules
            constraints = self._generate_constraints(profile)
            
            # Generate business context if possible
            business_context = await self._generate_business_context(profile)
            
            definition = ColumnDefinition(
                name=profile.name,
                description=enhanced_description,
                data_type=profile.data_type,
                semantic_type=profile.inferred_type,
                nullable=profile.null_count > 0,
                unique=profile.unique_count == profile.total_count,
                constraints=constraints,
                sample_values=profile.sample_values[:5],  # Limit sample values
                statistics=profile.statistics,
                business_context=business_context,
                data_quality_score=self._calculate_quality_score(profile)
            )
            
            definitions.append(definition)
        
        return definitions
    
    async def _enhance_column_description(self, profile, analysis_result: AnalysisResult) -> str:
        """Enhance column description with LLM insights"""
        try:
            # Prepare context
            context = {
                "column_profile": {
                    "name": profile.name,
                    "type": profile.data_type,
                    "semantic_type": profile.inferred_type,
                    "null_percentage": (profile.null_count / profile.total_count) * 100,
                    "unique_percentage": (profile.unique_count / profile.total_count) * 100,
                    "sample_values": profile.sample_values[:5],
                    "statistics": profile.statistics,
                    "patterns": profile.patterns
                },
                "dataset_context": {
                    "total_columns": len(analysis_result.column_profiles),
                    "related_columns": [name for name in analysis_result.column_profiles.keys() 
                                      if name != profile.name][:5]
                }
            }
            
            prompt = COLUMN_DEFINITION_PROMPT.format(
                column_name=profile.name,
                context=json.dumps(context, indent=2)
            )
            
            response = await self.llm_client.generate(prompt)
            return response.strip()
        
        except Exception as e:
            print(f"Warning: Could not enhance description for {profile.name}: {e}")
            return profile.description or f"Column containing {profile.data_type} data"
    
    async def _generate_business_context(self, profile) -> str:
        """Generate business context for a column"""
        try:
            # Simple heuristics for business context
            name_lower = profile.name.lower()
            
            if 'id' in name_lower:
                return "Identifier field - likely used for referencing and joining data"
            elif 'name' in name_lower:
                return "Name field - contains descriptive text for identification"
            elif 'date' in name_lower or 'time' in name_lower:
                return "Temporal field - used for time-based analysis and tracking"
            elif 'amount' in name_lower or 'price' in name_lower or 'cost' in name_lower:
                return "Financial field - important for monetary calculations and analysis"
            elif profile.inferred_type == 'categorical':
                return "Categorical field - useful for grouping and segmentation analysis"
            elif profile.inferred_type == 'email':
                return "Contact information - can be used for communication and user identification"
            else:
                return "Data field - requires domain expertise for complete business context"
        
        except Exception as e:
            return "Business context analysis unavailable"
    
    def _generate_constraints(self, profile) -> List[str]:
        """Generate data constraints based on profile analysis"""
        constraints = []
        
        # Null constraints
        if profile.null_count == 0:
            constraints.append("NOT NULL - No missing values allowed")
        elif profile.null_count > 0:
            null_percentage = (profile.null_count / profile.total_count) * 100
            constraints.append(f"Nullable - {null_percentage:.1f}% missing values")
        
        # Uniqueness constraints
        if profile.unique_count == profile.total_count:
            constraints.append("UNIQUE - All values are unique")
        elif profile.unique_count == 1:
            constraints.append("CONSTANT - Only one unique value")
        
        # Type-specific constraints
        if profile.statistics:
            if 'min' in profile.statistics and 'max' in profile.statistics:
                constraints.append(f"Range: {profile.statistics['min']} to {profile.statistics['max']}")
        
        # Pattern constraints
        for pattern in profile.patterns:
            constraints.append(f"Pattern: {pattern}")
        
        return constraints
    
    def _calculate_quality_score(self, profile) -> float:
        """Calculate data quality score for a column"""
        score = 1.0
        
        # Penalize for missing values
        null_ratio = profile.null_count / profile.total_count
        score -= null_ratio * 0.3
        
        # Penalize for very low or very high cardinality (context dependent)
        if profile.unique_count == 1:  # All same values
            score -= 0.2
        elif profile.unique_count == profile.total_count and profile.inferred_type != 'identifier':
            score -= 0.1  # High cardinality when not expected
        
        # Bonus for having patterns
        if profile.patterns:
            score += 0.1
        
        return max(0.0, min(1.0, score))
    
    async def _generate_dataset_description(self, analysis_result: AnalysisResult) -> str:
        """Generate overall dataset description"""
        try:
            summary = {
                "dimensions": f"{analysis_result.data_profile.total_rows} rows × {analysis_result.data_profile.total_columns} columns",
                "data_types": analysis_result.data_profile.data_types,
                "column_summary": {
                    name: {
                        "type": profile.inferred_type,
                        "quality": self._calculate_quality_score(profile)
                    }
                    for name, profile in analysis_result.column_profiles.items()
                },
                "insights": analysis_result.insights[:3]  # Top 3 insights
            }
            
            prompt = DICTIONARY_GENERATION_PROMPT.format(
                summary=json.dumps(summary, indent=2)
            )
            
            response = await self.llm_client.generate(prompt)
            return response.strip()
        
        except Exception as e:
            print(f"Warning: Could not generate dataset description: {e}")
            return f"Dataset with {analysis_result.data_profile.total_rows} rows and {analysis_result.data_profile.total_columns} columns"
    
    def _generate_quality_assessment(self, analysis_result: AnalysisResult) -> Dict[str, Any]:
        """Generate data quality assessment"""
        total_values = sum(profile.total_count for profile in analysis_result.column_profiles.values())
        total_missing = sum(profile.null_count for profile in analysis_result.column_profiles.values())
        
        completeness = 1 - (total_missing / total_values) if total_values > 0 else 0
        
        # Calculate column quality scores
        column_scores = {
            name: self._calculate_quality_score(profile)
            for name, profile in analysis_result.column_profiles.items()
        }
        
        overall_quality = sum(column_scores.values()) / len(column_scores) if column_scores else 0
        
        return {
            "overall_score": round(overall_quality, 2),
            "completeness": round(completeness, 2),
            "column_scores": column_scores,
            "issues": self._identify_quality_issues(analysis_result),
            "recommendations": self._generate_quality_recommendations(analysis_result)
        }
    
    def _identify_quality_issues(self, analysis_result: AnalysisResult) -> List[str]:
        """Identify data quality issues"""
        issues = []
        
        for name, profile in analysis_result.column_profiles.items():
            # High missing values
            null_percentage = (profile.null_count / profile.total_count) * 100
            if null_percentage > 50:
                issues.append(f"Column '{name}' has {null_percentage:.1f}% missing values")
            
            # Constant values
            if profile.unique_count == 1:
                issues.append(f"Column '{name}' has only one unique value")
            
            # Potential duplicates in identifier columns
            if profile.inferred_type == 'identifier' and profile.unique_count < profile.total_count:
                issues.append(f"Column '{name}' appears to be an identifier but has duplicate values")
        
        return issues
    
    def _generate_quality_recommendations(self, analysis_result: AnalysisResult) -> List[str]:
        """Generate data quality improvement recommendations"""
        recommendations = []
        
        for name, profile in analysis_result.column_profiles.items():
            null_percentage = (profile.null_count / profile.total_count) * 100
            
            if null_percentage > 20:
                recommendations.append(f"Consider data imputation or collection improvement for column '{name}'")
            
            if profile.unique_count == 1:
                recommendations.append(f"Column '{name}' may be redundant - consider removal")
            
            if profile.inferred_type == 'text' and profile.unique_count < 20:
                recommendations.append(f"Column '{name}' might benefit from categorical encoding")
        
        return recommendations
    
    async def _generate_usage_recommendations(self, analysis_result: AnalysisResult) -> List[str]:
        """Generate usage recommendations for the dataset"""
        recommendations = []
        
        # Identify potential target variables
        numeric_cols = [name for name, profile in analysis_result.column_profiles.items() 
                       if profile.inferred_type in ['numeric', 'integer']]
        categorical_cols = [name for name, profile in analysis_result.column_profiles.items() 
                          if profile.inferred_type == 'categorical']
        id_cols = [name for name, profile in analysis_result.column_profiles.items() 
                  if profile.inferred_type == 'identifier']
        
        if numeric_cols:
            recommendations.append(f"Numeric columns {numeric_cols[:3]} suitable for regression analysis")
        
        if categorical_cols:
            recommendations.append(f"Categorical columns {categorical_cols[:3]} suitable for classification tasks")
        
        if id_cols:
            recommendations.append(f"Identifier columns {id_cols} useful for data joining and referencing")
        
        # Correlation insights
        if 'high_correlations' in analysis_result.correlations:
            high_corr = analysis_result.correlations['high_correlations']
            if high_corr:
                recommendations.append("High correlations detected - consider feature selection for ML models")
        
        # Missing data recommendations
        total_missing = sum(profile.null_count for profile in analysis_result.column_profiles.values())
        if total_missing > 0:
            recommendations.append("Missing data present - consider imputation strategies before analysis")
        
        return recommendations