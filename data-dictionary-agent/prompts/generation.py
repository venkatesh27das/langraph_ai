"""
Dictionary Generation Prompts for creating comprehensive data dictionaries
"""
from typing import Dict, Any, List


class DictionaryGenerationPrompts:
    """Collection of prompts for data dictionary generation tasks"""
    
    @staticmethod
    def dataset_description_generation(file_name: str, total_rows: int, total_columns: int,
                                     column_types: Dict[str, int], key_columns: List[str],
                                     sample_data_insights: str) -> str:
        """Prompt for generating overall dataset description"""
        
        type_summary = ", ".join([f"{v} {k} columns" for k, v in column_types.items()])
        key_cols_str = ", ".join(key_columns[:5]) if key_columns else "Not identified"
        
        return f"""
You are a technical writer creating documentation for a data catalog.

Generate a comprehensive dataset description for:

Dataset: {file_name}
Size: {total_rows:,} rows × {total_columns} columns
Column Types: {type_summary}
Key Columns: {key_cols_str}

Additional Context:
{sample_data_insights}

Write a professional dataset description (2-3 paragraphs) that includes:
1. What the dataset represents and its primary purpose
2. Key characteristics and data structure
3. Potential business applications and use cases
4. Data collection context (if inferable)

Keep it informative yet accessible to both technical and business users.

Dataset Description:"""

    @staticmethod
    def glossary_generation(column_names: List[str], business_terms: List[str],
                          semantic_types: List[str]) -> str:
        """Prompt for generating a business glossary"""
        
        unique_terms = set()
        
        # Extract potential business terms from column names
        for col in column_names:
            # Split on common separators and add meaningful terms
            parts = col.lower().replace('_', ' ').replace('-', ' ').split()
            for part in parts:
                if len(part) > 3:  # Avoid short words like 'id', 'no'
                    unique_terms.add(part)
        
        # Add explicit business terms
        unique_terms.update([term.lower() for term in business_terms])
        
        terms_list = sorted(list(unique_terms))[:20]  # Limit to 20 terms
        
        return f"""
You are a business analyst creating a glossary for a data dictionary.

Create definitions for these business terms found in the dataset:
{', '.join(terms_list)}

Column Names Context: {', '.join(column_names[:10])}
Semantic Types Present: {', '.join(set(semantic_types))}

For each term, provide:
- Clear, concise definition (1-2 sentences)
- Business context when relevant
- Common usage in data analysis

Format as: Term: Definition

Focus on terms that would be unclear to someone unfamiliar with the business domain.

Glossary Terms:"""

    @staticmethod
    def data_lineage_inference(file_name: str, column_profiles: List[Dict[str, Any]],
                             identifier_columns: List[str]) -> str:
        """Prompt for inferring potential data lineage and sources"""
        
        id_cols = ", ".join(identifier_columns) if identifier_columns else "None identified"
        
        # Identify potential foreign keys or reference columns
        ref_columns = []
        for col in column_profiles:
            col_name = col.get('name', '').lower()
            if any(suffix in col_name for suffix in ['_id', '_key', '_ref', '_code']):
                ref_columns.append(col.get('name'))
        
        ref_cols_str = ", ".join(ref_columns[:5]) if ref_columns else "None identified"
        
        return f"""
You are a data architect analyzing data lineage and relationships.

Dataset: {file_name}
Identifier Columns: {id_cols}
Potential Reference Columns: {ref_cols_str}
Total Columns: {len(column_profiles)}

Infer potential data lineage including:
1. Likely source systems or databases
2. Potential upstream data feeds
3. Common integration patterns
4. Relationships with other datasets

Provide insights about where this data might originate and how it connects to other systems.

Data Lineage Insights:"""

    @staticmethod
    def compliance_considerations(column_profiles: List[Dict[str, Any]], 
                                pii_columns: List[str], sensitive_columns: List[str]) -> str:
        """Prompt for identifying compliance and privacy considerations"""
        
        pii_str = ", ".join(pii_columns) if pii_columns else "None detected"
        sensitive_str = ", ".join(sensitive_columns) if sensitive_columns else "None detected"
        
        return f"""
You are a data privacy officer reviewing this dataset for compliance considerations.

Dataset Analysis:
- Total Columns: {len(column_profiles)}
- Potential PII Columns: {pii_str}
- Sensitive Data Columns: {sensitive_str}

Identify compliance considerations including:
1. Data privacy regulations (GDPR, CCPA, etc.)
2. Industry-specific compliance requirements
3. Data retention and deletion policies
4. Access control recommendations
5. Anonymization/pseudonymization opportunities

Provide specific, actionable compliance guidance.

Compliance Considerations:"""

    @staticmethod
    def data_dictionary_summary(total_columns: int, data_quality_score: float,
                              key_insights: List[str], main_challenges: List[str]) -> str:
        """Prompt for generating an executive summary of the data dictionary"""
        
        insights_str = "\n".join([f"• {insight}" for insight in key_insights[:5]])
        challenges_str = "\n".join([f"• {challenge}" for challenge in main_challenges[:3]])
        
        return f"""
You are an executive briefing specialist. Create a concise executive summary for this data dictionary.

Dataset Metrics:
- Columns Analyzed: {total_columns}
- Overall Data Quality Score: {data_quality_score:.1f}/10
- Key Insights Identified: {len(key_insights)}
- Main Challenges: {len(main_challenges)}

Key Insights:
{insights_str}

Main Challenges:
{challenges_str}

Write a professional executive summary (3-4 sentences) that:
1. Highlights the dataset's business value
2. Summarizes data quality and readiness
3. Identifies key opportunities and risks
4. Provides clear next steps

Executive Summary:"""

    @staticmethod
    def integration_recommendations(column_profiles: List[Dict[str, Any]], 
                                  similar_datasets: List[str], join_keys: List[str]) -> str:
        """Prompt for generating data integration recommendations"""
        
        join_keys_str = ", ".join(join_keys) if join_keys else "None identified"
        similar_str = ", ".join(similar_datasets) if similar_datasets else "None specified"
        
        return f"""
You are a data integration specialist providing recommendations for combining datasets.

Current Dataset:
- Columns: {len(column_profiles)}
- Potential Join Keys: {join_keys_str}
- Similar Datasets: {similar_str}

Provide integration recommendations including:
1. Best practices for joining with other datasets
2. Key matching strategies and considerations
3. Potential data quality issues in joins
4. Recommended data preparation steps
5. Integration testing approaches

Focus on practical, implementable recommendations.

Integration Recommendations:"""

    @staticmethod
    def quality_improvement_plan(low_quality_columns: List[Dict[str, Any]], 
                               quality_issues: List[str], improvement_priorities: List[str]) -> str:
        """Prompt for generating a data quality improvement plan"""
        
        low_quality_names = [col.get('name', 'Unknown') for col in low_quality_columns]
        issues_str = "\n".join([f"• {issue}" for issue in quality_issues[:5]])
        priorities_str = "\n".join([f"• {priority}" for priority in improvement_priorities[:3]])
        
        return f"""
You are a data quality manager creating an improvement plan.

Quality Assessment:
- Low Quality Columns: {len(low_quality_names)} ({', '.join(low_quality_names[:5])})
- Total Issues Identified: {len(quality_issues)}

Key Issues:
{issues_str}

Improvement Priorities:
{priorities_str}

Create a structured quality improvement plan including:
1. Immediate actions (quick wins)
2. Medium-term improvements
3. Long-term data governance changes
4. Success metrics and monitoring
5. Resource requirements

Quality Improvement Plan:"""

    @staticmethod
    def documentation_completeness_check(column_definitions: List[Dict[str, Any]],
                                       missing_descriptions: List[str], 
                                       incomplete_metadata: List[str]) -> str:
        """Prompt for assessing documentation completeness"""
        
        total_cols = len(column_definitions)
        missing_count = len(missing_descriptions)
        incomplete_count = len(incomplete_metadata)
        
        completeness_score = ((total_cols - missing_count - incomplete_count) / total_cols * 100) if total_cols > 0 else 0
        
        return f"""
You are a documentation quality auditor reviewing this data dictionary.

Documentation Status:
- Total Columns: {total_cols}
- Missing Descriptions: {missing_count}
- Incomplete Metadata: {incomplete_count}
- Completeness Score: {completeness_score:.1f}%

Missing Descriptions: {', '.join(missing_descriptions[:5]) if missing_descriptions else 'None'}
Incomplete Metadata: {', '.join(incomplete_metadata[:5]) if incomplete_metadata else 'None'}

Provide a documentation quality assessment including:
1. Overall completeness rating
2. Critical gaps that need immediate attention
3. Recommendations for improving documentation
4. Maintenance and update procedures

Documentation Quality Assessment:"""

    @staticmethod
    def stakeholder_recommendations(dataset_type: str, business_domain: str,
                                  user_personas: List[str], use_cases: List[str]) -> str:
        """Prompt for generating stakeholder-specific recommendations"""
        
        personas_str = ", ".join(user_personas) if user_personas else "General business users"
        use_cases_str = "\n".join([f"• {use_case}" for use_case in use_cases[:5]])
        
        return f"""
You are a data product manager creating stakeholder guidance.

Dataset Context:
- Type: {dataset_type}
- Business Domain: {business_domain}
- Target Users: {personas_str}

Identified Use Cases:
{use_cases_str}

Provide stakeholder-specific recommendations for:
1. Data analysts and scientists
2. Business intelligence teams
3. Application developers
4. Business stakeholders
5. Data governance teams

Include specific guidance on how each group should approach and utilize this dataset.

Stakeholder Recommendations:"""