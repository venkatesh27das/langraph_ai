"""
SQL Agent for handling structured data queries.
Converts natural language to SQL and generates insights with visualizations.
"""

from typing import Dict, List, Any, Optional
import logging
import pandas as pd
from ..core.sql_generator import SQLGenerator
from ..core.data_dictionary import DataDictionary
from ..core.chart_generator import ChartGenerator
from ..core.lm_studio_client import LMStudioClient
from ..utils import execute_sql_query

logger = logging.getLogger(__name__)

class SQLAgent:
    """Agent for handling structured data queries using SQL generation"""
    
    def __init__(self, config: Dict[str, Any], llm_client: LMStudioClient):
        self.config = config
        self.llm_client = llm_client
        self.sql_generator = SQLGenerator(config['sql'], llm_client)
        self.data_dictionary = DataDictionary(config['database'])
        self.chart_generator = ChartGenerator()
        
    def process(self, user_input: str, conversation_history: List[Dict[str, str]]) -> Dict[str, Any]:
        """Process user query by generating and executing SQL"""
        try:
            # Get data dictionary for context
            schema_info = self.data_dictionary.get_schema_summary()
            
            # Generate SQL query
            sql_result = self.sql_generator.generate_sql(
                user_input, 
                schema_info, 
                conversation_history
            )
            
            if not sql_result.get("sql"):
                return {
                    "response": "I couldn't generate a valid SQL query for your request. Could you please rephrase your question or be more specific about what data you're looking for?",
                    "data": {}
                }
            
            # Execute SQL query
            query_result = self._execute_query(sql_result["sql"])
            
            if query_result.get("error"):
                return {
                    "response": f"I generated a SQL query but encountered an error: {query_result['error']}. Let me know if you'd like me to try a different approach.",
                    "data": {"sql": sql_result["sql"], "error": query_result["error"]}
                }
            
            # Process results
            df = query_result["data"]
            
            # Generate insights and response
            response = self._generate_insights_response(
                user_input, 
                sql_result["sql"], 
                df, 
                conversation_history
            )
            
            # Generate visualization if appropriate
            visualization = self._create_visualization(df, user_input, sql_result.get("chart_type"))
            
            return {
                "response": response,
                "data": {
                    "sql": sql_result["sql"],
                    "results": df.to_dict('records') if not df.empty else [],
                    "row_count": len(df),
                    "columns": df.columns.tolist() if not df.empty else []
                },
                "visualization": visualization
            }
            
        except Exception as e:
            logger.error(f"Error in SQL processing: {str(e)}")
            return {
                "response": "I encountered an error while processing your data query. Please try rephrasing your question or check if it relates to available data.",
                "data": {"error": str(e)}
            }
    
    def _execute_query(self, sql: str) -> Dict[str, Any]:
        """Execute SQL query and return results"""
        try:
            df = execute_sql_query(sql, self.config['database']['path'])
            return {"data": df, "error": None}
        except Exception as e:
            logger.error(f"SQL execution error: {str(e)}")
            return {"data": pd.DataFrame(), "error": str(e)}
    
    def _generate_insights_response(self, user_input: str, sql: str, df: pd.DataFrame, 
                                  history: List[Dict[str, str]]) -> str:
        """Generate natural language response with insights from query results"""
        
        if df.empty:
            return "Your query returned no results. You might want to try different criteria or check if the data exists."
        
        # Create summary of results
        summary = self._create_data_summary(df)
        
        # Build conversation context
        conversation_context = ""
        if history:
            recent_history = history[-2:]  # Last 2 exchanges
            conversation_context = "\n".join([
                f"User: {h.get('user', '')}\nAssistant: {h.get('bot', '')}" 
                for h in recent_history
            ])
        
        prompt = f"""You are a BI assistant analyzing data query results. 

Previous conversation:
{conversation_context}

User's question: {user_input}

SQL Query executed: {sql}

Data summary:
{summary}

Instructions:
1. Provide insights based on the data results
2. Answer the user's specific question directly
3. Highlight key findings or patterns
4. Keep response conversational and business-focused
5. If numbers are involved, provide context
6. Suggest follow-up questions if relevant
7. Don't repeat the SQL query in your response

Response:"""

        try:
            response = self.llm_client.generate(
                prompt, 
                max_tokens=self.config.get('max_response_tokens', 400),
                temperature=0.2
            )
            
            return response.strip()
            
        except Exception as e:
            logger.error(f"Error generating SQL insights response: {str(e)}")
            return f"I found {len(df)} records matching your query, but encountered an error generating insights. Here's what I found: {summary}"
    
    def _create_data_summary(self, df: pd.DataFrame) -> str:
        """Create a summary of the dataframe"""
        if df.empty:
            return "No data found."
        
        summary_parts = [f"Found {len(df)} records"]
        
        # Add column info
        summary_parts.append(f"Columns: {', '.join(df.columns.tolist())}")
        
        # Add key statistics for numeric columns
        numeric_cols = df.select_dtypes(include=['number']).columns
        if not numeric_cols.empty:
            for col in numeric_cols[:3]:  # Limit to first 3 numeric columns
                if col in df.columns:
                    stats = f"{col}: min={df[col].min()}, max={df[col].max()}, avg={df[col].mean():.2f}"
                    summary_parts.append(stats)
        
        # Add sample of categorical data
        categorical_cols = df.select_dtypes(include=['object']).columns
        if not categorical_cols.empty:
            for col in categorical_cols[:2]:  # Limit to first 2 categorical columns
                unique_values = df[col].nunique()
                if unique_values <= 10:
                    values = ', '.join(df[col].unique()[:5].astype(str))
                    summary_parts.append(f"{col} values: {values}")
                else:
                    summary_parts.append(f"{col}: {unique_values} unique values")
        
        return "\n".join(summary_parts)
    
    def _create_visualization(self, df: pd.DataFrame, user_input: str, 
                            suggested_chart_type: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Create visualization if appropriate"""
        try:
            if df.empty or len(df) > 1000:  # Skip if too much data
                return None
            
            # Determine chart type
            chart_type = suggested_chart_type or self._suggest_chart_type(df, user_input)
            
            if not chart_type:
                return None
            
            # Generate chart
            chart_data = self.chart_generator.create_chart(df, chart_type)
            
            return chart_data
            
        except Exception as e:
            logger.error(f"Error creating visualization: {str(e)}")
            return None
    
    def _suggest_chart_type(self, df: pd.DataFrame, user_input: str) -> Optional[str]:
        """Suggest appropriate chart type based on data and query"""
        if len(df) < 2:
            return None
        
        numeric_cols = df.select_dtypes(include=['number']).columns
        categorical_cols = df.select_dtypes(include=['object']).columns
        
        # Simple heuristics for chart type
        query_lower = user_input.lower()
        
        if 'trend' in query_lower or 'over time' in query_lower:
            return 'line'
        elif 'compare' in query_lower or 'vs' in query_lower:
            return 'bar'
        elif len(numeric_cols) >= 1 and len(categorical_cols) >= 1:
            if len(df) <= 20:
                return 'bar'
            else:
                return 'line'
        elif len(numeric_cols) >= 2:
            return 'scatter'
        
        return 'bar' if len(df) <= 50 else None
    
    def get_schema_info(self) -> Dict[str, Any]:
        """Get database schema information"""
        try:
            return self.data_dictionary.get_full_schema()
        except Exception as e:
            logger.error(f"Error getting schema info: {str(e)}")
            return {"error": str(e)}
    
    def validate_query(self, sql: str) -> Dict[str, Any]:
        """Validate SQL query without executing"""
        try:
            # Basic SQL validation (can be enhanced)
            sql_lower = sql.lower().strip()
            
            # Check for dangerous operations
            dangerous_keywords = ['drop', 'delete', 'truncate', 'alter', 'create', 'insert', 'update']
            for keyword in dangerous_keywords:
                if keyword in sql_lower:
                    return {"valid": False, "error": f"Query contains potentially dangerous keyword: {keyword}"}
            
            # Check if it's a SELECT query
            if not sql_lower.startswith('select'):
                return {"valid": False, "error": "Only SELECT queries are allowed"}
            
            return {"valid": True, "error": None}
            
        except Exception as e:
            return {"valid": False, "error": str(e)}