"""
LangGraph Workflow for Data Dictionary Generation
Orchestrates the data analysis and dictionary generation process
"""
import json
from typing import Dict, Any, Optional
from dataclasses import asdict

from langgraph.graph import Graph, Node
from langgraph.graph.state import BaseState

from core.data_loader import DataLoader
from core.llm_client import LMStudioClient
from agents.analyzer import DataAnalyzer
from agents.generator import DictionaryGenerator
from config import LMSTUDIO_BASE_URL, LMSTUDIO_MODEL, LMSTUDIO_API_KEY


class WorkflowState(BaseState):
    """State object for the workflow"""
    file_path: str
    output_format: str
    sample_size: int
    dataframe: Optional[Any] = None
    analysis_result: Optional[Any] = None
    dictionary: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class DataDictionaryWorkflow:
    """LangGraph workflow for data dictionary generation"""
    
    def __init__(self):
        self.llm_client = LMStudioClient(
            base_url=LMSTUDIO_BASE_URL,
            model=LMSTUDIO_MODEL,
            api_key=LMSTUDIO_API_KEY
        )
        self.data_loader = DataLoader()
        self.analyzer = DataAnalyzer(self.llm_client)
        self.generator = DictionaryGenerator(self.llm_client)
        
        # Build the workflow graph
        self.graph = self._build_graph()
    
    def _build_graph(self) -> Graph:
        """Build the LangGraph workflow"""
        
        # Define workflow nodes
        load_data_node = Node(
            name="load_data",
            function=self._load_data_step,
            description="Load and validate data from file"
        )
        
        analyze_data_node = Node(
            name="analyze_data",
            function=self._analyze_data_step,
            description="Analyze data structure and generate insights"
        )
        
        generate_dictionary_node = Node(
            name="generate_dictionary",
            function=self._generate_dictionary_step,
            description="Generate comprehensive data dictionary"
        )
        
        format_output_node = Node(
            name="format_output",
            function=self._format_output_step,
            description="Format output in requested format"
        )
        
        # Create graph
        graph = Graph()
        
        # Add nodes
        graph.add_node(load_data_node)
        graph.add_node(analyze_data_node)
        graph.add_node(generate_dictionary_node)
        graph.add_node(format_output_node)
        
        # Define edges (workflow flow)
        graph.add_edge("load_data", "analyze_data")
        graph.add_edge("analyze_data", "generate_dictionary")
        graph.add_edge("generate_dictionary", "format_output")
        
        # Set entry and exit points
        graph.set_entry_point("load_data")
        graph.set_finish_point("format_output")
        
        return graph
    
    async def run(self, file_path: str, output_format: str = 'json', 
                  sample_size: int = 1000) -> Dict[str, Any]:
        """
        Run the complete workflow
        """
        print("🚀 Starting Data Dictionary Generation Workflow")
        
        # Initialize state
        initial_state = WorkflowState(
            file_path=file_path,
            output_format=output_format,
            sample_size=sample_size
        )
        
        try:
            # Execute workflow
            final_state = await self._execute_workflow(initial_state)
            
            if final_state.error:
                raise Exception(final_state.error)
            
            return final_state.dictionary
        
        except Exception as e:
            print(f"❌ Workflow failed: {str(e)}")
            raise
    
    async def _execute_workflow(self, state: WorkflowState) -> WorkflowState:
        """Execute the workflow steps in sequence"""
        
        # Step 1: Load Data
        print("📊 Step 1: Loading data...")
        state = await self._load_data_step(state)
        if state.error:
            return state
        
        # Step 2: Analyze Data
        print("🔍 Step 2: Analyzing data...")
        state = await self._analyze_data_step(state)
        if state.error:
            return state
        
        # Step 3: Generate Dictionary
        print("📝 Step 3: Generating dictionary...")
        state = await self._generate_dictionary_step(state)
        if state.error:
            return state
        
        # Step 4: Format Output
        print("🎨 Step 4: Formatting output...")
        state = await self._format_output_step(state)
        
        return state
    
    async def _load_data_step(self, state: WorkflowState) -> WorkflowState:
        """Load data from file"""
        try:
            print(f"  📁 Loading file: {state.file_path}")
            
            # Load data using DataLoader
            df = await self.data_loader.load_data(state.file_path)
            
            if df is None or df.empty:
                state.error = "Failed to load data or file is empty"
                return state
            
            state.dataframe = df
            print(f"  ✅ Data loaded: {len(df)} rows, {len(df.columns)} columns")
            
            return state
        
        except Exception as e:
            state.error = f"Data loading failed: {str(e)}"
            return state
    
    async def _analyze_data_step(self, state: WorkflowState) -> WorkflowState:
        """Analyze the loaded data"""
        try:
            if state.dataframe is None:
                state.error = "No data available for analysis"
                return state
            
            print("  🔍 Running data analysis...")
            
            # Run analysis
            analysis_result = await self.analyzer.analyze(
                state.dataframe, 
                sample_size=state.sample_size
            )
            
            state.analysis_result = analysis_result
            print(f"  ✅ Analysis completed: {len(analysis_result.column_profiles)} columns analyzed")
            
            return state
        
        except Exception as e:
            state.error = f"Data analysis failed: {str(e)}"
            return state
    
    async def _generate_dictionary_step(self, state: WorkflowState) -> WorkflowState:
        """Generate the data dictionary"""
        try:
            if state.analysis_result is None:
                state.error = "No analysis result available for dictionary generation"
                return state
            
            print("  📝 Generating data dictionary...")
            
            # Generate dictionary
            dictionary = await self.generator.generate(
                state.analysis_result,
                state.file_path,
                state.output_format
            )
            
            # Convert to dict for JSON serialization
            state.dictionary = asdict(dictionary)
            print("  ✅ Dictionary generated successfully")
            
            return state
        
        except Exception as e:
            state.error = f"Dictionary generation failed: {str(e)}"
            return state
    
    async def _format_output_step(self, state: WorkflowState) -> WorkflowState:
        """Format the output in the requested format"""
        try:
            if state.dictionary is None:
                state.error = "No dictionary available for formatting"
                return state
            
            print(f"  🎨 Formatting output as {state.output_format}...")
            
            if state.output_format == 'json':
                # Already in dict format, just add metadata
                state.dictionary['format'] = 'json'
                state.dictionary['workflow_version'] = '1.0'
            
            elif state.output_format == 'markdown':
                # Convert to markdown format
                markdown_output = self._convert_to_markdown(state.dictionary)
                state.dictionary['markdown_output'] = markdown_output
                state.dictionary['format'] = 'markdown'
            
            elif state.output_format == 'yaml':
                # Add YAML metadata
                state.dictionary['format'] = 'yaml'
            
            # Add summary for quick reference
            state.dictionary['summary'] = {
                'total_columns': len(state.dictionary.get('column_definitions', [])),
                'total_rows': state.dictionary.get('metadata', {}).get('total_rows', 0),
                'missing_values_count': sum(
                    col.get('statistics', {}).get('null_count', 0) 
                    for col in state.dictionary.get('column_definitions', [])
                    if isinstance(col, dict)
                ),
                'data_types': {
                    col.get('name', 'unknown'): col.get('data_type', 'unknown')
                    for col in state.dictionary.get('column_definitions', [])
                    if isinstance(col, dict)
                }
            }
            
            print("  ✅ Output formatted successfully")
            return state
        
        except Exception as e:
            state.error = f"Output formatting failed: {str(e)}"
            return state
    
    def _convert_to_markdown(self, dictionary: Dict[str, Any]) -> str:
        """Convert dictionary to markdown format"""
        markdown = []
        
        # Title
        markdown.append("# Data Dictionary")
        markdown.append("")
        
        # Metadata
        if 'metadata' in dictionary:
            markdown.append("## Dataset Information")
            metadata = dictionary['metadata']
            markdown.append(f"- **Source File**: {metadata.get('source_file', 'N/A')}")
            markdown.append(f"- **Generated At**: {metadata.get('generated_at', 'N/A')}")
            markdown.append(f"- **Total Rows**: {metadata.get('total_rows', 'N/A'):,}")
            markdown.append(f"- **Total Columns**: {metadata.get('total_columns', 'N/A')}")
            markdown.append("")
        
        # Dataset Description
        if 'dataset_description' in dictionary:
            markdown.append("## Dataset Description")
            markdown.append(dictionary['dataset_description'])
            markdown.append("")
        
        # Column Definitions
        if 'column_definitions' in dictionary:
            markdown.append("## Column Definitions")
            markdown.append("")
            
            for col in dictionary['column_definitions']:
                if isinstance(col, dict):
                    markdown.append(f"### {col.get('name', 'Unknown')}")
                    markdown.append(f"**Description**: {col.get('description', 'N/A')}")
                    markdown.append(f"**Data Type**: {col.get('data_type', 'N/A')}")
                    markdown.append(f"**Semantic Type**: {col.get('semantic_type', 'N/A')}")
                    markdown.append(f"**Nullable**: {'Yes' if col.get('nullable', False) else 'No'}")
                    
                    if col.get('sample_values'):
                        markdown.append(f"**Sample Values**: {', '.join(map(str, col['sample_values'][:5]))}")
                    
                    if col.get('business_context'):
                        markdown.append(f"**Business Context**: {col['business_context']}")
                    
                    markdown.append("")
        
        # Data Quality
        if 'data_quality' in dictionary:
            quality = dictionary['data_quality']
            markdown.append("## Data Quality Assessment")
            markdown.append(f"- **Overall Score**: {quality.get('overall_score', 'N/A')}")
            markdown.append(f"- **Completeness**: {quality.get('completeness', 'N/A')}")
            
            if quality.get('issues'):
                markdown.append("### Issues Identified")
                for issue in quality['issues']:
                    markdown.append(f"- {issue}")
            markdown.append("")
        
        # Insights
        if 'insights' in dictionary:
            markdown.append("## Key Insights")
            for insight in dictionary['insights']:
                markdown.append(f"- {insight}")
            markdown.append("")
        
        # Usage Recommendations
        if 'usage_recommendations' in dictionary:
            markdown.append("## Usage Recommendations")
            for rec in dictionary['usage_recommendations']:
                markdown.append(f"- {rec}")
            markdown.append("")
        
        return "\n".join(markdown)