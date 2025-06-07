import json
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
from utils.llm_client import llm_client
from utils.file_handler import FileHandler

class CodeIdentifierAgent:
    def __init__(self):
        self.system_prompt = """You are a Code Identifier Agent. Your job is to analyze existing code files and identify what changes are needed based on configuration metadata.

You should:
1. Analyze existing code structure
2. Identify files that need modification
3. Suggest specific changes for each file
4. Prioritize changes by importance
5. Consider dependencies between files

Always provide detailed, actionable suggestions."""

    def identify_target_files(self, parsed_config: Dict[str, Any], project_path: Path) -> List[Dict[str, Any]]:
        """Identify files that need modification based on config"""
        
        # Find all relevant files in project
        python_files = FileHandler.find_files(project_path, ['.py'])
        config_files = FileHandler.find_files(project_path, ['.json', '.yaml', '.yml'])
        
        target_files = []
        
        # Analyze each Python file
        for file_path in python_files:
            file_info = FileHandler.get_file_info(file_path)
            file_structure = FileHandler.parse_python_file(file_path)
            file_content = FileHandler.read_file(file_path)
            
            if file_structure and file_content:
                analysis = self._analyze_file_relevance(
                    file_path, file_content, file_structure, parsed_config
                )
                
                if analysis['needs_modification']:
                    target_files.append({
                        'file_path': str(file_path),
                        'file_info': file_info,
                        'structure': file_structure,
                        'analysis': analysis,
                        'priority': analysis.get('priority', 'medium')
                    })
        
        # Sort by priority
        priority_order = {'high': 3, 'medium': 2, 'low': 1}
        target_files.sort(key=lambda x: priority_order.get(x['priority'], 0), reverse=True)
        
        return target_files
    
    def _analyze_file_relevance(self, file_path: Path, content: str, structure: Dict, config: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze if a file needs modification based on config"""
        
        prompt = f"""
Analyze this Python file to determine if it needs modification based on the configuration:

File: {file_path.name}
File Structure:
- Functions: {[f['name'] for f in structure.get('functions', [])]}
- Classes: {[c['name'] for c in structure.get('classes', [])]}
- Imports: {structure.get('imports', [])}

Configuration Requirements:
{json.dumps(config, indent=2)}

File Content (first 1000 chars):
{content[:1000]}

Determine:
1. Does this file need modification? (yes/no)
2. What type of modifications are needed?
3. What is the priority level? (high/medium/low)
4. What specific changes should be made?

Return JSON:
{{
    "needs_modification": true/false,
    "modification_type": "data_loading|data_transformation|output_handling|configuration|utility",
    "priority": "high|medium|low",
    "reason": "explanation of why modification is needed",
    "suggested_changes": [
        {{
            "type": "add_function|modify_function|add_import|modify_variable",
            "target": "function/class/variable name",
            "description": "detailed description of change"
        }}
    ]
}}
"""

        try:
            response = llm_client.chat_completion(
                messages=[{"role": "user", "content": prompt}],
                system_prompt=self.system_prompt
            )
            
            if response:
                cleaned_response = self._clean_json_response(response)
                return json.loads(cleaned_response)
            
            return {"needs_modification": False}
            
        except Exception as e:
            print(f"Error analyzing file {file_path}: {e}")
            return {"needs_modification": False}
    
    def suggest_file_changes(self, target_file: Dict[str, Any], parsed_config: Dict[str, Any]) -> Dict[str, Any]:
        """Generate detailed suggestions for file modifications"""
        
        file_path = target_file['file_path']
        file_content = FileHandler.read_file(Path(file_path))
        
        prompt = f"""
Generate detailed code modification suggestions for this file:

File Path: {file_path}
Current Analysis: {json.dumps(target_file['analysis'], indent=2)}
Configuration: {json.dumps(parsed_config, indent=2)}

Current File Content:
{file_content}

Provide specific, actionable suggestions:
1. Exact code changes needed
2. New functions/classes to add
3. Imports to add/modify
4. Configuration parameters to update
5. Error handling improvements

Return JSON with detailed suggestions:
{{
    "modifications": [
        {{
            "action": "add|modify|delete",
            "target_type": "function|class|import|variable",
            "target_name": "specific name",
            "line_number": 0,
            "old_code": "existing code if modifying",
            "new_code": "new/modified code",
            "explanation": "why this change is needed"
        }}
    ],
    "new_dependencies": [],
    "testing_suggestions": [],
    "potential_issues": []
}}
"""

        try:
            response = llm_client.chat_completion(
                messages=[{"role": "user", "content": prompt}],
                system_prompt=self.system_prompt
            )
            
            if response:
                cleaned_response = self._clean_json_response(response)
                suggestions = json.loads(cleaned_response)
                
                # Add metadata
                suggestions['file_path'] = file_path
                suggestions['timestamp'] = FileHandler.get_file_info(Path(file_path)).get('modified', 0)
                
                return suggestions
            
            return {"modifications": []}
            
        except Exception as e:
            print(f"Error generating suggestions for {file_path}: {e}")
            return {"modifications": []}
    
    def create_modification_plan(self, target_files: List[Dict[str, Any]], parsed_config: Dict[str, Any]) -> Dict[str, Any]:
        """Create a comprehensive modification plan"""
        
        plan = {
            "files_to_modify": [],
            "execution_order": [],
            "dependencies": [],
            "estimated_complexity": "low",
            "risks": [],
            "backup_required": True
        }
        
        # Generate suggestions for each file
        for target_file in target_files:
            suggestions = self.suggest_file_changes(target_file, parsed_config)
            
            plan["files_to_modify"].append({
                "file_path": target_file['file_path'],
                "priority": target_file['priority'],
                "suggestions": suggestions
            })
        
        # Determine execution order based on dependencies
        plan["execution_order"] = self._determine_execution_order(plan["files_to_modify"])
        
        # Assess overall complexity
        total_modifications = sum(len(f["suggestions"].get("modifications", [])) for f in plan["files_to_modify"])
        if total_modifications > 20:
            plan["estimated_complexity"] = "high"
        elif total_modifications > 10:
            plan["estimated_complexity"] = "medium"
        
        return plan
    
    def _determine_execution_order(self, files_to_modify: List[Dict[str, Any]]) -> List[str]:
        """Determine the order in which files should be modified"""
        
        # Simple heuristic: prioritize by file type and dependencies
        order = []
        
        # First: Configuration and utility files
        for file_info in files_to_modify:
            file_path = Path(file_info['file_path'])
            if 'config' in file_path.name.lower() or 'util' in file_path.name.lower():
                order.append(file_info['file_path'])
        
        # Second: Data loading files
        for file_info in files_to_modify:
            file_path = Path(file_info['file_path'])
            if 'load' in file_path.name.lower() or 'input' in file_path.name.lower():
                if file_info['file_path'] not in order:
                    order.append(file_info['file_path'])
        
        # Third: Processing/transformation files
        for file_info in files_to_modify:
            file_path = Path(file_info['file_path'])
            if 'process' in file_path.name.lower() or 'transform' in file_path.name.lower():
                if file_info['file_path'] not in order:
                    order.append(file_info['file_path'])
        
        # Fourth: Output files
        for file_info in files_to_modify:
            file_path = Path(file_info['file_path'])
            if 'output' in file_path.name.lower() or 'write' in file_path.name.lower():
                if file_info['file_path'] not in order:
                    order.append(file_info['file_path'])
        
        # Finally: Main/pipeline files
        for file_info in files_to_modify:
            if file_info['file_path'] not in order:
                order.append(file_info['file_path'])
        
        return order
    
    def _clean_json_response(self, response: str) -> str:
        """Clean LLM response to extract valid JSON"""
        if "```json" in response:
            start = response.find("```json") + 7
            end = response.find("```", start)
            response = response[start:end].strip()
        elif "```" in response:
            start = response.find("```") + 3
            end = response.find("```", start)
            response = response[start:end].strip()
        
        return response.strip