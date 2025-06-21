"""
Main entry point for Data Dictionary Agent
"""
import asyncio
import argparse
import json
from pathlib import Path
from typing import Optional
import sys

from config import SAMPLE_DATA_DIR, OUTPUTS_DIR, SUPPORTED_FORMATS
from core.data_loader import DataLoader
from agents.workflow import DataDictionaryWorkflow


async def main():
    parser = argparse.ArgumentParser(description="Generate data dictionary from CSV/Excel files")
    parser.add_argument("file_path", help="Path to CSV or Excel file")
    parser.add_argument("--output", "-o", help="Output file path (optional)")
    parser.add_argument("--format", "-f", choices=['json', 'yaml', 'markdown'], 
                       default='json', help="Output format")
    parser.add_argument("--sample", "-s", type=int, default=1000, 
                       help="Sample size for analysis (default: 1000)")
    
    args = parser.parse_args()
    
    # Validate file path
    file_path = Path(args.file_path)
    if not file_path.exists():
        print(f"Error: File {file_path} not found")
        sys.exit(1)
    
    if file_path.suffix.lower() not in SUPPORTED_FORMATS:
        print(f"Error: Unsupported file format. Supported formats: {SUPPORTED_FORMATS}")
        sys.exit(1)
    
    # Set output path
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = OUTPUTS_DIR / f"{file_path.stem}_dictionary.{args.format}"
    
    print(f"📊 Analyzing file: {file_path}")
    print(f"📝 Output will be saved to: {output_path}")
    print(f"🔍 Sample size: {args.sample}")
    
    try:
        # Initialize workflow
        workflow = DataDictionaryWorkflow()
        
        # Run the workflow
        result = await workflow.run(
            file_path=str(file_path),
            output_format=args.format,
            sample_size=args.sample
        )
        
        # Save results
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if args.format == 'json':
            with open(output_path, 'w') as f:
                json.dump(result, f, indent=2, default=str)
        elif args.format == 'yaml':
            import yaml
            with open(output_path, 'w') as f:
                yaml.dump(result, f, default_flow_style=False)
        elif args.format == 'markdown':
            with open(output_path, 'w') as f:
                f.write(result.get('markdown_output', str(result)))
        
        print(f"✅ Data dictionary generated successfully!")
        print(f"📄 Output saved to: {output_path}")
        
        # Print summary
        if 'summary' in result:
            print("\n📋 Summary:")
            print(f"  - Total columns: {result['summary'].get('total_columns', 'N/A')}")
            print(f"  - Total rows: {result['summary'].get('total_rows', 'N/A')}")
            print(f"  - Missing values: {result['summary'].get('missing_values_count', 'N/A')}")
            print(f"  - Data types: {len(result['summary'].get('data_types', {}))} unique")
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        sys.exit(1)


def demo():
    """Run a demo with sample data"""
    print("🚀 Running Data Dictionary Agent Demo")
    
    # Check for sample files
    sample_files = list(SAMPLE_DATA_DIR.glob("*.csv")) + list(SAMPLE_DATA_DIR.glob("*.xlsx"))
    
    if not sample_files:
        print("No sample files found. Please add some CSV or Excel files to the sample_data directory.")
        return
    
    # Use the first sample file
    sample_file = sample_files[0]
    print(f"Using sample file: {sample_file}")
    
    # Run the main function with sample file
    sys.argv = ['main.py', str(sample_file)]
    asyncio.run(main())


if __name__ == "__main__":
    if len(sys.argv) == 1:
        # No arguments provided, run demo
        demo()
    else:
        asyncio.run(main())