# Copyright (c) 2025-2026 NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""
Compare evaluation results from multiple models.

This script provides utilities to compare and visualize results from different
model evaluations.
"""

import argparse
import json
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd
from datetime import datetime


def load_results(file_path: str) -> Dict[str, Any]:
    """Load results from a JSON file."""
    with open(file_path, 'r') as f:
        return json.load(f)


def compare_models(results_files: List[str]) -> pd.DataFrame:
    """
    Compare results from multiple model evaluations.
    
    Args:
        results_files: List of paths to result JSON files
        
    Returns:
        DataFrame with comparison metrics
    """
    comparisons = []
    
    for file_path in results_files:
        results = load_results(file_path)
        # Was: summary = results.get('summary', results)
        summary = results['summary'] if 'summary' in results else results
        # Was: metadata = results.get('metadata', {})
        metadata = results['metadata'] if 'metadata' in results else {}
        
        row = {
            'file': Path(file_path).name,
            # Was: 'model': metadata.get('model', 'unknown'),
            'model': metadata['model'] if 'model' in metadata else 'unknown',
            # Was: 'use_tools': metadata.get('use_tools', False),
            'use_tools': metadata['use_tools'] if 'use_tools' in metadata else False,
            # Was: 'num_examples': summary.get('num_examples', 0),
            'num_examples': summary['num_examples'] if 'num_examples' in summary else 0,
            # Was: 'accuracy': summary.get('accuracy', 0.0),
            'accuracy': summary['accuracy'] if 'accuracy' in summary else None,
            # Was: 'average_score': summary.get('average_score', 0.0),
            'average_score': summary['average_score'] if 'average_score' in summary else None,
            # Was: 'error_rate': summary.get('error_rate', 0.0),
            'error_rate': summary['error_rate'] if 'error_rate' in summary else None,
        }
        
        # Add metrics by type if available
        if 'metrics_by_type' in summary:
            for qtype, metrics in summary['metrics_by_type'].items():
                # Was: row[f'{qtype}_accuracy'] = metrics.get('accuracy', 0.0)
                row[f'{qtype}_accuracy'] = metrics['accuracy'] if 'accuracy' in metrics else None
                # Was: row[f'{qtype}_count'] = metrics.get('count', 0)
                row[f'{qtype}_count'] = metrics['count'] if 'count' in metrics else 0
        
        comparisons.append(row)
    
    return pd.DataFrame(comparisons)


def print_comparison_table(df: pd.DataFrame):
    """Print a formatted comparison table."""
    print("\n" + "="*80)
    print("MODEL COMPARISON")
    print("="*80)
    
    # Basic info
    print("\nModels Evaluated:")
    for _, row in df.iterrows():
        tools_str = "with tools" if row['use_tools'] else "no tools"
        print(f"  - {row['model']} ({tools_str}): {row['file']}")
    
    # Overall metrics
    print("\nOverall Performance:")
    print("-" * 50)
    print(f"{'Model':<20} {'Tools':<10} {'Accuracy':<12} {'Avg Score':<12} {'Error Rate':<12}")
    print("-" * 50)
    
    for _, row in df.iterrows():
        tools_str = "Yes" if row['use_tools'] else "No"
        acc_str = f"{row['accuracy']:.2%}" if row['accuracy'] is not None else "N/A"
        avg_str = f"{row['average_score']:.3f}" if row['average_score'] is not None else "N/A"
        err_str = f"{row['error_rate']:.2%}" if row['error_rate'] is not None else "N/A"
        print(f"{row['model']:<20} {tools_str:<10} "
              f"{acc_str:<12} {avg_str:<12} "
              f"{err_str:<12}")
    
    # Performance by type if available
    type_cols = [col for col in df.columns if col.endswith('_accuracy')]
    if type_cols:
        print("\nPerformance by Question Type:")
        print("-" * 60)
        
        types = list(set(col.replace('_accuracy', '') for col in type_cols))
        
        # Header
        header = f"{'Model':<20} {'Tools':<10}"
        for qtype in types:
            header += f" {qtype:<15}"
        print(header)
        print("-" * 60)
        
        # Data
        for _, row in df.iterrows():
            tools_str = "Yes" if row['use_tools'] else "No"
            line = f"{row['model']:<20} {tools_str:<10}"
            for qtype in types:
                # Was: acc = row.get(f'{qtype}_accuracy', 0.0)
                acc = row[f'{qtype}_accuracy'] if f'{qtype}_accuracy' in row else None
                # Was: count = row.get(f'{qtype}_count', 0)
                count = row[f'{qtype}_count'] if f'{qtype}_count' in row else 0
                acc_str = f"{acc:.2%}" if acc is not None else "N/A"
                line += f" {acc_str:>6} ({count:>3})"
            print(line)
    
    print("="*80 + "\n")


def save_comparison_html(df: pd.DataFrame, output_path: str):
    """Save comparison as an HTML report."""
    html_content = f"""
    <html>
    <head>
        <title>Model Evaluation Comparison</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            h1 {{ color: #333; }}
            table {{ border-collapse: collapse; width: 100%; margin-top: 20px; }}
            th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
            th {{ background-color: #4CAF50; color: white; }}
            tr:nth-child(even) {{ background-color: #f2f2f2; }}
            .metric {{ text-align: right; }}
            .timestamp {{ color: #666; font-size: 0.9em; }}
        </style>
    </head>
    <body>
        <h1>Model Evaluation Comparison</h1>
        <p class="timestamp">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        
        <h2>Overall Performance</h2>
        <table>
            <tr>
                <th>Model</th>
                <th>Tools</th>
                <th>Examples</th>
                <th>Accuracy</th>
                <th>Average Score</th>
                <th>Error Rate</th>
            </tr>
    """
    
    for _, row in df.iterrows():
        tools_icon = "✓" if row['use_tools'] else "✗"
        html_content += f"""
            <tr>
                <td>{row['model']}</td>
                <td style="text-align: center;">{tools_icon}</td>
                <td class="metric">{row['num_examples']}</td>
                <td class="metric">{row['accuracy']:.2%}</td>
                <td class="metric">{row['average_score']:.3f}</td>
                <td class="metric">{row['error_rate']:.2%}</td>
            </tr>
        """
    
    html_content += """
        </table>
    </body>
    </html>
    """
    
    with open(output_path, 'w') as f:
        f.write(html_content)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Compare evaluation results from multiple models")
    parser.add_argument('results', nargs='+', help='Result JSON files to compare')
    parser.add_argument('--output-html', help='Save comparison as HTML report')
    parser.add_argument('--output-csv', help='Save comparison as CSV file')
    
    args = parser.parse_args()
    
    # Validate input files
    for file_path in args.results:
        if not Path(file_path).exists():
            print(f"Error: File not found: {file_path}")
            return
    
    # Compare models
    df = compare_models(args.results)
    
    # Print comparison
    print_comparison_table(df)
    
    # Save outputs if requested
    if args.output_html:
        save_comparison_html(df, args.output_html)
        print(f"HTML report saved to: {args.output_html}")
    
    if args.output_csv:
        df.to_csv(args.output_csv, index=False)
        print(f"CSV data saved to: {args.output_csv}")


if __name__ == '__main__':
    main()
