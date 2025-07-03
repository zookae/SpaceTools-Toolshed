#!/usr/bin/env python

# Copyright (c) 2025-2026 NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""
Merge multiple parquet datasets into a single file.
"""
import pandas as pd
import argparse
from pathlib import Path


def merge_datasets(input_paths, output_path):
    """
    Merge multiple parquet files into a single parquet file.
    
    Args:
        input_paths: List of paths to input parquet files
        output_path: Path to output parquet file
    """
    if len(input_paths) < 2:
        raise ValueError("At least 2 input files are required for merging")
    
    print(f"Merging {len(input_paths)} datasets...")
    dfs = []
    for i, path in enumerate(input_paths, 1):
        print(f"  [{i}/{len(input_paths)}] Loading {path}...")
        df = pd.read_parquet(path)
        print(f"    Rows: {len(df):,}")
        dfs.append(df)
    
    print(f"  Concatenating datasets...")
    merged_df = pd.concat(dfs, ignore_index=True)
    print(f"  Total rows after merge: {len(merged_df):,}")
    
    # Reassign unique sequential indices to extra_info['index']
    print(f"  Reassigning unique indices to extra_info['index']...")
    for i in range(len(merged_df)):
        if 'extra_info' in merged_df.iloc[i] and isinstance(merged_df.iloc[i]['extra_info'], dict):
            merged_df.iloc[i]['extra_info']['index'] = i
    
    print(f"  Verifying unique indices...")
    unique_indices = set()
    for i in range(len(merged_df)):
        if 'extra_info' in merged_df.iloc[i]:
            idx = merged_df.iloc[i]['extra_info'].get('index', -1)
            unique_indices.add(idx)
    print(f"    Unique extra_info['index'] values: {len(unique_indices)}")
    
    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged_df.to_parquet(output_path)
    
    print(f"\n✅ Merged dataset saved:")
    print(f"  Output: {output_path}")
    print(f"  Total rows: {len(merged_df):,}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Merge multiple parquet datasets into a single file",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  python merge_datasets.py \\
    --input data/depth.parquet data/object.parquet data/spatial.parquet data/vacant.parquet \\
    --output data/merged.parquet
        """
    )
    parser.add_argument("--input", required=True, nargs='+', 
                        help="Input parquet files (at least 2 required)")
    parser.add_argument("--output", required=True, 
                        help="Output parquet file")
    
    args = parser.parse_args()
    merge_datasets(args.input, args.output)
