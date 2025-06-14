#!/usr/bin/env python3
"""
Script to verify consistency between two lerobot datasets with different episode indices.
This script checks hash values of corresponding files in data and videos subfolders.
"""

import hashlib
import json
import os
import argparse
from pathlib import Path
import pandas as pd
from typing import Dict, List, Tuple, Set

def load_jsonl(file_path: str) -> List[Dict]:
    """Load JSONL file and return list of dictionaries."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data

def calculate_file_hash(file_path: Path) -> str:
    """Calculate SHA256 hash of a file."""
    hash_sha256 = hashlib.sha256()
    try:
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()
    except Exception as e:
        print(f"Error calculating hash for {file_path}: {e}")
        return ""

def calculate_parquet_content_hash(file_path: Path, exclude_columns: Set[str] = None) -> str:
    """
    Calculate hash of parquet file content excluding specified columns.
    This is useful when index columns are different but data content should be the same.
    """
    if exclude_columns is None:
        exclude_columns = {'episode_index', 'index'}
    
    try:
        df = pd.read_parquet(file_path)
        
        # Remove columns that are expected to be different
        content_df = df.drop(columns=[col for col in exclude_columns if col in df.columns])
        
        # Sort by frame_index to ensure consistent ordering
        if 'frame_index' in content_df.columns:
            content_df = content_df.sort_values('frame_index')
        
        # Convert to string representation for hashing
        content_str = content_df.to_string(index=False)
        
        # Calculate hash
        return hashlib.sha256(content_str.encode()).hexdigest()
    except Exception as e:
        print(f"Error calculating content hash for {file_path}: {e}")
        return ""

def build_episode_mapping(source_episodes: List[Dict], target_episodes: List[Dict], start_episode: int) -> Dict[int, int]:
    """
    Build mapping from source episode indices to target episode indices.
    
    Args:
        source_episodes: Original episodes list
        target_episodes: New episodes list  
        start_episode: Starting episode number in source
    
    Returns:
        Dictionary mapping source episode index to target episode index
    """
    mapping = {}
    
    # Filter source episodes from start_episode onwards
    filtered_source = [ep for ep in source_episodes if ep["episode_index"] >= start_episode]
    
    # Create mapping
    for i, source_ep in enumerate(filtered_source):
        if i < len(target_episodes):
            mapping[source_ep["episode_index"]] = target_episodes[i]["episode_index"]
    
    return mapping

def verify_videos(source_dir: Path, target_dir: Path, episode_mapping: Dict[int, int]) -> Tuple[int, int, List[str]]:
    """
    Verify video files between source and target datasets.
    
    Returns:
        Tuple of (total_files, matching_files, error_list)
    """
    print("Verifying video files...")
    total_files = 0
    matching_files = 0
    errors = []
    
    source_video_dir = source_dir / "videos" / "chunk-000"
    target_video_dir = target_dir / "videos" / "chunk-000"
    
    # Check both camera types
    for camera in ["observation.images.room", "observation.images.wrist"]:
        print(f"  Checking {camera} videos...")
        
        for source_idx, target_idx in episode_mapping.items():
            source_file = source_video_dir / camera / f"episode_{source_idx:06d}.mp4"
            target_file = target_video_dir / camera / f"episode_{target_idx:06d}.mp4"
            
            if source_file.exists() and target_file.exists():
                total_files += 1
                
                source_hash = calculate_file_hash(source_file)
                target_hash = calculate_file_hash(target_file)
                
                if source_hash == target_hash and source_hash != "":
                    matching_files += 1
                    print(f"    ✓ {camera}/episode_{source_idx:06d}.mp4 -> episode_{target_idx:06d}.mp4")
                else:
                    error_msg = f"    ✗ Hash mismatch: {camera}/episode_{source_idx:06d}.mp4 -> episode_{target_idx:06d}.mp4"
                    errors.append(error_msg)
                    print(error_msg)
            else:
                if not source_file.exists():
                    error_msg = f"    ✗ Source file missing: {source_file}"
                    errors.append(error_msg)
                    print(error_msg)
                if not target_file.exists():
                    error_msg = f"    ✗ Target file missing: {target_file}"
                    errors.append(error_msg)
                    print(error_msg)
    
    return total_files, matching_files, errors

def verify_data(source_dir: Path, target_dir: Path, episode_mapping: Dict[int, int]) -> Tuple[int, int, List[str]]:
    """
    Verify data files between source and target datasets.
    
    Returns:
        Tuple of (total_files, matching_files, error_list)
    """
    print("Verifying data files...")
    total_files = 0
    matching_files = 0
    errors = []
    
    source_data_dir = source_dir / "data" / "chunk-000"
    target_data_dir = target_dir / "data" / "chunk-000"
    
    for source_idx, target_idx in episode_mapping.items():
        source_file = source_data_dir / f"episode_{source_idx:06d}.parquet"
        target_file = target_data_dir / f"episode_{target_idx:06d}.parquet"
        
        if source_file.exists() and target_file.exists():
            total_files += 1
            
            # Calculate content hash excluding index columns
            source_hash = calculate_parquet_content_hash(source_file)
            target_hash = calculate_parquet_content_hash(target_file)
            
            if source_hash == target_hash and source_hash != "":
                matching_files += 1
                print(f"  ✓ episode_{source_idx:06d}.parquet -> episode_{target_idx:06d}.parquet")
            else:
                error_msg = f"  ✗ Content mismatch: episode_{source_idx:06d}.parquet -> episode_{target_idx:06d}.parquet"
                errors.append(error_msg)
                print(error_msg)
        else:
            if not source_file.exists():
                error_msg = f"  ✗ Source file missing: {source_file}"
                errors.append(error_msg)
                print(error_msg)
            if not target_file.exists():
                error_msg = f"  ✗ Target file missing: {target_file}"
                errors.append(error_msg)
                print(error_msg)
    
    return total_files, matching_files, errors

def verify_metadata(source_dir: Path, target_dir: Path, start_episode: int) -> List[str]:
    """
    Verify that metadata is consistent between datasets.
    
    Returns:
        List of error messages
    """
    print("Verifying metadata consistency...")
    errors = []
    
    try:
        # Load metadata
        source_episodes = load_jsonl(source_dir / "meta" / "episodes.jsonl")
        target_episodes = load_jsonl(target_dir / "meta" / "episodes.jsonl")
        
        # Check episode count
        expected_count = len([ep for ep in source_episodes if ep["episode_index"] >= start_episode])
        actual_count = len(target_episodes)
        
        if expected_count != actual_count:
            error_msg = f"Episode count mismatch: expected {expected_count}, got {actual_count}"
            errors.append(error_msg)
            print(f"  ✗ {error_msg}")
        else:
            print(f"  ✓ Episode count matches: {actual_count}")
        
        # Check episode tasks consistency
        source_filtered = [ep for ep in source_episodes if ep["episode_index"] >= start_episode]
        for i, (source_ep, target_ep) in enumerate(zip(source_filtered, target_episodes)):
            if source_ep["tasks"] != target_ep["tasks"]:
                error_msg = f"Task mismatch at episode {i}: {source_ep['tasks']} != {target_ep['tasks']}"
                errors.append(error_msg)
                print(f"  ✗ {error_msg}")
            
            if source_ep["length"] != target_ep["length"]:
                error_msg = f"Length mismatch at episode {i}: {source_ep['length']} != {target_ep['length']}"
                errors.append(error_msg)
                print(f"  ✗ {error_msg}")
        
        if not errors:
            print("  ✓ All episode metadata matches")
            
    except Exception as e:
        error_msg = f"Error verifying metadata: {e}"
        errors.append(error_msg)
        print(f"  ✗ {error_msg}")
    
    return errors

def main():
    parser = argparse.ArgumentParser(description="Verify consistency between two lerobot datasets")
    parser.add_argument("--source_dir", help="Path to source dataset directory")
    parser.add_argument("--target_dir", help="Path to target dataset directory")
    parser.add_argument("--start_episode", type=int, default=144,
                        help="Starting episode number from source dataset (default: 144)")
    
    args = parser.parse_args()
    
    source_path = Path(args.source_dir)
    target_path = Path(args.target_dir)
    
    # Validate directories
    if not source_path.exists():
        print(f"Error: Source directory {source_path} does not exist!")
        return 1
    
    if not target_path.exists():
        print(f"Error: Target directory {target_path} does not exist!")
        return 1
    
    print(f"Verifying datasets:")
    print(f"  Source: {source_path}")
    print(f"  Target: {target_path}")
    print(f"  Start episode: {args.start_episode}")
    print()
    
    try:
        # Load episode metadata to build mapping
        source_episodes = load_jsonl(source_path / "meta" / "episodes.jsonl")
        target_episodes = load_jsonl(target_path / "meta" / "episodes.jsonl")
        
        # Build episode mapping
        episode_mapping = build_episode_mapping(source_episodes, target_episodes, args.start_episode)
        
        print(f"Episode mapping (first 5): {dict(list(episode_mapping.items())[:5])}")
        print(f"Total episodes to verify: {len(episode_mapping)}")
        print()
        
        # Verify metadata
        metadata_errors = verify_metadata(source_path, target_path, args.start_episode)
        print()
        
        # Verify data files
        data_total, data_matching, data_errors = verify_data(source_path, target_path, episode_mapping)
        print()
        
        # Verify video files
        video_total, video_matching, video_errors = verify_videos(source_path, target_path, episode_mapping)
        print()
        
        # Summary
        total_errors = len(metadata_errors) + len(data_errors) + len(video_errors)
        
        print("=" * 60)
        print("VERIFICATION SUMMARY")
        print("=" * 60)
        print(f"Metadata errors: {len(metadata_errors)}")
        print(f"Data files - Total: {data_total}, Matching: {data_matching}, Errors: {len(data_errors)}")
        print(f"Video files - Total: {video_total}, Matching: {video_matching}, Errors: {len(video_errors)}")
        print(f"Total errors: {total_errors}")
        
        if total_errors == 0:
            print("✓ All files verified successfully! Datasets are consistent.")
            return 0
        else:
            print("✗ Verification failed! See errors above.")
            return 1
            
    except Exception as e:
        print(f"Error during verification: {e}")
        return 1

if __name__ == "__main__":
    exit(main()) 