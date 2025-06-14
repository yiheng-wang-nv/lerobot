#!/usr/bin/env python3
"""
Script to create a new lerobot dataset from existing dataset starting from episode 144.
This will copy all episodes from 144 onwards and re-index them starting from 0.
"""

import json
import os
import shutil
from pathlib import Path
import pandas as pd
import argparse
from typing import Dict, List, Any

def load_jsonl(file_path: str) -> List[Dict]:
    """Load JSONL file and return list of dictionaries."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data

def save_jsonl(data: List[Dict], file_path: str) -> None:
    """Save list of dictionaries to JSONL file."""
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

def save_json(data: Dict, file_path: str) -> None:
    """Save dictionary to JSON file."""
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

def create_new_dataset(source_dir: str, target_dir: str, start_episode: int = 144):
    """
    Create a new dataset from existing dataset starting from specified episode.
    
    Args:
        source_dir: Path to source dataset directory
        target_dir: Path to target dataset directory
        start_episode: Starting episode number (default: 144)
    """
    source_path = Path(source_dir)
    target_path = Path(target_dir)
    
    # Create target directory structure
    target_path.mkdir(parents=True, exist_ok=True)
    (target_path / "meta").mkdir(exist_ok=True)
    (target_path / "data" / "chunk-000").mkdir(parents=True, exist_ok=True)
    (target_path / "videos" / "chunk-000").mkdir(parents=True, exist_ok=True)
    
    # Load original metadata
    print("Loading original metadata...")
    episodes = load_jsonl(source_path / "meta" / "episodes.jsonl")
    episodes_stats = load_jsonl(source_path / "meta" / "episodes_stats.jsonl")
    
    with open(source_path / "meta" / "info.json", 'r', encoding='utf-8') as f:
        info = json.load(f)
    
    with open(source_path / "meta" / "modality.json", 'r', encoding='utf-8') as f:
        modality = json.load(f)
    
    # Filter episodes from start_episode onwards
    print(f"Filtering episodes from {start_episode} onwards...")
    filtered_episodes = [ep for ep in episodes if ep["episode_index"] >= start_episode]
    filtered_episodes_stats = [ep for ep in episodes_stats if ep["episode_index"] >= start_episode]
    
    # Create mapping from old episode index to new episode index
    episode_mapping = {}
    for i, episode in enumerate(filtered_episodes):
        old_index = episode["episode_index"]
        episode_mapping[old_index] = i
        print(f"Episode mapping: {old_index} -> {i}")
    
    # Re-index episodes starting from 0
    print("Re-indexing episodes...")
    for i, episode in enumerate(filtered_episodes):
        episode["episode_index"] = i
    
    # Update episode stats with new indices
    total_frames = 0
    for i, episode_stat in enumerate(filtered_episodes_stats):
        old_index = episode_stat["episode_index"]
        episode_stat["episode_index"] = i
        
        # Update frame indices and global index in stats
        episode_length = filtered_episodes[i]["length"]
        
        # Update stats to reflect new global indices
        if "stats" in episode_stat:
            stats = episode_stat["stats"]
            if "index" in stats:
                stats["index"]["min"] = [total_frames]
                stats["index"]["max"] = [total_frames + episode_length - 1]
                stats["index"]["mean"] = [total_frames + episode_length / 2 - 0.5]
                # Calculate standard deviation properly
                if episode_length > 1:
                    variance = sum((j - (episode_length - 1) / 2) ** 2 for j in range(episode_length)) / episode_length
                    stats["index"]["std"] = [variance ** 0.5]
                else:
                    stats["index"]["std"] = [0.0]
            
            if "episode_index" in stats:
                stats["episode_index"]["min"] = [i]
                stats["episode_index"]["max"] = [i]
                stats["episode_index"]["mean"] = [float(i)]
                stats["episode_index"]["std"] = [0.0]
        
        total_frames += episode_length
        print(f"Updated stats for episode {old_index} -> {i}")
    
    # Update info.json
    print("Updating info.json...")
    new_info = info.copy()
    new_info["total_episodes"] = len(filtered_episodes)
    new_info["total_frames"] = total_frames
    new_info["splits"]["train"] = f"0:{len(filtered_episodes)}"
    
    # Calculate total videos (2 cameras per episode)
    new_info["total_videos"] = len(filtered_episodes) * 2
    
    # Save new metadata
    print("Saving new metadata...")
    save_jsonl(filtered_episodes, target_path / "meta" / "episodes.jsonl")
    save_jsonl(filtered_episodes_stats, target_path / "meta" / "episodes_stats.jsonl")
    save_json(new_info, target_path / "meta" / "info.json")
    save_json(modality, target_path / "meta" / "modality.json")
    
    # Copy and update data files
    print("Copying and updating data files...")
    source_data_dir = source_path / "data" / "chunk-000"
    target_data_dir = target_path / "data" / "chunk-000"
    
    # Calculate cumulative frames for global index update
    cumulative_frames = [0]
    for episode in filtered_episodes:
        cumulative_frames.append(cumulative_frames[-1] + episode["length"])
    
    for old_index, new_index in episode_mapping.items():
        old_filename = f"episode_{old_index:06d}.parquet"
        new_filename = f"episode_{new_index:06d}.parquet"
        
        source_file = source_data_dir / old_filename
        target_file = target_data_dir / new_filename
        
        if source_file.exists():
            print(f"Copying {old_filename} -> {new_filename}")
            
            # Load parquet and update indices
            df = pd.read_parquet(source_file)
            
            # Update episode_index column
            if 'episode_index' in df.columns:
                df['episode_index'] = new_index
            
            # Update global index
            if 'index' in df.columns:
                frame_indices = df['frame_index'].values
                df['index'] = cumulative_frames[new_index] + frame_indices
            
            # Save updated parquet
            df.to_parquet(target_file, index=False)
        else:
            print(f"Warning: Source file {source_file} not found!")
    
    # Copy video files
    print("Copying video files...")
    source_video_dir = source_path / "videos" / "chunk-000"
    target_video_dir = target_path / "videos" / "chunk-000"
    
    # Create camera directories
    for camera in ["observation.images.room", "observation.images.wrist"]:
        (target_video_dir / camera).mkdir(exist_ok=True)
        
        for old_index, new_index in episode_mapping.items():
            old_filename = f"episode_{old_index:06d}.mp4"
            new_filename = f"episode_{new_index:06d}.mp4"
            
            source_file = source_video_dir / camera / old_filename
            target_file = target_video_dir / camera / new_filename
            
            if source_file.exists():
                print(f"Copying {camera}/{old_filename} -> {camera}/{new_filename}")
                shutil.copy2(source_file, target_file)
            else:
                print(f"Warning: Source video file {source_file} not found!")
    
    print(f"\nDataset creation completed!")
    print(f"Original episodes: {info['total_episodes']}")
    print(f"New episodes: {len(filtered_episodes)} (episodes {start_episode}-{max(episode_mapping.keys())} -> 0-{len(filtered_episodes)-1})")
    print(f"Original total frames: {info['total_frames']}")
    print(f"New total frames: {total_frames}")
    print(f"New dataset saved to: {target_path}")

def main():
    parser = argparse.ArgumentParser(description="Create new lerobot dataset from existing dataset")
    parser.add_argument("--source_dir", help="Path to source dataset directory")
    parser.add_argument("--target_dir", help="Path to target dataset directory")
    parser.add_argument("--start_episode", type=int, default=144, 
                        help="Starting episode number (default: 144)")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.source_dir):
        print(f"Error: Source directory {args.source_dir} does not exist!")
        return
    
    if os.path.exists(args.target_dir):
        response = input(f"Target directory {args.target_dir} already exists. Continue? (y/N): ")
        if response.lower() != 'y':
            print("Aborted.")
            return
    
    create_new_dataset(args.source_dir, args.target_dir, args.start_episode)

if __name__ == "__main__":
    main() 