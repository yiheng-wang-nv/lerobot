# lerobot/utils/update_data_index.py

import os
import sys
import json

# List of episode indices to remove
EPISODES_TO_REMOVE = [75, 101, 106, 153, 196]  # Add your episode indices here

def remove_from_jsonl(filepath, episode_indices_to_remove):
    new_lines = []
    removed_count = 0
    with open(filepath, "r") as f:
        for line in f:
            try:
                obj = json.loads(line)
                if obj.get("episode_index") in episode_indices_to_remove:
                    removed_count += 1
                    continue
                new_lines.append(line)
            except Exception:
                new_lines.append(line)
    with open(filepath, "w") as f:
        f.writelines(new_lines)
    if removed_count > 0:
        print(f"Removed {removed_count} episodes from {filepath}")

def remove_data_files(data_dir, episode_indices_to_remove):
    for episode_index in episode_indices_to_remove:
        ep_str = f"{episode_index:06d}"
        fname = f"episode_{ep_str}.parquet"
        fpath = os.path.join(data_dir, fname)
        if os.path.exists(fpath):
            os.remove(fpath)
            print(f"Removed data file: {fpath}")

def remove_video_files(video_dir, episode_indices_to_remove):
    for episode_index in episode_indices_to_remove:
        ep_str = f"{episode_index:06d}"
        fname = f"episode_{ep_str}.mp4"
        fpath = os.path.join(video_dir, fname)
        if os.path.exists(fpath):
            os.remove(fpath)
            print(f"Removed video file: {fpath}")

def cleanup_and_reindex():
    base_dir = "2_cameras_fps15_enhanced_gripper"
    meta_dir = os.path.join(base_dir, "meta")
    data_dir = os.path.join(base_dir, "data/chunk-000")
    video_dirs = [
        os.path.join(base_dir, "videos/chunk-000/observation.images.room"),
        os.path.join(base_dir, "videos/chunk-000/observation.images.wrist"),
    ]

    print(f"Removing episodes: {EPISODES_TO_REMOVE}")
    
    # Step 1: Remove files and meta entries for specified episodes
    # Remove from meta .jsonl files
    for fname in os.listdir(meta_dir):
        if fname.endswith(".jsonl"):
            fpath = os.path.join(meta_dir, fname)
            remove_from_jsonl(fpath, EPISODES_TO_REMOVE)

    # Remove data files
    if os.path.exists(data_dir):
        remove_data_files(data_dir, EPISODES_TO_REMOVE)

    # Remove video files
    for vdir in video_dirs:
        if os.path.exists(vdir):
            remove_video_files(vdir, EPISODES_TO_REMOVE)

    print("Removal complete. Starting reindexing...")

    # Step 2: Gather all remaining episode indices
    indices = set()
    # From data files
    if os.path.exists(data_dir):
        for fname in os.listdir(data_dir):
            if fname.startswith("episode_") and fname.endswith(".parquet"):
                idx = int(fname.split("_")[1].split(".")[0])
                indices.add(idx)
    # From video files
    for vdir in video_dirs:
        if os.path.exists(vdir):
            for fname in os.listdir(vdir):
                if fname.startswith("episode_") and fname.endswith(".mp4"):
                    idx = int(fname.split("_")[1].split(".")[0])
                    indices.add(idx)
    # From meta files
    for fname in os.listdir(meta_dir):
        if fname.endswith(".jsonl"):
            with open(os.path.join(meta_dir, fname), "r") as f:
                for line in f:
                    try:
                        obj = json.loads(line)
                        idx = obj.get("episode_index")
                        if idx is not None:
                            indices.add(idx)
                    except Exception:
                        continue

    # Step 3: Sort and create mapping: old_idx -> new_idx
    sorted_indices = sorted(indices)
    idx_map = {old: new for new, old in enumerate(sorted_indices)}
    
    print(f"Reindexing {len(sorted_indices)} episodes...")

    # Step 4: Rename data files
    if os.path.exists(data_dir):
        for old_idx, new_idx in idx_map.items():
            old_name = f"episode_{old_idx:06d}.parquet"
            new_name = f"episode_{new_idx:06d}.parquet"
            old_path = os.path.join(data_dir, old_name)
            new_path = os.path.join(data_dir, new_name)
            if os.path.exists(old_path) and old_path != new_path:
                os.rename(old_path, new_path)
                print(f"Renamed {old_path} -> {new_path}")

    # Step 5: Rename video files
    for vdir in video_dirs:
        if os.path.exists(vdir):
            for old_idx, new_idx in idx_map.items():
                old_name = f"episode_{old_idx:06d}.mp4"
                new_name = f"episode_{new_idx:06d}.mp4"
                old_path = os.path.join(vdir, old_name)
                new_path = os.path.join(vdir, new_name)
                if os.path.exists(old_path) and old_path != new_path:
                    os.rename(old_path, new_path)
                    print(f"Renamed {old_path} -> {new_path}")

    # Step 6: Update meta .jsonl files
    for fname in os.listdir(meta_dir):
        if fname.endswith(".jsonl"):
            fpath = os.path.join(meta_dir, fname)
            new_lines = []
            with open(fpath, "r") as f:
                for line in f:
                    try:
                        obj = json.loads(line)
                        old_idx = obj.get("episode_index")
                        if old_idx in idx_map:
                            obj["episode_index"] = idx_map[old_idx]
                        new_lines.append(json.dumps(obj) + "\n")
                    except Exception:
                        new_lines.append(line)
            with open(fpath, "w") as f:
                f.writelines(new_lines)
            print(f"Updated indices in {fpath}")

    # Step 7: Update info.json
    info_path = os.path.join(meta_dir, "info.json")
    if os.path.exists(info_path):
        with open(info_path, "r") as f:
            info = json.load(f)
        
        # Update total_episodes
        info["total_episodes"] = len(idx_map)
        
        # Update splits
        if "splits" in info:
            for split_name, split_range in info["splits"].items():
                parts = split_range.split(":")
                if len(parts) == 2 and parts[0].isdigit():
                    info["splits"][split_name] = f"{parts[0]}:{info['total_episodes']}"
        
        # Calculate total_frames from episodes.jsonl
        episodes_jsonl_path = os.path.join(meta_dir, "episodes.jsonl")
        total_frames = 0
        if os.path.exists(episodes_jsonl_path):
            with open(episodes_jsonl_path, "r") as f:
                for line in f:
                    try:
                        obj = json.loads(line)
                        total_frames += obj.get("length", 0)
                    except Exception:
                        continue
            info["total_frames"] = total_frames
        
        # Calculate total_videos as the sum of .mp4 files in both video dirs
        total_videos = 0
        for vdir in video_dirs:
            if os.path.exists(vdir):
                total_videos += len([f for f in os.listdir(vdir) if f.endswith(".mp4")])
        info["total_videos"] = total_videos
        
        with open(info_path, "w") as f:
            json.dump(info, f, indent=2)
        print(f"Updated info.json: {info['total_episodes']} episodes, {info['total_frames']} frames, {info['total_videos']} videos")

    print("Cleanup and reindexing complete!")

def main():
    if len(sys.argv) == 2 and sys.argv[1] == "--cleanup-reindex":
        cleanup_and_reindex()
        return
    
    print("Usage:")
    print("  python update_data_index.py --cleanup-reindex")
    print("  This will remove episodes specified in EPISODES_TO_REMOVE and reindex all remaining episodes.")

if __name__ == "__main__":
    main()
