# lerobot/utils/verify_reindex.py

import os
import json
import hashlib
from collections import defaultdict

# List of episode indices that were removed
REMOVED_EPISODES = [75, 101, 106, 153, 196]  # Update this to match your removals

def calculate_file_hash(filepath):
    """Calculate SHA256 hash of a file."""
    hash_sha256 = hashlib.sha256()
    try:
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()
    except Exception as e:
        print(f"Error hashing {filepath}: {e}")
        return None

def get_episode_index_from_filename(filename):
    """Extract episode index from filename like episode_000123.ext"""
    try:
        return int(filename.split("_")[1].split(".")[0])
    except:
        return None

def verify_data_files(source_dir, target_dir):
    """Verify data files (.parquet) have same content after reindexing."""
    print("=== Verifying Data Files ===")
    
    source_data_dir = os.path.join(source_dir, "data/chunk-000")
    target_data_dir = os.path.join(target_dir, "data/chunk-000")
    
    if not os.path.exists(source_data_dir) or not os.path.exists(target_data_dir):
        print("Data directories not found!")
        return False
    
    # Get source files (excluding removed episodes)
    source_files = {}
    for fname in os.listdir(source_data_dir):
        if fname.startswith("episode_") and fname.endswith(".parquet"):
            ep_idx = get_episode_index_from_filename(fname)
            if ep_idx is not None and ep_idx not in REMOVED_EPISODES:
                source_files[ep_idx] = fname
    
    # Get target files
    target_files = {}
    for fname in os.listdir(target_data_dir):
        if fname.startswith("episode_") and fname.endswith(".parquet"):
            ep_idx = get_episode_index_from_filename(fname)
            if ep_idx is not None:
                target_files[ep_idx] = fname
    
    # Create mapping: old_idx -> new_idx
    sorted_source_indices = sorted(source_files.keys())
    idx_mapping = {old: new for new, old in enumerate(sorted_source_indices)}
    
    print(f"Source episodes (after removal): {len(source_files)}")
    print(f"Target episodes: {len(target_files)}")
    
    if len(source_files) != len(target_files):
        print("❌ Episode count mismatch!")
        return False
    
    # Verify file contents
    all_match = True
    for old_idx, new_idx in idx_mapping.items():
        source_file = os.path.join(source_data_dir, source_files[old_idx])
        target_file = os.path.join(target_data_dir, target_files[new_idx])
        
        source_hash = calculate_file_hash(source_file)
        target_hash = calculate_file_hash(target_file)
        
        if source_hash != target_hash:
            print(f"❌ Hash mismatch: episode {old_idx} -> {new_idx}")
            all_match = False
        else:
            print(f"✅ episode {old_idx} -> {new_idx}: Hash match")
    
    return all_match

def verify_video_files(source_dir, target_dir):
    """Verify video files (.mp4) have same content after reindexing."""
    print("\n=== Verifying Video Files ===")
    
    video_subdirs = [
        "videos/chunk-000/observation.images.room",
        "videos/chunk-000/observation.images.wrist"
    ]
    
    all_match = True
    for subdir in video_subdirs:
        print(f"\nChecking {subdir}...")
        source_video_dir = os.path.join(source_dir, subdir)
        target_video_dir = os.path.join(target_dir, subdir)
        
        if not os.path.exists(source_video_dir) or not os.path.exists(target_video_dir):
            print(f"Video directories not found for {subdir}!")
            all_match = False
            continue
        
        # Get source files (excluding removed episodes)
        source_files = {}
        for fname in os.listdir(source_video_dir):
            if fname.startswith("episode_") and fname.endswith(".mp4"):
                ep_idx = get_episode_index_from_filename(fname)
                if ep_idx is not None and ep_idx not in REMOVED_EPISODES:
                    source_files[ep_idx] = fname
        
        # Get target files
        target_files = {}
        for fname in os.listdir(target_video_dir):
            if fname.startswith("episode_") and fname.endswith(".mp4"):
                ep_idx = get_episode_index_from_filename(fname)
                if ep_idx is not None:
                    target_files[ep_idx] = fname
        
        # Create mapping
        sorted_source_indices = sorted(source_files.keys())
        idx_mapping = {old: new for new, old in enumerate(sorted_source_indices)}
        
        if len(source_files) != len(target_files):
            print(f"❌ Episode count mismatch in {subdir}!")
            all_match = False
            continue
        
        # Verify file contents
        for old_idx, new_idx in idx_mapping.items():
            source_file = os.path.join(source_video_dir, source_files[old_idx])
            target_file = os.path.join(target_video_dir, target_files[new_idx])
            
            source_hash = calculate_file_hash(source_file)
            target_hash = calculate_file_hash(target_file)
            
            if source_hash != target_hash:
                print(f"❌ Hash mismatch: episode {old_idx} -> {new_idx}")
                all_match = False
            else:
                print(f"✅ episode {old_idx} -> {new_idx}: Hash match")
    
    return all_match

def verify_meta_files(source_dir, target_dir):
    """Verify meta .jsonl files have correct episode records after reindexing."""
    print("\n=== Verifying Meta Files ===")
    
    source_meta_dir = os.path.join(source_dir, "meta")
    target_meta_dir = os.path.join(target_dir, "meta")
    
    if not os.path.exists(source_meta_dir) or not os.path.exists(target_meta_dir):
        print("Meta directories not found!")
        return False
    
    all_match = True
    
    for fname in os.listdir(source_meta_dir):
        if fname.endswith(".jsonl"):
            print(f"\nChecking {fname}...")
            source_file = os.path.join(source_meta_dir, fname)
            target_file = os.path.join(target_meta_dir, fname)
            
            if not os.path.exists(target_file):
                print(f"❌ Target file {fname} not found!")
                all_match = False
                continue
            
            # Read source records (excluding removed episodes)
            source_records = {}
            with open(source_file, "r") as f:
                for line in f:
                    try:
                        obj = json.loads(line)
                        ep_idx = obj.get("episode_index")
                        if ep_idx is not None and ep_idx not in REMOVED_EPISODES:
                            # Remove episode_index for comparison
                            obj_copy = obj.copy()
                            del obj_copy["episode_index"]
                            source_records[ep_idx] = obj_copy
                    except Exception as e:
                        print(f"Error reading source {fname}: {e}")
            
            # Read target records
            target_records = {}
            with open(target_file, "r") as f:
                for line in f:
                    try:
                        obj = json.loads(line)
                        ep_idx = obj.get("episode_index")
                        if ep_idx is not None:
                            # Remove episode_index for comparison
                            obj_copy = obj.copy()
                            del obj_copy["episode_index"]
                            target_records[ep_idx] = obj_copy
                    except Exception as e:
                        print(f"Error reading target {fname}: {e}")
            
            # Create mapping
            sorted_source_indices = sorted(source_records.keys())
            idx_mapping = {old: new for new, old in enumerate(sorted_source_indices)}
            
            if len(source_records) != len(target_records):
                print(f"❌ Record count mismatch in {fname}!")
                all_match = False
                continue
            
            # Verify record contents
            for old_idx, new_idx in idx_mapping.items():
                if source_records[old_idx] != target_records[new_idx]:
                    print(f"❌ Record mismatch in {fname}: episode {old_idx} -> {new_idx}")
                    all_match = False
                else:
                    print(f"✅ episode {old_idx} -> {new_idx}: Record match")
    
    return all_match

def verify_info_json(source_dir, target_dir):
    """Verify info.json has correct updated values."""
    print("\n=== Verifying info.json ===")
    
    source_info_file = os.path.join(source_dir, "meta/info.json")
    target_info_file = os.path.join(target_dir, "meta/info.json")
    
    if not os.path.exists(source_info_file) or not os.path.exists(target_info_file):
        print("info.json files not found!")
        return False
    
    with open(source_info_file, "r") as f:
        source_info = json.load(f)
    
    with open(target_info_file, "r") as f:
        target_info = json.load(f)
    
    all_correct = True
    
    # Check total_episodes
    expected_episodes = source_info.get("total_episodes", 0) - len(REMOVED_EPISODES)
    actual_episodes = target_info.get("total_episodes", 0)
    if expected_episodes == actual_episodes:
        print(f"✅ total_episodes: {actual_episodes} (correct)")
    else:
        print(f"❌ total_episodes: expected {expected_episodes}, got {actual_episodes}")
        all_correct = False
    
    # Check total_frames (need to calculate from episodes.jsonl)
    episodes_jsonl = os.path.join(source_dir, "meta/episodes.jsonl")
    if os.path.exists(episodes_jsonl):
        removed_frames = 0
        with open(episodes_jsonl, "r") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    if obj.get("episode_index") in REMOVED_EPISODES:
                        removed_frames += obj.get("length", 0)
                except:
                    continue
        
        expected_frames = source_info.get("total_frames", 0) - removed_frames
        actual_frames = target_info.get("total_frames", 0)
        if expected_frames == actual_frames:
            print(f"✅ total_frames: {actual_frames} (correct)")
        else:
            print(f"❌ total_frames: expected {expected_frames}, got {actual_frames}")
            all_correct = False
    
    # Check total_videos
    expected_videos = source_info.get("total_videos", 0) - (len(REMOVED_EPISODES) * 2)  # 2 videos per episode
    actual_videos = target_info.get("total_videos", 0)
    if expected_videos == actual_videos:
        print(f"✅ total_videos: {actual_videos} (correct)")
    else:
        print(f"❌ total_videos: expected {expected_videos}, got {actual_videos}")
        all_correct = False
    
    # Check splits
    if "splits" in source_info and "splits" in target_info:
        for split_name, source_range in source_info["splits"].items():
            target_range = target_info["splits"].get(split_name, "")
            expected_range = f"0:{expected_episodes}"
            if target_range == expected_range:
                print(f"✅ splits[{split_name}]: {target_range} (correct)")
            else:
                print(f"❌ splits[{split_name}]: expected {expected_range}, got {target_range}")
                all_correct = False
    
    return all_correct

def main():
    if len(sys.argv) != 3:
        print("Usage: python verify_reindex.py <source_dir> <target_dir>")
        print("Example: python verify_reindex.py backup/2_cameras_fps15_enhanced_gripper 2_cameras_fps15_enhanced_gripper")
        sys.exit(1)
    
    source_dir = sys.argv[1]
    target_dir = sys.argv[2]
    
    print(f"Verifying reindex results...")
    print(f"Source: {source_dir}")
    print(f"Target: {target_dir}")
    print(f"Removed episodes: {REMOVED_EPISODES}")
    
    # Run all verifications
    data_ok = verify_data_files(source_dir, target_dir)
    video_ok = verify_video_files(source_dir, target_dir)
    meta_ok = verify_meta_files(source_dir, target_dir)
    info_ok = verify_info_json(source_dir, target_dir)
    
    print("\n" + "="*50)
    print("VERIFICATION SUMMARY:")
    print(f"Data files: {'✅ PASS' if data_ok else '❌ FAIL'}")
    print(f"Video files: {'✅ PASS' if video_ok else '❌ FAIL'}")
    print(f"Meta files: {'✅ PASS' if meta_ok else '❌ FAIL'}")
    print(f"Info.json: {'✅ PASS' if info_ok else '❌ FAIL'}")
    
    if all([data_ok, video_ok, meta_ok, info_ok]):
        print("\n🎉 ALL VERIFICATIONS PASSED!")
    else:
        print("\n⚠️  SOME VERIFICATIONS FAILED!")

if __name__ == "__main__":
    import sys
    main()
