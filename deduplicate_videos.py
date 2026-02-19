import os
import hashlib
import argparse
import shutil
import re
from collections import defaultdict

import random

def get_file_signature(filepath, num_blocks=3, block_size=4096):
    """
    Returns a fast signature using MD5 of 3 random blocks + file size.
    Random offsets are seeded by file size, so identical files always
    produce the same hash.
    """
    try:
        file_size = os.path.getsize(filepath)
        if file_size == 0:
            return "empty"
        
        # Seed RNG with file size so same-size files get same offsets
        rng = random.Random(file_size)
        
        hasher = hashlib.md5()
        hasher.update(str(file_size).encode('utf-8'))
        
        with open(filepath, 'rb') as f:
            for _ in range(num_blocks):
                offset = rng.randint(0, max(0, file_size - block_size))
                f.seek(offset)
                hasher.update(f.read(block_size))
        
        return hasher.hexdigest()
    except Exception as e:
        print(f"[WARN] Could not read {filepath}: {e}")
        return None

def normalize_filename(filename):
    """
    Strip common 'copy' patterns from filenames to find potential duplicates.
    E.g. "video(1).mp4" -> "video.mp4"
         "video - Copy.mp4" -> "video.mp4"
         "video_1.mp4" -> "video.mp4"
    """
    name, ext = os.path.splitext(filename)
    
    # Remove (1), (2), etc.
    name = re.sub(r'\s*\(\d+\)$', '', name)
    # Remove " - Copy" (and optional number)
    name = re.sub(r'\s*-\s*Copy(?:\s*\(\d+\))?$', '', name, flags=re.IGNORECASE)
    # Remove trailing underscore+number (e.g. _1, _01) often added by downloaders
    name = re.sub(r'_\d+$', '', name)
    
    return (name + ext).lower()

def _find_dups_in_groups(groups):
    """Given a dict of {key: [paths]}, verify duplicates with content hash."""
    duplicates = []
    
    for key, paths in groups.items():
        if len(paths) < 2:
            continue
        
        # Group by signature
        sigs = defaultdict(list)
        for p in paths:
            sig = get_file_signature(p)
            if sig:
                sigs[sig].append(p)
        
        # Any group with > 1 item is a confirmed duplicate set
        for sig, matching_paths in sigs.items():
            if len(matching_paths) > 1:
                # Keep the cleanest/shallowest file as original
                matching_paths.sort(key=lambda x: (
                    len(os.path.basename(x)),   # Shortest filename first
                    len(x.split(os.sep)),       # Shallower path first
                    x                           # Alphabetical
                ))
                
                original = matching_paths[0]
                for d in matching_paths[1:]:
                    duplicates.append((original, d, sig))
    
    return duplicates

def find_duplicates(root_folder):
    print(f"Scanning {root_folder}...")
    
    # Collect all video files
    all_files = []
    for dirpath, _, filenames in os.walk(root_folder):
        if "_duplicates" in os.path.normpath(dirpath).split(os.sep):
            continue
        for filename in filenames:
            ext = os.path.splitext(filename)[1].lower()
            if ext in ['.mp4', '.mov', '.avi', '.mkv', '.webm']:
                all_files.append(os.path.join(dirpath, filename))
    
    print(f"Found {len(all_files)} video files.")
    
    # ── Pass 1: Group by normalized filename ──
    print("\n[Pass 1] Checking for duplicates with similar filenames...")
    files_by_norm_name = defaultdict(list)
    for f in all_files:
        norm_name = normalize_filename(os.path.basename(f))
        files_by_norm_name[norm_name].append(f)
    
    pass1_dupes = _find_dups_in_groups(files_by_norm_name)
    print(f"[Pass 1] Found {len(pass1_dupes)} duplicates by filename match.")
    
    # Collect all files already identified as duplicates (to skip in pass 2)
    already_found = set()
    for orig, dupe, sig in pass1_dupes:
        already_found.add(orig)
        already_found.add(dupe)
    
    # ── Pass 2: Group remaining files by size ──
    remaining = [f for f in all_files if f not in already_found]
    print(f"\n[Pass 2] Checking {len(remaining)} remaining files by size...")
    
    files_by_size = defaultdict(list)
    for f in remaining:
        try:
            size = os.path.getsize(f)
            files_by_size[size].append(f)
        except OSError:
            pass
    
    pass2_dupes = _find_dups_in_groups(files_by_size)
    print(f"[Pass 2] Found {len(pass2_dupes)} duplicates by content match (different names).")
    
    all_dupes = pass1_dupes + pass2_dupes
    return all_dupes

def main():
    parser = argparse.ArgumentParser(
        description="Find and move duplicate video files based on filename patterns AND random-block content hashing.",
        epilog="By default, duplicates are MOVED to a '_duplicates' subfolder. Use --dry-run to only list them without moving."
    )
    parser.add_argument("folder", help="Folder to scan")
    parser.add_argument("--duplicates-folder", help="Custom folder to move duplicates to (default: '_duplicates' inside source folder)")
    parser.add_argument("--dry-run", action="store_true", help="Just list duplicates, do not move files")
    
    args = parser.parse_args()
    
    if not os.path.isdir(args.folder):
        print(f"Error: {args.folder} is not a directory.")
        return

    dupes = find_duplicates(args.folder)
    
    if not dupes:
        print("No duplicates found.")
        return
        
    print(f"\nFound {len(dupes)} duplicates:")
    for orig, dupe, sig in dupes:
        print(f"  Hash:   {sig}")
        print(f"  Keep:   {orig}")
        print(f"  Remove: {dupe}")
        print("-" * 40)
        
    if not args.dry_run:
        # Determine destination folder — always nest inside a '_duplicates' parent
        if args.duplicates_folder:
            dup_dir = os.path.join(os.path.abspath(args.duplicates_folder), "_duplicates")
        else:
            dup_dir = os.path.join(args.folder, "_duplicates")
            
        print(f"\nMoving duplicates to '{dup_dir}'...")
        os.makedirs(dup_dir, exist_ok=True)
        
        count = 0
        for orig, dupe, sig in dupes:
            try:
                # Create a structure inside duplicates folder that mirrors the original relative path
                rel_path = os.path.relpath(dupe, args.folder)
                target_path = os.path.join(dup_dir, rel_path)
                
                os.makedirs(os.path.dirname(target_path), exist_ok=True)
                
                # If target exists, ONLY THEN do we rename
                if os.path.exists(target_path):
                    base, ext = os.path.splitext(target_path)
                    counter = 1
                    while os.path.exists(target_path):
                        target_path = f"{base}_COPY_{counter}{ext}"
                        counter += 1
                
                shutil.move(dupe, target_path)
                print(f"  Moved: {rel_path} -> {os.path.basename(target_path)}")
                count += 1
            except Exception as e:
                print(f"  Error moving {dupe}: {e}")
        print(f"Done. Moved {count} files.")
        
    else:
        print("\n[DRY RUN] No files were moved.")
        print("Run without --dry-run to move these files.")

if __name__ == "__main__":
    main()
