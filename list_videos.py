import argparse
import os
import csv
from glob import glob

VIDEO_EXTENSIONS = {'.mp4', '.mkv', '.mov', '.avi', '.webm', '.flv', '.wmv', '.m4v'}

def main():
    parser = argparse.ArgumentParser(description="List all video files in a folder to a CSV file.")
    parser.add_argument("folder", help="Folder to scan for videos")
    parser.add_argument("--output", help="Output CSV file path (e.g. videos.csv). Defaults to 'video_list.csv' in current folder.")
    parser.add_argument("--no-recursive", dest="recursive", action="store_false", help="Do not scan recursively (flat search)")
    parser.set_defaults(recursive=True)
    
    args = parser.parse_args()
    
    source_dir = os.path.abspath(args.folder)
    
    if not os.path.exists(source_dir):
        print(f"[ERROR] Folder not found: {source_dir}")
        return

    # Determine output file
    if args.output:
        csv_path = args.output
    else:
        # Default to current working directory
        csv_path = os.path.join(os.getcwd(), "video_list.csv")
        
    print(f"Scanning: {source_dir}")
    print(f"Recursive: {args.recursive}")
    
    files_found = []
    
    # Efficient recursive walk if supported (glob usually ok)
    if args.recursive:
        # Using os.walk for better cross-platform + extension filtering control
        for root, dirs, files in os.walk(source_dir):
            for file in files:
                ext = os.path.splitext(file)[1].lower()
                if ext in VIDEO_EXTENSIONS:
                    files_found.append(os.path.join(root, file))
    else:
        for file in os.listdir(source_dir):
            ext = os.path.splitext(file)[1].lower()
            if ext in VIDEO_EXTENSIONS:
                files_found.append(os.path.join(source_dir, file))
                
    files_found.sort()
    count = len(files_found)
    
    print(f"Found {count} video files.")
    
    try:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["File Path"])
            
            for path in files_found:
                writer.writerow([path])
                
            # Add footer tally
            writer.writerow([])
            writer.writerow([f"Total Count: {count}"])
            
        print(f"Saved list to: {csv_path}")
        print(f"Total Count: {count}")
    except Exception as e:
        print(f"[ERROR] Could not write to CSV: {e}")

if __name__ == "__main__":
    main()
