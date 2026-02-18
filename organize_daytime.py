import argparse
import os
import shutil
import sys
import csv
import re
from glob import glob
from datetime import datetime

# Import from sibling script
import rename_video_files as rv

def parse_args():
    parser = argparse.ArgumentParser(description="Organize video files by date and time of day (Dawn to Dusk).")
    parser.add_argument("source", help="Source folder containing videos")
    parser.add_argument("destination", nargs="?", help="Destination folder (optional, defaults to Source)")
    parser.add_argument("--dawn", default="0600", help="Start time in HHMM format (default: 0600)")
    parser.add_argument("--dusk", default="1800", help="End time in HHMM format (default: 1800)")
    parser.add_argument("--no-recursive", action="store_false", dest="recursive", help="Disable recursive search (default: enabled)")
    parser.add_argument("--dry-run", action="store_true", help="Scan only, do not move files")
    return parser.parse_args()

def get_datetime_from_filename(filename):
    # Try parsing date/time from a filename that was already renamed by the tool.
    # Format: YYMMDD-DAY-HHMM-HHMM-ampm-...
    match = re.match(r'^(\d{6})-[A-Za-z]{3,4}-(\d{4})-(\d{4})-(am|pm)-', filename, re.IGNORECASE)
    if match:
        yymmdd = match.group(1)
        start_hhmm = match.group(2)
        ampm = match.group(4).lower()
        
        try:
            year = int("20" + yymmdd[0:2])
            month = int(yymmdd[2:4])
            day = int(yymmdd[4:6])
            
            hh = int(start_hhmm[:2])
            mm = int(start_hhmm[2:])
            
            if ampm == 'pm' and hh < 12:
                hh += 12
            if ampm == 'am' and hh == 12:
                hh = 0
                
            return datetime(year, month, day), hh * 100 + mm
        except ValueError:
            pass
    return None, None

def get_datetime_from_vlm(file_path):
    print(f"  [VLM] Extracting frames for {os.path.basename(file_path)}...")
    frames, duration = rv._extract_frames(file_path, num_frames=2)
    if not frames:
        return None, None
        
    model, processor = rv._load_vlm()
    if not model:
        return None, None
        
    found_dt = None
    found_time = None
    
    for frame in frames:
        cropped = rv._crop_timestamp_region(frame["path"])
        text = rv._read_timestamp_from_image(model, processor, cropped)
        print(f"  [VLM] Read: {text}")
        
        date_part, time_part, ampm, day = rv._parse_timestamp_parts(text)
        
        if date_part and time_part:
            try:
                year = int("20" + date_part[0:2])
                month = int(date_part[2:4])
                d = int(date_part[4:6])
                
                hh = int(time_part[:2])
                mm = int(time_part[2:])
                
                if ampm:
                    if ampm.lower() == 'pm' and hh < 12:
                        hh += 12
                    if ampm.lower() == 'am' and hh == 12:
                        hh = 0
                
                found_dt = datetime(year, month, d)
                found_time = hh * 100 + mm
                break
            except ValueError:
                continue
    
    rv._cleanup_frames(frames)
    return found_dt, found_time

def main():
    args = parse_args()
    
    if not os.path.exists(args.source):
        print(f"[ERROR] Source folder not found: {args.source}")
        return

    # Default destination to source if not provided
    if not args.destination:
        args.destination = args.source
        print(f"Destination not specified. Organizing into daily folders inside Source.")

    try:
        dawn = int(args.dawn)
        dusk = int(args.dusk)
    except ValueError:
        print("[ERROR] Dawn/Dusk must be HHMM integers (e.g. 0600)")
        return
    
    if args.recursive:
        files = sorted(glob(os.path.join(args.source, "**", "*.mp4"), recursive=True))
    else:
        files = sorted(glob(os.path.join(args.source, "*.mp4")))
        
    print(f"Found {len(files)} files in {args.source}")
    print(f"Filtering for Daytime: {dawn:04d} to {dusk:04d}")
    
    results = []
    
    for f in files:
        filename = os.path.basename(f)
        print(f"\nProcessing: {filename}")
        
        dt, time_val = get_datetime_from_filename(filename)
        
        if not dt:
            dt, time_val = get_datetime_from_vlm(f)
            
        if not dt:
            print("  [SKIP] Could not determine timestamp.")
            continue
            
        if dawn <= time_val <= dusk:
            print(f"  [MATCH] {time_val:04d} (keep)")
            date_str = dt.strftime("%y%m%d")
            dest_folder = os.path.join(args.destination, date_str)
            dest_path = os.path.join(dest_folder, filename)
            
            results.append({
                "file": f,
                "date": date_str,
                "dest": dest_path
            })
        else:
            print(f"  [SKIP] {time_val:04d} (outside range)")

    # Save CSV
    csv_file = "daytime_videos.csv"
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["file", "date", "dest"])
        writer.writeheader()
        writer.writerows(results)
    print(f"\n[REPORT] Saved list to {csv_file}")
    
    # Move files
    if not args.dry_run:
        print(f"\nMoving {len(results)} files...")
        count = 0
        for row in results:
            os.makedirs(os.path.dirname(row["dest"]), exist_ok=True)
            try:
                shutil.move(row["file"], row["dest"])
                print(f"  Moved: {os.path.basename(row['file'])} -> {row['date']}")
                count += 1
            except Exception as e:
                print(f"  [ERROR] Move failed: {e}")
        print(f"Done. Moved {count} files.")
    else:
        print("[DRY RUN] No files moved.")

if __name__ == "__main__":
    main()
